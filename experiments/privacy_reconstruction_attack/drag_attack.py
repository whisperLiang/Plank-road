from __future__ import annotations

import argparse
import csv
import inspect
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    from common.cuda_visibility import configure_default_cuda_visible_devices

    configure_default_cuda_visible_devices()

import torch
import torch.nn.functional as F
from loguru import logger

from config import load_runtime_config
from experiments.privacy_reconstruction_attack.attack_dataset import (
    AttackSample,
    group_samples_by_split,
    load_attack_samples,
    load_experiment_config,
    load_rgb_image,
    load_tensor,
    read_json,
    save_tensor_image,
    write_json,
)
from experiments.privacy_reconstruction_attack.boundary_feature_adapter import (
    BoundaryFeatureAdapter,
)
from experiments.privacy_reconstruction_attack.collect_attack_targets import (
    _build_trace_splitter,
    configure_object_detection_device,
)
from experiments.privacy_reconstruction_attack.pixel_dra_attack import (
    RuntimeInputAdapter,
    _freeze,
    _prediction_on_reconstruction,
)
from experiments.privacy_reconstruction_attack.reconstruction_metrics import (
    OptionalLPIPS,
    evaluate_reconstruction,
    total_variation,
)
from model_management.object_detection import Object_Detection


def _require_diffusers():
    try:
        from diffusers import DDIMScheduler, StableDiffusionPipeline  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "DRAG guided diffusion requires the optional dependency 'diffusers'. "
            "Install diffusers and a compatible transformers/accelerate stack, "
            "or disable drag.enabled. "
            "This script does not fall back to Pixel DRA."
        ) from exc
    return DDIMScheduler, StableDiffusionPipeline


def _torch_dtype(name: object) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    text = str(name or "float16").replace("torch.", "")
    dtype = getattr(torch, text, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {name!r}")
    return dtype


def _encode_prompt(pipe: Any, device: torch.device) -> torch.Tensor:
    if hasattr(pipe, "encode_prompt"):
        result = pipe.encode_prompt(
            prompt="",
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        return result[0] if isinstance(result, tuple) else result
    return pipe._encode_prompt("", device, 1, False)


def _decode_latents(pipe: Any, latents: torch.Tensor) -> torch.Tensor:
    scaling = float(getattr(getattr(pipe.vae, "config", None), "scaling_factor", 0.18215))
    decoded = pipe.vae.decode(latents / scaling).sample
    return (decoded / 2.0 + 0.5).clamp(0.0, 1.0)


def _scheduler_step(
    scheduler: Any,
    noise_pred: torch.Tensor,
    timestep: torch.Tensor,
    latents: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    kwargs = {}
    try:
        if "eta" in inspect.signature(scheduler.step).parameters:
            kwargs["eta"] = float(eta)
    except (TypeError, ValueError):
        pass
    return scheduler.step(noise_pred, timestep, latents, **kwargs).prev_sample


def _write_curve(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "internal_iteration",
                "feature_loss",
                "regularization_loss",
                "total_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _load_pipeline(config: Mapping[str, Any], device: torch.device) -> Any:
    DDIMScheduler, StableDiffusionPipeline = _require_diffusers()
    dtype = _torch_dtype(config.get("torch_dtype", "float16"))
    if device.type == "cpu" and dtype == torch.float16:
        logger.warning(
            "[DRAG] CPU execution requested with float16; using float32 for Stable Diffusion."
        )
        dtype = torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        str(config.get("checkpoint", "runwayml/stable-diffusion-v1-5")),
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.vae.eval()
    pipe.unet.eval()
    for module in (pipe.vae, pipe.unet, getattr(pipe, "text_encoder", None)):
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return pipe


def _guided_diffusion_sample(
    sample: AttackSample,
    *,
    pipe: Any,
    splitter: Any,
    runtime_adapter: RuntimeInputAdapter,
    adapter: BoundaryFeatureAdapter,
    drag_cfg: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, float, list[dict[str, float]]]:
    target_payload = adapter.load_payload(sample.path("boundary_payload_path"))
    prompt_embeds = _encode_prompt(pipe, device)
    image_size = int(drag_cfg.get("image_size", 512))
    height = width = image_size
    vae_scale_factor = 2 ** (len(getattr(pipe.vae.config, "block_out_channels", [1, 1, 1, 1])) - 1)
    latent_channels = int(getattr(pipe.unet.config, "in_channels", 4))
    latents = torch.randn(
        (1, latent_channels, height // vae_scale_factor, width // vae_scale_factor),
        device=device,
        dtype=prompt_embeds.dtype,
    )
    pipe.scheduler.set_timesteps(int(drag_cfg.get("num_inference_steps", 80)), device=device)
    latents = latents * float(getattr(pipe.scheduler, "init_noise_sigma", 1.0))
    guidance_rate = float(drag_cfg.get("guidance_rate", 0.2))
    max_grad_norm = float(drag_cfg.get("max_grad_norm", 0.02))
    internal_iterations = max(1, int(drag_cfg.get("num_internal_iterations", 4)))
    eta = float(drag_cfg.get("eta", 1.0))
    l2_weight = float(drag_cfg.get("l2_regularization_x", 0.01))
    tv_weight = float(drag_cfg.get("total_variation_x", 0.0))
    log_every = max(1, int(drag_cfg.get("log_every_n_steps", 20)))
    curve: list[dict[str, float]] = []
    final_feature_loss = float("nan")

    for step_index, timestep in enumerate(pipe.scheduler.timesteps, start=1):
        for internal_index in range(1, internal_iterations + 1):
            latents = latents.detach().requires_grad_(True)
            image = _decode_latents(pipe, latents).float()
            attack_image = F.interpolate(
                image,
                size=(runtime_adapter.height, runtime_adapter.width),
                mode="bilinear",
                align_corners=False,
            )
            pred_payload = adapter.model_forward_to_payload(
                splitter,
                runtime_adapter.to_runtime_input(attack_image),
            )
            feature_loss = adapter.feature_distance(pred_payload, target_payload)
            reg = l2_weight * image.square().mean() + tv_weight * total_variation(image)
            total = feature_loss + reg
            grad = torch.autograd.grad(total, latents)[0]
            grad_norm = grad.float().norm()
            if max_grad_norm > 0.0:
                grad = grad * min(1.0, max_grad_norm / (float(grad_norm.detach().cpu()) + 1.0e-12))
            latents = (latents - guidance_rate * grad).detach()
            final_feature_loss = float(feature_loss.detach().cpu())
            if (
                step_index == 1
                or step_index % log_every == 0
                or step_index == len(pipe.scheduler.timesteps)
            ):
                curve.append(
                    {
                        "step": float(step_index),
                        "internal_iteration": float(internal_index),
                        "feature_loss": final_feature_loss,
                        "regularization_loss": float(reg.detach().cpu()),
                        "total_loss": float(total.detach().cpu()),
                    }
                )
        with torch.no_grad():
            noise_pred = pipe.unet(latents, timestep, encoder_hidden_states=prompt_embeds).sample
            latents = _scheduler_step(pipe.scheduler, noise_pred, timestep, latents, eta)
        if (
            step_index == 1
            or step_index % log_every == 0
            or step_index == len(pipe.scheduler.timesteps)
        ):
            logger.info(
                "[DRAG] split={} sample={} step={} feature_loss={:.6g}",
                sample.split_name,
                sample.sample_id,
                step_index,
                final_feature_loss,
            )

    with torch.no_grad():
        image = _decode_latents(pipe, latents).float()
        image = F.interpolate(
            image,
            size=(runtime_adapter.height, runtime_adapter.width),
            mode="bilinear",
            align_corners=False,
        )
    return image.detach(), final_feature_loss, curve


def _run_one_sample(
    sample: AttackSample,
    *,
    pipe: Any,
    splitter: Any,
    runtime_adapter: RuntimeInputAdapter,
    adapter: BoundaryFeatureAdapter,
    teacher: Object_Detection,
    drag_cfg: Mapping[str, Any],
    metrics_cfg: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
    lpips_metric: OptionalLPIPS,
) -> None:
    try:
        image, final_feature_loss, curve = _guided_diffusion_sample(
            sample,
            pipe=pipe,
            splitter=splitter,
            runtime_adapter=runtime_adapter,
            adapter=adapter,
            drag_cfg=drag_cfg,
            device=device,
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "DRAG ran out of memory. Reduce num_inference_steps, "
                "num_internal_iterations, image_size, torch_dtype, or max_samples_per_split."
            ) from exc
        raise

    sample_out = output_dir / sample.split_name / sample.sample_id
    sample_out.mkdir(parents=True, exist_ok=True)
    save_tensor_image(sample_out / "recon.png", image)
    raw_rgb = load_rgb_image(sample.path("raw_image_path"))
    from experiments.privacy_reconstruction_attack.attack_dataset import save_rgb_image

    save_rgb_image(sample_out / "raw.png", raw_rgb)
    recon_rgb = load_rgb_image(sample_out / "recon.png")
    original_teacher = read_json(sample.path("teacher_prediction_path"))
    teacher_threshold = metrics_cfg.get("teacher_threshold")
    teacher_threshold_value = None if teacher_threshold is None else float(teacher_threshold)
    recon_teacher = _prediction_on_reconstruction(teacher, image, threshold=teacher_threshold_value)
    metrics = evaluate_reconstruction(
        raw_rgb,
        recon_rgb,
        original_teacher_prediction=original_teacher,
        recon_teacher_prediction=recon_teacher,
        feature_distance_final=final_feature_loss,
        lpips_metric=lpips_metric,
        object_iou_threshold=float(metrics_cfg.get("object_iou_threshold", 0.5)),
        psnr_norm_max=float(metrics_cfg.get("psnr_norm_max", 40.0)),
        lpips_norm_max=float(metrics_cfg.get("lpips_norm_max", 1.0)),
    )
    metrics.update(
        {
            "method": "drag",
            "split_name": sample.split_name,
            "split_point": sample.split_point,
            "sample_id": sample.sample_id,
            "frame_index": int(sample.metadata.get("frame_index", 0)),
            "privacy_leakage_score": float(sample.privacy_leakage_score),
            "recon_teacher_prediction": recon_teacher,
        }
    )
    write_json(sample_out / "metrics.json", metrics)
    _write_curve(sample_out / "feature_loss_curve.csv", curve)


def run_drag(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device}, but CUDA is not available.")
    configure_object_detection_device(device)
    config = load_experiment_config(args.config)
    drag_cfg = dict(config.get("drag") or {})
    if not bool(drag_cfg.get("enabled", True)):
        logger.info("[DRAG] disabled by config")
        return
    _require_diffusers()
    manifest = read_json(Path(args.targets_dir) / "manifest.json")
    runtime_config = load_runtime_config(manifest.get("yaml_path") or "./config/config.yaml")
    student = Object_Detection(runtime_config.client, "small inference")
    teacher = Object_Detection(runtime_config.server, "large inference")
    student.model.to(device).eval()
    teacher.model.to(device).eval()
    _freeze(student.model)
    _freeze(teacher.model)
    adapter = BoundaryFeatureAdapter.from_config(
        config.get("feature_distance") or {}, device=device
    )
    metrics_cfg = dict(config.get("metrics") or {})
    lpips_metric = OptionalLPIPS(device=device)
    if not lpips_metric.available:
        logger.warning("[DRAG] LPIPS unavailable; metric will be skipped: {}", lpips_metric.error)
    pipe = _load_pipeline(drag_cfg, device)

    grouped = group_samples_by_split(load_attack_samples(args.targets_dir))
    output_dir = Path(args.output_dir)
    for split_name, samples in grouped.items():
        max_samples = int(drag_cfg.get("max_samples_per_split", 10) or 10)
        selected = samples[: max(0, max_samples)]
        if not selected:
            continue
        sample_input = load_tensor(selected[0].path("model_input_tensor_path"), device=device)
        splitter = _build_trace_splitter(
            student,
            sample_input,
            split_point=selected[0].split_point,
            device=device,
        )
        runtime_adapter = RuntimeInputAdapter(
            student,
            tuple(int(dim) for dim in sample_input.shape),
            device=device,
        )
        for sample in selected:
            _run_one_sample(
                sample,
                pipe=pipe,
                splitter=splitter,
                runtime_adapter=runtime_adapter,
                adapter=adapter,
                teacher=teacher,
                drag_cfg=drag_cfg,
                metrics_cfg=metrics_cfg,
                output_dir=output_dir,
                device=device,
                lpips_metric=lpips_metric,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DRAG guided diffusion DRA.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--targets_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> None:
    run_drag(build_parser().parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
