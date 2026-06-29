from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    from common.cuda_visibility import configure_default_cuda_visible_devices

    configure_default_cuda_visible_devices()

import cv2
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
from experiments.privacy_reconstruction_attack.reconstruction_metrics import (
    OptionalLPIPS,
    evaluate_reconstruction,
    range_loss,
    total_variation,
)
from model_management.object_detection import Object_Detection
from model_management.payload import BoundaryPayload


def _freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


class RuntimeInputAdapter:
    def __init__(
        self, detector: Object_Detection, target_shape: tuple[int, ...], *, device: torch.device
    ) -> None:
        if len(target_shape) != 4 or int(target_shape[1]) != 3:
            raise ValueError(f"Expected BCHW model input shape, got {target_shape}.")
        self.detector = detector
        self.height = int(target_shape[-2])
        self.width = int(target_shape[-1])
        self.device = device
        self.is_rfdetr = hasattr(getattr(detector.model, "rfdetr", None), "means")
        self.mean = None
        self.std = None
        if self.is_rfdetr:
            means = getattr(detector.model.rfdetr, "means")
            stds = getattr(detector.model.rfdetr, "stds")
            self.mean = torch.as_tensor(means, dtype=torch.float32, device=device).view(1, 3, 1, 1)
            self.std = torch.as_tensor(stds, dtype=torch.float32, device=device).view(1, 3, 1, 1)

    def to_runtime_input(self, rgb_image: torch.Tensor) -> torch.Tensor:
        x = rgb_image
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW image tensor, got {tuple(x.shape)}.")
        if tuple(x.shape[-2:]) != (self.height, self.width):
            x = F.interpolate(
                x, size=(self.height, self.width), mode="bilinear", align_corners=False
            )
        if self.is_rfdetr and self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std.clamp_min(1.0e-12)
        return x


def _init_image(
    shape: tuple[int, ...],
    *,
    init: str,
    image_range: tuple[float, float],
    device: torch.device,
) -> torch.Tensor:
    low, high = image_range
    init = str(init or "random_noise").lower()
    if init == "gray":
        image = torch.full(shape, 0.5 * (float(low) + float(high)), device=device)
    elif init == "blurred":
        image = torch.rand(shape, device=device) * (float(high) - float(low)) + float(low)
        kernel = min(15, max(3, int(shape[-1]) // 8 * 2 + 1))
        image = F.avg_pool2d(image, kernel_size=kernel, stride=1, padding=kernel // 2)
    elif init == "random_noise":
        image = torch.rand(shape, device=device) * (float(high) - float(low)) + float(low)
    else:
        raise ValueError(f"Unsupported pixel_dra init: {init!r}")
    image = image.clamp(float(low) + 1.0e-4, float(high) - 1.0e-4)
    scaled = (image - float(low)) / max(float(high) - float(low), 1.0e-12)
    return torch.logit(scaled.clamp(1.0e-4, 1.0 - 1.0e-4)).detach().requires_grad_(True)


def _param_to_image(param: torch.Tensor, image_range: tuple[float, float]) -> torch.Tensor:
    low, high = image_range
    return torch.sigmoid(param) * (float(high) - float(low)) + float(low)


def _prediction_on_reconstruction(
    teacher: Object_Detection,
    image_tensor: torch.Tensor,
    *,
    threshold: float | None,
) -> dict[str, Any]:
    rgb = image_tensor.detach().cpu().clamp(0.0, 1.0)[0].permute(1, 2, 0).numpy()
    bgr = cv2.cvtColor((rgb * 255.0).round().astype("uint8"), cv2.COLOR_RGB2BGR)
    boxes, labels, scores = teacher.large_inference(bgr, threshold=threshold)
    from experiments.privacy_reconstruction_attack.attack_dataset import prediction_to_json

    return prediction_to_json(boxes or [], labels or [], scores or [], image_size=bgr.shape[:2])


def _payload_feature_distance_value(
    adapter: BoundaryFeatureAdapter,
    splitter: Any,
    runtime_adapter: RuntimeInputAdapter,
    image: torch.Tensor,
    target_payload: BoundaryPayload,
) -> torch.Tensor:
    pred_payload = adapter.model_forward_to_payload(
        splitter, runtime_adapter.to_runtime_input(image)
    )
    return adapter.feature_distance(pred_payload, target_payload)


def _write_curve(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "iteration",
                "feature_loss",
                "tv_loss",
                "l2_loss",
                "range_loss",
                "total_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _run_one_sample(
    sample: AttackSample,
    *,
    splitter: Any,
    runtime_adapter: RuntimeInputAdapter,
    adapter: BoundaryFeatureAdapter,
    teacher: Object_Detection,
    pixel_cfg: Mapping[str, Any],
    metrics_cfg: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
    lpips_metric: OptionalLPIPS,
) -> None:
    target_payload = adapter.load_payload(sample.path("boundary_payload_path"))
    target_input = load_tensor(sample.path("model_input_tensor_path"), device=device)
    shape = tuple(int(dim) for dim in target_input.shape)
    image_range = tuple(float(v) for v in list(pixel_cfg.get("image_range", [0.0, 1.0]))[:2])
    param = _init_image(
        shape,
        init=str(pixel_cfg.get("init", "random_noise")),
        image_range=image_range,
        device=device,
    )
    optimizer_name = str(pixel_cfg.get("optimizer", "Adam")).lower()
    if optimizer_name != "adam":
        raise ValueError("pixel_dra currently supports optimizer: Adam")
    optimizer = torch.optim.Adam([param], lr=float(pixel_cfg.get("lr", 0.05)))
    iterations = int(pixel_cfg.get("iterations", 800))
    log_every = max(1, int(pixel_cfg.get("log_every", 100)))
    tv_weight = float(pixel_cfg.get("tv_weight", 1.0e-4))
    l2_weight = float(pixel_cfg.get("l2_weight", 1.0e-5))
    range_weight = float(pixel_cfg.get("range_weight", 1.0))
    curve: list[dict[str, float]] = []
    final_feature_loss = None

    for iteration in range(1, iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        image = _param_to_image(param, image_range)
        feature_loss = _payload_feature_distance_value(
            adapter,
            splitter,
            runtime_adapter,
            image,
            target_payload,
        )
        tv = total_variation(image)
        l2 = image.square().mean()
        bounded = range_loss(image, image_range)
        loss = feature_loss + tv_weight * tv + l2_weight * l2 + range_weight * bounded
        loss.backward()
        optimizer.step()
        final_feature_loss = float(feature_loss.detach().cpu())
        if iteration == 1 or iteration % log_every == 0 or iteration == iterations:
            row = {
                "iteration": float(iteration),
                "feature_loss": float(feature_loss.detach().cpu()),
                "tv_loss": float(tv.detach().cpu()),
                "l2_loss": float(l2.detach().cpu()),
                "range_loss": float(bounded.detach().cpu()),
                "total_loss": float(loss.detach().cpu()),
            }
            curve.append(row)
            logger.info(
                "[PixelDRA] split={} sample={} iter={} feature_loss={:.6g}",
                sample.split_name,
                sample.sample_id,
                iteration,
                row["feature_loss"],
            )

    image = _param_to_image(param, image_range).detach()
    sample_out = output_dir / sample.split_name / sample.sample_id
    sample_out.mkdir(parents=True, exist_ok=True)
    save_tensor_image(sample_out / "recon.png", image)
    raw_rgb = load_rgb_image(sample.path("raw_image_path"))
    raw_copy = sample_out / "raw.png"
    if not raw_copy.exists():
        from experiments.privacy_reconstruction_attack.attack_dataset import save_rgb_image

        save_rgb_image(raw_copy, raw_rgb)
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
            "method": "pixel_dra",
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


def run_pixel_dra(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device}, but CUDA is not available.")
    configure_object_detection_device(device)
    config = load_experiment_config(args.config)
    pixel_cfg = dict(config.get("pixel_dra") or {})
    if not bool(pixel_cfg.get("enabled", True)):
        logger.info("[PixelDRA] disabled by config")
        return
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
        logger.warning(
            "[PixelDRA] LPIPS unavailable; metric will be skipped: {}", lpips_metric.error
        )

    grouped = group_samples_by_split(load_attack_samples(args.targets_dir))
    output_dir = Path(args.output_dir)
    for split_name, samples in grouped.items():
        max_samples = int(pixel_cfg.get("max_samples_per_split", len(samples)) or len(samples))
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
                splitter=splitter,
                runtime_adapter=runtime_adapter,
                adapter=adapter,
                teacher=teacher,
                pixel_cfg=pixel_cfg,
                metrics_cfg=metrics_cfg,
                output_dir=output_dir,
                device=device,
                lpips_metric=lpips_metric,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pixel optimization DRA.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--targets_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> None:
    run_pixel_dra(build_parser().parse_args(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
