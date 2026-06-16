from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import torch

from cloud.model_update import serialize_model_update
from cloud.training.baseline_workspace import (
    load_baseline_manifest,
    model_builder_kwargs,
    resolve_training_device,
    samples_from_baseline_manifest,
)
from cloud.training.freeze_modes import (
    build_optimizer,
    configure_fixed_prefix_training,
    run_freeze_training,
)
from model_management.fixed_split import (
    FIXED_SPLIT_PLAN_VERSION,
    SplitPlan,
    load_split_plan,
)
from model_management.model_zoo import build_detection_model
from model_management.split_model_adapters import (
    build_split_runtime_sample_input,
    build_split_training_loss,
)
from model_management.split_runtime import make_split_spec
from model_management.universal_model_split import (
    UniversalModelSplitter,
    build_split_retrain_optimizer,
    prepare_exact_split_runtime,
)


class CloudTorchLensFreezeTrainingStrategy:
    name = "freeze"

    def __init__(
        self,
        *,
        learner=None,
        runtime_factory: Callable[[torch.nn.Module, dict[str, Any], Path], Any] | None = None,
        model_builder: Callable[..., torch.nn.Module] | None = None,
        update_serializer: Callable[..., bytes] | None = None,
        loss_builder: Callable[[torch.nn.Module], Callable[[Any, Any], torch.Tensor]] | None = None,
    ) -> None:
        self.learner = learner
        self.runtime_factory = runtime_factory or build_default_torchlens_freeze_runtime
        self.model_builder = model_builder or build_detection_model
        self.update_serializer = update_serializer or serialize_model_update
        self.loss_builder = loss_builder or build_split_training_loss

    def train_from_workspace(
        self,
        workspace: str | Path,
        *,
        base_model_version: str = "0",
        result_model_version: str = "1",
    ) -> dict[str, Any]:
        workspace_path = Path(workspace)
        manifest = load_baseline_manifest(workspace_path)
        if manifest.get("training_strategy") != self.name:
            raise RuntimeError(f"freeze strategy received {manifest.get('training_strategy')!r}")
        training_cfg = dict(manifest.get("training_config") or {})
        device = resolve_training_device(training_cfg.get("device", "auto"))
        model_name = str(manifest.get("model_name", "") or "")
        if not model_name:
            raise RuntimeError("baseline trigger manifest is missing model_name")
        model = self.model_builder(
            model_name,
            pretrained=True,
            device=device,
            weights_path=str(manifest.get("weights_path", "") or "") or None,
            **model_builder_kwargs(manifest),
        )
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError(f"model_builder returned non-module: {type(model)!r}")
        model.to(device)
        runtime = self.runtime_factory(model, dict(manifest), workspace_path)
        _names, suffix_params = configure_fixed_prefix_training(model, runtime)
        optimizer = build_split_retrain_optimizer(
            model,
            runtime=runtime,
            learning_rate=float(training_cfg.get("learning_rate", 1e-3) or 1e-3),
            optimizer_name=str(training_cfg.get("optimizer_name", "adam") or "adam"),
            weight_decay=float(training_cfg.get("weight_decay", 0.0) or 0.0),
        )
        if optimizer is None:
            optimizer = build_optimizer(
                suffix_params,
                learning_rate=float(training_cfg.get("learning_rate", 1e-3) or 1e-3),
                optimizer_name=str(training_cfg.get("optimizer_name", "adam") or "adam"),
                weight_decay=float(training_cfg.get("weight_decay", 0.0) or 0.0),
            )
        samples = samples_from_baseline_manifest(
            workspace_path,
            manifest,
            teacher=getattr(self.learner, "large_od", None),
            allow_edge_targets=bool(training_cfg.get("allow_edge_targets", False)),
        )
        started = time.perf_counter()
        metrics = run_freeze_training(
            model=model,
            runtime=runtime,
            samples=samples,
            batch_size=int(training_cfg.get("batch_size", 32) or 32),
            epochs=int(training_cfg.get("num_epoch", 50) or 50),
            device=device,
            loss_fn=self.loss_builder(model),
            optimizer=optimizer,
        )
        update_bytes = self.update_serializer(
            model,
            model_name=model_name,
            checkpoint_path=str(workspace_path / "model_update" / "baseline_freeze_state.pt"),
            weights_metadata={
                "protocol_version": str(manifest.get("protocol_version", "")),
                "training_strategy": self.name,
                "source_base_model_version": str(base_model_version or "0"),
                "checkpoint_model_version": str(result_model_version or "1"),
                "baseline_method": str(manifest.get("baseline_method", "")),
                "window_id": str(manifest.get("window_id", "")),
            },
            metadata_path=str(workspace_path / "model_update" / "baseline_freeze_metadata.json"),
        )
        return {
            "success": True,
            "model_data": base64.b64encode(update_bytes).decode("ascii"),
            "message": (
                "[CloudTraining] strategy=freeze "
                f"samples={len(samples)} elapsed={time.perf_counter() - started:.3f}s"
            ),
            "metrics": metrics,
            "result_model_version": str(result_model_version or "1"),
        }


def build_default_torchlens_freeze_runtime(
    model: torch.nn.Module,
    manifest: dict[str, Any],
    workspace: Path,
) -> UniversalModelSplitter:
    training_cfg = dict(manifest.get("training_config") or {})
    device = resolve_training_device(training_cfg.get("device", "auto"))
    model.to(device)
    sample_input = _trace_sample_input(
        model,
        manifest,
        workspace,
        device=device,
    )
    split_plan = _load_cloud_freeze_split_plan(workspace, manifest)
    if split_plan is not None:
        return _build_freeze_runtime_from_split_plan(
            model,
            sample_input,
            manifest,
            split_plan,
            device=device,
            training_cfg=training_cfg,
        )
    boundary = _resolve_cloud_freeze_boundary(
        workspace,
        manifest=manifest,
        training_cfg=training_cfg,
    )
    mode = str(
        training_cfg.get("torchlens_mode")
        or manifest.get("torchlens_mode")
        or "generated_eager"
    )
    dynamic_batch_max = max(1, int(training_cfg.get("dynamic_batch_max", 64) or 64))
    return UniversalModelSplitter(device=device).trace(
        model,
        sample_input,
        boundary=boundary,
        mode=mode,
        model_name=str(manifest.get("model_name", "") or type(model).__name__),
        dynamic_batch_max=dynamic_batch_max,
    )


def _build_freeze_runtime_from_split_plan(
    model: torch.nn.Module,
    sample_input: Any,
    manifest: Mapping[str, Any],
    plan: SplitPlan,
    *,
    device: torch.device,
    training_cfg: Mapping[str, Any],
) -> UniversalModelSplitter:
    boundary = str(plan.logical_split_id or plan.canonical_split_key).strip()
    if not boundary:
        raise RuntimeError("cloud freeze split plan is missing a logical split boundary")
    mode = str(
        training_cfg.get("torchlens_mode")
        or manifest.get("torchlens_mode")
        or dict(plan.runtime_contract or {}).get("mode")
        or "generated_eager"
    )
    trace_batch_size = _first_tensor_batch_size(sample_input) or 1
    spec = make_split_spec(
        boundary,
        dynamic_batch=tuple(plan.dynamic_batch) if plan.dynamic_batch else (1, 64),
        trainable=True,
        trace_batch_mode=(
            str(plan.trace_batch_mode or "")
            or ("batch_gt1" if trace_batch_size > 1 else "batch_1")
        ),
        mode=mode,
    )
    runtime = prepare_exact_split_runtime(
        model,
        sample_input,
        spec,
        mode=spec.mode,
        expected_boundary_tensor_labels=plan.boundary_tensor_labels,
    )
    splitter = UniversalModelSplitter(device=device)
    splitter.bind_runtime(runtime, model=model, split_spec=spec)
    return splitter


def _load_cloud_freeze_split_plan(
    workspace: Path,
    manifest: Mapping[str, Any],
) -> SplitPlan | None:
    for key in ("split_plan", "fixed_split_plan"):
        value = manifest.get(key)
        if not isinstance(value, Mapping):
            continue
        plan = _split_plan_from_payload(value)
        if plan is not None:
            return plan
    for filename in ("fixed_split_plan.json", "split_plan.json"):
        path = workspace / filename
        if not path.exists():
            continue
        try:
            plan = load_split_plan(str(path))
        except (OSError, KeyError, TypeError, ValueError):
            continue
        if _usable_split_plan(plan):
            return plan
    return None


def _split_plan_from_payload(payload: Mapping[str, Any]) -> SplitPlan | None:
    try:
        plan = SplitPlan.from_dict(dict(payload))
    except (KeyError, TypeError, ValueError):
        return None
    return plan if _usable_split_plan(plan) else None


def _usable_split_plan(plan: SplitPlan | None) -> bool:
    return plan is not None and str(plan.plan_version or "") == FIXED_SPLIT_PLAN_VERSION


def _resolve_cloud_freeze_boundary(
    workspace: Path,
    *,
    manifest: Mapping[str, Any],
    training_cfg: Mapping[str, Any],
) -> str:
    configured = str(
        training_cfg.get("split_boundary") or manifest.get("split_boundary") or ""
    ).strip()
    if configured:
        return configured
    for payload in _candidate_split_payloads(workspace, manifest):
        boundary = _boundary_from_split_payload(payload)
        if boundary:
            return boundary
    return "auto"


def _candidate_split_payloads(workspace: Path, manifest: Mapping[str, Any]):
    for key in ("split_plan", "fixed_split_plan"):
        value = manifest.get(key)
        if isinstance(value, Mapping):
            yield dict(value)
    contract = manifest.get("runtime_contract")
    if isinstance(contract, Mapping):
        yield {"runtime_contract": dict(contract)}
    for filename in (
        "fixed_split_plan.json",
        "split_plan.json",
        "split_contract.json",
        "split_runtime_contract.json",
    ):
        path = workspace / filename
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping):
            yield dict(payload)


def _boundary_from_split_payload(payload: Mapping[str, Any]) -> str:
    runtime_contract = payload.get("runtime_contract")
    if isinstance(runtime_contract, Mapping):
        boundary = str(runtime_contract.get("logical_split_id") or "").strip()
        if boundary:
            return boundary
    for key in (
        "logical_split_id",
        "canonical_split_key",
        "edge_split_id",
        "candidate_id",
        "split_label",
    ):
        boundary = str(payload.get(key) or "").strip()
        if boundary:
            return boundary
    return ""


def _first_tensor_batch_size(value: object) -> int | None:
    if isinstance(value, torch.Tensor) and value.ndim > 0:
        return int(value.shape[0])
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor_batch_size(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor_batch_size(item)
            if found is not None:
                return found
    return None


def _trace_sample_input(
    model: torch.nn.Module,
    manifest: dict[str, Any],
    workspace: Path,
    *,
    device: torch.device,
) -> Any:
    for item in list(manifest.get("frames") or []):
        if not isinstance(item, dict):
            continue
        image_path = workspace / str(item.get("image_path", "") or "")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is not None:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return (
                torch.from_numpy(rgb)
                .permute(2, 0, 1)
                .float()
                .div(255.0)
                .unsqueeze(0)
                .to(device)
            )
    image_size = _trace_image_size(manifest)
    try:
        return build_split_runtime_sample_input(
            model,
            image_size=image_size,
            device=device,
        )
    except Exception:
        height, width = image_size
        return torch.zeros((1, 3, height, width), dtype=torch.float32, device=device)


def _trace_image_size(manifest: dict[str, Any]) -> tuple[int, int]:
    training_cfg = dict(manifest.get("training_config") or {})
    value = (
        training_cfg.get("trace_image_size")
        or manifest.get("trace_image_size")
        or manifest.get("tinynext_input_size")
        or 224
    )
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return max(1, int(value[0])), max(1, int(value[1]))
    size = max(1, int(value))
    return size, size
