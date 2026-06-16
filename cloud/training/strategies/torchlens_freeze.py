from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import torch

from cloud.model_update import serialize_model_update
from cloud.training.freeze_modes import (
    build_optimizer,
    configure_fixed_prefix_training,
    run_freeze_training,
)
from cloud.training.strategies.raw_freeze import (
    _load_manifest,
    _model_builder_kwargs,
    _resolve_device,
    _samples_from_manifest,
)
from model_management.model_zoo import build_detection_model
from model_management.split_model_adapters import (
    build_split_runtime_sample_input,
    build_split_training_loss,
)
from model_management.universal_model_split import (
    UniversalModelSplitter,
    build_split_retrain_optimizer,
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
        manifest = _load_manifest(workspace_path)
        if manifest.get("training_strategy") != self.name:
            raise RuntimeError(f"freeze strategy received {manifest.get('training_strategy')!r}")
        training_cfg = dict(manifest.get("training_config") or {})
        device = _resolve_device(training_cfg.get("device", "auto"))
        model_name = str(manifest.get("model_name", "") or "")
        if not model_name:
            raise RuntimeError("baseline trigger manifest is missing model_name")
        model = self.model_builder(
            model_name,
            pretrained=True,
            device=device,
            weights_path=str(manifest.get("weights_path", "") or "") or None,
            **_model_builder_kwargs(manifest),
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
        samples = _samples_from_manifest(
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
    device = _resolve_device(training_cfg.get("device", "auto"))
    model.to(device)
    sample_input = _trace_sample_input(
        model,
        manifest,
        workspace,
        device=device,
    )
    boundary = str(
        training_cfg.get("split_boundary") or manifest.get("split_boundary") or "auto"
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
