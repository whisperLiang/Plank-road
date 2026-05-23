"""Student detector adapter for real baseline execution."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from model_management.model_zoo import build_detection_model, get_model_family
from model_management.split_model_adapters import (
    get_split_runtime_input_resize_mode,
    get_split_runtime_model,
    prepare_split_runtime_input,
)
from model_management.fixed_split import (
    SplitConstraints,
    SplitPlan,
    apply_split_plan,
    load_or_compute_fixed_split_plan,
)
from model_management.universal_model_split import (
    UniversalModelSplitter,
    save_split_feature_cache,
    slice_boundary_payload_batch,
)


def resolve_torch_device(device: str) -> torch.device:
    requested = str(device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


@dataclass(frozen=True)
class StudentInferenceOutput:
    prediction_path: str
    confidence: float
    latency_ms: float
    num_detections: int
    feature_tensor_path: str | None = None


class StudentInferencer:
    """Real student detector inference wrapper with checkpoint load/save support."""

    MODEL_ALIASES = {"yolo26": "yolo26n"}

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        results_dir: str | Path,
        method_name: str,
        cache_features: bool = False,
        pretrained: bool = True,
        weights_path: str | Path | None = None,
        seed: int = 2026,
        fixed_split_constraints: SplitConstraints | None = None,
        fixed_split_cache_path: str | Path | None = None,
        fixed_split_validate_cached_plan: bool = True,
        feature_trace_batch_size: int = 1,
    ) -> None:
        normalized = self.MODEL_ALIASES.get(model_name.lower().replace("-", "_"), model_name)
        torch.manual_seed(int(seed))
        self.model_name = str(normalized).lower().replace("-", "_")
        self.device = resolve_torch_device(device)
        try:
            self.model = build_detection_model(
                self.model_name,
                pretrained=bool(pretrained),
                device=self.device,
                weights_path=None if weights_path is None else str(weights_path),
            )
        except Exception as exc:
            raise NotImplementedError(
                f"Student model {model_name!r} could not be initialized by the real "
                "detection adapter. Provide a supported model artifact."
            ) from exc
        self.model_family = get_model_family(self.model_name)
        self.model.eval()
        self.results_dir = Path(results_dir)
        self.method_name = method_name
        self.cache_features = bool(cache_features)
        self.feature_trace_batch_size = 1 if int(feature_trace_batch_size) <= 1 else 2
        self.prediction_dir = self.results_dir / "predictions" / method_name
        self.feature_cache_dir = self.results_dir / "feature_cache" / method_name
        self._feature_splitter: UniversalModelSplitter | None = None
        self.fixed_split_constraints = fixed_split_constraints
        self.fixed_split_cache_path = (
            Path(fixed_split_cache_path) if fixed_split_cache_path is not None else None
        )
        self.fixed_split_validate_cached_plan = bool(fixed_split_validate_cached_plan)
        self.fixed_split_plan: SplitPlan | None = None

    def infer(self, frame_path: str | Path, *, device_id: int, frame_index: int) -> StudentInferenceOutput:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise FileNotFoundError(f"Unable to read frame image: {frame_path}")
        tensor = self._bgr_to_detection_tensor(frame)
        start = time.perf_counter()
        # The same model instance is later reused for training/split tracing.
        # no_grad avoids autograd work without leaving inference-mode tensors in
        # Ultralytics head caches that a later backward-capable forward may reuse.
        with torch.no_grad():
            outputs = self.model([tensor])
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - start) * 1000.0

        detections = self._outputs_to_detections(outputs)
        confidence = (
            sum(float(det["score"]) for det in detections) / len(detections)
            if detections
            else 0.0
        )
        pred_path = self.prediction_dir / f"edge_{device_id}" / f"{frame_index:08d}.json"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w", encoding="utf-8") as f:
            json.dump(detections, f)

        feature_tensor_path = None
        if self.cache_features:
            feature_tensor_path = self._cache_split_feature(
                frame,
                device_id=device_id,
                frame_index=frame_index,
            )

        return StudentInferenceOutput(
            prediction_path=str(pred_path),
            confidence=confidence,
            latency_ms=latency_ms,
            num_detections=len(detections),
            feature_tensor_path=feature_tensor_path,
        )

    def _bgr_to_detection_tensor(self, frame: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()
        return tensor.div_(255.0).to(self.device)

    def _ensure_feature_splitter(self, sample_input: torch.Tensor) -> UniversalModelSplitter:
        trace_input = self._prepare_feature_trace_input(sample_input)
        if self._feature_splitter is None:
            self._feature_splitter = UniversalModelSplitter(device=self.device).trace(
                get_split_runtime_model(self.model),
                trace_input,
                model_name=self.model_name,
                model_family=self.model_family,
            )
        self._bind_fixed_split_plan(self._feature_splitter, trace_input)
        return self._feature_splitter

    def _prepare_feature_trace_input(self, sample_input: torch.Tensor) -> torch.Tensor:
        return self._pad_tensor_batch(
            sample_input,
            min_batch_size=self.feature_trace_batch_size,
        )

    def ensure_fixed_split_plan(self, frame_path: str | Path) -> SplitPlan | None:
        if self.fixed_split_constraints is None:
            return None
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise FileNotFoundError(f"Unable to read frame image: {frame_path}")
        sample_input = prepare_split_runtime_input(self.model, frame, device=self.device)
        if not isinstance(sample_input, torch.Tensor):
            raise NotImplementedError(
                f"Fixed split selection for {type(self.model).__name__} requires tensor runtime input"
            )
        self._ensure_feature_splitter(sample_input)
        return self.fixed_split_plan

    def _bind_fixed_split_plan(
        self,
        splitter: UniversalModelSplitter,
        sample_input: torch.Tensor,
    ) -> SplitPlan | None:
        if self.fixed_split_constraints is None:
            return None
        if self.fixed_split_plan is not None:
            self._apply_fixed_split_plan(splitter, self.fixed_split_plan)
            return self.fixed_split_plan
        plan = load_or_compute_fixed_split_plan(
            get_split_runtime_model(self.model),
            self.fixed_split_constraints,
            sample_input=sample_input,
            device=self.device,
            model_name=self.model_name,
            cache_path=(
                None
                if self.fixed_split_cache_path is None
                else str(self.fixed_split_cache_path)
            ),
            splitter=splitter,
            validate_cached_plan=self.fixed_split_validate_cached_plan,
            input_resize_mode=get_split_runtime_input_resize_mode(self.model) or "direct_resize",
            front_version="0",
        )
        self._apply_fixed_split_plan(splitter, plan)
        self.fixed_split_plan = plan
        return plan

    @staticmethod
    def _apply_fixed_split_plan(splitter: UniversalModelSplitter, plan: SplitPlan) -> None:
        current = getattr(splitter, "current_candidate", None)
        current_id = getattr(current, "candidate_id", None)
        if plan.candidate_id is not None and str(current_id) == str(plan.candidate_id):
            return
        apply_split_plan(splitter, plan)

    def _cache_split_feature(self, frame: np.ndarray, *, device_id: int, frame_index: int) -> str:
        sample_input = prepare_split_runtime_input(self.model, frame, device=self.device)
        if not isinstance(sample_input, torch.Tensor):
            raise NotImplementedError(
                f"Split feature caching for {type(self.model).__name__} requires tensor runtime input"
            )
        splitter = self._ensure_feature_splitter(sample_input)
        runtime_input = self._prepare_feature_trace_input(sample_input)
        with torch.no_grad():
            boundary = splitter.run_prefix(runtime_input)
        if int(runtime_input.shape[0]) != int(sample_input.shape[0]):
            boundary = slice_boundary_payload_batch(
                boundary,
                start=0,
                length=int(sample_input.shape[0]),
            )
        cache_root = self.feature_cache_dir / f"edge_{int(device_id)}"
        save_split_feature_cache(
            str(cache_root),
            str(int(frame_index)),
            boundary,
            input_image_size=[int(frame.shape[0]), int(frame.shape[1])],
            input_tensor_shape=[int(dim) for dim in sample_input.shape],
            input_resize_mode=get_split_runtime_input_resize_mode(self.model),
        )
        return str(cache_root / "features" / f"{int(frame_index)}.pt")

    @staticmethod
    def _pad_tensor_batch(tensor: torch.Tensor, *, min_batch_size: int) -> torch.Tensor:
        min_batch_size = max(1, int(min_batch_size))
        batch_size = int(tensor.shape[0]) if tensor.ndim > 0 else 0
        if batch_size >= min_batch_size:
            return tensor
        if batch_size <= 0:
            raise RuntimeError("Split runtime tensor input must include a batch dimension.")
        padding = [tensor[-1:]] * (min_batch_size - batch_size)
        return torch.cat([tensor, *padding], dim=0)

    @staticmethod
    def _outputs_to_detections(outputs: Any) -> list[dict[str, Any]]:
        if isinstance(outputs, dict):
            outputs = [outputs]
        if not isinstance(outputs, (list, tuple)) or not outputs:
            raise RuntimeError("Student detector did not return a detection output list")
        output = outputs[0]
        boxes = output.get("boxes", [])
        scores = output.get("scores", [])
        labels = output.get("labels", [])
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().tolist()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().tolist()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().tolist()
        detections: list[dict[str, Any]] = []
        for box, score, label in zip(boxes, scores, labels):
            detections.append(
                {
                    "bbox": [float(v) for v in box],
                    "score": float(score),
                    "class_id": int(label),
                }
            )
        return detections

    def save_checkpoint(self, checkpoint_path: str | Path) -> str:
        out = Path(checkpoint_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": self.model_name,
                "state_dict": self.model.state_dict(),
            },
            out,
        )
        return str(out)

    def load_checkpoint(self, checkpoint_path: str | Path) -> float:
        start = time.perf_counter()
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self._feature_splitter = None
        return time.perf_counter() - start
