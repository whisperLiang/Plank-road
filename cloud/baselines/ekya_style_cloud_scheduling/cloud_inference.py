from __future__ import annotations

import copy
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from cloud.baselines.ekya_style_cloud_scheduling.config import (
    EkyaStyleCloudSchedulingConfig,
)
from cloud.baselines.ekya_style_cloud_scheduling.protocol import (
    DetectionResultPacket,
    FrameUploadPacket,
    now_s,
)


class CloudInferenceEngine:
    def __init__(
        self,
        config: EkyaStyleCloudSchedulingConfig,
        *,
        detector: Any | None = None,
        runtime_config: object | None = None,
    ) -> None:
        self.config = config
        self._detector = detector
        self._runtime_config = runtime_config
        self._lock = threading.RLock()
        self._model_version = "0"

    @property
    def model_version(self) -> str:
        return self._model_version

    def infer(
        self,
        *,
        packet: FrameUploadPacket,
        frame_bgr: np.ndarray | None,
        timestamp_cloud_receive: float,
    ) -> DetectionResultPacket:
        detector = self._ensure_detector()
        timestamp_inference_start = now_s()
        if frame_bgr is None:
            boxes, labels, scores = [], [], []
        else:
            with self._lock:
                boxes, labels, scores = _infer_detector(
                    detector,
                    frame_bgr,
                    threshold=float(self.config.cloud_inference.score_threshold),
                )
        timestamp_inference_end = now_s()
        class_names = _class_names_for_labels(self.config.class_names, labels)
        return DetectionResultPacket(
            method=packet.method,
            run_id=packet.run_id,
            edge_id=int(packet.edge_id),
            camera_id=int(packet.camera_id),
            task_id=int(packet.task_id),
            chunk_id=int(packet.chunk_id),
            frame_idx=int(packet.frame_idx),
            video_name=packet.video_name,
            timestamp_edge_capture=float(packet.timestamp_edge_capture),
            timestamp_edge_send=float(packet.timestamp_edge_send),
            timestamp_cloud_receive=float(timestamp_cloud_receive),
            timestamp_inference_start=float(timestamp_inference_start),
            timestamp_inference_end=float(timestamp_inference_end),
            timestamp_cloud_send=now_s(),
            image_shape=tuple(packet.image_shape),
            boxes_xyxy=[list(map(float, box)) for box in list(boxes or [])],
            labels=[int(label) for label in list(labels or [])],
            scores=[float(score) for score in list(scores or [])],
            class_names=class_names,
            model_version=self._model_version,
            encoded_frame_jpeg=None,
        )

    def adopt_checkpoint(self, checkpoint_path: str, *, model_version: str) -> None:
        if not checkpoint_path:
            return
        detector = self._ensure_detector()
        model = getattr(detector, "model", detector)
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError("Ekya checkpoint adoption requires a torch.nn.Module detector")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict) or not state:
            raise RuntimeError("Ekya checkpoint is missing trained model weights")
        with self._lock:
            model.load_state_dict(state, strict=False)
            model.eval()
            self._model_version = str(model_version)

    def export_state_dict(self) -> dict[str, torch.Tensor]:
        detector = self._ensure_detector()
        model = getattr(detector, "model", detector)
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError("Ekya model export requires a torch.nn.Module detector")
        with self._lock:
            state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
                if torch.is_tensor(value)
            }
        if not state:
            raise RuntimeError("Ekya model export produced no model weights")
        return state

    def build_student_model_clone(self) -> torch.nn.Module:
        if self._runtime_config is not None:
            from model_management.model_zoo import build_detection_model

            od_config = self._object_detection_config(self._runtime_config)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = build_detection_model(
                str(getattr(od_config, "lightweight", None) or self.config.student_model),
                pretrained=True,
                device=device,
                weights_path=getattr(od_config, "weights_path", None),
            )
            model.to(device)
            model.eval()
            return model
        detector = self._ensure_detector()
        model = getattr(detector, "model", detector)
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError("Ekya model clone requires a torch.nn.Module detector")
        with self._lock:
            clone = copy.deepcopy(model)
        clone.eval()
        return clone

    def _ensure_detector(self) -> Any:
        if self._detector is not None:
            return self._detector
        from model_management.object_detection import Object_Detection

        runtime_config = self._runtime_config
        if runtime_config is None:
            raise RuntimeError("runtime_config is required to build cloud inference detector")
        self._detector = Object_Detection(
            self._object_detection_config(runtime_config),
            type="small inference",
        )
        return self._detector

    def _object_detection_config(self, runtime_config: object) -> object:
        if hasattr(runtime_config, "lightweight"):
            return runtime_config
        return SimpleNamespace(
            lightweight=(
                getattr(runtime_config, "edge_model_name", None)
                or getattr(runtime_config, "lightweight", None)
                or self.config.student_model
            ),
            golden=getattr(runtime_config, "golden", None) or self.config.teacher_model,
            weights_path=getattr(runtime_config, "weights_path", None),
            tinynext_input_size=getattr(runtime_config, "tinynext_input_size", None),
        )


def _infer_detector(detector: Any, frame_bgr: np.ndarray, *, threshold: float):
    if hasattr(detector, "small_inference"):
        _unused, boxes, labels, scores = detector.small_inference(frame_bgr)
        return boxes or [], labels or [], scores or []
    if hasattr(detector, "infer_sample"):
        artifacts = detector.infer_sample(frame_bgr)
        return (
            getattr(artifacts, "final_detection_boxes", []) or [],
            getattr(artifacts, "final_detection_labels", []) or [],
            getattr(artifacts, "final_detection_scores", []) or [],
        )
    if callable(detector):
        value = detector(frame_bgr, threshold=threshold)
        if isinstance(value, tuple) and len(value) >= 3:
            return value[0], value[1], value[2]
    raise RuntimeError(f"unsupported cloud inference detector: {type(detector)!r}")


def _class_names_for_labels(configured: tuple[str, ...], labels: list[int]) -> list[str]:
    names = []
    for label in labels:
        index = int(label)
        if 0 <= index < len(configured):
            names.append(str(configured[index]))
        else:
            names.append(f"class_{index}")
    return names
