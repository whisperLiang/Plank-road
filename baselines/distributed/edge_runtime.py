from __future__ import annotations

import base64
import io
import threading
import zipfile
from pathlib import Path
from typing import Any

import cv2
import grpc
import torch
from loguru import logger

from baselines.distributed.messages import BaselineFramePayload, json_dumps, json_loads, now_ms
from baselines.distributed.metrics import DistributedMetricsWriter
from baselines.method_factory import create_policy
from baselines.training import (
    BASELINE_FROZEN_RATIO_TRAINING_STRATEGY,
    BaselineFrozenRatioTrainer,
    build_baseline_training_bundle,
)
from config.baseline import default_run_id, validate_baseline_method
from edge.diff import DiffProcessor
from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc
from model_management.model_delta_payload import require_state_dict_delta_payload
from model_management.object_detection import Object_Detection
from tools.grpc_options import grpc_message_options
from tools.video_processor import VideoProcessor


class BaselineGrpcTransport:
    def __init__(self, server_ip: str) -> None:
        self.channel = grpc.insecure_channel(str(server_ip), options=grpc_message_options())
        self.stub = message_transmission_pb2_grpc.MessageTransmissionStub(self.channel)

    def close(self) -> None:
        self.channel.close()

    def register_edge(self, *, payload: BaselineFramePayload) -> None:
        reply = self.stub.RegisterEdge(
            message_transmission_pb2.BaselineRegisterEdgeRequest(
                run_id=payload.run_id,
                baseline_method=payload.baseline_method,
                edge_id=payload.edge_id,
                model_name=payload.model_name,
                model_version=payload.model_version,
                video_source=payload.video_source,
                timestamp_ms=payload.timestamp_ms,
            )
        )
        if not bool(reply.success):
            raise RuntimeError(reply.message)

    def upload_frame(self, payload: BaselineFramePayload) -> None:
        request = _frame_payload_to_proto(payload)
        if payload.baseline_method == "accuracy_trigger_cloud_retraining":
            reply = self.stub.UploadKeyFrame(request)
        else:
            reply = self.stub.UploadFrame(request)
        if not bool(reply.success):
            raise RuntimeError(reply.message)

    def request_cloud_inference(self, payload: BaselineFramePayload) -> dict[str, Any]:
        reply = self.stub.RequestCloudInference(
            message_transmission_pb2.BaselineInferenceRequest(
                run_id=payload.run_id,
                baseline_method=payload.baseline_method,
                edge_id=payload.edge_id,
                frame_id=payload.frame_id,
            )
        )
        if not bool(reply.success):
            raise RuntimeError(reply.message)
        return {
            "frame_id": int(reply.frame_id),
            "cloud_prediction": json_loads(reply.cloud_prediction_json),
            "confidence": float(reply.confidence),
            "message": reply.message,
            "success": bool(reply.success),
        }

    def request_training(
        self,
        *,
        payload: BaselineFramePayload,
        frame_ids: list[int],
        training_config: dict[str, Any],
    ) -> dict[str, Any]:
        reply = self.stub.RequestTraining(
            message_transmission_pb2.BaselineTrainingRequest(
                run_id=payload.run_id,
                baseline_method=payload.baseline_method,
                edge_id=payload.edge_id,
                training_strategy=BASELINE_FROZEN_RATIO_TRAINING_STRATEGY,
                frame_ids=[int(value) for value in frame_ids],
                payload_json=json_dumps(
                    {
                        "frame_ids": [int(value) for value in frame_ids],
                        "training_config": dict(training_config),
                    }
                ),
            )
        )
        if not bool(reply.accepted):
            raise RuntimeError(reply.message)
        return {
            "job_id": reply.job_id,
            "status": reply.status,
            "queue_position": int(reply.queue_position),
            "message": reply.message,
            "protocol_version": reply.protocol_version,
            "result_model_version": reply.result_model_version,
        }

    def poll_training_job(self, payload: BaselineFramePayload, job_id: str) -> dict[str, Any]:
        reply = self.stub.PollTrainingJob(
            message_transmission_pb2.BaselineTrainingStatusRequest(
                run_id=payload.run_id,
                baseline_method=payload.baseline_method,
                edge_id=payload.edge_id,
                job_id=str(job_id),
            )
        )
        return {
            "found": bool(reply.found),
            "job_id": reply.job_id,
            "status": reply.status,
            "message": reply.message,
            "result_available": bool(reply.result_available),
            "queue_position": int(reply.queue_position),
            "result_model_version": reply.result_model_version,
        }

    def download_model_update(self, payload: BaselineFramePayload, job_id: str) -> dict[str, Any]:
        reply = self.stub.DownloadModelUpdate(
            message_transmission_pb2.BaselineModelUpdateRequest(
                run_id=payload.run_id,
                baseline_method=payload.baseline_method,
                edge_id=payload.edge_id,
                job_id=str(job_id),
            )
        )
        if not bool(reply.success):
            raise RuntimeError(reply.message)
        return {
            "job_id": reply.job_id,
            "status": reply.status,
            "model_data": reply.model_data,
            "message": reply.message,
            "model_version": reply.model_version or reply.result_model_version,
            "protocol_version": reply.protocol_version,
            "result_model_version": reply.result_model_version,
        }


class BaselineEdgeRuntime:
    def __init__(
        self,
        *,
        config: object,
        baseline_method: str,
        run_id: str | None,
        edge_id: int,
        server_ip: str = "",
        cache_path: str = "./cache",
        video_path: str = "",
        transport: object | None = None,
        edge_detector: object | None = None,
    ) -> None:
        self.config = config
        self.baseline_method = validate_baseline_method(baseline_method)
        self.run_id = str(run_id or default_run_id(self.baseline_method))
        self.edge_id = int(edge_id)
        self.server_ip = str(server_ip or "")
        self.cache_path = str(cache_path or "./cache")
        self.video_path = str(video_path or getattr(config.source, "video_path", ""))
        baseline_cfg = getattr(config, "baseline", None)
        self.results_root = str(
            getattr(baseline_cfg, "results_root", "results/baselines_distributed")
        )
        method_cfg = getattr(baseline_cfg, self.baseline_method, None)
        self.policy = create_policy(self.baseline_method, method_cfg)
        self.metrics = DistributedMetricsWriter(
            results_root=self.results_root,
            run_id=self.run_id,
            baseline_method=self.baseline_method,
            edge_id=self.edge_id,
        )
        self.transport = transport
        self.edge_detector = edge_detector
        self.model_version = "0"
        self._training_lock = threading.Lock()
        self._training_buffer: list[dict[str, Any]] = []
        self._active_training_job: dict[str, Any] | None = None
        self._local_training_thread: threading.Thread | None = None
        self._local_training_result: dict[str, Any] | None = None
        self._local_training_error: str = ""
        self._training_config = _training_config_dict(getattr(baseline_cfg, "training", None))
        self._training_window_size = max(
            1,
            int(self._training_config.get("training_window_size", 8) or 8),
        )
        self._min_training_samples = max(
            1,
            int(self._training_config.get("min_training_samples", 1) or 1),
        )
        if self.transport is None and self.policy.requires_cloud:
            self.transport = BaselineGrpcTransport(self.server_ip)

    @property
    def metrics_path(self) -> Path:
        return self.metrics.path

    def close(self) -> None:
        if self.transport is not None and hasattr(self.transport, "close"):
            self.transport.close()

    def is_keyframe(self, frame: object, frame_id: int, state: dict[str, Any]) -> bool:
        if not self.policy.frame_filter_enabled:
            return True
        if not bool(getattr(self.config, "diff_flag", True)):
            return True
        feature_name = str(getattr(self.config, "feature", "edge"))
        processor = state.get("diff_processor")
        if processor is None:
            processor = DiffProcessor.str_to_class(feature_name)()
            state["diff_processor"] = processor
        current = processor.get_frame_feature(frame)
        previous = state.get("previous_feature")
        state["previous_feature"] = current
        if previous is None:
            return True
        diff_value = processor.cal_frame_diff(current, previous)
        accumulated = float(state.get("diff", 0.0)) + float(diff_value)
        if accumulated >= float(getattr(self.config, "diff_thresh", 0.0004)):
            state["diff"] = 0.0
            return True
        state["diff"] = accumulated
        logger.debug("baseline frame {} filtered by diff={}", frame_id, accumulated)
        return False

    def process_frame(
        self,
        *,
        frame: object | None,
        frame_id: int,
        is_keyframe: bool,
        edge_prediction: dict[str, Any] | None = None,
    ) -> BaselineFramePayload | None:
        decision = self.policy.decide_frame(frame_id=frame_id, is_keyframe=is_keyframe)
        if edge_prediction is None and decision.upload_prediction:
            edge_prediction = self._edge_prediction_for_frame(frame)
        if (
            edge_prediction is None
            and self.baseline_method == "pure_edge_local_updating"
            and frame is not None
        ):
            edge_prediction = self._edge_prediction_for_frame(frame)
        edge_prediction_data = dict(edge_prediction or {})
        local_raw_frame = (
            _encode_frame(frame)
            if decision.upload_frame or self.baseline_method == "pure_edge_local_updating"
            else b""
        )
        payload = BaselineFramePayload(
            run_id=self.run_id,
            baseline_method=self.baseline_method,
            edge_id=self.edge_id,
            frame_id=int(frame_id),
            timestamp_ms=now_ms(),
            model_name=str(getattr(self.config, "lightweight", "")),
            model_version=str(self.model_version),
            video_source=self.video_path,
            upload_mode=decision.upload_mode,
            is_keyframe=decision.is_keyframe,
            edge_prediction=edge_prediction_data,
            confidence=_safe_float(edge_prediction_data.get("confidence", 0.0)),
            entropy=_safe_float(edge_prediction_data.get("entropy", 0.0)),
            quality_metadata={
                "decision_reason": decision.reason,
                "training_strategy": decision.training_strategy,
                **decision.metadata,
            },
            raw_frame=local_raw_frame if decision.upload_frame else b"",
        )
        self.metrics.record(
            "frame_decision",
            frame_id=int(frame_id),
            upload_frame=decision.upload_frame,
            is_keyframe=decision.is_keyframe,
            upload_mode=decision.upload_mode,
            training_strategy=decision.training_strategy,
        )
        if decision.upload_frame and self.transport is not None:
            if hasattr(self.transport, "register_edge"):
                self.transport.register_edge(payload=payload)
            self.transport.upload_frame(payload)
            if decision.request_cloud_inference and hasattr(
                self.transport,
                "request_cloud_inference",
            ):
                cloud_result = self.transport.request_cloud_inference(payload)
                self.metrics.record(
                    "cloud_inference_result",
                    frame_id=int(frame_id),
                    result=cloud_result,
                )
        self._record_training_sample(payload, raw_frame=local_raw_frame)
        self._poll_active_training(payload)
        self._maybe_start_training(payload)
        return payload if decision.upload_frame else None

    def _edge_prediction_for_frame(self, frame: object | None) -> dict[str, Any]:
        if self.baseline_method not in {
            "accuracy_trigger_cloud_retraining",
            "pure_edge_local_updating",
        } or frame is None:
            return {}
        if self.edge_detector is None:
            self.edge_detector = Object_Detection(self.config, type="small inference")
        infer_sample = getattr(self.edge_detector, "infer_sample", None)
        if infer_sample is None:
            return {}
        artifacts = infer_sample(frame)
        if isinstance(artifacts, dict):
            return dict(artifacts)
        scores = _jsonable_list(getattr(artifacts, "final_detection_scores", []) or [])
        entropy = getattr(artifacts, "logit_entropy", None)
        if entropy is None:
            entropy = getattr(artifacts, "feature_spectral_entropy", 0.0)
        return {
            "boxes": _jsonable_list(getattr(artifacts, "final_detection_boxes", []) or []),
            "labels": _jsonable_list(getattr(artifacts, "final_detection_labels", []) or []),
            "scores": scores,
            "confidence": _safe_float(getattr(artifacts, "confidence", 0.0)),
            "entropy": _safe_float(entropy),
        }

    def _record_training_sample(
        self,
        payload: BaselineFramePayload,
        *,
        raw_frame: bytes,
    ) -> None:
        if not raw_frame:
            return
        prediction = (
            payload.teacher_prediction
            or payload.cloud_prediction
            or payload.edge_prediction
        )
        if (
            self.baseline_method == "pure_edge_local_updating"
            and not prediction.get("boxes")
            and not prediction.get("labels")
        ):
            return
        with self._training_lock:
            self._training_buffer.append(
                {
                    "frame_id": int(payload.frame_id),
                    "raw_frame": bytes(raw_frame),
                    "teacher_prediction": dict(payload.teacher_prediction),
                    "cloud_prediction": dict(payload.cloud_prediction),
                    "edge_prediction": dict(payload.edge_prediction),
                    "quality_metadata": dict(payload.quality_metadata),
                }
            )
            self._training_buffer = self._training_buffer[-self._training_window_size :]

    def _maybe_start_training(self, payload: BaselineFramePayload) -> None:
        with self._training_lock:
            if len(self._training_buffer) < self._min_training_samples:
                return
            if self._active_training_job is not None:
                return
            if (
                self.baseline_method == "pure_edge_local_updating"
                and self._local_training_thread is not None
                and self._local_training_thread.is_alive()
            ):
                return
            frame_ids = [int(item["frame_id"]) for item in self._training_buffer]
        if self.baseline_method == "pure_edge_local_updating":
            self._start_local_training(payload)
            return
        if self.transport is None or not hasattr(self.transport, "request_training"):
            return
        try:
            result = self.transport.request_training(
                payload=payload,
                frame_ids=frame_ids,
                training_config=self._training_config,
            )
        except Exception as exc:
            logger.warning("baseline cloud training request failed: {}", exc)
            self.metrics.record(
                "training_request_failed",
                frame_id=int(payload.frame_id),
                message=str(exc),
            )
            return
        with self._training_lock:
            self._active_training_job = {
                "job_id": str(result["job_id"]),
                "submitted_model_version": str(self.model_version),
                "payload": payload,
            }
        self.metrics.record(
            "training_job_submitted",
            frame_id=int(payload.frame_id),
            job_id=str(result["job_id"]),
            status=str(result.get("status", "")),
            queue_position=int(result.get("queue_position", -1)),
        )

    def _poll_active_training(self, payload: BaselineFramePayload) -> None:
        if self.baseline_method == "pure_edge_local_updating":
            self._collect_local_training_result()
            return
        with self._training_lock:
            active = dict(self._active_training_job or {})
        if not active or self.transport is None or not hasattr(self.transport, "poll_training_job"):
            return
        job_id = str(active.get("job_id", ""))
        if not job_id:
            return
        try:
            status = self.transport.poll_training_job(payload, job_id)
        except Exception as exc:
            logger.warning("baseline training status poll failed: {}", exc)
            return
        status_text = str(status.get("status", "")).upper()
        if status_text not in {"SUCCEEDED", "FAILED", "STALE", "CANCELLED"}:
            return
        try:
            if status_text == "SUCCEEDED" and bool(status.get("result_available", False)):
                update = self.transport.download_model_update(payload, job_id)
                self._apply_model_update(
                    update.get("model_data", ""),
                    result_model_version=str(
                        update.get("result_model_version")
                        or update.get("model_version")
                        or ""
                    ),
                )
                self.metrics.record(
                    "training_model_update_applied",
                    frame_id=int(payload.frame_id),
                    job_id=job_id,
                    result_model_version=str(self.model_version),
                )
            else:
                self.metrics.record(
                    "training_job_terminal",
                    frame_id=int(payload.frame_id),
                    job_id=job_id,
                    status=status_text,
                    message=str(status.get("message", "")),
                )
        finally:
            with self._training_lock:
                self._active_training_job = None

    def _start_local_training(self, payload: BaselineFramePayload) -> None:
        with self._training_lock:
            frames = list(self._training_buffer)
        thread = threading.Thread(
            target=self._run_local_training,
            args=(payload, frames, str(self.model_version)),
            name=f"baseline-local-training-edge-{self.edge_id}",
            daemon=True,
        )
        with self._training_lock:
            self._local_training_thread = thread
            self._local_training_result = None
            self._local_training_error = ""
        thread.start()
        self.metrics.record(
            "local_training_started",
            frame_id=int(payload.frame_id),
            samples=len(frames),
        )

    def _run_local_training(
        self,
        payload: BaselineFramePayload,
        frames: list[dict[str, Any]],
        base_model_version: str,
    ) -> None:
        workspace = Path(self.cache_path) / "baseline_local_training"
        workspace.mkdir(parents=True, exist_ok=True)
        try:
            bundle = build_baseline_training_bundle(
                run_id=payload.run_id,
                baseline_method=payload.baseline_method,
                edge_id=payload.edge_id,
                model_name=payload.model_name,
                model_version=base_model_version,
                frames=frames,
                training_config=self._training_config,
                window_id=f"local-{payload.frame_id}",
                weights_path=str(getattr(self.config, "weights_path", "") or ""),
                tinynext_input_size=getattr(self.config, "tinynext_input_size", None),
            )
            _extract_bundle(bundle, workspace)
            trainer = BaselineFrozenRatioTrainer(config=self._training_config)
            result = trainer.train_from_workspace(
                workspace,
                base_model_version=base_model_version,
            )
            with self._training_lock:
                self._local_training_result = result
        except Exception as exc:
            with self._training_lock:
                self._local_training_error = str(exc)

    def _collect_local_training_result(self) -> None:
        with self._training_lock:
            thread = self._local_training_thread
            if thread is not None and thread.is_alive():
                return
            result = self._local_training_result
            error = self._local_training_error
            self._local_training_thread = None
            self._local_training_result = None
            self._local_training_error = ""
        if error:
            logger.warning("pure-edge baseline local training failed: {}", error)
            self.metrics.record("local_training_failed", message=error)
            return
        if result:
            self._apply_model_update(
                str(result.get("model_data", "")),
                result_model_version=str(result.get("result_model_version", "")),
            )
            self.metrics.record(
                "local_training_model_update_applied",
                result_model_version=str(self.model_version),
            )

    def _apply_model_update(self, model_data: str, *, result_model_version: str = "") -> None:
        if not model_data:
            raise RuntimeError("baseline model update payload is empty")
        if self.edge_detector is None:
            self.edge_detector = Object_Detection(self.config, type="small inference")
        update_payload = require_state_dict_delta_payload(
            torch.load(
                io.BytesIO(base64.b64decode(model_data)),
                map_location="cpu",
                weights_only=False,
            )
        )
        state_dict = dict(update_payload["state_dict"])
        with self.edge_detector.model_lock:
            self.edge_detector.model.load_state_dict(state_dict, strict=False)
            self.edge_detector.model.eval()
            self.edge_detector.refresh_thresholds_from_model()
        previous_version = str(self.model_version or "0")
        self.model_version = str(result_model_version or update_payload.get("result_model_version") or "")
        if not self.model_version:
            try:
                self.model_version = str(int(previous_version) + 1)
            except (TypeError, ValueError):
                self.model_version = "1"

    def run(self) -> None:
        frame_state: dict[str, Any] = {}
        with VideoProcessor(self.config.source) as video:
            frame_id = 0
            while True:
                frame = next(video)
                if frame is None:
                    break
                keyframe = self.is_keyframe(frame, frame_id, frame_state)
                self.process_frame(frame=frame, frame_id=frame_id, is_keyframe=keyframe)
                frame_id += 1


def _encode_frame(frame: object | None) -> bytes:
    if frame is None:
        return b""
    ok, encoded = cv2.imencode(".jpg", frame)
    return bytes(encoded.tobytes()) if ok else b""


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _jsonable_list(value: object) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return [value]
    return [_jsonable_item(item) for item in value]


def _jsonable_item(value: object):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable_item(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_item(item) for key, item in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def _training_config_dict(config: object | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in (
        "trainable_param_ratio",
        "freeze_order",
        "batch_size",
        "num_epoch",
        "learning_rate",
        "optimizer_name",
        "weight_decay",
        "min_training_samples",
        "training_window_size",
        "microprofile_epochs",
        "microprofile_max_samples",
        "device",
    ):
        if isinstance(config, dict):
            if name in config:
                result[name] = config[name]
        elif config is not None and hasattr(config, name):
            result[name] = getattr(config, name)
    result.setdefault("trainable_param_ratio", 0.3)
    result.setdefault("freeze_order", "forward_module_order")
    result.setdefault("batch_size", 32)
    result.setdefault("num_epoch", 50)
    result.setdefault("learning_rate", 1e-3)
    return result


def _extract_bundle(bundle: bytes, workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(bundle), "r") as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"unsafe baseline bundle member: {member.filename}")
            target = workspace / member_path
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))


def _frame_payload_to_proto(payload: BaselineFramePayload):
    return message_transmission_pb2.BaselineFrameRequest(
        run_id=payload.run_id,
        baseline_method=payload.baseline_method,
        edge_id=payload.edge_id,
        frame_id=payload.frame_id,
        timestamp_ms=payload.timestamp_ms,
        model_name=payload.model_name,
        model_version=payload.model_version,
        video_source=payload.video_source,
        upload_mode=payload.upload_mode,
        is_keyframe=payload.is_keyframe,
        edge_prediction_json=json_dumps(payload.edge_prediction),
        cloud_prediction_json=json_dumps(payload.cloud_prediction),
        teacher_prediction_json=json_dumps(payload.teacher_prediction),
        confidence=payload.confidence,
        entropy=payload.entropy,
        quality_metadata_json=json_dumps(payload.quality_metadata),
        raw_frame=payload.raw_frame,
        raw_frame_ref=payload.raw_frame_ref,
        feature_ref_json=json_dumps(payload.feature_ref),
        metrics_ref=payload.metrics_ref,
        job_id=payload.job_id,
    )
