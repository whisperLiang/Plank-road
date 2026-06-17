from __future__ import annotations

import base64
import io
import json
import time
import zipfile
from typing import Any, Iterable, Mapping

import cv2
import grpc

from baselines.distributed.messages import BaselineFramePayload, json_dumps, json_loads
from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc
from tools.grpc_options import grpc_message_options

BASELINE_TRAINING_PROTOCOL_VERSION = "baseline-training-trigger.v1"
ALLOWED_BASELINE_TRAINING_STRATEGIES = {"freeze"}


def encode_frame(frame: object | None) -> bytes:
    if frame is None:
        return b""
    ok, encoded = cv2.imencode(".jpg", frame)
    return bytes(encoded.tobytes()) if ok else b""


def build_baseline_training_bundle(
    *,
    run_id: str,
    baseline_method: str,
    edge_id: int,
    model_name: str,
    model_version: str,
    training_strategy: str,
    window_id: str,
    samples: Iterable[Mapping[str, Any]],
    training_config: Mapping[str, Any] | None = None,
    weights_path: str = "",
    tinynext_input_size: int | None = None,
    base_model_update_model_data: str = "",
    trigger_metadata: Mapping[str, Any] | None = None,
) -> bytes:
    strategy = validate_baseline_training_strategy(training_strategy)
    frame_entries: list[dict[str, Any]] = []
    edge_predictions: dict[str, dict[str, Any]] = {}
    teacher_predictions: dict[str, dict[str, Any]] = {}
    quality_metadata: dict[str, dict[str, Any]] = {}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in samples:
            frame_id = int(item["frame_id"])
            raw_frame = bytes(item.get("raw_frame", b"") or b"")
            if not raw_frame:
                continue
            frame_name = f"frames/{frame_id}.jpg"
            archive.writestr(frame_name, raw_frame)
            edge_prediction = dict(item.get("edge_prediction") or {})
            teacher_prediction = dict(item.get("teacher_prediction") or {})
            metadata = dict(item.get("quality_metadata") or {})
            edge_predictions[str(frame_id)] = edge_prediction
            if teacher_prediction:
                teacher_predictions[str(frame_id)] = teacher_prediction
            quality_metadata[str(frame_id)] = metadata
            frame_entry = {
                "frame_id": frame_id,
                "image_path": frame_name,
                "is_keyframe": bool(item.get("is_keyframe", False)),
                "edge_prediction": edge_prediction,
                "quality_metadata": metadata,
            }
            if teacher_prediction:
                frame_entry["teacher_prediction"] = teacher_prediction
            frame_entries.append(frame_entry)
        if not frame_entries:
            raise RuntimeError("baseline training bundle contains no raw frames")
        normalized_training_config = dict(training_config or {})
        if str(baseline_method) == "accuracy_trigger_cloud_retraining":
            normalized_training_config.setdefault("trainable_param_ratio", 0.3)
        manifest: dict[str, Any] = {
            "protocol_version": BASELINE_TRAINING_PROTOCOL_VERSION,
            "run_id": str(run_id),
            "baseline_method": str(baseline_method),
            "edge_id": int(edge_id),
            "model_name": str(model_name),
            "model_version": str(model_version or "0"),
            "training_strategy": strategy,
            "window_id": str(window_id),
            "frame_ids": [int(item["frame_id"]) for item in frame_entries],
            "edge_predictions": edge_predictions,
            "teacher_predictions": teacher_predictions,
            "quality_metadata": quality_metadata,
            "weights_path": str(weights_path or ""),
            "training_config": normalized_training_config,
            "frames": frame_entries,
        }
        manifest.update(dict(trigger_metadata or {}))
        if tinynext_input_size is not None and str(model_name).lower().startswith("tinynext"):
            manifest["tinynext_input_size"] = int(tinynext_input_size)
        update_data = str(base_model_update_model_data or "")
        if update_data:
            try:
                archive.writestr("base_model_update.pt", base64.b64decode(update_data))
                manifest["base_model_update_path"] = "base_model_update.pt"
            except Exception:
                manifest["base_model_update_model_data"] = update_data
        archive.writestr(
            "baseline_trigger_manifest.json",
            json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8"),
        )
    return buffer.getvalue()


def validate_baseline_training_strategy(value: object) -> str:
    strategy = str(value or "").strip()
    if strategy not in ALLOWED_BASELINE_TRAINING_STRATEGIES:
        raise ValueError("baseline training_strategy must be freeze")
    return strategy


class BaselineUploadClient:
    def __init__(self, server_ip: str) -> None:
        self.server_ip = str(server_ip)
        self.channel = grpc.insecure_channel(self.server_ip, options=grpc_message_options())
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

    def request_cloud_inference(
        self,
        payload: BaselineFramePayload,
        *,
        timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        reply = self.stub.RequestCloudInference(
            message_transmission_pb2.BaselineInferenceRequest(
                run_id=payload.run_id,
                baseline_method=payload.baseline_method,
                edge_id=payload.edge_id,
                frame_id=payload.frame_id,
            ),
            timeout=float(timeout_sec) if timeout_sec is not None else None,
        )
        if not bool(reply.success):
            raise RuntimeError(reply.message)
        return {
            "frame_id": int(reply.frame_id),
            "cloud_prediction": json_loads(reply.cloud_prediction_json),
            "confidence": float(reply.confidence),
            "message": reply.message,
            "success": bool(reply.success),
            "timestamp_ms": int(reply.timestamp_ms or int(time.time() * 1000)),
        }

    def poll_command(
        self,
        *,
        run_id: str,
        baseline_method: str,
        edge_id: int,
    ) -> list[dict[str, Any]]:
        reply = self.stub.PollCommand(
            message_transmission_pb2.BaselineCommandRequest(
                run_id=str(run_id),
                baseline_method=str(baseline_method),
                edge_id=int(edge_id),
                timestamp_ms=int(time.time() * 1000),
            )
        )
        if not bool(reply.success):
            raise RuntimeError(reply.message)
        commands: list[dict[str, Any]] = []
        for item in list(reply.command_json):
            value = json_loads(item)
            if value:
                commands.append(value)
        return commands

    def ack_command(
        self,
        *,
        run_id: str,
        baseline_method: str,
        edge_id: int,
        command_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        metrics = {"acked_commands": [str(command_id)]}
        metrics.update(dict(metadata or {}))
        reply = self.stub.Heartbeat(
            message_transmission_pb2.BaselineHeartbeatRequest(
                run_id=str(run_id),
                baseline_method=str(baseline_method),
                edge_id=int(edge_id),
                timestamp_ms=int(time.time() * 1000),
                metrics_json=json_dumps(metrics),
            )
        )
        if not bool(reply.success):
            raise RuntimeError(reply.message)

    def get_training_job_status(self, *, edge_id: int, job_id: str):
        return self.stub.get_training_job_status(
            message_transmission_pb2.TrainingJobStatusRequest(
                edge_id=int(edge_id),
                job_id=str(job_id),
            )
        )

    def download_trained_model(self, *, edge_id: int, job_id: str):
        return self.stub.download_trained_model(
            message_transmission_pb2.DownloadTrainedModelRequest(
                edge_id=int(edge_id),
                job_id=str(job_id),
            )
        )

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
