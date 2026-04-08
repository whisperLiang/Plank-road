import json
import os
import uuid

import grpc
from loguru import logger

from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc
from tools.grpc_options import grpc_message_options
import zipfile
import io

from edge.sample_store import EdgeSampleStore, HIGH_CONFIDENCE, LOW_CONFIDENCE
from model_management.fixed_split import SplitPlan
from model_management.continual_learning_bundle import (
    CONTINUAL_LEARNING_PROTOCOL_VERSION,
)


def _server_workspace_hint(edge_id: int, request_kind: str) -> str:
    return f"edge_{int(edge_id)}/{request_kind}"

def pack_training_payload(cache_path, all_frame_indices, drift_frame_indices=None):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        if drift_frame_indices is not None:
            # Split learning mode
            meta_path = os.path.join(cache_path, "features", "split_meta.json")
            if os.path.exists(meta_path):
                zf.write(meta_path, arcname="features/split_meta.json")
            for idx in all_frame_indices:
                feat_path = os.path.join(cache_path, "features", f"{idx}.pt")
                if os.path.exists(feat_path):
                    zf.write(feat_path, arcname=f"features/{idx}.pt")
            for idx in drift_frame_indices:
                frame_path = os.path.join(cache_path, "frames", f"{idx}.jpg")
                if os.path.exists(frame_path):
                    zf.write(frame_path, arcname=f"frames/{idx}.jpg")
        else:
            # Full frame train mode
            for idx in all_frame_indices:
                frame_path = os.path.join(cache_path, "frames", f"{idx}.jpg")
                if os.path.exists(frame_path):
                    zf.write(frame_path, arcname=f"frames/{idx}.jpg")
    return buf.getvalue()


def pack_continual_learning_bundle(
    sample_store: EdgeSampleStore,
    *,
    edge_id: int,
    send_low_conf_features: bool,
    split_plan: SplitPlan,
    model_id: str,
    model_version: str,
) -> tuple[bytes, dict]:
    records = [
        record
        for record in sample_store.list_records()
        if record.split_config_id == split_plan.split_config_id
        and record.model_id == str(model_id)
        and record.model_version == str(model_version)
    ]
    manifest = {
        "protocol_version": CONTINUAL_LEARNING_PROTOCOL_VERSION,
        "edge_id": int(edge_id),
        "model": {
            "model_id": str(model_id),
            "model_version": str(model_version),
        },
        "split_plan": split_plan.to_dict(),
        "training_mode": {
            "send_low_conf_features": bool(send_low_conf_features),
            "low_confidence_mode": "raw+feature" if send_low_conf_features else "raw-only",
        },
        "drift_sample_ids": [record.sample_id for record in records if record.drift_flag],
        "samples": [],
    }

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for record in records:
            include_feature = (
                record.confidence_bucket == HIGH_CONFIDENCE
                or (record.confidence_bucket == LOW_CONFIDENCE and send_low_conf_features)
            )
            sample_entry = record.to_dict()
            if not include_feature:
                sample_entry["feature_relpath"] = None
                sample_entry["feature_bytes"] = 0
            manifest["samples"].append(sample_entry)

            for path in sample_store.iter_existing_paths(record):
                relpath = os.path.relpath(path, sample_store.root_dir).replace("\\", "/")
                if not include_feature and relpath == record.feature_relpath:
                    continue
                zf.write(path, arcname=relpath)

        zf.writestr(
            "bundle_manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
        )
    return buf.getvalue(), manifest

import socket

def is_network_connected(address):
    ip, port = address.split(':')[0], int(address.split(':')[1])
    try:
        socket.create_connection((ip, port), timeout=1)
        return True
    except OSError:
        return False


def request_cloud_training(server_ip, edge_id, frame_indices, cache_path, num_epoch=0):
    """Send selected frame indices to the cloud for GT annotation and edge-model
    fine-tuning.  Returns ``(success, model_data_b64, message)``.

    Parameters
    ----------
    server_ip : str
        gRPC server address, e.g. ``"192.168.1.1:50051"``.
    edge_id : int
        Identifier of this edge node.
    frame_indices : list[int]
        Frame indices (relative to ``cache_path``) chosen for retraining.
    cache_path : str
        Absolute path to the local frame cache directory shared with the cloud
        (or accessible by both).
    num_epoch : int
        Number of fine-tuning epochs; 0 lets the cloud use its configured default.

    Returns
    -------
    tuple[bool, str, str]
        ``(success, base64_model_state_dict, message)``
    """
    try:
        channel = grpc.insecure_channel(server_ip, options=grpc_message_options())
        stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
        req = message_transmission_pb2.TrainRequest(
            edge_id=int(edge_id),
            frame_indices=[int(index) for index in frame_indices],
            cache_path=_server_workspace_hint(edge_id, "train_model"),
            num_epoch=int(num_epoch),
            payload_zip=pack_training_payload(cache_path, frame_indices),
        )
        reply = stub.train_model_request(req)
        return reply.success, reply.model_data, reply.message
    except Exception as exc:
        logger.exception("request_cloud_training failed: {}", exc)
        return False, "", str(exc)


def request_cloud_split_training(
    server_ip, edge_id, all_frame_indices, drift_frame_indices,
    cache_path, num_epoch=0,
):
    """Send frame indices and drift info to cloud for **split-learning**
    continual learning.

    The cloud will:
      1. Annotate **only** drift frames with the large model.
      2. Train the server-side model (rpn + roi_heads) on **all** cached
         backbone features (using pseudo-labels for non-drift frames).
      3. Return the updated edge-model state-dict.

    Parameters
    ----------
    server_ip : str
        gRPC server address.
    edge_id : int
    all_frame_indices : list[int]
        Every frame index that has cached backbone features.
    drift_frame_indices : list[int]
        Subset of *all_frame_indices* where drift was detected.
    cache_path : str
        Shared cache directory containing ``features/`` and ``frames/`` dirs.
    num_epoch : int
        Training epochs; 0 → cloud default.

    Returns
    -------
    tuple[bool, str, str]
        ``(success, base64_model_state_dict, message)``
    """
    try:
        channel = grpc.insecure_channel(server_ip, options=grpc_message_options())
        stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
        req = message_transmission_pb2.SplitTrainRequest(
            edge_id=int(edge_id),
            all_frame_indices=[int(index) for index in all_frame_indices],
            drift_frame_indices=[int(index) for index in (drift_frame_indices or [])],
            cache_path=_server_workspace_hint(edge_id, "split_train"),
            num_epoch=int(num_epoch),
            payload_zip=pack_training_payload(cache_path, all_frame_indices, drift_frame_indices),
        )
        reply = stub.split_train_request(req)
        return reply.success, reply.model_data, reply.message
    except Exception as exc:
        logger.exception("request_cloud_split_training failed: {}", exc)
        return False, "", str(exc)


def request_continual_learning(
    server_ip: str,
    *,
    edge_id: int,
    cache_path: str,
    sample_store: EdgeSampleStore,
    split_plan: SplitPlan,
    model_id: str,
    model_version: str,
    send_low_conf_features: bool,
    num_epoch: int = 0,
):
    try:
        payload_zip, manifest = pack_continual_learning_bundle(
            sample_store,
            edge_id=edge_id,
            send_low_conf_features=send_low_conf_features,
            split_plan=split_plan,
            model_id=model_id,
            model_version=model_version,
        )
        channel = grpc.insecure_channel(
            server_ip,
            options=grpc_message_options(),
        )
        stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
        req = message_transmission_pb2.ContinualLearningRequest(
            protocol_version=manifest["protocol_version"],
            edge_id=int(edge_id),
            cache_path=_server_workspace_hint(edge_id, "continual_learning"),
            num_epoch=int(num_epoch),
            send_low_conf_features=bool(send_low_conf_features),
            payload_zip=payload_zip,
        )
        reply = stub.continual_learning_request(req)
        return reply.success, reply.model_data, reply.message
    except Exception as exc:
        logger.exception("request_continual_learning failed: {}", exc)
        return False, "", str(exc)


def submit_training_job(
    server_ip: str,
    *,
    edge_id: int,
    request_id: str,
    job_type: int,
    cache_path: str,
    num_epoch: int = 0,
    protocol_version: str = "",
    send_low_conf_features: bool = False,
    frame_indices: list[int] | None = None,
    all_frame_indices: list[int] | None = None,
    drift_frame_indices: list[int] | None = None,
    payload_zip: bytes = b"",
):
    try:
        channel = grpc.insecure_channel(server_ip, options=grpc_message_options())
        stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
        req = message_transmission_pb2.SubmitTrainingJobRequest(
            protocol_version=str(protocol_version or ""),
            edge_id=int(edge_id),
            request_id=str(request_id or ""),
            job_type=int(job_type),
            cache_path=str(cache_path or ""),
            num_epoch=int(num_epoch),
            send_low_conf_features=bool(send_low_conf_features),
            frame_indices=[int(index) for index in (frame_indices or [])],
            all_frame_indices=[int(index) for index in (all_frame_indices or [])],
            drift_frame_indices=[int(index) for index in (drift_frame_indices or [])],
            payload_zip=payload_zip,
        )
        return stub.submit_training_job(req)
    except Exception as exc:
        logger.exception("submit_training_job failed: {}", exc)
        return None


def get_training_job_status(
    server_ip: str,
    *,
    edge_id: int,
    job_id: str,
):
    try:
        channel = grpc.insecure_channel(server_ip, options=grpc_message_options())
        stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
        req = message_transmission_pb2.TrainingJobStatusRequest(
            edge_id=int(edge_id),
            job_id=str(job_id or ""),
        )
        return stub.get_training_job_status(req)
    except Exception as exc:
        logger.exception("get_training_job_status failed: {}", exc)
        return None


def download_trained_model(
    server_ip: str,
    *,
    edge_id: int,
    job_id: str,
):
    try:
        channel = grpc.insecure_channel(server_ip, options=grpc_message_options())
        stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
        req = message_transmission_pb2.DownloadTrainedModelRequest(
            edge_id=int(edge_id),
            job_id=str(job_id or ""),
        )
        reply = stub.download_trained_model(req)
        return reply.success, reply.model_data, reply.message
    except Exception as exc:
        logger.exception("download_trained_model failed: {}", exc)
        return False, "", str(exc)


def submit_continual_learning_job(
    server_ip: str,
    *,
    edge_id: int,
    sample_store: EdgeSampleStore,
    split_plan: SplitPlan,
    model_id: str,
    model_version: str,
    send_low_conf_features: bool,
    num_epoch: int = 0,
    request_id: str | None = None,
):
    try:
        payload_zip, manifest = pack_continual_learning_bundle(
            sample_store,
            edge_id=edge_id,
            send_low_conf_features=send_low_conf_features,
            split_plan=split_plan,
            model_id=model_id,
            model_version=model_version,
        )
        reply = submit_training_job(
            server_ip,
            edge_id=edge_id,
            request_id=str(request_id or uuid.uuid4().hex),
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING,
            cache_path=_server_workspace_hint(edge_id, "continual_learning"),
            num_epoch=num_epoch,
            protocol_version=manifest["protocol_version"],
            send_low_conf_features=bool(send_low_conf_features),
            payload_zip=payload_zip,
        )
        if reply is None:
            return False, "", "submit_training_job failed"
        return bool(reply.accepted), str(reply.job_id), str(reply.message)
    except Exception as exc:
        logger.exception("submit_continual_learning_job failed: {}", exc)
        return False, "", str(exc)
