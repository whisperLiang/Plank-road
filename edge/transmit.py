import json
import os
import tarfile
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import grpc
import torch
from loguru import logger

from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc
from tools.grpc_options import grpc_message_options
import zipfile
import io

from edge.quality_assessor import LOW_QUALITY
from edge.sample_store import EdgeSampleStore
from model_management.fixed_split import SplitPlan

LOW_QUALITY_TRIGGER_PROTOCOL_VERSION = "low-quality-trigger-shard.v1"


def _server_workspace_hint(edge_id: int, request_kind: str) -> str:
    return f"edge_{int(edge_id)}/{request_kind}"


def _format_bytes(num_bytes: int | float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} GiB"


def _quality_sort_key(record) -> tuple[float, str, str]:
    quality = record.quality_score
    return (
        0.0 if bool(getattr(record, "in_drift_window", False)) else 1.0,
        float("inf") if quality is None else float(quality),
        str(record.timestamp),
        str(record.sample_id),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_sample_filename(sample_id: str, suffix: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(sample_id)
    ).strip("._")
    return f"{safe or uuid.uuid4().hex}{suffix}"


def _tensor_only_features(intermediate: Any) -> dict[str, torch.Tensor]:
    if hasattr(intermediate, "tensors"):
        source = dict(getattr(intermediate, "tensors"))
    elif isinstance(intermediate, torch.Tensor):
        source = {"payload": intermediate}
    elif isinstance(intermediate, Mapping):
        source = dict(intermediate.get("tensors") or intermediate)
    else:
        raise TypeError(f"Unsupported intermediate feature type: {type(intermediate)!r}")
    tensors: dict[str, torch.Tensor] = {}
    for label, value in source.items():
        if isinstance(value, torch.Tensor):
            tensors[str(label)] = value.detach().cpu()
    if not tensors:
        raise ValueError("Intermediate feature payload did not contain tensors.")
    return tensors


def _feature_sample_payload(intermediate: Any) -> dict[str, Any]:
    return {"tensors": _tensor_only_features(intermediate)}


def _record_abs_path(sample_store: EdgeSampleStore, relpath: str | None) -> str | None:
    if relpath is None:
        return None
    return os.path.join(sample_store.root_dir, str(relpath).replace("/", os.sep))


def _low_quality_trigger_source_bytes(
    sample_store: EdgeSampleStore,
    record,
    *,
    send_low_conf_features: bool,
) -> int:
    total = 0
    for relpath in (record.raw_relpath, record.feature_relpath if send_low_conf_features else None):
        path = _record_abs_path(sample_store, relpath)
        if path and os.path.exists(path):
            total += os.path.getsize(path)
    return int(total)


def _select_low_quality_trigger_records(
    sample_store: EdgeSampleStore,
    records: Sequence,
    *,
    send_low_conf_features: bool,
    bundle_cap_bytes: int | None,
) -> tuple[list, dict[str, Any]]:
    cap = None if bundle_cap_bytes is None else max(1, int(bundle_cap_bytes))
    selected = []
    selected_bytes = 0
    omitted = 0
    for record in sorted(
        [record for record in records if record.quality_bucket == LOW_QUALITY],
        key=_quality_sort_key,
    ):
        raw_path = _record_abs_path(sample_store, record.raw_relpath)
        if raw_path is None or not os.path.exists(raw_path):
            omitted += 1
            continue
        source_bytes = _low_quality_trigger_source_bytes(
            sample_store,
            record,
            send_low_conf_features=send_low_conf_features,
        )
        protected = bool(getattr(record, "in_drift_window", False))
        if cap is not None and not protected and selected and selected_bytes + source_bytes > cap:
            omitted += 1
            continue
        selected.append(record)
        selected_bytes += source_bytes
    return selected, {
        "policy": "low_quality_trigger_raw_with_optional_features",
        "bundle_cap_bytes": 0 if cap is None else int(cap),
        "selected_sample_count": len(selected),
        "omitted_sample_count": omitted + max(0, len(records) - len(selected) - omitted),
        "source_total_bytes": int(selected_bytes),
        "zip_payload_bytes": 0,
    }


def _chunks(items: Sequence, size: int) -> list[list]:
    shard_size = max(1, int(size))
    return [list(items[index : index + shard_size]) for index in range(0, len(items), shard_size)]


def _write_low_quality_raw_tar(
    sample_store: EdgeSampleStore,
    records: Sequence,
) -> tuple[bytes, list[dict[str, str]]]:
    tar_buffer = io.BytesIO()
    manifest_entries: list[dict[str, str]] = []
    with tarfile.open(fileobj=tar_buffer, mode="w") as tf:
        for record in records:
            sample_id = str(record.sample_id)
            raw_path = _record_abs_path(sample_store, record.raw_relpath)
            if raw_path is None or not os.path.exists(raw_path):
                raise FileNotFoundError(raw_path or record.raw_relpath or sample_id)
            suffix = os.path.splitext(raw_path)[1] or ".jpg"
            raw_name = f"raw/{_safe_sample_filename(sample_id, suffix)}"
            tf.add(raw_path, arcname=raw_name, recursive=False)
            manifest_entries.append(
                {
                    "sample_id": sample_id,
                    "raw_file": raw_name,
                }
            )
        manifest_bytes = (
            "\n".join(
                json.dumps(entry, sort_keys=True, separators=(",", ":"))
                for entry in manifest_entries
            )
            + ("\n" if manifest_entries else "")
        ).encode("utf-8")
        manifest_info = tarfile.TarInfo("manifest.jsonl")
        manifest_info.size = len(manifest_bytes)
        manifest_info.mtime = int(time.time())
        tf.addfile(manifest_info, io.BytesIO(manifest_bytes))
    return tar_buffer.getvalue(), manifest_entries


def _write_low_quality_feature_shard(
    sample_store: EdgeSampleStore,
    records: Sequence,
    *,
    runtime_contract: Mapping[str, object] | None = None,
) -> tuple[bytes | None, list[str]]:
    feature_payload = {"schema_version": 1, "samples": {}}
    contract_payload = dict(runtime_contract or {})
    sample_ids: list[str] = []
    for record in records:
        sample_id = str(record.sample_id)
        feature_path = _record_abs_path(sample_store, record.feature_relpath)
        if feature_path is None or not os.path.exists(feature_path):
            continue
        try:
            feature_sample = _feature_sample_payload(sample_store.load_intermediate(record))
            if contract_payload:
                feature_sample["runtime_contract"] = contract_payload
                if contract_payload.get("feature_layout_id"):
                    feature_sample["feature_layout_id"] = str(
                        contract_payload.get("feature_layout_id")
                    )
            feature_payload["samples"][sample_id] = feature_sample
        except Exception as exc:
            logger.warning(
                "Skipping optional low-quality feature for sample {}: {}",
                sample_id,
                exc,
            )
            feature_payload["samples"].pop(sample_id, None)
            continue
        sample_ids.append(sample_id)
    if not sample_ids:
        return None, []
    buffer = io.BytesIO()
    torch.save(feature_payload, buffer)
    return buffer.getvalue(), sample_ids


def pack_low_quality_trigger_bundle(
    sample_store: EdgeSampleStore,
    *,
    edge_id: int,
    send_low_conf_features: bool,
    split_plan: SplitPlan,
    model_id: str,
    model_version: str,
    model_metadata: Mapping[str, object] | None = None,
    bundle_cap_bytes: int | None = None,
    shard_size: int | None = None,
) -> tuple[bytes, dict]:
    zip_path, manifest, _ = pack_low_quality_trigger_bundle_to_file(
        sample_store,
        edge_id=edge_id,
        send_low_conf_features=send_low_conf_features,
        split_plan=split_plan,
        model_id=model_id,
        model_version=model_version,
        model_metadata=model_metadata,
        bundle_cap_bytes=bundle_cap_bytes,
        shard_size=shard_size,
    )
    try:
        with open(zip_path, "rb") as handle:
            return handle.read(), manifest
    finally:
        try:
            os.remove(zip_path)
        except OSError:
            pass


def pack_low_quality_trigger_bundle_to_file(
    sample_store: EdgeSampleStore,
    *,
    edge_id: int,
    send_low_conf_features: bool,
    split_plan: SplitPlan,
    model_id: str,
    model_version: str,
    model_metadata: Mapping[str, object] | None = None,
    bundle_cap_bytes: int | None = None,
    shard_size: int | None = None,
    output_dir: str | None = None,
) -> tuple[str, dict, dict]:
    pack_started = time.perf_counter()
    records = [
        record
        for record in sample_store.list_records()
        if record.split_config_id == split_plan.split_config_id
        and record.model_id == str(model_id)
        and str(getattr(record, "front_version", "0") or "0")
        == str(getattr(split_plan, "front_version", "0") or "0")
        and record.quality_bucket == LOW_QUALITY
    ]
    selected, selection_policy = _select_low_quality_trigger_records(
        sample_store,
        records,
        send_low_conf_features=send_low_conf_features,
        bundle_cap_bytes=bundle_cap_bytes,
    )
    resolved_shard_size = max(1, int(shard_size or 64))
    model_meta = {
        "model_id": str(model_id),
        "model_version": str(model_version),
    }
    for key, value in dict(model_metadata or {}).items():
        if value is None:
            continue
        model_meta[str(key)] = value

    manifest = {
        "protocol_version": LOW_QUALITY_TRIGGER_PROTOCOL_VERSION,
        "edge_id": int(edge_id),
        "model_id": str(model_id),
        "model_version": str(model_version),
        "front_version": str(getattr(split_plan, "front_version", "0") or "0"),
        "split_config_id": str(split_plan.split_config_id),
        "canonical_split_key": str(getattr(split_plan, "canonical_split_key", "") or ""),
        "edge_split_id": str(getattr(split_plan, "edge_split_id", "") or ""),
        "input_tensor_shape": [
            int(dim) for dim in list(getattr(split_plan, "input_tensor_shape", []) or [])
        ],
        "input_resize_mode": str(getattr(split_plan, "input_resize_mode", "") or "direct_resize"),
        "runtime_contract": dict(getattr(split_plan, "runtime_contract", {}) or {}),
        "upload_mode": "raw+feature" if send_low_conf_features else "raw-only",
        "created_at": _utc_now(),
        "model": model_meta,
        "split_plan": split_plan.to_dict(),
        "training_mode": {
            "send_low_conf_features": bool(send_low_conf_features),
            "low_quality_mode": "raw+feature" if send_low_conf_features else "raw-only",
        },
        "selection_policy": selection_policy,
        "shard_size": resolved_shard_size,
        "sample_count": len(selected),
        "raw_shards": [],
        "feature_shards": [],
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f"low_trigger_edge_{int(edge_id)}_",
        suffix=".zip",
        dir=output_dir,
        delete=False,
    )
    zip_path = handle.name
    handle.close()
    try:
        raw_shard_payloads: list[tuple[str, bytes, list[str]]] = []
        feature_shard_payloads: list[tuple[str, bytes, list[str]]] = []
        for shard_index, shard_records in enumerate(_chunks(selected, resolved_shard_size), 1):
            sample_ids = [str(record.sample_id) for record in shard_records]
            raw_shard_name = f"raw_shards/low_raw_shard_{shard_index:06d}.tar"
            tar_bytes, _manifest_entries = _write_low_quality_raw_tar(
                sample_store,
                shard_records,
            )
            raw_shard_payloads.append((raw_shard_name, tar_bytes, sample_ids))
            if send_low_conf_features:
                feature_shard_name = f"feature_shards/low_feature_shard_{shard_index:06d}.pt"
                feature_payload, feature_sample_ids = _write_low_quality_feature_shard(
                    sample_store,
                    shard_records,
                    runtime_contract=dict(getattr(split_plan, "runtime_contract", {}) or {}),
                )
                if feature_payload is not None and feature_sample_ids:
                    feature_shard_payloads.append(
                        (
                            feature_shard_name,
                            feature_payload,
                            feature_sample_ids,
                        )
                    )
        manifest["raw_shards"] = [
            {
                "shard_id": f"edge{int(edge_id)}_low_raw_{index:06d}",
                "file": name,
                "sample_count": len(sample_ids),
            }
            for index, (name, _payload, sample_ids) in enumerate(raw_shard_payloads, 1)
        ]
        manifest["feature_shards"] = [
            {
                "shard_id": f"edge{int(edge_id)}_low_feature_{index:06d}",
                "file": name,
                "sample_count": len(sample_ids),
            }
            for index, (name, _payload, sample_ids) in enumerate(feature_shard_payloads, 1)
        ]

        def _write_zip() -> None:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
                for name, payload, _sample_ids in raw_shard_payloads:
                    zf.writestr(name, payload, compress_type=zipfile.ZIP_STORED)
                for name, payload, _sample_ids in feature_shard_payloads:
                    zf.writestr(name, payload, compress_type=zipfile.ZIP_STORED)
                zf.writestr(
                    "trigger_manifest.json",
                    json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
                    compress_type=zipfile.ZIP_STORED,
                )

        for _ in range(3):
            _write_zip()
            zip_payload_bytes = os.path.getsize(zip_path)
            if int(manifest["selection_policy"]["zip_payload_bytes"]) == int(zip_payload_bytes):
                break
            manifest["selection_policy"]["zip_payload_bytes"] = int(zip_payload_bytes)
        _write_zip()
        zip_payload_bytes = os.path.getsize(zip_path)
        manifest["selection_policy"]["zip_payload_bytes"] = int(zip_payload_bytes)
        stats = dict(manifest["selection_policy"])
        stats.update(
            {
                "pack_elapsed_sec": float(time.perf_counter() - pack_started),
                "zip_path": zip_path,
                "zip_payload_bytes": int(zip_payload_bytes),
                "sample_count": len(selected),
                "raw_shard_count": len(manifest["raw_shards"]),
                "feature_shard_count": len(manifest["feature_shards"]),
            }
        )
        return zip_path, manifest, stats
    except Exception:
        try:
            os.remove(zip_path)
        except OSError:
            pass
        raise


def pack_training_payload(cache_path, frame_indices):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for idx in frame_indices:
            frame_path = os.path.join(cache_path, "frames", f"{idx}.jpg")
            if os.path.exists(frame_path):
                zf.write(frame_path, arcname=f"frames/{idx}.jpg")
    return buf.getvalue()


import socket

def is_network_connected(address):
    ip, port = address.split(':')[0], int(address.split(':')[1])
    try:
        socket.create_connection((ip, port), timeout=1)
        return True
    except OSError:
        return False


def request_cloud_training(server_ip, edge_id, frame_indices, cache_path):
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
            payload_zip=pack_training_payload(cache_path, frame_indices),
        )
        reply = stub.train_model_request(req)
        return reply.success, reply.model_data, reply.message
    except Exception as exc:
        logger.exception("request_cloud_training failed: {}", exc)
        return False, "", str(exc)


def submit_training_job(
    server_ip: str,
    *,
    edge_id: int,
    request_id: str,
    job_type: int,
    cache_path: str,
    protocol_version: str = "",
    send_low_conf_features: bool = False,
    frame_indices: list[int] | None = None,
    payload_zip: bytes = b"",
    channel=None,
):
    owned_channel = channel is None
    request_started = time.perf_counter()
    try:
        if channel is None:
            channel = grpc.insecure_channel(server_ip, options=grpc_message_options())
        stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
        payload_size = len(payload_zip or b"")
        logger.info(
            "Submitting training job request_id={} edge_id={} job_type={} "
            "payload_zip={} server={}",
            request_id,
            edge_id,
            job_type,
            _format_bytes(payload_size),
            server_ip,
        )
        req = message_transmission_pb2.SubmitTrainingJobRequest(
            protocol_version=str(protocol_version or ""),
            edge_id=int(edge_id),
            request_id=str(request_id or ""),
            job_type=int(job_type),
            cache_path=str(cache_path or ""),
            send_low_conf_features=bool(send_low_conf_features),
            frame_indices=[int(index) for index in (frame_indices or [])],
            payload_zip=payload_zip,
        )
        reply = stub.submit_training_job(req)
        logger.info(
            "submit_training_job reply request_id={} accepted={} job_id={} "
            "status={} elapsed={:.3f}s",
            request_id,
            bool(reply.accepted),
            reply.job_id,
            reply.status,
            time.perf_counter() - request_started,
        )
        return reply
    except Exception as exc:
        logger.exception(
            "submit_training_job failed after {:.3f}s: {}",
            time.perf_counter() - request_started,
            exc,
        )
        return None
    finally:
        if owned_channel and channel is not None:
            channel.close()


def _build_proto_or_namespace(
    names: Sequence[str],
    fields: Mapping[str, Any],
):
    for name in names:
        message_cls = getattr(message_transmission_pb2, name, None)
        if message_cls is None:
            continue
        descriptor = getattr(message_cls, "DESCRIPTOR", None)
        if descriptor is not None:
            allowed = set(descriptor.fields_by_name.keys())
            return message_cls(
                **{
                    key: value
                    for key, value in fields.items()
                    if key in allowed
                }
            )
        return message_cls(**dict(fields))
    return SimpleNamespace(**dict(fields))


def submit_sample_sync(
    server_ip: str,
    *,
    edge_id: int,
    request_id: str,
    protocol_version: str,
    sync_type: str,
    model_id: str,
    model_version: str,
    split_config_id: str,
    payload_zip: bytes,
    cache_path: str | None = None,
    channel=None,
):
    owned_channel = channel is None
    request_started = time.perf_counter()
    try:
        if channel is None:
            channel = grpc.insecure_channel(server_ip, options=grpc_message_options())
        stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
        req = _build_proto_or_namespace(
            (
                "SampleSyncRequest",
                "SyncSamplesRequest",
                "SampleSyncBundleRequest",
            ),
            {
                "protocol_version": str(protocol_version or ""),
                "edge_id": int(edge_id),
                "model_id": str(model_id or ""),
                "model_version": str(model_version or ""),
                "split_config_id": str(split_config_id or ""),
                "sync_type": str(sync_type or ""),
                "payload_zip": payload_zip or b"",
            },
        )
        reply = stub.sync_samples(req)
        logger.info(
            "submit_sample_sync reply request_id={} elapsed={:.3f}s",
            request_id,
            time.perf_counter() - request_started,
        )
        return reply
    except Exception as exc:
        logger.exception(
            "submit_sample_sync failed after {:.3f}s: {}",
            time.perf_counter() - request_started,
            exc,
        )
        return None
    finally:
        if owned_channel and channel is not None:
            channel.close()


def get_training_job_status(
    server_ip: str,
    *,
    edge_id: int,
    job_id: str,
    channel=None,
):
    owned_channel = channel is None
    try:
        if channel is None:
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
    finally:
        if owned_channel and channel is not None:
            channel.close()


def download_trained_model(
    server_ip: str,
    *,
    edge_id: int,
    job_id: str,
    channel=None,
):
    owned_channel = channel is None
    try:
        if channel is None:
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
    finally:
        if owned_channel and channel is not None:
            channel.close()


def submit_continual_learning_job(
    server_ip: str,
    *,
    edge_id: int,
    sample_store: EdgeSampleStore,
    split_plan: SplitPlan,
    model_id: str,
    model_version: str,
    model_metadata: Mapping[str, object] | None = None,
    send_low_conf_features: bool,
    bundle_cap_bytes: int | None = None,
    trigger_shard_size: int | None = None,
    bandwidth_mbps: float = 0.0,
    request_id: str | None = None,
    channel=None,
):
    try:
        pack_started = time.perf_counter()
        stats = sample_store.stats()
        logger.info(
            "Packing low-quality trigger bundle for edge {} "
            "(samples={}, high_quality={}, low_quality={}, "
            "send_low_conf_features={}, model_id={}, model_version={})",
            edge_id,
            int(stats.get("total_samples", 0)),
            int(stats.get("high_quality_count", 0)),
            int(stats.get("low_quality_count", 0)),
            bool(send_low_conf_features),
            model_id,
            model_version,
        )
        payload_zip, manifest = pack_low_quality_trigger_bundle(
            sample_store,
            edge_id=edge_id,
            send_low_conf_features=send_low_conf_features,
            split_plan=split_plan,
            model_id=model_id,
            model_version=model_version,
            model_metadata=model_metadata,
            bundle_cap_bytes=bundle_cap_bytes,
            shard_size=trigger_shard_size,
        )
        zip_payload_bytes = len(payload_zip)
        selection_policy = dict(manifest.get("selection_policy", {}) or {})
        estimated_upload_sec = None
        if bandwidth_mbps > 0.0 and zip_payload_bytes > 0:
            estimated_upload_sec = (
                zip_payload_bytes * 8.0 / (float(bandwidth_mbps) * 1_000_000.0)
            )
        logger.info(
            "Packed low-quality trigger bundle for edge {} "
            "(total_pack_time={:.3f}s, "
            "samples={}, raw_shards={}, feature_shards={}, "
            "source_total_bytes={}, cap={}, zip_payload={}, "
            "estimated_upload_sec={}).",
            edge_id,
            time.perf_counter() - pack_started,
            int(manifest.get("sample_count", 0) or 0),
            len(manifest.get("raw_shards", []) or []),
            len(manifest.get("feature_shards", []) or []),
            _format_bytes(int(selection_policy.get("source_total_bytes", 0))),
            _format_bytes(int(selection_policy.get("bundle_cap_bytes", 0))),
            _format_bytes(zip_payload_bytes),
            (
                f"{estimated_upload_sec:.3f}s"
                if estimated_upload_sec is not None
                else "unknown"
            ),
        )
        upload_started = time.perf_counter()
        reply = submit_training_job(
            server_ip,
            edge_id=edge_id,
            request_id=str(request_id or uuid.uuid4().hex),
            job_type=message_transmission_pb2.TRAINING_JOB_TYPE_CONTINUAL_LEARNING,
            cache_path=_server_workspace_hint(edge_id, "continual_learning"),
            protocol_version=manifest["protocol_version"],
            send_low_conf_features=bool(send_low_conf_features),
            payload_zip=payload_zip,
            channel=channel,
        )
        upload_elapsed = time.perf_counter() - upload_started
        upload_mbps = (
            zip_payload_bytes * 8.0 / upload_elapsed / 1_000_000.0
            if upload_elapsed > 0.0 and zip_payload_bytes > 0
            else 0.0
        )
        if reply is None:
            logger.error(
                "Low-quality trigger upload failed for edge {} "
                "(elapsed={:.3f}s, average_speed={:.3f} Mbps, zip_payload={}).",
                edge_id,
                upload_elapsed,
                upload_mbps,
                _format_bytes(zip_payload_bytes),
            )
            return False, "", "submit_training_job failed"
        logger.info(
            "Low-quality trigger upload completed for edge {} "
            "(actual_upload_time={:.3f}s, upload_speed={:.3f} Mbps, zip_payload={}).",
            edge_id,
            upload_elapsed,
            upload_mbps,
            _format_bytes(zip_payload_bytes),
        )
        return bool(reply.accepted), str(reply.job_id), str(reply.message)
    except Exception as exc:
        logger.exception("submit_continual_learning_job failed: {}", exc)
        return False, "", str(exc)
