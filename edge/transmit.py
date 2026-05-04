import json
import os
import tempfile
import time
import uuid

import grpc
from loguru import logger

from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc
from tools.grpc_options import grpc_message_options
import zipfile
import io

from edge.quality_assessor import HIGH_QUALITY, LOW_QUALITY
from edge.sample_store import EdgeSampleStore
from model_management.fixed_split import SplitPlan
from model_management.continual_learning_bundle import (
    CONTINUAL_LEARNING_PROTOCOL_VERSION,
)

DEFAULT_CL_BUNDLE_MAX_BYTES = 32 * 1024 * 1024


def _server_workspace_hint(edge_id: int, request_kind: str) -> str:
    return f"edge_{int(edge_id)}/{request_kind}"


def _format_bytes(num_bytes: int | float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} GiB"


def _manifest_payload_summary(manifest: dict) -> dict[str, int]:
    samples = list(manifest.get("samples") or [])
    return {
        "samples": len(samples),
        "drift_window_samples": sum(1 for sample in samples if bool(sample.get("in_drift_window", False))),
        "feature_bytes": sum(int(sample.get("feature_bytes", 0) or 0) for sample in samples),
        "raw_bytes": sum(int(sample.get("raw_bytes", 0) or 0) for sample in samples),
    }


def _entry_for_relpath(
    sample_store: EdgeSampleStore,
    relpath: str | None,
) -> dict[str, object] | None:
    if relpath is None:
        return None
    normalized = str(relpath).replace("\\", "/")
    path = os.path.join(sample_store.root_dir, normalized.replace("/", os.sep))
    if not os.path.exists(path):
        return None
    return {
        "relpath": normalized,
        "path": path,
        "bytes": os.path.getsize(path),
    }


def _quality_sort_key(record) -> tuple[float, str, str]:
    quality = record.quality_score
    return (
        0.0 if bool(getattr(record, "in_drift_window", False)) else 1.0,
        float("inf") if quality is None else float(quality),
        str(record.timestamp),
        str(record.sample_id),
    )


def _build_bundle_sample(
    sample_store: EdgeSampleStore,
    record,
    *,
    include_feature: bool,
    include_raw: bool,
) -> dict[str, object] | None:
    entries = []
    sample_entry = record.to_dict()
    try:
        sample_entry["inference_result"] = sample_store.load_inference_result(record)
        sample_entry["result_relpath"] = None
    except Exception:
        result_entry = _entry_for_relpath(sample_store, record.result_relpath)
        if result_entry is None:
            return None
        entries.append(result_entry)

    feature_entry = None
    if include_feature:
        feature_entry = _entry_for_relpath(sample_store, record.feature_relpath)
        if feature_entry is None:
            return None
        entries.append(feature_entry)

    raw_entry = None
    if include_raw:
        raw_entry = _entry_for_relpath(sample_store, record.raw_relpath)
        if raw_entry is None:
            return None
        entries.append(raw_entry)

    if feature_entry is None:
        sample_entry["feature_relpath"] = None
        sample_entry["feature_bytes"] = 0
        sample_entry["has_feature"] = False
    else:
        sample_entry["feature_bytes"] = int(feature_entry["bytes"])

    if raw_entry is None:
        sample_entry["raw_relpath"] = None
        sample_entry["raw_bytes"] = 0
        sample_entry["has_raw_sample"] = False
    else:
        sample_entry["raw_bytes"] = int(raw_entry["bytes"])

    return {
        "record": record,
        "sample": sample_entry,
        "files": entries,
    }


def _select_bundle_records(
    sample_store: EdgeSampleStore,
    records: list,
    *,
    send_low_conf_features: bool,
    bundle_cap_bytes: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    cap = max(1, int(bundle_cap_bytes))
    selected_by_id: dict[str, dict[str, object]] = {}
    selected_order: list[str] = []
    included_files: dict[str, dict[str, object]] = {}
    source_total_bytes = 0

    def _add_or_update(
        record,
        *,
        include_feature: bool,
        include_raw: bool,
        protected: bool,
    ) -> bool:
        nonlocal source_total_bytes
        current = selected_by_id.get(record.sample_id)
        if current is not None:
            current_sample = dict(current["sample"])
            include_feature = include_feature or current_sample.get("feature_relpath") is not None
            include_raw = include_raw or current_sample.get("raw_relpath") is not None

        candidate = _build_bundle_sample(
            sample_store,
            record,
            include_feature=include_feature,
            include_raw=include_raw,
        )
        if candidate is None:
            return False

        new_files = [
            entry
            for entry in candidate["files"]
            if str(entry["relpath"]) not in included_files
        ]
        added_bytes = sum(int(entry["bytes"]) for entry in new_files)
        if not protected and source_total_bytes + added_bytes > cap:
            return False

        if current is None:
            selected_order.append(str(record.sample_id))
        candidate["protected"] = bool(protected or (current or {}).get("protected", False))
        selected_by_id[str(record.sample_id)] = candidate
        for entry in new_files:
            included_files[str(entry["relpath"])] = entry
        source_total_bytes += added_bytes
        return True

    for record in records:
        if record.quality_bucket == HIGH_QUALITY:
            _add_or_update(
                record,
                include_feature=True,
                include_raw=False,
                protected=False,
            )

    low_quality_records = [
        record
        for record in records
        if record.quality_bucket == LOW_QUALITY
    ]
    for record in sorted(low_quality_records, key=_quality_sort_key):
        _add_or_update(
            record,
            include_feature=bool(send_low_conf_features),
            include_raw=True,
            protected=bool(getattr(record, "in_drift_window", False)),
        )

    selected = [selected_by_id[sample_id] for sample_id in selected_order]
    samples = [entry["sample"] for entry in selected]
    policy = {
        "policy": "quality_capped_high_features_low_quality_raw",
        "bundle_cap_bytes": cap,
        "selected_sample_count": len(selected),
        "omitted_sample_count": max(0, len(records) - len(selected)),
        "drift_window_selected_count": sum(
            1 for sample in samples if bool(sample.get("in_drift_window", False))
        ),
        "source_total_bytes": int(source_total_bytes),
        "source_feature_bytes": sum(
            int(sample.get("feature_bytes", 0) or 0) for sample in samples
        ),
        "source_raw_bytes": sum(
            int(sample.get("raw_bytes", 0) or 0) for sample in samples
        ),
        "zip_payload_bytes": 0,
    }
    return selected, policy


def _estimate_stored_zip_bytes(
    file_entries: list[dict[str, object]],
    manifest_bytes: bytes,
) -> int:
    total = 22
    for entry in file_entries:
        name_bytes = str(entry["relpath"]).encode("utf-8")
        total += int(entry["bytes"]) + 30 + len(name_bytes)
        total += 46 + len(name_bytes)
    manifest_name = b"bundle_manifest.json"
    total += len(manifest_bytes) + 30 + len(manifest_name)
    total += 46 + len(manifest_name)
    return int(total)


def _write_continual_learning_zip(
    zip_path: str,
    *,
    file_entries: list[dict[str, object]],
    manifest: dict,
) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for entry in file_entries:
            zf.write(
                str(entry["path"]),
                arcname=str(entry["relpath"]),
                compress_type=zipfile.ZIP_STORED,
            )
        zf.writestr(
            "bundle_manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
            compress_type=zipfile.ZIP_STORED,
        )


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
    bundle_cap_bytes: int | None = None,
) -> tuple[bytes, dict]:
    zip_path, manifest, _ = pack_continual_learning_bundle_to_file(
        sample_store,
        edge_id=edge_id,
        send_low_conf_features=send_low_conf_features,
        split_plan=split_plan,
        model_id=model_id,
        model_version=model_version,
        bundle_cap_bytes=bundle_cap_bytes,
    )
    try:
        with open(zip_path, "rb") as handle:
            return handle.read(), manifest
    finally:
        try:
            os.remove(zip_path)
        except OSError:
            pass


def pack_continual_learning_bundle_to_file(
    sample_store: EdgeSampleStore,
    *,
    edge_id: int,
    send_low_conf_features: bool,
    split_plan: SplitPlan,
    model_id: str,
    model_version: str,
    bundle_cap_bytes: int | None = None,
    output_dir: str | None = None,
) -> tuple[str, dict, dict]:
    selection_started = time.perf_counter()
    records = [
        record
        for record in sample_store.list_records()
        if record.split_config_id == split_plan.split_config_id
        and record.model_id == str(model_id)
        and record.model_version == str(model_version)
    ]
    selected, selection_policy = _select_bundle_records(
        sample_store,
        records,
        send_low_conf_features=send_low_conf_features,
        bundle_cap_bytes=(
            DEFAULT_CL_BUNDLE_MAX_BYTES
            if bundle_cap_bytes is None
            else int(bundle_cap_bytes)
        ),
    )
    selection_elapsed = time.perf_counter() - selection_started
    cap = int(selection_policy["bundle_cap_bytes"])
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
            "low_quality_mode": "raw+feature" if send_low_conf_features else "raw-only",
        },
        "selection_policy": selection_policy,
        "samples": [],
    }

    def _refresh_manifest() -> list[dict[str, object]]:
        file_map: dict[str, dict[str, object]] = {}
        for selected_entry in selected:
            for file_entry in selected_entry["files"]:
                file_map[str(file_entry["relpath"])] = file_entry
        file_entries = list(file_map.values())
        samples = [entry["sample"] for entry in selected]
        manifest["samples"] = samples
        policy = manifest["selection_policy"]
        policy["selected_sample_count"] = len(samples)
        policy["omitted_sample_count"] = max(0, len(records) - len(samples))
        policy["drift_window_selected_count"] = sum(
            1 for sample in samples if bool(sample.get("in_drift_window", False))
        )
        policy["source_total_bytes"] = sum(
            int(entry["bytes"]) for entry in file_entries
        )
        policy["source_feature_bytes"] = sum(
            int(sample.get("feature_bytes", 0) or 0) for sample in samples
        )
        policy["source_raw_bytes"] = sum(
            int(sample.get("raw_bytes", 0) or 0) for sample in samples
        )
        return file_entries

    def _update_estimated_zip_bytes(file_entries: list[dict[str, object]]) -> int:
        for _ in range(3):
            manifest_bytes = json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            estimated_zip_bytes = _estimate_stored_zip_bytes(file_entries, manifest_bytes)
            if int(manifest["selection_policy"]["zip_payload_bytes"]) == estimated_zip_bytes:
                return estimated_zip_bytes
            manifest["selection_policy"]["zip_payload_bytes"] = estimated_zip_bytes
        return int(manifest["selection_policy"]["zip_payload_bytes"])

    while True:
        file_entries = _refresh_manifest()
        estimated_zip_bytes = _update_estimated_zip_bytes(file_entries)
        removable_index = next(
            (
                index
                for index in range(len(selected) - 1, -1, -1)
                if not bool(selected[index].get("protected", False))
            ),
            None,
        )
        if estimated_zip_bytes <= cap or removable_index is None:
            break
        selected.pop(removable_index)

    file_entries = _refresh_manifest()
    _update_estimated_zip_bytes(file_entries)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f"cl_bundle_edge_{int(edge_id)}_",
        suffix=".zip",
        dir=output_dir,
        delete=False,
    )
    zip_path = handle.name
    handle.close()

    zip_started = time.perf_counter()
    try:
        _write_continual_learning_zip(
            zip_path,
            file_entries=file_entries,
            manifest=manifest,
        )
        zip_payload_bytes = os.path.getsize(zip_path)
        if zip_payload_bytes != int(manifest["selection_policy"]["zip_payload_bytes"]):
            manifest["selection_policy"]["zip_payload_bytes"] = int(zip_payload_bytes)
            _write_continual_learning_zip(
                zip_path,
                file_entries=file_entries,
                manifest=manifest,
            )
            zip_payload_bytes = os.path.getsize(zip_path)
        manifest["selection_policy"]["zip_payload_bytes"] = int(zip_payload_bytes)
        stats = dict(manifest["selection_policy"])
        stats.update(
            {
                "selection_elapsed_sec": float(selection_elapsed),
                "zip_write_elapsed_sec": float(time.perf_counter() - zip_started),
                "zip_path": zip_path,
                "zip_payload_bytes": int(zip_payload_bytes),
            }
        )
        return zip_path, manifest, stats
    except Exception:
        try:
            os.remove(zip_path)
        except OSError:
            pass
        raise

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


def request_cloud_split_training(
    server_ip, edge_id, all_frame_indices, drift_frame_indices,
    cache_path,
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
    protocol_version: str = "",
    send_low_conf_features: bool = False,
    frame_indices: list[int] | None = None,
    all_frame_indices: list[int] | None = None,
    drift_frame_indices: list[int] | None = None,
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
            all_frame_indices=[int(index) for index in (all_frame_indices or [])],
            drift_frame_indices=[int(index) for index in (drift_frame_indices or [])],
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
    send_low_conf_features: bool,
    bundle_cap_bytes: int | None = None,
    bandwidth_mbps: float = 0.0,
    request_id: str | None = None,
    channel=None,
):
    try:
        pack_started = time.perf_counter()
        stats = sample_store.stats()
        logger.info(
            "Packing continual learning bundle for edge {} "
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
        payload_zip, manifest = pack_continual_learning_bundle(
            sample_store,
            edge_id=edge_id,
            send_low_conf_features=send_low_conf_features,
            split_plan=split_plan,
            model_id=model_id,
            model_version=model_version,
            bundle_cap_bytes=bundle_cap_bytes,
        )
        payload_summary = _manifest_payload_summary(manifest)
        zip_payload_bytes = len(payload_zip)
        selection_policy = dict(manifest.get("selection_policy", {}) or {})
        estimated_upload_sec = None
        if bandwidth_mbps > 0.0 and zip_payload_bytes > 0:
            estimated_upload_sec = (
                zip_payload_bytes * 8.0 / (float(bandwidth_mbps) * 1_000_000.0)
            )
        logger.info(
            "Packed continual learning bundle for edge {} "
            "(total_pack_time={:.3f}s, "
            "(manifest_samples={}, drift_window_samples={}, source_feature_bytes={}, "
            "source_raw_bytes={}, source_total_bytes={}, cap={}, zip_payload={}, "
            "estimated_upload_sec={}).",
            edge_id,
            time.perf_counter() - pack_started,
            payload_summary["samples"],
            payload_summary["drift_window_samples"],
            _format_bytes(payload_summary["feature_bytes"]),
            _format_bytes(payload_summary["raw_bytes"]),
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
                "Continual learning upload failed for edge {} "
                "(elapsed={:.3f}s, average_speed={:.3f} Mbps, zip_payload={}).",
                edge_id,
                upload_elapsed,
                upload_mbps,
                _format_bytes(zip_payload_bytes),
            )
            return False, "", "submit_training_job failed"
        logger.info(
            "Continual learning upload completed for edge {} "
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
