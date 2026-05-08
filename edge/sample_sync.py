from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Callable

import torch
from loguru import logger

import edge.transmit as transmit
from edge.quality_assessor import HIGH_QUALITY
from edge.sample_store import EdgeSampleStore, StoredSampleRecord
from model_management.payload import BoundaryPayload


HIGH_QUALITY_SYNC_PROTOCOL_VERSION = "high-quality-feature-label-shard.v1"
UPLOAD_LEDGER_VERSION = "edge-sample-upload-ledger.v1"
UPLOAD_LEDGER_FILENAME = "upload_ledger.json"
UPLOAD_PENDING = "pending"
UPLOAD_UPLOADED = "uploaded"
UPLOAD_COMMITTED = "committed"
UPLOAD_FAILED = "failed"
_RETRYABLE_STATES = {UPLOAD_PENDING, UPLOAD_UPLOADED, UPLOAD_FAILED}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _first_nonempty(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _atomic_json_dump(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{threading.get_ident()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _chunks(items: Sequence[StoredSampleRecord], size: int) -> list[list[StoredSampleRecord]]:
    shard_size = max(1, int(size))
    return [list(items[index : index + shard_size]) for index in range(0, len(items), shard_size)]


def _tensor_only_features(intermediate: Any) -> dict[str, torch.Tensor]:
    if isinstance(intermediate, BoundaryPayload):
        source = dict(intermediate.tensors)
    elif isinstance(intermediate, torch.Tensor):
        source = {"payload": intermediate}
    elif isinstance(intermediate, Mapping):
        source = dict(intermediate)
    else:
        raise TypeError(f"Unsupported intermediate feature type: {type(intermediate)!r}")

    tensors: dict[str, torch.Tensor] = {}
    for label, value in source.items():
        if isinstance(value, torch.Tensor):
            tensors[str(label)] = value.detach().cpu()
    if not tensors:
        raise ValueError("Intermediate feature payload did not contain tensors.")
    return tensors


def _boundary_payload_metadata(intermediate: Any) -> dict[str, Any] | None:
    if not isinstance(intermediate, BoundaryPayload):
        return None
    return {
        "split_id": str(intermediate.split_id),
        "graph_signature": str(intermediate.graph_signature),
        "batch_size": int(intermediate.batch_size),
        "schema": dict(intermediate.schema or {}),
        "requires_grad": dict(intermediate.requires_grad or {}),
        "weight_version": intermediate.weight_version,
        "passthrough_inputs": dict(intermediate.passthrough_inputs or {}),
    }


def _feature_sample_payload(intermediate: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"tensors": _tensor_only_features(intermediate)}
    metadata = _boundary_payload_metadata(intermediate)
    if metadata is not None:
        payload["boundary"] = metadata
    return payload


def _image_size_from_value(value: object) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        height = int(value[0])
        width = int(value[1])
        if height > 0 and width > 0:
            return height, width
    return None


def _model_input_size_from_shape(value: object) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        height = int(value[-2])
        width = int(value[-1])
        if height > 0 and width > 0:
            return height, width
    return None


def _project_box_to_model_input(
    box: object,
    *,
    original_size: tuple[int, int],
    model_input_size: tuple[int, int],
    resize_mode: str,
) -> list[float]:
    values = [float(value) for value in list(box or [])[:4]]
    if len(values) < 4:
        return values
    orig_h, orig_w = original_size
    model_h, model_w = model_input_size
    if str(resize_mode).strip().lower() == "letterbox":
        scale = min(float(model_w) / float(orig_w), float(model_h) / float(orig_h))
        resized_w = float(orig_w) * scale
        resized_h = float(orig_h) * scale
        pad_x = (float(model_w) - resized_w) * 0.5
        pad_y = (float(model_h) - resized_h) * 0.5
        values[0] = values[0] * scale + pad_x
        values[2] = values[2] * scale + pad_x
        values[1] = values[1] * scale + pad_y
        values[3] = values[3] * scale + pad_y
    else:
        values[0] = values[0] * (float(model_w) / float(orig_w))
        values[2] = values[2] * (float(model_w) / float(orig_w))
        values[1] = values[1] * (float(model_h) / float(orig_h))
        values[3] = values[3] * (float(model_h) / float(orig_h))
    values[0] = max(0.0, min(float(model_w), values[0]))
    values[2] = max(0.0, min(float(model_w), values[2]))
    values[1] = max(0.0, min(float(model_h), values[1]))
    values[3] = max(0.0, min(float(model_h), values[3]))
    return values


def _project_labels_to_model_input(
    labels: Mapping[str, Any],
    *,
    record: StoredSampleRecord,
) -> dict[str, Any]:
    original_size = _image_size_from_value(getattr(record, "input_image_size", None))
    model_input_size = _model_input_size_from_shape(
        getattr(record, "input_tensor_shape", None)
    )
    if original_size is None or model_input_size is None:
        return dict(labels)
    if original_size == model_input_size:
        return dict(labels)
    projected = dict(labels)
    projected["boxes"] = [
        _project_box_to_model_input(
            box,
            original_size=original_size,
            model_input_size=model_input_size,
            resize_mode=str(getattr(record, "input_resize_mode", "") or "direct_resize"),
        )
        for box in list(labels.get("boxes") or [])
    ]
    return projected


def _training_labels(result: Mapping[str, Any], record: StoredSampleRecord) -> dict[str, Any]:
    labels = _project_labels_to_model_input(
        {
            "boxes": list(result.get("boxes") or []),
            "labels": list(result.get("labels") or []),
        },
        record=record,
    )
    return {
        "boxes": list(labels.get("boxes") or []),
        "labels": list(labels.get("labels") or []),
    }


def pack_high_quality_sync_bundle(
    sample_store: EdgeSampleStore,
    records: Sequence[StoredSampleRecord],
    *,
    edge_id: int,
    shard_size: int,
    request_id: str | None = None,
    split_context: Mapping[str, Any] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    zip_path, manifest, _stats = pack_high_quality_sync_bundle_to_file(
        sample_store,
        records,
        edge_id=edge_id,
        shard_size=shard_size,
        request_id=request_id,
        split_context=split_context,
    )
    try:
        with open(zip_path, "rb") as handle:
            return handle.read(), manifest
    finally:
        try:
            os.remove(zip_path)
        except OSError:
            pass


def pack_high_quality_sync_bundle_to_file(
    sample_store: EdgeSampleStore,
    records: Sequence[StoredSampleRecord],
    *,
    edge_id: int,
    shard_size: int,
    request_id: str | None = None,
    split_context: Mapping[str, Any] | None = None,
    output_dir: str | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    selected = [
        record
        for record in records
        if record.quality_bucket == HIGH_QUALITY and record.feature_relpath is not None
    ]
    resolved_request_id = str(request_id or uuid.uuid4().hex)
    resolved_shard_size = max(1, int(shard_size))
    context = dict(split_context or {})
    first_record = selected[0] if selected else None
    model_id = str(context.get("model_id") or getattr(first_record, "model_id", "") or "")
    model_version = str(
        context.get("model_version") or getattr(first_record, "model_version", "") or ""
    )
    split_config_id = str(
        context.get("split_config_id")
        or getattr(first_record, "split_config_id", "")
        or ""
    )
    split_label = context.get("split_label")
    boundary_tensor_labels = [
        str(label) for label in list(context.get("boundary_tensor_labels", []) or [])
    ]
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f"sample_sync_edge_{int(edge_id)}_",
        suffix=".zip",
        dir=output_dir,
        delete=False,
    )
    zip_path = handle.name
    handle.close()

    manifest: dict[str, Any] = {
        "protocol_version": HIGH_QUALITY_SYNC_PROTOCOL_VERSION,
        "edge_id": int(edge_id),
        "model_id": model_id,
        "model_version": model_version,
        "split_config_id": split_config_id,
        "split_label": None if split_label is None else str(split_label),
        "boundary_tensor_labels": boundary_tensor_labels,
        "request_id": resolved_request_id,
        "created_at": _utc_now(),
        "shard_size": resolved_shard_size,
        "shards": [],
    }

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for shard_index, shard_records in enumerate(_chunks(selected, resolved_shard_size), 1):
                shard_id = f"edge{int(edge_id)}_high_{shard_index:06d}"
                feature_name = f"feature_shards/high_feature_shard_{shard_index:06d}.pt"
                label_name = f"label_shards/high_label_shard_{shard_index:06d}.jsonl"
                feature_payload = {"schema_version": 1, "samples": {}}
                label_lines = []
                sample_ids = []
                for record in shard_records:
                    sample_id = str(record.sample_id)
                    intermediate = sample_store.load_intermediate(record)
                    result = sample_store.load_inference_result(record)
                    feature_payload["samples"][sample_id] = _feature_sample_payload(intermediate)
                    labels = _training_labels(result, record)
                    label_lines.append(
                        json.dumps(
                            {
                                "sample_id": sample_id,
                                "boxes": labels["boxes"],
                                "labels": labels["labels"],
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                    sample_ids.append(sample_id)

                feature_buffer = io.BytesIO()
                torch.save(feature_payload, feature_buffer)
                zf.writestr(
                    feature_name,
                    feature_buffer.getvalue(),
                    compress_type=zipfile.ZIP_STORED,
                )
                zf.writestr(
                    label_name,
                    ("\n".join(label_lines) + ("\n" if label_lines else "")).encode("utf-8"),
                    compress_type=zipfile.ZIP_STORED,
                )
                manifest["shards"].append(
                    {
                        "shard_id": shard_id,
                        "feature_file": feature_name,
                        "label_file": label_name,
                        "sample_count": len(sample_ids),
                    }
                )
            zf.writestr(
                "bundle_manifest.json",
                json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
                compress_type=zipfile.ZIP_STORED,
            )
        stats = {
            "sample_count": len(selected),
            "shard_count": len(manifest["shards"]),
            "zip_path": zip_path,
            "zip_payload_bytes": os.path.getsize(zip_path),
        }
        return zip_path, manifest, stats
    except Exception:
        try:
            os.remove(zip_path)
        except OSError:
            pass
        raise


class HighQualitySampleSyncer:
    def __init__(
        self,
        sample_store: EdgeSampleStore,
        *,
        server_ip: str,
        edge_id: int,
        sample_pool_config: object | None = None,
        shard_size: int | None = None,
        sync_interval_sec: float | None = None,
        enabled: bool = True,
        context_provider: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.sample_store = sample_store
        self.server_ip = str(server_ip)
        self.edge_id = int(edge_id)
        self.shard_size = max(
            1,
            int(
                getattr(
                    sample_pool_config,
                    "shard_size",
                    64 if shard_size is None else shard_size,
                )
            ),
        )
        self.sync_interval_sec = max(
            0.1,
            float(
                getattr(
                    sample_pool_config,
                    "sync_interval_sec",
                    30.0 if sync_interval_sec is None else sync_interval_sec,
                )
            ),
        )
        self.enabled = bool(getattr(sample_pool_config, "enabled", enabled))
        self.ledger_path = os.path.join(self.sample_store.root_dir, UPLOAD_LEDGER_FILENAME)
        self._ledger_lock = threading.RLock()
        self._condition = threading.Condition()
        self._flush_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._context_provider = context_provider

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"edge-high-quality-sync-{self.edge_id}",
            daemon=False,
        )
        self._thread.start()
        self.enqueue_existing_high_quality()

    def notify_sample(self, record: StoredSampleRecord) -> None:
        if not self.enabled or record.quality_bucket != HIGH_QUALITY:
            return
        self._mark_samples([str(record.sample_id)], UPLOAD_PENDING)
        with self._condition:
            self._condition.notify()

    def enqueue_existing_high_quality(self) -> None:
        if not self.enabled:
            return
        sample_ids = []
        for record in self.sample_store.list_records(quality_bucket=HIGH_QUALITY):
            state = self._sample_state(str(record.sample_id))
            if state != UPLOAD_COMMITTED:
                sample_ids.append(str(record.sample_id))
        if sample_ids:
            self._mark_samples(sample_ids, UPLOAD_PENDING, preserve_attempts=True)
            with self._condition:
                self._condition.notify()

    def flush(self, *, timeout: float | None = None, include_partial: bool = True) -> bool:
        if not self.enabled:
            return True
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        if not self._acquire_flush_lock(deadline):
            return False
        try:
            record_groups = self._select_retryable_record_groups(
                include_partial=include_partial
            )
            if not record_groups:
                return True
            all_ok = True
            for records in record_groups:
                if deadline is not None and time.monotonic() >= deadline:
                    return False
                if not self._flush_record_group(records, deadline=deadline):
                    all_ok = False
            return all_ok
        finally:
            self._flush_lock.release()

    def _flush_record_group(
        self,
        records: Sequence[StoredSampleRecord],
        *,
        deadline: float | None,
    ) -> bool:
        if not records:
            return True
        request_id = uuid.uuid4().hex
        sample_ids = [str(record.sample_id) for record in records]
        shard_by_sample = {
            str(record.sample_id): f"edge{self.edge_id}_high_{(index // self.shard_size) + 1:06d}"
            for index, record in enumerate(records)
        }
        zip_path = ""
        try:
            split_context = self._split_context_for_records(records)
            zip_path, manifest, stats = pack_high_quality_sync_bundle_to_file(
                self.sample_store,
                records,
                edge_id=self.edge_id,
                shard_size=self.shard_size,
                request_id=request_id,
                split_context=split_context,
                output_dir=os.path.join(self.sample_store.root_dir, "sync_tmp"),
            )
            if deadline is not None and time.monotonic() >= deadline:
                return False
            with open(zip_path, "rb") as handle:
                payload_zip = handle.read()
            reply = transmit.submit_sample_sync(
                self.server_ip,
                edge_id=self.edge_id,
                request_id=request_id,
                protocol_version=manifest["protocol_version"],
                sync_type="HIGH_QUALITY_FEATURE_LABEL_SHARD",
                model_id=str(manifest.get("model_id", "")),
                model_version=str(manifest.get("model_version", "")),
                split_config_id=str(manifest.get("split_config_id", "")),
                payload_zip=payload_zip,
            )
            if not _reply_succeeded(reply):
                self._mark_samples(sample_ids, UPLOAD_FAILED, error=_reply_message(reply))
                return False
            self._mark_samples(
                sample_ids,
                UPLOAD_COMMITTED,
                shard_by_sample=shard_by_sample,
            )
            partial = any(
                int(shard.get("sample_count", 0) or 0) < self.shard_size
                for shard in list(manifest.get("shards", []) or [])
            )
            logger.info(
                "[ShardCL][Upload] high-quality sync edge_id={} "
                "shard_size={} samples={} shards={} partial={} payload_bytes={}",
                self.edge_id,
                self.shard_size,
                len(sample_ids),
                int(stats.get("shard_count", 0)),
                partial,
                int(stats.get("zip_payload_bytes", 0)),
            )
            return True
        except Exception as exc:
            self._mark_samples(sample_ids, UPLOAD_FAILED, error=str(exc))
            logger.exception("High-quality sample sync failed: {}", exc)
            return False
        finally:
            if zip_path:
                try:
                    os.remove(zip_path)
                except OSError:
                    pass

    def close(self, *, timeout: float | None = None) -> bool:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        thread = self._thread
        if thread is not None and thread.is_alive():
            join_timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(timeout=join_timeout)
        flush_timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
        flushed = self.flush(timeout=flush_timeout, include_partial=True)
        thread_stopped = thread is None or not thread.is_alive()
        return flushed and thread_stopped

    def _run(self) -> None:
        next_partial_flush = time.monotonic() + self.sync_interval_sec
        while not self._stop_event.is_set():
            action: str | None = None
            with self._condition:
                pending_count = self._retryable_count()
                full_group_count = self._retryable_full_group_count()
                now = time.monotonic()
                if full_group_count >= self.shard_size:
                    action = "full"
                elif pending_count > 0 and now >= next_partial_flush:
                    action = "partial"
                else:
                    if pending_count == 0:
                        next_partial_flush = now + self.sync_interval_sec
                    wait_for = max(0.1, next_partial_flush - now)
                    self._condition.wait(timeout=wait_for)
                    continue

            if action == "full":
                self.flush(timeout=self.sync_interval_sec, include_partial=False)
            elif action == "partial":
                self.flush(timeout=self.sync_interval_sec, include_partial=True)
            next_partial_flush = time.monotonic() + self.sync_interval_sec

    def _acquire_flush_lock(self, deadline: float | None) -> bool:
        if deadline is None:
            self._flush_lock.acquire()
            return True
        timeout = max(0.0, deadline - time.monotonic())
        return self._flush_lock.acquire(timeout=timeout)

    def _empty_ledger(self) -> dict[str, Any]:
        return {
            "schema_version": UPLOAD_LEDGER_VERSION,
            "created_at": _utc_now(),
            "samples": {},
        }

    def _load_ledger_unlocked(self) -> dict[str, Any]:
        if not os.path.exists(self.ledger_path):
            return self._empty_ledger()
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            logger.warning("Ignoring unreadable sample upload ledger at {}.", self.ledger_path)
            return self._empty_ledger()
        if not isinstance(payload, dict):
            return self._empty_ledger()
        samples = payload.get("samples")
        if not isinstance(samples, dict):
            payload["samples"] = {}
        payload.setdefault("schema_version", UPLOAD_LEDGER_VERSION)
        return payload

    def _write_ledger_unlocked(self, payload: Mapping[str, Any]) -> None:
        _atomic_json_dump(self.ledger_path, payload)

    def _sample_state(self, sample_id: str) -> str | None:
        with self._ledger_lock:
            entry = self._load_ledger_unlocked().get("samples", {}).get(str(sample_id))
        if isinstance(entry, Mapping):
            return str(entry.get("sync_state") or entry.get("state") or "")
        return None

    def _split_context_for_records(self, records: Sequence[StoredSampleRecord]) -> dict[str, Any]:
        provider_context: dict[str, Any] = {}
        if callable(self._context_provider):
            try:
                provider_context.update(dict(self._context_provider() or {}))
            except Exception as exc:
                logger.warning("High-quality sync context provider failed: {}", exc)
        first_record = records[0] if records else None
        context = {
            "model_id": _first_nonempty(
                getattr(first_record, "model_id", ""),
                provider_context.get("model_id"),
            ),
            "model_version": _first_nonempty(
                getattr(first_record, "model_version", ""),
                provider_context.get("model_version"),
            ),
            "split_config_id": _first_nonempty(
                getattr(first_record, "split_config_id", ""),
                provider_context.get("split_config_id"),
            ),
        }
        provider_split = str(provider_context.get("split_config_id") or "").strip()
        if provider_split and provider_split == context["split_config_id"]:
            context["split_label"] = provider_context.get("split_label")
            context["boundary_tensor_labels"] = list(
                provider_context.get("boundary_tensor_labels", []) or []
            )
        return context

    def _mark_samples(
        self,
        sample_ids: Sequence[str],
        state: str,
        *,
        error: str | None = None,
        preserve_attempts: bool = False,
        shard_by_sample: Mapping[str, str] | None = None,
    ) -> None:
        if not sample_ids:
            return
        now = _utc_now()
        with self._ledger_lock:
            ledger = self._load_ledger_unlocked()
            samples = ledger.setdefault("samples", {})
            for sample_id in sample_ids:
                key = str(sample_id)
                previous = samples.get(key) if isinstance(samples.get(key), Mapping) else {}
                attempts = int(previous.get("attempts", 0) or 0)
                if state == UPLOAD_PENDING and not preserve_attempts:
                    attempts += 1
                entry = {
                    "sample_id": key,
                    "sync_state": state,
                    "attempts": attempts,
                    "updated_at": now,
                }
                shard_id = (
                    str(shard_by_sample[key])
                    if shard_by_sample and key in shard_by_sample
                    else previous.get("shard_id")
                )
                if shard_id:
                    entry["shard_id"] = shard_id
                if state in {UPLOAD_UPLOADED, UPLOAD_COMMITTED}:
                    entry["uploaded_at"] = previous.get("uploaded_at") or now
                elif previous.get("uploaded_at"):
                    entry["uploaded_at"] = previous.get("uploaded_at")
                if state == UPLOAD_COMMITTED:
                    entry["committed_at"] = now
                elif previous.get("committed_at"):
                    entry["committed_at"] = previous.get("committed_at")
                if error:
                    entry["last_error"] = str(error)
                elif previous.get("last_error") and state != UPLOAD_COMMITTED:
                    entry["last_error"] = previous.get("last_error")
                samples[key] = entry
            ledger["updated_at"] = now
            self._write_ledger_unlocked(ledger)

    def _retryable_count(self) -> int:
        with self._ledger_lock:
            samples = self._load_ledger_unlocked().get("samples", {})
            return sum(
                1
                for entry in samples.values()
                if isinstance(entry, Mapping)
                and str(entry.get("sync_state") or entry.get("state")) in _RETRYABLE_STATES
            )

    @staticmethod
    def _record_context_key(record: StoredSampleRecord) -> tuple[str, str, str]:
        return (
            str(getattr(record, "model_id", "") or ""),
            str(getattr(record, "model_version", "") or ""),
            str(getattr(record, "split_config_id", "") or ""),
        )

    def _retryable_records_by_context(self) -> dict[tuple[str, str, str], list[StoredSampleRecord]]:
        with self._ledger_lock:
            samples = dict(self._load_ledger_unlocked().get("samples", {}))
        groups: dict[tuple[str, str, str], list[StoredSampleRecord]] = {}
        for record in self.sample_store.list_records(quality_bucket=HIGH_QUALITY):
            entry = samples.get(str(record.sample_id)) or {}
            state = str(entry.get("sync_state") or entry.get("state") or "")
            if state not in _RETRYABLE_STATES:
                continue
            groups.setdefault(self._record_context_key(record), []).append(record)
        return groups

    def _retryable_full_group_count(self) -> int:
        groups = self._retryable_records_by_context()
        return max(
            ((len(records) // self.shard_size) * self.shard_size for records in groups.values()),
            default=0,
        )

    def _select_retryable_record_groups(
        self,
        *,
        include_partial: bool,
    ) -> list[list[StoredSampleRecord]]:
        groups = self._retryable_records_by_context()
        if include_partial:
            return [records for records in groups.values() if records]
        record_groups: list[list[StoredSampleRecord]] = []
        for records in groups.values():
            full_count = (len(records) // self.shard_size) * self.shard_size
            if full_count:
                record_groups.append(records[:full_count])
        return record_groups


def _reply_succeeded(reply: object | None) -> bool:
    if reply is None:
        return False
    for field_name in ("success", "accepted", "committed"):
        if hasattr(reply, field_name):
            return bool(getattr(reply, field_name))
    status = str(getattr(reply, "status", "") or "").upper()
    return status not in {"FAILED", "ERROR", "REJECTED"}


def _reply_message(reply: object | None) -> str:
    if reply is None:
        return "sync_samples returned no reply"
    return str(getattr(reply, "message", "") or getattr(reply, "status", "") or "")


def _reply_sample_ids(
    reply: object | None,
    field_name: str,
    *,
    default: Sequence[str],
) -> list[str]:
    if reply is None or not hasattr(reply, field_name):
        return [str(sample_id) for sample_id in default]
    value = getattr(reply, field_name)
    return [str(sample_id) for sample_id in (value or [])]
