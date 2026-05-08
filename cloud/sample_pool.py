from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import torch

from model_management.payload import BoundaryPayload, SplitPayload


POOL_MANIFEST_FIELDS = (
    "model_id",
    "model_version",
    "split_config_id",
    "split_label",
    "boundary_tensor_labels",
)

_QUALITY_METADATA_FIELDS = {
    "quality_bucket",
    "quality_score",
    "risk_score",
    "risk_reasons",
    "evidence_count",
    "covered_evidence_count",
    "uncovered_evidence_count",
    "uncovered_evidence_rate",
    "candidate_uncovered_score",
    "motion_uncovered_score",
    "track_uncovered_score",
    "window_id",
    "in_drift_window",
}

_RAW_METADATA_FIELDS = {
    "raw_relpath",
    "raw_bytes",
    "raw_sha256",
    "source_raw_relpath",
    "source_raw_bytes",
    "source_raw_sha256",
    "raw_cache_key",
    "has_raw_sample",
    "frame_relpath",
    "frame_file_size",
    "input_image_size",
    "input_tensor_shape",
    "input_resize_mode",
}

_LABEL_FIELDS = {
    "pseudo_boxes",
    "pseudo_labels",
    "pseudo_scores",
}


def _stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _read_json(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _atomic_json_dump(path: str, payload: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{threading.get_ident()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def _atomic_torch_save(payload: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{threading.get_ident()}"
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _atomic_text_write(path: str, payload: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp-{threading.get_ident()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(payload)
    os.replace(tmp_path, path)


def _normalise_relpath(path: str) -> str:
    return str(path).replace("\\", "/")


def _resolve_relpath(root_dir: str, relpath: str) -> str:
    return os.path.join(root_dir, str(relpath).replace("/", os.sep))


def _pool_manifest_from_bundle_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    model_meta = dict(manifest.get("model", {}) or {})
    split_plan = dict(manifest.get("split_plan", {}) or {})
    return {
        "model_id": str(manifest.get("model_id") or model_meta.get("model_id", "") or ""),
        "model_version": str(
            manifest.get("model_version") or model_meta.get("model_version", "") or ""
        ),
        "split_config_id": str(
            manifest.get("split_config_id") or split_plan.get("split_config_id", "") or ""
        ),
        "split_label": (
            None
            if (manifest.get("split_label") if "split_label" in manifest else split_plan.get("split_label")) is None
            else str(manifest.get("split_label") if "split_label" in manifest else split_plan.get("split_label"))
        ),
        "boundary_tensor_labels": [
            str(label)
            for label in list(
                manifest.get("boundary_tensor_labels")
                or split_plan.get("boundary_tensor_labels", [])
                or []
            )
        ],
    }


def _coerce_boundary_payload(payload: Any) -> BoundaryPayload:
    if isinstance(payload, BoundaryPayload):
        return payload
    if isinstance(payload, torch.Tensor):
        return SplitPayload.from_mapping({"payload": payload}, primary_label="payload")
    if isinstance(payload, Mapping):
        if "tensors" in payload and isinstance(payload.get("tensors"), Mapping):
            tensors = dict(payload.get("tensors") or {})
            return SplitPayload.from_mapping(
                tensors,
                primary_label=next(reversed(tensors), None) if tensors else None,
            )
        return SplitPayload.from_mapping(dict(payload))
    raise TypeError(f"Unsupported split feature payload: {type(payload)!r}")


def _labels_from_result(result: Mapping[str, Any] | None) -> dict[str, list[Any]]:
    result = dict(result or {})
    labels = {
        "boxes": list(result.get("boxes") or []),
        "labels": list(result.get("labels") or []),
    }
    if "scores" in result:
        labels["scores"] = list(result.get("scores") or [])
    return labels


def _class_counts(labels: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in list(labels.get("labels") or []):
        key = str(label)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _dominant_class(class_counts: Mapping[str, int]) -> int | None:
    if not class_counts:
        return None
    label = sorted(
        ((int(count), str(label)) for label, count in class_counts.items()),
        key=lambda item: (-item[0], item[1]),
    )[0][1]
    try:
        return int(label)
    except (TypeError, ValueError):
        return None


def _object_count(labels: Mapping[str, Any]) -> int:
    boxes = list(labels.get("boxes") or [])
    label_values = list(labels.get("labels") or [])
    if boxes and label_values:
        return min(len(boxes), len(label_values))
    return max(len(boxes), len(label_values))


def _created_at_text(value: object | None = None) -> str:
    if value in (None, ""):
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    return str(value)


def _created_at_sort_value(value: object) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return 0.0


def _sanitize_feature_record(record: Mapping[str, Any]) -> dict[str, Any]:
    removed = _QUALITY_METADATA_FIELDS | _RAW_METADATA_FIELDS | _LABEL_FIELDS
    return {str(key): value for key, value in dict(record).items() if str(key) not in removed}


def _feature_record_from_tensors(
    sample_id: str,
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    clean_tensors = {
        str(label): tensor.detach().cpu()
        for label, tensor in dict(tensors or {}).items()
        if isinstance(tensor, torch.Tensor)
    }
    return {
        "sample_id": str(sample_id),
        "intermediate": SplitPayload.from_mapping(
            clean_tensors,
            primary_label=next(reversed(clean_tensors), None) if clean_tensors else None,
        ),
        "boundary_tensor_labels": list(clean_tensors.keys()),
        "split_label": next(reversed(clean_tensors), None) if clean_tensors else None,
    }


def _feature_sample_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    intermediate = dict(record).get("intermediate", record)
    boundary = _coerce_boundary_payload(intermediate)
    return {
        "tensors": {
            str(label): tensor.detach().cpu()
            for label, tensor in dict(boundary.tensors).items()
            if isinstance(tensor, torch.Tensor)
        }
    }


def _record_from_feature_payload(
    payload: Mapping[str, Any],
    *,
    sample: Mapping[str, Any],
    split_plan: Mapping[str, Any],
) -> dict[str, Any]:
    intermediate = _coerce_boundary_payload(payload.get("intermediate"))
    boundary_labels = list(
        getattr(
            intermediate,
            "boundary_tensor_labels",
            list(getattr(intermediate, "tensors", {}).keys()),
        )
        or []
    )
    return _sanitize_feature_record(
        {
            "intermediate": intermediate,
            "candidate_id": (
                getattr(intermediate, "candidate_id", None)
                or getattr(intermediate, "split_id", None)
                or split_plan.get("candidate_id")
            ),
            "boundary_tensor_labels": boundary_labels
            or list(split_plan.get("boundary_tensor_labels", []) or []),
            "split_index": getattr(intermediate, "split_index", None)
            or split_plan.get("split_index"),
            "split_label": (
                getattr(intermediate, "split_label", None)
                or getattr(intermediate, "split_id", None)
                or split_plan.get("split_label")
            ),
            "split_plan_candidate_id": split_plan.get("candidate_id"),
            "split_plan_split_index": split_plan.get("split_index"),
            "split_plan_split_label": split_plan.get("split_label"),
            "split_plan_boundary_tensor_labels": list(
                split_plan.get("boundary_tensor_labels", []) or []
            ),
            "sample_id": str(sample.get("sample_id", "")),
        }
    )


@dataclass(frozen=True)
class FeatureLabelRecord:
    sample_id: str
    feature_record: dict[str, Any]
    labels: dict[str, list[Any]]


class FeatureLabelShardReader:
    """LRU reader for cloud sample-pool feature and label shards."""

    def __init__(self, root_dir: str, *, cache_size: int = 4) -> None:
        self.root_dir = os.path.abspath(root_dir)
        self.cache_size = max(2, min(4, int(cache_size)))
        self._feature_cache: OrderedDict[str, Mapping[str, Any]] = OrderedDict()
        self._label_cache: OrderedDict[str, Mapping[str, Any]] = OrderedDict()

    def _load_shard(
        self,
        cache: OrderedDict[str, Mapping[str, Any]],
        relpath: str,
        key_name: str,
    ) -> Mapping[str, Any]:
        shard_key = _normalise_relpath(relpath)
        cached = cache.get(shard_key)
        if cached is not None:
            cache.move_to_end(shard_key)
            return cached
        shard_path = _resolve_relpath(self.root_dir, shard_key)
        if shard_key.endswith(".jsonl"):
            records = {}
            with open(shard_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if isinstance(entry, Mapping) and entry.get("sample_id"):
                        records[str(entry["sample_id"])] = dict(entry)
        else:
            payload = torch.load(
                shard_path,
                map_location="cpu",
                weights_only=False,
            )
            if isinstance(payload, Mapping) and key_name in payload:
                records = payload[key_name]
            elif key_name == "samples" and isinstance(payload, Mapping) and "records" in payload:
                records = payload["records"]
            else:
                records = payload
        if not isinstance(records, Mapping):
            raise TypeError(f"Unsupported shard payload in {shard_key!r}: {type(records)!r}")
        cache[shard_key] = records
        cache.move_to_end(shard_key)
        while len(cache) > self.cache_size:
            cache.popitem(last=False)
        return records

    def load_feature(self, shard: str, key: str) -> dict[str, Any]:
        records = self._load_shard(self._feature_cache, shard, "samples")
        value = records[str(key)]
        if not isinstance(value, Mapping):
            raise TypeError(f"Unsupported feature record for {key!r}: {type(value)!r}")
        if "tensors" in value:
            return _feature_record_from_tensors(str(key), dict(value.get("tensors") or {}))
        return dict(value)

    def load_label(self, shard: str, key: str) -> dict[str, list[Any]]:
        records = self._load_shard(self._label_cache, shard, "labels")
        value = records[str(key)]
        if not isinstance(value, Mapping):
            raise TypeError(f"Unsupported label record for {key!r}: {type(value)!r}")
        return _labels_from_result(value)

    def read(self, entry: Mapping[str, Any]) -> FeatureLabelRecord:
        sample_id = str(entry["sample_id"])
        feature_record = self.load_feature(
            str(entry["feature_shard"]),
            str(entry["feature_key"]),
        )
        labels = self.load_label(
            str(entry["label_shard"]),
            str(entry["label_key"]),
        )
        return FeatureLabelRecord(
            sample_id=sample_id,
            feature_record=feature_record,
            labels=labels,
        )

    def training_record(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        record = self.read(entry)
        training_record = dict(record.feature_record)
        training_record["pseudo_boxes"] = list(record.labels.get("boxes") or [])
        training_record["pseudo_labels"] = list(record.labels.get("labels") or [])
        if "scores" in record.labels:
            training_record["pseudo_scores"] = list(record.labels.get("scores") or [])
        return training_record


class CloudSamplePool:
    """Durable cloud-side pool of split features and trainable labels."""

    def __init__(
        self,
        root_dir: str,
        *,
        model_id: str | None = None,
        model_version: str | None = None,
        split_config_id: str | None = None,
        split_label: str | None = None,
        boundary_tensor_labels: list[str] | tuple[str, ...] | None = None,
        max_active_samples: int | None = None,
        max_samples: int | None = None,
        shard_size: int = 64,
        reader_cache_size: int = 4,
    ) -> None:
        self.root_dir = os.path.abspath(root_dir)
        self.db_path = os.path.join(self.root_dir, "samples.sqlite")
        self.manifest_path = os.path.join(self.root_dir, "pool_manifest.json")
        self.feature_dir = os.path.join(self.root_dir, "features")
        self.label_dir = os.path.join(self.root_dir, "labels")
        resolved_max_active = max_active_samples if max_active_samples is not None else max_samples
        self.max_active_samples = (
            None
            if resolved_max_active in (None, "", 0)
            else max(1, int(resolved_max_active))
        )
        self.shard_size = max(1, int(shard_size))
        self.reader = FeatureLabelShardReader(
            self.root_dir,
            cache_size=reader_cache_size,
        )
        self._lock = threading.RLock()
        self._replacement_index: dict[str, Any] = {}
        os.makedirs(self.feature_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)
        self._init_db()
        initial_manifest = {
            "model_id": str(model_id or ""),
            "model_version": str(model_version or ""),
            "split_config_id": str(split_config_id or ""),
            "split_label": split_label,
            "boundary_tensor_labels": [
                str(label) for label in list(boundary_tensor_labels or [])
            ],
        }
        if any(initial_manifest.get(field) for field in POOL_MANIFEST_FIELDS):
            self._ensure_manifest(initial_manifest)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _init_db(self) -> None:
        os.makedirs(self.root_dir, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS samples (
                    sample_id TEXT PRIMARY KEY,
                    feature_shard TEXT NOT NULL,
                    feature_key TEXT NOT NULL,
                    label_shard TEXT NOT NULL,
                    label_key TEXT NOT NULL,
                    object_count INTEGER NOT NULL,
                    class_counts_json TEXT NOT NULL,
                    dominant_class INTEGER,
                    created_at TEXT,
                    active INTEGER DEFAULT 1
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_samples_active_created "
                "ON samples(active, created_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_samples_active_dominant "
                "ON samples(active, dominant_class)"
            )

    def _read_manifest(self) -> dict[str, Any]:
        payload = _read_json(self.manifest_path)
        return {field: payload.get(field) for field in POOL_MANIFEST_FIELDS}

    def _ensure_manifest(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        expected = {
            "model_id": str(manifest.get("model_id", "") or ""),
            "model_version": str(manifest.get("model_version", "") or ""),
            "split_config_id": str(manifest.get("split_config_id", "") or ""),
            "split_label": manifest.get("split_label"),
            "boundary_tensor_labels": [
                str(label)
                for label in list(manifest.get("boundary_tensor_labels", []) or [])
            ],
        }
        existing = self._read_manifest()
        if not any(existing.get(field) for field in POOL_MANIFEST_FIELDS):
            _atomic_json_dump(self.manifest_path, expected)
            return expected

        merged = dict(existing)
        for field in POOL_MANIFEST_FIELDS:
            old_value = existing.get(field)
            new_value = expected.get(field)
            if old_value in (None, "", []):
                merged[field] = new_value
                continue
            if new_value in (None, "", []):
                continue
            if old_value != new_value:
                raise RuntimeError(
                    "Cloud sample pool manifest mismatch for "
                    f"{field}: existing={old_value!r}, incoming={new_value!r}."
                )
        _atomic_json_dump(self.manifest_path, merged)
        return merged

    @staticmethod
    def manifest_from_bundle(bundle_root: str) -> dict[str, Any]:
        manifest = _read_json(os.path.join(bundle_root, "bundle_manifest.json"))
        return _pool_manifest_from_bundle_manifest(manifest)

    def _next_shard_name(self, prefix: str, suffix: str) -> str:
        directory = self.feature_dir if prefix == "feature_shard" else self.label_dir
        existing = [
            name
            for name in os.listdir(directory)
            if name.startswith(f"{prefix}_") and name.endswith(suffix)
        ]
        return f"{prefix}_{len(existing) + 1:06d}{suffix}"

    def list_active_samples(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT sample_id, feature_shard, feature_key, label_shard, label_key,
                       object_count, class_counts_json, dominant_class, created_at, active
                FROM samples
                WHERE active = 1
                ORDER BY created_at ASC, sample_id ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def deactivate_sample(self, sample_id: str) -> bool:
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "UPDATE samples SET active = 0 WHERE sample_id = ? AND active = 1",
                (str(sample_id),),
            )
            changed = cursor.rowcount > 0
        if changed:
            self.rebuild_replacement_index()
        return bool(changed)

    def rebuild_replacement_index(self) -> dict[str, Any]:
        rows = self.list_active_samples()
        by_dominant: dict[str, list[dict[str, Any]]] = {}
        by_class: dict[str, list[dict[str, Any]]] = {}
        aggregate_counts: dict[str, int] = {}
        for row in rows:
            dominant = str(row.get("dominant_class") or "")
            by_dominant.setdefault(dominant, []).append(row)
            try:
                counts = json.loads(str(row.get("class_counts_json") or "{}"))
            except json.JSONDecodeError:
                counts = {}
            if isinstance(counts, Mapping):
                for label, count in counts.items():
                    class_id = str(label)
                    object_count = int(count)
                    aggregate_counts[class_id] = aggregate_counts.get(class_id, 0) + object_count
                    if object_count > 0:
                        by_class.setdefault(class_id, []).append(row)
        for bucket in list(by_dominant.values()) + list(by_class.values()):
            bucket.sort(
                key=lambda row: (
                    int(row.get("object_count") or 0),
                    _created_at_sort_value(row.get("created_at")),
                    str(row.get("sample_id") or ""),
                )
            )
        self._replacement_index = {
            "active_count": len(rows),
            "by_dominant": by_dominant,
            "by_class": by_class,
            "aggregate_counts": aggregate_counts,
        }
        return dict(self._replacement_index)

    def select_victim_for_new_sample(
        self,
        sample: Mapping[str, Any] | None = None,
        *,
        force: bool = False,
        **sample_fields: Any,
    ) -> str | None:
        if self.max_active_samples is None:
            return None
        candidate = dict(sample or {})
        candidate.update(sample_fields)
        labels = (
            candidate.get("labels")
            or candidate.get("label")
            or candidate.get("target")
        )
        if not isinstance(labels, Mapping):
            labels = {
                "boxes": candidate.get("boxes", []),
                "labels": candidate.get("labels", []),
                "scores": candidate.get("scores", []),
            }
        candidate_counts = _class_counts(labels)
        candidate_dominant = _dominant_class(candidate_counts) or ""
        index = self.rebuild_replacement_index()
        if not force and int(index.get("active_count", 0)) < int(self.max_active_samples):
            return None
        by_class = dict(index.get("by_class") or {})
        aggregate_counts = dict(index.get("aggregate_counts") or {})
        for class_id, _count in sorted(
            aggregate_counts.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        ):
            victim_bucket = list(by_class.get(str(class_id)) or [])
            if victim_bucket:
                return str(victim_bucket[0]["sample_id"])
        active_rows = self.list_active_samples()
        if not active_rows:
            return None
        active_rows.sort(
            key=lambda row: (
                int(row.get("object_count") or 0),
                _created_at_sort_value(row.get("created_at")),
                str(row.get("sample_id") or ""),
            )
        )
        return str(active_rows[0]["sample_id"])

    def _active_sample_count(self) -> int:
        with self._connection() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM samples WHERE active = 1"
                ).fetchone()[0]
            )

    def _active_sample_exists(self, sample_id: str) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM samples WHERE sample_id = ? AND active = 1",
                (str(sample_id),),
            ).fetchone()
        return row is not None

    def _enforce_capacity_for_pending(self, samples: list[Mapping[str, Any]]) -> int:
        if self.max_active_samples is None:
            return 0
        reserved_new_ids: set[str] = set()
        replacement_count = 0
        for sample in samples:
            sample_id = str(sample.get("sample_id", "") or "")
            if not sample_id:
                continue
            if sample_id in reserved_new_ids or self._active_sample_exists(sample_id):
                continue
            while (
                self._active_sample_count() + len(reserved_new_ids) + 1
                > int(self.max_active_samples)
            ):
                victim = self.select_victim_for_new_sample(sample, force=True)
                if victim is None or victim == sample_id:
                    break
                if not self.deactivate_sample(victim):
                    break
                replacement_count += 1
            reserved_new_ids.add(sample_id)
        return replacement_count

    def _trim_to_capacity(self) -> int:
        if self.max_active_samples is None:
            return 0
        replacement_count = 0
        while self._active_sample_count() > int(self.max_active_samples):
            victim = self.select_victim_for_new_sample({}, force=True)
            if victim is None:
                break
            if not self.deactivate_sample(victim):
                break
            replacement_count += 1
        return replacement_count

    def _prepare_db_rows(
        self,
        samples: list[Mapping[str, Any]],
        *,
        feature_shard: str,
        label_shard: str,
    ) -> list[tuple[Any, ...]]:
        rows = []
        for sample in samples:
            sample_id = str(sample["sample_id"])
            labels = _labels_from_result(sample.get("labels"))
            class_counts = _class_counts(labels)
            rows.append(
                (
                    sample_id,
                    feature_shard,
                    sample_id,
                    label_shard,
                    sample_id,
                    _object_count(labels),
                    _stable_json(class_counts),
                    _dominant_class(class_counts),
                    _created_at_text(sample.get("created_at")),
                    1,
                )
            )
        return rows

    def append_feature_label_shard(self, samples: list[Mapping[str, Any]]) -> int:
        prepared = []
        for sample in samples:
            if not sample.get("sample_id"):
                continue
            feature_record = (
                sample.get("feature_record")
                or sample.get("record")
                or sample.get("feature")
            )
            labels = (
                sample.get("labels")
                or sample.get("label")
                or sample.get("target")
            )
            if not isinstance(feature_record, Mapping) or not isinstance(labels, Mapping):
                continue
            sample_id = str(sample["sample_id"])
            prepared.append(
                {
                    "sample_id": sample_id,
                    "feature_record": _sanitize_feature_record(feature_record),
                    "labels": _labels_from_result(labels),
                    "created_at": _created_at_text(sample.get("created_at")),
                }
            )
        if not prepared:
            return 0

        feature_shard = _normalise_relpath(
            os.path.join("features", self._next_shard_name("feature_shard", ".pt"))
        )
        label_shard = _normalise_relpath(
            os.path.join("labels", self._next_shard_name("label_shard", ".jsonl"))
        )
        feature_payload = {
            "schema_version": 1,
            "samples": {
                sample["sample_id"]: _feature_sample_from_record(sample["feature_record"])
                for sample in prepared
            }
        }
        label_payload = "\n".join(
            json.dumps(
                {
                    "sample_id": sample["sample_id"],
                    "boxes": list(sample["labels"].get("boxes") or []),
                    "labels": list(sample["labels"].get("labels") or []),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            for sample in prepared
        )
        if label_payload:
            label_payload += "\n"
        rows = self._prepare_db_rows(
            prepared,
            feature_shard=feature_shard,
            label_shard=label_shard,
        )
        with self._lock:
            _atomic_torch_save(feature_payload, _resolve_relpath(self.root_dir, feature_shard))
            _atomic_text_write(_resolve_relpath(self.root_dir, label_shard), label_payload)
            with self._connection() as connection:
                connection.executemany(
                    """
                    INSERT OR REPLACE INTO samples (
                        sample_id, feature_shard, feature_key, label_shard, label_key,
                        object_count, class_counts_json, dominant_class, created_at, active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        self.rebuild_replacement_index()
        return len(prepared)

    def add_trainable_sample(
        self,
        sample: Mapping[str, Any] | None = None,
        **sample_fields: Any,
    ) -> str:
        payload = dict(sample or {})
        payload.update(sample_fields)
        sample_id = str(payload["sample_id"])
        self._enforce_capacity_for_pending([payload])
        self.append_feature_label_shard([payload])
        self._trim_to_capacity()
        return sample_id

    def ingest_low_quality_processed_samples(self, samples: list[Mapping[str, Any]]) -> int:
        pending: list[Mapping[str, Any]] = []
        for sample in samples:
            sample_id = str(sample.get("sample_id", "") or "")
            if not sample_id:
                continue
            pending.append(sample)
        self._enforce_capacity_for_pending(pending)
        added = 0
        for offset in range(0, len(pending), self.shard_size):
            added += self.append_feature_label_shard(
                list(pending[offset:offset + self.shard_size])
            )
        self._trim_to_capacity()
        return added

    def ingest_high_quality_feature_label_bundle(self, bundle_root: str) -> int:
        manifest_path = os.path.join(bundle_root, "bundle_manifest.json")
        manifest = _read_json(manifest_path)
        self._ensure_manifest(_pool_manifest_from_bundle_manifest(manifest))
        split_plan = dict(manifest.get("split_plan", {}) or {})
        trainable_samples: list[dict[str, Any]] = []
        shards = list(manifest.get("shards", []) or [])
        if shards:
            for shard in shards:
                if not isinstance(shard, Mapping):
                    continue
                feature_file = shard.get("feature_file") or shard.get("feature_shard")
                label_file = shard.get("label_file") or shard.get("label_shard")
                if not feature_file or not label_file:
                    continue
                feature_path = _resolve_relpath(bundle_root, str(feature_file))
                label_path = _resolve_relpath(bundle_root, str(label_file))
                if not os.path.exists(feature_path) or not os.path.exists(label_path):
                    continue
                feature_payload = torch.load(feature_path, map_location="cpu", weights_only=False)
                feature_samples = (
                    feature_payload.get("samples")
                    if isinstance(feature_payload, Mapping)
                    else None
                )
                if not isinstance(feature_samples, Mapping):
                    continue
                labels_by_id: dict[str, dict[str, Any]] = {}
                with open(label_path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        label_payload = json.loads(line)
                        if isinstance(label_payload, Mapping) and label_payload.get("sample_id"):
                            labels_by_id[str(label_payload["sample_id"])] = dict(label_payload)
                for sample_id, feature_value in feature_samples.items():
                    sample_key = str(sample_id)
                    if not isinstance(feature_value, Mapping):
                        continue
                    tensors = dict(feature_value.get("tensors") or {})
                    if not tensors or sample_key not in labels_by_id:
                        continue
                    trainable_samples.append(
                        {
                            "sample_id": sample_key,
                            "feature_record": _feature_record_from_tensors(sample_key, tensors),
                    "labels": _labels_from_result(labels_by_id[sample_key]),
                            "created_at": _created_at_text(),
                        }
                    )
        else:
            for sample in list(manifest.get("samples", []) or []):
                if not isinstance(sample, Mapping):
                    continue
                if str(sample.get("quality_bucket", "") or "") != "high_quality":
                    continue
                sample_id = str(sample.get("sample_id", "") or "")
                feature_relpath = sample.get("feature_relpath")
                if not sample_id or not feature_relpath:
                    continue
                feature_path = _resolve_relpath(bundle_root, str(feature_relpath))
                if not os.path.exists(feature_path):
                    continue
                payload = torch.load(feature_path, map_location="cpu", weights_only=False)
                if not isinstance(payload, Mapping):
                    continue
                result_payload = sample.get("inference_result")
                if not isinstance(result_payload, Mapping):
                    result_relpath = sample.get("result_relpath")
                    if result_relpath:
                        result_payload = _read_json(_resolve_relpath(bundle_root, str(result_relpath)))
                    else:
                        result_payload = {}
                trainable_samples.append(
                    {
                        "sample_id": sample_id,
                        "feature_record": _record_from_feature_payload(
                            payload,
                            sample=sample,
                            split_plan=split_plan,
                        ),
                        "labels": _labels_from_result(result_payload),
                        "created_at": _created_at_text(),
                    }
                )
        return self.ingest_low_quality_processed_samples(trainable_samples)

    def maybe_compact(self, *, force: bool = False) -> bool:
        with self._connection() as connection:
            active_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM samples WHERE active = 1"
                ).fetchone()[0]
            )
            inactive_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM samples WHERE active = 0"
                ).fetchone()[0]
            )
        if inactive_count <= 0:
            return False
        if not force and inactive_count < max(64, active_count):
            return False
        active_entries = self.list_active_samples()
        active_samples = []
        for entry in active_entries:
            record = self.reader.read(entry)
            active_samples.append(
                {
                    "sample_id": record.sample_id,
                    "feature_record": record.feature_record,
                    "labels": record.labels,
                    "created_at": _created_at_text(entry.get("created_at")),
                }
            )
        old_feature_dir = self.feature_dir
        old_label_dir = self.label_dir
        with self._lock:
            for directory in (old_feature_dir, old_label_dir):
                shutil.rmtree(directory, ignore_errors=True)
                os.makedirs(directory, exist_ok=True)
            with self._connection() as connection:
                connection.execute("DELETE FROM samples")
            self.reader = FeatureLabelShardReader(
                self.root_dir,
                cache_size=self.reader.cache_size,
            )
            self.append_feature_label_shard(active_samples)
        return True
