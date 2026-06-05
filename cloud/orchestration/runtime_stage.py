from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from cloud.contracts import validate_fixed_split_plan


FIXED_SPLIT_DYNAMIC_BATCH = (2, 64)
FIXED_SPLIT_DYNAMIC_BATCH_MIN = FIXED_SPLIT_DYNAMIC_BATCH[0]
FIXED_SPLIT_DYNAMIC_BATCH_MAX = FIXED_SPLIT_DYNAMIC_BATCH[1]


def _json_fingerprint(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def fixed_split_plan_runtime_contract(split_plan: Mapping[str, object]) -> dict[str, object]:
    return validate_fixed_split_plan(split_plan)


def fixed_split_boundary_from_plan(split_plan: Mapping[str, object]) -> str:
    boundary = fixed_split_plan_runtime_contract(split_plan).get("logical_split_id") or "auto"
    boundary = str(boundary)
    if boundary != "auto" and not boundary.startswith("after:"):
        boundary = f"after:{boundary}"
    return boundary


def fixed_split_dynamic_batch_from_plan(
    split_plan: Mapping[str, object],
    default: tuple[int, int] | None,
) -> tuple[int, int] | None:
    raw = split_plan.get("dynamic_batch")
    if raw is None:
        return default
    try:
        lower, upper = list(raw)[:2]
    except (TypeError, ValueError):
        return default
    lower_int = max(1, int(lower))
    upper_int = max(lower_int, int(upper))
    return lower_int, upper_int


def fixed_split_trace_batch_mode_from_plan(split_plan: Mapping[str, object]) -> str:
    mode = str(split_plan.get("trace_batch_mode") or "").strip()
    return mode if mode in {"batch_1", "batch_gt1"} else "batch_gt1"


def fixed_split_trace_batch_size_from_plan(
    split_plan: Mapping[str, object],
    default: int,
) -> int:
    raw = split_plan.get("trace_batch_size")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return max(1, int(default))


def cloud_fixed_split_dynamic_batch(
    split_plan: Mapping[str, object],
    *,
    model_family: str | None,
) -> tuple[int, int] | None:
    family = str(model_family or "").lower()
    default = (
        (1, FIXED_SPLIT_DYNAMIC_BATCH_MAX)
        if family == "rfdetr"
        else FIXED_SPLIT_DYNAMIC_BATCH
    )
    return fixed_split_dynamic_batch_from_plan(split_plan, default)


def cloud_fixed_split_trace_batch_mode(
    split_plan: Mapping[str, object],
    *,
    model_family: str | None,
) -> str:
    if str(model_family or "").lower() == "rfdetr":
        return "batch_gt1"
    return fixed_split_trace_batch_mode_from_plan(split_plan)


def cloud_fixed_split_trace_batch_size(
    split_plan: Mapping[str, object],
    *,
    model_family: str | None,
    default: int,
) -> int:
    if str(model_family or "").lower() == "rfdetr":
        return max(FIXED_SPLIT_DYNAMIC_BATCH_MIN, int(default))
    return fixed_split_trace_batch_size_from_plan(split_plan, default)


def fixed_split_validation_batches(
    *,
    model_family: str | None,
    trace_batch_size: int,
    runtime_batch_size: int | None,
    dynamic_batch: tuple[int, int] | None,
) -> list[int]:
    if str(model_family or "").lower() != "rfdetr":
        return []
    lower, upper = dynamic_batch or FIXED_SPLIT_DYNAMIC_BATCH
    max_batch = min(
        int(upper),
        max(int(trace_batch_size), 4, int(runtime_batch_size or trace_batch_size)),
    )
    candidates = [int(trace_batch_size), 4, max_batch]
    if int(lower) <= 1:
        candidates.insert(0, 1)
    return sorted({batch for batch in candidates if int(lower) <= batch <= int(upper)})


def fixed_split_manifest_has_rebuildable_raw_samples(
    manifest: Mapping[str, object],
) -> bool:
    samples = [
        sample
        for sample in list(manifest.get("samples", []) or [])
        if isinstance(sample, Mapping)
    ]
    if not samples:
        return False
    return all(sample.get("raw_relpath") is not None for sample in samples)


def fixed_split_runtime_validation_signature(
    *,
    model_family: str | None,
    batch_sizes: list[int],
) -> str | None:
    if not batch_sizes:
        return None
    return _json_fingerprint(
        {
            "kind": "fixed-split-train-smoke",
            "version": 1,
            "model_family": str(model_family or ""),
            "batch_sizes": [int(batch_size) for batch_size in batch_sizes],
        }
    )


__all__ = [
    "FIXED_SPLIT_DYNAMIC_BATCH",
    "FIXED_SPLIT_DYNAMIC_BATCH_MAX",
    "FIXED_SPLIT_DYNAMIC_BATCH_MIN",
    "cloud_fixed_split_dynamic_batch",
    "cloud_fixed_split_trace_batch_mode",
    "cloud_fixed_split_trace_batch_size",
    "fixed_split_boundary_from_plan",
    "fixed_split_dynamic_batch_from_plan",
    "fixed_split_manifest_has_rebuildable_raw_samples",
    "fixed_split_plan_runtime_contract",
    "fixed_split_runtime_validation_signature",
    "fixed_split_trace_batch_mode_from_plan",
    "fixed_split_trace_batch_size_from_plan",
    "fixed_split_validation_batches",
]
