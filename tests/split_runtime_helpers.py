from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

from model_management.model_zoo import build_detection_model
from model_management.payload import SplitPayload


NEW_DETECTORS = ("rfdetr_nano", "tinynext_s")


def detector_kwargs(model_name: str) -> dict[str, object]:
    if model_name == "rfdetr_nano":
        return {
            "confidence": 0.01,
            "resolution": 96,
            "num_queries": 80,
        }
    return {"confidence": 0.01}


def build_public_detector(model_name: str):
    return build_detection_model(
        model_name,
        pretrained=False,
        device="cpu",
        **detector_kwargs(model_name),
    ).eval()


def collect_valid_candidates(
    splitter,
    candidates: Iterable,
    *,
    minimum_valid: int,
):
    valid = []
    for candidate in candidates:
        report = splitter.validate_candidate(candidate)
        if report["success"]:
            valid.append((candidate, report))
        if len(valid) >= minimum_valid:
            break
    assert len(valid) >= minimum_valid
    return valid


def payload_without_boundary(payload: SplitPayload, removed_label: str) -> SplitPayload:
    reduced = OrderedDict(
        (label, tensor)
        for label, tensor in payload.tensors.items()
        if label != removed_label
    )
    return SplitPayload(
        tensors=reduced,
        metadata=dict(payload.metadata),
        candidate_id=payload.candidate_id,
        boundary_tensor_labels=list(reduced.keys()),
        primary_label=next(reversed(reduced.keys())) if reduced else None,
        split_index=payload.split_index,
        split_label=payload.split_label,
    )
