from __future__ import annotations

import pytest

from cloud.contracts import LOW_QUALITY_TRIGGER_PROTOCOL_VERSION, validate_low_quality_manifest
from model_management.fixed_split import FIXED_SPLIT_PLAN_VERSION
from model_management.split_contract import build_runtime_contract


def _runtime_contract() -> dict[str, object]:
    return build_runtime_contract(
        logical_split_id="after:test",
        trace_signature="trace-a",
        trace_device_type="cpu",
        runtime_backend="torchlens_native",
        boundary_tensor_labels=["boundary"],
        boundary_schema={
            "boundary": {
                "canonical_id": "boundary",
                "torchlens_label": "boundary",
                "module_path": "fake",
                "op_type": "conv",
                "shape": (1, 2, 3),
                "dtype": "torch.float32",
                "requires_grad": False,
                "role": "primary",
                "output_index": None,
                "device_policy": "runtime",
            }
        },
        model_id="yolo26n",
        model_version="1",
        input_tensor_shape=[1, 3, 32, 32],
        input_resize_mode="direct_resize",
        feature_layout={"boundary": {"shape": [1, 2, 3], "dtype": "float32"}},
    )


def _low_quality_manifest() -> dict[str, object]:
    runtime_contract = _runtime_contract()
    return {
        "protocol_version": LOW_QUALITY_TRIGGER_PROTOCOL_VERSION,
        "model_id": "yolo26n",
        "model_version": "1",
        "split_config_id": "split-a",
        "input_tensor_shape": [1, 3, 32, 32],
        "input_resize_mode": "direct_resize",
        "runtime_contract": runtime_contract,
        "split_plan": {
            "plan_version": FIXED_SPLIT_PLAN_VERSION,
            "split_config_id": "split-a",
            "runtime_contract": runtime_contract,
        },
    }


def test_low_quality_manifest_requires_runtime_contract() -> None:
    manifest = _low_quality_manifest()
    manifest.pop("runtime_contract")

    with pytest.raises(RuntimeError, match="runtime_contract"):
        validate_low_quality_manifest(manifest)


def test_low_quality_manifest_rejects_old_fixed_split_plan() -> None:
    manifest = _low_quality_manifest()
    manifest["split_plan"] = {
        **dict(manifest["split_plan"]),
        "plan_version": "fixed-split.v9",
    }

    with pytest.raises(RuntimeError, match="Unsupported fixed split plan version"):
        validate_low_quality_manifest(manifest)
