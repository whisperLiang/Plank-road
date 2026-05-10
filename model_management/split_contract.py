from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import torch


SPLIT_RUNTIME_CONTRACT_VERSION = "split-runtime-contract.v1"


def _stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sanitize_segment(value: object) -> str:
    text = str(value or "").strip()
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text) or "unknown"


def _atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=directory,
        delete=False,
    )
    try:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        handle.close()
        os.replace(handle.name, path)
    finally:
        try:
            handle.close()
        except Exception:
            pass
        if os.path.exists(handle.name):
            try:
                os.remove(handle.name)
            except OSError:
                pass


def contract_path(
    root_dir: str,
    *,
    edge_id: int | str,
    model_id: str,
    split_config_id: str,
) -> str:
    return os.path.join(
        root_dir,
        f"edge_{_sanitize_segment(edge_id)}",
        _sanitize_segment(model_id),
        f"{_sanitize_segment(split_config_id)}.json",
    )


def normalise_feature_tensors(value: object) -> dict[str, torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return {"payload": value.detach().cpu()}
    if not isinstance(value, Mapping):
        raise TypeError(f"Unsupported feature payload: {type(value).__name__}")
    source = value.get("feature") if isinstance(value.get("feature"), Mapping) else value
    if isinstance(source, Mapping) and isinstance(source.get("tensors"), Mapping):
        source = source["tensors"]
    tensors = {
        str(label): tensor.detach().cpu()
        for label, tensor in dict(source or {}).items()
        if isinstance(tensor, torch.Tensor)
    }
    if not tensors:
        raise ValueError("Feature payload did not contain any tensors.")
    return tensors


def feature_layout_from_tensors(
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    layout: dict[str, dict[str, Any]] = {}
    for label, tensor in sorted(dict(tensors).items()):
        if not isinstance(tensor, torch.Tensor):
            continue
        shape = [int(dim) for dim in tensor.shape]
        layout[str(label)] = {
            "dtype": str(tensor.dtype),
            "shape_without_batch": shape[1:] if shape else [],
        }
    if not layout:
        raise ValueError("Cannot compute feature layout without tensor features.")
    return layout


def feature_layout_id(layout: Mapping[str, Mapping[str, Any]]) -> str:
    return hashlib.sha1(_stable_json(layout).encode("utf-8")).hexdigest()


def feature_layout_matches(
    tensors: Mapping[str, torch.Tensor],
    layout: Mapping[str, Mapping[str, Any]],
) -> bool:
    actual = feature_layout_from_tensors(tensors)
    return _stable_json(actual) == _stable_json(layout)


@dataclass
class SplitRuntimeContract:
    contract_version: str
    edge_id: str
    model_id: str
    split_config_id: str
    canonical_split_key: str
    edge_split_id: str
    cloud_batch_split_id: str
    input_tensor_shape: list[int]
    input_resize_mode: str
    boundary_tensor_labels: list[str]
    feature_layout_id: str
    front_version: str
    tail_version: str | None = None
    feature_layout: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        edge_id: int | str,
        model_id: str,
        split_config_id: str,
        canonical_split_key: str,
        edge_split_id: str,
        cloud_batch_split_id: str,
        input_tensor_shape: list[int] | tuple[int, ...],
        input_resize_mode: str,
        boundary_tensor_labels: list[str] | tuple[str, ...],
        front_version: str,
        feature_tensors: Mapping[str, torch.Tensor],
        tail_version: str | None = None,
    ) -> "SplitRuntimeContract":
        layout = feature_layout_from_tensors(feature_tensors)
        return cls(
            contract_version=SPLIT_RUNTIME_CONTRACT_VERSION,
            edge_id=str(edge_id),
            model_id=str(model_id),
            split_config_id=str(split_config_id),
            canonical_split_key=str(canonical_split_key),
            edge_split_id=str(edge_split_id),
            cloud_batch_split_id=str(cloud_batch_split_id),
            input_tensor_shape=[int(dim) for dim in input_tensor_shape],
            input_resize_mode=str(input_resize_mode or "direct_resize"),
            boundary_tensor_labels=[str(label) for label in boundary_tensor_labels],
            feature_layout_id=feature_layout_id(layout),
            front_version=str(front_version or "0"),
            tail_version=None if tail_version is None else str(tail_version),
            feature_layout=layout,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SplitRuntimeContract":
        return cls(
            contract_version=str(
                payload.get("contract_version") or SPLIT_RUNTIME_CONTRACT_VERSION
            ),
            edge_id=str(payload["edge_id"]),
            model_id=str(payload["model_id"]),
            split_config_id=str(payload["split_config_id"]),
            canonical_split_key=str(payload["canonical_split_key"]),
            edge_split_id=str(payload["edge_split_id"]),
            cloud_batch_split_id=str(payload["cloud_batch_split_id"]),
            input_tensor_shape=[int(dim) for dim in payload.get("input_tensor_shape", [])],
            input_resize_mode=str(payload.get("input_resize_mode") or "direct_resize"),
            boundary_tensor_labels=[
                str(label) for label in list(payload.get("boundary_tensor_labels", []) or [])
            ],
            feature_layout_id=str(payload["feature_layout_id"]),
            front_version=str(payload.get("front_version") or "0"),
            tail_version=(
                None if payload.get("tail_version") is None else str(payload.get("tail_version"))
            ),
            feature_layout={
                str(label): dict(spec)
                for label, spec in dict(payload.get("feature_layout") or {}).items()
                if isinstance(spec, Mapping)
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, root_dir: str) -> str:
        path = contract_path(
            root_dir,
            edge_id=self.edge_id,
            model_id=self.model_id,
            split_config_id=self.split_config_id,
        )
        _atomic_write_json(path, self.to_dict())
        return path

    @classmethod
    def load(
        cls,
        root_dir: str,
        *,
        edge_id: int | str,
        model_id: str,
        split_config_id: str,
    ) -> "SplitRuntimeContract" | None:
        path = contract_path(
            root_dir,
            edge_id=edge_id,
            model_id=model_id,
            split_config_id=split_config_id,
        )
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise TypeError(f"Unsupported SplitRuntimeContract payload at {path}")
        return cls.from_dict(payload)

    def validate_reference(self, *, split_config_id: str, front_version: str) -> str | None:
        if str(split_config_id) != self.split_config_id:
            return "contract_mismatch"
        if str(front_version or "0") != self.front_version:
            return "front_version_mismatch"
        return None

    def validate_feature_layout(self, tensors: Mapping[str, torch.Tensor]) -> bool:
        return feature_layout_matches(tensors, self.feature_layout)


__all__ = [
    "SPLIT_RUNTIME_CONTRACT_VERSION",
    "SplitRuntimeContract",
    "contract_path",
    "feature_layout_from_tensors",
    "feature_layout_id",
    "feature_layout_matches",
    "normalise_feature_tensors",
]
