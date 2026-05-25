from __future__ import annotations

import gzip
import os
from collections.abc import Mapping
from dataclasses import fields
from typing import Any

import torch
from ariadne import BoundaryPayload


BOUNDARY_CACHE_PROTOCOL = "ariadne-boundary-v2"


def _payload_field_names() -> set[str]:
    try:
        return {field.name for field in fields(BoundaryPayload)}
    except TypeError:
        return set(getattr(BoundaryPayload, "__annotations__", {}) or {})


def _new_payload_like(payload: BoundaryPayload, **changes: Any) -> BoundaryPayload:
    names = _payload_field_names()
    values = {
        name: getattr(payload, name)
        for name in names
        if hasattr(payload, name)
    }
    values.update(changes)
    if names:
        values = {key: value for key, value in values.items() if key in names}
    return BoundaryPayload(**values)


def _runtime_from(value: Any) -> Any:
    if value is None:
        return None
    ensure_runtime = getattr(value, "_ensure_runtime", None)
    if callable(ensure_runtime):
        try:
            return ensure_runtime()
        except RuntimeError:
            pass
    return getattr(value, "runtime", value)


def _runtime_batch_symbol(runtime: Any) -> str:
    shape_env = getattr(getattr(runtime, "trace_plan", None), "shape_env", None)
    return str(getattr(shape_env, "batch_symbol", None) or "B")


def _same_tensor(left: torch.Tensor, right: torch.Tensor) -> bool:
    if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype:
        return False
    try:
        return bool(torch.equal(left.detach().cpu(), right.detach().cpu()))
    except Exception:
        return False


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and _same_tensor(left, right)
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if tuple(left.keys()) != tuple(right.keys()):
            return False
        return all(_same_value(left[key], right[key]) for key in left.keys())
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(_same_value(a, b) for a, b in zip(left, right))
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same_value(a, b) for a, b in zip(left, right))
    return left == right


class BoundaryPayloadCacheCodec:
    def __init__(self, runtime: Any) -> None:
        self.runtime = _runtime_from(runtime)
        self.batch_symbol = _runtime_batch_symbol(self.runtime)

    def validate(self, payload: BoundaryPayload) -> BoundaryPayload:
        if not isinstance(payload, BoundaryPayload):
            raise TypeError(f"Expected BoundaryPayload, got {type(payload).__name__}.")
        validate_boundary = getattr(self.runtime, "validate_boundary", None)
        if callable(validate_boundary):
            validate_boundary(payload)
        return payload

    def split_batch(
        self,
        payload: BoundaryPayload,
        actual_batch_size: int | None = None,
    ) -> list[BoundaryPayload]:
        self.validate(payload)
        payload_batch_size = int(getattr(payload, "batch_size", 0) or 0)
        if payload_batch_size <= 0:
            raise RuntimeError("BoundaryPayload cache split requires a positive batch_size.")
        actual = payload_batch_size if actual_batch_size is None else int(actual_batch_size)
        if actual < 0 or actual > payload_batch_size:
            raise RuntimeError(
                "Cannot split "
                f"{actual} sample(s) from BoundaryPayload batch_size={payload_batch_size}."
            )

        schema = dict(getattr(payload, "schema", {}) or {})
        if not schema:
            raise RuntimeError("BoundaryPayload cache split requires Ariadne boundary schema.")

        sample_tensors: list[dict[str, torch.Tensor]] = [dict() for _ in range(actual)]
        for label, tensor in dict(getattr(payload, "tensors", {}) or {}).items():
            if not isinstance(tensor, torch.Tensor):
                continue
            label = str(label)
            spec = schema.get(label)
            if spec is None:
                raise RuntimeError(f"Boundary tensor {label!r} is missing from payload schema.")
            plan = self._tensor_batch_plan(label, tensor, spec, payload_batch_size)
            if plan is None:
                for sample in sample_tensors:
                    sample[label] = tensor
                continue
            axis, multiplier = plan
            for sample_index, sample in enumerate(sample_tensors):
                sample[label] = tensor.narrow(axis, sample_index * multiplier, multiplier)

        sample_passthrough = self._split_passthrough_value(
            dict(getattr(payload, "passthrough_inputs", {}) or {}),
            batch_size=payload_batch_size,
            actual_batch_size=actual,
        )
        sample_values = self._split_values(
            tuple(getattr(payload, "values", ()) or ()),
            tuple(getattr(payload, "value_schema", ()) or ()),
            batch_size=payload_batch_size,
            actual_batch_size=actual,
        )

        payloads = [
            _new_payload_like(
                payload,
                batch_size=1,
                tensors=sample_tensors[index],
                passthrough_inputs=sample_passthrough[index],
                values=sample_values[index],
            )
            for index in range(actual)
        ]
        for sample_payload in payloads:
            self.validate(sample_payload)
        return payloads

    def collate(self, payloads: list[BoundaryPayload]) -> BoundaryPayload:
        if not payloads:
            raise RuntimeError("Cannot collate an empty BoundaryPayload cache batch.")
        for payload in payloads:
            self.validate(payload)

        first = payloads[0]
        schema = dict(getattr(first, "schema", {}) or {})
        if not schema:
            raise RuntimeError("BoundaryPayload cache collate requires Ariadne boundary schema.")
        requires_grad = dict(getattr(first, "requires_grad", {}) or {})
        value_schema = tuple(getattr(first, "value_schema", ()) or ())

        for payload in payloads[1:]:
            if str(payload.split_id) != str(first.split_id):
                raise RuntimeError(
                    "Cannot collate BoundaryPayload objects with different split_id."
                )
            if str(payload.graph_signature) != str(first.graph_signature):
                raise RuntimeError(
                    "Cannot collate BoundaryPayload objects with different graph_signature."
                )
            if dict(getattr(payload, "schema", {}) or {}) != schema:
                raise RuntimeError("Cannot collate BoundaryPayload objects with different schema.")
            if dict(getattr(payload, "requires_grad", {}) or {}) != requires_grad:
                raise RuntimeError(
                    "Cannot collate BoundaryPayload objects with different requires_grad metadata."
                )
            if tuple(getattr(payload, "value_schema", ()) or ()) != value_schema:
                raise RuntimeError(
                    "Cannot collate BoundaryPayload objects with different value_schema."
                )

        batch_size = sum(int(getattr(payload, "batch_size", 0) or 0) for payload in payloads)
        if batch_size <= 0:
            raise RuntimeError("Cannot collate BoundaryPayload objects with empty batch_size.")

        batched_tensors: dict[str, torch.Tensor] = {}
        labels = list(dict(getattr(first, "tensors", {}) or {}).keys())
        for label in labels:
            label = str(label)
            spec = schema.get(label)
            if spec is None:
                raise RuntimeError(f"Boundary tensor {label!r} is missing from payload schema.")
            pieces = [dict(getattr(payload, "tensors", {}) or {})[label] for payload in payloads]
            first_piece = pieces[0]
            if not isinstance(first_piece, torch.Tensor):
                continue
            plan = self._tensor_batch_plan(
                label,
                first_piece,
                spec,
                int(getattr(first, "batch_size", 0) or 0),
            )
            if plan is None:
                for piece in pieces[1:]:
                    if not _same_tensor(first_piece, piece):
                        raise RuntimeError(
                            f"Boundary tensor {label!r} is schema-shared but differs "
                            "across samples."
                        )
                batched_tensors[label] = first_piece
                continue
            axis, multiplier = plan
            for payload, piece in zip(payloads, pieces, strict=True):
                expected = multiplier * int(getattr(payload, "batch_size", 0) or 0)
                if int(piece.shape[axis]) != expected:
                    raise RuntimeError(
                        f"Boundary tensor {label!r} dimension {axis} is {int(piece.shape[axis])}; "
                        f"expected {expected} from {getattr(spec, 'symbolic_shape', ())[axis]}."
                    )
            target_device = first_piece.device
            batched_tensors[label] = torch.cat(
                [piece.to(target_device) for piece in pieces],
                dim=axis,
            )

        batched_passthrough = self._collate_passthrough_values(
            [dict(getattr(payload, "passthrough_inputs", {}) or {}) for payload in payloads],
            [int(getattr(payload, "batch_size", 0) or 0) for payload in payloads],
        )
        batched_values = self._collate_values(
            [tuple(getattr(payload, "values", ()) or ()) for payload in payloads],
            value_schema,
            [int(getattr(payload, "batch_size", 0) or 0) for payload in payloads],
        )
        result = _new_payload_like(
            first,
            batch_size=batch_size,
            tensors=batched_tensors,
            passthrough_inputs=batched_passthrough,
            values=batched_values,
        )
        self.validate(result)
        return result

    def save(
        self,
        path: str | os.PathLike[str],
        payload: BoundaryPayload,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.validate(payload)
        record = dict(metadata or {})
        record.update(
            {
                "cache_protocol": BOUNDARY_CACHE_PROTOCOL,
                "intermediate": payload,
            }
        )
        os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
        with gzip.open(path, "wb", compresslevel=1) as handle:
            torch.save(record, handle)
        return record

    def load(self, path: str | os.PathLike[str]) -> BoundaryPayload:
        try:
            with gzip.open(path, "rb") as handle:
                record = torch.load(handle, map_location="cpu", weights_only=False)
        except gzip.BadGzipFile:
            record = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(record, BoundaryPayload):
            return self.validate(record)
        if not isinstance(record, Mapping):
            raise TypeError(f"Unsupported boundary cache record: {type(record).__name__}.")
        payload = record.get("intermediate") or record.get("boundary_payload")
        if not isinstance(payload, BoundaryPayload):
            raise TypeError("Boundary cache record did not contain a BoundaryPayload.")
        return self.validate(payload)

    def _dimension_multiplier(self, value: Any) -> int | None:
        if isinstance(value, str):
            return 1 if value == self.batch_symbol else None
        expression = getattr(value, "expression", None)
        if expression != self.batch_symbol:
            return None
        offset = int(getattr(value, "offset", 0) or 0)
        if offset != 0:
            raise RuntimeError(
                f"Cannot split affine boundary dimension with non-zero offset: {value}."
            )
        multiplier = int(getattr(value, "multiplier", 1) or 1)
        if multiplier <= 0:
            raise RuntimeError(f"Cannot split boundary dimension with multiplier={multiplier}.")
        return multiplier

    def _tensor_batch_plan(
        self,
        label: str,
        tensor: torch.Tensor,
        spec: Any,
        batch_size: int,
    ) -> tuple[int, int] | None:
        symbolic_shape = tuple(getattr(spec, "symbolic_shape", ()) or ())
        if tensor.ndim == 0 or not symbolic_shape:
            return None
        if tensor.ndim != len(symbolic_shape):
            raise RuntimeError(
                f"Boundary tensor {label!r} rank {tensor.ndim} does not match schema rank "
                f"{len(symbolic_shape)}."
            )
        batch_dims: list[tuple[int, int]] = []
        for axis, dim in enumerate(symbolic_shape):
            multiplier = self._dimension_multiplier(dim)
            if multiplier is not None:
                batch_dims.append((axis, multiplier))
        if not batch_dims:
            return None
        if len(batch_dims) > 1:
            raise RuntimeError(
                f"Boundary tensor {label!r} has multiple batch-derived schema dimensions."
            )
        axis, multiplier = batch_dims[0]
        expected = int(batch_size) * multiplier
        actual = int(tensor.shape[axis])
        if actual != expected:
            raise RuntimeError(
                f"Boundary tensor {label!r} dimension {axis} is {actual}; "
                f"expected {expected} from {symbolic_shape[axis]}."
            )
        return axis, multiplier

    def _split_passthrough_value(
        self,
        value: Any,
        *,
        batch_size: int,
        actual_batch_size: int,
    ) -> list[Any]:
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return [value for _ in range(actual_batch_size)]
            dim0 = int(value.shape[0])
            if dim0 == int(batch_size):
                multiplier = 1
            elif batch_size > 0 and dim0 > 0 and dim0 % int(batch_size) == 0:
                multiplier = dim0 // int(batch_size)
            else:
                return [value for _ in range(actual_batch_size)]
            return [
                value.narrow(0, index * multiplier, multiplier)
                for index in range(actual_batch_size)
            ]
        if isinstance(value, Mapping):
            pieces = {
                key: self._split_passthrough_value(
                    item,
                    batch_size=batch_size,
                    actual_batch_size=actual_batch_size,
                )
                for key, item in value.items()
            }
            return [
                {key: split_values[index] for key, split_values in pieces.items()}
                for index in range(actual_batch_size)
            ]
        if isinstance(value, tuple):
            pieces = [
                self._split_passthrough_value(
                    item,
                    batch_size=batch_size,
                    actual_batch_size=actual_batch_size,
                )
                for item in value
            ]
            return [tuple(item[index] for item in pieces) for index in range(actual_batch_size)]
        if isinstance(value, list):
            pieces = [
                self._split_passthrough_value(
                    item,
                    batch_size=batch_size,
                    actual_batch_size=actual_batch_size,
                )
                for item in value
            ]
            return [[item[index] for item in pieces] for index in range(actual_batch_size)]
        return [value for _ in range(actual_batch_size)]

    def _collate_passthrough_values(
        self,
        values: list[Any],
        batch_sizes: list[int],
    ) -> Any:
        first = values[0]
        if isinstance(first, torch.Tensor):
            tensors = values
            if not all(isinstance(value, torch.Tensor) for value in tensors):
                raise RuntimeError("Cannot collate mixed tensor/non-tensor passthrough values.")
            if first.ndim == 0:
                if all(_same_tensor(first, value) for value in tensors[1:]):
                    return first
                raise RuntimeError(
                    "Scalar passthrough tensors differ across BoundaryPayload samples."
                )
            multipliers: list[int] = []
            batch_like = True
            for tensor, batch_size in zip(tensors, batch_sizes, strict=True):
                if batch_size <= 0 or tensor.ndim == 0 or int(tensor.shape[0]) % batch_size != 0:
                    batch_like = False
                    break
                multipliers.append(int(tensor.shape[0]) // batch_size)
            same_shared = all(_same_tensor(first, value) for value in tensors[1:])
            if batch_like and (multipliers and (multipliers[0] == 1 or not same_shared)):
                if any(multiplier != multipliers[0] for multiplier in multipliers):
                    raise RuntimeError(
                        "Passthrough tensor batch multipliers differ across samples."
                    )
                target_device = first.device
                return torch.cat([tensor.to(target_device) for tensor in tensors], dim=0)
            if same_shared:
                return first
            raise RuntimeError("Cannot determine how to collate passthrough tensor values.")
        if isinstance(first, Mapping):
            keys = tuple(first.keys())
            for value in values[1:]:
                if not isinstance(value, Mapping) or tuple(value.keys()) != keys:
                    raise RuntimeError("Passthrough mapping keys differ across samples.")
            return {
                key: self._collate_passthrough_values(
                    [value[key] for value in values],
                    batch_sizes,
                )
                for key in keys
            }
        if isinstance(first, tuple):
            length = len(first)
            for value in values[1:]:
                if not isinstance(value, tuple) or len(value) != length:
                    raise RuntimeError("Passthrough tuple shapes differ across samples.")
            return tuple(
                self._collate_passthrough_values(
                    [value[index] for value in values],
                    batch_sizes,
                )
                for index in range(length)
            )
        if isinstance(first, list):
            length = len(first)
            for value in values[1:]:
                if not isinstance(value, list) or len(value) != length:
                    raise RuntimeError("Passthrough list shapes differ across samples.")
            return [
                self._collate_passthrough_values(
                    [value[index] for value in values],
                    batch_sizes,
                )
                for index in range(length)
            ]
        if all(_same_value(first, value) for value in values[1:]):
            return first
        raise RuntimeError("Passthrough values differ across samples and cannot be collated.")

    def _split_values(
        self,
        values: tuple[Any, ...],
        value_schema: tuple[Any, ...],
        *,
        batch_size: int,
        actual_batch_size: int,
    ) -> list[tuple[Any, ...]]:
        if not values and not value_schema:
            return [() for _ in range(actual_batch_size)]
        if len(values) != len(value_schema):
            raise RuntimeError(
                f"BoundaryPayload has {len(values)} value(s), but value_schema has "
                f"{len(value_schema)} item(s)."
            )
        split_items = [
            self._split_value_by_spec(
                value,
                spec,
                batch_size=batch_size,
                actual_batch_size=actual_batch_size,
            )
            for value, spec in zip(values, value_schema, strict=True)
        ]
        return [
            tuple(item[index] for item in split_items)
            for index in range(actual_batch_size)
        ]

    def _split_value_by_spec(
        self,
        value: Any,
        spec: Any,
        *,
        batch_size: int,
        actual_batch_size: int,
    ) -> list[Any]:
        kind = type(spec).__name__
        if kind == "BoundarySequenceValueSpec":
            multiplier = self._dimension_multiplier(getattr(spec, "length_expr", None))
            if multiplier is not None:
                expected = int(batch_size) * multiplier
                if len(value) != expected:
                    raise RuntimeError(
                        f"Boundary sequence {getattr(spec, 'label', '')!r} length {len(value)}; "
                        f"expected {expected} from {getattr(spec, 'length_expr', None)}."
                    )
                return [
                    self._sequence_like(
                        value,
                        value[index * multiplier : (index + 1) * multiplier],
                        spec,
                    )
                    for index in range(actual_batch_size)
                ]
            return [value for _ in range(actual_batch_size)]
        if kind in {"BoundaryTupleValueSpec", "BoundaryListValueSpec"}:
            item_specs = tuple(getattr(spec, "items", ()) or ())
            split_items = [
                self._split_value_by_spec(
                    item,
                    item_spec,
                    batch_size=batch_size,
                    actual_batch_size=actual_batch_size,
                )
                for item, item_spec in zip(value, item_specs, strict=True)
            ]
            if kind == "BoundaryTupleValueSpec":
                return [
                    tuple(item[index] for item in split_items)
                    for index in range(actual_batch_size)
                ]
            return [[item[index] for item in split_items] for index in range(actual_batch_size)]
        if kind == "BoundaryDictValueSpec":
            item_specs = tuple(getattr(spec, "items", ()) or ())
            split_items = {
                key: self._split_value_by_spec(
                    value[key],
                    item_spec,
                    batch_size=batch_size,
                    actual_batch_size=actual_batch_size,
                )
                for key, item_spec in item_specs
            }
            return [
                {key: split_values[index] for key, split_values in split_items.items()}
                for index in range(actual_batch_size)
            ]
        if kind == "BoundarySliceValueSpec":
            starts = self._split_value_by_spec(
                value.start,
                getattr(spec, "start"),
                batch_size=batch_size,
                actual_batch_size=actual_batch_size,
            )
            stops = self._split_value_by_spec(
                value.stop,
                getattr(spec, "stop"),
                batch_size=batch_size,
                actual_batch_size=actual_batch_size,
            )
            steps = self._split_value_by_spec(
                value.step,
                getattr(spec, "step"),
                batch_size=batch_size,
                actual_batch_size=actual_batch_size,
            )
            return [
                slice(starts[index], stops[index], steps[index])
                for index in range(actual_batch_size)
            ]
        return [value for _ in range(actual_batch_size)]

    def _collate_values(
        self,
        values: list[tuple[Any, ...]],
        value_schema: tuple[Any, ...],
        batch_sizes: list[int],
    ) -> tuple[Any, ...]:
        if not values and not value_schema:
            return ()
        if any(len(value) != len(value_schema) for value in values):
            raise RuntimeError("BoundaryPayload values do not match value_schema during collate.")
        return tuple(
            self._collate_value_by_spec(
                [value[index] for value in values],
                spec,
                batch_sizes,
            )
            for index, spec in enumerate(value_schema)
        )

    def _collate_value_by_spec(
        self,
        values: list[Any],
        spec: Any,
        batch_sizes: list[int],
    ) -> Any:
        first = values[0]
        kind = type(spec).__name__
        if kind == "BoundarySequenceValueSpec":
            multiplier = self._dimension_multiplier(getattr(spec, "length_expr", None))
            if multiplier is not None:
                pieces: list[Any] = []
                for value, batch_size in zip(values, batch_sizes, strict=True):
                    expected = int(batch_size) * multiplier
                    if len(value) != expected:
                        label = getattr(spec, "label", "")
                        raise RuntimeError(
                            f"Boundary sequence {label!r} length {len(value)}; "
                            f"expected {expected} from {getattr(spec, 'length_expr', None)}."
                        )
                    pieces.extend(list(value))
                return self._sequence_like(first, pieces, spec)
            return self._shared_value(values)
        if kind in {"BoundaryTupleValueSpec", "BoundaryListValueSpec"}:
            item_specs = tuple(getattr(spec, "items", ()) or ())
            collated = [
                self._collate_value_by_spec(
                    [value[index] for value in values],
                    item_spec,
                    batch_sizes,
                )
                for index, item_spec in enumerate(item_specs)
            ]
            return tuple(collated) if kind == "BoundaryTupleValueSpec" else collated
        if kind == "BoundaryDictValueSpec":
            return {
                key: self._collate_value_by_spec(
                    [value[key] for value in values],
                    item_spec,
                    batch_sizes,
                )
                for key, item_spec in tuple(getattr(spec, "items", ()) or ())
            }
        if kind == "BoundarySliceValueSpec":
            return slice(
                self._collate_value_by_spec(
                    [value.start for value in values],
                    getattr(spec, "start"),
                    batch_sizes,
                ),
                self._collate_value_by_spec(
                    [value.stop for value in values],
                    getattr(spec, "stop"),
                    batch_sizes,
                ),
                self._collate_value_by_spec(
                    [value.step for value in values],
                    getattr(spec, "step"),
                    batch_sizes,
                ),
            )
        return self._shared_value(values)

    @staticmethod
    def _sequence_like(original: Any, items: Any, spec: Any) -> Any:
        if isinstance(original, tuple) or getattr(spec, "container_type", None) == "tuple":
            return tuple(items)
        return list(items)

    @staticmethod
    def _shared_value(values: list[Any]) -> Any:
        first = values[0]
        if all(_same_value(first, value) for value in values[1:]):
            return first
        raise RuntimeError("BoundaryPayload values differ across samples and cannot be collated.")


__all__ = ["BOUNDARY_CACHE_PROTOCOL", "BoundaryPayloadCacheCodec"]
