from __future__ import annotations

import gzip
import os
from collections.abc import Mapping
from dataclasses import fields, replace
from typing import Any

import torch
from ariadne import BoundaryPayload
from ariadne.pattern.boundary_value import BoundaryTensorRef


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


def _first_module_device(module: Any) -> torch.device | None:
    if not isinstance(module, torch.nn.Module):
        return None
    for parameter in module.parameters(recurse=True):
        return parameter.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return None


def _runtime_variant_for_payload(runtime: Any, payload: BoundaryPayload) -> Any:
    for variant in tuple(getattr(runtime, "variants", ()) or ()):
        if (
            getattr(variant, "graph_signature", None) == payload.graph_signature
            and getattr(variant, "split_id", None) == payload.split_id
        ):
            return variant
    return runtime


def _runtime_payload_schema(
    runtime: Any,
    payload: BoundaryPayload,
) -> dict[str, Any] | None:
    if runtime is None:
        return None
    resolved_runtime = _runtime_variant_for_payload(runtime, payload)
    for schema in (
        getattr(getattr(resolved_runtime, "candidate", None), "boundary_schema", None),
        getattr(resolved_runtime, "schema", None),
    ):
        if not isinstance(schema, Mapping):
            continue
        runtime_schema = {str(label): spec for label, spec in dict(schema).items()}
        tensor_labels = {str(label) for label in dict(getattr(payload, "tensors", {}) or {})}
        if tensor_labels and tensor_labels != set(runtime_schema):
            continue
        return runtime_schema
    return None


def _runtime_payload_value_schema(
    runtime: Any,
    payload: BoundaryPayload,
) -> tuple[Any, ...] | None:
    if runtime is None:
        return None
    resolved_runtime = _runtime_variant_for_payload(runtime, payload)
    boundary_value_schema = getattr(resolved_runtime, "_boundary_value_schema", None)
    if callable(boundary_value_schema):
        try:
            return tuple(boundary_value_schema())
        except (AttributeError, KeyError, TypeError):
            pass

    candidate = getattr(resolved_runtime, "candidate", None)
    value_schema_by_label = getattr(candidate, "boundary_value_schema", None)
    boundary_order = getattr(getattr(resolved_runtime, "segments", None), "boundary_order", None)
    if isinstance(value_schema_by_label, Mapping) and boundary_order is not None:
        try:
            return tuple(value_schema_by_label[label] for label in boundary_order)
        except (KeyError, TypeError):
            pass

    value_schema = getattr(resolved_runtime, "value_schema", None)
    if value_schema is not None:
        try:
            return tuple(value_schema)
        except TypeError:
            return None
    return None


def _runtime_requires_grad(
    schema: Mapping[str, Any],
    payload: BoundaryPayload,
) -> dict[str, bool]:
    payload_requires_grad = dict(getattr(payload, "requires_grad", {}) or {})
    payload_tensors = dict(getattr(payload, "tensors", {}) or {})
    requires_grad: dict[str, bool] = {}
    for label, spec in dict(schema).items():
        tensor = payload_tensors.get(label)
        fallback = bool(getattr(tensor, "requires_grad", False))
        requires_grad[str(label)] = bool(
            getattr(spec, "requires_grad", payload_requires_grad.get(label, fallback))
        )
    return requires_grad


def _sequence_element_spec(spec: Any, index: int) -> Any:
    element_spec = getattr(spec, "element_spec", None)
    if type(element_spec).__name__ == "BoundaryTensorValueSpec":
        try:
            return replace(element_spec, label=f"{getattr(spec, 'label')}.{index}")
        except TypeError:
            return element_spec
    return element_spec


def _runtime_payload_value(value: Any, spec: Any) -> Any:
    spec_type = type(spec).__name__
    if spec_type == "BoundaryTensorValueSpec":
        label = str(getattr(spec, "label", "") or "")
        return BoundaryTensorRef(label) if label else value
    if spec_type == "BoundaryTupleValueSpec":
        source = tuple(value) if isinstance(value, (tuple, list)) else ()
        items = tuple(getattr(spec, "items", ()) or ())
        return tuple(
            _runtime_payload_value(source[index] if index < len(source) else None, item_spec)
            for index, item_spec in enumerate(items)
        )
    if spec_type == "BoundaryListValueSpec":
        source = list(value) if isinstance(value, (tuple, list)) else []
        items = tuple(getattr(spec, "items", ()) or ())
        return [
            _runtime_payload_value(source[index] if index < len(source) else None, item_spec)
            for index, item_spec in enumerate(items)
        ]
    if spec_type == "BoundaryDictValueSpec":
        source = dict(value) if isinstance(value, Mapping) else {}
        return {
            key: _runtime_payload_value(source.get(key), item_spec)
            for key, item_spec in tuple(getattr(spec, "items", ()) or ())
        }
    if spec_type == "BoundarySliceValueSpec":
        return slice(
            _runtime_payload_value(getattr(value, "start", None), getattr(spec, "start", None)),
            _runtime_payload_value(getattr(value, "stop", None), getattr(spec, "stop", None)),
            _runtime_payload_value(getattr(value, "step", None), getattr(spec, "step", None)),
        )
    if spec_type == "BoundarySequenceValueSpec":
        if isinstance(value, tuple):
            source = value
        elif isinstance(value, list):
            source = tuple(value)
        else:
            source = ()
        rebuilt = [
            _runtime_payload_value(item, _sequence_element_spec(spec, index))
            for index, item in enumerate(source)
        ]
        return tuple(rebuilt) if getattr(spec, "container_type", None) == "tuple" else rebuilt
    return value


def _runtime_payload_values(
    payload: BoundaryPayload,
    value_schema: tuple[Any, ...],
) -> tuple[Any, ...]:
    if not value_schema:
        return ()
    values = tuple(getattr(payload, "values", ()) or ())
    return tuple(
        _runtime_payload_value(values[index] if index < len(values) else None, spec)
        for index, spec in enumerate(value_schema)
    )


def _runtime_payload_device(runtime: Any, payload: BoundaryPayload) -> torch.device | None:
    if runtime is None:
        return None
    resolved_runtime = _runtime_variant_for_payload(runtime, payload)
    schema = _runtime_payload_schema(resolved_runtime, payload)
    if schema is None:
        schema = getattr(payload, "schema", None)

    schema_device_type: str | None = None
    if isinstance(schema, Mapping):
        for label in dict(getattr(payload, "tensors", {}) or {}):
            spec = schema.get(str(label))
            device_type = getattr(spec, "device_type", None)
            if device_type:
                schema_device_type = str(device_type)
                break

    for module_name in ("suffix_segment", "prefix_segment", "training_prefix_segment"):
        module_device = _first_module_device(getattr(resolved_runtime, module_name, None))
        if module_device is None:
            continue
        if schema_device_type is None or module_device.type == schema_device_type:
            return module_device

    if schema_device_type:
        return torch.device(schema_device_type)
    return None


def _payload_values_on_device(value: Any, device: torch.device) -> bool:
    if isinstance(value, torch.Tensor):
        return value.device == device and value.is_contiguous()
    if isinstance(value, Mapping):
        return all(_payload_values_on_device(item, device) for item in value.values())
    if isinstance(value, tuple):
        return all(_payload_values_on_device(item, device) for item in value)
    if isinstance(value, list):
        return all(_payload_values_on_device(item, device) for item in value)
    return True


def _move_payload_value_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        moved = value.to(device)
        return moved if moved.is_contiguous() else moved.contiguous()
    if isinstance(value, Mapping):
        return {
            key: _move_payload_value_to_device(item, device)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_move_payload_value_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_payload_value_to_device(item, device) for item in value]
    return value


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

    def to_runtime_device(self, payload: BoundaryPayload) -> BoundaryPayload:
        if not isinstance(payload, BoundaryPayload):
            raise TypeError(f"Expected BoundaryPayload, got {type(payload).__name__}.")
        device = _runtime_payload_device(self.runtime, payload)
        runtime_schema = _runtime_payload_schema(self.runtime, payload)
        runtime_value_schema = _runtime_payload_value_schema(self.runtime, payload)
        schema = dict(getattr(payload, "schema", {}) or {})
        requires_grad = dict(getattr(payload, "requires_grad", {}) or {})
        value_schema = tuple(getattr(payload, "value_schema", ()) or ())
        values = tuple(getattr(payload, "values", ()) or ())
        if runtime_schema is not None:
            schema = runtime_schema
            requires_grad = _runtime_requires_grad(runtime_schema, payload)
        if runtime_value_schema is not None:
            value_schema = runtime_value_schema
            values = _runtime_payload_values(payload, value_schema)
        if device is None:
            if (
                schema == dict(getattr(payload, "schema", {}) or {})
                and requires_grad == dict(getattr(payload, "requires_grad", {}) or {})
                and value_schema == tuple(getattr(payload, "value_schema", ()) or ())
                and values == tuple(getattr(payload, "values", ()) or ())
            ):
                return payload
            return replace(
                payload,
                schema=schema,
                requires_grad=requires_grad,
                value_schema=value_schema,
                values=values,
            )
        tensors = dict(getattr(payload, "tensors", {}) or {})
        passthrough = dict(getattr(payload, "passthrough_inputs", {}) or {})
        tensors_on_device = _payload_values_on_device(tensors, device)
        passthrough_on_device = _payload_values_on_device(passthrough, device)
        values_on_device = _payload_values_on_device(values, device)
        if (
            tensors_on_device
            and passthrough_on_device
            and values_on_device
            and schema == dict(getattr(payload, "schema", {}) or {})
            and requires_grad == dict(getattr(payload, "requires_grad", {}) or {})
            and value_schema == tuple(getattr(payload, "value_schema", ()) or ())
            and values == tuple(getattr(payload, "values", ()) or ())
        ):
            return payload
        return replace(
            payload,
            tensors=(
                tensors
                if tensors_on_device
                else _move_payload_value_to_device(tensors, device)
            ),
            passthrough_inputs=(
                passthrough
                if passthrough_on_device
                else _move_payload_value_to_device(passthrough, device)
            ),
            values=values if values_on_device else _move_payload_value_to_device(values, device),
            schema=schema,
            requires_grad=requires_grad,
            value_schema=value_schema,
        )

    def split_batch(
        self,
        payload: BoundaryPayload,
        actual_batch_size: int | None = None,
    ) -> list[BoundaryPayload]:
        payload = self.to_runtime_device(payload)
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
        payloads = [self.to_runtime_device(payload) for payload in payloads]
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
        payload = self.validate(self.to_runtime_device(payload))
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
            record = self.to_runtime_device(record)
            return self.validate(record)
        if not isinstance(record, Mapping):
            raise TypeError(f"Unsupported boundary cache record: {type(record).__name__}.")
        payload = record.get("intermediate") or record.get("boundary_payload")
        if not isinstance(payload, BoundaryPayload):
            raise TypeError("Boundary cache record did not contain a BoundaryPayload.")
        payload = self.to_runtime_device(payload)
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
