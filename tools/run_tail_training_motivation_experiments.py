from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np
import torch
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import model_management.object_detection as object_detection_module
from cloud_server import _evaluate_detection_proxy_map
from config import load_runtime_config
from model_management.fixed_split import (
    SplitConstraints,
    load_or_compute_fixed_split_plan,
    min_edge_parameters_for_privacy,
)
from model_management.model_zoo import (
    ensure_local_model_artifact,
    get_model_artifact_path,
    get_model_detection_thresholds,
    get_model_family,
    set_detection_finetune_mode,
)
from model_management.object_detection import Object_Detection
from model_management.split_model_adapters import (
    build_split_training_loss,
    get_split_runtime_input_resize_mode,
    get_split_runtime_model,
    prepare_split_runtime_input,
)
from model_management.universal_model_split import (
    BoundaryPayload,
    SplitCandidate,
    UniversalModelSplitter,
    build_split_retrain_optimizer,
    collect_suffix_trainable_parameters,
    prepare_split_train_batches_once,
    save_split_feature_cache,
)


DEFAULT_MODES = ("full", "freeze", "split_cached", "split_rebuild")
DEFAULT_SAMPLE_COUNTS = (64, 128, 256)
DEFAULT_EPOCHS = (1, 3, 5, 10, 20)
DEFAULT_BOUNDARY_QUANTILES = (0.25, 0.50, 0.75)
BUCKET_LABELS = ("Early", "Middle", "Late")


@dataclass(frozen=True)
class CandidateChoice:
    bucket: str
    target_ratio: float | None
    candidate: SplitCandidate


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare full, freeze-prefix, and cached split-tail training.",
    )
    parser.add_argument("--yaml-path", default="./config/config.yaml")
    parser.add_argument("--video-path", default="./video_data/road.mp4")
    parser.add_argument("--edge-model", default="rfdetr_nano")
    parser.add_argument("--golden-model", default="rtdetr_x")
    parser.add_argument(
        "--sample-counts",
        nargs="+",
        type=int,
        default=list(DEFAULT_SAMPLE_COUNTS),
    )
    parser.add_argument(
        "--epochs",
        nargs="+",
        type=int,
        default=list(DEFAULT_EPOCHS),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--boundary-quantiles",
        nargs="+",
        type=float,
        default=list(DEFAULT_BOUNDARY_QUANTILES),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=DEFAULT_MODES,
        default=list(DEFAULT_MODES),
    )
    parser.add_argument("--output-root", default="./tmp/tail_training_motivation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat the full experiment with seeds seed+i and aggregate mean/std in plots.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for model construction and training.",
    )
    return parser.parse_args(argv)


def _safe_segment(value: object, *, max_len: int = 64) -> str:
    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    text = text.strip("._-") or "item"
    if len(text) <= max_len:
        return text
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{text[: max_len - 11]}_{digest}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_ready(dict(row)), ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def _csv_value(value: Any) -> Any:
    value = _json_ready(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_summary_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _mean_std(values: list[Any]) -> tuple[float | None, float | None]:
    numeric = [
        float(value)
        for value in values
        if value is not None and np.isfinite(float(value))
    ]
    if not numeric:
        return None, None
    if len(numeric) == 1:
        return float(numeric[0]), 0.0
    return float(np.mean(numeric)), float(np.std(numeric, ddof=1))


def _aggregate_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("mode"),
            row.get("split_bucket"),
            row.get("candidate_id"),
            int(row.get("sample_count") or 0),
            int(row.get("epochs") or 0),
        )
        groups.setdefault(key, []).append(row)

    metric_fields = (
        "total_wall_time",
        "feature_reconstruction_time",
        "feature_load_time",
        "training_time",
        "epoch_time_mean",
        "batch_time_mean",
        "peak_cuda_memory_allocated",
        "peak_cuda_memory_reserved",
        "trainable_parameter_count",
        "total_parameter_count",
        "prefix_parameter_count",
        "suffix_parameter_count",
        "prefix_parameter_ratio",
        "boundary_payload_bytes",
        "proxy_mAP@0.5 before",
        "proxy_mAP@0.5 after",
        "delta proxy_mAP@0.5",
        "final_loss",
    )
    aggregated: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        mode, split_bucket, candidate_id, sample_count, epochs = key
        success_items = [item for item in items if bool(item.get("success"))]
        row = {
            "mode": mode,
            "split_bucket": split_bucket,
            "candidate_id": candidate_id,
            "sample_count": sample_count,
            "epochs": epochs,
            "run_count": len(items),
            "success_count": len(success_items),
            "failure_count": len(items) - len(success_items),
            "success_rate": len(success_items) / float(max(1, len(items))),
        }
        for field in metric_fields:
            mean, std = _mean_std([item.get(field) for item in success_items])
            row[f"{field}_mean"] = mean
            row[f"{field}_std"] = std
        aggregated.append(row)
    return aggregated


def _write_aggregate_summary_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    _write_summary_csv(path, _aggregate_rows(rows))


def _set_random_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_peak(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0, 0
    return (
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def _clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(dict(json.loads(line)))
    return rows


def _resolve_local_weights_path(model_name: str) -> str:
    artifact_path = ensure_local_model_artifact(model_name)
    expected_path = get_model_artifact_path(model_name)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Failed to prepare local weights for {model_name}: {expected_path}"
        )
    return str(artifact_path)


def _select_sample_frame_ids(
    total_frames: int,
    sample_counts: list[int],
    *,
    seed: int,
) -> dict[int, list[int]]:
    if total_frames <= 0:
        raise RuntimeError("Video contains no readable frames.")
    max_count = max(int(count) for count in sample_counts)
    if max_count <= 0:
        raise ValueError("--sample-counts values must be positive.")
    if max_count > total_frames:
        raise RuntimeError(
            f"Requested {max_count} samples but video only has {total_frames} frame(s)."
        )

    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(np.arange(1, total_frames + 1))
    selected: dict[int, list[int]] = {}
    for count in sample_counts:
        count = int(count)
        if count <= 0:
            raise ValueError("--sample-counts values must be positive.")
        selected[count] = sorted(int(value) for value in permutation[:count].tolist())
    return selected


def _sample_video_frames(
    video_path: Path,
    sample_counts: list[int],
    *,
    seed: int,
) -> tuple[dict[int, np.ndarray], dict[int, list[int]]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        frames: dict[int, np.ndarray] = {}
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frame_index += 1
                frames[frame_index] = frame
        finally:
            capture.release()
        selected = _select_sample_frame_ids(len(frames), sample_counts, seed=seed)
        selected_ids = {frame_id for ids in selected.values() for frame_id in ids}
        return {frame_id: frames[frame_id] for frame_id in selected_ids}, selected

    selected = _select_sample_frame_ids(total_frames, sample_counts, seed=seed)
    needed_ids = {frame_id for ids in selected.values() for frame_id in ids}
    frames_by_id: dict[int, np.ndarray] = {}
    frame_index = 0
    try:
        while len(frames_by_id) < len(needed_ids):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame_index += 1
            if frame_index in needed_ids:
                frames_by_id[frame_index] = frame
    finally:
        capture.release()

    missing = sorted(needed_ids - set(frames_by_id))
    if missing:
        raise RuntimeError(f"Failed to read sampled frame(s): {missing[:8]}")
    return frames_by_id, selected


def _write_raw_frames(frame_dir: Path, frames_by_id: Mapping[int, np.ndarray]) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for frame_id, frame in frames_by_id.items():
        path = frame_dir / f"{int(frame_id)}.jpg"
        if not path.exists():
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"Failed to write sampled frame: {path}")


def _teacher_target_from_prediction(
    pred_boxes: Any,
    pred_class: Any,
    pred_score: Any = None,
) -> dict[str, Any]:
    del pred_score
    boxes = list(pred_boxes or [])
    labels = list(pred_class or [])
    count = min(len(boxes), len(labels))
    if count <= 0:
        return {"boxes": [], "labels": []}
    return {
        "boxes": [
            [float(coord) for coord in list(box)[:4]]
            for box in boxes[:count]
        ],
        "labels": [int(label) for label in labels[:count]],
    }


def _load_or_collect_teacher_annotations(
    *,
    cache_path: Path,
    detector: Object_Detection,
    frames_by_id: Mapping[int, np.ndarray],
    frame_ids: list[int],
    golden_model: str,
    video_path: Path,
    threshold: float,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, dict[str, Any]], float]:
    expected_meta = {
        "golden_model": str(golden_model),
        "video_path": str(video_path.resolve()),
        "frame_ids": [int(frame_id) for frame_id in frame_ids],
        "threshold": float(threshold),
    }
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("metadata") == expected_meta:
            return dict(payload.get("annotations") or {}), 0.0

    annotations: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    _synchronize(device)
    try:
        batch_size = max(1, int(batch_size))
        for start in range(0, len(frame_ids), batch_size):
            batch_ids = frame_ids[start : start + batch_size]
            batch_frames = [frames_by_id[int(frame_id)] for frame_id in batch_ids]
            predictions = detector.large_inference_batch(
                batch_frames,
                threshold=float(threshold),
            )
            if len(predictions) != len(batch_ids):
                raise RuntimeError(
                    "Teacher batch inference returned "
                    f"{len(predictions)} result(s) for {len(batch_ids)} frame(s)."
                )
            for frame_id, prediction in zip(batch_ids, predictions):
                pred_boxes = pred_class = pred_score = None
                if isinstance(prediction, (list, tuple)):
                    if len(prediction) >= 1:
                        pred_boxes = prediction[0]
                    if len(prediction) >= 2:
                        pred_class = prediction[1]
                    if len(prediction) >= 3:
                        pred_score = prediction[2]
                annotations[str(int(frame_id))] = _teacher_target_from_prediction(
                    pred_boxes,
                    pred_class,
                    pred_score,
                )
    finally:
        _synchronize(device)
    elapsed = time.perf_counter() - started
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"metadata": expected_meta, "annotations": annotations},
            handle,
            indent=2,
            sort_keys=True,
        )
    return annotations, elapsed


def _candidate_prefix_ratio(candidate: SplitCandidate) -> float:
    total = int(getattr(candidate, "total_parameter_count", 0) or 0)
    if total <= 0:
        return float(getattr(candidate, "edge_parameter_ratio", 0.0) or 0.0)
    return float(getattr(candidate, "edge_parameter_count", 0) or 0) / float(total)


def _candidate_satisfies_constraints(
    candidate: SplitCandidate,
    constraints: SplitConstraints,
) -> bool:
    if not bool(getattr(candidate, "is_trainable_tail", False)):
        return False
    if int(getattr(candidate, "boundary_count", 0) or 0) > int(constraints.max_boundary_count):
        return False
    if int(getattr(candidate, "estimated_payload_bytes", 0) or 0) > int(
        constraints.max_payload_bytes
    ):
        return False
    if _candidate_prefix_ratio(candidate) > float(constraints.max_layer_freezing_ratio):
        return False
    if float(constraints.privacy_leakage_upper_bound) > 0.0:
        required_prefix_params = min_edge_parameters_for_privacy(
            float(constraints.privacy_leakage_upper_bound),
            epsilon=float(constraints.privacy_leakage_epsilon),
        )
        if int(getattr(candidate, "edge_parameter_count", 0) or 0) < required_prefix_params:
            return False
    return True


def _filter_candidates(
    candidates: list[SplitCandidate],
    constraints: SplitConstraints,
) -> list[SplitCandidate]:
    eligible = [
        candidate
        for candidate in candidates
        if _candidate_satisfies_constraints(candidate, constraints)
    ]
    eligible.sort(
        key=lambda candidate: (
            _candidate_prefix_ratio(candidate),
            int(getattr(candidate, "estimated_payload_bytes", 0) or 0),
            str(getattr(candidate, "candidate_id", "")),
        )
    )
    return eligible


def _select_candidate_choices(
    eligible_candidates: list[SplitCandidate],
    auto_candidate: SplitCandidate,
    boundary_quantiles: list[float],
) -> list[CandidateChoice]:
    if not eligible_candidates:
        raise RuntimeError("No trainable split candidates satisfy the fixed_split constraints.")
    quantiles = list(boundary_quantiles)
    if len(quantiles) != 3:
        raise ValueError("--boundary-quantiles must contain exactly three values.")

    choices: list[CandidateChoice] = []
    for bucket, target in zip(BUCKET_LABELS, quantiles):
        target = float(target)
        candidate = min(
            eligible_candidates,
            key=lambda item: (
                abs(_candidate_prefix_ratio(item) - target),
                int(getattr(item, "estimated_payload_bytes", 0) or 0),
                str(getattr(item, "candidate_id", "")),
            ),
        )
        choices.append(CandidateChoice(bucket=bucket, target_ratio=target, candidate=candidate))

    choices.append(CandidateChoice(bucket="Auto", target_ratio=None, candidate=auto_candidate))
    return choices


def _snapshot_model_state(model: torch.nn.Module) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key, value in model.state_dict().items():
        if torch.is_tensor(value):
            snapshot[key] = value.detach().cpu().clone()
        else:
            snapshot[key] = copy.deepcopy(value)
    return snapshot


def _restore_model_state(model: torch.nn.Module, state: Mapping[str, Any]) -> None:
    model.load_state_dict(dict(state), strict=False)


def _count_parameters(parameters: Any) -> int:
    return int(sum(int(parameter.numel()) for parameter in parameters))


def _total_parameter_count(model: torch.nn.Module) -> int:
    return _count_parameters(model.parameters())


def _resolve_suffix_trainable_parameters(
    split_model: torch.nn.Module,
    splitter: Any,
    *,
    collector: Callable[[Any], list[torch.nn.Parameter]] = collect_suffix_trainable_parameters,
) -> tuple[list[torch.nn.Parameter], list[str]]:
    params = list(collector(splitter))
    param_ids = {id(parameter) for parameter in params}
    names = [
        name
        for name, parameter in split_model.named_parameters()
        if id(parameter) in param_ids
    ]
    return params, names


def _first_tensor_shape(value: Any) -> list[int] | None:
    if isinstance(value, torch.Tensor):
        return [int(dim) for dim in value.shape]
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor_shape(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor_shape(item)
            if found is not None:
                return found
    return None


def _runtime_input_batch_size(value: Any) -> int:
    shape = _first_tensor_shape(value)
    if shape and len(shape) >= 4:
        return int(shape[0])
    return 1


def _combine_runtime_inputs(inputs: list[Any]) -> Any:
    if not inputs:
        raise RuntimeError("Cannot combine an empty runtime-input batch.")
    if all(isinstance(item, torch.Tensor) for item in inputs):
        tensors = [
            item if int(item.ndim) >= 4 else item.unsqueeze(0)
            for item in inputs
            if isinstance(item, torch.Tensor)
        ]
        return torch.cat(tensors, dim=0)
    if all(isinstance(item, list) for item in inputs):
        combined: list[Any] = []
        for item in inputs:
            combined.extend(item)
        return combined
    if all(isinstance(item, tuple) for item in inputs):
        combined_items = []
        length = len(inputs[0])
        if not all(len(item) == length for item in inputs):
            raise RuntimeError("Cannot batch runtime input tuples with different lengths.")
        for index in range(length):
            combined_items.append(_combine_runtime_inputs([item[index] for item in inputs]))
        return tuple(combined_items)
    raise RuntimeError(
        "Unsupported mixed runtime input batch: "
        + ", ".join(type(item).__name__ for item in inputs)
    )


def _make_trace_input(sample_input: Any, trace_batch_size: int) -> Any:
    trace_batch_size = max(1, int(trace_batch_size))
    current_batch = max(1, _runtime_input_batch_size(sample_input))
    if current_batch >= trace_batch_size:
        return sample_input
    return _combine_runtime_inputs([sample_input for _ in range(trace_batch_size)])


def _splitter_dynamic_batch_min(splitter: UniversalModelSplitter) -> int:
    split_spec = getattr(splitter, "split_spec", None)
    dynamic_batch = getattr(split_spec, "dynamic_batch", None)
    if isinstance(dynamic_batch, (list, tuple)) and dynamic_batch:
        return max(1, int(dynamic_batch[0]))
    return 1


def _slice_batch_value(value: Any, index: int, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim > 0:
            leading = int(value.shape[0])
            if leading == int(batch_size):
                return value[index:index + 1].contiguous()
            if leading > 0 and leading % int(batch_size) == 0:
                chunk = leading // int(batch_size)
                return value[index * chunk:(index + 1) * chunk].contiguous()
        return value
    if isinstance(value, Mapping):
        return {
            key: _slice_batch_value(item, index, batch_size)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_slice_batch_value(item, index, batch_size) for item in value)
    if isinstance(value, list):
        return [_slice_batch_value(item, index, batch_size) for item in value]
    return value


def _split_boundary_payload_batch(
    payload: BoundaryPayload,
    *,
    batch_size: int,
) -> list[BoundaryPayload]:
    if int(getattr(payload, "batch_size", 0)) != int(batch_size):
        raise RuntimeError(
            "BoundaryPayload batch size does not match the split request "
            f"(payload_batch={getattr(payload, 'batch_size', None)}, expected={batch_size})."
        )
    return [
        BoundaryPayload(
            split_id=payload.split_id,
            graph_signature=payload.graph_signature,
            batch_size=1,
            tensors={
                label: _slice_batch_value(tensor, index, batch_size)
                for label, tensor in dict(payload.tensors).items()
            },
            schema=dict(payload.schema),
            requires_grad=dict(payload.requires_grad),
            weight_version=payload.weight_version,
            passthrough_inputs=_slice_batch_value(
                dict(payload.passthrough_inputs or {}),
                index,
                batch_size,
            ),
        )
        for index in range(int(batch_size))
    ]


def _target_with_metadata(
    annotation: Mapping[str, Any] | None,
    *,
    frame: np.ndarray,
    runtime_input: Any,
    resize_mode: str | None,
) -> dict[str, Any]:
    target = {
        "boxes": list((annotation or {}).get("boxes") or []),
        "labels": list((annotation or {}).get("labels") or []),
    }
    target["_split_meta"] = {
        "input_image_size": [int(frame.shape[0]), int(frame.shape[1])],
        "input_tensor_shape": _first_tensor_shape(runtime_input),
        "input_resize_mode": resize_mode or "direct_resize",
    }
    return target


def _prepare_raw_batch(
    *,
    model: torch.nn.Module,
    frame_ids: list[int],
    frames_by_id: Mapping[int, np.ndarray],
    annotations: Mapping[str, Mapping[str, Any]],
    device: torch.device,
    resize_mode: str | None,
) -> tuple[Any, list[dict[str, Any]]]:
    runtime_inputs: list[Any] = []
    targets: list[dict[str, Any]] = []
    for frame_id in frame_ids:
        frame = frames_by_id[int(frame_id)]
        runtime_input = prepare_split_runtime_input(model, frame, device=device)
        runtime_inputs.append(runtime_input)
        targets.append(
            _target_with_metadata(
                annotations.get(str(int(frame_id))),
                frame=frame,
                runtime_input=runtime_input,
                resize_mode=resize_mode,
            )
        )
    return _combine_runtime_inputs(runtime_inputs), targets


@contextmanager
def _forbid_prefix_execution(splitter: Any):
    runtime = splitter._ensure_runtime() if hasattr(splitter, "_ensure_runtime") else None
    original_runtime_prefix = getattr(runtime, "run_prefix", None)
    original_edge_forward = getattr(splitter, "edge_forward", None)
    original_run_prefix = getattr(splitter, "run_prefix", None)

    def _blocked_prefix(*_: Any, **__: Any) -> Any:
        raise RuntimeError("Prefix forward is forbidden during split-tail training.")

    if runtime is not None and original_runtime_prefix is not None:
        setattr(runtime, "run_prefix", _blocked_prefix)
    if original_edge_forward is not None:
        setattr(splitter, "edge_forward", _blocked_prefix)
    if original_run_prefix is not None:
        setattr(splitter, "run_prefix", _blocked_prefix)
    try:
        yield
    finally:
        if runtime is not None and original_runtime_prefix is not None:
            setattr(runtime, "run_prefix", original_runtime_prefix)
        if original_edge_forward is not None:
            setattr(splitter, "edge_forward", original_edge_forward)
        if original_run_prefix is not None:
            setattr(splitter, "run_prefix", original_run_prefix)


def _resolve_experiment_learning_rate(config: Any, model_name: str) -> float:
    cl_cfg = getattr(config, "continual_learning", None)
    family = get_model_family(str(model_name))
    if family == "tinynext":
        return float(getattr(cl_cfg, "tinynext_fixed_split_learning_rate", 1e-3))
    if family == "rfdetr":
        return float(getattr(cl_cfg, "rfdetr_fixed_split_learning_rate", 1e-4))
    if family in {"yolo", "detr", "rtdetr"}:
        return float(getattr(cl_cfg, "wrapper_fixed_split_learning_rate", 3e-5))
    return float(getattr(cl_cfg, "split_learning_rate", 1e-3))


def _optimizer_overrides(model_name: str) -> dict[str, Any]:
    if get_model_family(str(model_name)) in {"rfdetr", "yolo", "tinynext"}:
        return {
            "optimizer_name": "adamw",
            "weight_decay": 1e-4,
            "grad_clip_norm": 1.0,
            "shuffle_samples": True,
        }
    return {"optimizer_name": "adam", "weight_decay": 0.0, "shuffle_samples": False}


def _make_optimizer(
    split_model: torch.nn.Module,
    *,
    runtime: Any,
    learning_rate: float,
    optimizer_config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    optimizer = build_split_retrain_optimizer(
        split_model,
        runtime=runtime,
        learning_rate=float(learning_rate),
        optimizer_name=str(optimizer_config.get("optimizer_name", "adam")),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
        grad_clip_norm=optimizer_config.get("grad_clip_norm"),
    )
    if optimizer is None:
        raise RuntimeError("No trainable parameters were available for this run.")
    return optimizer


def _shuffled_epoch_batches(
    sample_ids: list[int],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    epoch: int,
) -> list[list[int]]:
    ids = list(sample_ids)
    if shuffle and len(ids) > 1:
        rng = np.random.default_rng(int(seed) + int(epoch))
        order = rng.permutation(np.arange(len(ids))).tolist()
        ids = [ids[index] for index in order]
    return [ids[start : start + batch_size] for start in range(0, len(ids), batch_size)]


def _train_raw_loop(
    *,
    edge_model: torch.nn.Module,
    split_model: torch.nn.Module,
    model_name: str,
    frames_by_id: Mapping[int, np.ndarray],
    sample_ids: list[int],
    annotations: Mapping[str, Mapping[str, Any]],
    num_epoch: int,
    batch_size: int,
    device: torch.device,
    loss_fn: Callable[[Any, Any], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    seed: int,
    shuffle_samples: bool,
) -> dict[str, Any]:
    resize_mode = get_split_runtime_input_resize_mode(edge_model)
    epoch_times: list[float] = []
    batch_times: list[float] = []
    losses: list[float] = []
    _synchronize(device)
    training_started = time.perf_counter()
    for epoch in range(int(num_epoch)):
        set_detection_finetune_mode(edge_model, model_name)
        epoch_started = time.perf_counter()
        epoch_losses: list[float] = []
        for batch_ids in _shuffled_epoch_batches(
            sample_ids,
            batch_size=max(1, int(batch_size)),
            shuffle=bool(shuffle_samples),
            seed=int(seed),
            epoch=epoch,
        ):
            _synchronize(device)
            batch_started = time.perf_counter()
            inputs, targets = _prepare_raw_batch(
                model=edge_model,
                frame_ids=batch_ids,
                frames_by_id=frames_by_id,
                annotations=annotations,
                device=device,
                resize_mode=resize_mode,
            )
            outputs = split_model(inputs)
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _synchronize(device)
            batch_times.append(time.perf_counter() - batch_started)
            loss_value = float(loss.detach().cpu().item())
            epoch_losses.append(loss_value)
            losses.append(loss_value)
        epoch_times.append(time.perf_counter() - epoch_started)
    _synchronize(device)
    training_time = time.perf_counter() - training_started
    return {
        "training_time": float(training_time),
        "epoch_time_mean": float(np.mean(epoch_times)) if epoch_times else None,
        "batch_time_mean": float(np.mean(batch_times)) if batch_times else None,
        "final_loss": float(losses[-1]) if losses else None,
        "epoch_times": epoch_times,
        "batch_times": batch_times,
    }


def _boundary_payload_bytes(payload: BoundaryPayload) -> int:
    total = 0
    for tensor in dict(getattr(payload, "tensors", {}) or {}).values():
        if isinstance(tensor, torch.Tensor):
            total += int(tensor.numel()) * int(tensor.element_size())
    return int(total)


def _rebuild_feature_cache(
    *,
    splitter: UniversalModelSplitter,
    edge_model: torch.nn.Module,
    frames_by_id: Mapping[int, np.ndarray],
    sample_ids: list[int],
    cache_path: Path,
    device: torch.device,
) -> tuple[float, dict[str, Mapping[str, Any]], int]:
    cache_path.mkdir(parents=True, exist_ok=True)
    records: dict[str, Mapping[str, Any]] = {}
    payload_bytes = 0
    resize_mode = get_split_runtime_input_resize_mode(edge_model)
    _synchronize(device)
    started = time.perf_counter()
    try:
        runtime_min_batch = _splitter_dynamic_batch_min(splitter)
        for start in range(0, len(sample_ids), runtime_min_batch):
            chunk_ids = list(sample_ids[start:start + runtime_min_batch])
            runtime_inputs = [
                prepare_split_runtime_input(edge_model, frames_by_id[int(frame_id)], device=device)
                for frame_id in chunk_ids
            ]
            while len(runtime_inputs) < runtime_min_batch:
                runtime_inputs.append(runtime_inputs[-1])
            batch_input = _combine_runtime_inputs(runtime_inputs)
            with torch.inference_mode():
                batch_payload = splitter.edge_forward(batch_input)
            split_payloads = _split_boundary_payload_batch(
                batch_payload,
                batch_size=runtime_min_batch,
            )
            for frame_id, runtime_input, payload in zip(
                chunk_ids,
                runtime_inputs[:len(chunk_ids)],
                split_payloads[:len(chunk_ids)],
                strict=True,
            ):
                frame = frames_by_id[int(frame_id)]
                payload_bytes += _boundary_payload_bytes(payload)
                record = save_split_feature_cache(
                    str(cache_path),
                    int(frame_id),
                    payload,
                    input_image_size=[int(frame.shape[0]), int(frame.shape[1])],
                    input_tensor_shape=_first_tensor_shape(runtime_input),
                    input_resize_mode=resize_mode or "direct_resize",
                )
                records[str(int(frame_id))] = record
    finally:
        _synchronize(device)
    return time.perf_counter() - started, records, payload_bytes


def _train_split_loop(
    *,
    split_model: torch.nn.Module,
    splitter: UniversalModelSplitter,
    cache_path: Path,
    sample_ids: list[int],
    annotations: Mapping[str, Mapping[str, Any]],
    num_epoch: int,
    batch_size: int,
    device: torch.device,
    loss_fn: Callable[[Any, Any], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    seed: int,
    shuffle_samples: bool,
    preloaded_records: Mapping[Any, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    del split_model
    _synchronize(device)
    load_started = time.perf_counter()
    prepared_batches = prepare_split_train_batches_once(
        splitter=splitter,
        cache_path=str(cache_path),
        all_indices=[int(sample_id) for sample_id in sample_ids],
        annotations=annotations,
        batch_size=max(1, int(batch_size)),
        device=device,
        preloaded_records=preloaded_records,
        move_to_device=True,
        validate=True,
    )
    _synchronize(device)
    feature_load_time = time.perf_counter() - load_started
    if not prepared_batches:
        raise RuntimeError("Split-tail training did not prepare any batches.")

    epoch_times: list[float] = []
    batch_times: list[float] = []
    losses: list[float] = []
    _synchronize(device)
    training_started = time.perf_counter()
    with _forbid_prefix_execution(splitter):
        for epoch in range(int(num_epoch)):
            epoch_started = time.perf_counter()
            epoch_batches = list(prepared_batches)
            if shuffle_samples and len(epoch_batches) > 1:
                rng = np.random.default_rng(int(seed) + int(epoch))
                order = rng.permutation(np.arange(len(epoch_batches))).tolist()
                epoch_batches = [epoch_batches[index] for index in order]
            for prepared_batch in epoch_batches:
                _synchronize(device)
                batch_started = time.perf_counter()
                if prepared_batch.validated and hasattr(splitter, "train_suffix_fast"):
                    loss, _grads = splitter.train_suffix_fast(
                        prepared_batch.boundary,
                        prepared_batch.targets,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                    )
                else:
                    loss, _grads = splitter.train_suffix(
                        prepared_batch.boundary,
                        prepared_batch.targets,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                    )
                _synchronize(device)
                batch_times.append(time.perf_counter() - batch_started)
                losses.append(float(loss.detach().cpu().item()))
            epoch_times.append(time.perf_counter() - epoch_started)
    _synchronize(device)
    training_time = time.perf_counter() - training_started
    return {
        "feature_load_time": float(feature_load_time),
        "training_time": float(training_time),
        "epoch_time_mean": float(np.mean(epoch_times)) if epoch_times else None,
        "batch_time_mean": float(np.mean(batch_times)) if batch_times else None,
        "final_loss": float(losses[-1]) if losses else None,
        "epoch_times": epoch_times,
        "batch_times": batch_times,
    }


def _evaluate_proxy_map(
    *,
    model: torch.nn.Module,
    model_name: str,
    frame_dir: Path,
    annotations: Mapping[str, Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
    split_cache_path: Path | None = None,
    splitter: UniversalModelSplitter | None = None,
    split_candidate: SplitCandidate | None = None,
) -> dict[str, Any]:
    threshold_low, threshold_high = get_model_detection_thresholds(model, model_name)
    return dict(
        _evaluate_detection_proxy_map(
            model,
            frame_dir=str(frame_dir),
            gt_annotations=annotations,
            device=device,
            threshold_low=float(threshold_low),
            threshold_high=float(threshold_high),
            model_name=model_name,
            inference_batch_size=max(1, int(batch_size)),
            split_cache_path=str(split_cache_path) if split_cache_path is not None else None,
            splitter=splitter,
            split_candidate=split_candidate,
        )
    )


def _base_result_row(
    *,
    mode: str,
    edge_model: str,
    golden_model: str,
    sample_count: int,
    epochs: int,
    batch_size: int,
    split_bucket: str | None,
    target_prefix_ratio: float | None,
    candidate: SplitCandidate | None,
    sampled_frame_indices: list[int],
    graph_build_time: float,
    candidate_enumeration_time: float,
    teacher_annotation_time: float,
    repeat_index: int,
    base_seed: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "edge_model": edge_model,
        "golden_model": golden_model,
        "sample_count": int(sample_count),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "split_bucket": split_bucket,
        "target_prefix_ratio": target_prefix_ratio,
        "candidate_id": getattr(candidate, "candidate_id", None),
        "sampled_frame_indices": [int(item) for item in sampled_frame_indices],
        "repeat_index": int(repeat_index),
        "base_seed": int(base_seed),
        "seed": int(seed),
        "device": str(device),
        "success": False,
        "failure_reason": None,
        "total_wall_time": 0.0,
        "graph_build_time": float(graph_build_time),
        "candidate_enumeration_time": float(candidate_enumeration_time),
        "teacher_annotation_time": float(teacher_annotation_time),
        "feature_reconstruction_time": 0.0,
        "feature_load_time": 0.0,
        "training_time": 0.0,
        "epoch_time_mean": None,
        "batch_time_mean": None,
        "peak_cuda_memory_allocated": 0,
        "peak_cuda_memory_reserved": 0,
        "trainable_parameter_count": None,
        "total_parameter_count": None,
        "prefix_parameter_count": None,
        "suffix_parameter_count": None,
        "prefix_parameter_ratio": None,
        "boundary_payload_bytes": int(getattr(candidate, "estimated_payload_bytes", 0) or 0)
        if candidate is not None
        else 0,
        "proxy_mAP@0.5 before": None,
        "proxy_mAP@0.5 after": None,
        "delta proxy_mAP@0.5": None,
        "final_loss": None,
        "epoch_times": [],
        "batch_times": [],
    }


def _mark_failure(row: dict[str, Any], reason: object) -> dict[str, Any]:
    row["success"] = False
    row["failure_reason"] = str(reason)
    return row


def _update_map_metrics(
    row: dict[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    before_map = before.get("map")
    after_map = after.get("map")
    row["proxy_mAP@0.5 before"] = None if before_map is None else float(before_map)
    row["proxy_mAP@0.5 after"] = None if after_map is None else float(after_map)
    if before_map is not None and after_map is not None:
        row["delta proxy_mAP@0.5"] = float(after_map) - float(before_map)


def _label_for_row(row: Mapping[str, Any]) -> str:
    mode = str(row.get("mode") or "")
    bucket = row.get("split_bucket")
    if bucket:
        return f"{mode}/{bucket}"
    return mode


def _successful_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if bool(row.get("success"))]


def _speedup_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    successful = _successful_rows(rows)
    freeze_by_key = {
        (
            int(row.get("repeat_index") or 0),
            int(row.get("sample_count") or 0),
            int(row.get("epochs") or 0),
            str(row.get("candidate_id")),
        ): row
        for row in successful
        if row.get("mode") == "freeze"
    }
    speedups: list[dict[str, Any]] = []
    for row in successful:
        if row.get("mode") not in {"split_cached", "split_rebuild"}:
            continue
        key = (
            int(row.get("repeat_index") or 0),
            int(row.get("sample_count") or 0),
            int(row.get("epochs") or 0),
            str(row.get("candidate_id")),
        )
        baseline = freeze_by_key.get(key)
        if not baseline:
            continue
        split_time = float(row.get("training_time") or 0.0)
        freeze_time = float(baseline.get("training_time") or 0.0)
        if split_time <= 0.0:
            continue
        speedups.append(
            {
                "mode": row.get("mode"),
                "split_bucket": row.get("split_bucket"),
                "candidate_id": row.get("candidate_id"),
                "sample_count": int(row.get("sample_count") or 0),
                "epochs": int(row.get("epochs") or 0),
                "prefix_parameter_ratio": float(row.get("prefix_parameter_ratio") or 0.0),
                "speedup": freeze_time / split_time,
            }
        )
    return speedups


def _time_reduction_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    successful = _successful_rows(rows)
    freeze_by_key = {
        (
            int(row.get("repeat_index") or 0),
            int(row.get("sample_count") or 0),
            int(row.get("epochs") or 0),
            str(row.get("candidate_id")),
        ): row
        for row in successful
        if row.get("mode") == "freeze"
    }
    reductions: list[dict[str, Any]] = []
    for row in successful:
        if row.get("mode") not in {"split_cached", "split_rebuild"}:
            continue
        key = (
            int(row.get("repeat_index") or 0),
            int(row.get("sample_count") or 0),
            int(row.get("epochs") or 0),
            str(row.get("candidate_id")),
        )
        baseline = freeze_by_key.get(key)
        if not baseline:
            continue
        split_time = row.get("training_time")
        freeze_time = baseline.get("training_time")
        if split_time is None or freeze_time is None:
            continue
        reductions.append(
            {
                "mode": row.get("mode"),
                "split_bucket": row.get("split_bucket"),
                "candidate_id": row.get("candidate_id"),
                "sample_count": int(row.get("sample_count") or 0),
                "epochs": int(row.get("epochs") or 0),
                "repeat_index": int(row.get("repeat_index") or 0),
                "reduction": float(freeze_time) - float(split_time),
            }
        )
    return reductions


def _plot_mean_std_line(
    ax: Any,
    rows: list[Mapping[str, Any]],
    *,
    x_field: str,
    y_field: str,
    label: str,
    marker: str = "o",
) -> None:
    by_x: dict[int, list[Any]] = {}
    for row in rows:
        by_x.setdefault(int(row.get(x_field) or 0), []).append(row.get(y_field))
    xs = sorted(by_x)
    means: list[float] = []
    stds: list[float] = []
    for x in xs:
        mean, std = _mean_std(by_x[x])
        if mean is None:
            continue
        means.append(mean)
        stds.append(float(std or 0.0))
    if not means:
        return
    xs = xs[: len(means)]
    ax.plot(xs, means, marker=marker, linewidth=1.6, label=label)
    lower = [mean - std for mean, std in zip(means, stds)]
    upper = [mean + std for mean, std in zip(means, stds)]
    if any(std > 0.0 for std in stds):
        ax.fill_between(xs, lower, upper, alpha=0.16)


def _write_overview_plot(rows: list[Mapping[str, Any]], output_root: Path) -> None:
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        logger.warning("matplotlib is unavailable; skipping overview plot: {}", exc)
        return

    successful = _successful_rows(rows)
    aggregate = _aggregate_rows(rows)
    speedups = _speedup_rows(rows)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), constrained_layout=True)
    ax_time, ax_speed, ax_pareto, ax_status = axes.ravel()

    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in successful:
        key = (
            str(row.get("mode")),
            str(row.get("split_bucket") or "Full"),
            int(row.get("sample_count") or 0),
        )
        grouped.setdefault(key, []).append(row)
    for key, items in sorted(grouped.items()):
        mode, bucket, sample_count = key
        _plot_mean_std_line(
            ax_time,
            items,
            x_field="epochs",
            y_field="training_time",
            label=f"{mode}/{bucket}/n={sample_count}",
        )
    ax_time.set_title("A. Training Time Across Repeats")
    ax_time.set_xlabel("Epochs")
    ax_time.set_ylabel("Training time (s)")
    if grouped:
        ax_time.legend(ncol=2)
    ax_time.grid(alpha=0.25)

    speedup_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in speedups:
        key = (
            str(row.get("mode")),
            str(row.get("split_bucket")),
            str(row.get("candidate_id")),
        )
        speedup_groups.setdefault(key, []).append(row)
    for key, items in sorted(speedup_groups.items()):
        mode, bucket, _candidate_id = key
        x_mean, x_std = _mean_std([item.get("prefix_parameter_ratio") for item in items])
        y_mean, y_std = _mean_std([item.get("speedup") for item in items])
        if x_mean is None or y_mean is None:
            continue
        ax_speed.errorbar(
            x_mean,
            y_mean,
            xerr=x_std or 0.0,
            yerr=y_std or 0.0,
            marker="o",
            capsize=3,
            linestyle="none",
            label=f"{mode}/{bucket}",
        )
    ax_speed.axhline(1.0, color="0.45", linestyle="--", linewidth=1.0)
    ax_speed.set_title("B. Speedup vs Freeze@c")
    ax_speed.set_xlabel("Prefix parameter ratio")
    ax_speed.set_ylabel("Speedup (mean +/- std)")
    if speedup_groups:
        ax_speed.legend(ncol=2)
    ax_speed.grid(alpha=0.25)

    pareto_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in successful:
        if row.get("delta proxy_mAP@0.5") is None:
            continue
        key = (str(row.get("mode")), str(row.get("split_bucket") or "Full"))
        pareto_groups.setdefault(key, []).append(row)
    for key, items in sorted(pareto_groups.items()):
        x_mean, x_std = _mean_std([item.get("training_time") for item in items])
        y_mean, y_std = _mean_std([item.get("delta proxy_mAP@0.5") for item in items])
        if x_mean is None or y_mean is None:
            continue
        ax_pareto.errorbar(
            x_mean,
            y_mean,
            xerr=x_std or 0.0,
            yerr=y_std or 0.0,
            marker="o",
            capsize=3,
            linestyle="none",
            label="/".join(key),
        )
    ax_pareto.axhline(0.0, color="0.55", linestyle=":", linewidth=1.0)
    ax_pareto.set_title("C. Time-Accuracy Pareto")
    ax_pareto.set_xlabel("Training time (s)")
    ax_pareto.set_ylabel("Delta proxy mAP@0.5")
    if pareto_groups:
        ax_pareto.legend(ncol=2)
    ax_pareto.grid(alpha=0.25)

    labels = [_label_for_row(row) for row in aggregate]
    success_rates = [float(row.get("success_rate") or 0.0) for row in aggregate]
    if labels:
        y_positions = np.arange(len(labels))
        colors = ["#2ca25f" if rate >= 1.0 else "#de8f05" for rate in success_rates]
        ax_status.barh(y_positions, success_rates, color=colors, alpha=0.88)
        ax_status.set_yticks(y_positions)
        ax_status.set_yticklabels(labels)
        ax_status.set_xlim(0.0, 1.05)
        for pos, row in zip(y_positions, aggregate):
            text = f"{int(row['success_count'])}/{int(row['run_count'])}"
            ax_status.text(
                min(1.01, float(row.get("success_rate") or 0.0) + 0.02),
                pos,
                text,
                va="center",
                fontsize=8,
            )
    ax_status.set_title("D. Successful Runs")
    ax_status.set_xlabel("Success rate")
    ax_status.grid(axis="x", alpha=0.25)

    repeat_count = len({int(row.get("repeat_index") or 0) for row in rows})
    sample_counts = sorted({int(row.get("sample_count") or 0) for row in rows})
    epoch_counts = sorted({int(row.get("epochs") or 0) for row in rows})
    fig.suptitle(
        "Tail Training Motivation Overview "
        f"(repeats={repeat_count}, samples={sample_counts}, epochs={epoch_counts})",
        fontsize=13,
    )
    fig.savefig(plots_dir / "tail_training_motivation_overview.pdf")
    plt.close(fig)


def _write_training_time_reduction_boxplot(
    rows: list[Mapping[str, Any]],
    output_root: Path,
) -> None:
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:
        logger.warning("matplotlib is unavailable; skipping reduction boxplot: {}", exc)
        return

    reductions = _time_reduction_rows(rows)
    bucket_order = [*BUCKET_LABELS, "Auto"]
    mode_styles = (
        ("split_cached", "Best-case: Cached", "solid", "o"),
        ("split_rebuild", "Worst-case: Rebuild", (0, (4, 2)), "s"),
    )
    fill_color = "#19f419"
    edge_color = "#083bff"

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    plotted = False
    legend_handles: list[Any] = []
    rng = np.random.default_rng(20240509)

    for mode_index, (mode, label, linestyle, marker) in enumerate(mode_styles):
        offset = -0.18 if mode_index == 0 else 0.18
        data: list[list[float]] = []
        positions: list[float] = []
        for bucket_index, bucket in enumerate(bucket_order, start=1):
            values: list[float] = []
            for row in reductions:
                if row.get("mode") != mode or row.get("split_bucket") != bucket:
                    continue
                reduction = row.get("reduction")
                if reduction is None:
                    continue
                value = float(reduction)
                if np.isfinite(value):
                    values.append(value)
            if values:
                data.append(values)
                positions.append(float(bucket_index) + offset)
        if not data:
            continue

        plotted = True
        boxplot = ax.boxplot(
            data,
            positions=positions,
            widths=0.30,
            patch_artist=True,
            manage_ticks=False,
            showmeans=True,
            boxprops={
                "facecolor": fill_color,
                "edgecolor": edge_color,
                "linewidth": 1.6,
                "linestyle": linestyle,
            },
            medianprops={"color": edge_color, "linewidth": 1.5},
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": edge_color,
                "markersize": 4.5,
            },
            whiskerprops={"color": edge_color, "linewidth": 1.3, "linestyle": linestyle},
            capprops={"color": edge_color, "linewidth": 1.3, "linestyle": linestyle},
            flierprops={
                "marker": marker,
                "markerfacecolor": "white",
                "markeredgecolor": edge_color,
                "markersize": 4,
                "alpha": 0.55,
            },
        )
        for patch in boxplot["boxes"]:
            patch.set_linestyle(linestyle)
            patch.set_alpha(0.78)

        for position, values in zip(positions, data):
            jitter = rng.uniform(-0.045, 0.045, size=len(values)) if len(values) > 1 else [0.0]
            ax.scatter(
                [position + float(delta) for delta in jitter],
                values,
                marker=marker,
                facecolors="white",
                edgecolors=edge_color,
                linewidths=0.7,
                s=20,
                alpha=0.72,
                zorder=3,
            )

        legend_handles.append(
            Patch(
                facecolor=fill_color,
                edgecolor=edge_color,
                linewidth=1.6,
                linestyle=linestyle,
                label=label,
                alpha=0.78,
            )
        )

    ax.axhline(0.0, color="0.42", linestyle="--", linewidth=1.0)
    ax.set_xticks(np.arange(1, len(bucket_order) + 1))
    ax.set_xticklabels(bucket_order)
    ax.set_xlabel("Split candidate")
    ax.set_ylabel("Reduction (s)")
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.32)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    repeat_count = len({int(row.get("repeat_index") or 0) for row in rows})
    sample_counts = sorted({int(row.get("sample_count") or 0) for row in rows})
    epoch_counts = sorted({int(row.get("epochs") or 0) for row in rows})
    ax.set_title(
        "Training Time Reduction vs Freeze@c\n"
        f"repeats={repeat_count}, samples={sample_counts}, epochs={epoch_counts}",
        pad=14,
    )
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.26),
            ncol=2,
            frameon=True,
            fancybox=False,
            edgecolor="0.35",
        )
    if not plotted:
        ax.text(
            0.5,
            0.52,
            "No successful SplitTrain and Freeze@c pairs to compare.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="0.35",
        )

    fig.tight_layout()
    fig.savefig(plots_dir / "training_time_reduction_boxplot.pdf")
    plt.close(fig)


def _write_split_position_time_accuracy_plot(
    rows: list[Mapping[str, Any]],
    output_root: Path,
) -> None:
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:
        logger.warning("matplotlib is unavailable; skipping split position plot: {}", exc)
        return

    successful = _successful_rows(rows)
    bucket_order = [
        bucket
        for bucket in [*BUCKET_LABELS, "Auto"]
        if any(row.get("split_bucket") == bucket for row in successful)
    ]
    if not bucket_order:
        return

    mode_styles = (
        {
            "mode": "freeze",
            "label": "Freeze@c",
            "offset": -0.26,
            "face": "#d9d9d9",
            "edge": "#4d4d4d",
            "line": "solid",
            "marker": "o",
        },
        {
            "mode": "split_cached",
            "label": "SplitTrain-Cached",
            "offset": 0.0,
            "face": "#33ff33",
            "edge": "#083bff",
            "line": "solid",
            "marker": "s",
        },
        {
            "mode": "split_rebuild",
            "label": "SplitTrain-Rebuild",
            "offset": 0.26,
            "face": "#caff1a",
            "edge": "#083bff",
            "line": (0, (4, 2)),
            "marker": "^",
        },
    )

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax_time = plt.subplots(figsize=(8.4, 4.2))
    ax_acc = ax_time.twinx()
    rng = np.random.default_rng(20240510)
    x_positions = np.arange(1, len(bucket_order) + 1, dtype=float)
    plotted_time = False
    plotted_accuracy = False
    accuracy_values: list[float] = []

    for style in mode_styles:
        mode = str(style["mode"])
        data: list[list[float]] = []
        positions: list[float] = []
        accuracy_means: list[float | None] = []
        accuracy_stds: list[float] = []
        accuracy_positions: list[float] = []

        for bucket_index, bucket in enumerate(bucket_order, start=1):
            items = [
                row
                for row in successful
                if row.get("mode") == mode and row.get("split_bucket") == bucket
            ]
            times = [
                float(row.get("training_time"))
                for row in items
                if row.get("training_time") is not None
                and np.isfinite(float(row.get("training_time")))
            ]
            if times:
                data.append(times)
                positions.append(float(bucket_index) + float(style["offset"]))

            mean, std = _mean_std(
                [
                    row.get("proxy_mAP@0.5 after")
                    for row in items
                    if row.get("proxy_mAP@0.5 after") is not None
                ]
            )
            if mean is not None:
                accuracy_means.append(mean)
                accuracy_stds.append(float(std or 0.0))
                accuracy_positions.append(float(bucket_index) + float(style["offset"]))
                accuracy_values.append(mean)

        if data:
            plotted_time = True
            boxplot = ax_time.boxplot(
                data,
                positions=positions,
                widths=0.22,
                patch_artist=True,
                manage_ticks=False,
                showmeans=True,
                boxprops={
                    "facecolor": str(style["face"]),
                    "edgecolor": str(style["edge"]),
                    "linewidth": 1.6,
                    "linestyle": style["line"],
                },
                medianprops={"color": str(style["edge"]), "linewidth": 1.4},
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "white",
                    "markeredgecolor": str(style["edge"]),
                    "markersize": 4,
                },
                whiskerprops={
                    "color": str(style["edge"]),
                    "linewidth": 1.2,
                    "linestyle": style["line"],
                },
                capprops={"color": str(style["edge"]), "linewidth": 1.2},
                flierprops={
                    "marker": str(style["marker"]),
                    "markerfacecolor": "white",
                    "markeredgecolor": str(style["edge"]),
                    "markersize": 4,
                    "alpha": 0.6,
                },
            )
            for patch in boxplot["boxes"]:
                patch.set_alpha(0.78)
                patch.set_linestyle(style["line"])
            for position, values in zip(positions, data):
                jitter = rng.uniform(-0.035, 0.035, size=len(values)) if len(values) > 1 else [0.0]
                ax_time.scatter(
                    [position + float(delta) for delta in jitter],
                    values,
                    marker=str(style["marker"]),
                    facecolors="white",
                    edgecolors=str(style["edge"]),
                    linewidths=0.7,
                    s=18,
                    alpha=0.65,
                    zorder=3,
                )

        if accuracy_positions:
            plotted_accuracy = True
            ax_acc.errorbar(
                accuracy_positions,
                accuracy_means,
                yerr=accuracy_stds,
                marker=str(style["marker"]),
                linestyle=style["line"],
                linewidth=1.5,
                color=str(style["edge"]),
                markerfacecolor=str(style["face"]),
                markeredgecolor=str(style["edge"]),
                capsize=3,
                alpha=0.9,
            )

    labels: list[str] = []
    for bucket in bucket_order:
        ratios = [
            float(row.get("prefix_parameter_ratio"))
            for row in successful
            if row.get("split_bucket") == bucket
            and row.get("prefix_parameter_ratio") is not None
            and np.isfinite(float(row.get("prefix_parameter_ratio")))
        ]
        if ratios:
            labels.append(f"{bucket}\n{float(np.mean(ratios)):.0%}")
        else:
            labels.append(bucket)

    ax_time.set_xticks(x_positions)
    ax_time.set_xticklabels(labels)
    ax_time.set_xlabel("Split position")
    ax_time.set_ylabel("Training time (s)")
    ax_acc.set_ylabel("Proxy mAP@0.5 after")
    ax_time.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.34)
    ax_time.set_axisbelow(True)
    ax_time.set_xlim(0.45, len(bucket_order) + 0.55)
    ax_time.set_ylim(bottom=0.0)

    if accuracy_values:
        min_acc = min(accuracy_values)
        max_acc = max(accuracy_values)
        pad = max(0.01, (max_acc - min_acc) * 0.18)
        ax_acc.set_ylim(max(0.0, min_acc - pad), min(1.0, max_acc + pad))

    for spine in ("top",):
        ax_time.spines[spine].set_visible(False)
        ax_acc.spines[spine].set_visible(False)

    legend_handles = [
        Patch(
            facecolor=str(style["face"]),
            edgecolor=str(style["edge"]),
            linewidth=1.6,
            linestyle=style["line"],
            label=str(style["label"]),
            alpha=0.78,
        )
        for style in mode_styles
    ]
    ax_time.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="0.35",
    )

    if not plotted_time and not plotted_accuracy:
        ax_time.text(
            0.5,
            0.52,
            "No successful split-position runs to plot.",
            ha="center",
            va="center",
            transform=ax_time.transAxes,
            color="0.35",
        )

    fig.subplots_adjust(top=0.82, bottom=0.19, left=0.10, right=0.88)
    fig.savefig(plots_dir / "split_position_time_accuracy_dual_axis.pdf")
    fig.savefig(plots_dir / "split_position_time_accuracy_dual_axis.png", dpi=220)
    plt.close(fig)


def _run_one_experiment(
    *,
    mode: str,
    edge_model: torch.nn.Module,
    split_model: torch.nn.Module,
    model_name: str,
    golden_model: str,
    initial_state: Mapping[str, Any],
    splitter: UniversalModelSplitter,
    choice: CandidateChoice | None,
    cached_feature_path: Path | None,
    cached_feature_failure: str | None,
    frame_dir: Path,
    frames_by_id: Mapping[int, np.ndarray],
    sampled_frame_indices: list[int],
    annotations: Mapping[str, Mapping[str, Any]],
    sample_count: int,
    epochs: int,
    batch_size: int,
    output_root: Path,
    graph_build_time: float,
    candidate_enumeration_time: float,
    teacher_annotation_time: float,
    learning_rate: float,
    optimizer_config: Mapping[str, Any],
    repeat_index: int,
    base_seed: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    candidate = choice.candidate if choice is not None else None
    row = _base_result_row(
        mode=mode,
        edge_model=model_name,
        golden_model=golden_model,
        sample_count=sample_count,
        epochs=epochs,
        batch_size=batch_size,
        split_bucket=choice.bucket if choice is not None else None,
        target_prefix_ratio=choice.target_ratio if choice is not None else None,
        candidate=candidate,
        sampled_frame_indices=sampled_frame_indices,
        graph_build_time=graph_build_time,
        candidate_enumeration_time=candidate_enumeration_time,
        teacher_annotation_time=teacher_annotation_time,
        repeat_index=repeat_index,
        base_seed=base_seed,
        seed=seed,
        device=device,
    )
    del output_root
    run_started = time.perf_counter()
    _reset_cuda_peak(device)
    optimizer = None
    try:
        _set_random_seed(seed)
        _restore_model_state(edge_model, initial_state)
        edge_model.to(device)
        loss_fn = build_split_training_loss(edge_model)
        if loss_fn is None:
            raise RuntimeError(f"No split-training loss is available for {model_name}.")

        total_params = _total_parameter_count(split_model)
        row["total_parameter_count"] = total_params

        split_cache_path: Path | None = None
        split_candidate: SplitCandidate | None = None

        if mode == "full":
            for parameter in split_model.parameters():
                parameter.requires_grad_(True)
            trainable_params = [
                parameter
                for parameter in split_model.parameters()
                if parameter.requires_grad
            ]
            row["trainable_parameter_count"] = _count_parameters(trainable_params)
            row["prefix_parameter_count"] = 0
            row["suffix_parameter_count"] = total_params
            row["prefix_parameter_ratio"] = 0.0
            row["boundary_payload_bytes"] = 0
            optimizer = _make_optimizer(
                split_model,
                runtime=None,
                learning_rate=learning_rate,
                optimizer_config=optimizer_config,
            )
            before_metrics = _evaluate_proxy_map(
                model=edge_model,
                model_name=model_name,
                frame_dir=frame_dir,
                annotations=annotations,
                device=device,
                batch_size=batch_size,
            )
            train_metrics = _train_raw_loop(
                edge_model=edge_model,
                split_model=split_model,
                model_name=model_name,
                frames_by_id=frames_by_id,
                sample_ids=sampled_frame_indices,
                annotations=annotations,
                num_epoch=epochs,
                batch_size=batch_size,
                device=device,
                loss_fn=loss_fn,
                optimizer=optimizer,
                seed=seed,
                shuffle_samples=bool(optimizer_config.get("shuffle_samples", False)),
            )
            after_metrics = _evaluate_proxy_map(
                model=edge_model,
                model_name=model_name,
                frame_dir=frame_dir,
                annotations=annotations,
                device=device,
                batch_size=batch_size,
            )
        else:
            if candidate is None:
                raise RuntimeError(f"{mode} requires a split candidate.")
            splitter.split(candidate=candidate)
            suffix_params, suffix_names = _resolve_suffix_trainable_parameters(
                split_model,
                splitter,
            )
            suffix_param_count = _count_parameters(suffix_params)
            row["suffix_parameter_names"] = suffix_names
            row["trainable_parameter_count"] = suffix_param_count
            row["suffix_parameter_count"] = suffix_param_count
            row["prefix_parameter_count"] = max(0, total_params - suffix_param_count)
            row["prefix_parameter_ratio"] = (
                float(row["prefix_parameter_count"]) / float(total_params)
                if total_params > 0
                else 0.0
            )
            row["boundary_payload_bytes"] = int(
                getattr(candidate, "estimated_payload_bytes", 0) or 0
            )
            optimizer_runtime = splitter if mode.startswith("split_") else None
            optimizer = _make_optimizer(
                split_model,
                runtime=optimizer_runtime,
                learning_rate=learning_rate,
                optimizer_config=optimizer_config,
            )

            if mode == "freeze":
                before_metrics = _evaluate_proxy_map(
                    model=edge_model,
                    model_name=model_name,
                    frame_dir=frame_dir,
                    annotations=annotations,
                    device=device,
                    batch_size=batch_size,
                )
                train_metrics = _train_raw_loop(
                    edge_model=edge_model,
                    split_model=split_model,
                    model_name=model_name,
                    frames_by_id=frames_by_id,
                    sample_ids=sampled_frame_indices,
                    annotations=annotations,
                    num_epoch=epochs,
                    batch_size=batch_size,
                    device=device,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    seed=seed,
                    shuffle_samples=bool(optimizer_config.get("shuffle_samples", False)),
                )
                after_metrics = _evaluate_proxy_map(
                    model=edge_model,
                    model_name=model_name,
                    frame_dir=frame_dir,
                    annotations=annotations,
                    device=device,
                    batch_size=batch_size,
                )
            else:
                if mode == "split_cached":
                    if cached_feature_failure:
                        raise RuntimeError(cached_feature_failure)
                    if cached_feature_path is None:
                        raise RuntimeError("Missing cached BoundaryPayload feature cache.")
                    split_cache_path = cached_feature_path
                elif mode == "split_rebuild":
                    split_cache_path = (
                        Path("split_rebuild")
                        / _safe_segment(choice.bucket)
                        / _safe_segment(candidate.candidate_id)
                        / f"samples_{sample_count}"
                        / f"epochs_{epochs}"
                    )
                    split_cache_path = frame_dir.parent / split_cache_path
                    rebuild_time, _records, actual_bytes = _rebuild_feature_cache(
                        splitter=splitter,
                        edge_model=edge_model,
                        frames_by_id=frames_by_id,
                        sample_ids=sampled_frame_indices,
                        cache_path=split_cache_path,
                        device=device,
                    )
                    row["feature_reconstruction_time"] = float(rebuild_time)
                    if actual_bytes > 0:
                        row["boundary_payload_bytes_actual"] = int(actual_bytes)
                else:
                    raise RuntimeError(f"Unsupported mode: {mode}")

                split_candidate = candidate
                before_metrics = _evaluate_proxy_map(
                    model=edge_model,
                    model_name=model_name,
                    frame_dir=frame_dir,
                    annotations=annotations,
                    device=device,
                    batch_size=batch_size,
                )
                train_metrics = _train_split_loop(
                    split_model=split_model,
                    splitter=splitter,
                    cache_path=split_cache_path,
                    sample_ids=sampled_frame_indices,
                    annotations=annotations,
                    num_epoch=epochs,
                    batch_size=batch_size,
                    device=device,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    seed=seed,
                    shuffle_samples=bool(optimizer_config.get("shuffle_samples", False)),
                )
                after_metrics = _evaluate_proxy_map(
                    model=edge_model,
                    model_name=model_name,
                    frame_dir=frame_dir,
                    annotations=annotations,
                    device=device,
                    batch_size=batch_size,
                )

        row.update({key: value for key, value in train_metrics.items() if key in row})
        row["success"] = True
        _update_map_metrics(row, before_metrics, after_metrics)
    except Exception as exc:  # noqa: BLE001 - experiment rows must capture failures and continue.
        logger.exception(
            "Experiment failed mode={} samples={} epochs={} bucket={} candidate={}: {}",
            mode,
            sample_count,
            epochs,
            choice.bucket if choice is not None else None,
            getattr(candidate, "candidate_id", None),
            exc,
        )
        _mark_failure(row, exc)
    finally:
        allocated, reserved = _cuda_peak(device)
        row["peak_cuda_memory_allocated"] = allocated
        row["peak_cuda_memory_reserved"] = reserved
        row["total_wall_time"] = float(time.perf_counter() - run_started)
        try:
            if optimizer is not None:
                del optimizer
        finally:
            _restore_model_state(edge_model, initial_state)
            for parameter in split_model.parameters():
                parameter.grad = None
            _clear_cuda_cache()
    return row


def _write_plots(rows: list[Mapping[str, Any]], output_root: Path) -> None:
    _write_overview_plot(rows, output_root)
    _write_split_position_time_accuracy_plot(rows, output_root)
    _write_training_time_reduction_boxplot(rows, output_root)
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        logger.warning("matplotlib is unavailable; skipping PDF plot generation: {}", exc)
        return

    successful = [dict(row) for row in rows if bool(row.get("success"))]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in successful:
        key = (
            str(row.get("mode")),
            str(row.get("split_bucket") or "Full"),
            int(row.get("sample_count") or 0),
        )
        grouped.setdefault(key, []).append(row)
    for (mode, bucket, sample_count), items in sorted(grouped.items()):
        items.sort(key=lambda item: int(item.get("epochs") or 0))
        ax.plot(
            [int(item.get("epochs") or 0) for item in items],
            [float(item.get("training_time") or 0.0) for item in items],
            marker="o",
            linewidth=1.5,
            label=f"{mode}/{bucket}/n={sample_count}",
        )
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Training time (s)")
    ax.set_title("Training Time vs Epochs")
    if grouped:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plots_dir / "training_time_vs_epochs.pdf")
    plt.close(fig)

    speedup_rows = _speedup_rows(rows)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for mode in ("split_cached", "split_rebuild"):
        items = [row for row in speedup_rows if row["mode"] == mode]
        if not items:
            continue
        ax.scatter(
            [row["prefix_parameter_ratio"] for row in items],
            [row["speedup"] for row in items],
            label=mode,
            alpha=0.85,
        )
    ax.axhline(1.0, color="0.5", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Prefix parameter ratio")
    ax.set_ylabel("Speedup vs Freeze@c")
    ax.set_title("Speedup vs Split Depth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "speedup_vs_split_depth.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for mode in DEFAULT_MODES:
        items = [
            row
            for row in successful
            if row.get("mode") == mode and row.get("delta proxy_mAP@0.5") is not None
        ]
        if not items:
            continue
        ax.scatter(
            [float(row.get("training_time") or 0.0) for row in items],
            [float(row.get("delta proxy_mAP@0.5") or 0.0) for row in items],
            label=mode,
            alpha=0.85,
        )
    ax.set_xlabel("Training time (s)")
    ax.set_ylabel("Delta proxy mAP@0.5")
    ax.set_title("Time-Accuracy Pareto")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "time_accuracy_pareto.pdf")
    plt.close(fig)


def _prepare_configs(args: argparse.Namespace) -> tuple[Any, Any]:
    base_config = load_runtime_config(Path(args.yaml_path))
    client_cfg = copy.deepcopy(base_config.client)
    server_cfg = copy.deepcopy(base_config.server)
    client_cfg.lightweight = str(args.edge_model)
    client_cfg.weights_path = _resolve_local_weights_path(str(args.edge_model))
    server_cfg.edge_model_name = str(args.edge_model)
    server_cfg.golden = str(args.golden_model)
    server_cfg.weights_path = _resolve_local_weights_path(str(args.golden_model))
    server_cfg.continual_learning.num_epoch = max(int(epoch) for epoch in args.epochs)
    server_cfg.continual_learning.batch_size = int(args.batch_size)
    server_cfg.das.enabled = False
    return client_cfg, server_cfg


def _build_split_setup(
    *,
    edge_model: torch.nn.Module,
    edge_model_name: str,
    first_frame: np.ndarray,
    trace_batch_size: int,
    fixed_split_cfg: Any,
    cache_path: Path,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    UniversalModelSplitter,
    list[CandidateChoice],
    float,
    float,
]:
    split_model = get_split_runtime_model(edge_model)
    split_model.to(device)
    loss_fn = build_split_training_loss(edge_model)
    splitter = UniversalModelSplitter(device=device)
    splitter.trainability_loss_fn = loss_fn
    sample_input = prepare_split_runtime_input(edge_model, first_frame, device=device)
    sample_input = _make_trace_input(sample_input, trace_batch_size)
    model_family = get_model_family(edge_model_name)

    _synchronize(device)
    graph_started = time.perf_counter()
    splitter.trace(
        split_model,
        sample_input,
        model_name=edge_model_name,
        model_family=model_family,
    )
    _synchronize(device)
    graph_build_time = time.perf_counter() - graph_started

    constraints = SplitConstraints.from_config(fixed_split_cfg)
    _synchronize(device)
    enum_started = time.perf_counter()
    candidates = splitter.enumerate_candidates(
        max_candidates=constraints.max_candidates,
        max_boundary_count=constraints.max_boundary_count,
        max_payload_bytes=constraints.max_payload_bytes,
    )
    eligible = _filter_candidates(candidates, constraints)
    auto_plan = load_or_compute_fixed_split_plan(
        split_model,
        constraints,
        sample_input=sample_input,
        device=device,
        model_name=edge_model_name,
        cache_path=str(cache_path),
        splitter=splitter,
    )
    auto_candidate = splitter.split(candidate_id=auto_plan.candidate_id)
    if not _candidate_satisfies_constraints(auto_candidate, constraints):
        raise RuntimeError(
            "The fixed split planner returned a candidate that does not satisfy "
            "the experiment's trainable/constraint filters."
        )
    choices = _select_candidate_choices(
        eligible,
        auto_candidate,
        list(DEFAULT_BOUNDARY_QUANTILES),
    )
    _synchronize(device)
    candidate_enumeration_time = time.perf_counter() - enum_started
    return split_model, splitter, choices, graph_build_time, candidate_enumeration_time


def _replace_quantile_choices(
    choices: list[CandidateChoice],
    boundary_quantiles: list[float],
    eligible_candidates: list[SplitCandidate],
) -> list[CandidateChoice]:
    auto = next(choice.candidate for choice in choices if choice.bucket == "Auto")
    return _select_candidate_choices(eligible_candidates, auto, boundary_quantiles)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = output_root / "results.jsonl"
    summary_path = output_root / "summary.csv"
    aggregate_summary_path = output_root / "aggregate_summary.csv"
    for path in (results_path, summary_path, aggregate_summary_path):
        if path.exists():
            path.unlink()

    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device requested CUDA, but torch.cuda.is_available() is false.")
    object_detection_module.device = device
    _set_random_seed(int(args.seed))
    repeats = max(1, int(args.repeats))

    client_cfg, server_cfg = _prepare_configs(args)

    logger.info("Sampling frames from {}", args.video_path)
    frames_by_id: dict[int, np.ndarray] = {}
    sample_ids_by_repeat_count: dict[int, dict[int, list[int]]] = {}
    for repeat_index in range(repeats):
        repeat_frames, repeat_sample_ids = _sample_video_frames(
            Path(args.video_path),
            [int(count) for count in args.sample_counts],
            seed=int(args.seed) + repeat_index,
        )
        frames_by_id.update(repeat_frames)
        sample_ids_by_repeat_count[repeat_index] = repeat_sample_ids
    max_sample_ids = sorted(
        {
            frame_id
            for selected in sample_ids_by_repeat_count.values()
            for ids in selected.values()
            for frame_id in ids
        }
    )
    frame_dir = output_root / "frames"
    _write_raw_frames(frame_dir, frames_by_id)

    logger.info("Loading edge model {} and teacher {}", args.edge_model, args.golden_model)
    edge_detector = Object_Detection(client_cfg, type="small inference")
    teacher_detector = Object_Detection(server_cfg, type="large inference")
    edge_detector.model.to(device)
    teacher_detector.model.to(device)

    teacher_threshold = float(
        getattr(server_cfg.continual_learning, "teacher_annotation_threshold", 0.5)
    )
    annotations, teacher_annotation_time = _load_or_collect_teacher_annotations(
        cache_path=output_root / "teacher_labels.json",
        detector=teacher_detector,
        frames_by_id=frames_by_id,
        frame_ids=max_sample_ids,
        golden_model=str(args.golden_model),
        video_path=Path(args.video_path),
        threshold=teacher_threshold,
        batch_size=max(
            1,
            int(getattr(server_cfg.continual_learning, "teacher_batch_size", args.batch_size)),
        ),
        device=device,
    )

    first_frame = frames_by_id[max_sample_ids[0]]
    split_model, splitter, default_choices, graph_build_time, candidate_enumeration_time = (
        _build_split_setup(
            edge_model=edge_detector.model,
            edge_model_name=str(args.edge_model),
            first_frame=first_frame,
            trace_batch_size=int(getattr(server_cfg.continual_learning, "trace_batch_size", 2)),
            fixed_split_cfg=client_cfg.split_learning.fixed_split,
            cache_path=output_root / "fixed_split_plan.json",
            device=device,
        )
    )
    constraints = SplitConstraints.from_config(client_cfg.split_learning.fixed_split)
    eligible_candidates = _filter_candidates(
        splitter.enumerate_candidates(
            max_candidates=constraints.max_candidates,
            max_boundary_count=constraints.max_boundary_count,
            max_payload_bytes=constraints.max_payload_bytes,
        ),
        constraints,
    )
    choices = _replace_quantile_choices(
        default_choices,
        [float(value) for value in args.boundary_quantiles],
        eligible_candidates,
    )

    logger.info("Selected split candidates:")
    for choice in choices:
        logger.info(
            "  {} target={} candidate={} prefix_ratio={:.4f} payload={} bytes",
            choice.bucket,
            choice.target_ratio,
            choice.candidate.candidate_id,
            _candidate_prefix_ratio(choice.candidate),
            int(choice.candidate.estimated_payload_bytes),
        )

    initial_state = _snapshot_model_state(edge_detector.model)
    cached_feature_paths: dict[str, Path] = {}
    cached_feature_failures: dict[str, str] = {}
    if "split_cached" in set(args.modes):
        unique_candidates = {
            str(choice.candidate.candidate_id): choice.candidate
            for choice in choices
        }
        for candidate_id, candidate in unique_candidates.items():
            cache_path = output_root / "cached_features" / _safe_segment(candidate_id)
            try:
                logger.info("Building pre-cached BoundaryPayload records for {}", candidate_id)
                splitter.split(candidate=candidate)
                _rebuild_feature_cache(
                    splitter=splitter,
                    edge_model=edge_detector.model,
                    frames_by_id=frames_by_id,
                    sample_ids=max_sample_ids,
                    cache_path=cache_path,
                    device=device,
                )
                cached_feature_paths[candidate_id] = cache_path
            except Exception as exc:  # noqa: BLE001 - cached mode rows should report failures.
                cached_feature_failures[candidate_id] = str(exc)
                logger.exception("Failed to prebuild cached features for {}: {}", candidate_id, exc)
        _restore_model_state(edge_detector.model, initial_state)

    learning_rate = _resolve_experiment_learning_rate(server_cfg, str(args.edge_model))
    optimizer_config = _optimizer_overrides(str(args.edge_model))
    rows: list[dict[str, Any]] = []

    for repeat_index in range(repeats):
        run_seed = int(args.seed) + repeat_index
        for sample_count in [int(count) for count in args.sample_counts]:
            sampled_ids = list(sample_ids_by_repeat_count[repeat_index][int(sample_count)])
            sample_annotations = {
                str(frame_id): dict(annotations.get(str(frame_id), {"boxes": [], "labels": []}))
                for frame_id in sampled_ids
            }
            for epochs in [int(epoch) for epoch in args.epochs]:
                if "full" in args.modes:
                    row = _run_one_experiment(
                        mode="full",
                        edge_model=edge_detector.model,
                        split_model=split_model,
                        model_name=str(args.edge_model),
                        golden_model=str(args.golden_model),
                        initial_state=initial_state,
                        splitter=splitter,
                        choice=None,
                        cached_feature_path=None,
                        cached_feature_failure=None,
                        frame_dir=frame_dir,
                        frames_by_id=frames_by_id,
                        sampled_frame_indices=sampled_ids,
                        annotations=sample_annotations,
                        sample_count=sample_count,
                        epochs=epochs,
                        batch_size=int(args.batch_size),
                        output_root=output_root,
                        graph_build_time=graph_build_time,
                        candidate_enumeration_time=candidate_enumeration_time,
                        teacher_annotation_time=teacher_annotation_time,
                        learning_rate=learning_rate,
                        optimizer_config=optimizer_config,
                        repeat_index=repeat_index,
                        base_seed=int(args.seed),
                        seed=run_seed,
                        device=device,
                    )
                    rows.append(row)
                    _append_jsonl(results_path, row)
                for choice in choices:
                    candidate_id = str(choice.candidate.candidate_id)
                    for mode in [item for item in args.modes if item != "full"]:
                        row = _run_one_experiment(
                            mode=mode,
                            edge_model=edge_detector.model,
                            split_model=split_model,
                            model_name=str(args.edge_model),
                            golden_model=str(args.golden_model),
                            initial_state=initial_state,
                            splitter=splitter,
                            choice=choice,
                            cached_feature_path=cached_feature_paths.get(candidate_id),
                            cached_feature_failure=cached_feature_failures.get(candidate_id),
                            frame_dir=frame_dir,
                            frames_by_id=frames_by_id,
                            sampled_frame_indices=sampled_ids,
                            annotations=sample_annotations,
                            sample_count=sample_count,
                            epochs=epochs,
                            batch_size=int(args.batch_size),
                            output_root=output_root,
                            graph_build_time=graph_build_time,
                            candidate_enumeration_time=candidate_enumeration_time,
                            teacher_annotation_time=teacher_annotation_time,
                            learning_rate=learning_rate,
                            optimizer_config=optimizer_config,
                            repeat_index=repeat_index,
                            base_seed=int(args.seed),
                            seed=run_seed,
                            device=device,
                        )
                        rows.append(row)
                        _append_jsonl(results_path, row)

    _write_summary_csv(summary_path, rows)
    _write_aggregate_summary_csv(aggregate_summary_path, rows)
    _write_plots(rows, output_root)
    logger.info("Wrote {}", results_path)
    logger.info("Wrote {}", summary_path)
    logger.info("Wrote {}", aggregate_summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
