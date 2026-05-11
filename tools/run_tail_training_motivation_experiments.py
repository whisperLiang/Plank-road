"""Freeze vs rebuilt vs cached Ariadne split tail-training motivation experiment.

All three modes (``freeze``, ``split_rebuild`` and ``split_cached``) implement
fixed-prefix tail training:

* the prefix segment is a frozen feature extractor (parameters frozen,
  ``BatchNorm`` running stats frozen, dropout off, ``eval`` state, forward under
  ``torch.no_grad``);
* the suffix segment is the only trainable part, with trainable parameters
  resolved from the Ariadne runtime (``runtime.candidate.suffix_nodes``);
* ``freeze``, ``split_rebuild`` and ``split_cached`` train **the same** set of
  suffix parameters, with the same optimizer configuration, using the same
  ``Ariadne`` ``run_prefix``/``train_suffix`` code paths;
* ``split_rebuild`` rebuilds the boundary feature cache **exactly once** before
  training and then reuses it for every epoch;
* ``split_cached`` reuses the cache built before the repeat loop, so it must
  never pay a feature-rebuild cost.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import re
import sys
import time
from dataclasses import dataclass, replace
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
from model_management.model_zoo import (
    ensure_local_model_artifact,
    get_model_artifact_path,
    get_model_detection_thresholds,
)
from model_management.object_detection import Object_Detection
from model_management.split_model_adapters import (
    build_split_training_loss,
    get_split_runtime_input_resize_mode,
    get_split_runtime_model,
    prepare_split_runtime_input,
)
from model_management.split_runtime import (
    SplitRuntimeConfig,
    build_split_runtime,
    get_split_runtime_metadata,
    maybe_warmup_runtime,
)
from model_management.universal_model_split import (
    _suffix_parameter_names,
    build_split_retrain_optimizer,
    collect_suffix_trainable_parameters,
)


DEFAULT_MODES = ("freeze", "split_rebuild", "split_cached")
DEFAULT_SAMPLE_COUNT = 512
DEFAULT_EPOCHS = 10
DEFAULT_REPEAT = 5
DEFAULT_SPLIT_BOUNDARIES = ("percent:25", "percent:50", "percent:75")
SPLIT_BUCKET_BY_BOUNDARY = {
    "percent:25": "Early25%",
    "percent:50": "Middle50%",
    "percent:75": "Late75%",
}
BUCKET_LABELS = tuple(SPLIT_BUCKET_BY_BOUNDARY[boundary] for boundary in DEFAULT_SPLIT_BOUNDARIES)
UNSTABLE_SPLIT_ID_FRAGMENTS = (
    ".self_attn",
    ".multihead_attn",
    ".mlp.",
    ".fc1",
    ".fc2",
    ".q_proj",
    ".k_proj",
    ".v_proj",
    ".out_proj",
    ".bn",
    ".norm",
    ".dropout",
    ".activation",
    ".cv1",
    ".cv2",
)


@dataclass(frozen=True)
class SplitChoice:
    bucket: str
    boundary: str
    resolved_boundary: str | None = None


@dataclass(frozen=True)
class CachedSplitBatch:
    """Precomputed boundary payload for a single training mini-batch.

    Storing the full :class:`BoundaryPayload` (not just the tensors) keeps the
    Ariadne replay metadata needed for suffix training: ``split_id``,
    ``graph_signature``, ``passthrough_inputs``, ``schema`` and
    ``primary_label``.
    """

    sample_ids: tuple[int, ...]
    boundary: Any
    boundary_split_id: str
    boundary_graph_signature: str
    targets: tuple[Any, ...]


@dataclass(frozen=True)
class CachedSplitRuntime:
    percent: str
    split_id: str
    graph_signature: str
    runtime: Any
    cached_batches: list[CachedSplitBatch]
    feature_rebuild_time: float
    runtime_build_time: float
    suffix_param_names: tuple[str, ...]

    @property
    def cached_sample_count(self) -> int:
        return sum(len(batch.sample_ids) for batch in self.cached_batches)


@dataclass(frozen=True)
class PreflightReport:
    percent: str
    actual_split_id: str
    graph_signature: str
    trainable_parameter_names: tuple[str, ...]
    trainable_parameter_count: int
    full_loss: float
    split_loss: float
    full_output_max_diff: float
    tolerance: float


@dataclass(frozen=True)
class _PreparedBatch:
    sample_ids: tuple[int, ...]
    boundary: Any
    targets: tuple[Any, ...]



def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare freeze, rebuilt Ariadne split, and cached Ariadne split training.",
    )
    parser.add_argument("--yaml-path", default="./config/config.yaml")
    parser.add_argument("--video-path", default="./video_data/road.mp4")
    parser.add_argument("--edge-model", default="rfdetr_nano")
    parser.add_argument("--golden-model", default="rtdetr_x")
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=DEFAULT_MODES,
        default=list(DEFAULT_MODES),
    )
    parser.add_argument(
        "--split-boundaries",
        nargs="+",
        default=list(DEFAULT_SPLIT_BOUNDARIES),
    )
    parser.add_argument(
        "--ariadne-mode",
        choices=("generated_eager", "compiled"),
        default="generated_eager",
    )
    parser.add_argument("--dynamic-batch-max", type=int, default=64)
    parser.add_argument("--output-root", default="./tmp/tail_training_motivation")
    parser.add_argument("--seed", type=int, default=42)
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
            row.get("split_boundary"),
            int(row.get("sample_count") or 0),
            int(row.get("epochs") or 0),
        )
        groups.setdefault(key, []).append(row)

    metric_fields = (
        "train_time_sec",
        "suffix_train_time_sec",
        "feature_rebuild_time_sec",
        "total_update_time_sec",
        "metric_before",
        "metric_after",
        "metric_delta",
        "final_loss",
        "runtime_build_time_sec",
    )
    aggregated: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        mode, split_bucket, split_boundary, sample_count, epochs = key
        row = {
            "mode": mode,
            "split_bucket": split_bucket,
            "split_boundary": split_boundary,
            "sample_count": sample_count,
            "epochs": epochs,
            "run_count": len(items),
        }
        for field in metric_fields:
            mean, std = _mean_std([item.get(field) for item in items])
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


def _clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cuda_sdp_flags() -> dict[str, Any]:
    if not hasattr(torch.backends, "cuda"):
        return {
            "flash_sdp": None,
            "mem_efficient_sdp": None,
            "math_sdp": None,
            "cudnn_sdp": None,
        }
    cuda_backend = torch.backends.cuda
    cudnn_enabled = None
    if hasattr(cuda_backend, "cudnn_sdp_enabled"):
        cudnn_enabled = cuda_backend.cudnn_sdp_enabled()
    return {
        "flash_sdp": cuda_backend.flash_sdp_enabled(),
        "mem_efficient_sdp": cuda_backend.mem_efficient_sdp_enabled(),
        "math_sdp": cuda_backend.math_sdp_enabled(),
        "cudnn_sdp": cudnn_enabled,
    }


def _log_cuda_sdp_flags(message: str) -> None:
    flags = _cuda_sdp_flags()
    logger.info(
        "{}: flash_sdp={}, mem_efficient_sdp={}, math_sdp={}, cudnn_sdp={}",
        message,
        flags["flash_sdp"],
        flags["mem_efficient_sdp"],
        flags["math_sdp"],
        flags["cudnn_sdp"],
    )


def _force_cuda_math_sdp(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    logger.info(
        "Forced CUDA SDPA backend to math mode: "
        "flash_sdp={}, mem_efficient_sdp={}, math_sdp={}",
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
    )


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
    sample_count: int,
    *,
    seed: int,
) -> list[int]:
    if total_frames <= 0:
        raise RuntimeError("Video contains no readable frames.")
    sample_count = int(sample_count)
    if sample_count <= 0:
        raise ValueError("--sample-count must be positive.")
    if sample_count > total_frames:
        raise RuntimeError(
            f"Requested {sample_count} samples but video only has {total_frames} frame(s)."
        )

    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(np.arange(1, total_frames + 1))
    return sorted(int(value) for value in permutation[:sample_count].tolist())


def _sample_video_frames(
    video_path: Path,
    sample_count: int,
    *,
    seed: int,
) -> tuple[dict[int, np.ndarray], list[int]]:
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
        selected = _select_sample_frame_ids(len(frames), sample_count, seed=seed)
        return {frame_id: frames[frame_id] for frame_id in selected}, selected

    selected = _select_sample_frame_ids(total_frames, sample_count, seed=seed)
    needed_ids = set(selected)
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
        if not path.exists() and not cv2.imwrite(str(path), frame):
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
        "boxes": [[float(coord) for coord in list(box)[:4]] for box in boxes[:count]],
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
            for frame_id, prediction in zip(batch_ids, predictions, strict=True):
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
    for parameter in model.parameters():
        parameter.grad = None


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
    if not shape:
        raise RuntimeError("Runtime input does not contain a batched tensor.")
    return int(shape[0])


def _combine_runtime_inputs(inputs: list[Any]) -> Any:
    if not inputs:
        raise ValueError("Cannot combine an empty runtime input list.")
    first = inputs[0]
    if isinstance(first, torch.Tensor):
        tensors = [item for item in inputs if isinstance(item, torch.Tensor)]
        if len(tensors) != len(inputs):
            raise TypeError("Runtime input batch contains mixed tensor/non-tensor values.")
        if all(tensor.ndim > 0 and int(tensor.shape[0]) == 1 for tensor in tensors):
            return torch.cat(tensors, dim=0)
        return torch.stack(tensors, dim=0)
    if isinstance(first, tuple):
        return tuple(
            _combine_runtime_inputs([item[index] for item in inputs])
            for index in range(len(first))
        )
    if isinstance(first, list):
        return [
            _combine_runtime_inputs([item[index] for item in inputs])
            for index in range(len(first))
        ]
    if isinstance(first, Mapping):
        keys = list(first.keys())
        return {key: _combine_runtime_inputs([item[key] for item in inputs]) for key in keys}
    return list(inputs)


def _target_with_metadata(
    frame_id: int,
    annotation: Mapping[str, Any] | None,
    runtime_input: Any,
    frame: np.ndarray,
    resize_mode: str | None,
) -> dict[str, Any]:
    target = {
        "boxes": list((annotation or {}).get("boxes") or []),
        "labels": list((annotation or {}).get("labels") or []),
        "label_coordinate_space": "original_xyxy",
        "label_image_size": [int(frame.shape[0]), int(frame.shape[1])],
        "label_resize_mode": resize_mode or "direct_resize",
    }
    target["_split_meta"] = {
        "sample_id": int(frame_id),
        "input_tensor_shape": _first_tensor_shape(runtime_input),
        "input_image_size": [int(frame.shape[0]), int(frame.shape[1])],
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
) -> tuple[Any, list[Any]]:
    runtime_inputs: list[Any] = []
    targets: list[Any] = []
    for frame_id in frame_ids:
        frame = frames_by_id[int(frame_id)]
        runtime_input = prepare_split_runtime_input(model, frame, device=device)
        runtime_inputs.append(runtime_input)
        targets.append(
            _target_with_metadata(
                int(frame_id),
                annotations.get(str(int(frame_id))),
                runtime_input,
                frame,
                resize_mode,
            )
        )
    return _combine_runtime_inputs(runtime_inputs), targets


def _make_trace_batch(
    *,
    model: torch.nn.Module,
    frames_by_id: Mapping[int, np.ndarray],
    sample_ids: list[int],
    device: torch.device,
    trace_batch_size: int,
) -> torch.Tensor:
    if int(trace_batch_size) <= 1:
        raise ValueError("Ariadne batch_gt1 tracing requires trace_batch_size > 1.")
    if len(sample_ids) < int(trace_batch_size):
        raise ValueError("--sample-count must be at least the Ariadne trace batch size.")
    runtime_inputs = [
        prepare_split_runtime_input(model, frames_by_id[int(frame_id)], device=device)
        for frame_id in sample_ids[: int(trace_batch_size)]
    ]
    batch = _combine_runtime_inputs(runtime_inputs)
    if not isinstance(batch, torch.Tensor):
        raise TypeError("Ariadne split experiments expect a tensor runtime input.")
    if _runtime_input_batch_size(batch) <= 1:
        raise RuntimeError("Ariadne example batch must contain at least two samples.")
    return batch


def _split_choices(boundaries: list[str]) -> list[SplitChoice]:
    choices: list[SplitChoice] = []
    for boundary in boundaries:
        if boundary not in SPLIT_BUCKET_BY_BOUNDARY:
            raise ValueError(
                "Unsupported split boundary for this experiment: "
                f"{boundary}. Expected percent:25, percent:50, or percent:75."
            )
        choices.append(SplitChoice(bucket=SPLIT_BUCKET_BY_BOUNDARY[boundary], boundary=boundary))
    return choices


def _is_stable_split_id(split_id: str) -> bool:
    return not any(fragment in str(split_id) for fragment in UNSTABLE_SPLIT_ID_FRAGMENTS)


def _module_level_boundary_for_split_id(split_id: str) -> str | None:
    text = str(split_id)
    for pattern in (
        r"^(after:.*?\.encoder\.layer\.\d+)(?:\.|$)",
        r"^(after:.*?\.decoder\.layers\.\d+)(?:\.|$)",
        r"^(after:.*?\.projector\.stages\.\d+)(?:\.|$)",
        r"^(after:.*?\.layer\.\d+)(?:\.|$)",
        r"^(after:.*?\.layers\.\d+)(?:\.|$)",
        r"^(after:.*?\.stages\.\d+)(?:\.|$)",
    ):
        match = re.match(pattern, text)
        if match:
            return match.group(1)
    return text if _is_stable_split_id(text) else None


def _runtime_boundary_for_choice(choice: SplitChoice) -> str:
    return str(choice.resolved_boundary or choice.boundary)


def _ordered_epoch_batches(
    sample_ids: list[int],
    *,
    batch_size: int,
) -> list[list[int]]:
    """Chunk ``sample_ids`` into fixed-order batches of ``batch_size``.

    The tail-training motivation experiment uses the same sample order for all
    three modes, so a deterministic chunking is all we need; there is no
    per-epoch reshuffle that could desynchronise freeze vs split.
    """
    ids = list(sample_ids)
    batches = [ids[start : start + batch_size] for start in range(0, len(ids), batch_size)]
    if any(len(batch) < 2 for batch in batches):
        raise ValueError(
            "Ariadne batch_gt1 experiments require every training batch to contain "
            "at least two samples. Adjust --sample-count or --batch-size to avoid "
            "a singleton final batch."
        )
    return batches


def _resolve_experiment_learning_rate(config: Any, model_name: str) -> float:
    cl_cfg = config.continual_learning
    normalized = str(model_name).lower()
    if "tinynext" in normalized:
        return float(getattr(cl_cfg, "tinynext_fixed_split_learning_rate", 1e-3))
    if "rfdetr" in normalized:
        return float(getattr(cl_cfg, "rfdetr_fixed_split_learning_rate", 1e-4))
    if "yolo" in normalized:
        return float(getattr(cl_cfg, "wrapper_fixed_split_learning_rate", 3e-5))
    return float(getattr(cl_cfg, "split_learning_rate", 1e-3))


def _optimizer_overrides(model_name: str) -> dict[str, Any]:
    normalized = str(model_name).lower()
    if "rfdetr" in normalized:
        return {"optimizer_name": "adamw", "weight_decay": 1e-4, "grad_clip_norm": 1.0}
    if "tinynext" in normalized:
        return {"optimizer_name": "adam", "weight_decay": 0.0, "grad_clip_norm": 5.0}
    return {"optimizer_name": "adam", "weight_decay": 0.0, "grad_clip_norm": None}



# ---------------------------------------------------------------------------
# Ariadne runtime helpers
# ---------------------------------------------------------------------------


def _require_runtime_split_id(runtime: Any) -> str:
    split_id = getattr(runtime, "split_id", None)
    if split_id is None:
        split_id = get_split_runtime_metadata(runtime).get("actual_split_id")
    if not split_id:
        raise RuntimeError("Ariadne runtime did not expose an authoritative split_id.")
    return str(split_id)


def _require_runtime_graph_signature(runtime: Any) -> str:
    graph_signature = getattr(runtime, "graph_signature", None)
    if not graph_signature:
        graph_signature = get_split_runtime_metadata(runtime).get("graph_signature")
    if not graph_signature:
        raise RuntimeError("Ariadne runtime did not expose a graph_signature.")
    return str(graph_signature)


def _require_boundary_split_id(boundary: Any) -> str:
    split_id = getattr(boundary, "split_id", None)
    if not split_id:
        raise RuntimeError("Cached Ariadne boundary payload did not expose split_id.")
    return str(split_id)


def _require_boundary_graph_signature(boundary: Any) -> str:
    graph_signature = getattr(boundary, "graph_signature", None)
    if not graph_signature:
        raise RuntimeError("Cached Ariadne boundary payload did not expose graph_signature.")
    return str(graph_signature)


def _contiguous_tensor_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().contiguous()
    if isinstance(value, Mapping):
        return {key: _contiguous_tensor_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_contiguous_tensor_tree(item) for item in value)
    if isinstance(value, list):
        return [_contiguous_tensor_tree(item) for item in value]
    return value


def _contiguous_boundary_payload(boundary: Any) -> Any:
    tensors = getattr(boundary, "tensors", None)
    if not isinstance(tensors, Mapping):
        return boundary
    contiguous_tensors = {
        key: _contiguous_tensor_tree(value)
        for key, value in tensors.items()
    }
    passthrough_inputs = getattr(boundary, "passthrough_inputs", None)
    contiguous_passthrough = (
        _contiguous_tensor_tree(passthrough_inputs)
        if isinstance(passthrough_inputs, Mapping)
        else passthrough_inputs
    )
    if contiguous_tensors is tensors and contiguous_passthrough is passthrough_inputs:
        return boundary
    return replace(
        boundary,
        tensors=contiguous_tensors,
        passthrough_inputs=contiguous_passthrough,
    )


def _raise_cached_split_id_mismatch(
    *,
    cached_sample_split_id: str,
    cached_runtime_split_id: str,
    percent: str,
    sample_index: int,
) -> None:
    raise RuntimeError(
        "Cached Ariadne boundary split_id mismatch before split_cached training: "
        f"cached sample split_id={cached_sample_split_id!r}; "
        f"cached runtime split_id={cached_runtime_split_id!r}; "
        f"percent={percent!r}; "
        f"sample index={int(sample_index)}. "
        "The cache must be rebuilt with the same SplitPlan used for training."
    )


def _validate_cached_split_runtime(cached_split: CachedSplitRuntime) -> None:
    runtime_split_id = _require_runtime_split_id(cached_split.runtime)
    runtime_graph_signature = _require_runtime_graph_signature(cached_split.runtime)
    if runtime_split_id != cached_split.split_id:
        raise RuntimeError(
            "Cached Ariadne runtime split_id changed before split_cached training: "
            f"cached sample split_id={cached_split.split_id!r}; "
            f"cached runtime split_id={runtime_split_id!r}; "
            f"percent={cached_split.percent!r}; sample index=0. "
            "The cache must be rebuilt with the same SplitPlan used for training."
        )
    if runtime_graph_signature != cached_split.graph_signature:
        raise RuntimeError(
            "Cached Ariadne runtime graph_signature changed before split_cached training: "
            f"cached graph_signature={cached_split.graph_signature!r}; "
            f"cached runtime graph_signature={runtime_graph_signature!r}; "
            f"percent={cached_split.percent!r}; sample index=0."
        )
    for sample_index, cached_batch in enumerate(cached_split.cached_batches):
        boundary_split_id = _require_boundary_split_id(cached_batch.boundary)
        boundary_graph_signature = _require_boundary_graph_signature(cached_batch.boundary)
        if boundary_split_id != cached_batch.boundary_split_id:
            raise RuntimeError(
                "Cached Ariadne boundary split_id metadata mismatch: "
                f"cached sample split_id={boundary_split_id!r}; "
                f"recorded sample split_id={cached_batch.boundary_split_id!r}; "
                f"cached runtime split_id={cached_split.split_id!r}; "
                f"percent={cached_split.percent!r}; sample index={sample_index}. "
                "The cache must be rebuilt with the same SplitPlan used for training."
            )
        if boundary_graph_signature != cached_batch.boundary_graph_signature:
            raise RuntimeError(
                "Cached Ariadne boundary graph_signature metadata mismatch: "
                f"cached sample graph_signature={boundary_graph_signature!r}; "
                f"recorded sample graph_signature={cached_batch.boundary_graph_signature!r}; "
                f"percent={cached_split.percent!r}; sample index={sample_index}."
            )
        if boundary_split_id != cached_split.split_id:
            _raise_cached_split_id_mismatch(
                cached_sample_split_id=boundary_split_id,
                cached_runtime_split_id=cached_split.split_id,
                percent=cached_split.percent,
                sample_index=sample_index,
            )
        if boundary_graph_signature != cached_split.graph_signature:
            raise RuntimeError(
                "Cached Ariadne boundary graph_signature does not match runtime: "
                f"boundary graph_signature={boundary_graph_signature!r}; "
                f"runtime graph_signature={cached_split.graph_signature!r}; "
                f"percent={cached_split.percent!r}; sample index={sample_index}."
            )


# ---------------------------------------------------------------------------
# Fixed-prefix model configuration
# ---------------------------------------------------------------------------


def _configure_fixed_prefix_training(
    split_model: torch.nn.Module,
    runtime: Any,
) -> tuple[tuple[str, ...], list[torch.nn.Parameter]]:
    """Apply the fixed-prefix + trainable-suffix regime to ``split_model``.

    * suffix params -> ``requires_grad=True`` and ``.train()`` module state;
    * every other parameter -> ``requires_grad=False``;
    * the whole model is placed in ``eval`` mode first, and only the modules
      owning suffix parameters are switched to ``train`` mode. This keeps
      prefix ``BatchNorm`` running stats, dropout, and any other
      train-mode-only behaviour frozen.
    """
    suffix_names = tuple(_suffix_parameter_names(runtime))
    suffix_name_set = set(suffix_names)

    split_model.eval()
    for parameter in split_model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None

    suffix_params: list[torch.nn.Parameter] = []
    modules_with_suffix: set[str] = set()
    named_parameters = dict(split_model.named_parameters())
    for name, parameter in named_parameters.items():
        if name in suffix_name_set:
            parameter.requires_grad_(True)
            suffix_params.append(parameter)
            module_name = name.rsplit(".", 1)[0]
            modules_with_suffix.add(module_name)

    missing = [name for name in suffix_name_set if name not in named_parameters]
    if missing:
        raise RuntimeError(
            "Suffix trainable parameters missing from split model: " + ", ".join(missing)
        )

    module_lookup = dict(split_model.named_modules())
    for module_name in modules_with_suffix:
        module = module_lookup.get(module_name)
        if module is None:
            continue
        module.train()

    return suffix_names, suffix_params


def _collect_frozen_batchnorm_stats(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    snapshot: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            running_mean = getattr(module, "running_mean", None)
            running_var = getattr(module, "running_var", None)
            if isinstance(running_mean, torch.Tensor):
                snapshot[f"{module_name}.running_mean"] = running_mean.detach().clone()
            if isinstance(running_var, torch.Tensor):
                snapshot[f"{module_name}.running_var"] = running_var.detach().clone()
    return snapshot


# ---------------------------------------------------------------------------
# Trainable-suffix loop (shared by freeze / split_rebuild / split_cached)
# ---------------------------------------------------------------------------


def _train_suffix_loop(
    *,
    runtime: Any,
    prepared_batches: list[_PreparedBatch],
    epochs: int,
    device: torch.device,
    loss_fn: Callable[[Any, Any], torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Train the Ariadne suffix over a fixed list of precomputed batches.

    All three modes funnel into this single loop so the only difference
    between them is *how* ``prepared_batches[i].boundary`` was produced.
    """
    epoch_times: list[float] = []
    batch_times: list[float] = []
    losses: list[float] = []
    _synchronize(device)
    training_started = time.perf_counter()
    for _epoch in range(int(epochs)):
        epoch_started = time.perf_counter()
        for prepared in prepared_batches:
            _synchronize(device)
            batch_started = time.perf_counter()
            boundary = _contiguous_boundary_payload(prepared.boundary)
            targets = list(copy.deepcopy(prepared.targets))
            loss, _boundary_grads = runtime.train_suffix(
                boundary,
                targets,
                loss_fn=loss_fn,
                optimizer=optimizer,
            )
            _synchronize(device)
            batch_times.append(time.perf_counter() - batch_started)
            losses.append(float(loss.detach().cpu().item()))
        epoch_times.append(time.perf_counter() - epoch_started)
    _synchronize(device)
    return {
        "suffix_train_time_sec": float(time.perf_counter() - training_started),
        "epoch_time_mean_sec": float(np.mean(epoch_times)) if epoch_times else None,
        "batch_time_mean_sec": float(np.mean(batch_times)) if batch_times else None,
        "final_loss": float(losses[-1]) if losses else None,
    }


def _build_cached_batches(
    *,
    runtime: Any,
    percent: str,
    split_id: str,
    graph_signature: str,
    edge_model: torch.nn.Module,
    frames_by_id: Mapping[int, np.ndarray],
    sample_ids: list[int],
    annotations: Mapping[str, Mapping[str, Any]],
    batch_size: int,
    device: torch.device,
) -> tuple[list[CachedSplitBatch], float]:
    resize_mode = get_split_runtime_input_resize_mode(edge_model)
    batches: list[CachedSplitBatch] = []
    _synchronize(device)
    started = time.perf_counter()
    for batch_ids in _ordered_epoch_batches(sample_ids, batch_size=max(2, int(batch_size))):
        inputs, targets = _prepare_raw_batch(
            model=edge_model,
            frame_ids=batch_ids,
            frames_by_id=frames_by_id,
            annotations=annotations,
            device=device,
            resize_mode=resize_mode,
        )
        with torch.no_grad():
            boundary = runtime.run_prefix(inputs)
        boundary_split_id = _require_boundary_split_id(boundary)
        boundary_graph_signature = _require_boundary_graph_signature(boundary)
        if boundary_split_id != split_id:
            _raise_cached_split_id_mismatch(
                cached_sample_split_id=boundary_split_id,
                cached_runtime_split_id=split_id,
                percent=percent,
                sample_index=len(batches),
            )
        if boundary_graph_signature != graph_signature:
            raise RuntimeError(
                "Ariadne boundary graph_signature differs from runtime: "
                f"boundary={boundary_graph_signature!r}; runtime={graph_signature!r}; "
                f"percent={percent!r}; sample index={len(batches)}."
            )
        boundary = _contiguous_boundary_payload(boundary)
        batches.append(
            CachedSplitBatch(
                sample_ids=tuple(int(item) for item in batch_ids),
                boundary=boundary,
                boundary_split_id=boundary_split_id,
                boundary_graph_signature=boundary_graph_signature,
                targets=tuple(copy.deepcopy(target) for target in targets),
            )
        )
    _synchronize(device)
    return batches, float(time.perf_counter() - started)


def _prepared_batches_from_cache(
    cached_batches: list[CachedSplitBatch],
) -> list[_PreparedBatch]:
    return [
        _PreparedBatch(
            sample_ids=batch.sample_ids,
            boundary=batch.boundary,
            targets=batch.targets,
        )
        for batch in cached_batches
    ]



# ---------------------------------------------------------------------------
# Preflight equivalence check
# ---------------------------------------------------------------------------


def _walk_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_tensors(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_tensors(item)


def _max_abs_output_diff(expected: Any, actual: Any) -> float:
    expected_tensors = list(_walk_tensors(expected))
    actual_tensors = list(_walk_tensors(actual))
    if not expected_tensors or not actual_tensors:
        return 0.0
    pairs = list(zip(expected_tensors, actual_tensors))
    if not pairs:
        return 0.0
    max_diff = 0.0
    for expected_tensor, actual_tensor in pairs:
        if expected_tensor.shape != actual_tensor.shape:
            return float("inf")
        diff = (expected_tensor.detach().float() - actual_tensor.detach().float()).abs().max()
        max_diff = max(max_diff, float(diff.item()))
    return float(max_diff)


def _preflight_equivalence_check(
    *,
    runtime: Any,
    split_model: torch.nn.Module,
    edge_model: torch.nn.Module,
    choice: SplitChoice,
    frames_by_id: Mapping[int, np.ndarray],
    sample_ids: list[int],
    annotations: Mapping[str, Mapping[str, Any]],
    batch_size: int,
    device: torch.device,
    loss_fn: Callable[[Any, Any], torch.Tensor],
    freeze_trainable_names: tuple[str, ...],
    split_trainable_names: tuple[str, ...],
    cached_split: CachedSplitRuntime | None,
    runtime_split_id: str,
    runtime_graph_signature: str,
    loss_tolerance: float = 5e-2,
    output_tolerance: float = 1e-1,
) -> PreflightReport:
    """Fail loudly if freeze / split paths are not equivalent on one batch."""
    if freeze_trainable_names != split_trainable_names:
        raise RuntimeError(
            "Freeze vs split trainable parameter names differ. "
            f"freeze={freeze_trainable_names}; split={split_trainable_names}."
        )

    if cached_split is not None:
        if cached_split.split_id != runtime_split_id:
            raise RuntimeError(
                "Preflight cached split_id does not match live runtime: "
                f"cached={cached_split.split_id!r}; runtime={runtime_split_id!r}; "
                f"percent={choice.boundary!r}."
            )
        if tuple(cached_split.suffix_param_names) != tuple(split_trainable_names):
            raise RuntimeError(
                "Preflight cached suffix parameter names differ from live runtime. "
                f"cached={cached_split.suffix_param_names}; "
                f"runtime={split_trainable_names}."
            )

    check_batch_size = max(2, min(int(batch_size), len(sample_ids)))
    probe_ids = list(sample_ids[:check_batch_size])
    resize_mode = get_split_runtime_input_resize_mode(edge_model)
    inputs, targets = _prepare_raw_batch(
        model=edge_model,
        frame_ids=probe_ids,
        frames_by_id=frames_by_id,
        annotations=annotations,
        device=device,
        resize_mode=resize_mode,
    )

    previous_training = split_model.training
    split_model.eval()
    try:
        with torch.no_grad():
            full_output = split_model(inputs)
            boundary = runtime.run_prefix(inputs)
            suffix_output = runtime.run_suffix(_contiguous_boundary_payload(boundary))
            full_loss = loss_fn(full_output, copy.deepcopy(targets))
            split_loss = loss_fn(suffix_output, copy.deepcopy(targets))
    finally:
        split_model.train(previous_training)

    boundary_split_id = _require_boundary_split_id(boundary)
    boundary_graph_signature = _require_boundary_graph_signature(boundary)
    if boundary_split_id != runtime_split_id:
        raise RuntimeError(
            "Preflight boundary split_id mismatch: "
            f"boundary={boundary_split_id!r}; runtime={runtime_split_id!r}."
        )
    if boundary_graph_signature != runtime_graph_signature:
        raise RuntimeError(
            "Preflight boundary graph_signature mismatch: "
            f"boundary={boundary_graph_signature!r}; runtime={runtime_graph_signature!r}."
        )

    full_loss_value = float(full_loss.detach().cpu().item())
    split_loss_value = float(split_loss.detach().cpu().item())
    loss_diff = abs(full_loss_value - split_loss_value)
    if loss_diff > float(loss_tolerance):
        raise RuntimeError(
            "Preflight full-path vs split-path loss mismatch: "
            f"full={full_loss_value}; split={split_loss_value}; "
            f"diff={loss_diff}; tolerance={loss_tolerance}; "
            f"percent={choice.boundary!r}; split_id={runtime_split_id!r}."
        )

    output_max_diff = _max_abs_output_diff(full_output, suffix_output)
    if not np.isfinite(output_max_diff) or output_max_diff > float(output_tolerance):
        raise RuntimeError(
            "Preflight full-path vs split-path output mismatch: "
            f"max_abs_diff={output_max_diff}; tolerance={output_tolerance}; "
            f"percent={choice.boundary!r}; split_id={runtime_split_id!r}."
        )

    return PreflightReport(
        percent=choice.boundary,
        actual_split_id=runtime_split_id,
        graph_signature=runtime_graph_signature,
        trainable_parameter_names=tuple(split_trainable_names),
        trainable_parameter_count=sum(
            int(param.numel())
            for param in collect_suffix_trainable_parameters(runtime)
        ),
        full_loss=full_loss_value,
        split_loss=split_loss_value,
        full_output_max_diff=output_max_diff,
        tolerance=float(loss_tolerance),
    )


# ---------------------------------------------------------------------------
# Runtime construction / boundary resolution
# ---------------------------------------------------------------------------


def _build_runtime_for_boundary(
    *,
    split_model: torch.nn.Module,
    example_batch: torch.Tensor,
    boundary: str,
    args: argparse.Namespace,
    warmup: bool,
) -> tuple[Any, float]:
    config = SplitRuntimeConfig(
        boundary=str(boundary),
        dynamic_batch=(2, max(2, int(args.dynamic_batch_max), int(args.batch_size))),
        trace_batch_size=2,
        mode=str(args.ariadne_mode),
        trainable=True,
    )
    _log_cuda_sdp_flags("CUDA SDPA backend flags before Ariadne runtime construction")
    started = time.perf_counter()
    runtime = build_split_runtime(split_model, example_batch, config)
    if warmup:
        maybe_warmup_runtime(runtime, example_batch)
    return runtime, float(time.perf_counter() - started)


def _build_runtime_for_choice(
    *,
    split_model: torch.nn.Module,
    example_batch: torch.Tensor,
    choice: SplitChoice,
    args: argparse.Namespace,
) -> tuple[Any, float]:
    runtime_boundary = _runtime_boundary_for_choice(choice)
    runtime, elapsed = _build_runtime_for_boundary(
        split_model=split_model,
        example_batch=example_batch,
        boundary=runtime_boundary,
        args=args,
        warmup=True,
    )
    split_id = _require_runtime_split_id(runtime)
    logger.info(
        "Selected percent boundary {} -> runtime boundary {} -> exact Ariadne "
        "split_id {} stable_boundary_passed={}",
        choice.boundary,
        runtime_boundary,
        split_id,
        _is_stable_split_id(split_id),
    )
    return runtime, elapsed


def _resolve_stable_split_choices(
    *,
    split_model: torch.nn.Module,
    example_batch: torch.Tensor,
    choices: list[SplitChoice],
    args: argparse.Namespace,
) -> list[SplitChoice]:
    resolved: list[SplitChoice] = []
    for choice in choices:
        probe_runtime, _elapsed = _build_runtime_for_boundary(
            split_model=split_model,
            example_batch=example_batch,
            boundary=choice.boundary,
            args=args,
            warmup=False,
        )
        probe_split_id = _require_runtime_split_id(probe_runtime)
        del probe_runtime
        _clear_cuda_cache()
        stable_boundary = probe_split_id
        stable_passed = _is_stable_split_id(probe_split_id)
        if not stable_passed:
            stable_boundary = _module_level_boundary_for_split_id(probe_split_id)
            if stable_boundary is None:
                raise RuntimeError(
                    "Ariadne percent boundary resolved to an unstable training split "
                    f"with no module-level fallback: percent={choice.boundary!r}, "
                    f"split_id={probe_split_id!r}."
                )
            logger.info(
                "Percent boundary {} initially resolved to unstable split_id {}; "
                "trying stable module-level boundary {}.",
                choice.boundary,
                probe_split_id,
                stable_boundary,
            )
            stable_runtime, _stable_elapsed = _build_runtime_for_boundary(
                split_model=split_model,
                example_batch=example_batch,
                boundary=stable_boundary,
                args=args,
                warmup=False,
            )
            stable_split_id = _require_runtime_split_id(stable_runtime)
            del stable_runtime
            _clear_cuda_cache()
            stable_passed = _is_stable_split_id(stable_split_id)
            if not stable_passed:
                raise RuntimeError(
                    "Ariadne stable boundary fallback still resolved to an unstable "
                    "training split: "
                    f"percent={choice.boundary!r}, requested={stable_boundary!r}, "
                    f"split_id={stable_split_id!r}."
                )
            stable_boundary = stable_split_id
        logger.info(
            "Selected percent boundary {} -> exact Ariadne split_id {} "
            "stable_boundary_passed={}",
            choice.boundary,
            stable_boundary,
            stable_passed,
        )
        resolved.append(
            SplitChoice(
                bucket=choice.bucket,
                boundary=choice.boundary,
                resolved_boundary=stable_boundary,
            )
        )
    return resolved


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _evaluate_proxy_map(
    *,
    model: torch.nn.Module,
    model_name: str,
    frame_dir: Path,
    annotations: Mapping[str, Mapping[str, Any]],
    device: torch.device,
    batch_size: int,
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
        )
    )


def _metric_value(metrics: Mapping[str, Any]) -> float | None:
    value = metrics.get("map")
    return None if value is None else float(value)


# ---------------------------------------------------------------------------
# Result row builders
# ---------------------------------------------------------------------------


def _base_result_row(
    *,
    mode: str,
    choice: SplitChoice,
    metadata: Mapping[str, Any],
    edge_model: str,
    golden_model: str,
    sample_count: int,
    epochs: int,
    batch_size: int,
    repeat_id: int,
    seed: int,
    ariadne_mode: str,
    teacher_annotation_time: float,
    sampled_frame_indices: list[int],
) -> dict[str, Any]:
    return {
        "mode": mode,
        "split_bucket": choice.bucket,
        "split_boundary": choice.boundary,
        "resolved_split_boundary": _runtime_boundary_for_choice(choice),
        "actual_split_id": metadata.get("actual_split_id"),
        "graph_signature": metadata.get("graph_signature"),
        "repeat_id": int(repeat_id),
        "sample_count": int(sample_count),
        "epochs": int(epochs),
        "train_time_sec": 0.0,
        "suffix_train_time_sec": 0.0,
        "feature_rebuild_time_sec": 0.0,
        "total_update_time_sec": 0.0,
        "metric_before": None,
        "metric_after": None,
        "metric_delta": None,
        "batch_size": int(batch_size),
        "ariadne_mode": ariadne_mode,
        "edge_model": edge_model,
        "golden_model": golden_model,
        "seed": int(seed),
        "teacher_annotation_time_sec": float(teacher_annotation_time),
        "runtime_build_time_sec": 0.0,
        "final_loss": None,
        "epoch_time_mean_sec": None,
        "batch_time_mean_sec": None,
        "sampled_frame_indices": [int(item) for item in sampled_frame_indices],
        **dict(metadata),
    }


def _update_metrics(
    row: dict[str, Any],
    before_metrics: Mapping[str, Any],
    after_metrics: Mapping[str, Any],
) -> None:
    before = _metric_value(before_metrics)
    after = _metric_value(after_metrics)
    row["metric_before"] = before
    row["metric_after"] = after
    row["metric_delta"] = None if before is None or after is None else after - before


def _prepare_configs(args: argparse.Namespace) -> tuple[Any, Any]:
    base_config = load_runtime_config(Path(args.yaml_path))
    client_cfg = copy.deepcopy(base_config.client)
    server_cfg = copy.deepcopy(base_config.server)
    client_cfg.lightweight = str(args.edge_model)
    client_cfg.weights_path = _resolve_local_weights_path(str(args.edge_model))
    server_cfg.edge_model_name = str(args.edge_model)
    server_cfg.golden = str(args.golden_model)
    server_cfg.weights_path = _resolve_local_weights_path(str(args.golden_model))
    server_cfg.continual_learning.num_epoch = int(args.epochs)
    server_cfg.continual_learning.batch_size = int(args.batch_size)
    server_cfg.das.enabled = False
    return client_cfg, server_cfg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_split_time_accuracy_subplots(
    rows: list[Mapping[str, Any]],
    output_root: Path,
) -> None:
    """Two-subplot figure. Top: training-time boxplots. Bottom: mAP boxplots.

    Both panels share the split-position x-axis. Each panel groups the three
    modes side by side. The displayed training time follows the experiment's
    definition:

    * ``freeze``        -> ``suffix_train_time_sec`` (raw-input fixed-prefix
      forward + suffix train, measured inside the training loop);
    * ``split_rebuild`` -> ``feature_rebuild_time_sec + suffix_train_time_sec``;
    * ``split_cached``  -> ``suffix_train_time_sec`` (cached features, no
      per-run rebuild cost).

    Teacher annotation, sample collection, and the one-off runtime build are
    not mixed into the training-time boxplots.
    """
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib is unavailable; skipping split-position subplot figure: {}", exc)
        return

    modes = [mode for mode in DEFAULT_MODES if any(row.get("mode") == mode for row in rows)]
    if not modes:
        logger.warning("No recognised modes found in rows; skipping subplot figure.")
        return

    n_modes = len(modes)
    total_spread = 0.48
    box_width = min(0.12, total_spread / max(n_modes, 1) * 0.85)
    offsets = np.linspace(-total_spread / 2, total_spread / 2, n_modes) if n_modes > 1 else [0.0]
    mode_offsets = {mode: float(offsets[i]) for i, mode in enumerate(modes)}

    mode_faces = {
        "freeze": "#6aa6d8",
        "split_rebuild": "#f2c94c",
        "split_cached": "#65b96a",
    }
    mode_edges = {
        "freeze": "#24567a",
        "split_rebuild": "#8f6b00",
        "split_cached": "#266b32",
    }
    _fallback_faces = ["#d08080", "#80d0d0", "#d0a0d0"]
    _fallback_edges = ["#803030", "#307070", "#703070"]
    for i, mode in enumerate(modes):
        if mode not in mode_faces:
            mode_faces[mode] = _fallback_faces[i % len(_fallback_faces)]
            mode_edges[mode] = _fallback_edges[i % len(_fallback_edges)]

    bucket_positions = {bucket: index + 1 for index, bucket in enumerate(BUCKET_LABELS)}

    def _training_time(row: Mapping[str, Any]) -> float | None:
        mode = row.get("mode")
        suffix = row.get("suffix_train_time_sec")
        rebuild = row.get("feature_rebuild_time_sec") or 0.0
        try:
            suffix_value = None if suffix is None else float(suffix)
        except (TypeError, ValueError):
            return None
        try:
            rebuild_value = 0.0 if rebuild is None else float(rebuild)
        except (TypeError, ValueError):
            rebuild_value = 0.0
        if suffix_value is None:
            return None
        if mode == "split_rebuild":
            return suffix_value + rebuild_value
        return suffix_value

    def _collect_training_time(bucket: str, mode: str) -> list[float]:
        result: list[float] = []
        for row in rows:
            if row.get("split_bucket") != bucket or row.get("mode") != mode:
                continue
            value = _training_time(row)
            if value is None or not np.isfinite(value):
                continue
            result.append(float(value))
        return result

    def _collect_metric(bucket: str, mode: str) -> list[float]:
        result: list[float] = []
        for row in rows:
            if row.get("split_bucket") != bucket or row.get("mode") != mode:
                continue
            value = row.get("metric_after")
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                result.append(number)
        return result

    def _draw_boxes(
        ax: "plt.Axes",
        data: list[float],
        position: float,
        mode: str,
    ) -> bool:
        if not data:
            return False
        bp = ax.boxplot(
            data,
            positions=[position],
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
            showfliers=True,
            showmeans=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(mode_faces[mode])
            patch.set_edgecolor(mode_edges[mode])
            patch.set_linewidth(1.1)
            patch.set_alpha(0.82)
        for key in ("whiskers", "caps"):
            for line in bp[key]:
                line.set_color(mode_edges[mode])
                line.set_linewidth(0.9)
        for line in bp["medians"]:
            line.set_color(mode_edges[mode])
            line.set_linewidth(1.6)
        for flier in bp["fliers"]:
            flier.set_markerfacecolor(mode_faces[mode])
            flier.set_markeredgecolor(mode_edges[mode])
            flier.set_markersize(3.0)
            flier.set_alpha(0.7)
        return True

    fig, (ax_time, ax_acc) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7.5, 6.0),
        gridspec_kw={"height_ratios": [1, 1]},
    )

    plotted_time = False
    plotted_acc = False

    for bucket in BUCKET_LABELS:
        base = float(bucket_positions[bucket])
        for mode in modes:
            offset = mode_offsets[mode]
            pos = base + offset

            time_vals = _collect_training_time(bucket, mode)
            if _draw_boxes(ax_time, time_vals, pos, mode):
                plotted_time = True

            acc_vals = [v * 100.0 for v in _collect_metric(bucket, mode)]
            if _draw_boxes(ax_acc, acc_vals, pos, mode):
                plotted_acc = True

    if not plotted_time and not plotted_acc:
        logger.warning("No finite values to plot; skipping subplot figure.")
        plt.close(fig)
        return

    x_ticks = [bucket_positions[b] for b in BUCKET_LABELS]
    x_lim = (0.45, len(BUCKET_LABELS) + 0.55)

    for ax in (ax_time, ax_acc):
        ax.set_xticks(x_ticks)
        ax.set_xlim(x_lim)
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_time.set_xticklabels([])
    ax_time.set_ylabel("Training time (s)", fontsize=9)
    ax_time.set_ylim(bottom=0.0)

    ax_acc.set_xticklabels(BUCKET_LABELS, fontsize=9)
    ax_acc.set_xlabel("Split position", fontsize=9)
    ax_acc.set_ylabel("mAP (%)", fontsize=9)

    legend_handles = [
        Patch(facecolor=mode_faces[mode], edgecolor=mode_edges[mode], label=mode)
        for mode in modes
    ]
    ax_time.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=len(modes),
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        edgecolor="0.75",
    )

    fig.tight_layout()
    stem = "freeze_vs_split_cached_vs_rebuild_by_position"
    fig.savefig(plots_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(plots_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved subplot figure to {}", plots_dir / f"{stem}.pdf")



# ---------------------------------------------------------------------------
# Per-mode runners
# ---------------------------------------------------------------------------


def _run_freeze_mode(
    *,
    runtime: Any,
    edge_model: torch.nn.Module,
    frames_by_id: Mapping[int, np.ndarray],
    sample_ids: list[int],
    annotations: Mapping[str, Mapping[str, Any]],
    batch_size: int,
    epochs: int,
    device: torch.device,
    loss_fn: Callable[[Any, Any], torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """Fixed-prefix freeze training via the Ariadne runtime.

    Each batch:

    1. prepare raw inputs for the ordered sample ids;
    2. run the Ariadne prefix under ``torch.no_grad`` (prefix is frozen and in
       ``eval`` state) to produce a BoundaryPayload;
    3. call ``runtime.train_suffix`` to update suffix parameters.

    Because freeze and split_rebuild use the exact same code path here, the
    only valid source of behaviour difference comes from *when* features were
    computed (every batch here; once before training for the split modes).
    """
    resize_mode = get_split_runtime_input_resize_mode(edge_model)
    epoch_times: list[float] = []
    batch_times: list[float] = []
    losses: list[float] = []
    _synchronize(device)
    training_started = time.perf_counter()
    for _epoch in range(int(epochs)):
        epoch_started = time.perf_counter()
        for batch_ids in _ordered_epoch_batches(
            sample_ids,
            batch_size=max(2, int(batch_size)),
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
            with torch.no_grad():
                boundary = runtime.run_prefix(inputs)
            boundary = _contiguous_boundary_payload(boundary)
            loss, _grads = runtime.train_suffix(
                boundary,
                targets,
                loss_fn=loss_fn,
                optimizer=optimizer,
            )
            _synchronize(device)
            batch_times.append(time.perf_counter() - batch_started)
            losses.append(float(loss.detach().cpu().item()))
        epoch_times.append(time.perf_counter() - epoch_started)
    _synchronize(device)
    return {
        "suffix_train_time_sec": float(time.perf_counter() - training_started),
        "feature_rebuild_time_sec": 0.0,
        "epoch_time_mean_sec": float(np.mean(epoch_times)) if epoch_times else None,
        "batch_time_mean_sec": float(np.mean(batch_times)) if batch_times else None,
        "final_loss": float(losses[-1]) if losses else None,
    }


def _run_split_rebuild_mode(
    *,
    runtime: Any,
    edge_model: torch.nn.Module,
    frames_by_id: Mapping[int, np.ndarray],
    sample_ids: list[int],
    annotations: Mapping[str, Mapping[str, Any]],
    batch_size: int,
    epochs: int,
    device: torch.device,
    loss_fn: Callable[[Any, Any], torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """split_rebuild: rebuild features once, reuse for every epoch."""
    split_id = _require_runtime_split_id(runtime)
    graph_signature = _require_runtime_graph_signature(runtime)
    cached_batches, feature_rebuild_time = _build_cached_batches(
        runtime=runtime,
        percent="split_rebuild",
        split_id=split_id,
        graph_signature=graph_signature,
        edge_model=edge_model,
        frames_by_id=frames_by_id,
        sample_ids=sample_ids,
        annotations=annotations,
        batch_size=int(batch_size),
        device=device,
    )
    prepared = _prepared_batches_from_cache(cached_batches)
    train_metrics = _train_suffix_loop(
        runtime=runtime,
        prepared_batches=prepared,
        epochs=epochs,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )
    train_metrics["feature_rebuild_time_sec"] = float(feature_rebuild_time)
    return train_metrics


def _run_split_cached_mode(
    *,
    cached_split: CachedSplitRuntime,
    epochs: int,
    device: torch.device,
    loss_fn: Callable[[Any, Any], torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    """split_cached: reuse the pre-built boundary payload cache."""
    _validate_cached_split_runtime(cached_split)
    prepared = _prepared_batches_from_cache(cached_split.cached_batches)
    train_metrics = _train_suffix_loop(
        runtime=cached_split.runtime,
        prepared_batches=prepared,
        epochs=epochs,
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )
    train_metrics["feature_rebuild_time_sec"] = 0.0
    return train_metrics


# Public alias for tests that import the legacy name.
def _train_split_cached_loop(
    *,
    cached_split: CachedSplitRuntime,
    epochs: int,
    loss_fn: Callable[[Any, Any], torch.Tensor],
    optimizer: torch.optim.Optimizer | None,
    seed: int,
    shuffle_samples: bool,
    device: torch.device,
) -> dict[str, Any]:
    del seed, shuffle_samples  # ordered deterministic loop; API compat only
    metrics = _run_split_cached_mode(
        cached_split=cached_split,
        epochs=int(epochs),
        device=device,
        loss_fn=loss_fn,
        optimizer=optimizer,
    )
    metrics["train_time_sec"] = float(metrics.get("suffix_train_time_sec", 0.0))
    return metrics



# ---------------------------------------------------------------------------
# Per-repeat experiment driver
# ---------------------------------------------------------------------------


def _assert_trainable_parameter_equivalence(rows: list[Mapping[str, Any]]) -> None:
    by_boundary: dict[str, dict[str, tuple[tuple[str, ...], int]]] = {}
    for row in rows:
        boundary = str(row.get("split_boundary"))
        mode = str(row.get("mode"))
        names = tuple(row.get("trainable_parameter_names") or ())
        count = int(row.get("trainable_parameter_count") or 0)
        by_boundary.setdefault(boundary, {})[mode] = (names, count)
    for boundary, modes_map in by_boundary.items():
        entries = list(modes_map.items())
        if len(entries) < 2:
            continue
        reference_mode, (reference_names, reference_count) = entries[0]
        for mode, (names, count) in entries[1:]:
            if names != reference_names:
                raise RuntimeError(
                    "Trainable parameter names differ across modes at boundary "
                    f"{boundary!r}: {reference_mode}={reference_names}; "
                    f"{mode}={names}."
                )
            if count != reference_count:
                raise RuntimeError(
                    "Trainable parameter counts differ across modes at boundary "
                    f"{boundary!r}: {reference_mode}={reference_count}; "
                    f"{mode}={count}."
                )


def _run_one_experiment(
    *,
    mode: str,
    choice: SplitChoice,
    edge_model: torch.nn.Module,
    split_model: torch.nn.Module,
    model_name: str,
    golden_model: str,
    initial_state: Mapping[str, Any],
    example_batch: torch.Tensor,
    cached_split: CachedSplitRuntime | None,
    shared_runtime: Any,
    shared_runtime_build_time: float,
    frame_dir: Path,
    frames_by_id: Mapping[int, np.ndarray],
    sampled_frame_indices: list[int],
    annotations: Mapping[str, Mapping[str, Any]],
    sample_count: int,
    epochs: int,
    batch_size: int,
    teacher_annotation_time: float,
    learning_rate: float,
    optimizer_config: Mapping[str, Any],
    repeat_id: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    del example_batch  # retained for API symmetry; runtime reused across modes
    _set_random_seed(seed)
    _restore_model_state(edge_model, initial_state)
    edge_model.to(device)
    split_model.to(device)
    loss_fn = build_split_training_loss(edge_model)
    if loss_fn is None:
        raise RuntimeError(f"No split-training loss is available for {model_name}.")

    update_started = time.perf_counter()
    runtime = shared_runtime
    runtime_build_time = float(shared_runtime_build_time)
    if mode == "split_cached":
        if cached_split is None:
            raise RuntimeError("Missing cached Ariadne split runtime.")
        _validate_cached_split_runtime(cached_split)
        if cached_split.runtime is not runtime:
            raise RuntimeError(
                "split_cached runtime does not match the shared per-boundary runtime."
            )
        logger.info(
            "Using cached Ariadne runtime for {} split_id={} samples={}",
            cached_split.percent,
            cached_split.split_id,
            cached_split.cached_sample_count,
        )
    elif mode not in {"freeze", "split_rebuild"}:
        raise RuntimeError(f"Unsupported mode: {mode}")

    metadata = get_split_runtime_metadata(runtime)
    row = _base_result_row(
        mode=mode,
        choice=choice,
        metadata=metadata,
        edge_model=model_name,
        golden_model=golden_model,
        sample_count=sample_count,
        epochs=epochs,
        batch_size=batch_size,
        repeat_id=repeat_id,
        seed=seed,
        ariadne_mode=str(args.ariadne_mode),
        teacher_annotation_time=teacher_annotation_time,
        sampled_frame_indices=sampled_frame_indices,
    )
    row["runtime_build_time_sec"] = runtime_build_time

    # Freeze the prefix, mark only suffix parameters trainable. This MUST
    # happen before the optimizer is constructed so the optimizer only sees
    # suffix parameters for all three modes.
    suffix_param_names, _suffix_params = _configure_fixed_prefix_training(
        split_model, runtime
    )
    row["trainable_parameter_names"] = list(suffix_param_names)
    row["trainable_parameter_count"] = sum(
        int(param.numel()) for param in collect_suffix_trainable_parameters(runtime)
    )

    # Snapshot prefix BatchNorm stats so we can assert they stayed frozen.
    prefix_bn_snapshot = _collect_frozen_batchnorm_stats(split_model)

    optimizer = build_split_retrain_optimizer(
        split_model,
        runtime=runtime,
        learning_rate=float(learning_rate),
        optimizer_name=str(optimizer_config.get("optimizer_name", "adam")),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
        grad_clip_norm=optimizer_config.get("grad_clip_norm"),
    )
    if optimizer is None:
        raise RuntimeError("No trainable suffix parameters were available for this run.")
    optimizer_param_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    suffix_name_set = set(suffix_param_names)
    suffix_param_ids = {
        id(param)
        for name, param in split_model.named_parameters()
        if name in suffix_name_set
    }
    if optimizer_param_ids != suffix_param_ids:
        raise RuntimeError(
            "Optimizer parameter set does not match suffix trainable parameters."
        )

    before_metrics = _evaluate_proxy_map(
        model=edge_model,
        model_name=model_name,
        frame_dir=frame_dir,
        annotations=annotations,
        device=device,
        batch_size=batch_size,
    )

    if mode == "freeze":
        train_metrics = _run_freeze_mode(
            runtime=runtime,
            edge_model=edge_model,
            frames_by_id=frames_by_id,
            sample_ids=sampled_frame_indices,
            annotations=annotations,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
    elif mode == "split_rebuild":
        train_metrics = _run_split_rebuild_mode(
            runtime=runtime,
            edge_model=edge_model,
            frames_by_id=frames_by_id,
            sample_ids=sampled_frame_indices,
            annotations=annotations,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
    else:
        assert cached_split is not None
        train_metrics = _run_split_cached_mode(
            cached_split=cached_split,
            epochs=epochs,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )

    after_metrics = _evaluate_proxy_map(
        model=edge_model,
        model_name=model_name,
        frame_dir=frame_dir,
        annotations=annotations,
        device=device,
        batch_size=batch_size,
    )

    # Verify that frozen prefix BatchNorm running stats were not updated.
    post_bn_snapshot = _collect_frozen_batchnorm_stats(split_model)
    for key, before_tensor in prefix_bn_snapshot.items():
        after_tensor = post_bn_snapshot.get(key)
        if after_tensor is None:
            continue
        module_name = key.rsplit(".", 1)[0]
        if any(
            param_name.startswith(module_name + ".")
            for param_name in suffix_param_names
        ):
            continue
        if not torch.equal(before_tensor.to(after_tensor.device), after_tensor):
            raise RuntimeError(
                "Frozen prefix BatchNorm running statistic drifted during training: "
                f"key={key}; mode={mode}; split_id={metadata.get('actual_split_id')!r}."
            )

    suffix_train = float(train_metrics.get("suffix_train_time_sec", 0.0))
    feature_rebuild = float(train_metrics.get("feature_rebuild_time_sec", 0.0))
    row.update(train_metrics)
    row["suffix_train_time_sec"] = suffix_train
    row["feature_rebuild_time_sec"] = feature_rebuild
    # "train_time_sec" matches the value used for the plot's training-time
    # boxplot (freeze: suffix_train; split_rebuild: rebuild + suffix_train;
    # split_cached: suffix_train).
    if mode == "split_rebuild":
        row["train_time_sec"] = suffix_train + feature_rebuild
    else:
        row["train_time_sec"] = suffix_train
    row["total_update_time_sec"] = float(time.perf_counter() - update_started)
    _update_metrics(row, before_metrics, after_metrics)
    return row


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


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
    if int(args.batch_size) < 2:
        raise ValueError("--batch-size must be at least 2 for Ariadne batch_gt1 tracing.")
    _force_cuda_math_sdp(device)
    object_detection_module.device = device
    _set_random_seed(int(args.seed))
    sample_count = int(args.sample_count)
    epochs = int(args.epochs)
    repeat = max(1, int(args.repeat))
    choices = _split_choices([str(boundary) for boundary in args.split_boundaries])
    modes = tuple(str(mode) for mode in args.modes)

    client_cfg, server_cfg = _prepare_configs(args)

    logger.info("Sampling {} frames from {}", sample_count, args.video_path)
    frames_by_id, sampled_ids = _sample_video_frames(
        Path(args.video_path),
        sample_count,
        seed=int(args.seed),
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
        frame_ids=sampled_ids,
        golden_model=str(args.golden_model),
        video_path=Path(args.video_path),
        threshold=teacher_threshold,
        batch_size=max(
            1,
            int(getattr(server_cfg.continual_learning, "teacher_batch_size", args.batch_size)),
        ),
        device=device,
    )
    sample_annotations = {
        str(frame_id): dict(annotations.get(str(frame_id), {"boxes": [], "labels": []}))
        for frame_id in sampled_ids
    }

    split_model = get_split_runtime_model(edge_detector.model)
    example_batch = _make_trace_batch(
        model=edge_detector.model,
        frames_by_id=frames_by_id,
        sample_ids=sampled_ids,
        device=device,
        trace_batch_size=2,
    )
    choices = _resolve_stable_split_choices(
        split_model=split_model,
        example_batch=example_batch,
        choices=choices,
        args=args,
    )
    initial_state = _snapshot_model_state(edge_detector.model)
    learning_rate = _resolve_experiment_learning_rate(server_cfg, str(args.edge_model))
    optimizer_config = _optimizer_overrides(str(args.edge_model))

    cached_by_boundary: dict[str, CachedSplitRuntime] = {}
    runtime_by_boundary: dict[str, Any] = {}
    runtime_build_time_by_boundary: dict[str, float] = {}

    # Build exactly one Ariadne runtime per boundary and reuse it for all
    # three modes. Ariadne variant enumeration is not strictly deterministic
    # across separate build_split_runtime calls, so rebuilding per-mode would
    # cause the suffix parameter sets to drift between modes.
    for choice in choices:
        _restore_model_state(edge_detector.model, initial_state)
        runtime, build_time = _build_runtime_for_choice(
            split_model=split_model,
            example_batch=example_batch,
            choice=choice,
            args=args,
        )
        runtime_by_boundary[choice.boundary] = runtime
        runtime_build_time_by_boundary[choice.boundary] = float(build_time)

    if "split_cached" in set(modes):
        for choice in choices:
            _restore_model_state(edge_detector.model, initial_state)
            runtime = runtime_by_boundary[choice.boundary]
            split_id = _require_runtime_split_id(runtime)
            graph_signature = _require_runtime_graph_signature(runtime)
            suffix_param_names = tuple(_suffix_parameter_names(runtime))
            _configure_fixed_prefix_training(split_model, runtime)
            split_model.eval()
            logger.info(
                "Prebuilding cached Ariadne boundaries for {} using split_id {}",
                choice.boundary,
                split_id,
            )
            cached_batches, feature_rebuild_time = _build_cached_batches(
                runtime=runtime,
                percent=choice.boundary,
                split_id=split_id,
                graph_signature=graph_signature,
                edge_model=edge_detector.model,
                frames_by_id=frames_by_id,
                sample_ids=sampled_ids,
                annotations=sample_annotations,
                batch_size=int(args.batch_size),
                device=device,
            )
            cached_split = CachedSplitRuntime(
                percent=choice.boundary,
                split_id=split_id,
                graph_signature=graph_signature,
                runtime=runtime,
                cached_batches=cached_batches,
                feature_rebuild_time=feature_rebuild_time,
                runtime_build_time=runtime_build_time_by_boundary[choice.boundary],
                suffix_param_names=suffix_param_names,
            )
            _validate_cached_split_runtime(cached_split)
            cached_by_boundary[choice.boundary] = cached_split
            logger.info(
                "Cached {} boundary batch(es) for {} split_id={} samples={}",
                len(cached_batches),
                choice.boundary,
                split_id,
                cached_split.cached_sample_count,
            )
        _restore_model_state(edge_detector.model, initial_state)

    # Preflight equivalence check: probe one batch per boundary BEFORE the
    # repeat loop so the experiment fails fast if freeze vs split paths would
    # disagree.
    preflight_reports: list[PreflightReport] = []
    preflight_loss_fn = build_split_training_loss(edge_detector.model)
    for choice in choices:
        _restore_model_state(edge_detector.model, initial_state)
        runtime = runtime_by_boundary[choice.boundary]
        runtime_split_id = _require_runtime_split_id(runtime)
        runtime_graph_signature = _require_runtime_graph_signature(runtime)
        suffix_names, _ = _configure_fixed_prefix_training(split_model, runtime)
        report = _preflight_equivalence_check(
            runtime=runtime,
            split_model=split_model,
            edge_model=edge_detector.model,
            choice=choice,
            frames_by_id=frames_by_id,
            sample_ids=sampled_ids,
            annotations=sample_annotations,
            batch_size=int(args.batch_size),
            device=device,
            loss_fn=preflight_loss_fn,
            freeze_trainable_names=suffix_names,
            split_trainable_names=suffix_names,
            cached_split=cached_by_boundary.get(choice.boundary),
            runtime_split_id=runtime_split_id,
            runtime_graph_signature=runtime_graph_signature,
        )
        preflight_reports.append(report)
        logger.info(
            "Preflight OK percent={} split_id={} trainable_params={} "
            "full_loss={:.6f} split_loss={:.6f} output_max_diff={:.3e}",
            report.percent,
            report.actual_split_id,
            report.trainable_parameter_count,
            report.full_loss,
            report.split_loss,
            report.full_output_max_diff,
        )
        _clear_cuda_cache()
    preflight_path = output_root / "preflight.jsonl"
    if preflight_path.exists():
        preflight_path.unlink()
    for report in preflight_reports:
        _append_jsonl(
            preflight_path,
            {
                "percent": report.percent,
                "actual_split_id": report.actual_split_id,
                "graph_signature": report.graph_signature,
                "trainable_parameter_names": list(report.trainable_parameter_names),
                "trainable_parameter_count": report.trainable_parameter_count,
                "full_loss": report.full_loss,
                "split_loss": report.split_loss,
                "full_output_max_diff": report.full_output_max_diff,
                "tolerance": report.tolerance,
            },
        )

    rows: list[dict[str, Any]] = []
    for repeat_id in range(repeat):
        run_seed = int(args.seed) + repeat_id
        for choice in choices:
            cached_split = cached_by_boundary.get(choice.boundary)
            shared_runtime = runtime_by_boundary[choice.boundary]
            shared_runtime_build_time = runtime_build_time_by_boundary[choice.boundary]
            for mode in modes:
                row = _run_one_experiment(
                    mode=str(mode),
                    choice=choice,
                    edge_model=edge_detector.model,
                    split_model=split_model,
                    model_name=str(args.edge_model),
                    golden_model=str(args.golden_model),
                    initial_state=initial_state,
                    example_batch=example_batch,
                    cached_split=cached_split,
                    shared_runtime=shared_runtime,
                    shared_runtime_build_time=shared_runtime_build_time,
                    frame_dir=frame_dir,
                    frames_by_id=frames_by_id,
                    sampled_frame_indices=sampled_ids,
                    annotations=sample_annotations,
                    sample_count=sample_count,
                    epochs=epochs,
                    batch_size=int(args.batch_size),
                    teacher_annotation_time=teacher_annotation_time,
                    learning_rate=learning_rate,
                    optimizer_config=optimizer_config,
                    repeat_id=repeat_id,
                    seed=run_seed,
                    args=args,
                    device=device,
                )
                rows.append(row)
                _append_jsonl(results_path, row)
                _restore_model_state(edge_detector.model, initial_state)
                _clear_cuda_cache()

    _assert_trainable_parameter_equivalence(rows)

    _write_summary_csv(summary_path, rows)
    _write_aggregate_summary_csv(aggregate_summary_path, rows)
    plot_split_time_accuracy_subplots(rows, output_root)
    logger.info("Wrote {}", results_path)
    logger.info("Wrote {}", summary_path)
    logger.info("Wrote {}", aggregate_summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
