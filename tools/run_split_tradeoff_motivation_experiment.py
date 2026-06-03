"""
Split Model Tradeoff Motivation Experiment
===========================================

Visualizes intermediate feature size and privacy leakage tradeoffs
for arbitrary detection models under different TorchLens split candidates.

This script performs split candidate profiling and plotting without
participating in training or modifying the fixed_split/split_runtime/retrain pipelines.

Usage:
    python tools/run_split_tradeoff_motivation_experiment.py \\
        --model tinynext \\
        --device cpu \\
        --input-size 640 640 \\
        --initial-input-size 640 640 \\
        --max-candidates 64 \\
        --output-dir results/split_tradeoff/tinynext

Output:
    - split_tradeoff_candidates.csv
    - split_tradeoff_candidates.json
    - split_payload_privacy_by_depth.pdf/png
    - split_pareto_tradeoff.pdf/png
    - split_constraint_feasibility.pdf/png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from loguru import logger
except ModuleNotFoundError:
    class _FallbackLogger:
        def _log(self, level: str, message: str) -> None:
            print(f"{level}: {message}", file=sys.stderr)

        def debug(self, message: str) -> None:
            self._log("DEBUG", message)

        def info(self, message: str) -> None:
            self._log("INFO", message)

        def warning(self, message: str) -> None:
            self._log("WARNING", message)

        def error(self, message: str) -> None:
            self._log("ERROR", message)

    logger = _FallbackLogger()

try:
    import torch
except ModuleNotFoundError as exc:
    _TORCH_IMPORT_ERROR = exc

    class _MissingDevice:
        def __init__(self, spec: str):
            self.spec = str(spec)
            self.type = self.spec.split(":", 1)[0]

        def __str__(self) -> str:
            return self.spec

    class _MissingCuda:
        @staticmethod
        def empty_cache() -> None:
            return None

    class _MissingTorch:
        cuda = _MissingCuda()

        class nn:
            class Module:
                pass

        @staticmethod
        def manual_seed(seed: int) -> None:
            return None

        @staticmethod
        def device(spec: str) -> _MissingDevice:
            return _MissingDevice(spec)

        @staticmethod
        def zeros(*args, **kwargs):
            raise RuntimeError("torch is required to create model inputs") from _TORCH_IMPORT_ERROR

    torch = _MissingTorch()  # type: ignore[assignment]

try:
    from torchlens.split import SplitRuntime
except Exception as exc:
    _TORCHLENS_IMPORT_ERROR = exc

    class SplitRuntime:
        pass

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import load_runtime_config
except Exception as exc:
    _CONFIG_IMPORT_ERROR = exc

    def load_runtime_config(*args, **kwargs):
        raise RuntimeError("config runtime dependencies are required") from _CONFIG_IMPORT_ERROR

try:
    from model_management.model_zoo import build_detection_model, list_available_models
except Exception as exc:
    _MODEL_ZOO_IMPORT_ERROR = exc

    def build_detection_model(*args, **kwargs):
        raise RuntimeError("model_management.model_zoo is required to build models") from _MODEL_ZOO_IMPORT_ERROR

    def list_available_models() -> list[str]:
        return []

try:
    from model_management.split_candidate import SplitCandidate
except Exception as exc:
    _SPLIT_CANDIDATE_IMPORT_ERROR = exc

    @dataclass
    class SplitCandidate:
        candidate_id: str
        edge_nodes: list[str]
        cloud_nodes: list[str]
        boundary_edges: list[Any]
        boundary_tensor_labels: list[str]
        edge_input_labels: list[str]
        cloud_input_labels: list[str]
        cloud_output_labels: list[str]
        estimated_edge_flops: float
        estimated_cloud_flops: float
        estimated_payload_bytes: int
        estimated_privacy_risk: float
        estimated_latency: float
        is_trainable_tail: bool
        legacy_layer_index: int | None
        boundary_count: int
        edge_parameter_count: int
        total_parameter_count: int
        edge_parameter_ratio: float
        metadata: dict[str, Any] = field(default_factory=dict)

try:
    from model_management.universal_model_split import UniversalModelSplitter
except Exception as exc:
    _UNIVERSAL_SPLITTER_IMPORT_ERROR = exc

    class UniversalModelSplitter:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "model_management.universal_model_split is required to trace models"
            ) from _UNIVERSAL_SPLITTER_IMPORT_ERROR

try:
    from model_management.split_model_adapters import (
        build_split_runtime_sample_input,
        get_split_runtime_model,
    )
except Exception as exc:
    _SPLIT_ADAPTER_IMPORT_ERROR = exc

    def build_split_runtime_sample_input(*args, **kwargs):
        raise RuntimeError(
            "model_management.split_model_adapters is required to prepare trace inputs"
        ) from _SPLIT_ADAPTER_IMPORT_ERROR

    def get_split_runtime_model(model):
        return model


# ───────────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────────

PRIVACY_LEAKAGE_EPSILON = 1e-12
DEFAULT_CONFIG_PATH = "config/config.yaml"
DEFAULT_LIGHTWEIGHT_MODELS = ["yolo26", "tinynext", "rfdetr", "yolov8s"]
MODEL_ALIAS_BUILD_CANDIDATES = {
    "yolo26": ("yolo26n", "yolo26s", "yolo26"),
    "tinynext": ("tinynext_s", "tinynext_m", "tinynext"),
    "rfdetr": ("rfdetr_nano", "rfdetr_small", "rfdetr"),
}
MODEL_SUMMARY_FIELDNAMES = [
    "model",
    "status",
    "error",
    "candidate_count",
    "valid_candidate_count",
    "validation_passed_count",
    "trainable_candidate_count",
    "payload_min_mb",
    "payload_max_mb",
    "payload_mean_mb",
    "payload_median_mb",
    "payload_spread_ratio",
    "payload_spread_log10",
    "privacy_min",
    "privacy_max",
    "privacy_mean",
    "privacy_spread",
    "pareto_candidate_count",
    "valid_ratio",
    "trainable_ratio",
    "nontrivial_score",
    "motivation_strength_score",
    "recommended_as_main_figure",
]


# ───────────────────────────────────────────────────────────────────────
# Data Structures
# ───────────────────────────────────────────────────────────────────────


@dataclass
class CandidateRecord:
    """Records profiling metrics for a single split candidate."""

    candidate_id: str
    legacy_layer_index: int | None
    canonical_split_key: str
    boundary_tensor_count: int
    boundary_tensor_labels: str  # JSON array string
    boundary_shape_summary: str  # JSON string
    payload_bytes: int
    payload_mb: float
    input_tensor_bytes: int
    payload_ratio_to_input: float
    edge_parameter_count: int
    total_parameter_count: int
    edge_parameter_ratio: float
    privacy_leakage_official: float
    privacy_leakage_log10: float
    privacy_leakage_score: float
    estimated_edge_flops: float
    estimated_cloud_flops: float
    estimated_latency: float
    is_trainable_tail: bool
    validation_passed: bool
    replay_success_rate: float
    tail_trainability: bool
    measured_edge_latency: float | None = None
    measured_cloud_latency: float | None = None
    measured_end_to_end_latency: float | None = None
    stability_score: float | None = None
    validation_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values for cleaner JSON."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ExperimentMetadata:
    """Metadata for the entire experiment run."""

    model_name: str
    input_height: int
    input_width: int
    initial_input_height: int
    initial_input_width: int
    initial_input_bytes: int
    device: str
    max_candidates: int | None
    max_boundary_count: int
    max_payload_mb: int
    privacy_epsilon: float
    validate_candidates: bool
    candidate_count: int
    trace_signature: str | None = None
    random_seed: int = 42
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelSummary:
    """Summary metrics for one model in the all-model experiment."""

    model: str
    status: str
    error: str = ""
    candidate_count: int = 0
    valid_candidate_count: int = 0
    validation_passed_count: int = 0
    trainable_candidate_count: int = 0
    payload_min_mb: float = 0.0
    payload_max_mb: float = 0.0
    payload_mean_mb: float = 0.0
    payload_median_mb: float = 0.0
    payload_spread_ratio: float = 0.0
    payload_spread_log10: float = 0.0
    privacy_min: float = 0.0
    privacy_max: float = 0.0
    privacy_mean: float = 0.0
    privacy_spread: float = 0.0
    pareto_candidate_count: int = 0
    valid_ratio: float = 0.0
    trainable_ratio: float = 0.0
    nontrivial_score: float = 0.0
    motivation_strength_score: float = 0.0
    recommended_as_main_figure: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelExperimentResult:
    """Artifacts produced for one model run."""

    summary: ModelSummary
    records: list[CandidateRecord] = field(default_factory=list)
    metadata: ExperimentMetadata | None = None
    output_dir: Path | None = None


# ───────────────────────────────────────────────────────────────────────
# Utility Functions
# ───────────────────────────────────────────────────────────────────────


def safe_estimate_privacy_leakage(
    edge_parameter_count: int | float,
    *,
    epsilon: float = PRIVACY_LEAKAGE_EPSILON,
) -> float:
    """Safely compute privacy leakage score from edge parameter count."""
    denominator = max(0.0, float(edge_parameter_count)) + max(0.0, float(epsilon))
    if denominator <= 0.0:
        return float("inf")
    return 1.0 / denominator


def safe_log10(value: float) -> float:
    """Safely compute log10, handling inf values."""
    if not math.isfinite(value):
        return 0.0  # Represent inf as 0 in log scale for plotting
    if value <= 0:
        return 0.0
    return math.log10(value)


def normalize_candidate_limit(max_candidates: int | None) -> int | None:
    """Return None when candidate enumeration should be uncapped."""
    if max_candidates is None:
        return None
    max_candidates = int(max_candidates)
    return max_candidates if max_candidates > 0 else None


def format_candidate_limit(max_candidates: int | None) -> str:
    """Format candidate limit for logs and reports."""
    limit = normalize_candidate_limit(max_candidates)
    return str(limit) if limit is not None else "all"


def clip01(value: float) -> float:
    """Clip a numeric score into [0, 1]."""
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def normalize_model_name(model_name: str) -> str:
    """Normalize user-facing model names consistently with model_zoo."""
    return str(model_name).strip().lower().replace("-", "_")


def resolve_model_build_name(
    model_name: str,
    available_models: Sequence[str] | None = None,
) -> str:
    """Resolve experiment aliases such as yolo26/tinynext/rfdetr to buildable names."""
    normalized = normalize_model_name(model_name)
    available = {normalize_model_name(name) for name in available_models or []}

    if not available:
        try:
            available = {normalize_model_name(name) for name in list_available_models()}
        except Exception as exc:
            logger.warning(f"Failed to inspect model_zoo registry: {exc}")

    if normalized in available:
        return normalized

    for candidate in MODEL_ALIAS_BUILD_CANDIDATES.get(normalized, ()):
        candidate_normalized = normalize_model_name(candidate)
        if not available or candidate_normalized in available:
            return candidate_normalized

    return normalized


def discover_supported_lightweight_models() -> list[str]:
    """Discover supported lightweight detector representatives for batch runs."""
    try:
        available = [normalize_model_name(name) for name in list_available_models()]
        discovered = [
            model
            for model in DEFAULT_LIGHTWEIGHT_MODELS
            if resolve_model_build_name(model, available) in set(available)
        ]
        if discovered:
            logger.info(f"Discovered lightweight models from model_zoo: {discovered}")
            return discovered
    except Exception as exc:
        logger.warning(f"Model discovery from model_zoo failed: {exc}")

    logger.info(f"Using fallback lightweight model list: {DEFAULT_LIGHTWEIGHT_MODELS}")
    return list(DEFAULT_LIGHTWEIGHT_MODELS)


def parse_models_argument(models_arg: str) -> list[str]:
    """Parse --models values while preserving user order."""
    if normalize_model_name(models_arg) == "all":
        return discover_supported_lightweight_models()

    models = [part.strip() for part in str(models_arg).split(",") if part.strip()]
    if not models:
        raise ValueError("--models must be 'all' or a comma-separated model list")
    return models


def resolve_requested_models(args: argparse.Namespace) -> list[str]:
    """Resolve CLI model arguments into a concrete model list."""
    resolved = getattr(args, "resolved_models", None)
    if resolved:
        return list(resolved)

    models_arg = getattr(args, "models", None)
    if models_arg:
        if getattr(args, "model", None):
            logger.warning("--models was provided; ignoring --model")
        return parse_models_argument(models_arg)

    model_name = getattr(args, "model", None) or get_default_model_name()
    return [model_name]


def safe_model_dir_name(model_name: str) -> str:
    """Return a filesystem-safe directory name for a model label."""
    normalized = normalize_model_name(model_name)
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in normalized)


def create_deterministic_sample_input(
    model_name: str,
    input_height: int,
    input_width: int,
    device: torch.device,
) -> Any:
    """Create a deterministic sample input compatible with the model.
    
    Returns batched tensor [1,3,H,W] as default format.
    """
    logger.info(f"Creating sample input for {model_name} ({input_height}x{input_width})")

    # Use batched tensor format [1,3,H,W] as default
    try:
        sample_input = torch.zeros(1, 3, input_height, input_width, device=device)
        logger.debug("Created Tensor[1,3,H,W] format")
        return sample_input
    except Exception as e:
        logger.error(f"Failed to create input: {e}")
        raise RuntimeError(
            f"Cannot create deterministic sample input for {model_name}"
        ) from e


def create_split_runtime_sample_input(
    model: torch.nn.Module,
    model_name: str,
    input_height: int,
    input_width: int,
    device: torch.device,
) -> Any:
    """Create a sample input for the model object actually used by split runtime."""
    logger.info(
        f"Creating split runtime sample input for {model_name} "
        f"({input_height}x{input_width})"
    )
    try:
        sample_input = build_split_runtime_sample_input(
            model,
            image_size=(input_height, input_width),
            device=device,
        )
        return sample_input
    except Exception as exc:
        logger.warning(
            f"Split adapter sample input failed for {model_name}: {exc}; "
            "falling back to Tensor[1,3,H,W]"
        )
        return create_deterministic_sample_input(
            model_name,
            input_height,
            input_width,
            device,
        )


def get_default_model_name() -> str:
    """Read default model name from config or use fallback."""
    try:
        config = load_runtime_config(DEFAULT_CONFIG_PATH)
        model_name = getattr(config.client, "lightweight", None)
        if model_name:
            logger.info(f"Using default model from config: {model_name}")
            return model_name
    except Exception as e:
        logger.warning(f"Failed to read config: {e}")
    
    logger.info("Using fallback default model: yolov8s")
    return "yolov8s"


def build_model_safe(model_name: str, device: torch.device) -> torch.nn.Module:
    """Build a detection model with proper error handling."""
    logger.info(f"Building model: {model_name} on {device}")
    try:
        model = build_detection_model(model_name, pretrained=True)
        model = model.to(device)
        model.eval()
        logger.info(f"Model built successfully: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to build model {model_name}: {e}")
        raise RuntimeError(f"Cannot build model '{model_name}': {e}") from e


# ───────────────────────────────────────────────────────────────────────
# Tracing and Candidate Enumeration
# ───────────────────────────────────────────────────────────────────────


def trace_model_with_splitter(
    model: torch.nn.Module,
    sample_input: Any,
    model_name: str,
    input_height: int,
    input_width: int,
    device: torch.device,
) -> tuple[UniversalModelSplitter, SplitRuntime]:
    """Trace model using UniversalModelSplitter with fallback input formats."""
    logger.info("Tracing model with UniversalModelSplitter...")
    
    splitter = UniversalModelSplitter(device=device)
    
    # Try primary input format (batched tensor)
    try:
        logger.debug("Attempting trace with batched tensor [1,3,H,W]...")
        traced = splitter.trace(model, sample_input, model_name=model_name)
        runtime = getattr(traced, "runtime", None)
        if runtime is None:
            raise RuntimeError("UniversalModelSplitter.trace() did not produce a runtime")
        plan = getattr(runtime, "plan", None)
        logger.info(f"Trace successful with batched tensor format. "
                   f"Prefix nodes: {len(getattr(plan, 'prefix_nodes', ()) or ())}, "
                   f"Suffix nodes: {len(getattr(plan, 'suffix_nodes', ()) or ())}")
        return splitter, runtime
    except Exception as e:
        logger.warning(f"Trace failed with batched tensor: {e}")
    
    # Try fallback format (list of single tensors)
    try:
        logger.debug("Attempting trace with list[Tensor[3,H,W]]...")
        fallback_input = [torch.zeros(3, input_height, input_width, device=device)]
        traced = splitter.trace(model, fallback_input, model_name=model_name)
        runtime = getattr(traced, "runtime", None)
        if runtime is None:
            raise RuntimeError("UniversalModelSplitter.trace() did not produce a runtime")
        plan = getattr(runtime, "plan", None)
        logger.info(f"Trace successful with list format. "
                   f"Prefix nodes: {len(getattr(plan, 'prefix_nodes', ()) or ())}, "
                   f"Suffix nodes: {len(getattr(plan, 'suffix_nodes', ()) or ())}")
        return splitter, runtime
    except Exception as e:
        logger.warning(f"Trace failed with list format: {e}")
    
    # Both formats failed
    logger.error("Trace failed with both input formats")
    raise RuntimeError(
        f"Failed to trace model {model_name} with any input format. "
        f"Tried: Tensor[1,3,H,W] and list[Tensor[3,H,W]]"
    )


def enumerate_candidates(
    splitter: UniversalModelSplitter,
    runtime: SplitRuntime,
    max_candidates: int | None = None,
    max_boundary_count: int = 8,
    max_payload_bytes: int = 128 * 1024 * 1024,
) -> list[SplitCandidate]:
    """Enumerate split candidates from traced runtime."""
    max_candidates = normalize_candidate_limit(max_candidates)
    logger.info(
        f"Enumerating split candidates (max={format_candidate_limit(max_candidates)}, "
        f"max_boundary={max_boundary_count}, "
        f"max_payload={max_payload_bytes / (1024*1024):.1f} MB)..."
    )

    try:
        del runtime
        candidates = splitter.enumerate_candidates(
            max_candidates=max_candidates,
            max_boundary_count=max_boundary_count,
            max_payload_bytes=max_payload_bytes,
        )
        logger.info(f"Enumerated {len(candidates)} valid candidates")
        return list(candidates)
    except Exception as e:
        logger.error(f"Candidate enumeration failed: {e}")
        raise RuntimeError(f"Failed to enumerate candidates: {e}") from e


# ───────────────────────────────────────────────────────────────────────
# Candidate Profiling
# ───────────────────────────────────────────────────────────────────────


def profile_candidates(
    candidates: list[SplitCandidate],
    sample_input: Any,
    runtime: SplitRuntime,
    splitter: UniversalModelSplitter,
    input_size_bytes: int,
    privacy_epsilon: float = PRIVACY_LEAKAGE_EPSILON,
    validate: bool = False,
    initial_input_shape: Sequence[int] | None = None,
) -> list[CandidateRecord]:
    """Profile all candidates and create records."""
    logger.info(f"Profiling {len(candidates)} candidates...")
    
    records: list[CandidateRecord] = []
    
    for idx, candidate in enumerate(candidates):
        logger.debug(f"Profiling candidate {idx + 1}/{len(candidates)}: {candidate.candidate_id}")
        
        try:
            record = _profile_single_candidate(
                candidate,
                idx,
                sample_input,
                runtime,
                splitter,
                input_size_bytes,
                privacy_epsilon,
                validate,
                initial_input_shape,
            )
            records.append(record)
        except Exception as e:
            logger.warning(f"Failed to profile candidate {candidate.candidate_id}: {e}")
            # Still create a record with error information
            record = _create_error_record(candidate, str(e), input_size_bytes)
            records.append(record)
    
    if records and not any(record.legacy_layer_index == 0 for record in records):
        records.insert(
            0,
            _create_initial_input_record(
                sample_input,
                input_size_bytes,
                records,
                initial_input_shape=initial_input_shape,
            ),
        )
    
    logger.info(f"Profiled {len(records)} candidates")
    return records


def _profile_single_candidate(
    candidate: SplitCandidate,
    index: int,
    sample_input: Any,
    runtime: SplitRuntime,
    splitter: UniversalModelSplitter,
    input_size_bytes: int,
    privacy_epsilon: float,
    validate: bool,
    initial_input_shape: Sequence[int] | None = None,
) -> CandidateRecord:
    """Profile a single candidate."""
    
    # Compute privacy metrics
    privacy_leakage_official = safe_estimate_privacy_leakage(
        candidate.edge_parameter_count,
        epsilon=privacy_epsilon,
    )
    privacy_leakage_score = 1.0 - candidate.edge_parameter_ratio
    privacy_leakage_score = max(0.0, min(1.0, privacy_leakage_score))
    
    # Compute payload metrics
    payload_bytes = _display_payload_bytes(candidate, input_size_bytes)
    payload_mb = payload_bytes / (1024 * 1024)
    payload_ratio = (
        float(payload_bytes) / float(input_size_bytes)
        if input_size_bytes > 0
        else 0.0
    )
    
    # Boundary tensor summary
    boundary_labels_json = json.dumps(candidate.boundary_tensor_labels)
    
    # Boundary shape summary
    if candidate.legacy_layer_index == 0 and initial_input_shape is not None:
        boundary_shape_json = json.dumps(_initial_input_shape_summary(initial_input_shape))
    else:
        boundary_shape = _get_boundary_shape_summary(candidate, runtime)
        boundary_shape_json = json.dumps(boundary_shape)
    
    # Validation
    validation_passed = True
    validation_error = None
    if validate:
        try:
            validation_passed = _validate_candidate(candidate, splitter, sample_input)
        except Exception as e:
            validation_passed = False
            validation_error = str(e)
    
    # Canonical split key
    canonical_split_key = candidate.metadata.get("canonical_split_key", candidate.candidate_id)
    
    return CandidateRecord(
        candidate_id=candidate.candidate_id,
        legacy_layer_index=candidate.legacy_layer_index,
        canonical_split_key=canonical_split_key,
        boundary_tensor_count=candidate.boundary_count,
        boundary_tensor_labels=boundary_labels_json,
        boundary_shape_summary=boundary_shape_json,
        payload_bytes=payload_bytes,
        payload_mb=payload_mb,
        input_tensor_bytes=input_size_bytes,
        payload_ratio_to_input=payload_ratio,
        edge_parameter_count=candidate.edge_parameter_count,
        total_parameter_count=candidate.total_parameter_count,
        edge_parameter_ratio=candidate.edge_parameter_ratio,
        privacy_leakage_official=privacy_leakage_official,
        privacy_leakage_log10=safe_log10(privacy_leakage_official),
        privacy_leakage_score=privacy_leakage_score,
        estimated_edge_flops=candidate.estimated_edge_flops,
        estimated_cloud_flops=candidate.estimated_cloud_flops,
        estimated_latency=candidate.estimated_latency,
        is_trainable_tail=candidate.is_trainable_tail,
        validation_passed=validation_passed,
        replay_success_rate=1.0 if validation_passed else 0.0,
        tail_trainability=candidate.is_trainable_tail,
        validation_error=validation_error,
    )


def _display_payload_bytes(candidate: SplitCandidate, input_size_bytes: int) -> int:
    """Return the payload size to show for split trade-off outputs."""
    if candidate.legacy_layer_index == 0 and input_size_bytes > 0:
        return int(input_size_bytes)
    return int(candidate.estimated_payload_bytes)


def _sample_input_shape_summary(sample_input: Any) -> list[list[Any]]:
    """Return a compact shape summary for the initial model input."""
    if isinstance(sample_input, list):
        return [
            [f"input_{idx}", list(item.shape)]
            for idx, item in enumerate(sample_input)
            if hasattr(item, "shape")
        ]
    if hasattr(sample_input, "shape"):
        return [["input", list(sample_input.shape)]]
    return [["input", None]]


def _initial_input_shape_summary(initial_input_shape: Sequence[int]) -> list[list[Any]]:
    return [["input", [int(dim) for dim in initial_input_shape]]]


def _create_initial_input_record(
    sample_input: Any,
    input_size_bytes: int,
    profiled_records: Sequence[CandidateRecord],
    *,
    initial_input_shape: Sequence[int] | None = None,
) -> CandidateRecord:
    """Create the layer-0 baseline that represents sending the raw input."""
    total_parameter_count = max(
        (record.total_parameter_count for record in profiled_records),
        default=0,
    )
    privacy_leakage_official = safe_estimate_privacy_leakage(0)
    payload_mb = input_size_bytes / (1024 * 1024)
    return CandidateRecord(
        candidate_id="initial_input",
        legacy_layer_index=0,
        canonical_split_key="initial_input",
        boundary_tensor_count=1,
        boundary_tensor_labels=json.dumps(["input"]),
        boundary_shape_summary=json.dumps(
            _initial_input_shape_summary(initial_input_shape)
            if initial_input_shape is not None
            else _sample_input_shape_summary(sample_input)
        ),
        payload_bytes=int(input_size_bytes),
        payload_mb=payload_mb,
        input_tensor_bytes=int(input_size_bytes),
        payload_ratio_to_input=1.0 if input_size_bytes > 0 else 0.0,
        edge_parameter_count=0,
        total_parameter_count=total_parameter_count,
        edge_parameter_ratio=0.0,
        privacy_leakage_official=privacy_leakage_official,
        privacy_leakage_log10=safe_log10(privacy_leakage_official),
        privacy_leakage_score=1.0,
        estimated_edge_flops=0.0,
        estimated_cloud_flops=0.0,
        estimated_latency=0.0,
        is_trainable_tail=True,
        validation_passed=True,
        replay_success_rate=1.0,
        tail_trainability=True,
    )


def _get_boundary_shape_summary(candidate: SplitCandidate, runtime: SplitRuntime) -> list:
    """Extract boundary tensor shape information."""
    shapes = []
    
    # Try to get from metadata first
    if "boundary_shape_summary" in candidate.metadata:
        return candidate.metadata["boundary_shape_summary"]
    schema = candidate.metadata.get("boundary_schema")
    if isinstance(schema, Sequence) and not isinstance(schema, (str, bytes)):
        return [
            [
                str(item.get("canonical_id") or item.get("torchlens_label") or ""),
                list(item.get("symbolic_shape") or []),
            ]
            for item in schema
            if isinstance(item, Mapping)
        ]
    
    # Try to get from TraceGraph if available.
    try:
        graph = getattr(runtime, "trace_graph", None)
        if graph is not None and hasattr(graph, "nodes"):
            for label in candidate.boundary_tensor_labels:
                node = dict(getattr(graph, "nodes", {}) or {}).get(str(label))
                shape = getattr(node, "shape", None) or getattr(node, "tensor_shape", None)
                shapes.append([label, list(shape) if shape is not None else None])
    except Exception as e:
        logger.debug(f"Failed to extract boundary shape: {e}")
    
    return shapes


def _validate_candidate(
    candidate: SplitCandidate,
    splitter: UniversalModelSplitter,
    sample_input: Any,
) -> bool:
    """Validate a candidate through the Plank-road split facade."""
    del sample_input
    if not hasattr(splitter, "validate_candidate"):
        return True
    
    try:
        result = splitter.validate_candidate(candidate)
        if isinstance(result, Mapping):
            return bool(result.get("success", result.get("valid", False)))
        return bool(result)
    except Exception as e:
        logger.debug(f"Validation error for {candidate.candidate_id}: {e}")
        return False


def _create_error_record(
    candidate: SplitCandidate,
    error_msg: str,
    input_size_bytes: int = 0,
) -> CandidateRecord:
    """Create an error record for a failed candidate."""
    privacy_score = 1.0 - candidate.edge_parameter_ratio
    privacy_score = max(0.0, min(1.0, privacy_score))
    payload_bytes = _display_payload_bytes(candidate, input_size_bytes)
    
    return CandidateRecord(
        candidate_id=candidate.candidate_id,
        legacy_layer_index=candidate.legacy_layer_index,
        canonical_split_key=candidate.candidate_id,
        boundary_tensor_count=candidate.boundary_count,
        boundary_tensor_labels=json.dumps(candidate.boundary_tensor_labels),
        boundary_shape_summary="null",
        payload_bytes=payload_bytes,
        payload_mb=payload_bytes / (1024 * 1024),
        input_tensor_bytes=input_size_bytes,
        payload_ratio_to_input=(
            float(payload_bytes) / float(input_size_bytes)
            if input_size_bytes > 0
            else 0.0
        ),
        edge_parameter_count=candidate.edge_parameter_count,
        total_parameter_count=candidate.total_parameter_count,
        edge_parameter_ratio=candidate.edge_parameter_ratio,
        privacy_leakage_official=float("inf"),
        privacy_leakage_log10=0.0,
        privacy_leakage_score=privacy_score,
        estimated_edge_flops=0.0,
        estimated_cloud_flops=0.0,
        estimated_latency=0.0,
        is_trainable_tail=candidate.is_trainable_tail,
        validation_passed=False,
        replay_success_rate=0.0,
        tail_trainability=candidate.is_trainable_tail,
        validation_error=error_msg,
    )


# ───────────────────────────────────────────────────────────────────────
# Output Functions
# ───────────────────────────────────────────────────────────────────────


def save_candidates_csv(
    records: list[CandidateRecord],
    output_path: Path,
) -> None:
    """Save candidate records to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not records:
        logger.warning("No records to save, creating empty CSV with headers only")
    
    # Define field order
    fieldnames = [
        "candidate_id",
        "legacy_layer_index",
        "canonical_split_key",
        "boundary_tensor_count",
        "boundary_tensor_labels",
        "boundary_shape_summary",
        "payload_bytes",
        "payload_mb",
        "input_tensor_bytes",
        "payload_ratio_to_input",
        "edge_parameter_count",
        "total_parameter_count",
        "edge_parameter_ratio",
        "privacy_leakage_official",
        "privacy_leakage_log10",
        "privacy_leakage_score",
        "estimated_edge_flops",
        "estimated_cloud_flops",
        "estimated_latency",
        "is_trainable_tail",
        "validation_passed",
        "replay_success_rate",
        "tail_trainability",
        "validation_error",
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if records:
            for record in records:
                writer.writerow(record.to_dict())
    
    logger.info(f"Saved {len(records)} records to {output_path}")


def save_candidates_json(
    records: list[CandidateRecord],
    metadata: ExperimentMetadata,
    output_path: Path,
) -> None:
    """Save candidate records and metadata to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "metadata": metadata.to_dict(),
        "candidates": [r.to_dict() for r in records],
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Saved {len(records)} records and metadata to {output_path}")


def compute_pareto_frontier(records: list[CandidateRecord]) -> list[int]:
    """Compute Pareto frontier indices.
    
    We minimize payload_mb and minimize privacy_leakage_score.
    A record is on the frontier if no other record is strictly better in both objectives.
    """
    if not records:
        return []
    
    frontier_indices = []
    
    for i, rec_i in enumerate(records):
        is_dominated = False
        
        for j, rec_j in enumerate(records):
            if i == j:
                continue
            
            # For Pareto frontier with minimization of both payload and privacy leakage:
            # rec_j dominates rec_i if:
            # - rec_j has smaller or equal payload AND
            # - rec_j has equal or lower privacy leakage AND
            # - at least one is strictly better
            payload_better_or_equal = rec_j.payload_mb <= rec_i.payload_mb
            privacy_better_or_equal = rec_j.privacy_leakage_score <= rec_i.privacy_leakage_score
            
            if payload_better_or_equal and privacy_better_or_equal:
                # Check if strictly better in at least one
                payload_strictly_better = rec_j.payload_mb < rec_i.payload_mb
                privacy_strictly_better = rec_j.privacy_leakage_score < rec_i.privacy_leakage_score
                
                if payload_strictly_better or privacy_strictly_better:
                    is_dominated = True
                    break
        
        if not is_dominated:
            frontier_indices.append(i)
    
    return sorted(frontier_indices)


def _finite_float_values(values: Sequence[float]) -> list[float]:
    """Return finite float values only."""
    return [float(value) for value in values if math.isfinite(float(value))]


def _mean(values: Sequence[float]) -> float:
    finite = _finite_float_values(values)
    return float(np.mean(finite)) if finite else 0.0


def _median(values: Sequence[float]) -> float:
    finite = _finite_float_values(values)
    return float(np.median(finite)) if finite else 0.0


def _candidate_depth_sort_key(record: CandidateRecord, index: int) -> tuple[float, float]:
    if record.legacy_layer_index is not None:
        return (0.0, float(record.legacy_layer_index))
    if math.isfinite(float(record.edge_parameter_ratio)):
        return (1.0, float(record.edge_parameter_ratio))
    return (2.0, float(index))


def select_recommended_candidate_index(records: list[CandidateRecord]) -> int | None:
    """Select a balanced candidate on the payload/privacy frontier."""
    if not records:
        return None

    eligible = [
        idx for idx, record in enumerate(records)
        if record.validation_passed and record.is_trainable_tail
    ]
    if not eligible:
        eligible = [
            idx for idx, record in enumerate(records)
            if record.is_trainable_tail
        ]
    if not eligible:
        eligible = list(range(len(records)))

    eligible_records = [records[idx] for idx in eligible]
    eligible_frontier = compute_pareto_frontier(eligible_records)
    if eligible_frontier:
        eligible = [eligible[idx] for idx in eligible_frontier]

    payload_values = _finite_float_values([records[idx].payload_mb for idx in eligible])
    privacy_values = _finite_float_values([
        records[idx].privacy_leakage_score for idx in eligible
    ])
    if not payload_values or not privacy_values:
        return eligible[0]

    payload_min = min(payload_values)
    payload_max = max(payload_values)
    privacy_min = min(privacy_values)
    privacy_max = max(privacy_values)

    def candidate_distance(index: int) -> tuple[float, tuple[float, float], str]:
        record = records[index]
        payload_range = max(payload_max - payload_min, 1e-12)
        privacy_range = max(privacy_max - privacy_min, 1e-12)
        payload_distance = (float(record.payload_mb) - payload_min) / payload_range
        privacy_distance = (float(record.privacy_leakage_score) - privacy_min) / privacy_range
        distance = math.hypot(payload_distance, privacy_distance)
        return (distance, _candidate_depth_sort_key(record, index), record.candidate_id)

    return min(eligible, key=candidate_distance)


def compute_nontrivial_score(records: list[CandidateRecord]) -> float:
    """Score whether the candidate set exposes a non-obvious split trade-off."""
    if not records:
        return 0.0

    payload_min_index = min(range(len(records)), key=lambda idx: records[idx].payload_mb)
    privacy_min_index = min(
        range(len(records)),
        key=lambda idx: records[idx].privacy_leakage_score,
    )
    privacy_max_index = max(
        range(len(records)),
        key=lambda idx: records[idx].privacy_leakage_score,
    )
    _ = privacy_max_index

    score = 0.0
    if records[payload_min_index].candidate_id != records[privacy_min_index].candidate_id:
        score += 0.4

    if len(compute_pareto_frontier(records)) > 1:
        score += 0.4

    recommended_index = select_recommended_candidate_index(records)
    sorted_by_depth = sorted(
        range(len(records)),
        key=lambda idx: _candidate_depth_sort_key(records[idx], idx),
    )
    if recommended_index is not None and sorted_by_depth:
        earliest_index = sorted_by_depth[0]
        latest_index = sorted_by_depth[-1]
        if recommended_index not in {earliest_index, latest_index}:
            score += 0.2

    return clip01(score)


def compute_model_summary(
    model_name: str,
    status: str,
    records: list[CandidateRecord] | None = None,
    *,
    error: str = "",
) -> ModelSummary:
    """Compute all ranking metrics for one model."""
    records = records or []
    candidate_count = len(records)
    if candidate_count == 0:
        return ModelSummary(model=model_name, status=status, error=error)

    payload_values = _finite_float_values([record.payload_mb for record in records])
    privacy_values = _finite_float_values([
        record.privacy_leakage_score for record in records
    ])

    payload_min = min(payload_values) if payload_values else 0.0
    payload_max = max(payload_values) if payload_values else 0.0
    payload_spread_ratio = (
        payload_max / max(payload_min, 1e-12)
        if payload_values
        else 0.0
    )
    payload_spread_log10 = safe_log10(payload_spread_ratio)

    privacy_min = min(privacy_values) if privacy_values else 0.0
    privacy_max = max(privacy_values) if privacy_values else 0.0
    privacy_spread = privacy_max - privacy_min

    validation_passed_count = sum(1 for record in records if record.validation_passed)
    trainable_candidate_count = sum(1 for record in records if record.is_trainable_tail)
    valid_candidate_count = sum(
        1
        for record in records
        if record.validation_passed and record.is_trainable_tail
    )
    valid_ratio = valid_candidate_count / candidate_count if candidate_count else 0.0
    trainable_ratio = (
        trainable_candidate_count / candidate_count if candidate_count else 0.0
    )
    pareto_candidate_count = len(compute_pareto_frontier(records))
    nontrivial_score = compute_nontrivial_score(records)

    candidate_count_score = min(candidate_count / 64.0, 1.0)
    normalized_payload_spread = min(payload_spread_log10 / 2.0, 1.0)
    privacy_spread_score = clip01(privacy_spread)
    pareto_score = min(pareto_candidate_count / 8.0, 1.0)
    valid_ratio_score = clip01(valid_ratio)
    motivation_strength_score = (
        0.25 * candidate_count_score
        + 0.25 * normalized_payload_spread
        + 0.20 * privacy_spread_score
        + 0.15 * pareto_score
        + 0.10 * valid_ratio_score
        + 0.05 * nontrivial_score
    )

    return ModelSummary(
        model=model_name,
        status=status,
        error=error,
        candidate_count=candidate_count,
        valid_candidate_count=valid_candidate_count,
        validation_passed_count=validation_passed_count,
        trainable_candidate_count=trainable_candidate_count,
        payload_min_mb=payload_min,
        payload_max_mb=payload_max,
        payload_mean_mb=_mean(payload_values),
        payload_median_mb=_median(payload_values),
        payload_spread_ratio=payload_spread_ratio,
        payload_spread_log10=payload_spread_log10,
        privacy_min=privacy_min,
        privacy_max=privacy_max,
        privacy_mean=_mean(privacy_values),
        privacy_spread=privacy_spread,
        pareto_candidate_count=pareto_candidate_count,
        valid_ratio=valid_ratio,
        trainable_ratio=trainable_ratio,
        nontrivial_score=nontrivial_score,
        motivation_strength_score=motivation_strength_score,
        recommended_as_main_figure=False,
    )


def rank_model_summaries(summaries: list[ModelSummary]) -> list[ModelSummary]:
    """Rank models and mark the best motivation-figure candidate."""
    for summary in summaries:
        summary.recommended_as_main_figure = False

    ok_summaries = [
        summary
        for summary in summaries
        if summary.status == "ok" and summary.candidate_count > 0
    ]
    ok_summaries.sort(
        key=lambda summary: (-summary.motivation_strength_score, summary.model)
    )
    if ok_summaries:
        ok_summaries[0].recommended_as_main_figure = True

    ok_summary_ids = {id(summary) for summary in ok_summaries}
    failures = [summary for summary in summaries if id(summary) not in ok_summary_ids]
    return ok_summaries + failures


# ───────────────────────────────────────────────────────────────────────
# Plotting Functions
# ───────────────────────────────────────────────────────────────────────


def _is_initial_input_record(record: CandidateRecord) -> bool:
    return record.legacy_layer_index == 0 or record.candidate_id == "initial_input"


def _initial_input_plot_label(record: CandidateRecord) -> str:
    return f"layer 0 input\n{record.payload_mb:.2f} MB"


def plot_payload_privacy_by_depth(
    records: list[CandidateRecord],
    output_dir: Path,
    top_k_labels: int = 8,
) -> None:
    """Create payload and privacy leakage plot by split depth."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plotting")
        return
    
    if not records:
        logger.warning("No records to plot")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by edge_parameter_ratio for x-axis
    sorted_records = sorted(records, key=lambda r: r.edge_parameter_ratio)
    
    x = np.arange(len(sorted_records))
    payload_mb = [r.payload_mb for r in sorted_records]
    privacy_score = [r.privacy_leakage_score for r in sorted_records]
    initial_indices = [
        idx for idx, record in enumerate(sorted_records)
        if _is_initial_input_record(record)
    ]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Top plot: Payload
    bar_colors = [
        "#c2410c" if _is_initial_input_record(record) else "steelblue"
        for record in sorted_records
    ]
    bar_edges = [
        "#7c2d12" if _is_initial_input_record(record) else "navy"
        for record in sorted_records
    ]
    ax1.bar(x, payload_mb, color=bar_colors, alpha=0.7, edgecolor=bar_edges, linewidth=0.5)
    ax1.set_ylabel("Payload (MB)", fontsize=11)
    ax1.set_title("Intermediate Feature Size", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.set_ylim([0, max(payload_mb) * 1.2 if payload_mb else 1])
    
    # Bottom plot: Privacy leakage score
    ax2.plot(x, privacy_score, marker="o", color="darkred", linewidth=1.5, markersize=4, alpha=0.8)
    ax2.fill_between(x, privacy_score, alpha=0.2, color="darkred")
    ax2.set_ylabel("Privacy Leakage Score", fontsize=11)
    ax2.set_xlabel("Split Candidate Ordered by Edge Parameter Ratio", fontsize=11)
    ax2.set_title("Privacy Leakage Score", fontsize=12, fontweight="bold")
    ax2.set_ylim([0, 1.05])
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    
    for idx in initial_indices:
        record = sorted_records[idx]
        ax1.axvline(idx, color="#c2410c", linestyle=":", linewidth=1.2, alpha=0.8)
        ax2.axvline(idx, color="#c2410c", linestyle=":", linewidth=1.2, alpha=0.8)
        ax1.annotate(
            _initial_input_plot_label(record),
            (idx, payload_mb[idx]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#7c2d12",
        )
        ax2.scatter(
            [idx],
            [privacy_score[idx]],
            s=100,
            marker="*",
            c="#c2410c",
            edgecolors="black",
            linewidth=0.6,
            zorder=5,
        )
        ax2.annotate(
            "0",
            (idx, privacy_score[idx]),
            xytext=(6, -14),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="#7c2d12",
        )
    if initial_indices:
        ax2.set_xticks(initial_indices)
        ax2.set_xticklabels(["0\ninput" for _idx in initial_indices], fontsize=9)
    
    # Keep other candidate labels off the dense full-candidate plots.
    _ = top_k_labels
    
    plt.tight_layout()
    
    # Save
    pdf_path = output_dir / "split_payload_privacy_by_depth.pdf"
    png_path = output_dir / "split_payload_privacy_by_depth.png"
    plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved payload/privacy plot to {pdf_path} and {png_path}")


def plot_pareto_tradeoff(
    records: list[CandidateRecord],
    output_dir: Path,
    top_k_labels: int = 8,
) -> None:
    """Create Pareto frontier scatter plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plotting")
        return
    
    if not records:
        logger.warning("No records to plot")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    payload_mb = np.array([r.payload_mb for r in records])
    privacy_score = np.array([r.privacy_leakage_score for r in records])
    edge_param_ratio = np.array([r.edge_parameter_ratio for r in records])
    is_trainable = np.array([r.is_trainable_tail for r in records])
    is_initial = np.array([_is_initial_input_record(r) for r in records])
    
    # Compute Pareto frontier
    frontier_indices = compute_pareto_frontier(records)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot trainable vs untrainable
    trainable_mask = is_trainable & ~is_initial
    untrainable_mask = (~is_trainable) & ~is_initial
    
    scatter1 = ax.scatter(
        payload_mb[trainable_mask],
        privacy_score[trainable_mask],
        c=edge_param_ratio[trainable_mask],
        cmap="viridis",
        s=80,
        alpha=0.7,
        marker="o",
        edgecolors="black",
        linewidth=0.5,
        label="Trainable tail",
    )
    
    ax.scatter(
        payload_mb[untrainable_mask],
        privacy_score[untrainable_mask],
        c=edge_param_ratio[untrainable_mask],
        cmap="viridis",
        s=80,
        alpha=0.7,
        marker="x",
        linewidth=1.5,
        label="Non-trainable tail",
    )
    
    if is_initial.any():
        ax.scatter(
            payload_mb[is_initial],
            privacy_score[is_initial],
            c="#c2410c",
            s=180,
            marker="*",
            edgecolors="black",
            linewidth=0.8,
            label="Initial input (layer 0)",
            zorder=6,
        )
        for idx in np.where(is_initial)[0]:
            ax.annotate(
                _initial_input_plot_label(records[int(idx)]),
                (payload_mb[idx], privacy_score[idx]),
                xytext=(8, -28),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color="#7c2d12",
                arrowprops={"arrowstyle": "->", "color": "#7c2d12", "lw": 0.8},
            )
    
    # Highlight Pareto frontier
    if frontier_indices:
        frontier_payloads = payload_mb[frontier_indices]
        frontier_privacy = privacy_score[frontier_indices]
        ax.plot(
            frontier_payloads,
            frontier_privacy,
            "r--",
            linewidth=1.5,
            alpha=0.5,
            label="Pareto frontier",
        )
    
    ax.set_xlabel("Intermediate Feature Size (MB)", fontsize=11)
    ax.set_ylabel("Privacy Leakage Score", fontsize=11)
    ax.set_title("Split Candidate Pareto Tradeoff", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3, linestyle="--")
    
    cbar = plt.colorbar(scatter1, ax=ax, pad=0.02)
    cbar.set_label("Edge Parameter Ratio", fontsize=10)
    
    ax.legend(loc="best", fontsize=10)
    
    # Add top-k labels
    if top_k_labels > 0 and frontier_indices:
        for idx in frontier_indices[:top_k_labels]:
            if _is_initial_input_record(records[idx]):
                continue
            # Use the candidate's legacy layer index if available, otherwise use record index
            label = f"{records[idx].legacy_layer_index}" if records[idx].legacy_layer_index is not None else f"#{idx}"
            ax.annotate(
                label,
                (payload_mb[idx], privacy_score[idx]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )
    
    plt.tight_layout()
    
    # Save
    pdf_path = output_dir / "split_pareto_tradeoff.pdf"
    png_path = output_dir / "split_pareto_tradeoff.png"
    plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved Pareto plot to {pdf_path} and {png_path}")


def plot_constraint_feasibility(
    records: list[CandidateRecord],
    output_dir: Path,
    privacy_bound: float | None = None,
    max_freezing_ratio: float | None = None,
) -> None:
    """Create constraint feasibility plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plotting")
        return
    
    if not records:
        logger.warning("No records to plot")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    payload_mb = np.array([r.payload_mb for r in records])
    privacy_score = np.array([r.privacy_leakage_score for r in records])
    is_initial = np.array([_is_initial_input_record(r) for r in records])
    is_valid = np.array([
        r.validation_passed and r.is_trainable_tail
        for r in records
    ])
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot valid vs invalid
    valid_mask = is_valid & ~is_initial
    invalid_mask = (~is_valid) & ~is_initial
    
    ax.scatter(
        payload_mb[valid_mask],
        privacy_score[valid_mask],
        c="green",
        s=100,
        alpha=0.6,
        marker="o",
        edgecolors="darkgreen",
        linewidth=1,
        label="Valid (trainable + passed)",
    )
    
    ax.scatter(
        payload_mb[invalid_mask],
        privacy_score[invalid_mask],
        c="red",
        s=100,
        alpha=0.6,
        marker="x",
        linewidth=2,
        label="Invalid",
    )
    
    if is_initial.any():
        ax.scatter(
            payload_mb[is_initial],
            privacy_score[is_initial],
            c="#c2410c",
            s=180,
            marker="*",
            edgecolors="black",
            linewidth=0.8,
            label="Initial input (layer 0)",
            zorder=6,
        )
        for idx in np.where(is_initial)[0]:
            ax.annotate(
                _initial_input_plot_label(records[int(idx)]),
                (payload_mb[idx], privacy_score[idx]),
                xytext=(8, -28),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color="#7c2d12",
                arrowprops={"arrowstyle": "->", "color": "#7c2d12", "lw": 0.8},
            )
    
    # Add constraint boundaries if provided
    if privacy_bound is not None:
        ax.axhline(y=privacy_bound, color="blue", linestyle="--", linewidth=1.5, alpha=0.7,
                  label=f"Privacy bound ({privacy_bound:.2f})")
    
    if max_freezing_ratio is not None:
        ax.axvline(x=max_freezing_ratio * (max(payload_mb) if payload_mb.size > 0 else 1), 
                  color="orange", linestyle="--", linewidth=1.5, alpha=0.7,
                  label="Max freezing ratio")
    
    ax.set_xlabel("Intermediate Feature Size (MB)", fontsize=11)
    ax.set_ylabel("Privacy Leakage Score", fontsize=11)
    ax.set_title("Split Candidate Constraint Feasibility", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=10)
    
    plt.tight_layout()
    
    # Save
    pdf_path = output_dir / "split_constraint_feasibility.pdf"
    png_path = output_dir / "split_constraint_feasibility.png"
    plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Saved feasibility plot to {pdf_path} and {png_path}")


def save_model_ranking_csv(summaries: list[ModelSummary], output_path: Path) -> None:
    """Save all-model summary/ranking metrics as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MODEL_SUMMARY_FIELDNAMES)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.to_dict())
    logger.info(f"Saved model ranking CSV to {output_path}")


def save_model_ranking_json(summaries: list[ModelSummary], output_path: Path) -> None:
    """Save all-model summary/ranking metrics as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    recommended = next(
        (summary.model for summary in summaries if summary.recommended_as_main_figure),
        None,
    )
    payload = {
        "recommended_model": recommended,
        "models": [summary.to_dict() for summary in summaries],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved model ranking JSON to {output_path}")


def _format_metric(value: float, digits: int = 4) -> str:
    if not math.isfinite(float(value)):
        return "0"
    return f"{float(value):.{digits}f}"


def render_model_ranking_markdown(
    summaries: list[ModelSummary],
    args: argparse.Namespace,
) -> str:
    """Render the automatic all-model ranking report."""
    recommended = next(
        (summary for summary in summaries if summary.recommended_as_main_figure),
        None,
    )
    failures = [summary for summary in summaries if summary.status != "ok"]

    lines = [
        "# Split Trade-off Motivation Model Ranking",
        "",
        "## Experiment Settings",
        "",
        f"- Input size: {args.input_size[0]} x {args.input_size[1]}",
        f"- Max candidates: {format_candidate_limit(args.max_candidates)}",
        f"- Max boundary count: {args.max_boundary_count}",
        f"- Max payload MB: {args.max_payload_mb}",
        f"- Validate candidates: {bool(args.validate_candidates)}",
        "",
        "## Model Ranking",
        "",
        (
            "| Rank | Model | Status | Motivation score | Candidates | "
            "Payload spread log10 | Privacy spread | Pareto candidates | "
            "Valid ratio | Main figure | Error |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    rank = 0
    for summary in summaries:
        if summary.status == "ok" and summary.candidate_count > 0:
            rank += 1
            rank_label = str(rank)
        else:
            rank_label = "-"
        error = summary.error.replace("\n", " ").replace("|", "/")
        lines.append(
            "| {rank} | {model} | {status} | {score} | {candidates} | {payload} | "
            "{privacy} | {pareto} | {valid} | {recommended} | {error} |".format(
                rank=rank_label,
                model=summary.model,
                status=summary.status,
                score=_format_metric(summary.motivation_strength_score),
                candidates=summary.candidate_count,
                payload=_format_metric(summary.payload_spread_log10),
                privacy=_format_metric(summary.privacy_spread),
                pareto=summary.pareto_candidate_count,
                valid=_format_metric(summary.valid_ratio),
                recommended="yes" if summary.recommended_as_main_figure else "no",
                error=error,
            )
        )

    lines.extend(["", "## Recommended Model", ""])
    if recommended is None:
        lines.append("No status=ok model with candidates was available for a main figure recommendation.")
    else:
        lines.extend([
            f"Recommended as main figure: **{recommended.model}**.",
            "",
            "Recommendation basis:",
            f"- candidate_count: {recommended.candidate_count}",
            (
                "- payload_spread_log10: "
                f"{_format_metric(recommended.payload_spread_log10)} "
                f"(payload_spread_ratio: {_format_metric(recommended.payload_spread_ratio)})"
            ),
            f"- privacy_spread: {_format_metric(recommended.privacy_spread)}",
            f"- pareto_candidate_count: {recommended.pareto_candidate_count}",
            f"- valid_ratio: {_format_metric(recommended.valid_ratio)}",
        ])
        if recommended.valid_ratio < 0.2:
            lines.extend([
                "",
                (
                    "Warning: this model has a low valid_ratio (< 0.2). "
                    "It may show a clear trade-off, but the trainable/valid "
                    "candidate proportion is low, so use it cautiously as the main figure."
                ),
            ])

    lines.extend(["", "## Failed Models", ""])
    if not failures:
        lines.append("No models failed.")
    else:
        for summary in failures:
            reason = summary.error or "No error message recorded."
            lines.append(f"- {summary.model}: {summary.status}; {reason}")

    lines.extend([
        "",
        "## Conclusion",
        "",
        (
            "The ranking is based on split-tradeoff expressiveness rather than detection accuracy. "
            "A higher score means that the model exposes a clearer difference among split candidates "
            "in communication cost, privacy leakage, and feasibility. Therefore, the top-ranked model "
            "is the most suitable one for the motivation figure."
        ),
        "",
        (
            "motivation_strength_score is only used to choose the model for the motivation "
            "experiment figure. It is not a detection accuracy metric and does not imply that "
            "the selected model has the best detection performance."
        ),
    ])
    return "\n".join(lines) + "\n"


def save_model_ranking_markdown(
    summaries: list[ModelSummary],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    """Save the automatic all-model ranking report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_model_ranking_markdown(summaries, args), encoding="utf-8")
    logger.info(f"Saved model ranking Markdown to {output_path}")


def _plot_all_model_bar(
    summaries: list[ModelSummary],
    output_dir: Path,
    *,
    metric_name: str,
    y_label: str,
    filename_base: str,
    annotate_recommended: bool = False,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping all-model bar plot")
        return

    if not summaries:
        logger.warning("No model summaries to plot")
        return

    sorted_summaries = sorted(
        summaries,
        key=lambda summary: (
            -float(getattr(summary, metric_name)),
            summary.model,
        ),
    )
    model_names = [summary.model for summary in sorted_summaries]
    values = [float(getattr(summary, metric_name)) for summary in sorted_summaries]
    colors = [
        "#2b6cb0" if not summary.recommended_as_main_figure else "#c2410c"
        for summary in sorted_summaries
    ]

    fig_width = max(7.0, 1.3 * len(model_names))
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))
    bars = ax.bar(model_names, values, color=colors, alpha=0.85)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Model")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.tick_params(axis="x", rotation=25)

    if annotate_recommended:
        for bar, summary in zip(bars, sorted_summaries):
            if summary.recommended_as_main_figure:
                ax.annotate(
                    "recommended",
                    (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{filename_base}.pdf"
    png_path = output_dir / f"{filename_base}.png"
    plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved all-model bar plot to {pdf_path} and {png_path}")


def plot_all_models_pareto_overlay(
    records_by_model: Mapping[str, list[CandidateRecord]],
    output_dir: Path,
) -> None:
    """Plot all successful models' candidates in one payload/privacy scatter."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping all-model Pareto overlay")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

    plotted = False
    for idx, (model_name, records) in enumerate(records_by_model.items()):
        if not records:
            continue
        regular_records = [
            record for record in records
            if not _is_initial_input_record(record)
        ]
        initial_records = [
            record for record in records
            if _is_initial_input_record(record)
        ]
        payload_mb = [record.payload_mb for record in regular_records]
        privacy_score = [record.privacy_leakage_score for record in regular_records]
        color = f"C{idx}"
        ax.scatter(
            payload_mb,
            privacy_score,
            c=color,
            s=70,
            alpha=0.7,
            marker=markers[idx % len(markers)],
            label=model_name,
            edgecolors="black",
            linewidth=0.4,
        )
        if initial_records:
            ax.scatter(
                [record.payload_mb for record in initial_records],
                [record.privacy_leakage_score for record in initial_records],
                c=color,
                s=170,
                alpha=0.95,
                marker="*",
                label=f"{model_name} input",
                edgecolors="black",
                linewidth=0.8,
                zorder=6,
            )
            for record in initial_records:
                ax.annotate(
                    "0",
                    (record.payload_mb, record.privacy_leakage_score),
                    xytext=(6, -12),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    color=color,
                )
        plotted = True

    ax.set_xlabel("Intermediate Feature Size (MB)", fontsize=11)
    ax.set_ylabel("Privacy Leakage Score", fontsize=11)
    ax.set_title("All Models Split Candidate Trade-off", fontsize=12, fontweight="bold")
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3, linestyle="--")
    if plotted:
        ax.legend(loc="best", fontsize=10)
    else:
        ax.text(
            0.5,
            0.5,
            "No successful model candidates",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )

    plt.tight_layout()
    pdf_path = output_dir / "all_models_pareto_overlay.pdf"
    png_path = output_dir / "all_models_pareto_overlay.png"
    plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved all-model Pareto overlay to {pdf_path} and {png_path}")


def save_all_model_outputs(
    summaries: list[ModelSummary],
    records_by_model: Mapping[str, list[CandidateRecord]],
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    """Save ranking tables, markdown, and all-model comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_model_ranking_csv(summaries, output_dir / "model_ranking.csv")
    save_model_ranking_json(summaries, output_dir / "model_ranking.json")
    save_model_ranking_markdown(summaries, args, output_dir / "model_ranking.md")
    _plot_all_model_bar(
        summaries,
        output_dir,
        metric_name="motivation_strength_score",
        y_label="Motivation Strength Score",
        filename_base="all_models_motivation_score",
        annotate_recommended=True,
    )
    _plot_all_model_bar(
        summaries,
        output_dir,
        metric_name="payload_spread_log10",
        y_label="Payload Spread (log10 ratio)",
        filename_base="all_models_payload_spread",
    )
    _plot_all_model_bar(
        summaries,
        output_dir,
        metric_name="privacy_spread",
        y_label="Privacy Spread",
        filename_base="all_models_privacy_spread",
    )
    plot_all_models_pareto_overlay(records_by_model, output_dir)


# ───────────────────────────────────────────────────────────────────────
# Main Experiment
# ───────────────────────────────────────────────────────────────────────


def compute_raw_input_size_bytes(
    input_hw: Sequence[int],
    *,
    channels: int = 3,
    bytes_per_channel: int = 1,
) -> int:
    """Compute the shared raw RGB frame size used as the layer-0 baseline."""
    if len(input_hw) != 2:
        raise ValueError("input_hw must contain height and width")
    height, width = (int(input_hw[0]), int(input_hw[1]))
    if height <= 0 or width <= 0:
        raise ValueError("input height and width must be positive")
    return height * width * int(channels) * int(bytes_per_channel)


def _make_failure_result(
    model_name: str,
    status: str,
    error: str,
    output_dir: Path,
) -> ModelExperimentResult:
    return ModelExperimentResult(
        summary=ModelSummary(model=model_name, status=status, error=error),
        records=[],
        metadata=None,
        output_dir=output_dir,
    )


def run_single_model_experiment(
    args: argparse.Namespace,
    model_name: str,
    output_dir: Path,
    device: torch.device,
) -> ModelExperimentResult:
    """Run trace, enumeration, profiling, validation, and plotting for one model."""
    display_model_name = normalize_model_name(model_name)
    build_model_name = resolve_model_build_name(display_model_name)

    logger.info("-" * 70)
    logger.info(f"Running split tradeoff experiment for {display_model_name}")
    logger.info(f"Build target: {build_model_name}")
    logger.info("-" * 70)

    try:
        model = build_model_safe(build_model_name, device)
    except Exception as exc:
        error = str(exc)
        logger.error(f"Model build failed for {display_model_name}: {error}")
        return _make_failure_result(display_model_name, "build_failed", error, output_dir)

    try:
        trace_model = get_split_runtime_model(model)
        trace_model = trace_model.to(device)
        trace_model.eval()
        sample_input = create_split_runtime_sample_input(
            model,
            build_model_name,
            args.input_size[0],
            args.input_size[1],
            device,
        )
        splitter, runtime = trace_model_with_splitter(
            trace_model,
            sample_input,
            build_model_name,
            args.input_size[0],
            args.input_size[1],
            device,
        )
        _ = splitter
    except Exception as exc:
        error = str(exc)
        logger.error(f"Trace failed for {display_model_name}: {error}")
        return _make_failure_result(display_model_name, "trace_failed", error, output_dir)

    try:
        candidates = enumerate_candidates(
            splitter,
            runtime,
            max_candidates=args.max_candidates,
            max_boundary_count=args.max_boundary_count,
            max_payload_bytes=args.max_payload_mb * 1024 * 1024,
        )
    except Exception as exc:
        error = str(exc)
        logger.error(f"Candidate enumeration failed for {display_model_name}: {error}")
        return _make_failure_result(display_model_name, "no_candidates", error, output_dir)

    if not candidates:
        error = "No candidates enumerated"
        logger.error(f"{display_model_name}: {error}")
        return _make_failure_result(display_model_name, "no_candidates", error, output_dir)

    try:
        records = profile_candidates(
            candidates,
            sample_input,
            runtime,
            splitter,
            args.initial_input_bytes,
            privacy_epsilon=args.privacy_epsilon,
            validate=args.validate_candidates,
            initial_input_shape=args.initial_input_shape,
        )
    except Exception as exc:
        error = str(exc)
        logger.error(f"Candidate profiling failed for {display_model_name}: {error}")
        return _make_failure_result(display_model_name, "profile_failed", error, output_dir)

    metadata = ExperimentMetadata(
        model_name=display_model_name,
        input_height=args.input_size[0],
        input_width=args.input_size[1],
        initial_input_height=args.initial_input_shape[0],
        initial_input_width=args.initial_input_shape[1],
        initial_input_bytes=args.initial_input_bytes,
        device=str(device),
        max_candidates=args.max_candidates,
        max_boundary_count=args.max_boundary_count,
        max_payload_mb=args.max_payload_mb,
        privacy_epsilon=args.privacy_epsilon,
        validate_candidates=args.validate_candidates,
        candidate_count=len(records),
        random_seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_candidates_csv(records, output_dir / "split_tradeoff_candidates.csv")
    save_candidates_json(records, metadata, output_dir / "split_tradeoff_candidates.json")

    if args.format in ["pdf", "png", "both"]:
        plot_payload_privacy_by_depth(records, output_dir, top_k_labels=args.top_k_labels)
        plot_pareto_tradeoff(records, output_dir, top_k_labels=args.top_k_labels)
        plot_constraint_feasibility(records, output_dir)

    summary = compute_model_summary(display_model_name, "ok", records)
    return ModelExperimentResult(
        summary=summary,
        records=records,
        metadata=metadata,
        output_dir=output_dir,
    )


def run_all_models_experiment(args: argparse.Namespace) -> list[ModelExperimentResult]:
    """Run the split tradeoff motivation experiment for all requested models."""
    logger.info("=" * 70)
    logger.info("Split Model Tradeoff Motivation Experiment")
    logger.info("=" * 70)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    models = resolve_requested_models(args)
    output_root = Path(args.output_dir)
    multi_model_layout = bool(getattr(args, "multi_model_layout", len(models) > 1))
    initial_input_shape = list(getattr(args, "initial_input_size", None) or args.input_size)
    args.initial_input_shape = initial_input_shape
    args.initial_input_bytes = compute_raw_input_size_bytes(initial_input_shape)
    logger.info(f"Models to test: {models}")
    logger.info(f"Output root: {output_root}")
    logger.info(
        "Shared raw input baseline: "
        f"{initial_input_shape[0]}x{initial_input_shape[1]}x3 uint8 = "
        f"{args.initial_input_bytes / (1024 * 1024):.2f} MB"
    )

    results: list[ModelExperimentResult] = []
    for model_name in models:
        model_output_dir = (
            output_root / safe_model_dir_name(model_name)
            if multi_model_layout
            else output_root
        )
        try:
            result = run_single_model_experiment(
                args,
                model_name,
                model_output_dir,
                device,
            )
        except Exception as exc:
            error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            logger.error(f"Unexpected failure for model {model_name}: {error}")
            result = _make_failure_result(
                normalize_model_name(model_name),
                "trace_failed",
                error,
                model_output_dir,
            )
        results.append(result)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ranked_summaries = rank_model_summaries([result.summary for result in results])
    records_by_model = {
        result.summary.model: result.records
        for result in results
        if result.summary.status == "ok" and result.records
    }
    save_all_model_outputs(ranked_summaries, records_by_model, args, output_root)

    logger.info("=" * 70)
    logger.info(f"Experiment completed. Results saved to: {output_root}")
    logger.info("=" * 70)
    return results


def run_experiment(args: argparse.Namespace) -> None:
    """Run the complete split tradeoff motivation experiment."""
    run_all_models_experiment(args)


# ───────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Split model tradeoff motivation experiment"
    )
    
    # Model and device
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Detection model name (default: read from config or yolov8s)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Models to test: 'all' or comma-separated names "
            "(for example: yolo26,tinynext,rfdetr,yolov8s). "
            "Overrides --model when provided."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (default: cpu)",
    )
    
    # Input
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[640, 640],
        help="Input size H W (default: 640 640)",
    )
    parser.add_argument(
        "--initial-input-size",
        type=int,
        nargs=2,
        default=None,
        help=(
            "Raw input frame size H W for the shared layer-0 baseline "
            "(default: same as --input-size)"
        ),
    )
    
    # Candidate enumeration
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help=(
            "Maximum number of candidates to enumerate. Omit, 0, or negative "
            "means enumerate all candidates that satisfy boundary/payload filters."
        ),
    )
    parser.add_argument(
        "--max-boundary-count",
        type=int,
        default=8,
        help="Maximum number of boundary tensors (default: 8)",
    )
    parser.add_argument(
        "--max-payload-mb",
        type=int,
        default=128,
        help="Maximum payload size in MB (default: 128)",
    )
    
    # Privacy
    parser.add_argument(
        "--privacy-epsilon",
        type=float,
        default=PRIVACY_LEAKAGE_EPSILON,
        help=f"Privacy leakage epsilon (default: {PRIVACY_LEAKAGE_EPSILON})",
    )
    
    # Validation and output
    parser.add_argument(
        "--validate-candidates",
        action="store_true",
        default=False,
        help="Validate candidates using UniversalModelSplitter.validate_candidate()",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory (default: results/split_tradeoff/{model_name} "
            "for --model, results/split_tradeoff/all_models for --models)"
        ),
    )
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=["pdf", "png", "both"],
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--top-k-labels",
        type=int,
        default=8,
        help="Top-k candidates to label in plots (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    args = parser.parse_args()

    try:
        resolved_models = resolve_requested_models(args)
    except Exception as e:
        logger.error(f"Failed to resolve models: {e}")
        sys.exit(2)

    args.resolved_models = resolved_models
    args.multi_model_layout = bool(args.models is not None or len(resolved_models) > 1)

    # Set default output dir if not provided
    if args.output_dir is None:
        if args.multi_model_layout:
            args.output_dir = "results/split_tradeoff/all_models"
        else:
            args.output_dir = f"results/split_tradeoff/{safe_model_dir_name(resolved_models[0])}"
    
    # Run experiment
    try:
        run_experiment(args)
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
