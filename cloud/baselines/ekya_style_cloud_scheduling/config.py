from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

METHOD = "ekya_style_cloud_scheduling"


@dataclass(frozen=True)
class CandidateHyperparameters:
    id: str
    epochs: int
    train_batch_size: int
    test_batch_size: int
    learning_rate: float
    subsample: float

    @classmethod
    def from_value(cls, value: Mapping[str, Any] | object) -> "CandidateHyperparameters":
        missing = [
            key
            for key in (
                "id",
                "epochs",
                "train_batch_size",
                "test_batch_size",
                "learning_rate",
                "subsample",
            )
            if _get(value, key, None) in (None, "")
        ]
        if missing:
            raise ValueError(
                "ekya_style_cloud_scheduling.microprofile.candidate_hyperparameters "
                f"entry is missing: {', '.join(missing)}"
            )
        candidate = cls(
            id=str(_get(value, "id")),
            epochs=int(_get(value, "epochs")),
            train_batch_size=int(_get(value, "train_batch_size")),
            test_batch_size=int(_get(value, "test_batch_size")),
            learning_rate=float(_get(value, "learning_rate")),
            subsample=float(_get(value, "subsample")),
        )
        candidate.validate()
        return candidate

    def validate(self) -> None:
        if not self.id:
            raise ValueError("candidate hyperparameter id must be non-empty")
        for name, value in (
            ("epochs", self.epochs),
            ("train_batch_size", self.train_batch_size),
            ("test_batch_size", self.test_batch_size),
        ):
            if int(value) <= 0:
                raise ValueError(f"candidate {self.id}: {name} must be positive")
        if self.learning_rate <= 0:
            raise ValueError(f"candidate {self.id}: learning_rate must be positive")
        if self.subsample <= 0 or self.subsample > 1:
            raise ValueError(f"candidate {self.id}: subsample must be in (0, 1]")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "epochs": int(self.epochs),
            "train_batch_size": int(self.train_batch_size),
            "test_batch_size": int(self.test_batch_size),
            "learning_rate": float(self.learning_rate),
            "subsample": float(self.subsample),
        }


@dataclass(frozen=True)
class EdgeStreamingConfig:
    enabled: bool = True
    upload_format: str = "jpeg"
    jpeg_quality: int = 85
    max_inflight_frames: int = 4
    upload_queue_size: int = 8
    result_queue_size: int = 8
    drop_stale_results: bool = True
    display_cloud_results_only: bool = True


@dataclass(frozen=True)
class CloudInferenceConfig:
    score_threshold: float = 0.3
    batch_size: int = 1
    high_priority: bool = True
    async_result_return: bool = True
    result_queue_size: int = 8
    drop_stale_display_packets: bool = True


@dataclass(frozen=True)
class TeacherLabelingConfig:
    enabled: bool = True
    batch_size: int = 1
    score_threshold: float = 0.3
    cache_labels: bool = True
    run_async: bool = True


@dataclass(frozen=True)
class MicroprofileConfig:
    enabled: bool = True
    microprofile_epochs: int = 1
    microprofile_subsample_rate: float = 0.25
    resources_per_trial: float = 0.25
    metric: str = "map"
    prediction_model: str = "simple_linear"
    candidate_hyperparameters: tuple[CandidateHyperparameters, ...] = field(
        default_factory=tuple
    )


@dataclass(frozen=True)
class SchedulerConfig:
    name: str = "ekya_thief_style"
    retraining_period_s: float = 64.0
    inference_resource_floor: float = 0.5
    microprofile_resource_fraction: float = 0.25
    steal_increment: float = 0.1
    allow_inference_only_when_no_gain: bool = True
    fail_on_microprofile_overrun: bool = False
    protect_inference_from_training: bool = True
    warm_start_retraining: bool = False


@dataclass(frozen=True)
class RetrainingConfig:
    enabled: bool = True
    adopt_only_if_improved: bool = True
    min_map_gain_to_adopt: float = 0.0
    max_concurrent_train_jobs: int = 1
    save_checkpoints: bool = True
    run_async: bool = True
    trainable_param_ratio: float | None = None


@dataclass(frozen=True)
class LoggingConfig:
    result_schema_version: int = 1
    log_internal_ids: bool = False
    diagnostics: bool = False


@dataclass(frozen=True)
class EkyaStyleCloudSchedulingConfig:
    enabled: bool
    run_id: str
    student_model: str
    teacher_model: str
    video_path: str
    video_name: str
    offline_cloud_video_debug: bool
    num_frames: int
    window_size: int
    seed: int
    class_names: tuple[str, ...]
    result_root: Path
    output_dir: Path
    edge_streaming: EdgeStreamingConfig
    cloud_inference: CloudInferenceConfig
    teacher_labeling: TeacherLabelingConfig
    microprofile: MicroprofileConfig
    scheduler: SchedulerConfig
    retraining: RetrainingConfig
    logging: LoggingConfig
    allow_model_override: bool = False

    def validate(self) -> None:
        if not self.allow_model_override and self.student_model != "rfdetr_nano":
            raise ValueError(
                "ekya_style_cloud_scheduling.student_model must be rfdetr_nano "
                "unless allow_model_override=true"
            )
        if not self.allow_model_override and self.teacher_model != "rtdetr_x":
            raise ValueError(
                "ekya_style_cloud_scheduling.teacher_model must be rtdetr_x "
                "unless allow_model_override=true"
            )
        if not self.edge_streaming.enabled:
            raise ValueError("ekya_style_cloud_scheduling.edge_streaming.enabled must be true")
        if not self.edge_streaming.display_cloud_results_only:
            raise ValueError(
                "ekya_style_cloud_scheduling.edge_streaming.display_cloud_results_only "
                "must be true"
            )
        if self.window_size <= 0:
            raise ValueError("ekya_style_cloud_scheduling.window_size must be positive")
        if self.num_frames < self.window_size:
            raise ValueError(
                "ekya_style_cloud_scheduling.num_frames must be >= window_size"
            )
        if not self.microprofile.candidate_hyperparameters:
            raise ValueError(
                "ekya_style_cloud_scheduling.microprofile.candidate_hyperparameters "
                "must not be empty"
            )
        if self.edge_streaming.upload_format != "jpeg":
            raise ValueError(
                "ekya_style_cloud_scheduling.edge_streaming.upload_format currently "
                "supports only jpeg"
            )
        if not 1 <= int(self.edge_streaming.jpeg_quality) <= 100:
            raise ValueError("ekya_style_cloud_scheduling.jpeg_quality must be in [1, 100]")


def parse_ekya_style_config(
    runtime_config: object,
    *,
    run_id: str,
    video_path: str | None = None,
    result_root: str | Path | None = None,
) -> EkyaStyleCloudSchedulingConfig:
    server = _get(runtime_config, "server", runtime_config)
    client = _get(runtime_config, "client", None)
    server_baselines = _get(server, "baselines", None)
    section = _get(server_baselines, METHOD, None)
    if section is None:
        baseline = _get(runtime_config, "baseline", None)
        section = _get(baseline, METHOD, None)
    section = section or {}

    resolved_run_id = str(
        run_id or _get(_get(runtime_config, "baseline", None), "run_id", "") or ""
    )
    if not resolved_run_id:
        raise ValueError("run_id must be non-empty for ekya_style_cloud_scheduling")

    source = _get(client, "source", None)
    resolved_video_path = str(
        video_path
        or _get(section, "video_path", "")
        or _get(source, "video_path", "")
        or "./video_data/road.mp4"
    )
    resolved_result_root = Path(
        result_root
        or _get(section, "result_root", "")
        or _get(section, "results_root", "")
        or "results/cloud"
    )
    output_dir = resolved_result_root / resolved_run_id / "baselines" / METHOD
    candidates = tuple(
        CandidateHyperparameters.from_value(item)
        for item in list(
            _get(
                _get(section, "microprofile", None),
                "candidate_hyperparameters",
                _default_candidates(),
            )
            or []
        )
    )
    config = EkyaStyleCloudSchedulingConfig(
        enabled=bool(_get(section, "enabled", True)),
        run_id=resolved_run_id,
        student_model=str(
            _get(section, "student_model", "")
            or _get(server, "edge_model_name", "")
            or "rfdetr_nano"
        ),
        teacher_model=str(
            _get(section, "teacher_model", "") or _get(server, "golden", "") or "rtdetr_x"
        ),
        video_path=resolved_video_path,
        video_name=Path(resolved_video_path).name,
        offline_cloud_video_debug=bool(_get(section, "offline_cloud_video_debug", False)),
        num_frames=int(_get(section, "num_frames", 512)),
        window_size=int(_get(section, "window_size", 64)),
        seed=int(_get(section, "seed", 42)),
        class_names=tuple(str(value) for value in list(_get(client, "class_names", []) or [])),
        result_root=resolved_result_root,
        output_dir=output_dir,
        edge_streaming=_edge_streaming_config(_get(section, "edge_streaming", None)),
        cloud_inference=_cloud_inference_config(_get(section, "cloud_inference", None)),
        teacher_labeling=_teacher_labeling_config(_get(section, "teacher_labeling", None)),
        microprofile=_microprofile_config(
            _get(section, "microprofile", None),
            candidates=candidates,
        ),
        scheduler=_scheduler_config(_get(section, "scheduler", None)),
        retraining=_retraining_config(_get(section, "retraining", None)),
        logging=_logging_config(_get(section, "logging", None)),
        allow_model_override=bool(_get(section, "allow_model_override", False)),
    )
    config.validate()
    return config


def _edge_streaming_config(value: object) -> EdgeStreamingConfig:
    return EdgeStreamingConfig(
        enabled=bool(_get(value, "enabled", True)),
        upload_format=str(_get(value, "upload_format", "jpeg") or "jpeg").lower(),
        jpeg_quality=int(_get(value, "jpeg_quality", 85)),
        max_inflight_frames=int(_get(value, "max_inflight_frames", 4)),
        upload_queue_size=int(_get(value, "upload_queue_size", 8)),
        result_queue_size=int(_get(value, "result_queue_size", 8)),
        drop_stale_results=bool(_get(value, "drop_stale_results", True)),
        display_cloud_results_only=bool(_get(value, "display_cloud_results_only", True)),
    )


def _cloud_inference_config(value: object) -> CloudInferenceConfig:
    return CloudInferenceConfig(
        score_threshold=float(_get(value, "score_threshold", 0.3)),
        batch_size=int(_get(value, "batch_size", 1)),
        high_priority=bool(_get(value, "high_priority", True)),
        async_result_return=bool(_get(value, "async_result_return", True)),
        result_queue_size=int(_get(value, "result_queue_size", 8)),
        drop_stale_display_packets=bool(_get(value, "drop_stale_display_packets", True)),
    )


def _teacher_labeling_config(value: object) -> TeacherLabelingConfig:
    return TeacherLabelingConfig(
        enabled=bool(_get(value, "enabled", True)),
        batch_size=int(_get(value, "batch_size", 1)),
        score_threshold=float(_get(value, "score_threshold", 0.3)),
        cache_labels=bool(_get(value, "cache_labels", True)),
        run_async=bool(_get(value, "run_async", True)),
    )


def _microprofile_config(
    value: object,
    *,
    candidates: tuple[CandidateHyperparameters, ...],
) -> MicroprofileConfig:
    return MicroprofileConfig(
        enabled=bool(_get(value, "enabled", True)),
        microprofile_epochs=int(_get(value, "microprofile_epochs", 1)),
        microprofile_subsample_rate=float(_get(value, "microprofile_subsample_rate", 0.25)),
        resources_per_trial=float(_get(value, "resources_per_trial", 0.25)),
        metric=str(_get(value, "metric", "map") or "map"),
        prediction_model=str(_get(value, "prediction_model", "simple_linear") or "simple_linear"),
        candidate_hyperparameters=candidates,
    )


def _scheduler_config(value: object) -> SchedulerConfig:
    return SchedulerConfig(
        name=str(_get(value, "name", "ekya_thief_style") or "ekya_thief_style"),
        retraining_period_s=float(_get(value, "retraining_period_s", 64.0)),
        inference_resource_floor=float(_get(value, "inference_resource_floor", 0.5)),
        microprofile_resource_fraction=float(_get(value, "microprofile_resource_fraction", 0.25)),
        steal_increment=float(_get(value, "steal_increment", 0.1)),
        allow_inference_only_when_no_gain=bool(
            _get(value, "allow_inference_only_when_no_gain", True)
        ),
        fail_on_microprofile_overrun=bool(_get(value, "fail_on_microprofile_overrun", False)),
        protect_inference_from_training=bool(_get(value, "protect_inference_from_training", True)),
        warm_start_retraining=bool(_get(value, "warm_start_retraining", False)),
    )


def _retraining_config(value: object) -> RetrainingConfig:
    ratio = _get(value, "trainable_param_ratio", None)
    return RetrainingConfig(
        enabled=bool(_get(value, "enabled", True)),
        adopt_only_if_improved=bool(_get(value, "adopt_only_if_improved", True)),
        min_map_gain_to_adopt=float(_get(value, "min_map_gain_to_adopt", 0.0)),
        max_concurrent_train_jobs=int(_get(value, "max_concurrent_train_jobs", 1)),
        save_checkpoints=bool(_get(value, "save_checkpoints", True)),
        run_async=bool(_get(value, "run_async", True)),
        trainable_param_ratio=None if ratio in (None, "") else float(ratio),
    )


def _logging_config(value: object) -> LoggingConfig:
    return LoggingConfig(
        result_schema_version=int(_get(value, "result_schema_version", 1)),
        log_internal_ids=bool(_get(value, "log_internal_ids", False)),
        diagnostics=bool(_get(value, "diagnostics", False)),
    )


def _default_candidates() -> list[dict[str, Any]]:
    return [
        {
            "id": "hp_small",
            "epochs": 1,
            "train_batch_size": 2,
            "test_batch_size": 1,
            "learning_rate": 0.00001,
            "subsample": 0.25,
        },
        {
            "id": "hp_medium",
            "epochs": 2,
            "train_batch_size": 2,
            "test_batch_size": 1,
            "learning_rate": 0.00001,
            "subsample": 0.5,
        },
        {
            "id": "hp_large",
            "epochs": 3,
            "train_batch_size": 2,
            "test_batch_size": 1,
            "learning_rate": 0.000005,
            "subsample": 1.0,
        },
    ]


def _get(value: object, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return default
    return getattr(value, name, default)
