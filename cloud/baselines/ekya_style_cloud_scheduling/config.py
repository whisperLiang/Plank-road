from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

METHOD = "ekya_style_cloud_scheduling"


@dataclass(frozen=True)
class FixedTrainingConfig:
    epochs: int
    train_batch_size: int
    test_batch_size: int
    learning_rate: float
    hp_id: str = "fixed"
    subsample: float = 1.0

    def validate(self) -> None:
        if not self.hp_id:
            raise ValueError("fixed training hp_id must be non-empty")
        for name, value in (
            ("epochs", self.epochs),
            ("train_batch_size", self.train_batch_size),
            ("test_batch_size", self.test_batch_size),
        ):
            if int(value) <= 0:
                raise ValueError(f"fixed training config: {name} must be positive")
        if self.learning_rate <= 0:
            raise ValueError("fixed training config: learning_rate must be positive")
        if float(self.subsample) != 1.0:
            raise ValueError("fixed training config: subsample must be 1.0")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.hp_id,
            "epochs": int(self.epochs),
            "train_batch_size": int(self.train_batch_size),
            "test_batch_size": int(self.test_batch_size),
            "learning_rate": float(self.learning_rate),
            "subsample": float(self.subsample),
        }


@dataclass(frozen=True)
class EdgeStreamingConfig:
    jpeg_quality: int = 85
    upload_queue_size: int = 8


@dataclass(frozen=True)
class CloudInferenceConfig:
    score_threshold: float = 0.3
    batch_size: int = 1


@dataclass(frozen=True)
class TeacherLabelingConfig:
    batch_size: int = 1
    score_threshold: float = 0.3


@dataclass(frozen=True)
class MicroprofileConfig:
    microprofile_epochs: int = 1


@dataclass(frozen=True)
class DatasetConfig:
    train_val_split: float = 0.75
    min_train_samples: int = 1
    min_val_samples: int = 1


@dataclass(frozen=True)
class EvaluationConfig:
    score_threshold: float = 0.3
    iou_threshold: float = 0.5


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
    adopt_only_if_improved: bool = True
    min_map_gain_to_adopt: float = 0.0
    drop_training_when_active_same_connection: bool = True
    training_admission_scope: str = "edge_camera"
    max_concurrent_train_jobs: int = 1
    train_mode: str = "full"
    trainable_param_ratio: float | None = None
    optimizer_name: str = "adamw"
    weight_decay: float = 0.0


@dataclass(frozen=True)
class EkyaStyleCloudSchedulingConfig:
    run_id: str
    student_model: str
    teacher_model: str
    video_path: str
    video_name: str
    num_frames: int
    window_size: int
    seed: int
    class_names: tuple[str, ...]
    result_root: Path
    output_dir: Path
    edge_streaming: EdgeStreamingConfig
    cloud_inference: CloudInferenceConfig
    teacher_labeling: TeacherLabelingConfig
    fixed_training: FixedTrainingConfig
    microprofile: MicroprofileConfig
    dataset: DatasetConfig
    evaluation: EvaluationConfig
    scheduler: SchedulerConfig
    retraining: RetrainingConfig
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
        if self.window_size <= 0:
            raise ValueError("ekya_style_cloud_scheduling.window_size must be positive")
        if self.num_frames < self.window_size:
            raise ValueError("ekya_style_cloud_scheduling.num_frames must be >= window_size")
        self.fixed_training.validate()
        if self.microprofile.microprofile_epochs <= 0:
            raise ValueError(
                "ekya_style_cloud_scheduling.microprofile.microprofile_epochs must be positive"
            )
        if not 1 <= int(self.edge_streaming.jpeg_quality) <= 100:
            raise ValueError("ekya_style_cloud_scheduling.jpeg_quality must be in [1, 100]")
        if self.dataset.train_val_split <= 0.0 or self.dataset.train_val_split >= 1.0:
            raise ValueError(
                "ekya_style_cloud_scheduling.dataset.train_val_split must be in (0, 1)"
            )
        if self.dataset.min_train_samples <= 0 or self.dataset.min_val_samples <= 0:
            raise ValueError(
                "ekya_style_cloud_scheduling.dataset min sample counts must be positive"
            )
        if self.evaluation.score_threshold < 0.0:
            raise ValueError(
                "ekya_style_cloud_scheduling.evaluation.score_threshold must be non-negative"
            )
        if self.evaluation.iou_threshold <= 0.0 or self.evaluation.iou_threshold > 1.0:
            raise ValueError(
                "ekya_style_cloud_scheduling.evaluation.iou_threshold must be in (0, 1]"
            )
        train_mode = str(self.retraining.train_mode or "").strip().lower()
        if train_mode not in {"full", "freeze"}:
            raise ValueError(
                "ekya_style_cloud_scheduling.retraining.train_mode must be full or freeze"
            )
        if train_mode == "freeze" and self.retraining.trainable_param_ratio is None:
            raise ValueError(
                "ekya_style_cloud_scheduling.retraining.trainable_param_ratio is required "
                "when train_mode=freeze"
            )
        if self.retraining.trainable_param_ratio is not None and (
            self.retraining.trainable_param_ratio <= 0.0
            or self.retraining.trainable_param_ratio > 1.0
        ):
            raise ValueError(
                "ekya_style_cloud_scheduling.retraining.trainable_param_ratio must be in (0, 1]"
            )
        if str(self.retraining.optimizer_name or "").strip().lower() not in {
            "adamw",
            "adam",
            "sgd",
        }:
            raise ValueError(
                "ekya_style_cloud_scheduling.retraining.optimizer_name must be adamw, adam, or sgd"
            )
        if self.retraining.weight_decay < 0.0:
            raise ValueError("ekya_style_cloud_scheduling.retraining.weight_decay must be >= 0")
        scope = str(
            self.retraining.training_admission_scope or "edge_camera"
        ).strip().lower()
        if scope not in {"edge_camera", "edge_only", "global"}:
            raise ValueError(
                "ekya_style_cloud_scheduling.retraining.training_admission_scope "
                "must be edge_camera, edge_only, or global"
            )


def parse_ekya_style_config(
    runtime_config: object,
    *,
    run_id: str,
    video_path: str | None = None,
    result_root: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> EkyaStyleCloudSchedulingConfig:
    server = _get(runtime_config, "server", runtime_config)
    client = _get(runtime_config, "client", None)
    baseline = _get(runtime_config, "baseline", None)
    server_baselines = _get(server, "baselines", None)
    section = _get(server_baselines, METHOD, None)
    if section is None:
        raise ValueError(
            "server.baselines.ekya_style_cloud_scheduling is required for "
            "ekya_style_cloud_scheduling"
        )

    resolved_run_id = str(run_id or _get(baseline, "run_id", "") or "")
    if not resolved_run_id:
        raise ValueError("run_id must be non-empty for ekya_style_cloud_scheduling")

    source = _get(client, "source", None)
    student_model = str(
        _required_value(
            _configured_value(
                _get(section, "student_model", None), _get(server, "edge_model_name", None)
            ),
            "ekya_style_cloud_scheduling.student_model",
        )
    )
    teacher_model = str(
        _required_value(
            _configured_value(_get(section, "teacher_model", None), _get(server, "golden", None)),
            "ekya_style_cloud_scheduling.teacher_model",
        )
    )
    resolved_video_path = str(
        _required_value(
            video_path
            or _configured_value(
                _get(section, "video_path", None), _get(source, "video_path", None)
            ),
            "ekya_style_cloud_scheduling.video_path",
        )
    )
    resolved_result_root = Path(
        result_root or _get(section, "result_root", "") or "results/cloud"
    )
    resolved_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else resolved_result_root / resolved_run_id / "baselines" / METHOD
    )
    microprofile_section = _get(section, "microprofile", None)
    accuracy_cfg = _get(baseline, "accuracy_trigger_cloud_retraining", None)
    config = EkyaStyleCloudSchedulingConfig(
        run_id=resolved_run_id,
        student_model=student_model,
        teacher_model=teacher_model,
        video_path=resolved_video_path,
        video_name=Path(resolved_video_path).name,
        num_frames=int(
            _required_value(
                _configured_value(
                    _get(section, "num_frames", None), _get(source, "max_count", None)
                ),
                "ekya_style_cloud_scheduling.num_frames",
            )
        ),
        window_size=int(
            _required_value(
                _configured_value(
                    _get(section, "window_size", None),
                    _get(accuracy_cfg, "trigger_window_size", None),
                ),
                "ekya_style_cloud_scheduling.window_size",
            )
        ),
        seed=int(_get(section, "seed", 42)),
        class_names=tuple(str(value) for value in list(_get(client, "class_names", []) or [])),
        result_root=resolved_result_root,
        output_dir=resolved_output_dir,
        edge_streaming=_edge_streaming_config(_get(section, "edge_streaming", None)),
        cloud_inference=_cloud_inference_config(
            _get(section, "cloud_inference", None),
            client=client,
        ),
        teacher_labeling=_teacher_labeling_config(
            _get(section, "teacher_labeling", None),
            server=server,
        ),
        fixed_training=_fixed_training_config(
            server=server,
            baseline=baseline,
            student_model=student_model,
        ),
        microprofile=_microprofile_config(
            microprofile_section,
            baseline=baseline,
        ),
        dataset=_dataset_config(_get(section, "dataset", None), server=server, baseline=baseline),
        evaluation=_evaluation_config(_get(section, "evaluation", None), baseline=baseline),
        scheduler=_scheduler_config(_get(section, "scheduler", None)),
        retraining=_retraining_config(
            _get(section, "retraining", None), server=server, baseline=baseline
        ),
        allow_model_override=bool(_get(section, "allow_model_override", False)),
    )
    config.validate()
    return config


def _configured_value(value: Any, default: Any) -> Any:
    return default if value is None or value == "" else value


def _required_value(value: Any, name: str) -> Any:
    if value is None or value == "":
        raise ValueError(f"{name} must be configured")
    return value


def _fixed_training_config(
    *,
    server: object,
    baseline: object | None,
    student_model: str,
) -> FixedTrainingConfig:
    continual_learning = _get(server, "continual_learning", None)
    baseline_training = _get(baseline, "training", None)
    batch_size = _configured_value(
        _get(continual_learning, "batch_size", None),
        _get(baseline_training, "batch_size", 1),
    )
    config = FixedTrainingConfig(
        epochs=int(
            _configured_value(
                _get(continual_learning, "num_epoch", None),
                _get(baseline_training, "num_epoch", 1),
            )
        ),
        train_batch_size=int(batch_size),
        test_batch_size=int(batch_size),
        learning_rate=float(
            _fixed_training_learning_rate(
                server=server,
                baseline=baseline,
                student_model=student_model,
            )
        ),
        hp_id="fixed",
        subsample=1.0,
    )
    config.validate()
    return config


def _fixed_training_learning_rate(
    *,
    server: object,
    baseline: object | None,
    student_model: str,
) -> float:
    continual_learning = _get(server, "continual_learning", None)
    baseline_training = _get(baseline, "training", None)
    family = _model_family(student_model)
    family_field = {
        "rfdetr": "rfdetr_fixed_split_learning_rate",
        "tinynext": "tinynext_fixed_split_learning_rate",
    }.get(family)
    if family_field:
        value = _get(continual_learning, family_field, None)
        if value not in (None, ""):
            return float(value)
    value = _get(continual_learning, "split_learning_rate", None)
    if value not in (None, ""):
        return float(value)
    return float(_get(baseline_training, "learning_rate", 1.0e-3))


def _model_family(model_name: str) -> str:
    normalized = str(model_name or "").strip().lower()
    if normalized.startswith("rfdetr") or normalized.startswith("rf-detr"):
        return "rfdetr"
    if normalized.startswith("tinynext"):
        return "tinynext"
    if normalized.startswith("yolo"):
        return "yolo"
    return normalized


def _edge_streaming_config(value: object) -> EdgeStreamingConfig:
    return EdgeStreamingConfig(
        jpeg_quality=int(_get(value, "jpeg_quality", 85)),
        upload_queue_size=int(_get(value, "upload_queue_size", 8)),
    )


def _cloud_inference_config(value: object, *, client: object | None) -> CloudInferenceConfig:
    return CloudInferenceConfig(
        score_threshold=float(
            _configured_value(
                _get(value, "score_threshold", None),
                _get(client, "final_detection_threshold", 0.0),
            )
        ),
        batch_size=int(_get(value, "batch_size", 1)),
    )


def _teacher_labeling_config(value: object, *, server: object) -> TeacherLabelingConfig:
    continual_learning = _get(server, "continual_learning", None)
    return TeacherLabelingConfig(
        batch_size=int(
            _required_value(
                _configured_value(
                    _get(value, "batch_size", None),
                    _get(continual_learning, "teacher_batch_size", None),
                ),
                "ekya_style_cloud_scheduling.teacher_labeling.batch_size",
            )
        ),
        score_threshold=float(
            _required_value(
                _configured_value(
                    _get(value, "score_threshold", None),
                    _get(continual_learning, "teacher_annotation_threshold", None),
                ),
                "ekya_style_cloud_scheduling.teacher_labeling.score_threshold",
            )
        ),
    )


def _microprofile_config(
    value: object,
    *,
    baseline: object | None,
) -> MicroprofileConfig:
    baseline_training = _get(baseline, "training", None)
    return MicroprofileConfig(
        microprofile_epochs=int(
            _configured_value(
                _get(value, "microprofile_epochs", None),
                _get(baseline_training, "microprofile_epochs", 1),
            )
        ),
    )


def _dataset_config(value: object, *, server: object, baseline: object | None) -> DatasetConfig:
    continual_learning = _get(server, "continual_learning", None)
    baseline_training = _get(baseline, "training", None)
    default_train_val_split = 1.0 - float(
        _get(continual_learning, "proxy_eval_validation_fraction", 0.25)
    )
    return DatasetConfig(
        train_val_split=float(
            _configured_value(_get(value, "train_val_split", None), default_train_val_split)
        ),
        min_train_samples=int(
            _configured_value(
                _get(value, "min_train_samples", None),
                _get(baseline_training, "min_training_samples", 1),
            )
        ),
        min_val_samples=int(_configured_value(_get(value, "min_val_samples", None), 1)),
    )


def _evaluation_config(value: object, *, baseline: object | None) -> EvaluationConfig:
    accuracy_cfg = _get(baseline, "accuracy_trigger_cloud_retraining", None)
    return EvaluationConfig(
        score_threshold=float(
            _configured_value(
                _get(value, "score_threshold", None),
                _get(accuracy_cfg, "agreement_score_threshold", 0.0),
            )
        ),
        iou_threshold=float(
            _configured_value(
                _get(value, "iou_threshold", None),
                _get(accuracy_cfg, "agreement_iou_threshold", 0.5),
            )
        ),
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


def _retraining_config(
    value: object,
    *,
    server: object,
    baseline: object | None,
) -> RetrainingConfig:
    continual_learning = _get(server, "continual_learning", None)
    baseline_training = _get(baseline, "training", None)
    accuracy_cfg = _get(baseline, "accuracy_trigger_cloud_retraining", None)
    train_mode = (
        str(
            _configured_value(
                _get(value, "train_mode", None),
                _get(accuracy_cfg, "training_strategy", "full"),
            )
            or "full"
        )
        .strip()
        .lower()
    )
    ratio = _get(value, "trainable_param_ratio", None)
    if ratio in (None, "") and train_mode == "freeze":
        ratio = _get(accuracy_cfg, "trainable_param_ratio", None)
    return RetrainingConfig(
        adopt_only_if_improved=bool(_get(value, "adopt_only_if_improved", True)),
        min_map_gain_to_adopt=float(_get(value, "min_map_gain_to_adopt", 0.0)),
        drop_training_when_active_same_connection=bool(
            _get(value, "drop_training_when_active_same_connection", True)
        ),
        training_admission_scope=str(
            _get(value, "training_admission_scope", "edge_camera") or "edge_camera"
        )
        .strip()
        .lower(),
        max_concurrent_train_jobs=int(
            _configured_value(
                _get(value, "max_concurrent_train_jobs", None),
                _get(continual_learning, "max_concurrent_jobs", 1),
            )
        ),
        train_mode=train_mode,
        trainable_param_ratio=None if ratio in (None, "") else float(ratio),
        optimizer_name=str(
            _get(baseline_training, "optimizer_name", "adamw") or "adamw"
        )
        .strip()
        .lower(),
        weight_decay=float(_get(baseline_training, "weight_decay", 0.0)),
    )


def _get(value: object, name: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(name, default)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return default
    return getattr(value, name, default)
