from __future__ import annotations

import pytest

from config.runtime import load_runtime_config


def test_default_config_loads_for_main_runtime() -> None:
    config = load_runtime_config("./config/config.yaml")

    assert config.baseline.enabled is False
    assert config.server.edge_model_name == "yolo26n"
    assert config.client.feature_upload.sync_high_quality is False
    assert config.client.resource_aware_trigger.max_training_samples == 128
    assert config.client.resource_aware_trigger.min_training_samples_by_model == {
        "rfdetr_nano": 64
    }
    assert config.client.resource_aware_trigger.max_training_samples_by_model == {
        "rfdetr_nano": 64
    }
    assert config.client.resource_aware_trigger.bootstrap_without_drift is True
    assert config.server.continual_learning.training_replay_fraction_by_model == {
        "yolo26n": 0.25
    }
    assert config.server.continual_learning.training_frame_count_by_model == {
        "rfdetr_nano": 64
    }


def test_high_quality_feature_sync_flag_is_loaded(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
client:
  feature_upload:
    sync_high_quality: false
""",
        encoding="utf-8",
    )

    config = load_runtime_config(path)

    assert config.client.feature_upload.sync_high_quality is False


@pytest.mark.parametrize("yaml_value", ["1", '"false"'])
def test_high_quality_feature_sync_flag_requires_boolean(tmp_path, yaml_value: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
client:
  feature_upload:
    sync_high_quality: {yaml_value}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sync_high_quality"):
        load_runtime_config(path)


@pytest.mark.parametrize("field_name", ["enabled", "retain_empty_predictions"])
@pytest.mark.parametrize("yaml_value", ["1", '"false"'])
def test_teacher_sampling_flags_require_boolean(
    tmp_path,
    field_name: str,
    yaml_value: str,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
client:
  sample_quality:
    teacher_sampling:
      {field_name}: {yaml_value}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field_name):
        load_runtime_config(path)


@pytest.mark.parametrize("yaml_value", ["1", '"false"'])
def test_feature_update_on_anomaly_requires_boolean(tmp_path, yaml_value: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
client:
  sample_quality:
    boundary_feature_entropy:
      update_on_anomaly: {yaml_value}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="update_on_anomaly"):
        load_runtime_config(path)


def test_training_upload_cap_cannot_be_smaller_than_trigger_minimum(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
client:
  resource_aware_trigger:
    min_training_samples: 8
    max_training_samples: 7
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_training_samples"):
        load_runtime_config(path)


def test_model_training_minimum_cannot_exceed_inherited_upload_cap(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
client:
  resource_aware_trigger:
    min_training_samples: 8
    max_training_samples: 8
    min_training_samples_by_model:
      yolo26n: 9
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_training_samples_by_model"):
        load_runtime_config(path)


@pytest.mark.parametrize("yaml_value", ["1", '"true"'])
def test_bootstrap_without_drift_requires_boolean(tmp_path, yaml_value: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
client:
  resource_aware_trigger:
    bootstrap_without_drift: {yaml_value}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bootstrap_without_drift"):
        load_runtime_config(path)


def test_ekya_uses_common_server_models_without_method_override(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
baseline:
  enabled: true
  method: Ekya
server:
  edge_model_name: yolo26n
  golden: rtdetr_x
""",
        encoding="utf-8",
    )

    config = load_runtime_config(path)

    assert config.server.edge_model_name == "yolo26n"
    assert config.server.golden == "rtdetr_x"


@pytest.mark.parametrize(
    "field",
    ["student_model", "teacher_model", "allow_model_override"],
)
def test_removed_ekya_model_override_fields_are_rejected(tmp_path, field: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
server:
  baselines:
    Ekya:
      {field}: legacy_value
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field):
        load_runtime_config(path)


@pytest.mark.parametrize(
    ("env_name", "attribute_path", "expected"),
    [
        (
            "PLANK_ROAD__BASELINE__SURGEON__TRAIN_SAMPLE_COUNT",
            "baseline.SURGEON.train_sample_count",
            3,
        ),
        (
            "PLANK_ROAD__BASELINE__CATR__TRIGGER_WINDOW_SIZE",
            "baseline.CATR.trigger_window_size",
            7,
        ),
        (
            "PLANK_ROAD__SERVER__BASELINES__EKYA__WINDOW_SIZE",
            "server.baselines.Ekya.window_size",
            9,
        ),
    ],
)
def test_env_overrides_preserve_canonical_baseline_section_names(
    tmp_path,
    monkeypatch,
    env_name: str,
    attribute_path: str,
    expected: int,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")
    monkeypatch.setenv(env_name, str(expected))

    value = load_runtime_config(path)
    for attribute in attribute_path.split("."):
        value = getattr(value, attribute)

    assert value == expected


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("suffix_num_threads", "inference_num_threads"),
        ("suffix_thread_tuning_iterations", "no longer used"),
    ],
)
def test_removed_fixed_split_thread_fields_are_rejected(
    tmp_path,
    field: str,
    replacement: str,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
client:
  split_learning:
    fixed_split:
      {field}: 4
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=replacement):
        load_runtime_config(path)


@pytest.mark.parametrize("field", ["max_candidates", "privacy_metric_lower_bound"])
def test_removed_fixed_split_compatibility_fields_are_rejected(
    tmp_path,
    field: str,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
client:
  split_learning:
    fixed_split:
      {field}: 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field):
        load_runtime_config(path)


def test_removed_surgeon_tta_steps_is_rejected(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
baseline:
  SURGEON:
    tta_steps: 4
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tta_steps"):
        load_runtime_config(path)


@pytest.mark.parametrize("yaml_value", ["0", "-1", "true", '"4"'])
def test_inference_num_threads_requires_positive_integer(tmp_path, yaml_value: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
client:
  split_learning:
    fixed_split:
      inference_num_threads: {yaml_value}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="inference_num_threads"):
        load_runtime_config(path)


def test_experiment_results_config_is_shared_and_validated(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
experiment_results:
  root_dir: cloud-results
  local_root_dir: edge-results
  max_artifact_bytes: 1024
""",
        encoding="utf-8",
    )
    config = load_runtime_config(path)
    assert config.client.experiment_results is config.experiment_results
    assert config.server.experiment_results is config.experiment_results
    assert config.experiment_results.local_root_dir == "edge-results"
    assert config.experiment_results.upload_enabled is True


def test_experiment_run_config_is_loaded_and_normalized(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
experiment_run:
  experiment_id: suwon5a_weather
  scenario: Rainy Scene
  video_path: ./video_data/rainy.mp4
  max_count: 2000
  edge_count: 1
  repeat: r02
""",
        encoding="utf-8",
    )
    config = load_runtime_config(path)

    assert config.experiment_run.experiment_id == "suwon5a_weather"
    assert config.experiment_run.scenario == "rainy-scene"
    assert config.experiment_run.video_path == "./video_data/rainy.mp4"
    assert config.experiment_run.max_count == 2000
    assert config.experiment_run.edge_count == 1
    assert config.experiment_run.repeat == 2
    assert config.client.source.video_path == "./video_data/rainy.mp4"
    assert config.client.source.max_count == 2000
    assert config.client.source.scenario_name == "rainy-scene"


def test_experiment_run_defaults_do_not_force_specific_scenario(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")

    config = load_runtime_config(path)

    assert config.experiment_run.experiment_id == "default_experiment"
    assert config.experiment_run.scenario == ""
    assert config.experiment_run.video_path == ""
    assert config.experiment_run.max_count == 1000
    assert config.experiment_run.edge_count == 1
    assert config.experiment_run.repeat == 1
    assert config.client.source.max_count == 1000


def test_pure_edge_training_frame_count_override_does_not_change_shared_baseline(
    tmp_path,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
baseline:
  training:
    training_frame_count: 128
  SURGEON:
    training_frame_count: 32
""",
        encoding="utf-8",
    )

    config = load_runtime_config(path)

    assert config.baseline.training.training_frame_count == 128
    assert config.baseline.SURGEON.training_frame_count == 32


def test_pure_edge_train_sample_count_does_not_change_shared_baseline(
    tmp_path,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
baseline:
  training:
    training_frame_count: 128
  SURGEON:
    train_sample_count: 32
""",
        encoding="utf-8",
    )

    config = load_runtime_config(path)

    assert config.baseline.training.training_frame_count == 128
    assert config.baseline.SURGEON.train_sample_count == 32


def test_pure_edge_adaptive_entropy_gate_config_loads(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
baseline:
  SURGEON:
    entropy_margin_ratio: 0.4
    adaptive_entropy_gate: true
    max_entropy_margin_ratio: 0.7
""",
        encoding="utf-8",
    )

    config = load_runtime_config(path)

    assert config.baseline.SURGEON.adaptive_entropy_gate is True
    assert config.baseline.SURGEON.max_entropy_margin_ratio == pytest.approx(0.7)


def test_pure_edge_universal_tta_defaults_load(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("baseline: {}\n", encoding="utf-8")

    config = load_runtime_config(path)
    surgeon = config.baseline.SURGEON

    assert surgeon.train_sample_count == 16
    assert surgeon.guard_sample_count == 8
    assert surgeon.num_epoch == 30
    assert surgeon.max_selected_logit_count == 256
    assert surgeon.reference_consistency_weight == pytest.approx(0.05)
    assert surgeon.max_foreground_growth_ratio == pytest.approx(2.0)
    assert surgeon.max_foreground_fraction_increase == pytest.approx(0.02)
    assert surgeon.max_reference_kl == pytest.approx(0.10)
    assert surgeon.max_relative_param_delta == pytest.approx(0.02)


def test_pure_edge_train_and_guard_samples_must_fit_window(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
baseline:
  training:
    training_frame_count: 100
  SURGEON:
    train_sample_count: 96
    guard_sample_count: 8
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"train_sample_count \+ guard_sample_count"):
        load_runtime_config(path)


def test_pure_edge_max_selected_logits_must_cover_minimum(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
baseline:
  SURGEON:
    min_selected_logit_count: 16
    max_selected_logit_count: 8
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_selected_logit_count must be >="):
        load_runtime_config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_foreground_growth_ratio", 0.9),
        ("max_foreground_fraction_increase", 1.1),
        ("max_reference_kl", -0.1),
        ("max_relative_param_delta", 1.1),
    ],
)
def test_pure_edge_guard_limits_are_validated(tmp_path, field: str, value: float) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
baseline:
  SURGEON:
    {field}: {value}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field):
        load_runtime_config(path)


def test_pure_edge_adaptive_entropy_gate_requires_boolean(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
baseline:
  SURGEON:
    adaptive_entropy_gate: "false"
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="baseline.SURGEON.adaptive_entropy_gate must be a boolean",
    ):
        load_runtime_config(path)


@pytest.mark.parametrize(
    ("entropy_margin", "max_entropy_margin", "message"),
    [
        (0.8, 0.7, "entropy_margin_ratio must be <="),
        (0.4, 1.1, "max_entropy_margin_ratio must be <= 1"),
        (0.4, -0.1, "max_entropy_margin_ratio"),
    ],
)
def test_pure_edge_entropy_gate_margins_are_validated(
    tmp_path,
    entropy_margin: float,
    max_entropy_margin: float,
    message: str,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
baseline:
  SURGEON:
    entropy_margin_ratio: {entropy_margin}
    max_entropy_margin_ratio: {max_entropy_margin}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_runtime_config(path)


def test_pure_edge_training_frame_count_override_requires_positive_value(
    tmp_path,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
baseline:
  SURGEON:
    training_frame_count: 0
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="baseline.SURGEON.training_frame_count",
    ):
        load_runtime_config(path)


@pytest.mark.parametrize(
    ("yaml_value", "message"),
    [
        ("0", "baseline.SURGEON.train_sample_count"),
        ("129", "train_sample_count must be <="),
    ],
)
def test_pure_edge_train_sample_count_is_validated(
    tmp_path,
    yaml_value: str,
    message: str,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
baseline:
  training:
    training_frame_count: 128
  SURGEON:
    train_sample_count: {yaml_value}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_runtime_config(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("experiment_id", '""', "experiment_run.experiment_id"),
        ("max_count", "0", "experiment_run.max_count"),
        ("edge_count", "0", "edge_count must be a positive integer"),
        ("repeat", "0", "repeat must be a positive integer"),
    ],
)
def test_experiment_run_config_is_validated(
    tmp_path,
    field: str,
    value: str,
    message: str,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
experiment_run:
  experiment_id: suwon5a_weather
  scenario: rainy
  edge_count: 1
  repeat: 1
  {field}: {value}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_runtime_config(path)


def test_removed_experiment_result_config_fields_are_rejected(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
experiment_results:
  comparison_id: comparison-a
  upload_to_cloud: true
client:
  source:
    video_path: road.mp4
    max_count: 2000
    scenario_name: road
    video_slug: road
baseline:
  run_id: old-run
server:
  edge_affine_workers:
    run_id: old-main
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="removed by the current experiment layout"):
        load_runtime_config(path)


@pytest.mark.parametrize(
    "ekya_yaml",
    [
        "enabled: true",
        "edge_streaming:\n        display_cloud_results_only: true",
        "cloud_inference:\n        drop_stale_display_packets: true",
        "teacher_labeling:\n        enabled: true",
        "microprofile:\n        prediction_model: simple_linear",
        "microprofile:\n        candidate_hyperparameters: []",
        "retraining:\n        save_checkpoints: true",
        "retraining:\n        optimizer_name: adamw",
    ],
)
def test_removed_ekya_fixed_behavior_fields_are_rejected(tmp_path, ekya_yaml: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
server:
  baselines:
    Ekya:
      {ekya_yaml}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no longer supports"):
        load_runtime_config(path)
