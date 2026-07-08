from __future__ import annotations

import pytest

from config.runtime import load_runtime_config


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


def test_experiment_run_config_is_loaded_and_normalized(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
experiment_run:
  experiment_id: suwon5a_weather
  scenario: Rainy Scene
  edge_count: 1
  repeat: r02
""",
        encoding="utf-8",
    )
    config = load_runtime_config(path)

    assert config.experiment_run.experiment_id == "suwon5a_weather"
    assert config.experiment_run.scenario == "rainy-scene"
    assert config.experiment_run.edge_count == 1
    assert config.experiment_run.repeat == 2


def test_experiment_run_defaults_do_not_force_specific_scenario(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")

    config = load_runtime_config(path)

    assert config.experiment_run.experiment_id == "default_experiment"
    assert config.experiment_run.scenario == ""
    assert config.experiment_run.edge_count == 1
    assert config.experiment_run.repeat == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("experiment_id", '""', "experiment_run.experiment_id"),
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
  scenario: sunny
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
    ekya_style_cloud_scheduling:
      {ekya_yaml}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no longer supports"):
        load_runtime_config(path)
