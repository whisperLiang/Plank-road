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
  comparison_id: comparison-a
  root_dir: cloud-results
  local_root_dir: edge-results
  max_artifact_bytes: 1024
  pure_edge_remote_sync:
    target: user@example.com:/srv/plank-road
    timeout_sec: 15
""",
        encoding="utf-8",
    )
    config = load_runtime_config(path)
    assert config.client.experiment_results is config.experiment_results
    assert config.server.experiment_results is config.experiment_results
    assert config.experiment_results.local_root_dir == "edge-results"
    assert config.experiment_results.pure_edge_remote_sync.target == (
        "user@example.com:/srv/plank-road"
    )
    assert config.experiment_results.pure_edge_remote_sync.timeout_sec == 15


def test_experiment_results_upload_requires_edge_summary(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
experiment_results:
  enabled: true
  upload_to_cloud: true
  include_edge_summary: false
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="include_edge_summary"):
        load_runtime_config(path)


@pytest.mark.parametrize(
    "ekya_yaml",
    [
        "enabled: true",
        "edge_streaming:\n        display_cloud_results_only: true",
        "cloud_inference:\n        drop_stale_display_packets: true",
        "teacher_labeling:\n        enabled: true",
        "microprofile:\n        prediction_model: simple_linear",
        "retraining:\n        save_checkpoints: true",
        "logging:\n        result_schema_version: 1",
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
