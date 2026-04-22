from __future__ import annotations

import pytest
import copy

from config import load_runtime_config
from model_management.fixed_split import SplitConstraints


def test_load_runtime_config_builds_typed_sections(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
client:
  edge_id: 42
  source:
    video_path: ./demo.mp4
    max_count: 12
server:
  golden: yolo26x
  workspace_root: ./cache/cloud-workspace
""".strip(),
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)

    assert config.client.edge_id == 42
    assert config.client.source.video_path == "./demo.mp4"
    assert config.client.source.max_count == 12
    assert config.client.retrain.cache_path == "./cache"
    assert config.client.final_detection_threshold == 0.5
    assert config.server.golden == "yolo26x"
    assert config.server.workspace_root == "./cache/cloud-workspace"
    assert config.server.listen_address == "[::]:50051"


def test_load_runtime_config_applies_environment_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
client:
  server_ip: 10.0.0.1:50051
server:
  listen_address: "[::]:50051"
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("PLANK_ROAD__CLIENT__SERVER_IP", "127.0.0.1:60000")
    monkeypatch.setenv("PLANK_ROAD__SERVER__LISTEN_ADDRESS", "[::]:60001")

    config = load_runtime_config(config_path)

    assert config.client.server_ip == "127.0.0.1:60000"
    assert config.server.listen_address == "[::]:60001"


def test_runtime_config_sections_support_deepcopy(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
client:
  server_ip: 10.0.0.1:50051
server:
  listen_address: "[::]:50051"
""".strip(),
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)
    client_copy = copy.deepcopy(config.client)
    server_copy = copy.deepcopy(config.server)

    assert client_copy.server_ip == config.client.server_ip
    assert server_copy.listen_address == config.server.listen_address


def test_load_runtime_config_reads_final_detection_threshold(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
client:
  final_detection_threshold: 0.65
  server_ip: 10.0.0.1:50051
server:
  listen_address: "[::]:50051"
""".strip(),
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)

    assert config.client.final_detection_threshold == 0.65


def test_load_runtime_config_reads_das_strategy(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
client:
  server_ip: 10.0.0.1:50051
server:
  listen_address: "[::]:50051"
  das:
    enabled: true
    strategy: entropy
    probe_samples: 4
""".strip(),
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)

    assert config.server.das.enabled is True
    assert config.server.das.strategy == "entropy"
    assert config.server.das.probe_samples == 4


def test_load_runtime_config_reads_fixed_split_privacy_leakage_bound(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
client:
  server_ip: 10.0.0.1:50051
  split_learning:
    fixed_split:
      privacy_leakage_upper_bound: 0.02
      privacy_leakage_epsilon: 1.0e-9
server:
  listen_address: "[::]:50051"
""".strip(),
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)
    constraints = SplitConstraints.from_config(config.client.split_learning.fixed_split)

    assert constraints.privacy_leakage_upper_bound == 0.02
    assert constraints.privacy_leakage_epsilon == 1.0e-9


def test_fixed_split_constraints_accept_legacy_privacy_metric_name(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
client:
  server_ip: 10.0.0.1:50051
  split_learning:
    fixed_split:
      privacy_metric_lower_bound: 0.03
server:
  listen_address: "[::]:50051"
""".strip(),
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)
    constraints = SplitConstraints.from_config(config.client.split_learning.fixed_split)

    assert constraints.privacy_leakage_upper_bound == 0.03


def test_load_runtime_config_rejects_invalid_das_strategy(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
client:
  server_ip: 10.0.0.1:50051
server:
  listen_address: "[::]:50051"
  das:
    strategy: invalid
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="server.das.strategy"):
        load_runtime_config(config_path)


def test_load_runtime_config_reads_cloud_continual_learning_batch_size(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
client:
  server_ip: 10.0.0.1:50051
server:
  listen_address: "[::]:50051"
  continual_learning:
    batch_size: 4
""".strip(),
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)

    assert config.server.continual_learning.batch_size == 4


def test_load_runtime_config_reads_rfdetr_target_steps_per_round(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
client:
  server_ip: 10.0.0.1:50051
server:
  listen_address: "[::]:50051"
  continual_learning:
    batch_size: 16
    rfdetr_fixed_split_target_steps_per_round: 6
""".strip(),
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)

    assert config.server.continual_learning.rfdetr_fixed_split_target_steps_per_round == 6


def test_load_runtime_config_reads_cloud_proxy_eval_knobs(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
client:
  server_ip: 10.0.0.1:50051
server:
  listen_address: "[::]:50051"
  continual_learning:
    batch_size: 4
    teacher_batch_size: 3
    proxy_eval_max_samples: 24
    proxy_eval_threshold_candidates: [0.08, 0.148, 0.2]
    proxy_eval_frame_cache_enabled: true
""".strip(),
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)

    assert config.server.continual_learning.batch_size == 4
    assert config.server.continual_learning.teacher_batch_size == 3
    assert config.server.continual_learning.proxy_eval_max_samples == 24
    assert config.server.continual_learning.proxy_eval_threshold_candidates == [
        0.08,
        0.148,
        0.2,
    ]
    assert config.server.continual_learning.proxy_eval_frame_cache_enabled is True


def test_load_runtime_config_defaults_teacher_batch_size_to_cloud_batch_size(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
client:
  server_ip: 10.0.0.1:50051
server:
  listen_address: "[::]:50051"
  continual_learning:
    batch_size: 7
""".strip(),
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)

    assert config.server.continual_learning.teacher_batch_size == 7


def test_load_runtime_config_rfdetr_target_steps_per_round_defaults_to_four(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
client:
  server_ip: 10.0.0.1:50051
server:
  listen_address: "[::]:50051"
  continual_learning:
    batch_size: 7
""".strip(),
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)

    assert config.server.continual_learning.rfdetr_fixed_split_target_steps_per_round == 4


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("batch_size", 0),
        ("rfdetr_fixed_split_target_steps_per_round", 0),
    ],
)
def test_load_runtime_config_rejects_invalid_cloud_batch_reconstruction_settings(
    tmp_path,
    field_name,
    field_value,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
client:
  server_ip: 10.0.0.1:50051
server:
  listen_address: "[::]:50051"
  continual_learning:
    {field_name}: {field_value}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=f"server.continual_learning.{field_name}"):
        load_runtime_config(config_path)


@pytest.mark.parametrize(
    ("field_block", "match_text"),
    [
        ("teacher_batch_size: 0", "server.continual_learning.teacher_batch_size"),
        ("proxy_eval_max_samples: -1", "server.continual_learning.proxy_eval_max_samples"),
        (
            "proxy_eval_threshold_candidates: []",
            "server.continual_learning.proxy_eval_threshold_candidates",
        ),
        (
            "proxy_eval_frame_cache_enabled: 1",
            "server.continual_learning.proxy_eval_frame_cache_enabled",
        ),
    ],
)
def test_load_runtime_config_rejects_invalid_cloud_proxy_eval_settings(
    tmp_path,
    field_block,
    match_text,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
client:
  server_ip: 10.0.0.1:50051
server:
  listen_address: "[::]:50051"
  continual_learning:
    batch_size: 4
    {field_block}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=match_text):
        load_runtime_config(config_path)


@pytest.mark.parametrize(
    ("section", "field_name", "match_text"),
    [
        ("client.retrain", "batch_size", "client.retrain.batch_size has been removed"),
        ("client.retrain", "num_epoch", "client.retrain.num_epoch has been removed"),
        ("server.continual_learning", "trace_batch_size", "trace_batch_size has been removed"),
        ("server.continual_learning", "rebuild_batch_size", "rebuild_batch_size has been removed"),
        (
            "server.continual_learning",
            "min_wrapper_fixed_split_num_epoch",
            "min_wrapper_fixed_split_num_epoch has been removed",
        ),
        (
            "server.continual_learning",
            "min_rfdetr_fixed_split_num_epoch",
            "min_rfdetr_fixed_split_num_epoch has been removed",
        ),
    ],
)
def test_load_runtime_config_rejects_removed_cloud_fixed_split_fields(
    tmp_path,
    section,
    field_name,
    match_text,
):
    if section == "client.retrain":
        client_retrain_block = f"""
  retrain:
    cache_path: ./cache
    {field_name}: 2
""".rstrip()
        server_cl_block = """
  continual_learning:
    batch_size: 4
""".rstrip()
    else:
        client_retrain_block = """
  retrain:
    cache_path: ./cache
""".rstrip()
        server_cl_block = f"""
  continual_learning:
    batch_size: 4
    {field_name}: 2
""".rstrip()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
client:
  server_ip: 10.0.0.1:50051
{client_retrain_block}
server:
  listen_address: "[::]:50051"
{server_cl_block}
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=match_text):
        load_runtime_config(config_path)


def test_load_runtime_config_proxy_eval_frame_cache_defaults_to_true(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
client:
  server_ip: 10.0.0.1:50051
server:
  listen_address: "[::]:50051"
  continual_learning:
    batch_size: 4
""".strip(),
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)

    assert config.server.continual_learning.proxy_eval_frame_cache_enabled is True


def test_load_runtime_config_proxy_eval_max_samples_defaults_to_zero(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
client:
  server_ip: 10.0.0.1:50051
server:
  listen_address: "[::]:50051"
  continual_learning:
    batch_size: 4
""".strip(),
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)

    assert config.server.continual_learning.proxy_eval_max_samples == 0


def test_load_runtime_config_proxy_eval_threshold_candidates_defaults_to_none(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
client:
  server_ip: 10.0.0.1:50051
server:
  listen_address: "[::]:50051"
  continual_learning:
    batch_size: 4
""".strip(),
        encoding="utf-8",
    )

    config = load_runtime_config(config_path)

    assert config.server.continual_learning.proxy_eval_threshold_candidates is None


def test_load_runtime_config_rejects_out_of_range_threshold_candidate(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
client:
  server_ip: 10.0.0.1:50051
server:
  listen_address: "[::]:50051"
  continual_learning:
    batch_size: 4
    proxy_eval_threshold_candidates: [0.1, 1.5]
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be within \\[0, 1\\]"):
        load_runtime_config(config_path)
