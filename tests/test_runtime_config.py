from __future__ import annotations

import pytest

from config import load_runtime_config


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
    assert config.client.runtime_resize.enabled is False
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
    monkeypatch.setenv("PLANK_ROAD__CLIENT__RUNTIME_RESIZE__ENABLED", "true")
    monkeypatch.setenv("PLANK_ROAD__CLIENT__RUNTIME_RESIZE__HEIGHT", "320")
    monkeypatch.setenv("PLANK_ROAD__CLIENT__RUNTIME_RESIZE__WIDTH", "544")
    monkeypatch.setenv("PLANK_ROAD__SERVER__LISTEN_ADDRESS", "[::]:60001")

    config = load_runtime_config(config_path)

    assert config.client.server_ip == "127.0.0.1:60000"
    assert config.client.runtime_resize.enabled is True
    assert config.client.runtime_resize.height == 320
    assert config.client.runtime_resize.width == 544
    assert config.server.listen_address == "[::]:60001"


def test_load_runtime_config_rejects_invalid_resize_dimensions(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
client:
  runtime_resize:
    enabled: true
    height: 0
    width: 640
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="runtime_resize.height"):
        load_runtime_config(config_path)
