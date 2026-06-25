from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from baselines.distributed.cloud_controller import DistributedBaselineController
from baselines.distributed.messages import BaselineFramePayload
from baselines.method_factory import create_policy, registered_methods
from baselines.runtime import BaselineEdgeAdapter, stable_window_id
from common.experiment_results import collect_edge_artifacts
from config.baseline import PLANK_ROAD_BASELINE_ERROR
from config.runtime import RuntimeConfig, load_runtime_config
from edge_client import (
    _configure_baseline_client_runtime,
    _experiment_result_upload_enabled,
    _prepare_experiment_run_dir,
    _resolve_baseline_run_id,
    _upload_experiment_run_artifacts_if_enabled,
    _validate_startup_config,
)
from grpc_server import message_transmission_pb2

PROJECT_ROOT = Path(__file__).resolve().parents[1]




def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        source=SimpleNamespace(video_path=str(PROJECT_ROOT / "video_data" / "road.mp4")),
        diff_flag=True,
        diff_thresh=0.0004,
        feature="edge",
        lightweight="tiny",
        weights_path="",
        tinynext_input_size=None,
        baseline=SimpleNamespace(
            results_root=str(tmp_path / "results"),
            pure_edge_local_updating=SimpleNamespace(
                label_source="pseudo_label",
                local_metrics=True,
                upload_metrics_to_cloud=False,
                upload_frames_to_cloud=False,
                use_cloud_teacher=False,
            ),
            accuracy_trigger_cloud_retraining=SimpleNamespace(
                reuse_plank_road_frame_filter=True,
                upload_keyframes_only=True,
                trigger_on_cloud_comparison=True,
                training_strategy="freeze",
                trainable_param_ratio=0.3,
                training_failure_backoff_sec=30.0,
                return_model_update=True,
            ),
            edge=SimpleNamespace(split_runtime_policy="disabled"),
            training=SimpleNamespace(
                batch_size=2,
                num_epoch=1,
                learning_rate=1e-3,
                min_training_samples=1,
                training_window_size=8,
            ),
        ),
    )


class RecordingTransport:
    def __init__(self) -> None:
        self.uploaded: list[BaselineFramePayload] = []
        self.uploaded_windows: list[object] = []
        self.inference_requests: list[int] = []
        self.training_requests: list[dict[str, object]] = []
        self.registered: BaselineFramePayload | None = None

    def close(self) -> None:
        pass

    def register_edge(self, *, payload: BaselineFramePayload) -> None:
        self.registered = payload

    def upload_frame(self, payload: BaselineFramePayload) -> None:
        self.uploaded.append(payload)

    def upload_accuracy_trigger_window(self, payload) -> None:
        self.uploaded_windows.append(payload)

    def get_training_job_status(self, *, edge_id: int, job_id: str):
        del edge_id, job_id
        return message_transmission_pb2.TrainingJobStatusReply(
            found=True,
            status="RUNNING",
            result_available=False,
        )






class FailingTrainingTransport(RecordingTransport):
    def get_training_job_status(self, *, edge_id: int, job_id: str):
        del edge_id, job_id
        return message_transmission_pb2.TrainingJobStatusReply(
            found=True,
            status="FAILED",
            result_available=False,
            message="boom",
        )








class AccuracyCommandTrainingTransport(RecordingTransport):
    def __init__(self) -> None:
        super().__init__()
        self.commands: list[dict[str, object]] = []
        self.acked: list[dict[str, object]] = []

    def poll_command(self, *, run_id: str, baseline_method: str, edge_id: int):
        del run_id, baseline_method, edge_id
        return list(self.commands)

    def ack_command(
        self,
        *,
        run_id: str,
        baseline_method: str,
        edge_id: int,
        command_id: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.acked.append(
            {
                "run_id": run_id,
                "baseline_method": baseline_method,
                "edge_id": int(edge_id),
                "command_id": command_id,
                "metadata": dict(metadata or {}),
            }
        )
        self.commands = [item for item in self.commands if item.get("command_id") != command_id]

    def get_training_job_status(self, *, edge_id: int, job_id: str):
        return message_transmission_pb2.TrainingJobStatusReply(
            found=True,
            job_id=str(job_id),
            edge_id=int(edge_id),
            status="SUCCEEDED",
            result_available=True,
            result_model_version="1",
        )

    def download_trained_model(self, *, edge_id: int, job_id: str):
        del edge_id
        return message_transmission_pb2.DownloadTrainedModelReply(
            success=True,
            job_id=str(job_id),
            status="SUCCEEDED",
            model_data="model-update",
            result_model_version="1",
        )


class FakeTask:
    def __init__(self, *, source: str, model_version: str = "0") -> None:
        self.result_source = source
        self.timing_ms = {"inference": 3.5}
        self.inference_artifacts = {
            "boxes": [[1, 2, 3, 4]],
            "labels": [5],
            "scores": [0.9],
            "confidence": 0.9,
            "entropy": 0.25,
            "model_version": model_version,
            "result_source": source,
        }


class FakeEdge:
    model_version = "0"

    def apply_model_update(self, *args, **kwargs) -> None:
        del args, kwargs


class RecordingEdge(FakeEdge):
    def __init__(self) -> None:
        self.model_version = "0"
        self.updates: list[dict[str, object]] = []

    def apply_model_update(self, model_data, **kwargs) -> None:
        self.updates.append({"model_data": model_data, **kwargs})
        self.model_version = str(kwargs.get("result_model_version") or "1")




def test_legacy_baseline_files_are_removed() -> None:
    removed = [
        "launch_multi" + "_edge.py",
        "multi" + "_edge" + "_runner.py",
        "tools/run_baselines" + "_real.py",
        "tools/baselines" + "_real_common.py",
        "config/experiment.py",
        "config/experiment.yaml",
        "config/baselines_real_advantage.yaml",
        "baselines/plank_road" + "_multi_device.py",
        "multi" + "_edge",
        "baselines/distributed/edge_runtime.py",
    ]
    for relpath in removed:
        assert not (PROJECT_ROOT / relpath).exists(), relpath


def test_only_two_baseline_methods_are_registered() -> None:
    assert registered_methods() == (
        "pure_edge_local_updating",
        "accuracy_trigger_cloud_retraining",
    )
    with pytest.raises(ValueError, match="not a baseline method"):
        create_policy("plank_road" + "_multi_device")
    assert str(PLANK_ROAD_BASELINE_ERROR).startswith("plank_road" + "_multi_device")


def test_removed_ekya_baseline_config_section_is_rejected(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
baseline:
  ekya_style_centralized_scheduling:
    cloud_inference: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="removed and no longer supported"):
        load_runtime_config(path)


def test_baseline_defaults_to_freeze_and_disabled_edge_split_runtime() -> None:
    config = RuntimeConfig()

    assert config.baseline.accuracy_trigger_cloud_retraining.training_strategy == "freeze"
    assert (
        config.baseline.accuracy_trigger_cloud_retraining.trainable_param_ratio
        == pytest.approx(0.3)
    )
    assert config.baseline.accuracy_trigger_cloud_retraining.training_failure_backoff_sec == 30.0
    assert config.baseline.accuracy_trigger_cloud_retraining.trigger_window_size == 8
    assert config.baseline.accuracy_trigger_cloud_retraining.min_history_windows == 2
    assert config.baseline.accuracy_trigger_cloud_retraining.accuracy_drop_sigma == pytest.approx(
        1.0
    )
    assert config.baseline.accuracy_trigger_cloud_retraining.history_decay == pytest.approx(0.9)
    assert config.baseline.accuracy_trigger_cloud_retraining.metric == "teacher_f1"
    assert config.baseline.accuracy_trigger_cloud_retraining.agreement_iou_threshold == (
        pytest.approx(0.5)
    )
    assert config.baseline.accuracy_trigger_cloud_retraining.agreement_score_threshold == (
        pytest.approx(0.0)
    )
    assert (
        config.baseline.accuracy_trigger_cloud_retraining.agreement_empty_empty_policy
        == "exclude"
    )
    assert config.baseline.accuracy_trigger_cloud_retraining.absolute_accuracy_floor == (
        pytest.approx(0.6)
    )
    assert config.baseline.edge.split_runtime_policy == "disabled"






@pytest.mark.parametrize(
    "yaml_body",
    [
        """
sample_pool:
  max_samples: 8
baseline:
  training:
    training_window_size: 9
  accuracy_trigger_cloud_retraining:
    trigger_window_size: 8
""",
        """
sample_pool:
  max_samples: 8
baseline:
  training:
    training_window_size: 8
  accuracy_trigger_cloud_retraining:
    trigger_window_size: 9
""",
    ],
)
def test_sample_pool_capacity_must_cover_baseline_windows(tmp_path, yaml_body: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml_body, encoding="utf-8")

    with pytest.raises(ValueError, match="sample_pool.max_samples"):
        load_runtime_config(path)


def test_accuracy_trigger_agreement_policy_is_validated(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
baseline:
  accuracy_trigger_cloud_retraining:
    agreement_empty_empty_policy: background_bonus
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="agreement_empty_empty_policy"):
        load_runtime_config(path)


def test_default_baseline_edge_disables_main_cl_side_effects(tmp_path) -> None:
    config = _config(tmp_path)
    config.retrain = SimpleNamespace(flag=True, cache_path=str(tmp_path / "cache"))
    config.resource_aware_trigger = SimpleNamespace(enabled=True)
    config.sample_pool = SimpleNamespace(enabled=True)
    config.split_learning = SimpleNamespace(enabled=True)

    policy = _configure_baseline_client_runtime(config, config.baseline)

    assert policy == "disabled"
    assert config.retrain.flag is False
    assert config.resource_aware_trigger.enabled is False
    assert config.sample_pool.enabled is False
    assert config.split_learning.enabled is False


def test_baseline_edge_rejects_replay_only_runtime_policy(tmp_path) -> None:
    config = _config(tmp_path)
    config.baseline.edge.split_runtime_policy = "replay_only"

    with pytest.raises(ValueError, match="split_runtime_policy must be disabled"):
        _configure_baseline_client_runtime(config, config.baseline)


def test_pure_edge_adapter_uses_shared_artifacts_without_cloud(tmp_path) -> None:
    adapter = BaselineEdgeAdapter(
        config=_config(tmp_path),
        baseline_method="pure_edge_local_updating",
        run_id="pure-run",
        edge_id=1,
        transport=None,
    )
    try:
        adapter.before_video_start(FakeEdge())
        adapter.on_sampled_inference_result(
            frame=np.zeros((8, 8, 3), dtype=np.uint8),
            frame_index=1,
            task=FakeTask(source="inference"),
            detection_boxes=[[9, 9, 10, 10]],
            detection_class=[8],
            detection_score=[0.4],
            latency_ms=1.0,
        )
        assert adapter.transport is None
        assert adapter.metrics_path.exists()
        assert "frame_decision" in adapter.metrics_path.read_text(encoding="utf-8")
    finally:
        adapter.close()


def test_pure_edge_startup_validation_can_skip_cloud_address(tmp_path) -> None:
    config = _config(tmp_path)
    config.edge_id = 1
    config.server_ip = ""
    config.retrain = SimpleNamespace(cache_path=str(tmp_path / "cache"))

    _validate_startup_config(config, require_server_ip=False)
    with pytest.raises(ValueError, match="server_ip"):
        _validate_startup_config(config, require_server_ip=True)


def test_cloud_backed_baselines_require_explicit_run_id() -> None:
    with pytest.raises(ValueError, match="--run_id is required"):
        _resolve_baseline_run_id("accuracy_trigger_cloud_retraining", None)
    assert _resolve_baseline_run_id("accuracy_trigger_cloud_retraining", "run-a") == "run-a"
    assert _resolve_baseline_run_id("pure_edge_local_updating", None) is None


def test_prepare_experiment_run_dir_overwrites_existing_enabled_run(tmp_path) -> None:
    run_dir = tmp_path / "comparison" / "accuracy_trigger_cloud_retraining" / "edge_1" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.jsonl").write_text("old\n", encoding="utf-8")
    nested = run_dir / "nested"
    nested.mkdir()
    (nested / "edge_summary.json").write_text("old\n", encoding="utf-8")

    _prepare_experiment_run_dir(run_dir, enabled=True)

    assert run_dir.is_dir()
    assert list(run_dir.iterdir()) == []

    (run_dir / "keep.jsonl").write_text("kept\n", encoding="utf-8")
    _prepare_experiment_run_dir(run_dir, enabled=False)
    assert (run_dir / "keep.jsonl").read_text(encoding="utf-8") == "kept\n"


def test_pure_edge_shutdown_upload_disabled_but_local_artifacts_collected(tmp_path) -> None:
    run_dir = tmp_path / "pure-edge-run"
    run_dir.mkdir()
    inference_path = run_dir / "latest_inference_results.jsonl"
    metrics_path = run_dir / "metrics.jsonl"
    inference_path.write_text('{"frame_index": 1}\n', encoding="utf-8")
    metrics_path.write_text('{"event": "surgeon_tta_done"}\n', encoding="utf-8")
    (run_dir / "edge_summary.json").write_text(
        '{"method": "pure_edge_local_updating"}\n',
        encoding="utf-8",
    )
    experiment_results = SimpleNamespace(
        upload_to_cloud=True,
        upload_on_shutdown=True,
        include_inference_results=True,
        include_baseline_metrics=True,
        include_edge_summary=True,
        include_trigger_manifest=False,
        include_runtime_logs=False,
        max_artifact_bytes=1024 * 1024,
    )

    artifacts = collect_edge_artifacts(
        method="pure_edge_local_updating",
        run_id="pure-run",
        edge_id=1,
        comparison_id="comparison",
        config=experiment_results,
        inference_result_path=inference_path,
        baseline_metrics_path=metrics_path,
        cache_path=tmp_path / "cache",
    )

    assert "latest_inference_results.jsonl" in artifacts
    assert "metrics.jsonl" in artifacts
    assert "edge_summary.json" in artifacts
    assert not _experiment_result_upload_enabled(
        mode="baseline",
        baseline_method="pure_edge_local_updating",
        experiment_results=experiment_results,
        disable_experiment_result_upload=False,
    )


@pytest.mark.parametrize(
    ("mode", "baseline_method", "disabled", "expected"),
    [
        ("main", None, False, True),
        ("baseline", "accuracy_trigger_cloud_retraining", False, True),
        ("main", None, True, False),
        ("baseline", "accuracy_trigger_cloud_retraining", True, False),
        ("baseline", "pure_edge_local_updating", False, False),
    ],
)
def test_shutdown_experiment_upload_enablement_preserves_non_pure_edge_modes(
    mode: str,
    baseline_method: str | None,
    disabled: bool,
    expected: bool,
) -> None:
    experiment_results = SimpleNamespace(upload_to_cloud=True, upload_on_shutdown=True)

    assert (
        _experiment_result_upload_enabled(
            mode=mode,
            baseline_method=baseline_method,
            experiment_results=experiment_results,
            disable_experiment_result_upload=disabled,
        )
        is expected
    )


def test_shutdown_upload_helper_does_not_call_uploader_for_pure_edge() -> None:
    calls: list[dict[str, object]] = []

    class FakeUploader:
        def __init__(self, server_ip: str, enabled: bool) -> None:
            calls.append({"event": "init", "server_ip": server_ip, "enabled": enabled})

        def upload_run_artifacts(self, **kwargs) -> bool:
            calls.append({"event": "upload", **kwargs})
            return True

    uploaded = _upload_experiment_run_artifacts_if_enabled(
        server_ip="127.0.0.1:1",
        mode="baseline",
        baseline_method="pure_edge_local_updating",
        experiment_results=SimpleNamespace(upload_to_cloud=True, upload_on_shutdown=True),
        disable_experiment_result_upload=False,
        comparison_id="comparison",
        run_id="pure-run",
        method="pure_edge_local_updating",
        edge_id=1,
        artifacts={"metrics.jsonl": "{}\n"},
        uploader_cls=FakeUploader,
    )

    assert uploaded is False
    assert calls == []


def test_shutdown_upload_helper_still_calls_uploader_for_accuracy_trigger() -> None:
    calls: list[dict[str, object]] = []

    class FakeUploader:
        def __init__(self, server_ip: str, enabled: bool) -> None:
            calls.append({"event": "init", "server_ip": server_ip, "enabled": enabled})

        def upload_run_artifacts(self, **kwargs) -> bool:
            calls.append({"event": "upload", **kwargs})
            return True

    uploaded = _upload_experiment_run_artifacts_if_enabled(
        server_ip="127.0.0.1:1",
        mode="baseline",
        baseline_method="accuracy_trigger_cloud_retraining",
        experiment_results=SimpleNamespace(upload_to_cloud=True, upload_on_shutdown=True),
        disable_experiment_result_upload=False,
        comparison_id="comparison",
        run_id="acc-run",
        method="accuracy_trigger_cloud_retraining",
        edge_id=1,
        artifacts={"metrics.jsonl": "{}\n"},
        uploader_cls=FakeUploader,
    )

    assert uploaded is True
    assert calls[0] == {"event": "init", "server_ip": "127.0.0.1:1", "enabled": True}
    assert calls[1]["event"] == "upload"
    assert calls[1]["method"] == "accuracy_trigger_cloud_retraining"


def test_accuracy_adapter_uploads_keyframes_without_local_training_submit(tmp_path) -> None:
    transport = RecordingTransport()
    adapter = BaselineEdgeAdapter(
        config=_config(tmp_path),
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="acc-run",
        edge_id=2,
        server_ip="127.0.0.1:1",
        transport=transport,
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    try:
        adapter.before_video_start(FakeEdge())
        adapter.on_sampled_inference_result(
            frame=frame,
            frame_index=1,
            task=FakeTask(source="cached"),
            detection_boxes=[],
            detection_class=[],
            detection_score=[],
            latency_ms=1.0,
        )
        adapter.on_sampled_inference_result(
            frame=frame,
            frame_index=2,
            task=FakeTask(source="inference", model_version="0"),
            detection_boxes=[],
            detection_class=[],
            detection_score=[],
            latency_ms=1.0,
        )
        time.sleep(0.1)

        assert transport.uploaded == []
        assert transport.uploaded_windows == []
        adapter.close()
        assert len(transport.uploaded_windows) == 1
        window = transport.uploaded_windows[0]
        assert [sample.frame_id for sample in window.selected_samples] == [2]
        sample = window.selected_samples[0]
        assert sample.edge_prediction["boxes"] == [[1, 2, 3, 4]]
        assert sample.confidence == pytest.approx(0.9)
        assert sample.entropy == pytest.approx(0.25)
        assert sample.quality_metadata["training_strategy"] == "freeze"
        assert transport.training_requests == []
        metric_rows = [
            json.loads(line)
            for line in Path(adapter.metrics_path).read_text(encoding="utf-8").splitlines()
        ]
        upload = next(row for row in metric_rows if row["event"] == "bundle_upload_done")
        assert upload["raw_frame_bytes"] > 0
        assert upload["feature_bytes"] == 0
        assert upload["prediction_metadata_bytes"] > 0
        assert upload["total_upload_bytes"] >= upload["raw_frame_bytes"]
        assert upload["upload_ms"] >= 0
    finally:
        adapter.close()


def test_accuracy_adapter_flushes_buffer_on_model_version_change(tmp_path) -> None:
    config = _config(tmp_path)
    config.baseline.accuracy_trigger_cloud_retraining.trigger_window_size = 8
    transport = RecordingTransport()
    adapter = BaselineEdgeAdapter(
        config=config,
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="acc-run",
        edge_id=2,
        server_ip="127.0.0.1:1",
        transport=transport,
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    try:
        adapter.before_video_start(FakeEdge())
        adapter.on_sampled_inference_result(
            frame=frame,
            frame_index=1,
            task=FakeTask(source="inference", model_version="0"),
            detection_boxes=[],
            detection_class=[],
            detection_score=[],
            latency_ms=1.0,
        )
        adapter.on_sampled_inference_result(
            frame=frame,
            frame_index=2,
            task=FakeTask(source="inference", model_version="1"),
            detection_boxes=[],
            detection_class=[],
            detection_score=[],
            latency_ms=1.0,
        )

        _wait_until(lambda: len(transport.uploaded_windows) == 1)
        first = transport.uploaded_windows[0]
        assert first.model_version == "0"
        assert [sample.frame_id for sample in first.selected_samples] == [1]

        adapter.close()
        assert len(transport.uploaded_windows) == 2
        second = transport.uploaded_windows[1]
        assert second.model_version == "1"
        assert [sample.frame_id for sample in second.selected_samples] == [2]
    finally:
        adapter.close()


def test_accuracy_adapter_close_drains_queued_windows_before_partial_flush(tmp_path) -> None:
    config = _config(tmp_path)
    config.baseline.accuracy_trigger_cloud_retraining.trigger_window_size = 2
    transport = RecordingTransport()
    adapter = BaselineEdgeAdapter(
        config=config,
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="acc-run",
        edge_id=2,
        server_ip="127.0.0.1:1",
        transport=transport,
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    try:
        adapter.before_video_start(FakeEdge())
        for frame_index in (1, 2, 3):
            adapter.on_sampled_inference_result(
                frame=frame,
                frame_index=frame_index,
                task=FakeTask(source="inference", model_version="0"),
                detection_boxes=[],
                detection_class=[],
                detection_score=[],
                latency_ms=1.0,
            )

        adapter.close()

        assert [
            [sample.frame_id for sample in window.selected_samples]
            for window in transport.uploaded_windows
        ] == [[1, 2], [3]]
    finally:
        adapter.close()


def test_accuracy_adapter_never_enters_local_training_backoff(tmp_path) -> None:
    config = _config(tmp_path)
    config.baseline.training.training_window_size = 1
    config.baseline.accuracy_trigger_cloud_retraining.trigger_window_size = 1
    config.baseline.accuracy_trigger_cloud_retraining.training_failure_backoff_sec = 30.0
    transport = FailingTrainingTransport()
    adapter = BaselineEdgeAdapter(
        config=config,
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="acc-run",
        edge_id=2,
        server_ip="127.0.0.1:1",
        transport=transport,
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    try:
        adapter.before_video_start(FakeEdge())
        adapter.on_sampled_inference_result(
            frame=frame,
            frame_index=2,
            task=FakeTask(source="inference", model_version="0"),
            detection_boxes=[],
            detection_class=[],
            detection_score=[],
            latency_ms=1.0,
        )
        _wait_until(lambda: len(transport.uploaded_windows) == 1)
        time.sleep(0.1)
        assert transport.training_requests == []
        assert adapter._training_state.active_job is None

        adapter.on_sampled_inference_result(
            frame=frame,
            frame_index=2,
            task=FakeTask(source="inference", model_version="0"),
            detection_boxes=[],
            detection_class=[],
            detection_score=[],
            latency_ms=1.0,
        )
        time.sleep(0.1)

        assert transport.training_requests == []
    finally:
        adapter.close()














def test_accuracy_adapter_validates_cloud_command_and_acks_after_update(tmp_path) -> None:
    transport = AccuracyCommandTrainingTransport()
    edge = RecordingEdge()
    adapter = BaselineEdgeAdapter(
        config=_config(tmp_path),
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="acc-run",
        edge_id=2,
        server_ip="127.0.0.1:1",
        transport=transport,
    )
    try:
        adapter.before_video_start(edge)
        transport.commands = [
            {
                "type": "baseline_training_job_available",
                "command_id": "bad-cmd",
                "run_id": "wrong-run",
                "baseline_method": "accuracy_trigger_cloud_retraining",
                "edge_id": 2,
                "job_id": "job-bad",
                "window_id": "window-bad",
                "base_model_version": "0",
            }
        ]
        adapter._discover_cloud_scheduled_training()
        assert adapter._cloud_scheduled_active_job is None
        assert transport.acked == []

        transport.commands = [
            {
                "type": "baseline_training_job_available",
                "command_id": "cmd-1",
                "run_id": "acc-run",
                "baseline_method": "accuracy_trigger_cloud_retraining",
                "edge_id": 2,
                "job_id": "job-1",
                "window_id": "window-1",
                "base_model_version": "0",
            }
        ]
        adapter._last_command_poll_at = 0.0
        adapter._discover_cloud_scheduled_training()

        assert adapter._cloud_scheduled_active_job is not None
        assert adapter._cloud_scheduled_active_job.job_id == "job-1"
        assert transport.acked == []

        adapter._poll_cloud_scheduled_training()

        assert edge.updates
        assert edge.updates[0]["submitted_model_version"] == "0"
        assert edge.updates[0]["result_model_version"] == "1"
        assert transport.acked
        ack = transport.acked[0]
        assert ack["command_id"] == "cmd-1"
        update_ack = ack["metadata"]["accuracy_trigger_model_update_applied"]
        assert update_ack["job_id"] == "job-1"
        assert update_ack["base_model_version"] == "0"
        assert update_ack["result_model_version"] == "1"
    finally:
        adapter.close()


def test_stable_window_id_includes_strategy_ratio_and_sorts_frames() -> None:
    first = stable_window_id(
        run_id="run-a",
        baseline_method="accuracy_trigger_cloud_retraining",
        training_strategy="freeze",
        trainable_param_ratio=0.3,
        edge_id=1,
        model_version="0",
        frame_ids=[5, 1, 3],
    )
    reordered = stable_window_id(
        run_id="run-a",
        baseline_method="accuracy_trigger_cloud_retraining",
        training_strategy="freeze",
        trainable_param_ratio=0.3,
        edge_id=1,
        model_version="0",
        frame_ids=[1, 3, 5],
    )
    different_strategy = stable_window_id(
        run_id="run-a",
        baseline_method="accuracy_trigger_cloud_retraining",
        training_strategy="diagnostic",
        trainable_param_ratio=0.3,
        edge_id=1,
        model_version="0",
        frame_ids=[1, 3, 5],
    )
    different_ratio = stable_window_id(
        run_id="run-a",
        baseline_method="accuracy_trigger_cloud_retraining",
        training_strategy="freeze",
        trainable_param_ratio=0.5,
        edge_id=1,
        model_version="0",
        frame_ids=[1, 3, 5],
    )
    assert first == reordered
    assert first != different_strategy
    assert first != different_ratio


def test_cloud_controller_no_longer_exposes_training_state_machine() -> None:
    controller = DistributedBaselineController(
        baseline_method="accuracy_trigger_cloud_retraining",
        run_id="run-a",
        results_root="unused",
        sample_pool_max_samples=64,
        strict_run_id=False,
    )
    assert not hasattr(controller, "request_training")
    assert not hasattr(controller, "poll_training_job")
    assert not hasattr(controller, "download_model_update")






















def test_baseline_teacher_annotator_uses_configured_wait_timeout(tmp_path) -> None:
    from cloud_server import _build_baseline_teacher_annotator

    class FakeTeacherDetector:
        def large_inference(self, _frame, threshold=None):
            del threshold
            return [[1, 1, 4, 4]], [1], [0.9]

    config = SimpleNamespace(
        golden="rtdetr_x",
        continual_learning=SimpleNamespace(
            teacher_annotation_threshold=0.6,
            teacher_annotation=SimpleNamespace(
                cache_root_dir=str(tmp_path / "teacher-cache"),
                cache_enabled=True,
                wait_timeout_sec=0.25,
                worker_batch_size=1,
                worker_max_queue_size=8,
                worker_max_retries=0,
                oom_retry_enabled=True,
                min_worker_batch_size=1,
            ),
        ),
    )

    annotator = _build_baseline_teacher_annotator(
        config,
        FakeTeacherDetector(),
        heavy_gpu_lease=None,
        log_internal_ids=False,
    )
    try:
        assert annotator.wait_timeout_sec == pytest.approx(0.25)
    finally:
        annotator.close()


def test_baseline_teacher_annotator_rejects_disabled_cache(tmp_path) -> None:
    from cloud_server import _build_baseline_teacher_annotator

    config = SimpleNamespace(
        golden="rtdetr_x",
        continual_learning=SimpleNamespace(
            teacher_annotation_threshold=0.6,
            teacher_annotation=SimpleNamespace(
                cache_root_dir=str(tmp_path / "teacher-cache"),
                cache_enabled=False,
            ),
        ),
    )

    with pytest.raises(ValueError, match="cache_enabled must be true"):
        _build_baseline_teacher_annotator(
            config,
            object(),
            heavy_gpu_lease=None,
            log_internal_ids=False,
        )




































def test_upload_client_rejects_raw_freeze_strategy() -> None:
    from baselines.runtime.upload_client import validate_baseline_training_strategy

    with pytest.raises(ValueError, match="freeze"):
        validate_baseline_training_strategy("raw_freeze")


def test_production_code_has_no_old_baseline_training_fallbacks() -> None:
    roots = [
        PROJECT_ROOT / "cloud",
        PROJECT_ROOT / "baselines",
        PROJECT_ROOT / "config",
        PROJECT_ROOT / "grpc_server",
        PROJECT_ROOT / "cloud_server.py",
        PROJECT_ROOT / "edge_client.py",
    ]
    banned = [
        "raw_freeze",
        "CloudRawFreezeTrainingStrategy",
        "BaselineFrozenRatioTrainer",
        "TRAINING_JOB_TYPE_BASELINE_FROZEN_RATIO",
        "frozen_ratio_training",
        "BaselineEdgeRuntime",
        "CloudTorchLensFreezeTrainingStrategy",
        "CloudPending",
        "submit_training_bundle",
        "BaselineTrainingRequest",
        "RequestTraining",
        "PollTrainingJob",
        "DownloadModelUpdate",
        "_teacher_annotation_inference_adapter",
    ]
    for path in _iter_text_files(roots):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{token} found in {path.relative_to(PROJECT_ROOT)}"












def _iter_text_files(paths):
    for path in paths:
        if path.is_file():
            yield path
            continue
        for child in path.rglob("*.py"):
            if "__pycache__" not in child.parts:
                yield child


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition was not satisfied before timeout")
