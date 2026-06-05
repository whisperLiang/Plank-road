from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cloud.orchestration.sample_stage import CanonicalSampleStage
from cloud.training import FixedSplitProxyEvaluator


class FakeLargeOD:
    def __init__(self) -> None:
        self.model_name = "fake_teacher"
        self.model = SimpleNamespace(label_schema="coco_91", class_names=["person"])

    def large_inference_batch(self, images, threshold=None):
        del threshold
        return [([[1, 2, 3, 4]], [1], [0.9]) for _image in images]

    def large_inference(self, image, threshold=None):
        del image, threshold
        return ([[1, 2, 3, 4]], [1], [0.9])


def _config(tmp_path: Path) -> SimpleNamespace:
    teacher_annotation = SimpleNamespace(
        async_enabled=False,
        cache_enabled=True,
        wait_timeout_sec=0.0,
        worker_batch_size=4,
        worker_max_queue_size=64,
        worker_max_retries=0,
        oom_retry_enabled=True,
        min_worker_batch_size=1,
        cache_root_dir=str(tmp_path / "teacher_cache"),
    )
    feature_cache = SimpleNamespace(
        shard_root_dir=str(tmp_path / "feature_shards"),
        view_root_dir=str(tmp_path / "training_views"),
        storage_format="safetensors_shard",
        accepted_storage_formats=["safetensors_shard", "npy_memmap_shard"],
        materialization_mode="direct_ref",
        view_source="canonical_active",
    )
    continual_learning = SimpleNamespace(
        num_epoch=1,
        max_concurrent_jobs=1,
        batch_size=2,
        trace_batch_size=2,
        feature_cache_mode="auto",
        feature_cache=feature_cache,
        teacher_annotation_threshold=0.5,
        teacher_batch_size=2,
        teacher_annotation=teacher_annotation,
        proxy_eval_interval_rounds=1,
        proxy_eval_patience=0,
        proxy_eval_min_delta=0.0,
        proxy_eval_max_samples=0,
        proxy_eval_frame_cache_enabled=True,
        split_learning_rate=1e-3,
        wrapper_fixed_split_learning_rate=3e-5,
        tinynext_fixed_split_learning_rate=1e-3,
        rfdetr_fixed_split_learning_rate=1e-4,
        tinynext_fixed_split_target_steps_per_round=4,
        yolo_fixed_split_target_steps_per_round=4,
        rfdetr_fixed_split_target_steps_per_round=4,
    )
    sample_pool = SimpleNamespace(
        enabled=True,
        root_dir=str(tmp_path / "pool"),
        staging_root=str(tmp_path / "staging"),
        split_contract_root=str(tmp_path / "contracts"),
        max_samples=32,
        shard_size=1,
        enable_timing_logs=False,
        enable_coordinate_debug=False,
    )
    return SimpleNamespace(
        edge_model_name="yolo26n",
        golden="fake_teacher",
        weights_path="",
        workspace_root=str(tmp_path / "workspace"),
        continual_learning=continual_learning,
        sample_pool=sample_pool,
        das=None,
        tinynext_input_size=320,
    )


def test_cloud_orchestrator_imports_initializes_and_alias_survives(tmp_path) -> None:
    from cloud.orchestrator import CloudContinualLearner, CloudFixedSplitOrchestrator

    learner = CloudFixedSplitOrchestrator(_config(tmp_path), FakeLargeOD())
    alias = CloudContinualLearner(_config(tmp_path / "alias"), FakeLargeOD())
    try:
        assert learner.max_concurrent_jobs == 1
        assert learner.settings.batch_size == 2
        assert Path(learner.weight_folder) == Path.cwd() / "model_management" / "models"
        assert isinstance(alias, CloudFixedSplitOrchestrator)
    finally:
        learner.close()
        alias.close()


def test_orchestrator_does_not_import_private_proxy_helpers() -> None:
    source = Path("cloud/orchestrator.py").read_text(encoding="utf-8")

    assert "from cloud.training." + "proxy_eval" not in source
    assert "import _" not in source


def test_fixed_split_pipeline_does_not_import_private_proxy_helpers() -> None:
    source = Path("cloud/orchestration/fixed_split_pipeline.py").read_text(
        encoding="utf-8",
    )

    assert "from cloud.training." + "proxy_eval import (" not in source
    assert "_evaluate_detection_proxy_map" not in source
    assert "_snapshot_model_state" not in source
    assert "_proxy_metrics_indicate_dead_detector" not in source
    assert "_fixed_split_proxy_rejection_reason" not in source
    assert "_load" + "_cached_split_batches" not in source


def test_fixed_split_dependencies_does_not_bridge_private_proxy_helpers() -> None:
    source = Path("cloud/orchestration/fixed_split_dependencies.py").read_text(
        encoding="utf-8",
    )

    assert "import cloud.training." + "proxy_eval as " + "_proxy" + "_eval" not in source
    assert "_proxy" + "_eval." not in source
    assert "from cloud.training." + "proxy_eval import " + "_" not in source


def test_sample_stage_preserves_canonical_rebuild_argument_order() -> None:
    calls = {}

    class FakePool:
        def rebuild_canonical_training_pool(self, **kwargs):
            calls.update(kwargs)
            return {"generation_commit": {"active": 0}}, []

    existing_active = [{"sample_id": "active"}]
    pending_high_quality = [{"sample_id": "pending"}]
    new_low_quality = [{"sample_id": "low"}]

    CanonicalSampleStage(FakePool()).rebuild(
        split_contract=object(),
        existing_active=existing_active,
        pending_high_quality=pending_high_quality,
        new_low_quality=new_low_quality,
    )

    assert calls["existing_active_samples"] is existing_active
    assert calls["pending_high_quality_samples"] is pending_high_quality
    assert calls["new_low_quality_samples"] is new_low_quality


def test_full_image_retrain_remains_rejected_public_stub(tmp_path) -> None:
    from cloud.orchestrator import CloudFixedSplitOrchestrator

    learner = CloudFixedSplitOrchestrator(_config(tmp_path), FakeLargeOD())
    try:
        success, model_data, message = learner.get_ground_truth_and_retrain(
            1,
            [1],
            str(tmp_path),
        )
    finally:
        learner.close()

    assert success is False
    assert model_data == ""
    assert "legacy full-image retrain has been removed" in message


def test_legacy_low_quality_bundle_manifest_is_rejected(tmp_path) -> None:
    from cloud.orchestrator import CloudFixedSplitOrchestrator

    bundle_dir = tmp_path / "legacy_bundle"
    bundle_dir.mkdir()
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps({"protocol_version": "legacy"}),
        encoding="utf-8",
    )
    learner = CloudFixedSplitOrchestrator(_config(tmp_path), FakeLargeOD())
    try:
        success, model_data, message = learner.get_ground_truth_and_fixed_split_retrain(
            1,
            str(bundle_dir),
        )
    finally:
        learner.close()

    assert success is False
    assert model_data == ""
    assert "trigger_manifest.json" in message
    assert "legacy bundle_manifest.json uploads are no longer supported" in message


def test_proxy_evaluator_rejects_and_rolls_back_bad_candidate() -> None:
    evaluator = FixedSplitProxyEvaluator(
        device=torch.device("cpu"),
        default_batch_size=2,
        max_samples=8,
    )
    model = torch.nn.Linear(1, 1)
    baseline_state = evaluator.snapshot_model_state(model)
    with torch.no_grad():
        model.weight.add_(10.0)

    decision = evaluator.decide(
        {"map": 0.5, "evaluated_samples": 4, "nonempty_predictions": 4},
        {"map": 0.4, "evaluated_samples": 4, "nonempty_predictions": 4},
    )

    assert decision.accepted is False
    assert "regressed" in str(decision.reason)
    assert evaluator.rollback_if_rejected(
        model,
        baseline_state=baseline_state,
        decision=decision,
    )
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor.cpu(), baseline_state[name])


def test_proxy_evaluator_dead_detector_is_rejected() -> None:
    evaluator = FixedSplitProxyEvaluator(
        device=torch.device("cpu"),
        default_batch_size=2,
        max_samples=8,
    )

    decision = evaluator.decide(
        {"map": 0.5, "evaluated_samples": 4, "nonempty_predictions": 4},
        {
            "map": 0.0,
            "evaluated_samples": 4,
            "nonempty_predictions": 0,
            "total_prediction_boxes": 0,
        },
    )

    assert decision.accepted is False
    assert "no detections" in str(decision.reason)


def test_proxy_evaluator_accepts_non_regressing_metrics() -> None:
    evaluator = FixedSplitProxyEvaluator(
        device=torch.device("cpu"),
        default_batch_size=2,
        max_samples=8,
    )

    decision = evaluator.decide(
        {"map": 0.5, "evaluated_samples": 4, "nonempty_predictions": 3},
        {"map": 0.5, "evaluated_samples": 4, "nonempty_predictions": 3},
    )

    assert decision.accepted is True
    assert decision.reason is None
    assert decision.summary is not None


def test_tinynext_dead_subset_fast_path_skips_full_baseline_recheck(monkeypatch) -> None:
    evaluator = FixedSplitProxyEvaluator(
        device=torch.device("cpu"),
        default_batch_size=2,
        max_samples=30,
    )
    gt_annotations = {
        f"sample-{index:02d}": {"boxes": [[1, 2, 3, 4]], "labels": [1]}
        for index in range(30)
    }

    def fake_calibrate(*args, **kwargs):
        del args, kwargs
        return (
            {
                "map": 0.0,
                "evaluated_samples": 24,
                "nonempty_predictions": 0,
                "total_prediction_boxes": 0,
            },
            0.5,
            0.4,
        )

    monkeypatch.setattr(
        "cloud.training.proxy_eval._calibrate_tinynext_proxy_thresholds",
        fake_calibrate,
    )

    metrics = evaluator.evaluate_tinynext(
        torch.nn.Linear(1, 1),
        frame_dir="unused",
        gt_annotations=gt_annotations,
        model_name="tinynext",
        stage_label="baseline",
        allow_dead_baseline_fast_path=True,
    )

    assert metrics["full_proxy_evaluation_skipped"] == 1
    assert metrics["subset_proxy_sample_count"] == 24
    assert metrics["full_proxy_sample_count"] == 30


def test_sample_stage_rebuild_uses_three_way_canonical_merge_order() -> None:
    class FakePool:
        def __init__(self) -> None:
            self.kwargs = None

        def rebuild_canonical_training_pool(self, **kwargs):
            self.kwargs = kwargs
            return {"validation": {"accepted_low_quality": 1}}, ["kept"]

    pool = FakePool()
    existing_active = [{"sample_id": "active"}]
    pending_high_quality = [{"sample_id": "pending"}]
    new_low_quality = [{"sample_id": "low"}]
    contract = object()

    result = CanonicalSampleStage(pool).rebuild(
        split_contract=contract,
        existing_active=existing_active,
        pending_high_quality=pending_high_quality,
        new_low_quality=new_low_quality,
    )

    assert pool.kwargs == {
        "split_contract": contract,
        "existing_active_samples": existing_active,
        "pending_high_quality_samples": pending_high_quality,
        "new_low_quality_samples": new_low_quality,
    }
    assert result.rebuild_stats["validation"]["accepted_low_quality"] == 1
    assert result.kept_records == ["kept"]
