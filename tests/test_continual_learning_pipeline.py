import io
import json
import zipfile

import torch

from edge.sample_store import EdgeSampleStore, HIGH_CONFIDENCE, LOW_CONFIDENCE
from edge.transmit import pack_continual_learning_bundle
from model_management.fixed_split import (
    SplitConstraints,
    SplitPlan,
    load_or_compute_fixed_split_plan,
)
from model_management.payload import SplitPayload
from model_management.continual_learning_bundle import prepare_split_training_cache


def _dummy_plan() -> SplitPlan:
    return SplitPlan(
        split_config_id="plan-1",
        model_name="dummy-model",
        candidate_id="candidate-1",
        split_index=3,
        split_label="layer3",
        boundary_tensor_labels=["layer3"],
        payload_bytes=128,
        privacy_metric=0.4,
        privacy_risk=0.6,
        layer_freezing_ratio=0.5,
        constraints={
            "privacy_metric_lower_bound": 0.0,
            "max_layer_freezing_ratio": 1.0,
            "validate_candidates": True,
            "max_candidates": 24,
            "max_boundary_count": 8,
            "max_payload_bytes": 32 * 1024 * 1024,
        },
        trace_signature="sig",
    )


def _payload() -> SplitPayload:
    return SplitPayload.from_mapping({"payload": torch.ones(1, 2, 2)}, primary_label="payload")


def test_fixed_split_is_computed_once_and_reused(tmp_path, monkeypatch):
    calls = {"count": 0}
    dummy_plan = _dummy_plan()

    class DummySplitter:
        def __init__(self):
            self.graph = object()
            self.model = object()

        def enumerate_candidates(self, **kwargs):
            return []

    def _fake_compute(*args, **kwargs):
        calls["count"] += 1
        return dummy_plan

    monkeypatch.setattr("model_management.fixed_split._trace_signature", lambda splitter: "sig")
    monkeypatch.setattr("model_management.fixed_split.compute_fixed_split_for_model", _fake_compute)
    monkeypatch.setattr(
        "model_management.fixed_split.validate_split_plan",
        lambda splitter, plan: {"success": True, "validation_passed": True},
    )

    constraints = SplitConstraints()
    splitter = DummySplitter()
    cache_path = str(tmp_path / "fixed_split_plan.json")
    model = torch.nn.Linear(1, 1)

    first = load_or_compute_fixed_split_plan(
        model,
        constraints,
        sample_input=[torch.rand(1)],
        splitter=splitter,
        cache_path=cache_path,
        model_name="dummy-model",
    )
    second = load_or_compute_fixed_split_plan(
        model,
        constraints,
        sample_input=[torch.rand(1)],
        splitter=splitter,
        cache_path=cache_path,
        model_name="dummy-model",
    )

    assert calls["count"] == 1
    assert first.split_config_id == second.split_config_id


def test_high_confidence_sample_saves_feature_and_result_without_raw(tmp_path):
    store = EdgeSampleStore(str(tmp_path))
    record = store.store_sample(
        sample_id="high-1",
        frame_index=1,
        confidence=0.95,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        confidence_bucket=HIGH_CONFIDENCE,
        inference_result={"boxes": [[1, 2, 3, 4]], "labels": [1], "scores": [0.95]},
        intermediate=_payload(),
        raw_frame=None,
    )

    assert record.has_feature is True
    assert record.has_raw_sample is False
    assert (tmp_path / "features" / "high-1.pt").exists()
    assert (tmp_path / "results" / "high-1.json").exists()
    assert not (tmp_path / "raw" / "high-1.jpg").exists()


def test_low_confidence_sample_saves_feature_result_and_raw(tmp_path, sample_bgr_frame):
    store = EdgeSampleStore(str(tmp_path))
    record = store.store_sample(
        sample_id="low-1",
        frame_index=2,
        confidence=0.2,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        confidence_bucket=LOW_CONFIDENCE,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=_payload(),
        raw_frame=sample_bgr_frame,
    )

    assert record.has_feature is True
    assert record.has_raw_sample is True
    assert (tmp_path / "features" / "low-1.pt").exists()
    assert (tmp_path / "results" / "low-1.json").exists()
    assert (tmp_path / "raw" / "low-1.jpg").exists()


def test_bundle_always_includes_high_conf_features_and_results(tmp_path, sample_bgr_frame):
    store = EdgeSampleStore(str(tmp_path))
    high = store.store_sample(
        sample_id="high-1",
        frame_index=1,
        confidence=0.9,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        confidence_bucket=HIGH_CONFIDENCE,
        inference_result={"boxes": [[1, 2, 3, 4]], "labels": [1], "scores": [0.9]},
        intermediate=_payload(),
    )
    low = store.store_sample(
        sample_id="low-1",
        frame_index=2,
        confidence=0.2,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        confidence_bucket=LOW_CONFIDENCE,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=_payload(),
        raw_frame=sample_bgr_frame,
    )

    payload_zip, manifest = pack_continual_learning_bundle(
        store,
        edge_id=1,
        send_low_conf_features=False,
        split_plan=_dummy_plan(),
        model_id="model-a",
        model_version="0",
    )
    with zipfile.ZipFile(io.BytesIO(payload_zip), "r") as zf:
        names = set(zf.namelist())
        bundle_manifest = json.loads(zf.read("bundle_manifest.json"))

    sample_map = {sample["sample_id"]: sample for sample in bundle_manifest["samples"]}
    assert high.feature_relpath in names
    assert high.result_relpath in names
    assert low.result_relpath in names
    assert low.raw_relpath in names
    assert low.feature_relpath not in names
    assert sample_map["low-1"]["feature_relpath"] is None
    assert manifest["training_mode"]["low_confidence_mode"] == "raw-only"


def test_bundle_includes_low_conf_features_when_decision_requests_them(tmp_path, sample_bgr_frame):
    store = EdgeSampleStore(str(tmp_path))
    low = store.store_sample(
        sample_id="low-1",
        frame_index=2,
        confidence=0.2,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        confidence_bucket=LOW_CONFIDENCE,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=_payload(),
        raw_frame=sample_bgr_frame,
    )

    payload_zip, manifest = pack_continual_learning_bundle(
        store,
        edge_id=1,
        send_low_conf_features=True,
        split_plan=_dummy_plan(),
        model_id="model-a",
        model_version="0",
    )
    with zipfile.ZipFile(io.BytesIO(payload_zip), "r") as zf:
        names = set(zf.namelist())

    assert low.feature_relpath in names
    assert low.raw_relpath in names
    assert manifest["training_mode"]["low_confidence_mode"] == "raw+feature"


def test_server_reconstructs_low_conf_features_only_in_raw_only_mode(tmp_path, sample_bgr_frame):
    store = EdgeSampleStore(str(tmp_path / "store"))
    store.store_sample(
        sample_id="low-1",
        frame_index=2,
        confidence=0.2,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        confidence_bucket=LOW_CONFIDENCE,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=_payload(),
        raw_frame=sample_bgr_frame,
    )

    provider_calls = {"count": 0}

    def _provider(raw_path, sample, manifest):
        provider_calls["count"] += 1
        return _payload()

    raw_only_zip, _ = pack_continual_learning_bundle(
        store,
        edge_id=1,
        send_low_conf_features=False,
        split_plan=_dummy_plan(),
        model_id="model-a",
        model_version="0",
    )
    raw_only_root = tmp_path / "raw_only_bundle"
    with zipfile.ZipFile(io.BytesIO(raw_only_zip), "r") as zf:
        zf.extractall(raw_only_root)
    prepare_split_training_cache(
        str(raw_only_root),
        str(tmp_path / "raw_only_cache"),
        feature_provider=_provider,
    )
    assert provider_calls["count"] == 1

    provider_calls["count"] = 0
    raw_plus_zip, _ = pack_continual_learning_bundle(
        store,
        edge_id=1,
        send_low_conf_features=True,
        split_plan=_dummy_plan(),
        model_id="model-a",
        model_version="0",
    )
    raw_plus_root = tmp_path / "raw_plus_bundle"
    with zipfile.ZipFile(io.BytesIO(raw_plus_zip), "r") as zf:
        zf.extractall(raw_plus_root)
    prepare_split_training_cache(
        str(raw_plus_root),
        str(tmp_path / "raw_plus_cache"),
        feature_provider=_provider,
    )
    assert provider_calls["count"] == 0
