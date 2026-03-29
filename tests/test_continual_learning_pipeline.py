import io
import json
import zipfile
from collections import OrderedDict
from types import SimpleNamespace

import torch

from edge.sample_store import EdgeSampleStore, HIGH_CONFIDENCE, LOW_CONFIDENCE
from edge.transmit import pack_continual_learning_bundle
from model_management.fixed_split import (
    compute_fixed_split_for_model,
    SplitConstraints,
    SplitPlan,
    apply_split_plan,
    load_or_compute_fixed_split_plan,
)
from model_management.payload import SplitPayload
from model_management.continual_learning_bundle import prepare_split_training_cache
from model_management.split_candidate import SplitCandidate
from model_management.universal_model_split import load_split_feature_cache


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


def _planned_payload(plan: SplitPlan | None = None) -> SplitPayload:
    active_plan = plan or _dummy_plan()
    return SplitPayload(
        tensors=OrderedDict([("payload", torch.ones(1, 2, 2))]),
        candidate_id=active_plan.candidate_id,
        boundary_tensor_labels=list(active_plan.boundary_tensor_labels),
        primary_label="payload",
        split_index=active_plan.split_index,
        split_label=active_plan.split_label,
    )


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


def test_fixed_split_validates_only_lowest_payload_group_until_success():
    constraints = SplitConstraints()

    def _candidate(candidate_id: str, *, edge_nodes: list[str], payload_bytes: int, layer_index: int) -> SplitCandidate:
        return SplitCandidate(
            candidate_id=candidate_id,
            edge_nodes=edge_nodes,
            cloud_nodes=[label for label in ["n1", "n2", "n3"] if label not in edge_nodes],
            boundary_edges=[],
            boundary_tensor_labels=[edge_nodes[-1]],
            edge_input_labels=[],
            cloud_input_labels=[],
            cloud_output_labels=["n3"],
            estimated_edge_flops=1.0,
            estimated_cloud_flops=1.0,
            estimated_payload_bytes=payload_bytes,
            estimated_privacy_risk=1.0,
            estimated_latency=float(layer_index),
            is_trainable_tail=True,
            legacy_layer_index=layer_index,
            boundary_count=1,
        )

    candidates = [
        _candidate("candidate-low-invalid", edge_nodes=["n1"], payload_bytes=10, layer_index=1),
        _candidate("candidate-low-valid", edge_nodes=["n1", "n2"], payload_bytes=10, layer_index=2),
        _candidate("candidate-high-valid", edge_nodes=["n1"], payload_bytes=20, layer_index=3),
    ]

    reports = {
        "candidate-low-invalid": {
            "success": False,
            "edge_latency": 0.1,
            "cloud_latency": 0.1,
            "end_to_end_latency": 0.2,
            "tail_trainability": False,
            "stability_score": 0.0,
            "error": "mismatch",
        },
        "candidate-low-valid": {
            "success": True,
            "edge_latency": 0.1,
            "cloud_latency": 0.2,
            "end_to_end_latency": 0.3,
            "tail_trainability": True,
            "stability_score": 1.0,
            "error": None,
        },
        "candidate-high-valid": {
            "success": True,
            "edge_latency": 0.05,
            "cloud_latency": 0.05,
            "end_to_end_latency": 0.1,
            "tail_trainability": True,
            "stability_score": 1.0,
            "error": None,
        },
    }

    class DummyRuntime:
        def __init__(self):
            self.graph = SimpleNamespace(
                relevant_labels=["n1", "n2", "n3"],
                nodes={
                    "n1": SimpleNamespace(
                        has_trainable_params=True,
                        tensor_shape=(1, 4, 4),
                        containing_module="m.n1",
                    ),
                    "n2": SimpleNamespace(
                        has_trainable_params=True,
                        tensor_shape=(1, 4, 4),
                        containing_module="m.n2",
                    ),
                    "n3": SimpleNamespace(
                        has_trainable_params=True,
                        tensor_shape=(1, 4, 4),
                        containing_module="m.n3",
                    ),
                },
            )
            self.model = object()
            self.candidates = candidates
            self._candidate_enumeration_config = (
                constraints.max_candidates,
                constraints.max_boundary_count,
                constraints.max_payload_bytes,
            )
            self.validation_calls: list[str] = []

        def _ensure_ready(self):
            return self.model, self.graph

        def validate_candidate(self, candidate):
            self.validation_calls.append(candidate.candidate_id)
            return dict(reports[candidate.candidate_id])

    runtime = DummyRuntime()
    plan = compute_fixed_split_for_model(
        torch.nn.Linear(1, 1),
        constraints,
        sample_input=[torch.rand(1)],
        splitter=runtime,
        model_name="dummy-model",
    )

    assert plan.candidate_id == "candidate-low-valid"
    assert runtime.validation_calls == [
        "candidate-low-invalid",
        "candidate-low-valid",
    ]


def test_apply_split_plan_falls_back_from_boundary_labels_to_candidate_and_split_index():
    plan = SplitPlan(
        split_config_id="plan-1",
        model_name="dummy-model",
        candidate_id="candidate-2",
        split_index=7,
        split_label="layer7",
        boundary_tensor_labels=["missing-boundary"],
        payload_bytes=128,
        privacy_metric=0.4,
        privacy_risk=0.6,
        layer_freezing_ratio=0.5,
        constraints={},
        trace_signature="sig",
    )

    class CandidateRuntime:
        def __init__(self):
            self.calls = []

        def split(self, *, boundary_tensor_labels=None, candidate_id=None, layer_index=None):
            self.calls.append(
                {
                    "boundary_tensor_labels": boundary_tensor_labels,
                    "candidate_id": candidate_id,
                    "layer_index": layer_index,
                }
            )
            if boundary_tensor_labels is not None:
                raise KeyError("missing boundary labels")
            if candidate_id is not None:
                return "candidate-match"
            raise AssertionError("layer_index fallback should not be used here")

    runtime = CandidateRuntime()
    assert apply_split_plan(runtime, plan) == "candidate-match"
    assert runtime.calls == [
        {"boundary_tensor_labels": ["missing-boundary"], "candidate_id": None, "layer_index": None},
        {"boundary_tensor_labels": None, "candidate_id": "candidate-2", "layer_index": None},
    ]

    class LayerRuntime:
        def __init__(self):
            self.calls = []

        def split(self, *, boundary_tensor_labels=None, candidate_id=None, layer_index=None):
            self.calls.append(
                {
                    "boundary_tensor_labels": boundary_tensor_labels,
                    "candidate_id": candidate_id,
                    "layer_index": layer_index,
                }
            )
            if boundary_tensor_labels is not None or candidate_id is not None:
                raise KeyError("fallback")
            return "layer-match"

    runtime = LayerRuntime()
    assert apply_split_plan(runtime, plan) == "layer-match"
    assert runtime.calls == [
        {"boundary_tensor_labels": ["missing-boundary"], "candidate_id": None, "layer_index": None},
        {"boundary_tensor_labels": None, "candidate_id": "candidate-2", "layer_index": None},
        {"boundary_tensor_labels": None, "candidate_id": None, "layer_index": 7},
    ]


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


def test_bundle_filters_records_to_current_split_plan_and_model(tmp_path):
    store = EdgeSampleStore(str(tmp_path))
    keep = store.store_sample(
        sample_id="keep-1",
        frame_index=1,
        confidence=0.9,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        confidence_bucket=HIGH_CONFIDENCE,
        inference_result={"boxes": [[1, 2, 3, 4]], "labels": [1], "scores": [0.9]},
        intermediate=_payload(),
        drift_flag=True,
    )
    store.store_sample(
        sample_id="old-plan",
        frame_index=2,
        confidence=0.9,
        split_config_id="plan-old",
        model_id="model-a",
        model_version="0",
        confidence_bucket=HIGH_CONFIDENCE,
        inference_result={"boxes": [[1, 2, 3, 4]], "labels": [1], "scores": [0.9]},
        intermediate=_payload(),
        drift_flag=True,
    )
    store.store_sample(
        sample_id="old-model",
        frame_index=3,
        confidence=0.9,
        split_config_id="plan-1",
        model_id="model-b",
        model_version="0",
        confidence_bucket=HIGH_CONFIDENCE,
        inference_result={"boxes": [[1, 2, 3, 4]], "labels": [1], "scores": [0.9]},
        intermediate=_payload(),
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

    sample_ids = [sample["sample_id"] for sample in bundle_manifest["samples"]]
    assert sample_ids == ["keep-1"]
    assert bundle_manifest["drift_sample_ids"] == ["keep-1"]
    assert keep.feature_relpath in names
    assert "features/old-plan.pt" not in names
    assert "features/old-model.pt" not in names


def test_server_reconstructs_low_conf_features_only_in_raw_only_mode(tmp_path, sample_bgr_frame):
    store = EdgeSampleStore(str(tmp_path / "store"))
    plan = _dummy_plan()
    store.store_sample(
        sample_id="low-1",
        frame_index=2,
        confidence=0.2,
        split_config_id="plan-1",
        model_id="model-a",
        model_version="0",
        confidence_bucket=LOW_CONFIDENCE,
        inference_result={"boxes": [], "labels": [], "scores": []},
        intermediate=_planned_payload(plan),
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
        split_plan=plan,
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
        split_plan=plan,
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


def test_prepare_split_training_cache_accepts_matching_split_index_when_boundary_labels_drift(tmp_path):
    bundle_root = tmp_path / "bundle"
    (bundle_root / "features").mkdir(parents=True)
    (bundle_root / "results").mkdir()

    payload = SplitPayload(
        tensors=OrderedDict([("payload", torch.ones(1, 2, 2))]),
        candidate_id="candidate-old",
        boundary_tensor_labels=["old-boundary"],
        primary_label="payload",
        split_index=3,
        split_label="payload",
    )
    torch.save({"intermediate": payload}, bundle_root / "features" / "sample-1.pt")
    (bundle_root / "results" / "sample-1.json").write_text(
        json.dumps({"boxes": [], "labels": [], "scores": []}),
        encoding="utf-8",
    )

    manifest = {
        "protocol_version": "edge-cl-bundle.v1",
        "edge_id": 1,
        "model": {"model_id": "model-a", "model_version": "0"},
        "split_plan": {
            **_dummy_plan().to_dict(),
            "candidate_id": "candidate-new",
            "split_index": 3,
            "boundary_tensor_labels": ["new-boundary"],
        },
        "drift_sample_ids": [],
        "samples": [
            {
                "sample_id": "sample-1",
                "frame_index": 1,
                "confidence": 0.9,
                "confidence_bucket": HIGH_CONFIDENCE,
                "drift_flag": False,
                "feature_relpath": "features/sample-1.pt",
                "feature_bytes": (bundle_root / "features" / "sample-1.pt").stat().st_size,
                "result_relpath": "results/sample-1.json",
                "metadata_relpath": "metadata/sample-1.json",
                "raw_relpath": None,
                "raw_bytes": 0,
                "has_feature": True,
                "has_raw_sample": False,
                "split_config_id": "plan-1",
                "model_id": "model-a",
                "model_version": "0",
                "input_image_size": None,
                "input_tensor_shape": None,
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ],
    }
    (bundle_root / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    info = prepare_split_training_cache(
        str(bundle_root),
        str(tmp_path / "prepared_cache"),
    )

    assert info["all_sample_ids"] == ["sample-1"]


def test_prepare_split_training_cache_backfills_input_image_size_from_raw_sample(tmp_path, sample_bgr_frame):
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

    payload_zip, _ = pack_continual_learning_bundle(
        store,
        edge_id=1,
        send_low_conf_features=False,
        split_plan=_dummy_plan(),
        model_id="model-a",
        model_version="0",
    )
    bundle_root = tmp_path / "bundle"
    with zipfile.ZipFile(io.BytesIO(payload_zip), "r") as zf:
        zf.extractall(bundle_root)

    manifest = json.loads((bundle_root / "bundle_manifest.json").read_text())
    assert manifest["samples"][0]["input_image_size"] is None

    cache_root = tmp_path / "prepared_cache"
    prepare_split_training_cache(
        str(bundle_root),
        str(cache_root),
        feature_provider=lambda *_: _payload(),
    )

    record = load_split_feature_cache(str(cache_root), "low-1")
    assert record["input_image_size"] == list(sample_bgr_frame.shape[:2])


def test_prepare_split_training_cache_skips_incompatible_feature_only_samples(tmp_path):
    bundle_root = tmp_path / "bundle"
    (bundle_root / "features").mkdir(parents=True)
    (bundle_root / "results").mkdir()

    payload = SplitPayload(
        tensors=OrderedDict([("payload", torch.ones(1, 2, 2))]),
        candidate_id="candidate-old",
        boundary_tensor_labels=["old-boundary"],
        primary_label="payload",
        split_index=99,
        split_label="payload",
    )
    torch.save({"intermediate": payload}, bundle_root / "features" / "old-1.pt")
    (bundle_root / "results" / "old-1.json").write_text(
        json.dumps({"boxes": [], "labels": [], "scores": []}),
        encoding="utf-8",
    )

    manifest = {
        "protocol_version": "edge-cl-bundle.v1",
        "edge_id": 1,
        "model": {"model_id": "model-a", "model_version": "0"},
        "split_plan": _dummy_plan().to_dict(),
        "drift_sample_ids": [],
        "samples": [
            {
                "sample_id": "old-1",
                "frame_index": 1,
                "confidence": 0.9,
                "confidence_bucket": HIGH_CONFIDENCE,
                "drift_flag": False,
                "feature_relpath": "features/old-1.pt",
                "feature_bytes": (bundle_root / "features" / "old-1.pt").stat().st_size,
                "result_relpath": "results/old-1.json",
                "metadata_relpath": "metadata/old-1.json",
                "raw_relpath": None,
                "raw_bytes": 0,
                "has_feature": True,
                "has_raw_sample": False,
                "split_config_id": "plan-old",
                "model_id": "model-a",
                "model_version": "0",
                "input_image_size": None,
                "input_tensor_shape": None,
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ],
    }
    (bundle_root / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    info = prepare_split_training_cache(
        str(bundle_root),
        str(tmp_path / "prepared_cache"),
    )

    assert info["all_sample_ids"] == []
    assert info["drift_sample_ids"] == []
