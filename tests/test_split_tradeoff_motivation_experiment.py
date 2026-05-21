"""
Tests for Split Tradeoff Motivation Experiment
================================================

Tests core functionality including:
- Privacy leakage calculations
- Payload ratio calculations
- Pareto frontier computation
- CSV/JSON serialization
- Validation error handling
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tools.run_split_tradeoff_motivation_experiment import (
    CandidateRecord,
    ExperimentMetadata,
    ModelExperimentResult,
    ModelSummary,
    SplitCandidate,
    compute_raw_input_size_bytes,
    compute_pareto_frontier,
    enumerate_candidates,
    format_candidate_limit,
    compute_model_summary,
    compute_nontrivial_score,
    profile_candidates,
    rank_model_summaries,
    render_model_ranking_markdown,
    run_all_models_experiment,
    safe_estimate_privacy_leakage,
    safe_log10,
    save_candidates_csv,
    save_candidates_json,
    save_model_ranking_csv,
    save_model_ranking_json,
)


def make_candidate_record(
    candidate_id: str,
    *,
    legacy_layer_index: int,
    payload_mb: float,
    privacy_score: float,
    trainable: bool = True,
    validation_passed: bool = True,
) -> CandidateRecord:
    payload_bytes = int(payload_mb * 1024 * 1024)
    edge_parameter_ratio = 1.0 - privacy_score
    return CandidateRecord(
        candidate_id=candidate_id,
        legacy_layer_index=legacy_layer_index,
        canonical_split_key=candidate_id,
        boundary_tensor_count=1,
        boundary_tensor_labels="[]",
        boundary_shape_summary="[]",
        payload_bytes=payload_bytes,
        payload_mb=payload_mb,
        input_tensor_bytes=1024 * 1024,
        payload_ratio_to_input=payload_mb,
        edge_parameter_count=max(1, int(edge_parameter_ratio * 1000)),
        total_parameter_count=1000,
        edge_parameter_ratio=edge_parameter_ratio,
        privacy_leakage_official=0.01,
        privacy_leakage_log10=-2.0,
        privacy_leakage_score=privacy_score,
        estimated_edge_flops=100.0,
        estimated_cloud_flops=1000.0,
        estimated_latency=10.0,
        is_trainable_tail=trainable,
        validation_passed=validation_passed,
        replay_success_rate=1.0 if validation_passed else 0.0,
        tail_trainability=trainable,
    )


def make_cli_args(tmp_path: Path, models: str = "good,bad") -> SimpleNamespace:
    return SimpleNamespace(
        model=None,
        models=models,
        resolved_models=[part.strip() for part in models.split(",")],
        multi_model_layout=True,
        device="cpu",
        input_size=[640, 640],
        max_candidates=64,
        max_boundary_count=8,
        max_payload_mb=128,
        privacy_epsilon=1e-12,
        validate_candidates=False,
        output_dir=str(tmp_path),
        format="both",
        top_k_labels=8,
        seed=42,
    )


# ───────────────────────────────────────────────────────────────────────
# Test Privacy Leakage Calculations
# ───────────────────────────────────────────────────────────────────────


class TestPrivacyLeakageCalculations:
    """Test privacy_leakage_official and privacy_leakage_score calculations."""

    def test_privacy_leakage_with_zero_edge_params(self):
        """Privacy leakage should be very large (1/epsilon) when edge_parameter_count is 0."""
        result = safe_estimate_privacy_leakage(0, epsilon=1e-12)
        # When count=0, result should be 1 / (0 + 1e-12) = 1e12, which is very large
        assert result == 1e12
        
        # With larger epsilon, should be smaller
        result2 = safe_estimate_privacy_leakage(0, epsilon=1e-6)
        assert result2 == 1e6
        assert result2 < result

    def test_privacy_leakage_with_positive_edge_params(self):
        """Privacy leakage should decrease with more edge parameters."""
        result1 = safe_estimate_privacy_leakage(100, epsilon=1e-12)
        result2 = safe_estimate_privacy_leakage(1000, epsilon=1e-12)
        
        assert result1 > result2  # Fewer params => higher privacy risk
        assert result1 > 0
        assert result2 > 0

    def test_privacy_leakage_with_custom_epsilon(self):
        """Privacy leakage should respect custom epsilon."""
        result1 = safe_estimate_privacy_leakage(100, epsilon=0.0)
        result2 = safe_estimate_privacy_leakage(100, epsilon=10.0)
        
        assert result1 > result2
        assert abs(result1 - 1.0 / 100) < 1e-10

    def test_privacy_leakage_score_formula(self):
        """Privacy leakage score = 1 - edge_parameter_ratio should be in [0,1]."""
        # With edge_parameter_ratio = 0.2
        edge_param_ratio = 0.2
        privacy_score = 1.0 - edge_param_ratio
        privacy_score = max(0.0, min(1.0, privacy_score))
        
        assert 0.0 <= privacy_score <= 1.0
        assert privacy_score == 0.8

    def test_privacy_leakage_score_clipping(self):
        """Privacy leakage score should be clipped to [0,1]."""
        score = 1.0 - (-0.5)  # Would be 1.5
        clipped = max(0.0, min(1.0, score))
        assert clipped == 1.0
        
        score = 1.0 - 1.5  # Would be -0.5
        clipped = max(0.0, min(1.0, score))
        assert clipped == 0.0


# ───────────────────────────────────────────────────────────────────────
# Test Payload Ratio Calculations
# ───────────────────────────────────────────────────────────────────────


class TestPayloadRatioCalculations:
    """Test payload_ratio_to_input calculations."""

    def test_payload_ratio_basic(self):
        """Payload ratio should be payload_bytes / input_tensor_bytes."""
        payload_bytes = 1024 * 100  # 100 KB
        input_bytes = 1024 * 1024  # 1 MB
        
        ratio = float(payload_bytes) / float(input_bytes)
        assert abs(ratio - 100.0 / 1024) < 1e-6

    def test_payload_ratio_with_zero_input(self):
        """Payload ratio with zero input bytes should be handled safely."""
        payload_bytes = 1024
        input_bytes = 0
        
        ratio = (
            float(payload_bytes) / float(input_bytes)
            if input_bytes > 0
            else 0.0
        )
        assert ratio == 0.0

    def test_payload_mb_conversion(self):
        """Payload MB conversion should be correct."""
        payload_bytes = 1024 * 1024 * 2  # 2 MB
        payload_mb = payload_bytes / (1024 * 1024)
        
        assert abs(payload_mb - 2.0) < 1e-6

    def test_raw_input_size_uses_uint8_rgb_frame_bytes(self):
        """The layer-0 baseline should be raw image bytes, not float tensors."""
        assert compute_raw_input_size_bytes([640, 640]) == 640 * 640 * 3

    def test_layer_zero_payload_displays_initial_input_size(self):
        """Layer index 0 should show the shared raw input size."""
        input_size_bytes = compute_raw_input_size_bytes([640, 640])
        candidate = SplitCandidate(
            candidate_id="layer0",
            edge_nodes=[],
            cloud_nodes=["conv"],
            boundary_edges=[],
            boundary_tensor_labels=["input"],
            edge_input_labels=[],
            cloud_input_labels=["input"],
            cloud_output_labels=[],
            estimated_edge_flops=0.0,
            estimated_cloud_flops=1.0,
            estimated_payload_bytes=512,
            estimated_privacy_risk=0.0,
            estimated_latency=0.0,
            is_trainable_tail=True,
            legacy_layer_index=0,
            boundary_count=1,
            edge_parameter_count=0,
            total_parameter_count=100,
            edge_parameter_ratio=0.0,
            metadata={"boundary_shape_summary": []},
        )

        records = profile_candidates(
            [candidate],
            sample_input=MagicMock(),
            runtime=SimpleNamespace(),
            input_size_bytes=input_size_bytes,
            initial_input_shape=[640, 640],
        )

        assert records[0].payload_bytes == input_size_bytes
        assert records[0].payload_mb == pytest.approx(640 * 640 * 3 / 1024 / 1024)
        assert records[0].payload_ratio_to_input == pytest.approx(1.0)
        assert json.loads(records[0].boundary_shape_summary) == [["input", [640, 640]]]

    def test_missing_layer_zero_prepends_initial_input_record(self):
        """Candidate sets that start after layer 0 should still show raw input size."""
        input_size_bytes = compute_raw_input_size_bytes([640, 640])
        candidate = SplitCandidate(
            candidate_id="layer1",
            edge_nodes=["conv"],
            cloud_nodes=[],
            boundary_edges=[],
            boundary_tensor_labels=["node_0"],
            edge_input_labels=[],
            cloud_input_labels=["node_0"],
            cloud_output_labels=[],
            estimated_edge_flops=1.0,
            estimated_cloud_flops=1.0,
            estimated_payload_bytes=512,
            estimated_privacy_risk=0.0,
            estimated_latency=0.0,
            is_trainable_tail=True,
            legacy_layer_index=1,
            boundary_count=1,
            edge_parameter_count=10,
            total_parameter_count=100,
            edge_parameter_ratio=0.1,
            metadata={"boundary_shape_summary": []},
        )

        records = profile_candidates(
            [candidate],
            sample_input=SimpleNamespace(shape=(1, 3, 640, 640)),
            runtime=SimpleNamespace(),
            input_size_bytes=input_size_bytes,
            initial_input_shape=[640, 640],
        )

        assert [record.legacy_layer_index for record in records] == [0, 1]
        assert records[0].candidate_id == "initial_input"
        assert records[0].payload_bytes == input_size_bytes
        assert records[0].payload_ratio_to_input == pytest.approx(1.0)
        assert json.loads(records[0].boundary_shape_summary) == [["input", [640, 640]]]
        assert records[1].payload_bytes == 512


# ───────────────────────────────────────────────────────────────────────
# Test Pareto Frontier Computation
# ───────────────────────────────────────────────────────────────────────


class TestParetoFrontierComputation:
    """Test Pareto frontier calculation."""

    def test_pareto_frontier_empty_records(self):
        """Pareto frontier of empty records should be empty."""
        result = compute_pareto_frontier([])
        assert result == []

    def test_pareto_frontier_single_record(self):
        """Single record should be on the frontier."""
        record = CandidateRecord(
            candidate_id="c1",
            legacy_layer_index=0,
            canonical_split_key="c1",
            boundary_tensor_count=1,
            boundary_tensor_labels="[]",
            boundary_shape_summary="[]",
            payload_bytes=1024,
            payload_mb=0.001,
            input_tensor_bytes=1024 * 1024,
            payload_ratio_to_input=0.001,
            edge_parameter_count=100,
            total_parameter_count=1000,
            edge_parameter_ratio=0.1,
            privacy_leakage_official=0.01,
            privacy_leakage_log10=-2.0,
            privacy_leakage_score=0.9,
            estimated_edge_flops=100.0,
            estimated_cloud_flops=1000.0,
            estimated_latency=10.0,
            is_trainable_tail=True,
            validation_passed=True,
            replay_success_rate=1.0,
            tail_trainability=True,
        )
        
        result = compute_pareto_frontier([record])
        assert result == [0]

    def test_pareto_frontier_obvious_dominance(self):
        """Obvious dominance should be detected."""
        records = [
            CandidateRecord(
                candidate_id="dominated",
                legacy_layer_index=0,
                canonical_split_key="c1",
                boundary_tensor_count=1,
                boundary_tensor_labels="[]",
                boundary_shape_summary="[]",
                payload_bytes=1024,
                payload_mb=10.0,  # Higher payload (worse)
                input_tensor_bytes=1024 * 1024,
                payload_ratio_to_input=0.01,
                edge_parameter_count=100,
                total_parameter_count=1000,
                edge_parameter_ratio=0.1,
                privacy_leakage_official=0.01,
                privacy_leakage_log10=-2.0,
                privacy_leakage_score=0.8,  # Higher privacy leakage (worse)
                estimated_edge_flops=100.0,
                estimated_cloud_flops=1000.0,
                estimated_latency=10.0,
                is_trainable_tail=True,
                validation_passed=True,
                replay_success_rate=1.0,
                tail_trainability=True,
            ),
            CandidateRecord(
                candidate_id="pareto",
                legacy_layer_index=0,
                canonical_split_key="c2",
                boundary_tensor_count=1,
                boundary_tensor_labels="[]",
                boundary_shape_summary="[]",
                payload_bytes=512,
                payload_mb=5.0,  # Lower payload (better)
                input_tensor_bytes=1024 * 1024,
                payload_ratio_to_input=0.005,
                edge_parameter_count=200,
                total_parameter_count=1000,
                edge_parameter_ratio=0.2,
                privacy_leakage_official=0.005,
                privacy_leakage_log10=-2.3,
                privacy_leakage_score=0.5,  # Lower privacy leakage (better)
                estimated_edge_flops=100.0,
                estimated_cloud_flops=1000.0,
                estimated_latency=10.0,
                is_trainable_tail=True,
                validation_passed=True,
                replay_success_rate=1.0,
                tail_trainability=True,
            ),
        ]
        
        result = compute_pareto_frontier(records)
        # Only the second record should be on frontier (it dominates the first)
        assert result == [1], f"Expected [1], got {result}"


class TestLog10Conversion:
    """Test safe log10 conversion."""

    def test_safe_log10_normal_values(self):
        """Log10 of normal values should work."""
        result = safe_log10(100.0)
        assert abs(result - 2.0) < 1e-6

    def test_safe_log10_infinity(self):
        """Log10 of infinity should return 0."""
        result = safe_log10(float("inf"))
        assert result == 0.0

    def test_safe_log10_zero(self):
        """Log10 of zero should return 0."""
        result = safe_log10(0.0)
        assert result == 0.0

    def test_safe_log10_negative(self):
        """Log10 of negative should return 0."""
        result = safe_log10(-10.0)
        assert result == 0.0


# ───────────────────────────────────────────────────────────────────────
# Test CSV/JSON Serialization
# ───────────────────────────────────────────────────────────────────────


class TestSerialization:
    """Test CSV and JSON serialization."""

    @pytest.fixture
    def sample_records(self):
        """Create sample records for testing."""
        return [
            CandidateRecord(
                candidate_id=f"candidate_{i}",
                legacy_layer_index=i,
                canonical_split_key=f"split_{i}",
                boundary_tensor_count=i + 1,
                boundary_tensor_labels=json.dumps([f"tensor_{j}" for j in range(i + 1)]),
                boundary_shape_summary=json.dumps([]),
                payload_bytes=1024 * (i + 1),
                payload_mb=0.001 * (i + 1),
                input_tensor_bytes=1024 * 1024,
                payload_ratio_to_input=0.001 * (i + 1),
                edge_parameter_count=100 * (i + 1),
                total_parameter_count=1000,
                edge_parameter_ratio=0.1 * (i + 1),
                privacy_leakage_official=0.01 / (i + 1),
                privacy_leakage_log10=math.log10(0.01 / (i + 1)) if (0.01 / (i + 1)) > 0 else 0.0,
                privacy_leakage_score=0.9 - 0.1 * i,
                estimated_edge_flops=100.0,
                estimated_cloud_flops=1000.0,
                estimated_latency=10.0,
                is_trainable_tail=True,
                validation_passed=True,
                replay_success_rate=1.0,
                tail_trainability=True,
            )
            for i in range(3)
        ]

    def test_candidate_record_to_dict(self, sample_records):
        """CandidateRecord should convert to dict."""
        record = sample_records[0]
        d = record.to_dict()
        
        assert isinstance(d, dict)
        assert d["candidate_id"] == "candidate_0"
        assert d["payload_mb"] == 0.001

    def test_save_candidates_csv(self, sample_records):
        """CSV serialization should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            save_candidates_csv(sample_records, csv_path)
            
            assert csv_path.exists()
            
            # Read and verify
            with open(csv_path, "r") as f:
                lines = f.readlines()
            
            assert len(lines) == 4  # Header + 3 records
            assert "candidate_id" in lines[0]

    def test_save_candidates_json(self, sample_records):
        """JSON serialization should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "test.json"
            metadata = ExperimentMetadata(
                model_name="test_model",
                input_height=640,
                input_width=640,
                initial_input_height=640,
                initial_input_width=640,
                initial_input_bytes=640 * 640 * 3,
                device="cpu",
                max_candidates=128,
                max_boundary_count=8,
                max_payload_mb=128,
                privacy_epsilon=1e-12,
                validate_candidates=False,
                candidate_count=3,
            )
            
            save_candidates_json(sample_records, metadata, json_path)
            
            assert json_path.exists()
            
            # Read and verify
            with open(json_path, "r") as f:
                data = json.load(f)
            
            assert "metadata" in data
            assert "candidates" in data
            assert len(data["candidates"]) == 3
            assert data["metadata"]["model_name"] == "test_model"

    def test_csv_handles_empty_records(self):
        """CSV save should handle empty records gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "empty.csv"
            save_candidates_csv([], csv_path)
            
            # Should create file with just headers
            assert csv_path.exists()
            with open(csv_path, "r") as f:
                lines = f.readlines()
            assert len(lines) >= 1  # At least header


# ───────────────────────────────────────────────────────────────────────
# Test Validation Error Handling
# ───────────────────────────────────────────────────────────────────────


class TestValidationErrorHandling:
    """Test that validation errors don't break the experiment."""

    def test_candidate_record_with_validation_error(self):
        """Record with validation error should still serialize."""
        record = CandidateRecord(
            candidate_id="failed_validation",
            legacy_layer_index=0,
            canonical_split_key="c1",
            boundary_tensor_count=1,
            boundary_tensor_labels="[]",
            boundary_shape_summary="[]",
            payload_bytes=1024,
            payload_mb=0.001,
            input_tensor_bytes=1024 * 1024,
            payload_ratio_to_input=0.001,
            edge_parameter_count=100,
            total_parameter_count=1000,
            edge_parameter_ratio=0.1,
            privacy_leakage_official=0.01,
            privacy_leakage_log10=-2.0,
            privacy_leakage_score=0.9,
            estimated_edge_flops=100.0,
            estimated_cloud_flops=1000.0,
            estimated_latency=10.0,
            is_trainable_tail=True,
            validation_passed=False,
            replay_success_rate=0.0,
            tail_trainability=True,
            validation_error="Split replay failed: output shape mismatch",
        )
        
        d = record.to_dict()
        assert d["validation_passed"] is False
        assert "output shape mismatch" in d["validation_error"]

    def test_metadata_with_missing_trace_signature(self):
        """Metadata should handle missing trace_signature gracefully."""
        metadata = ExperimentMetadata(
            model_name="test_model",
            input_height=640,
            input_width=640,
            initial_input_height=640,
            initial_input_width=640,
            initial_input_bytes=640 * 640 * 3,
            device="cpu",
            max_candidates=128,
            max_boundary_count=8,
            max_payload_mb=128,
            privacy_epsilon=1e-12,
            validate_candidates=False,
            candidate_count=0,
            trace_signature=None,
        )
        
        d = metadata.to_dict()
        assert d["trace_signature"] is None


class TestAllModelRanking:
    """Test all-model summary, scoring, ranking, and reports."""

    def test_unlimited_candidate_enumeration_does_not_truncate(self, monkeypatch):
        raw_candidates = [
            SimpleNamespace(
                split_id=f"candidate_{idx}",
                prefix_nodes=[f"node_{idx}"],
                suffix_nodes=[],
                boundary_nodes=[f"boundary_{idx}"],
                cost=SimpleNamespace(boundary_bytes=1024),
                trainable_suffix=True,
            )
            for idx in range(5)
        ]
        captured_kwargs = {}

        def fake_enumerate_frontier_splits(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return tuple(raw_candidates)

        monkeypatch.setattr(
            "tools.run_split_tradeoff_motivation_experiment.enumerate_frontier_splits",
            fake_enumerate_frontier_splits,
        )
        runtime = SimpleNamespace(trace_plan=SimpleNamespace(nodes=[]))

        candidates = enumerate_candidates(
            runtime,
            max_candidates=None,
            max_boundary_count=8,
            max_payload_bytes=1024 * 1024,
        )

        assert len(candidates) == 5
        assert "max_splits" not in captured_kwargs
        assert format_candidate_limit(None) == "all"
        assert format_candidate_limit(0) == "all"
        assert format_candidate_limit(3) == "3"

    def test_motivation_strength_score_formula(self):
        records = [
            make_candidate_record("early", legacy_layer_index=0, payload_mb=1.0, privacy_score=0.9),
            make_candidate_record("middle", legacy_layer_index=1, payload_mb=10.0, privacy_score=0.8),
            make_candidate_record("late", legacy_layer_index=2, payload_mb=100.0, privacy_score=0.1),
            make_candidate_record(
                "invalid",
                legacy_layer_index=3,
                payload_mb=200.0,
                privacy_score=0.2,
                trainable=False,
                validation_passed=False,
            ),
        ]

        summary = compute_model_summary("demo", "ok", records)

        expected = (
            0.25 * (4 / 64.0)
            + 0.25 * 1.0
            + 0.20 * 0.8
            + 0.15 * (3 / 8.0)
            + 0.10 * (3 / 4.0)
            + 0.05 * 1.0
        )
        assert summary.candidate_count == 4
        assert summary.valid_candidate_count == 3
        assert summary.trainable_candidate_count == 3
        assert summary.pareto_candidate_count == 3
        assert summary.payload_spread_ratio == pytest.approx(200.0)
        assert summary.payload_spread_log10 == pytest.approx(math.log10(200.0))
        assert summary.privacy_spread == pytest.approx(0.8)
        assert summary.nontrivial_score == pytest.approx(1.0)
        assert summary.motivation_strength_score == pytest.approx(expected)

    def test_nontrivial_score_formula(self):
        records = [
            make_candidate_record("payload_min", legacy_layer_index=0, payload_mb=1.0, privacy_score=0.9),
            make_candidate_record("privacy_min", legacy_layer_index=1, payload_mb=10.0, privacy_score=0.1),
            make_candidate_record("privacy_max", legacy_layer_index=2, payload_mb=100.0, privacy_score=0.95),
        ]

        assert compute_nontrivial_score(records) == pytest.approx(0.8)

    def test_single_model_failure_does_not_block_other_summaries(self, tmp_path, monkeypatch):
        good_records = [
            make_candidate_record("a", legacy_layer_index=0, payload_mb=1.0, privacy_score=0.4),
            make_candidate_record("b", legacy_layer_index=1, payload_mb=4.0, privacy_score=0.8),
        ]
        good_summary = compute_model_summary("good", "ok", good_records)

        def fake_run_single(args, model_name, output_dir, device):
            if model_name == "bad":
                raise RuntimeError("trace exploded")
            return ModelExperimentResult(
                summary=good_summary,
                records=good_records,
                metadata=None,
                output_dir=output_dir,
            )

        monkeypatch.setattr(
            "tools.run_split_tradeoff_motivation_experiment.run_single_model_experiment",
            fake_run_single,
        )
        monkeypatch.setattr(
            "tools.run_split_tradeoff_motivation_experiment.save_all_model_outputs",
            lambda summaries, records_by_model, args, output_dir: None,
        )

        results = run_all_models_experiment(make_cli_args(tmp_path))
        summaries = {result.summary.model: result.summary for result in results}

        assert summaries["good"].status == "ok"
        assert summaries["bad"].status == "trace_failed"
        assert "trace exploded" in summaries["bad"].error

    def test_ranking_selects_highest_score_model(self):
        low = ModelSummary(
            model="low",
            status="ok",
            candidate_count=2,
            motivation_strength_score=0.25,
        )
        high = ModelSummary(
            model="high",
            status="ok",
            candidate_count=2,
            motivation_strength_score=0.75,
        )
        failed = ModelSummary(model="failed", status="build_failed", error="missing weights")

        ranked = rank_model_summaries([low, failed, high])

        assert ranked[0].model == "high"
        assert ranked[0].recommended_as_main_figure is True
        assert low.recommended_as_main_figure is False
        assert failed.recommended_as_main_figure is False

    def test_model_ranking_markdown_contains_recommendation_and_failures(self, tmp_path):
        recommended = ModelSummary(
            model="best",
            status="ok",
            candidate_count=8,
            payload_spread_log10=1.5,
            payload_spread_ratio=32.0,
            privacy_spread=0.7,
            pareto_candidate_count=4,
            valid_ratio=0.75,
            motivation_strength_score=0.8,
            recommended_as_main_figure=True,
        )
        failed = ModelSummary(model="broken", status="build_failed", error="weights missing")
        args = make_cli_args(tmp_path, models="best,broken")
        args.max_candidates = None

        markdown = render_model_ranking_markdown([recommended, failed], args)

        assert "Recommended as main figure: **best**" in markdown
        assert "- Max candidates: all" in markdown
        assert "broken: build_failed; weights missing" in markdown
        assert "split-tradeoff expressiveness rather than detection accuracy" in markdown
        assert "candidate_count: 8" in markdown
        assert "payload_spread_log10" in markdown
        assert "privacy_spread" in markdown
        assert "pareto_candidate_count" in markdown
        assert "valid_ratio" in markdown

    def test_all_model_summary_serializes_to_csv_and_json(self, tmp_path):
        summaries = rank_model_summaries([
            ModelSummary(
                model="a",
                status="ok",
                candidate_count=2,
                motivation_strength_score=0.6,
            ),
            ModelSummary(model="b", status="trace_failed", error="trace failed"),
        ])

        csv_path = tmp_path / "model_ranking.csv"
        json_path = tmp_path / "model_ranking.json"
        save_model_ranking_csv(summaries, csv_path)
        save_model_ranking_json(summaries, json_path)

        assert csv_path.exists()
        assert json_path.exists()
        assert "motivation_strength_score" in csv_path.read_text(encoding="utf-8")
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["recommended_model"] == "a"
        assert [item["model"] for item in data["models"]] == ["a", "b"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
