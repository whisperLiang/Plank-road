from __future__ import annotations

from model_management.candidate_selector import SplitCandidateSelector
from model_management.split_candidate import CandidateProfile, SplitCandidate


def _candidate(index: int) -> SplitCandidate:
    return SplitCandidate(
        candidate_id=f"after:layer{index}",
        edge_nodes=[f"layer{index}"],
        cloud_nodes=[],
        boundary_edges=[],
        boundary_tensor_labels=[f"node_{index}"],
        edge_input_labels=[],
        cloud_input_labels=[],
        cloud_output_labels=[],
        estimated_edge_flops=float(index),
        estimated_cloud_flops=float(10 - index),
        estimated_payload_bytes=1024 * index,
        estimated_privacy_risk=0.1 * index,
        estimated_latency=float(index),
        is_trainable_tail=True,
        is_validated=True,
        boundary_count=1,
    )


def _profile(candidate: SplitCandidate) -> CandidateProfile:
    return CandidateProfile(
        candidate_id=candidate.candidate_id,
        edge_flops=candidate.estimated_edge_flops,
        cloud_flops=candidate.estimated_cloud_flops,
        payload_bytes=candidate.estimated_payload_bytes,
        boundary_tensor_count=candidate.boundary_count,
        boundary_shape_summary=[(candidate.boundary_tensor_labels[0], ("B", 8))],
        estimated_privacy_leakage=candidate.estimated_privacy_risk,
        measured_edge_latency=0.01,
        measured_cloud_latency=0.02,
        measured_end_to_end_latency=candidate.estimated_latency,
        replay_success_rate=1.0,
        tail_trainability=True,
        stability_score=1.0,
        validation_passed=True,
    )


def test_candidate_selector_operates_on_profiled_candidate_sets():
    candidates = [_candidate(index) for index in range(1, 4)]
    profiles = [_profile(candidate) for candidate in candidates]

    selector = SplitCandidateSelector(candidates, profiles, alpha=0.2, epsilon=0.0)
    chosen = selector.select_candidate(bandwidth=12.0, edge_load=0.25, cloud_load=0.15)
    assert chosen in {candidate.candidate_id for candidate in candidates}

    context = selector.fit_context(chosen, bandwidth=12.0, edge_load=0.25, cloud_load=0.15)
    assert context.shape[0] == selector.feature_dim

    selector.update_reward(chosen, reward=1.25, context=context)
    assert selector.states[chosen].num_updates == 1
    assert selector.states[chosen].historical_reward > 0.0

    selector.invalidate_candidate(chosen)
    fallback = selector.select_candidate(bandwidth=8.0, edge_load=0.4, cloud_load=0.2)
    assert fallback != chosen
