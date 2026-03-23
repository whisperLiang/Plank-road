from __future__ import annotations

import torch
import torchvision.models as tv_models

from model_management.candidate_profiler import profile_candidates
from model_management.candidate_selector import SplitCandidateSelector
from model_management.universal_model_split import UniversalModelSplitter


def test_candidate_profiler_and_selector_operate_on_candidate_sets():
    model = tv_models.resnet18(weights=None).eval()
    sample = torch.randn(1, 3, 32, 32)

    splitter = UniversalModelSplitter(device="cpu")
    splitter.trace(model, sample)
    candidates = splitter.enumerate_candidates(max_candidates=6)
    assert len(candidates) >= 2

    profiles = profile_candidates(splitter, candidates[:3], validate=True, validation_runs=1)
    assert len(profiles) == 3
    assert all(profile.validation_passed for profile in profiles)

    selector = SplitCandidateSelector(candidates[:3], profiles, alpha=0.2, epsilon=0.0)
    chosen = selector.select_candidate(bandwidth=12.0, edge_load=0.25, cloud_load=0.15)
    assert chosen in {candidate.candidate_id for candidate in candidates[:3]}

    context = selector.fit_context(chosen, bandwidth=12.0, edge_load=0.25, cloud_load=0.15)
    assert context.shape[0] == selector.feature_dim

    selector.update_reward(chosen, reward=1.25, context=context)
    assert selector.states[chosen].num_updates == 1
    assert selector.states[chosen].historical_reward > 0.0

    selector.invalidate_candidate(chosen)
    fallback = selector.select_candidate(bandwidth=8.0, edge_load=0.4, cloud_load=0.2)
    assert fallback != chosen

    splitter_selector = splitter.create_split_selector(profiles)
    pick = splitter_selector.select_candidate(bandwidth=10.0, edge_load=0.1, cloud_load=0.1)
    assert pick in {candidate.candidate_id for candidate in splitter.candidates}
