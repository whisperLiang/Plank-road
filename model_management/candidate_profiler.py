from __future__ import annotations

import time
from typing import Sequence

from model_management.split_candidate import CandidateProfile, SplitCandidate


def profile_candidates(
    runtime,
    candidates: Sequence[SplitCandidate],
    *,
    validate: bool = True,
    validation_runs: int = 1,
) -> list[CandidateProfile]:
    profiles: list[CandidateProfile] = []
    for candidate in candidates:
        edge_latency = 0.0
        cloud_latency = 0.0
        end_to_end_latency = 0.0
        successes = 0
        stability = 0.0
        trainable = candidate.is_trainable_tail
        error: str | None = None

        if validate:
            trainable = True
            for _ in range(max(1, validation_runs)):
                start = time.perf_counter()
                report = runtime.validate_candidate(candidate)
                elapsed = time.perf_counter() - start
                end_to_end_latency += elapsed
                edge_latency += float(report.get("edge_latency", 0.0))
                cloud_latency += float(report.get("cloud_latency", 0.0))
                successes += int(report.get("success", False))
                stability += float(report.get("stability_score", 0.0))
                trainable = trainable and bool(report.get("tail_trainability", candidate.is_trainable_tail))
                error = report.get("error", error)
            runs = float(max(1, validation_runs))
            replay_success_rate = successes / runs
            stability_score = stability / runs if stability else replay_success_rate
            edge_latency /= runs
            cloud_latency /= runs
            end_to_end_latency /= runs
        else:
            replay_success_rate = 0.0
            stability_score = 0.0

        profile = CandidateProfile(
            candidate_id=candidate.candidate_id,
            edge_flops=candidate.estimated_edge_flops,
            cloud_flops=candidate.estimated_cloud_flops,
            payload_bytes=candidate.estimated_payload_bytes,
            boundary_tensor_count=candidate.boundary_count,
            boundary_shape_summary=[
                (label, runtime.graph.nodes[label].tensor_shape)
                for label in candidate.boundary_tensor_labels
            ],
            estimated_privacy_leakage=candidate.estimated_privacy_risk,
            measured_edge_latency=edge_latency,
            measured_cloud_latency=cloud_latency,
            measured_end_to_end_latency=end_to_end_latency,
            replay_success_rate=replay_success_rate,
            tail_trainability=trainable,
            stability_score=stability_score,
            validation_passed=error is None and replay_success_rate >= 1.0,
            metadata={"error": error} if error else {},
        )
        profiles.append(profile)
    return profiles
