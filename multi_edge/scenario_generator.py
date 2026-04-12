"""Scenario generator for multi-device experiments.

Generates per-device profiles and synthetic inference streams that
simulate varying drift severity, bandwidth constraints, and local
training budgets. Supports concurrent drift bursts for stress testing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from baselines.base_method import InferenceResult


# ── Profile value tables ─────────────────────────────────────────────

# Maps profile names to numeric parameters used by the simulation.
# These are intentionally simple approximations, not realistic models.

DRIFT_PROFILES = {
    "low": {
        "base_confidence": 0.85,
        "drift_probability": 0.02,
        "confidence_noise_std": 0.05,
        "drift_confidence_drop": 0.10,
    },
    "medium": {
        "base_confidence": 0.75,
        "drift_probability": 0.08,
        "confidence_noise_std": 0.10,
        "drift_confidence_drop": 0.20,
    },
    "high": {
        "base_confidence": 0.60,
        "drift_probability": 0.20,
        "confidence_noise_std": 0.15,
        "drift_confidence_drop": 0.30,
    },
}

BANDWIDTH_PROFILES = {
    "low": {"effective_bw_bytes_per_sec": 500_000, "latency_overhead_ms": 50.0},
    "medium": {"effective_bw_bytes_per_sec": 5_000_000, "latency_overhead_ms": 10.0},
    "high": {"effective_bw_bytes_per_sec": 50_000_000, "latency_overhead_ms": 2.0},
}

LOCAL_TRAIN_BUDGET_PROFILES = {
    "low": {"sec_per_epoch": 5.0},
    "medium": {"sec_per_epoch": 1.0},
    "high": {"sec_per_epoch": 0.3},
}


@dataclass
class DeviceProfile:
    """Assigned profile for one simulated device."""
    device_id: int
    drift_profile: str = "medium"
    bandwidth_profile: str = "medium"
    local_train_budget_profile: str = "medium"

    @property
    def drift_params(self) -> dict[str, Any]:
        return DRIFT_PROFILES.get(self.drift_profile, DRIFT_PROFILES["medium"])

    @property
    def bandwidth_params(self) -> dict[str, Any]:
        return BANDWIDTH_PROFILES.get(self.bandwidth_profile, BANDWIDTH_PROFILES["medium"])

    @property
    def local_train_params(self) -> dict[str, Any]:
        return LOCAL_TRAIN_BUDGET_PROFILES.get(
            self.local_train_budget_profile, LOCAL_TRAIN_BUDGET_PROFILES["medium"]
        )


class ScenarioGenerator:
    """Generates device profiles and synthetic inference streams.

    Attributes:
        num_devices: Number of devices to simulate.
        total_frames: Frames per device in the simulation.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_devices: int = 1,
        total_frames: int = 300,
        seed: int = 42,
    ) -> None:
        self.num_devices = num_devices
        self.total_frames = total_frames
        self.rng = random.Random(seed)

    def generate_uniform_profiles(
        self,
        drift: str = "medium",
        bandwidth: str = "medium",
        local_budget: str = "medium",
    ) -> list[DeviceProfile]:
        """All devices share the same profile (homogeneous scenario)."""
        return [
            DeviceProfile(
                device_id=i + 1,
                drift_profile=drift,
                bandwidth_profile=bandwidth,
                local_train_budget_profile=local_budget,
            )
            for i in range(self.num_devices)
        ]

    def generate_heterogeneous_profiles(
        self,
        drift_options: list[str] | None = None,
        bandwidth_options: list[str] | None = None,
        budget_options: list[str] | None = None,
    ) -> list[DeviceProfile]:
        """Assign random profiles from provided options."""
        drifts = drift_options or ["low", "medium", "high"]
        bws = bandwidth_options or ["low", "medium", "high"]
        budgets = budget_options or ["low", "medium", "high"]
        return [
            DeviceProfile(
                device_id=i + 1,
                drift_profile=self.rng.choice(drifts),
                bandwidth_profile=self.rng.choice(bws),
                local_train_budget_profile=self.rng.choice(budgets),
            )
            for i in range(self.num_devices)
        ]

    def generate_concurrent_drift_burst(
        self,
        burst_start_frame: int | None = None,
        burst_duration: int = 30,
        burst_drift: str = "high",
    ) -> list[DeviceProfile]:
        """Generate profiles where multiple devices enter drift in a
        nearby time window. The burst parameters are stored in the
        profile metadata (used by ``generate_stream``).

        Returns homogeneous medium profiles with burst info attached.
        """
        if burst_start_frame is None:
            burst_start_frame = self.total_frames // 3
        profiles = self.generate_uniform_profiles(drift="low")
        for p in profiles:
            # Store burst metadata as extra attributes
            p._burst_start = burst_start_frame + self.rng.randint(-5, 5)  # type: ignore[attr-defined]
            p._burst_duration = burst_duration  # type: ignore[attr-defined]
            p._burst_drift = burst_drift  # type: ignore[attr-defined]
        return profiles

    def generate_stream(
        self,
        profile: DeviceProfile,
    ) -> list[InferenceResult]:
        """Synthesize a sequence of inference results for one device.

        Uses profile drift parameters to simulate varying confidence,
        occasional drift flags, and proxy mAP degradation.
        """
        params = profile.drift_params
        base_conf = params["base_confidence"]
        drift_prob = params["drift_probability"]
        noise_std = params["confidence_noise_std"]
        drift_drop = params["drift_confidence_drop"]

        bw_params = profile.bandwidth_params
        latency_overhead = bw_params["latency_overhead_ms"]

        # Check for burst metadata
        burst_start = getattr(profile, "_burst_start", None)
        burst_duration = getattr(profile, "_burst_duration", 0)
        burst_drift_name = getattr(profile, "_burst_drift", "high")
        burst_params = DRIFT_PROFILES.get(burst_drift_name, DRIFT_PROFILES["high"])

        results: list[InferenceResult] = []
        for frame_idx in range(self.total_frames):
            # Determine if we are in a drift burst
            in_burst = False
            if burst_start is not None:
                if burst_start <= frame_idx < burst_start + burst_duration:
                    in_burst = True

            if in_burst:
                eff_conf = burst_params["base_confidence"]
                eff_drift_prob = burst_params["drift_probability"]
                eff_noise = burst_params["confidence_noise_std"]
                eff_drop = burst_params["drift_confidence_drop"]
            else:
                eff_conf = base_conf
                eff_drift_prob = drift_prob
                eff_noise = noise_std
                eff_drop = drift_drop

            # Simulate confidence + drift
            drift_flag = self.rng.random() < eff_drift_prob
            noise = self.rng.gauss(0.0, eff_noise)
            confidence = max(0.0, min(1.0, eff_conf + noise - (eff_drop if drift_flag else 0.0)))

            # Proxy mAP correlates loosely with confidence
            proxy_map = max(0.0, min(1.0, confidence * 0.9 + self.rng.gauss(0.0, 0.05)))

            # Base inference latency + bandwidth overhead + noise
            latency_ms = 10.0 + latency_overhead + abs(self.rng.gauss(0.0, 3.0))

            results.append(InferenceResult(
                device_id=profile.device_id,
                frame_index=frame_idx,
                confidence=confidence,
                proxy_map=proxy_map,
                latency_ms=latency_ms,
                drift_flag=drift_flag,
                num_detections=max(0, int(5 + self.rng.gauss(0, 2))),
            ))

        return results
