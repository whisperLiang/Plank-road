"""
Resource-Aware Continual Learning Trigger & Split-Point Selector
================================================================

Implements "Edge-side Joint Optimization Decision + Cloud-side Asynchronous Resource Pricing"
to jointly decide:

    1. **Whether to trigger continual learning** — considering cloud
       resource pricing, network bandwidth, and intermediate-feature
       privacy constraints.
    2. **Where to place the split point** — selecting the layer that best
       balances latency, bandwidth cost, privacy leakage, and cloud
       compute cost.

At each decision epoch, we evaluate candidate split points by comparing
resource costs (based on async shadow prices from the cloud) against 
performance gains.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

from loguru import logger


# ── ResourceAwareCLTrigger ───────────────────────────────────────────

class ResourceAwareCLTrigger:
    """Multi-queue Lyapunov policy for resource-aware CL triggering.

    Extends RCCDA's single-queue policy to four virtual queues that
    track update rate, cloud cost, bandwidth, and privacy budgets.

    Parameters
    ----------
    pi_bar : float
        Maximum allowed average CL trigger rate.
    V : float
        Lyapunov trade-off: larger → more responsive to loss changes.
    K_p, K_d : float
        PD controller gains for loss error / derivative.
    lambda_cloud : float
        Average cloud-cost budget per round (∈ (0, 1]).
    lambda_bw : float
        Average bandwidth budget per round (∈ (0, 1]).
    lambda_priv : float
        Average privacy-leakage budget per round (∈ (0, 1]).
    w_cloud, w_bw, w_priv : float
        Relative importance weights for each queue in the Lyapunov
        drift expression.  Higher → stricter enforcement.
    confidence_threshold : float
        Confidence score below which a detection is a "loss".
    """

    def __init__(
        self,
        # RCCDA core
        pi_bar: float = 0.1,
        V: float = 10.0,
        K_p: float = 1.0,
        K_d: float = 0.5,
        # Multi-queue budgets
        lambda_cloud: float = 0.5,
        lambda_bw: float = 0.5,
        lambda_priv: float = 0.3,
        # Queue weights
        w_cloud: float = 1.0,
        w_bw: float = 1.0,
        w_priv: float = 1.0,
        # Loss
        confidence_threshold: float = 0.5,
        # For Refactored Joint-Optimization
        client = None,
        split_profiles: dict = None,
        max_tolerable_privacy: float = 1.0,
        base_train_cost: float = 10.0,
    ) -> None:
        # ── RCCDA PD parameters ──
        self.pi_bar = pi_bar
        self.V = V
        self.K_p = K_p
        self.K_d = K_d
        self.confidence_threshold = confidence_threshold

        # ── Multi-queue budgets ──
        self.lambda_cloud = lambda_cloud
        self.lambda_bw = lambda_bw
        self.lambda_priv = lambda_priv

        # ── Queue weights ──
        self.w_cloud = w_cloud
        self.w_bw = w_bw
        self.w_priv = w_priv

        # ── Joint-Optimization parameters ──
        self.client = client
        self.split_profiles = split_profiles or {}
        self.max_tolerable_privacy = max_tolerable_privacy
        self.base_train_cost = base_train_cost

        # ── Virtual queues (initialised to 0) ──
        self.Q_update: float = 0.0     # update-rate queue
        self.Q_cloud: float = 0.0      # cloud-cost queue
        self.Q_bw: float = 0.0         # bandwidth queue
        self.Q_priv: float = 0.0       # privacy queue

        # ── Loss tracking (RCCDA-style) ──
        self.loss_best: float = float("inf")
        self.loss_prev: float = 0.0
        self.loss_initial: Optional[float] = None

        # ── Statistics ──
        self.step: int = 0
        self.trigger_count: int = 0
        self.history: List[Dict] = []

    # ──────── reset ──────────────────────────────────────────────────

    def reset(self) -> None:
        self.Q_update = 0.0
        self.Q_cloud = 0.0
        self.Q_bw = 0.0
        self.Q_priv = 0.0
        self.loss_best = float("inf")
        self.loss_prev = 0.0
        self.loss_initial = None
        self.step = 0
        self.trigger_count = 0
        self.history.clear()

    # ──────── Joint Optimization Refactor ────────────────────────────

    def evaluate_and_trigger(self, drift_severity: float) -> dict:
        """
        Execute joint optimization decision in the edge inference loop.
        Uses cached cloud prices for asynchronous operations.
        
        Evaluates potential split strategies and returns the optimal action.
        """
        if not self.client:
            logger.warning("Cloud client not initialized. Returning DO_NOTHING.")
            return {"action": "DO_NOTHING"}

        prices = self.client.get_cached_prices()
        best_action = {"action": "DO_NOTHING"}
        min_net_value = 0.0  # gating threshold: cost must be less than gain (net_value = cost - gain < 0)
        
        for s in self.split_profiles:
            profile = self.split_profiles[s]
            
            # 1. Hard constraint filtering
            if profile.get('privacy', 1.0) > self.max_tolerable_privacy:
                continue
                
            # 2. Calculate joint cost: (resource cost) - (performance gain)
            # Assuming 'bw' is network bandwidth used and 'gain' provides the performance gain
            cost = (prices.get('price_comp', float('inf')) * self.base_train_cost) + \
                   (prices.get('price_bw', float('inf')) * profile.get('bw', 0.0))
                   
            gain_func = profile.get('gain')
            actual_gain = gain_func(drift_severity) if callable(gain_func) else profile.get('gain', 0.0)
            gain = self.V * actual_gain
            
            net_value = cost - gain
            
            # 3. Search for optimal solution
            if net_value < min_net_value:
                min_net_value = net_value
                best_action = {"action": "TRIGGER", "split_point": s}
                
        return best_action

    # ──────── Properties ─────────────────────────────────────────────

    @property
    def effective_trigger_rate(self) -> float:
        """Empirical average CL trigger rate so far."""
        if self.step == 0:
            return 0.0
        return self.trigger_count / self.step

    @property
    def queue_snapshot(self) -> Dict[str, float]:
        return {
            "Q_update": self.Q_update,
            "Q_cloud": self.Q_cloud,
            "Q_bw": self.Q_bw,
            "Q_priv": self.Q_priv,
        }

# ── Factory ──────────────────────────────────────────────────────────

def create_resource_aware_trigger(config, client=None, split_profiles=None) -> ResourceAwareCLTrigger:
    """Build a ``ResourceAwareCLTrigger`` from the config object.
    """
    ra = getattr(config, "resource_aware_trigger", None)
    dd = getattr(config, "drift_detection", None)

    def _get(key, default, *sources):
        for src in sources:
            if src is not None:
                v = getattr(src, key, None)
                if v is not None:
                    return v
        return default

    return ResourceAwareCLTrigger(
        pi_bar=float(_get("pi_bar", 0.1, ra, dd)),
        V=float(_get("V", 10.0, ra, dd)),
        K_p=float(_get("K_p", 1.0, ra, dd)),
        K_d=float(_get("K_d", 0.5, ra, dd)),
        lambda_cloud=float(_get("lambda_cloud", 0.5, ra)),
        lambda_bw=float(_get("lambda_bw", 0.5, ra)),
        lambda_priv=float(_get("lambda_priv", 0.3, ra)),
        w_cloud=float(_get("w_cloud", 1.0, ra)),
        w_bw=float(_get("w_bw", 1.0, ra)),
        w_priv=float(_get("w_priv", 1.0, ra)),
        confidence_threshold=float(_get("confidence_threshold", 0.5, ra, dd)),
        client=client,
        split_profiles=split_profiles or {},
        max_tolerable_privacy=float(_get("max_tolerable_privacy", 1.0, ra)),
        base_train_cost=float(_get("base_train_cost", 10.0, ra)),
    )
    ra = getattr(config, "resource_aware_trigger", None)
    dd = getattr(config, "drift_detection", None)

    def _get(key, default, *sources):
        for src in sources:
            if src is not None:
                v = getattr(src, key, None)
                if v is not None:
                    return v
        return default

    return ResourceAwareCLTrigger(
        pi_bar=float(_get("pi_bar", 0.1, ra, dd)),
        V=float(_get("V", 10.0, ra, dd)),
        K_p=float(_get("K_p", 1.0, ra, dd)),
        K_d=float(_get("K_d", 0.5, ra, dd)),
        lambda_cloud=float(_get("lambda_cloud", 0.5, ra)),
        lambda_bw=float(_get("lambda_bw", 0.5, ra)),
        lambda_priv=float(_get("lambda_priv", 0.3, ra)),
        w_cloud=float(_get("w_cloud", 1.0, ra)),
        w_bw=float(_get("w_bw", 1.0, ra)),
        w_priv=float(_get("w_priv", 1.0, ra)),
        confidence_threshold=float(_get("confidence_threshold", 0.5, ra, dd)),
    )
