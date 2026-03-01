"""
Resource-Aware Continual Learning Trigger & Split-Point Selector
================================================================

Extends the RCCDA (NeurIPS 2025) Lyapunov drift-plus-penalty framework
to jointly decide:

    1. **Whether to trigger continual learning** — considering cloud
       resource state, network bandwidth, intermediate-feature privacy
       leakage, and edge-side loss degradation.
    2. **Where to place the split point** — selecting the layer that best
       balances latency, bandwidth cost, privacy leakage, and cloud
       compute cost under the Lyapunov objective.

Reference
---------
Adampi210 / RCCDA_resource_constrained_concept_drift_adaptation_code
"Adaptive Model Updates in the Presence of Concept Drift under a
Constrained Resource Budget" — NeurIPS 2025.

Multi-Queue Lyapunov Formulation
---------------------------------
We maintain **four virtual queues**:

    Q_update(t+1) = max(0, Q_update(t) + u(t) − π̄)
        → guarantees  avg(retrains) ≤ π̄

    Q_cloud(t+1) = max(0, Q_cloud(t) + c_cloud(t)·u(t) − λ_cloud)
        → guarantees  avg cloud cost  ≤ λ_cloud

    Q_bw(t+1) = max(0, Q_bw(t) + c_bw(t)·u(t) − λ_bw)
        → bounds average bandwidth usage

    Q_priv(t+1) = max(0, Q_priv(t) + l_priv(t)·u(t) − λ_priv)
        → bounds cumulative privacy leakage

At each decision epoch *t* we solve the Lyapunov drift-plus-penalty
minimisation greedily:

    u*(t) = 1   iff   V · penalty(t) > Σ_q w_q · Q_q(t) + 0.5

where  penalty(t) = K_p · e_t + K_d · Δe_t   is the PD loss term from
RCCDA, and the sum runs over all queues weighted by the respective
instantaneous costs.

Split-point selection is performed analogously: for each candidate layer
*k*, compute a Lyapunov-weighted score and pick the maximiser.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger


# ── helpers ──────────────────────────────────────────────────────────

def _score_to_loss(avg_score: float, threshold: float = 0.5) -> float:
    """Map mean detection confidence to a [0, 1] loss proxy."""
    if avg_score is None or avg_score <= 0:
        return 1.0
    return float(max(0.0, (threshold - avg_score) / threshold))


def _estimate_privacy_leakage(
    feature_numel: int,
    input_numel: int = 3 * 224 * 224,
) -> float:
    """Heuristic privacy-leakage proxy.

    Uses the compression ratio  dim(smashed) / dim(input) as a [0,1]
    upper-bound surrogate for mutual information between the raw input
    and the intermediate representation.  A deeper split (smaller
    smashed data) → lower leakage.

    For a more rigorous estimate one could plug in a trained mutual-
    information neural estimator (MINE), but this simple ratio suffices
    for the Lyapunov decision and matches the HSFL strategy.
    """
    if input_numel <= 0:
        return 1.0
    return float(min(1.0, feature_numel / input_numel))


def _estimate_bandwidth_cost(
    feature_bytes: int,
    bandwidth_mbps: float = 10.0,
) -> float:
    """Normalised transmission cost in [0, 1].

    = transfer_time / 1 second, clamped to [0, 1].
    """
    if bandwidth_mbps <= 0:
        return 1.0
    transfer_sec = (feature_bytes / 1e6) / (bandwidth_mbps / 8.0)
    return float(min(1.0, transfer_sec))


def _cloud_cost_from_state(cloud_state: "CloudResourceState") -> float:
    """Scalar ∈ [0, 1] summarising how *busy* the cloud is.

    Uses a weighted combination of GPU utilisation, CPU utilisation,
    memory pressure, and training-queue depth.
    """
    gpu = cloud_state.gpu_utilization       # [0, 1]
    cpu = cloud_state.cpu_utilization       # [0, 1]
    mem = cloud_state.memory_utilization    # [0, 1]
    q_depth = min(cloud_state.train_queue_size / max(cloud_state.max_queue_size, 1), 1.0)

    # Weighted combination — GPU is the most critical resource
    cost = 0.4 * gpu + 0.2 * cpu + 0.2 * mem + 0.2 * q_depth
    return float(min(1.0, cost))


# ── data classes ─────────────────────────────────────────────────────

@dataclass
class CloudResourceState:
    """Snapshot of the cloud server's resource utilisation.

    All utilisation values are fractions in [0.0, 1.0].
    """
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    train_queue_size: int = 0
    max_queue_size: int = 10
    # Optional: measured round-trip latency (seconds)
    rtt_seconds: float = 0.0
    timestamp: float = 0.0

    def is_stale(self, max_age_sec: float = 30.0) -> bool:
        """Return True if the snapshot is older than *max_age_sec*."""
        return (time.time() - self.timestamp) > max_age_sec


@dataclass
class SplitCandidate:
    """Metadata for one candidate split point."""
    layer_index: int
    layer_name: str = ""
    # Smashed-data size in elements
    feature_numel: int = 0
    # Estimated bytes to transmit
    feature_bytes: int = 0
    # Fraction of total FLOPs executed on the edge
    edge_flops_ratio: float = 0.5
    # Privacy leakage proxy ∈ [0, 1]
    privacy_leakage: float = 0.5
    # Estimated cloud computation cost ∈ [0, 1]
    cloud_compute_cost: float = 0.5


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

    # ──────── CL trigger decision ────────────────────────────────────

    def should_trigger_cl(
        self,
        avg_confidence: float,
        cloud_state: Optional[CloudResourceState] = None,
        bandwidth_mbps: float = 10.0,
        feature_bytes: int = 0,
        feature_numel: int = 0,
    ) -> bool:
        """Decide whether to trigger continual learning this round.

        Parameters
        ----------
        avg_confidence : float
            Mean detection confidence from the latest inference window.
        cloud_state : CloudResourceState, optional
            Latest cloud resource snapshot.  If *None*, the cloud-cost
            queue is updated with zero cost (optimistic).
        bandwidth_mbps : float
            Estimated current bandwidth in Mbps.
        feature_bytes : int
            Bytes of the intermediate feature that would be transmitted.
        feature_numel : int
            Number of elements in the intermediate feature (for
            privacy estimation).

        Returns
        -------
        bool
            True → trigger CL this round.
        """
        # ── Loss proxy ──
        loss_curr = _score_to_loss(avg_confidence, self.confidence_threshold)
        if self.loss_initial is None:
            self.loss_initial = loss_curr
        if loss_curr < self.loss_best:
            self.loss_best = loss_curr

        e_t = loss_curr - self.loss_best           # ≥ 0
        delta_e = loss_curr - self.loss_prev       # positive → degrading

        # ── PD penalty (performance pressure to update) ──
        pd_term = self.K_p * e_t + self.K_d * delta_e

        # ── Instantaneous costs ──
        c_cloud = _cloud_cost_from_state(cloud_state) if cloud_state else 0.0
        c_bw = _estimate_bandwidth_cost(feature_bytes, bandwidth_mbps)
        c_priv = _estimate_privacy_leakage(feature_numel) if feature_numel > 0 else 0.0

        # ── Lyapunov drift-plus-penalty condition ──
        # The "benefit" of triggering: V * pd_term
        # The "threshold" accumulates queue back-pressure:
        threshold = (
            self.Q_update
            + self.w_cloud * self.Q_cloud * c_cloud
            + self.w_bw * self.Q_bw * c_bw
            + self.w_priv * self.Q_priv * c_priv
            + 0.5 - self.pi_bar
        )

        should_update: bool = (self.V * pd_term) > threshold

        # ── Update virtual queues ──
        u = 1.0 if should_update else 0.0
        self.Q_update = max(0.0, self.Q_update + u - self.pi_bar)
        self.Q_cloud = max(0.0, self.Q_cloud + c_cloud * u - self.lambda_cloud)
        self.Q_bw = max(0.0, self.Q_bw + c_bw * u - self.lambda_bw)
        self.Q_priv = max(0.0, self.Q_priv + c_priv * u - self.lambda_priv)

        # ── Book-keeping ──
        self.loss_prev = loss_curr
        self.step += 1
        if should_update:
            self.trigger_count += 1

        record = {
            "step": self.step,
            "loss": loss_curr,
            "e_t": e_t,
            "delta_e": delta_e,
            "pd_term": pd_term,
            "c_cloud": c_cloud,
            "c_bw": c_bw,
            "c_priv": c_priv,
            "Q_update": self.Q_update,
            "Q_cloud": self.Q_cloud,
            "Q_bw": self.Q_bw,
            "Q_priv": self.Q_priv,
            "threshold": threshold,
            "V_pd": self.V * pd_term,
            "decision": should_update,
        }
        self.history.append(record)

        if should_update:
            logger.info(
                "[ResourceCLTrigger] CL triggered at step {}: "
                "loss={:.4f}, pd={:.4f}, threshold={:.3f}, "
                "Q=[u={:.2f}, c={:.2f}, bw={:.2f}, p={:.2f}]",
                self.step, loss_curr, pd_term, threshold,
                self.Q_update, self.Q_cloud, self.Q_bw, self.Q_priv,
            )
        return should_update

    # ──────── Split-point selection ──────────────────────────────────

    def select_split_point(
        self,
        candidates: List[SplitCandidate],
        cloud_state: Optional[CloudResourceState] = None,
        bandwidth_mbps: float = 10.0,
    ) -> int:
        """Choose the best split point via Lyapunov-weighted objective.

        For each candidate *k* we compute:

            score(k) = − w_cloud · Q_cloud · cloud_cost(k)
                       − w_bw    · Q_bw    · bw_cost(k)
                       − w_priv  · Q_priv  · privacy(k)

        The candidate with the **highest** (least negative) score is
        selected.  When all queues are near zero, this degenerates to a
        simple "minimise total cost" selector.

        Parameters
        ----------
        candidates : list of SplitCandidate
            Pre-computed metadata for each candidate layer.
        cloud_state : CloudResourceState, optional
        bandwidth_mbps : float
            Estimated current bandwidth in Mbps.

        Returns
        -------
        int
            Layer index of the selected split point.
        """
        if not candidates:
            raise ValueError("No split candidates provided")

        if len(candidates) == 1:
            return candidates[0].layer_index

        c_cloud_base = _cloud_cost_from_state(cloud_state) if cloud_state else 0.0

        best_score = float("-inf")
        best_idx = candidates[0].layer_index

        for cand in candidates:
            bw_cost = _estimate_bandwidth_cost(cand.feature_bytes, bandwidth_mbps)
            # Cloud compute cost scales with the fraction of FLOPs on the cloud
            cloud_cost = c_cloud_base * (1.0 - cand.edge_flops_ratio)
            priv = cand.privacy_leakage

            score = (
                - self.w_cloud * self.Q_cloud * cloud_cost
                - self.w_bw * self.Q_bw * bw_cost
                - self.w_priv * self.Q_priv * priv
            )

            if score > best_score:
                best_score = score
                best_idx = cand.layer_index

        logger.info(
            "[ResourceCLTrigger] Selected split point: layer {} "
            "(score={:.4f}, Q_cloud={:.2f}, Q_bw={:.2f}, Q_priv={:.2f})",
            best_idx, best_score, self.Q_cloud, self.Q_bw, self.Q_priv,
        )
        return best_idx

    # ──────── Combined: trigger + split selection ────────────────────

    def decide(
        self,
        avg_confidence: float,
        cloud_state: Optional[CloudResourceState] = None,
        bandwidth_mbps: float = 10.0,
        feature_bytes: int = 0,
        feature_numel: int = 0,
        split_candidates: Optional[List[SplitCandidate]] = None,
    ) -> Tuple[bool, Optional[int]]:
        """Joint decision: trigger CL + (optionally) select split point.

        Returns
        -------
        (should_trigger, selected_split_layer_index_or_None)
        """
        trigger = self.should_trigger_cl(
            avg_confidence=avg_confidence,
            cloud_state=cloud_state,
            bandwidth_mbps=bandwidth_mbps,
            feature_bytes=feature_bytes,
            feature_numel=feature_numel,
        )

        selected_split: Optional[int] = None
        if trigger and split_candidates:
            selected_split = self.select_split_point(
                candidates=split_candidates,
                cloud_state=cloud_state,
                bandwidth_mbps=bandwidth_mbps,
            )

        return trigger, selected_split

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


# ── Helpers: build SplitCandidate list from UniversalModelSplitter ───

def build_split_candidates(
    splitter,
    input_numel: int = 3 * 224 * 224,
    only_parametric: bool = False,
) -> List[SplitCandidate]:
    """Create ``SplitCandidate`` list from a traced ``UniversalModelSplitter``.

    Parameters
    ----------
    splitter : UniversalModelSplitter
        Must have been traced (``splitter.layer_info`` populated).
    input_numel : int
        Number of elements in the model input (for privacy estimation).
    only_parametric : bool
        If True, only consider layers with trainable parameters.

    Returns
    -------
    list of SplitCandidate
    """
    candidates: List[SplitCandidate] = []
    layers = splitter.layer_info
    if not layers:
        return candidates

    total_flops = sum(l.flops for l in layers) or 1.0

    cum_flops = 0.0
    for i, linfo in enumerate(layers):
        cum_flops += linfo.flops

        if only_parametric and linfo.params == 0:
            continue

        # Estimate smashed-data size
        numel = 1
        for d in linfo.output_shape:
            numel *= d
        feat_bytes = numel * 4  # float32

        candidates.append(SplitCandidate(
            layer_index=i,
            layer_name=linfo.label,
            feature_numel=numel,
            feature_bytes=feat_bytes,
            edge_flops_ratio=cum_flops / total_flops,
            privacy_leakage=_estimate_privacy_leakage(numel, input_numel),
            cloud_compute_cost=1.0 - cum_flops / total_flops,
        ))

    return candidates


# ── Query cloud resource helper ──────────────────────────────────────

def query_cloud_resource(server_ip: str, timeout_sec: float = 5.0) -> Optional[CloudResourceState]:
    """Query the cloud server for its resource utilisation via gRPC.

    Returns *None* on failure (timeout, network error, etc.).
    """
    try:
        import grpc
        from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc

        channel = grpc.insecure_channel(server_ip)
        stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
        req = message_transmission_pb2.ResourceRequest(edge_id=0)
        reply = stub.query_resource(req, timeout=timeout_sec)
        return CloudResourceState(
            cpu_utilization=reply.cpu_utilization,
            gpu_utilization=reply.gpu_utilization,
            memory_utilization=reply.memory_utilization,
            train_queue_size=int(reply.train_queue_size),
            max_queue_size=int(reply.max_queue_size) or 10,
            rtt_seconds=0.0,
            timestamp=time.time(),
        )
    except Exception as exc:
        logger.debug("[ResourceCLTrigger] query_cloud_resource failed: {}", exc)
        return None


def estimate_bandwidth(server_ip: str, probe_bytes: int = 4096) -> float:
    """Rough bandwidth estimate (Mbps) via a small probe message.

    Falls back to a conservative default on failure.
    """
    try:
        import grpc
        from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc

        payload = "x" * probe_bytes
        channel = grpc.insecure_channel(server_ip)
        stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
        req = message_transmission_pb2.BandwidthProbeRequest(payload=payload)
        t0 = time.time()
        reply = stub.bandwidth_probe(req, timeout=5.0)
        rtt = time.time() - t0
        if rtt <= 0:
            return 10.0
        # Round-trip bytes / half-RTT → estimate one-way throughput
        mbps = (probe_bytes * 8 / 1e6) / (rtt / 2.0)
        return max(0.1, mbps)
    except Exception:
        return 10.0  # conservative default


# ── Factory ──────────────────────────────────────────────────────────

def create_resource_aware_trigger(config) -> ResourceAwareCLTrigger:
    """Build a ``ResourceAwareCLTrigger`` from the config object.

    Reads ``config.resource_aware_trigger`` (or falls back to
    ``config.drift_detection`` for shared RCCDA parameters).
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
    )
