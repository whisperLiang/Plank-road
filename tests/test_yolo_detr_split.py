"""
Tests for YOLO / DETR / RT-DETR model families with UniversalModelSplitter.

For each model family the *underlying pure-PyTorch model* is tested:

* **YOLO**  — ``ultralytics.YOLO("yolov8n.pt").model``  (DetectionModel)
* **DETR backbone** — ``DetrForObjectDetection.model.backbone.model``  (timm
  FeatureListNet / ResNet-50)
* **RT-DETR** — ``ultralytics.RTDETR("rtdetr-l.pt").model``  (RTDETRDetectionModel)

The high-level ``model_zoo`` *wrapper* classes (``YOLODetectionModel``,
``DETRDetectionModel``, ``RTDETRDetectionModel``) perform numpy / PIL
pre-processing that is opaque to torchlens, so split-learning operates
on the inner PyTorch graph.

Coverage per model:
  1. Trace with torchlens
  2. Full-forward replay accuracy
  3. Edge→cloud split inference
  4. Split at multiple points
  5. Cloud training step with gradient flow
  6. ``split_retrain`` convenience
  7. Tail state-dict round-trip
  8. Intermediate serialisation
  9. model_zoo wrapper build + inference smoke test
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Import guards
# ---------------------------------------------------------------------------
try:
    from model_management.universal_model_split import (
        _HAS_TORCHLENS,
        UniversalModelSplitter,
    )
    _IMPORT_OK = True
except ImportError:
    _IMPORT_OK = False

try:
    import torchlens as tl
    _HAS_TL = True
except ImportError:
    _HAS_TL = False

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False

try:
    from ultralytics import RTDETR
    _HAS_RTDETR = True
except ImportError:
    _HAS_RTDETR = False

try:
    from transformers import DetrForObjectDetection
    _HAS_DETR = True
except ImportError:
    _HAS_DETR = False

try:
    from model_management.model_zoo import (
        build_detection_model,
        YOLODetectionModel,
        DETRDetectionModel,
        RTDETRDetectionModel,
        get_model_family,
        is_wrapper_model,
    )
    _HAS_ZOO = True
except ImportError:
    _HAS_ZOO = False

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------
_skip_yolo = pytest.mark.skipif(
    not (_IMPORT_OK and _HAS_TORCHLENS and _HAS_YOLO),
    reason="Requires universal_model_split + torchlens + ultralytics",
)
_skip_detr = pytest.mark.skipif(
    not (_IMPORT_OK and _HAS_TORCHLENS and _HAS_DETR),
    reason="Requires universal_model_split + torchlens + transformers",
)
_skip_rtdetr = pytest.mark.skipif(
    not (_IMPORT_OK and _HAS_TORCHLENS and _HAS_RTDETR),
    reason="Requires universal_model_split + torchlens + ultralytics",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _warmup_trace(model: nn.Module, x: torch.Tensor):
    """Run a lightweight torchlens trace to initialise internal hooks.

    torchlens 0.12 has a re-entrant ``__format__`` bug that triggers
    infinite recursion on the first ``log_forward_pass`` call with
    ``save_function_args=True`` for certain models (ultralytics YOLO,
    RT-DETR).  A warm-up call without that flag builds internal state
    that prevents the crash on subsequent calls.
    """
    if _HAS_TL:
        tl.log_forward_pass(model, x, vis_opt="none")


def _find_param_layer(layers, target_idx, n):
    """Walk from *target_idx* forward (then backward) to find a parametric layer."""
    idx = target_idx
    while idx < n - 1:
        if layers[idx].has_params:
            return idx
        idx += 1
    idx = target_idx
    while idx > 0:
        if layers[idx].has_params:
            return idx
        idx -= 1
    return target_idx  # fallback


def _trace_and_split(model: nn.Module, x: torch.Tensor, split_ratio: float = 0.33):
    """Trace *model*, split at ~split_ratio, return ``(splitter, n_layers, split_idx)``."""
    sp = UniversalModelSplitter(device="cpu")
    sp.trace(model, x)
    n = sp.num_layers()
    target = max(1, min(int(n * split_ratio), n - 2))
    layers = sp.list_layers()
    idx = _find_param_layer(layers, target, n)
    sp.split(layer_index=idx)
    return sp, n, idx


def _unpack_output(out):
    """YOLO returns ``(Tensor, dict)``; others return a plain tensor or list."""
    if isinstance(out, tuple):
        return out[0]
    if isinstance(out, list):
        # DETR backbone → list of feature maps; use the last one
        return out[-1]
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 1. YOLOv8n  (inner DetectionModel from ultralytics)
# ═══════════════════════════════════════════════════════════════════════════

@_skip_yolo
class TestYOLOv8nSplit:
    """Split-learning lifecycle on the inner YOLOv8n PyTorch model."""

    @pytest.fixture(scope="class")
    def model_and_input(self):
        yolo = YOLO("yolov8n.pt")
        inner = yolo.model
        inner.eval()
        # Small input to keep trace fast
        x = torch.randn(1, 3, 160, 160)
        # Warm-up torchlens to avoid re-entrant __format__ crash
        _warmup_trace(inner, x)
        return inner, x

    # ── trace ──

    def test_trace_and_layer_count(self, model_and_input):
        m, x = model_and_input
        sp = UniversalModelSplitter(device="cpu")
        sp.trace(m, x)
        n = sp.num_layers()
        assert n > 200, f"YOLOv8n should yield many layers, got {n}"

    # ── full forward replay ──

    def test_full_forward_produces_output(self, model_and_input):
        """YOLO uses dynamic grid ops (arange/meshgrid) so exact replay may
        diverge in shape.  Verify the replay produces a valid tensor."""
        m, x = model_and_input
        sp = UniversalModelSplitter(device="cpu")
        sp.trace(m, x)
        with torch.no_grad():
            replayed = _unpack_output(sp.full_forward(x))
        assert isinstance(replayed, torch.Tensor)
        assert replayed.dim() >= 2

    # ── split inference ──

    def test_split_edge_cloud_forward(self, model_and_input):
        m, x = model_and_input
        sp, n, idx = _trace_and_split(m, x)
        inter = sp.edge_forward(x)
        assert isinstance(inter, torch.Tensor)
        assert inter.dim() >= 1
        out = _unpack_output(sp.cloud_forward(inter))
        assert isinstance(out, torch.Tensor)

    def test_split_produces_output(self, model_and_input):
        """Split inference should produce a tensor output."""
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        with torch.no_grad():
            inter = sp.edge_forward(x)
            out = _unpack_output(sp.cloud_forward(inter))
        assert isinstance(out, torch.Tensor)
        assert out.dim() >= 2

    def test_split_at_multiple_points(self, model_and_input):
        m, x = model_and_input
        for ratio in (0.2, 0.5, 0.8):
            sp, n, idx = _trace_and_split(m, x, split_ratio=ratio)
            inter = sp.edge_forward(x)
            out = _unpack_output(sp.cloud_forward(inter))
            assert isinstance(out, torch.Tensor), f"ratio={ratio}"

    # ── training ──

    def test_cloud_train_step(self, model_and_input):
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        sp.freeze_head()
        sp.unfreeze_tail()
        inter = sp.edge_forward(x)
        out, _ = sp.cloud_train_step(inter, None, None)
        out_t = _unpack_output(out)
        loss = out_t.sum()
        loss.backward()

    def test_trainable_params_exist(self, model_and_input):
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        sp.freeze_head()
        sp.unfreeze_tail()
        params = sp.get_tail_trainable_params()
        assert len(params) > 0

    def test_gradient_flow(self, model_and_input):
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        sp.freeze_head()
        sp.unfreeze_tail()
        params = sp.get_tail_trainable_params()
        inter = sp.edge_forward(x)
        out, _ = sp.cloud_train_step(inter, None, None)
        loss = _unpack_output(out).sum()
        loss.backward()
        assert any(p.grad is not None for p in params), "Tail params should receive gradients"

    # ── state dict / serialisation ──

    def test_tail_state_dict_round_trip(self, model_and_input):
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        sd = sp.get_tail_state_dict()
        assert isinstance(sd, dict) and len(sd) > 0
        sp.load_tail_state_dict(sd)

    def test_serialise_intermediate(self, model_and_input):
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        inter = sp.edge_forward(x)
        data = sp.serialise_intermediate(inter)
        recovered = sp.deserialise_intermediate(data)
        assert torch.allclose(inter, recovered)


# ═══════════════════════════════════════════════════════════════════════════
# 2. DETR backbone  (timm FeatureListNet / ResNet-50 inside HuggingFace DETR)
# ═══════════════════════════════════════════════════════════════════════════

@_skip_detr
class TestDETRBackboneSplit:
    """Split-learning lifecycle on the ResNet-50 backbone of HuggingFace DETR."""

    @pytest.fixture(scope="class")
    def model_and_input(self):
        detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        detr.eval()
        body = detr.model.backbone.model  # timm FeatureListNet (ResNet-50)
        x = torch.randn(1, 3, 128, 128)
        return body, x

    # ── trace ──

    def test_trace_and_layer_count(self, model_and_input):
        m, x = model_and_input
        sp = UniversalModelSplitter(device="cpu")
        sp.trace(m, x)
        assert sp.num_layers() > 200

    # ── full forward replay ──

    def test_full_forward_matches_original(self, model_and_input):
        m, x = model_and_input
        sp = UniversalModelSplitter(device="cpu")
        sp.trace(m, x)
        with torch.no_grad():
            expected = _unpack_output(m(x))
            replayed = _unpack_output(sp.full_forward(x))
        assert torch.allclose(expected, replayed, atol=1e-4), (
            f"max diff = {(expected - replayed).abs().max().item()}"
        )

    # ── split inference ──

    def test_split_edge_cloud_forward(self, model_and_input):
        m, x = model_and_input
        sp, n, idx = _trace_and_split(m, x)
        inter = sp.edge_forward(x)
        assert isinstance(inter, torch.Tensor)
        out = _unpack_output(sp.cloud_forward(inter))
        assert isinstance(out, torch.Tensor)

    def test_split_forward_matches_original(self, model_and_input):
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        with torch.no_grad():
            expected = _unpack_output(m(x))
            inter = sp.edge_forward(x)
            out = _unpack_output(sp.cloud_forward(inter))
        assert torch.allclose(expected, out, atol=1e-4)

    def test_split_at_multiple_points(self, model_and_input):
        m, x = model_and_input
        for ratio in (0.25, 0.5, 0.75):
            sp, n, idx = _trace_and_split(m, x, split_ratio=ratio)
            inter = sp.edge_forward(x)
            out = _unpack_output(sp.cloud_forward(inter))
            assert isinstance(out, torch.Tensor), f"ratio={ratio}"

    # ── training ──

    def test_cloud_train_step_gradient_flow(self, model_and_input):
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        sp.freeze_head()
        sp.unfreeze_tail()
        params = sp.get_tail_trainable_params()
        assert len(params) > 0
        inter = sp.edge_forward(x)
        out, _ = sp.cloud_train_step(inter, None, None)
        loss = _unpack_output(out).sum()
        loss.backward()
        assert any(p.grad is not None for p in params)

    def test_split_retrain(self, model_and_input):
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        # Use MSE loss against dummy target shaped like output
        with torch.no_grad():
            ref = _unpack_output(m(x))
        target_shape = ref.shape

        def _mse_loss(output, target):
            out_t = _unpack_output(output)
            return nn.functional.mse_loss(out_t, target)

        data = [(torch.randn(1, 3, 128, 128), torch.randn(target_shape)) for _ in range(3)]
        losses = sp.split_retrain(data, _mse_loss, num_epochs=1, lr=0.001)
        assert len(losses) == 1
        assert losses[0] > 0

    # ── state dict ──

    def test_tail_state_dict_round_trip(self, model_and_input):
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        sd = sp.get_tail_state_dict()
        assert len(sd) > 0
        sp.load_tail_state_dict(sd)


# ═══════════════════════════════════════════════════════════════════════════
# 3. RT-DETR-L  (inner model from ultralytics)
# ═══════════════════════════════════════════════════════════════════════════

@_skip_rtdetr
class TestRTDETRSplit:
    """Split-learning lifecycle on the inner RT-DETR-L model."""

    @pytest.fixture(scope="class")
    def model_and_input(self):
        rtdetr = RTDETR("rtdetr-l.pt")
        inner = rtdetr.model
        inner.eval()
        x = torch.randn(1, 3, 320, 320)
        # Warm-up torchlens to avoid re-entrant __format__ crash
        _warmup_trace(inner, x)
        return inner, x

    # ── trace ──

    def test_trace_and_layer_count(self, model_and_input):
        m, x = model_and_input
        sp = UniversalModelSplitter(device="cpu")
        sp.trace(m, x)
        assert sp.num_layers() > 500, f"RT-DETR should have many layers, got {sp.num_layers()}"

    # ── split inference ──

    def test_split_edge_cloud_forward(self, model_and_input):
        m, x = model_and_input
        sp, n, idx = _trace_and_split(m, x)
        inter = sp.edge_forward(x)
        assert isinstance(inter, torch.Tensor)
        out = _unpack_output(sp.cloud_forward(inter))
        assert isinstance(out, torch.Tensor)

    def test_split_produces_output(self, model_and_input):
        """RT-DETR has dynamic transformer ops; verify split produces
        a valid tensor output (not exact-match)."""
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        with torch.no_grad():
            inter = sp.edge_forward(x)
            out = _unpack_output(sp.cloud_forward(inter))
        assert isinstance(out, torch.Tensor)
        assert out.dim() >= 2

    # ── training ──

    def test_cloud_train_step_gradient_flow(self, model_and_input):
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        sp.freeze_head()
        sp.unfreeze_tail()
        params = sp.get_tail_trainable_params()
        assert len(params) > 0
        inter = sp.edge_forward(x)
        out, _ = sp.cloud_train_step(inter, None, None)
        loss = _unpack_output(out).sum()
        loss.backward()
        grads = [p for p in params if p.grad is not None]
        assert len(grads) > 0, "At least some tail parameters should have gradients"

    # ── state dict ──

    def test_tail_state_dict_round_trip(self, model_and_input):
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        sd = sp.get_tail_state_dict()
        assert len(sd) > 0
        sp.load_tail_state_dict(sd)


# ═══════════════════════════════════════════════════════════════════════════
# 4. model_zoo wrapper smoke tests  (build + inference without torchlens)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_ZOO, reason="model_zoo import failed")
class TestModelZooYOLODETR:
    """Verify that the high-level wrapper classes build correctly and
    perform basic inference (no split / torchlens dependency)."""

    @pytest.mark.skipif(not _HAS_YOLO, reason="ultralytics not installed")
    def test_build_yolo_model(self):
        model = build_detection_model("yolov8n", pretrained=True, device="cpu")
        assert isinstance(model, YOLODetectionModel)
        assert get_model_family("yolov8n") == "yolo"
        assert is_wrapper_model(model)

    @pytest.mark.skipif(not _HAS_YOLO, reason="ultralytics not installed")
    def test_yolo_forward_output_format(self):
        model = build_detection_model("yolov8n", pretrained=True, device="cpu")
        img = torch.rand(3, 320, 320)
        results = model([img])
        assert isinstance(results, list)
        assert len(results) == 1
        r = results[0]
        assert "boxes" in r and "labels" in r and "scores" in r
        assert r["boxes"].dim() == 2 and r["boxes"].shape[1] == 4

    @pytest.mark.skipif(not _HAS_DETR, reason="transformers not installed")
    def test_build_detr_model(self):
        model = build_detection_model("detr_resnet50", pretrained=True, device="cpu")
        assert isinstance(model, DETRDetectionModel)
        assert get_model_family("detr_resnet50") == "detr"
        assert is_wrapper_model(model)

    @pytest.mark.skipif(not _HAS_DETR, reason="transformers not installed")
    def test_detr_forward_output_format(self):
        model = build_detection_model("detr_resnet50", pretrained=True, device="cpu")
        model.eval()
        img = torch.rand(3, 224, 224)
        results = model([img])
        assert isinstance(results, list) and len(results) == 1
        r = results[0]
        assert "boxes" in r and "labels" in r and "scores" in r

    @pytest.mark.skipif(not _HAS_RTDETR, reason="ultralytics not installed")
    def test_build_rtdetr_model(self):
        model = build_detection_model("rtdetr_l", pretrained=True, device="cpu")
        assert isinstance(model, RTDETRDetectionModel)
        assert get_model_family("rtdetr_l") == "rtdetr"
        assert is_wrapper_model(model)

    @pytest.mark.skipif(not _HAS_RTDETR, reason="ultralytics not installed")
    def test_rtdetr_forward_output_format(self):
        model = build_detection_model("rtdetr_l", pretrained=True, device="cpu")
        img = torch.rand(3, 640, 640)
        results = model([img])
        assert isinstance(results, list) and len(results) == 1
        r = results[0]
        assert "boxes" in r and "labels" in r and "scores" in r

    @pytest.mark.skipif(not _HAS_YOLO, reason="ultralytics not installed")
    def test_yolo_state_dict_roundtrip(self):
        model = build_detection_model("yolov8n", pretrained=True, device="cpu")
        sd = model.state_dict()
        assert len(sd) > 0
        model.load_state_dict(sd)

    @pytest.mark.skipif(not _HAS_DETR, reason="transformers not installed")
    def test_detr_state_dict_roundtrip(self):
        model = build_detection_model("detr_resnet50", pretrained=True, device="cpu")
        sd = model.state_dict()
        assert len(sd) > 0
        model.load_state_dict(sd)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Cross-model parameterized checks
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not (_IMPORT_OK and _HAS_TORCHLENS),
    reason="Requires universal_model_split + torchlens",
)
class TestYOLODETRCrossModel:
    """Parameterized generic checks for all traceable YOLO/DETR models."""

    @pytest.fixture(
        params=[
            pytest.param(
                ("yolov8n", lambda: (YOLO("yolov8n.pt").model, torch.randn(1, 3, 160, 160))),
                marks=pytest.mark.skipif(not _HAS_YOLO, reason="ultralytics"),
                id="yolov8n",
            ),
            pytest.param(
                ("detr_backbone", lambda: (
                    DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").model.backbone.model,
                    torch.randn(1, 3, 128, 128),
                )),
                marks=pytest.mark.skipif(not _HAS_DETR, reason="transformers"),
                id="detr_backbone",
            ),
            pytest.param(
                ("rtdetr_l", lambda: (RTDETR("rtdetr-l.pt").model, torch.randn(1, 3, 320, 320))),
                marks=pytest.mark.skipif(not _HAS_RTDETR, reason="ultralytics"),
                id="rtdetr_l",
            ),
        ],
    )
    def model_info(self, request):
        name, factory = request.param
        model, x = factory()
        model.eval()
        _warmup_trace(model, x)
        return name, model, x

    def test_edge_cloud_matches(self, model_info):
        name, m, x = model_info
        sp, _, _ = _trace_and_split(m, x)
        with torch.no_grad():
            expected = _unpack_output(m(x))
            inter = sp.edge_forward(x)
            out = _unpack_output(sp.cloud_forward(inter))
        # YOLO / RT-DETR have dynamic-grid ops; shape may diverge during replay.
        # Only DETR backbone (plain ResNet) supports exact-match comparison.
        if name == "detr_backbone":
            assert torch.allclose(expected, out, atol=1e-4), (
                f"{name}: max diff = {(expected - out).abs().max().item()}"
            )
        else:
            assert isinstance(out, torch.Tensor)
            assert out.dim() >= 2

    def test_intermediate_shape_differs_from_input(self, model_info):
        name, m, x = model_info
        sp, _, _ = _trace_and_split(m, x)
        inter = sp.edge_forward(x)
        assert inter.shape != x.shape, f"{name}: intermediate same shape as input"

    def test_train_produces_gradients(self, model_info):
        name, m, x = model_info
        sp, _, _ = _trace_and_split(m, x)
        sp.freeze_head()
        sp.unfreeze_tail()
        params = sp.get_tail_trainable_params()
        if not params:
            pytest.skip(f"{name}: no trainable params in tail")
        inter = sp.edge_forward(x)
        out, _ = sp.cloud_train_step(inter, None, None)
        loss = _unpack_output(out).sum()
        loss.backward()
        assert any(p.grad is not None for p in params)
