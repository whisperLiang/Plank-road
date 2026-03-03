"""
Tests for model_zoo models with UniversalModelSplitter — split, inference & training.

Verifies that real-world model architectures from model_zoo.py (and their
underlying torchvision components) can be:

1. Traced via torchlens
2. Split at various points
3. Run through edge→cloud inference with faithful output
4. Trained via split-learning (freeze head, train tail)

Models tested
-------------
* ResNet-18          — standard classification backbone
* MobileNet-V2       — lightweight classification backbone
* VGG-11             — purely sequential classification model
* Faster R-CNN backbone (ResNet-50 via model_zoo ``build_detection_model``)
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
    import torchvision.models as tv_models
    _HAS_TV = True
except ImportError:
    _HAS_TV = False

try:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    _HAS_TV_DET = True
except ImportError:
    _HAS_TV_DET = False

try:
    from model_management.model_zoo import (
        build_detection_model,
        list_available_models,
        get_model_family,
        is_wrapper_model,
        model_has_roi_heads,
        _HAS_TV_DETECTION,
    )
    _HAS_ZOO = True
except ImportError:
    _HAS_ZOO = False

_skip = pytest.mark.skipif(
    not (_IMPORT_OK and _HAS_TORCHLENS and _HAS_TV),
    reason="Requires universal_model_split + torchlens + torchvision",
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _trace_and_split(model: nn.Module, x: torch.Tensor, split_ratio: float = 0.5):
    """Trace + split at ~split_ratio of total layers and return splitter.

    Avoids splitting at buffer / constant layers (no ``func_applied``)
    since those are not meaningful computation boundaries.
    """
    sp = UniversalModelSplitter(device="cpu")
    sp.trace(model, x)
    n = sp.num_layers()
    target = max(1, min(int(n * split_ratio), n - 2))

    # Walk from target towards the end until we find a non-buffer layer
    layers = sp.list_layers()
    idx = target
    while idx < n - 1:
        info = layers[idx]
        # Buffer layers have func_name == '' or 'None'; prefer parametric layers
        if info.has_params:
            break
        idx += 1
    else:
        # Fall back: walk backward from target
        idx = target
        while idx > 0 and not layers[idx].has_params:
            idx -= 1

    sp.split(layer_index=idx)
    return sp, n, idx


# ═══════════════════════════════════════════════════════════════════════════
# 1. ResNet-18  (classification, residual connections)
# ═══════════════════════════════════════════════════════════════════════════

@_skip
class TestResNet18Split:
    """Full split-learning lifecycle on ResNet-18."""

    @pytest.fixture(scope="class")
    def model_and_input(self):
        m = tv_models.resnet18(weights=None)
        m.eval()
        x = torch.randn(1, 3, 32, 32)
        return m, x

    def test_trace_and_layer_count(self, model_and_input):
        m, x = model_and_input
        sp = UniversalModelSplitter(device="cpu")
        sp.trace(m, x)
        n = sp.num_layers()
        assert n > 50, f"ResNet-18 should have many layers, got {n}"
        layers = sp.list_layers()
        assert len(layers) == n

    def test_full_forward_matches_original(self, model_and_input):
        m, x = model_and_input
        sp = UniversalModelSplitter(device="cpu")
        sp.trace(m, x)
        with torch.no_grad():
            expected = m(x)
            replayed = sp.full_forward(x)
        assert torch.allclose(expected, replayed, atol=1e-5), (
            f"max diff = {(expected - replayed).abs().max().item()}"
        )

    def test_split_edge_cloud_forward(self, model_and_input):
        m, x = model_and_input
        sp, n, idx = _trace_and_split(m, x)
        inter = sp.edge_forward(x)
        assert isinstance(inter, torch.Tensor)
        assert inter.dim() >= 1
        out = sp.cloud_forward(inter)
        assert out.shape == (1, 1000)

    def test_split_forward_matches_original(self, model_and_input):
        """edge_forward → cloud_forward should equal model(x)."""
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        with torch.no_grad():
            expected = m(x)
            inter = sp.edge_forward(x)
            out = sp.cloud_forward(inter)
        assert torch.allclose(expected, out, atol=1e-5)

    def test_split_at_multiple_points(self, model_and_input):
        """Try splitting at 25%, 50%, 75% of layers."""
        m, x = model_and_input
        for ratio in (0.25, 0.5, 0.75):
            sp, n, idx = _trace_and_split(m, x, split_ratio=ratio)
            inter = sp.edge_forward(x)
            out = sp.cloud_forward(inter)
            assert out.shape == (1, 1000), f"Failed at ratio={ratio}"

    def test_cloud_train_step(self, model_and_input):
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        sp.freeze_head()
        sp.unfreeze_tail()
        inter = sp.edge_forward(x)
        target = torch.tensor([5])
        loss_fn = nn.CrossEntropyLoss()
        out, loss = sp.cloud_train_step(inter, target, loss_fn)
        assert loss.dim() == 0
        assert loss.item() > 0
        loss.backward()

    def test_trainable_params_exist(self, model_and_input):
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        sp.freeze_head()
        sp.unfreeze_tail()
        params = sp.get_tail_trainable_params()
        assert len(params) > 0, "Tail should have trainable params"
        for p in params:
            assert p.requires_grad

    def test_split_retrain(self, model_and_input):
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        train_data = [(torch.randn(1, 3, 32, 32), torch.tensor([i % 10])) for i in range(4)]
        loss_fn = nn.CrossEntropyLoss()
        losses = sp.split_retrain(train_data, loss_fn, num_epochs=1, lr=0.01)
        assert len(losses) == 1
        assert losses[0] > 0

    def test_tail_state_dict_round_trip(self, model_and_input):
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        sd = sp.get_tail_state_dict()
        assert isinstance(sd, dict)
        assert len(sd) > 0
        sp.load_tail_state_dict(sd)

    def test_serialise_intermediate(self, model_and_input):
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        inter = sp.edge_forward(x)
        data = sp.serialise_intermediate(inter)
        recovered = sp.deserialise_intermediate(data)
        assert torch.allclose(inter, recovered)


# ═══════════════════════════════════════════════════════════════════════════
# 2. MobileNet-V2  (lightweight, inverted residual blocks)
# ═══════════════════════════════════════════════════════════════════════════

@_skip
class TestMobileNetV2Split:

    @pytest.fixture(scope="class")
    def model_and_input(self):
        m = tv_models.mobilenet_v2(weights=None)
        m.eval()
        x = torch.randn(1, 3, 32, 32)
        return m, x

    def test_trace_layers(self, model_and_input):
        m, x = model_and_input
        sp = UniversalModelSplitter(device="cpu")
        sp.trace(m, x)
        assert sp.num_layers() > 100

    def test_full_forward_matches(self, model_and_input):
        m, x = model_and_input
        sp = UniversalModelSplitter(device="cpu")
        sp.trace(m, x)
        with torch.no_grad():
            expected = m(x)
            replayed = sp.full_forward(x)
        assert torch.allclose(expected, replayed, atol=1e-5)

    def test_split_inference(self, model_and_input):
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        with torch.no_grad():
            expected = m(x)
            inter = sp.edge_forward(x)
            out = sp.cloud_forward(inter)
        assert torch.allclose(expected, out, atol=1e-5)

    def test_split_train_step(self, model_and_input):
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        sp.freeze_head()
        sp.unfreeze_tail()
        inter = sp.edge_forward(x)
        loss_fn = nn.CrossEntropyLoss()
        out, loss = sp.cloud_train_step(inter, torch.tensor([0]), loss_fn)
        assert loss.dim() == 0
        loss.backward()

    def test_split_retrain(self, model_and_input):
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        data = [(torch.randn(1, 3, 32, 32), torch.tensor([i % 5])) for i in range(3)]
        losses = sp.split_retrain(data, nn.CrossEntropyLoss(), num_epochs=1, lr=0.01)
        assert len(losses) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 3. VGG-11  (purely sequential architecture)
# ═══════════════════════════════════════════════════════════════════════════

@_skip
class TestVGG11Split:

    @pytest.fixture(scope="class")
    def model_and_input(self):
        m = tv_models.vgg11(weights=None)
        m.eval()
        x = torch.randn(1, 3, 32, 32)
        return m, x

    def test_full_forward_matches(self, model_and_input):
        m, x = model_and_input
        sp = UniversalModelSplitter(device="cpu")
        sp.trace(m, x)
        with torch.no_grad():
            expected = m(x)
            replayed = sp.full_forward(x)
        assert torch.allclose(expected, replayed, atol=1e-5)

    def test_split_inference(self, model_and_input):
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        with torch.no_grad():
            expected = m(x)
            inter = sp.edge_forward(x)
            out = sp.cloud_forward(inter)
        assert torch.allclose(expected, out, atol=1e-5)

    def test_split_train_and_retrain(self, model_and_input):
        m, x = model_and_input
        sp, _, _ = _trace_and_split(m, x)
        data = [(torch.randn(1, 3, 32, 32), torch.tensor([3])) for _ in range(3)]
        losses = sp.split_retrain(data, nn.CrossEntropyLoss(), num_epochs=1, lr=0.01)
        assert len(losses) == 1
        assert losses[0] > 0


# ═══════════════════════════════════════════════════════════════════════════
# 4. Faster R-CNN backbone (from model_zoo / torchvision detection)
# ═══════════════════════════════════════════════════════════════════════════

_skip_det = pytest.mark.skipif(
    not (_IMPORT_OK and _HAS_TORCHLENS and _HAS_TV_DET),
    reason="Requires torchlens + torchvision detection",
)


@_skip_det
class TestFasterRCNNBackboneSplit:
    """Split the ResNet-50 backbone as used in Faster R-CNN detection models
    built by ``model_zoo.build_detection_model('fasterrcnn_resnet50_fpn')``.
    """

    @pytest.fixture(scope="class")
    def backbone_and_input(self):
        det = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        body = det.backbone.body  # IntermediateLayerGetter(ResNet50)
        body.eval()
        x = torch.randn(1, 3, 64, 64)
        return body, x

    def test_trace_layers(self, backbone_and_input):
        body, x = backbone_and_input
        sp = UniversalModelSplitter(device="cpu")
        sp.trace(body, x)
        assert sp.num_layers() > 100

    def test_split_inference(self, backbone_and_input):
        body, x = backbone_and_input
        sp, n, idx = _trace_and_split(body, x)
        inter = sp.edge_forward(x)
        assert isinstance(inter, torch.Tensor)
        out = sp.cloud_forward(inter)
        assert isinstance(out, torch.Tensor)

    def test_split_train_step(self, backbone_and_input):
        body, x = backbone_and_input
        sp, _, _ = _trace_and_split(body, x)
        sp.freeze_head()
        sp.unfreeze_tail()
        params = sp.get_tail_trainable_params()
        assert len(params) > 0
        inter = sp.edge_forward(x)
        # Use MSE loss against a dummy target sized like the output
        out, _ = sp.cloud_train_step(inter, None, None)
        # Compute a manual loss so backward flows through the replay graph
        loss = out.sum()
        loss.backward()
        # At least some parameters should receive gradients
        has_grad = any(p.grad is not None for p in params)
        assert has_grad, "At least some tail params should have gradients"

    def test_state_dict_round_trip(self, backbone_and_input):
        body, x = backbone_and_input
        sp, _, _ = _trace_and_split(body, x)
        sd = sp.get_tail_state_dict()
        assert len(sd) > 0
        sp.load_tail_state_dict(sd)


# ═══════════════════════════════════════════════════════════════════════════
# 5. model_zoo helper functions
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_ZOO, reason="model_zoo import failed")
class TestModelZooHelpers:

    def test_list_available_models(self):
        models = list_available_models()
        assert isinstance(models, list)
        # At minimum, torchvision built-ins should be present if available
        if _HAS_TV_DETECTION:
            assert "fasterrcnn_resnet50_fpn" in models

    def test_get_model_family(self):
        assert get_model_family("fasterrcnn_resnet50_fpn") == "fasterrcnn"
        assert get_model_family("retinanet_resnet50_fpn") == "retinanet"
        assert get_model_family("ssd300_vgg16") == "ssd"
        assert get_model_family("fcos_resnet50_fpn") == "fcos"
        assert get_model_family("yolov8s") == "yolo"
        assert get_model_family("detr_resnet50") == "detr"
        assert get_model_family("rtdetr_l") == "rtdetr"
        assert get_model_family("unknown_model") == "unknown"

    def test_is_wrapper_model(self):
        assert is_wrapper_model("yolov8s") is True
        assert is_wrapper_model("detr_resnet50") is True
        assert is_wrapper_model("rtdetr_l") is True
        assert is_wrapper_model("fasterrcnn_resnet50_fpn") is False

    def test_model_has_roi_heads(self):
        assert model_has_roi_heads("fasterrcnn_resnet50_fpn") is True
        assert model_has_roi_heads("retinanet_resnet50_fpn") is False
        assert model_has_roi_heads("yolov8s") is False

    @pytest.mark.skipif(not _HAS_TV_DETECTION, reason="torchvision detection not available")
    def test_build_fasterrcnn(self):
        model = build_detection_model(
            "fasterrcnn_resnet50_fpn", pretrained=False, device="cpu",
        )
        assert isinstance(model, nn.Module)
        assert hasattr(model, "backbone")
        assert hasattr(model, "roi_heads")

    @pytest.mark.skipif(not _HAS_TV_DETECTION, reason="torchvision detection not available")
    def test_build_retinanet(self):
        model = build_detection_model(
            "retinanet_resnet50_fpn", pretrained=False, device="cpu",
        )
        assert isinstance(model, nn.Module)

    def test_build_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown detection model"):
            build_detection_model("totally_fake_model_xyz")


# ═══════════════════════════════════════════════════════════════════════════
# 6. Cross-model: shared split-inference consistency checks
# ═══════════════════════════════════════════════════════════════════════════

@_skip
class TestCrossModelConsistency:
    """Parameterized checks that hold for every traceable model."""

    @pytest.fixture(
        params=[
            ("resnet18", lambda: tv_models.resnet18(weights=None)),
            ("mobilenet_v2", lambda: tv_models.mobilenet_v2(weights=None)),
            ("vgg11", lambda: tv_models.vgg11(weights=None)),
        ],
        ids=["resnet18", "mobilenet_v2", "vgg11"],
    )
    def model_info(self, request):
        name, factory = request.param
        m = factory()
        m.eval()
        x = torch.randn(1, 3, 32, 32)
        return name, m, x

    def test_edge_cloud_output_matches_model(self, model_info):
        """For every model, edge→cloud output must match model(x)."""
        name, m, x = model_info
        sp, _, _ = _trace_and_split(m, x)
        with torch.no_grad():
            expected = m(x)
            inter = sp.edge_forward(x)
            out = sp.cloud_forward(inter)
        assert torch.allclose(expected, out, atol=1e-5), (
            f"{name}: max diff = {(expected - out).abs().max().item()}"
        )

    def test_intermediate_smaller_than_input(self, model_info):
        """Intermediate activations at mid-split should differ from input shape."""
        name, m, x = model_info
        sp, _, _ = _trace_and_split(m, x)
        inter = sp.edge_forward(x)
        # Intermediate should be a different shape than the input
        assert inter.shape != x.shape, f"{name}: intermediate same shape as input"

    def test_train_step_produces_nonzero_loss(self, model_info):
        name, m, x = model_info
        sp, _, _ = _trace_and_split(m, x)
        sp.freeze_head()
        sp.unfreeze_tail()
        if not sp.get_tail_trainable_params():
            pytest.skip(f"{name}: no trainable params in tail at this split point")
        inter = sp.edge_forward(x)
        loss_fn = nn.CrossEntropyLoss()
        out, loss = sp.cloud_train_step(inter, torch.tensor([0]), loss_fn)
        assert loss.item() > 0
