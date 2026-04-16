"""
Tests for model_management.activation_sparsity — Dynamic Activation Sparsity.

Covers:
* ActivationClipper (clip / reshape round-trip, varying clip ratios)
* AutoFreezeConv2d (forward match with nn.Conv2d when sparsity off)
* DASBatchNorm2d (forward match with nn.BatchNorm2d when sparsity off)
* AutoFreezeFC (forward match with nn.Linear when sparsity off)
* compute_tgi (gradient importance computation)
* DASTrainer (module replacement, activate/deactivate, memory stats,
  probe_and_set_ratios, das_train_step)
* apply_das_to_model / apply_das_to_tail convenience functions
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from model_management.activation_sparsity import (
    ActivationClipper,
    AutoFreezeConv2d,
    AutoFreezeFC,
    DASBatchNorm2d,
    DASGroupNorm,
    DASLayerNorm,
    DASTrainer,
    apply_das_to_model,
    apply_das_to_tail,
    compute_tgi,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_cnn():
    """A tiny CNN for DASTrainer tests."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 5),
    )


def _norm_augmented_model():
    class _NormAugmentedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1)
            self.gn = nn.GroupNorm(2, 4)
            self.pool = nn.AdaptiveAvgPool2d((2, 2))
            self.flatten = nn.Flatten()
            self.ln = nn.LayerNorm(16)
            self.fc = nn.Linear(16, 3)

        def forward(self, x):
            x = self.conv(x)
            x = self.gn(x)
            x = torch.relu(x)
            x = self.pool(x)
            x = self.flatten(x)
            x = self.ln(x)
            return self.fc(x)

    return _NormAugmentedModel()


# ═══════════════════════════════════════════════════════════════════════════
# 1.  ActivationClipper
# ═══════════════════════════════════════════════════════════════════════════

class TestActivationClipper:

    def test_clip_ratio_zero_keeps_all(self):
        clipper = ActivationClipper(clip_ratio=0.0)
        x = torch.randn(2, 3, 4)

        class Ctx:
            pass
        ctx = Ctx()
        clipped = clipper.clip(x, ctx)
        assert clipped.shape[0] == x.numel()
        restored = clipper.reshape(clipped, ctx)
        # Should recover original data
        assert torch.allclose(restored.view(x.shape), x)

    def test_clip_ratio_half(self):
        clipper = ActivationClipper(clip_ratio=0.5)
        x = torch.arange(10, dtype=torch.float32)

        class Ctx:
            pass
        ctx = Ctx()
        clipped = clipper.clip(x, ctx)
        # keep_n = ceil(10 * 0.5) = 5
        assert clipped.shape[0] == 5
        restored = clipper.reshape(clipped, ctx)
        assert restored.shape[0] == 10
        # The 5 largest-magnitude elements should be preserved
        nonzero = restored.nonzero(as_tuple=True)[0]
        assert len(nonzero) == 5

    def test_clip_ratio_high(self):
        clipper = ActivationClipper(clip_ratio=0.9)
        x = torch.randn(100)

        class Ctx:
            pass
        ctx = Ctx()
        clipped = clipper.clip(x, ctx)
        # keep_n = ceil(100 * (1-0.9)) = 10
        assert clipped.shape[0] == 10

    def test_clip_ratio_one_prunes_all_elements(self):
        clipper = ActivationClipper(clip_ratio=1.0)
        x = torch.arange(6, dtype=torch.float32)

        class Ctx:
            pass
        ctx = Ctx()
        clipped = clipper.clip(x, ctx)
        assert clipped.numel() == 0
        restored = clipper.reshape(clipped, ctx)
        assert torch.equal(restored, torch.zeros_like(x))

    def test_clip_empty_tensor(self):
        clipper = ActivationClipper(clip_ratio=0.5)
        x = torch.tensor([])

        class Ctx:
            pass
        ctx = Ctx()
        clipped = clipper.clip(x, ctx)
        # Empty tensor: clip_ratio > 0 but numel == 0 → returns empty clone
        assert clipped.numel() == 0

    def test_clip_ratio_clamped(self):
        c1 = ActivationClipper(clip_ratio=-0.5)
        assert c1.clip_ratio == 0.0
        c2 = ActivationClipper(clip_ratio=1.5)
        assert c2.clip_ratio == 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 2.  AutoFreezeConv2d
# ═══════════════════════════════════════════════════════════════════════════

class TestAutoFreezeConv2d:

    def test_forward_matches_conv2d_when_sparsity_off(self):
        ref = nn.Conv2d(3, 8, 3, padding=1)
        af = AutoFreezeConv2d(3, 8, 3, padding=1)
        af.load_state_dict(ref.state_dict())
        af.sparsity_signal = False

        x = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            y_ref = ref(x)
            y_af = af(x)
        assert torch.allclose(y_ref, y_af, atol=1e-5)

    def test_forward_with_sparsity_on(self):
        af = AutoFreezeConv2d(3, 8, 3, padding=1)
        af.sparsity_signal = True
        af.clip_ratio = 0.3

        x = torch.randn(1, 3, 8, 8, requires_grad=True)
        y = af(x)
        assert y.shape == (1, 8, 8, 8)
        # Backward should work
        y.sum().backward()
        assert x.grad is not None

    def test_activation_size_tracked(self):
        af = AutoFreezeConv2d(3, 8, 3, padding=1)
        af.sparsity_signal = False
        x = torch.randn(2, 3, 4, 4)
        af(x)
        assert af.activation_size == 2 * 3 * 4 * 4

    def test_bn_only_blocks_weight_grad(self):
        af = AutoFreezeConv2d(3, 8, 3, padding=1, bn_only=True)
        af.sparsity_signal = False
        x = torch.randn(1, 3, 8, 8, requires_grad=True)
        y = af(x)
        y.sum().backward()
        # bn_only should block weight gradient
        assert af.weight.grad is None

    def test_backward_produces_bias_grad(self):
        af = AutoFreezeConv2d(3, 8, 3, padding=1)
        af.sparsity_signal = True
        af.clip_ratio = 0.4

        x = torch.randn(1, 3, 8, 8, requires_grad=True)
        y = af(x)
        y.sum().backward()
        assert af.bias is not None
        assert af.bias.grad is not None


# ═══════════════════════════════════════════════════════════════════════════
# 3.  DASBatchNorm2d
# ═══════════════════════════════════════════════════════════════════════════

class TestDASBatchNorm2d:

    def test_forward_matches_bn_when_sparsity_off(self):
        ref = nn.BatchNorm2d(8)
        das = DASBatchNorm2d(8)
        das.load_state_dict(ref.state_dict())
        das.sparsity_signal = False

        x = torch.randn(2, 8, 4, 4)
        ref.eval()
        das.eval()
        with torch.no_grad():
            y_ref = ref(x)
            y_das = das(x)
        assert torch.allclose(y_ref, y_das, atol=1e-5)

    def test_forward_with_sparsity_on_train_mode(self):
        das = DASBatchNorm2d(8)
        das.sparsity_signal = True
        das.clip_ratio = 0.3
        das.train()

        x = torch.randn(2, 8, 4, 4, requires_grad=True)
        y = das(x)
        assert y.shape == x.shape
        y.sum().backward()
        # Should complete without error

    def test_sparse_backward_produces_affine_grads(self):
        das = DASBatchNorm2d(8)
        das.sparsity_signal = True
        das.clip_ratio = 0.5
        das.train()

        x = torch.randn(2, 8, 4, 4, requires_grad=True)
        y = das(x)
        y.sum().backward()
        assert das.weight is not None and das.weight.grad is not None
        assert das.bias is not None and das.bias.grad is not None

    def test_activation_size_tracked(self):
        das = DASBatchNorm2d(8)
        das.sparsity_signal = False
        x = torch.randn(2, 8, 4, 4)
        das.eval()
        das(x)
        assert das.activation_size == 2 * 8 * 4 * 4


# ═══════════════════════════════════════════════════════════════════════════
# 4.  AutoFreezeFC
# ═══════════════════════════════════════════════════════════════════════════

class TestDASGroupNorm:

    def test_forward_matches_group_norm_when_sparsity_off(self):
        ref = nn.GroupNorm(2, 8)
        das = DASGroupNorm(2, 8)
        das.load_state_dict(ref.state_dict())
        das.sparsity_signal = False

        x = torch.randn(2, 8, 4, 4)
        with torch.no_grad():
            y_ref = ref(x)
            y_das = das(x)
        assert torch.allclose(y_ref, y_das, atol=1e-5)

    def test_sparse_backward_produces_affine_grads(self):
        das = DASGroupNorm(2, 8)
        das.sparsity_signal = True
        das.clip_ratio = 0.5

        x = torch.randn(2, 8, 4, 4, requires_grad=True)
        y = das(x)
        y.sum().backward()

        assert x.grad is not None
        assert das.weight is not None and das.weight.grad is not None
        assert das.bias is not None and das.bias.grad is not None


class TestDASLayerNorm:

    def test_forward_matches_layer_norm_when_sparsity_off(self):
        ref = nn.LayerNorm(16)
        das = DASLayerNorm(16)
        das.load_state_dict(ref.state_dict())
        das.sparsity_signal = False

        x = torch.randn(2, 16)
        with torch.no_grad():
            y_ref = ref(x)
            y_das = das(x)
        assert torch.allclose(y_ref, y_das, atol=1e-5)

    def test_sparse_backward_produces_affine_grads(self):
        das = DASLayerNorm(16)
        das.sparsity_signal = True
        das.clip_ratio = 0.5

        x = torch.randn(2, 16, requires_grad=True)
        y = das(x)
        y.sum().backward()

        assert x.grad is not None
        assert das.weight is not None and das.weight.grad is not None
        assert das.bias is not None and das.bias.grad is not None


class TestAutoFreezeFC:

    def test_forward_matches_linear_when_sparsity_off(self):
        ref = nn.Linear(10, 5)
        af = AutoFreezeFC(10, 5)
        af.load_state_dict(ref.state_dict())
        af.sparsity_signal = False

        x = torch.randn(2, 10)
        with torch.no_grad():
            y_ref = ref(x)
            y_af = af(x)
        assert torch.allclose(y_ref, y_af, atol=1e-5)

    def test_forward_with_sparsity_on(self):
        af = AutoFreezeFC(10, 5)
        af.sparsity_signal = True
        af.clip_ratio = 0.5

        x = torch.randn(2, 10, requires_grad=True)
        y = af(x)
        assert y.shape == (2, 5)
        y.sum().backward()
        assert x.grad is not None

    def test_bn_only_blocks_weight_grad(self):
        af = AutoFreezeFC(10, 5, bn_only=True)
        af.sparsity_signal = False
        x = torch.randn(2, 10, requires_grad=True)
        y = af(x)
        y.sum().backward()
        assert af.weight.grad is None

    def test_backward_produces_bias_grad(self):
        af = AutoFreezeFC(10, 5)
        af.sparsity_signal = True
        af.clip_ratio = 0.5

        x = torch.randn(2, 10, requires_grad=True)
        y = af(x)
        y.sum().backward()
        assert af.bias is not None
        assert af.bias.grad is not None


# ═══════════════════════════════════════════════════════════════════════════
# 5.  compute_tgi
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeTGI:

    def test_basic_tgi(self):
        params = [torch.randn(3, 3), torch.randn(5, 5)]
        grads = [torch.randn(3, 3), torch.randn(5, 5)]
        names = ["layer1.weight", "layer2.weight"]
        memories = [9.0, 25.0]
        mem_sum = 34.0

        tgi = compute_tgi(params, grads, names, memories, mem_sum)
        assert isinstance(tgi, dict)
        assert set(tgi.keys()) == set(names)
        for v in tgi.values():
            assert isinstance(v, float)
            assert v >= 0  # TGI should be non-negative

    def test_tgi_zero_grad(self):
        params = [torch.randn(3, 3)]
        grads = [torch.zeros(3, 3)]
        names = ["w"]
        memories = [9.0]
        mem_sum = 9.0

        tgi = compute_tgi(params, grads, names, memories, mem_sum)
        # GI = 0 → TGI = 0
        assert tgi["w"] == 0.0

    def test_tgi_single_layer_memory_importance_zero(self):
        """MI = log(M_total / M_layer) = log(1) = 0 when single layer."""
        params = [torch.randn(4, 4)]
        grads = [torch.ones(4, 4)]
        names = ["w"]
        memories = [16.0]
        mem_sum = 16.0  # MI = log(16/16) = 0

        tgi = compute_tgi(params, grads, names, memories, mem_sum)
        assert tgi["w"] == 0.0

    def test_tgi_larger_memory_lower_mi(self):
        """Layers with larger activation memory have lower MI."""
        params = [torch.ones(2, 2), torch.ones(2, 2)]
        grads = [torch.ones(2, 2), torch.ones(2, 2)]
        names = ["small", "big"]
        memories = [1.0, 100.0]
        mem_sum = 101.0

        tgi = compute_tgi(params, grads, names, memories, mem_sum)
        # "small" has MI = log(101/1) > MI of "big" = log(101/100)
        assert tgi["small"] > tgi["big"]


# ═══════════════════════════════════════════════════════════════════════════
# 6.  DASTrainer
# ═══════════════════════════════════════════════════════════════════════════

class TestDASTrainer:

    @pytest.fixture
    def cnn_trainer(self):
        model = _small_cnn()
        return DASTrainer(model, device="cpu")

    def test_module_replacement_counts(self, cnn_trainer):
        """Conv2d, BN2d, Linear should each be replaced once."""
        assert cnn_trainer._n_conv == 1
        assert cnn_trainer._n_bn == 1
        assert cnn_trainer._n_fc == 1

    def test_das_modules_found(self, cnn_trainer):
        modules = cnn_trainer._das_modules()
        assert len(modules) == 3  # 1 conv + 1 bn + 1 fc
        types = {type(m) for m in modules.values()}
        assert AutoFreezeConv2d in types
        assert DASBatchNorm2d in types
        assert AutoFreezeFC in types

    def test_activate_deactivate_sparsity(self, cnn_trainer):
        cnn_trainer.activate_sparsity()
        for _, mod in cnn_trainer._das_modules().items():
            assert mod.sparsity_signal is True

        cnn_trainer.deactivate_sparsity()
        for _, mod in cnn_trainer._das_modules().items():
            assert mod.sparsity_signal is False

    def test_activate_with_custom_ratios(self, cnn_trainer):
        modules = cnn_trainer._das_modules()
        ratios = {name + ".weight": 0.42 for name in modules}
        cnn_trainer.activate_sparsity(pruning_ratios=ratios)
        for name, mod in modules.items():
            assert mod.sparsity_signal is True
            assert abs(mod.clip_ratio - 0.42) < 1e-6

    def test_get_memory_stats(self, cnn_trainer):
        # Run a forward pass first so activation_size is populated
        x = torch.randn(1, 3, 8, 8)
        cnn_trainer.model(x)

        stats = cnn_trainer.get_memory_stats()
        assert "total_activation_elements" in stats
        assert "cached_elements" in stats
        assert "compression_ratio" in stats
        assert "per_layer" in stats
        assert stats["total_activation_elements"] > 0

    def test_pruning_ratios_property(self, cnn_trainer):
        assert cnn_trainer.pruning_ratios == {}
        cnn_trainer._pruning_ratios = {"a": 0.5}
        assert cnn_trainer.pruning_ratios == {"a": 0.5}
        # Should be a copy
        cnn_trainer.pruning_ratios["b"] = 0.1
        assert "b" not in cnn_trainer._pruning_ratios

    def test_probe_and_set_ratios(self, cnn_trainer):
        x = torch.randn(4, 3, 8, 8)

        def loss_fn(logits):
            return logits.sum()

        ratios = cnn_trainer.probe_and_set_ratios(x, loss_fn)
        assert isinstance(ratios, dict)
        # All values in [0, 1]
        for v in ratios.values():
            assert 0.0 <= v <= 1.0

    def test_das_train_step(self, cnn_trainer):
        x = torch.randn(4, 3, 8, 8)
        targets = torch.randint(0, 5, (4,))
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(cnn_trainer.model.parameters(), lr=0.01)

        loss_val = cnn_trainer.das_train_step(x, targets, loss_fn, optimizer)
        assert isinstance(loss_val, float)
        assert loss_val >= 0

    def test_model_output_unchanged_with_sparsity_off(self, cnn_trainer):
        """With sparsity off, output should be deterministic."""
        cnn_trainer.deactivate_sparsity()
        cnn_trainer.model.eval()
        x = torch.randn(1, 3, 8, 8)
        with torch.no_grad():
            y1 = cnn_trainer.model(x)
            y2 = cnn_trainer.model(x)
        assert torch.allclose(y1, y2)

    def test_get_layer_memories(self, cnn_trainer):
        x = torch.randn(1, 3, 8, 8)
        cnn_trainer.model(x)
        mem_dict, mem_sum = cnn_trainer._get_layer_memories()
        assert isinstance(mem_dict, dict)
        assert mem_sum > 0

    def test_entropy_strategy_enables_entropy_tracking(self):
        trainer = DASTrainer(_small_cnn(), strategy="entropy", device="cpu")
        assert trainer.strategy == "entropy"
        assert trainer.use_spectral_entropy is True
        assert all(mod.track_spectral_entropy for mod in trainer._das_modules().values())

    def test_refresh_pruning_ratios_from_entropy(self):
        trainer = DASTrainer(_small_cnn(), strategy="entropy", device="cpu")
        x = torch.randn(2, 3, 8, 8)
        trainer.model(x)

        ratios = trainer.refresh_pruning_ratios_from_entropy()

        assert isinstance(ratios, dict)
        assert ratios
        for value in ratios.values():
            assert 0.0 <= value <= 1.0

    def test_zero_importance_scores_disable_pruning(self, cnn_trainer):
        ratios = cnn_trainer._importance_scores_to_pruning_ratios(
            {"layer1.weight": 0.0, "layer2.weight": 0.0}
        )
        assert ratios == {"layer1.weight": 0.0, "layer2.weight": 0.0}


# ═══════════════════════════════════════════════════════════════════════════
# 7.  apply_das_to_model / apply_das_to_tail
# ═══════════════════════════════════════════════════════════════════════════

class TestDASTrainerNormVariants:

    def test_replaces_group_norm_and_layer_norm(self):
        trainer = DASTrainer(_norm_augmented_model(), device="cpu")

        assert trainer._n_conv == 1
        assert trainer._n_gn == 1
        assert trainer._n_ln == 1
        assert trainer._n_fc == 1

        modules = trainer._das_modules()
        types = {type(m) for m in modules.values()}
        assert AutoFreezeConv2d in types
        assert DASGroupNorm in types
        assert DASLayerNorm in types
        assert AutoFreezeFC in types


class TestApplyDAS:

    def test_apply_das_to_model(self):
        model = _small_cnn()
        trainer = apply_das_to_model(model, device="cpu")
        assert isinstance(trainer, DASTrainer)
        assert trainer._n_conv >= 1

    def test_apply_das_to_model_entropy_strategy(self):
        model = _small_cnn()
        trainer = apply_das_to_model(model, strategy="entropy", device="cpu")
        assert trainer.strategy == "entropy"
        assert trainer.use_spectral_entropy is True

    def test_apply_das_to_tail(self):
        """Apply DAS only to a sub-module."""

        class TwoPartModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU())
                self.tail = nn.Sequential(
                    nn.Conv2d(8, 16, 3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(16, 5),
                )

            def forward(self, x):
                return self.tail(self.head(x))

        model = TwoPartModel()
        trainer = apply_das_to_tail(model, ["tail"], device="cpu")
        assert isinstance(trainer, DASTrainer)
        # Only tail modules should be replaced
        assert trainer._n_conv >= 1
        assert trainer._n_bn >= 1
        # Head conv should NOT be replaced
        assert isinstance(model.head[0], nn.Conv2d)
        assert not isinstance(model.head[0], AutoFreezeConv2d)

    def test_apply_das_to_tail_missing_module(self):
        model = _small_cnn()
        trainer = apply_das_to_tail(model, ["nonexistent"], device="cpu")
        # Should not crash, just skip the missing module
        assert trainer._n_conv == 0
        assert trainer._n_bn == 0
        assert trainer._n_fc == 0

    def test_apply_das_to_tail_replaces_group_norm_and_layer_norm(self):
        class TwoPartNormModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
                self.tail = nn.Sequential(
                    nn.GroupNorm(2, 4),
                    nn.AdaptiveAvgPool2d((2, 2)),
                    nn.Flatten(),
                    nn.LayerNorm(16),
                    nn.Linear(16, 3),
                )

            def forward(self, x):
                return self.tail(self.head(x))

        model = TwoPartNormModel()
        trainer = apply_das_to_tail(model, ["tail"], device="cpu")

        assert isinstance(trainer, DASTrainer)
        assert trainer._n_gn == 1
        assert trainer._n_ln == 1
        assert trainer._n_fc == 1
        assert isinstance(model.head[0], nn.Conv2d)
        assert not isinstance(model.head[0], AutoFreezeConv2d)
        assert isinstance(model.tail[0], DASGroupNorm)
        assert isinstance(model.tail[3], DASLayerNorm)
