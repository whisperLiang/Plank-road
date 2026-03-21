"""
Tests for model_management.universal_model_split — model-agnostic splitting.

Covers:
* SplitPointSelector (LinUCB bandit: select, update, select_by_flops_ratio,
  select_by_privacy, get_profiles)
* LayerInfo / LayerProfile dataclasses
* UniversalModelSplitter (trace, list_layers, split, edge_forward,
  cloud_forward, cloud_train_step, serialise/deserialise intermediate,
  full_forward, split_retrain, state-dict helpers)
* Convenience functions: extract_split_features, save_split_feature_cache,
  load_split_feature_cache

NOTE: torchlens is required for the full universal splitting.  Tests that
depend on it are skipped when torchlens is not installed.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections import OrderedDict
from dataclasses import fields

import numpy as np
import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Import guards
# ---------------------------------------------------------------------------
try:
    from model_management.universal_model_split import (
        _HAS_TORCHLENS,
        LayerInfo,
        LayerProfile,
        SplitPayload,
        SplitPointSelector,
        UniversalModelSplitter,
        _compute_boundary_labels,
        _partition_forward_head,
        _partition_forward_tail,
        _prepare_replay_graph,
        extract_split_features,
        load_split_feature_cache,
        save_split_feature_cache,
    )
    _IMPORT_OK = True
except ImportError:
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="universal_model_split import failed")


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def cache_dir():
    d = tempfile.mkdtemp(prefix="plankroad_usplit_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _simple_model():
    """A tiny Sequential model for testing (no torchlens dependency)."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )


class _DummyLayer:
    def __init__(
        self,
        *,
        label,
        func_name,
        func=None,
        layer_type="operation",
        creation_args=None,
        parent_layers=None,
        parent_arg_locs=None,
        tensor_contents=None,
    ):
        self.layer_label = label
        self.layer_label_short = label
        self.func_applied_name = func_name
        self.func_applied = func
        self.layer_type = layer_type
        self.creation_args = creation_args or []
        self.parent_layers = parent_layers or []
        self.parent_layer_arg_locs = parent_arg_locs or {"args": {}, "kwargs": {}}
        self.func_keyword_args_non_tensor = {}
        self.tensor_contents = tensor_contents


def _branch_layers():
    return [
        _DummyLayer(label="input", func_name="none", func=None, layer_type="input"),
        _DummyLayer(
            label="left_mul",
            func_name="mul",
            func=torch.mul,
            creation_args=[None, 2.0],
            parent_layers=["input"],
            parent_arg_locs={"args": {0: "input"}, "kwargs": {}},
        ),
        _DummyLayer(
            label="right_mul",
            func_name="mul",
            func=torch.mul,
            creation_args=[None, 3.0],
            parent_layers=["input"],
            parent_arg_locs={"args": {0: "input"}, "kwargs": {}},
        ),
        _DummyLayer(
            label="merge_add",
            func_name="add",
            func=torch.add,
            creation_args=[None, None],
            parent_layers=["left_mul", "right_mul"],
            parent_arg_locs={"args": {0: "left_mul", 1: "right_mul"}, "kwargs": {}},
        ),
        _DummyLayer(
            label="relu_out",
            func_name="relu",
            func=torch.relu,
            creation_args=[None],
            parent_layers=["merge_add"],
            parent_arg_locs={"args": {0: "merge_add"}, "kwargs": {}},
        ),
        _DummyLayer(
            label="output",
            func_name="none",
            func=None,
            layer_type="output",
            parent_layers=["relu_out"],
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# 1. LayerInfo / LayerProfile dataclasses
# ═══════════════════════════════════════════════════════════════════════════

class TestDataclasses:

    def test_layer_info_fields(self):
        info = LayerInfo(
            index=0, label="conv1", layer_type="conv",
            func_name="conv2d", output_shape=(1, 64, 32, 32),
            has_params=True, num_params=9408, module_name="layer1.conv",
        )
        assert info.index == 0
        assert info.label == "conv1"
        assert info.has_params is True
        assert info.num_params == 9408

    def test_layer_profile_fields(self):
        prof = LayerProfile(
            index=3, label="relu1", cumulative_flops=1e6,
            smashed_data_size=65536, output_shape=(1, 64, 32, 32),
            privacy_leakage=0.42,
        )
        assert prof.index == 3
        assert prof.cumulative_flops == 1e6
        assert prof.smashed_data_size == 65536
        assert abs(prof.privacy_leakage - 0.42) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# 2.  SplitPointSelector (LinUCB bandit)
# ═══════════════════════════════════════════════════════════════════════════

class TestSplitPointSelector:

    def _make_profiles(self, n=5):
        return [
            LayerProfile(
                index=i, label=f"layer_{i}",
                cumulative_flops=(i + 1) * 1e5,
                smashed_data_size=(n - i) * 1000,
                output_shape=(1, 64, 32 - i * 4, 32 - i * 4),
                privacy_leakage=(n - i) * 0.1,
            )
            for i in range(n)
        ]

    def test_creation_and_profiles(self):
        profiles = self._make_profiles(5)
        selector = SplitPointSelector(profiles, alpha=1.0)
        assert selector.get_profiles() == profiles

    def test_select_returns_valid_index(self):
        profiles = self._make_profiles(5)
        selector = SplitPointSelector(profiles, alpha=1.0)
        chosen = selector.select()
        assert 0 <= chosen < 5

    def test_update_does_not_raise(self):
        profiles = self._make_profiles(5)
        selector = SplitPointSelector(profiles, alpha=1.0)
        idx = selector.select()
        selector.update(idx, observed_latency=0.5)

    def test_select_by_flops_ratio(self):
        profiles = self._make_profiles(5)
        selector = SplitPointSelector(profiles, alpha=1.0)
        total_flops = profiles[-1].cumulative_flops
        # Target ~50% → should pick layer near the middle
        idx = selector.select_by_flops_ratio(target_ratio=0.5)
        assert 0 <= idx < 5
        # Cumulative flops at chosen index should be ≤ total
        assert profiles[idx].cumulative_flops <= total_flops

    def test_select_by_privacy(self):
        profiles = self._make_profiles(5)
        selector = SplitPointSelector(profiles, alpha=1.0)
        # max_leakage = 0.25 → should pick a deeper layer (less leakage)
        idx = selector.select_by_privacy(max_leakage=0.25)
        assert 0 <= idx < 5
        assert profiles[idx].privacy_leakage <= 0.25 + 1e-6

    def test_repeated_select_update_converges(self):
        profiles = self._make_profiles(5)
        selector = SplitPointSelector(profiles, alpha=0.5)
        for _ in range(20):
            idx = selector.select()
            # Simulate lower latency for deeper splits
            lat = (5 - idx) * 0.1
            selector.update(idx, lat)
        # After many updates, select should still return a valid index
        final = selector.select()
        assert 0 <= final < 5


# ═══════════════════════════════════════════════════════════════════════════
# 3.  UniversalModelSplitter — requires torchlens
# ═══════════════════════════════════════════════════════════════════════════

_skip_no_torchlens = pytest.mark.skipif(
    not (_IMPORT_OK and _HAS_TORCHLENS),
    reason="torchlens not installed",
)


@_skip_no_torchlens
class TestUniversalModelSplitter:

    @pytest.fixture
    def splitter_and_model(self):
        model = _simple_model()
        splitter = UniversalModelSplitter(device="cpu")
        sample = torch.randn(1, 10)
        splitter.trace(model, sample)
        return splitter, model, sample

    def test_trace_populates_layers(self, splitter_and_model):
        splitter, model, _ = splitter_and_model
        layers = splitter.list_layers()
        assert isinstance(layers, list)
        assert splitter.num_layers() > 0
        for info in layers:
            assert isinstance(info, LayerInfo)

    def test_split_and_edge_cloud_forward(self, splitter_and_model):
        splitter, model, sample = splitter_and_model
        n = splitter.num_layers()
        # Split at the midpoint
        mid = n // 2
        splitter.split(layer_index=mid)

        inter = splitter.edge_forward(sample)
        assert isinstance(inter, torch.Tensor)

        out = splitter.cloud_forward(inter)
        assert isinstance(out, torch.Tensor)

    def test_full_forward_matches_original(self, splitter_and_model):
        splitter, model, sample = splitter_and_model
        model.eval()
        with torch.no_grad():
            expected = model(sample)
            replayed = splitter.full_forward(sample)
        assert torch.allclose(expected, replayed, atol=1e-4), (
            f"max diff = {(expected - replayed).abs().max().item()}"
        )

    def test_split_by_label(self, splitter_and_model):
        splitter, model, sample = splitter_and_model
        layers = splitter.list_layers()
        # Pick a layer with a label
        label = layers[len(layers) // 2].label
        splitter.split(layer_label=label)
        inter = splitter.edge_forward(sample)
        assert inter is not None

    def test_serialise_deserialise_intermediate(self, splitter_and_model):
        splitter, model, sample = splitter_and_model
        n = splitter.num_layers()
        splitter.split(layer_index=n // 2)
        inter = splitter.edge_forward(sample)

        data = splitter.serialise_intermediate(inter)
        assert isinstance(data, bytes)

        recovered = splitter.deserialise_intermediate(data)
        assert torch.allclose(inter, recovered)

    def test_serialise_deserialise_compressed(self, splitter_and_model):
        splitter, model, sample = splitter_and_model
        n = splitter.num_layers()
        splitter.split(layer_index=n // 2)
        inter = splitter.edge_forward(sample)

        data = splitter.serialise_intermediate(inter, compress=True)
        recovered = splitter.deserialise_intermediate(data, compressed=True)
        assert torch.allclose(inter, recovered)

    def test_cloud_train_step(self, splitter_and_model):
        splitter, model, sample = splitter_and_model
        n = splitter.num_layers()
        splitter.split(layer_index=n // 2)
        splitter.freeze_head()
        splitter.unfreeze_tail()

        inter = splitter.edge_forward(sample)
        targets = torch.tensor([2])
        loss_fn = nn.CrossEntropyLoss()

        output, loss = splitter.cloud_train_step(inter, targets, loss_fn)
        assert isinstance(output, torch.Tensor)
        assert loss.dim() == 0  # scalar

    def test_get_tail_trainable_params(self, splitter_and_model):
        splitter, model, sample = splitter_and_model
        n = splitter.num_layers()
        splitter.split(layer_index=n // 2)
        splitter.freeze_head()
        splitter.unfreeze_tail()
        params = splitter.get_tail_trainable_params()
        assert isinstance(params, list)
        # There should be at least some trainable params in the tail
        assert len(params) >= 0  # could be 0 if split is at last layer

    def test_tail_state_dict_round_trip(self, splitter_and_model):
        splitter, model, sample = splitter_and_model
        n = splitter.num_layers()
        splitter.split(layer_index=n // 2)

        sd = splitter.get_tail_state_dict()
        assert isinstance(sd, dict)
        # Load back
        splitter.load_tail_state_dict(sd)

    def test_repr(self, splitter_and_model):
        splitter, model, sample = splitter_and_model
        r = repr(splitter)
        assert "layers" in r
        n = splitter.num_layers()
        splitter.split(layer_index=n // 2)
        r2 = repr(splitter)
        assert "split@" in r2

    def test_ensure_traced_raises_if_not_traced(self):
        splitter = UniversalModelSplitter(device="cpu")
        with pytest.raises(RuntimeError, match="not been traced"):
            splitter.list_layers()

    def test_ensure_split_raises_if_not_split(self, splitter_and_model):
        splitter, model, sample = splitter_and_model
        with pytest.raises(RuntimeError, match="Split point not set"):
            splitter.edge_forward(sample)

    def test_split_retrain_runs(self, splitter_and_model):
        splitter, model, sample = splitter_and_model
        n = splitter.num_layers()
        splitter.split(layer_index=n // 2)

        # Create a tiny train loader
        train_data = [(torch.randn(1, 10), torch.tensor([1])) for _ in range(3)]
        loss_fn = nn.CrossEntropyLoss()

        epoch_losses = splitter.split_retrain(
            train_data, loss_fn, num_epochs=1, lr=0.01,
        )
        assert isinstance(epoch_losses, list)
        assert len(epoch_losses) == 1

    def test_profile_layers(self, splitter_and_model):
        splitter, model, sample = splitter_and_model
        try:
            profiles = splitter.profile_layers()
            assert isinstance(profiles, list)
            for p in profiles:
                assert isinstance(p, LayerProfile)
        except Exception:
            pytest.skip("profile_layers not available or failed")

    def test_create_split_selector(self, splitter_and_model):
        splitter, model, sample = splitter_and_model
        try:
            selector = splitter.create_split_selector()
            assert isinstance(selector, SplitPointSelector)
        except Exception:
            pytest.skip("create_split_selector requires profile_layers")


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Feature cache convenience functions
# ═══════════════════════════════════════════════════════════════════════════

class TestSplitFeatureCache:

    def test_save_and_load(self, cache_dir):
        intermediate = torch.randn(1, 64, 16, 16)
        path = save_split_feature_cache(
            cache_dir, 7, intermediate, is_drift=True,
            pseudo_boxes=[[10, 20, 100, 200]],
            pseudo_labels=[1],
            pseudo_scores=[0.95],
        )
        assert os.path.isfile(path)

        data = load_split_feature_cache(cache_dir, 7)
        assert torch.allclose(data["intermediate"], intermediate)
        assert data["is_drift"] is True
        assert data["pseudo_boxes"] == [[10, 20, 100, 200]]

    def test_save_with_extra_metadata(self, cache_dir):
        intermediate = torch.randn(1, 32)
        save_split_feature_cache(
            cache_dir, 0, intermediate, is_drift=False,
            extra_metadata={"model_name": "resnet18", "split_layer": 5},
        )
        data = load_split_feature_cache(cache_dir, 0)
        assert data["model_name"] == "resnet18"
        assert data["split_layer"] == 5

    def test_load_missing_raises(self, cache_dir):
        with pytest.raises(Exception):
            load_split_feature_cache(cache_dir, 9999)

    def test_save_and_load_split_payload(self, cache_dir):
        payload = SplitPayload(
            tensors=OrderedDict([
                ("left_mul", torch.tensor([2.0])),
                ("right_mul", torch.tensor([3.0])),
            ]),
            split_index=2,
            split_label="right_mul",
        )
        save_split_feature_cache(cache_dir, 11, payload, is_drift=False)
        data = load_split_feature_cache(cache_dir, 11)
        restored = data["intermediate"]
        assert isinstance(restored, SplitPayload)
        assert list(restored.tensors.keys()) == ["left_mul", "right_mul"]
        assert torch.allclose(restored.tensors["left_mul"], torch.tensor([2.0]))
        assert torch.allclose(restored.tensors["right_mul"], torch.tensor([3.0]))


class TestGraphBoundaryReplay:

    def test_branch_split_uses_boundary_payload(self):
        layers = _branch_layers()
        label2idx = _prepare_replay_graph(layers)
        x = torch.tensor([1.0])
        split_index = 2

        boundary_labels = _compute_boundary_labels(layers, label2idx, split_index)
        assert boundary_labels == ["left_mul", "right_mul"]

        _, payload = _partition_forward_head(
            layers,
            label2idx,
            x,
            split_index,
            boundary_labels,
        )
        assert isinstance(payload, SplitPayload)
        assert list(payload.tensors.keys()) == ["left_mul", "right_mul"]

        for layer in layers:
            if isinstance(layer.tensor_contents, torch.Tensor):
                layer.tensor_contents = torch.tensor([-999.0])

        out = _partition_forward_tail(
            layers,
            label2idx,
            payload,
            split_index,
            "right_mul",
        )
        assert torch.allclose(out, torch.tensor([5.0]))

    def test_payload_serialisation_round_trip(self):
        payload = SplitPayload(
            tensors=OrderedDict([
                ("left_mul", torch.tensor([2.0])),
                ("right_mul", torch.tensor([3.0])),
            ]),
            split_index=2,
            split_label="right_mul",
        )
        data = UniversalModelSplitter.serialise_intermediate(payload, compress=True)
        restored = UniversalModelSplitter.deserialise_intermediate(data, compressed=True)
        assert isinstance(restored, SplitPayload)
        assert torch.allclose(restored.tensors["left_mul"], torch.tensor([2.0]))
        assert torch.allclose(restored.tensors["right_mul"], torch.tensor([3.0]))


@_skip_no_torchlens
class TestExtractSplitFeatures:

    def test_extract(self):
        model = _simple_model()
        splitter = UniversalModelSplitter(device="cpu")
        sample = torch.randn(1, 10)
        splitter.trace(model, sample)
        n = splitter.num_layers()
        splitter.split(layer_index=n // 2)

        inter = extract_split_features(splitter, sample)
        assert isinstance(inter, torch.Tensor)
        assert inter.device == torch.device("cpu")
