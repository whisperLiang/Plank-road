from types import SimpleNamespace

import torch

from model_management.model_delta_payload import build_state_dict_delta_payload


def test_delta_payload_uses_unregistered_wrapper_inner_parameter_names():
    class WrappedDetector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            inner = torch.nn.Linear(2, 1)
            inner.bias.requires_grad_(False)
            self.rfdetr = SimpleNamespace(model=SimpleNamespace(model=inner))

        def parameters(self, recurse=True):
            return self.rfdetr.model.model.parameters(recurse=recurse)

        def state_dict(self, *args, **kwargs):
            return self.rfdetr.model.model.state_dict(*args, **kwargs)

    payload = build_state_dict_delta_payload(
        WrappedDetector(),
        model_name="rfdetr_nano",
        base_model_version="0",
        result_model_version="1",
    )

    assert set(payload["state_dict"]) == {"weight"}


def test_delta_payload_maps_registered_wrapper_prefix_to_inner_state_dict():
    class WrappedDetector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.detr = torch.nn.Linear(2, 1)
            self.detr.bias.requires_grad_(False)

        def state_dict(self, *args, **kwargs):
            return self.detr.state_dict(*args, **kwargs)

    payload = build_state_dict_delta_payload(
        WrappedDetector(),
        model_name="detr_resnet50",
        base_model_version="0",
        result_model_version="1",
    )

    assert set(payload["state_dict"]) == {"weight"}
