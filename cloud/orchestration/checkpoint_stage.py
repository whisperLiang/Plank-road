from __future__ import annotations

import base64
import hashlib
from collections.abc import Mapping

import torch

from cloud.model_update import serialize_model_update


def file_sha1(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointStage:
    def serialize_encoded_update(
        self,
        model: torch.nn.Module,
        *,
        model_name: str,
        checkpoint_path: str,
        weights_metadata: Mapping[str, object] | None,
        metadata_path: str | None,
    ) -> str:
        return base64.b64encode(
            serialize_model_update(
                model,
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                weights_metadata=weights_metadata,
                metadata_path=metadata_path,
            )
        ).decode("utf-8")
