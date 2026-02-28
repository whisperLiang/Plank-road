"""
Model Split Utilities for Split-Learning-based Continual Learning
=================================================================

Splits a Faster R-CNN model at the **backbone** boundary for HSFL-style
continual learning (Hybrid Split Federated Learning).

Architecture split
------------------
  Edge (client):  ``transform + backbone``  →  intermediate FPN features
  Cloud (server):  ``rpn + roi_heads``      →  losses / detections

Design highlights
-----------------
* Only **drift** samples are annotated by the cloud's large model.
* **All** samples (drift + non-drift) contribute intermediate features.
* Non-drift samples use edge pseudo-labels.
* Only ``roi_heads`` is trainable; ``backbone`` and ``rpn`` are frozen,
  so **no gradient exchange** between edge and cloud is needed.

Reference
---------
  HSFL (ICWS 2023): "HSFL: Efficient and Privacy-Preserving Offloading
  for Split and Federated Learning in IoT Services"
  https://github.com/SASA-cloud/ICWS-23-HSFL
"""

from __future__ import annotations

import io
import os
import random
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from loguru import logger

try:
    from torchvision.models.detection.image_list import ImageList
except ImportError:  # older torchvision
    from torchvision.models.detection.transform import ImageList


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Feature extraction  (edge side)
# ═══════════════════════════════════════════════════════════════════════════

def extract_backbone_features(
    model: torch.nn.Module,
    frame: np.ndarray,
    device: torch.device | None = None,
) -> Tuple[OrderedDict, List[Tuple[int, int]], Tuple[int, int], List[Tuple[int, int]]]:
    """Run ``transform → backbone`` of a Faster R-CNN model on a BGR frame.

    Parameters
    ----------
    model : torchvision Faster R-CNN model
    frame : np.ndarray – BGR, HWC
    device : computation device (inferred from *model* if ``None``)

    Returns
    -------
    features_cpu   : OrderedDict[str, Tensor] – FPN feature maps (CPU)
    image_sizes    : [(H_trans, W_trans)]      – sizes after transform
    tensor_shape   : (H_pad, W_pad)            – padded batch spatial dims
    original_sizes : [(H_orig, W_orig)]        – original frame sizes
    """
    if device is None:
        device = next(model.parameters()).device

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_t = transforms.ToTensor()(img_pil).to(device)
    orig_h, orig_w = img_t.shape[-2], img_t.shape[-1]

    was_training = model.training
    model.eval()
    with torch.no_grad():
        images, _ = model.transform([img_t], None)
        features = model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

    if was_training:
        model.train()

    feats_cpu = OrderedDict((k, v.cpu()) for k, v in features.items())
    return (
        feats_cpu,
        images.image_sizes,                    # [(H_trans, W_trans)]
        tuple(images.tensors.shape[-2:]),       # (H_pad, W_pad)
        [(orig_h, orig_w)],                     # original sizes
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Feature cache I/O
# ═══════════════════════════════════════════════════════════════════════════

def save_feature_cache(
    cache_path: str,
    frame_index: int,
    features: OrderedDict,
    image_sizes: list,
    tensor_shape: tuple,
    original_sizes: list,
    is_drift: bool,
    pseudo_boxes: list | None = None,
    pseudo_labels: list | None = None,
    pseudo_scores: list | None = None,
) -> str:
    """Persist one frame's backbone features to ``<cache>/features/<idx>.pt``."""
    feat_dir = os.path.join(cache_path, "features")
    os.makedirs(feat_dir, exist_ok=True)
    out_path = os.path.join(feat_dir, f"{frame_index}.pt")
    torch.save(
        {
            "features":       dict(features),
            "image_sizes":    image_sizes,
            "tensor_shape":   tensor_shape,
            "original_sizes": original_sizes,
            "is_drift":       is_drift,
            "pseudo_boxes":   pseudo_boxes or [],
            "pseudo_labels":  pseudo_labels or [],
            "pseudo_scores":  pseudo_scores or [],
        },
        out_path,
    )
    return out_path


def load_feature_cache(cache_path: str, frame_index: int) -> dict:
    """Load one frame's cached features and metadata."""
    path = os.path.join(cache_path, "features", f"{frame_index}.pt")
    data = torch.load(path, map_location="cpu")
    data["features"] = OrderedDict(data["features"])
    return data


def list_cached_features(cache_path: str) -> list[int]:
    """Return sorted list of frame indices that have cached features."""
    feat_dir = os.path.join(cache_path, "features")
    if not os.path.isdir(feat_dir):
        return []
    indices = []
    for fn in os.listdir(feat_dir):
        if fn.endswith(".pt"):
            try:
                indices.append(int(fn[:-3]))
            except ValueError:
                pass
    return sorted(indices)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Target-coordinate transform
# ═══════════════════════════════════════════════════════════════════════════

def transform_targets_to_feature_space(
    targets: List[Dict[str, torch.Tensor]],
    original_sizes: List[Tuple[int, int]],
    image_sizes: List[Tuple[int, int]],
) -> List[Dict[str, torch.Tensor]]:
    """Scale target boxes from **original-image** coords → **transformed** coords.

    Faster R-CNN's ``GeneralizedRCNNTransform`` resizes images (and targets)
    before feeding them to the backbone.  Since our features are already in
    the transformed space the targets must be scaled accordingly.
    """
    out: list[dict] = []
    for tgt, (oh, ow), (th, tw) in zip(targets, original_sizes, image_sizes):
        t = dict(tgt)
        if "boxes" in t and len(t["boxes"]) > 0:
            boxes = t["boxes"].clone().float()
            boxes[:, 0] *= tw / ow   # x1
            boxes[:, 1] *= th / oh   # y1
            boxes[:, 2] *= tw / ow   # x2
            boxes[:, 3] *= th / oh   # y2
            t["boxes"] = boxes
        out.append(t)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Server-side helpers  (cloud side)
# ═══════════════════════════════════════════════════════════════════════════

def build_image_list(
    image_sizes: List[Tuple[int, int]],
    tensor_shape: Tuple[int, int],
    device: torch.device,
) -> ImageList:
    """Build a minimal ``ImageList`` whose shape satisfies the RPN anchor
    generator (which reads ``images.tensors.shape[-2:]``)."""
    n = len(image_sizes)
    h, w = tensor_shape
    dummy = torch.zeros(n, 3, h, w, device=device)
    return ImageList(dummy, image_sizes)


def server_side_train_step(
    model: torch.nn.Module,
    features: OrderedDict,
    image_sizes: list,
    tensor_shape: tuple,
    targets: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> dict:
    """Forward through *rpn + roi_heads* in **train** mode and return the
    loss dictionary.

    Parameters
    ----------
    model        : Full Faster R-CNN; only ``roi_heads`` should be trainable.
    features     : Pre-computed backbone features (single frame, batch=1).
    image_sizes  : ``[(H_trans, W_trans)]`` after transform.
    tensor_shape : ``(H_pad, W_pad)`` padded spatial size.
    targets      : ``[{"boxes": Tensor, "labels": Tensor}]`` in *transformed*
                   coordinate space.
    device       : Computation device.

    Returns
    -------
    dict[str, Tensor] – loss dictionary (rpn + roi_heads losses).
    """
    feat = OrderedDict((k, v.to(device)) for k, v in features.items())
    tgts = [{k: v.to(device) for k, v in t.items()} for t in targets]
    imgs = build_image_list(image_sizes, tensor_shape, device)

    proposals, proposal_losses = model.rpn(imgs, feat, tgts)
    _, detector_losses = model.roi_heads(feat, proposals, image_sizes, tgts)

    losses: dict = {}
    losses.update(proposal_losses)
    losses.update(detector_losses)
    return losses


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Full split-learning training loop  (cloud side)
# ═══════════════════════════════════════════════════════════════════════════

def split_retrain(
    model: torch.nn.Module,
    cache_path: str,
    all_indices: List[int],
    gt_annotations: Dict[int, dict],
    device: torch.device,
    num_epoch: int = 2,
    lr: float = 0.005,
) -> None:
    """Train ``rpn + roi_heads`` on cached backbone features.

    Parameters
    ----------
    model : Faster R-CNN with frozen backbone & rpn, trainable roi_heads.
    cache_path : Directory containing ``features/<idx>.pt`` files.
    all_indices : All frame indices to use for training.
    gt_annotations : ``{frame_index: {"boxes": [[x1,y1,x2,y2], ...],
                       "labels": [int, ...]}}`` ground-truth for drift frames.
                     Non-drift frames use cached pseudo-labels.
    device : Computation device.
    num_epoch : Number of training epochs.
    lr : Learning rate.
    """
    # Ensure only roi_heads is trainable
    for p in model.parameters():
        p.requires_grad = False
    for p in model.roi_heads.parameters():
        p.requires_grad = True

    roi_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(roi_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    model.train()
    # Keep backbone in eval (batch-norm frozen) — it's not used anyway
    model.backbone.eval()

    gt_set = set(gt_annotations.keys())

    for epoch in range(num_epoch):
        epoch_loss = 0.0
        n_samples = 0
        indices = all_indices.copy()
        random.shuffle(indices)

        for idx in indices:
            try:
                data = load_feature_cache(cache_path, idx)
            except Exception as exc:
                logger.warning("[SplitRetrain] Cannot load features for frame {}: {}", idx, exc)
                continue

            features = data["features"]
            image_sizes = data["image_sizes"]
            tensor_shape = data["tensor_shape"]
            original_sizes = data["original_sizes"]

            # ------ Build targets ------
            if idx in gt_set:
                # Ground-truth from cloud annotation (drift frame)
                gt = gt_annotations[idx]
                boxes = gt["boxes"]
                labels = gt["labels"]
            else:
                # Pseudo-labels from edge inference (non-drift frame)
                boxes = data["pseudo_boxes"]
                labels = data["pseudo_labels"]

            if not boxes or not labels:
                continue  # skip frames with no annotations

            targets = [{
                "boxes":  torch.tensor(boxes,  dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
            }]

            # Scale boxes from original-image space → transformed space
            targets = transform_targets_to_feature_space(
                targets, original_sizes, image_sizes
            )

            # ------ Forward & backward ------
            try:
                loss_dict = server_side_train_step(
                    model, features, image_sizes, tensor_shape, targets, device
                )
                total_loss = sum(loss_dict.values())

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                n_samples += 1
            except Exception as exc:
                logger.warning("[SplitRetrain] Training error on frame {}: {}", idx, exc)
                continue

        lr_scheduler.step()
        avg = epoch_loss / max(n_samples, 1)
        logger.info(
            "[SplitRetrain] Epoch {}/{} — samples={}, avg_loss={:.4f}",
            epoch + 1, num_epoch, n_samples, avg,
        )
