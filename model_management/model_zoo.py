"""
Unified Model Zoo — Detection Model Factory
============================================

Provides a single ``build_detection_model(name)`` entry-point that returns a
PyTorch model **whose forward() always produces the torchvision detection
output format**::

    [{"boxes": Tensor[N,4], "labels": Tensor[N], "scores": Tensor[N]}]

This makes YOLO, DETR, RT-DETR, and any future model family a drop-in
replacement for the existing Faster R-CNN pipeline.

Supported model families
------------------------

**Torchvision built-in (Faster R-CNN / RetinaNet / SSD / FCOS)**

    Directly instantiated via ``torchvision.models.detection.*``.
    These already follow the target format natively.

**DETR family (via torchvision or huggingface)**

    - ``detr_resnet50``  — Facebook DETR (torchvision hub / HF)
    - ``detr_resnet101``
    - ``conditional_detr_resnet50``
    - ``rtdetr_r50vd`` / ``rtdetr_r101vd`` — Baidu RT-DETR (ultralytics)

**YOLO family (via ultralytics)**

    - ``yolov5s`` / ``yolov5m`` / ``yolov5l`` / ``yolov5x``
    - ``yolov8n`` / ``yolov8s`` / ``yolov8m`` / ``yolov8l`` / ``yolov8x``
    - ``yolov10n`` / ``yolov10s`` / ``yolov10m`` / ``yolov10l`` / ``yolov10x``
    - ``yolo11n`` / ``yolo11s`` / ``yolo11m`` / ``yolo11l`` / ``yolo11x``
    - ``yolo12n`` / ``yolo12s`` / ``yolo12m`` / ``yolo12l`` / ``yolo12x``

Usage
-----
>>> from model_management.model_zoo import build_detection_model
>>> model = build_detection_model("yolov8s", num_classes=91, pretrained=True)
>>> model.eval()
>>> results = model([img_tensor])
>>> results[0]["boxes"], results[0]["labels"], results[0]["scores"]
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger
from torchvision import transforms

# ---------------------------------------------------------------------------
# Required dependencies
# ---------------------------------------------------------------------------
import ultralytics
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    retinanet_resnet50_fpn,
    ssd300_vgg16,
    ssdlite320_mobilenet_v3_large,
    fcos_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    RetinaNet_ResNet50_FPN_Weights,
    SSD300_VGG16_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    FCOS_ResNet50_FPN_Weights,
)

_HAS_ULTRALYTICS = True
_HAS_HF_DETR = True
_HAS_TV_DETECTION = True


# ═══════════════════════════════════════════════════════════════════════
#  COCO 80-class → 91-index mapping (used by YOLO → torchvision label)
# ═══════════════════════════════════════════════════════════════════════

# COCO dataset has 80 categories but uses IDs 1–90 (with gaps).  YOLO
# outputs a contiguous 0-79 class index; torchvision detection models
# use the original COCO IDs (1-indexed).  This LUT maps yolo_cls →
# torchvision label.
COCO_80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
    39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75,
    76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


# ═══════════════════════════════════════════════════════════════════════
#  1. YOLO Wrapper  (ultralytics)
# ═══════════════════════════════════════════════════════════════════════

class YOLODetectionModel(nn.Module):
    """Wraps an Ultralytics YOLO model to produce torchvision-compatible output.

    Parameters
    ----------
    model_name : str
        Any name accepted by ``ultralytics.YOLO()``, e.g.
        ``"yolov8s.pt"``, ``"yolov5m.pt"``, ``"yolo11n.pt"``.
    confidence : float
        Minimum confidence threshold for NMS.
    device : str or torch.device
        Target device.
    num_classes : int
        Number of classes (used for label mapping).
    """

    def __init__(
        self,
        model_name: str = "yolov8s.pt",
        confidence: float = 0.01,
        device: str | torch.device = "cpu",
        num_classes: int = 91,
    ):
        super().__init__()
        if not _HAS_ULTRALYTICS:
            raise ImportError(
                "ultralytics is required for YOLO models.  "
                "Install: pip install ultralytics"
            )
        from ultralytics import YOLO
        self.yolo = YOLO(model_name)
        self.yolo.to(device)
        self.confidence = confidence
        self._device = device
        self.num_classes = num_classes
        # If COCO 80-class, use the mapping; otherwise identity
        self._map_labels = num_classes >= 91

    def forward(
        self, images: List[torch.Tensor], targets=None
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        images : list of Tensor[C, H, W]
            Batch of images (float, 0–1 range).
        targets : ignored (for compatibility with train() mode)

        Returns
        -------
        list of dict  with keys ``boxes``, ``labels``, ``scores``
        """
        results: List[Dict[str, torch.Tensor]] = []
        for img_tensor in images:
            # ultralytics accepts numpy HWC uint8 or PIL or path
            img_np = (
                img_tensor.detach().cpu().permute(1, 2, 0).numpy() * 255
            ).astype("uint8")

            preds = self.yolo.predict(
                img_np, conf=self.confidence, verbose=False,
            )

            if preds and len(preds) > 0 and preds[0].boxes is not None:
                boxes_xyxy = preds[0].boxes.xyxy.cpu()
                scores = preds[0].boxes.conf.cpu()
                cls_ids = preds[0].boxes.cls.cpu().long()

                # Map YOLO 80-class indices → COCO 91-class IDs
                if self._map_labels:
                    labels = torch.tensor(
                        [COCO_80_TO_91[c] if c < 80 else int(c) + 1 for c in cls_ids],
                        dtype=torch.int64,
                    )
                else:
                    labels = cls_ids + 1  # 0-indexed → 1-indexed

                results.append({
                    "boxes": boxes_xyxy.float(),
                    "labels": labels,
                    "scores": scores.float(),
                })
            else:
                results.append({
                    "boxes": torch.zeros(0, 4),
                    "labels": torch.zeros(0, dtype=torch.int64),
                    "scores": torch.zeros(0),
                })
        return results

    def train(self, mode: bool = True):
        """YOLO training is done via ultralytics CLI / API — not via this wrapper."""
        return self

    def eval(self):
        return self.train(False)

    def parameters(self, recurse=True):
        """Return underlying YOLO parameters for device placement etc."""
        return self.yolo.model.parameters(recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.yolo.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.yolo.model.load_state_dict(state_dict, strict=strict)

    def to(self, device):
        self._device = device
        self.yolo.to(device)
        return self


# ═══════════════════════════════════════════════════════════════════════
#  2. DETR Wrapper  (HuggingFace transformers)
# ═══════════════════════════════════════════════════════════════════════

class DETRDetectionModel(nn.Module):
    """Wraps a HuggingFace DETR model to produce torchvision-compatible output.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model id, e.g. ``"facebook/detr-resnet-50"``.
    confidence : float
        Score threshold for filtering predictions.
    device : str or torch.device
    num_classes : int
        Number of detection classes (including background for DETR).
    """

    def __init__(
        self,
        model_name_or_path: str = "facebook/detr-resnet-50",
        confidence: float = 0.5,
        device: str | torch.device = "cpu",
        num_classes: int = 91,
        pretrained: bool = True,
    ):
        super().__init__()
        if not _HAS_HF_DETR:
            raise ImportError(
                "transformers is required for DETR models.  "
                "Install: pip install transformers"
            )
        if pretrained:
            self.processor = DetrImageProcessor.from_pretrained(model_name_or_path)
            self.detr = DetrForObjectDetection.from_pretrained(model_name_or_path)
        else:
            cfg = DetrConfig(
                num_labels=num_classes,
                use_timm_backbone=False,
                backbone=None,
                backbone_config={
                    "model_type": "resnet",
                    "num_channels": 3,
                    "embedding_size": 64,
                    "hidden_sizes": [64, 128, 256, 512],
                    "depths": [2, 2, 2, 2],
                    "hidden_act": "relu",
                },
            )
            self.processor = DetrImageProcessor()
            self.detr = DetrForObjectDetection(cfg)
        self.detr.to(device)
        self.confidence = confidence
        self._device = device
        self.num_classes = num_classes

    def forward(
        self, images: List[torch.Tensor], targets=None
    ) -> List[Dict[str, torch.Tensor]]:
        results: List[Dict[str, torch.Tensor]] = []
        for img_tensor in images:
            # Convert tensor → PIL for the processor
            pil_img = transforms.ToPILImage()(img_tensor.cpu())
            inputs = self.processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            if self.training:
                outputs = self.detr(**inputs)
            else:
                with torch.no_grad():
                    outputs = self.detr(**inputs)

            # Post-process: convert logits → boxes in image coords
            target_sizes = torch.tensor(
                [pil_img.size[::-1]], device=self._device
            )  # (H, W)
            post = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.confidence,
            )[0]

            results.append({
                "boxes": post["boxes"].cpu().float(),
                "labels": post["labels"].cpu().long(),
                "scores": post["scores"].cpu().float(),
            })
        return results

    def train(self, mode: bool = True):
        self.detr.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self, recurse=True):
        return self.detr.parameters(recurse)

    def state_dict(self, *args, **kwargs):
        return self.detr.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.detr.load_state_dict(state_dict, strict=strict)

    def to(self, device):
        self._device = device
        self.detr.to(device)
        return self


# ═══════════════════════════════════════════════════════════════════════
#  3. RT-DETR Wrapper  (ultralytics)
# ═══════════════════════════════════════════════════════════════════════

class RTDETRDetectionModel(nn.Module):
    """Wraps an Ultralytics RT-DETR model (real-time DETR) to produce
    torchvision-compatible output.

    Parameters
    ----------
    model_name : str
        e.g. ``"rtdetr-l.pt"``, ``"rtdetr-x.pt"``
    confidence : float
    device : str or torch.device
    num_classes : int
    """

    def __init__(
        self,
        model_name: str = "rtdetr-l.pt",
        confidence: float = 0.01,
        device: str | torch.device = "cpu",
        num_classes: int = 91,
    ):
        super().__init__()
        if not _HAS_ULTRALYTICS:
            raise ImportError(
                "ultralytics is required for RT-DETR models.  "
                "Install: pip install ultralytics"
            )
        from ultralytics import RTDETR
        self.rtdetr = RTDETR(model_name)
        self.rtdetr.to(device)
        self.confidence = confidence
        self._device = device
        self.num_classes = num_classes
        self._map_labels = num_classes >= 91

    def forward(
        self, images: List[torch.Tensor], targets=None
    ) -> List[Dict[str, torch.Tensor]]:
        results: List[Dict[str, torch.Tensor]] = []
        for img_tensor in images:
            img_np = (
                img_tensor.detach().cpu().permute(1, 2, 0).numpy() * 255
            ).astype("uint8")
            preds = self.rtdetr.predict(img_np, conf=self.confidence, verbose=False)

            if preds and len(preds) > 0 and preds[0].boxes is not None:
                boxes_xyxy = preds[0].boxes.xyxy.cpu()
                scores = preds[0].boxes.conf.cpu()
                cls_ids = preds[0].boxes.cls.cpu().long()

                if self._map_labels:
                    labels = torch.tensor(
                        [COCO_80_TO_91[c] if c < 80 else int(c) + 1 for c in cls_ids],
                        dtype=torch.int64,
                    )
                else:
                    labels = cls_ids + 1

                results.append({
                    "boxes": boxes_xyxy.float(),
                    "labels": labels,
                    "scores": scores.float(),
                })
            else:
                results.append({
                    "boxes": torch.zeros(0, 4),
                    "labels": torch.zeros(0, dtype=torch.int64),
                    "scores": torch.zeros(0),
                })
        return results

    def train(self, mode: bool = True):
        return self

    def eval(self):
        return self.train(False)

    def parameters(self, recurse=True):
        return self.rtdetr.model.parameters(recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self.rtdetr.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.rtdetr.model.load_state_dict(state_dict, strict=strict)

    def to(self, device):
        self._device = device
        self.rtdetr.to(device)
        return self


# ═══════════════════════════════════════════════════════════════════════
#  4. Torchvision built-in (RetinaNet / SSD / FCOS)
# ═══════════════════════════════════════════════════════════════════════

# These already output the standard torchvision detection format, so we
# just need a factory function to instantiate them.

_TORCHVISION_BUILTIN = {
    # Faster R-CNN
    "fasterrcnn_resnet50_fpn":                 lambda **kw: fasterrcnn_resnet50_fpn(**kw),
    "fasterrcnn_mobilenet_v3_large_fpn":       lambda **kw: fasterrcnn_mobilenet_v3_large_fpn(**kw),
    "fasterrcnn_mobilenet_v3_large_320_fpn":   lambda **kw: fasterrcnn_mobilenet_v3_large_320_fpn(**kw),
    # RetinaNet
    "retinanet_resnet50_fpn":                  lambda **kw: retinanet_resnet50_fpn(**kw),
    # SSD
    "ssd300_vgg16":                            lambda **kw: ssd300_vgg16(**kw),
    "ssdlite320_mobilenet_v3_large":           lambda **kw: ssdlite320_mobilenet_v3_large(**kw),
    # FCOS
    "fcos_resnet50_fpn":                       lambda **kw: fcos_resnet50_fpn(**kw),
} if _HAS_TV_DETECTION else {}

_TORCHVISION_WEIGHT_DEFAULTS = {
    "fasterrcnn_resnet50_fpn": FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
    "fasterrcnn_mobilenet_v3_large_fpn": FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
    "fasterrcnn_mobilenet_v3_large_320_fpn": FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
    "retinanet_resnet50_fpn": RetinaNet_ResNet50_FPN_Weights.DEFAULT,
    "ssd300_vgg16": SSD300_VGG16_Weights.DEFAULT,
    "ssdlite320_mobilenet_v3_large": SSDLite320_MobileNet_V3_Large_Weights.DEFAULT,
    "fcos_resnet50_fpn": FCOS_ResNet50_FPN_Weights.DEFAULT,
}


# ═══════════════════════════════════════════════════════════════════════
#  5. YOLO / DETR name mapping
# ═══════════════════════════════════════════════════════════════════════

# Maps user-facing config names → (wrapper_class, constructor_arg)
_YOLO_MODELS: Dict[str, str] = {
    # YOLOv5
    "yolov5n": "yolov5nu.pt",
    "yolov5s": "yolov5su.pt",
    "yolov5m": "yolov5mu.pt",
    "yolov5l": "yolov5lu.pt",
    "yolov5x": "yolov5xu.pt",
    # YOLOv8
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
    "yolov8l": "yolov8l.pt",
    "yolov8x": "yolov8x.pt",
    # YOLOv10
    "yolov10n": "yolov10n.pt",
    "yolov10s": "yolov10s.pt",
    "yolov10m": "yolov10m.pt",
    "yolov10l": "yolov10l.pt",
    "yolov10x": "yolov10x.pt",
    # YOLO11
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo11m": "yolo11m.pt",
    "yolo11l": "yolo11l.pt",
    "yolo11x": "yolo11x.pt",
    # YOLO12
    "yolo12n": "yolo12n.pt",
    "yolo12s": "yolo12s.pt",
    "yolo12m": "yolo12m.pt",
    "yolo12l": "yolo12l.pt",
    "yolo12x": "yolo12x.pt",
}

_DETR_MODELS: Dict[str, str] = {
    "detr_resnet50":         "facebook/detr-resnet-50",
    "detr_resnet101":        "facebook/detr-resnet-101",
    "conditional_detr_resnet50": "microsoft/conditional-detr-resnet-50",
}

_RTDETR_MODELS: Dict[str, str] = {
    "rtdetr_l":  "rtdetr-l.pt",
    "rtdetr_x":  "rtdetr-x.pt",
}


# ═══════════════════════════════════════════════════════════════════════
#  6. Unified factory
# ═══════════════════════════════════════════════════════════════════════

def build_detection_model(
    name: str,
    num_classes: int = 91,
    pretrained: bool = True,
    device: str | torch.device = "cpu",
    weights_path: Optional[str] = None,
    confidence: float = 0.01,
    **kwargs,
) -> nn.Module:
    """Build any supported detection model by name.

    Parameters
    ----------
    name : str
        Model identifier (case-insensitive).  See the module docstring
        for the full list.
    num_classes : int
        Number of detection classes.  91 = COCO (default).
    pretrained : bool
        If ``True``, load pre-trained weights (COCO).
    device : str or torch.device
    weights_path : str, optional
        Path to a local ``.pth`` / ``.pt`` weights file.
        If provided, overrides the default pre-trained weights.
    confidence : float
        Confidence threshold for wrapper models (YOLO / DETR).
    **kwargs
        Extra keyword arguments forwarded to the model constructor.

    Returns
    -------
    nn.Module
        A model whose ``forward([img_tensor])`` returns the standard
        torchvision detection output format.
    """
    name_lower = name.lower().replace("-", "_")

    # ── 1. Torchvision built-in ──
    if name_lower in _TORCHVISION_BUILTIN:
        build_fn = _TORCHVISION_BUILTIN[name_lower]
        if pretrained:
            try:
                model = build_fn(weights=_TORCHVISION_WEIGHT_DEFAULTS[name_lower], **kwargs)
            except TypeError:
                # Backward compatibility for older torchvision APIs.
                model = build_fn(pretrained=True, **kwargs)
        else:
            try:
                model = build_fn(weights=None, weights_backbone=None, **kwargs)
            except TypeError:
                # Some constructors do not expose weights_backbone.
                try:
                    model = build_fn(weights=None, **kwargs)
                except TypeError:
                    model = build_fn(pretrained=False, pretrained_backbone=False, **kwargs)

        if weights_path and os.path.isfile(weights_path):
            state = torch.load(weights_path, map_location=device)
            model.load_state_dict(state)
            logger.info("[ModelZoo] Loaded weights from {}", weights_path)

        model.to(device)
        model.eval()
        logger.info("[ModelZoo] Built torchvision model: {}", name)
        return model

    # ── 2. YOLO ──
    if name_lower in _YOLO_MODELS:
        pt_name = _YOLO_MODELS[name_lower]
        # If user has a local weights file, use it
        if weights_path and os.path.isfile(weights_path):
            pt_name = weights_path
        model = YOLODetectionModel(
            model_name=pt_name,
            confidence=confidence,
            device=device,
            num_classes=num_classes,
        )
        logger.info("[ModelZoo] Built YOLO model: {} ({})", name, pt_name)
        return model

    # ── 3. DETR (HuggingFace) ──
    if name_lower in _DETR_MODELS:
        hf_id = _DETR_MODELS[name_lower]
        if weights_path and os.path.isdir(weights_path):
            hf_id = weights_path
        model = DETRDetectionModel(
            model_name_or_path=hf_id,
            confidence=confidence,
            device=device,
            num_classes=num_classes,
            pretrained=pretrained,
        )
        logger.info("[ModelZoo] Built DETR model: {} ({})", name, hf_id)
        return model

    # ── 4. RT-DETR (ultralytics) ──
    if name_lower in _RTDETR_MODELS:
        pt_name = _RTDETR_MODELS[name_lower]
        if weights_path and os.path.isfile(weights_path):
            pt_name = weights_path
        model = RTDETRDetectionModel(
            model_name=pt_name,
            confidence=confidence,
            device=device,
            num_classes=num_classes,
        )
        logger.info("[ModelZoo] Built RT-DETR model: {} ({})", name, pt_name)
        return model

    # ── 5. Fallback: try torchvision eval ──
    try:
        import torchvision.models.detection as _tv_det_module
        fn = getattr(_tv_det_module, name_lower, None)
        if fn is not None and callable(fn):
            if pretrained:
                try:
                    model = fn(weights="DEFAULT", **kwargs)
                except TypeError:
                    model = fn(pretrained=True, **kwargs)
            else:
                try:
                    model = fn(weights=None, weights_backbone=None, **kwargs)
                except TypeError:
                    try:
                        model = fn(weights=None, **kwargs)
                    except TypeError:
                        model = fn(**kwargs)
            if weights_path and os.path.isfile(weights_path):
                model.load_state_dict(
                    torch.load(weights_path, map_location=device)
                )
            model.to(device)
            model.eval()
            logger.info("[ModelZoo] Built torchvision fallback: {}", name)
            return model
    except Exception:
        pass

    raise ValueError(
        f"Unknown detection model: '{name}'.  "
        f"Available: {list_available_models()}"
    )


# ═══════════════════════════════════════════════════════════════════════
#  7. Helpers
# ═══════════════════════════════════════════════════════════════════════

def list_available_models() -> List[str]:
    """Return sorted list of all available model names."""
    models = list(_TORCHVISION_BUILTIN.keys())
    models += list(_YOLO_MODELS.keys())
    models += list(_DETR_MODELS.keys())
    models += list(_RTDETR_MODELS.keys())
    return sorted(set(models))


def is_wrapper_model(model_or_name) -> bool:
    """Return True if model is a non-torchvision wrapper (YOLO/DETR/RT-DETR).

    Accepts an ``nn.Module`` instance **or** a model name string.
    """
    if isinstance(model_or_name, str):
        return get_model_family(model_or_name) in ('yolo', 'detr', 'rtdetr')
    return isinstance(model_or_name, (YOLODetectionModel, DETRDetectionModel, RTDETRDetectionModel))


def model_has_roi_heads(model_or_name) -> bool:
    """Return True if the model has standard torchvision ``roi_heads``.

    Accepts an ``nn.Module`` instance **or** a model name string.
    """
    if isinstance(model_or_name, str):
        return get_model_family(model_or_name) == 'fasterrcnn'
    return hasattr(model_or_name, "roi_heads")


def get_model_family(name: str) -> str:
    """Return family string: 'fasterrcnn', 'retinanet', 'ssd', 'fcos',
    'yolo', 'detr', 'rtdetr', or 'unknown'."""
    name_lower = name.lower().replace("-", "_")
    if "fasterrcnn" in name_lower:
        return "fasterrcnn"
    if "retinanet" in name_lower:
        return "retinanet"
    if "ssd" in name_lower:
        return "ssd"
    if "fcos" in name_lower:
        return "fcos"
    if name_lower in _YOLO_MODELS:
        return "yolo"
    if name_lower in _DETR_MODELS:
        return "detr"
    if name_lower in _RTDETR_MODELS:
        return "rtdetr"
    return "unknown"
