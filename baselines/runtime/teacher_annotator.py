"""Teacher/oracle annotation for real baseline runs."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class TeacherAnnotation:
    label_path: str
    latency_sec: float
    from_cache: bool


class TeacherAnnotator:
    """Generate or load pseudo labels for real baseline runs."""

    SUPPORTED_TEACHERS = {"cv_oracle"}

    def __init__(
        self,
        *,
        teacher_model: str | None,
        results_dir: str | Path,
        reuse_cache: bool = True,
        allow_cv_oracle: bool = False,
    ) -> None:
        raw_model = str(teacher_model or "").strip()
        possible_path = Path(raw_model) if raw_model else None
        if possible_path is not None and possible_path.exists() and possible_path.is_dir():
            self.teacher_model = "ground_truth_dir"
            self.ground_truth_dir = possible_path
            teacher_identity = f"ground_truth_dir:{possible_path.resolve()}"
        else:
            self.teacher_model = (raw_model or "cv_oracle").lower().replace("-", "_")
            self.ground_truth_dir = None
            teacher_identity = f"teacher_model:{self.teacher_model}"
        if self.teacher_model == "cv_oracle" and not allow_cv_oracle:
            raise RuntimeError(
                "cv_oracle is a smoke-test image-processing oracle, not a paper teacher. "
                "Pass --quick-smoke or provide a ground-truth label directory."
            )
        if self.teacher_model not in self.SUPPORTED_TEACHERS and self.teacher_model != "ground_truth_dir":
            raise NotImplementedError(
                f"Teacher model {teacher_model!r} is not supported by the real baseline runner. "
                "Provide a directory of detection JSON labels, or use cv_oracle only for quick smoke."
            )
        self.cache_dir = Path(results_dir) / "teacher_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_namespace = hashlib.sha1(teacher_identity.encode("utf-8")).hexdigest()[:12]
        self.reuse_cache = bool(reuse_cache)

    def annotate(self, frame_path: str | Path) -> TeacherAnnotation:
        cache_path = self._cache_path(frame_path)
        if self.reuse_cache and cache_path.exists():
            return TeacherAnnotation(str(cache_path), 0.0, True)

        start = time.perf_counter()
        if self.teacher_model == "ground_truth_dir":
            source = self._ground_truth_label_path(frame_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, cache_path)
            latency = time.perf_counter() - start
            return TeacherAnnotation(str(cache_path), latency, False)

        detections = self._cv_oracle(frame_path)
        latency = time.perf_counter() - start
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(detections, f)
        return TeacherAnnotation(str(cache_path), latency, False)

    def _cache_path(self, frame_path: str | Path) -> Path:
        path = Path(frame_path)
        digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{self.cache_namespace}_{path.stem}_{digest}.json"

    def _ground_truth_label_path(self, frame_path: str | Path) -> Path:
        if self.ground_truth_dir is None:
            raise RuntimeError("Ground-truth label directory is not configured")
        stem = Path(frame_path).stem
        for candidate in (
            self.ground_truth_dir / f"{stem}.json",
            self.ground_truth_dir / f"{int(stem):08d}.json" if stem.isdigit() else self.ground_truth_dir / f"{stem}.json",
        ):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No ground-truth label JSON found for frame {frame_path} in {self.ground_truth_dir}"
        )

    @staticmethod
    def _cv_oracle(frame_path: str | Path) -> list[dict[str, Any]]:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise FileNotFoundError(f"Unable to read frame for teacher annotation: {frame_path}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if float(np.mean(mask)) > 127.0:
            mask = cv2.bitwise_not(mask)
        contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = gray.shape[:2]
        min_area = max(16.0, 0.0025 * float(height * width))
        detections: list[dict[str, Any]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 1 or h <= 1:
                continue
            detections.append(
                {
                    "bbox": [float(x), float(y), float(x + w), float(y + h)],
                    "score": 1.0,
                    "class_id": 1,
                }
            )
        detections.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        return detections
