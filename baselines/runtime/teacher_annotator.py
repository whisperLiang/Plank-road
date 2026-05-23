"""Teacher label loading for real baseline runs."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from baselines.runtime.label_mapping import (
    normalize_detection_for_target,
    normalize_label_schema,
)


@dataclass(frozen=True)
class TeacherAnnotation:
    label_path: str
    latency_sec: float
    from_cache: bool


class TeacherAnnotator:
    """Load cached teacher labels from a real label directory."""

    def __init__(
        self,
        *,
        teacher_model: str | None,
        results_dir: str | Path,
        reuse_cache: bool = True,
        teacher_label_schema: str = "coco_91",
        target_label_schema: str = "target",
        target_class_names: Sequence[str] | None = None,
    ) -> None:
        raw_model = str(teacher_model or "").strip()
        if not raw_model:
            raise ValueError("teacher_model must be an existing teacher label directory")
        ground_truth_dir = Path(raw_model)
        if not ground_truth_dir.exists() or not ground_truth_dir.is_dir():
            raise FileNotFoundError(f"teacher_model must be an existing teacher label directory: {ground_truth_dir}")
        self.teacher_model = "teacher_label_dir"
        self.ground_truth_dir = ground_truth_dir
        teacher_identity = f"teacher_label_dir:{ground_truth_dir.resolve()}"
        self.cache_dir = Path(results_dir) / "teacher_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._teacher_identity = teacher_identity
        self.reuse_cache = bool(reuse_cache)
        self.teacher_label_schema = normalize_label_schema(teacher_label_schema)
        self.target_label_schema = normalize_label_schema(target_label_schema)
        self.target_class_names = [str(item) for item in (target_class_names or [])]

    def configure_target(
        self,
        *,
        label_schema: str | None,
        class_names: Sequence[str] | None = None,
    ) -> None:
        self.target_label_schema = normalize_label_schema(label_schema)
        self.target_class_names = [str(item) for item in (class_names or [])]
        if (
            self.teacher_label_schema != "target"
            and self.teacher_label_schema != self.target_label_schema
            and self.target_label_schema == "zero_based"
            and not self.target_class_names
        ):
            raise ValueError(
                "Baseline teacher labels cannot be mapped safely: "
                "teacher_label_schema is not target/zero_based, student model uses "
                "zero_based labels, and experiment.class_names is empty."
            )

    def annotate(self, frame_path: str | Path) -> TeacherAnnotation:
        source = self._ground_truth_label_path(frame_path)
        labels = self._load_and_validate_label_json(source)
        cache_path = self._cache_path(frame_path)
        if self.reuse_cache and cache_path.exists():
            return TeacherAnnotation(str(cache_path), 0.0, True)

        start = time.perf_counter()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        normalized = self._normalize_labels(labels)
        if normalized == labels:
            shutil.copy2(source, cache_path)
        else:
            with cache_path.open("w", encoding="utf-8") as handle:
                json.dump(normalized, handle, ensure_ascii=False)
        latency = time.perf_counter() - start
        return TeacherAnnotation(str(cache_path), latency, False)

    def _cache_path(self, frame_path: str | Path) -> Path:
        path = Path(frame_path)
        digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{self._cache_namespace()}_{path.stem}_{digest}.json"

    def _cache_namespace(self) -> str:
        mapping_identity = json.dumps(
            {
                "teacher": self._teacher_identity,
                "teacher_label_schema": self.teacher_label_schema,
                "target_label_schema": self.target_label_schema,
                "target_class_names": self.target_class_names,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha1(mapping_identity.encode("utf-8")).hexdigest()[:12]

    def _ground_truth_label_path(self, frame_path: str | Path) -> Path:
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
    def _load_and_validate_label_json(label_path: Path) -> list[dict[str, object]]:
        with label_path.open("r", encoding="utf-8") as f:
            labels = json.load(f)
        if not isinstance(labels, list):
            raise ValueError(f"Teacher label JSON must be a list: {label_path}")
        for item in labels:
            if not isinstance(item, dict):
                raise ValueError(f"Teacher label item must be an object in {label_path}: {item!r}")
            bbox = item.get("bbox", item.get("box"))
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"Teacher label bbox must contain four numbers in {label_path}: {item!r}")
            for value in bbox:
                float(value)
            float(item.get("score", 1.0))
            int(item.get("class_id", item.get("label", 0)))
        return [dict(item) for item in labels]

    def _normalize_labels(self, labels: list[dict[str, object]]) -> list[dict[str, object]]:
        normalized: list[dict[str, object]] = []
        for item in labels:
            mapped = normalize_detection_for_target(
                item,
                source_label_schema=self.teacher_label_schema,
                target_label_schema=self.target_label_schema,
                target_class_names=self.target_class_names,
            )
            if mapped is not None:
                normalized.append(mapped)
        return normalized
