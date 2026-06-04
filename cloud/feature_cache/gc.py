from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from typing import Any

from loguru import logger

from cloud.feature_cache.feature_store import STORE_VERSION
from cloud.feature_cache.types import FeatureCacheGCResult


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return dict(payload) if isinstance(payload, Mapping) else {}


class FeatureCacheGC:
    def __init__(
        self,
        *,
        store_root_dir: str,
        view_root_dir: str | None = None,
        max_live_generations: int = 3,
        dry_run: bool = False,
    ) -> None:
        self.store_root_dir = os.path.abspath(str(store_root_dir))
        self.version_root = os.path.join(self.store_root_dir, STORE_VERSION)
        self.view_root_dir = (
            None if view_root_dir in (None, "") else os.path.abspath(str(view_root_dir))
        )
        self.max_live_generations = max(0, int(max_live_generations))
        self.dry_run = bool(dry_run)

    @staticmethod
    def _normalise_live_paths(paths: Iterable[str] | None) -> set[str]:
        return {
            os.path.abspath(str(path))
            for path in list(paths or [])
            if str(path or "").strip()
        }

    def _view_live_paths(self) -> set[str]:
        live: set[str] = set()
        if not self.view_root_dir or not os.path.isdir(self.view_root_dir):
            return live
        views = [
            os.path.join(self.view_root_dir, name)
            for name in os.listdir(self.view_root_dir)
            if os.path.isdir(os.path.join(self.view_root_dir, name))
        ]
        views.sort(key=lambda path: os.path.getmtime(path), reverse=True)
        for view_dir in views[: self.max_live_generations]:
            for filename in ("view_manifest.json", "metadata_index.json"):
                path = os.path.join(view_dir, filename)
                if not os.path.exists(path):
                    continue
                try:
                    payload = _read_json(path)
                except Exception:
                    continue
                samples = payload.get("samples")
                if isinstance(samples, Mapping):
                    iterable = samples.values()
                elif isinstance(samples, list):
                    iterable = samples
                else:
                    iterable = []
                for sample in iterable:
                    if not isinstance(sample, Mapping):
                        continue
                    ref = sample.get("feature_ref")
                    if isinstance(ref, Mapping) and ref.get("path"):
                        live.add(os.path.abspath(str(ref.get("path"))))
                    elif sample.get("feature_relpath"):
                        relpath = str(sample.get("feature_relpath"))
                        live.add(
                            os.path.abspath(
                                relpath if os.path.isabs(relpath) else os.path.join(view_dir, relpath)
                            )
                        )
        return live

    def collect(
        self,
        *,
        live_feature_paths: Iterable[str] | None = None,
        dry_run: bool | None = None,
    ) -> FeatureCacheGCResult:
        effective_dry_run = self.dry_run if dry_run is None else bool(dry_run)
        live = self._normalise_live_paths(live_feature_paths)
        live.update(self._view_live_paths())
        result = FeatureCacheGCResult(dry_run=effective_dry_run)
        if not os.path.isdir(self.version_root):
            logger.info(
                "[FeatureCache][GC] dry_run={} scanned=0 retained=0 deleted=0 deleted_bytes=0",
                effective_dry_run,
            )
            return result

        for root, _dirs, files in os.walk(self.version_root):
            for filename in files:
                if not filename.endswith(".pt"):
                    continue
                path = os.path.abspath(os.path.join(root, filename))
                result.scanned_files += 1
                if path in live:
                    result.retained_files += 1
                    if len(result.retained_files_preview) < 10:
                        result.retained_files_preview.append(path)
                    continue
                size = os.path.getsize(path) if os.path.exists(path) else 0
                result.orphan_files.append(path)
                if effective_dry_run:
                    continue
                try:
                    os.remove(path)
                    meta_path = f"{os.path.splitext(path)[0]}.meta.json"
                    if os.path.exists(meta_path):
                        os.remove(meta_path)
                    result.deleted_files += 1
                    result.deleted_bytes += int(size)
                except OSError as exc:
                    result.errors[path] = str(exc)

        logger.info(
            "[FeatureCache][GC] dry_run={} scanned={} retained={} orphan={} deleted={} deleted_bytes={} errors={}",
            effective_dry_run,
            result.scanned_files,
            result.retained_files,
            len(result.orphan_files),
            result.deleted_files,
            result.deleted_bytes,
            len(result.errors),
        )
        return result


__all__ = ["FeatureCacheGC"]
