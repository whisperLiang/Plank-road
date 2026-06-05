from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureCacheSettings:
    store_root_dir: str
    view_root_dir: str
    storage_format: str
    accepted_storage_formats: tuple[str, ...]
    materialization_mode: str
    view_source: str


@dataclass(frozen=True)
class TeacherAnnotationSettings:
    async_enabled: bool
    cache_enabled: bool
    wait_timeout_sec: float
    worker_batch_size: int
    worker_max_queue_size: int
    worker_max_retries: int
    oom_retry_enabled: bool
    min_worker_batch_size: int
    cache_root_dir: str


@dataclass(frozen=True)
class SamplePoolSettings:
    enabled: bool
    root_dir: str
    staging_root: str
    split_contract_root: str
    max_active_samples: int | None
    shard_size: int


@dataclass(frozen=True)
class OrchestrationSettings:
    edge_model_name: str
    workspace_root: str
    default_num_epoch: int
    max_concurrent_jobs: int
    batch_size: int
    trace_batch_size: int
    feature_cache: FeatureCacheSettings
    teacher_annotation: TeacherAnnotationSettings
    sample_pool: SamplePoolSettings

    @classmethod
    def from_config(cls, config: Any) -> "OrchestrationSettings":
        cl_cfg = getattr(config, "continual_learning", None)
        feature_cache_cfg = getattr(cl_cfg, "feature_cache", None) if cl_cfg is not None else None
        sample_pool_cfg = getattr(config, "sample_pool", None)
        workspace_root = os.path.abspath(str(getattr(config, "workspace_root", "./cache/server_workspace")))
        sample_pool_root = os.path.abspath(
            str(
                getattr(
                    sample_pool_cfg,
                    "root_dir",
                    os.path.join(workspace_root, "cloud_sample_pool"),
                )
            )
        )
        teacher_cfg = getattr(cl_cfg, "teacher_annotation", None) if cl_cfg is not None else None
        raw_sample_pool_max = (
            getattr(sample_pool_cfg, "max_samples", None)
            if sample_pool_cfg is not None
            else getattr(cl_cfg, "sample_pool_max_active_samples", None)
            if cl_cfg
            else None
        )
        return cls(
            edge_model_name=str(getattr(config, "edge_model_name", "rfdetr_nano")),
            workspace_root=workspace_root,
            default_num_epoch=int(getattr(cl_cfg, "num_epoch", 2)) if cl_cfg else 2,
            max_concurrent_jobs=int(getattr(cl_cfg, "max_concurrent_jobs", 2)) if cl_cfg else 2,
            batch_size=int(getattr(cl_cfg, "batch_size", 2)) if cl_cfg else 2,
            trace_batch_size=int(getattr(cl_cfg, "trace_batch_size", 2)) if cl_cfg else 2,
            feature_cache=FeatureCacheSettings(
                store_root_dir=os.path.abspath(
                    str(
                        getattr(
                            feature_cache_cfg,
                            "shard_root_dir",
                            getattr(feature_cache_cfg, "store_root_dir", "./cache/cloud_feature_shards"),
                        )
                    )
                ),
                view_root_dir=os.path.abspath(
                    str(getattr(feature_cache_cfg, "view_root_dir", "./cache/cloud_training_views"))
                ),
                storage_format=str(getattr(feature_cache_cfg, "storage_format", "safetensors_shard")).strip().lower(),
                accepted_storage_formats=tuple(
                    str(item).strip().lower()
                    for item in list(
                        getattr(
                            feature_cache_cfg,
                            "accepted_storage_formats",
                            ["safetensors_shard", "npy_memmap_shard"],
                        )
                        or []
                    )
                ),
                materialization_mode=str(getattr(feature_cache_cfg, "materialization_mode", "direct_ref")).strip().lower(),
                view_source=str(getattr(feature_cache_cfg, "view_source", "canonical_active")).strip().lower(),
            ),
            teacher_annotation=TeacherAnnotationSettings(
                async_enabled=bool(getattr(teacher_cfg, "async_enabled", False)),
                cache_enabled=bool(getattr(teacher_cfg, "cache_enabled", True)),
                wait_timeout_sec=float(getattr(teacher_cfg, "wait_timeout_sec", 0.5)),
                worker_batch_size=int(getattr(teacher_cfg, "worker_batch_size", 16)),
                worker_max_queue_size=int(getattr(teacher_cfg, "worker_max_queue_size", 4096)),
                worker_max_retries=int(getattr(teacher_cfg, "worker_max_retries", 2)),
                oom_retry_enabled=bool(getattr(teacher_cfg, "oom_retry_enabled", True)),
                min_worker_batch_size=int(getattr(teacher_cfg, "min_worker_batch_size", 1)),
                cache_root_dir=os.path.abspath(
                    str(getattr(teacher_cfg, "cache_root_dir", "./cache/teacher_label_cache"))
                ),
            ),
            sample_pool=SamplePoolSettings(
                enabled=bool(getattr(sample_pool_cfg, "enabled", True)) if sample_pool_cfg is not None else True,
                root_dir=sample_pool_root,
                staging_root=os.path.abspath(
                    str(
                        getattr(
                            sample_pool_cfg,
                            "staging_root",
                            os.path.join(os.path.dirname(sample_pool_root), "cloud_sample_staging"),
                        )
                    )
                ),
                split_contract_root=os.path.abspath(
                    str(
                        getattr(
                            sample_pool_cfg,
                            "split_contract_root",
                            os.path.join(os.path.dirname(workspace_root), "split_contracts"),
                        )
                    )
                ),
                max_active_samples=None if raw_sample_pool_max in (None, "", 0) else int(raw_sample_pool_max),
                shard_size=max(1, int(getattr(sample_pool_cfg, "shard_size", 64))) if sample_pool_cfg is not None else 64,
            ),
        )
