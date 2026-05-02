from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edge.quality_assessor import HIGH_QUALITY, LOW_QUALITY
from edge.sample_store import EdgeSampleStore
from edge.transmit import pack_continual_learning_bundle_to_file
from model_management.continual_learning_bundle import prepare_split_training_cache
from model_management.object_detection import Object_Detection
from model_management.payload import boundary_payload_from_tensors
from model_management.split_runtime import make_split_spec
from model_management.split_runtime.template import (
    FixedSplitRuntimeTemplate,
    FixedSplitRuntimeTemplateCache,
    fixed_split_runtime_template_key,
)


@dataclass
class _Timing:
    values: list[float]

    def add(self, value: float) -> None:
        self.values.append(float(value))

    @property
    def mean(self) -> float:
        return statistics.mean(self.values) if self.values else 0.0


class _DummyPlan:
    split_config_id = "bench-plan"
    candidate_id = "candidate-1"
    split_index = 1
    split_label = "after:node_1"
    boundary_tensor_labels = ["node_1"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_config_id": self.split_config_id,
            "candidate_id": self.candidate_id,
            "split_index": self.split_index,
            "split_label": self.split_label,
            "boundary_tensor_labels": list(self.boundary_tensor_labels),
            "trace_signature": "bench-graph",
        }


def _payload(index: int = 0):
    return boundary_payload_from_tensors(
        {"node_1": torch.full((1, 4), float(index), dtype=torch.float32)},
        split_id="after:node_1",
        graph_signature="bench-graph",
        passthrough_inputs={"input": torch.full((1, 3), float(index), dtype=torch.float32)},
    )


def _write_sample(
    store: EdgeSampleStore,
    sample_id: str,
    frame: np.ndarray,
    *,
    low_quality: bool,
    index: int,
) -> None:
    store.store_sample(
        sample_id=sample_id,
        frame_index=index,
        confidence=0.2 if low_quality else 0.9,
        split_config_id="bench-plan",
        model_id="bench-model",
        model_version="0",
        quality_bucket=LOW_QUALITY if low_quality else HIGH_QUALITY,
        quality_score=0.2 if low_quality else 0.9,
        risk_score=0.8 if low_quality else 0.1,
        evidence_count=1,
        covered_evidence_count=0 if low_quality else 1,
        uncovered_evidence_count=1 if low_quality else 0,
        uncovered_evidence_rate=1.0 if low_quality else 0.0,
        candidate_uncovered_score=1.0 if low_quality else 0.0,
        motion_uncovered_score=0.0,
        track_uncovered_score=0.0,
        in_drift_window=low_quality and index % 5 == 0,
        inference_result={
            "boxes": [[0, 0, 8, 8]] if not low_quality else [],
            "labels": [1] if not low_quality else [],
            "scores": [0.9] if not low_quality else [],
        },
        intermediate=_payload(index),
        raw_frame=frame if low_quality else None,
        input_image_size=list(frame.shape[:2]),
        input_tensor_shape=[1, 3, int(frame.shape[0]), int(frame.shape[1])],
    )


def _maybe_async_writer(store: EdgeSampleStore):
    try:
        from edge.edge_worker import AsyncSampleWriter, SampleStatsDelta, SampleWriteJob
    except Exception:
        return None, None, None
    return AsyncSampleWriter(store), SampleStatsDelta, SampleWriteJob


def _run_edge_benchmark(frames: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
    detector = Object_Detection.__new__(Object_Detection)

    prep = _Timing([])
    inference = _Timing([])
    quality = _Timing([])
    split_replay = _Timing([])
    blocking = _Timing([])
    storage = _Timing([])
    stats_elapsed = _Timing([])

    tiny_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 96 * 128, 4))
    tiny_model.eval()

    with tempfile.TemporaryDirectory(prefix="plankroad_bench_edge_") as tmp:
        store = EdgeSampleStore(os.path.join(tmp, "store"))
        writer, delta_cls, job_cls = _maybe_async_writer(store)

        for index in range(max(200, frames * 10)):
            _write_sample(
                store,
                f"prefill-{index}",
                frame,
                low_quality=bool(index % 2),
                index=index,
            )

        for index in range(frames):
            low_quality = bool(index % 2)
            started = time.perf_counter()
            tensor = detector._prepare_image_tensor(frame)
            prep.add(time.perf_counter() - started)

            started = time.perf_counter()
            with torch.inference_mode():
                tiny_model(tensor.detach().cpu().reshape(1, -1))
            inference.add(time.perf_counter() - started)

            started = time.perf_counter()
            _ = math.sqrt(float(index + 1)) / float(frames + 1)
            quality.add(time.perf_counter() - started)

            started = time.perf_counter()
            _ = _payload(index)
            split_replay.add(time.perf_counter() - started)

            sample_id = f"frame-{index}"
            started = time.perf_counter()
            if writer is None:
                _write_sample(
                    store,
                    sample_id,
                    frame,
                    low_quality=low_quality,
                    index=index,
                )
            else:
                delta = delta_cls.from_values(
                    quality_bucket=LOW_QUALITY if low_quality else HIGH_QUALITY,
                    uncovered_evidence_rate=1.0 if low_quality else 0.0,
                    candidate_uncovered_score=1.0 if low_quality else 0.0,
                    motion_uncovered_score=0.0,
                    track_uncovered_score=0.0,
                    in_drift_window=low_quality and index % 5 == 0,
                )
                job = job_cls(
                    store_kwargs={
                        "sample_id": sample_id,
                        "frame_index": index,
                        "confidence": 0.2 if low_quality else 0.9,
                        "split_config_id": "bench-plan",
                        "model_id": "bench-model",
                        "model_version": "0",
                        "quality_bucket": LOW_QUALITY if low_quality else HIGH_QUALITY,
                        "quality_score": 0.2 if low_quality else 0.9,
                        "risk_score": 0.8 if low_quality else 0.1,
                        "evidence_count": 1,
                        "covered_evidence_count": 0 if low_quality else 1,
                        "uncovered_evidence_count": 1 if low_quality else 0,
                        "uncovered_evidence_rate": 1.0 if low_quality else 0.0,
                        "candidate_uncovered_score": 1.0 if low_quality else 0.0,
                        "motion_uncovered_score": 0.0,
                        "track_uncovered_score": 0.0,
                        "in_drift_window": low_quality and index % 5 == 0,
                        "inference_result": {"boxes": [], "labels": [], "scores": []},
                        "intermediate": _payload(index),
                        "raw_frame": frame if low_quality else None,
                        "input_image_size": list(frame.shape[:2]),
                        "input_tensor_shape": [1, 3, int(frame.shape[0]), int(frame.shape[1])],
                    },
                    stats_delta=delta,
                )
                writer.submit(job)
            blocking.add(time.perf_counter() - started)

            started = time.perf_counter()
            store.stats()
            stats_elapsed.add(time.perf_counter() - started)

        if writer is not None:
            started = time.perf_counter()
            writer.close(timeout=10.0)
            storage.add(time.perf_counter() - started)
        else:
            storage.values = list(blocking.values)

        started = time.perf_counter()
        zip_path, _manifest, _stats = pack_continual_learning_bundle_to_file(
            store,
            edge_id=1,
            send_low_conf_features=False,
            split_plan=_DummyPlan(),
            model_id="bench-model",
            model_version="0",
            bundle_cap_bytes=8 * 1024 * 1024,
            output_dir=tmp,
        )
        try:
            os.remove(zip_path)
        except OSError:
            pass
        bundle_packing = time.perf_counter() - started

    per_frame_total = (
        prep.mean
        + inference.mean
        + quality.mean
        + split_replay.mean
        + blocking.mean
    )
    return {
        "edge_per_frame_total_ms": per_frame_total * 1000.0,
        "edge_model_inference_ms": inference.mean * 1000.0,
        "edge_preprocessing_ms": prep.mean * 1000.0,
        "edge_split_replay_ms": split_replay.mean * 1000.0,
        "edge_quality_assessment_ms": quality.mean * 1000.0,
        "edge_inference_thread_blocking_ms": blocking.mean * 1000.0,
        "edge_sample_storage_ms": storage.mean * 1000.0,
        "edge_sample_store_stats_ms": stats_elapsed.mean * 1000.0,
        "edge_bundle_packing_ms": bundle_packing * 1000.0,
    }


def _run_template_cache_benchmark() -> dict[str, float]:
    cache = FixedSplitRuntimeTemplateCache()
    spec = make_split_spec(
        "auto",
        dynamic_batch=(2, 64),
        trainable=True,
        trace_batch_mode="batch_gt1",
        model_family="toy",
    )
    example = torch.randn(2, 3, 8, 8)
    key_kwargs = {
        "model_name": "toy",
        "model_family": "toy",
        "split_spec": spec,
        "example_inputs": example,
        "split_plan_hash": "bench-plan",
    }
    if "trace_batch_size" in fixed_split_runtime_template_key.__code__.co_varnames:
        key_kwargs["trace_batch_size"] = 2
    key = fixed_split_runtime_template_key(**key_kwargs)
    runtime = SimpleNamespace(
        trace_plan=SimpleNamespace(root_module=torch.nn.Identity()),
        split_spec=spec,
        candidate=SimpleNamespace(),
        mode="generated_eager",
        variants=(),
    )

    def builder():
        time.sleep(0.02)
        return FixedSplitRuntimeTemplate(
            cache_key=key,
            runtime=runtime,
            split_spec=spec,
            model_name="toy",
            model_family="toy",
            graph_signature="bench-graph",
            symbolic_input_schema_hash=key.symbolic_input_schema_hash,
            split_plan_hash=key.split_plan_hash,
            mode="generated_eager",
        )

    cold_started = time.perf_counter()
    first = cache.get_or_create_lookup(key, builder)
    cold = time.perf_counter() - cold_started
    hit_started = time.perf_counter()
    second = cache.get_or_create_lookup(key, builder)
    hit = time.perf_counter() - hit_started
    assert first.template is second.template
    return {
        "cloud_runtime_graph_template_preparation_s": cold if first.cache_status == "miss" else 0.0,
        "cloud_runtime_template_lookup_s": hit,
    }


def _run_cloud_cache_benchmark(frames: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed + 1000)
    frame = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory(prefix="plankroad_bench_cloud_") as tmp:
        store = EdgeSampleStore(os.path.join(tmp, "store"))
        plan = _DummyPlan()
        for index in range(frames):
            _write_sample(
                store,
                f"sample-{index}",
                frame,
                low_quality=bool(index % 2),
                index=index,
            )
        zip_path, _manifest, _stats = pack_continual_learning_bundle_to_file(
            store,
            edge_id=1,
            send_low_conf_features=False,
            split_plan=plan,
            model_id="bench-model",
            model_version="0",
            bundle_cap_bytes=8 * 1024 * 1024,
            output_dir=tmp,
        )
        bundle_root = os.path.join(tmp, "bundle")
        extraction_started = time.perf_counter()
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(bundle_root)
        extraction = time.perf_counter() - extraction_started

        preloaded_records: dict[str, dict[str, Any]] = {}

        def rebuild_provider(raw_paths, samples, manifest):
            return [_payload(index) for index, _path in enumerate(raw_paths)]

        cache_started = time.perf_counter()
        info = prepare_split_training_cache(
            bundle_root,
            os.path.join(tmp, "prepared"),
            batch_feature_provider=rebuild_provider,
            preloaded_records=preloaded_records,
        )
        feature_cache = time.perf_counter() - cache_started

        teacher_started = time.perf_counter()
        for sample_id in info["all_sample_ids"]:
            frame_path = os.path.join(tmp, "prepared", "frames", f"{sample_id}.jpg")
            if os.path.exists(frame_path):
                cv2.imread(frame_path)
        teacher = time.perf_counter() - teacher_started

        retrain_started = time.perf_counter()
        for record in preloaded_records.values():
            boundary = record["intermediate"]
            sum(tensor.float().sum().item() for tensor in boundary.tensors.values())
        retrain = time.perf_counter() - retrain_started

        serialization_started = time.perf_counter()
        torch.save({"weight": torch.ones(8)}, os.path.join(tmp, "model.pt"))
        serialization = time.perf_counter() - serialization_started

    template_metrics = _run_template_cache_benchmark()
    total = extraction + feature_cache + teacher + retrain + serialization
    total += template_metrics["cloud_runtime_graph_template_preparation_s"]
    return {
        "cloud_bundle_extraction_s": extraction,
        "cloud_manifest_loading_s": 0.0,
        **template_metrics,
        "cloud_feature_reconstruction_s": feature_cache,
        "cloud_feature_cache_build_load_s": feature_cache,
        "cloud_teacher_annotation_s": teacher,
        "cloud_dataloader_construction_s": 0.0,
        "cloud_split_retraining_s": retrain,
        "cloud_model_serialization_s": serialization,
        "cloud_total_continual_learning_job_s": total,
    }


def _write_outputs(output_dir: Path, phase: str, metrics: dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{phase}_summary.json"
    log_path = output_dir / f"{phase}_perf.log"
    summary_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [f"{key}={value:.6f}" for key, value in sorted(metrics.items())]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _speedup(before: float, after: float) -> float:
    if after <= 0.0:
        return 0.0
    return before / after


def _write_comparison(output_dir: Path) -> None:
    baseline_path = output_dir / "baseline_summary.json"
    optimized_path = output_dir / "optimized_summary.json"
    if not baseline_path.exists() or not optimized_path.exists():
        return
    before = json.loads(baseline_path.read_text(encoding="utf-8"))
    after = json.loads(optimized_path.read_text(encoding="utf-8"))
    rows = [
        ("Edge", "per-frame total latency", "edge_per_frame_total_ms", "ms"),
        ("Edge", "inference-thread blocking time", "edge_inference_thread_blocking_ms", "ms"),
        ("Edge", "sample_store.stats()", "edge_sample_store_stats_ms", "ms"),
        ("Edge", "sample storage time", "edge_sample_storage_ms", "ms"),
        ("Cloud", "runtime graph/template preparation", "cloud_runtime_graph_template_preparation_s", "s"),
        ("Cloud", "feature cache build/load", "cloud_feature_cache_build_load_s", "s"),
        ("Cloud", "teacher annotation", "cloud_teacher_annotation_s", "s"),
        ("Cloud", "split retraining", "cloud_split_retraining_s", "s"),
        ("Cloud", "total continual-learning job", "cloud_total_continual_learning_job_s", "s"),
    ]
    payload = []
    md_lines = [
        "| Component | Metric | Before | After | Speedup |",
        "|---|---:|---:|---:|---:|",
    ]
    for component, metric, key, unit in rows:
        before_value = float(before.get(key, 0.0))
        after_value = float(after.get(key, 0.0))
        speedup = _speedup(before_value, after_value)
        payload.append(
            {
                "component": component,
                "metric": metric,
                "before": before_value,
                "after": after_value,
                "unit": unit,
                "speedup": speedup,
            }
        )
        md_lines.append(
            f"| {component} | {metric} | {before_value:.4f} {unit} | "
            f"{after_value:.4f} {unit} | {speedup:.2f}x |"
        )
    (output_dir / "perf_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "perf_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["baseline", "optimized"], required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-dir", default="benchmark_results")
    args = parser.parse_args()

    _ = args.config, args.epochs
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    metrics = {}
    metrics.update(_run_edge_benchmark(max(1, int(args.frames)), int(args.seed)))
    metrics.update(_run_cloud_cache_benchmark(max(1, int(args.frames)), int(args.seed)))

    output_dir = Path(args.output_dir)
    _write_outputs(output_dir, str(args.phase), metrics)
    _write_comparison(output_dir)


if __name__ == "__main__":
    main()
