"""Run the real baseline advantage experiment matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.experiment_utils import display_name_for_method, validate_method_name
from baselines.runtime.student_inferencer import resolve_torch_device
from config.experiment import ExperimentConfig
from tools.baselines_real_common import (
    TRAINING_BREAKDOWN_FIELDNAMES,
    UPDATE_EVENT_FIELDNAMES,
    UPLOAD_EVENT_FIELDNAMES,
    compute_capacity_summary,
    run_one_experiment_case,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _run_variants(method_name: str, raw: dict[str, Any]) -> list[str]:
    ablation = raw.get("plank_road_ablation", {}) or {}
    if method_name != "plank_road_multi_device":
        return ["default"]
    if not bool(ablation.get("enabled", False)):
        return ["default"]
    variants = ablation.get("variants", {}) or {}
    return list(variants.keys()) or ["full"]


def _variant_settings(method_name: str, variant: str, raw: dict[str, Any]) -> dict[str, Any] | None:
    if method_name != "plank_road_multi_device":
        return None
    ablation = raw.get("plank_road_ablation", {}) or {}
    variants = ablation.get("variants", {}) or {}
    settings = variants.get(variant)
    if not isinstance(settings, dict):
        return None
    return dict(settings)


def _video_paths(raw: dict[str, Any]) -> str:
    videos = raw.get("dataset", {}).get("videos", []) or []
    paths = [str(item["path"]) for item in videos if item.get("path")]
    if not paths:
        raise ValueError("dataset.videos must include at least one path")
    return ",".join(paths)


def _teacher_label_dir(raw: dict[str, Any]) -> str:
    model = raw.get("model", {}) or {}
    teacher = model.get("teacher_label_dir")
    if teacher:
        return str(teacher)
    videos = raw.get("dataset", {}).get("videos", []) or []
    labels = [str(item["labels"]) for item in videos if item.get("labels")]
    if labels:
        return labels[0]
    raise ValueError("model.teacher_label_dir or dataset.videos[].labels is required")


def _build_config(
    *,
    raw: dict[str, Any],
    method_name: str,
    variant: str,
    repeat_id: int,
    num_edges: int,
    bandwidth_mbps: float,
    max_concurrent_train_jobs: int,
    run_id: str,
    run_dir: Path,
) -> ExperimentConfig:
    exp = raw.get("experiment", {}) or {}
    dataset = raw.get("dataset", {}) or {}
    model = raw.get("model", {}) or {}
    runtime = raw.get("runtime", {}) or {}
    device = str(runtime.get("device", "cpu"))
    resolved_device = resolve_torch_device(device)
    if str(resolved_device) != device:
        print(f"[advantage] Requested {device}, using {resolved_device}.", file=sys.stderr)
        device = str(resolved_device)
    return ExperimentConfig(
        method=method_name,
        num_devices=int(num_edges),
        total_frames=int(dataset.get("total_frames", 512)),
        results_dir=str(run_dir),
        video_path=_video_paths(raw),
        student_model=str(model.get("student_model", "yolo26")),
        teacher_model=_teacher_label_dir(raw),
        initial_checkpoint=model.get("initial_checkpoint"),
        seed=int(exp.get("seed", 2026)),
        repeat_id=int(repeat_id),
        run_id=run_id,
        method_variant=variant if method_name == "plank_road_multi_device" else "default",
        window_seconds=dataset.get("window_seconds", 10),
        window_frames=dataset.get("window_frames"),
        batch_size=int(runtime.get("batch_size", 2)),
        epochs=int(runtime.get("epochs", 1)),
        device=device,
        bandwidth_mbps=float(bandwidth_mbps),
        max_concurrent_train_jobs=int(max_concurrent_train_jobs),
        reuse_teacher_cache=True,
        quick_smoke=False,
        f1_threshold=runtime.get("f1_threshold"),
        map50_threshold=runtime.get("map50_threshold"),
        recovery_sla_sec=float(runtime.get("recovery_sla_sec", 120)),
        latency_sla_ms=runtime.get("latency_sla_ms"),
        capacity_mode=True,
    )


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    exp = raw.get("experiment", {}) or {}
    runtime = raw.get("runtime", {}) or {}
    methods = [str(method) for method in exp.get("methods", [])]
    if not methods:
        raise ValueError("experiment.methods must not be empty")
    for method in methods:
        validate_method_name(method)

    teacher_dir = Path(_teacher_label_dir(raw))
    if not teacher_dir.exists() or not teacher_dir.is_dir():
        raise FileNotFoundError(f"teacher_label_dir does not exist: {teacher_dir}")

    results_dir = Path(exp.get("results_dir", "results/baselines_real_advantage"))
    results_dir.mkdir(parents=True, exist_ok=True)
    repeats = int(exp.get("repeats", 1))
    all_summary_rows: list[dict[str, Any]] = []
    all_update_rows: list[dict[str, Any]] = []
    all_upload_rows: list[dict[str, Any]] = []
    all_breakdown_rows: list[dict[str, Any]] = []

    for repeat_id in range(repeats):
        for method_name in methods:
            for variant in _run_variants(method_name, raw):
                for num_edges in _as_list(runtime.get("num_edges", [1])):
                    for bandwidth in _as_list(runtime.get("bandwidth_mbps", [50])):
                        for jobs in _as_list(runtime.get("max_concurrent_train_jobs", [1])):
                            safe_variant = variant.replace("/", "_")
                            run_id = (
                                f"r{repeat_id}_{method_name}_{safe_variant}_"
                                f"e{int(num_edges)}_bw{float(bandwidth):g}_q{int(jobs)}"
                            )
                            run_dir = results_dir / "runs" / run_id
                            config = _build_config(
                                raw=raw,
                                method_name=method_name,
                                variant=variant,
                                repeat_id=repeat_id,
                                num_edges=int(num_edges),
                                bandwidth_mbps=float(bandwidth),
                                max_concurrent_train_jobs=int(jobs),
                                run_id=run_id,
                                run_dir=run_dir,
                            )
                            variant_settings = _variant_settings(method_name, variant, raw)
                            result = run_one_experiment_case(
                                config=config,
                                method_names=[method_name],
                                root_results=run_dir,
                                method_variants={method_name: variant},
                                method_variant_configs=(
                                    {method_name: variant_settings}
                                    if variant_settings is not None
                                    else {}
                                ),
                                write_outputs=True,
                            )
                            summary = result["method_summaries"][0]
                            all_summary_rows.append(summary)
                            all_update_rows.extend(result["update_event_rows"])
                            all_upload_rows.extend(result["upload_event_rows"])
                            all_breakdown_rows.extend(result["training_breakdown_rows"])
                            print(
                                "[advantage] "
                                f"method={method_name} "
                                f"display_name={display_name_for_method(method_name)} "
                                f"method_variant={summary.get('method_variant', 'default')} "
                                f"num_edges={num_edges} "
                                f"bandwidth_mbps={bandwidth} "
                                f"repeat_id={repeat_id} "
                                f"summary_path={run_dir / 'summary.json'} "
                                f"sla_satisfied={summary.get('sla_satisfied')}"
                            )

    capacity_rows = compute_capacity_summary(all_summary_rows)
    write_csv(results_dir / "all_summary.csv", all_summary_rows)
    write_csv(results_dir / "all_update_events.csv", all_update_rows, UPDATE_EVENT_FIELDNAMES)
    write_csv(results_dir / "all_upload_events.csv", all_upload_rows, UPLOAD_EVENT_FIELDNAMES)
    write_csv(results_dir / "all_training_breakdown.csv", all_breakdown_rows, TRAINING_BREAKDOWN_FIELDNAMES)
    write_csv(results_dir / "capacity_summary.csv", capacity_rows)
    print(f"Wrote advantage experiment matrix to {results_dir}")


if __name__ == "__main__":
    main()
