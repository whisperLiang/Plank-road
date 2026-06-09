"""Run one real-execution baseline experiment over video detection streams."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.experiment_utils import display_name_for_method
from baselines.runtime.student_inferencer import resolve_torch_device
from config.experiment import ExperimentConfig, VALID_METHODS
from config.runtime import load_runtime_config
from tools.baselines_real_common import run_one_experiment_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Video file or image directory. Repeat with comma-separated paths.")
    parser.add_argument("--methods", default=",".join(VALID_METHODS))
    parser.add_argument("--student-model", default="yolo26")
    parser.add_argument("--student-weights", help="Optional local student weights path.")
    parser.add_argument(
        "--tinynext-input-size",
        type=int,
        help="TinyNeXt square input size. Defaults to client.tinynext_input_size from --runtime-config.",
    )
    parser.add_argument(
        "--tinynext-anchor-profile",
        help="TinyNeXt anchor profile. Defaults to client.tinynext_anchor_profile from --runtime-config.",
    )
    parser.add_argument(
        "--runtime-config",
        default="config/config.yaml",
        help="Runtime config used as a fallback source for client.class_names.",
    )
    parser.add_argument(
        "--class-names",
        default="",
        help="Comma-separated student model class names in zero-based order.",
    )
    parser.add_argument(
        "--teacher-label-schema",
        default="coco_91",
        help="Teacher JSON label schema, e.g. coco_91 or target.",
    )
    parser.add_argument("--teacher-model", required=True, help="Existing directory of teacher label JSON files.")
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--window-frames", type=int)
    parser.add_argument("--total-frames", type=int, default=128)
    parser.add_argument("--num-edges", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--results-dir", default="results/baselines_real_smoke")
    parser.add_argument("--reuse-teacher-cache", action="store_true")
    parser.add_argument("--quick-smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeat-id", type=int, default=0)
    parser.add_argument("--method-variant", default="default")
    parser.add_argument("--bandwidth-mbps", type=float, default=50.0)
    parser.add_argument("--max-concurrent-train-jobs", type=int, default=1)
    parser.add_argument("--real-sleep-upload", action="store_true")
    parser.add_argument("--f1-threshold", type=float)
    parser.add_argument("--map50-threshold", type=float)
    parser.add_argument("--recovery-sla-sec", type=float, default=120.0)
    parser.add_argument("--latency-sla-ms", type=float)
    parser.add_argument("--capacity-mode", action="store_true")
    return parser.parse_args()


def _parse_class_names(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _runtime_client_defaults(path: str | Path | None) -> dict[str, object]:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    try:
        client = load_runtime_config(config_path).client
    except Exception as exc:
        print(
            f"[run_baselines_real] Could not read client defaults from {config_path}: {exc}",
            file=sys.stderr,
        )
        return {}
    anchor_profile = str(getattr(client, "tinynext_anchor_profile", "") or "").strip()
    return {
        "class_names": [str(item) for item in getattr(client, "class_names", [])],
        "tinynext_input_size": int(getattr(client, "tinynext_input_size", 0) or 0) or None,
        "tinynext_anchor_profile": anchor_profile or None,
    }


def _resolve_class_names(args: argparse.Namespace, defaults: dict[str, object]) -> list[str]:
    explicit = _parse_class_names(args.class_names)
    if explicit:
        return explicit
    return [str(item) for item in defaults.get("class_names", [])]


def main() -> None:
    args = parse_args()
    root_results = Path(args.results_dir)
    resolved_device = resolve_torch_device(args.device)
    if str(resolved_device) != str(args.device):
        print(f"[run_baselines_real] Requested {args.device}, using {resolved_device}.", file=sys.stderr)
        args.device = str(resolved_device)

    method_names = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(method_names) - set(VALID_METHODS))
    if unknown:
        raise ValueError(f"Unknown baseline methods: {unknown}. Valid methods: {VALID_METHODS}")

    runtime_defaults = _runtime_client_defaults(args.runtime_config)
    base_config = ExperimentConfig(
        method=method_names[0],
        num_devices=args.num_edges,
        total_frames=args.total_frames,
        results_dir=str(root_results),
        video_path=args.video,
        student_model=args.student_model,
        student_weights_path=args.student_weights,
        tinynext_input_size=(
            args.tinynext_input_size
            if args.tinynext_input_size is not None
            else runtime_defaults.get("tinynext_input_size")
        ),
        tinynext_anchor_profile=(
            args.tinynext_anchor_profile
            if args.tinynext_anchor_profile is not None
            else runtime_defaults.get("tinynext_anchor_profile")
        ),
        class_names=_resolve_class_names(args, runtime_defaults),
        teacher_label_schema=args.teacher_label_schema,
        teacher_model=args.teacher_model,
        initial_checkpoint=args.initial_checkpoint,
        seed=args.seed,
        repeat_id=args.repeat_id,
        run_id=root_results.name,
        method_variant=args.method_variant if method_names[0] == "plank_road_multi_device" else "default",
        window_seconds=args.window_seconds,
        window_frames=args.window_frames,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        bandwidth_mbps=args.bandwidth_mbps,
        max_concurrent_train_jobs=args.max_concurrent_train_jobs,
        real_sleep_upload=args.real_sleep_upload,
        reuse_teacher_cache=args.reuse_teacher_cache,
        quick_smoke=args.quick_smoke,
        f1_threshold=args.f1_threshold,
        map50_threshold=args.map50_threshold,
        recovery_sla_sec=args.recovery_sla_sec,
        latency_sla_ms=args.latency_sla_ms,
        capacity_mode=args.capacity_mode,
    )
    method_variants = {
        method_name: args.method_variant
        for method_name in method_names
        if method_name == "plank_road_multi_device" and args.method_variant != "default"
    }
    print(f"[run_baselines_real] teacher_label_dir={args.teacher_model}")
    result = run_one_experiment_case(
        config=base_config,
        method_names=method_names,
        root_results=root_results,
        method_variants=method_variants,
        write_outputs=True,
    )
    for summary in result["method_summaries"]:
        print(
            "[run_baselines_real] "
            f"method={summary['method_name']} "
            f"display_name={display_name_for_method(summary['method_name'])} "
            f"method_variant={summary.get('method_variant', 'default')} "
            f"num_edges={summary.get('num_edges')} "
            f"bandwidth_mbps={summary.get('bandwidth_mbps')} "
            f"sla_satisfied={summary.get('sla_satisfied')} "
            f"summary_path={root_results / 'summary.json'}"
        )
    print(f"Wrote real baseline results to {root_results}")


if __name__ == "__main__":
    main()
