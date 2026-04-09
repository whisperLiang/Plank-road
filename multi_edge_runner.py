#!/usr/bin/env python3
"""Multi-device experiment runner for the four-method baseline framework.

This is the main entry point for running scalability, drift-burst,
and heterogeneous-resource experiments across all four methods:

- plank_road_multi_device
- ekya_style_centralized_scheduling
- accuracy_trigger_cloud_retraining
- pure_edge_local_updating

Usage examples
--------------
Run a single method with 4 devices::

    python multi_edge_runner.py --config config/experiment.yaml

Run all four methods for device-count scaling::

    python multi_edge_runner.py --config config/experiment.yaml \
        --experiment scaling --num_devices 1 2 4 8

Run concurrent drift burst experiment::

    python multi_edge_runner.py --config config/experiment.yaml \
        --experiment drift_burst --num_devices 4

Run heterogeneous resource experiment::

    python multi_edge_runner.py --config config/experiment.yaml \
        --experiment heterogeneous --num_devices 4

Run Plank-road ablation (raw-only vs raw+feature)::

    python multi_edge_runner.py --config config/experiment.yaml \
        --experiment ablation --num_devices 4

Results are written to ``results/`` (configurable via YAML).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

from baselines.base_method import BaseMethod, InferenceResult
from baselines.method_factory import create_method
from config.experiment import ExperimentConfig, VALID_METHODS, load_experiment_config
from multi_edge.scenario_generator import DeviceProfile, ScenarioGenerator


# -- Core simulation loop -----------------------------------------------------


def run_single_experiment(
    method: BaseMethod,
    profiles: list[DeviceProfile],
    total_frames: int = 300,
    seed: int = 42,
) -> None:
    """Drive one experiment run through the unified BaseMethod interface.

    For each frame, iterates over all devices, feeds synthetic
    inference results, checks triggers, and executes updates.
    """
    gen = ScenarioGenerator(
        num_devices=len(profiles),
        total_frames=total_frames,
        seed=seed,
    )
    set_local_train_budget = getattr(method, 'set_local_train_budget', None)
    set_upload_bandwidth = getattr(method, 'set_upload_bandwidth', None)

    # Pre-generate streams per device.
    streams: dict[int, list[InferenceResult]] = {}
    for profile in profiles:
        streams[profile.device_id] = gen.generate_stream(profile)

        # Set device metadata on the metrics collector.
        dev_metrics = method.metrics.get_device(profile.device_id)
        dev_metrics.drift_profile = profile.drift_profile
        dev_metrics.bandwidth_profile = profile.bandwidth_profile
        dev_metrics.local_train_budget_profile = profile.local_train_budget_profile

        if callable(set_local_train_budget):
            budget = profile.local_train_params.get('sec_per_epoch', 1.0)
            set_local_train_budget(profile.device_id, budget)
        if callable(set_upload_bandwidth):
            effective_bw = profile.bandwidth_params.get(
                'effective_bw_bytes_per_sec',
                5 * 1024 * 1024,
            )
            set_upload_bandwidth(profile.device_id, effective_bw)

    # Simulate frame-by-frame, round-robin across devices.
    for frame_idx in range(total_frames):
        for profile in profiles:
            device_id = profile.device_id
            result = streams[device_id][frame_idx]

            # 1. Process inference result.
            method.on_inference_result(result)

            # 2. Check trigger.
            if method.should_trigger(device_id):
                # 3. Build update plan.
                plan = method.build_update_plan(device_id)
                # 4. Execute update.
                method.execute_update(plan)

    logger.info(
        'Experiment completed: method={}, devices={}, frames={}',
        method.name(),
        len(profiles),
        total_frames,
    )


# -- Experiment runners -------------------------------------------------------


def run_scaling_experiment(
    base_config: ExperimentConfig,
    num_devices_list: list[int] | None = None,
    methods: list[str] | None = None,
    seed: int = 42,
) -> dict:
    """Experiment 1: Device-count scaling.

    Compares all methods across varying num_devices.
    """
    methods = methods or VALID_METHODS
    if num_devices_list is None:
        num_devices_list = list(base_config.scenario.num_devices_candidates)
    all_results = {}

    for method_name in methods:
        for num_devices in num_devices_list:
            logger.info('Scaling: method={}, num_devices={}', method_name, num_devices)
            config = _clone_config(base_config, method=method_name, num_devices=num_devices)
            method = create_method(config)

            gen = ScenarioGenerator(
                num_devices=num_devices,
                total_frames=config.total_frames,
                seed=seed,
            )
            profiles = gen.generate_uniform_profiles()
            run_single_experiment(
                method,
                profiles,
                total_frames=config.total_frames,
                seed=seed,
            )

            results_dir = str(Path(config.results_dir) / f'scaling_{method_name}_n{num_devices}')
            summary_path, csv_path = method.metrics.finalize_and_export(results_dir)
            key = f'{method_name}_n{num_devices}'
            all_results[key] = {
                'summary': str(summary_path),
                'csv': str(csv_path),
                'overall': method.metrics.compute_overall().to_dict(),
            }

    return all_results


def run_drift_burst_experiment(
    base_config: ExperimentConfig,
    num_devices: int | None = None,
    methods: list[str] | None = None,
    seed: int = 42,
) -> dict:
    """Experiment 2: Concurrent drift burst.

    Multiple devices enter high drift in nearby time windows.
    """
    methods = methods or VALID_METHODS
    if num_devices is None:
        num_devices = base_config.num_devices
    all_results = {}

    for method_name in methods:
        logger.info('Drift burst: method={}, num_devices={}', method_name, num_devices)
        config = _clone_config(base_config, method=method_name, num_devices=num_devices)
        method = create_method(config)

        gen = ScenarioGenerator(
            num_devices=num_devices,
            total_frames=config.total_frames,
            seed=seed,
        )
        profiles = gen.generate_concurrent_drift_burst(
            burst_start_frame=config.total_frames // 3,
            burst_duration=30,
            burst_drift='high',
        )
        run_single_experiment(
            method,
            profiles,
            total_frames=config.total_frames,
            seed=seed,
        )

        results_dir = str(Path(config.results_dir) / f'drift_burst_{method_name}_n{num_devices}')
        summary_path, csv_path = method.metrics.finalize_and_export(results_dir)
        all_results[method_name] = {
            'summary': str(summary_path),
            'csv': str(csv_path),
            'overall': method.metrics.compute_overall().to_dict(),
        }

    return all_results


def run_heterogeneous_experiment(
    base_config: ExperimentConfig,
    num_devices: int | None = None,
    methods: list[str] | None = None,
    seed: int = 42,
) -> dict:
    """Experiment 3: Heterogeneous resource conditions.

    Different devices have different bandwidth / local_train_budget /
    drift profiles.
    """
    methods = methods or VALID_METHODS
    if num_devices is None:
        num_devices = base_config.num_devices
    all_results = {}

    for method_name in methods:
        logger.info('Heterogeneous: method={}, num_devices={}', method_name, num_devices)
        config = _clone_config(base_config, method=method_name, num_devices=num_devices)
        method = create_method(config)

        gen = ScenarioGenerator(
            num_devices=num_devices,
            total_frames=config.total_frames,
            seed=seed,
        )
        profiles = gen.generate_heterogeneous_profiles(
            drift_options=list(config.scenario.drift_profiles),
            bandwidth_options=list(config.scenario.bandwidth_profiles),
            budget_options=list(config.scenario.local_train_budget_profiles),
        )
        run_single_experiment(
            method,
            profiles,
            total_frames=config.total_frames,
            seed=seed,
        )

        results_dir = str(
            Path(config.results_dir) / f'heterogeneous_{method_name}_n{num_devices}'
        )
        summary_path, csv_path = method.metrics.finalize_and_export(results_dir)
        all_results[method_name] = {
            'summary': str(summary_path),
            'csv': str(csv_path),
            'overall': method.metrics.compute_overall().to_dict(),
        }

    return all_results


def run_ablation_experiment(
    base_config: ExperimentConfig,
    num_devices: int | None = None,
    seed: int = 42,
) -> dict:
    """Experiment 4: Internal Plank-road ablation.

    Compares raw-only vs raw+feature upload modes.
    """
    if num_devices is None:
        num_devices = base_config.num_devices
    all_results = {}

    for upload_mode in ['raw_only', 'raw+feature']:
        logger.info('Ablation: upload_mode={}, num_devices={}', upload_mode, num_devices)
        config = _clone_config(
            base_config,
            method='plank_road_multi_device',
            num_devices=num_devices,
        )
        # Override upload mode.
        config.plank_road_multi_device.upload_mode_default = upload_mode
        config.plank_road_multi_device.allow_resource_aware_feature_upload = (
            upload_mode == 'raw+feature'
        )

        method = create_method(config)
        gen = ScenarioGenerator(
            num_devices=num_devices,
            total_frames=config.total_frames,
            seed=seed,
        )
        profiles = gen.generate_uniform_profiles(drift='medium')
        run_single_experiment(
            method,
            profiles,
            total_frames=config.total_frames,
            seed=seed,
        )

        label = upload_mode.replace('+', '_plus_')
        results_dir = str(Path(config.results_dir) / f'ablation_{label}_n{num_devices}')
        summary_path, csv_path = method.metrics.finalize_and_export(results_dir)
        all_results[upload_mode] = {
            'summary': str(summary_path),
            'csv': str(csv_path),
            'overall': method.metrics.compute_overall().to_dict(),
        }

    return all_results


# -- Helpers ------------------------------------------------------------------


def _clone_config(
    base: ExperimentConfig,
    *,
    method: str | None = None,
    num_devices: int | None = None,
) -> ExperimentConfig:
    """Create an isolated copy with optional overrides."""
    import copy

    cfg = copy.deepcopy(base)
    if method is not None:
        cfg.method = method
    if num_devices is not None:
        cfg.num_devices = num_devices
    return cfg


# -- CLI ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Multi-device experiment runner for Plank-road baselines.',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/experiment.yaml',
        help='Path to experiment YAML config.',
    )
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['single', 'scaling', 'drift_burst', 'heterogeneous', 'ablation', 'all'],
        default='single',
        help='Which experiment to run.',
    )
    parser.add_argument(
        '--num_devices',
        type=int,
        nargs='*',
        default=None,
        help='Device counts (for scaling: 1 2 4 8; for others: single value).',
    )
    parser.add_argument(
        '--methods',
        type=str,
        nargs='*',
        default=None,
        help='Methods to evaluate (default: all four).',
    )
    parser.add_argument(
        '--total_frames',
        type=int,
        default=None,
        help='Override total frames per device.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility.',
    )

    args = parser.parse_args()
    config = load_experiment_config(args.config)

    if args.total_frames is not None:
        config.total_frames = args.total_frames

    methods = args.methods

    if args.experiment == 'single':
        # Run a single method as specified in config.
        num_devices = (args.num_devices or [config.num_devices])[0]
        config = _clone_config(config, num_devices=num_devices)
        method = create_method(config)
        gen = ScenarioGenerator(
            num_devices=num_devices,
            total_frames=config.total_frames,
            seed=args.seed,
        )
        profiles = gen.generate_uniform_profiles()
        run_single_experiment(method, profiles, total_frames=config.total_frames, seed=args.seed)
        summary, csv = method.metrics.finalize_and_export(config.results_dir)
        logger.info('Results: {} , {}', summary, csv)

    elif args.experiment == 'scaling':
        results = run_scaling_experiment(
            config,
            args.num_devices,
            methods=methods,
            seed=args.seed,
        )
        _write_combined_results(config.results_dir, 'scaling', results)

    elif args.experiment == 'drift_burst':
        n = (args.num_devices or [config.num_devices])[0]
        results = run_drift_burst_experiment(
            config,
            num_devices=n,
            methods=methods,
            seed=args.seed,
        )
        _write_combined_results(config.results_dir, 'drift_burst', results)

    elif args.experiment == 'heterogeneous':
        n = (args.num_devices or [config.num_devices])[0]
        results = run_heterogeneous_experiment(
            config,
            num_devices=n,
            methods=methods,
            seed=args.seed,
        )
        _write_combined_results(config.results_dir, 'heterogeneous', results)

    elif args.experiment == 'ablation':
        n = (args.num_devices or [config.num_devices])[0]
        results = run_ablation_experiment(config, num_devices=n, seed=args.seed)
        _write_combined_results(config.results_dir, 'ablation', results)

    elif args.experiment == 'all':
        devices_list = args.num_devices or list(config.scenario.num_devices_candidates)
        n_default = devices_list[-1] if devices_list else config.num_devices

        r1 = run_scaling_experiment(config, devices_list, methods=methods, seed=args.seed)
        _write_combined_results(config.results_dir, 'scaling', r1)

        r2 = run_drift_burst_experiment(
            config,
            num_devices=n_default,
            methods=methods,
            seed=args.seed,
        )
        _write_combined_results(config.results_dir, 'drift_burst', r2)

        r3 = run_heterogeneous_experiment(
            config,
            num_devices=n_default,
            methods=methods,
            seed=args.seed,
        )
        _write_combined_results(config.results_dir, 'heterogeneous', r3)

        r4 = run_ablation_experiment(config, num_devices=n_default, seed=args.seed)
        _write_combined_results(config.results_dir, 'ablation', r4)


def _write_combined_results(results_dir: str, experiment_name: str, results: dict) -> None:
    """Write a combined JSON with all sub-experiment overall metrics."""
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{experiment_name}_combined.json'
    with path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info('Combined results: {}', path)


if __name__ == '__main__':
    main()
