import argparse
import csv
import json
import queue
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

if __name__ == "__main__":
    from common.cuda_visibility import configure_default_cuda_visible_devices

    configure_default_cuda_visible_devices()

import cv2
import grpc
from loguru import logger

from baselines.runtime import BaselineEdgeAdapter
from baselines.runtime.upload_client import encode_frame_for_raw_upload
from common.experiment_results import (
    EKYA_METHOD,
    RECAP_METHOD,
    SURGEON_METHOD,
    ExperimentIdentity,
    ExperimentJsonlWriter,
    collect_edge_artifacts,
    edge_run_dir,
    normalize_scenario_slug,
)
from common.logging_sanitizer import log_diagnostic_debug, summarize_path
from common.video_identity import (
    VideoIdentity,
    is_remote_video_source,
    resolve_video_identity,
)
from config import load_runtime_config
from config.baseline import validate_baseline_method
from edge.box_motion import compensate_boxes_between_frames
from edge.edge_worker import EdgeWorker
from edge.experiment_result_uploader import ExperimentResultUploader
from edge.info import TASK_STATE
from edge.replay_frame_archiver import ReplayFrameArchiver
from edge.task import Task
from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc
from model_management.utils import draw_detection
from tools.file_op import clear_folder
from tools.grpc_options import grpc_message_options
from tools.video_processor import VideoProcessor


def _task_state_name(task: Task) -> str:
    if task.state == TASK_STATE.TIMEOUT:
        return "Timeout"
    if task.ref is not None:
        return "Cached"
    return "Finished"


def _write_task_result(
    handle,
    task: Task,
    *,
    model_name: str = "",
    model_version: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> None:
    detection_boxes, detection_class, detection_score = task.get_result()
    latency_ms = None
    if task.end_time is not None:
        latency_ms = max(0.0, (float(task.end_time) - float(task.start_time)) * 1000.0)
    timing_ms = {
        str(name): float(value)
        for name, value in dict(getattr(task, "timing_ms", {}) or {}).items()
    }
    extra = dict(metadata or {})
    timestamp_ms = extra.pop("timestamp_ms", None)
    if timestamp_ms is None:
        timestamp_ms = getattr(task, "capture_timestamp_ms", None)
    if timestamp_ms is None:
        timestamp_ms = int(float(task.start_time) * 1000)
    payload = {
        "frame_index": int(task.frame_index),
        "timestamp_ms": int(timestamp_ms),
        "start_time": float(task.start_time),
        "end_time": float(task.end_time) if task.end_time is not None else None,
        "latency_ms": latency_ms,
        "timing_ms": timing_ms,
        "state": _task_state_name(task),
        "result_source": task.result_source,
        "ref": int(task.ref) if task.ref is not None else None,
        "model_name": str(model_name or ""),
        "model_version": str(model_version or ""),
        "result": {
            "labels": list(detection_class),
            "boxes": [list(box) for box in detection_boxes],
            "scores": [float(score) for score in detection_score],
        },
    }
    payload.update(extra)
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_buffered_task_result(
    handle,
    task: Task,
    *,
    unflushed_count: int,
    flush_every_n_frames: int,
    model_name: str = "",
    model_version: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> int:
    _write_task_result(
        handle,
        task,
        model_name=model_name,
        model_version=model_version,
        metadata=metadata,
    )
    pending = int(unflushed_count) + 1
    if pending >= max(1, int(flush_every_n_frames)):
        handle.flush()
        return 0
    return pending


def _resolve_display_label_config(config, edge: EdgeWorker):
    class_names = getattr(config, "class_names", None) or None
    label_schema = getattr(config, "label_schema", None) or None
    detector = getattr(edge, "small_object_detection", None)
    model = getattr(detector, "model", None)
    if label_schema is None:
        label_schema = getattr(model, "label_schema", None)
    return class_names, label_schema


def _overlay_lines(frame, lines: list[str]) -> None:
    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 26


def _build_display_frame(
    frame,
    *,
    frame_index: int,
    detection_boxes: list,
    detection_class: list,
    detection_score: list,
    mode: str,
    sampled: bool,
    latency_ms: float | None = None,
    ref: int | None = None,
    latest_result_frame: int | None = None,
    show_boxes: bool = True,
    detection_count: int | None = None,
    class_names=None,
    label_schema: str | None = None,
):
    display_boxes = detection_boxes if show_boxes else []
    display_class = detection_class if show_boxes else []
    display_score = detection_score if show_boxes else []
    rendered = draw_detection(
        frame,
        display_boxes,
        display_class,
        display_score,
        class_names=class_names,
        label_schema=label_schema,
    )
    lines = [
        f"Frame: {frame_index}",
        f"Detections: {detection_count if detection_count is not None else len(display_boxes)}",
        f"Mode: {mode}",
        f"Sampled: {'yes' if sampled else 'no'}",
    ]
    if latency_ms is not None:
        lines.append(f"Latency: {latency_ms:.1f} ms")
    if latest_result_frame is not None and latest_result_frame != frame_index:
        lines.append(f"Latest result frame: {latest_result_frame}")
        if detection_count:
            lines.append(f"Overlay: reused ({frame_index - latest_result_frame} frames old)")
    if ref is not None:
        lines.append(f"Reference frame: {ref}")
    lines.append("Press q or ESC to quit")
    _overlay_lines(rendered, lines)
    return rendered


def _valid_host_port(value: str) -> bool:
    server_ip = str(value or "").strip()
    if not server_ip:
        return False

    if server_ip.startswith("["):
        closing_index = server_ip.find("]")
        if closing_index <= 1:
            return False
        if closing_index + 1 >= len(server_ip) or server_ip[closing_index + 1] != ":":
            return False
        port = server_ip[closing_index + 2 :]
        return _valid_port(port)

    host, separator, port = server_ip.rpartition(":")
    return bool(separator and host.strip() and _valid_port(port))


def _valid_port(value: str) -> bool:
    try:
        port = int(str(value).strip())
    except (TypeError, ValueError):
        return False
    return 0 < port <= 65535


def _is_uri_or_camera_source(value: str) -> bool:
    source = str(value or "").strip()
    if not source:
        return False
    if source.isdigit():
        return True
    return "://" in source


def _rtsp_enabled(config) -> bool:
    return bool(getattr(getattr(config.source, "rtsp", None), "flag", False))


def _validate_startup_config(config, *, require_server_ip: bool = True) -> None:
    edge_id = int(getattr(config, "edge_id", 0) or 0)
    if edge_id <= 0:
        raise ValueError("edge_id must be a positive integer")

    server_ip = str(getattr(config, "server_ip", "") or "").strip()
    if require_server_ip and not _valid_host_port(server_ip):
        raise ValueError("server_ip must be a non-empty host:port value")

    cache_path = str(getattr(config.retrain, "cache_path", "") or "").strip()
    if not cache_path:
        raise ValueError("cache_path must be non-empty")

    video_path = str(getattr(config.source, "video_path", "") or "").strip()
    if _rtsp_enabled(config):
        _resolve_video_identity(config)
        return
    if not video_path:
        raise ValueError("experiment_run.video_path is required unless RTSP is enabled")
    if _is_uri_or_camera_source(video_path):
        _resolve_video_identity(config)
        return
    if not Path(video_path).expanduser().exists():
        raise FileNotFoundError(f"video_path does not exist: {video_path}")
    _resolve_video_identity(config)


def _effective_video_source(config) -> str:
    if _rtsp_enabled(config):
        rtsp = config.source.rtsp
        return f"rtsp://{getattr(rtsp, 'ip_address', '')}/channel/{getattr(rtsp, 'channel', '')}"
    return str(getattr(config.source, "video_path", "") or "").strip()


def _resolve_video_identity(config, *, remote_frames_saved: bool | None = None) -> VideoIdentity:
    replay_config = getattr(config.source, "teacher_replay", None)
    if remote_frames_saved is None:
        remote_frames_saved = bool(getattr(replay_config, "save_sampled_frames", False))
    return resolve_video_identity(
        _effective_video_source(config),
        configured_video_slug=getattr(config.source, "video_slug", ""),
        configured_scenario_name=getattr(config.source, "scenario_name", ""),
        remote_frames_saved=bool(remote_frames_saved),
    )


def _split_learning_status(config) -> str:
    return (
        "enabled"
        if bool(getattr(getattr(config, "split_learning", None), "enabled", False))
        else "disabled"
    )


def _baseline_requires_cloud(baseline_method: str) -> bool:
    return validate_baseline_method(baseline_method) != SURGEON_METHOD


def _experiment_method_for_runtime(method: str | None) -> str:
    return EKYA_METHOD if str(method or "") == EKYA_METHOD else str(method or "")


def _create_experiment_identity(
    *,
    experiment_id: str | None,
    scenario: str | None,
    edge_count: int | str | None,
    repeat: int | str | None,
    method: str,
    video_identity: VideoIdentity,
) -> ExperimentIdentity:
    scenario_slug = (
        normalize_scenario_slug(scenario)
        if str(scenario or "").strip()
        else normalize_scenario_slug(video_identity.scenario_name or video_identity.video_slug)
    )
    return ExperimentIdentity.create(
        experiment_id=str(experiment_id or "default_experiment"),
        scenario_slug=scenario_slug,
        edge_count=1 if edge_count is None else edge_count,
        repeat=1 if repeat is None else repeat,
        method=method,
    )


def _experiment_result_upload_enabled(
    *,
    mode: str,
    baseline_method: str | None,
    experiment_results: object,
) -> bool:
    enabled = bool(getattr(experiment_results, "enabled", False))
    upload_enabled = bool(getattr(experiment_results, "upload_enabled", enabled))
    return enabled and upload_enabled


def _upload_experiment_run_artifacts_if_enabled(
    *,
    server_ip: str,
    mode: str,
    baseline_method: str | None,
    experiment_results: object,
    identity: ExperimentIdentity,
    run_id: str,
    method: str,
    edge_id: int,
    artifacts: Mapping[str, Any],
    uploader_cls=ExperimentResultUploader,
) -> bool:
    if not _experiment_result_upload_enabled(
        mode=mode,
        baseline_method=baseline_method,
        experiment_results=experiment_results,
    ):
        return False
    uploader = uploader_cls(str(server_ip), enabled=True)
    return bool(
        uploader.upload_run_artifacts(
            experiment_id=identity.experiment_id,
            scenario_slug=identity.scenario_slug,
            edge_count=identity.edge_count,
            repeat=identity.repeat,
            run_id=run_id,
            method=method,
            edge_id=int(edge_id),
            artifacts=artifacts,
        )
    )


def _baseline_split_runtime_policy(baseline_config) -> str:
    edge_cfg = getattr(baseline_config, "edge", None)
    policy = (
        str(getattr(edge_cfg, "split_runtime_policy", "disabled") or "disabled").strip().lower()
    )
    if policy != "disabled":
        raise ValueError("baseline.edge.split_runtime_policy must be disabled")
    return policy


def _configure_baseline_client_runtime(config, baseline_config) -> str:
    policy = _baseline_split_runtime_policy(baseline_config)
    config.baseline = baseline_config
    if getattr(config, "retrain", None) is not None:
        config.retrain.flag = False
    if getattr(config, "resource_aware_trigger", None) is not None:
        config.resource_aware_trigger.enabled = False
    split_learning = getattr(config, "split_learning", None)
    if split_learning is not None:
        split_learning.enabled = False
    if policy == "disabled":
        logger.info("[BaselineEdge] split_runtime_policy=disabled; fixed-split runtime skipped.")
    return policy


def _prepare_experiment_run_dir(run_dir: Path, *, enabled: bool) -> None:
    if not enabled:
        run_dir.mkdir(parents=True, exist_ok=True)
        return
    if run_dir.exists() or run_dir.is_symlink():
        logger.warning(
            "Experiment result run path already exists; overwriting local results: {}",
            run_dir,
        )
        if run_dir.is_dir() and not run_dir.is_symlink():
            shutil.rmtree(run_dir)
        else:
            run_dir.unlink()
    run_dir.mkdir(parents=True, exist_ok=False)


def _log_startup_config(config) -> None:
    logger.info(
        "edge client effective startup config: edge_id={}, server_ip={}, "
        "model={}, video_source={}, split_learning={}",
        config.edge_id,
        config.server_ip,
        config.lightweight,
        summarize_path(_effective_video_source(config)),
        _split_learning_status(config),
    )
    log_diagnostic_debug(
        config,
        "edge client startup paths",
        lambda: {
            "cache_path": config.retrain.cache_path,
            "video_path": _effective_video_source(config),
            "weights_path": getattr(config, "weights_path", None),
        },
    )


def _run_video_loop(
    config,
    edge: EdgeWorker,
    *,
    headless: bool = False,
    baseline_adapter: BaselineEdgeAdapter | None = None,
    result_path: Path,
    experiment_metrics: ExperimentJsonlWriter | None = None,
    method: str = "",
    run_id: str,
    video_identity: VideoIdentity | None = None,
    replay_archiver: ReplayFrameArchiver | None = None,
) -> int:
    result_path.parent.mkdir(parents=True, exist_ok=True)

    window_name = f"Edge {config.edge_id} Inference"
    window_created = False
    display_class_names, display_label_schema = _resolve_display_label_config(config, edge)
    identity = video_identity or _resolve_video_identity(config)
    if display_class_names:
        logger.info(
            "Using {} configured detection class name(s) for display.",
            len(display_class_names),
        )
    elif str(display_label_schema or "").strip().lower() == "zero_based":
        logger.info(
            "Model uses zero-based labels; display will show class_<id>. "
            "Set client.class_names in config.yaml to show custom names."
        )

    last_visual = {
        "boxes": [],
        "labels": [],
        "scores": [],
        "mode": "Waiting",
        "latency_ms": None,
        "ref": None,
        "frame_index": None,
        "frame": None,
    }
    if baseline_adapter is not None:
        baseline_adapter.before_video_start(edge)

    sampled_frame_count = 0
    with result_path.open("w", encoding="utf-8") as result_file:
        flush_every_n_frames = max(
            1,
            int(getattr(config, "flush_every_n_frames", 30)),
        )
        unflushed_result_count = 0
        with VideoProcessor(config.source) as video:
            video_fps = float(video.fps or 0.0)
            if video_fps <= 0:
                video_fps = 25.0
                logger.warning(
                    "Video FPS unavailable, falling back to {} FPS for display.",
                    video_fps,
                )
            logger.info("the video fps is {}", video_fps)

            if config.interval == 0:
                raise ValueError("config.interval must not be 0")

            logger.info("Take the frame interval is {}", config.interval)
            display_delay_ms = max(1, int(1000 / video_fps))
            index = 0
            split_runtime_prepared = False

            while True:
                frame = next(video)
                if frame is None:
                    logger.info("The video finished")
                    break

                if not split_runtime_prepared:
                    if getattr(edge, "split_learning_enabled", False):
                        logger.info(
                            "Preparing fixed split runtime before starting inference frames."
                        )
                        edge.ensure_fixed_split_runtime(
                            frame,
                            tuple(int(value) for value in frame.shape[:2]),
                        )
                    split_runtime_prepared = True

                if not headless and not window_created:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    window_created = True

                index += 1
                sampled = index % config.interval == 0

                if sampled:
                    sampled_frame_count += 1
                    start_time = time.time()
                    task = Task(config.edge_id, index, frame, start_time, frame.shape)
                    task.capture_timestamp_ms = int(start_time * 1000)
                    edge.submit_task(task)
                    waited = task.wait_until_done(timeout=float(config.wait_thresh) + 5.0)
                    if not waited:
                        task.end_time = time.time()
                        task.state = TASK_STATE.TIMEOUT
                        task.result_source = "timeout"
                        logger.warning("[EdgeVideo] inference timeout.")
                        log_diagnostic_debug(
                            config,
                            "[EdgeVideo] inference timeout diagnostics",
                            lambda: {"frame_id": index},
                        )

                    detection_boxes, detection_class, detection_score = task.get_result()
                    latency_ms = None
                    if task.end_time is not None:
                        latency_ms = max(0.0, (task.end_time - task.start_time) * 1000.0)

                    if task.state == TASK_STATE.TIMEOUT:
                        mode = "Timeout"
                    elif task.ref is not None:
                        mode = "Cached"
                    else:
                        mode = "Inference"

                    last_visual = {
                        "boxes": [list(box) for box in detection_boxes],
                        "labels": list(detection_class),
                        "scores": [float(score) for score in detection_score],
                        "mode": mode,
                        "latency_ms": latency_ms,
                        "ref": task.ref,
                        "frame_index": index,
                        "frame": frame.copy(),
                    }
                    replay_frame_path = None
                    frame_replayable = bool(identity.frame_replayable)
                    if is_remote_video_source(identity.video_source):
                        replay_frame_path = (
                            replay_archiver.enqueue(index, frame)
                            if replay_archiver is not None
                            else None
                        )
                        frame_replayable = replay_frame_path is not None
                    unflushed_result_count = _write_buffered_task_result(
                        result_file,
                        task,
                        unflushed_count=unflushed_result_count,
                        flush_every_n_frames=flush_every_n_frames,
                        model_name=str(getattr(config, "lightweight", "") or ""),
                        model_version=str(getattr(edge, "model_version", "0") or "0"),
                        metadata={
                            "video_source": identity.video_source,
                            "video_slug": identity.video_slug,
                            "scenario_name": identity.scenario_name,
                            "edge_id": int(config.edge_id),
                            "run_id": str(run_id),
                            "method": str(method),
                            "frame_replayable": bool(frame_replayable),
                            "replay_frame_path": replay_frame_path or "",
                            "label_schema": str(display_label_schema or ""),
                            "class_names": list(display_class_names or []),
                        },
                    )
                    if experiment_metrics is not None:
                        experiment_metrics.write(
                            {
                                "event": "frame_inference",
                                "timestamp_ms": int(time.time() * 1000),
                                "frame_id": int(index),
                                "latency_ms": latency_ms,
                                "result_source": str(task.result_source or ""),
                                "model_version": str(getattr(edge, "model_version", "0") or "0"),
                                "num_detections": len(detection_boxes),
                                "timing_ms": dict(getattr(task, "timing_ms", {}) or {}),
                            }
                        )
                    if baseline_adapter is not None:
                        baseline_adapter.on_sampled_inference_result(
                            frame=frame,
                            frame_index=index,
                            task=task,
                            detection_boxes=last_visual["boxes"],
                            detection_class=last_visual["labels"],
                            detection_score=last_visual["scores"],
                            latency_ms=latency_ms,
                        )
                    display_visual = (
                        baseline_adapter.display_visual(last_visual)
                        if baseline_adapter is not None
                        else last_visual
                    )
                    display_frame = _build_display_frame(
                        frame,
                        frame_index=index,
                        detection_boxes=display_visual["boxes"],
                        detection_class=display_visual["labels"],
                        detection_score=display_visual["scores"],
                        mode=display_visual["mode"],
                        sampled=True,
                        latency_ms=display_visual.get("latency_ms"),
                        ref=display_visual.get("ref"),
                        latest_result_frame=display_visual.get("frame_index"),
                        show_boxes=bool(display_visual["boxes"]),
                        detection_count=len(display_visual["boxes"]),
                        class_names=display_class_names,
                        label_schema=display_label_schema,
                    )
                else:
                    display_boxes = last_visual["boxes"]
                    display_labels = last_visual["labels"]
                    display_scores = last_visual["scores"]
                    if display_boxes and last_visual.get("frame") is not None:
                        compensated_boxes, keep_indices = compensate_boxes_between_frames(
                            display_boxes,
                            last_visual["frame"],
                            frame,
                        )
                        kept = [
                            (box, display_labels[item_index], display_scores[item_index])
                            for box, item_index in zip(compensated_boxes, keep_indices)
                            if item_index < len(display_labels) and item_index < len(display_scores)
                        ]
                        display_boxes = [item[0] for item in kept]
                        display_labels = [item[1] for item in kept]
                        display_scores = [item[2] for item in kept]
                    elif display_boxes:
                        display_boxes = []
                        display_labels = []
                        display_scores = []
                    local_visual = {
                        "boxes": display_boxes,
                        "labels": display_labels,
                        "scores": display_scores,
                        "mode": last_visual["mode"],
                        "latency_ms": last_visual["latency_ms"],
                        "ref": last_visual["ref"],
                        "frame_index": last_visual["frame_index"],
                        "frame": last_visual.get("frame"),
                    }
                    if baseline_adapter is not None:
                        baseline_adapter.on_unsampled_frame(
                            frame=frame,
                            frame_index=index,
                            latest_visual=local_visual,
                        )
                    display_visual = (
                        baseline_adapter.display_visual(local_visual)
                        if baseline_adapter is not None
                        else local_visual
                    )
                    display_frame = _build_display_frame(
                        frame,
                        frame_index=index,
                        detection_boxes=display_visual["boxes"],
                        detection_class=display_visual["labels"],
                        detection_score=display_visual["scores"],
                        mode=display_visual["mode"],
                        sampled=False,
                        latency_ms=display_visual.get("latency_ms"),
                        ref=display_visual.get("ref"),
                        latest_result_frame=display_visual.get("frame_index"),
                        show_boxes=bool(display_visual["boxes"]),
                        detection_count=len(display_visual["boxes"]),
                        class_names=display_class_names,
                        label_schema=display_label_schema,
                    )

                if not headless:
                    cv2.imshow(window_name, display_frame)
                    key = cv2.waitKey(display_delay_ms) & 0xFF
                    if key in (27, ord("q")):
                        logger.info("Video display stopped by user.")
                        break
        result_file.flush()

    logger.info("Saved local inference results: records_file={}.", result_path.name)
    compatibility_path = Path("log") / "client" / "latest_inference_results.jsonl"
    compatibility_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(result_path, compatibility_path)
    log_diagnostic_debug(
        config,
        "edge inference result path",
        lambda: {"result_path": str(result_path)},
    )
    return sampled_frame_count


def _preserve_experiment_results_cache(config, preserve: set[str]) -> bool:
    experiment_results = getattr(config, "experiment_results", None)
    if experiment_results is None:
        return True
    cache_root = Path(str(config.retrain.cache_path)).resolve()
    local_root = Path(
        str(getattr(experiment_results, "local_root_dir", "cache/experiment_results"))
    ).resolve()
    try:
        relative = local_root.relative_to(cache_root)
    except ValueError:
        return True
    if not relative.parts:
        logger.warning(
            "Experiment result local_root_dir equals retrain cache_path; "
            "skipping cache cleanup to preserve archived runs."
        )
        return False
    if relative.parts:
        preserve.add(relative.parts[0])
    return True


def _write_edge_summary(
    path: Path,
    *,
    config,
    identity: ExperimentIdentity,
    method: str,
    run_id: str,
    sampled_frame_count: int,
    video_identity: VideoIdentity | None = None,
    replay_snapshot_failures: Mapping[int, str] | None = None,
    edge: EdgeWorker | None = None,
) -> None:
    video_info = video_identity or _resolve_video_identity(config)
    class_names, label_schema = _resolve_display_label_config(
        config,
        edge,
    )
    payload = {
        "experiment_id": identity.experiment_id,
        "scenario_slug": identity.scenario_slug,
        "edge_count": identity.edge_count,
        "repeat": identity.repeat,
        "run_id": run_id,
        "method": method,
        "edge_id": int(config.edge_id),
        "video_source": video_info.video_source,
        "video_slug": video_info.video_slug,
        "scenario_name": video_info.scenario_name,
        "frame_replayable": bool(video_info.frame_replayable),
        "label_schema": str(label_schema or ""),
        "class_names": list(class_names or []),
        "student_model": str(getattr(config, "lightweight", "") or ""),
        "teacher_model": str(getattr(config, "experiment_teacher_model", "") or ""),
        "sampled_frame_count": int(sampled_frame_count),
        "offline_result_archival": method == "SURGEON",
        "archive_upload_excluded_from_communication_cost": True,
        "completed_at_ms": int(time.time() * 1000),
        "replay_snapshot_failures": {
            str(key): value for key, value in sorted(dict(replay_snapshot_failures or {}).items())
        },
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_trigger_manifest_from_metrics(run_dir: Path) -> None:
    metrics_path = run_dir / "edge_metrics.jsonl"
    if not metrics_path.is_file():
        return
    decisions = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("event") != "resource_trigger_decision":
                continue
            decisions.append(
                {
                    key: payload.get(key)
                    for key in (
                        "timestamp_ms",
                        "frame_id",
                        "window_id",
                        "trigger_decision",
                        "trigger_reason",
                        "send_low_conf_features",
                        "bandwidth_mbps",
                        "bundle_cap_bytes",
                    )
                }
            )
    if not decisions:
        return
    (run_dir / "trigger_manifest.json").write_text(
        json.dumps(
            {
                "decision_count": len(decisions),
                "decisions": decisions,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _run_ekya_style_edge_stream(
    *,
    runtime_config,
    config,
    baseline_run_id: str,
    headless: bool,
    output_dir: Path | None = None,
) -> Path:
    from cloud.baselines.Ekya.config import parse_ekya_style_config
    from cloud.baselines.Ekya.unified_logger import DISPLAY_FIELDS

    ekya_config = parse_ekya_style_config(
        runtime_config,
        run_id=baseline_run_id,
        video_path=_effective_video_source(config),
    )
    edge_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else (Path("results") / "edge" / baseline_run_id / "baselines" / EKYA_METHOD)
    )
    edge_output_dir.mkdir(parents=True, exist_ok=True)
    display_events_path = edge_output_dir / "display_events.csv"
    result_path = edge_output_dir / "latest_cloud_inference_results.jsonl"
    _write_csv_header(display_events_path, DISPLAY_FIELDS)
    window_name = f"Edge {config.edge_id} Cloud Inference"
    window_created = False
    request_queue: queue.Queue[message_transmission_pb2.EkyaClientMessage | None] = queue.Queue(
        maxsize=max(1, int(ekya_config.edge_streaming.upload_queue_size))
    )

    def request_iter():
        while True:
            item = request_queue.get()
            try:
                if item is None:
                    return
                yield item
            finally:
                request_queue.task_done()

    channel = grpc.insecure_channel(str(config.server_ip), options=grpc_message_options())
    stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
    responses = stub.EkyaFrameStream(request_iter())
    logger.info(
        "ekya_style edge streaming start: run_id={} edge_id={} video={} output={}",
        baseline_run_id,
        int(config.edge_id),
        summarize_path(_effective_video_source(config)),
        edge_output_dir,
    )
    try:
        with result_path.open("w", encoding="utf-8") as result_file:
            with VideoProcessor(config.source) as video:
                video_fps = float(video.fps or 25.0)
                display_delay_ms = max(1, int(1000 / max(video_fps, 1.0)))
                frame_idx = 0
                while frame_idx < int(ekya_config.num_frames):
                    frame = next(video)
                    if frame is None:
                        break
                    frame_idx += 1
                    timestamp_capture = time.time()
                    encoded = encode_frame_for_raw_upload(frame)
                    timestamp_send = time.time()
                    task_id = (frame_idx - 1) // max(1, int(ekya_config.window_size))
                    upload = message_transmission_pb2.EkyaFrameUpload(
                        method=EKYA_METHOD,
                        run_id=baseline_run_id,
                        edge_id=int(config.edge_id),
                        camera_id=0,
                        task_id=int(task_id),
                        chunk_id=int(task_id),
                        frame_idx=int(frame_idx),
                        video_name=ekya_config.video_name,
                        timestamp_edge_capture=float(timestamp_capture),
                        timestamp_edge_send=float(timestamp_send),
                        image_shape=[int(frame.shape[0]), int(frame.shape[1])],
                        encoded_frame_jpeg=encoded,
                    )
                    request_queue.put(
                        message_transmission_pb2.EkyaClientMessage(frame_upload=upload)
                    )
                    result = _next_ekya_detection_result(responses, frame_idx)
                    timestamp_receive = time.time()
                    boxes, labels, scores, class_names = _ekya_result_lists(result)
                    display_frame = _build_display_frame(
                        frame,
                        frame_index=frame_idx,
                        detection_boxes=boxes,
                        detection_class=labels,
                        detection_score=scores,
                        mode="Cloud",
                        sampled=True,
                        latency_ms=max(0.0, (timestamp_receive - timestamp_capture) * 1000.0),
                        show_boxes=True,
                        detection_count=len(boxes),
                        class_names=class_names or getattr(config, "class_names", None),
                    )
                    if not headless and not window_created:
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        window_created = True
                    if not headless:
                        cv2.imshow(window_name, display_frame)
                        key = cv2.waitKey(display_delay_ms) & 0xFF
                        if key in (27, ord("q")):
                            logger.info("Ekya cloud display stopped by user.")
                            break
                    timestamp_display = time.time()
                    display_event = message_transmission_pb2.EkyaDisplayEvent(
                        method=EKYA_METHOD,
                        run_id=baseline_run_id,
                        edge_id=int(config.edge_id),
                        camera_id=0,
                        task_id=int(result.task_id),
                        chunk_id=int(result.chunk_id),
                        frame_idx=int(frame_idx),
                        timestamp_edge_capture=float(result.timestamp_edge_capture),
                        timestamp_edge_send=float(result.timestamp_edge_send),
                        timestamp_edge_receive=float(timestamp_receive),
                        timestamp_edge_display=float(timestamp_display),
                        displayed=True,
                        drop_reason="",
                    )
                    request_queue.put(
                        message_transmission_pb2.EkyaClientMessage(display_event=display_event)
                    )
                    _append_display_event_row(display_events_path, DISPLAY_FIELDS, display_event)
                    result_file.write(
                        json.dumps(
                            {
                                "method": EKYA_METHOD,
                                "run_id": baseline_run_id,
                                "edge_id": int(config.edge_id),
                                "frame_index": int(frame_idx),
                                "timestamp_ms": int(timestamp_capture * 1000),
                                "model_name": ekya_config.student_model,
                                "model_version": str(result.model_version),
                                "result_source": "cloud_inference",
                                "latency_ms": max(
                                    0.0,
                                    (timestamp_display - timestamp_capture) * 1000.0,
                                ),
                                "result": {
                                    "boxes": boxes,
                                    "labels": labels,
                                    "scores": scores,
                                },
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    result_file.flush()
    finally:
        _close_ekya_frame_stream(
            request_queue=request_queue,
            responses=responses,
            channel=channel,
            run_id=baseline_run_id,
            edge_id=int(config.edge_id),
        )
    logger.info(
        "ekya_style edge streaming complete: display_events={}",
        display_events_path,
    )
    return display_events_path


def _next_ekya_detection_result(responses, frame_idx: int):
    while True:
        message = next(responses)
        payload_type = message.WhichOneof("payload")
        if payload_type == "error":
            raise RuntimeError(message.error.message)
        if payload_type == "detection_result":
            result = message.detection_result
            if int(result.frame_idx) == int(frame_idx):
                return result


def _close_ekya_frame_stream(
    *,
    request_queue: queue.Queue,
    responses,
    channel,
    run_id: str,
    edge_id: int,
) -> None:
    try:
        request_queue.put(
            message_transmission_pb2.EkyaClientMessage(
                close=message_transmission_pb2.EkyaStreamClose(
                    run_id=str(run_id),
                    edge_id=int(edge_id),
                )
            ),
            timeout=5.0,
        )
        request_queue.put(None, timeout=5.0)
        _wait_ekya_stream_close_ack(responses)
    except Exception as exc:
        logger.warning("Ekya stream close handshake failed: {}", exc)
    finally:
        channel.close()


def _wait_ekya_stream_close_ack(responses) -> None:
    for message in responses:
        payload_type = message.WhichOneof("payload")
        if payload_type == "error":
            raise RuntimeError(message.error.message)
        if payload_type != "ack":
            continue
        if not bool(message.ack.success):
            raise RuntimeError(message.ack.message)
        if str(message.ack.message) == "stream closed":
            return
    raise RuntimeError("Ekya stream ended before close ack")


def _ekya_result_lists(result) -> tuple[list[list[float]], list[int], list[float], list[str]]:
    boxes = []
    labels = []
    scores = []
    class_names = []
    for detection in list(result.detections):
        boxes.append(
            [
                float(detection.x1),
                float(detection.y1),
                float(detection.x2),
                float(detection.y2),
            ]
        )
        labels.append(int(detection.label))
        scores.append(float(detection.score))
        class_names.append(str(detection.class_name))
    return boxes, labels, scores, class_names


def _write_csv_header(path: Path, fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()


def _count_csv_records(path: Path) -> int:
    if not Path(path).is_file():
        return 0
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _row in csv.DictReader(handle))


def _append_display_event_row(path: Path, fields: list[str], event) -> None:
    row = {
        "method": EKYA_METHOD,
        "run_id": event.run_id,
        "edge_id": int(event.edge_id),
        "camera_id": int(event.camera_id),
        "task_id": int(event.task_id),
        "chunk_id": int(event.chunk_id),
        "frame_idx": int(event.frame_idx),
        "timestamp_edge_capture": float(event.timestamp_edge_capture),
        "timestamp_edge_send": float(event.timestamp_edge_send),
        "timestamp_edge_receive": float(event.timestamp_edge_receive),
        "timestamp_edge_display": float(event.timestamp_edge_display),
        "edge_upload_to_result_latency_ms": max(
            0.0,
            (float(event.timestamp_edge_receive) - float(event.timestamp_edge_send)) * 1000.0,
        ),
        "edge_render_latency_ms": max(
            0.0,
            (float(event.timestamp_edge_display) - float(event.timestamp_edge_receive)) * 1000.0,
        ),
        "edge_e2e_display_latency_ms": max(
            0.0,
            (float(event.timestamp_edge_display) - float(event.timestamp_edge_capture)) * 1000.0,
        ),
        "displayed": "true" if bool(event.displayed) else "false",
        "drop_reason": event.drop_reason,
    }
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writerow(row)


if __name__ == "__main__":
    from tools.logging_config import configure_logging

    configure_logging()

    parser = argparse.ArgumentParser(description="configuration description")
    parser.add_argument(
        "--yaml_path",
        default="./config/config.yaml",
        help="input the path of *.yaml",
    )
    parser.add_argument(
        "--edge_id",
        type=int,
        default=None,
        help="override client.edge_id for multi-edge deployment",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default=None,
        help="override client.retrain.cache_path (must be unique per edge)",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="override experiment_run.video_path for this edge process",
    )
    parser.add_argument("--server_ip", type=str, default=None, help="override client.server_ip")
    parser.add_argument(
        "--max_count",
        type=int,
        default=None,
        help="override experiment_run.max_count for this edge process",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without OpenCV display windows",
    )
    parser.add_argument("--mode", choices=("main", "baseline"), default="main")
    parser.add_argument(
        "--baseline_method",
        default=None,
        help="baseline method for baseline mode: SURGEON, CATR, or Ekya",
    )
    parser.add_argument("--experiment_id", default=None, help="experiment id")
    parser.add_argument("--scenario", default=None, help="experiment scenario slug/name")
    parser.add_argument("--edge_count", type=int, default=None, help="number of edge devices")
    parser.add_argument("--repeat", default=None, help="repeat index, e.g. 1 or r01")
    parser.add_argument(
        "--experiment_results_root",
        default=None,
        help="override edge local experiment result staging root",
    )
    args = parser.parse_args()
    requested_baseline_method = None
    if args.baseline_method is not None:
        try:
            requested_baseline_method = validate_baseline_method(args.baseline_method)
        except ValueError as exc:
            parser.error(str(exc))
    if requested_baseline_method == EKYA_METHOD and args.mode != "baseline":
        args.mode = "baseline"

    runtime_config = load_runtime_config(args.yaml_path)
    config = runtime_config.client
    experiment_run = runtime_config.experiment_run
    config.experiment_teacher_model = str(getattr(runtime_config.server, "golden", "") or "")
    experiment_results = runtime_config.experiment_results
    if args.experiment_results_root is not None:
        experiment_results.local_root_dir = args.experiment_results_root

    # Apply per-edge CLI overrides for multi-edge deployment
    if args.edge_id is not None:
        config.edge_id = args.edge_id
    if args.cache_path is not None:
        config.retrain.cache_path = args.cache_path
    elif args.edge_id is not None:
        # Auto-isolate cache per edge_id when only --edge_id is specified
        config.retrain.cache_path = f"./cache/edge_{args.edge_id}"
    if args.video_path is not None:
        config.source.video_path = args.video_path
    if args.scenario is not None:
        # Keep per-frame replay metadata aligned with the experiment identity.
        # Previously --scenario only changed the run directory, while the
        # configured scenario leaked into every prediction row.
        config.source.scenario_name = normalize_scenario_slug(args.scenario)
    if args.max_count is not None:
        config.source.max_count = args.max_count
    if args.server_ip is not None:
        config.server_ip = args.server_ip

    baseline_method = None
    if args.mode == "baseline":
        baseline_method = requested_baseline_method or runtime_config.baseline.method
        try:
            baseline_method = validate_baseline_method(baseline_method)
        except ValueError as exc:
            parser.error(str(exc))

    require_server_ip = (
        args.mode == "main"
        or _baseline_requires_cloud(baseline_method)
        or _experiment_result_upload_enabled(
            mode=args.mode,
            baseline_method=baseline_method,
            experiment_results=experiment_results,
        )
    )
    try:
        _validate_startup_config(config, require_server_ip=require_server_ip)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    method = (
        _experiment_method_for_runtime(baseline_method)
        if args.mode == "baseline"
        else RECAP_METHOD
    )
    video_identity = _resolve_video_identity(config)
    experiment_identity = _create_experiment_identity(
        experiment_id=(
            args.experiment_id if args.experiment_id is not None else experiment_run.experiment_id
        ),
        scenario=args.scenario if args.scenario is not None else experiment_run.scenario,
        edge_count=(args.edge_count if args.edge_count is not None else experiment_run.edge_count),
        repeat=args.repeat if args.repeat is not None else experiment_run.repeat,
        method=method,
        video_identity=video_identity,
    )
    run_id = experiment_identity.run_id
    config.experiment_identity = experiment_identity

    baseline_adapter = None
    if args.mode == "baseline":
        runtime_config.baseline.enabled = True
        runtime_config.baseline.method = baseline_method
        logger.add(
            f"log/client/baseline_{baseline_method}_edge_{config.edge_id}_{{time}}.log",
            level="INFO",
            rotation="500 MB",
        )
        if baseline_method == EKYA_METHOD:
            config.baseline = runtime_config.baseline
            if getattr(config, "retrain", None) is not None:
                config.retrain.flag = False
            if getattr(config, "resource_aware_trigger", None) is not None:
                config.resource_aware_trigger.enabled = False
            split_learning = getattr(config, "split_learning", None)
            if split_learning is not None:
                split_learning.enabled = False
            logger.info("[EkyaStyleEdge] split/runtime/triggers disabled for cloud streaming.")
        else:
            _configure_baseline_client_runtime(config, runtime_config.baseline)
        logger.info(
            "baseline edge effective startup config: run_id={}, baseline_method={}, "
            "edge_id={}, server_ip={}, video_source={}, split_learning={}",
            run_id,
            baseline_method,
            config.edge_id,
            config.server_ip,
            summarize_path(_effective_video_source(config)),
            _split_learning_status(config),
        )
        log_diagnostic_debug(
            config,
            "baseline edge startup paths",
            lambda: {
                "cache_path": config.retrain.cache_path,
                "video_path": _effective_video_source(config),
            },
        )
    else:
        logger.add(
            f"log/client/edge_{config.edge_id}_{{time}}.log",
            level="INFO",
            rotation="500 MB",
        )

    preserve_cache_entries = {"pytest_tmp"}
    if bool(getattr(getattr(config, "split_learning", None), "enabled", False)):
        preserve_cache_entries.add("fixed_split_plan.json")
    clear_retrain_cache = _preserve_experiment_results_cache(
        config,
        preserve_cache_entries,
    )
    _log_startup_config(config)
    if clear_retrain_cache:
        clear_folder(config.retrain.cache_path, preserve=preserve_cache_entries)

    if args.mode == "baseline" and baseline_method == EKYA_METHOD:
        run_dir = None
        experiment_log_sink = None
        try:
            if bool(experiment_results.enabled):
                run_dir = edge_run_dir(
                    str(experiment_results.local_root_dir),
                    experiment_identity.experiment_id,
                    experiment_identity.scenario_slug,
                    experiment_identity.edge_count,
                    experiment_identity.repeat,
                    method,
                    int(config.edge_id),
                    run_id,
                )
                _prepare_experiment_run_dir(
                    run_dir,
                    enabled=True,
                )
            display_events_path = _run_ekya_style_edge_stream(
                runtime_config=runtime_config,
                config=config,
                baseline_run_id=run_id,
                headless=args.headless,
                output_dir=run_dir,
            )
            if run_dir is not None and bool(experiment_results.enabled):
                sampled_frame_count = _count_csv_records(display_events_path)
                _write_edge_summary(
                    run_dir / "edge_summary.json",
                    config=config,
                    identity=experiment_identity,
                    method=method,
                    run_id=run_id,
                    sampled_frame_count=sampled_frame_count,
                    video_identity=video_identity,
                )
                artifacts = collect_edge_artifacts(
                    method=method,
                    run_id=run_id,
                    edge_id=int(config.edge_id),
                    experiment_id=experiment_identity.experiment_id,
                    scenario_slug=experiment_identity.scenario_slug,
                    edge_count=experiment_identity.edge_count,
                    repeat=experiment_identity.repeat,
                    config=experiment_results,
                    inference_result_path=run_dir / "latest_cloud_inference_results.jsonl",
                    baseline_metrics_path=None,
                    cache_path=Path(config.retrain.cache_path),
                )
                _upload_experiment_run_artifacts_if_enabled(
                    server_ip=str(config.server_ip),
                    mode=args.mode,
                    baseline_method=baseline_method,
                    experiment_results=experiment_results,
                    identity=experiment_identity,
                    run_id=run_id,
                    method=method,
                    edge_id=int(config.edge_id),
                    artifacts=artifacts,
                )
        finally:
            if experiment_log_sink is not None:
                logger.remove(experiment_log_sink)
            if not args.headless:
                cv2.destroyAllWindows()
        raise SystemExit(0)

    run_dir = edge_run_dir(
        str(experiment_results.local_root_dir),
        experiment_identity.experiment_id,
        experiment_identity.scenario_slug,
        experiment_identity.edge_count,
        experiment_identity.repeat,
        method,
        int(config.edge_id),
        run_id,
    )
    _prepare_experiment_run_dir(
        run_dir,
        enabled=bool(experiment_results.enabled),
    )
    result_path = run_dir / "latest_inference_results.jsonl"
    experiment_metrics = (
        ExperimentJsonlWriter(run_dir / "edge_metrics.jsonl")
        if method == RECAP_METHOD and bool(experiment_results.enabled)
        else None
    )
    config.experiment_metrics_writer = experiment_metrics
    experiment_log_sink = None

    edge = EdgeWorker(config)
    if args.mode == "baseline":
        baseline_adapter = BaselineEdgeAdapter(
            config=config,
            baseline_method=baseline_method,
            run_id=run_id,
            edge_id=int(config.edge_id),
            server_ip=str(config.server_ip),
            cache_path=str(config.retrain.cache_path),
            video_path=_effective_video_source(config),
        )

    replay_config = getattr(config.source, "teacher_replay", None)
    replay_archiver = ReplayFrameArchiver(
        run_dir,
        enabled=bool(
            is_remote_video_source(video_identity.video_source)
            and getattr(replay_config, "save_sampled_frames", False)
        ),
        jpeg_quality=int(getattr(replay_config, "jpeg_quality", 90)),
        queue_size=int(getattr(replay_config, "queue_size", 64)),
        archive_chunk_max_bytes=int(getattr(replay_config, "archive_chunk_max_bytes", 67108864)),
    )
    sampled_frame_count = 0
    try:
        sampled_frame_count = _run_video_loop(
            config,
            edge,
            headless=args.headless,
            baseline_adapter=baseline_adapter,
            result_path=result_path,
            experiment_metrics=experiment_metrics,
            method=method,
            run_id=run_id,
            video_identity=video_identity,
            replay_archiver=replay_archiver,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        try:
            replay_archiver.close()
        except Exception as exc:
            logger.warning("Teacher replay snapshot archival failed: {}", exc)
        if baseline_adapter is not None:
            baseline_adapter.close()
        edge.close()
        try:
            _write_trigger_manifest_from_metrics(run_dir)
            _write_edge_summary(
                run_dir / "edge_summary.json",
                config=config,
                identity=experiment_identity,
                method=method,
                run_id=run_id,
                sampled_frame_count=sampled_frame_count,
                video_identity=video_identity,
                replay_snapshot_failures=replay_archiver.failures,
                edge=edge,
            )
            baseline_metrics_path = (
                Path(baseline_adapter.metrics_path)
                if baseline_adapter is not None
                else (run_dir / "edge_metrics.jsonl")
            )
            if bool(experiment_results.enabled):
                artifacts = collect_edge_artifacts(
                    method=method,
                    run_id=run_id,
                    edge_id=int(config.edge_id),
                    experiment_id=experiment_identity.experiment_id,
                    scenario_slug=experiment_identity.scenario_slug,
                    edge_count=experiment_identity.edge_count,
                    repeat=experiment_identity.repeat,
                    config=experiment_results,
                    inference_result_path=result_path,
                    baseline_metrics_path=baseline_metrics_path,
                    cache_path=Path(config.retrain.cache_path),
                )
                _upload_experiment_run_artifacts_if_enabled(
                    server_ip=str(config.server_ip),
                    mode=args.mode,
                    baseline_method=baseline_method,
                    experiment_results=experiment_results,
                    identity=experiment_identity,
                    run_id=run_id,
                    method=method,
                    edge_id=int(config.edge_id),
                    artifacts=artifacts,
                )
        except Exception as exc:
            logger.warning("Experiment result archival failed during shutdown: {}", exc)
        finally:
            if experiment_log_sink is not None:
                logger.remove(experiment_log_sink)
            if not args.headless:
                cv2.destroyAllWindows()
