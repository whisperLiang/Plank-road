import argparse
import json
import time
from pathlib import Path

if __name__ == "__main__":
    from common.cuda_visibility import configure_default_cuda_visible_devices

    configure_default_cuda_visible_devices()

import cv2
from loguru import logger

from baselines.runtime import BaselineEdgeAdapter
from common.logging_sanitizer import log_diagnostic_debug, summarize_path
from config import load_runtime_config
from config.baseline import validate_baseline_method
from edge.box_motion import compensate_boxes_between_frames
from edge.edge_worker import EdgeWorker
from edge.info import TASK_STATE
from edge.task import Task
from model_management.utils import draw_detection
from tools.file_op import clear_folder
from tools.video_processor import VideoProcessor


def _task_state_name(task: Task) -> str:
    if task.state == TASK_STATE.TIMEOUT:
        return "Timeout"
    if task.ref is not None:
        return "Cached"
    return "Finished"


def _write_task_result(handle, task: Task) -> None:
    detection_boxes, detection_class, detection_score = task.get_result()
    latency_ms = None
    if task.end_time is not None:
        latency_ms = max(0.0, (float(task.end_time) - float(task.start_time)) * 1000.0)
    timing_ms = {
        str(name): float(value)
        for name, value in dict(getattr(task, "timing_ms", {}) or {}).items()
    }
    payload = {
        "frame_index": int(task.frame_index),
        "start_time": float(task.start_time),
        "end_time": float(task.end_time) if task.end_time is not None else None,
        "latency_ms": latency_ms,
        "timing_ms": timing_ms,
        "state": _task_state_name(task),
        "result_source": task.result_source,
        "ref": int(task.ref) if task.ref is not None else None,
        "result": {
            "labels": list(detection_class),
            "boxes": [list(box) for box in detection_boxes],
            "scores": [float(score) for score in detection_score],
        },
    }
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


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
        return
    if not video_path:
        raise ValueError("client.source.video_path is required unless RTSP is enabled")
    if _is_uri_or_camera_source(video_path):
        return
    if not Path(video_path).expanduser().exists():
        raise FileNotFoundError(f"video_path does not exist: {video_path}")


def _effective_video_source(config) -> str:
    if _rtsp_enabled(config):
        rtsp = config.source.rtsp
        return f"rtsp://{getattr(rtsp, 'ip_address', '')}/channel/{getattr(rtsp, 'channel', '')}"
    return str(getattr(config.source, "video_path", "") or "").strip()


def _split_learning_status(config) -> str:
    return (
        "enabled"
        if bool(getattr(getattr(config, "split_learning", None), "enabled", False))
        else "disabled"
    )


def _baseline_requires_cloud(baseline_method: str) -> bool:
    return validate_baseline_method(baseline_method) != "pure_edge_local_updating"


def _baseline_split_runtime_policy(baseline_config) -> str:
    edge_cfg = getattr(baseline_config, "edge", None)
    policy = (
        str(getattr(edge_cfg, "split_runtime_policy", "disabled") or "disabled")
        .strip()
        .lower()
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
    if getattr(config, "sample_pool", None) is not None:
        config.sample_pool.enabled = False
    split_learning = getattr(config, "split_learning", None)
    if split_learning is not None:
        split_learning.enabled = False
    if policy == "disabled":
        logger.info("[BaselineEdge] split_runtime_policy=disabled; fixed-split runtime skipped.")
    return policy


def _resolve_baseline_run_id(baseline_method: str, run_id: str | None) -> str | None:
    value = str(run_id or "").strip()
    if _baseline_requires_cloud(baseline_method) and not value:
        raise ValueError(
            "--run_id is required for cloud-backed baseline mode so every edge "
            "joins the same cloud run"
        )
    return value or None


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
) -> None:
    result_path = Path("log") / "client" / "latest_inference_results.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    window_name = f"Edge {config.edge_id} Inference"
    window_created = False
    display_class_names, display_label_schema = _resolve_display_label_config(config, edge)
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

    with result_path.open("w", encoding="utf-8") as result_file:
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
                    start_time = time.time()
                    task = Task(config.edge_id, index, frame, start_time, frame.shape)
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
                    _write_task_result(result_file, task)
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

    logger.info("Saved local inference results: records_file={}.", result_path.name)
    log_diagnostic_debug(
        config,
        "edge inference result path",
        lambda: {"result_path": str(result_path)},
    )


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
        help="override client.source.video_path",
    )
    parser.add_argument("--server_ip", type=str, default=None, help="override client.server_ip")
    parser.add_argument(
        "--max_count",
        type=int,
        default=None,
        help="override client.source.max_count",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without OpenCV display windows",
    )
    parser.add_argument("--mode", choices=("main", "baseline"), default="main")
    parser.add_argument("--baseline_method", default=None, help="baseline method for baseline mode")
    parser.add_argument("--run_id", default=None, help="baseline run id")
    args = parser.parse_args()

    runtime_config = load_runtime_config(args.yaml_path)
    config = runtime_config.client

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
    if args.max_count is not None:
        config.source.max_count = args.max_count
    if args.server_ip is not None:
        config.server_ip = args.server_ip

    baseline_method = None
    if args.mode == "baseline":
        baseline_method = args.baseline_method or runtime_config.baseline.method
        try:
            baseline_method = validate_baseline_method(baseline_method)
        except ValueError as exc:
            parser.error(str(exc))

    baseline_run_id = None
    if args.mode == "baseline":
        try:
            baseline_run_id = _resolve_baseline_run_id(
                baseline_method,
                args.run_id or runtime_config.baseline.run_id,
            )
        except ValueError as exc:
            parser.error(str(exc))

    require_server_ip = args.mode == "main" or _baseline_requires_cloud(baseline_method)
    try:
        _validate_startup_config(config, require_server_ip=require_server_ip)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    baseline_adapter = None
    if args.mode == "baseline":
        runtime_config.baseline.enabled = True
        runtime_config.baseline.method = baseline_method
        runtime_config.baseline.run_id = baseline_run_id
        logger.add(
            f"log/client/baseline_{baseline_method}_edge_{config.edge_id}_{{time}}.log",
            level="INFO",
            rotation="500 MB",
        )
        _configure_baseline_client_runtime(config, runtime_config.baseline)
        logger.info(
            "baseline edge effective startup config: run_id={}, baseline_method={}, "
            "edge_id={}, server_ip={}, video_source={}, split_learning={}",
            runtime_config.baseline.run_id or "<auto-local>",
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
    _log_startup_config(config)
    clear_folder(config.retrain.cache_path, preserve=preserve_cache_entries)
    edge = EdgeWorker(config)
    if args.mode == "baseline":
        baseline_adapter = BaselineEdgeAdapter(
            config=config,
            baseline_method=baseline_method,
            run_id=runtime_config.baseline.run_id,
            edge_id=int(config.edge_id),
            server_ip=str(config.server_ip),
            cache_path=str(config.retrain.cache_path),
            video_path=_effective_video_source(config),
        )

    try:
        _run_video_loop(
            config,
            edge,
            headless=args.headless,
            baseline_adapter=baseline_adapter,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        if baseline_adapter is not None:
            baseline_adapter.close()
        edge.close()
        if not args.headless:
            cv2.destroyAllWindows()
