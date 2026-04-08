import argparse
import json
import time
from pathlib import Path

import cv2
from loguru import logger

from config import load_runtime_config
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
    payload = {
        "frame_index": int(task.frame_index),
        "start_time": float(task.start_time),
        "end_time": float(task.end_time) if task.end_time is not None else None,
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
):
    display_boxes = detection_boxes if show_boxes else []
    display_class = detection_class if show_boxes else []
    display_score = detection_score if show_boxes else []
    rendered = draw_detection(frame, display_boxes, display_class, display_score)
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


def _run_video_loop(config, edge: EdgeWorker) -> None:
    result_path = Path("log") / "client" / "latest_inference_results.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    window_name = f"Edge {config.edge_id} Inference"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_visual = {
        "boxes": [],
        "labels": [],
        "scores": [],
        "mode": "Waiting",
        "latency_ms": None,
        "ref": None,
        "frame_index": None,
    }

    with result_path.open("w", encoding="utf-8") as result_file:
        with VideoProcessor(config.source) as video:
            video_fps = float(video.fps or 0.0)
            if video_fps <= 0:
                video_fps = 25.0
                logger.warning("Video FPS unavailable, falling back to {} FPS for display.", video_fps)
            logger.info("the video fps is {}", video_fps)

            if config.interval == 0:
                raise ValueError("config.interval must not be 0")

            logger.info("Take the frame interval is {}", config.interval)
            display_delay_ms = max(1, int(1000 / video_fps))
            index = 0

            while True:
                frame = next(video)
                if frame is None:
                    logger.info("The video finished")
                    break

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
                        logger.warning("Inference timeout for frame {}", index)

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

                    show_boxes = bool(detection_boxes)
                    last_visual = {
                        "boxes": [list(box) for box in detection_boxes],
                        "labels": list(detection_class),
                        "scores": [float(score) for score in detection_score],
                        "mode": mode,
                        "latency_ms": latency_ms,
                        "ref": task.ref,
                        "frame_index": index if task.ref is None else task.ref,
                    }
                    _write_task_result(result_file, task)
                    display_frame = _build_display_frame(
                        frame,
                        frame_index=index,
                        detection_boxes=last_visual["boxes"],
                        detection_class=last_visual["labels"],
                        detection_score=last_visual["scores"],
                        mode=mode,
                        sampled=True,
                        latency_ms=latency_ms,
                        ref=task.ref,
                        latest_result_frame=last_visual["frame_index"],
                        show_boxes=show_boxes,
                        detection_count=len(last_visual["boxes"]),
                    )
                else:
                    display_frame = _build_display_frame(
                        frame,
                        frame_index=index,
                        detection_boxes=last_visual["boxes"],
                        detection_class=last_visual["labels"],
                        detection_score=last_visual["scores"],
                        mode=last_visual["mode"],
                        sampled=False,
                        latency_ms=last_visual["latency_ms"],
                        ref=last_visual["ref"],
                        latest_result_frame=last_visual["frame_index"],
                        show_boxes=bool(last_visual["boxes"]),
                        detection_count=len(last_visual["boxes"]),
                    )

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(display_delay_ms) & 0xFF
                if key in (27, ord("q")):
                    logger.info("Video display stopped by user.")
                    break

    logger.info("Saved local inference results to {}", result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="configuration description")
    parser.add_argument("--yaml_path", default="./config/config.yaml", help="input the path of *.yaml")
    parser.add_argument("--edge_id", type=int, default=None, help="override client.edge_id for multi-edge deployment")
    parser.add_argument("--cache_path", type=str, default=None, help="override client.retrain.cache_path (must be unique per edge)")
    parser.add_argument("--video_path", type=str, default=None, help="override client.source.video_path")
    parser.add_argument("--server_ip", type=str, default=None, help="override client.server_ip")
    args = parser.parse_args()

    config = load_runtime_config(args.yaml_path).client

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
    if args.server_ip is not None:
        config.server_ip = args.server_ip

    logger.add(
        f"log/client/edge_{config.edge_id}_{{time}}.log",
        level="INFO",
        rotation="500 MB",
    )

    clear_folder(config.retrain.cache_path)
    edge = EdgeWorker(config)

    try:
        _run_video_loop(config, edge)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        edge.close()
        cv2.destroyAllWindows()
