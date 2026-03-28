import argparse
import json
import time
from pathlib import Path

import cv2
import munch
import yaml
from loguru import logger

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
):
    rendered = draw_detection(frame, detection_boxes, detection_class, detection_score)
    lines = [
        f"Frame: {frame_index}",
        f"Detections: {len(detection_boxes)}",
        f"Mode: {mode}",
        f"Sampled: {'yes' if sampled else 'no'}",
    ]
    if latency_ms is not None:
        lines.append(f"Latency: {latency_ms:.1f} ms")
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

                    last_visual = {
                        "boxes": [list(box) for box in detection_boxes],
                        "labels": list(detection_class),
                        "scores": [float(score) for score in detection_score],
                        "mode": mode,
                        "latency_ms": latency_ms,
                        "ref": task.ref,
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
    args = parser.parse_args()

    with open(args.yaml_path, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    config = munch.munchify(config).client
    logger.add("log/client/client_{time}.log", level="INFO", rotation="500 MB")

    clear_folder(config.retrain.cache_path)
    edge = EdgeWorker(config)

    try:
        _run_video_loop(config, edge)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cv2.destroyAllWindows()
