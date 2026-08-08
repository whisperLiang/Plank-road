"""Load one edge detector and run one video frame through the production wrapper."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_management.object_detection import Object_Detection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    weights = Path(args.weights)
    video = Path(args.video)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if not video.is_file():
        raise FileNotFoundError(video)

    capture = cv2.VideoCapture(str(video))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"could not read first frame from {video}")

    config = SimpleNamespace(
        lightweight=args.model,
        weights_path=str(weights),
        tinynext_input_size=640,
    )
    started = time.perf_counter()
    detector = Object_Detection(config, "small inference")
    loaded_at = time.perf_counter()
    result = detector.infer_sample(frame)
    finished_at = time.perf_counter()
    predictions = getattr(result, "predictions", None)
    if predictions is None:
        predictions = getattr(result, "detections", None)
    try:
        prediction_count = len(predictions) if predictions is not None else None
    except TypeError:
        prediction_count = None
    print(
        json.dumps(
            {
                "model": args.model,
                "device": str(next(detector.model.parameters()).device),
                "torch": torch.__version__,
                "cuda": torch.cuda.is_available(),
                "frame_shape": list(frame.shape),
                "prediction_count": prediction_count,
                "load_seconds": round(loaded_at - started, 3),
                "inference_seconds": round(finished_at - loaded_at, 3),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
