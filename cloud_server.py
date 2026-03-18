
import argparse
import base64
import io
import json
import os
import random
import threading
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import numpy as np
import pandas as pd
import torch
import torch
print(torch.__version__)
print(torch.version.cuda)
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")

from queue import Queue
from datetime import datetime
import yaml
import munch
import grpc
from concurrent import futures
from torch.utils.data import DataLoader

from loguru import logger
from database.database import DataBase
from edge.info import TASK_STATE
from grpc_server.rpc_server import MessageTransmissionServicer

from model_management.object_detection import Object_Detection
from model_management.detection_dataset import TrafficDataset
from model_management.detection_metric import RetrainMetric
from model_management.model_info import model_lib
from model_management.model_split import (
    load_feature_cache,
    list_cached_features,
    split_retrain,
)

# Universal model splitting (optional — requires torchlens)
from model_management.universal_model_split import (
    UniversalModelSplitter,
    universal_split_retrain,
    load_split_feature_cache,
)

from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc


def _collate_fn(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# Cloud-side Continual Learning
# ---------------------------------------------------------------------------

class CloudContinualLearner:
    """Performs ground-truth labelling and model retraining on the cloud side.

    Workflow triggered when the edge detects drift:
      1. Edge sends selected frame indices and the path of its local cache.
      2. Cloud runs the large model on each frame to obtain ground-truth boxes.
      3. Cloud saves a CSV annotation file inside the cache directory.
      4. Cloud retrains a **fresh copy** of the lightweight edge model.
      5. Cloud returns the updated state-dict bytes (base-64 encoded).

    The edge model weights are kept separately from the cloud inference model.
    """

    annotation_cols = (
        "frame_index", "target_id",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "score", "object_category",
    )

    def __init__(self, config, large_object_detection: Object_Detection):
        self.config = config
        self.large_od = large_object_detection
        self.lock = threading.Lock()

        # Name of the lightweight model to retrain (mirrors edge model)
        self.edge_model_name = getattr(config, "edge_model_name", "fasterrcnn_mobilenet_v3_large_fpn")
        self.weight_folder = os.path.join(
            os.path.dirname(__file__), "model_management", "models"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Default training hyper-parameters (overridable from config)
        cl_cfg = getattr(config, "continual_learning", None)
        self.default_num_epoch = int(getattr(cl_cfg, "num_epoch", 2)) if cl_cfg else 2

        # Dynamic Activation Sparsity (SURGEON) config
        das_cfg = getattr(config, "das", None)
        self.das_enabled = bool(getattr(das_cfg, "enabled", False)) if das_cfg else False
        self.das_bn_only = bool(getattr(das_cfg, "bn_only", False)) if das_cfg else False
        self.das_probe_samples = int(getattr(das_cfg, "probe_samples", 10)) if das_cfg else 10

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_ground_truth_and_retrain(
        self,
        edge_id: int,
        frame_indices: list[int],
        cache_path: str,
        num_epoch: int = 0,
    ) -> tuple[bool, str, str]:
        """Label frames with large model, retrain edge model, return weights.

        Returns
        -------
        (success, base64_model_data, message)
        """
        if num_epoch <= 0:
            num_epoch = self.default_num_epoch

        if not frame_indices:
            return False, "", "No frame indices provided."

        with self.lock:
            try:
                logger.info(
                    f"[CL] Starting cloud retraining for edge {edge_id}: "
                    f"{len(frame_indices)} frames, {num_epoch} epochs."
                )
                annotation_path = self._generate_annotations(
                    edge_id, frame_indices, cache_path
                )
                model_bytes = self._retrain_edge_model(
                    cache_path, frame_indices, num_epoch
                )
                encoded = base64.b64encode(model_bytes).decode("utf-8")
                logger.success(
                    f"[CL] Retraining done for edge {edge_id}. "
                    f"Model size: {len(model_bytes) // 1024} KB."
                )
                return True, encoded, "Retraining successful"
            except Exception as exc:
                logger.exception(f"[CL] Retraining failed for edge {edge_id}: {exc}")
                return False, "", str(exc)

    # ------------------------------------------------------------------
    # Split-learning continual learning 
    # ------------------------------------------------------------------

    def get_ground_truth_and_split_retrain(
        self,
        edge_id: int,
        all_frame_indices: list[int],
        drift_frame_indices: list[int],
        cache_path: str,
        num_epoch: int = 0,
    ) -> tuple[bool, str, str]:
        """Label **only** drift frames with the large model, then train the
        server-side model (rpn + roi_heads) on **all** cached backbone
        features using split learning.

        Non-drift frames use the edge's pseudo-labels (cached alongside
        the features).  Drift frames receive ground-truth from the cloud's
        large model.

        Returns
        -------
        (success, base64_model_data, message)
        """
        if num_epoch <= 0:
            num_epoch = self.default_num_epoch

        if not all_frame_indices:
            return False, "", "No frame indices provided."

        with self.lock:
            try:
                drift_set = set(drift_frame_indices or [])
                logger.info(
                    "[SplitCL] Starting split-learning retraining for edge {}: "
                    "{} total frames, {} drift frames, {} epochs.",
                    edge_id, len(all_frame_indices), len(drift_set), num_epoch,
                )

                # 1. Annotate **only** drift frames with the large model
                gt_annotations: dict[int, dict] = {}
                frame_dir = os.path.join(cache_path, "frames")
                for idx in drift_frame_indices or []:
                    img_path = os.path.join(frame_dir, f"{idx}.jpg")
                    if not os.path.exists(img_path):
                        logger.warning("[SplitCL] Drift frame {} not found, skipping.", idx)
                        continue
                    frame = cv2.imread(img_path)
                    if frame is None:
                        continue
                    pred_boxes, pred_class, pred_score = self.large_od.large_inference(frame)
                    if pred_boxes is None:
                        continue
                    gt_annotations[idx] = {
                        "boxes":  pred_boxes,
                        "labels": pred_class,
                    }
                logger.info(
                    "[SplitCL] Annotated {} drift frames with large model.",
                    len(gt_annotations),
                )

                # 2. Build the lightweight model for split training
                model_bytes = self._split_retrain_edge_model(
                    cache_path, all_frame_indices, gt_annotations, num_epoch
                )

                encoded = base64.b64encode(model_bytes).decode("utf-8")
                logger.success(
                    "[SplitCL] Split retraining done for edge {}. "
                    "Model size: {} KB.",
                    edge_id, len(model_bytes) // 1024,
                )
                return True, encoded, "Split retraining successful"
            except Exception as exc:
                logger.exception("[SplitCL] Split retraining failed for edge {}: {}", edge_id, exc)
                return False, "", str(exc)

    def _split_retrain_edge_model(
        self,
        cache_path: str,
        all_indices: list[int],
        gt_annotations: dict[int, dict],
        num_epoch: int,
    ) -> bytes:
        """Fine-tune the lightweight model via split learning; return state-dict bytes.

        Automatically detects whether the cached features were produced by the
        *universal* model splitter (``features/split_meta.json`` present) or
        by the legacy Faster R-CNN backbone extractor, and dispatches to the
        appropriate training routine.
        """
        import json as _json
        from model_management.model_zoo import build_detection_model, model_has_roi_heads, is_wrapper_model

        model_name = self.edge_model_name

        if is_wrapper_model(model_name):
            raise NotImplementedError(
                f"[SplitCL] {model_name} is a wrapper model (YOLO/DETR/RT-DETR) and "
                f"does not support torchvision-style split retraining."
            )

        tmp_model = build_detection_model(model_name, pretrained=False, device=self.device)

        # Load existing edge-model weights
        edge_weights = os.path.join(self.weight_folder, "tmp_edge_model.pth")
        if os.path.exists(edge_weights):
            state = torch.load(edge_weights, map_location=self.device)
        else:
            if model_name in model_lib:
                official_path = os.path.join(
                    self.weight_folder, model_lib[model_name]["model_path"]
                )
                if os.path.exists(official_path):
                    state = torch.load(official_path, map_location=self.device)
                else:
                    state = None
            else:
                state = None

        if state is not None:
            tmp_model.load_state_dict(state)

        tmp_model.to(self.device)

        # ---- Detect universal split features ----
        meta_path = os.path.join(cache_path, "features", "split_meta.json")
        use_universal = False
        split_index = None

        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as _mf:
                    meta = _json.load(_mf)
                use_universal = meta.get("universal", False)
                split_index = meta.get("split_index")
                logger.info(
                    "[SplitCL] Detected universal split features — split_index={}",
                    split_index,
                )
            except Exception as exc:
                logger.warning("[SplitCL] Failed to read split_meta.json: {}", exc)

        if use_universal and split_index is not None:
            # ---- Universal split-learning path (any model, any layer) ----
            sample_input = torch.rand(1, 3, 224, 224).to(self.device)
            universal_split_retrain(
                model=tmp_model,
                sample_input=sample_input,
                split_layer=int(split_index),
                cache_path=cache_path,
                all_indices=all_indices,
                gt_annotations=gt_annotations,
                device=self.device,
                num_epoch=num_epoch,
                das_enabled=self.das_enabled,
                das_bn_only=self.das_bn_only,
                das_probe_samples=self.das_probe_samples,
            )
        else:
            # ---- Legacy Faster R-CNN split-learning path ----
            split_retrain(
                model=tmp_model,
                cache_path=cache_path,
                all_indices=all_indices,
                gt_annotations=gt_annotations,
                device=self.device,
                num_epoch=num_epoch,
                das_enabled=self.das_enabled,
                das_bn_only=self.das_bn_only,
                das_probe_samples=self.das_probe_samples,
            )

        # Persist weights so future retraining can fine-tune further
        torch.save(tmp_model.state_dict(), edge_weights)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        buf = io.BytesIO()
        torch.save(tmp_model.state_dict(), buf)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Internal helpers (original full-image retraining)
    # ------------------------------------------------------------------

    def _generate_annotations(
        self, edge_id: int, frame_indices: list[int], cache_path: str
    ) -> str:
        """Run large model on frames; save annotation CSV; return its path."""
        frame_dir = os.path.join(cache_path, "frames")
        annotation_path = os.path.join(cache_path, "annotation.txt")
        import cv2

        rows: list[tuple] = []
        for idx in frame_indices:
            img_path = os.path.join(frame_dir, f"{idx}.jpg")
            if not os.path.exists(img_path):
                logger.warning(f"[CL] Frame {idx} not found at {img_path}, skipping.")
                continue
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            pred_boxes, pred_class, pred_score = self.large_od.large_inference(frame)
            if pred_boxes is None:
                pred_boxes, pred_class, pred_score = [], [], []
            for box, label, score in zip(pred_boxes, pred_class, pred_score):
                rows.append((idx, label, box[0], box[1], box[2], box[3], score, ""))

        if rows:
            np.savetxt(
                annotation_path,
                rows,
                fmt=["%d", "%d", "%f", "%f", "%f", "%f", "%f", "%s"],
                delimiter=",",
            )
        logger.info(f"[CL] Saved {len(rows)} annotations to {annotation_path}")
        return annotation_path

    def _retrain_edge_model(
        self, cache_path: str, frame_indices: list[int], num_epoch: int
    ) -> bytes:
        """Fine-tune the lightweight model; return its state-dict as bytes."""
        from model_management.model_zoo import build_detection_model, model_has_roi_heads, is_wrapper_model

        model_name = self.edge_model_name

        if is_wrapper_model(model_name):
            raise NotImplementedError(
                f"[CL] {model_name} is a wrapper model (YOLO/DETR/RT-DETR) and "
                f"does not support torchvision-style retraining."
            )

        # Build a fresh copy from the saved edge model weights
        tmp_model = build_detection_model(model_name, pretrained=False, device=self.device)

        # Prefer the initialised edge-model weights if they exist
        edge_weights = os.path.join(self.weight_folder, "tmp_edge_model.pth")
        if os.path.exists(edge_weights):
            state = torch.load(edge_weights, map_location=self.device)
        else:
            # Fall back to the model-lib path
            if model_name in model_lib:
                official_path = os.path.join(
                    self.weight_folder, model_lib[model_name]["model_path"]
                )
                if os.path.exists(official_path):
                    state = torch.load(official_path, map_location=self.device)
                    logger.info(f"[CL] Loaded official weights from {official_path}")
                else:
                    logger.warning("[CL] No pre-trained weights; training from random init.")
                    state = None
            else:
                state = None

        if state is not None:
            tmp_model.load_state_dict(state)

        tmp_model.to(self.device)

        # Freeze backbone; fine-tune only head (model-family aware)
        for param in tmp_model.parameters():
            param.requires_grad = False
        if model_has_roi_heads(model_name):
            for param in tmp_model.roi_heads.parameters():
                param.requires_grad = True
        else:
            head_attrs = ['head', 'classification_head', 'regression_head']
            found = False
            for attr in head_attrs:
                if hasattr(tmp_model, attr):
                    for param in getattr(tmp_model, attr).parameters():
                        param.requires_grad = True
                    found = True
            if not found:
                all_params = list(tmp_model.parameters())
                for p in all_params[int(len(all_params) * 0.8):]:
                    p.requires_grad = True

        dataset = TrafficDataset(root=cache_path, select_index=frame_indices)
        if len(dataset) == 0:
            raise ValueError("TrafficDataset is empty — no annotated frames found.")

        data_loader = DataLoader(dataset=dataset, batch_size=2, collate_fn=_collate_fn)
        tr_metric = RetrainMetric()

        roi_params = [p for p in tmp_model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(roi_params, lr=0.005, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        tmp_model.train()
        for epoch in range(num_epoch):
            for images, targets in tr_metric.log_iter(epoch, num_epoch, data_loader):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                loss_dict = tmp_model(images, targets)
                losses = sum(loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                tr_metric.update(loss_dict, losses)
            lr_scheduler.step()

        # Persist to cloud-side tmp so future retraining can fine-tune further
        torch.save(tmp_model.state_dict(), edge_weights)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Serialise state-dict to bytes
        buf = io.BytesIO()
        torch.save(tmp_model.state_dict(), buf)
        return buf.getvalue()


class CloudServer:
    def __init__(self, config):
        self.config = config
        self.server_id = config.server_id
        self.large_object_detection = Object_Detection(config, type='large inference')

        # Cloud-side continual learner (retrains the edge lightweight model)
        self.continual_learner = CloudContinualLearner(config, self.large_object_detection)

        #create database and tables
        self.database = DataBase(self.config.database)
        self.database.use_database()
        self.database.create_table(self.config.edge_ids)

        # start the thread for local process
        self.local_queue = Queue(config.local_queue_maxsize)
        self.local_processor = threading.Thread(target=self.cloud_local, daemon=True)
        self.local_processor.start()


    def cloud_local(self):
        while True:
            task = self.local_queue.get(block=True)
            if time.time() - task.start_time >= self.config.wait_thresh:
                end_time = time.time()
                task.end_time = end_time
                task.state = TASK_STATE.TIMEOUT
                self.update_table(task)
                continue
            task.frame_cloud = task.frame_edge
            frame = task.frame_cloud
            high_boxes, high_class, high_score = self.large_object_detection.large_inference(frame)
            # scale the small result
            scale = task.raw_shape[0] / frame.shape[0]
            if high_boxes:
                high_boxes = (np.array(high_boxes) * scale).tolist()
                task.add_result(high_boxes, high_class, high_score)
            end_time = time.time()
            task.end_time = end_time
            # upload the result to database
            task.state = TASK_STATE.FINISHED
            self.update_table(task)


    def update_table(self, task):
        if task.state == TASK_STATE.FINISHED:
            state = "Finished"
        elif task.state == TASK_STATE.TIMEOUT:
            state = "Timeout"
        else:
            state = ""
        detection_boxes, detection_class, detection_score = task.get_result()
        result = {
            'labels': detection_class,
            'boxes': detection_boxes,
            'scores': detection_score
        }
        # upload the result to database
        data = (
            task.end_time,
            str(result),
            state,
            task.frame_index)
        self.database.update_data(task.edge_id, data)


    def start_server(self):
        logger.info("cloud server is starting")
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        message_transmission_pb2_grpc.add_MessageTransmissionServicer_to_server(
            MessageTransmissionServicer(
                self.local_queue,
                self.server_id,
                self.large_object_detection,
                continual_learner=self.continual_learner,
            ),
            server,
        )
        server.add_insecure_port('[::]:50051')
        server.start()
        logger.info("cloud server is listening")
        server.wait_for_termination()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="configuration description")
    parser.add_argument("--yaml_path", default="./config/config.yaml", help="input the path of *.yaml")
    args = parser.parse_args()
    with open(args.yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    # provide class-like access for dict
    config = munch.munchify(config)
    server_config = config.server
    cloud_server = CloudServer(server_config)
    cloud_server.start_server()