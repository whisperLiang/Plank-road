import base64
import io
import os
import threading
import time
from concurrent import futures
from queue import Queue

import grpc
import torch
from loguru import logger

from database.database import DataBase
from difference.diff import DiffProcessor
from edge.drift_detector import CompositeDriftDetector
from edge.info import TASK_STATE
from edge.resource_aware_trigger import (
    CloudResourceState,
    PendingTrainingStats,
    ResourceAwareCLTrigger,
    TrainingDecision,
    create_resource_aware_trigger,
    estimate_bandwidth,
    query_cloud_resource,
)
from edge.sample_store import EdgeSampleStore, HIGH_CONFIDENCE, LOW_CONFIDENCE
from edge.task import Task
from edge.transmit import request_continual_learning
from grpc_server import message_transmission_pb2_grpc
from grpc_server.rpc_server import MessageTransmissionServicer
from model_management.fixed_split import (
    SplitConstraints,
    SplitPlan,
    load_or_compute_fixed_split_plan,
)
from model_management.object_detection import InferenceArtifacts, Object_Detection
from model_management.universal_model_split import UniversalModelSplitter


class EdgeWorker:
    def __init__(self, config):
        self.config = config
        self.edge_id = config.edge_id

        self.edge_processor = DiffProcessor.str_to_class(config.feature)()
        self.small_object_detection = Object_Detection(config, type="small inference")
        self.drift_detector = CompositeDriftDetector(config)

        self.resource_trigger: ResourceAwareCLTrigger | None = None
        self._cloud_state: CloudResourceState | None = None
        ra_cfg = getattr(config, "resource_aware_trigger", None)
        self.resource_trigger_enabled = bool(getattr(ra_cfg, "enabled", False)) if ra_cfg else False
        if self.resource_trigger_enabled:
            self.resource_trigger = create_resource_aware_trigger(config)
            logger.info(
                "Resource-aware CL trigger enabled (pi_bar={}, V={}, lambda_cloud={}, lambda_bw={})",
                self.resource_trigger.pi_bar,
                self.resource_trigger.V,
                self.resource_trigger.lambda_cloud,
                self.resource_trigger.lambda_bw,
            )

        self.database = DataBase(self.config.database)
        self.database.use_database()
        self.database.clear_table(self.edge_id)
        self.data_lock = threading.Lock()

        self.queue_info = {f"{i + 1}": 0 for i in range(self.config.edge_num)}
        self.frame_cache = Queue(config.frame_cache_maxsize)
        self.local_queue = Queue(config.local_queue_maxsize)

        self.collect_flag = bool(self.config.retrain.flag)
        self.retrain_flag = False
        self.pending_training_decision: TrainingDecision | None = None
        self.sample_store = EdgeSampleStore(
            os.path.join(self.config.retrain.cache_path, "sample_store")
        )
        self.model_id = getattr(self.small_object_detection, "model_name", "edge-model")
        self.model_version = "0"
        self.bundle_cache_path = os.path.join(self.config.retrain.cache_path, "server_bundle")
        self.sample_confidence_threshold = float(
            getattr(
                ra_cfg if ra_cfg is not None else getattr(config, "drift_detection", None),
                "confidence_threshold",
                0.5,
            )
        )

        self.split_learning_enabled = True
        self.universal_split_enabled = False
        self.universal_splitter: UniversalModelSplitter | None = None
        self.fixed_split_plan: SplitPlan | None = None
        self._init_fixed_split_runtime()

        self.diff = 0.0
        self.key_task = None
        self.diff_processor = threading.Thread(target=self.diff_worker, daemon=True)
        self.local_processor = threading.Thread(target=self.local_worker, daemon=True)
        self.retrain_processor = threading.Thread(target=self.retrain_worker, daemon=True)
        self.diff_processor.start()
        self.local_processor.start()
        self.retrain_processor.start()

    def _init_fixed_split_runtime(self) -> None:
        sl_cfg = getattr(self.config, "split_learning", None)
        fixed_split_cfg = getattr(sl_cfg, "fixed_split", None) if sl_cfg else None
        try:
            self.universal_splitter = UniversalModelSplitter(
                device=next(self.small_object_detection.model.parameters()).device,
            )
            sample_input = self.small_object_detection.build_split_sample_input((224, 224))
            self.universal_splitter.trace(
                self.small_object_detection.model,
                sample_input,
            )
            constraints = SplitConstraints.from_config(fixed_split_cfg)
            cache_path = os.path.join(self.config.retrain.cache_path, "fixed_split_plan.json")
            self.fixed_split_plan = load_or_compute_fixed_split_plan(
                self.small_object_detection.model,
                constraints,
                sample_input=sample_input,
                device=next(self.small_object_detection.model.parameters()).device,
                model_name=self.model_id,
                cache_path=cache_path,
                splitter=self.universal_splitter,
            )
            self.universal_split_enabled = True
            logger.info(
                "Fixed split plan ready (split_config_id={}, split_index={}, payload_bytes={})",
                self.fixed_split_plan.split_config_id,
                self.fixed_split_plan.split_index,
                self.fixed_split_plan.payload_bytes,
            )
        except Exception as exc:
            logger.exception("Failed to initialise fixed split plan: {}", exc)
            self.split_learning_enabled = False
            self.universal_split_enabled = False
            self.universal_splitter = None
            self.fixed_split_plan = None

    def _next_sample_id(self, task: Task) -> str:
        return f"{task.frame_index}-{int(task.start_time * 1000)}"

    def _make_training_decision(
        self,
        *,
        confidence: float,
        drift_detected: bool,
        stats: PendingTrainingStats,
    ) -> TrainingDecision:
        if self.resource_trigger_enabled and self.resource_trigger is not None:
            try:
                if self._cloud_state is None or self._cloud_state.is_stale(30.0):
                    self._cloud_state = query_cloud_resource(
                        self.config.server_ip,
                        edge_id=self.edge_id,
                        timeout_sec=3.0,
                    )
                bandwidth_mbps = estimate_bandwidth(self.config.server_ip)
                return self.resource_trigger.decide(
                    avg_confidence=confidence,
                    drift_detected=drift_detected,
                    cloud_state=self._cloud_state,
                    bandwidth_mbps=bandwidth_mbps,
                    sample_stats=stats,
                )
            except Exception as exc:
                logger.warning("Resource-aware trigger decision failed: {}", exc)

        should_train = (
            stats.total_samples >= max(1, int(getattr(self.config.retrain, "collect_num", 1)))
            or drift_detected
        )
        return TrainingDecision(
            train_now=bool(should_train),
            send_low_conf_features=False,
            urgency=1.0 if should_train else 0.0,
            compute_pressure=0.0,
            bandwidth_pressure=0.0,
            reason="Fallback trigger without low-confidence feature upload.",
        )

    def diff_worker(self):
        if not self.config.diff_flag:
            while True:
                task = self.frame_cache.get(block=True)
                self._insert_task_row(task)
                self.decision_worker(task)

        task = self.frame_cache.get(block=True)
        frame = task.frame_edge
        self.pre_frame_feature = self.edge_processor.get_frame_feature(frame)
        self.key_task = task
        self._insert_task_row(task)
        self.decision_worker(task)

        while True:
            task = self.frame_cache.get(block=True)
            self._insert_task_row(task)
            frame = task.frame_edge
            self.frame_feature = self.edge_processor.get_frame_feature(frame)
            self.diff += self.edge_processor.cal_frame_diff(
                self.frame_feature,
                self.pre_frame_feature,
            )
            self.pre_frame_feature = self.frame_feature
            if self.diff >= self.config.diff_thresh:
                self.diff = 0.0
                self.key_task = task
                self.decision_worker(task)
            else:
                task.end_time = time.time()
                task.ref = self.key_task.frame_index
                task.state = TASK_STATE.FINISHED
                self.update_table(task)

    def _insert_task_row(self, task: Task) -> None:
        data = (
            task.frame_index,
            task.start_time,
            None,
            "",
            "",
        )
        with self.data_lock:
            self.database.insert_data(self.edge_id, data)

    def update_table(self, task):
        if task.state == TASK_STATE.FINISHED:
            state = "Finished"
        elif task.state == TASK_STATE.TIMEOUT:
            state = "Timeout"
        else:
            state = ""
        if task.ref is not None:
            result = {"ref": task.ref}
        else:
            detection_boxes, detection_class, detection_score = task.get_result()
            result = {
                "labels": detection_class,
                "boxes": detection_boxes,
                "scores": detection_score,
            }
        data = (
            task.end_time,
            str(result),
            state,
            task.frame_index,
        )
        with self.data_lock:
            self.database.update_data(task.edge_id, data)

    def local_worker(self):
        while True:
            task = self.local_queue.get(block=True)
            if time.time() - task.start_time >= self.config.wait_thresh:
                task.end_time = time.time()
                task.state = TASK_STATE.TIMEOUT
                self.update_table(task)
                continue

            self.queue_info[f"{self.edge_id}"] = self.local_queue.qsize()
            current_frame = task.frame_edge
            inference = self.small_object_detection.infer_sample(
                current_frame,
                splitter=self.universal_splitter if self.universal_split_enabled else None,
            )

            task.add_result(
                inference.detection_boxes or None,
                inference.detection_class or None,
                inference.detection_score or None,
            )

            if (
                self.collect_flag
                and self.split_learning_enabled
                and inference.intermediate is not None
                and self.fixed_split_plan is not None
            ):
                self.collect_data(task, current_frame, inference)

            task.end_time = time.time()
            task.state = TASK_STATE.FINISHED
            self.update_table(task)

    def collect_data(self, task: Task, frame, inference: InferenceArtifacts) -> None:
        confidence = float(inference.confidence)
        drift_detected = self.drift_detector.update(confidence)
        confidence_bucket = (
            HIGH_CONFIDENCE
            if confidence >= self.sample_confidence_threshold
            else LOW_CONFIDENCE
        )
        save_raw = confidence_bucket == LOW_CONFIDENCE or drift_detected
        sample_id = self._next_sample_id(task)
        self.sample_store.store_sample(
            sample_id=sample_id,
            frame_index=task.frame_index,
            confidence=confidence,
            split_config_id=self.fixed_split_plan.split_config_id,
            model_id=self.model_id,
            model_version=self.model_version,
            confidence_bucket=confidence_bucket,
            inference_result={
                "boxes": inference.detection_boxes,
                "labels": inference.detection_class,
                "scores": inference.detection_score,
            },
            intermediate=inference.intermediate,
            drift_flag=drift_detected,
            raw_frame=frame if save_raw else None,
        )

        if self.retrain_flag:
            return

        stats = PendingTrainingStats.from_mapping(self.sample_store.stats())
        decision = self._make_training_decision(
            confidence=confidence,
            drift_detected=drift_detected,
            stats=stats,
        )
        if decision.train_now and stats.total_samples > 0:
            self.pending_training_decision = decision
            self.retrain_flag = True
            self.collect_flag = False
            logger.info(
                "Continual learning triggered (samples={}, send_low_conf_features={}, reason={})",
                stats.total_samples,
                decision.send_low_conf_features,
                decision.reason,
            )

    def retrain_worker(self):
        while True:
            if not self.retrain_flag:
                time.sleep(0.2)
                continue

            decision = self.pending_training_decision
            if self.fixed_split_plan is None or decision is None:
                self.retrain_flag = False
                self.collect_flag = True
                self.pending_training_decision = None
                time.sleep(0.2)
                continue

            num_epoch = int(getattr(self.config.retrain, "num_epoch", 0))
            success, model_b64, msg = request_continual_learning(
                self.config.server_ip,
                edge_id=self.edge_id,
                cache_path=self.bundle_cache_path,
                sample_store=self.sample_store,
                split_plan=self.fixed_split_plan,
                model_id=self.model_id,
                model_version=self.model_version,
                send_low_conf_features=decision.send_low_conf_features,
                num_epoch=num_epoch,
            )

            if success and model_b64:
                try:
                    buf = io.BytesIO(base64.b64decode(model_b64))
                    state_dict = torch.load(buf, map_location="cpu", weights_only=False)
                    with self.small_object_detection.model_lock:
                        self.small_object_detection.model.load_state_dict(state_dict)
                        self.small_object_detection.model.eval()
                    self.model_version = str(int(self.model_version) + 1)
                    self.sample_store.clear()
                    self.drift_detector.reset()
                    logger.success("Edge model updated from cloud successfully")
                except Exception as exc:
                    logger.exception("Failed to load cloud-returned model weights: {}", exc)
            else:
                logger.error("Cloud continual learning failed: {}", msg)

            self.pending_training_decision = None
            self.retrain_flag = False
            self.collect_flag = True
            time.sleep(0.2)

    def start_edge_server(self):
        logger.info("edge {} server is starting".format(self.edge_id))
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        message_transmission_pb2_grpc.add_MessageTransmissionServicer_to_server(
            MessageTransmissionServicer(
                self.local_queue,
                self.edge_id,
                self.small_object_detection,
            ),
            server,
        )
        server.add_insecure_port("[::]:50050")
        server.start()
        logger.success("edge {} server is listening".format(self.edge_id))
        server.wait_for_termination()

    def decision_worker(self, task):
        task.edge_process = True
        self.local_queue.put(task, block=True)
