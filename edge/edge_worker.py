import base64
import io
import os
import sys
import threading
import time
from concurrent import futures

import random

import cv2
import grpc
import numpy as np
import torch
from queue import Queue
from loguru import logger

from database.database import DataBase
from difference.diff import DiffProcessor
from edge.drift_detector import CompositeDriftDetector
from edge.info import TASK_STATE
from edge.resample import history_sample
from edge.task import Task
from edge.transmit import is_network_connected, request_cloud_training, request_cloud_split_training
from grpc_server import message_transmission_pb2_grpc, message_transmission_pb2
from grpc_server.rpc_server import MessageTransmissionServicer

from tools.convert_tool import cv2_to_base64
from tools.file_op import clear_folder, creat_folder, sample_files
from tools.preprocess import frame_resize
from model_management.object_detection import Object_Detection
# Resource-aware CL trigger (RCCDA-inspired multi-queue Lyapunov)
from edge.resource_aware_trigger import (
    ResourceAwareCLTrigger,
    CloudResourceState,
    SplitCandidate,
    build_split_candidates,
    query_cloud_resource,
    estimate_bandwidth,
    create_resource_aware_trigger,
)

# Universal model splitting (optional — requires torchlens)
from model_management.universal_model_split import (
    UniversalModelSplitter,
    extract_split_features,
    save_split_feature_cache,
)

from apscheduler.schedulers.background import BackgroundScheduler

from tools.video_processor import VideoProcessor




class EdgeWorker:
    def __init__(self, config):

        self.config = config
        self.edge_id = config.edge_id

        self.edge_processor = DiffProcessor.str_to_class(config.feature)()
        self.small_object_detection = Object_Detection(config, type='small inference')

        # Drift detector (RCCDA-inspired)
        self.drift_detector = CompositeDriftDetector(config)

        # Resource-aware CL trigger (RCCDA multi-queue Lyapunov)
        # When enabled, this replaces the simple drift_detector trigger with
        # a Lyapunov drift-plus-penalty policy that considers cloud resource
        # state, bandwidth, and intermediate-feature privacy leakage.
        self.resource_trigger: ResourceAwareCLTrigger | None = None  # type: ignore
        self._cloud_state: CloudResourceState | None = None          # type: ignore
        self._split_candidates: list | None = None
        ra_cfg = getattr(config, 'resource_aware_trigger', None)
        self.resource_trigger_enabled = (
            ra_cfg is not None
            and bool(getattr(ra_cfg, 'enabled', False))
        )
        if self.resource_trigger_enabled:
            self.resource_trigger = create_resource_aware_trigger(config)
            logger.info(
                "Resource-aware CL trigger enabled "
                "(pi_bar={}, V={}, λ_cloud={}, λ_bw={}, λ_priv={})",
                self.resource_trigger.pi_bar,
                self.resource_trigger.V,
                self.resource_trigger.lambda_cloud,
                self.resource_trigger.lambda_bw,
                self.resource_trigger.lambda_priv,
            )

        # create database and tables
        self.database = DataBase(self.config.database)
        self.database.use_database()
        self.database.clear_table(self.edge_id)
        self.data_lock = threading.Lock()

        self.queue_info = {'{}'.format(i + 1): 0 for i in range(self.config.edge_num)}
        self.frame_cache = Queue(config.frame_cache_maxsize)
        self.local_queue = Queue(config.local_queue_maxsize)

        self.pred_res = []
        self.avg_scores = []
        self.select_index = []

        # start the thread for diff
        self.diff = 0
        self.key_task = None
        self.diff_processor = threading.Thread(target=self.diff_worker,daemon=True)
        self.diff_processor.start()

        # start the thread for edge server
        #self.edge_server = threading.Thread(target=self.start_edge_server, daemon=True)
        #self.edge_server.start()

        # start the thread for local process
        self.local_processor = threading.Thread(target=self.local_worker,daemon=True)
        self.local_processor.start()

        # start the thread for retrain process
        self.collect_flag = self.config.retrain.flag
        self.collect_time = 0
        self.last_collect_time = 0
        self.collect_time_flag = True
        self.retrain_first = True
        self.retrain_flag = False
        self.cache_count = 0

        self.use_history = True

        # Split-learning mode 
        sl_cfg = getattr(self.config, 'split_learning', None)
        self.split_learning_enabled = bool(getattr(sl_cfg, 'enabled', False)) if sl_cfg else False
        self.drift_frame_indices: list[int] = []
        if self.split_learning_enabled:
            logger.info("Split-learning continual learning enabled")

        # Universal model splitting (model-agnostic, torchlens-based)
        self.universal_split_enabled = False
        self.universal_splitter: UniversalModelSplitter | None = None  # type: ignore
        if self.split_learning_enabled:
            us_cfg = getattr(sl_cfg, 'universal', None)
            if us_cfg and getattr(us_cfg, 'enabled', False):
                self.universal_split_enabled = True
                split_layer = getattr(us_cfg, 'split_layer', None)
                split_module = getattr(us_cfg, 'split_module', None)
                split_strategy = getattr(us_cfg, 'strategy', None)
                target_ratio = float(getattr(us_cfg, 'target_flops_ratio', 0.5))
                max_privacy = float(getattr(us_cfg, 'max_privacy_leakage', 0.1))
                latency_weight = float(getattr(us_cfg, 'latency_weight', 1.0))
                ucb_alpha = float(getattr(us_cfg, 'ucb_alpha', 0.25))
                only_parametric = bool(getattr(us_cfg, 'only_parametric', False))
                try:
                    self.universal_splitter = UniversalModelSplitter(
                        device=next(self.small_object_detection.model.parameters()).device,
                    )
                    sample_input = torch.rand(1, 3, 224, 224).to(
                        next(self.small_object_detection.model.parameters()).device
                    )
                    self.universal_splitter.trace(
                        self.small_object_detection.model, sample_input,
                    )

                    # ---- Split-point selection (priority: explicit > strategy) ----
                    if split_module:
                        # Explicit module-name boundary (e.g. 'backbone')
                        idx = self.universal_splitter.find_best_split_by_module(split_module)
                        self.universal_splitter.split(layer_index=idx)
                    elif split_layer is not None:
                        # Explicit layer index or label
                        if isinstance(split_layer, str):
                            self.universal_splitter.split(layer_label=split_layer)
                        else:
                            self.universal_splitter.split(layer_index=int(split_layer))

                    logger.info(
                        "Universal model splitting enabled — split at layer {}",
                        self.universal_splitter.split_index,
                    )
                except Exception as exc:
                    logger.exception("Failed to init universal splitter: {}", exc)
                    self.universal_split_enabled = False
                    self.universal_splitter = None

        if self.split_learning_enabled and not self.universal_split_enabled:
            logger.warning(
                "Split-learning requested but universal model splitting is unavailable; "
                "disabling split-learning because legacy Faster R-CNN split path has been removed."
            )
            self.split_learning_enabled = False

        self.retrain_processor = threading.Thread(target=self.retrain_worker,daemon=True)
        self.retrain_processor.start()

        # start the thread pool for offload
        self.offloading_executor = futures.ThreadPoolExecutor(max_workers=config.offloading_max_worker,)




    def diff_worker(self):
        logger.info('the offloading policy is {}'.format(self.config.policy))
        if self.config.diff_flag:
            task = self.frame_cache.get(block=True)
            frame = task.frame_edge
            self.pre_frame_feature = self.edge_processor.get_frame_feature(frame)
            self.key_task = task
            # Create an entry for the task in the database table
            data = (
                task.frame_index,
                task.start_time,
                None,
                "",
                "",)
            with self.data_lock:
                self.database.insert_data(self.edge_id, data)
            task.edge_process = True
            self.local_queue.put(task, block=True)

            while True:
                # get task from cache
                task = self.frame_cache.get(block=True)
                # Create an entry for the task in the database table
                data = (
                    task.frame_index,
                    task.start_time,
                    None,
                    "",
                    "",)
                with self.data_lock:
                    self.database.insert_data(self.edge_id, data)

                frame = task.frame_edge
                self.frame_feature = self.edge_processor.get_frame_feature(frame)
                # calculate and accumulate the difference
                self.diff += self.edge_processor.cal_frame_diff(self.frame_feature, self.pre_frame_feature)
                self.pre_frame_feature = self.frame_feature
                # Process the video frame greater than a certain threshold
                if self.diff >= self.config.diff_thresh:
                    self.diff = 0.0
                    self.key_task = task
                    self.decision_worker(task)
                else:
                    task.end_time = time.time()
                    task.ref = self.key_task.frame_index
                    task.state = TASK_STATE.FINISHED
                    self.update_table(task)

    def update_table(self, task):
        if task.state == TASK_STATE.FINISHED:
            state = "Finished"
        elif task.state == TASK_STATE.TIMEOUT:
            state = "Timeout"
        else:
            state = ""
        if task.ref is not None:
            result = {'ref': task.ref}
        else:
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
        with self.data_lock:
            self.database.update_data(task.edge_id, data)


    def local_worker(self):
        while True:
            # get a inference task from local queue
            task = self.local_queue.get(block=True)
            if time.time() - task.start_time >= self.config.wait_thresh:
                end_time = time.time()
                task.end_time = end_time
                task.state = TASK_STATE.TIMEOUT
                self.update_table(task)
                continue
            self.queue_info['{}'.format(self.edge_id)] = self.local_queue.qsize()
            current_frame = task.frame_edge
            # get the query image and the small inference result
            split_payload = None
            if self.split_learning_enabled and self.universal_split_enabled and self.universal_splitter is not None:
                (
                    offloading_image,
                    detection_boxes,
                    detection_class,
                    detection_score,
                    split_payload,
                ) = self.small_object_detection.small_inference(
                    current_frame,
                    splitter=self.universal_splitter,
                    return_split_payload=True,
                )
            else:
                offloading_image, detection_boxes, detection_class, detection_score \
                    = self.small_object_detection.small_inference(current_frame)



            # collect data for retrain
            if self.collect_flag and self.retrain_first:
                if self.collect_time_flag:
                    self.last_collect_time = time.time()
                    self.collect_time_flag = False
                logger.debug("ctime L{}, {}".format(self.last_collect_time, self.collect_time))
                self.collect_data(
                    task,
                    current_frame,
                    detection_boxes,
                    detection_class,
                    detection_score,
                    split_payload=split_payload,
                )
            elif self.collect_flag:
                duration = time.time() - self.last_collect_time
                logger.debug("duration {}, L{} {}".format(duration,self.last_collect_time, self.collect_time))
                if duration > self.config.retrain.window:
                    if self.collect_time_flag:
                        self.collect_time = time.time()
                        self.collect_time_flag = False
                    self.collect_data(
                        task,
                        current_frame,
                        detection_boxes,
                        detection_class,
                        detection_score,
                        split_payload=split_payload,
                    )

            if detection_boxes is not None:
                task.add_result(detection_boxes, detection_class, detection_score)
                # whether to further inference
                if offloading_image is not None and task.edge_process is False:
                    # offload to cloud for further inference
                    task.frame_cloud = offloading_image
                    self.offloading_executor.submit(self.offload_worker, task)
                # local infer, upload result
                else:
                    end_time = time.time()
                    task.end_time = end_time
                    task.state = TASK_STATE.FINISHED
                    self.update_table(task)
            else:
                if offloading_image is not None and task.edge_process is False:
                    # offload to cloud for further inference
                    task.frame_cloud = offloading_image
                    self.offloading_executor.submit(self.offload_worker, task)
                end_time = time.time()
                task.end_time = end_time
                task.state = TASK_STATE.FINISHED
                # upload the result to database
                self.update_table(task)
                logger.info('target not detected')

    def offload_worker(self, task, destination_edge_id=None):
        new_height = self.config.new_height
        qp = self.config.quality
        # offload to cloud
        if destination_edge_id is None:
            frame_cloud = task.frame_cloud
            assert frame_cloud is not None
            detection_boxes, detection_class, detection_score = task.get_result()
            if len(detection_boxes) != 0:
                part_result = {'boxes': detection_boxes, 'labels': detection_class, 'scores': detection_score}
            else:
                part_result = ""
            # offload to the cloud directly
            if task.directly_cloud:
                new_frame = frame_resize(frame_cloud, new_height=new_height.directly_cloud)
                encoded_image = cv2_to_base64(new_frame, qp=qp.directly_cloud)
            # The task from another edge node is offloaded to the cloud after local inference
            elif task.other:
                if task.frame_cloud.shape[0] < new_height.another_cloud:
                    logger.error("the new height can not larger than the height of current frame.")
                new_frame = frame_resize(frame_cloud, new_height=new_height.another_cloud)
                encoded_image = cv2_to_base64(new_frame, qp=qp.another_cloud)
            # offload to the cloud after local inference
            else:
                new_frame = frame_resize(frame_cloud, new_height=new_height.local_cloud)
                encoded_image = cv2_to_base64(new_frame, qp=qp.local_cloud)

            msg_request = message_transmission_pb2.MessageRequest(
                source_edge_id=int(self.edge_id),
                frame_index=int(task.frame_index),
                start_time=str(task.start_time),
                frame=encoded_image,
                part_result=str(part_result),
                raw_shape=str(task.raw_shape),
                new_shape=str(new_frame.shape),
                note="",
            )
            try:
                channel = grpc.insecure_channel(self.config.server_ip)
                stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
                res = stub.task_processor(msg_request)
            except Exception as e:
                logger.exception("the cloud can not reply, {}".format(e))
                # put the task into local queue
                if task.directly_cloud:
                    task.edge_process = True
                    self.local_queue.put(task, block=True)
                # upload the result
                else:
                    end_time = time.time()
                    task.end_time = end_time
                    self.update_table(task)
            else:
                self.queue_info['{}'.format(self.edge_id)] = self.local_queue.qsize()

        # to another edge
        else:
            frame_edge = task.frame_edge
            new_frame = frame_resize(frame_edge, new_height=new_height.another)
            encoded_image = cv2_to_base64(new_frame, qp=qp.another)
            if task.edge_process is True:
                note = "edge process"
            msg_request = message_transmission_pb2.MessageRequest(
                source_edge_id=int(self.edge_id),
                frame_index=int(task.frame_index),
                start_time=str(task.start_time),
                frame=encoded_image,
                part_result="",
                raw_shape=str(task.raw_shape),
                new_shape=str(new_frame.shape),
                note=note,
            )
            destinations = self.config.destinations
            destination_edge_ip = destinations['ip'][destinations['id'] == destination_edge_id]
            try:
                channel = grpc.insecure_channel(destination_edge_ip)
                stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
                res = stub.task_processor(msg_request)
            except Exception as e:
                logger.exception("the edge id{}, ip {} can not reply, {}".format(destination_edge_id,destination_edge_ip,e))
                self.local_queue.put(task, block=True)
            else:
                logger.debug("forward to other edge")
                self.queue_info['{}'.format(self.edge_id)] = self.local_queue.qsize()
                self.queue_info['{}'.format(res.destination_edge_id)] = res.local_length

    # collect data for retrain
    def collect_data(
        self,
        task,
        frame,
        detection_boxes,
        detection_class,
        detection_score,
        split_payload=None,
    ):
        if detection_score is not None:
            creat_folder(self.config.retrain.cache_path)
            cv2.imwrite(os.path.join(self.config.retrain.cache_path,'frames', str(task.frame_index) + '.jpg'), frame)
            avg_score = float(np.mean(detection_score))
            self.avg_scores.append({task.frame_index: avg_score})
            self.cache_count += 1

            # Query the drift detector; it may trigger a retrain ahead of the
            # count-based threshold when concept drift is detected.
            drift_detected = self.drift_detector.update(avg_score)

            # ── Resource-aware CL trigger (RCCDA multi-queue Lyapunov) ──
            # When enabled, the resource-aware trigger *replaces* the drift
            # detector's trigger decision with a Lyapunov policy that
            # considers cloud resource state, bandwidth, and privacy.
            resource_trigger_fired = False
            if self.resource_trigger_enabled and self.resource_trigger is not None:
                try:
                    # 1. Query cloud resources (non-blocking, cached)
                    if self._cloud_state is None or self._cloud_state.is_stale(30.0):
                        self._cloud_state = query_cloud_resource(
                            self.config.server_ip, timeout_sec=3.0
                        )

                    # 2. Estimate bandwidth
                    bw_mbps = estimate_bandwidth(self.config.server_ip)

                    # 3. Estimate smashed-data size for current split
                    feat_numel = 0
                    feat_bytes = 0
                    if (
                        self.universal_split_enabled
                        and self.universal_splitter is not None
                        and self.universal_splitter.split_index is not None
                    ):
                        layers = self.universal_splitter.layer_info
                        if layers and self.universal_splitter.split_index < len(layers):
                            linfo = layers[self.universal_splitter.split_index]
                            feat_numel = 1
                            for d in linfo.output_shape:
                                feat_numel *= d
                            feat_bytes = feat_numel * 4  # float32

                    # 4. Make the joint CL-trigger + split-point decision
                    #    Build split candidates for Lyapunov split selection
                    if (
                        self.universal_split_enabled
                        and self.universal_splitter is not None
                        and self._split_candidates is None
                    ):
                        try:
                            self._split_candidates = build_split_candidates(
                                self.universal_splitter,
                                only_parametric=False,
                            )
                        except Exception:
                            self._split_candidates = []

                    trigger, new_split_idx = self.resource_trigger.decide(
                        avg_confidence=avg_score,
                        cloud_state=self._cloud_state,
                        bandwidth_mbps=bw_mbps,
                        feature_bytes=feat_bytes,
                        feature_numel=feat_numel,
                        split_candidates=self._split_candidates or None,
                    )
                    resource_trigger_fired = trigger

                    # 5. Apply Lyapunov-selected split point if changed
                    if (
                        new_split_idx is not None
                        and self.universal_splitter is not None
                        and new_split_idx != self.universal_splitter.split_index
                    ):
                        old_idx = self.universal_splitter.split_index
                        self.universal_splitter.split(layer_index=new_split_idx)
                        logger.info(
                            "[ResourceCLTrigger] Re-split: layer {} → {}",
                            old_idx, new_split_idx,
                        )
                        # Update split_meta.json so cloud uses the new split
                        import json as _json
                        meta_path = os.path.join(
                            self.config.retrain.cache_path, "features", "split_meta.json"
                        )
                        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
                        meta = {
                            "universal": True,
                            "split_index": new_split_idx,
                        }
                        with open(meta_path, "w") as _mf:
                            _json.dump(meta, _mf)
                        feat_dir = os.path.join(
                            self.config.retrain.cache_path, "features"
                        )
                        if os.path.isdir(feat_dir):
                            for fn in os.listdir(feat_dir):
                                if not fn.endswith(".pt"):
                                    continue
                                fp = os.path.join(feat_dir, fn)
                                if os.path.isfile(fp):
                                    try:
                                        os.remove(fp)
                                    except OSError:
                                        pass
                        self.select_index = []
                        self.avg_scores = []
                        self.drift_frame_indices = []
                        self.cache_count = 0
                        logger.info(
                            "[ResourceCLTrigger] Cleared cached split payloads after re-split."
                        )
                except Exception as exc:
                    logger.warning("[ResourceCLTrigger] Decision failed: {}", exc)
                    resource_trigger_fired = False

            # ---- Split learning: extract & cache backbone features ----
            if self.split_learning_enabled:
                try:
                    if self.universal_split_enabled and self.universal_splitter is not None:
                        # ---- Universal path (any model, any split layer) ----
                        from PIL import Image
                        from torchvision import transforms as T
                        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        img_t = T.ToTensor()(img_pil).unsqueeze(0)
                        dev = next(self.small_object_detection.model.parameters()).device
                        img_t = img_t.to(dev)
                        if split_payload is not None:
                            intermediate = split_payload
                        else:
                            with self.small_object_detection.model_lock:
                                intermediate = extract_split_features(
                                    self.universal_splitter, img_t,
                                )
                        save_split_feature_cache(
                            cache_path=self.config.retrain.cache_path,
                            frame_index=task.frame_index,
                            intermediate=intermediate,
                            is_drift=drift_detected,
                            pseudo_boxes=detection_boxes,
                            pseudo_labels=detection_class,
                            pseudo_scores=detection_score,
                            extra_metadata={
                                "split_index": self.universal_splitter.split_index,
                            },
                        )
                        # Persist split metadata so the cloud knows which
                        # path / layer index to use during retraining.
                        import json as _json
                        meta_path = os.path.join(
                            self.config.retrain.cache_path, "features", "split_meta.json"
                        )
                        if not os.path.exists(meta_path):
                            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
                            meta = {
                                "universal": True,
                                "split_index": self.universal_splitter.split_index,
                                "model_name": self.config.client.detection.model_name
                                    if hasattr(self.config, 'client') else "",
                            }
                            with open(meta_path, "w") as _mf:
                                _json.dump(meta, _mf)
                    if drift_detected:
                        self.drift_frame_indices.append(task.frame_index)
                    logger.debug("Cached features for frame {} (drift={})",
                                 task.frame_index, drift_detected)
                except Exception as exc:
                    logger.exception("Failed to extract/cache features for frame {}: {}",
                                     task.frame_index, exc)

            min_samples = max(getattr(self.config.retrain, 'select_num', 10), 10)
            count_trigger = self.cache_count >= self.config.retrain.collect_num

            # Resource-aware trigger supersedes drift_detected when enabled
            if self.resource_trigger_enabled:
                drift_trigger = resource_trigger_fired and self.cache_count >= min_samples
            else:
                drift_trigger = drift_detected and self.cache_count >= min_samples

            if count_trigger or drift_trigger:
                if drift_trigger and not count_trigger:
                    trigger_source = "resource-aware" if self.resource_trigger_enabled else "drift"
                    logger.info("{} trigger — starting cloud retraining early "
                                "(cached={} samples)", trigger_source, self.cache_count)

                if self.split_learning_enabled:
                    # Split learning: use ALL cached frames for training
                    self.select_index = [list(d.keys())[0] for d in self.avg_scores]
                else:
                    # Original: select worst-confidence frames
                    smallest_elements = sorted(self.avg_scores, key=lambda d: list(d.values())[0])[:self.config.retrain.select_num]
                    self.select_index = [list(d.keys())[0] for d in smallest_elements]

                logger.debug("the select index {}".format(self.select_index))
                self.pred_res = []
                self.collect_flag = False
                self.cache_count = 0
                self.retrain_flag = True


    # retrain — continual learning is performed entirely on the cloud
    def retrain_worker(self):
        while True:
            if self.retrain_flag:
                num_epoch = int(getattr(self.config.retrain, 'num_epoch', 0))

                if self.split_learning_enabled:
                    # ---- Split-learning path ----
                    _retrain_start = time.time()
                    logger.info(
                        "Sending {} features ({} drift) to cloud for split-learning CL",
                        len(self.select_index), len(self.drift_frame_indices),
                    )
                    success, model_b64, msg = request_cloud_split_training(
                        self.config.server_ip,
                        self.edge_id,
                        self.select_index,
                        self.drift_frame_indices,
                        self.config.retrain.cache_path,
                        num_epoch,
                    )
                else:
                    # ---- Original full-image path ----
                    _retrain_start = time.time()
                    logger.info("Sending {} frames to cloud for continual learning",
                                len(self.select_index))
                    success, model_b64, msg = request_cloud_training(
                        self.config.server_ip,
                        self.edge_id,
                        self.select_index,
                        self.config.retrain.cache_path,
                        num_epoch,
                    )

                _retrain_elapsed = time.time() - _retrain_start

                if success and model_b64:
                    try:
                        buf = io.BytesIO(base64.b64decode(model_b64))
                        state_dict = torch.load(buf, map_location='cpu')
                        with self.small_object_detection.model_lock:
                            self.small_object_detection.model.load_state_dict(state_dict)
                            self.small_object_detection.model.eval()
                        logger.success("Edge model updated from cloud successfully")
                    except Exception as exc:
                        logger.exception("Failed to load cloud-returned model weights: {}", exc)

                else:
                    logger.error("Cloud continual learning failed: {}", msg)

                self.retrain_flag = False
                if self.use_history:
                    self.select_index, self.avg_scores = history_sample(
                        self.select_index, self.avg_scores)
                    sample_files(
                        os.path.join(self.config.retrain.cache_path, 'frames'),
                        self.select_index)
                    # Also prune feature cache when using split learning
                    if self.split_learning_enabled:
                        feat_dir = os.path.join(self.config.retrain.cache_path, 'features')
                        if os.path.isdir(feat_dir):
                            sample_files(feat_dir, self.select_index)
                        # Keep only drift indices that survived history sampling
                        self.drift_frame_indices = [
                            i for i in self.drift_frame_indices if i in self.select_index
                        ]
                    self.cache_count = len(self.select_index)
                else:
                    clear_folder(self.config.retrain.cache_path)
                    # Also clear feature cache
                    if self.split_learning_enabled:
                        feat_dir = os.path.join(self.config.retrain.cache_path, 'features')
                        if os.path.isdir(feat_dir):
                            for fn in os.listdir(feat_dir):
                                fp = os.path.join(feat_dir, fn)
                                if os.path.isfile(fp):
                                    try:
                                        os.remove(fp)
                                    except OSError:
                                        pass
                        self.drift_frame_indices = []
                    self.select_index = []
                    self.avg_scores = []
                if self.retrain_first is False:
                    self.last_collect_time = self.collect_time
                self.retrain_first = False
                self.collect_time_flag = True
                self.collect_flag = True
            time.sleep(0.2)



    def start_edge_server(self):
        logger.info("edge {} server is starting".format(self.edge_id))
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        message_transmission_pb2_grpc.add_MessageTransmissionServicer_to_server(
            MessageTransmissionServicer(self.local_queue, self.edge_id, self.small_object_detection ), server)
        server.add_insecure_port('[::]:50050')
        server.start()
        logger.success("edge {} server is listening".format(self.edge_id))
        server.wait_for_termination()


    def decision_worker(self, task):
        policy = self.config.policy
        if policy == 'Edge-Local':
            task.edge_process = True
            self.local_queue.put(task, block=True)

        elif policy == 'Edge-Cloud-Assisted':
            self.local_queue.put(task, block=True)

        elif policy =='Edge-Cloud-Threshold':
            queue_thresh = self.config.queue_thresh
            if self.local_queue.qsize() <= queue_thresh:
                task.edge_process = True
                self.local_queue.put(task, block=True)
            else:
                task.frame_cloud = task.frame_edge
                task.directly_cloud = True
                self.offloading_executor.submit(self.offload_worker, task)

        elif policy == 'Edge-Shortest':
            shortest_id = min(self.queue_info, key=self.queue_info.get)
            task.edge_process = True
            if int(shortest_id) == self.edge_id or \
                    self.queue_info['{}'.format(shortest_id)] == self.queue_info['{}'.format(self.edge_id)]:
                self.local_queue.put(task, block=True)
            else:
                destinations = self.config.destinations
                destination_edge_ip = destinations['ip'][destinations['id'] == shortest_id]
                if is_network_connected(destination_edge_ip):
                    self.offloading_executor.submit(self.offload_worker, task, int(shortest_id))
                else:
                    logger.info("could not connect to {}".format(destination_edge_ip))
                    self.local_queue.put(task, block=True)

        elif policy == 'Shortest-Cloud-Threshold':
            queue_thresh = self.config.queue_thresh
            shortest_info = min(zip(self.queue_info.values(), self.queue_info.keys()))
            shortest_length = shortest_info[0]
            shortest_id = shortest_info[1]
            if shortest_length > queue_thresh:
                task.frame_cloud = task.frame_edge
                task.directly_cloud = True
                self.offloading_executor.submit(self.offload_worker, task)
            elif int(shortest_id) == self.edge_id:
                task.edge_process = True
                self.local_queue.put(task, block=True)
            else:
                task.edge_process = True
                self.offloading_executor.submit(self.offload_worker, task, int(shortest_id))

        else:
            logger.error('the policy does not exist.')

