import threading
from dataclasses import dataclass

import cv2
import pandas as pd
import torch
import os
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader
from torchvision.models.detection.backbone_utils import *
from model_management.detection_dataset import TrafficDataset
from model_management.detection_metric import RetrainMetric
from model_management.model_info import annotation_cols
from model_management.model_zoo import (
    build_detection_model, get_models_dir, is_wrapper_model, model_has_roi_heads, get_model_family,
)
from model_management.split_model_adapters import (
    build_split_runtime_sample_input,
    get_split_runtime_model,
    postprocess_split_runtime_output,
    prepare_split_runtime_input,
)
from PIL import Image
from torchvision import transforms
from mapcalc import calculate_map


@dataclass
class InferenceArtifacts:
    intermediate: object | None
    detection_boxes: list
    detection_class: list
    detection_score: list
    confidence: float
    input_tensor_shape: list[int] | None = None

def _collate_fn(batch):
    return tuple(zip(*batch))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.debug(device)

class Object_Detection:
    def __init__(self, config, type):
        self.type = type
        self.config = config
        self.init_model_flag = False
        self.model_lock = threading.Lock()

        self.record_r = open("record_r.txt", "w")
        self.record_p = open("record_p.txt", "w")

        if type == 'small inference':
            self.model_name = config.lightweight
            self.init_model_flag = True
        else:
            self.model_name = config.golden
        self.model = None
        self._tmp_weight_path = os.path.join(
            str(get_models_dir()),
            f"tmp_model_{self.model_name}.pth",
        )
        self.load_model()
        self.threshold_low = 0.2
        self.threshold_high = 0.6

    def load_model(self):
        explicit_weights_path = getattr(self.config, "weights_path", None)
        self.model = build_detection_model(
            self.model_name,
            pretrained=True,
            device=device,
            weights_path=explicit_weights_path,
        )

        self.model.to(device)
        if self.init_model_flag:
            self.init_model()
        self.model.eval()
        get_split_runtime_model(self.model).eval()

    def init_model(self):
        logger.debug("init_model")
        for param in self.model.parameters():
            param.requires_grad = False

        if model_has_roi_heads(self.model_name):
            # Faster R-CNN / legacy path: only fine-tune roi_heads
            for param in self.model.roi_heads.parameters():
                param.requires_grad = True
        elif is_wrapper_model(self.model_name):
            # YOLO / DETR / RT-DETR wrappers manage their own fine-tuning;
            # for now unfreeze the last 20 % of parameters as a heuristic.
            all_params = list(self.model.parameters())
            trainable_start = int(len(all_params) * 0.8)
            for p in all_params[trainable_start:]:
                p.requires_grad = True
        else:
            # Other torchvision models (RetinaNet / SSD / FCOS): fine-tune head
            head_attrs = ['head', 'classification_head', 'regression_head']
            found = False
            for attr in head_attrs:
                if hasattr(self.model, attr):
                    for param in getattr(self.model, attr).parameters():
                        param.requires_grad = True
                    found = True
            if not found:
                # Fallback: unfreeze last 20 % of parameters
                all_params = list(self.model.parameters())
                trainable_start = int(len(all_params) * 0.8)
                for p in all_params[trainable_start:]:
                    p.requires_grad = True

        torch.save(self.model.state_dict(), self._tmp_weight_path)

    def retrain(self, path, select_index):

        # Wrapper models (YOLO/DETR/RT-DETR) don't support torchvision-style
        # forward(images, targets) → loss_dict training through this path.
        if is_wrapper_model(self.model_name):
            logger.warning(
                "[Retrain] {} ({}) does not support torchvision-style retraining. "
                "Use the model's native training API (e.g. ultralytics CLI).",
                self.model_name, get_model_family(self.model_name),
            )
            return

        tmp_model = build_detection_model(self.model_name, pretrained=False, device=device)
        state_dict = torch.load(self._tmp_weight_path, map_location=device)
        tmp_model.load_state_dict(state_dict)
        tmp_model.to(device)

        # Freeze backbone, unfreeze head — model-family aware
        for param in tmp_model.parameters():
            param.requires_grad = False
        if model_has_roi_heads(self.model_name):
            for param in tmp_model.roi_heads.parameters():
                param.requires_grad = True
        else:
            head_attrs = ['head', 'classification_head', 'regression_head', 'roi_heads']
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

        dataset = TrafficDataset(root=path, select_index = select_index)
        data_loader = DataLoader(dataset=dataset, batch_size=2, collate_fn=_collate_fn, )
        tr_metric = RetrainMetric()

        # 训练设置
        num_epoch = self.config.retrain.num_epoch
        trainable_params = [p for p in tmp_model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(trainable_params, lr=0.005, momentum=0.9,weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        for epoch in range(num_epoch):
            tmp_model.train()
            for images, targets in tr_metric.log_iter(epoch, num_epoch, data_loader):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = tmp_model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                tr_metric.update(loss_dict, losses)
            # Update the learning rate
            lr_scheduler.step()
        torch.save(tmp_model.state_dict(), self._tmp_weight_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        state_dict = torch.load(self._tmp_weight_path, map_location=device)
        with self.model_lock:
            self.model.load_state_dict(state_dict)
            self.model.eval()

    def model_evaluation(self,cache_path, select_index):
        map = []
        frame_path = os.path.join(cache_path, 'frames')
        annotation_path = os.path.join(cache_path, 'annotation.txt')
        annotations_f = pd.read_csv(annotation_path, header=None, names=annotation_cols)
        test_model = build_detection_model(self.model_name, pretrained=False, device=device)
        state_dict = torch.load(self._tmp_weight_path, map_location=device)
        test_model.load_state_dict(state_dict)
        test_model.to(device)
        test_model.eval()
        for _id in select_index:
            logger.debug(_id)
            path = os.path.join(frame_path, str(_id)+'.jpg')
            frame = cv2.imread(path)
            pred_boxes, pred_class, pred_score = self.get_model_prediction(frame, self.threshold_high, test_model)
            pred = {'labels':pred_class, 'boxes': pred_boxes, 'scores':pred_score}
            annos = annotations_f[annotations_f['frame_index'] == _id]
            target_boxes = []
            target_labels = []
            for _idx, _label in annos.iterrows():
                label = _label['target_id']
                if label != 0:
                    x_min = _label['bbox_x1']
                    y_min = _label['bbox_y1']
                    x_max = _label['bbox_x2']
                    y_max = _label['bbox_y2']
                    target_boxes.append([x_min, y_min, x_max, y_max])
                    target_labels.append(label)
            target = {'labels':target_labels, 'boxes': target_boxes}
            if pred['labels'] is not None:
                cal_map = calculate_map(target, pred, 0.5)
                map.append(cal_map)
        if len(map):
            map = np.mean(map)
        else:
            map = 0.0
        logger.debug("retrain {}".format(map))
        self.record_r.write("{}\n".format(map))
        self.record_r.flush()
        # pretrained_model
        map = []
        state_dict = torch.load("./model_management/pretrained.pth", map_location=device)
        test_model.load_state_dict(state_dict)
        test_model.to(device)
        test_model.eval()
        logger.debug("pretrained")
        for _id in select_index:
            logger.debug(_id)
            path = os.path.join(frame_path, str(_id)+'.jpg')
            frame = cv2.imread(path)
            pred_boxes, pred_class, pred_score = self.get_model_prediction(frame, self.threshold_high, test_model)
            pred = {'labels':pred_class, 'boxes': pred_boxes, 'scores':pred_score}
            annos = annotations_f[annotations_f['frame_index'] == _id]
            target_boxes = []
            target_labels = []
            for _idx, _label in annos.iterrows():
                label = _label['target_id']
                if label != 0:
                    x_min = _label['bbox_x1']
                    y_min = _label['bbox_y1']
                    x_max = _label['bbox_x2']
                    y_max = _label['bbox_y2']
                    target_boxes.append([x_min, y_min, x_max, y_max])
                    target_labels.append(label)
            target = {'labels':target_labels, 'boxes': target_boxes}
            if pred['labels'] is not None:
                cal_map = calculate_map(target, pred, 0.5)
                #logger.debug(cal_map)
                map.append(cal_map)
        if len(map):
            map = np.mean(map)
        else:
            map = 0.0
        logger.debug("pre {}".format(map))
        self.record_p.write("{}\n".format(map))
        self.record_p.flush()

    def prepare_splitter_input(self, img):
        return prepare_split_runtime_input(self.model, img, device=device)

    def build_split_sample_input(self, image_size=(224, 224)):
        return build_split_runtime_sample_input(self.model, image_size=image_size, device=device)

    def get_split_runtime_model(self):
        return get_split_runtime_model(self.model)

    def infer_sample(self, img, splitter=None) -> InferenceArtifacts:
        split_payload = None
        input_tensor_shape = None
        with self.model_lock:
            if splitter is not None:
                splitter_input = self.prepare_splitter_input(img)
                if isinstance(splitter_input, torch.Tensor):
                    input_tensor_shape = [int(dim) for dim in splitter_input.shape]
                elif (
                    isinstance(splitter_input, (list, tuple))
                    and splitter_input
                    and isinstance(splitter_input[0], torch.Tensor)
                ):
                    input_tensor_shape = [int(dim) for dim in splitter_input[0].shape]
                replayed, split_payload = splitter.replay_inference(
                    splitter_input, return_split_output=True,
                )
                replayed = postprocess_split_runtime_output(
                    self.model,
                    replayed,
                    threshold=self.threshold_low,
                    model_input=splitter_input,
                    orig_image=img,
                )
                pred_boxes, pred_class, pred_score = self._parse_prediction_output(
                    replayed, self.threshold_low,
                )
            else:
                pred_boxes, pred_class, pred_score = self.get_model_prediction(img, self.threshold_low)

        if pred_boxes is None or pred_score is None:
            return InferenceArtifacts(
                intermediate=split_payload,
                detection_boxes=[],
                detection_class=[],
                detection_score=[],
                confidence=0.0,
                input_tensor_shape=input_tensor_shape,
            )

        confidence = float(np.mean(pred_score)) if len(pred_score) else 0.0
        try:
            prediction_index = [pred_score.index(x) for x in pred_score if x > self.threshold_high][-1]
        except IndexError:
            detection_boxes = []
            detection_class = []
            detection_score = []
        else:
            detection_boxes = pred_boxes[:prediction_index + 1]
            detection_class = pred_class[:prediction_index + 1]
            detection_score = pred_score[:prediction_index + 1]

        return InferenceArtifacts(
            intermediate=split_payload,
            detection_boxes=detection_boxes,
            detection_class=detection_class,
            detection_score=detection_score,
            confidence=confidence,
            input_tensor_shape=input_tensor_shape,
        )

    def small_inference(self, img, splitter=None, return_split_payload=False):
        artifacts = self.infer_sample(img, splitter=splitter)
        if return_split_payload:
            return (
                None,
                artifacts.detection_boxes or None,
                artifacts.detection_class or None,
                artifacts.detection_score or None,
                artifacts.intermediate,
            )
        return (
            None,
            artifacts.detection_boxes or None,
            artifacts.detection_class or None,
            artifacts.detection_score or None,
        )


    def large_inference(self, img, threshold=None):
        if threshold is None:
            threshold = self.threshold_high
        pred_boxes, pred_class, pred_score = self.get_model_prediction(img, float(threshold))
        return pred_boxes, pred_class, pred_score

    def get_model_prediction(self, img, threshold, model=None):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        img = self._prepare_image_tensor(img)
        #get the inference result
        if model is None:
            res = self.model([img])
        else:
            res = model([img])
        return self._parse_prediction_output(res, threshold)

    def _prepare_image_tensor(self, img):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img)
        return img.to(device)

    def _parse_prediction_output(self, res, threshold):
        if isinstance(res, tuple):
            res = res[0]
        if isinstance(res, dict):
            res = [res]
        if not isinstance(res, (list, tuple)) or len(res) == 0:
            return None, None, None

        first = res[0]
        if not isinstance(first, dict):
            return None, None, None

        labels_t = first.get('labels')
        boxes_t = first.get('boxes')
        scores_t = first.get('scores')
        if labels_t is None or boxes_t is None or scores_t is None:
            return None, None, None

        prediction_class = labels_t.detach().cpu().tolist()
        prediction_boxes = boxes_t.detach().cpu().tolist()
        prediction_score = scores_t.detach().cpu().tolist()

        try:
            prediction_t = [prediction_score.index(x) for x in prediction_score if x > threshold][-1]
        except IndexError:
            return None, None, None
        pred_boxes = prediction_boxes[:prediction_t + 1]
        pred_class = prediction_class[:prediction_t + 1]
        pred_score = prediction_score[:prediction_t + 1]
        return pred_boxes, pred_class, pred_score





