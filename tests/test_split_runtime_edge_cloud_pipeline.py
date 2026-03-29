from __future__ import annotations

import base64
import io
import json
import os
import queue
import threading
from concurrent import futures
from pathlib import Path
from types import SimpleNamespace

import cv2
import grpc
import numpy as np
import pytest
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

from edge.transmit import pack_training_payload
from cloud_server import CloudContinualLearner
from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc
from grpc_server.rpc_server import MessageTransmissionServicer
from model_management.model_zoo import (
    build_detection_model,
    build_model_sample_input,
    ensure_local_model_artifact,
)
from model_management.object_detection import Object_Detection
from model_management.payload import SplitPayload
from model_management.split_model_adapters import (
    build_split_training_loss,
    build_split_runtime_sample_input,
    get_split_runtime_model,
    postprocess_split_runtime_output,
    prepare_split_runtime_input,
)
from model_management.split_runtime import compare_outputs
from model_management.universal_model_split import (
    UniversalModelSplitter,
    save_split_feature_cache,
    universal_split_retrain,
)
from tools.grpc_options import grpc_message_options


REPO_ROOT = Path(r"D:\ProgramCode\Plank-road")
ROAD_VIDEO = REPO_ROOT / "video_data" / "road.mp4"


class TraceableFasterRCNNDetector(nn.Module):
    """Replay-friendly detector built from real Faster R-CNN backbone/RPN parts.

    The model keeps the public detection output format but avoids the heavy
    post-processing path that is not suitable for dependency-driven replay.
    """

    def __init__(self) -> None:
        super().__init__()
        detector = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None).eval()
        self.backbone = detector.backbone
        self.rpn_head = detector.rpn.head

    def _batchify(self, images):
        if isinstance(images, torch.Tensor):
            return images if images.dim() == 4 else images.unsqueeze(0)
        if isinstance(images, (list, tuple)):
            items = []
            for image in images:
                if image.dim() == 4:
                    items.extend([item for item in image])
                else:
                    items.append(image)
            return torch.stack(items, dim=0)
        raise TypeError(f"Unsupported input type: {type(images)!r}")

    def forward(self, images):
        x = self._batchify(images)
        features = self.backbone(x)
        feature_list = list(features.values()) if isinstance(features, dict) else list(features)
        objectness, bbox_deltas = self.rpn_head(feature_list)

        flat_scores = torch.cat([score.flatten(start_dim=1) for score in objectness], dim=1).sigmoid()
        flat_boxes = torch.cat(
            [delta.permute(0, 2, 3, 1).reshape(delta.shape[0], -1, 4) for delta in bbox_deltas],
            dim=1,
        )

        topk = min(12, flat_scores.shape[1])
        scores, indices = flat_scores.topk(topk, dim=1)
        picked = torch.gather(flat_boxes, 1, indices.unsqueeze(-1).expand(-1, -1, 4))

        centers = picked[..., :2].abs() * 8.0
        spans = picked[..., 2:].abs() * 8.0 + 1.0
        boxes = torch.cat([centers, centers + spans], dim=-1)
        labels = torch.ones((x.shape[0], topk), dtype=torch.int64, device=x.device)
        return [{"boxes": boxes[i], "labels": labels[i], "scores": scores[i]} for i in range(x.shape[0])]


def _read_video_frames(*, count: int, size: tuple[int, int]) -> list:
    assert ROAD_VIDEO.exists(), ROAD_VIDEO
    cap = cv2.VideoCapture(str(ROAD_VIDEO))
    frames = []
    try:
        while len(frames) < count:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(frame, size))
    finally:
        cap.release()
    assert len(frames) == count
    return frames


def _make_detector_with_model(model: nn.Module) -> Object_Detection:
    detector = Object_Detection.__new__(Object_Detection)
    detector.type = "small inference"
    detector.config = SimpleNamespace(retrain=SimpleNamespace(num_epoch=1))
    detector.model = model
    detector.model_lock = threading.Lock()
    detector.threshold_low = 0.1
    detector.threshold_high = 0.2
    return detector


def _build_traceable_sample_input(detector: Object_Detection):
    torch.manual_seed(0)
    return detector.build_split_sample_input((96, 96))


def _choose_trainable_candidate(splitter: UniversalModelSplitter):
    for candidate in splitter.candidates:
        report = splitter.validate_candidate(candidate)
        if report["success"] and splitter.get_tail_trainable_params(candidate):
            return candidate
    raise AssertionError("No validated trainable candidate found.")


def _build_traceable_pipeline():
    model = TraceableFasterRCNNDetector().eval()
    detector = _make_detector_with_model(model)
    splitter = UniversalModelSplitter(device="cpu")
    sample_input = _build_traceable_sample_input(detector)
    splitter.trace(model, sample_input)
    candidate = _choose_trainable_candidate(splitter)
    splitter.split(candidate=candidate)
    return detector, splitter, candidate


def _build_traceable_pipeline_for_boundary(boundary_tensor_labels):
    model = TraceableFasterRCNNDetector().eval()
    detector = _make_detector_with_model(model)
    splitter = UniversalModelSplitter(device="cpu")
    sample_input = _build_traceable_sample_input(detector)
    splitter.trace(model, sample_input)
    candidate = splitter.split(boundary_tensor_labels=boundary_tensor_labels)
    return detector, splitter, candidate


def _sample_traceable_trainable_candidate_boundaries(*, count: int, seed: int) -> list[list[str]]:
    model = TraceableFasterRCNNDetector().eval()
    detector = _make_detector_with_model(model)
    splitter = UniversalModelSplitter(device="cpu")
    splitter.trace(model, _build_traceable_sample_input(detector))
    sampled = splitter.sample_candidates(
        sample_size=count,
        seed=seed,
        require_validated=True,
        require_trainable=True,
        max_candidates=12,
    )
    assert sampled
    return [list(candidate.boundary_tensor_labels) for candidate in sampled]


def _frame_to_split_input(detector: Object_Detection, frame):
    return detector.prepare_splitter_input(frame)


def _tensor_dict(det_output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in det_output.items()}


def _perturbed_target(det_output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    target = _tensor_dict(det_output)
    if "boxes" in target:
        target["boxes"] = target["boxes"] + 0.25
    if "scores" in target:
        target["scores"] = (target["scores"] * 0.9).clamp(0.0, 1.0)
    return target


def _more_aggressive_target(det_output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    target = _tensor_dict(det_output)
    if "boxes" in target:
        target["boxes"] = target["boxes"] + 1.0
    if "scores" in target:
        target["scores"] = (target["scores"] * 0.8).clamp(0.0, 1.0)
    return target


def _changed_state_names(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> list[str]:
    return [
        name
        for name, tensor in after.items()
        if name in before and not torch.allclose(tensor, before[name])
    ]


def _build_original_splitter(model_name: str, sample_input, *, weights_path: str | None = None):
    model = build_detection_model(model_name, pretrained=False, device="cpu", weights_path=weights_path).eval()
    splitter = UniversalModelSplitter(device="cpu")
    splitter.trace(model, sample_input)
    candidate = _choose_trainable_candidate(splitter)
    splitter.split(candidate=candidate)
    return model, splitter, candidate


def test_edge_replay_outputs_intermediate_features_and_final_detections():
    detector, splitter, candidate = _build_traceable_pipeline()
    frame = _read_video_frames(count=1, size=(96, 96))[0]

    runtime_input = _frame_to_split_input(detector, frame)
    replayed, payload = splitter.replay_inference(runtime_input, return_split_output=True, candidate=candidate)
    expected = detector.model(runtime_input)
    ok, max_diff = compare_outputs(expected, replayed)
    assert ok, max_diff

    _, boxes, labels, scores, detector_payload = detector.small_inference(
        frame,
        splitter=splitter,
        return_split_payload=True,
    )
    assert isinstance(detector_payload, SplitPayload)
    assert detector_payload.candidate_id == candidate.candidate_id
    assert detector_payload.boundary_tensor_labels
    assert boxes is not None and len(boxes) > 0
    assert labels is not None and len(labels) == len(boxes)
    assert scores is not None and len(scores) == len(boxes)


def test_server_batch_training_updates_tail_weights(tmp_path):
    detector, splitter, candidate = _build_traceable_pipeline()
    frames = _read_video_frames(count=3, size=(96, 96))

    frame_indices = []
    targets = {}
    for index, frame in enumerate(frames):
        runtime_input = _frame_to_split_input(detector, frame)
        payload = splitter.edge_forward(runtime_input, candidate=candidate)
        targets[index] = _perturbed_target(detector.model(runtime_input)[0])
        save_split_feature_cache(
            str(tmp_path),
            index,
            payload.detach(),
            pseudo_boxes=targets[index]["boxes"].tolist(),
            pseudo_labels=targets[index]["labels"].tolist(),
            pseudo_scores=targets[index]["scores"].tolist(),
            extra_metadata={"input_tensor_shape": list(runtime_input[0].shape)},
        )
        frame_indices.append(index)

    splitter.freeze_head(candidate)
    splitter.unfreeze_tail(candidate)
    tail_params = splitter.get_tail_trainable_params(candidate)
    tail_param_ids = {id(param) for param in tail_params}
    before = {
        name: param.detach().clone()
        for name, param in detector.model.named_parameters()
        if id(param) in tail_param_ids
    }

    losses = universal_split_retrain(
        model=detector.model,
        sample_input=_build_traceable_sample_input(detector),
        cache_path=str(tmp_path),
        all_indices=frame_indices,
        gt_annotations=targets,
        device="cpu",
        num_epoch=1,
    )

    changed = [
        name
        for name, param in detector.model.named_parameters()
        if name in before and not torch.allclose(param.detach(), before[name])
    ]
    assert losses and losses[0] >= 0.0
    assert changed


class _GrpcSplitLearner:
    def __init__(self, target_map):
        self.model = TraceableFasterRCNNDetector().eval()
        self.target_map = target_map

    def get_ground_truth_and_split_retrain(
        self,
        edge_id: int,
        all_frame_indices: list[int],
        drift_frame_indices: list[int],
        cache_path: str,
        num_epoch: int = 1,
    ):
        universal_split_retrain(
            model=self.model,
            sample_input=[torch.rand(3, 96, 96)],
            cache_path=cache_path,
            all_indices=all_frame_indices,
            gt_annotations={index: self.target_map[index] for index in all_frame_indices},
            device="cpu",
            num_epoch=max(1, int(num_epoch)),
        )
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        return True, base64.b64encode(buffer.getvalue()).decode("utf-8"), "ok"


def _run_grpc_split_train_roundtrip(*, client_cache: Path, server_cache: Path, frame_indices, target_map):
    learner = _GrpcSplitLearner(target_map)
    initial_state = {name: tensor.detach().clone() for name, tensor in learner.model.state_dict().items()}

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=1),
        options=grpc_message_options(),
    )
    message_transmission_pb2_grpc.add_MessageTransmissionServicer_to_server(
        MessageTransmissionServicer(
            local_queue=queue.Queue(),
            id=0,
            object_detection=SimpleNamespace(large_inference=lambda frame: ([], [], [])),
            queue_info={},
            continual_learner=learner,
        ),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()

    try:
        with grpc.insecure_channel(
            f"127.0.0.1:{port}",
            options=grpc_message_options(),
        ) as channel:
            stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
            reply = stub.split_train_request(
                message_transmission_pb2.SplitTrainRequest(
                    edge_id=1,
                    all_frame_indices=json.dumps(frame_indices),
                    drift_frame_indices=json.dumps(frame_indices[:1]),
                    cache_path=str(server_cache),
                    num_epoch=1,
                    payload_zip=pack_training_payload(str(client_cache), frame_indices, frame_indices[:1]),
                )
            )
    finally:
        server.stop(None)

    returned_state = torch.load(io.BytesIO(base64.b64decode(reply.model_data)), map_location="cpu", weights_only=False)
    changed = [name for name, tensor in returned_state.items() if not torch.allclose(tensor, initial_state[name])]
    return reply, returned_state, changed


def test_feature_transfer_and_weight_download_over_grpc(tmp_path):
    detector, splitter, candidate = _build_traceable_pipeline()
    frames = _read_video_frames(count=3, size=(96, 96))

    client_cache = tmp_path / "client_cache"
    server_cache = tmp_path / "server_cache"
    client_cache.mkdir()
    server_cache.mkdir()

    frame_indices = []
    target_map = {}
    for index, frame in enumerate(frames):
        runtime_input = _frame_to_split_input(detector, frame)
        payload = splitter.edge_forward(runtime_input, candidate=candidate)
        target_map[index] = _perturbed_target(detector.model(runtime_input)[0])
        save_split_feature_cache(
            str(client_cache),
            index,
            payload.detach(),
            pseudo_boxes=target_map[index]["boxes"].tolist(),
            pseudo_labels=target_map[index]["labels"].tolist(),
            pseudo_scores=target_map[index]["scores"].tolist(),
            extra_metadata={"input_tensor_shape": list(runtime_input[0].shape)},
        )
        frame_indices.append(index)

    reply, returned_state, changed = _run_grpc_split_train_roundtrip(
        client_cache=client_cache,
        server_cache=server_cache,
        frame_indices=frame_indices,
        target_map=target_map,
    )
    assert reply.success, reply.message
    assert (server_cache / "features" / "0.pt").exists()
    assert (server_cache / "features" / "split_meta.json").exists()
    assert changed

    reloaded_model = TraceableFasterRCNNDetector().eval()
    reloaded_model.load_state_dict(returned_state)
    output = reloaded_model([torch.rand(3, 96, 96)])[0]
    assert output["boxes"].shape[-1] == 4
    assert output["scores"].numel() > 0


def test_real_yolov8_runs_end_to_end_on_road_video():
    model = build_detection_model(
        "yolov8n",
        pretrained=True,
        device="cpu",
    )
    frames = _read_video_frames(count=3, size=(640, 360))
    to_tensor = transforms.ToTensor()

    total_detections = 0
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = model([to_tensor(rgb)])[0]
        assert set(output.keys()) == {"boxes", "labels", "scores"}
        assert output["boxes"].shape[-1] == 4
        total_detections += int(output["boxes"].shape[0])

    assert total_detections > 0


def test_real_yolov8_split_runtime_pads_non_stride_aligned_inputs():
    model = build_detection_model(
        "yolov8n",
        pretrained=True,
        device="cpu",
    ).eval()
    core_model = get_split_runtime_model(model).eval()

    sample_input = build_split_runtime_sample_input(
        model,
        image_size=(360, 640),
        device="cpu",
    )
    runtime_input = prepare_split_runtime_input(
        model,
        np.zeros((360, 640, 3), dtype=np.uint8),
        device="cpu",
    )

    assert tuple(sample_input.shape[-2:]) == (384, 640)
    assert tuple(runtime_input.shape[-2:]) == (384, 640)

    output = core_model(runtime_input)
    assert isinstance(output, (list, tuple))


def test_real_yolov8_split_runtime_matches_wrapper_output():
    model = build_detection_model(
        "yolov8n",
        pretrained=True,
        device="cpu",
    ).eval()
    frame = _read_video_frames(count=1, size=(640, 360))[0]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transforms.ToTensor()(rgb)

    expected = model([input_tensor])[0]
    runtime_input = prepare_split_runtime_input(model, frame, device="cpu")
    raw_outputs = get_split_runtime_model(model).eval()(runtime_input)
    replayed = postprocess_split_runtime_output(
        model,
        raw_outputs,
        threshold=0.2,
        model_input=runtime_input,
        orig_image=frame,
    )[0]

    ok, max_diff = compare_outputs([expected], [replayed])
    assert ok, max_diff


def test_real_yolov8_fixed_split_recovers_trainable_tail_from_parameter_refs():
    model = build_detection_model(
        "yolov8n",
        pretrained=True,
        device="cpu",
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    all_params = list(model.parameters())
    trainable_start = int(len(all_params) * 0.8)
    for parameter in all_params[trainable_start:]:
        parameter.requires_grad = True

    splitter = UniversalModelSplitter(device="cpu")
    sample_input = build_split_runtime_sample_input(
        model,
        image_size=(1080, 1920),
        device="cpu",
    )
    splitter.trace(get_split_runtime_model(model).eval(), sample_input)
    candidates = splitter.enumerate_candidates(max_candidates=24, max_boundary_count=8, max_payload_bytes=32 * 1024 * 1024)

    trainable_labels = [
        label
        for label in splitter.graph.relevant_labels
        if splitter.graph.nodes[label].has_trainable_params
    ]
    assert trainable_labels
    assert any(candidate.is_trainable_tail for candidate in candidates)


def test_original_fasterrcnn_split_loss_makes_trainable_candidates_valid_and_trainable():
    model = build_detection_model(
        "fasterrcnn_mobilenet_v3_large_320_fpn",
        pretrained=False,
        device="cpu",
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.roi_heads.parameters():
        parameter.requires_grad = True

    splitter = UniversalModelSplitter(device="cpu")
    splitter.trainability_loss_fn = build_split_training_loss(model)
    sample_input = build_split_runtime_sample_input(model, image_size=(96, 96), device="cpu")
    splitter.trace(get_split_runtime_model(model).eval(), sample_input)
    candidates = splitter.enumerate_candidates(max_candidates=24, max_boundary_count=8, max_payload_bytes=32 * 1024 * 1024)

    chosen = None
    report = None
    for candidate in candidates:
        if not candidate.is_trainable_tail:
            continue
        candidate_report = splitter.validate_candidate(candidate)
        if candidate_report["success"] and candidate_report["tail_trainability"]:
            chosen = candidate
            report = candidate_report
            break

    assert chosen is not None
    assert report is not None and report["tail_trainability"]

    splitter.freeze_head(chosen)
    splitter.unfreeze_tail(chosen)
    optimizer = torch.optim.Adam(splitter.get_tail_trainable_params(chosen), lr=1e-3)
    payload = splitter.edge_forward(sample_input, candidate=chosen)
    before = splitter.get_tail_state_dict(chosen)
    _, loss = splitter.cloud_train_step(
        payload,
        targets={"boxes": [], "labels": [], "scores": []},
        loss_fn=build_split_training_loss(model),
        optimizer=optimizer,
        candidate=chosen,
    )
    after = splitter.get_tail_state_dict(chosen)

    assert bool(torch.isfinite(loss).item())
    assert _changed_state_names(before, after)


def test_random_trainable_candidates_replay_and_split_retrain(tmp_path):
    candidate_boundaries = _sample_traceable_trainable_candidate_boundaries(count=3, seed=19)
    frames = _read_video_frames(count=2, size=(96, 96))

    for candidate_index, boundary_tensor_labels in enumerate(candidate_boundaries):
        cache_path = tmp_path / f"candidate_{candidate_index}"
        detector, splitter, candidate = _build_traceable_pipeline_for_boundary(boundary_tensor_labels)

        frame_indices = []
        targets = {}
        for frame_index, frame in enumerate(frames):
            runtime_input = _frame_to_split_input(detector, frame)
            replayed, payload = splitter.replay_inference(
                runtime_input,
                return_split_output=True,
                candidate=candidate,
            )
            expected = detector.model(runtime_input)
            ok, max_diff = compare_outputs(expected, replayed)
            assert ok, (candidate.candidate_id, max_diff)

            targets[frame_index] = _more_aggressive_target(expected[0])
            save_split_feature_cache(
                str(cache_path),
                frame_index,
                payload.detach(),
                pseudo_boxes=targets[frame_index]["boxes"].tolist(),
                pseudo_labels=targets[frame_index]["labels"].tolist(),
                pseudo_scores=targets[frame_index]["scores"].tolist(),
                extra_metadata={"input_tensor_shape": list(runtime_input[0].shape)},
            )
            frame_indices.append(frame_index)

        before = splitter.get_tail_state_dict(candidate)

        losses = universal_split_retrain(
            model=detector.model,
            sample_input=_build_traceable_sample_input(detector),
            cache_path=str(cache_path),
            all_indices=frame_indices,
            gt_annotations=targets,
            device="cpu",
            num_epoch=2,
        )
        changed = [
            name
            for name, tensor in detector.model.state_dict().items()
            if name in before and not torch.allclose(tensor.detach().cpu(), before[name])
        ]
        assert losses and all(torch.isfinite(torch.tensor(losses)).tolist())
        assert changed, candidate.candidate_id


def test_random_trainable_candidates_transfer_and_weight_download_over_grpc(tmp_path):
    candidate_boundaries = _sample_traceable_trainable_candidate_boundaries(count=2, seed=29)
    frames = _read_video_frames(count=2, size=(96, 96))

    for candidate_index, boundary_tensor_labels in enumerate(candidate_boundaries):
        detector, splitter, candidate = _build_traceable_pipeline_for_boundary(boundary_tensor_labels)
        client_cache = tmp_path / f"client_cache_{candidate_index}"
        server_cache = tmp_path / f"server_cache_{candidate_index}"
        client_cache.mkdir()
        server_cache.mkdir()

        frame_indices = []
        target_map = {}
        for frame_index, frame in enumerate(frames):
            runtime_input = _frame_to_split_input(detector, frame)
            payload = splitter.edge_forward(runtime_input, candidate=candidate)
            target_map[frame_index] = _more_aggressive_target(detector.model(runtime_input)[0])
            save_split_feature_cache(
                str(client_cache),
                frame_index,
                payload.detach(),
                pseudo_boxes=target_map[frame_index]["boxes"].tolist(),
                pseudo_labels=target_map[frame_index]["labels"].tolist(),
                pseudo_scores=target_map[frame_index]["scores"].tolist(),
                extra_metadata={"input_tensor_shape": list(runtime_input[0].shape)},
            )
            frame_indices.append(frame_index)

        reply, returned_state, changed = _run_grpc_split_train_roundtrip(
            client_cache=client_cache,
            server_cache=server_cache,
            frame_indices=frame_indices,
            target_map=target_map,
        )
        assert reply.success, (candidate.candidate_id, reply.message)
        assert (server_cache / "features" / "0.pt").exists()
        assert (server_cache / "features" / "split_meta.json").exists()
        assert changed, candidate.candidate_id

        reloaded_model = TraceableFasterRCNNDetector().eval()
        reloaded_model.load_state_dict(returned_state)
        output = reloaded_model([torch.rand(3, 96, 96)])[0]
        assert output["boxes"].shape[-1] == 4
        assert output["scores"].numel() > 0


def test_original_ssd_service_split_retrain_updates_weights(tmp_path):
    model_name = "ssdlite320_mobilenet_v3_large"
    sample_input = build_model_sample_input(model_name, image_size=(96, 96), device="cpu")
    model, splitter, candidate = _build_original_splitter(model_name, sample_input)

    targets = {}
    frame_indices = []
    for index in range(3):
        torch.manual_seed(200 + index)
        runtime_input = [torch.rand(3, 96, 96)]
        payload = splitter.edge_forward(runtime_input, candidate=candidate)
        targets[index] = _perturbed_target(model(runtime_input)[0])
        save_split_feature_cache(
            str(tmp_path),
            index,
            payload.detach(),
            pseudo_boxes=targets[index]["boxes"].tolist(),
            pseudo_labels=targets[index]["labels"].tolist(),
            pseudo_scores=targets[index]["scores"].tolist(),
            extra_metadata={"input_tensor_shape": list(runtime_input[0].shape)},
        )
        frame_indices.append(index)

    cfg = SimpleNamespace(
        edge_model_name=model_name,
        continual_learning=SimpleNamespace(num_epoch=1),
        das=SimpleNamespace(enabled=False, bn_only=False, probe_samples=2),
    )
    learner = CloudContinualLearner(cfg, large_object_detection=SimpleNamespace(large_inference=lambda frame: ([], [], [])))
    learner.device = torch.device("cpu")
    learner.weight_folder = str(tmp_path / "weights")
    Path(learner.weight_folder).mkdir(exist_ok=True)
    torch.save(model.state_dict(), Path(learner.weight_folder) / "tmp_edge_model.pth")

    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    returned_state = torch.load(
        io.BytesIO(learner._split_retrain_edge_model(str(tmp_path), frame_indices, targets, num_epoch=1)),
        map_location="cpu",
        weights_only=False,
    )
    changed = _changed_state_names(before, returned_state)
    assert changed


def test_cloud_loader_uses_builder_for_official_yolo_weights(tmp_path, monkeypatch):
    cfg = SimpleNamespace(
        edge_model_name="yolov8n",
        continual_learning=SimpleNamespace(num_epoch=1),
        das=SimpleNamespace(enabled=False, bn_only=False, probe_samples=2),
    )
    learner = CloudContinualLearner(
        cfg,
        large_object_detection=SimpleNamespace(large_inference=lambda frame: ([], [], [])),
    )
    learner.device = torch.device("cpu")
    learner.weight_folder = str(tmp_path / "weights")
    Path(learner.weight_folder).mkdir(exist_ok=True)

    calls = []

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(2, 2)

    def fake_build_detection_model(name, pretrained, device, **kwargs):
        calls.append((name, pretrained, str(device)))
        return DummyModel().eval()

    def fail_on_torch_load(*args, **kwargs):
        raise AssertionError("Official YOLO fallback should not call torch.load directly.")

    monkeypatch.setattr("model_management.model_zoo.build_detection_model", fake_build_detection_model)
    monkeypatch.setattr("cloud_server.torch.load", fail_on_torch_load)

    model = learner._load_edge_training_model()

    assert isinstance(model, DummyModel)
    assert calls == [("yolov8n", True, "cpu")]


def test_original_fasterrcnn_service_split_retrain_recovers_cached_candidate_by_boundary_labels(tmp_path):
    if not ROAD_VIDEO.exists():
        pytest.skip(f"Missing video: {ROAD_VIDEO}")

    frames = _read_video_frames(count=2, size=(320, 192))
    runtime_inputs = [
        [transforms.ToTensor()(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))]
        for frame in frames
    ]
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(exist_ok=True)
    for index, frame in enumerate(frames):
        cv2.imwrite(str(frame_dir / f"{index}.jpg"), frame)
    model_name = "fasterrcnn_mobilenet_v3_large_320_fpn"
    checkpoint_path = ensure_local_model_artifact(model_name)
    model, splitter, candidate = _build_original_splitter(
        model_name,
        runtime_inputs[0],
        weights_path=str(checkpoint_path),
    )

    # Re-tracing with the service-side default sample input changes the frontier ordering,
    # so the cached candidate must be recovered by boundary labels rather than raw candidate_id.
    service_trace = UniversalModelSplitter(device="cpu")
    service_trace.trace(
        build_detection_model(model_name, pretrained=False, device="cpu", weights_path=str(checkpoint_path)).eval(),
        build_model_sample_input(model_name, image_size=(224, 224), device="cpu"),
    )
    service_trace.enumerate_candidates(max_candidates=24)
    matched = [
        item
        for item in service_trace.candidates
        if item.boundary_tensor_labels == candidate.boundary_tensor_labels
    ]
    assert matched
    assert all(item.candidate_id != candidate.candidate_id for item in matched)

    targets = {}
    frame_indices = []
    for index, runtime_input in enumerate(runtime_inputs):
        payload = splitter.edge_forward(runtime_input, candidate=candidate)
        targets[index] = _more_aggressive_target(model(runtime_input)[0])
        save_split_feature_cache(
            str(tmp_path),
            index,
            payload.detach(),
            pseudo_boxes=targets[index]["boxes"].tolist(),
            pseudo_labels=targets[index]["labels"].tolist(),
            pseudo_scores=targets[index]["scores"].tolist(),
            extra_metadata={"input_tensor_shape": list(runtime_input[0].shape)},
        )
        frame_indices.append(index)

    cfg = SimpleNamespace(
        edge_model_name=model_name,
        continual_learning=SimpleNamespace(num_epoch=1),
        das=SimpleNamespace(enabled=False, bn_only=False, probe_samples=2),
    )
    learner = CloudContinualLearner(cfg, large_object_detection=SimpleNamespace(large_inference=lambda frame: ([], [], [])))
    learner.device = torch.device("cpu")
    learner.weight_folder = str(tmp_path / "weights")
    Path(learner.weight_folder).mkdir(exist_ok=True)
    torch.save(model.state_dict(), Path(learner.weight_folder) / "tmp_edge_model.pth")

    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    returned_state = torch.load(
        io.BytesIO(learner._split_retrain_edge_model(str(tmp_path), frame_indices, targets, num_epoch=1)),
        map_location="cpu",
        weights_only=False,
    )
    changed = _changed_state_names(before, returned_state)
    assert changed
