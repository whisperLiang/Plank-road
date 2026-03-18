import json
import os

import cv2
import grpc
from loguru import logger

from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc
from tools.convert_tool import cv2_to_base64
import zipfile
import io

def pack_training_payload(cache_path, all_frame_indices, drift_frame_indices=None):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        if drift_frame_indices is not None:
            # Split learning mode
            meta_path = os.path.join(cache_path, "features", "split_meta.json")
            if os.path.exists(meta_path):
                zf.write(meta_path, arcname="features/split_meta.json")
            for idx in all_frame_indices:
                feat_path = os.path.join(cache_path, "features", f"{idx}.pt")
                if os.path.exists(feat_path):
                    zf.write(feat_path, arcname=f"features/{idx}.pt")
            for idx in drift_frame_indices:
                frame_path = os.path.join(cache_path, "frames", f"{idx}.jpg")
                if os.path.exists(frame_path):
                    zf.write(frame_path, arcname=f"frames/{idx}.jpg")
        else:
            # Full frame train mode
            for idx in all_frame_indices:
                frame_path = os.path.join(cache_path, "frames", f"{idx}.jpg")
                if os.path.exists(frame_path):
                    zf.write(frame_path, arcname=f"frames/{idx}.jpg")
    return buf.getvalue()



# send to cloud, get ground truth
def get_cloud_target(server_ip, select_index, cache_path):

    def requset_stream():
        for index in select_index:
            path = os.path.join(cache_path, 'frames', '{}.jpg'.format(index))
            logger.debug(path)
            frame = cv2.imread(path)
            encoded_image = cv2_to_base64(frame)
            frame_request = message_transmission_pb2.FrameRequest(
                frame=encoded_image,
                frame_shape=str(frame.shape),
                frame_index= index,
            )
            yield frame_request

    try:
        channel = grpc.insecure_channel(server_ip, options=[('grpc.max_send_message_length', 1024 * 1024 * 512), ('grpc.max_receive_message_length', 1024 * 1024 * 512)])
        stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
        res = stub.frame_processor(requset_stream())
        result_dict = eval(res.response)
    except Exception as e:
        logger.exception("the cloud can not reply, {}".format(e))
    else:
        logger.debug("res{}".format(result_dict))
    return result_dict



import socket

def is_network_connected(address):
    ip, port = address.split(':')[0], int(address.split(':')[1])
    try:
        socket.create_connection((ip, port), timeout=1)
        return True
    except OSError:
        return False


def request_cloud_training(server_ip, edge_id, frame_indices, cache_path, num_epoch=0):
    """Send selected frame indices to the cloud for GT annotation and edge-model
    fine-tuning.  Returns ``(success, model_data_b64, message)``.

    Parameters
    ----------
    server_ip : str
        gRPC server address, e.g. ``"192.168.1.1:50051"``.
    edge_id : int
        Identifier of this edge node.
    frame_indices : list[int]
        Frame indices (relative to ``cache_path``) chosen for retraining.
    cache_path : str
        Absolute path to the local frame cache directory shared with the cloud
        (or accessible by both).
    num_epoch : int
        Number of fine-tuning epochs; 0 lets the cloud use its configured default.

    Returns
    -------
    tuple[bool, str, str]
        ``(success, base64_model_state_dict, message)``
    """
    try:
        channel = grpc.insecure_channel(server_ip, options=[('grpc.max_send_message_length', 1024 * 1024 * 512), ('grpc.max_receive_message_length', 1024 * 1024 * 512)])
        stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
        req = message_transmission_pb2.TrainRequest(
            edge_id=int(edge_id),
            frame_indices=json.dumps(frame_indices),
            cache_path=cache_path,
            num_epoch=int(num_epoch),
            payload_zip=pack_training_payload(cache_path, frame_indices),
        )
        reply = stub.train_model_request(req)
        return reply.success, reply.model_data, reply.message
    except Exception as exc:
        logger.exception("request_cloud_training failed: {}", exc)
        return False, "", str(exc)


def request_cloud_split_training(
    server_ip, edge_id, all_frame_indices, drift_frame_indices,
    cache_path, num_epoch=0,
):
    """Send frame indices and drift info to cloud for **split-learning**
    continual learning.

    The cloud will:
      1. Annotate **only** drift frames with the large model.
      2. Train the server-side model (rpn + roi_heads) on **all** cached
         backbone features (using pseudo-labels for non-drift frames).
      3. Return the updated edge-model state-dict.

    Parameters
    ----------
    server_ip : str
        gRPC server address.
    edge_id : int
    all_frame_indices : list[int]
        Every frame index that has cached backbone features.
    drift_frame_indices : list[int]
        Subset of *all_frame_indices* where drift was detected.
    cache_path : str
        Shared cache directory containing ``features/`` and ``frames/`` dirs.
    num_epoch : int
        Training epochs; 0 → cloud default.

    Returns
    -------
    tuple[bool, str, str]
        ``(success, base64_model_state_dict, message)``
    """
    try:
        channel = grpc.insecure_channel(server_ip, options=[('grpc.max_send_message_length', 1024 * 1024 * 512), ('grpc.max_receive_message_length', 1024 * 1024 * 512)])
        stub = message_transmission_pb2_grpc.MessageTransmissionStub(channel)
        req = message_transmission_pb2.SplitTrainRequest(
            edge_id=int(edge_id),
            all_frame_indices=json.dumps(all_frame_indices),
            drift_frame_indices=json.dumps(drift_frame_indices or []),
            cache_path=cache_path,
            num_epoch=int(num_epoch),
            payload_zip=pack_training_payload(cache_path, all_frame_indices, drift_frame_indices),
        )
        reply = stub.split_train_request(req)
        return reply.success, reply.model_data, reply.message
    except Exception as exc:
        logger.exception("request_cloud_split_training failed: {}", exc)
        return False, "", str(exc)
