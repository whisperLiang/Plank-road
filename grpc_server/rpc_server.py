import json

from loguru import logger

from edge.task import Task
from tools.convert_tool import base64_to_cv2
from grpc_server import message_transmission_pb2, message_transmission_pb2_grpc


class MessageTransmissionServicer(message_transmission_pb2_grpc.MessageTransmissionServicer):
    def __init__(self, local_queue, id, object_detection, queue_info=None, continual_learner=None):
        self.local_queue = local_queue
        self.id = id
        self.queue_info = queue_info
        self.object_detection = object_detection
        # CloudContinualLearner instance; None on edge-side servicers
        self.continual_learner = continual_learner

    def task_processor(self, request, context):
        logger.debug("task_processor")
        base64_frame = request.frame
        frame_shape = tuple(int(s) for s in request.new_shape[1:-1].split(","))
        frame = base64_to_cv2(base64_frame).reshape(frame_shape)
        raw_shape = tuple(int(s) for s in request.raw_shape[1:-1].split(","))

        task = Task(request.source_edge_id, request.frame_index, frame, float(request.start_time), raw_shape)
        task.other = True

        if request.part_result != "":
            part_result = eval(request.part_result)
            if len(request.part_result['boxes']) != 0:
                task.add_result(part_result['boxes'], part_result['labels'], part_result['scores'])

        if request.note == "edge process":
            task.edge_process = True


        self.local_queue.put(task, block=True)

        reply = message_transmission_pb2.MessageReply(
            destination_id=self.id,
            local_length=self.local_queue.qsize(),
            response='offload to {} successfully'.format(self.id)
        )

        return reply

    def frame_processor(self, request_iterator, context):
        res_list = []
        for request in request_iterator:
            base64_frame = request.frame
            frame_shape = tuple(int(s) for s in request.frame_shape[1:-1].split(","))
            frame = base64_to_cv2(base64_frame).reshape(frame_shape)
            index = request.frame_index
            pred_boxes, pred_class, pred_score = self.object_detection.large_inference(frame)
            res = {
                'index': index,
                'boxes': pred_boxes,
                'labels': pred_class,
                'scores': pred_score
            }
            res_list.append(res)
        reply = message_transmission_pb2.FrameReply(
            response=str(res_list),
            frame_shape=str(frame_shape),
        )
        logger.debug(reply)
        return reply


    def get_queue_info(self, request, context):
        self.queue_info['{}'.format(request.source_edge_id)] = request.local_length
        reply = message_transmission_pb2.InfoReply(
            destination_id=self.id,
            local_length=self.local_queue.qsize(),
        )
        return reply

    def train_model_request(self, request, context):
        """Cloud-side continual learning: label frames with the large model then
        fine-tune the lightweight edge model and return the updated weights."""
        logger.info(
            "train_model_request from edge_id={} cache_path={} num_epoch={}",
            request.edge_id, request.cache_path, request.num_epoch,
        )
        if self.continual_learner is None:
            logger.error("train_model_request: continual_learner not configured")
            return message_transmission_pb2.TrainReply(
                success=False, model_data="", message="continual_learner not configured"
            )
        try:
            frame_indices = json.loads(request.frame_indices)
            success, model_data, message = self.continual_learner.get_ground_truth_and_retrain(
                request.edge_id,
                frame_indices,
                request.cache_path,
                int(request.num_epoch),
            )
        except Exception as exc:
            logger.exception("train_model_request error: {}", exc)
            success, model_data, message = False, "", str(exc)

        return message_transmission_pb2.TrainReply(
            success=success, model_data=model_data, message=message
        )

    def split_train_request(self, request, context):
        """Split-learning continual learning (HSFL-style): annotate only drift
        frames with the large model, then train server-side model on all cached
        backbone features and return the updated state-dict."""
        logger.info(
            "split_train_request from edge_id={} cache_path={} num_epoch={}",
            request.edge_id, request.cache_path, request.num_epoch,
        )
        if self.continual_learner is None:
            logger.error("split_train_request: continual_learner not configured")
            return message_transmission_pb2.SplitTrainReply(
                success=False, model_data="", message="continual_learner not configured"
            )
        try:
            all_indices = json.loads(request.all_frame_indices)
            drift_indices = json.loads(request.drift_frame_indices)
            success, model_data, message = \
                self.continual_learner.get_ground_truth_and_split_retrain(
                    request.edge_id,
                    all_indices,
                    drift_indices,
                    request.cache_path,
                    int(request.num_epoch),
                )
        except Exception as exc:
            logger.exception("split_train_request error: {}", exc)
            success, model_data, message = False, "", str(exc)

        return message_transmission_pb2.SplitTrainReply(
            success=success, model_data=model_data, message=message
        )


