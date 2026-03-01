from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class MessageRequest(_message.Message):
    __slots__ = ("source_edge_id", "frame_index", "start_time", "frame", "part_result", "raw_shape", "new_shape", "ref_list", "note")
    SOURCE_EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    FRAME_INDEX_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    PART_RESULT_FIELD_NUMBER: _ClassVar[int]
    RAW_SHAPE_FIELD_NUMBER: _ClassVar[int]
    NEW_SHAPE_FIELD_NUMBER: _ClassVar[int]
    REF_LIST_FIELD_NUMBER: _ClassVar[int]
    NOTE_FIELD_NUMBER: _ClassVar[int]
    source_edge_id: int
    frame_index: int
    start_time: str
    frame: str
    part_result: str
    raw_shape: str
    new_shape: str
    ref_list: str
    note: str
    def __init__(self, source_edge_id: _Optional[int] = ..., frame_index: _Optional[int] = ..., start_time: _Optional[str] = ..., frame: _Optional[str] = ..., part_result: _Optional[str] = ..., raw_shape: _Optional[str] = ..., new_shape: _Optional[str] = ..., ref_list: _Optional[str] = ..., note: _Optional[str] = ...) -> None: ...

class MessageReply(_message.Message):
    __slots__ = ("destination_id", "local_length", "response")
    DESTINATION_ID_FIELD_NUMBER: _ClassVar[int]
    LOCAL_LENGTH_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    destination_id: int
    local_length: int
    response: str
    def __init__(self, destination_id: _Optional[int] = ..., local_length: _Optional[int] = ..., response: _Optional[str] = ...) -> None: ...

class FrameRequest(_message.Message):
    __slots__ = ("frame", "frame_shape", "frame_index")
    FRAME_FIELD_NUMBER: _ClassVar[int]
    FRAME_SHAPE_FIELD_NUMBER: _ClassVar[int]
    FRAME_INDEX_FIELD_NUMBER: _ClassVar[int]
    frame: str
    frame_shape: str
    frame_index: int
    def __init__(self, frame: _Optional[str] = ..., frame_shape: _Optional[str] = ..., frame_index: _Optional[int] = ...) -> None: ...

class FrameReply(_message.Message):
    __slots__ = ("response", "frame_shape")
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    FRAME_SHAPE_FIELD_NUMBER: _ClassVar[int]
    response: str
    frame_shape: str
    def __init__(self, response: _Optional[str] = ..., frame_shape: _Optional[str] = ...) -> None: ...

class InfoRequest(_message.Message):
    __slots__ = ("source_edge_id", "local_length")
    SOURCE_EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    LOCAL_LENGTH_FIELD_NUMBER: _ClassVar[int]
    source_edge_id: int
    local_length: int
    def __init__(self, source_edge_id: _Optional[int] = ..., local_length: _Optional[int] = ...) -> None: ...

class InfoReply(_message.Message):
    __slots__ = ("destination_id", "local_length")
    DESTINATION_ID_FIELD_NUMBER: _ClassVar[int]
    LOCAL_LENGTH_FIELD_NUMBER: _ClassVar[int]
    destination_id: int
    local_length: int
    def __init__(self, destination_id: _Optional[int] = ..., local_length: _Optional[int] = ...) -> None: ...

class TrainRequest(_message.Message):
    __slots__ = ("edge_id", "frame_indices", "cache_path", "num_epoch")
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    FRAME_INDICES_FIELD_NUMBER: _ClassVar[int]
    CACHE_PATH_FIELD_NUMBER: _ClassVar[int]
    NUM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    edge_id: int
    frame_indices: str
    cache_path: str
    num_epoch: int
    def __init__(self, edge_id: _Optional[int] = ..., frame_indices: _Optional[str] = ..., cache_path: _Optional[str] = ..., num_epoch: _Optional[int] = ...) -> None: ...

class TrainReply(_message.Message):
    __slots__ = ("success", "model_data", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MODEL_DATA_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    model_data: str
    message: str
    def __init__(self, success: bool = ..., model_data: _Optional[str] = ..., message: _Optional[str] = ...) -> None: ...

class SplitTrainRequest(_message.Message):
    __slots__ = ("edge_id", "all_frame_indices", "drift_frame_indices", "cache_path", "num_epoch")
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    ALL_FRAME_INDICES_FIELD_NUMBER: _ClassVar[int]
    DRIFT_FRAME_INDICES_FIELD_NUMBER: _ClassVar[int]
    CACHE_PATH_FIELD_NUMBER: _ClassVar[int]
    NUM_EPOCH_FIELD_NUMBER: _ClassVar[int]
    edge_id: int
    all_frame_indices: str
    drift_frame_indices: str
    cache_path: str
    num_epoch: int
    def __init__(self, edge_id: _Optional[int] = ..., all_frame_indices: _Optional[str] = ..., drift_frame_indices: _Optional[str] = ..., cache_path: _Optional[str] = ..., num_epoch: _Optional[int] = ...) -> None: ...

class SplitTrainReply(_message.Message):
    __slots__ = ("success", "model_data", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MODEL_DATA_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    model_data: str
    message: str
    def __init__(self, success: bool = ..., model_data: _Optional[str] = ..., message: _Optional[str] = ...) -> None: ...

class ResourceRequest(_message.Message):
    __slots__ = ("edge_id",)
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    edge_id: int
    def __init__(self, edge_id: _Optional[int] = ...) -> None: ...

class ResourceReply(_message.Message):
    __slots__ = ("cpu_utilization", "gpu_utilization", "memory_utilization", "train_queue_size", "max_queue_size")
    CPU_UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    GPU_UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    MEMORY_UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    TRAIN_QUEUE_SIZE_FIELD_NUMBER: _ClassVar[int]
    MAX_QUEUE_SIZE_FIELD_NUMBER: _ClassVar[int]
    cpu_utilization: float
    gpu_utilization: float
    memory_utilization: float
    train_queue_size: int
    max_queue_size: int
    def __init__(self, cpu_utilization: float = ..., gpu_utilization: float = ..., memory_utilization: float = ..., train_queue_size: _Optional[int] = ..., max_queue_size: _Optional[int] = ...) -> None: ...

class BandwidthProbeRequest(_message.Message):
    __slots__ = ("payload",)
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    payload: str
    def __init__(self, payload: _Optional[str] = ...) -> None: ...

class BandwidthProbeReply(_message.Message):
    __slots__ = ("payload",)
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    payload: str
    def __init__(self, payload: _Optional[str] = ...) -> None: ...
