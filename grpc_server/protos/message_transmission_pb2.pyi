from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TrainingJobType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRAINING_JOB_TYPE_UNSPECIFIED: _ClassVar[TrainingJobType]
    TRAINING_JOB_TYPE_FULL_FRAME: _ClassVar[TrainingJobType]
    TRAINING_JOB_TYPE_SPLIT: _ClassVar[TrainingJobType]
    TRAINING_JOB_TYPE_CONTINUAL_LEARNING: _ClassVar[TrainingJobType]
TRAINING_JOB_TYPE_UNSPECIFIED: TrainingJobType
TRAINING_JOB_TYPE_FULL_FRAME: TrainingJobType
TRAINING_JOB_TYPE_SPLIT: TrainingJobType
TRAINING_JOB_TYPE_CONTINUAL_LEARNING: TrainingJobType

class TrainRequest(_message.Message):
    __slots__ = ("edge_id", "frame_indices", "cache_path", "payload_zip")
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    FRAME_INDICES_FIELD_NUMBER: _ClassVar[int]
    CACHE_PATH_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_ZIP_FIELD_NUMBER: _ClassVar[int]
    edge_id: int
    frame_indices: _containers.RepeatedScalarFieldContainer[int]
    cache_path: str
    payload_zip: bytes
    def __init__(self, edge_id: _Optional[int] = ..., frame_indices: _Optional[_Iterable[int]] = ..., cache_path: _Optional[str] = ..., payload_zip: _Optional[bytes] = ...) -> None: ...

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
    __slots__ = ("edge_id", "all_frame_indices", "drift_frame_indices", "cache_path", "payload_zip")
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    ALL_FRAME_INDICES_FIELD_NUMBER: _ClassVar[int]
    DRIFT_FRAME_INDICES_FIELD_NUMBER: _ClassVar[int]
    CACHE_PATH_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_ZIP_FIELD_NUMBER: _ClassVar[int]
    edge_id: int
    all_frame_indices: _containers.RepeatedScalarFieldContainer[int]
    drift_frame_indices: _containers.RepeatedScalarFieldContainer[int]
    cache_path: str
    payload_zip: bytes
    def __init__(self, edge_id: _Optional[int] = ..., all_frame_indices: _Optional[_Iterable[int]] = ..., drift_frame_indices: _Optional[_Iterable[int]] = ..., cache_path: _Optional[str] = ..., payload_zip: _Optional[bytes] = ...) -> None: ...

class SplitTrainReply(_message.Message):
    __slots__ = ("success", "model_data", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MODEL_DATA_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    model_data: str
    message: str
    def __init__(self, success: bool = ..., model_data: _Optional[str] = ..., message: _Optional[str] = ...) -> None: ...

class ContinualLearningRequest(_message.Message):
    __slots__ = ("protocol_version", "edge_id", "cache_path", "send_low_conf_features", "payload_zip")
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    CACHE_PATH_FIELD_NUMBER: _ClassVar[int]
    SEND_LOW_CONF_FEATURES_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_ZIP_FIELD_NUMBER: _ClassVar[int]
    protocol_version: str
    edge_id: int
    cache_path: str
    send_low_conf_features: bool
    payload_zip: bytes
    def __init__(self, protocol_version: _Optional[str] = ..., edge_id: _Optional[int] = ..., cache_path: _Optional[str] = ..., send_low_conf_features: bool = ..., payload_zip: _Optional[bytes] = ...) -> None: ...

class ContinualLearningReply(_message.Message):
    __slots__ = ("success", "model_data", "message", "protocol_version")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MODEL_DATA_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    success: bool
    model_data: str
    message: str
    protocol_version: str
    def __init__(self, success: bool = ..., model_data: _Optional[str] = ..., message: _Optional[str] = ..., protocol_version: _Optional[str] = ...) -> None: ...

class SubmitTrainingJobRequest(_message.Message):
    __slots__ = ("protocol_version", "edge_id", "request_id", "job_type", "cache_path", "send_low_conf_features", "frame_indices", "all_frame_indices", "drift_frame_indices", "payload_zip", "base_model_version")
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_TYPE_FIELD_NUMBER: _ClassVar[int]
    CACHE_PATH_FIELD_NUMBER: _ClassVar[int]
    SEND_LOW_CONF_FEATURES_FIELD_NUMBER: _ClassVar[int]
    FRAME_INDICES_FIELD_NUMBER: _ClassVar[int]
    ALL_FRAME_INDICES_FIELD_NUMBER: _ClassVar[int]
    DRIFT_FRAME_INDICES_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_ZIP_FIELD_NUMBER: _ClassVar[int]
    BASE_MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    protocol_version: str
    edge_id: int
    request_id: str
    job_type: TrainingJobType
    cache_path: str
    send_low_conf_features: bool
    frame_indices: _containers.RepeatedScalarFieldContainer[int]
    all_frame_indices: _containers.RepeatedScalarFieldContainer[int]
    drift_frame_indices: _containers.RepeatedScalarFieldContainer[int]
    payload_zip: bytes
    base_model_version: str
    def __init__(self, protocol_version: _Optional[str] = ..., edge_id: _Optional[int] = ..., request_id: _Optional[str] = ..., job_type: _Optional[_Union[TrainingJobType, str]] = ..., cache_path: _Optional[str] = ..., send_low_conf_features: bool = ..., frame_indices: _Optional[_Iterable[int]] = ..., all_frame_indices: _Optional[_Iterable[int]] = ..., drift_frame_indices: _Optional[_Iterable[int]] = ..., payload_zip: _Optional[bytes] = ..., base_model_version: _Optional[str] = ...) -> None: ...

class SubmitTrainingJobChunk(_message.Message):
    __slots__ = ("protocol_version", "edge_id", "request_id", "job_type", "cache_path", "send_low_conf_features", "frame_indices", "all_frame_indices", "drift_frame_indices", "payload_chunk", "base_model_version", "chunk_index", "total_payload_bytes")
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_TYPE_FIELD_NUMBER: _ClassVar[int]
    CACHE_PATH_FIELD_NUMBER: _ClassVar[int]
    SEND_LOW_CONF_FEATURES_FIELD_NUMBER: _ClassVar[int]
    FRAME_INDICES_FIELD_NUMBER: _ClassVar[int]
    ALL_FRAME_INDICES_FIELD_NUMBER: _ClassVar[int]
    DRIFT_FRAME_INDICES_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_CHUNK_FIELD_NUMBER: _ClassVar[int]
    BASE_MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    CHUNK_INDEX_FIELD_NUMBER: _ClassVar[int]
    TOTAL_PAYLOAD_BYTES_FIELD_NUMBER: _ClassVar[int]
    protocol_version: str
    edge_id: int
    request_id: str
    job_type: TrainingJobType
    cache_path: str
    send_low_conf_features: bool
    frame_indices: _containers.RepeatedScalarFieldContainer[int]
    all_frame_indices: _containers.RepeatedScalarFieldContainer[int]
    drift_frame_indices: _containers.RepeatedScalarFieldContainer[int]
    payload_chunk: bytes
    base_model_version: str
    chunk_index: int
    total_payload_bytes: int
    def __init__(self, protocol_version: _Optional[str] = ..., edge_id: _Optional[int] = ..., request_id: _Optional[str] = ..., job_type: _Optional[_Union[TrainingJobType, str]] = ..., cache_path: _Optional[str] = ..., send_low_conf_features: bool = ..., frame_indices: _Optional[_Iterable[int]] = ..., all_frame_indices: _Optional[_Iterable[int]] = ..., drift_frame_indices: _Optional[_Iterable[int]] = ..., payload_chunk: _Optional[bytes] = ..., base_model_version: _Optional[str] = ..., chunk_index: _Optional[int] = ..., total_payload_bytes: _Optional[int] = ...) -> None: ...

class SubmitTrainingJobReply(_message.Message):
    __slots__ = ("accepted", "job_id", "status", "queue_position", "message")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    QUEUE_POSITION_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    job_id: str
    status: str
    queue_position: int
    message: str
    def __init__(self, accepted: bool = ..., job_id: _Optional[str] = ..., status: _Optional[str] = ..., queue_position: _Optional[int] = ..., message: _Optional[str] = ...) -> None: ...

class TrainingJobStatusRequest(_message.Message):
    __slots__ = ("edge_id", "job_id")
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    edge_id: int
    job_id: str
    def __init__(self, edge_id: _Optional[int] = ..., job_id: _Optional[str] = ...) -> None: ...

class TrainingJobStatusReply(_message.Message):
    __slots__ = ("found", "job_id", "edge_id", "status", "queue_position", "message", "request_id", "job_type", "result_available", "submitted_at_ms", "started_at_ms", "finished_at_ms", "protocol_version", "base_model_version", "result_model_version")
    FOUND_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    QUEUE_POSITION_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_TYPE_FIELD_NUMBER: _ClassVar[int]
    RESULT_AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    SUBMITTED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    STARTED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    FINISHED_AT_MS_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    BASE_MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    RESULT_MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    found: bool
    job_id: str
    edge_id: int
    status: str
    queue_position: int
    message: str
    request_id: str
    job_type: TrainingJobType
    result_available: bool
    submitted_at_ms: int
    started_at_ms: int
    finished_at_ms: int
    protocol_version: str
    base_model_version: str
    result_model_version: str
    def __init__(self, found: bool = ..., job_id: _Optional[str] = ..., edge_id: _Optional[int] = ..., status: _Optional[str] = ..., queue_position: _Optional[int] = ..., message: _Optional[str] = ..., request_id: _Optional[str] = ..., job_type: _Optional[_Union[TrainingJobType, str]] = ..., result_available: bool = ..., submitted_at_ms: _Optional[int] = ..., started_at_ms: _Optional[int] = ..., finished_at_ms: _Optional[int] = ..., protocol_version: _Optional[str] = ..., base_model_version: _Optional[str] = ..., result_model_version: _Optional[str] = ...) -> None: ...

class DownloadTrainedModelRequest(_message.Message):
    __slots__ = ("edge_id", "job_id")
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    edge_id: int
    job_id: str
    def __init__(self, edge_id: _Optional[int] = ..., job_id: _Optional[str] = ...) -> None: ...

class DownloadTrainedModelReply(_message.Message):
    __slots__ = ("success", "job_id", "status", "model_data", "message", "protocol_version", "result_model_version")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MODEL_DATA_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    RESULT_MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    success: bool
    job_id: str
    status: str
    model_data: str
    message: str
    protocol_version: str
    result_model_version: str
    def __init__(self, success: bool = ..., job_id: _Optional[str] = ..., status: _Optional[str] = ..., model_data: _Optional[str] = ..., message: _Optional[str] = ..., protocol_version: _Optional[str] = ..., result_model_version: _Optional[str] = ...) -> None: ...

class CancelTrainingJobRequest(_message.Message):
    __slots__ = ("edge_id", "job_id")
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    edge_id: int
    job_id: str
    def __init__(self, edge_id: _Optional[int] = ..., job_id: _Optional[str] = ...) -> None: ...

class CancelTrainingJobReply(_message.Message):
    __slots__ = ("cancelled", "message")
    CANCELLED_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    cancelled: bool
    message: str
    def __init__(self, cancelled: bool = ..., message: _Optional[str] = ...) -> None: ...

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
    def __init__(self, cpu_utilization: _Optional[float] = ..., gpu_utilization: _Optional[float] = ..., memory_utilization: _Optional[float] = ..., train_queue_size: _Optional[int] = ..., max_queue_size: _Optional[int] = ...) -> None: ...

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
