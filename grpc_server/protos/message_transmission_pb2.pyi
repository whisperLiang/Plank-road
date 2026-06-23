from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TrainingJobType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRAINING_JOB_TYPE_UNSPECIFIED: _ClassVar[TrainingJobType]
    TRAINING_JOB_TYPE_FULL_FRAME: _ClassVar[TrainingJobType]
    TRAINING_JOB_TYPE_CONTINUAL_LEARNING: _ClassVar[TrainingJobType]
    TRAINING_JOB_TYPE_BASELINE_TRAINING: _ClassVar[TrainingJobType]
TRAINING_JOB_TYPE_UNSPECIFIED: TrainingJobType
TRAINING_JOB_TYPE_FULL_FRAME: TrainingJobType
TRAINING_JOB_TYPE_CONTINUAL_LEARNING: TrainingJobType
TRAINING_JOB_TYPE_BASELINE_TRAINING: TrainingJobType

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

class SampleSyncRequest(_message.Message):
    __slots__ = ("protocol_version", "edge_id", "model_id", "model_version", "split_config_id", "sync_type", "payload_zip")
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    SPLIT_CONFIG_ID_FIELD_NUMBER: _ClassVar[int]
    SYNC_TYPE_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_ZIP_FIELD_NUMBER: _ClassVar[int]
    protocol_version: str
    edge_id: int
    model_id: str
    model_version: str
    split_config_id: str
    sync_type: str
    payload_zip: bytes
    def __init__(self, protocol_version: _Optional[str] = ..., edge_id: _Optional[int] = ..., model_id: _Optional[str] = ..., model_version: _Optional[str] = ..., split_config_id: _Optional[str] = ..., sync_type: _Optional[str] = ..., payload_zip: _Optional[bytes] = ...) -> None: ...

class SampleSyncReply(_message.Message):
    __slots__ = ("success", "message", "committed_samples")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    COMMITTED_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    committed_samples: int
    def __init__(self, success: bool = ..., message: _Optional[str] = ..., committed_samples: _Optional[int] = ...) -> None: ...

class SubmitTrainingJobRequest(_message.Message):
    __slots__ = ("protocol_version", "edge_id", "request_id", "job_type", "cache_path", "send_low_conf_features", "frame_indices", "payload_zip", "base_model_version")
    PROTOCOL_VERSION_FIELD_NUMBER: _ClassVar[int]
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    JOB_TYPE_FIELD_NUMBER: _ClassVar[int]
    CACHE_PATH_FIELD_NUMBER: _ClassVar[int]
    SEND_LOW_CONF_FEATURES_FIELD_NUMBER: _ClassVar[int]
    FRAME_INDICES_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_ZIP_FIELD_NUMBER: _ClassVar[int]
    BASE_MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    protocol_version: str
    edge_id: int
    request_id: str
    job_type: TrainingJobType
    cache_path: str
    send_low_conf_features: bool
    frame_indices: _containers.RepeatedScalarFieldContainer[int]
    payload_zip: bytes
    base_model_version: str
    def __init__(self, protocol_version: _Optional[str] = ..., edge_id: _Optional[int] = ..., request_id: _Optional[str] = ..., job_type: _Optional[_Union[TrainingJobType, str]] = ..., cache_path: _Optional[str] = ..., send_low_conf_features: bool = ..., frame_indices: _Optional[_Iterable[int]] = ..., payload_zip: _Optional[bytes] = ..., base_model_version: _Optional[str] = ...) -> None: ...

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
    __slots__ = ("found", "job_id", "edge_id", "status", "queue_position", "message", "request_id", "job_type", "result_available", "submitted_at_ms", "started_at_ms", "finished_at_ms", "protocol_version", "base_model_version", "result_model_version", "worker_id")
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
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
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
    worker_id: str
    def __init__(self, found: bool = ..., job_id: _Optional[str] = ..., edge_id: _Optional[int] = ..., status: _Optional[str] = ..., queue_position: _Optional[int] = ..., message: _Optional[str] = ..., request_id: _Optional[str] = ..., job_type: _Optional[_Union[TrainingJobType, str]] = ..., result_available: bool = ..., submitted_at_ms: _Optional[int] = ..., started_at_ms: _Optional[int] = ..., finished_at_ms: _Optional[int] = ..., protocol_version: _Optional[str] = ..., base_model_version: _Optional[str] = ..., result_model_version: _Optional[str] = ..., worker_id: _Optional[str] = ...) -> None: ...

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

class ReportEdgeModelVersionRequest(_message.Message):
    __slots__ = ("edge_id", "model_id", "model_version")
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    edge_id: int
    model_id: str
    model_version: str
    def __init__(self, edge_id: _Optional[int] = ..., model_id: _Optional[str] = ..., model_version: _Optional[str] = ...) -> None: ...

class ReportEdgeModelVersionReply(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

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

class BaselineAck(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class BaselineRegisterEdgeRequest(_message.Message):
    __slots__ = ("run_id", "baseline_method", "edge_id", "model_name", "model_version", "video_source", "timestamp_ms")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    BASELINE_METHOD_FIELD_NUMBER: _ClassVar[int]
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    VIDEO_SOURCE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    baseline_method: str
    edge_id: int
    model_name: str
    model_version: str
    video_source: str
    timestamp_ms: int
    def __init__(self, run_id: _Optional[str] = ..., baseline_method: _Optional[str] = ..., edge_id: _Optional[int] = ..., model_name: _Optional[str] = ..., model_version: _Optional[str] = ..., video_source: _Optional[str] = ..., timestamp_ms: _Optional[int] = ...) -> None: ...

class BaselineHeartbeatRequest(_message.Message):
    __slots__ = ("run_id", "baseline_method", "edge_id", "timestamp_ms", "metrics_ref", "metrics_json")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    BASELINE_METHOD_FIELD_NUMBER: _ClassVar[int]
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    METRICS_REF_FIELD_NUMBER: _ClassVar[int]
    METRICS_JSON_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    baseline_method: str
    edge_id: int
    timestamp_ms: int
    metrics_ref: str
    metrics_json: str
    def __init__(self, run_id: _Optional[str] = ..., baseline_method: _Optional[str] = ..., edge_id: _Optional[int] = ..., timestamp_ms: _Optional[int] = ..., metrics_ref: _Optional[str] = ..., metrics_json: _Optional[str] = ...) -> None: ...

class BaselineFrameRequest(_message.Message):
    __slots__ = ("run_id", "baseline_method", "edge_id", "frame_id", "timestamp_ms", "model_name", "model_version", "video_source", "upload_mode", "is_keyframe", "edge_prediction_json", "cloud_prediction_json", "teacher_prediction_json", "confidence", "entropy", "quality_metadata_json", "raw_frame", "raw_frame_ref", "feature_ref_json", "metrics_ref", "job_id")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    BASELINE_METHOD_FIELD_NUMBER: _ClassVar[int]
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    VIDEO_SOURCE_FIELD_NUMBER: _ClassVar[int]
    UPLOAD_MODE_FIELD_NUMBER: _ClassVar[int]
    IS_KEYFRAME_FIELD_NUMBER: _ClassVar[int]
    EDGE_PREDICTION_JSON_FIELD_NUMBER: _ClassVar[int]
    CLOUD_PREDICTION_JSON_FIELD_NUMBER: _ClassVar[int]
    TEACHER_PREDICTION_JSON_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    ENTROPY_FIELD_NUMBER: _ClassVar[int]
    QUALITY_METADATA_JSON_FIELD_NUMBER: _ClassVar[int]
    RAW_FRAME_FIELD_NUMBER: _ClassVar[int]
    RAW_FRAME_REF_FIELD_NUMBER: _ClassVar[int]
    FEATURE_REF_JSON_FIELD_NUMBER: _ClassVar[int]
    METRICS_REF_FIELD_NUMBER: _ClassVar[int]
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    baseline_method: str
    edge_id: int
    frame_id: int
    timestamp_ms: int
    model_name: str
    model_version: str
    video_source: str
    upload_mode: str
    is_keyframe: bool
    edge_prediction_json: str
    cloud_prediction_json: str
    teacher_prediction_json: str
    confidence: float
    entropy: float
    quality_metadata_json: str
    raw_frame: bytes
    raw_frame_ref: str
    feature_ref_json: str
    metrics_ref: str
    job_id: str
    def __init__(self, run_id: _Optional[str] = ..., baseline_method: _Optional[str] = ..., edge_id: _Optional[int] = ..., frame_id: _Optional[int] = ..., timestamp_ms: _Optional[int] = ..., model_name: _Optional[str] = ..., model_version: _Optional[str] = ..., video_source: _Optional[str] = ..., upload_mode: _Optional[str] = ..., is_keyframe: bool = ..., edge_prediction_json: _Optional[str] = ..., cloud_prediction_json: _Optional[str] = ..., teacher_prediction_json: _Optional[str] = ..., confidence: _Optional[float] = ..., entropy: _Optional[float] = ..., quality_metadata_json: _Optional[str] = ..., raw_frame: _Optional[bytes] = ..., raw_frame_ref: _Optional[str] = ..., feature_ref_json: _Optional[str] = ..., metrics_ref: _Optional[str] = ..., job_id: _Optional[str] = ...) -> None: ...

class BaselineWindowSample(_message.Message):
    __slots__ = ("frame_id", "timestamp_ms", "raw_frame", "edge_prediction_json", "confidence", "entropy", "quality_metadata_json", "upload_mode", "is_keyframe")
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    RAW_FRAME_FIELD_NUMBER: _ClassVar[int]
    EDGE_PREDICTION_JSON_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    ENTROPY_FIELD_NUMBER: _ClassVar[int]
    QUALITY_METADATA_JSON_FIELD_NUMBER: _ClassVar[int]
    UPLOAD_MODE_FIELD_NUMBER: _ClassVar[int]
    IS_KEYFRAME_FIELD_NUMBER: _ClassVar[int]
    frame_id: int
    timestamp_ms: int
    raw_frame: bytes
    edge_prediction_json: str
    confidence: float
    entropy: float
    quality_metadata_json: str
    upload_mode: str
    is_keyframe: bool
    def __init__(self, frame_id: _Optional[int] = ..., timestamp_ms: _Optional[int] = ..., raw_frame: _Optional[bytes] = ..., edge_prediction_json: _Optional[str] = ..., confidence: _Optional[float] = ..., entropy: _Optional[float] = ..., quality_metadata_json: _Optional[str] = ..., upload_mode: _Optional[str] = ..., is_keyframe: bool = ...) -> None: ...

class BaselineWindowRequest(_message.Message):
    __slots__ = ("run_id", "baseline_method", "edge_id", "model_name", "model_version", "video_source", "window_id", "window_start_frame_id", "window_end_frame_id", "timestamp_ms", "selected_samples")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    BASELINE_METHOD_FIELD_NUMBER: _ClassVar[int]
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    VIDEO_SOURCE_FIELD_NUMBER: _ClassVar[int]
    WINDOW_ID_FIELD_NUMBER: _ClassVar[int]
    WINDOW_START_FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    WINDOW_END_FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    SELECTED_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    baseline_method: str
    edge_id: int
    model_name: str
    model_version: str
    video_source: str
    window_id: str
    window_start_frame_id: int
    window_end_frame_id: int
    timestamp_ms: int
    selected_samples: _containers.RepeatedCompositeFieldContainer[BaselineWindowSample]
    def __init__(self, run_id: _Optional[str] = ..., baseline_method: _Optional[str] = ..., edge_id: _Optional[int] = ..., model_name: _Optional[str] = ..., model_version: _Optional[str] = ..., video_source: _Optional[str] = ..., window_id: _Optional[str] = ..., window_start_frame_id: _Optional[int] = ..., window_end_frame_id: _Optional[int] = ..., timestamp_ms: _Optional[int] = ..., selected_samples: _Optional[_Iterable[_Union[BaselineWindowSample, _Mapping]]] = ...) -> None: ...

class BaselineInferenceRequest(_message.Message):
    __slots__ = ("run_id", "baseline_method", "edge_id", "frame_id")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    BASELINE_METHOD_FIELD_NUMBER: _ClassVar[int]
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    baseline_method: str
    edge_id: int
    frame_id: int
    def __init__(self, run_id: _Optional[str] = ..., baseline_method: _Optional[str] = ..., edge_id: _Optional[int] = ..., frame_id: _Optional[int] = ...) -> None: ...

class BaselineInferenceReply(_message.Message):
    __slots__ = ("success", "message", "run_id", "baseline_method", "edge_id", "frame_id", "cloud_prediction_json", "confidence", "timestamp_ms")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    BASELINE_METHOD_FIELD_NUMBER: _ClassVar[int]
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    CLOUD_PREDICTION_JSON_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    run_id: str
    baseline_method: str
    edge_id: int
    frame_id: int
    cloud_prediction_json: str
    confidence: float
    timestamp_ms: int
    def __init__(self, success: bool = ..., message: _Optional[str] = ..., run_id: _Optional[str] = ..., baseline_method: _Optional[str] = ..., edge_id: _Optional[int] = ..., frame_id: _Optional[int] = ..., cloud_prediction_json: _Optional[str] = ..., confidence: _Optional[float] = ..., timestamp_ms: _Optional[int] = ...) -> None: ...

class BaselineCommandRequest(_message.Message):
    __slots__ = ("run_id", "baseline_method", "edge_id", "timestamp_ms")
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    BASELINE_METHOD_FIELD_NUMBER: _ClassVar[int]
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    run_id: str
    baseline_method: str
    edge_id: int
    timestamp_ms: int
    def __init__(self, run_id: _Optional[str] = ..., baseline_method: _Optional[str] = ..., edge_id: _Optional[int] = ..., timestamp_ms: _Optional[int] = ...) -> None: ...

class BaselineCommandReply(_message.Message):
    __slots__ = ("success", "message", "command_json")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    COMMAND_JSON_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    command_json: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, success: bool = ..., message: _Optional[str] = ..., command_json: _Optional[_Iterable[str]] = ...) -> None: ...

class ExperimentResultArtifact(_message.Message):
    __slots__ = ("comparison_id", "run_id", "method", "edge_id", "relative_path", "content", "size_bytes", "sha256", "content_type", "is_final")
    COMPARISON_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    RELATIVE_PATH_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    SHA256_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_FIELD_NUMBER: _ClassVar[int]
    comparison_id: str
    run_id: str
    method: str
    edge_id: int
    relative_path: str
    content: bytes
    size_bytes: int
    sha256: str
    content_type: str
    is_final: bool
    def __init__(self, comparison_id: _Optional[str] = ..., run_id: _Optional[str] = ..., method: _Optional[str] = ..., edge_id: _Optional[int] = ..., relative_path: _Optional[str] = ..., content: _Optional[bytes] = ..., size_bytes: _Optional[int] = ..., sha256: _Optional[str] = ..., content_type: _Optional[str] = ..., is_final: bool = ...) -> None: ...

class UploadExperimentResultRequest(_message.Message):
    __slots__ = ("comparison_id", "run_id", "method", "edge_id", "artifacts")
    COMPARISON_ID_FIELD_NUMBER: _ClassVar[int]
    RUN_ID_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    EDGE_ID_FIELD_NUMBER: _ClassVar[int]
    ARTIFACTS_FIELD_NUMBER: _ClassVar[int]
    comparison_id: str
    run_id: str
    method: str
    edge_id: int
    artifacts: _containers.RepeatedCompositeFieldContainer[ExperimentResultArtifact]
    def __init__(self, comparison_id: _Optional[str] = ..., run_id: _Optional[str] = ..., method: _Optional[str] = ..., edge_id: _Optional[int] = ..., artifacts: _Optional[_Iterable[_Union[ExperimentResultArtifact, _Mapping]]] = ...) -> None: ...

class UploadExperimentResultResponse(_message.Message):
    __slots__ = ("accepted", "message", "stored_paths")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    STORED_PATHS_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    message: str
    stored_paths: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, accepted: bool = ..., message: _Optional[str] = ..., stored_paths: _Optional[_Iterable[str]] = ...) -> None: ...
