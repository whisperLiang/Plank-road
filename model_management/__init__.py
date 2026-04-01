# model_management package

# Re-export universal model splitting API
from model_management.universal_model_split import (  # noqa: F401
    UniversalModelSplitter,
    extract_split_features,
    save_split_feature_cache,
    load_split_feature_cache,
    universal_split_retrain,
    LayerInfo,
    LayerProfile,
    SplitPayload,
    SplitCandidate,
    CandidateProfile,
    SplitCandidateSelector,
    SplitPointSelector,
)

# Re-export Dynamic Activation Sparsity (SURGEON-style) API
from model_management.activation_sparsity import (  # noqa: F401
    DASTrainer,
    apply_das_to_model,
    apply_das_to_tail,
    AutoFreezeConv2d,
    DASBatchNorm2d,
    AutoFreezeFC,
    ActivationClipper,
    compute_tgi,
)

# Re-export Model Zoo (unified detection model factory)
from model_management.model_zoo import (  # noqa: F401
    build_detection_model,
    list_available_models,
    is_wrapper_model,
    model_has_roi_heads,
    get_model_family,
)
