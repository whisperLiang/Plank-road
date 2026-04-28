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
    SplitCandidate,
    CandidateProfile,
    SplitCandidateSelector,
    SplitPointSelector,
)
from model_management.payload import BoundaryPayload  # noqa: F401

# Re-export Dynamic Activation Sparsity (SURGEON-style) API
from model_management.activation_sparsity import (  # noqa: F401
    DASTrainer,
    apply_das_to_model,
    apply_das_to_tail,
    AutoFreezeConv2d,
    DASBatchNorm2d,
    DASGroupNorm,
    DASLayerNorm,
    AutoFreezeFC,
    ActivationClipper,
    compute_tgi,
)

# Model-zoo imports pull optional detector runtimes such as torchvision and
# ultralytics. Keep package import lightweight; callers that need model-zoo
# APIs should import model_management.model_zoo directly.
