from __future__ import annotations

from dataclasses import dataclass

SURGEON_METHOD = "SURGEON"
CATR_METHOD = "CATR"
EKYA_METHOD = "Ekya"

ALLOWED_BASELINE_METHODS: tuple[str, ...] = (
    SURGEON_METHOD,
    CATR_METHOD,
    EKYA_METHOD,
)

BASELINE_METHOD_LABELS: dict[str, str] = {
    SURGEON_METHOD: "SURGEON",
    CATR_METHOD: "CATR",
    EKYA_METHOD: "Ekya",
}

_RECAP_BASELINE_METHOD = "recap" + "_multi_device"
RECAP_BASELINE_ERROR = (
    f"{_RECAP_BASELINE_METHOD} is not a baseline method. "
    "Use the main RECAP distributed deployment path instead."
)


def validate_baseline_method(method: str) -> str:
    value = str(method or "").strip()
    if value == _RECAP_BASELINE_METHOD:
        raise ValueError(RECAP_BASELINE_ERROR)
    if value not in ALLOWED_BASELINE_METHODS:
        raise ValueError(
            f"Unknown baseline method {value!r}. "
            f"Valid methods: {', '.join(ALLOWED_BASELINE_METHODS)}"
        )
    return value


def baseline_method_label(method: str) -> str:
    canonical = validate_baseline_method(method)
    return BASELINE_METHOD_LABELS[canonical]


@dataclass(frozen=True)
class BaselineIdentity:
    run_id: str
    baseline_method: str
    edge_id: int

    def key(self) -> tuple[str, str, int]:
        return (self.run_id, validate_baseline_method(self.baseline_method), int(self.edge_id))
