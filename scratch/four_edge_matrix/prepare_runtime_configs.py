"""Generate the two immutable runtime YAML files used by the N=4 matrix."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "config" / "config.yaml"
OUTPUT_DIR = Path(__file__).resolve().parent

MODELS = {
    "yolo26n": "./model_management/models/yolo26n.pt",
    "rfdetr_nano": "./model_management/models/rf-detr-nano.pth",
}


def main() -> None:
    source = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    for model_name, weights_path in MODELS.items():
        config = copy.deepcopy(source)
        config["experiment_run"]["edge_count"] = 4
        config["experiment_run"]["repeat"] = 1
        config["experiment_run"]["max_count"] = 5000
        config["client"]["lightweight"] = model_name
        config["client"]["weights_path"] = weights_path
        config["server"]["edge_model_name"] = model_name
        config["server"]["weights_path"] = weights_path
        output = OUTPUT_DIR / f"config_{model_name}.yaml"
        output.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(output)


if __name__ == "__main__":
    main()
