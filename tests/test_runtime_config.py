from __future__ import annotations

import pytest

from config.runtime import load_runtime_config


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("suffix_num_threads", "inference_num_threads"),
        ("suffix_thread_tuning_iterations", "no longer used"),
    ],
)
def test_removed_fixed_split_thread_fields_are_rejected(
    tmp_path,
    field: str,
    replacement: str,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
client:
  split_learning:
    fixed_split:
      {field}: 4
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=replacement):
        load_runtime_config(path)


@pytest.mark.parametrize("yaml_value", ["0", "-1", "true", '"4"'])
def test_inference_num_threads_requires_positive_integer(tmp_path, yaml_value: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"""
client:
  split_learning:
    fixed_split:
      inference_num_threads: {yaml_value}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="inference_num_threads"):
        load_runtime_config(path)
