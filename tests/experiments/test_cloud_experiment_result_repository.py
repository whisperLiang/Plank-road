from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cloud.experiment_result_repository import (
    CloudExperimentManifestWriter,
    CloudExperimentResultRepository,
)
from grpc_server import message_transmission_pb2
from tools.experiments.experiment_common import read_csv
from tools.experiments.normalize_plank_road_baseline_logs import normalize


def _request(
    *,
    comparison_id: str = "comparison",
    run_id: str = "run-1",
    method: str = "plank_road",
    edge_id: int = 1,
    relative_path: str = "latest_inference_results.jsonl",
    content: bytes = b'{"frame_index": 1}\n',
):
    artifact = message_transmission_pb2.ExperimentResultArtifact(
        comparison_id=comparison_id,
        run_id=run_id,
        method=method,
        edge_id=edge_id,
        relative_path=relative_path,
        content=content,
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        content_type="application/json",
        is_final=True,
    )
    return message_transmission_pb2.UploadExperimentResultRequest(
        comparison_id=comparison_id,
        run_id=run_id,
        method=method,
        edge_id=edge_id,
        artifacts=[artifact],
    )


@pytest.mark.parametrize(
    "method",
    [
        "plank_road",
        "pure_edge_local_updating",
        "accuracy_trigger_cloud_retraining",
    ],
)
def test_repository_stores_all_supported_methods(tmp_path: Path, method: str) -> None:
    repository = CloudExperimentResultRepository(str(tmp_path))
    stored = repository.store_artifacts(_request(method=method))
    assert stored == [
        tmp_path
        / "comparison"
        / "raw_logs"
        / method
        / "edge_1"
        / "run-1"
        / "latest_inference_results.jsonl"
    ]
    assert stored[0].read_bytes() == b'{"frame_index": 1}\n'
    assert stored[0].with_name("uploaded_artifacts_manifest.json").is_file()


@pytest.mark.parametrize("relative_path", ["../secret", "/tmp/secret", r"..\\secret"])
def test_repository_rejects_unsafe_paths(tmp_path: Path, relative_path: str) -> None:
    repository = CloudExperimentResultRepository(str(tmp_path))
    with pytest.raises(ValueError):
        repository.store_artifacts(_request(relative_path=relative_path))


def test_repository_rejects_unknown_method_and_large_or_invalid_content(
    tmp_path: Path,
) -> None:
    repository = CloudExperimentResultRepository(str(tmp_path), max_artifact_bytes=4)
    with pytest.raises(ValueError, match="unknown experiment method"):
        repository.store_artifacts(_request(method="unknown", content=b"x"))
    with pytest.raises(ValueError, match="exceeds"):
        repository.store_artifacts(_request(content=b"12345"))
    bad = _request(content=b"1234")
    bad.artifacts[0].sha256 = "0" * 64
    with pytest.raises(ValueError, match="sha256 mismatch"):
        repository.store_artifacts(bad)


def test_repository_preserves_skipped_metadata_from_client_manifest(
    tmp_path: Path,
) -> None:
    repository = CloudExperimentResultRepository(str(tmp_path))
    client_manifest = json.dumps(
        {
            "artifacts": [
                {
                    "relative_path": "latest_inference_results.jsonl",
                    "size_bytes": 10_000,
                    "sha256": "",
                    "content_type": "application/json",
                    "status": "skipped_too_large",
                    "message": "artifact exceeds max_artifact_bytes=4",
                }
            ]
        }
    ).encode()

    assert repository.store_artifacts(
        _request(
            relative_path="uploaded_artifacts_manifest.json",
            content=client_manifest,
        )
    ) == []
    manifest_path = (
        tmp_path
        / "comparison"
        / "raw_logs"
        / "plank_road"
        / "edge_1"
        / "run-1"
        / "uploaded_artifacts_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["artifacts"] == [
        {
            "relative_path": "latest_inference_results.jsonl",
            "stored_path": "",
            "size_bytes": 10_000,
            "sha256": "",
            "content_type": "application/json",
            "is_final": False,
            "status": "skipped_too_large",
            "message": "artifact exceeds max_artifact_bytes=4",
        }
    ]


def test_repository_is_idempotent_and_preserves_conflicting_duplicate(
    tmp_path: Path,
) -> None:
    repository = CloudExperimentResultRepository(str(tmp_path))
    original = repository.store_artifacts(_request(content=b"first"))[0]
    assert repository.store_artifacts(_request(content=b"first")) == [original]
    duplicate = repository.store_artifacts(_request(content=b"second"))[0]
    assert duplicate != original
    assert ".duplicate." in duplicate.name
    assert original.read_bytes() == b"first"
    assert duplicate.read_bytes() == b"second"


def test_repository_uses_edge_summary_to_update_manifest(tmp_path: Path) -> None:
    writer = CloudExperimentManifestWriter(
        root_dir=str(tmp_path),
        comparison_id="comparison",
        student_model="student",
        teacher_model="teacher",
        log_timezone="UTC",
    )
    repository = CloudExperimentResultRepository(
        str(tmp_path),
        manifest_writer=writer,
    )
    summary = json.dumps(
        {
            "video_source": "./video_data/road.mp4",
            "student_model": "student",
            "teacher_model": "teacher",
        }
    ).encode()
    repository.store_artifacts(
        _request(relative_path="edge_summary.json", content=summary)
    )
    manifest = writer.manifest_path.read_text(encoding="utf-8")
    assert "scenario_name: road" in manifest
    assert "raw_logs/plank_road/edge_1/run-1" in manifest


def test_auto_repository_layout_normalizes_without_manual_log_copy(
    tmp_path: Path,
) -> None:
    writer = CloudExperimentManifestWriter(
        root_dir=str(tmp_path),
        comparison_id="comparison",
        student_model="student",
        teacher_model="teacher",
        log_timezone="UTC",
    )
    repository = CloudExperimentResultRepository(
        str(tmp_path),
        manifest_writer=writer,
    )
    runs = (
        ("plank_road", "main-r1"),
        ("pure_edge_local_updating", "pure-r1"),
        ("accuracy_trigger_cloud_retraining", "accuracy-r1"),
    )
    summary = json.dumps({"video_source": "road.mp4"}).encode()
    frame = (
        json.dumps(
            {
                "frame_index": 1,
                "start_time": 1.0,
                "latency_ms": 2.0,
                "result_source": "inference",
                "result": {"labels": [], "boxes": [], "scores": []},
            }
        )
        + "\n"
    ).encode()
    for method, run_id in runs:
        repository.store_artifacts(
            _request(
                method=method,
                run_id=run_id,
                relative_path="latest_inference_results.jsonl",
                content=frame,
            )
        )
        repository.store_artifacts(
            _request(
                method=method,
                run_id=run_id,
                relative_path="edge_summary.json",
                content=summary,
            )
        )

    comparison_dir = tmp_path / "comparison"
    normalize(comparison_dir)
    frames = read_csv(comparison_dir / "normalized" / "frame_metrics.csv")
    assert {row["method"] for row in frames} == {method for method, _ in runs}
