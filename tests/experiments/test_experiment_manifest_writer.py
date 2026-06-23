from __future__ import annotations

from pathlib import Path

import yaml

from cloud.experiment_result_repository import CloudExperimentManifestWriter


def test_manifest_writer_merges_methods_edges_and_preserves_notes(tmp_path: Path) -> None:
    writer = CloudExperimentManifestWriter(
        root_dir=str(tmp_path),
        comparison_id="comparison",
        student_model="student",
        teacher_model="teacher",
        log_timezone="UTC",
    )
    summary = {"video_source": "road.mp4"}
    writer.upsert_edge_run(
        method="plank_road",
        run_id="main-r1",
        edge_id=1,
        summary=summary,
    )
    payload = yaml.safe_load(writer.manifest_path.read_text(encoding="utf-8"))
    payload["scenarios"][0]["notes"] = "keep me"
    payload["custom"] = {"owner": "user"}
    writer.manifest_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    writer.upsert_edge_run(
        method="plank_road",
        run_id="main-r1",
        edge_id=2,
        summary=summary,
    )
    writer.upsert_edge_run(
        method="pure_edge_local_updating",
        run_id="pure-r1",
        edge_id=1,
        summary=summary,
    )
    writer.upsert_edge_run(
        method="accuracy_trigger_cloud_retraining",
        run_id="accuracy-r1",
        edge_id=1,
        summary=summary,
    )

    result = yaml.safe_load(writer.manifest_path.read_text(encoding="utf-8"))
    runs = {run["run_id"]: run for run in result["runs"]}
    assert runs["main-r1"]["edge_ids"] == [1, 2]
    assert set(runs["main-r1"]["raw_logs"]["edges"]) == {"1", "2"}
    assert "cloud" not in runs["pure-r1"]["raw_logs"]
    assert "cloud" in runs["main-r1"]["raw_logs"]
    assert "cloud" in runs["accuracy-r1"]["raw_logs"]
    assert result["scenarios"][0]["notes"] == "keep me"
    assert result["custom"] == {"owner": "user"}
    assert writer.index_path.is_file()
