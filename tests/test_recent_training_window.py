from __future__ import annotations

from cloud.orchestration.recent_training_window import RecentTrainingWindowStore


def _sample(sample_id: str, frame_id: int) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "frame_id": int(frame_id),
        "labels": {"boxes": [], "labels": []},
        "feature_ref": {"sample_id": sample_id, "path": f"{sample_id}.npy"},
    }


def test_recent_training_window_appends_dedupes_and_returns_latest_n(tmp_path) -> None:
    store = RecentTrainingWindowStore(str(tmp_path / "window"), max_samples=4)

    first = store.append_samples(
        [_sample("frame-1", 1), _sample("frame-2", 2), _sample("frame-3", 3)],
        sample_source="low_quality",
    )
    assert first.as_dict() == {
        "accepted": 3,
        "replaced": 0,
        "retained": 3,
        "dropped_old": 0,
    }

    replaced = store.append_samples(
        [
            {**_sample("frame-2", 2), "labels": {"boxes": [[1, 2, 3, 4]], "labels": [1]}},
            _sample("frame-4", 4),
        ],
        sample_source="high_quality",
    )

    assert replaced.replaced == 1
    assert store.sample_count() == 4
    assert [sample["sample_id"] for sample in store.latest_samples(2)] == [
        "frame-2",
        "frame-4",
    ]
    assert store.latest_samples(2)[0]["labels"]["labels"] == [1]
    assert "__recent_window_sequence" not in store.latest_samples(1)[0]


def test_recent_training_window_slides_without_consuming_selected_samples(tmp_path) -> None:
    store = RecentTrainingWindowStore(str(tmp_path / "window"), max_samples=3)

    store.append_samples([_sample("frame-1", 1), _sample("frame-2", 2)])
    assert [sample["sample_id"] for sample in store.latest_samples(3)] == [
        "frame-1",
        "frame-2",
    ]

    first_window = store.latest_samples(2)
    second_window = store.latest_samples(2)
    assert first_window == second_window

    slide = store.append_samples([_sample("frame-3", 3), _sample("frame-4", 4)])
    assert slide.dropped_old == 1
    assert [sample["sample_id"] for sample in store.latest_samples(3)] == [
        "frame-2",
        "frame-3",
        "frame-4",
    ]


def test_recent_training_window_reset_clears_initial_model_state(tmp_path) -> None:
    store = RecentTrainingWindowStore(str(tmp_path / "window"), max_samples=2)
    store.append_samples([_sample("frame-1", 1), _sample("frame-2", 2)])

    store.reset()

    assert store.latest_samples(2) == []
    assert store.sample_count() == 0
