from __future__ import annotations

from tests.test_split_runtime_edge_cloud_pipeline import (
    _assert_cross_batch_replay,
    _assert_cross_batch_train,
)


def test_yolo_cross_batch_split_replay():
    _assert_cross_batch_replay("yolo")


def test_rfdetr_cross_batch_split_replay():
    _assert_cross_batch_replay("rfdetr")


def test_tinynext_cross_batch_split_replay():
    _assert_cross_batch_replay("tinynext")


def test_yolo_cross_batch_split_train():
    _assert_cross_batch_train("yolo")


def test_rfdetr_cross_batch_split_train():
    _assert_cross_batch_train("rfdetr")


def test_tinynext_cross_batch_split_train():
    _assert_cross_batch_train("tinynext")
