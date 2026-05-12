from pathlib import Path

from baselines.accuracy_trigger_cloud_retraining import AccuracyTriggerCloudRetraining
from baselines.base_method import InferenceResult
from tests.baselines_real_helpers import build_context, make_config


def _feed_window(method, *, start: int, metric_f1: float, proxy_map: float):
    for offset in range(4):
        method.on_inference_result(
            InferenceResult(
                device_id=0,
                frame_index=start + offset,
                confidence=0.9,
                proxy_map=proxy_map,
                latency_ms=1.0,
                metric_f1=metric_f1,
                metric_map50=metric_f1,
                is_real=True,
            )
        )


def test_accuracy_trigger_uses_real_metric(tmp_path: Path):
    config = make_config("accuracy_trigger_cloud_retraining", total_frames=8)
    context = build_context(tmp_path, method_name="accuracy_trigger_cloud_retraining")

    high_proxy = AccuracyTriggerCloudRetraining(config, num_devices=1)
    high_proxy.set_context(context)
    _feed_window(high_proxy, start=0, metric_f1=0.9, proxy_map=0.1)
    assert not high_proxy.should_trigger(0)
    _feed_window(high_proxy, start=4, metric_f1=0.2, proxy_map=0.99)
    assert high_proxy.should_trigger(0)

    low_proxy = AccuracyTriggerCloudRetraining(config, num_devices=1)
    low_proxy.set_context(context)
    _feed_window(low_proxy, start=0, metric_f1=0.9, proxy_map=0.99)
    assert not low_proxy.should_trigger(0)
    _feed_window(low_proxy, start=4, metric_f1=0.2, proxy_map=0.01)
    assert low_proxy.should_trigger(0)
