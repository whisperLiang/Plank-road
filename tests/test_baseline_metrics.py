from __future__ import annotations

from baselines.metrics import MetricsCollector


def test_baseline_metrics_export_uses_metric_map50_not_proxy_map() -> None:
    collector = MetricsCollector(method_name="method-a", num_devices=1)
    device = collector.get_device(0)
    device.record_inference(
        latency_ms=12.0,
        confidence=0.75,
        metric_f1=0.5,
        metric_map50=0.6,
    )

    row = collector.device_rows()[0]
    overall = collector.compute_overall().to_dict()

    assert "avg_map50" in row
    assert row["avg_map50"] == 0.6
    assert overall["avg_map50"] == 0.6
    assert "avg_proxy_map" not in row
    assert "avg_proxy_map" not in overall
