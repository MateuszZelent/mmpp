from __future__ import annotations

from mmpp.analyze.hysteresis.animation import create_animation as create_animation_hysteresis
from mmpp.analyze.hysteresis.compat import dependency_report
from mmpp.analyze.hysteresis.metrics.registry import (
    get_registered_metric,
    register_metric,
)
from mmpp.ui.animation import create_animation as create_animation_shared


def test_hysteresis_wrappers_bind_to_shared_modules():
    assert callable(create_animation_hysteresis)
    assert create_animation_hysteresis is create_animation_shared

    report = dependency_report()
    assert isinstance(report, dict)
    assert "ipywidgets" in report


def test_hysteresis_metric_registry_roundtrip():
    @register_metric("stage1_registry_smoke")
    def _metric(_result):
        return 1.0

    resolved = get_registered_metric("stage1_registry_smoke")
    assert resolved is _metric
