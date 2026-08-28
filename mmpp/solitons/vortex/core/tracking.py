"""Compatibility wrapper for numerical core-tracking algorithms."""

from __future__ import annotations

from ..numerical.core import tracking as _numerical_tracking

SCIPY_AVAILABLE = _numerical_tracking.SCIPY_AVAILABLE
curve_fit = _numerical_tracking.curve_fit


def _sync_patchable_symbols() -> None:
    # Keep monkeypatch compatibility for tests that patch symbols on this module.
    _numerical_tracking.SCIPY_AVAILABLE = SCIPY_AVAILABLE
    _numerical_tracking.curve_fit = curve_fit


def track_core(*args, **kwargs):
    _sync_patchable_symbols()
    return _numerical_tracking.track_core(*args, **kwargs)


def track_core_lazy(*args, **kwargs):
    _sync_patchable_symbols()
    return _numerical_tracking.track_core_lazy(*args, **kwargs)


__all__ = ["SCIPY_AVAILABLE", "curve_fit", "track_core", "track_core_lazy"]
