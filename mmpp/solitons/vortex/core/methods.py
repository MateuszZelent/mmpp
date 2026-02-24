"""Compatibility wrapper for numerical core method registry."""

from __future__ import annotations

from ..numerical.core.methods import TRACKING_METHODS, TrackingMethod

__all__ = ["TrackingMethod", "TRACKING_METHODS"]
