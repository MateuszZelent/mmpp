"""Tracking method registry for vortex core tracking."""

from __future__ import annotations

from enum import Enum


class TrackingMethod(str, Enum):
    """Supported core tracking methods."""

    MAXIMUM = "maximum"
    CENTROID = "centroid"
    GAUSSIAN = "gaussian"


TRACKING_METHODS: set[str] = {item.value for item in TrackingMethod}


__all__ = ["TrackingMethod", "TRACKING_METHODS"]
