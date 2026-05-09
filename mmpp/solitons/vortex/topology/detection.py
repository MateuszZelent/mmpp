"""Compatibility wrapper for numerical topology detection."""

from __future__ import annotations

from ..numerical.topology.detection import detect_topology

detect_topology.__module__ = __name__

__all__ = ["detect_topology"]
