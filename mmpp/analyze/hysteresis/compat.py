"""Compatibility wrapper over shared UI dependency checks."""

from __future__ import annotations

from ...ui.compat import (
    dependency_report,
    has_ffmpeg,
    has_ipywidgets,
    has_module,
    has_pillow,
    has_scipy,
    in_notebook,
)

__all__ = [
    "has_module",
    "has_ipywidgets",
    "has_scipy",
    "has_ffmpeg",
    "has_pillow",
    "in_notebook",
    "dependency_report",
]
