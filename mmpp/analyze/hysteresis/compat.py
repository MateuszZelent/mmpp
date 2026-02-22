"""Compatibility helpers for optional hysteresis dependencies."""

from __future__ import annotations

import importlib.util
import shutil


def has_module(module_name: str) -> bool:
    """Return ``True`` when module can be imported."""
    return importlib.util.find_spec(str(module_name)) is not None


def has_ipywidgets() -> bool:
    """Return ``True`` when ipywidgets is installed."""
    return has_module("ipywidgets")


def has_scipy() -> bool:
    """Return ``True`` when scipy is installed."""
    return has_module("scipy")


def has_ffmpeg() -> bool:
    """Return ``True`` when ffmpeg binary is available in PATH."""
    return shutil.which("ffmpeg") is not None


def has_pillow() -> bool:
    """Return ``True`` when Pillow is installed."""
    return has_module("PIL")


def in_notebook() -> bool:
    """Best-effort notebook detection."""
    try:
        from IPython import get_ipython
    except Exception:
        return False

    shell = get_ipython()
    return bool(shell is not None and getattr(shell, "kernel", None) is not None)


def dependency_report() -> dict[str, bool]:
    """Return a snapshot of optional dependency availability."""
    return {
        "ipywidgets": has_ipywidgets(),
        "scipy": has_scipy(),
        "ffmpeg": has_ffmpeg(),
        "pillow": has_pillow(),
        "notebook": in_notebook(),
    }


__all__ = [
    "has_module",
    "has_ipywidgets",
    "has_scipy",
    "has_ffmpeg",
    "has_pillow",
    "in_notebook",
    "dependency_report",
]
