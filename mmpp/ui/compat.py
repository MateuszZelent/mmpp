"""Optional-dependency checks for interactive plotting and export."""

from __future__ import annotations

import importlib.util
import shutil


def has_module(module_name: str) -> bool:
    """Return True when module can be imported."""
    return importlib.util.find_spec(str(module_name)) is not None


def has_ipywidgets() -> bool:
    """Return True when ipywidgets is installed."""
    return has_module("ipywidgets")


def has_scipy() -> bool:
    """Return True when scipy is installed."""
    return has_module("scipy")


def has_ffmpeg() -> bool:
    """Return True when ffmpeg binary is available in PATH."""
    return shutil.which("ffmpeg") is not None


def has_pillow() -> bool:
    """Return True when Pillow is installed."""
    return has_module("PIL")


def has_cmcrameri() -> bool:
    """Return True when cmcrameri colormap package is installed."""
    return has_module("cmcrameri")


def has_cmocean() -> bool:
    """Return True when cmocean colormap package is installed."""
    return has_module("cmocean")


def in_notebook() -> bool:
    """Best-effort notebook detection."""
    try:
        from IPython import get_ipython
    except Exception:
        return False

    shell = get_ipython()
    return bool(shell is not None and getattr(shell, "kernel", None) is not None)


def dependency_report() -> dict[str, bool]:
    """Return availability snapshot of optional UI dependencies."""
    return {
        "ipywidgets": has_ipywidgets(),
        "scipy": has_scipy(),
        "ffmpeg": has_ffmpeg(),
        "pillow": has_pillow(),
        "cmcrameri": has_cmcrameri(),
        "cmocean": has_cmocean(),
        "notebook": in_notebook(),
    }


__all__ = [
    "has_module",
    "has_ipywidgets",
    "has_scipy",
    "has_ffmpeg",
    "has_pillow",
    "has_cmcrameri",
    "has_cmocean",
    "in_notebook",
    "dependency_report",
]
