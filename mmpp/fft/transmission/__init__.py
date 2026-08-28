"""Transmission analysis API for FFT module."""

from __future__ import annotations

from importlib import import_module

from .compute import (
    TransmissionCompute,
    TransmissionConfig,
    TransmissionModesResult,
    TransmissionResult,
)

_LAZY_EXPORTS = {
    "TransmissionPlotConfig": (".plot", "TransmissionPlotConfig"),
    "TransmissionPlotter": (".plot", "TransmissionPlotter"),
    "overlay_transmission": (".experimental", "overlay_transmission"),
    "overlay_experimental_transmission": (
        ".experimental",
        "overlay_experimental_transmission",
    ),
    "BatchTransmission": (".batch", "BatchTransmission"),
    "BatchTransmissionResult": (".batch", "BatchTransmissionResult"),
    "stack_results": (".batch", "stack_results"),
}


def __getattr__(name: str):
    """Load plotting, experimental, and batch helpers only when requested."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "TransmissionConfig",
    "TransmissionCompute",
    "TransmissionResult",
    "TransmissionModesResult",
    "TransmissionPlotConfig",
    "TransmissionPlotter",
    "overlay_transmission",
    "overlay_experimental_transmission",
    "BatchTransmission",
    "BatchTransmissionResult",
    "stack_results",
]
