"""Shared dataset-aware routing to the FFT spectrum and mode explorer."""

from __future__ import annotations

from typing import Any


def dataset_view(interface: Any) -> Any:
    """Return the dataset wrapper selected by a soliton interface."""
    bound_view = getattr(interface, "_dataset_view", None)
    if bound_view is not None:
        return bound_view
    dataset_name = interface.dataset_name
    if dataset_name is None:
        raise ValueError(
            "No magnetisation dataset is available for spectrum or mode analysis."
        )
    data = getattr(interface._job, dataset_name)
    slice_info = getattr(interface, "_slice_info", None)
    if slice_info is not None and hasattr(data, "__getitem__"):
        data = data[slice_info]
    return data


def interactive_spectrum_modes(interface: Any, **kwargs: Any) -> Any:
    """Open the combined FFT spectrum and spatial-mode explorer."""
    data = dataset_view(interface)
    fft = getattr(data, "fft", None)
    if fft is None:
        raise RuntimeError(
            "Interactive spectrum and mode visualization requires the FFT "
            "accessor. Install MMPP with the 'fft' and 'interactive' extras."
        )
    return fft.modes.interactive_spectrum(**kwargs)


__all__ = ["dataset_view", "interactive_spectrum_modes"]
