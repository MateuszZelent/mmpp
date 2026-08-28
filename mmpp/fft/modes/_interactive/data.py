"""Spectrum data-preparation helpers for :mod:`mmpp.fft.modes.interactive`."""

from __future__ import annotations

from typing import Any

import numpy as np

from .filters import (
    _component_from_label,
    _to_ghz,
    _to_power,
    apply_spectrum_filters,
    collapse_spectrum_components,
    detect_spectrum_peaks,
)


def load_spectrum_data(explorer: Any) -> None:
    """Load and normalize spectrum data from available source."""
    freqs: np.ndarray | None = None
    spectrum: np.ndarray | None = None
    frequencies_are_ghz = False
    component_hint = _component_from_label(explorer._component_label)
    single_component = component_hint is not None

    if explorer.spectrum_result is not None:
        if hasattr(explorer.spectrum_result, "frequencies_ghz"):
            freqs = np.asarray(explorer.spectrum_result.frequencies_ghz, dtype=float)
            frequencies_are_ghz = True
        else:
            freqs = np.asarray(
                getattr(explorer.spectrum_result, "frequencies", []), dtype=float
            )
        if hasattr(explorer.spectrum_result, "power"):
            spectrum = np.asarray(explorer.spectrum_result.power)
        elif hasattr(explorer.spectrum_result, "spectrum"):
            spectrum = _to_power(np.asarray(explorer.spectrum_result.spectrum))
        else:
            spectrum = None

        hint_from_result = _component_from_label(
            getattr(explorer.spectrum_result, "component_label", None)
        )
        component_hint = hint_from_result or component_hint
        single_component = bool(
            getattr(explorer.spectrum_result, "_single_component", False)
            or hint_from_result is not None
        )

    if (freqs is None or spectrum is None) and explorer.data_loader is not None:
        loaded_freqs, loaded_spectrum, loaded_label = (
            explorer.data_loader.load_spectrum()
        )
        freqs = np.asarray(loaded_freqs, dtype=float)
        frequencies_are_ghz = True
        spectrum = _to_power(np.asarray(loaded_spectrum))
        component_hint = _component_from_label(loaded_label) or component_hint
        single_component = component_hint is not None

    if (freqs is None or spectrum is None) and explorer.analyzer is not None:
        if getattr(explorer.analyzer, "frequencies", None) is not None:
            freqs = np.asarray(explorer.analyzer.frequencies, dtype=float)
            frequencies_are_ghz = True
        if getattr(explorer.analyzer, "spectrum", None) is not None:
            spectrum = _to_power(np.asarray(explorer.analyzer.spectrum))

    if freqs is None or spectrum is None:
        raise ValueError(
            "No spectrum data available. Provide spectrum_result, data_loader, or analyzer."
        )

    freqs_ghz = freqs if frequencies_are_ghz else _to_ghz(freqs)
    component_power = collapse_spectrum_components(
        _to_power(spectrum),
        component_hint,
        single_component=single_component,
    )

    trimmed: dict[str, np.ndarray] = {}
    for comp, values in component_power.items():
        arr = np.asarray(values, dtype=float)
        if arr.shape[0] != freqs_ghz.shape[0]:
            raise ValueError(
                f"Spectrum component '{comp}' has {arr.shape[0]} samples, "
                f"but the frequency axis has {freqs_ghz.shape[0]}"
            )
        if arr.shape[0] == 0:
            continue
        trimmed[comp] = arr

    if not trimmed:
        raise ValueError("Spectrum data is empty after preprocessing")

    explorer._raw_frequencies_ghz = freqs_ghz
    explorer._raw_component_power = trimmed
    explorer._available_components = list(explorer._raw_component_power.keys())


def recompute_filtered_spectrum(explorer: Any) -> None:
    """Recompute filtered traces and peak list from current filter state."""
    explorer._filtered_frequencies_ghz, filtered = apply_spectrum_filters(
        explorer._raw_frequencies_ghz,
        explorer._raw_component_power,
        explorer._filter_state,
    )

    selected = {
        comp: values
        for comp, values in filtered.items()
        if comp in explorer._spectrum_components
    }
    if not selected:
        # Keep mode-component selection independent and fallback only for
        # traces used in the top spectrum panel.
        selected = dict(filtered)

    explorer._filtered_component_power = selected

    if explorer._show_peaks:
        stacked = np.vstack(list(selected.values()))
        avg_trace = np.mean(stacked, axis=0)
        explorer._peaks = detect_spectrum_peaks(
            explorer._filtered_frequencies_ghz,
            avg_trace,
            min_prominence=explorer._peak_prominence,
            min_distance=explorer._peak_distance,
        )
    else:
        explorer._peaks = []


def initialize_frequency(explorer: Any, initial_frequency: float | None) -> None:
    """Set current frequency based on initial request, peaks, or center."""
    if initial_frequency is not None:
        explorer._current_frequency_ghz = float(initial_frequency)
        return

    if explorer._peaks:
        explorer._current_frequency_ghz = float(explorer._peaks[0][0])
        return

    if explorer._filtered_frequencies_ghz.size:
        center = explorer._filtered_frequencies_ghz.size // 2
        explorer._current_frequency_ghz = float(
            explorer._filtered_frequencies_ghz[center]
        )
        return

    explorer._current_frequency_ghz = None


__all__ = [
    "load_spectrum_data",
    "recompute_filtered_spectrum",
    "initialize_frequency",
]
