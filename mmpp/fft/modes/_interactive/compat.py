"""Backward-compatible plotting helper for :mod:`mmpp.fft.modes.interactive`."""

from __future__ import annotations

from typing import Any

import numpy as np

from .filters import (
    SpectrumFilterState,
    _component_from_label,
    _to_ghz,
    _to_power,
    apply_spectrum_filters,
    collapse_spectrum_components,
    detect_spectrum_peaks,
)


def _get_freq_scale(freq_unit: str) -> float:
    mapping = {
        "hz": 1e9,
        "khz": 1e6,
        "mhz": 1e3,
        "ghz": 1.0,
        "thz": 1e-3,
    }
    key = str(freq_unit).strip().lower()
    if key not in mapping:
        raise ValueError("freq_unit must be Hz, kHz, MHz, GHz, or THz")
    return float(mapping[key])


def plot(
    data_loader: Any,
    log_scale: bool = False,
    normalize: bool = True,
    freq_unit: str = "GHz",
    show_peaks: bool = True,
    freq_min: float | None = None,
    freq_max: float | None = None,
    smooth_filter: str = "none",
    smooth_window: int = 7,
    smooth_sigma: float = 1.0,
    baseline_mode: str = "none",
    clip_percentile_low: float = 0.0,
    clip_percentile_high: float = 100.0,
    soft_threshold_percentile: float = 0.0,
    peak_prominence: float = 0.05,
    peak_distance: int = 5,
    title: str | None = None,
    dpi: int = 100,
    figsize: tuple[float, float] = (12.0, 6.0),
) -> Any:
    """Simple static spectrum plot compatibility helper."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Matplotlib required") from exc

    frequencies, spectrum, component_label = data_loader.load_spectrum()
    freqs_ghz = _to_ghz(np.asarray(frequencies, dtype=float))
    component_hint = _component_from_label(component_label)
    power = collapse_spectrum_components(
        _to_power(np.asarray(spectrum)),
        component_hint,
        single_component=component_hint is not None,
    )

    data_fmin = float(np.nanmin(freqs_ghz))
    data_fmax = float(np.nanmax(freqs_ghz))
    init_fmin = data_fmin if freq_min is None else float(freq_min)
    init_fmax = data_fmax if freq_max is None else float(freq_max)
    init_fmin = float(np.clip(init_fmin, data_fmin, data_fmax))
    init_fmax = float(np.clip(init_fmax, data_fmin, data_fmax))
    if init_fmin > init_fmax:
        init_fmin, init_fmax = init_fmax, init_fmin

    state = SpectrumFilterState(
        freq_min=init_fmin,
        freq_max=init_fmax,
        smooth_filter=str(smooth_filter),
        smooth_window=int(smooth_window),
        smooth_sigma=float(smooth_sigma),
        baseline_mode=str(baseline_mode),
        clip_percentile_low=float(clip_percentile_low),
        clip_percentile_high=float(clip_percentile_high),
        soft_threshold_percentile=float(soft_threshold_percentile),
        normalize=bool(normalize),
        log_scale=bool(log_scale),
    )
    freqs_filtered, traces = apply_spectrum_filters(freqs_ghz, power, state)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    scale = _get_freq_scale(freq_unit)
    x = freqs_filtered * scale

    color_map = {"x": "#E76F51", "y": "#2A9D8F", "z": "#457B9D"}
    for comp, trace in traces.items():
        ax.plot(
            x,
            trace,
            linewidth=1.6,
            color=color_map.get(comp, "#4C78A8"),
            label=f"m_{comp}",
        )

    if show_peaks:
        stacked = np.vstack(list(traces.values()))
        peaks = detect_spectrum_peaks(
            freqs_filtered,
            np.mean(stacked, axis=0),
            min_prominence=float(peak_prominence),
            min_distance=int(peak_distance),
        )
        for freq_ghz, amp in peaks:
            ax.plot(freq_ghz * scale, amp, "o", color="#D62828", markersize=4)

    ax.set_xlabel(f"Frequency ({freq_unit})")
    ax.set_ylabel("log10(Power)" if log_scale else "Power")
    ax.set_title(title or "FFT Power Spectrum")
    ax.grid(True, alpha=0.25, linestyle="--")
    if len(traces) > 1:
        ax.legend(loc="upper right")

    try:
        fig.tight_layout()
    except Exception:
        pass

    return fig


__all__ = ["plot"]
