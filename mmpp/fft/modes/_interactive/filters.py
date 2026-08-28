"""Shared filtering and spectrum-prep helpers for interactive FMR UI."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

try:
    from scipy.signal import find_peaks as scipy_find_peaks
except ImportError:  # pragma: no cover - optional dependency
    scipy_find_peaks = None  # type: ignore[assignment]

from ...filters.postprocess import (
    apply_baseline as _shared_apply_baseline,
)
from ...filters.postprocess import (
    apply_percentile_clip as _shared_apply_percentile_clip,
)
from ...filters.postprocess import (
    apply_smoothing as _shared_apply_smoothing,
)
from ...filters.postprocess import (
    apply_soft_threshold as _shared_apply_soft_threshold,
)

CARTESIAN_COMPONENT_NAMES = ("x", "y", "z")
TOPOLOGICAL_COMPONENT_NAMES = ("+", "-", "rho", "phi")
COMPONENT_NAMES = list(CARTESIAN_COMPONENT_NAMES + TOPOLOGICAL_COMPONENT_NAMES)

COMPONENT_LABELS = {
    "x": r"$m_x$",
    "y": r"$m_y$",
    "z": r"$m_z$",
    "+": r"$m_{+}$",
    "-": r"$m_{-}$",
    "rho": r"$m_{\rho}$",
    "phi": r"$m_{\phi}$",
}
_COMPONENT_INDEX = {name: idx for idx, name in enumerate(CARTESIAN_COMPONENT_NAMES)}


@dataclass
class SpectrumFilterState:
    """Runtime filter state for spectrum processing."""

    freq_min: float
    freq_max: float
    smooth_filter: str = "none"
    smooth_window: int = 7
    smooth_sigma: float = 1.0
    baseline_mode: str = "none"
    clip_percentile_low: float = 0.0
    clip_percentile_high: float = 100.0
    soft_threshold_percentile: float = 0.0
    normalize: bool = True
    log_scale: bool = False

    def __post_init__(self) -> None:
        self.freq_min = float(self.freq_min)
        self.freq_max = float(self.freq_max)
        if not np.isfinite(self.freq_min) or not np.isfinite(self.freq_max):
            raise ValueError("Frequency filter bounds must be finite")

        self.smooth_filter = str(self.smooth_filter).strip().lower()
        valid_smoothing = {
            "none",
            "",
            "moving",
            "moving_average",
            "gaussian",
            "gaussian_smooth",
            "savgol",
            "savgol_smooth",
        }
        if self.smooth_filter not in valid_smoothing:
            raise ValueError(f"Unknown smoothing filter: {self.smooth_filter!r}")
        if (
            isinstance(self.smooth_window, (bool, np.bool_))
            or int(self.smooth_window) != self.smooth_window
            or int(self.smooth_window) < 1
        ):
            raise ValueError("smooth_window must be a positive integer")
        self.smooth_window = int(self.smooth_window)
        self.smooth_sigma = float(self.smooth_sigma)
        if not np.isfinite(self.smooth_sigma) or self.smooth_sigma < 0:
            raise ValueError("smooth_sigma must be finite and non-negative")

        self.baseline_mode = str(self.baseline_mode).strip().lower()
        if self.baseline_mode not in {"none", "mean", "median", "linear"}:
            raise ValueError(f"Unknown baseline mode: {self.baseline_mode!r}")
        self.clip_percentile_low = float(self.clip_percentile_low)
        self.clip_percentile_high = float(self.clip_percentile_high)
        if not (
            np.isfinite(self.clip_percentile_low)
            and np.isfinite(self.clip_percentile_high)
            and 0 <= self.clip_percentile_low <= self.clip_percentile_high <= 100
        ):
            raise ValueError("clip percentiles must satisfy 0 <= low <= high <= 100")
        self.soft_threshold_percentile = float(self.soft_threshold_percentile)
        if not (
            np.isfinite(self.soft_threshold_percentile)
            and 0 <= self.soft_threshold_percentile <= 100
        ):
            raise ValueError("soft_threshold_percentile must be in [0, 100]")
        if not isinstance(self.normalize, (bool, np.bool_)):
            raise TypeError("normalize must be boolean")
        if not isinstance(self.log_scale, (bool, np.bool_)):
            raise TypeError("log_scale must be boolean")
        self.normalize = bool(self.normalize)
        self.log_scale = bool(self.log_scale)


def _component_from_label(label: str | None) -> str | None:
    """Infer component key from a label like '$m_x$' or 'my'."""
    if not label:
        return None
    text = re.sub(r"[\s$\\{}_^]", "", str(label).strip().lower())
    match = re.fullmatch(r"m?([xyz])", text)
    if match:
        return match.group(1)
    return None


def _to_ghz(frequencies: np.ndarray) -> np.ndarray:
    """Convert frequencies to GHz when input appears to be in Hz."""
    freqs = np.asarray(frequencies, dtype=float)
    if freqs.size == 0:
        return freqs

    max_abs = float(np.nanmax(np.abs(freqs)))
    if max_abs > 1e6:
        return freqs / 1e9
    return freqs


def _to_power(spectrum: np.ndarray) -> np.ndarray:
    """Convert complex or amplitude spectrum to non-negative power-like data."""
    spec = np.asarray(spectrum)
    if spec.size == 0:
        return spec.astype(float)

    if np.iscomplexobj(spec):
        return np.abs(spec) ** 2

    spec = np.asarray(spec, dtype=float)
    if np.nanmin(spec) < 0:
        return np.abs(spec)
    return spec


def component_plot_label(component: str) -> str:
    """Return a publication-friendly label for a component key."""
    key = str(component).strip().lower()
    return COMPONENT_LABELS.get(key, f"$m_{{{key}}}$")


def _normalize_component_key(component: int | str) -> str | None:
    """Normalize a single component selector token to canonical key."""
    key: str | None = None
    if isinstance(component, int):
        if 0 <= component < len(CARTESIAN_COMPONENT_NAMES):
            key = CARTESIAN_COMPONENT_NAMES[component]
    elif isinstance(component, str):
        text = component.strip().lower().replace("_", "")
        if text.startswith("m") and len(text) > 1:
            text = text[1:]
        if text in COMPONENT_NAMES:
            key = text
    return key


def normalize_component_selection(
    components: Sequence[int | str] | None,
    available: Sequence[str] | None = None,
) -> list[str]:
    """Normalize mixed component input to canonical component keys."""
    if components is None:
        normalized = list(CARTESIAN_COMPONENT_NAMES)
    else:
        normalized = []
        for comp in components:
            key = _normalize_component_key(comp)
            if key is not None and key not in normalized:
                normalized.append(key)

    if available:
        available_set = {str(comp).strip().lower() for comp in available}
        allowed = [
            c
            for c in normalized
            if c in available_set or c in TOPOLOGICAL_COMPONENT_NAMES
        ]
        if allowed:
            return allowed
        return [str(available[0]).strip().lower()]

    return normalized or ["z"]


def normalize_spectrum_component_selection(
    components: Sequence[int | str] | None,
    available: Sequence[str] | None,
) -> list[str]:
    """Normalize selection strictly to available spectrum-trace components."""
    available_keys = []
    if available:
        for comp in available:
            key = str(comp).strip().lower()
            if key and key not in available_keys:
                available_keys.append(key)
    if not available_keys:
        return []

    normalized = normalize_component_selection(
        components,
        available=available_keys,
    )
    selected = [comp for comp in normalized if comp in available_keys]
    return selected or list(available_keys)


def _extract_cartesian_components(
    mode_array: np.ndarray,
    *,
    singleton_component: str = "z",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (mx, my, mz) arrays from a mode tensor with robust fallbacks."""
    arr = np.asanyarray(mode_array)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    if arr.ndim < 3:
        raise ValueError(f"Unsupported mode array shape: {arr.shape}")

    ny, nx = arr.shape[:2]
    zeros = np.zeros((ny, nx), dtype=arr.dtype)

    if arr.shape[-1] == 1:
        selected = arr[:, :, 0]
        if singleton_component == "x":
            return selected, zeros, zeros
        if singleton_component == "y":
            return zeros, selected, zeros
        if singleton_component == "z":
            return zeros, zeros, selected
        raise ValueError(
            "singleton_component must be one of 'x', 'y', or 'z', "
            f"got {singleton_component!r}"
        )
    if arr.shape[-1] == 2:
        return arr[:, :, 0], arr[:, :, 1], zeros
    return arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]


def resolve_mode_components(
    mode_array: np.ndarray,
    components: Sequence[str],
) -> dict[str, np.ndarray]:
    """Resolve requested cartesian/topological components for mode rendering."""
    requested = [
        key
        for key in normalize_component_selection(components)
        if key in COMPONENT_NAMES
    ]
    if not requested:
        requested = ["z"]

    arr = np.asarray(mode_array)
    singleton_component = "z"
    if arr.ndim >= 3 and arr.shape[-1] == 1:
        requested_cartesian = [
            key for key in requested if key in CARTESIAN_COMPONENT_NAMES
        ]
        requested_topological = [
            key for key in requested if key in TOPOLOGICAL_COMPONENT_NAMES
        ]
        if requested_topological:
            raise ValueError(
                "Circular/cylindrical components require both mx and my; "
                "the selected mode view contains only one component"
            )
        if len(requested_cartesian) != 1:
            raise ValueError(
                "A singleton mode tensor requires exactly one requested "
                "cartesian component so its identity is unambiguous"
            )
        singleton_component = requested_cartesian[0]

    m_x, m_y, m_z = _extract_cartesian_components(
        arr,
        singleton_component=singleton_component,
    )
    resolved: dict[str, np.ndarray] = {
        "x": m_x,
        "y": m_y,
        "z": m_z,
    }

    needs_topological = any(key in TOPOLOGICAL_COMPONENT_NAMES for key in requested)
    if needs_topological:
        try:
            from ..vortex_optics import VortexOptics

            resolved = VortexOptics.resolve_physical_components(
                m_x,
                m_y,
                m_z,
                list(requested),
            )
        except ImportError:
            # Optional visualization dependencies must not break cartesian data.
            pass

    return {key: resolved[key] for key in requested if key in resolved}


def collapse_spectrum_components(
    spectrum_power: np.ndarray,
    component_hint: str | None = None,
    *,
    single_component: bool = False,
) -> dict[str, np.ndarray]:
    """Collapse arbitrary spectrum shapes into per-component 1D traces."""
    spec = np.asarray(spectrum_power)
    if spec.ndim == 0:
        return {component_hint or "z": np.asarray([float(spec)])}

    if spec.ndim == 1:
        key = component_hint or "z"
        return {key: spec.astype(float, copy=False)}

    if single_component:
        reduced = np.mean(spec, axis=tuple(range(1, spec.ndim)))
        key = component_hint or "z"
        return {key: np.asarray(reduced, dtype=float)}

    if spec.shape[-1] <= 3:
        if spec.ndim == 2:
            traces = spec
        else:
            spatial_axes = tuple(range(1, spec.ndim - 1))
            traces = np.mean(spec, axis=spatial_axes)

        out: dict[str, np.ndarray] = {}
        n_comp = min(traces.shape[-1], 3)
        for idx in range(n_comp):
            out[CARTESIAN_COMPONENT_NAMES[idx]] = np.asarray(
                traces[:, idx], dtype=float
            )
        return out

    reduced = np.mean(spec, axis=tuple(range(1, spec.ndim)))
    key = component_hint or "z"
    return {key: np.asarray(reduced, dtype=float)}


def _apply_smoothing(
    values: np.ndarray,
    smooth_filter: str,
    smooth_window: int,
    smooth_sigma: float,
) -> np.ndarray:
    return _shared_apply_smoothing(
        values,
        smooth_filter=smooth_filter,
        smooth_window=smooth_window,
        smooth_sigma=smooth_sigma,
    )


def _remove_baseline(values: np.ndarray, baseline_mode: str) -> np.ndarray:
    return _shared_apply_baseline(values, mode=baseline_mode)


def _apply_percentile_clip(
    values: np.ndarray,
    clip_percentile_low: float,
    clip_percentile_high: float,
) -> np.ndarray:
    return _shared_apply_percentile_clip(
        values,
        low=clip_percentile_low,
        high=clip_percentile_high,
    )


def _apply_soft_threshold(values: np.ndarray, percentile: float) -> np.ndarray:
    return _shared_apply_soft_threshold(values, percentile=percentile)


def apply_spectrum_filters(
    frequencies_ghz: np.ndarray,
    component_power: dict[str, np.ndarray],
    filters: SpectrumFilterState,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Apply frequency range, smoothing, normalization and optional log transform."""
    freqs = np.asarray(frequencies_ghz, dtype=float)
    if freqs.size == 0:
        return freqs, {
            key: np.asarray(val, dtype=float) for key, val in component_power.items()
        }

    fmin = float(min(filters.freq_min, filters.freq_max))
    fmax = float(max(filters.freq_min, filters.freq_max))
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        finite_freqs = freqs[np.isfinite(freqs)]
        available = (
            f"{float(np.min(finite_freqs)):.6g}–{float(np.max(finite_freqs)):.6g} GHz"
            if finite_freqs.size
            else "no finite frequencies"
        )
        raise ValueError(
            f"Requested frequency range {fmin:.6g}–{fmax:.6g} GHz does not "
            f"overlap available data ({available})"
        )

    filtered_freqs = freqs[mask]
    output: dict[str, np.ndarray] = {}

    for comp, values in component_power.items():
        arr = np.asarray(values, dtype=float)
        if arr.shape[0] != freqs.shape[0]:
            raise ValueError(
                f"Spectrum component '{comp}' has {arr.shape[0]} samples, "
                f"but the frequency axis has {freqs.shape[0]}"
            )
        local_mask = mask
        local_freqs = freqs

        sub = arr[local_mask]
        if sub.size == 0:
            sub = np.asarray([0.0], dtype=float)
            local_freqs = np.asarray([fmin], dtype=float)
            filtered_freqs = local_freqs

        sub = _apply_smoothing(
            sub,
            smooth_filter=filters.smooth_filter,
            smooth_window=filters.smooth_window,
            smooth_sigma=filters.smooth_sigma,
        )
        sub = _remove_baseline(sub, filters.baseline_mode)
        sub = _apply_percentile_clip(
            sub,
            clip_percentile_low=filters.clip_percentile_low,
            clip_percentile_high=filters.clip_percentile_high,
        )
        sub = _apply_soft_threshold(sub, filters.soft_threshold_percentile)

        finite = np.isfinite(sub)
        if np.any(finite):
            sub = np.where(finite, sub, np.nanmedian(sub[finite]))
        else:
            sub = np.zeros_like(sub)

        min_val = float(np.nanmin(sub)) if sub.size else 0.0
        if np.isfinite(min_val) and min_val < 0:
            sub = sub - min_val

        if filters.normalize and sub.size:
            vmax = float(np.nanmax(sub))
            if vmax > 0:
                sub = sub / vmax

        if filters.log_scale:
            sub = np.log10(np.clip(sub, 1e-12, None))

        output[comp] = np.asarray(sub, dtype=float)

        if filtered_freqs.shape[0] != local_freqs[local_mask].shape[0]:
            filtered_freqs = local_freqs[local_mask]

    return filtered_freqs, output


def detect_spectrum_peaks(
    frequencies_ghz: np.ndarray,
    spectrum_1d: np.ndarray,
    min_prominence: float = 0.05,
    min_distance: int = 5,
) -> list[tuple[float, float]]:
    """Detect local peaks and return list of ``(frequency, amplitude)`` tuples."""
    freqs = np.asarray(frequencies_ghz, dtype=float)
    values = np.asarray(spectrum_1d, dtype=float)

    if freqs.ndim != 1 or values.ndim != 1:
        raise ValueError("frequencies and spectrum must be one-dimensional")
    if freqs.shape[0] != values.shape[0]:
        raise ValueError("frequencies and spectrum must have matching lengths")
    if not np.all(np.isfinite(freqs)):
        raise ValueError("frequencies must contain only finite values")
    prominence_fraction = float(min_prominence)
    if not np.isfinite(prominence_fraction) or not 0 <= prominence_fraction <= 1:
        raise ValueError("min_prominence must be a fraction in [0, 1]")
    if (
        isinstance(min_distance, (bool, np.bool_))
        or int(min_distance) != min_distance
        or int(min_distance) < 1
    ):
        raise ValueError("min_distance must be a positive integer")
    distance = int(min_distance)

    if freqs.size < 3:
        return []

    finite = np.isfinite(values)
    if not np.any(finite):
        return []

    data = values.copy()
    data[~finite] = np.nanmin(data[finite])
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    dynamic_range = data_max - data_min
    if dynamic_range <= 0:
        return []
    prominence = prominence_fraction * dynamic_range

    if scipy_find_peaks is not None:
        idx, _ = scipy_find_peaks(
            data,
            prominence=prominence,
            distance=distance,
        )
    else:
        candidates: list[int] = []
        for i in range(1, data.size - 1):
            if data[i] > data[i - 1] and data[i] > data[i + 1]:
                left_base = float(np.min(data[: i + 1]))
                right_base = float(np.min(data[i:]))
                if data[i] - max(left_base, right_base) >= prominence:
                    candidates.append(i)
        selected: list[int] = []
        for index in sorted(candidates, key=lambda i: data[i], reverse=True):
            if all(abs(index - other) >= distance for other in selected):
                selected.append(index)
        idx = np.asarray(selected, dtype=int)

    peaks = [(float(freqs[i]), float(data[i])) for i in idx]
    peaks.sort(key=lambda item: item[1], reverse=True)
    return peaks


__all__ = [
    "CARTESIAN_COMPONENT_NAMES",
    "TOPOLOGICAL_COMPONENT_NAMES",
    "COMPONENT_LABELS",
    "COMPONENT_NAMES",
    "_COMPONENT_INDEX",
    "component_plot_label",
    "SpectrumFilterState",
    "_component_from_label",
    "_to_ghz",
    "_to_power",
    "normalize_component_selection",
    "normalize_spectrum_component_selection",
    "resolve_mode_components",
    "collapse_spectrum_components",
    "apply_spectrum_filters",
    "detect_spectrum_peaks",
]
