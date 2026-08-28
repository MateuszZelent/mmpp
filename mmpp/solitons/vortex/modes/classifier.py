"""Mode classification logic for vortex dynamics."""

from __future__ import annotations

import warnings

import numpy as np

from ..core.models import TrajectoryResult
from ..spectrum.gyration import compute_breathing_spectrum, compute_gyration_spectrum
from .azimuthal import estimate_azimuthal_index
from .models import VortexModeResult
from .radial import estimate_radial_index

try:
    from scipy.signal import find_peaks

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    find_peaks = None  # type: ignore[assignment]
    SCIPY_AVAILABLE = False


def _simple_peak_indices(values: np.ndarray, max_peaks: int = 8) -> np.ndarray:
    """SciPy-free local maxima detector with amplitude sorting."""
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return np.array([], dtype=int)

    local = np.where((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]))[0] + 1
    if local.size == 0:
        return np.array([], dtype=int)

    order = np.argsort(values[local])[::-1]
    return local[order][:max_peaks]


def _find_peak_indices(
    power: np.ndarray,
    *,
    min_prominence: float = 0.05,
    max_modes: int = 8,
) -> np.ndarray:
    """Find dominant peak indices with optional SciPy implementation."""
    power = np.asarray(power, dtype=float)
    if power.size == 0:
        return np.array([], dtype=int)

    amplitude_floor = float(np.max(power)) * float(min_prominence)
    if amplitude_floor <= 0.0:
        return np.array([], dtype=int)

    if SCIPY_AVAILABLE and find_peaks is not None:
        idx, _ = find_peaks(power, prominence=amplitude_floor)
        if idx.size == 0:
            return idx
        order = np.argsort(power[idx])[::-1]
        return idx[order][:max_modes]

    warnings.warn(
        "SciPy is unavailable; using simple peak detector fallback.",
        RuntimeWarning,
        stacklevel=2,
    )
    idx = _simple_peak_indices(power, max_peaks=max_modes * 2)
    if idx.size == 0:
        return idx
    return idx[power[idx] >= amplitude_floor][:max_modes]


def _classify_mode_type(*, harmonic: float, source: str) -> tuple[str, float]:
    """Classify mode type from harmonic ratio and source spectrum."""
    harmonic_abs = abs(float(harmonic))
    source_norm = source.lower()

    if source_norm == "gyration":
        if harmonic_abs <= 1.4:
            return "gyration", float(
                np.clip(1.0 - abs(harmonic_abs - 1.0) / 0.4, 0.0, 1.0)
            )
        if harmonic_abs <= 2.4:
            return "azimuthal", float(
                np.clip(1.0 - abs(harmonic_abs - 2.0) / 0.6, 0.0, 1.0)
            )
        return "azimuthal", 0.4

    if source_norm == "breathing":
        if harmonic_abs >= 1.5:
            return "breathing", float(np.clip((harmonic_abs - 1.5) / 1.0, 0.5, 1.0))
        return "gyration", 0.3

    return "unknown", 0.1


def classify_modes_from_trajectory(
    trajectory: TrajectoryResult,
    *,
    spectrum_method: str = "welch",
    min_prominence: float = 0.05,
    max_modes: int = 6,
) -> list[VortexModeResult]:
    """Classify vortex dynamical modes based on trajectory-derived spectra."""
    gyr = compute_gyration_spectrum(trajectory, method=spectrum_method)
    br = compute_breathing_spectrum(trajectory, method=spectrum_method)

    if gyr.frequencies.size == 0:
        return []

    base_idx = int(np.argmax(gyr.power))
    base_freq = float(gyr.frequencies[base_idx])
    base_power = float(gyr.power[base_idx])
    if base_freq <= 0.0:
        return []

    results: list[VortexModeResult] = []
    rotation_sense = trajectory.rotation_sense

    def _collect(frequencies: np.ndarray, power: np.ndarray, source: str):
        peak_indices = _find_peak_indices(
            power,
            min_prominence=min_prominence,
            max_modes=max_modes,
        )

        for idx in peak_indices:
            freq = float(frequencies[idx])
            pwr = float(power[idx])
            if freq <= 0.0:
                continue

            harmonic = freq / base_freq
            mode_type, base_conf = _classify_mode_type(harmonic=harmonic, source=source)

            m_idx = estimate_azimuthal_index(
                mode_type=mode_type, rotation_sense=rotation_sense
            )
            n_idx = estimate_radial_index(mode_type=mode_type, harmonic=harmonic)

            rel_power = pwr / max(base_power, 1e-30)
            confidence = float(
                np.clip(0.7 * base_conf + 0.3 * min(rel_power, 1.0), 0.0, 1.0)
            )

            results.append(
                VortexModeResult(
                    m_index=int(m_idx),
                    n_index=int(n_idx),
                    mode_type=mode_type,
                    rotation_sense=rotation_sense,
                    confidence=confidence,
                    frequency_hz=freq,
                    power=pwr,
                    source=source,
                    metadata={
                        "harmonic": harmonic,
                        "base_frequency_hz": base_freq,
                        "relative_power": rel_power,
                        "spectrum_method": spectrum_method,
                    },
                )
            )

    _collect(gyr.frequencies, gyr.power, "gyration")
    _collect(br.frequencies, br.power, "breathing")

    if not results:
        return []

    # Merge near-duplicate peaks (same frequency from two sources) keeping higher confidence.
    results.sort(key=lambda item: item.frequency_hz)
    merged: list[VortexModeResult] = []
    merge_tol = max(base_freq * 0.05, 1e6)

    for item in results:
        if not merged:
            merged.append(item)
            continue

        prev = merged[-1]
        if abs(item.frequency_hz - prev.frequency_hz) <= merge_tol:
            if item.confidence > prev.confidence:
                merged[-1] = item
        else:
            merged.append(item)

    merged.sort(key=lambda item: item.power, reverse=True)
    return merged[:max_modes]


class VortexModesClassifier:
    """Stateful helper wrapping mode classification pipeline."""

    def __init__(self, trajectory: TrajectoryResult):
        self._trajectory = trajectory
        self._cache: list[VortexModeResult] | None = None

    def classify_all(
        self,
        *,
        spectrum_method: str = "welch",
        min_prominence: float = 0.05,
        max_modes: int = 6,
        force: bool = False,
    ) -> list[VortexModeResult]:
        """Classify all dominant modes."""
        if self._cache is None or force:
            self._cache = classify_modes_from_trajectory(
                self._trajectory,
                spectrum_method=spectrum_method,
                min_prominence=min_prominence,
                max_modes=max_modes,
            )
        return list(self._cache)

    def classify(
        self,
        *,
        frequency_hz: float,
        spectrum_method: str = "welch",
    ) -> VortexModeResult:
        """Classify nearest mode to the provided frequency."""
        modes = self.classify_all(spectrum_method=spectrum_method)
        if not modes:
            return VortexModeResult(
                m_index=0,
                n_index=0,
                mode_type="unknown",
                confidence=0.0,
                frequency_hz=float(frequency_hz),
                source="none",
            )

        target = float(frequency_hz)
        return min(modes, key=lambda item: abs(item.frequency_hz - target))

    def classify_fft_mode_data(self, mode_data, **kwargs) -> VortexModeResult:
        """Classify single FFT-mode object via legacy advanced classifier bridge."""
        try:
            from mmpp.fft.vortex_classifier import AdvancedVortexClassifier
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Legacy FFT vortex classifier is unavailable; cannot classify FFT mode data."
            ) from exc

        classifier = AdvancedVortexClassifier()
        legacy = classifier.classify_mode(mode_data, **kwargs)
        return VortexModeResult(
            m_index=int(getattr(legacy, "m_index", 0)),
            n_index=int(getattr(legacy, "n_index", 0)),
            l_index=getattr(legacy, "l_index", None),
            mode_type=str(getattr(legacy, "mode_type", "unknown")),
            rotation_sense=str(getattr(legacy, "rotation_sense", "unknown")),
            confidence=float(getattr(legacy, "confidence", 0.0)),
            frequency_hz=float(getattr(legacy, "frequency", 0.0)) * 1e9,
            power=float(getattr(legacy, "E_parallel_frac", 0.0)),
            source="fft_mode_data",
            metadata={
                "legacy_notes": list(getattr(legacy, "notes", [])),
                "legacy_core_position": tuple(
                    getattr(legacy, "core_position", (0.0, 0.0))
                ),
            },
        )
