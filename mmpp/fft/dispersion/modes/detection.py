"""
Automatic detection of Brillouin zone parameters from dispersion data.

Provides methods to estimate:
- Lattice constant from periodicity in S(k,f)
- Optimal number of BZ periods
- FBZ boundaries
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..models import DispersionResult1D

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LowIntensityInterval:
    """Frequency interval with weak observed response, not a proven band gap."""

    f_low: float
    f_high: float
    mean_relative_intensity: float
    status: str = "low_observed_intensity_candidate"


class BrillouinZoneDetector:
    """
    Automatic detection of Brillouin zone parameters from dispersion data.

    Methods
    -------
    detect_lattice_constant(result, method='autocorrelation')
        Estimate lattice constant from dispersion periodicity
    suggest_n_periods(k_axis, a)
        Suggest number of BZ periods needed for full coverage
    find_band_gaps(result)
        Detect frequency gaps in the dispersion

    Example
    -------
    >>> detector = BrillouinZoneDetector()
    >>> a = detector.detect_lattice_constant(dispersion_result)
    >>> print(f"Detected lattice constant: {a*1e9:.1f} nm")
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.last_detection: dict[str, Any] | None = None
        self._last_quality: float | None = None
        self._last_failure: str | None = None

    def detect_lattice_constant(
        self,
        result: DispersionResult1D,
        method: str = "autocorrelation",
        f_range: tuple[float, float] | None = None,
    ) -> float | None:
        """
        Detect lattice constant from periodicity in the dispersion relation.

        Parameters
        ----------
        result : DispersionResult1D
            Dispersion result to analyze
        method : str
            Detection method:
            - 'autocorrelation': Autocorrelation of k-averaged spectrum
            - 'fft': FFT of the dispersion to find periodicity
            - 'peak_spacing': Analyze spacing between dispersion branches
        f_range : tuple, optional
            Frequency range (Hz) to consider for detection

        Returns
        -------
        float or None
            Candidate lattice constant [m], or ``None`` when the data do not
            support a resolved periodicity. ``last_detection`` records method,
            quality, and failure status.
        """
        if method not in {"autocorrelation", "fft", "peak_spacing"}:
            raise ValueError(f"Unknown detection method: {method}")
        self._last_quality = None
        self._last_failure = None
        detector = {
            "autocorrelation": self._detect_via_autocorr,
            "fft": self._detect_via_fft,
            "peak_spacing": self._detect_via_peak_spacing,
        }[method]
        value = detector(result, f_range)
        self.last_detection = {
            "method": method,
            "value_m": value,
            "quality": self._last_quality,
            "status": "candidate" if value is not None else "not_detected",
            "reason": self._last_failure,
        }
        return value

    def _detect_via_autocorr(
        self,
        result: DispersionResult1D,
        f_range: tuple[float, float] | None = None,
    ) -> float | None:
        """
        Detect lattice constant via autocorrelation of the k-profile.

        The idea: if the dispersion has periodicity in k-space (due to BZ folding),
        the autocorrelation will show peaks at multiples of 2π/a.

        For magnonic crystals:
        - Typical lattice constants: 100nm - 2000nm
        - Corresponding BZ widths: 6.3e7 - 3.1e6 rad/m
        - We want to find the LARGEST significant periodicity (not fine structure)
        """
        S = result.S
        k_axis = result.k_axis
        f_axis = result.f_axis

        # Apply frequency filter - focus on positive frequencies with signal
        if f_range is not None:
            f_mask = (f_axis >= f_range[0]) & (f_axis <= f_range[1])
        else:
            # Auto select: positive frequencies, above noise floor
            f_mask = f_axis > 0
        S = S[:, f_mask]
        if S.size == 0 or k_axis.size < 8 or not np.all(np.isfinite(k_axis)):
            self._last_failure = "insufficient finite k/f samples"
            return None
        if np.any(np.diff(k_axis) <= 0):
            self._last_failure = "k axis must be strictly increasing"
            return None

        # Get k-profile weighted by log intensity to reduce dynamic range
        S_positive = np.maximum(S, 1e-20)
        S_log = np.log10(S_positive)
        S_mean = np.mean(S_log, axis=1)

        # Remove DC / mean to focus on periodic structure
        S_mean = S_mean - np.mean(S_mean)

        variation = float(np.max(np.abs(S_mean))) if S_mean.size else 0.0
        if not np.isfinite(variation) or variation <= 1e-10:
            self._last_failure = "k-profile has no resolved periodic variation"
            return None

        # Normalize
        S_mean = S_mean / (np.max(np.abs(S_mean)) + 1e-20)

        # Compute autocorrelation
        autocorr = np.correlate(S_mean, S_mean, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]  # Keep only positive lags

        # Normalize autocorrelation
        autocorr = autocorr / (autocorr[0] + 1e-20)

        dk = k_axis[1] - k_axis[0] if len(k_axis) > 1 else 1.0

        # Define physical constraints for magnonic crystals
        # Minimum lattice constant: 50nm → max period_k = 2π/50nm = 1.26e8 rad/m
        # Maximum lattice constant: 5μm → min period_k = 2π/5μm = 1.26e6 rad/m
        min_a = 50e-9
        max_a = 5e-6
        min_period_k = 2 * np.pi / max_a  # rad/m
        max_period_k = 2 * np.pi / min_a  # rad/m

        # Convert to lag indices
        min_lag_physical = max(5, int(min_period_k / dk))
        max_lag_physical = min(len(autocorr) - 1, int(max_period_k / dk))

        # Ensure valid range
        if min_lag_physical >= max_lag_physical or max_lag_physical <= 1:
            self._last_failure = "available k range cannot resolve physical periods"
            return None

        # Find ALL peaks in physical range
        peaks_in_range = []
        for lag in range(min_lag_physical, max_lag_physical):
            if (
                autocorr[lag] > autocorr[lag - 1]
                and autocorr[lag] > autocorr[lag + 1]
                and autocorr[lag] > 0.05
            ):  # Must be positive correlation
                period_k = lag * dk
                a = 2 * np.pi / period_k
                peaks_in_range.append(
                    {
                        "lag": lag,
                        "value": autocorr[lag],
                        "period_k": period_k,
                        "a": a,
                    }
                )

        if peaks_in_range:
            # Prefer the most prominent peak (highest correlation value)
            # But weight toward larger periods (larger a) as they are more likely BZ
            best_peak = max(
                peaks_in_range,
                key=lambda p: p["value"] * (1 + 0.1 * np.log10(p["a"] * 1e9)),
            )

            a = best_peak["a"]

            # Sanity check: reasonable for magnonic crystals
            if 50e-9 < a < 5e-6:
                logger.info(
                    "Autocorr detection: peak at lag %d (corr=%.2f, Δk=%.3e rad/m) → a=%.1f nm",
                    best_peak["lag"],
                    best_peak["value"],
                    best_peak["period_k"],
                    a * 1e9,
                )
                self._last_quality = float(np.clip(best_peak["value"], 0.0, 1.0))
                return a

        self._last_failure = "no autocorrelation peak met the periodicity criterion"
        return None

    def _detect_via_fft(
        self,
        result: DispersionResult1D,
        f_range: tuple[float, float] | None = None,
    ) -> float | None:
        """
        Detect lattice constant via FFT of the k-profile.

        Takes FFT of S(k) integrated over f to find spatial frequency peaks.
        Looking for periodicity in k-space that corresponds to BZ folding.
        """
        S = result.S
        k_axis = result.k_axis
        f_axis = result.f_axis

        if f_range is not None:
            f_mask = (f_axis >= f_range[0]) & (f_axis <= f_range[1])
        else:
            f_mask = f_axis > 0
        S = S[:, f_mask]
        if S.size == 0 or len(k_axis) < 8 or not np.all(np.isfinite(k_axis)):
            self._last_failure = "insufficient finite k/f samples"
            return None
        if np.any(np.diff(k_axis) <= 0):
            self._last_failure = "k axis must be strictly increasing"
            return None

        # Get k-profile using log to reduce dynamic range
        S_positive = np.maximum(S, 1e-20)
        S_k = np.mean(np.log10(S_positive), axis=1)
        S_k = S_k - np.mean(S_k)  # Remove DC

        # Apply window to reduce spectral leakage
        window = np.hanning(len(S_k))
        S_k = S_k * window

        # FFT of k-profile
        fft_result = np.fft.fft(S_k)
        fft_mag = np.abs(fft_result)

        # Frequency axis for the FFT
        n_k = len(k_axis)
        dk = k_axis[1] - k_axis[0] if n_k > 1 else 1.0
        fft_freq = np.fft.fftfreq(n_k, dk)  # cycles per rad/m

        # Only positive frequencies
        pos_mask = fft_freq > 0
        fft_freq = fft_freq[pos_mask]
        fft_mag = fft_mag[pos_mask]

        # Physical constraints for magnonic crystals: 50nm < a < 5μm
        # period_k = 2π/a, so freq = 1/period_k = a/(2π)
        min_a, max_a = 50e-9, 5e-6
        min_freq = min_a / (2 * np.pi)  # cycles per rad/m
        max_freq = max_a / (2 * np.pi)

        freq_mask = (fft_freq >= min_freq) & (fft_freq <= max_freq)

        if np.any(freq_mask):
            fft_freq_valid = fft_freq[freq_mask]
            fft_mag_valid = fft_mag[freq_mask]

            # Find peak in valid range
            peak_idx = np.argmax(fft_mag_valid)
            dominant_freq = fft_freq_valid[peak_idx]

            median_level = float(np.median(fft_mag_valid))
            peak_level = float(fft_mag_valid[peak_idx])
            prominence_ratio = peak_level / max(median_level, np.finfo(float).eps)
            if dominant_freq > 0 and prominence_ratio >= 3.0:
                # freq = a/(2π) → a = 2π * freq
                # Actually: period_k = 1/freq, a = 2π/period_k = 2π * freq
                a = 2 * np.pi * dominant_freq

                logger.info(
                    "FFT detection: dominant freq=%.3e cycles/rad → a=%.1f nm",
                    dominant_freq,
                    a * 1e9,
                )
                self._last_quality = float(
                    np.clip(1.0 - 1.0 / prominence_ratio, 0.0, 1.0)
                )
                return a

        self._last_failure = "no sufficiently prominent periodicity peak"
        return None

    def _detect_via_peak_spacing(
        self,
        result: DispersionResult1D,
        f_range: tuple[float, float] | None = None,
    ) -> float | None:
        """
        Detect lattice constant from spacing between dispersion branches.

        Analyzes at what k-spacing the dispersion pattern repeats.
        """
        S = result.S
        k_axis = result.k_axis
        f_axis = result.f_axis

        if f_range is not None:
            f_mask = (f_axis >= f_range[0]) & (f_axis <= f_range[1])
            S = S[:, f_mask]
            f_axis = f_axis[f_mask]

        # For each frequency, find peaks in S(k)
        all_peak_spacings: list[float] = []

        for i_f in range(S.shape[1]):
            S_k = S[:, i_f]
            peaks = self._find_peaks_simple(S_k, threshold=0.3)

            if len(peaks) >= 2:
                # Compute spacings between consecutive peaks
                peak_k = k_axis[peaks]
                spacings = np.diff(peak_k)
                all_peak_spacings.extend(spacings)

        if len(all_peak_spacings) > 0:
            # Most common spacing (mode of histogram)
            spacings = np.array(all_peak_spacings)

            # Filter outliers
            median_spacing = np.median(spacings)
            valid = (spacings > 0.5 * median_spacing) & (spacings < 2 * median_spacing)

            if np.any(valid):
                period_k = float(np.median(spacings[valid]))
                a = 2 * np.pi / period_k if period_k > 0 else float("nan")
                if np.isfinite(a) and 50e-9 < a < 5e-6:
                    scatter = float(
                        np.median(np.abs(spacings[valid] - period_k)) / period_k
                    )
                    self._last_quality = float(1.0 / (1.0 + scatter))
                    logger.info(
                        "Peak-spacing candidate: median Δk=%.3e → a=%.1f nm",
                        period_k,
                        a * 1e9,
                    )
                    return a

        self._last_failure = "no consistent positive peak spacing was detected"
        return None

    def _find_peaks_simple(
        self,
        signal: np.ndarray,
        threshold: float = 0.1,
    ) -> list[int]:
        """Simple peak finding without scipy."""
        peaks = []
        n = len(signal)
        abs_threshold = threshold * np.max(np.abs(signal))

        for i in range(1, n - 1):
            if signal[i] > abs_threshold:
                if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
                    peaks.append(i)

        return peaks

    def suggest_n_periods(
        self,
        k_axis: np.ndarray,
        lattice_constant: float,
    ) -> int:
        """
        Suggest number of BZ periods needed to cover the k-range.

        Parameters
        ----------
        k_axis : np.ndarray
            Wave vector axis [rad/m]
        lattice_constant : float
            Lattice constant [m]

        Returns
        -------
        int
            Suggested number of periods (typically 1-5)
        """
        k_range = np.abs(k_axis[-1] - k_axis[0])
        bz_width = 2 * np.pi / lattice_constant

        n_periods = int(np.ceil(k_range / bz_width / 2)) + 1
        n_periods = max(1, min(n_periods, 10))  # Clamp to [1, 10]

        logger.debug(
            "k_range=%.3e, bz_width=%.3e → suggest %d periods",
            k_range,
            bz_width,
            n_periods,
        )

        return n_periods

    def find_band_gaps(
        self,
        result: DispersionResult1D,
        threshold: float = 0.1,
    ) -> list[tuple[float, float]]:
        """
        Return legacy frequency tuples for low-observed-intensity intervals.

        An interval with weak excitation is not evidence that eigenmodes are
        absent. Use :meth:`find_low_intensity_intervals` to retain the status
        and measured relative intensity with each candidate.
        """
        return [
            (candidate.f_low, candidate.f_high)
            for candidate in self.find_low_intensity_intervals(
                result, threshold=threshold
            )
        ]

    def find_low_intensity_intervals(
        self,
        result: DispersionResult1D,
        threshold: float = 0.1,
    ) -> list[LowIntensityInterval]:
        """Find frequency intervals with low integrated observed intensity."""
        if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be finite and between 0 and 1")

        S = np.asarray(result.S, dtype=float)
        f_axis = np.asarray(result.f_axis, dtype=float)
        if S.ndim != 2 or S.shape[1] != f_axis.size:
            raise ValueError("result.S must have shape (Nk, Nf) matching f_axis")
        if S.size == 0 or not np.all(np.isfinite(S)):
            return []

        S_f = np.sum(np.maximum(S, 0.0), axis=0)
        max_intensity = float(np.max(S_f))
        if max_intensity <= 0:
            return []
        relative = S_f / max_intensity

        intervals: list[LowIntensityInterval] = []
        start: int | None = None
        for idx, value in enumerate(relative):
            if value < threshold and start is None:
                start = idx
            elif value >= threshold and start is not None:
                if idx - start > 2:
                    intervals.append(
                        LowIntensityInterval(
                            f_low=float(f_axis[start]),
                            f_high=float(f_axis[idx - 1]),
                            mean_relative_intensity=float(np.mean(relative[start:idx])),
                        )
                    )
                start = None
        if start is not None and relative.size - start > 2:
            intervals.append(
                LowIntensityInterval(
                    f_low=float(f_axis[start]),
                    f_high=float(f_axis[-1]),
                    mean_relative_intensity=float(np.mean(relative[start:])),
                )
            )

        logger.info("Found %d low-observed-intensity candidates", len(intervals))
        return intervals

    def estimate_effective_mass(
        self,
        result: DispersionResult1D,
        k_center: float = 0,
        dk_fit: float = 1e6,
    ) -> float | None:
        """
        Estimate effective magnon mass from dispersion curvature at k_center.

        For parabolic dispersion: ω = ω₀ + ℏk²/(2m*)
        → m* = ℏ/(d²ω/dk²)

        Parameters
        ----------
        result : DispersionResult1D
            Dispersion result
        k_center : float
            k value around which to fit [rad/m]
        dk_fit : float
            Width of k range to use for fitting [rad/m]

        Returns
        -------
        float or None
            Effective mass [kg], or None if fit fails
        """
        S = result.S
        k_axis = result.k_axis
        f_axis = result.f_axis

        # Select k range around center
        mask = np.abs(k_axis - k_center) <= dk_fit
        if np.sum(mask) < 5:
            logger.warning("Not enough points for mass estimation")
            return None

        k_fit = k_axis[mask]
        S_fit = S[mask, :]

        if f_axis.size < 3 or not np.all(np.isfinite(f_axis)):
            logger.warning("Frequency axis is insufficient for mass estimation")
            return None

        center_idx = int(np.argmin(np.abs(k_fit - k_center)))
        central_spectrum = S_fit[center_idx]
        central_peak = int(np.argmax(central_spectrum))
        f_peaks = np.full(k_fit.size, np.nan, dtype=float)
        f_peaks[center_idx] = f_axis[central_peak]

        def local_peaks(values: np.ndarray) -> np.ndarray:
            if values.size < 3:
                return np.array([int(np.argmax(values))], dtype=int)
            peaks = (
                np.flatnonzero(
                    (values[1:-1] >= values[:-2])
                    & (values[1:-1] >= values[2:])
                    & ((values[1:-1] > values[:-2]) | (values[1:-1] > values[2:]))
                )
                + 1
            )
            return peaks if peaks.size else np.array([int(np.argmax(values))])

        df = float(np.median(np.abs(np.diff(f_axis))))
        max_frequency_jump = max(5.0 * df, 0.1 * float(np.ptp(f_axis)))
        for direction in (-1, 1):
            previous_idx = center_idx
            indices = (
                range(center_idx - 1, -1, -1)
                if direction < 0
                else range(center_idx + 1, k_fit.size)
            )
            for idx in indices:
                candidates = local_peaks(S_fit[idx])
                distances = np.abs(f_axis[candidates] - f_peaks[previous_idx])
                order = np.argsort(distances)
                best = int(order[0])
                if distances[best] > max_frequency_jump:
                    logger.warning(
                        "Dispersion branch tracking failed near k=%g; refusing a mixed-branch fit",
                        k_fit[idx],
                    )
                    return None
                if order.size > 1 and distances[order[1]] - distances[best] <= 0.5 * df:
                    logger.warning(
                        "Ambiguous neighboring branches near k=%g; refusing effective-mass fit",
                        k_fit[idx],
                    )
                    return None
                f_peaks[idx] = f_axis[candidates[best]]
                previous_idx = idx

        if np.any(~np.isfinite(f_peaks)):
            return None

        # Fit parabola: ω(k) = a*k² + b*k + c
        try:
            coeffs = np.polyfit(k_fit - k_center, 2 * np.pi * f_peaks, 2)
            a = coeffs[0]  # curvature coefficient

            if a > 0:
                # m* = ℏ / (2a) where a = d²ω/dk²
                hbar = 1.054571817e-34  # J⋅s
                m_eff = hbar / (2 * a)

                logger.info(
                    "Estimated effective mass: %.3e kg (%.2f m_e)",
                    m_eff,
                    m_eff / 9.109e-31,
                )

                return m_eff
        except Exception as e:
            logger.warning("Mass estimation fit failed: %s", e)

        return None

    def __repr__(self) -> str:
        return f"BrillouinZoneDetector(verbose={self.verbose})"
