"""
Automatic detection of Brillouin zone parameters from dispersion data.

Provides methods to estimate:
- Lattice constant from periodicity in S(k,f)
- Optimal number of BZ periods
- FBZ boundaries
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..models import DispersionResult1D

logger = logging.getLogger(__name__)


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

    def detect_lattice_constant(
        self,
        result: DispersionResult1D,
        method: str = "autocorrelation",
        f_range: tuple[float, float] | None = None,
    ) -> float:
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
        float
            Estimated lattice constant [m]
        """
        if method == "autocorrelation":
            return self._detect_via_autocorr(result, f_range)
        elif method == "fft":
            return self._detect_via_fft(result, f_range)
        elif method == "peak_spacing":
            return self._detect_via_peak_spacing(result, f_range)
        else:
            raise ValueError(f"Unknown detection method: {method}")

    def _detect_via_autocorr(
        self,
        result: DispersionResult1D,
        f_range: tuple[float, float] | None = None,
    ) -> float:
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

        # Get k-profile weighted by log intensity to reduce dynamic range
        S_positive = np.maximum(S, 1e-20)
        S_log = np.log10(S_positive)
        S_mean = np.mean(S_log, axis=1)

        # Remove DC / mean to focus on periodic structure
        S_mean = S_mean - np.mean(S_mean)

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
        if min_lag_physical >= max_lag_physical:
            min_lag_physical = 5
            max_lag_physical = len(autocorr) // 2

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
                return a

        # Fallback: estimate from k-range assuming ~2 BZ visible
        k_range = k_axis[-1] - k_axis[0]
        a_fallback = 2 * np.pi / k_range * 2

        # Clamp to physical limits
        a_fallback = np.clip(a_fallback, 100e-9, 2000e-9)

        logger.warning(
            "Autocorr detection weak, using fallback: a=%.1f nm", a_fallback * 1e9
        )

        return a_fallback

    def _detect_via_fft(
        self,
        result: DispersionResult1D,
        f_range: tuple[float, float] | None = None,
    ) -> float:
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

            if dominant_freq > 0:
                # freq = a/(2π) → a = 2π * freq
                # Actually: period_k = 1/freq, a = 2π/period_k = 2π * freq
                a = 2 * np.pi * dominant_freq

                logger.info(
                    "FFT detection: dominant freq=%.3e cycles/rad → a=%.1f nm",
                    dominant_freq,
                    a * 1e9,
                )

                return a

        # Fallback
        return self._detect_via_autocorr(result, f_range)

    def _detect_via_peak_spacing(
        self,
        result: DispersionResult1D,
        f_range: tuple[float, float] | None = None,
    ) -> float:
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
            threshold = 0.3 * np.max(S_k)
            peaks = self._find_peaks_simple(S_k, threshold)

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
                period_k = np.median(spacings[valid])
                a = 2 * np.pi / period_k

                logger.info(
                    "Peak spacing detection: median Δk=%.3e → a=%.1f nm",
                    period_k,
                    a * 1e9,
                )

                return a

        # Fallback
        return self._detect_via_autocorr(result, f_range)

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
        Find frequency gaps in the dispersion relation.

        Parameters
        ----------
        result : DispersionResult1D
            Dispersion result to analyze
        threshold : float
            Relative intensity threshold for gap detection

        Returns
        -------
        List[Tuple[float, float]]
            List of (f_low, f_high) tuples defining gaps [Hz]
        """
        S = result.S
        f_axis = result.f_axis

        # Sum over k to get total intensity at each frequency
        S_f = np.sum(S, axis=0)

        # Normalize
        S_f = S_f / (np.max(S_f) + 1e-20)

        # Find regions below threshold
        gaps = []
        in_gap = False
        gap_start = 0

        for i, val in enumerate(S_f):
            if val < threshold and not in_gap:
                in_gap = True
                gap_start = i
            elif val >= threshold and in_gap:
                in_gap = False
                if i - gap_start > 2:  # Minimum gap width
                    gaps.append((f_axis[gap_start], f_axis[i - 1]))

        # Handle gap extending to end
        if in_gap and len(f_axis) - gap_start > 2:
            gaps.append((f_axis[gap_start], f_axis[-1]))

        logger.info("Found %d band gaps", len(gaps))

        return gaps

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

        # Find peak frequency at each k
        f_peaks: Any = []
        for i in range(len(k_fit)):
            i_max = np.argmax(S_fit[i, :])
            f_peaks.append(f_axis[i_max])

        f_peaks = np.array(f_peaks)

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
