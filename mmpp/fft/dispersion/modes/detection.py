"""
Automatic detection of Brillouin zone parameters from dispersion data.

Provides methods to estimate:
- Lattice constant from periodicity in S(k,f)
- Optimal number of BZ periods
- FBZ boundaries
"""

from __future__ import annotations
import logging
from typing import List, Optional, Tuple, TYPE_CHECKING
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
        result: "DispersionResult1D",
        method: str = "autocorrelation",
        f_range: Optional[Tuple[float, float]] = None,
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
        result: "DispersionResult1D",
        f_range: Optional[Tuple[float, float]] = None,
    ) -> float:
        """
        Detect lattice constant via autocorrelation of the k-profile.
        
        The idea: if the dispersion has periodicity in k-space (due to BZ folding),
        the autocorrelation will show peaks at multiples of 2π/a.
        """
        S = result.S
        k_axis = result.k_axis
        f_axis = result.f_axis
        
        # Apply frequency filter if specified
        if f_range is not None:
            f_mask = (f_axis >= f_range[0]) & (f_axis <= f_range[1])
            S = S[:, f_mask]
        
        # Average over frequency to get k-profile
        # Weight by intensity to emphasize strong features
        S_mean = np.sum(S, axis=1)
        
        # Normalize
        S_mean = S_mean / (np.max(S_mean) + 1e-20)
        
        # Compute autocorrelation
        autocorr = np.correlate(S_mean, S_mean, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Keep only positive lags
        
        # Normalize autocorrelation
        autocorr = autocorr / (autocorr[0] + 1e-20)
        
        # Find first significant peak after zero
        # Skip the first few points (zero-lag region)
        min_lag = max(3, len(autocorr) // 20)
        
        peaks = self._find_peaks_simple(autocorr[min_lag:], threshold=0.1)
        
        if len(peaks) > 0:
            # First significant peak
            peak_lag = peaks[0] + min_lag
            
            # Convert lag to k-space distance
            dk = k_axis[1] - k_axis[0] if len(k_axis) > 1 else 1.0
            period_k = peak_lag * dk
            
            # period_k = 2π/a → a = 2π/period_k
            a = 2 * np.pi / period_k
            
            logger.info(
                "Autocorr detection: peak at lag %d (Δk=%.3e rad/m) → a=%.1f nm",
                peak_lag, period_k, a * 1e9
            )
            
            return a
        
        # Fallback: use k-range as rough estimate
        k_range = k_axis[-1] - k_axis[0]
        a_fallback = 2 * np.pi / k_range * 2  # Assume ~2 BZ visible
        
        logger.warning(
            "Autocorr detection failed, using fallback: a=%.1f nm",
            a_fallback * 1e9
        )
        
        return a_fallback
    
    def _detect_via_fft(
        self,
        result: "DispersionResult1D",
        f_range: Optional[Tuple[float, float]] = None,
    ) -> float:
        """
        Detect lattice constant via FFT of the k-profile.
        
        Takes FFT of S(k) integrated over f to find spatial frequency peaks.
        """
        S = result.S
        k_axis = result.k_axis
        f_axis = result.f_axis
        
        if f_range is not None:
            f_mask = (f_axis >= f_range[0]) & (f_axis <= f_range[1])
            S = S[:, f_mask]
        
        # Get k-profile
        S_k = np.sum(S, axis=1)
        
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
        
        # Find dominant frequency (excluding DC)
        if len(fft_mag) > 0:
            peak_idx = np.argmax(fft_mag)
            dominant_freq = fft_freq[peak_idx]  # cycles per rad/m
            
            # Convert to period in k-space
            if dominant_freq > 0:
                period_k = 1.0 / dominant_freq  # rad/m per cycle
                # period_k = 2π/a → a = 2π/period_k
                a = 2 * np.pi / period_k
                
                logger.info(
                    "FFT detection: dominant freq=%.3e → a=%.1f nm",
                    dominant_freq, a * 1e9
                )
                
                return a
        
        # Fallback
        return self._detect_via_autocorr(result, f_range)
    
    def _detect_via_peak_spacing(
        self,
        result: "DispersionResult1D",
        f_range: Optional[Tuple[float, float]] = None,
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
        all_peak_spacings = []
        
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
                    period_k, a * 1e9
                )
                
                return a
        
        # Fallback
        return self._detect_via_autocorr(result, f_range)
    
    def _find_peaks_simple(
        self,
        signal: np.ndarray,
        threshold: float = 0.1,
    ) -> List[int]:
        """Simple peak finding without scipy."""
        peaks = []
        n = len(signal)
        abs_threshold = threshold * np.max(np.abs(signal))
        
        for i in range(1, n - 1):
            if signal[i] > abs_threshold:
                if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
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
            k_range, bz_width, n_periods
        )
        
        return n_periods
    
    def find_band_gaps(
        self,
        result: "DispersionResult1D",
        threshold: float = 0.1,
    ) -> List[Tuple[float, float]]:
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
                    gaps.append((f_axis[gap_start], f_axis[i-1]))
        
        # Handle gap extending to end
        if in_gap and len(f_axis) - gap_start > 2:
            gaps.append((f_axis[gap_start], f_axis[-1]))
        
        logger.info("Found %d band gaps", len(gaps))
        
        return gaps
    
    def estimate_effective_mass(
        self,
        result: "DispersionResult1D",
        k_center: float = 0,
        dk_fit: float = 1e6,
    ) -> Optional[float]:
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
        f_peaks = []
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
                    m_eff, m_eff / 9.109e-31
                )
                
                return m_eff
        except Exception as e:
            logger.warning("Mass estimation fit failed: %s", e)
        
        return None
    
    def __repr__(self) -> str:
        return f"BrillouinZoneDetector(verbose={self.verbose})"
