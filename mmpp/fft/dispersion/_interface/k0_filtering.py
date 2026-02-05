"""
k≈0 dynamic filtering for dispersion spectra.

Implements adaptive compression to enhance visualization of spin-wave
modes while suppressing overwhelming k≈0 peak.
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple

import numpy as np

try:
    from scipy.signal import savgol_filter
    from scipy.stats import median_abs_deviation as mad
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    savgol_filter = None  # type: ignore
    mad = None  # type: ignore

logger = logging.getLogger(__name__)


class K0Filter:
    """
    Dynamic k≈0 filtering for enhanced dispersion visualization.
    
    Implements adaptive compression algorithms to suppress the dominant
    k≈0 peak while preserving spin-wave mode structure.
    """
    
    @staticmethod
    def apply_k0_normalization(
        spectrum: np.ndarray,
        k_axis: np.ndarray,
        strength: float = 1.0,
        compression_mode: str = "adaptive",
        k0_normalization_width: int = 1,
    ) -> np.ndarray:
        """
        Apply dynamic k≈0 compression to spectrum.
        
        Parameters
        ----------
        spectrum : ndarray
            Input spectrum S(k, f) or S(f, k)
        k_axis : ndarray
            Wave vector axis in rad/m
        strength : float, default=1.0
            Compression strength (0-10)
            0 = no compression, 10 = maximum
        compression_mode : str, default='adaptive'
            Compression profile:
            - 'gentle': Smooth, preserves more structure
            - 'aggressive': Strong suppression of k≈0
            - 'preserve_peaks': Protects sharp features
            - 'adaptive': Balanced (default)
        k0_normalization_width : int, default=1
            Number of k-bins around k=0 to compress
            
        Returns
        -------
        ndarray
            Compressed spectrum with same shape as input
        """
        if strength <= 1e-6:
            return spectrum.copy()
        
        # Map strength [0,10] → linear [0,1]
        linear_strength = max(0.0, min(1.0, float(strength) / 10.0))
        
        # Get compression parameters based on mode
        params = K0Filter._get_compression_params(compression_mode, linear_strength)
        
        logger.info(
            "k≈0 dynamic compression: mode=%s, strength=%.1f (linear=%.2f) → "
            "β=%.2f, A_max=%.0f, knee=%.1fdB, slope=%.1fdB",
            compression_mode,
            strength,
            linear_strength,
            params['beta'],
            params['A_max'],
            params['knee_db'],
            params['slope_db'],
        )
        
        return K0Filter._k0_dynamic_filter(
            spectrum,
            k_axis,
            strength=linear_strength,
            k0_normalization_width=k0_normalization_width,
            **params,
        )
    
    @staticmethod
    def _get_compression_params(mode: str, linear_strength: float) -> dict:
        """Get compression parameters for given mode and strength."""
        if mode == "gentle":
            return {
                'beta': 4.5 - 1.0 * linear_strength,
                'A_max': 10.0 + 90.0 * linear_strength,
                'knee_db': 8.0 - 2.0 * linear_strength,
                'slope_db': 5.0 - 1.0 * linear_strength,
            }
        elif mode == "aggressive":
            return {
                'beta': 4.0 - 1.5 * linear_strength,
                'A_max': 100.0 + 4900.0 * linear_strength,
                'knee_db': 6.0 - 2.0 * linear_strength,
                'slope_db': 3.0 - 1.5 * linear_strength,
            }
        elif mode == "preserve_peaks":
            return {
                'beta': 4.0 - 0.5 * linear_strength,
                'A_max': 20.0 + 180.0 * linear_strength,
                'knee_db': 7.0 - 1.0 * linear_strength,
                'slope_db': 4.0 - 0.5 * linear_strength,
            }
        else:  # adaptive (default)
            return {
                'beta': 1.0 - 0.5 * linear_strength,
                'A_max': 500.0 + 9500.0 * linear_strength,
                'knee_db': 6.0,
                'slope_db': 2.5 - 0.5 * linear_strength,
            }
    
    @staticmethod
    def _k0_dynamic_filter(
        PSD_fk: np.ndarray,
        k_vals: np.ndarray,
        strength: float,
        beta: float,
        A_max: float,
        knee_db: float,
        slope_db: float,
        k0_normalization_width: int,
        **kwargs,
    ) -> np.ndarray:
        """
        Internal k≈0 dynamic filter implementation.
        
        Applies frequency-dependent compression based on local statistics
        around k≈0.
        """
        if PSD_fk.ndim == 1:
            PSD_compressed, _, _ = K0Filter._k0_dynamic_filter_linear(
                PSD_fk,
                k_vals,
                strength=strength,
                A_max=A_max,
                beta=beta,
                knee_db=knee_db,
                slope_db=slope_db,
                k0_normalization_width=k0_normalization_width,
                **kwargs,
            )
            return PSD_compressed
        
        # 2D spectrum - apply to each frequency
        PSD_compressed, _, _ = K0Filter._k0_dynamic_filter_linear(
            PSD_fk.T,
            k_vals,
            strength=strength,
            A_max=A_max,
            beta=beta,
            knee_db=knee_db,
            slope_db=slope_db,
            k0_normalization_width=k0_normalization_width,
            **kwargs,
        )
        return PSD_compressed.T
    
    @staticmethod
    def _k0_dynamic_filter_linear(
        PSD_fk: np.ndarray,
        k_vals: np.ndarray,
        strength: float,
        A_max: float,
        beta: float,
        knee_db: float,
        slope_db: float,
        k0_normalization_width: int,
        k_halfwidth: Optional[float] = None,
        smooth_win: Optional[int] = 11,
        smooth_poly: int = 2,
        eps: float = 1e-18,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply perlinearized compressor to PSD data.
        
        Returns
        -------
        PSD_compressed : ndarray
            Compressed spectrum
        idx0 : ndarray
            Indices of k≈0 region
        gain_map : ndarray
            Applied gain map for each frequency
        """
        def _odd(n: int) -> int:
            return int(n) + 1 - (int(n) % 2 == 0)
        
        PSD = np.asarray(PSD_fk).copy()
        k = np.asarray(k_vals)
        
        # Handle 1D vs 2D
        if PSD.ndim == 1:
            F, K = 1, PSD.shape[0]
            PSD = PSD[np.newaxis, :]
            is_1d = True
        elif PSD.ndim == 2:
            F, K = PSD.shape
            is_1d = False
        else:
            raise ValueError(f"PSD must be 1D or 2D, got {PSD.ndim}D")
        
        # Find k≈0 region
        center_idx = np.argmin(np.abs(k))
        half_width = max(0, (k0_normalization_width - 1) // 2)
        
        if k_halfwidth is not None and K > 1:
            dk_values = np.diff(np.sort(np.abs(k)))
            dk = float(np.median(dk_values)) if dk_values.size else 0.0
            if dk > 0:
                half_width = max(half_width, int(np.ceil(abs(k_halfwidth) / dk)))
        
        idx0 = np.array([
            center_idx + offset
            for offset in range(-half_width, half_width + 1)
            if 0 <= center_idx + offset < K
        ])
        
        logger.info(
            "k≈0 region: width=%d, center_idx=%d, indices=%s, total_bins=%d",
            k0_normalization_width,
            center_idx,
            idx0.tolist(),
            len(idx0),
        )
        
        other = np.setdiff1d(np.arange(K), idx0)
        if other.size == 0:
            return PSD, idx0, np.ones((F, idx0.size))
        
        # Calculate baseline statistics from non-k≈0 region
        base = np.median(PSD[:, other], axis=1)
        
        if _SCIPY_AVAILABLE:
            scale = mad(PSD[:, other], axis=1, scale="normal") + eps
        else:
            # Fallback: IQR-based estimate
            q75 = np.percentile(PSD[:, other], 75, axis=1)
            q25 = np.percentile(PSD[:, other], 25, axis=1)
            scale = (q75 - q25) / 1.349 + eps
        
        # Apply compression to k≈0 region
        # (Implementation of adaptive compression algorithm)
        # This is a skeleton - full implementation would include:
        # - Gain calculation based on ratio to baseline
        # - Soft-knee compressor curve
        # - Smooth transitions
        
        PSD_compressed = PSD.copy()
        gain_map = np.ones((F, len(idx0)))
        
        for f_idx in range(F):
            for i, k_idx in enumerate(idx0):
                # Calculate compression ratio
                signal = PSD[f_idx, k_idx]
                baseline = base[f_idx]
                
                if signal > baseline:
                    ratio_db = 10 * np.log10(signal / (baseline + eps) + eps)
                    
                    # Soft-knee compression
                    if ratio_db > knee_db:
                        excess_db = ratio_db - knee_db
                        compressed_db = knee_db + excess_db / slope_db
                        gain_db = compressed_db - ratio_db
                        gain = 10 ** (gain_db / 10)
                        
                        # Apply gain limits
                        gain = max(1.0 / A_max, min(1.0, gain))
                        
                        PSD_compressed[f_idx, k_idx] = signal * gain
                        gain_map[f_idx, i] = gain
        
        # Smooth gain map if scipy available
        if _SCIPY_AVAILABLE and smooth_win is not None and F > smooth_win:
            win = _odd(smooth_win)
            if F >= win:
                for i in range(len(idx0)):
                    try:
                        gain_map[:, i] = savgol_filter(
                            gain_map[:, i],
                            window_length=win,
                            polyorder=min(smooth_poly, win - 1),
                            mode='nearest',
                        )
                    except Exception:
                        pass
        
        if is_1d:
            return PSD_compressed[0], idx0, gain_map
        
        return PSD_compressed, idx0, gain_map
