"""
Mode extraction utilities for spatial mode profile reconstruction.

Implements the algorithm from Rychły et al. for extracting 2D spatial
mode profiles m(x, y) from dispersion results using BZ folding.
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...models import DispersionResult1D

logger = logging.getLogger(__name__)


class ModeExtractor:
    """Handles extraction of 2D spatial mode profiles from dispersion data."""
    
    def __init__(self, result: DispersionResult1D):
        """
        Initialize extractor with dispersion result.
        
        Parameters
        ----------
        result : DispersionResult1D
            Dispersion result containing S_complex data.
        """
        self.result = result
        
    def extract_spatial_mode(
        self,
        k_0: float,
        f_0: float,
        lattice_constant: float,
        n_bz: int = 3,
        k_direction: str = "both",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract 2D spatial mode profile m(x, y) using pre-computed S_complex.
        
        Algorithm (following Rychły et al.):
        1. Use S_complex from dispersion result (already FFT'd!)
        2. Select frequency f_0 and create mask for k_0 ± n·G (BZ replicas)
        3. IFFT only over k → propagation axis (phase preserved!)
        4. Result: M(x, y) spatial profile of the mode
        
        This is FAST - no re-computation of FFT! Uses cached S_complex.
        
        Parameters
        ----------
        k_0 : float
            Target wave vector in rad/m
        f_0 : float
            Target frequency in Hz
        lattice_constant : float
            Lattice constant in meters
        n_bz : int, default=3
            Number of Brillouin zones to include in mask (±n_bz)
        k_direction : str, default='both'
            Direction filter: 'both', 'positive', or 'negative'
            
        Returns
        -------
        x_axis : ndarray
            x-axis in meters
        y_axis : ndarray
            y-axis in meters
        mode_2d : ndarray
            2D spatial mode profile m(x, y) (real part)
        """
        # Check if we have complex spectrum
        if self.result.S_complex is None:
            raise ValueError(
                "Mode visualization requires complex spectrum S_complex.\n"
                "This should be automatically computed with dispersion.\n"
                "Try recomputing with force=True."
            )
        
        # Get axes and data
        axis = self.result.axis  # 'x' or 'y'
        k_axis = self.result.k_axis.copy()
        f_axis = self.result.f_axis.copy()
        S_complex = self.result.S_complex  # Shape: (Nk, Nf) or (N_orth, Nk, Nf)
        
        # Get grid spacings from result
        dx = self.result.dx if self.result.dx > 0 else 1e-9
        
        # Determine if we have orthogonal spectra
        if S_complex.ndim == 3:
            # Shape: (N_orth, Nk, Nf) - we have spatial variation in orthogonal direction
            N_orth, N_k, N_f = S_complex.shape
            has_orth = True
            logger.info(f"Using orthogonal spectra: {N_orth} positions")
        else:
            # Shape: (Nk, Nf) - averaged over orthogonal direction
            N_k, N_f = S_complex.shape
            N_orth = 1
            has_orth = False
            logger.info("Using averaged spectrum (no orthogonal variation)")
        
        # ===== STEP 1: Select frequency f_0 =====
        idx_f = np.argmin(np.abs(f_axis - f_0))
        f_selected = f_axis[idx_f]
        
        logger.info(f"Selected frequency: f={f_selected/1e9:.3f} GHz (requested: {f_0/1e9:.3f} GHz)")
        
        # ===== STEP 2: Create BZ mask for k_0 ± n·G =====
        mask = self._create_bz_mask(
            k_axis=k_axis,
            k_0=k_0,
            lattice_constant=lattice_constant,
            n_bz=n_bz,
            k_direction=k_direction,
        )
        
        logger.info(f"BZ mask: {np.sum(mask)} k-points selected out of {len(k_axis)}")
        
        # ===== STEP 3: Extract slice at f_0 and apply mask =====
        S_filtered = self._extract_and_filter(S_complex, idx_f, mask, has_orth)
        
        # ===== STEP 4: IFFT over k → spatial axis =====
        M_mode = self._ifft_to_spatial(S_filtered, has_orth)
        
        # ===== STEP 5: Construct spatial axes =====
        prop_axis, orth_axis = self._construct_spatial_axes(
            N_k=N_k,
            N_orth=N_orth,
            k_axis=k_axis,
            dx=dx,
            has_orth=has_orth,
        )
        
        # ===== STEP 6: Assign to x, y based on propagation axis =====
        x_axis, y_axis, mode_2d = self._assign_xy_axes(
            axis=axis,
            prop_axis=prop_axis,
            orth_axis=orth_axis,
            M_mode=M_mode,
        )
        
        logger.info(
            f"Mode profile shape: {mode_2d.shape}, "
            f"x: {x_axis.min()*1e6:.1f}-{x_axis.max()*1e6:.1f} μm, "
            f"y: {y_axis.min()*1e6:.1f}-{y_axis.max()*1e6:.1f} μm"
        )

        return x_axis, y_axis, mode_2d
    
    def _create_bz_mask(
        self,
        k_axis: np.ndarray,
        k_0: float,
        lattice_constant: float,
        n_bz: int,
        k_direction: str,
    ) -> np.ndarray:
        """Create BZ folding mask for k_0 ± n·G."""
        G = 2 * np.pi / lattice_constant
        dk = np.abs(k_axis[1] - k_axis[0]) if len(k_axis) > 1 else 1.0
        
        # Mask width: 2 k-bins
        delta_k = dk * 2
        
        mask = np.zeros(len(k_axis), dtype=bool)
        
        for n in range(-n_bz, n_bz + 1):
            k_target = k_0 + n * G
            
            # Apply k-direction filter
            if k_direction == "positive" and k_target < 0:
                continue
            if k_direction == "negative" and k_target > 0:
                continue
            
            # Find k-values within delta_k of target
            mask |= np.abs(k_axis - k_target) < delta_k
        
        return mask
    
    def _extract_and_filter(
        self,
        S_complex: np.ndarray,
        idx_f: int,
        mask: np.ndarray,
        has_orth: bool,
    ) -> np.ndarray:
        """Extract frequency slice and apply spatial mask."""
        if has_orth:
            # S_complex shape: (N_orth, Nk, Nf)
            # Extract at f_0: (N_orth, Nk)
            S_at_f = S_complex[:, :, idx_f]
            
            # Apply mask: zero non-selected k
            S_filtered = S_at_f.copy()
            S_filtered[:, ~mask] = 0
        else:
            # S_complex shape: (Nk, Nf)
            # Extract at f_0: (Nk,)
            S_at_f = S_complex[:, idx_f]
            
            # Apply mask
            S_filtered = S_at_f.copy()
            S_filtered[~mask] = 0
        
        return S_filtered
    
    def _ifft_to_spatial(
        self,
        S_filtered: np.ndarray,
        has_orth: bool,
    ) -> np.ndarray:
        """Perform IFFT over k to get spatial representation."""
        # Undo fftshift before IFFT
        if has_orth:
            S_unshift = np.fft.ifftshift(S_filtered, axes=1)  # Unshift k-axis (axis=1)
            M_mode = np.fft.ifft(S_unshift, axis=1)  # IFFT along k
            # M_mode shape: (N_orth, N_prop)
        else:
            S_unshift = np.fft.ifftshift(S_filtered)
            M_mode = np.fft.ifft(S_unshift)
            # M_mode shape: (N_prop,)
            # Expand to 2D for consistency
            M_mode = M_mode[np.newaxis, :]  # → (1, N_prop)
        
        return M_mode
    
    def _construct_spatial_axes(
        self,
        N_k: int,
        N_orth: int,
        k_axis: np.ndarray,
        dx: float,
        has_orth: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct spatial axes from k-space parameters."""
        N_prop = N_k  # Propagation axis length = k-axis length
        
        # Propagation axis
        dk = np.abs(k_axis[1] - k_axis[0]) if len(k_axis) > 1 else 1.0
        L_prop = 2 * np.pi / dk if dk > 0 else N_prop * dx
        prop_axis = np.linspace(0, L_prop, N_prop, endpoint=False)
        
        # Orthogonal axis
        if has_orth and self.result.orth_axis is not None:
            orth_axis = self.result.orth_axis
        else:
            # Fallback - assume same spacing
            orth_axis = np.arange(N_orth) * dx
        
        return prop_axis, orth_axis
    
    def _assign_xy_axes(
        self,
        axis: str,
        prop_axis: np.ndarray,
        orth_axis: np.ndarray,
        M_mode: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Assign propagation/orthogonal axes to x/y based on FFT axis."""
        # M_mode shape: (N_orth, N_prop)
        # For axis='x': N_orth=N_y, N_prop=N_x → M_mode is (N_y, N_x) ✓
        # For axis='y': N_orth=N_x, N_prop=N_y → M_mode is (N_x, N_y), need transpose
        if axis == "x":
            x_axis = prop_axis
            y_axis = orth_axis
            # M_mode shape: (N_y, N_x) - already correct for m[y, x] indexing
            mode_2d = M_mode
        else:  # axis == 'y'
            x_axis = orth_axis
            y_axis = prop_axis
            # M_mode shape: (N_x, N_y) - need transpose to (N_y, N_x) for m[y, x]
            mode_2d = M_mode.T
        
        # Take real part (shows oscillation structure)
        mode_2d = np.real(mode_2d)
        
        return x_axis, y_axis, mode_2d
