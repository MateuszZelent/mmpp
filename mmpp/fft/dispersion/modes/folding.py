"""
Brillouin Zone folding algorithms for magnonic crystal dispersion analysis.

Implements algorithms to fold extended-zone dispersion relations into the
first Brillouin zone, preserving information about band origin.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

from .models import BrillouinZoneConfig, DispersionMode, FoldedDispersionResult

if TYPE_CHECKING:
    from ..models import DispersionResult1D

logger = logging.getLogger(__name__)


class BrillouinZoneFolding:
    """
    Handles Brillouin zone folding for periodic magnonic structures.
    
    The algorithm:
    1. Generate reciprocal lattice vectors G_n = n * 2π/a
    2. For each (k, ω) point in the dispersion, find all k' = k + G 
       that fall within the first Brillouin zone
    3. Track the origin (which BZ the point came from)
    4. Group modes into branches
    
    Parameters
    ----------
    lattice_constant : float
        Real-space period of the magnonic crystal [m]
    n_periods : int
        Number of BZ periods to consider on each side (total 2n+1 G vectors)
    lattice_type : str
        Type of lattice: '1d', 'square', 'hexagonal'
        
    Example
    -------
    >>> folder = BrillouinZoneFolding(lattice_constant=470e-9, n_periods=3)
    >>> folded = folder.fold_dispersion(dispersion_result)
    >>> print(folded.summary())
    """
    
    def __init__(
        self,
        lattice_constant: float,
        n_periods: int = 3,
        lattice_type: str = "1d",
    ):
        self.a = lattice_constant
        self.n_periods = n_periods
        self.lattice_type = lattice_type
        
        # Generate reciprocal lattice vectors
        self.G_vectors = self._generate_G_vectors()
        
        # BZ boundaries
        self.k_bz = np.pi / self.a  # FBZ boundary: [-π/a, π/a]
        
        self.config = BrillouinZoneConfig(
            lattice_constant=lattice_constant,
            n_periods=n_periods,
            lattice_type=lattice_type,
            auto_detect=False,
        )
        
        logger.debug(
            "BrillouinZoneFolding initialized: a=%.1f nm, n_periods=%d, "
            "k_bz=±%.3f rad/μm, %d G vectors",
            self.a * 1e9, self.n_periods, self.k_bz / 1e6, len(self.G_vectors)
        )
    
    def _generate_G_vectors(self) -> np.ndarray:
        """Generate reciprocal lattice vectors G_n = n * 2π/a."""
        G = 2 * np.pi / self.a
        indices = np.arange(-self.n_periods, self.n_periods + 1)
        return indices * G
    
    def is_in_fbz(self, k: float, tolerance: float = 1e-10) -> bool:
        """Check if wavevector k is within the first Brillouin zone."""
        return np.abs(k) <= self.k_bz + tolerance
    
    def fold_k_to_fbz(self, k: float) -> Tuple[float, int, float]:
        """
        Fold a single k value to the first Brillouin zone.
        
        Parameters
        ----------
        k : float
            Original wavevector [rad/m]
            
        Returns
        -------
        k_folded : float
            Wavevector folded to FBZ [rad/m]
        origin_bz : int
            Index of the source BZ (0 = FBZ, 1 = second BZ, etc.)
        G_applied : float
            Reciprocal lattice vector used for folding
        """
        # Check each G vector
        for G in self.G_vectors:
            k_mapped = k + G
            if self.is_in_fbz(k_mapped):
                # Determine origin BZ from G
                origin_bz = int(round(-G * self.a / (2 * np.pi)))
                return k_mapped, origin_bz, G
        
        # Fallback: use modulo folding
        bz_width = 2 * self.k_bz
        k_folded = ((k + self.k_bz) % bz_width) - self.k_bz
        
        # Estimate origin BZ
        n_folds = int(round((k - k_folded) / bz_width))
        G_applied = n_folds * bz_width
        
        logger.debug(
            "Fallback folding for k=%.3e: k_folded=%.3e, origin=%d",
            k, k_folded, n_folds
        )
        
        return k_folded, n_folds, G_applied
    
    def _find_peaks_in_spectrum(
        self,
        spectrum: np.ndarray,
        threshold: float,
        min_distance: int = 3,
    ) -> np.ndarray:
        """
        Find peaks in a 1D spectrum above threshold.
        
        Parameters
        ----------
        spectrum : np.ndarray
            1D spectral intensity array
        threshold : float
            Minimum intensity for peak detection
        min_distance : int
            Minimum distance between peaks
            
        Returns
        -------
        peak_indices : np.ndarray
            Indices of detected peaks, sorted by intensity (descending)
        """
        # Simple peak detection without scipy dependency
        peaks = []
        n = len(spectrum)
        
        for i in range(1, n - 1):
            if spectrum[i] > threshold:
                # Check if local maximum
                is_peak = True
                for j in range(max(0, i - min_distance), min(n, i + min_distance + 1)):
                    if j != i and spectrum[j] > spectrum[i]:
                        is_peak = False
                        break
                if is_peak:
                    peaks.append(i)
        
        # Also check if any point is a global maximum
        if len(peaks) == 0 and np.max(spectrum) > threshold:
            peaks = [np.argmax(spectrum)]
        
        # Sort by intensity (descending)
        peaks = sorted(peaks, key=lambda i: spectrum[i], reverse=True)
        
        return np.array(peaks, dtype=int)
    
    def fold_dispersion(
        self,
        result: "DispersionResult1D",
        peak_threshold: float = 0.01,
        min_peak_distance: int = 3,
        max_modes_per_k: int = 10,
    ) -> FoldedDispersionResult:
        """
        Fold a full dispersion relation to the first Brillouin zone.
        
        This is the main method for band folding. It:
        1. Reverses the fftshift on k-axis to get original FFT ordering
        2. For each k, finds spectral peaks (modes)
        3. Folds each (k, ω) point to the FBZ
        4. Tracks origin BZ for each mode
        5. Groups modes into branches
        
        Parameters
        ----------
        result : DispersionResult1D
            Original dispersion result (typically fftshifted for visualization)
        peak_threshold : float
            Relative threshold for peak detection (fraction of max intensity)
        min_peak_distance : int
            Minimum frequency bins between peaks
        max_modes_per_k : int
            Maximum number of modes to detect per k value
            
        Returns
        -------
        FoldedDispersionResult
            Folded dispersion with mode tracking
        """
        S = result.S.copy()
        k_axis = result.k_axis.copy()
        f_axis = result.f_axis.copy()
        
        # Handle fftshift: if k_axis appears shifted (centered), reverse it
        # Detection: check if k_axis is roughly symmetric around 0
        k_center = (k_axis[0] + k_axis[-1]) / 2
        is_shifted = np.abs(k_center) < 0.1 * np.abs(k_axis).max()
        
        if is_shifted:
            logger.debug("Detected fftshifted k-axis, reversing shift")
            # Reverse the fftshift to get original FFT ordering
            k_original = np.fft.ifftshift(k_axis)
            S_original = np.fft.ifftshift(S, axes=0)
        else:
            k_original = k_axis
            S_original = S
        
        # Only consider positive frequencies
        f_positive_mask = f_axis >= 0
        f_positive = f_axis[f_positive_mask]
        S_positive = S_original[:, f_positive_mask]
        
        # Calculate absolute threshold
        global_max = np.max(S_positive)
        abs_threshold = peak_threshold * global_max
        
        logger.info(
            "Folding dispersion: %d k-points, %d f-points, threshold=%.2e (%.1f%% of max)",
            len(k_original), len(f_positive), abs_threshold, peak_threshold * 100
        )
        
        modes: List[DispersionMode] = []
        
        # For each k value, find peaks and fold to FBZ
        for i_k, k in enumerate(k_original):
            spectrum_at_k = S_positive[i_k, :]
            
            # Find peaks
            peak_indices = self._find_peaks_in_spectrum(
                spectrum_at_k,
                abs_threshold,
                min_peak_distance,
            )
            
            # Limit number of modes
            peak_indices = peak_indices[:max_modes_per_k]
            
            for branch_idx, i_f in enumerate(peak_indices):
                omega = f_positive[i_f]
                intensity = spectrum_at_k[i_f]
                
                # Fold k to FBZ
                k_folded, origin_bz, G_applied = self.fold_k_to_fbz(k)
                
                mode = DispersionMode(
                    k=k_folded,
                    omega=omega,
                    branch_index=branch_idx,
                    origin_G=G_applied,
                    origin_BZ=origin_bz,
                    intensity=intensity,
                    k_original=k,
                )
                modes.append(mode)
        
        logger.info("Found %d modes before grouping", len(modes))
        
        # Group into branches based on BZ origin and branch index
        branches = self._group_into_branches(modes)
        
        # Get unique k values in FBZ
        k_fbz = np.unique([m.k for m in modes])
        k_fbz.sort()
        
        result_folded = FoldedDispersionResult(
            modes=modes,
            k_fbz=k_fbz,
            branches=branches,
            bz_config=self.config,
            original_result=result,
        )
        
        logger.info(
            "Folding complete: %d modes in %d branches",
            result_folded.n_modes, result_folded.n_branches
        )
        
        return result_folded
    
    def _group_into_branches(self, modes: List[DispersionMode]) -> Dict[int, List[DispersionMode]]:
        """
        Group modes into branches based on origin BZ and continuity.
        
        Uses a combined index: branch_id = origin_BZ * 100 + local_branch_index
        This allows distinguishing between same-indexed branches from different BZs.
        """
        branches: Dict[int, List[DispersionMode]] = {}
        
        for mode in modes:
            # Combined branch ID
            branch_id = mode.origin_BZ * 100 + mode.branch_index
            
            if branch_id not in branches:
                branches[branch_id] = []
            branches[branch_id].append(mode)
        
        # Re-index branches sequentially
        reindexed: Dict[int, List[DispersionMode]] = {}
        for new_idx, (old_idx, branch_modes) in enumerate(sorted(branches.items())):
            # Update mode branch indices
            for mode in branch_modes:
                mode.branch_index = new_idx
            reindexed[new_idx] = branch_modes
        
        return reindexed
    
    def fold_k_array(self, k_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fold an array of k values to FBZ.
        
        Returns
        -------
        k_folded : np.ndarray
            Folded k values
        origins : np.ndarray
            BZ origin for each k
        G_applied : np.ndarray
            G vector applied to each k
        """
        n = len(k_array)
        k_folded = np.zeros(n)
        origins = np.zeros(n, dtype=int)
        G_applied = np.zeros(n)
        
        for i, k in enumerate(k_array):
            k_folded[i], origins[i], G_applied[i] = self.fold_k_to_fbz(k)
        
        return k_folded, origins, G_applied
    
    # =========================================================================
    # KROK 4: Generowanie maski dla wybranego modu (Rychły et al. 2015)
    # =========================================================================
    
    def create_mode_mask(
        self,
        k_axis: np.ndarray,
        f_axis: np.ndarray,
        k_0: float,
        f_0: float,
        delta_k: Optional[float] = None,
        delta_f: Optional[float] = None,
        include_all_copies: bool = True,
    ) -> np.ndarray:
        """
        Create a mask in Fourier space for extracting a specific mode.
        
        The mask identifies all periodic copies of the mode at k_0 + n·G
        where G = 2π/a is the reciprocal lattice vector.
        
        Maska(f, k_y) = 1 if |f - f_0| < Δf AND |k_y - (k_0 + n·G)| < Δk
                       0 otherwise
        
        Parameters
        ----------
        k_axis : np.ndarray
            Wave vector axis [rad/m]
        f_axis : np.ndarray
            Frequency axis [Hz]
        k_0 : float
            Mode wave vector in FBZ [rad/m]
        f_0 : float
            Mode frequency [Hz]
        delta_k : float, optional
            Half-width in k-space [rad/m]. Default: 0.1 * G
        delta_f : float, optional
            Half-width in frequency [Hz]. Default: 0.5 GHz
        include_all_copies : bool
            If True, include all periodic copies (k_0 + n·G)
            If False, only the mode at k_0
            
        Returns
        -------
        mask : np.ndarray
            2D boolean mask of shape (len(k_axis), len(f_axis))
        """
        G = 2 * np.pi / self.a
        
        # Default widths
        if delta_k is None:
            delta_k = 0.1 * G
        if delta_f is None:
            delta_f = 0.5e9  # 0.5 GHz
        
        # Initialize mask
        mask = np.zeros((len(k_axis), len(f_axis)), dtype=bool)
        
        # Frequency condition (same for all copies)
        f_condition = np.abs(f_axis - f_0) < delta_f
        
        # Determine k positions to include
        if include_all_copies:
            # All periodic copies: k_0 + n·G for n in [-n_periods, n_periods]
            k_positions = [k_0 + n * G for n in range(-self.n_periods, self.n_periods + 1)]
        else:
            k_positions = [k_0]
        
        # Apply k conditions
        for k_target in k_positions:
            k_condition = np.abs(k_axis - k_target) < delta_k
            # Combine: mask[i_k, i_f] = True if both conditions met
            for i_k in np.where(k_condition)[0]:
                mask[i_k, :] |= f_condition
        
        logger.debug(
            "Created mode mask: k_0=%.3f rad/μm, f_0=%.2f GHz, "
            "%d k-positions, %d points selected",
            k_0 / 1e6, f_0 / 1e9, len(k_positions), np.sum(mask)
        )
        
        return mask
    
    def create_mode_mask_from_mode(
        self,
        mode: DispersionMode,
        k_axis: np.ndarray,
        f_axis: np.ndarray,
        delta_k: Optional[float] = None,
        delta_f: Optional[float] = None,
    ) -> np.ndarray:
        """
        Create a mask for a specific DispersionMode object.
        
        Parameters
        ----------
        mode : DispersionMode
            The mode to create mask for
        k_axis, f_axis : np.ndarray
            Fourier space axes
        delta_k, delta_f : float, optional
            Filter widths
            
        Returns
        -------
        mask : np.ndarray
            2D boolean mask
        """
        return self.create_mode_mask(
            k_axis=k_axis,
            f_axis=f_axis,
            k_0=mode.k,
            f_0=mode.omega,
            delta_k=delta_k,
            delta_f=delta_f,
            include_all_copies=True,
        )
    
    # =========================================================================
    # KROK 5: Inverse FFT — powrót do przestrzeni rzeczywistej
    # =========================================================================
    
    def extract_mode_profile(
        self,
        result: "DispersionResult1D",
        k_0: float,
        f_0: float,
        delta_k: Optional[float] = None,
        delta_f: Optional[float] = None,
        return_complex: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract the spatial profile of a specific mode using inverse FFT.
        
        Algorithm (Rychły et al. 2015):
        1. Apply mask in Fourier space to select mode and its periodic copies
        2. Perform inverse FFT to get real-space amplitude
        3. Return the mode envelope in position space
        
        Parameters
        ----------
        result : DispersionResult1D
            Original dispersion result containing S(k, f) spectrum
        k_0 : float
            Mode wave vector in FBZ [rad/m]
        f_0 : float
            Mode frequency [Hz]
        delta_k : float, optional
            Filter half-width in k [rad/m]
        delta_f : float, optional
            Filter half-width in frequency [Hz]
        return_complex : bool
            If True, return complex field; if False, return real part
            
        Returns
        -------
        y_axis : np.ndarray
            Position axis [m] (reconstructed from k_axis)
        mode_profile : np.ndarray
            Mode amplitude profile Re[IFFT{M̃_filtered}] or complex
        mask_info : dict
            Information about the applied mask
        """
        S = result.S.copy()
        k_axis = result.k_axis.copy()
        f_axis = result.f_axis.copy()
        
        # Create mask
        mask = self.create_mode_mask(
            k_axis=k_axis,
            f_axis=f_axis,
            k_0=k_0,
            f_0=f_0,
            delta_k=delta_k,
            delta_f=delta_f,
            include_all_copies=True,
        )
        
        # Apply mask to spectrum
        # S has shape (N_k, N_f)
        S_filtered = S * mask.astype(float)
        
        # Find frequency index closest to f_0
        i_f0 = np.argmin(np.abs(f_axis - f_0))
        
        # Extract 1D spectrum at this frequency
        S_at_f0 = S_filtered[:, i_f0]
        
        # Also create complex representation for proper IFFT
        # We need to include phase information if available
        # For amplitude-only data, we assume phase = 0 which gives symmetric profile
        
        # Reconstruct position axis from k_axis
        # Δk = 2π/L → L = 2π/Δk
        if len(k_axis) > 1:
            dk = np.abs(k_axis[1] - k_axis[0])
            L_y = 2 * np.pi / dk
            N_y = len(k_axis)
            y_axis = np.linspace(0, L_y, N_y, endpoint=False)
        else:
            y_axis = np.array([0])
        
        # Perform inverse FFT
        # Need to handle fftshift: if k_axis is centered, undo shift first
        k_center = (k_axis[0] + k_axis[-1]) / 2
        is_shifted = np.abs(k_center) < 0.1 * np.abs(k_axis).max()
        
        if is_shifted:
            S_for_ifft = np.fft.ifftshift(S_at_f0)
        else:
            S_for_ifft = S_at_f0
        
        # Inverse FFT: k → y
        mode_profile_complex = np.fft.ifft(S_for_ifft)
        
        if return_complex:
            mode_profile = mode_profile_complex
        else:
            mode_profile = np.real(mode_profile_complex)
        
        mask_info = {
            "k_0": k_0,
            "f_0": f_0,
            "delta_k": delta_k if delta_k is not None else 0.1 * (2 * np.pi / self.a),
            "delta_f": delta_f if delta_f is not None else 0.5e9,
            "n_points_masked": np.sum(mask),
            "frequency_index": i_f0,
        }
        
        logger.info(
            "Extracted mode profile: k_0=%.3f rad/μm, f_0=%.2f GHz, "
            "%d spatial points",
            k_0 / 1e6, f_0 / 1e9, len(y_axis)
        )
        
        return y_axis, mode_profile, mask_info
    
    def extract_mode_profile_from_mode(
        self,
        mode: DispersionMode,
        result: "DispersionResult1D",
        delta_k: Optional[float] = None,
        delta_f: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Extract spatial profile for a DispersionMode object.
        
        Parameters
        ----------
        mode : DispersionMode
            The mode to extract
        result : DispersionResult1D
            Original dispersion data
        delta_k, delta_f : float, optional
            Filter widths
            
        Returns
        -------
        y_axis, mode_profile, mask_info
        """
        return self.extract_mode_profile(
            result=result,
            k_0=mode.k,
            f_0=mode.omega,
            delta_k=delta_k,
            delta_f=delta_f,
        )
    
    def extract_mode_time_evolution(
        self,
        result: "DispersionResult1D",
        k_0: float,
        f_0: float,
        delta_k: Optional[float] = None,
        delta_f: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract mode profile with full time evolution using 2D inverse FFT.
        
        Returns m_mode(t, y) = IFFT_2D{M̃_filtered(f, k_y)}
        
        Parameters
        ----------
        result : DispersionResult1D
            Dispersion result
        k_0 : float
            Mode k in FBZ [rad/m]
        f_0 : float
            Mode frequency [Hz]
        delta_k, delta_f : float, optional
            Filter widths
            
        Returns
        -------
        t_axis : np.ndarray
            Time axis [s]
        y_axis : np.ndarray
            Position axis [m]
        mode_evolution : np.ndarray
            2D array m(t, y) of shape (N_t, N_y)
        """
        S = result.S.copy()
        k_axis = result.k_axis.copy()
        f_axis = result.f_axis.copy()
        
        # Create mask
        mask = self.create_mode_mask(
            k_axis=k_axis,
            f_axis=f_axis,
            k_0=k_0,
            f_0=f_0,
            delta_k=delta_k,
            delta_f=delta_f,
            include_all_copies=True,
        )
        
        # Apply mask
        S_filtered = S * mask.astype(float)
        
        # Handle fftshift
        k_center = (k_axis[0] + k_axis[-1]) / 2
        is_k_shifted = np.abs(k_center) < 0.1 * np.abs(k_axis).max()
        
        f_center = (f_axis[0] + f_axis[-1]) / 2
        is_f_shifted = np.abs(f_center) < 0.1 * np.abs(f_axis).max()
        
        S_for_ifft = S_filtered
        if is_k_shifted:
            S_for_ifft = np.fft.ifftshift(S_for_ifft, axes=0)
        if is_f_shifted:
            S_for_ifft = np.fft.ifftshift(S_for_ifft, axes=1)
        
        # 2D inverse FFT: (k, f) → (y, t)
        # Note: S has shape (N_k, N_f), result will be (N_y, N_t)
        mode_evolution_yt = np.fft.ifft2(S_for_ifft)
        
        # Transpose to (N_t, N_y) for conventional time-space ordering
        mode_evolution = np.real(mode_evolution_yt.T)
        
        # Reconstruct axes
        N_k, N_f = S.shape
        
        # Position axis from k
        if N_k > 1:
            dk = np.abs(k_axis[1] - k_axis[0])
            L_y = 2 * np.pi / dk
            y_axis = np.linspace(0, L_y, N_k, endpoint=False)
        else:
            y_axis = np.array([0])
        
        # Time axis from f
        if N_f > 1:
            df = np.abs(f_axis[1] - f_axis[0])
            T_total = 1.0 / df
            t_axis = np.linspace(0, T_total, N_f, endpoint=False)
        else:
            t_axis = np.array([0])
        
        logger.info(
            "Extracted mode time evolution: shape (%d, %d), "
            "T=%.2f ns, L=%.2f μm",
            len(t_axis), len(y_axis), T_total * 1e9, L_y * 1e6
        )
        
        return t_axis, y_axis, mode_evolution
    
    def __repr__(self) -> str:
        return (
            f"BrillouinZoneFolding(a={self.a*1e9:.1f} nm, "
            f"n_periods={self.n_periods}, "
            f"k_bz=±{self.k_bz/1e6:.3f} rad/μm)"
        )
