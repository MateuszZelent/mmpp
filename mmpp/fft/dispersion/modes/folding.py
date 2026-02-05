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
        
        **Physical Theory: Periodic Lattice and Brillouin Zone Folding**
        
        For a periodic lattice with lattice constant a, the translational symmetry
        m(x + a) = m(x) leads to equivalence of wave vectors in reciprocal space:
        
            k ≡ k + n·G    where G = 2π/a, n = 0, ±1, ±2, ...
        
        The same physical wave can be represented in different Brillouin zones.
        To properly reconstruct the spatial mode profile, we must sum ALL periodic 
        copies:
        
            M_physical(f₀, y) = Σₙ M̃_FFT(f₀, k₀ + n·G) exp(i(k₀ + n·G)y)
        
        This is equivalent to applying a mask that selects all k₀ ± n·G positions
        and then performing IFFT:
        
            M_physical(f₀, y) = IFFT_k { Σₙ δ(k - (k₀ + n·G)) · M̃_FFT(f₀, k) }
        
        **Algorithm:**
        The mask identifies all periodic copies at k = k₀ + n·G:
        
            Mask(f, k) = 1  if |f - f₀| < Δf AND |k - (k₀ + n·G)| < Δk for any n
                       = 0  otherwise
        
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
            If True, include all periodic copies (k_0 + n·G) - PHYSICALLY CORRECT
            If False, only the mode at k_0 - may lose amplitude and distort phase
            
        Returns
        -------
        mask : np.ndarray
            2D boolean mask of shape (len(k_axis), len(f_axis))
            
        References
        ----------
        Rychły et al. (2015) - Mode reconstruction in magnonic crystals
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
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Extract the spatial profile of a specific mode using inverse FFT.
        
        **Physical Theory: Full Mode Reconstruction with Periodic Copies**
        
        For a periodic lattice (period a), wave vectors differing by reciprocal
        lattice vector G = 2π/a represent the same physical wave:
        
            k ≡ k + n·G    for any integer n
        
        The FFT dispersion spectrum M̃(f, k) contains information folded across
        multiple Brillouin zones. To reconstruct the TRUE physical mode, we MUST
        sum all periodic copies:
        
            M_physical(f₀, y) = Σₙ M̃(f₀, k₀ + n·G) exp(i(k₀ + n·G)y)
        
        **Why is this necessary?**
        1. **Coherent addition**: Different BZ copies add constructively, 
           increasing the mode amplitude to its true physical value
        2. **Phase preservation**: Each copy carries phase information that
           affects the spatial structure
        3. **Spatial periodicity**: The sum automatically enforces m(y + a) = m(y)
        
        **Algorithm:**
        1. Create mask selecting all k₀ + n·G positions in the FFT data
        2. Apply mask to COMPLEX spectrum M̃(f, k) (preserves phase!)
        3. Perform IFFT: spatial reconstruction automatically sums all copies
        4. Result: M(y) = physically correct mode profile
        
        **Important**: Uses S_complex (phase-preserving) when available. If only
        S (power spectrum) is available, assumes zero phase → symmetric profile
        (less accurate).
        
        Parameters
        ----------
        result : DispersionResult1D
            Dispersion result containing S_complex (preferred) or S spectrum
        k_0 : float
            Mode wave vector in FBZ [rad/m]
        f_0 : float
            Mode frequency [Hz]
        delta_k : float, optional
            Filter half-width in k [rad/m]. Default: 0.1 * G
        delta_f : float, optional
            Filter half-width in frequency [Hz]. Default: 0.5 GHz
        return_complex : bool
            If True, return complex M(y); if False, return Re[M(y)]
            
        Returns
        -------
        y_axis : np.ndarray
            Position axis [m] (reconstructed from k_axis)
        mode_profile : np.ndarray
            Mode spatial profile: Re[M(y)] or complex M(y)
        mask_info : dict
            Information about the applied mask (k_0, f_0, number of points, etc.)
            
        References
        ----------
        Rychły et al. (2015) - Magnonic crystal mode reconstruction
        
        See Also
        --------
        create_mode_mask : Creates the BZ-folding mask
        """
        from .extraction import extract_mode_profile_1d

        if getattr(result, "S_complex", None) is None:
            raise ValueError(
                "Mode profile reconstruction requires complex spectrum S_complex "
                "(phase information). Recompute dispersion with avg_over_orthogonal=False."
            )

        G = 2 * np.pi / self.a
        dk_val = 0.1 * G if delta_k is None else float(delta_k)
        df_val = 0.5e9 if delta_f is None else float(delta_f)

        prop_axis, profile_complex, info = extract_mode_profile_1d(
            result,
            k_0=float(k_0),
            f_0=float(f_0),
            lattice_constant=float(self.a),
            n_bz=int(self.n_periods),
            k_direction="both",
            k_margin_bins=0,
            f_margin_bins=0,
            neighbor_reduce="mean",
            orth_reduce="mean",
            delta_k=dk_val,
            delta_f=df_val,
        )

        mode_profile = profile_complex if return_complex else np.real(profile_complex)

        mask_info = {
            "k_0": float(k_0),
            "f_0": float(f_0),
            "delta_k": float(dk_val),
            "delta_f": float(df_val),
            "n_periods": int(self.n_periods),
            "k_bins_selected": int(info.get("k_bins_selected", 0)),
            "f_bins_selected": int(info.get("f_bins_selected", 0)),
            # Backwards compatible key (historically counted 2D mask pixels).
            "n_points_masked": int(info.get("k_bins_selected", 0)) * int(info.get("f_bins_selected", 0)),
            "frequency_index": int(np.argmin(np.abs(np.asarray(result.f_axis) - float(f_0)))),
            "orth_reduce": str(info.get("orth_reduce", "mean")),
        }

        logger.info(
            "Extracted mode profile: k_0=%.3f rad/um, f_0=%.2f GHz, points=%d",
            float(k_0) / 1e6,
            float(f_0) / 1e9,
            int(prop_axis.size),
        )

        return prop_axis, mode_profile, mask_info
    
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
        include_negative_frequency: bool = True,
        return_complex: bool = False,
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
        from .extraction import canonicalize_s_complex, build_bz_k_mask

        if getattr(result, "S_complex", None) is None:
            raise ValueError(
                "Mode time evolution requires complex spectrum S_complex (phase information)."
            )

        k_axis = np.asarray(result.k_axis)
        f_axis = np.asarray(result.f_axis)

        S_complex, has_orth = canonicalize_s_complex(result.S_complex, k_axis=k_axis, f_axis=f_axis)
        if has_orth:
            # Collapse orthogonal dimension (linear -> equivalent to averaging after IFFT).
            S_complex = np.mean(S_complex, axis=0)

        a = float(self.a)
        G = 2 * np.pi / a
        dk_val = 0.1 * G if delta_k is None else float(delta_k)
        df_val = 0.5e9 if delta_f is None else float(delta_f)

        k_mask = build_bz_k_mask(
            k_axis,
            k_0=float(k_0),
            lattice_constant=a,
            n_bz=int(self.n_periods),
            k_direction="both",
            k_margin_bins=0,
            delta_k=dk_val,
        )

        # Select +/- frequency neighborhoods if requested.
        if include_negative_frequency:
            f_mask = (np.abs(f_axis - float(f_0)) < df_val) | (np.abs(f_axis + float(f_0)) < df_val)
        else:
            f_mask = np.abs(f_axis - float(f_0)) < df_val
        if not np.any(f_mask):
            idx = int(np.argmin(np.abs(f_axis - float(f_0))))
            f_mask = np.zeros(f_axis.size, dtype=bool)
            f_mask[idx] = True

        S_filtered = np.asarray(S_complex, dtype=np.complex128) * k_mask[:, None] * f_mask[None, :]

        # Undo fftshift before IFFT (k and f are stored shifted for visualization).
        if np.all(np.diff(k_axis) > 0):
            S_filtered = np.fft.ifftshift(S_filtered, axes=0)
        if np.all(np.diff(f_axis) > 0):
            S_filtered = np.fft.ifftshift(S_filtered, axes=1)

        # 2D inverse FFT: (k, f) -> (prop, t)
        mode_evolution_prop_t = np.fft.ifft2(S_filtered)
        mode_evolution_t_prop = mode_evolution_prop_t.T  # (Nt, Nprop)

        N_prop, N_t = mode_evolution_prop_t.shape
        dx = float(getattr(result, "dx", 0.0) or 0.0)
        if dx > 0:
            prop_axis = np.arange(N_prop, dtype=float) * dx
        else:
            if k_axis.size > 1:
                dk = float(np.abs(k_axis[1] - k_axis[0]))
                L = 2 * np.pi / dk if dk > 0 else float(N_prop)
            else:
                L = float(N_prop)
            prop_axis = np.linspace(0.0, L, N_prop, endpoint=False)

        dt = float(getattr(result, "dt", 0.0) or 0.0)
        if dt > 0:
            t_axis = np.arange(N_t, dtype=float) * dt
        else:
            # Fallback from df.
            df = float(np.abs(f_axis[1] - f_axis[0])) if f_axis.size > 1 else 1.0
            T_total = 1.0 / df if df > 0 else float(N_t)
            t_axis = np.linspace(0.0, T_total, N_t, endpoint=False)

        mode_evolution = mode_evolution_t_prop if return_complex else np.real(mode_evolution_t_prop)

        logger.info(
            "Extracted mode time evolution: shape=%s, T=%.2f ns, L=%.2f um",
            mode_evolution.shape,
            float(t_axis[-1] - t_axis[0]) * 1e9 if t_axis.size > 1 else 0.0,
            float(prop_axis[-1] - prop_axis[0]) * 1e6 if prop_axis.size > 1 else 0.0,
        )

        return t_axis, prop_axis, mode_evolution
    
    def __repr__(self) -> str:
        return (
            f"BrillouinZoneFolding(a={self.a*1e9:.1f} nm, "
            f"n_periods={self.n_periods}, "
            f"k_bz=±{self.k_bz/1e6:.3f} rad/μm)"
        )
