"""
Data models for Brillouin zone folding and dispersion mode analysis.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..models import DispersionResult1D


@dataclass
class BrillouinZoneConfig:
    """Configuration for Brillouin zone analysis."""
    
    lattice_constant: float  # Real-space period [m]
    n_periods: int = 3  # Number of BZ periods to consider for folding
    lattice_type: str = "1d"  # '1d', 'square', 'hexagonal'
    auto_detect: bool = True  # Whether to auto-detect from data
    
    @property
    def bz_width(self) -> float:
        """Width of first Brillouin zone [rad/m]."""
        return 2 * np.pi / self.lattice_constant
    
    @property
    def k_bz_boundary(self) -> float:
        """Boundary of first BZ: ±π/a [rad/m]."""
        return np.pi / self.lattice_constant
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "lattice_constant": self.lattice_constant,
            "n_periods": self.n_periods,
            "lattice_type": self.lattice_type,
            "auto_detect": self.auto_detect,
            "bz_width": self.bz_width,
            "k_bz_boundary": self.k_bz_boundary,
        }


@dataclass
class DispersionMode:
    """A single dispersion mode point after BZ folding."""
    
    k: float  # Wave vector in FBZ [rad/m]
    omega: float  # Angular frequency [rad/s] or frequency [Hz]
    branch_index: int  # Index of the dispersion branch
    origin_G: float  # Reciprocal lattice vector used for folding [rad/m]
    origin_BZ: int  # Index of the source Brillouin zone (0 = FBZ)
    intensity: float  # Spectral intensity at this point
    
    # Original (unfolded) coordinates
    k_original: float = 0.0  # Original k before folding
    
    def __repr__(self) -> str:
        return (
            f"DispersionMode(k={self.k/1e6:.3f} rad/μm, "
            f"f={self.omega/1e9:.3f} GHz, "
            f"branch={self.branch_index}, "
            f"from BZ {self.origin_BZ})"
        )


@dataclass
class FoldedDispersionResult:
    """Result of Brillouin zone folding analysis."""
    
    modes: List[DispersionMode]  # All detected modes
    k_fbz: np.ndarray  # Unique k values in FBZ [rad/m]
    branches: Dict[int, List[DispersionMode]]  # Modes grouped by branch
    bz_config: BrillouinZoneConfig  # Configuration used
    original_result: "DispersionResult1D"  # Original unfolded data
    
    # Processed arrays for fast plotting
    _k_array: Optional[np.ndarray] = field(default=None, repr=False)
    _f_array: Optional[np.ndarray] = field(default=None, repr=False)
    _intensity_array: Optional[np.ndarray] = field(default=None, repr=False)
    _origin_array: Optional[np.ndarray] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Build cached arrays for fast access."""
        if self.modes:
            self._k_array = np.array([m.k for m in self.modes])
            self._f_array = np.array([m.omega for m in self.modes])
            self._intensity_array = np.array([m.intensity for m in self.modes])
            self._origin_array = np.array([m.origin_BZ for m in self.modes])
    
    @property
    def n_modes(self) -> int:
        """Total number of detected modes."""
        return len(self.modes)
    
    @property
    def n_branches(self) -> int:
        """Number of distinct branches."""
        return len(self.branches)
    
    @property
    def k_values(self) -> np.ndarray:
        """All k values as array [rad/m]."""
        return self._k_array if self._k_array is not None else np.array([])
    
    @property
    def f_values(self) -> np.ndarray:
        """All frequency values as array [Hz]."""
        return self._f_array if self._f_array is not None else np.array([])
    
    @property
    def intensities(self) -> np.ndarray:
        """All intensity values as array."""
        return self._intensity_array if self._intensity_array is not None else np.array([])
    
    @property
    def origins(self) -> np.ndarray:
        """All BZ origin indices as array."""
        return self._origin_array if self._origin_array is not None else np.array([])
    
    def filter_frequency(self, f_min: float = 0, f_max: float = np.inf) -> "FoldedDispersionResult":
        """Return new result filtered by frequency range."""
        filtered_modes = [m for m in self.modes if f_min <= m.omega <= f_max]
        
        # Rebuild branches
        new_branches: Dict[int, List[DispersionMode]] = {}
        for mode in filtered_modes:
            if mode.branch_index not in new_branches:
                new_branches[mode.branch_index] = []
            new_branches[mode.branch_index].append(mode)
        
        return FoldedDispersionResult(
            modes=filtered_modes,
            k_fbz=self.k_fbz,
            branches=new_branches,
            bz_config=self.bz_config,
            original_result=self.original_result,
        )
    
    def filter_intensity(self, min_intensity: float) -> "FoldedDispersionResult":
        """Return new result filtered by minimum intensity."""
        filtered_modes = [m for m in self.modes if m.intensity >= min_intensity]
        
        new_branches: Dict[int, List[DispersionMode]] = {}
        for mode in filtered_modes:
            if mode.branch_index not in new_branches:
                new_branches[mode.branch_index] = []
            new_branches[mode.branch_index].append(mode)
        
        return FoldedDispersionResult(
            modes=filtered_modes,
            k_fbz=self.k_fbz,
            branches=new_branches,
            bz_config=self.bz_config,
            original_result=self.original_result,
        )
    
    def get_branch(self, branch_index: int) -> List[DispersionMode]:
        """Get all modes belonging to a specific branch."""
        return self.branches.get(branch_index, [])
    
    def get_branch_arrays(self, branch_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get (k, f) arrays for a specific branch, sorted by k."""
        modes = self.get_branch(branch_index)
        if not modes:
            return np.array([]), np.array([])
        
        k_vals = np.array([m.k for m in modes])
        f_vals = np.array([m.omega for m in modes])
        
        # Sort by k
        order = np.argsort(k_vals)
        return k_vals[order], f_vals[order]
    
    def find_mode_nearest(self, k: float, f: float) -> Optional[DispersionMode]:
        """
        Find the mode nearest to the specified (k, f) point.
        
        Parameters
        ----------
        k : float
            Wave vector [rad/m]
        f : float
            Frequency [Hz]
            
        Returns
        -------
        DispersionMode or None
            Nearest mode, or None if no modes exist
        """
        if not self.modes:
            return None
        
        # Normalize for distance calculation
        k_scale = self.bz_config.k_bz_boundary
        f_scale = np.max(self.f_values) if len(self.f_values) > 0 else 1e9
        
        min_dist = float('inf')
        nearest = None
        
        for mode in self.modes:
            dist = ((mode.k - k) / k_scale) ** 2 + ((mode.omega - f) / f_scale) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest = mode
        
        return nearest
    
    def find_modes_at_frequency(
        self, 
        f: float, 
        tolerance: float = 0.1e9
    ) -> List[DispersionMode]:
        """
        Find all modes at a specific frequency.
        
        Parameters
        ----------
        f : float
            Target frequency [Hz]
        tolerance : float
            Frequency tolerance [Hz], default 0.1 GHz
            
        Returns
        -------
        List[DispersionMode]
            Modes within tolerance of target frequency
        """
        return [m for m in self.modes if np.abs(m.omega - f) <= tolerance]
    
    def find_modes_at_k(
        self, 
        k: float, 
        tolerance: Optional[float] = None
    ) -> List[DispersionMode]:
        """
        Find all modes at a specific wave vector.
        
        Parameters
        ----------
        k : float
            Target wave vector [rad/m]
        tolerance : float, optional
            k tolerance [rad/m]. Default: 1% of BZ width
            
        Returns
        -------
        List[DispersionMode]
            Modes within tolerance of target k
        """
        if tolerance is None:
            tolerance = 0.01 * self.bz_config.bz_width
        return [m for m in self.modes if np.abs(m.k - k) <= tolerance]
    
    def get_modes_by_origin(self, origin_bz: int) -> List[DispersionMode]:
        """
        Get all modes originating from a specific Brillouin zone.
        
        Parameters
        ----------
        origin_bz : int
            BZ index (0 = first BZ, ±1 = second BZ, etc.)
            
        Returns
        -------
        List[DispersionMode]
            Modes from the specified BZ
        """
        return [m for m in self.modes if m.origin_BZ == origin_bz]
    
    def get_strongest_modes(self, n: int = 10) -> List[DispersionMode]:
        """
        Get the n strongest modes by intensity.
        
        Parameters
        ----------
        n : int
            Number of modes to return
            
        Returns
        -------
        List[DispersionMode]
            Top n modes sorted by intensity (descending)
        """
        sorted_modes = sorted(self.modes, key=lambda m: m.intensity, reverse=True)
        return sorted_modes[:n]
    
    def summary(self) -> str:
        """Generate a text summary of the folded dispersion."""
        lines = [
            "=" * 60,
            "Folded Dispersion Result Summary",
            "=" * 60,
            f"Lattice constant: {self.bz_config.lattice_constant * 1e9:.1f} nm",
            f"BZ boundary: ±{self.bz_config.k_bz_boundary / 1e6:.3f} rad/μm",
            f"Periods considered: {self.bz_config.n_periods}",
            f"Total modes found: {self.n_modes}",
            f"Number of branches: {self.n_branches}",
            "-" * 60,
            "Branches:",
        ]
        
        for branch_idx in sorted(self.branches.keys()):
            branch_modes = self.branches[branch_idx]
            k_min = min(m.k for m in branch_modes) / 1e6
            k_max = max(m.k for m in branch_modes) / 1e6
            f_min = min(m.omega for m in branch_modes) / 1e9
            f_max = max(m.omega for m in branch_modes) / 1e9
            lines.append(
                f"  Branch {branch_idx}: {len(branch_modes)} modes, "
                f"k=[{k_min:.2f}, {k_max:.2f}] rad/μm, "
                f"f=[{f_min:.2f}, {f_max:.2f}] GHz"
            )
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (
            f"FoldedDispersionResult("
            f"n_modes={self.n_modes}, "
            f"n_branches={self.n_branches}, "
            f"a={self.bz_config.lattice_constant*1e9:.1f} nm)"
        )
