"""
Core data models for spin-wave dispersion analysis.

Defines result structures and configuration classes for dispersion calculations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import numpy as np


@dataclass
class DispersionConfig:
    """Configuration parameters for dispersion analysis."""
    
    # Time domain processing
    dt: float = 1e-12  # Time step [s] - default value
    time_window: Optional[str] = "hann"  # Window function for time domain
    detrend: str = "mean"  # Detrending method: 'mean', 'initial', None
    
    # Spatial processing  
    dx: Optional[float] = None  # Grid spacing in x [m]
    dy: Optional[float] = None  # Grid spacing in y [m]
    dz: Optional[float] = None  # Grid spacing in z [m] 
    space_window: Optional[str] = None  # Window function for spatial domain
    avg_over_orthogonal: bool = True  # Average over orthogonal directions
    orthogonal_avg_mode: str = "magnetization"  # How to collapse orthogonal axis

    # Component selection
    component: str = "perp"  # 'perp', 'mx', 'my', 'mz', 'sum'
    
    # Brillouin zone folding
    fold_period: Optional[float] = None  # Real-space period [m] for BZ folding
    fold_agg: str = "sum"  # Aggregation method: 'sum' or 'max'
    
    # Branch tracking
    dk_max: float = 1e5  # Max k-deviation for sampling [rad/m]
    df_max: Optional[float] = None  # Max f-deviation for branch tracking [Hz]
    min_prominence: float = 0.0  # Minimum peak prominence for detection


@dataclass  
class DispersionResult1D:
    """Results from 1D dispersion analysis S(k,f)."""
    
    # Core data
    S: np.ndarray  # Spectral power (Nk, Nf)
    k_axis: np.ndarray  # Wave vector axis [rad/m]
    f_axis: np.ndarray  # Frequency axis [Hz]
    
    # Analysis parameters
    axis: str  # Propagation direction: 'x' or 'y'
    component: str  # Analyzed component
    config: DispersionConfig
    
    # Optional folded data
    S_folded: Optional[np.ndarray] = None  # Folded spectrum
    k_folded: Optional[np.ndarray] = None  # Folded k-axis
    fold_period: Optional[float] = None  # Folding period

    # Optional local spectra when orthogonal averaging is disabled
    S_local: Optional[np.ndarray] = None  # (N_orthogonal, Nk, Nf)
    orth_axis: Optional[np.ndarray] = None  # Coordinate values along orthogonal axis
    orth_axis_label: Optional[str] = None  # Name of orthogonal axis ('x' or 'y')
    
    # Complex FFT data for mode reconstruction (avoids re-computing FFT)
    S_complex: Optional[np.ndarray] = None  # Complex spectrum (Nk, Nf) or (N_orth, Nk, Nf)
    
    # Metadata
    dt: float = 0.0
    dx: float = 0.0
    flipx: bool = True  # Whether k-axis was flipped to correct FFT convention
    notes: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.notes is None:
            self.notes = []
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of dispersion array (Nk, Nf)."""
        return self.S.shape
    
    @property 
    def k_range(self) -> Tuple[float, float]:
        """Wave vector range [rad/m]."""
        return (self.k_axis.min(), self.k_axis.max())
        
    @property
    def f_range(self) -> Tuple[float, float]:
        """Frequency range [Hz]."""
        return (self.f_axis.min(), self.f_axis.max())
        
    @property
    def is_folded(self) -> bool:
        """Whether BZ folding was applied."""
        return self.S_folded is not None
        
    def get_active_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get currently active S, k, f data (folded if available)."""
        if self.is_folded and self.S_folded is not None and self.k_folded is not None:
            return self.S_folded, self.k_folded, self.f_axis
        return self.S, self.k_axis, self.f_axis
    
    def sample_at_k(self, k_query: float, dk_max: Optional[float] = None) -> Tuple[float, float]:
        """Sample dispersion at given k, return (k_eff, f_peak)."""
        if dk_max is None:
            dk_max = self.config.dk_max
            
        S, k_axis, f_axis = self.get_active_data()
        
        mask = np.abs(k_axis - k_query) <= dk_max
        if not np.any(mask):
            raise ValueError(f"No k within {dk_max} of {k_query}")
            
        S_slice = S[mask, :].sum(axis=0)
        idx = np.argmax(S_slice)
        f_peak = f_axis[idx]
        k_eff = float(np.average(k_axis[mask], weights=S[mask, idx] + 1e-12))
        
        return k_eff, float(f_peak)

    def select_orthogonal_slice(self, index: int) -> "DispersionResult1D":
        """Create a new result containing a single orthogonal slice."""
        if self.S_local is None:
            raise ValueError("No orthogonal slices stored; recompute with avg_over_orthogonal=False")
        if index < 0 or index >= self.S_local.shape[0]:
            raise IndexError(f"Orthogonal index {index} out of bounds (0..{self.S_local.shape[0]-1})")

        slice_notes = list(self.notes or []) + [f"Orthogonal slice #{index}"]
        slice_result = DispersionResult1D(
            S=self.S_local[index],
            k_axis=self.k_axis,
            f_axis=self.f_axis,
            axis=self.axis,
            component=self.component,
            config=self.config,
            S_folded=self.S_folded,
            k_folded=self.k_folded,
            fold_period=self.fold_period,
            dt=self.dt,
            dx=self.dx,
            notes=slice_notes,
        )
        return slice_result


@dataclass
class DispersionResult2D:
    """Results from 2D dispersion analysis S(kx,ky,f)."""
    
    # Core data
    S: np.ndarray  # Spectral power (Nkx, Nky, Nf)
    kx_axis: np.ndarray  # kx wave vector axis [rad/m]
    ky_axis: np.ndarray  # ky wave vector axis [rad/m] 
    f_axis: np.ndarray  # Frequency axis [Hz]
    
    # Analysis parameters
    component: str  # Analyzed component
    config: DispersionConfig
    
    # Metadata
    dt: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    notes: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.notes is None:
            self.notes = []
            
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of dispersion array (Nkx, Nky, Nf)."""
        return self.S.shape
        
    @property
    def kx_range(self) -> Tuple[float, float]:
        """kx range [rad/m]."""
        return (self.kx_axis.min(), self.kx_axis.max())
        
    @property  
    def ky_range(self) -> Tuple[float, float]:
        """ky range [rad/m]."""
        return (self.ky_axis.min(), self.ky_axis.max())
        
    @property
    def f_range(self) -> Tuple[float, float]:
        """Frequency range [Hz]."""
        return (self.f_axis.min(), self.f_axis.max())
        
    def slice_1d(self, direction: str, k_value: float = 0.0, dk_max: Optional[float] = None) -> DispersionResult1D:
        """Extract 1D slice along kx or ky direction."""
        if dk_max is None:
            dk_max = self.config.dk_max
            
        if direction == 'kx':
            # Slice along kx at fixed ky
            mask = np.abs(self.ky_axis - k_value) <= dk_max
            if not np.any(mask):
                raise ValueError(f"No ky within {dk_max} of {k_value}")
            S_1d = self.S[:, mask, :].mean(axis=1)  # Average over ky slice
            k_axis = self.kx_axis
            dx = self.dx
            axis = 'x'
            
        elif direction == 'ky':
            # Slice along ky at fixed kx  
            mask = np.abs(self.kx_axis - k_value) <= dk_max
            if not np.any(mask):
                raise ValueError(f"No kx within {dk_max} of {k_value}")
            S_1d = self.S[mask, :, :].mean(axis=0)  # Average over kx slice
            k_axis = self.ky_axis  
            dx = self.dy
            axis = 'y'
        else:
            raise ValueError("direction must be 'kx' or 'ky'")
            
        return DispersionResult1D(
            S=S_1d,
            k_axis=k_axis,
            f_axis=self.f_axis,
            axis=axis,
            component=self.component,
            config=self.config,
            dt=self.dt,
            dx=dx,
            notes=(self.notes or []) + [f"1D slice from 2D at {direction}={k_value}"]
        )


@dataclass
class DispersionBranch:
    """A tracked dispersion branch f(k)."""
    
    # Branch data
    k_path: np.ndarray  # Wave vector path [rad/m]
    f_values: np.ndarray  # Frequencies [Hz] 
    amplitudes: np.ndarray  # Spectral amplitudes at (k,f) points
    
    # Branch properties  
    branch_id: int = 0  # Branch identifier
    mode_type: Optional[str] = None  # Mode classification
    group_velocity: Optional[np.ndarray] = None  # dω/dk [Hz⋅m]
    
    # Analysis metadata
    tracking_config: Optional[Dict[str, Any]] = None
    notes: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.tracking_config is None:
            self.tracking_config = {}
        if self.notes is None:
            self.notes = []
            
    @property
    def length(self) -> int:
        """Number of points in branch."""
        return len(self.k_path)
        
    @property  
    def k_range(self) -> Tuple[float, float]:
        """Wave vector range [rad/m]."""
        return (self.k_path.min(), self.k_path.max())
        
    @property
    def f_range(self) -> Tuple[float, float]:
        """Frequency range [Hz]."""
        return (self.f_values.min(), self.f_values.max())
        
    def compute_group_velocity(self, smooth: bool = True) -> np.ndarray:
        """Compute group velocity dω/dk = 2π⋅df/dk."""
        if smooth:
            # Use gradient with smoothing
            vg = 2 * np.pi * np.gradient(self.f_values, self.k_path)
        else:
            # Simple finite differences
            dk = np.diff(self.k_path)
            df = np.diff(self.f_values)
            vg_interior = 2 * np.pi * df / dk
            # Pad edges
            vg = np.concatenate([[vg_interior[0]], vg_interior, [vg_interior[-1]]])
            
        self.group_velocity = vg
        return vg
        
    def interpolate_at_k(self, k_query: np.ndarray) -> np.ndarray:
        """Interpolate branch frequencies at query k values.""" 
        return np.interp(k_query, self.k_path, self.f_values)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'k_path': self.k_path.tolist(),
            'f_values': self.f_values.tolist(), 
            'amplitudes': self.amplitudes.tolist(),
            'branch_id': self.branch_id,
            'mode_type': self.mode_type,
            'group_velocity': self.group_velocity.tolist() if self.group_velocity is not None else None,
            'tracking_config': self.tracking_config or {},
            'notes': self.notes or [],
            'k_range': self.k_range,
            'f_range': self.f_range
        }
