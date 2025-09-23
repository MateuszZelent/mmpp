"""
Data models for FMR mode analysis.

This module contains lightweight data structures used throughout
the FMR mode analysis system, without matplotlib dependencies.
"""

from dataclasses import dataclass
from typing import Union, Optional, Any
import numpy as np


@dataclass
class Peak:
    """Peak data structure for spectrum analysis."""

    idx: int
    freq: float
    amplitude: float

    def __post_init__(self):
        """Validate peak data."""
        if self.idx < 0:
            raise ValueError("Peak index must be non-negative")
        if self.freq < 0:
            raise ValueError("Peak frequency must be non-negative")
        if self.amplitude < 0:
            raise ValueError("Peak amplitude must be non-negative")

    @property
    def is_valid(self) -> bool:
        """Check if peak data is valid."""
        return (
            isinstance(self.idx, int) and self.idx >= 0 and
            isinstance(self.freq, (int, float)) and self.freq >= 0 and
            isinstance(self.amplitude, (int, float)) and self.amplitude >= 0
        )

    def __str__(self) -> str:
        return f"Peak(freq={self.freq:.3f} GHz, amp={self.amplitude:.3f})"

    def __repr__(self) -> str:
        return f"Peak(idx={self.idx}, freq={self.freq}, amplitude={self.amplitude})"


class FMRModeData:
    """Container for FMR mode data at a specific frequency."""

    def __init__(
        self,
        frequency: float,
        mode_array: np.ndarray,
        extent: Optional[tuple[float, float, float, float]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize FMR mode data.

        Parameters:
        -----------
        frequency : float
            Frequency in GHz
        mode_array : np.ndarray
            Complex mode array with shape (ny, nx, 3) for spatial x-y and magnetization components
        extent : tuple, optional
            Spatial extent [x_min, x_max, y_min, y_max] in nm
        metadata : dict, optional
            Additional metadata
        """
        self.frequency = frequency
        self.mode_array = mode_array
        self.extent = extent or (0, mode_array.shape[1], 0, mode_array.shape[0])
        self.metadata = metadata or {}

        # Validate input
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input parameters."""
        if not isinstance(self.mode_array, np.ndarray):
            raise TypeError("mode_array must be numpy array")
        
        if self.mode_array.ndim != 3:
            raise ValueError(f"mode_array must be 3D, got {self.mode_array.ndim}D")
        
        if self.mode_array.shape[2] != 3:
            raise ValueError(f"mode_array must have 3 components, got {self.mode_array.shape[2]}")
        
        if not isinstance(self.frequency, (int, float)):
            raise TypeError("frequency must be numeric")
        
        if self.frequency < 0:
            raise ValueError("frequency must be non-negative")
        
        if len(self.extent) != 4:
            raise ValueError("extent must have 4 elements [x_min, x_max, y_min, y_max]")

    @property
    def shape(self) -> tuple[int, int, int]:
        """Get shape of mode array."""
        return self.mode_array.shape

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Get spatial dimensions (ny, nx)."""
        return self.mode_array.shape[:2]

    @property
    def magnitude(self) -> np.ndarray:
        """Get magnitude of mode for each component."""
        return np.abs(self.mode_array)

    @property
    def phase(self) -> np.ndarray:
        """Get phase of mode for each component."""
        return np.angle(self.mode_array)

    @property
    def total_magnitude(self) -> np.ndarray:
        """Get total magnitude across all components."""
        return np.sqrt(np.sum(self.magnitude**2, axis=2))

    @property
    def max_amplitude(self) -> float:
        """Get maximum amplitude across all components."""
        return float(np.max(self.total_magnitude))

    @property
    def spatial_extent_nm(self) -> tuple[float, float, float, float]:
        """Get spatial extent in nanometers."""
        return self.extent

    @property
    def width_nm(self) -> float:
        """Get width in nanometers."""
        return self.extent[1] - self.extent[0]

    @property
    def height_nm(self) -> float:
        """Get height in nanometers."""
        return self.extent[3] - self.extent[2]

    def get_component(self, component: Union[int, str]) -> np.ndarray:
        """
        Get specific magnetization component.

        Parameters:
        -----------
        component : int or str
            Component index (0, 1, 2) or name ('x', 'y', 'z', 'mx', 'my', 'mz')

        Returns:
        --------
        np.ndarray
            Complex array for specified component
        """
        if isinstance(component, str):
            # Handle magnitude component specially
            if component.lower() == 'magnitude':
                # Return magnitude across all components
                return np.sqrt(np.sum(np.abs(self.mode_array)**2, axis=-1))
            
            component_map = {
                'x': 0, 'mx': 0,
                'y': 1, 'my': 1,
                'z': 2, 'mz': 2
            }
            if component.lower() not in component_map:
                raise ValueError(f"Unknown component name: {component}")
            component = component_map[component.lower()]
        
        if not isinstance(component, int):
            raise TypeError("component must be int or str")
        
        if component < 0 or component >= 3:
            raise ValueError(f"component index must be 0, 1, or 2, got {component}")
        
        return self.mode_array[:, :, component]

    def get_component_magnitude(self, component: Union[int, str]) -> np.ndarray:
        """Get magnitude for specific component."""
        return np.abs(self.get_component(component))

    def get_component_phase(self, component: Union[int, str]) -> np.ndarray:
        """Get phase for specific component."""
        return np.angle(self.get_component(component))

    def get_real_part(self, component: Union[int, str]) -> np.ndarray:
        """Get real part for specific component."""
        return np.real(self.get_component(component))

    def get_imaginary_part(self, component: Union[int, str]) -> np.ndarray:
        """Get imaginary part for specific component."""
        return np.imag(self.get_component(component))

    def interpolate_to_grid(self, new_shape: tuple[int, int]) -> 'FMRModeData':
        """
        Interpolate mode data to new grid size.
        
        Parameters:
        -----------
        new_shape : tuple[int, int]
            New spatial shape (ny, nx)
            
        Returns:
        --------
        FMRModeData
            New instance with interpolated data
        """
        from scipy.ndimage import zoom
        
        old_shape = self.spatial_shape
        zoom_factors = (new_shape[0] / old_shape[0], new_shape[1] / old_shape[1], 1)
        
        # Interpolate real and imaginary parts separately
        real_part = np.real(self.mode_array)
        imag_part = np.imag(self.mode_array)
        
        real_interp = zoom(real_part, zoom_factors, order=1)
        imag_interp = zoom(imag_part, zoom_factors, order=1)
        
        new_array = real_interp + 1j * imag_interp
        
        return FMRModeData(
            frequency=self.frequency,
            mode_array=new_array,
            extent=self.extent,
            metadata=self.metadata.copy()
        )

    def crop_to_region(self, x_range: tuple[float, float], y_range: tuple[float, float]) -> 'FMRModeData':
        """
        Crop mode data to specified spatial region.
        
        Parameters:
        -----------
        x_range : tuple[float, float]
            X range in nm (x_min, x_max)
        y_range : tuple[float, float]
            Y range in nm (y_min, y_max)
            
        Returns:
        --------
        FMRModeData
            New instance with cropped data
        """
        x_min, x_max, y_min, y_max = self.extent
        nx, ny = self.spatial_shape[1], self.spatial_shape[0]
        
        # Convert spatial coordinates to array indices
        x_idx_min = int((x_range[0] - x_min) / (x_max - x_min) * nx)
        x_idx_max = int((x_range[1] - x_min) / (x_max - x_min) * nx)
        y_idx_min = int((y_range[0] - y_min) / (y_max - y_min) * ny)
        y_idx_max = int((y_range[1] - y_min) / (y_max - y_min) * ny)
        
        # Ensure indices are within bounds
        x_idx_min = max(0, min(x_idx_min, nx))
        x_idx_max = max(0, min(x_idx_max, nx))
        y_idx_min = max(0, min(y_idx_min, ny))
        y_idx_max = max(0, min(y_idx_max, ny))
        
        # Crop the array
        cropped_array = self.mode_array[y_idx_min:y_idx_max, x_idx_min:x_idx_max, :]
        
        # Update extent
        new_extent = (x_range[0], x_range[1], y_range[0], y_range[1])
        
        return FMRModeData(
            frequency=self.frequency,
            mode_array=cropped_array,
            extent=new_extent,
            metadata=self.metadata.copy()
        )

    def compute_statistics(self) -> dict[str, float]:
        """
        Compute basic statistics for the mode data.
        
        Returns:
        --------
        dict[str, float]
            Dictionary with statistical measures
        """
        total_mag = self.total_magnitude
        
        return {
            'max_amplitude': float(np.max(total_mag)),
            'mean_amplitude': float(np.mean(total_mag)),
            'std_amplitude': float(np.std(total_mag)),
            'total_power': float(np.sum(total_mag**2)),
            'spatial_extent_x': self.width_nm,
            'spatial_extent_y': self.height_nm,
            'frequency_ghz': self.frequency
        }

    def __str__(self) -> str:
        return (f"FMRModeData(freq={self.frequency:.3f} GHz, "
                f"shape={self.spatial_shape}, "
                f"extent={self.width_nm:.1f}×{self.height_nm:.1f} nm²)")

    def __repr__(self) -> str:
        return (f"FMRModeData(frequency={self.frequency}, "
                f"mode_array.shape={self.mode_array.shape}, "
                f"extent={self.extent})")
    
    def to_magnitude_phase(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert mode data to magnitude and phase arrays.
        
        Returns:
        --------
        tuple[np.ndarray, np.ndarray]
            Magnitude and phase arrays with shape (spatial_shape)
        """
        magnitude = np.abs(self.mode_array)
        phase = np.angle(self.mode_array)
        return magnitude, phase

    def copy(self) -> 'FMRModeData':
        """Create a deep copy of the mode data."""
        return FMRModeData(
            frequency=self.frequency,
            mode_array=self.mode_array.copy(),
            extent=self.extent,
            metadata=self.metadata.copy()
        )