"""
Data models for FMR mode analysis.

This module contains lightweight data structures used throughout
the FMR mode analysis system, without matplotlib dependencies.
"""

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np


@dataclass
class Peak:
    """Peak data structure for spectrum analysis."""

    idx: int
    freq: float
    amplitude: float

    def __post_init__(self):
        """Validate peak data."""
        if isinstance(self.idx, (bool, np.bool_)) or not isinstance(self.idx, Integral):
            raise TypeError("Peak index must be an integer")
        self.idx = int(self.idx)
        if self.idx < 0:
            raise ValueError("Peak index must be non-negative")
        if isinstance(self.freq, (bool, np.bool_)) or not isinstance(self.freq, Real):
            raise TypeError("Peak frequency must be numeric")
        if isinstance(self.amplitude, (bool, np.bool_)) or not isinstance(
            self.amplitude, Real
        ):
            raise TypeError("Peak amplitude must be numeric")
        self.freq = float(self.freq)
        self.amplitude = float(self.amplitude)
        if not np.isfinite(self.freq) or self.freq < 0:
            raise ValueError("Peak frequency must be finite and non-negative")
        if not np.isfinite(self.amplitude) or self.amplitude < 0:
            raise ValueError("Peak amplitude must be finite and non-negative")

    @property
    def is_valid(self) -> bool:
        """Check if peak data is valid."""
        return (
            isinstance(self.idx, int)
            and self.idx >= 0
            and np.isfinite(self.freq)
            and self.freq >= 0
            and np.isfinite(self.amplitude)
            and self.amplitude >= 0
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
        extent: tuple[float, float, float, float] | None = None,
        metadata: dict[str, Any] | None = None,
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
        self.extent = (
            (0, mode_array.shape[1], 0, mode_array.shape[0])
            if extent is None
            else tuple(extent)
        )
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary or None")
        self.metadata = dict(metadata or {})

        # Validate input
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input parameters."""
        if not isinstance(self.mode_array, np.ndarray):
            raise TypeError("mode_array must be numpy array")

        if self.mode_array.ndim != 3:
            raise ValueError(f"mode_array must be 3D, got {self.mode_array.ndim}D")

        if self.mode_array.shape[2] != 3:
            raise ValueError(
                f"mode_array must have 3 components, got {self.mode_array.shape[2]}"
            )

        if isinstance(self.frequency, (bool, np.bool_)) or not isinstance(
            self.frequency, Real
        ):
            raise TypeError("frequency must be numeric")
        self.frequency = float(self.frequency)
        if not np.isfinite(self.frequency) or self.frequency < 0:
            raise ValueError("frequency must be finite and non-negative")

        if len(self.extent) != 4:
            raise ValueError("extent must have 4 elements [x_min, x_max, y_min, y_max]")
        try:
            self.extent = tuple(float(value) for value in self.extent)
        except (TypeError, ValueError) as exc:
            raise TypeError("extent values must be numeric") from exc
        if not np.all(np.isfinite(self.extent)):
            raise ValueError("extent values must be finite")
        if self.extent[0] >= self.extent[1] or self.extent[2] >= self.extent[3]:
            raise ValueError("extent bounds must be strictly increasing")
        if self.mode_array.shape[0] == 0 or self.mode_array.shape[1] == 0:
            raise ValueError("mode_array spatial dimensions must be non-empty")

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

    def get_component(self, component: int | str) -> np.ndarray:
        """
        Get specific magnetization component.

        Parameters:
        -----------
        component : int or str
            Component index (0, 1, 2) or name:
            - Cartesian: 'x', 'y', 'z', 'mx', 'my', 'mz'
            - Circular: '+', '-' using (mx ± i*my)/sqrt(2)
            - Cylindrical: 'rho', 'phi' (radial/azimuthal)
            - Special: 'magnitude' (total magnitude)

        Returns:
        --------
        np.ndarray
            Complex array for specified component (2D spatial array)
        """
        if isinstance(component, str):
            # Handle magnitude component specially
            if component.lower() == "magnitude":
                # Return magnitude across all components
                return np.sqrt(np.sum(np.abs(self.mode_array) ** 2, axis=-1))

            # Circular basis (helical/chiral modes)
            if component in ["+", "-"]:
                from .vortex_optics import VortexOptics

                m_x = self.mode_array[:, :, 0]
                m_y = self.mode_array[:, :, 1]
                m_plus, m_minus = VortexOptics.to_circular_basis(m_x, m_y)
                return m_plus if component == "+" else m_minus

            # Cylindrical basis (radial/azimuthal modes)
            if component in ["rho", "phi"]:
                from .vortex_optics import VortexOptics

                m_x = self.mode_array[:, :, 0]
                m_y = self.mode_array[:, :, 1]
                # Use geometric center by default
                m_rho, m_phi = VortexOptics.to_cylindrical_basis(m_x, m_y, center=None)
                return m_rho if component == "rho" else m_phi

            # Standard Cartesian components
            component_map = {"x": 0, "mx": 0, "y": 1, "my": 1, "z": 2, "mz": 2}
            if component.lower() not in component_map:
                raise ValueError(
                    f"Unknown component: {component}. "
                    f"Supported: x/y/z (Cartesian), +/- (circular), rho/phi (cylindrical)"
                )
            component = component_map[component.lower()]

        if isinstance(component, (bool, np.bool_)) or not isinstance(
            component, Integral
        ):
            raise TypeError("component must be int or str")
        component = int(component)
        if component < 0 or component >= 3:
            raise ValueError(f"component index must be 0, 1, or 2, got {component}")

        return self.mode_array[:, :, component]

    def get_component_magnitude(self, component: int | str) -> np.ndarray:
        """Get magnitude for specific component."""
        return np.abs(self.get_component(component))

    def get_component_phase(self, component: int | str) -> np.ndarray:
        """Get phase for specific component."""
        return np.angle(self.get_component(component))

    def get_real_part(self, component: int | str) -> np.ndarray:
        """Get real part for specific component."""
        return np.real(self.get_component(component))

    def get_imaginary_part(self, component: int | str) -> np.ndarray:
        """Get imaginary part for specific component."""
        return np.imag(self.get_component(component))

    def interpolate_to_grid(self, new_shape: tuple[int, int]) -> "FMRModeData":
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
        if not isinstance(new_shape, tuple) or len(new_shape) != 2:
            raise TypeError("new_shape must be a (ny, nx) tuple")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or int(value) < 1
            for value in new_shape
        ):
            raise ValueError("new_shape dimensions must be positive integers")

        from scipy.ndimage import zoom

        old_shape = self.spatial_shape
        target_shape = (int(new_shape[0]), int(new_shape[1]))
        zoom_factors = (
            target_shape[0] / old_shape[0],
            target_shape[1] / old_shape[1],
            1,
        )

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
            metadata=self.metadata.copy(),
        )

    def crop_to_region(
        self, x_range: tuple[float, float], y_range: tuple[float, float]
    ) -> "FMRModeData":
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
        try:
            requested_x0, requested_x1 = (float(value) for value in x_range)
            requested_y0, requested_y1 = (float(value) for value in y_range)
        except (TypeError, ValueError) as exc:
            raise ValueError("x_range and y_range must contain two numbers") from exc
        requested = (requested_x0, requested_x1, requested_y0, requested_y1)
        if not np.all(np.isfinite(requested)):
            raise ValueError("Crop ranges must be finite")
        if requested_x0 >= requested_x1 or requested_y0 >= requested_y1:
            raise ValueError("Crop ranges must be strictly increasing")

        x_min, x_max, y_min, y_max = self.extent
        nx, ny = self.spatial_shape[1], self.spatial_shape[0]

        crop_x0 = max(requested_x0, x_min)
        crop_x1 = min(requested_x1, x_max)
        crop_y0 = max(requested_y0, y_min)
        crop_y1 = min(requested_y1, y_max)
        if crop_x0 >= crop_x1 or crop_y0 >= crop_y1:
            raise ValueError("Requested crop does not overlap the mode extent")

        # Convert spatial coordinates to array indices
        x_idx_min = int(np.floor((crop_x0 - x_min) / (x_max - x_min) * nx))
        x_idx_max = int(np.ceil((crop_x1 - x_min) / (x_max - x_min) * nx))
        y_idx_min = int(np.floor((crop_y0 - y_min) / (y_max - y_min) * ny))
        y_idx_max = int(np.ceil((crop_y1 - y_min) / (y_max - y_min) * ny))

        # Ensure indices are within bounds
        x_idx_min = max(0, min(x_idx_min, nx))
        x_idx_max = max(0, min(x_idx_max, nx))
        y_idx_min = max(0, min(y_idx_min, ny))
        y_idx_max = max(0, min(y_idx_max, ny))

        # Crop the array
        cropped_array = self.mode_array[y_idx_min:y_idx_max, x_idx_min:x_idx_max, :]

        # Update extent
        new_extent = (
            x_min + x_idx_min / nx * (x_max - x_min),
            x_min + x_idx_max / nx * (x_max - x_min),
            y_min + y_idx_min / ny * (y_max - y_min),
            y_min + y_idx_max / ny * (y_max - y_min),
        )

        return FMRModeData(
            frequency=self.frequency,
            mode_array=cropped_array,
            extent=new_extent,
            metadata=self.metadata.copy(),
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
            "max_amplitude": float(np.max(total_mag)),
            "mean_amplitude": float(np.mean(total_mag)),
            "std_amplitude": float(np.std(total_mag)),
            "total_power": float(np.sum(total_mag**2)),
            "spatial_extent_x": self.width_nm,
            "spatial_extent_y": self.height_nm,
            "frequency_ghz": self.frequency,
        }

    def __str__(self) -> str:
        return (
            f"FMRModeData(freq={self.frequency:.3f} GHz, "
            f"shape={self.spatial_shape}, "
            f"extent={self.width_nm:.1f}×{self.height_nm:.1f} nm²)"
        )

    def __repr__(self) -> str:
        return (
            f"FMRModeData(frequency={self.frequency}, "
            f"mode_array.shape={self.mode_array.shape}, "
            f"extent={self.extent})"
        )

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

    def copy(self) -> "FMRModeData":
        """Create a deep copy of the mode data."""
        return FMRModeData(
            frequency=self.frequency,
            mode_array=self.mode_array.copy(),
            extent=self.extent,
            metadata=self.metadata.copy(),
        )
