"""
Mode Data Loader Module

Provides slice-aware data loading for FMR mode analysis.
Supports component extraction and time slicing from DatasetAwareWrapper.
"""

import hashlib
import logging
from dataclasses import dataclass, field

import numpy as np

# Get logger
log = logging.getLogger("mmpp.fft.modes")

# Component labels for magnetization
COMPONENT_LABELS = [r"$m_x$", r"$m_y$", r"$m_z$"]
COMPONENT_NAMES = ["mx", "my", "mz"]


@dataclass
class ModeDataContext:
    """Context for mode data loading with slice support.

    Attributes
    ----------
    zarr_path : str
        Path to zarr file
    dataset_name : str
        Dataset name (e.g., 'm', 'm_x11')
    slice_info : tuple, optional
        Slice info from DatasetAwareWrapper, e.g., (slice(0,200), ..., 1)
    z_layer : int
        Z-layer index (default: -1 for top layer)
    component_index : int, optional
        Magnetization component: 0=mx, 1=my, 2=mz
        If None, all components are loaded
    time_slice : slice, optional
        Extracted time slice from slice_info
    """

    zarr_path: str
    dataset_name: str
    slice_info: tuple | None = None
    z_layer: int = -1
    component_index: int | None = None
    time_slice: slice | None = None
    mode_group: str | None = None
    time_step_scale: float = 1.0

    # Derived fields (set in __post_init__)
    component_label: str | None = field(default=None, init=False)

    def __post_init__(self):
        """Parse slice_info to extract component and time slice."""
        if self.slice_info is not None:
            self._parse_slice_info()

        # Set component label
        if self.component_index is not None and 0 <= self.component_index <= 2:
            self.component_label = COMPONENT_LABELS[self.component_index]

        if self.mode_group is None:
            self.mode_group = f"modes/{self.dataset_name}"
            if self.slice_info is not None:
                identity = f"{self.slice_info!r};dt_scale={float(self.time_step_scale)}"
                view_id = hashlib.blake2b(identity.encode(), digest_size=8).hexdigest()
                self.mode_group = f"{self.mode_group}/views/{view_id}"

    def _parse_slice_info(self):
        """Parse slice_info tuple to extract component and time slice."""
        if not isinstance(self.slice_info, tuple):
            return

        # Check last element for component index
        if len(self.slice_info) > 0:
            last = self.slice_info[-1]
            if isinstance(last, (int, np.integer)):
                idx = int(last)
                if idx in (0, 1, 2):
                    self.component_index = idx
                    log.debug(f"Extracted component index {idx} from slice_info")
                elif idx == -1:
                    self.component_index = 2
                    log.debug("Extracted component index 2 (from -1) from slice_info")
            elif isinstance(last, slice):
                step = 1 if last.step is None else int(last.step)
                if step == 1 and isinstance(last.start, (int, np.integer)):
                    start = int(last.start)
                    if start in (0, 1, 2) and isinstance(last.stop, (int, np.integer)):
                        if int(last.stop) == start + 1:
                            self.component_index = start
                            log.debug(
                                f"Extracted component index {start} from singleton slice_info"
                            )
                    elif start == -1 and last.stop is None:
                        self.component_index = 2
                        log.debug(
                            "Extracted component index 2 from slice_info[-1:None]"
                        )

        # Check first element for time slice
        if len(self.slice_info) > 0:
            first = self.slice_info[0]
            if isinstance(first, slice):
                self.time_slice = first
                log.debug(f"Extracted time slice {first} from slice_info")


class ModeDataLoader:
    """Loads mode data respecting slice_info from user selection.

    This class handles loading spectrum and mode data from zarr files,
    applying any user-specified slicing (time range, component selection).

    Parameters
    ----------
    context : ModeDataContext
        Configuration context for data loading

    Examples
    --------
    >>> context = ModeDataContext(
    ...     zarr_path="/path/to/data.zarr",
    ...     dataset_name="m",
    ...     slice_info=(slice(0, 200), ..., 1)  # First 200 timesteps, my
    ... )
    >>> loader = ModeDataLoader(context)
    >>> freqs, spectrum = loader.load_spectrum()
    """

    def __init__(self, context: ModeDataContext):
        self.context = context
        self._zarr_file = None
        self._frequencies = None
        self._spectrum = None
        self._modes_path = None
        self._freqs_path = None
        self._spectrum_path = None

    @property
    def zarr_file(self):
        """Lazy load zarr file."""
        if self._zarr_file is None:
            try:
                import zarr

                self._zarr_file = zarr.open(self.context.zarr_path, mode="r")
                log.debug(f"Opened zarr file: {self.context.zarr_path}")
            except Exception as e:
                log.error(f"Failed to open zarr file: {e}")
                raise
        return self._zarr_file

    @property
    def component_label(self) -> str | None:
        """Get label for selected component."""
        return self.context.component_label

    @property
    def has_single_component(self) -> bool:
        """Check if single component is selected."""
        return self.context.component_index is not None

    def _resolve_paths(self) -> tuple[str | None, str | None, str | None]:
        """Resolve zarr paths for modes, frequencies, and spectrum."""
        dset = self.context.dataset_name

        # Modes and frequencies paths
        exact_mode_group = self.context.mode_group or f"modes/{dset}"
        is_view = "/views/" in exact_mode_group
        base_paths = [exact_mode_group]
        if not is_view:
            base_paths.append(f"tmodes/{dset}")
        modes_path = None
        freqs_path = None

        for base in base_paths:
            if f"{base}/arr" in self.zarr_file and f"{base}/freqs" in self.zarr_file:
                modes_path = f"{base}/arr"
                freqs_path = f"{base}/freqs"
                break

        # Spectrum path candidates - start with exact matches
        spectrum_candidates = [
            f"{exact_mode_group}/power_sum",
            f"{exact_mode_group}/power_max",
        ]
        if not is_view:
            spectrum_candidates.extend(
                [
                    f"fft/{dset}_z-1_m1/spectrum",
                    f"fft/{dset}_z0_m1/spectrum",
                    f"fft/{dset}/spectrum",
                    f"fft/{dset}/spec",
                    f"fft/{dset}/sum",
                ]
            )

        spectrum_path = None

        # First try exact matches
        for path in spectrum_candidates:
            if path in self.zarr_file:
                spectrum_path = path
                log.debug(f"Found spectrum at: {path}")
                break

        return modes_path, freqs_path, spectrum_path

    def load_spectrum(self) -> tuple[np.ndarray, np.ndarray, str | None]:
        """Load spectrum data with component extraction.

        Returns
        -------
        frequencies : np.ndarray
            Frequency array in GHz
        spectrum : np.ndarray
            Power spectrum (1D if single component, 2D if all components)
        component_label : str or None
            Label for component if single component selected
        """
        modes_path, freqs_path, spectrum_path = self._resolve_paths()
        dset = self.context.dataset_name

        if spectrum_path is None:
            raise RuntimeError(
                f"Spectrum data not found for dataset '{self.context.dataset_name}'. "
                "Run compute_modes() first."
            )

        if spectrum_path.endswith(("/power_sum", "/power_max")):
            power_group_path = spectrum_path.rsplit("/", 1)[0]
            power_group = self.zarr_file[power_group_path]
            if (
                getattr(power_group, "attrs", {}).get("power_definition")
                != "abs_fft_squared"
            ):
                raise RuntimeError(
                    f"Mode summary '{spectrum_path}' has an unknown legacy power "
                    "definition; recompute modes with force=True"
                )

        # Load spectrum first to determine size
        spectrum = np.array(self.zarr_file[spectrum_path])

        # Handle complex spectra
        if np.iscomplexobj(spectrum):
            spectrum = np.abs(spectrum) ** 2

        # Determine spectrum length (first dim is usually frequency)
        spec_len = spectrum.shape[0]

        # Try to find matching frequencies
        frequencies = None

        # Option 1: Use modes frequencies if available and match
        if freqs_path is not None:
            freqs_candidate = np.array(self.zarr_file[freqs_path])
            if len(freqs_candidate) == spec_len:
                frequencies = freqs_candidate
                log.debug(f"Using modes frequencies: {len(frequencies)} points")

        # Option 2: only a full-dataset loader may use legacy global FFT axes.
        is_view = "/views/" in (self.context.mode_group or "")
        if frequencies is None and not is_view:
            fft_freqs_candidates = [
                f"fft/{dset}_z-1_m1/frequencies",
                f"fft/{dset}_z0_m1/frequencies",
                f"fft/{dset}/frequencies",
                f"fft/{dset}/freqs",
            ]
            for path in fft_freqs_candidates:
                if path in self.zarr_file:
                    freqs_candidate = np.array(self.zarr_file[path])
                    if len(freqs_candidate) == spec_len:
                        frequencies = freqs_candidate
                        log.debug(f"Using FFT frequencies from {path}")
                        break

            # If still not found and spectrum_path contains slice hash,
            # try frequencies from same directory
            if (
                frequencies is None
                and spectrum_path
                and spectrum_path.endswith("/spectrum")
                and "_s" in spectrum_path.rsplit("/", 1)[0]
            ):
                freq_path_from_spectrum = spectrum_path.replace(
                    "/spectrum", "/frequencies"
                )
                if freq_path_from_spectrum in self.zarr_file:
                    freqs_candidate = np.array(self.zarr_file[freq_path_from_spectrum])
                    if len(freqs_candidate) == spec_len:
                        frequencies = freqs_candidate
                        log.debug(
                            f"Using frequencies from sliced FFT: {freq_path_from_spectrum}"
                        )

        if frequencies is None:
            raise RuntimeError(
                f"No frequency axis matches spectrum length {spec_len}; "
                "recompute the exact mode/FFT view instead of inventing frequencies"
            )

        # Apply component extraction if specified
        if self.context.component_index is not None:
            if spectrum.ndim > 1:
                # Extract specific component
                comp_idx = self.context.component_index
                if spectrum.shape[-1] == 3:
                    spectrum = spectrum[..., comp_idx]
                    log.debug(f"Extracted component {comp_idx} from spectrum")
                else:
                    # Try to handle other dimensions
                    log.warning(
                        f"Cannot extract component {comp_idx} from spectrum "
                        f"with shape {spectrum.shape}"
                    )

        # Ensure spectrum is 1D or 2D (freqs, [components])
        if spectrum.ndim > 2:
            # Average over spatial dimensions, keeping frequency and components
            if spectrum.shape[-1] == 3:
                # (freqs, ..., 3) -> average spatial -> (freqs, 3)
                spatial_axes = tuple(range(1, spectrum.ndim - 1))
                spectrum = np.mean(spectrum, axis=spatial_axes)
            else:
                # Average all non-frequency dimensions
                spatial_axes = tuple(range(1, spectrum.ndim))
                spectrum = np.mean(spectrum, axis=spatial_axes)
            log.debug(f"Averaged spectrum to shape {spectrum.shape}")

        if len(frequencies) != spectrum.shape[0]:
            raise ValueError(
                f"Frequency/spectrum length mismatch: {len(frequencies)} vs "
                f"{spectrum.shape[0]}"
            )

        return frequencies, spectrum, self.context.component_label

    def load_mode_at_frequency(
        self, frequency: float, z_layer: int | None = None
    ) -> tuple[np.ndarray, float, dict]:
        """Load mode data at specified frequency.

        Parameters
        ----------
        frequency : float
            Target frequency in GHz
        z_layer : int, optional
            Z-layer override (uses context z_layer if not specified)

        Returns
        -------
        mode_data : np.ndarray
            Complex mode array with shape (ny, nx, 3) or (ny, nx) if single component
        actual_frequency : float
            Actual frequency loaded (nearest to requested)
        metadata : dict
            Additional metadata about the loaded mode
        """
        modes_path, freqs_path, _ = self._resolve_paths()

        if modes_path is None or freqs_path is None:
            raise RuntimeError(
                f"Mode data not found for dataset '{self.context.dataset_name}'. "
                "Run compute_modes() first."
            )

        z = z_layer if z_layer is not None else self.context.z_layer

        # Load frequencies and find nearest index
        frequencies = np.array(self.zarr_file[freqs_path])
        freq_idx = np.argmin(np.abs(frequencies - frequency))
        actual_freq = frequencies[freq_idx]

        if abs(actual_freq - frequency) > 0.1:
            log.warning(
                f"Requested frequency {frequency:.3f} GHz not exact, "
                f"using {actual_freq:.3f} GHz"
            )

        # Support canonical 5D and legacy single-layer 4D caches.
        mode_shape = self.zarr_file[modes_path].shape
        if len(mode_shape) == 5:
            n_z = mode_shape[1]
        elif len(mode_shape) == 4:
            n_z = 1
        else:
            raise ValueError(f"Unsupported mode cache shape: {mode_shape}")

        # Handle negative z indexing
        if z < 0:
            z = n_z + z
        if z < 0 or z >= n_z:
            raise ValueError(f"z_layer {z} out of range for {n_z} layer(s)")

        if len(mode_shape) == 5:
            mode_data = self.zarr_file[modes_path][freq_idx, z, :, :, :]
        else:
            mode_data = self.zarr_file[modes_path][freq_idx, :, :, :]

        material_mask = None
        mask_path = f"{modes_path.rsplit('/', 1)[0]}/material_mask"
        if mask_path in self.zarr_file:
            stored_mask = np.asarray(self.zarr_file[mask_path], dtype=bool)
            if stored_mask.ndim == 2:
                material_mask = stored_mask
            elif stored_mask.ndim == 3 and z < stored_mask.shape[0]:
                material_mask = stored_mask[z]
            if material_mask is not None and material_mask.shape != mode_data.shape[:2]:
                raise ValueError(
                    f"Stored material mask {material_mask.shape} does not match "
                    f"mode spatial shape {mode_data.shape[:2]}"
                )

        # Apply component extraction if specified
        if self.context.component_index is not None:
            if mode_data.shape[-1] == 1:
                mode_data = mode_data[:, :, 0]
            elif mode_data.shape[-1] == 3:
                mode_data = mode_data[:, :, self.context.component_index]
            else:
                raise ValueError(
                    f"Cannot select component from mode shape {mode_data.shape}"
                )
            log.debug(f"Extracted component {self.context.component_index}")

        if material_mask is not None:
            outside = ~material_mask
            if mode_data.ndim > 2:
                outside = np.broadcast_to(outside[..., None], mode_data.shape)
            mode_data = np.ma.array(mode_data, mask=outside, copy=False)

        metadata = {
            "frequency_index": freq_idx,
            "actual_frequency": actual_freq,
            "requested_frequency": frequency,
            "z_layer": z,
            "component": self.context.component_index,
            "component_label": self.context.component_label,
            "material_mask": material_mask,
        }

        return mode_data, actual_freq, metadata

    def get_available_frequencies(self) -> np.ndarray | None:
        """Get array of available frequencies."""
        _, freqs_path, _ = self._resolve_paths()
        if freqs_path is not None:
            return np.array(self.zarr_file[freqs_path])
        return None
