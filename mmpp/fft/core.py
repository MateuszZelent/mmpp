"""
FFT Core Module

Main FFT class providing unified interface for FFT analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..cli.logging_config import get_mmpp_logger

# Import from our own modules
from ._compute_loading import resolve_dt_from_metadata
from .compute_fft import FFTCompute, FFTComputeResult
from .method_helpers import CallableMethodHelper
from .spectrum.compute import (
    build_cache_key,
    compute_fft_cached,
    format_slice_identifier,
)
from .transmission.interface import FFTTransmissionInterface

if TYPE_CHECKING:
    from .plot import FFTPlotter

# Get logger for FFT core
log = get_mmpp_logger("mmpp.fft")


def _spectral_power_trace(spectrum: Any) -> np.ndarray:
    """Reduce a frequency-first spectrum to mean spectral power per bin."""
    values = np.asarray(spectrum)
    if values.ndim == 0 or values.shape[0] == 0:
        raise ValueError("Spectrum must be a non-empty frequency-first array")
    if not np.isfinite(values).all():
        raise ValueError("Spectrum must contain only finite values")
    power = np.abs(values) ** 2
    if power.ndim > 1:
        power = np.mean(power, axis=tuple(range(1, power.ndim)))
    return np.asarray(power, dtype=float)


# Import mode visualization capabilities
try:
    from .modes import FFTModeInterface, FMRModeAnalyzer
    from .modes.interface import FFTModeInterfaceNew  # New refactored interface

    MODES_AVAILABLE = True
except ImportError:
    MODES_AVAILABLE = False

try:
    from .dispersion import (
        FFTDispersionInterface,
        find_peaks_1d,
    )

    DISPERSION_AVAILABLE = True
except ImportError:
    DISPERSION_AVAILABLE = False
    find_peaks_1d = None  # type: ignore

from .spectrum import (
    SpectrumFilterChain,
    SpectrumHelper,
    SpectrumResult,
)


def generate_pastel_colors(n: int) -> list:
    """Generate n distinct pastel-ish RGBA colors for component overlays."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba

        colors = plt.get_cmap("Accent")(np.linspace(0, 1, max(int(n), 3)))
        return [to_rgba(c) for c in colors[: int(n)]]
    except ImportError:
        return [(0.4 + 0.15 * i, 0.6, 0.8, 1.0) for i in range(max(1, int(n)))]


class FFTHelpAccessor:
    """Callable helper namespace for major FFT API methods."""

    def __init__(self, fft_like: Any, owner: str = "fft"):
        self._fft = fft_like
        self._owner = owner

    def _method(
        self,
        name: str,
        description: str,
        examples: list[str] | None = None,
    ) -> CallableMethodHelper:
        target = getattr(self._fft, name)
        return CallableMethodHelper(
            owner=self._owner,
            name=name,
            target=target,
            description=description,
            examples=examples or [],
        )

    @property
    def spectrum(self) -> CallableMethodHelper:
        return self._method(
            "spectrum",
            "Compute FFT spectrum and return SpectrumResult.",
            ["data.fft.help.spectrum()", "data.fft.help.spectrum(tmin=0, tmax=500)"],
        )

    @property
    def filters(self) -> CallableMethodHelper:
        return self._method(
            "filters",
            "Create fluent filter chain for spectrum workflows.",
            ["data.fft.help.filters(remove_static=True).spectrum()"],
        )

    @property
    def frequencies(self) -> CallableMethodHelper:
        return self._method(
            "frequencies",
            "Return FFT frequency axis (Hz).",
            ["data.fft.help.frequencies()"],
        )

    @property
    def power(self) -> CallableMethodHelper:
        return self._method(
            "power",
            "Return power spectrum |FFT|^2.",
            ["data.fft.help.power()"],
        )

    @property
    def magnitude(self) -> CallableMethodHelper:
        return self._method(
            "magnitude",
            "Return amplitude spectrum |FFT|.",
            ["data.fft.help.magnitude()"],
        )

    @property
    def phase(self) -> CallableMethodHelper:
        return self._method(
            "phase",
            "Return phase spectrum arg(FFT).",
            ["data.fft.help.phase()"],
        )

    @property
    def plot_spectrum(self) -> CallableMethodHelper:
        return self._method(
            "plot_spectrum",
            "Legacy quick-look power plot (use spec.plot.spectrum for modular API).",
            ["data.fft.help.plot_spectrum(log_scale=True)"],
        )

    @property
    def plot_modes(self) -> CallableMethodHelper:
        return self._method(
            "plot_modes",
            "Legacy static mode plot entrypoint (GHz).",
            ["data.fft.help.plot_modes(frequency=9.5)"],
        )

    @property
    def interactive_spectrum(self) -> CallableMethodHelper:
        return self._method(
            "interactive_spectrum",
            "Legacy interactive spectrum entrypoint (prefer data.fft.modes).",
            ["data.fft.help.interactive_spectrum(dpi=140)"],
        )

    def __repr__(self) -> str:
        return (
            "<FFTHelpAccessor: spectrum, filters, frequencies, power, magnitude, "
            "phase, plot_spectrum, plot_modes, interactive_spectrum>"
        )

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import api_help_html

        return api_help_html(
            self,
            title="FFT helper API help",
            prefix=f"{self._owner}.help",
            properties=[
                ("spectrum", "Callable helper for spectrum(...)"),
                ("filters", "Callable helper for filters(...)"),
                ("frequencies", "Callable helper for frequencies(...)"),
                ("power", "Callable helper for power(...)"),
                ("magnitude", "Callable helper for magnitude(...)"),
                ("phase", "Callable helper for phase(...)"),
                ("plot_spectrum", "Callable helper for legacy plot_spectrum(...)"),
                ("plot_modes", "Callable helper for legacy plot_modes(...)"),
                (
                    "interactive_spectrum",
                    "Callable helper for legacy interactive_spectrum(...)",
                ),
            ],
            subtitle=(
                "Each property returns a callable method helper with its own "
                "signature and examples."
            ),
        )


class FFT:
    """
    Main FFT analysis class providing numpy.fft-like interface.

    This class aggregates FFT computation and plotting capabilities
    for MMPP job results.
    """

    # Feature availability flags
    MODES_AVAILABLE = MODES_AVAILABLE
    DISPERSION_AVAILABLE = DISPERSION_AVAILABLE

    def __init__(self, job_result, mmpp_instance: Any | None = None):
        """
        Initialize FFT analyzer for a job result.

        Parameters:
        -----------
        job_result : ZarrJobResult
            Job result to analyze
        mmpp_instance : MMPP, optional
            Reference to parent MMPP instance
        """
        self.job_result = job_result
        self.mmpp = mmpp_instance

        # Initialize compute engine with debug mode from parent MMPP if available
        debug_mode = getattr(mmpp_instance, "debug", False) if mmpp_instance else False
        self._compute = FFTCompute(debug=debug_mode)

        # Initialize plotter (lazy loaded)
        self._plotter: FFTPlotter | None = None

        # Transmission interface (lazy)
        self._transmission_interface: FFTTransmissionInterface | None = None

        # Cache for FFT results
        self._cache: dict[Any, Any] = {}

    @property
    def plotter(self) -> FFTPlotter:
        """Get plotter instance (lazy initialization)."""
        if self._plotter is None:
            from .plot import FFTPlotter

            self._plotter = FFTPlotter([self.job_result], self.mmpp)
        return self._plotter

    @property
    def transmission(self) -> FFTTransmissionInterface:
        """Transmission analysis helper."""

        if self._transmission_interface is None:
            self._transmission_interface = FFTTransmissionInterface(
                self,
                self._compute,
                self.job_result,
            )
        return self._transmission_interface

    @property
    def helpers(self) -> FFTHelpAccessor:
        """Helper namespace for major FFT methods."""
        return FFTHelpAccessor(self, owner="fft")

    @property
    def help(self) -> FFTHelpAccessor:
        """Alias for :attr:`helpers`."""
        return self.helpers

    def _format_slice_identifier(self, slice_info: Any | None) -> str:
        """Create a deterministic identifier for slice_info for caching/saving."""
        return format_slice_identifier(slice_info)

    def _get_cache_key(
        self,
        dataset_name: str,
        z_layer: int,
        method: int,
        slice_identifier: str | None = None,
        **kwargs,
    ) -> str:
        """Generate cache key for FFT results."""
        return build_cache_key(
            dataset_name,
            z_layer,
            method,
            slice_identifier=slice_identifier,
            **kwargs,
        )

    def _compute_fft(
        self,
        dataset_name: str | None = None,
        z_layer: int = -1,
        method: int = 1,
        use_cache: bool = True,
        save: bool = False,
        force: bool = False,
        save_dataset_name: str | None = None,
        slice_info: Any | None = None,
        **kwargs,
    ) -> FFTComputeResult:
        """
        Compute FFT with caching and optional saving.

        Parameters:
        -----------
        dataset_name : str, optional
            Dataset name (default: auto-select largest m dataset)
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        use_cache : bool, optional
            Use memory cache (default: True)
        save : bool, optional
            Save result to zarr file (default: False)
        force : bool, optional
            Force recalculation and overwrite existing (default: False)
        save_dataset_name : str, optional
            Custom name for saved dataset (default: auto-generated)
        slice_info : Any, optional
            Optional slicing (e.g., [:1000, ..., 0]) applied before FFT and used in caching
        **kwargs : Any
            Additional FFT configuration options

        Returns:
        --------
        FFTComputeResult
            FFT computation result
        """
        return compute_fft_cached(
            compute_engine=self._compute,
            job_result=self.job_result,
            cache=self._cache,
            dataset_name=dataset_name,
            z_layer=z_layer,
            method=method,
            use_cache=use_cache,
            save=save,
            force=force,
            save_dataset_name=save_dataset_name,
            slice_info=slice_info,
            **kwargs,
        )

    @property
    def spectrum(self) -> SpectrumHelper:
        """Get spectrum helper (shows help when accessed, callable to compute).

        When accessed directly in notebook, shows usage help.
        When called, computes FFT spectrum.

        Examples
        --------
        >>> job[0].fft.spectrum  # Shows help
        >>> job[0].fft.spectrum()  # Computes spectrum
        """
        return SpectrumHelper(self)

    def filters(self, **filters: Any) -> SpectrumFilterChain:
        """Create fluent filter chain for spectrum calculations.

        Examples
        --------
        >>> job[0].fft.filters(remove_static=True).spectrum()
        >>> job[0].fft.filters(post={"normalize": True, "log_transform": True}).spectrum()
        """
        return SpectrumFilterChain(self.spectrum, filters)

    def _spectrum_impl(
        self,
        dset: str = "m",
        z_layer: int = -1,
        method: int = 1,
        save: bool = False,
        force: bool = False,
        save_dataset_name: str | None = None,
        slice_info: Any | None = None,
        tmin: int | None = None,
        tmax: int | None = None,
        find_peaks: dict | None = None,
        fmin: float | None = None,
        fmax: float | None = None,
        **kwargs,
    ):
        """
        Compute FFT spectrum.

        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m")
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        save : bool, optional
            Save result to zarr file (default: False)
        force : bool, optional
            Force recalculation and overwrite existing (default: False)
        save_dataset_name : str, optional
            Custom name for saved dataset (default: auto-generated)
        slice_info : Any, optional
            Slicing info for data selection (e.g., (slice(0,100), ...))
            If tmin/tmax are provided, they take precedence for time slicing.
        tmin : int, optional
            Start time index for slicing (default: None, use all)
            Creates slice_info=(slice(tmin, tmax), ...) internally.
        tmax : int, optional
            End time index for slicing (default: None, use all)
        find_peaks : dict, optional
            If provided, detect peaks in the spectrum. Dictionary with parameters:
            - 'min_prominence': float, minimum peak prominence (default: 0.0)
            Example: find_peaks={'min_prominence': 0.1}
        fmin : float, optional
            Minimum frequency in Hz to include in result (default: None, include all)
        fmax : float, optional
            Maximum frequency in Hz to include in result (default: None, include all)
        **kwargs : Any
            Additional FFT configuration options

        Returns:
        --------
        tuple[np.ndarray, np.ndarray] or tuple[np.ndarray, np.ndarray, dict]
            If find_peaks is None: (frequencies, complex FFT spectrum)
            If find_peaks is provided: (frequencies, complex FFT spectrum, peaks_info)
            where peaks_info is a dict with 'indices', 'frequencies', and 'amplitudes'

        Examples:
        ---------
        >>> # Basic usage
        >>> freqs, spec = job[0].fft.spectrum()
        >>>
        >>> # With time slicing
        >>> freqs, spec = job[0].fft.spectrum(tmin=0, tmax=1000)
        >>>
        >>> # Equivalent using slice notation
        >>> freqs, spec = job[0].m[:1000,...].fft.spectrum()
        """
        if tmin is not None or tmax is not None:
            slice_info = self._merge_time_slice(slice_info, tmin=tmin, tmax=tmax)

        fft_result = self._compute_fft(
            dset,
            z_layer,
            method,
            save=save,
            force=force,
            save_dataset_name=save_dataset_name,
            slice_info=slice_info,
            **kwargs,
        )

        frequencies = fft_result.frequencies
        spectrum = fft_result.spectrum

        # Apply frequency range filtering
        if fmin is not None or fmax is not None:
            freq_mask = np.ones(len(frequencies), dtype=bool)
            if fmin is not None:
                freq_mask &= frequencies >= fmin
            if fmax is not None:
                freq_mask &= frequencies <= fmax
            frequencies = frequencies[freq_mask]
            spectrum = (
                spectrum[freq_mask] if spectrum.ndim == 1 else spectrum[freq_mask, ...]
            )

        peaks_info = None
        if find_peaks is not None:
            if not isinstance(find_peaks, dict):
                raise TypeError("find_peaks must be a dictionary or None")
            unknown_peak_options = sorted(set(find_peaks) - {"min_prominence"})
            if unknown_peak_options:
                raise TypeError(
                    "Unknown find_peaks option(s): " + ", ".join(unknown_peak_options)
                )
            # Find peaks in spectrum
            if find_peaks_1d is None:
                log.warning(
                    "Peak finding requested but dispersion module not available. Install required dependencies."
                )
            else:
                # Extract parameters
                min_prominence = find_peaks.get("min_prominence", 0.0)

                spectrum_for_peaks = _spectral_power_trace(spectrum)

                # Find peaks
                peak_indices = find_peaks_1d(
                    spectrum_for_peaks, min_prominence=min_prominence
                )

                # Create peaks info dictionary
                # Use amplitudes from the spectrum used for peak finding
                peaks_info = {
                    "indices": peak_indices,
                    "frequencies": frequencies[peak_indices],
                    "amplitudes": np.sqrt(spectrum_for_peaks[peak_indices]),
                    "powers": spectrum_for_peaks[peak_indices],
                }

                log.info(
                    f"Found {len(peak_indices)} peaks with prominence >= {min_prominence}"
                )

        # Determine whether a specific magnetization component was selected.
        component_label = None
        component_index = self._extract_component_index(slice_info)
        component_selected = component_index is not None
        if component_index == 0:
            component_label = r"$m_x$"
        elif component_index == 1:
            component_label = r"$m_y$"
        elif component_index == 2:
            component_label = r"$m_z$"

        # Mark the spectrum result to indicate single-component selection
        result = SpectrumResult(
            frequencies,
            spectrum,
            peaks_info,
            component_label=component_label,
            source_job=self.job_result,
            source_fft=self,
            mode_context={
                "dset": dset,
                "slice_info": slice_info,
                "preloaded_data": kwargs.get("preloaded_data"),
                "time_step_scale": kwargs.get("time_step_scale", 1.0),
            },
            scaling=fft_result.metadata.get("scaling", "raw"),
            spectrum_kind=fft_result.metadata.get("spectrum_kind", "complex"),
            power_quantity=fft_result.metadata.get("power_quantity", "raw_power"),
        )
        result._single_component = component_selected
        return result

    @staticmethod
    def _merge_time_slice(
        slice_info: Any | None,
        *,
        tmin: int | None,
        tmax: int | None,
    ) -> Any:
        """Override only the time axis while preserving other slice selections."""
        time_slice = slice(tmin, tmax)
        if slice_info is None:
            return (time_slice, Ellipsis)
        if slice_info is Ellipsis:
            return (time_slice, Ellipsis)
        if not isinstance(slice_info, tuple):
            return (time_slice, slice_info)
        if len(slice_info) == 0 or slice_info[0] is Ellipsis:
            return (time_slice, *slice_info)
        merged = list(slice_info)
        merged[0] = time_slice
        return tuple(merged)

    def frequencies(
        self,
        dset: str = "m",
        z_layer: int = -1,
        method: int = 1,
        save: bool = False,
        force: bool = False,
        save_dataset_name: str | None = None,
        slice_info: Any | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Get frequency array for FFT.

        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m")
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        save : bool, optional
            Save result to zarr file (default: False)
        force : bool, optional
            Force recalculation and overwrite existing (default: False)
        save_dataset_name : str, optional
            Custom name for saved dataset (default: auto-generated)
        **kwargs : Any
            Additional FFT configuration options

        Returns:
        --------
        np.ndarray
            Frequency array
        """
        # Try to compute frequencies efficiently without loading data
        try:
            log.debug(f"Attempting fast frequency calculation for dataset '{dset}'")
            frequencies = self._compute_frequencies_fast(
                dset, slice_info=slice_info, **kwargs
            )
            log.debug(
                f"✓ Fast frequency calculation successful (shape: {frequencies.shape})"
            )
            return frequencies
        except Exception as e:
            # Fallback to full FFT computation
            log.debug(
                f"Fast frequency calculation failed: {e}, falling back to full FFT"
            )
            result = self._compute_fft(
                dset,
                z_layer,
                method,
                save=save,
                force=force,
                save_dataset_name=save_dataset_name,
                slice_info=slice_info,
                **kwargs,
            )
            return result.frequencies

    def _compute_frequencies_fast(
        self,
        dataset_name: str | None = None,
        slice_info: Any | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute frequency array without loading full dataset.
        Only reads metadata (dt and shape) from zarr.
        """
        # Auto-select largest m dataset if none specified
        if dataset_name is None:
            dataset_name = self.job_result.get_largest_m_dataset()

        if not isinstance(dataset_name, str):
            dataset_name = str(dataset_name)

        # Get dataset metadata without materializing the data
        try:
            data_set = None
            zarr_group = None

            # Prefer existing job_result handle to avoid reopening the store
            try:
                zarr_group = self.job_result.z
                data_set = self.job_result.get_dset(dataset_name)
            except Exception as access_error:
                log.debug(
                    f"Fast frequency metadata lookup via job_result failed for {dataset_name}: {access_error}"
                )

            if data_set is None:
                import zarr

                zarr_group = zarr.open(self.job_result.path, mode="r")
                if dataset_name not in zarr_group:
                    raise ValueError(
                        f"Dataset '{dataset_name}' not found in {self.job_result.path}"
                    )
                data_set = zarr_group[dataset_name]

            data_shape = getattr(data_set, "shape", None)
            if not data_shape:
                raise ValueError(
                    f"Could not determine shape for dataset {dataset_name}"
                )

            n_timesteps = data_shape[0]
            n_timesteps = self._apply_time_slice_length(n_timesteps, slice_info)
            if n_timesteps <= 0:
                raise ValueError(f"Dataset {dataset_name} has no time dimension")

            # Respect optional tmax override if provided
            tmax = kwargs.get("tmax")
            if tmax is not None:
                try:
                    tmax_int = int(tmax)
                    if tmax_int > 0:
                        n_timesteps = min(n_timesteps, tmax_int)
                except (TypeError, ValueError):
                    log.debug(f"Ignoring invalid tmax value: {tmax}")

            dt = None
            if hasattr(data_set, "dt"):
                try:
                    dt = data_set.dt
                    log.debug(f"Using dt from data_set.dt property: {dt}")
                except AttributeError:
                    pass  # Fall through to manual checks
            if dt is None:
                job_meta = type(
                    "_JobMeta", (), {"attrs": getattr(zarr_group, "attrs", {})}
                )()
                dt = resolve_dt_from_metadata(
                    data_set=data_set, job=job_meta, logger=log
                )

            # Determine FFT length (same logic as in compute_fft)
            fft_length = n_timesteps

            zero_padding = kwargs.get("zero_padding", self._compute.config.zero_padding)
            nfft = kwargs.get("nfft", self._compute.config.nfft)

            if nfft is not None:
                if int(nfft) < int(n_timesteps):
                    raise ValueError(
                        f"Requested nfft ({nfft}) must be greater than or equal to data length ({n_timesteps})"
                    )
                fft_length = nfft
            elif zero_padding:
                next_power_two = 1 << (n_timesteps - 1).bit_length()
                if next_power_two > n_timesteps:
                    fft_length = next_power_two

            # Compute frequencies
            frequencies = np.fft.rfftfreq(fft_length, dt)
            return frequencies

        except Exception as e:
            raise RuntimeError(
                f"Failed to compute frequencies from metadata: {e}"
            ) from e

    def _apply_time_slice_length(self, n_timesteps: int, slice_info: Any | None) -> int:
        """Estimate resulting time length after applying slice info."""
        if slice_info is None or n_timesteps <= 0:
            return n_timesteps

        slice_tuple = slice_info if isinstance(slice_info, tuple) else (slice_info,)
        if not slice_tuple:
            return n_timesteps

        first = slice_tuple[0]
        if first is Ellipsis or first is None:
            return n_timesteps

        if isinstance(first, slice):
            start, stop, step = first.indices(n_timesteps)
            if step == 0:
                return n_timesteps
            length = max(0, (stop - start + (step - 1)) // step)
            return max(0, min(length, n_timesteps))

        if isinstance(first, (int, np.integer)):
            return 1

        return n_timesteps

    @staticmethod
    def _extract_component_index(slice_info: Any | None) -> int | None:
        """Extract selected component index from a dataset slice descriptor."""
        if slice_info is None:
            return None

        entries = slice_info if isinstance(slice_info, tuple) else (slice_info,)
        if not entries:
            return None

        last_entry = None
        for entry in reversed(entries):
            if entry is Ellipsis:
                continue
            last_entry = entry
            break

        if last_entry is None:
            return None

        if isinstance(last_entry, (int, np.integer)):
            idx = int(last_entry)
            if idx in (0, 1, 2):
                return idx
            if idx == -1:
                return 2
            return None

        if isinstance(last_entry, slice):
            step = 1 if last_entry.step is None else int(last_entry.step)
            if step != 1:
                return None

            start = last_entry.start
            stop = last_entry.stop
            if isinstance(start, (int, np.integer)):
                start_int = int(start)
                if start_int in (0, 1, 2) and isinstance(stop, (int, np.integer)):
                    if int(stop) == start_int + 1:
                        return start_int
                if start_int == -1 and stop is None:
                    return 2

        return None

    def power(
        self,
        dset: str = "m",
        z_layer: int = -1,
        method: int = 1,
        save: bool = False,
        force: bool = False,
        save_dataset_name: str | None = None,
        slice_info: Any | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute power spectrum.

        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m")
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        save : bool, optional
            Save result to zarr file (default: False)
        force : bool, optional
            Force recalculation and overwrite existing (default: False)
        save_dataset_name : str, optional
            Custom name for saved dataset (default: auto-generated)
        **kwargs : Any
            Additional FFT configuration options

        Returns:
        --------
        np.ndarray
            Power spectrum (|FFT|^2)
        """
        result = self.spectrum(
            dset,
            z_layer,
            method,
            save=save,
            force=force,
            save_dataset_name=save_dataset_name,
            slice_info=slice_info,
            **kwargs,
        )
        return result.power

    def phase(
        self,
        dset: str = "m",
        z_layer: int = -1,
        method: int = 1,
        slice_info: Any | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute phase spectrum.

        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m")
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        **kwargs : Any
            Additional FFT configuration options

        Returns:
        --------
        np.ndarray
            Phase spectrum
        """
        result = self.spectrum(dset, z_layer, method, slice_info=slice_info, **kwargs)
        return result.phase

    def magnitude(
        self,
        dset: str = "m",
        z_layer: int = -1,
        method: int = 1,
        slice_info: Any | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute magnitude spectrum.

        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m")
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        slice_info : Any, optional
            Optional slicing applied before FFT
        \\*\\*kwargs : Any
            Additional FFT configuration options

        Returns:
        --------
        np.ndarray
            Magnitude spectrum (\\|FFT\\|)
        """
        result = self.spectrum(dset, z_layer, method, slice_info=slice_info, **kwargs)
        return result.amplitude

    def plot_spectrum(
        self,
        dset: str = "m",
        ax: Any | None = None,
        method: int = 1,
        z_layer: int = -1,
        log_scale: bool = True,
        normalize: bool = False,
        save: bool = True,
        force: bool = False,
        save_dataset_name: str | None = None,
        slice_info: Any | None = None,
        slice_identifier: str | None = None,
        **kwargs,
    ) -> tuple[Any, Any]:
        """
        Plot power spectrum.

        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m")
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on. If None, creates new figure.
        method : int, optional
            FFT method (default: 1)
        z_layer : int, optional
            Z-layer (default: -1)
        log_scale : bool, optional
            Use logarithmic scale (default: True)
        normalize : bool, optional
            Normalize spectrum (default: False)
        save : bool, optional
            Save FFT result to zarr file (default: True)
        force : bool, optional
            Force recalculation and overwrite existing (default: False)
        save_dataset_name : str, optional
            Custom name for saved dataset (default: auto-generated)
        slice_info : Any, optional
            Optional slicing applied before FFT calculation.
        slice_identifier : str, optional
            Optional deterministic slice identifier used in save/cache naming.
            If omitted and slice_info is provided, it is derived automatically.
        **kwargs : Any
            Additional plotting options

        Returns:
        --------
        tuple
            (figure, axes) matplotlib objects
        """
        if slice_identifier is None and slice_info is not None:
            resolved_identifier = self._format_slice_identifier(slice_info)
            if resolved_identifier != "slice=None":
                slice_identifier = resolved_identifier

        return self.plotter.power_spectrum(
            dataset_name=dset,
            ax=ax,
            method=method,
            z_layer=z_layer,
            log_scale=log_scale,
            normalize=normalize,
            save=save,
            force=force,
            save_dataset_name=save_dataset_name,
            slice_info=slice_info,
            slice_identifier=slice_identifier,
            **kwargs,
        )

    def clear_cache(self):
        """Clear FFT computation cache."""
        self._cache.clear()

    def __repr__(self) -> str:
        """Rich documentation display for FFT interface."""
        try:
            return self._rich_fft_display()
        except Exception:
            return self._basic_fft_display()

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        try:
            return self._html_fft_display()
        except Exception:
            return ""

    def _html_fft_display(self) -> str:
        import uuid as _uuid

        from mmpp._repr_helpers import (
            NODE_COLOR_ANALYSIS,
            NODE_COLOR_COMPUTE,
            NODE_COLOR_PLOT,
            NODE_COLOR_UTIL,
            accessors_section_html,
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        job_result = self.job_result
        job_name = getattr(job_result, "name", "unknown")
        job_path = getattr(job_result, "path", "")
        cache_size = len(self._cache)
        modes_ok = MODES_AVAILABLE
        dispersion_ok = DISPERSION_AVAILABLE
        uid = str(_uuid.uuid4())[:8]

        status = metrics_section_html(
            [
                ("job", job_name, None),
                ("path", job_path, None),
                ("cache entries", cache_size, None),
                (
                    "modes",
                    "available" if modes_ok else "unavailable",
                    "#22c55e" if modes_ok else "#ef4444",
                ),
                (
                    "dispersion",
                    "available" if dispersion_ok else "unavailable",
                    "#22c55e" if dispersion_ok else "#ef4444",
                ),
            ]
        )

        accessors = accessors_section_html(
            [
                (
                    "Compute:",
                    [
                        (".spectrum()", NODE_COLOR_COMPUTE),
                        (".filters(**f).spectrum()", NODE_COLOR_COMPUTE),
                        (".power()", NODE_COLOR_COMPUTE),
                        (".frequencies()", NODE_COLOR_COMPUTE),
                        (".magnitude()", NODE_COLOR_COMPUTE),
                        (".phase()", NODE_COLOR_COMPUTE),
                    ],
                ),
                (
                    "Analysis:",
                    [
                        (".modes", NODE_COLOR_ANALYSIS),
                        (".dispersion", NODE_COLOR_ANALYSIS),
                        (".transmission", NODE_COLOR_ANALYSIS),
                    ],
                ),
                (
                    "Plotting:",
                    [
                        (".plot_spectrum()", NODE_COLOR_PLOT),
                        (".interactive_spectrum()", NODE_COLOR_PLOT),
                        (".plotter", NODE_COLOR_PLOT),
                    ],
                ),
                (
                    "Utilities:",
                    [
                        (".clear_cache()", NODE_COLOR_UTIL),
                        (".helpers", NODE_COLOR_UTIL),
                    ],
                ),
            ]
        )

        examples = examples_section_html(
            "\n".join(
                [
                    "# Preferred: access FFT through a dataset",
                    "data = job[0].m_layer13[:200, ...]",
                    "result = data.fft.spectrum()",
                    "result.plot.spectrum(log_scale=True, freq_unit='GHz')",
                    "",
                    "# Job-level FFT (auto-selects largest m dataset)",
                    "result = job[0].fft.spectrum()",
                    "",
                    "# Fluent filter chain",
                    "job[0].fft.filters(remove_static=True).spectrum()",
                    "",
                    "# Frequency range & peak detection",
                    "result = job[0].fft.spectrum(fmin=1e9, fmax=20e9,",
                    "                            find_peaks={'min_prominence': 0.1})",
                    "",
                    "# Analysis sub-interfaces",
                    "job[0].fft.modes.interactive_spectrum(dpi=150)",
                    "job[0].fft.dispersion.plot_dispersion(axis='x')",
                ]
            )
        )

        api = api_help_html(
            self,
            title="FFT API help",
            prefix="job[0].fft",
            subtitle="Live signatures generated from the FFT interface.",
            properties=[
                (
                    "spectrum",
                    "Spectrum namespace with computation and plotting helpers",
                ),
                ("modes", "FMR mode analysis namespace"),
                ("dispersion", "Dispersion relation analysis namespace"),
                ("transmission", "Transmission / absorption analysis namespace"),
            ],
            methods=[
                "filters",
                "power",
                "frequencies",
                "magnitude",
                "phase",
                "plot_spectrum",
                "plot_modes",
                "interactive_spectrum",
                "clear_cache",
            ],
            chrome=False,
        )

        return node_card_html(
            "FFT Analysis Interface",
            icon="🔬",
            subtitle="Job-level FFT namespace for spectra, filters, modes, dispersion and transmission.",
            badge=("ready", "#22c55e"),
            sections=[status, accessors, examples],
            api=api,
            uid=f"fft-job-{uid}",
        )

    def _rich_fft_display(self) -> str:
        """Create rich documentation display with panels and proper styling."""
        try:
            import io

            from rich.columns import Columns
            from rich.console import Console
            from rich.panel import Panel
            from rich.syntax import Syntax
            from rich.table import Table
            from rich.text import Text

            console = Console(file=io.StringIO(), width=120, force_terminal=True)

            # Get basic info
            path = self.job_result.path
            cache_size = len(self._cache)
            has_modes = MODES_AVAILABLE

            # Summary panel content
            summary_text = Text()
            summary_text.append("🔬 MMPP FFT Analysis Interface\n", style="bold cyan")
            summary_text.append(f"📁 Job Path: {path}\n", style="dim")
            summary_text.append(f"💾 Cache Entries: {cache_size}\n", style="dim")
            summary_text.append(
                f"🎯 Mode Analysis: {'✓ Available' if has_modes else '✗ Unavailable'}\n",
                style="green" if has_modes else "red",
            )

            # Core methods panel content
            core_methods_text = Text()
            core_methods_text.append("🔧 Core FFT Methods:\n", style="bold yellow")
            methods = [
                ("spectrum()", "Get (freqs, complex FFT) tuple"),
                ("frequencies()", "Get frequency array"),
                ("power()", "Get power spectrum |FFT|²"),
                ("magnitude()", "Get magnitude |FFT|"),
                ("phase()", "Get phase spectrum"),
                ("spectrum(tmin=,tmax=)", "FFT with time slicing"),
                ("plot_spectrum(ax=)", "Plot on existing axes"),
                ("transmission", "Transmission analysis interface"),
                ("clear_cache()", "Clear computation cache"),
            ]

            for method, desc in methods:
                core_methods_text.append("  • ", style="dim")
                core_methods_text.append(method, style="code")
                core_methods_text.append(f" - {desc}\n", style="dim")

            # Plotting methods panel content
            plotting_methods_text = Text()
            plotting_methods_text.append("📈 Plotting Toolkit:\n", style="bold magenta")
            plotting_methods = [
                ("plot_spectrum(log_scale=True)", "Quick-look power spectrum"),
                ("plotter.power_spectrum(normalize=True)", "Overlay multiple results"),
                (
                    "plotter.power_spectrum(save_path='fft.png')",
                    "Export publication figure",
                ),
                ("plot_modes(frequency=..., z_layer=-1)", "Static mode grid"),
                ("modes.save_modes_animation()", "Animated mode evolution"),
            ]

            for method, desc in plotting_methods:
                plotting_methods_text.append("  • ", style="dim")
                plotting_methods_text.append(method, style="code")
                plotting_methods_text.append(f" - {desc}\n", style="dim")

            # Mode methods panel content (if available)
            if has_modes:
                mode_methods_text = Text()
                mode_methods_text.append(
                    "🌊 Mode Analysis Methods:\n", style="bold blue"
                )
                mode_methods = [
                    ("modes", "Access mode interface"),
                    ("[index]", "Index-based mode access"),
                    ("plot_modes(frequency)", "Plot modes at frequency"),
                    ("interactive_spectrum()", "Interactive spectrum+modes"),
                ]

                for method, desc in mode_methods:
                    mode_methods_text.append("  • ", style="dim")
                    mode_methods_text.append(method, style="code")
                    mode_methods_text.append(f" - {desc}\n", style="dim")
            else:
                mode_methods_text = Text()
                mode_methods_text.append(
                    "🌊 Mode Analysis: Not Available\n", style="bold red"
                )
                mode_methods_text.append(
                    "Install mode visualization dependencies to enable", style="dim"
                )

            # Batch operations panel content
            batch_methods_text = Text()
            batch_methods_text.append(
                "📦 Batch Operations (job[:].fft):\n", style="bold green"
            )
            batch_methods = [
                ("spectrum.compute_all()", "Compute all spectra in batch"),
                ("transmission.compute_all()", "Compute all transmissions"),
                ("batch[i].plot()", "Plot individual spectrum"),
                ("batch.plot_heatmap(param)", "2D heatmap vs parameter"),
                ("batch.save(path)", "Save batch to zarr/pickle"),
            ]

            for method, desc in batch_methods:
                batch_methods_text.append("  • ", style="dim")
                batch_methods_text.append(method, style="code")
                batch_methods_text.append(f" - {desc}\n", style="dim")

            # Parameters table
            params_table = Table(show_header=False, box=None, padding=(0, 1))
            params_table.add_column("Parameter", style="bold yellow")
            params_table.add_column("Description", style="white")
            params_table.add_column("Values", style="cyan")

            params = [
                (
                    "dset",
                    "Dataset name",
                    "Auto-selected or explicit: 'm', 'm_x11', 'm_y11'",
                ),
                ("z_layer", "Z-layer index", "-1 (top), 0 (bottom), 1, 2, ..."),
                ("method", "FFT method", "1 (default), 2"),
                ("ax", "Existing matplotlib axes", "None (create new) or Axes"),
                ("fmin/fmax", "Frequency range filter", "None or float (Hz)"),
                ("tmin/tmax", "Time index range", "None or int (start:stop)"),
                ("slice_info", "Advanced slicing", "tuple e.g. (slice(0,100), ..., 1)"),
                ("find_peaks", "Peak detection", "None or {'min_prominence': 0.1}"),
                ("save", "Save to zarr", "True/False"),
                ("force", "Force recalculation", "True/False"),
                ("zero_padding", "Pad to power-of-two", "True/False (default: True)"),
                ("nfft", "Manual FFT length", "int or None (auto)"),
                (
                    "filter_type",
                    "Preprocessing filter",
                    "remove_mean, savgol_smooth, high_pass, band_pass",
                ),
                (
                    "window",
                    "Window function",
                    "hann (default), flattop, nuttall, blackman",
                ),
                ("dpi", "Plot resolution", "int (e.g., 100, 300)"),
                ("log_scale", "Logarithmic Y-scale", "True (default) / False"),
                ("normalize", "Normalize power", "True/False (default: False)"),
                ("show_peaks", "Show peak markers", "True (default) / False"),
                (
                    "freq_unit",
                    "Frequency display unit",
                    "Hz, kHz, MHz, GHz (default), THz",
                ),
            ]

            for param, desc, values in params:
                params_table.add_row(param, desc, values)

            example_code = """# Basic FFT operations
freqs, spectrum = job[0].fft.spectrum()
power = job[0].fft.power()

# Time slicing - two equivalent ways:
freqs, spec = job[0].fft.spectrum(tmin=0, tmax=1000)
freqs, spec = job[0].m[:1000,...].fft.spectrum()  # slice notation

# Frequency range filtering
freqs, spec = job[0].fft.spectrum(fmin=5e9, fmax=25e9)

# Advanced filters
job[0].fft.spectrum(filter_type="savgol_smooth")

# Batch spectrum
batch = job[:].fft.spectrum.compute_all(
    extract_parameters=["B0"],
)
batch[0].plot()                    # Plot single spectrum
batch.plot_heatmap("B0")           # 2D heatmap vs B0"""

            syntax = Syntax(
                example_code, "python", theme="monokai", background_color="default"
            )

            # Build panels
            with console.capture() as capture:
                # Main summary panel
                console.print(
                    Panel.fit(
                        summary_text,
                        title="[bold cyan]MMPP FFT Interface[/bold cyan]",
                        border_style="cyan",
                    )
                )
                console.print("")

                # Method panels side by side
                console.print(
                    Columns(
                        [
                            Panel.fit(
                                core_methods_text,
                                title="[bold yellow]Core Methods[/bold yellow]",
                                border_style="yellow",
                            ),
                            Panel.fit(
                                plotting_methods_text,
                                title="[bold magenta]Plotting[/bold magenta]",
                                border_style="magenta",
                            ),
                            Panel.fit(
                                mode_methods_text,
                                title="[bold blue]Mode Methods[/bold blue]",
                                border_style="blue" if has_modes else "red",
                            ),
                            Panel.fit(
                                batch_methods_text,
                                title="[bold green]Batch Operations[/bold green]",
                                border_style="green",
                            ),
                        ]
                    )
                )
                console.print("")

                # Parameters panel
                console.print(
                    Panel.fit(
                        params_table,
                        title="[bold green]Common Parameters[/bold green]",
                        border_style="green",
                    )
                )
                console.print("")

                # Examples panel
                console.print(
                    Panel.fit(
                        syntax,
                        title="[bold magenta]Usage Examples[/bold magenta]",
                        border_style="magenta",
                    )
                )

            return capture.get()

        except Exception:
            # Fallback to basic text display if rich fails
            return self._basic_fft_display_enhanced()

    def _basic_fft_display(self) -> str:
        """Fallback basic display if rich display fails."""
        return f"FFT(path='{self.job_result.path}', cache_entries={len(self._cache)})"

    def _basic_fft_display_enhanced(self) -> str:
        """Enhanced fallback display with more details if rich display fails."""
        path = self.job_result.path
        cache_size = len(self._cache)
        has_modes = MODES_AVAILABLE

        output = []
        output.append("=" * 70)
        output.append("🔬 MMPP FFT Analysis Interface")
        output.append("=" * 70)
        output.append(f"📁 Job Path: {path}")
        output.append(f"💾 Cache Entries: {cache_size}")
        output.append(
            f"🎯 Mode Analysis: {'✓ Available' if has_modes else '✗ Unavailable'}"
        )
        output.append("")

        # Core FFT Methods
        output.append("🔧 CORE FFT METHODS:")
        output.append("─" * 50)
        methods = [
            (
                "spectrum()",
                "Get (freqs, complex FFT spectrum)",
                "freqs, spectrum = job[0].fft.spectrum('m', z_layer=-1)",
            ),
            ("frequencies()", "Get frequency array", "job[0].fft.frequencies()"),
            ("power()", "Get power spectrum |FFT|²", "job[0].fft.power()"),
            ("magnitude()", "Get magnitude |FFT|", "job[0].fft.magnitude()"),
            ("phase()", "Get phase spectrum", "job[0].fft.phase()"),
            (
                "plot_spectrum()",
                "Plot power spectrum",
                "fig, ax = job[0].fft.plot_spectrum()",
            ),
            ("clear_cache()", "Clear computation cache", "job[0].fft.clear_cache()"),
        ]

        for method, desc, example in methods:
            output.append(f"  • {method:<15} {desc}")
            output.append(f"    └─ {example}")

        output.append("")

        # Plotting toolkit
        output.append("📈 PLOTTING TOOLKIT:")
        output.append("─" * 50)
        plot_methods = [
            (
                "plot_spectrum(log_scale=True)",
                "Quick-look spectrum",
                "job[0].fft.plot_spectrum(log_scale=True)",
            ),
            (
                "plotter.power_spectrum(normalize=True)",
                "Overlay multiple jobs",
                "job[0].fft.plotter.power_spectrum(normalize=True)",
            ),
            (
                "plotter.power_spectrum(save_path='fft.png')",
                "Export PNG/ publication",
                "job[0].fft.plotter.power_spectrum(save_path='fft.png')",
            ),
            (
                "plot_modes(frequency, z_layer)",
                "Static mode panels",
                "job[0].fft.plot_modes(frequency=10.4, z_layer=-1)",
            ),
            (
                "modes.save_modes_animation()",
                "Animated mode evolution",
                "job[0].fft.modes.save_modes_animation(frequency=10.4)",
            ),
        ]

        for method, desc, example in plot_methods:
            output.append(f"  • {method:<40} {desc}")
            output.append(f"    └─ {example}")

        output.append("")

        # Mode Analysis (if available)
        if has_modes:
            output.append("🌊 MODE ANALYSIS METHODS:")
            output.append("─" * 50)
            mode_methods = [
                (
                    "modes",
                    "Access mode interface",
                    "job[0].fft.modes.interactive_spectrum()",
                ),
                (
                    "[index]",
                    "Index-based mode access",
                    "job[0].fft[0][200].plot_modes()",
                ),
                (
                    "plot_modes()",
                    "Plot modes at frequency",
                    "job[0].fft.plot_modes(frequency=1.5)",
                ),
                (
                    "interactive_spectrum()",
                    "Interactive spectrum+modes",
                    "job[0].fft.interactive_spectrum()",
                ),
            ]

            for method, desc, example in mode_methods:
                output.append(f"  • {method:<20} {desc}")
                output.append(f"    └─ {example}")
        else:
            output.append("🌊 MODE ANALYSIS: Not Available")
            output.append("   Install mode visualization dependencies to enable")

        output.append("")

        # Common Parameters
        output.append("⚙️  COMMON PARAMETERS:")
        output.append("─" * 50)
        params = [
            ("dset", "Dataset name", "'m', 'm_x11', 'm_y11'"),
            ("z_layer", "Z-layer index", "-1 (top), 0 (bottom), 1, 2, ..."),
            ("method", "FFT method", "1 (default), 2"),
            ("save", "Save to zarr", "True/False"),
            ("force", "Force recalculation", "True/False"),
            ("zero_padding", "Pad to power-of-two length", "True/False"),
            ("nfft", "Manual FFT length", "int or None"),
        ]

        for param, desc, values in params:
            output.append(f"  • {param:<12} {desc:<20} {values}")

        output.append("")

        # Quick Examples
        output.append("🚀 QUICK START EXAMPLES:")
        output.append("─" * 50)
        examples = [
            "# Basic FFT operations",
            "power = job[0].fft.power('m')",
            "freqs = job[0].fft.frequencies()",
            "freqs_fft, spectrum = job[0].fft.spectrum(save=True, force=True)",
            "",
            "# Plotting",
            "fig, ax = job[0].fft.plot_spectrum(log_scale=True)",
            "job[0].fft.plotter.power_spectrum(save_path='fft_publication.png')",
            "",
            "# Mode analysis (if available)",
            "job[0].fft.modes.interactive_spectrum()",
            "job[0].fft[0][200].plot_modes()  # Elegant syntax",
            "job[0].fft.plot_modes(frequency=1.5)",
            "",
            "# Advanced usage",
            "job[0].fft.plotter.power_spectrum(normalize=True)",
            "job[0].fft.modes.save_modes_animation(frequency=10.4, save_path='mode.gif')",
            "help(job[0].fft.spectrum)  # Detailed documentation",
        ]

        for example in examples:
            output.append(f"  {example}")

        output.append("")
        output.append("=" * 70)
        output.append("📖 For detailed docs: help(job[0].fft.spectrum)")
        output.append("🔧 Clear cache: job[0].fft.clear_cache()")
        output.append("=" * 70)

        return "\n".join(output)

    @property
    def modes(self) -> FFTModeInterfaceNew:
        """
        Get mode visualization interface.

        Supports slice propagation for component selection.

        Returns:
        --------
        FFTModeInterfaceNew
            Interface for mode operations with slice support

        Examples:
        ---------
        >>> job[0].fft.modes.interactive_spectrum()
        >>> job[0].m[:200,...,1].fft.modes.interactive_spectrum(dpi=150)  # my only
        >>> job[0].fft.modes.plot_modes(frequency=1.5)
        """
        if not MODES_AVAILABLE:
            raise ImportError(
                "Mode visualization not available. Check modes module import."
            )

        if not hasattr(self, "_mode_interface_new"):
            self._mode_interface_new = FFTModeInterfaceNew(0, self)
        return self._mode_interface_new

    @property
    def dispersion(self) -> FFTDispersionInterface:
        """
        Get spin-wave dispersion analysis interface.

        Returns:
        --------
        FFTDispersionInterface
            Interface for dispersion operations

        Examples:
        ---------
        >>> job[0].fft.dispersion.plot_dispersion()
        >>> job[0].fft.dispersion.compute_1d(axis="x")
        >>> job[0].m_layer.fft.dispersion.plot_branch()
        """
        if not DISPERSION_AVAILABLE:
            raise ImportError(
                "Dispersion analysis not available. Check dispersion module import."
            )

        if not hasattr(self, "_dispersion_interface"):
            self._dispersion_interface = FFTDispersionInterface(self)
        return self._dispersion_interface

    def __getitem__(self, index: int) -> FFTModeInterface:
        """
        Get FFT result by index for mode operations.

        Parameters:
        -----------
        index : int
            FFT result index (usually 0 for latest)

        Returns:
        --------
        FFTModeInterface
            Interface for mode operations at specific FFT result

        Examples:
        ---------
        >>> job[0].fft[0].interactive_spectrum()
        >>> job[0].fft[0][200].plot_modes()
        """
        if not MODES_AVAILABLE:
            raise ImportError(
                "Mode visualization not available. Check modes module import."
            )

        return FFTModeInterface(index, self)

    def plot_modes(
        self, frequency: float, dset: str = "m", z_layer: int = 0, **kwargs
    ) -> tuple[Any, Any]:
        """
        Plot FMR modes at specific frequency.

        Parameters:
        -----------
        frequency : float
            Frequency in GHz
        dset : str
            Dataset name
        z_layer : int
            Z-layer index
        **kwargs
            Additional arguments for mode plotting

        Returns:
        --------
        Tuple[Figure, np.ndarray]
            Matplotlib figure and axes
        """
        if not MODES_AVAILABLE:
            raise ImportError(
                "Mode visualization not available. Check modes module import."
            )

        # Create temporary mode analyzer
        debug_mode = getattr(self.mmpp, "debug", False) if self.mmpp else False
        log_level = getattr(self.mmpp, "log_level", None) if self.mmpp else None
        analyzer = FMRModeAnalyzer(
            self.job_result.path,
            dataset_name=dset,
            debug=debug_mode,
            log_level=log_level,
        )
        return analyzer.plot_modes(frequency=frequency, z_layer=z_layer, **kwargs)

    def interactive_spectrum(self, dset: str = "m", **kwargs) -> Any:
        """
        Create interactive spectrum plot with mode visualization.

        Parameters:
        -----------
        dset : str
            Dataset name
        **kwargs
            Additional arguments for interactive plotting

        Returns:
        --------
        Figure
            Interactive matplotlib figure
        """
        if not MODES_AVAILABLE:
            raise ImportError(
                "Mode visualization not available. Check modes module import."
            )

        # Create temporary mode analyzer
        debug_mode = getattr(self.mmpp, "debug", False) if self.mmpp else False
        log_level = getattr(self.mmpp, "log_level", None) if self.mmpp else None
        analyzer = FMRModeAnalyzer(
            self.job_result.path,
            dataset_name=dset,
            debug=debug_mode,
            log_level=log_level,
        )
        return analyzer.interactive_spectrum(**kwargs)
