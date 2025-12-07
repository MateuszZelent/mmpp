"""Batch spectrum processing for multiple simulation results.

This module provides batch processing capabilities for FFT spectrum analysis,
enabling parallel computation across multiple jobs with caching and
visualization as parametric heatmaps.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..cli.logging_config import get_mmpp_logger
from ..cache import CacheKey, serialize_for_json

log = get_mmpp_logger("mmpp.fft.spectrum_batch")

# Optional imports
try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Axes = Any

try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False

@dataclass
class SpectrumEntry:
    """Single spectrum result from batch computation.
    
    Provides direct access to spectrum data for a single job.
    
    Attributes
    ----------
    frequencies : np.ndarray
        Frequency array in Hz
    spectrum : np.ndarray
        Complex FFT spectrum
    power : np.ndarray
        Power spectrum (|FFT|²)
    path : str
        Path to source zarr file
    parameters : Dict[str, Any]
        Extracted job parameters
    index : int
        Index in batch
    """
    
    frequencies: np.ndarray
    spectrum: np.ndarray
    power: np.ndarray
    path: str
    parameters: Dict[str, Any]
    index: int
    
    def plot(self, ax: Optional[Any] = None, freq_unit: str = "GHz", 
             log_scale: bool = True, **kwargs) -> Tuple[Any, Any]:
        """Plot this spectrum.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on
        freq_unit : str
            Frequency unit: "Hz", "kHz", "MHz", "GHz", "THz"
        log_scale : bool
            Use logarithmic Y-scale
        **kwargs
            Additional matplotlib plot kwargs
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required")
        
        freq_scales = {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12}
        freq_scale = freq_scales.get(freq_unit, 1e9)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        else:
            fig = ax.figure
        
        ax.plot(self.frequencies / freq_scale, self.power, **kwargs)
        ax.set_xlabel(f"Frequency ({freq_unit})")
        ax.set_ylabel("Power")
        if log_scale:
            ax.set_yscale('log')
        
        # Add parameter info to title
        param_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items() if v is not None)
        ax.set_title(f"Spectrum [{self.index}]: {param_str}" if param_str else f"Spectrum [{self.index}]")
        
        return fig, ax


class BatchSpectrumResult:
    """Container for batch FFT spectrum computation results.
    
    Stores multiple spectrum results along with their associated
    simulation parameters for batch visualization and analysis.
    
    Attributes
    ----------
    frequencies : np.ndarray
        Shared frequency array (n_freqs,)
    spectra : List[np.ndarray]
        List of complex FFT spectra, one per job
    powers : List[np.ndarray]
        List of power spectra (|FFT|²), one per job
    parameters : Dict[str, List[Any]]
        Extracted job parameters, keyed by parameter name
    job_paths : List[str]
        Paths to source zarr files
    config_dict : Dict[str, Any]
        FFT configuration used
    dataset_name : str
        Dataset name used for computation
    z_layer : int
        Z-layer used for computation
    """
    
    frequencies: np.ndarray
    spectra: List[np.ndarray]
    powers: List[np.ndarray]
    parameters: Dict[str, List[Any]]
    job_paths: List[str]
    config_dict: Dict[str, Any] = field(default_factory=dict)
    dataset_name: str = "m"
    z_layer: int = -1
    
    def __len__(self) -> int:
        """Return number of results in batch."""
        return len(self.spectra)
    
    def __getitem__(self, index: int) -> SpectrumEntry:
        """Get individual spectrum result by index.
        
        Parameters
        ----------
        index : int
            Index of spectrum to retrieve
            
        Returns
        -------
        SpectrumEntry
            Spectrum entry with frequencies, spectrum, power, path, parameters
            
        Examples
        --------
        >>> entry = batch_spectra[0]
        >>> entry.frequencies  # frequency array
        >>> entry.power        # power spectrum
        >>> entry.parameters   # extracted params dict
        >>> entry.plot()       # quick plot
        """
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for batch of {len(self)} spectra")
        
        return SpectrumEntry(
            frequencies=self.frequencies,
            spectrum=self.spectra[index],
            power=self.powers[index],
            path=self.job_paths[index],
            parameters={k: v[index] for k, v in self.parameters.items()},
            index=index,
        )
    
    def __iter__(self):
        """Iterate over spectrum entries."""
        for i in range(len(self)):
            yield self[i]
    
    def __repr__(self) -> str:
        """String representation."""
        params = list(self.parameters.keys())
        return (
            f"BatchSpectrumResult({len(self)} spectra, "
            f"frequencies={len(self.frequencies)}, "
            f"parameters={params})"
        )
    
    def get_parameter_values(self, param_name: str) -> np.ndarray:
        """Get array of parameter values.
        
        Parameters
        ----------
        param_name : str
            Name of parameter to retrieve
            
        Returns
        -------
        np.ndarray
            Array of parameter values
        """
        if param_name not in self.parameters:
            raise KeyError(
                f"Parameter '{param_name}' not found. "
                f"Available: {list(self.parameters.keys())}"
            )
        return np.array(self.parameters[param_name])
    
    def to_stacked_array(self, field: str = "power") -> np.ndarray:
        """Stack all spectra into 2D array.
        
        Parameters
        ----------
        field : str
            Field to stack: "power" (default) or "spectrum"
            
        Returns
        -------
        np.ndarray
            Stacked array with shape (n_jobs, n_freqs)
        """
        if field == "power":
            return np.stack(self.powers, axis=0)
        elif field == "spectrum":
            return np.stack(self.spectra, axis=0)
        else:
            raise ValueError(f"Unknown field: {field}")
    
    def save(self, path: Union[str, Path]) -> None:
        """Save batch result to file.
        
        Parameters
        ----------
        path : str or Path
            Path to save file (pickle or zarr based on extension)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix == ".zarr" and ZARR_AVAILABLE:
            self._save_zarr(path)
        else:
            # Use pickle for .pkl or unknown extensions
            with open(path, "wb") as f:
                pickle.dump(self, f)
        
        log.info(f"Saved batch spectrum result to {path}")
    
    def _save_zarr(self, path: Path) -> None:
        """Save to zarr format."""
        z = zarr.open(str(path), mode="w")
        
        z.create_dataset("frequencies", data=self.frequencies)
        z.create_dataset("spectra", data=np.stack(self.spectra, axis=0))
        z.create_dataset("powers", data=np.stack(self.powers, axis=0))
        
        z.attrs["job_paths"] = self.job_paths
        z.attrs["dataset_name"] = self.dataset_name
        z.attrs["z_layer"] = self.z_layer
        z.attrs["config_dict"] = json.dumps(self.config_dict, default=str)
        
        # Save parameters
        params_group = z.create_group("parameters")
        for name, values in self.parameters.items():
            try:
                params_group.create_dataset(name, data=np.array(values))
            except Exception:
                params_group.attrs[name] = json.dumps(values, default=str)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "BatchSpectrumResult":
        """Load batch result from file.
        
        Parameters
        ----------
        path : str or Path
            Path to load from
            
        Returns
        -------
        BatchSpectrumResult
            Loaded result
        """
        path = Path(path)
        
        if path.suffix == ".zarr" and ZARR_AVAILABLE:
            return cls._load_zarr(path)
        else:
            with open(path, "rb") as f:
                return pickle.load(f)
    
    @classmethod
    def _load_zarr(cls, path: Path) -> "BatchSpectrumResult":
        """Load from zarr format."""
        z = zarr.open(str(path), mode="r")
        
        frequencies = np.array(z["frequencies"])
        spectra_stacked = np.array(z["spectra"])
        powers_stacked = np.array(z["powers"])
        
        # Unstack to lists
        spectra = [spectra_stacked[i] for i in range(spectra_stacked.shape[0])]
        powers = [powers_stacked[i] for i in range(powers_stacked.shape[0])]
        
        # Load parameters
        parameters = {}
        if "parameters" in z:
            params_group = z["parameters"]
            for name in params_group.keys():
                parameters[name] = np.array(params_group[name]).tolist()
            for name, value in params_group.attrs.items():
                parameters[name] = json.loads(value)
        
        return cls(
            frequencies=frequencies,
            spectra=spectra,
            powers=powers,
            parameters=parameters,
            job_paths=z.attrs.get("job_paths", []),
            dataset_name=z.attrs.get("dataset_name", "m"),
            z_layer=z.attrs.get("z_layer", -1),
            config_dict=json.loads(z.attrs.get("config_dict", "{}")),
        )
    
    def plot_heatmap(
        self,
        parameter: str,
        ax: Optional[Any] = None,
        freq_unit: str = "GHz",
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        log_scale: bool = True,
        normalize: str = "per_row",
        cmap: str = "viridis",
        colorbar: bool = True,
        title: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """Plot 2D heatmap of power spectrum vs parameter.
        
        Parameters
        ----------
        parameter : str
            Parameter name for Y-axis
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on
        freq_unit : str
            Frequency unit: "Hz", "kHz", "MHz", "GHz", "THz"
        fmin, fmax : float, optional
            Frequency range limits (in displayed units)
        log_scale : bool
            Use logarithmic color scale
        normalize : str
            Normalization mode: "per_row", "global", or "none"
        cmap : str
            Matplotlib colormap
        colorbar : bool
            Show colorbar
        title : str, optional
            Plot title
            
        Returns
        -------
        Tuple[Figure, Axes]
            Matplotlib figure and axes
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for plotting")
        
        # Get frequency scaling
        freq_scales = {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12}
        freq_scale = freq_scales.get(freq_unit, 1e9)
        frequencies_scaled = self.frequencies / freq_scale
        
        # Apply frequency range filter
        freq_mask = np.ones(len(frequencies_scaled), dtype=bool)
        if fmin is not None:
            freq_mask &= frequencies_scaled >= fmin
        if fmax is not None:
            freq_mask &= frequencies_scaled <= fmax
        
        frequencies_display = frequencies_scaled[freq_mask]
        
        # Get parameter values and sort
        param_values = self.get_parameter_values(parameter)
        sort_idx = np.argsort(param_values)
        
        # Build 2D data matrix
        data_matrix = []
        for idx in sort_idx:
            power = self.powers[idx][freq_mask]
            data_matrix.append(power)
        
        data_matrix = np.array(data_matrix)
        param_sorted = param_values[sort_idx]
        
        # Normalize
        if normalize == "per_row":
            row_max = np.max(data_matrix, axis=1, keepdims=True)
            row_max[row_max == 0] = 1
            data_matrix = data_matrix / row_max
        elif normalize == "global":
            global_max = np.max(data_matrix)
            if global_max > 0:
                data_matrix = data_matrix / global_max
        
        # Apply log scale
        if log_scale:
            data_matrix = np.log10(data_matrix + 1e-10)
        
        # Create figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
        
        # Plot heatmap
        extent = [
            frequencies_display[0], frequencies_display[-1],
            param_sorted[0], param_sorted[-1]
        ]
        
        im = ax.imshow(
            data_matrix,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=cmap,
            **kwargs,
        )
        
        ax.set_xlabel(f"Frequency ({freq_unit})")
        ax.set_ylabel(parameter)
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Power Spectrum vs {parameter}")
        
        if colorbar:
            label = "log₁₀(Power)" if log_scale else "Power"
            if normalize != "none":
                label += f" ({normalize})"
            fig.colorbar(im, ax=ax, label=label)
        
        return fig, ax


class BatchSpectrum:
    """Batch spectrum computation handler.
    
    Enables parallel computation of FFT spectrum analysis across
    multiple simulation results with unified configuration.
    """
    
    def __init__(
        self,
        results: List[Any],
        mmpp_ref: Any,
        dataset_name: Optional[str] = None,
        slice_info: Optional[Any] = None,
    ):
        """Initialize batch spectrum processor.
        
        Parameters
        ----------
        results : List[Any]
            List of ZarrJobResult objects
        mmpp_ref : Any
            Reference to MMPP instance
        dataset_name : str, optional
            Dataset name for dataset-specific processing
        slice_info : Any, optional
            Slice information for dataset subsetting
        """
        self.results = results
        self.mmpp_ref = mmpp_ref
        self.dataset_name = dataset_name
        self.slice_info = slice_info
    
    def __call__(
        self,
        force: bool = False,
        filter_type: Optional[List[str]] = None,
        find_peaks: Optional[dict] = None,
        **kwargs,
    ) -> "MultiSpectrumResult":
        """Compute spectra for all jobs and return MultiSpectrumResult for overlay plotting.
        
        This enables the fluent API:
            job[:].m[...,2].fft.spectrum().plot_spectrum()
        
        Parameters
        ----------
        force : bool, default False
            Force recomputation
        filter_type : list[str], optional
            List of filters to apply
        find_peaks : dict, optional
            Peak detection parameters
        **kwargs
            Additional FFT parameters
            
        Returns
        -------
        MultiSpectrumResult
            Collection of spectra with .plot() method for overlay visualization
        """
        from .core import SpectrumResult, MultiSpectrumResult
        
        spectra = []
        
        for result in self.results:
            try:
                # Get dataset-specific FFT
                if self.dataset_name:
                    data_wrapper = result[self.dataset_name]
                    if self.slice_info is not None:
                        data_wrapper = data_wrapper[self.slice_info]
                    fft_obj = data_wrapper.fft
                else:
                    from .core import FFT
                    fft_obj = FFT(result, self.mmpp_ref)
                
                # Compute spectrum
                spectrum_result = fft_obj._spectrum_impl(
                    force=force,
                    filter_type=filter_type,
                    find_peaks=find_peaks,
                    **kwargs,
                )
                
                # Set source job for auto-labeling
                spectrum_result._source_job = result
                
                spectra.append(spectrum_result)
                
            except Exception as e:
                log.warning(f"Failed to compute spectrum for {result}: {e}")
        
        return MultiSpectrumResult(spectra)
    
    def compute_all(
        self,
        dataset_name: Optional[str] = None,
        z_layer: int = -1,
        method: int = 1,
        slice_info: Optional[Any] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        use_cache: bool = True,
        save: bool = True,
        force: bool = False,
        extract_parameters: Optional[List[str]] = None,
        save_batch: bool = True,
        batch_cache_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> BatchSpectrumResult:
        """Compute spectrum for all results in batch.
        
        Parameters
        ----------
        dataset_name : str, optional
            Dataset name to use
        z_layer : int
            Z-layer index (default: -1)
        method : int
            FFT method (default: 1)
        slice_info : Any, optional
            Slice information for data subsetting
        parallel : bool
            Use parallel processing (default: True)
        max_workers : int, optional
            Max worker threads (None for auto)
        use_cache : bool
            Use individual result caching (default: True)
        save : bool
            Save individual results to cache (default: True)
        force : bool
            Force recomputation (default: False)
        extract_parameters : List[str], optional
            Parameters to extract from job attributes
        save_batch : bool
            Save entire batch result (default: True)
        batch_cache_dir : str or Path, optional
            Directory for batch cache files
        **kwargs
            Additional FFT configuration options
            
        Returns
        -------
        BatchSpectrumResult
            Container with all computed results and parameters,
            accessible via batch[0], batch[1], etc.
        """
        from ..fft import FFT
        
        # Use provided values or fall back to instance values
        active_dataset = dataset_name or self.dataset_name
        active_slice = slice_info if slice_info is not None else self.slice_info
        
        # Default parameters to extract
        if extract_parameters is None:
            extract_parameters = ["B0", "d", "p", "thickness", "period", "bias_field", "bex"]
        
        # Generate batch cache key
        batch_key = CacheKey.for_batch(
            analysis_type="batch_spectrum",
            job_paths=[r.path for r in self.results],
            dataset_name=active_dataset or "m",
            config=kwargs,
            slice_info=active_slice,
            extract_parameters=extract_parameters,
        )
        
        # Determine batch cache directory
        if batch_cache_dir is None:
            if self.results:
                first_path = Path(self.results[0].path)
                batch_cache_dir = first_path.parent / ".mmpp_batch_cache"
            else:
                batch_cache_dir = Path(".mmpp_batch_cache")
        else:
            batch_cache_dir = Path(batch_cache_dir)
        
        batch_cache_file = batch_cache_dir / f"{batch_key.to_entry_name()}.zarr"
        
        # Try to load from cache
        if not force and save_batch and batch_cache_file.exists():
            try:
                log.info(f"Found cached batch result: {batch_cache_file}")
                cached = BatchSpectrumResult.load(batch_cache_file)
                
                if len(cached) == len(self.results):
                    log.info(f"✅ Loaded {len(cached)} spectra from cache")
                    return cached
                else:
                    log.warning(
                        f"Cache mismatch: {len(cached)} cached vs "
                        f"{len(self.results)} expected. Recomputing..."
                    )
            except Exception as e:
                log.warning(f"Failed to load cache: {e}. Recomputing...")
        
        log.info(f"Starting batch spectrum computation for {len(self.results)} results")
        
        # Storage for results
        computed_spectra = []
        computed_powers = []
        computed_frequencies = None
        parameters: Dict[str, List[Any]] = {p: [] for p in extract_parameters}
        job_paths = []
        errors = []
        computation_times = []
        
        def compute_single(result_info):
            """Compute spectrum for single result."""
            i, result = result_info
            start_time = time.time()
            
            try:
                log.debug(f"Computing spectrum {i + 1}/{len(self.results)}: {result.path}")
                
                fft_analyzer = FFT(result, self.mmpp_ref)
                
                # Compute spectrum
                freqs, spectrum = fft_analyzer.spectrum(
                    dset=active_dataset,
                    z_layer=z_layer,
                    method=method,
                    slice_info=active_slice,
                    save=save,
                    force=force,
                    **kwargs,
                )
                
                # Extract parameters
                extracted = {}
                for param in extract_parameters:
                    if hasattr(result, "attributes") and isinstance(result.attributes, dict):
                        extracted[param] = result.attributes.get(param)
                    else:
                        extracted[param] = None
                
                return {
                    "success": True,
                    "frequencies": freqs,
                    "spectrum": spectrum,
                    "power": np.abs(spectrum) ** 2,
                    "path": str(result.path),
                    "parameters": extracted,
                    "time": time.time() - start_time,
                }
                
            except Exception as e:
                log.error(f"Failed for {result.path}: {e}")
                gc.collect()
                return {
                    "success": False,
                    "path": str(result.path),
                    "error": str(e),
                    "time": time.time() - start_time,
                }
            finally:
                gc.collect()
        
        # Execute computations
        if parallel and len(self.results) > 1:
            if max_workers is None:
                max_workers = min(len(self.results), os.cpu_count() or 8)
            
            log.info(f"Using parallel execution with {max_workers} workers")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(compute_single, (i, r)): r
                    for i, r in enumerate(self.results)
                }
                
                try:
                    from tqdm import tqdm
                    iterator = tqdm(
                        as_completed(futures),
                        total=len(self.results),
                        desc="Computing spectra",
                        unit="result",
                    )
                except ImportError:
                    iterator = as_completed(futures)
                
                for future in iterator:
                    result_data = future.result()
                    computation_times.append(result_data["time"])
                    
                    if result_data["success"]:
                        if computed_frequencies is None:
                            computed_frequencies = result_data["frequencies"]
                        
                        computed_spectra.append(result_data["spectrum"])
                        computed_powers.append(result_data["power"])
                        job_paths.append(result_data["path"])
                        
                        for param, value in result_data["parameters"].items():
                            parameters[param].append(value)
                    else:
                        errors.append({
                            "path": result_data["path"],
                            "error": result_data.get("error", "Unknown"),
                        })
        else:
            log.info("Using sequential execution")
            
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    enumerate(self.results),
                    total=len(self.results),
                    desc="Computing spectra",
                    unit="result",
                )
            except ImportError:
                iterator = enumerate(self.results)
            
            for i, result in iterator:
                result_data = compute_single((i, result))
                computation_times.append(result_data["time"])
                
                if result_data["success"]:
                    if computed_frequencies is None:
                        computed_frequencies = result_data["frequencies"]
                    
                    computed_spectra.append(result_data["spectrum"])
                    computed_powers.append(result_data["power"])
                    job_paths.append(result_data["path"])
                    
                    for param, value in result_data["parameters"].items():
                        parameters[param].append(value)
                else:
                    errors.append({
                        "path": result_data["path"],
                        "error": result_data.get("error", "Unknown"),
                    })
        
        # Log summary
        successful = len(computed_spectra)
        failed = len(errors)
        avg_time = np.mean(computation_times) if computation_times else 0
        total_time = sum(computation_times)
        
        log.info(f"Batch spectrum: {successful} successful, {failed} failed")
        log.info(f"Total: {total_time:.2f}s, Average: {avg_time:.2f}s per result")
        
        if errors:
            log.warning(f"Errors in {len(errors)} computations:")
            for err in errors[:3]:
                log.warning(f"  {err['path']}: {err['error']}")
        
        # Clean up parameters - remove those with all None values
        parameters = {
            k: v for k, v in parameters.items()
            if any(val is not None for val in v)
        }
        
        if not computed_spectra:
            raise RuntimeError(
                f"All {len(self.results)} spectrum computations failed."
            )
        
        # Create batch result
        batch_result = BatchSpectrumResult(
            frequencies=computed_frequencies,
            spectra=computed_spectra,
            powers=computed_powers,
            parameters=parameters,
            job_paths=job_paths,
            config_dict=kwargs,
            dataset_name=active_dataset or "m",
            z_layer=z_layer,
        )
        
        # Save batch if requested
        if save_batch:
            try:
                batch_result.save(batch_cache_file)
                log.info(f"✅ Saved batch result to {batch_cache_file}")
            except Exception as e:
                log.warning(f"Failed to save batch: {e}")
        
        return batch_result


__all__ = [
    "BatchSpectrum",
    "BatchSpectrumResult",
]
