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
from dataclasses import dataclass
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
    
    def __init__(
        self,
        frequencies: np.ndarray,
        spectra: List[np.ndarray],
        powers: List[np.ndarray],
        parameters: Dict[str, List[Any]],
        job_paths: List[str],
        config_dict: Optional[Dict[str, Any]] = None,
        dataset_name: str = "m",
        z_layer: int = -1,
    ):
        """Initialize batch spectrum result.
        
        Parameters
        ----------
        frequencies : np.ndarray
            Shared frequency array
        spectra : List[np.ndarray]
            List of complex FFT spectra
        powers : List[np.ndarray]
            List of power spectra
        parameters : Dict[str, List[Any]]
            Extracted job parameters
        job_paths : List[str]
            Paths to source zarr files
        config_dict : Dict[str, Any], optional
            FFT configuration used
        dataset_name : str, default="m"
            Dataset name used
        z_layer : int, default=-1
            Z-layer used
        """
        self.frequencies = frequencies
        self.spectra = spectra
        self.powers = powers
        self.parameters = parameters
        self.job_paths = job_paths
        self.config_dict = config_dict if config_dict is not None else {}
        self.dataset_name = dataset_name
        self.z_layer = z_layer
    
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
    
    def show_parameters(self) -> None:
        """Print summary of all extracted parameters.
        
        Shows which parameters vary across the batch and their ranges.
        Useful for determining which parameter to use for heatmap plotting.
        """
        print(f"📊 Batch Spectrum Parameters Summary")
        print(f"{'='*60}")
        print(f"Total spectra: {len(self)}")
        print(f"Frequencies: {len(self.frequencies)} points")
        print(f"\nExtracted parameters:")
        
        if not self.parameters:
            print("  (no parameters extracted)")
            return
        
        varying = []
        constant = []
        
        for param_name, values in self.parameters.items():
            non_none_values = [v for v in values if v is not None]
            if not non_none_values:
                continue
                
            unique_values = np.unique(non_none_values)
            if len(unique_values) > 1:
                arr = np.array(non_none_values)
                varying.append({
                    'name': param_name,
                    'n_unique': len(unique_values),
                    'min': arr.min(),
                    'max': arr.max(),
                })
            else:
                constant.append({
                    'name': param_name,
                    'value': unique_values[0],
                })
        
        if varying:
            print(f"\n  ✓ Varying parameters (good for heatmap):")
            for p in sorted(varying, key=lambda x: x['n_unique'], reverse=True):
                print(f"    • {p['name']}: {p['n_unique']} unique values "
                      f"[{p['min']:.3g} to {p['max']:.3g}]")
        
        if constant:
            print(f"\n  ○ Constant parameters:")
            for p in constant:
                print(f"    • {p['name']}: {p['value']:.3g}")
        
        print(f"\n💡 Usage:")
        if varying:
            print(f"   result.plot_heatmap()  # Auto-selects '{varying[0]['name']}'")
            print(f"   result.plot_heatmap(parameter='{varying[0]['name']}')  # Explicit")
        else:
            print(f"   (No varying parameters - cannot create heatmap)")
        print(f"{'='*60}\n")
    
    def _apply_folding(
        self, 
        param_values: np.ndarray, 
        sort_idx: np.ndarray, 
        folding_period: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply folding to angular parameter values with mirroring.
        
        This method replicates and mirrors data to fill a complete angular period.
        For example, if data covers 0-90°, it can be folded to 0-360° by:
        - 0→90°: original data
        - 90→180°: mirrored (90←0)
        - 180→270°: original again
        - 270→360°: mirrored again
        
        Parameters
        ----------
        param_values : np.ndarray
            Original parameter values (e.g., phi values)
        sort_idx : np.ndarray
            Sorting indices for param_values
        folding_period : float
            Period for folding (e.g., 360 for degrees, 2π for radians)
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Folded parameter values and corresponding indices
        """
        # Sort parameter values
        sorted_params = param_values[sort_idx]
        
        # Calculate range coverage
        param_min = sorted_params.min()
        param_max = sorted_params.max()
        param_range = param_max - param_min
        
        # If already covers full period, no folding needed
        if param_range >= 0.95 * folding_period:
            return param_values, sort_idx
        
        # Determine number of replications needed
        n_replications = int(np.ceil(folding_period / param_range))
        
        # Create folded parameter values with mirroring
        folded_params = []
        folded_indices = []
        
        for i in range(n_replications):
            if i % 2 == 0:
                # Even replication: forward (original order)
                offset = i * param_range
                for val, idx in zip(sorted_params, sort_idx):
                    new_val = val - param_min + offset
                    if new_val < folding_period:
                        folded_params.append(new_val)
                        folded_indices.append(idx)
            else:
                # Odd replication: backward (mirrored)
                offset = (i + 1) * param_range
                for val, idx in zip(sorted_params[::-1], sort_idx[::-1]):
                    new_val = offset - (val - param_min)
                    if new_val < folding_period:
                        folded_params.append(new_val)
                        folded_indices.append(idx)
        
        # Sort by folded parameter values
        folded_params = np.array(folded_params)
        folded_indices = np.array(folded_indices)
        new_sort_idx = np.argsort(folded_params)
        
        return folded_params[new_sort_idx], folded_indices[new_sort_idx]
    
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
        import zarr
        
        # Detect zarr version
        zarr_major = int(zarr.__version__.split('.')[0])
        
        if zarr_major >= 3:
            # Zarr v3 API
            store = zarr.DirectoryStore(str(path))
            root = zarr.open_group(store=store, mode="w")
            
            root.create_dataset("frequencies", data=self.frequencies, chunks=None)
            root.create_dataset("spectra", data=np.stack(self.spectra, axis=0), chunks=None)
            root.create_dataset("powers", data=np.stack(self.powers, axis=0), chunks=None)
            
            root.attrs["job_paths"] = self.job_paths
            root.attrs["dataset_name"] = self.dataset_name
            root.attrs["z_layer"] = self.z_layer
            root.attrs["config_dict"] = json.dumps(self.config_dict, default=str)
            
            # Save parameters
            params_group = root.create_group("parameters")
            for name, values in self.parameters.items():
                try:
                    params_group.create_dataset(name, data=np.array(values), chunks=None)
                except Exception:
                    params_group.attrs[name] = json.dumps(values, default=str)
        else:
            # Zarr v2 API
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
        import zarr
        
        # Detect zarr version
        zarr_major = int(zarr.__version__.split('.')[0])
        
        if zarr_major >= 3:
            # Zarr v3 API
            store = zarr.DirectoryStore(str(path))
            root = zarr.open_group(store=store, mode="r")
            
            frequencies = np.array(root["frequencies"][:])
            spectra_stacked = np.array(root["spectra"][:])
            powers_stacked = np.array(root["powers"][:])
            
            # Unstack to lists
            spectra = [spectra_stacked[i] for i in range(spectra_stacked.shape[0])]
            powers = [powers_stacked[i] for i in range(powers_stacked.shape[0])]
            
            # Load parameters
            parameters = {}
            if "parameters" in root:
                params_group = root["parameters"]
                for name in params_group.keys():
                    parameters[name] = np.array(params_group[name][:]).tolist()
                for name, value in params_group.attrs.items():
                    parameters[name] = json.loads(value)
            
            return cls(
                frequencies=frequencies,
                spectra=spectra,
                powers=powers,
                parameters=parameters,
                job_paths=root.attrs.get("job_paths", []),
                dataset_name=root.attrs.get("dataset_name", "m"),
                z_layer=root.attrs.get("z_layer", -1),
                config_dict=json.loads(root.attrs.get("config_dict", "{}")),
            )
        else:
            # Zarr v2 API
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
        parameter: Optional[str] = None,
        ax: Optional[Any] = None,
        freq_unit: str = "GHz",
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        log_scale: bool = True,
        normalize: str = "per_row",
        cmap: str = "viridis",
        colorbar: bool = True,
        title: Optional[str] = None,
        folding: Optional[Union[float, str]] = None,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """Plot 2D heatmap of power spectrum vs parameter.
        
        Parameters
        ----------
        parameter : str, optional
            Parameter name for Y-axis. If None, automatically detects
            the first parameter with varying values
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
        folding : float or "auto", optional
            For angular parameters (phi, theta, angle), fold/replicate data
            to cover full range. E.g., folding=360 for degrees, folding=2*np.pi for radians.
            If "auto", automatically detects units and applies appropriate folding.
        verbose : bool
            Print parameter detection info (default: False)
            
        Returns
        -------
        Tuple[Figure, Axes]
            Matplotlib figure and axes
            
        Examples
        --------
        >>> # Auto-fold phi from 0-90° to 0-360°
        >>> result.plot_heatmap(folding=360)
        
        >>> # Auto-detect folding based on units  
        >>> result.plot_heatmap(folding="auto")
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for plotting")
        
        # Auto-detect parameter if not provided
        if parameter is None:
            # Find parameters with varying values
            varying_params = []
            for param_name, values in self.parameters.items():
                unique_values = np.unique([v for v in values if v is not None])
                if len(unique_values) > 1:
                    varying_params.append((param_name, len(unique_values)))
            
            if not varying_params:
                raise ValueError(
                    "No varying parameters found! All extracted parameters have constant values.\n"
                    f"Available parameters: {list(self.parameters.keys())}\n"
                    "Hint: Check if parameters were correctly extracted during compute_all()"
                )
            
            # Use the parameter with most unique values (most likely the swapping parameter)
            varying_params.sort(key=lambda x: x[1], reverse=True)
            parameter = varying_params[0][0]
            
            # Print available parameters only if verbose
            if verbose:
                print(f"🔍 Auto-detected swapping parameter: '{parameter}'")
                print(f"\n📊 Available varying parameters:")
                for param_name, n_unique in varying_params:
                    values = self.get_parameter_values(param_name)
                    print(f"   - {param_name}: {n_unique} unique values "
                          f"(range: {values.min():.3g} to {values.max():.3g})")
                print(f"\nUsing '{parameter}' for heatmap Y-axis.")
                print(f"To use a different parameter, call: result.plot_heatmap(parameter='...')\n")
        
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
        
        # Handle angular folding
        param_unit = ""
        angular_params = ["phi", "theta", "angle", "psi", "alpha", "beta", "gamma"]
        is_angular = parameter.lower() in angular_params
        
        if folding is not None and is_angular:
            # Auto-detect units if folding="auto"
            if isinstance(folding, str) and folding.lower() == "auto":
                max_val = param_values.max()
                if max_val <= 7:  # Likely radians (2π ≈ 6.28)
                    folding_period = 2 * np.pi
                    param_unit = " (rad)"
                else:  # Likely degrees
                    folding_period = 360.0
                    param_unit = " (°)"
            else:
                folding_period = float(folding)
                # Detect units from folding value
                if folding_period <= 7:
                    param_unit = " (rad)"
                else:
                    param_unit = " (°)"
            
            # Apply folding
            param_values, sort_idx = self._apply_folding(
                param_values, sort_idx, folding_period
            )
        elif is_angular:
            # Just add unit label even without folding
            max_val = param_values.max()
            if max_val <= 7:
                param_unit = " (rad)"
            else:
                param_unit = " (°)"
        
        # Build 2D data matrix
        data_matrix = []
        for idx in sort_idx:
            power = self.powers[idx][freq_mask]
            # Ensure power is 1D
            if power.ndim > 1:
                power = power.squeeze()
            data_matrix.append(power)
        
        data_matrix = np.array(data_matrix)
        # After folding, param_values are already sorted folded values
        # sort_idx contains indices to self.powers (for building data_matrix)
        # So we just use param_values directly (they're already sorted if folding was applied)
        param_sorted = param_values if folding is not None and is_angular else param_values[sort_idx]
        
        # Ensure data_matrix is 2D (n_params, n_freqs)
        if data_matrix.ndim != 2:
            raise ValueError(
                f"Expected 2D data matrix, got shape {data_matrix.shape}. "
                f"Power spectra should be 1D arrays."
            )
        
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
        
        # Plot heatmap with parameter on X-axis, frequency on Y-axis
        # data_matrix shape is (n_params, n_freqs)
        # We want: X = parameter, Y = frequency
        extent = [
            param_sorted[0], param_sorted[-1],  # X-axis: parameter
            frequencies_display[0], frequencies_display[-1]  # Y-axis: frequency
        ]
        
        im = ax.imshow(
            data_matrix.T,  # Transpose so params are on X, freqs on Y
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=cmap,
            **kwargs,
        )
        
        ax.set_xlabel(f"{parameter}{param_unit}")
        ax.set_ylabel(f"Frequency ({freq_unit})")
        
        if title:
            ax.set_title(title)
        # Else: no default title (removed auto-generated title)
        
        if colorbar:
            label = "log₁₀(Power)" if log_scale else "Power"
            if normalize != "none":
                label += f" ({normalize})"
            cbar = fig.colorbar(im, ax=ax, label=label)
            # Remove outline/frame from colorbar
            cbar.outline.set_visible(False)
        
        return fig, ax
    
    def plot_experimental_data(
        self,
        peaks: str,
        errors: str,
        shift: float = 0.0,
        target_field: Optional[float] = None,
        field_tolerance: float = 0.01,
        marker: str = 'o',
        color: str = 'cyan',
        s: float = 36,
        error_color: Optional[str] = None,
        error_linewidth: float = 1.5,
        label: str = 'Experimental',
        ax: Optional[Any] = None,
        **heatmap_kwargs
    ) -> Tuple[Any, Any]:
        """Plot heatmap with experimental peak positions overlaid.
        
        Loads experimental FMR peak data from CSV files and overlays as scatter
        points with error bars on simulation heatmap. Automatically handles
        folding replication to match simulation data.
        
        Parameters
        ----------
        peaks : str
            Path to CSV file with peak positions. Must contain columns:
            'Angle (°)', 'Field (T)', 'fres (GHz)', 'FWHM (GHz)', 'phi (rad)'
        errors : str
            Path to CSV file with errors. Same structure as peaks file,
            'fres (GHz)' column contains frequency errors
        shift : float
            Angular shift to apply to experimental angles (in radians).
            Use to align experimental and simulation coordinate systems.
            Default 0.0 (no shift). Common values: np.pi, np.pi/2
        target_field : float, optional
            If provided, only show experimental points for this field value (Tesla)
        field_tolerance : float
            Tolerance for field matching (Tesla), default 0.01
        marker : str
            Matplotlib marker style, default 'o'
        color : str
            Marker face color, default 'cyan'
        s : float
            Marker size (area in points^2), default 36
        error_color : str, optional
            Error bar color. If None, uses same as marker color
        error_linewidth : float
            Error bar line width, default 1.5
        label : str
            Legend label for experimental points, default 'Experimental'
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on. If None, creates new figure
        **heatmap_kwargs
            Additional arguments passed to plot_heatmap()
            (e.g., folding, cmap, normalize, fmin, fmax)
            
        Returns
        -------
        Tuple[Figure, Axes]
            Matplotlib figure and axes
            
        Examples
        --------
        >>> result.plot_experimental_data(
        ...     peaks="peaks.csv",
        ...     errors="errors.csv",
        ...     shift=np.pi,
        ...     target_field=0.2,
        ...     color='cyan',
        ...     s=64,
        ...     folding=2*np.pi,
        ...     cmap='hot'
        ... )
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for plotting")
        
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Pandas required for loading experimental data")
        
        # Load experimental data
        peaks_df = pd.read_csv(peaks)
        errors_df = pd.read_csv(errors)
        
        exp_data = {
            'angles_deg': peaks_df['Angle (°)'].values,
            'angles_rad': peaks_df['phi (rad)'].values,
            'fields': peaks_df['Field (T)'].values,
            'fres': peaks_df['fres (GHz)'].values,
            'fres_err': errors_df['fres (GHz)'].values,
            'fwhm': peaks_df['FWHM (GHz)'].values
        }
        
        # Create heatmap only if ax not provided
        if ax is None:
            fig, ax = self.plot_heatmap(**heatmap_kwargs)
        else:
            # Use existing axes - only overlay experimental data
            fig = ax.figure
            # Extract folding from kwargs if present (needed for replication)
            # Don't create new heatmap
        
        # Filter by field if requested
        if target_field is not None:
            mask = np.abs(exp_data['fields'] - target_field) < field_tolerance
            angles_deg = exp_data['angles_deg'][mask]
            fres = exp_data['fres'][mask]
            fres_err = exp_data['fres_err'][mask]
            field_info = f" @ {target_field} T"
        else:
            angles_deg = exp_data['angles_deg']
            fres = exp_data['fres']
            fres_err = exp_data['fres_err']
            field_info = ""
        
        # Convert angles: shift negative to positive (e.g., -180..180 → 0..360), then to radians
        # Step 1: Shift negative angles (e.g., -180° becomes 0°, -90° becomes 90°, etc.)
        angles_deg_shifted = np.where(angles_deg < 0, angles_deg + 360, angles_deg)
        
        # Step 2: Convert to radians
        angles_rad = np.deg2rad(angles_deg_shifted)
        
        # Step 3: Apply user's shift
        angles = angles_rad + shift
        
        # Handle folding - replicate points to match simulation
        if 'folding' in heatmap_kwargs:
            folding = heatmap_kwargs['folding']
            if folding == 'auto':
                # Auto-detect units
                varying_param = self.varying_parameters[0] if self.varying_parameters else None
                if varying_param and varying_param['name'] in ['phi', 'theta', 'angle']:
                    values = varying_param['values']
                    if np.max(values) > 10:
                        folding = 360
                    else:
                        folding = 2 * np.pi
            
            if isinstance(folding, (int, float)):
                # Normalize angles to [0, folding]
                angles = angles % folding
                # Replicate with mirroring
                angles, fres, fres_err = self._replicate_experimental_points(
                    angles, fres, fres_err, folding
                )
        
        # Set error color
        if error_color is None:
            error_color = color
        
        # Calculate marker size (convert area to radius for errorbar)
        markersize = np.sqrt(s / np.pi)
        
        # Overlay scatter points with error bars
        ax.errorbar(
            angles, fres, yerr=fres_err,
            fmt=marker, color=color, markersize=markersize,
            markeredgecolor='black', markeredgewidth=0.5,
            ecolor=error_color, elinewidth=error_linewidth, capsize=3,
            label=label + field_info, zorder=10
        )
        
        # Update legend
        ax.legend(loc='best', framealpha=0.9)
        
        return fig, ax
    
    def _replicate_experimental_points(
        self,
        angles: np.ndarray,
        fres: np.ndarray,
        fres_err: np.ndarray,
        folding: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Replicate experimental points to fill folding period with mirroring.
        
        Parameters
        ----------
        angles : np.ndarray
            Angle values
        fres : np.ndarray
            Resonance frequencies
        fres_err : np.ndarray
            Frequency errors
        folding : float
            Folding period
            
        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Replicated angles, frequencies, and errors
        """
        angle_min = angles.min()
        angle_max = angles.max()
        original_span = angle_max - angle_min
        
        # Determine number of replications needed
        n_copies = int(np.ceil(folding / original_span))
        
        # Collect replicated data
        angles_list = []
        fres_list = []
        fres_err_list = []
        
        for i in range(n_copies):
            if i % 2 == 0:
                # Forward copy
                new_angles = angles + i * original_span
            else:
                # Mirrored copy (backward)
                new_angles = 2 * (i * original_span + angle_min) - angles + original_span
            
            # Only keep points within [0, folding]
            mask = (new_angles >= 0) & (new_angles < folding)
            if np.any(mask):
                angles_list.append(new_angles[mask])
                fres_list.append(fres[mask])
                fres_err_list.append(fres_err[mask])
        
        # Concatenate all copies
        if len(angles_list) > 0:
            angles_rep = np.concatenate(angles_list)
            fres_rep = np.concatenate(fres_list)
            fres_err_rep = np.concatenate(fres_err_list)
        else:
            angles_rep = angles
            fres_rep = fres
            fres_err_rep = fres_err
        
        return angles_rep, fres_rep, fres_err_rep
    
    def overlay_experimental(
        self,
        exp_frequencies: np.ndarray,
        exp_data: np.ndarray,
        parameter_value: Optional[float] = None,
        ax: Optional[Any] = None,
        label: str = "Experimental",
        color: str = "red",
        **plot_kwargs
    ) -> Tuple[Any, Any]:
        """Overlay experimental data on spectrum plot.
        
        Parameters
        ----------
        exp_frequencies : np.ndarray
            Experimental frequency values (in Hz or same unit as simulation)
        exp_data : np.ndarray
            Experimental power/intensity data
        parameter_value : float, optional
            Parameter value (e.g., field, angle) for which to plot simulation data.
            If None, uses first spectrum.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        label : str
            Label for experimental data in legend
        color : str
            Color for experimental data
        **plot_kwargs : dict
            Additional kwargs for experimental data plot (linewidth, marker, etc.)
            
        Returns
        -------
        Tuple[Figure, Axes]
            Matplotlib figure and axes
            
        Examples
        --------
        >>> # Load experimental data
        >>> exp_freq = np.array([...])  # in Hz
        >>> exp_power = np.array([...])
        >>> 
        >>> # Overlay on simulation at specific angle
        >>> result.overlay_experimental(exp_freq, exp_power, 
        ...                            parameter_value=0.785,  # phi = 45°
        ...                            label="Experiment",
        ...                            color="red")
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for plotting")
        
        # Create figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
        
        # Find closest simulation spectrum to parameter_value
        if parameter_value is not None and self.parameters:
            # Get first varying parameter
            param_name = None
            for name, values in self.parameters.items():
                if len(np.unique(values)) > 1:
                    param_name = name
                    break
            
            if param_name:
                param_values = self.get_parameter_values(param_name)
                idx = np.argmin(np.abs(param_values - parameter_value))
            else:
                idx = 0
        else:
            idx = 0
        
        # Plot simulation data
        sim_freq = self.frequencies / 1e9  # Convert to GHz
        sim_power = self.powers[idx]
        if sim_power.ndim > 1:
            sim_power = sim_power.squeeze()
        
        ax.plot(sim_freq, sim_power, label="Simulation", color="blue", linewidth=2)
        
        # Plot experimental data
        exp_freq_ghz = exp_frequencies / 1e9  # Convert to GHz if needed
        default_kwargs = {"linewidth": 1.5, "linestyle": "--", "alpha": 0.8}
        default_kwargs.update(plot_kwargs)
        
        ax.plot(exp_freq_ghz, exp_data, label=label, color=color, **default_kwargs)
        
        # Styling
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Power (a.u.)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if parameter_value is not None and param_name:
            ax.set_title(f"{param_name} = {parameter_value:.3f}")
        
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
    
    def __call__(self, **kwargs) -> BatchSpectrumResult:
        """Compute spectrum for all results in batch.
        
        This enables the fluent API (analogous to batch transmission):
            result = jobs[:].m_layer13[:, ..., 0:1].fft.spectrum(
                filter_type=["remove_mean", "remove_static"],
                window_function="hann",
                component_weights=(1, 0, 0),
                use_cache=True,
                save=True,
                ...
            )
        
        Parameters
        ----------
        **kwargs
            All arguments are forwarded to compute_all()
            
        Returns
        -------
        BatchSpectrumResult
            Batch spectrum result container with heatmap plotting capabilities
            
        Examples
        --------
        >>> # Compute batch spectrum with caching
        >>> result = jobs[:].m_layer13[:, ..., 0:1].fft.spectrum(
        ...     filter_type=["remove_mean", "remove_static"],
        ...     window_function="hann",
        ...     component_weights=(1, 0, 0),
        ...     use_cache=True,
        ...     save=True,
        ...     extract_parameters=["B0", "d"],
        ... )
        >>> # Plot heatmap
        >>> result.plot_heatmap("B0", fmax=50)
        """
        return self.compute_all(**kwargs)
    
    def overlay(
        self,
        force: bool = False,
        filter_type: Optional[List[str]] = None,
        find_peaks: Optional[dict] = None,
        **kwargs,
    ) -> "MultiSpectrumResult":
        """Compute spectra for all jobs and return MultiSpectrumResult for overlay plotting.
        
        This method is useful when you want to plot multiple spectra overlaid
        on a single figure with auto-generated labels.
        
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
            
        Examples
        --------
        >>> # Compute and plot overlaid spectra
        >>> multi = jobs[:10].m[..., 2].fft.spectrum.overlay()
        >>> multi.plot(freq_unit="GHz", log_scale=True)
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
                log.warning(f"Failed to compute spectrum for {getattr(result, 'path', str(result))}: {e}")
        
        return MultiSpectrumResult(spectra)
    
    def compute_all(
        self,
        dataset_name: Optional[str] = None,
        z_layer: int = -1,
        method: int = 1,
        slice_info: Optional[Any] = None,
        # FFT configuration options
        filter_type: Optional[List[str]] = None,
        window_function: str = "none",
        component_weights: tuple = (1, 0, 0),
        normalize: str = "none",
        engine: str = "auto",
        # Peak detection
        find_peaks: Optional[dict] = None,
        # Time/frequency filtering
        tmin: Optional[int] = None,
        tmax: Optional[int] = None,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        # Batch execution control
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
            Dataset name to use (e.g., 'm_layer13')
        z_layer : int, default=-1
            Z-layer index
        method : int, default=1
            FFT method
        slice_info : Any, optional
            Slice information for data subsetting
        filter_type : list[str], optional
            Filters to apply before FFT (e.g., ["remove_mean", "remove_static"])
        window_function : str, default="none"
            Temporal window function: "none", "hann", "hamming", "blackman", etc.
        component_weights : tuple, default=(1, 0, 0)
            Component weights for magnetization (mx, my, mz)
        normalize : str, default="none"
            Normalization mode
        engine : str, default="auto"
            FFT engine: "auto", "numpy", "scipy"
        find_peaks : dict, optional
            Peak detection parameters (e.g., {'height': 0.1, 'prominence': 0.05})
        tmin, tmax : int, optional
            Time range limits in indices (for temporal slicing)
        fmin, fmax : float, optional
            Frequency range limits in Hz (for post-FFT filtering)
        parallel : bool, default=True
            Use parallel processing
        max_workers : int, optional
            Max worker threads (None for auto-detect)
        use_cache : bool, default=True
            Use individual result caching
        save : bool, default=True
            Save individual results to cache
        force : bool, default=False
            Force recomputation (ignore cache)
        extract_parameters : List[str], optional
            Parameter names to extract from job attributes for plotting
        save_batch : bool, default=True
            Save entire batch result for future loading
        batch_cache_dir : str or Path, optional
            Directory for batch cache files
        **kwargs
            Additional FFT configuration options passed to FFT.spectrum()
            
        Returns
        -------
        BatchSpectrumResult
            Container with all computed results and parameters,
            accessible via batch[0], batch[1], etc.
            
        Examples
        --------
        >>> # Compute batch spectrum with full configuration
        >>> result = jobs[:].m_layer13[:, ..., 0:1].fft.spectrum.compute_all(
        ...     filter_type=["remove_mean", "remove_static"],
        ...     window_function="hann",
        ...     component_weights=(1, 0, 0),
        ...     extract_parameters=["B0", "d"],
        ...     fmin=5e9,
        ...     fmax=25e9,
        ...     use_cache=True,
        ...     save=True,
        ... )
        >>> # Plot heatmap
        >>> result.plot_heatmap("B0", fmax=50)
        """
        from ..fft import FFT
        
        # Use provided values or fall back to instance values
        active_dataset = dataset_name or self.dataset_name
        active_slice = slice_info if slice_info is not None else self.slice_info
        
        # Default parameters to extract
        if extract_parameters is None:
            extract_parameters = [
                # Magnetic field
                "B0", "Bext", "bex", "bias_field", "applied_field",
                # Geometry
                "d", "p", "thickness", "period", "latticeconst",
                # Angles
                "phi", "theta", "angle",
            ]
        
        # Build complete config dictionary for cache key
        config_for_cache = {
            "filter_type": filter_type,
            "window_function": window_function,
            "component_weights": component_weights,
            "normalize": normalize,
            "engine": engine,
            "find_peaks": find_peaks,
            "tmin": tmin,
            "tmax": tmax,
            "fmin": fmin,
            "fmax": fmax,
            "z_layer": z_layer,
            "method": method,
            **kwargs,
        }
        
        # Generate batch cache key
        batch_key = CacheKey.for_batch(
            analysis_type="batch_spectrum",
            job_paths=[r.path for r in self.results],
            dataset_name=active_dataset or "m",
            config=config_for_cache,
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
        
        batch_cache_file = batch_cache_dir / f"{batch_key.to_entry_name()}.pkl"
        
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
                
                # Compute spectrum with all parameters
                freqs, spectrum = fft_analyzer.spectrum(
                    dset=active_dataset,
                    z_layer=z_layer,
                    method=method,
                    slice_info=active_slice,
                    save=save,
                    force=force,
                    filter_type=filter_type,
                    window_function=window_function,
                    component_weights=component_weights,
                    normalize=normalize,
                    engine=engine,
                    find_peaks=find_peaks,
                    tmin=tmin,
                    tmax=tmax,
                    fmin=fmin,
                    fmax=fmax,
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
            config_dict=config_for_cache,
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
    "SpectrumEntry",
]
