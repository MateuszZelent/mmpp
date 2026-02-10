"""Batch spectrum processing for multiple simulation results.

This module provides batch processing capabilities for FFT spectrum analysis,
enabling parallel computation across multiple jobs with caching and
visualization as parametric heatmaps.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ....cli.logging_config import get_mmpp_logger

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
        """Apply angular folding using the shared plotting helper."""
        from .plotting import apply_folding

        return apply_folding(param_values, sort_idx, folding_period)
    
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
        dpi: Optional[int] = None,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """Plot 2D heatmap of power spectrum vs parameter."""
        from .plotting import plot_heatmap

        return plot_heatmap(
            self,
            parameter=parameter,
            ax=ax,
            freq_unit=freq_unit,
            fmin=fmin,
            fmax=fmax,
            log_scale=log_scale,
            normalize=normalize,
            cmap=cmap,
            colorbar=colorbar,
            title=title,
            folding=folding,
            verbose=verbose,
            dpi=dpi,
            figsize=figsize,
            **kwargs,
        )
    
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
        alpha: float = 1.0,
        error_color: Optional[str] = None,
        error_linewidth: float = 1.5,
        label: str = 'Experimental',
        ax: Optional[Any] = None,
        **heatmap_kwargs
    ) -> Tuple[Any, Any]:
        """Plot heatmap with experimental peak positions overlaid."""
        from .plotting import plot_experimental_data

        return plot_experimental_data(
            self,
            peaks=peaks,
            errors=errors,
            shift=shift,
            target_field=target_field,
            field_tolerance=field_tolerance,
            marker=marker,
            color=color,
            s=s,
            alpha=alpha,
            error_color=error_color,
            error_linewidth=error_linewidth,
            label=label,
            ax=ax,
            **heatmap_kwargs,
        )
    
    def _replicate_experimental_points(
        self,
        angles: np.ndarray,
        fres: np.ndarray,
        fres_err: np.ndarray,
        folding: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Replicate experimental points to fill folding period with mirroring."""
        from .plotting import replicate_experimental_points

        return replicate_experimental_points(angles, fres, fres_err, folding)
    
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
        """Overlay experimental data on spectrum plot."""
        from .plotting import overlay_experimental

        return overlay_experimental(
            self,
            exp_frequencies=exp_frequencies,
            exp_data=exp_data,
            parameter_value=parameter_value,
            ax=ax,
            label=label,
            color=color,
            **plot_kwargs,
        )




__all__ = [
    "BatchSpectrumResult",
    "SpectrumEntry",
]
