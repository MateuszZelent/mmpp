"""Batch transmission processing for multiple simulation results.

This module provides batch processing capabilities for FFT transmission analysis,
enabling parallel computation across multiple jobs and visualization of results
as parametric heatmaps.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np

# Try to use joblib for parallel processing (much better for numpy/scipy CPU-bound tasks)
try:
    from joblib import Parallel, delayed
    _USE_JOBLIB = True
except ImportError:
    _USE_JOBLIB = False

from ...cli.logging_config import get_mmpp_logger
from .compute import TransmissionConfig, TransmissionResult
from .interface import FFTTransmissionInterface

log = get_mmpp_logger("mmpp.fft.transmission.batch")

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    Axes = Any


class BatchTransmissionResult:
    """Container for batch transmission computation results.
    
    Stores multiple TransmissionResult objects along with their associated
    simulation parameters for batch visualization and analysis.
    """
    
    def __init__(
        self,
        results: List[TransmissionResult],
        parameters: Dict[str, List[Any]],
        job_paths: List[str],
    ):
        """Initialize batch transmission result.
        
        Parameters
        ----------
        results : List[TransmissionResult]
            List of individual transmission results
        parameters : Dict[str, List[Any]]
            Dictionary mapping parameter names to lists of values
        job_paths : List[str]
            List of job paths corresponding to results
        """
        self.results = results
        self.parameters = parameters
        self.job_paths = job_paths
        
        # Validate consistency
        n = len(results)
        if len(job_paths) != n:
            raise ValueError(
                f"Inconsistent lengths: {len(results)} results vs {len(job_paths)} paths"
            )
        for param_name, param_values in parameters.items():
            if len(param_values) != n:
                raise ValueError(
                    f"Parameter '{param_name}' has {len(param_values)} values, "
                    f"expected {n}"
                )
    
    def __len__(self) -> int:
        """Return number of results in batch."""
        return len(self.results)
    
    def __getitem__(self, index: int) -> TransmissionResult:
        """Get individual transmission result by index."""
        return self.results[index]
    
    def __iter__(self):
        """Iterate over transmission results."""
        return iter(self.results)
    
    def save(self, path: Union[str, Path]) -> None:
        """Save batch result to file.
        
        Parameters
        ----------
        path : str or Path
            Path to save the batch result (pickle format)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare serializable data
        data = {
            "results": self.results,
            "parameters": self.parameters,
            "job_paths": self.job_paths,
            "version": "1.0",
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        log.info(f"Saved batch result to {path} ({len(self.results)} results)")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "BatchTransmissionResult":
        """Load batch result from file.
        
        Parameters
        ----------
        path : str or Path
            Path to load the batch result from
            
        Returns
        -------
        BatchTransmissionResult
            Loaded batch result
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Batch result file not found: {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle version compatibility
        version = data.get("version", "0.0")
        if version != "1.0":
            log.warning(f"Loading batch result with version {version}, current is 1.0")
        
        result = cls(
            results=data["results"],
            parameters=data["parameters"],
            job_paths=data["job_paths"],
        )
        
        log.info(f"Loaded batch result from {path} ({len(result.results)} results)")
        return result
    
    def get_parameter_values(self, param_name: str) -> np.ndarray:
        """Get array of parameter values.
        
        Parameters
        ----------
        param_name : str
            Name of the parameter to retrieve
            
        Returns
        -------
        np.ndarray
            Array of parameter values
        """
        if param_name not in self.parameters:
            available = ", ".join(self.parameters.keys())
            raise KeyError(
                f"Parameter '{param_name}' not found. "
                f"Available parameters: {available}"
            )
        return np.array(self.parameters[param_name])

    @staticmethod
    def _normalize_mode(normalize: Union[bool, str, None]) -> str:
        """Convert user-facing normalize flag to canonical mode string."""
        if normalize is True:
            return "per_column"
        if normalize is False or normalize is None or normalize == "none":
            return "none"
        if isinstance(normalize, str):
            return normalize.lower()
        return "per_column"

    def _prepare_crosssection_heatmap_data(
        self,
        swapping_parameter: str,
        x: float,
        x_width: Optional[float],
        freq_unit: str,
        trim_0f: int,
        fmin: Optional[float],
        fmax: Optional[float],
        normalize_mode: str,
        flip: bool,
        disable_averaging: bool,
        param_scale: float,
        verbose: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract normalized heatmap data without plotting."""
        param_values = self.get_parameter_values(swapping_parameter)

        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 plot_transmission_crosssection_heatmap()")
            print(f"{'='*60}")
            print(f"  swapping_parameter: {swapping_parameter}")
            print(f"  x: {x} ({x*1e9:.1f} nm)")
            print(f"  x_width: {x_width}")
            print(f"  normalize_mode: {normalize_mode}")
            print(f"  freq_unit: {freq_unit}")
            print(f"  fmin/fmax: {fmin}/{fmax}")
            print(f"  trim_0f: {trim_0f}")
            print(
                f"  param_values: {param_values[:5]}"
                f"{'...' if len(param_values) > 5 else ''}"
            )
            print(f"  n_results: {len(self.results)}")

        cross_sections = []
        frequencies = None

        for i, result in enumerate(self.results):
            try:
                freq, cross_section = self._extract_crosssection(
                    result,
                    x,
                    x_width,
                    trim_0f,
                    fmin,
                    fmax,
                    freq_unit,
                    normalize=False,
                    flip=flip,
                    disable_averaging=disable_averaging,
                    verbose=(verbose and i == 0),
                )

                if frequencies is None:
                    frequencies = freq
                    n_freq_expected = len(freq)
                else:
                    if len(freq) != n_freq_expected:
                        try:
                            from scipy.interpolate import interp1d

                            interp_func = interp1d(
                                freq,
                                cross_section,
                                kind="linear",
                                bounds_error=False,
                                fill_value=0.0,
                            )
                            cross_section = interp_func(frequencies)
                        except Exception:
                            cross_section = np.interp(
                                frequencies, freq, cross_section, left=0.0, right=0.0
                            )
                        if verbose and i == 1:
                            log.info(
                                "Interpolating result %d from %d to %d frequency points",
                                i,
                                len(freq),
                                n_freq_expected,
                            )

                cross_sections.append(cross_section)

            except Exception as e:
                log.error(f"Failed to extract cross-section from result {i}: {e}")
                if frequencies is not None:
                    cross_sections.append(np.full_like(frequencies, np.nan))
                else:
                    raise

        if frequencies is None:
            raise ValueError("No cross-sections were extracted; frequencies is None.")

        heatmap_data = np.array(cross_sections)

        # Sort by parameter values so imshow displays increasing parameter axis
        sort_indices = np.argsort(param_values)
        param_values = param_values[sort_indices]
        heatmap_data = heatmap_data[sort_indices, :]

        if verbose:
            print(f"\n  heatmap_data shape: {heatmap_data.shape}")
            print(
                f"  param_values (sorted): "
                f"[{param_values[0]:.6f}, ..., {param_values[-1]:.6f}]"
            )
            print(
                f"  heatmap_data range (raw): "
                f"[{heatmap_data.min():.4e}, {heatmap_data.max():.4e}]"
            )

        if normalize_mode == "per_column":
            for i in range(heatmap_data.shape[0]):
                col_max = heatmap_data[i, :].max()
                if col_max > 0:
                    heatmap_data[i, :] = heatmap_data[i, :] / col_max
            if verbose:
                print("  Applied per_column normalization: each simulation -> [0, 1]")
        elif normalize_mode == "global":
            global_max = heatmap_data.max()
            if global_max > 0:
                heatmap_data = heatmap_data / global_max
            if verbose:
                print("  Applied global normalization: entire heatmap -> [0, 1]")
        elif normalize_mode == "none" and verbose:
            print("  No normalization applied (raw values)")

        if verbose:
            print(
                f"  heatmap_data range (after norm): "
                f"[{heatmap_data.min():.4e}, {heatmap_data.max():.4e}]"
            )

        param_values_scaled = param_values * param_scale

        if verbose:
            print(f"  param_scale: {param_scale}")
            print(
                f"  param_values_scaled: "
                f"[{param_values_scaled[0]:.6f}, ..., {param_values_scaled[-1]:.6f}]"
            )

        return frequencies, heatmap_data, param_values_scaled, param_values
    
    def plot_transmission_crosssection_heatmap(
        self,
        swapping_parameter: str,
        x: float,
        x_width: Optional[float] = None,
        freq_unit: str = "GHz",
        trim_0f: int = 0,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        normalize: Union[bool, str] = "per_column",
        cmap: str = "inferno",
        ax: Optional[Axes] = None,
        mark_on_ax: Optional[Axes] = None,
        flip: bool = False,
        log_scale: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        disable_averaging: bool = False,
        interpolation: str = "nearest",
        colorbar_label: Optional[str] = None,
        param_scale: float = 1.0,
        param_label: Optional[str] = None,
        title: Optional[str] = None,
        figsize: tuple = (10, 6),
        dpi: int = 100,
        verbose: bool = False,
        # Publication-quality options
        colorbar_inset: bool = False,
        colorbar_position: str = "upper right",
        colorbar_width: str = "45%",
        colorbar_height: str = "4%",
        colorbar_bg_alpha: float = 0.5,
        show_grid: bool = False,
        grid_alpha: float = 0.3,
        grid_color: str = "white",
        grid_linestyle: str = "--",
        grid_axis: str = "y",
        **kwargs,
    ):
        """Plot 2D heatmap of transmission cross-sections vs parameter.
        
        Creates a 2D visualization showing how transmission cross-section
        at a specific x position varies with the swapping parameter.
        
        ALGORITHM FLOW:
        ---------------
        1. For each simulation result in batch:
           a. Extract transmission[:, x_index] or average over x_width
           b. Apply trim_0f (remove N lowest frequency points)
           c. Apply fmin/fmax frequency limits
           d. Apply normalization (per_column, global, or none)
        2. Stack all cross-sections into 2D array (n_params × n_freq)
        3. Transpose to (n_freq × n_params) for imshow
        4. Apply log_scale if requested
        5. Plot with imshow
        
        NORMALIZATION MODES:
        -------------------
        - "per_column" (default): Each cross-section normalized to [0, 1] independently
          → Shows RELATIVE spectral shape, loses absolute intensity comparison
        - "global": Entire heatmap normalized to [0, 1]
          → Preserves relative intensity between simulations
        - False/None/"none": No normalization, raw FFT amplitudes
          → Raw values, may need vmin/vmax adjustment
        
        Parameters
        ----------
        swapping_parameter : str
            Name of the parameter to use for the x-axis (e.g., "bex", "d", "B0")
            Must match a parameter extracted during compute_all()
        x : float
            X position for cross-section extraction.
            - If > 1e-6: interpreted as meters (e.g., 24700e-9 = 24700 nm)
            - If <= 1e-6: interpreted as cell index
        x_width : float, optional
            Width for averaging around x position (in same units as x).
            If None, extracts single column at x.
        freq_unit : str, default="GHz"
            Frequency unit for display: "Hz", "kHz", "MHz", "GHz", "THz"
        trim_0f : int, default=0
            Number of lowest frequency points to remove (DC component etc.)
        fmin : float, optional
            Minimum frequency to display (in freq_unit)
        fmax : float, optional
            Maximum frequency to display (in freq_unit)
        normalize : bool or str, default="per_column"
            Normalization mode:
            - "per_column" or True: Normalize each column to [0, 1]
            - "global": Normalize entire heatmap to [0, 1]
            - False, None, "none": No normalization
        cmap : str, default="inferno"
            Matplotlib colormap name
        ax : Axes, optional
            Matplotlib axes to plot on. If None, creates new figure.
        mark_on_ax : Axes, optional
            Additional axes to mark the x position on (e.g., dispersion plot)
        flip : bool, default=False
            Flip the frequency axis
        log_scale : bool, default=False
            Use logarithmic color scale (applies log10)
        vmin, vmax : float, optional
            Explicit color scale limits. Overrides automatic scaling.
        disable_averaging : bool, default=False
            If True, always extract single column even if x_width is specified
        interpolation : str, default="nearest"
            Interpolation method for imshow: "nearest", "bilinear", "bicubic", etc.
        colorbar_label : str, optional
            Custom label for colorbar. If None, auto-generates based on settings.
        param_scale : float, default=1.0
            Scale factor for parameter values. E.g., 1000 to convert T → mT.
        param_label : str, optional
            Custom label for parameter axis. E.g., "Applied Field (mT)".
            If None, uses swapping_parameter name.
        title : str, optional
            Custom title for the plot. If None, auto-generates.
        figsize : tuple, default=(10, 6)
            Figure size in inches (width, height). Only used if ax is None.
        dpi : int, default=100
            Figure resolution in dots per inch. Only used if ax is None.
        verbose : bool, default=False
            Print detailed information about extraction process
        **kwargs
            Additional arguments passed to imshow
            
        Returns
        -------
        fig : Figure
            Matplotlib figure
        ax : Axes
            Matplotlib axes
        img : AxesImage
            The image object (can be used to update colorbar, etc.)
            
        Examples
        --------
        >>> # Basic usage with per-column normalization (default)
        >>> batch_result.plot_transmission_crosssection_heatmap(
        ...     swapping_parameter="bex",
        ...     x=24700e-9,
        ...     fmax=50,
        ... )
        
        >>> # With unit conversion (T → mT) and custom label
        >>> batch_result.plot_transmission_crosssection_heatmap(
        ...     swapping_parameter="bex",
        ...     x=24700e-9,
        ...     param_scale=1000,  # T → mT
        ...     param_label="Applied Field (mT)",
        ...     fmax=50,
        ... )
        
        >>> # Global normalization to compare absolute intensities
        >>> batch_result.plot_transmission_crosssection_heatmap(
        ...     swapping_parameter="bex",
        ...     x=24700e-9,
        ...     normalize="global",
        ...     fmax=50,
        ... )
        
        >>> # Raw values with explicit limits
        >>> batch_result.plot_transmission_crosssection_heatmap(
        ...     swapping_parameter="bex",
        ...     x=24700e-9,
        ...     normalize=False,
        ...     vmin=0, vmax=1e-8,
        ...     log_scale=True,
        ... )
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for plotting")
        normalize_mode = self._normalize_mode(normalize)

        frequencies, heatmap_data, param_values_scaled, _ = (
            self._prepare_crosssection_heatmap_data(
                swapping_parameter=swapping_parameter,
                x=x,
                x_width=x_width,
                freq_unit=freq_unit,
                trim_0f=trim_0f,
                fmin=fmin,
                fmax=fmax,
                normalize_mode=normalize_mode,
                flip=flip,
                disable_averaging=disable_averaging,
                param_scale=param_scale,
                verbose=verbose,
            )
        )
        
        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.figure
        
        # Prepare for imshow (transpose so frequency is on y-axis)
        plot_data = heatmap_data.T  # Now (n_frequencies, n_params)
        
        # Apply log scale if requested
        if log_scale:
            plot_data = np.log10(plot_data + 1e-10)  # Add small value to avoid log(0)
            if verbose:
                print(f"  Applied log10 scale")
        
        # Create imshow with SCALED parameter values
        extent = [
            param_values_scaled.min(), param_values_scaled.max(),
            frequencies.min(), frequencies.max()
        ]
        
        if verbose:
            print(f"  extent: {extent}")
        
        img = ax.imshow(
            plot_data,
            aspect='auto',
            origin='lower',
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
            **kwargs,
        )
        
        # Labels and formatting
        if param_label is not None:
            ax.set_xlabel(param_label, fontsize=12)
        else:
            ax.set_xlabel(f"{swapping_parameter}", fontsize=12)
        ax.set_ylabel(f"Frequency ({freq_unit})", fontsize=12)
        
        if title is not None:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(
                f"Transmission Cross-section at x={x*1e9:.1f} nm",
                fontsize=14
            )
        
        # Add grid if requested
        if show_grid:
            ax.grid(
                True,
                axis=grid_axis,
                alpha=grid_alpha,
                color=grid_color,
                linestyle=grid_linestyle,
                zorder=5,
            )
        
        # Determine colorbar label
        if colorbar_label is not None:
            cbar_label = colorbar_label
        elif log_scale:
            cbar_label = "log₁₀(T)"
        elif normalize_mode == "per_column":
            cbar_label = "T (norm.)"
        elif normalize_mode == "global":
            cbar_label = "T (global)"
        else:
            cbar_label = "Transmission"
        
        # Determine vmin/vmax for colorbar labels
        actual_vmin = vmin if vmin is not None else float(plot_data.min())
        actual_vmax = vmax if vmax is not None else float(plot_data.max())
        
        # Colorbar - inset or external
        if colorbar_inset:
            # Create publication-quality inset colorbar
            # Background box
            cbbox = inset_axes(
                ax, width=colorbar_width, height=colorbar_height, 
                loc=colorbar_position,
                bbox_to_anchor=(0.0, 0.92, 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0,
            )
            for spine in cbbox.spines.values():
                spine.set_visible(False)
            cbbox.tick_params(
                axis='both', left=False, top=False, right=False, bottom=False,
                labelleft=False, labeltop=False, labelright=False, labelbottom=False
            )
            cbbox.set_facecolor([0, 0, 0, colorbar_bg_alpha])
            
            # Inner colorbar
            cbar_ax = inset_axes(cbbox, '85%', '35%', loc='upper center', borderpad=0)
            cbar = fig.colorbar(img, cax=cbar_ax, orientation="horizontal")
            cbar.set_ticks([])
            cbar.ax.set_xticklabels([])
            cbar.outline.set_visible(False)
            
            # Format values
            def format_val(v):
                if abs(v) >= 1000 or (abs(v) < 0.01 and v != 0):
                    return f"{v:.2e}"
                elif abs(v) < 10:
                    return f"{v:.3f}"
                else:
                    return f"{v:.1f}"
            
            # Labels
            cbar_ax.text(0.08, -1.2, format_val(actual_vmin), fontsize=10, ha='center', va='center',
                        color='white', fontweight='bold', transform=cbar_ax.transAxes)
            cbar_ax.text(0.5, -1.2, cbar_label, fontsize=11, ha='center', va='center',
                        color='white', fontweight='bold', transform=cbar_ax.transAxes)
            cbar_ax.text(0.92, -1.2, format_val(actual_vmax), fontsize=10, ha='center', va='center',
                        color='white', fontweight='bold', transform=cbar_ax.transAxes)
        else:
            # Standard external colorbar
            cbar = plt.colorbar(img, ax=ax)
            cbar.set_label(cbar_label, fontsize=11)
        
        # Mark position on reference axis if provided
        if mark_on_ax is not None:
            try:
                # Add vertical line at x position
                mark_on_ax.axvline(
                    x * 1e9,  # Convert to nm for marking
                    color='red',
                    linestyle='--',
                    linewidth=2,
                    alpha=0.7,
                    label=f'x={x*1e9:.1f} nm'
                )
                mark_on_ax.legend()
            except Exception as e:
                log.warning(f"Failed to mark position on reference axis: {e}")
        
        if verbose:
            print(f"{'='*60}\n")
        
        return fig, ax, img

    def plot_transmission_crosssection_heatmap_difference(
        self,
        other: "BatchTransmissionResult",
        swapping_parameter: str,
        x: float,
        x_width: Optional[float] = None,
        freq_unit: str = "GHz",
        trim_0f: int = 0,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
        normalize: Union[bool, str] = "per_column",
        cmap: str = "coolwarm",
        ax: Optional[Axes] = None,
        flip: bool = False,
        disable_averaging: bool = False,
        interpolation: str = "nearest",
        param_scale: float = 1.0,
        param_label: Optional[str] = None,
        title: Optional[str] = None,
        figsize: tuple = (10, 6),
        dpi: int = 100,
        symmetric_clim: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        colorbar_label: Optional[str] = None,
        verbose: bool = False,
        align_strategy: str = "strict",
        atol: float = 1e-9,
        rtol: float = 1e-6,
        **kwargs,
    ):
        """Plot a difference heatmap between two batch results.

        The function extracts matching transmission cross-sections from both
        batches, applies the same normalization, aligns frequency grids when
        needed, and plots ``self - other`` using a diverging colormap.

        Parameters
        ----------
        align_strategy : {"strict", "intersect"}, default "strict"
            - "strict": require identical parameter arrays (len and values).
            - "intersect": keep only matching parameter values (within atol/rtol).
        atol, rtol : float
            Tolerances for parameter matching when align_strategy="intersect".
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for plotting")
        if not isinstance(other, BatchTransmissionResult):
            raise TypeError("other must be a BatchTransmissionResult")

        normalize_mode = self._normalize_mode(normalize)

        freq_a, heatmap_a, param_values_scaled_a, param_values_a = (
            self._prepare_crosssection_heatmap_data(
                swapping_parameter=swapping_parameter,
                x=x,
                x_width=x_width,
                freq_unit=freq_unit,
                trim_0f=trim_0f,
                fmin=fmin,
                fmax=fmax,
                normalize_mode=normalize_mode,
                flip=flip,
                disable_averaging=disable_averaging,
                param_scale=param_scale,
                verbose=verbose,
            )
        )
        freq_b, heatmap_b, _, param_values_b = (
            other._prepare_crosssection_heatmap_data(
                swapping_parameter=swapping_parameter,
                x=x,
                x_width=x_width,
                freq_unit=freq_unit,
                trim_0f=trim_0f,
                fmin=fmin,
                fmax=fmax,
                normalize_mode=normalize_mode,
                flip=flip,
                disable_averaging=disable_averaging,
                param_scale=param_scale,
                verbose=verbose,
            )
        )

        def _match_params(
            params_a: np.ndarray,
            params_b: np.ndarray,
            atol_match: float,
            rtol_match: float,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Match parameter values from A to B, respecting tolerances."""
            matched_idx_a = []
            matched_idx_b = []
            matched_vals = []
            used_b = set()
            for i, val in enumerate(params_a):
                candidates = np.where(np.isclose(val, params_b, atol=atol_match, rtol=rtol_match))[0]
                candidates = [c for c in candidates if c not in used_b]
                if len(candidates) == 0:
                    continue
                j = candidates[0]
                used_b.add(j)
                matched_idx_a.append(i)
                matched_idx_b.append(j)
                matched_vals.append(val)
            return (
                np.array(matched_idx_a, dtype=int),
                np.array(matched_idx_b, dtype=int),
                np.array(matched_vals),
            )

        if len(param_values_a) != len(param_values_b) or not np.allclose(
            param_values_a, param_values_b, rtol=rtol, atol=atol
        ):
            if align_strategy == "intersect":
                idx_a, idx_b, matched_vals = _match_params(param_values_a, param_values_b, atol, rtol)
                if len(idx_a) == 0:
                    raise ValueError(
                        "No overlapping parameter values between batches (after intersection). "
                        "Provide matching batches or use a different align_strategy."
                    )
                if verbose:
                    log.info(
                        "Aligning on intersection of parameters (%d values kept out of %d/%d)",
                        len(idx_a),
                        len(param_values_a),
                        len(param_values_b),
                    )
                param_values_a = param_values_a[idx_a]
                param_values_b = param_values_b[idx_b]
                param_values_scaled_a = param_values_scaled_a[idx_a]
                heatmap_a = heatmap_a[idx_a, :]
                heatmap_b = heatmap_b[idx_b, :]
            else:
                raise ValueError(
                    "Batch parameter values do not align. "
                    "Ensure both batches were computed with the same swapping_parameter, "
                    "or call with align_strategy='intersect'."
                )

        def _interp_heatmap(
            source_freq: np.ndarray,
            source_data: np.ndarray,
            target_freq: np.ndarray,
        ) -> np.ndarray:
            """Interpolate heatmap rows onto target_freq (handles flipped axes)."""
            source_freq_arr = np.asarray(source_freq)
            target_freq_arr = np.asarray(target_freq)

            source_desc = np.any(np.diff(source_freq_arr) < 0)
            target_desc = np.any(np.diff(target_freq_arr) < 0)

            if source_desc:
                source_freq_arr = source_freq_arr[::-1]
                source_data = source_data[:, ::-1]
            if target_desc:
                target_freq_arr = target_freq_arr[::-1]

            interpolated = np.array(
                [
                    np.interp(
                        target_freq_arr,
                        source_freq_arr,
                        row,
                        left=0.0,
                        right=0.0,
                    )
                    for row in source_data
                ]
            )

            if target_desc:
                interpolated = interpolated[:, ::-1]
            return interpolated

        freq_target = freq_a
        heatmap_b_aligned = heatmap_b
        if len(freq_a) != len(freq_b) or not np.allclose(
            freq_a, freq_b, rtol=1e-6, atol=1e-9
        ):
            heatmap_b_aligned = _interp_heatmap(freq_b, heatmap_b, freq_target)
            if verbose:
                log.info(
                    "Aligned second batch heatmap to %d frequency points", len(freq_target)
                )

        diff_data = heatmap_a - heatmap_b_aligned
        plot_data = diff_data.T

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.figure

        extent = [
            param_values_scaled_a.min(),
            param_values_scaled_a.max(),
            freq_target.min(),
            freq_target.max(),
        ]

        if symmetric_clim and vmin is None and vmax is None:
            abs_max = np.nanmax(np.abs(plot_data))
            if not np.isfinite(abs_max) or abs_max == 0:
                abs_max = 1.0
            vmin_plot, vmax_plot = -abs_max, abs_max
        else:
            vmin_plot, vmax_plot = vmin, vmax

        img = ax.imshow(
            plot_data,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin_plot,
            vmax=vmax_plot,
            interpolation=interpolation,
            **kwargs,
        )

        ax.set_ylabel(f"Frequency ({freq_unit})", fontsize=12)
        ax.set_xlabel(
            param_label if param_label is not None else swapping_parameter,
            fontsize=12,
        )

        default_title = f"Difference heatmap (batch1 - batch2) at x={x*1e9:.1f} nm"
        ax.set_title(title or default_title, fontsize=14)

        if colorbar_label is not None:
            cbar_label = colorbar_label
        elif normalize_mode == "per_column":
            cbar_label = "Delta T (norm.)"
        elif normalize_mode == "global":
            cbar_label = "Delta T (global)"
        else:
            cbar_label = "Delta T"

        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label(cbar_label, fontsize=11)

        return fig, ax, img
    
    def _extract_crosssection(
        self,
        result: TransmissionResult,
        x: float,
        x_width: Optional[float],
        trim_0f: int,
        fmin: Optional[float],
        fmax: Optional[float],
        freq_unit: str,
        normalize: bool,
        flip: bool,
        disable_averaging: bool,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract and process cross-section from a single result.
        
        This method replicates the logic from TransmissionResult.plot_transmission_crosssection
        to ensure consistency.
        
        Parameters
        ----------
        result : TransmissionResult
            Transmission result object
        x : float
            X position (meters if > 1e-6, else index)
        x_width : float, optional
            Width for averaging
        trim_0f : int
            Number of low-frequency points to trim
        fmin : float, optional
            Minimum frequency (in freq_unit)
        fmax : float, optional
            Maximum frequency (in freq_unit)
        freq_unit : str
            Frequency unit
        normalize : bool
            Whether to normalize this cross-section
        flip : bool
            Flip frequency axis
        disable_averaging : bool
            Disable averaging even if x_width specified
        verbose : bool
            Print debug info
            
        Returns
        -------
        freq_displayed : np.ndarray
            Frequencies in display units
        cross_section : np.ndarray
            Cross-section values
        """
        from .plot import FREQ_SCALE
        
        # Get data
        transmission = result.transmission
        frequencies = result.frequencies.copy()  # Copy to avoid modifying original
        x_positions = result.x_positions
        
        if verbose:
            print(f"\n  _extract_crosssection():")
            print(f"    transmission.shape: {transmission.shape}")
            print(f"    x_positions range: [{x_positions.min():.1f}, {x_positions.max():.1f}] nm")
        
        # Determine if x is in physical units or index
        # Use 1e-6 as threshold (1 micrometer) - anything larger is definitely meters
        dx = result.dx if hasattr(result, 'dx') else result.config.metadata.get("dx", None)
        
        if dx is not None and x > 1e-6:
            # x is in physical units (meters) - convert to index
            x_nm = x * 1e9  # Convert to nm for comparison with x_positions
            # Find closest x_position
            x_index = np.abs(x_positions - x_nm).argmin()
            if verbose:
                print(f"    x={x} m → x_nm={x_nm:.1f} nm → x_index={x_index}")
                print(f"    actual x at index: {x_positions[x_index]:.1f} nm")
        else:
            # x is an index
            x_index = int(x)
            if verbose:
                print(f"    x={x} (treated as index) → x_index={x_index}")
        
        # Handle x_width averaging
        if x_width is not None and not disable_averaging:
            if dx is not None:
                # x_width in meters
                width_nm = x_width * 1e9
                # Find indices within range
                x_min = x_positions[x_index] - width_nm / 2
                x_max = x_positions[x_index] + width_nm / 2
                mask = (x_positions >= x_min) & (x_positions <= x_max)
                indices = np.where(mask)[0]
                
                if len(indices) == 0:
                    indices = [x_index]
                
                if verbose:
                    print(f"    x_width={x_width} m → {width_nm:.1f} nm")
                    print(f"    averaging over indices {indices[0]}..{indices[-1]} ({len(indices)} points)")
                
                cross_section = transmission[:, indices].mean(axis=1)
            else:
                width_cells = int(x_width)
                half_width = width_cells // 2
                start_idx = max(0, x_index - half_width)
                end_idx = min(transmission.shape[1], x_index + half_width + 1)
                cross_section = transmission[:, start_idx:end_idx].mean(axis=1)
        else:
            # Single column
            if x_index < 0 or x_index >= transmission.shape[1]:
                raise IndexError(
                    f"x_index {x_index} out of bounds for shape {transmission.shape}"
                )
            cross_section = transmission[:, x_index].copy()
            if verbose:
                print(f"    extracting single column at index {x_index}")
        
        if verbose:
            print(f"    cross_section.shape: {cross_section.shape}")
            print(f"    cross_section range: [{cross_section.min():.4e}, {cross_section.max():.4e}]")
        
        # Trim low frequencies
        if trim_0f > 0:
            frequencies = frequencies[trim_0f:]
            cross_section = cross_section[trim_0f:]
            if verbose:
                print(f"    trimmed {trim_0f} low-frequency points")
        
        # Get frequency scale
        scale = FREQ_SCALE.get(freq_unit, 1.0)
        freq_displayed = frequencies * scale
        
        # Apply frequency limits
        if fmin is not None:
            mask = freq_displayed >= fmin
            freq_displayed = freq_displayed[mask]
            cross_section = cross_section[mask]
            if verbose:
                print(f"    applied fmin={fmin} {freq_unit}")
        
        if fmax is not None:
            mask = freq_displayed <= fmax
            freq_displayed = freq_displayed[mask]
            cross_section = cross_section[mask]
            if verbose:
                print(f"    applied fmax={fmax} {freq_unit}")
        
        # Normalize (if requested - but usually we do this globally in the heatmap function)
        if normalize:
            max_val = cross_section.max()
            if max_val > 0:
                cross_section = cross_section / max_val
            if verbose:
                print(f"    normalized to [0, 1]")
        
        # Flip if requested
        if flip:
            cross_section = cross_section[::-1]
            freq_displayed = freq_displayed[::-1]
        
        if verbose:
            print(f"    final cross_section range: [{cross_section.min():.4e}, {cross_section.max():.4e}]")
            print(f"    final freq range: [{freq_displayed.min():.2f}, {freq_displayed.max():.2f}] {freq_unit}")
        
        return freq_displayed, cross_section


class BatchTransmission:
    """Batch transmission computation handler.
    
    Enables parallel computation of transmission analysis across
    multiple simulation results with unified configuration.
    """
    
    def __init__(
        self,
        results: List[Any],
        mmpp_ref: Any,
        dataset_name: Optional[str] = None,
        slice_info: Optional[Any] = None,
    ):
        """Initialize batch transmission processor.
        
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
    
    def compute_all(
        self,
        config: Optional[TransmissionConfig] = None,
        dataset_name: Optional[str] = None,
        slice_info: Optional[Any] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        use_cache: bool = True,
        save: bool = False,
        force: bool = False,
        cache_path: Optional[Union[str, Path]] = None,
        extract_parameters: Optional[List[str]] = None,
        save_batch: bool = True,
        batch_cache_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> BatchTransmissionResult:
        """Compute transmission for all results in batch.
        
        Supports caching of the entire batch result. If a cached batch with matching
        configuration exists, it will be loaded automatically unless force=True.
        
        Parameters
        ----------
        config : TransmissionConfig, optional
            Transmission configuration. If None, created from kwargs
        dataset_name : str, optional
            Dataset name to use (e.g., 'm_layer13'). If None, uses instance dataset_name
        slice_info : Any, optional
            Slice information for dataset subsetting. If None, uses instance slice_info
        parallel : bool, default=True
            Whether to use parallel processing
        max_workers : int, optional
            Maximum number of worker threads (None for auto)
        use_cache : bool, default=True
            Whether to use cached results for individual computations
        save : bool, default=False
            Whether to save individual results to cache
        force : bool, default=False
            Force recomputation even if cached batch exists
        cache_path : str or Path, optional
            Path for individual result cache storage
        extract_parameters : List[str], optional
            List of parameter names to extract from job attributes.
            If None, attempts to extract common parameters like B0, d, p
        save_batch : bool, default=True
            Whether to save the entire batch result to a file for future use.
            The file is saved in batch_cache_dir with a hash-based filename.
        batch_cache_dir : str or Path, optional
            Directory for batch cache files. If None, uses first job's parent 
            directory with '.mmpp_batch_cache' subfolder.
        **kwargs
            Additional arguments for TransmissionConfig if config is None
            
        Returns
        -------
        BatchTransmissionResult
            Container with all computed results and parameters
            
        Examples
        --------
        >>> # First run - computes and saves
        >>> batch_result = job[:50].fft.transmission.compute_all(
        ...     spatial_window=150,
        ...     extract_parameters=["bex", "d", "p"],
        ...     save_batch=True,
        ... )
        
        >>> # Second run - loads from cache automatically
        >>> batch_result = job[:50].fft.transmission.compute_all(
        ...     spatial_window=150,  # Same parameters!
        ...     extract_parameters=["bex", "d", "p"],
        ... )
        
        >>> # Force recomputation
        >>> batch_result = job[:50].fft.transmission.compute_all(
        ...     spatial_window=150,
        ...     force=True,  # Ignore cache
        ... )
        """
        from ...fft import FFT
        
        # Use provided dataset_name/slice_info or fall back to instance values
        active_dataset_name = dataset_name if dataset_name is not None else self.dataset_name
        active_slice_info = slice_info if slice_info is not None else self.slice_info
        
        # Determine parameters to extract
        if extract_parameters is None:
            # Common simulation parameters
            extract_parameters = ["B0", "d", "p", "thickness", "period", "bias_field"]
        
        # Build config from kwargs if not provided
        if config is None:
            config = TransmissionConfig(**kwargs)
        
        # Generate cache hash for batch result
        batch_cache_hash = self._generate_batch_cache_hash(
            config, active_dataset_name, active_slice_info, extract_parameters
        )
        
        # Determine batch cache directory
        if batch_cache_dir is None:
            # Use first result's parent directory
            if self.results:
                first_path = Path(self.results[0].path)
                batch_cache_dir = first_path.parent / ".mmpp_batch_cache"
            else:
                batch_cache_dir = Path(".mmpp_batch_cache")
        else:
            batch_cache_dir = Path(batch_cache_dir)
        
        batch_cache_file = batch_cache_dir / f"batch_{batch_cache_hash}.pkl"
        
        # Try to load from cache if not forcing recomputation
        if not force and save_batch and batch_cache_file.exists():
            try:
                log.info(f"Found cached batch result: {batch_cache_file}")
                cached_result = BatchTransmissionResult.load(batch_cache_file)
                
                # Verify the cached result matches our expectations
                if len(cached_result.results) == len(self.results):
                    log.info(f"✅ Loaded {len(cached_result.results)} results from cache")
                    return cached_result
                else:
                    log.warning(
                        f"Cache mismatch: {len(cached_result.results)} cached vs "
                        f"{len(self.results)} expected. Recomputing..."
                    )
            except Exception as e:
                log.warning(f"Failed to load cached batch result: {e}. Recomputing...")
        
        log.info(f"Starting batch transmission computation for {len(self.results)} results")
        log.info(f"Parallel: {parallel}, Use cache: {use_cache}, Save: {save}")
        
        successful = 0
        failed = 0
        errors = []
        computed_results = []
        computation_times = []
        
        # Initialize parameter storage
        parameters: Dict[str, List[Any]] = {param: [] for param in extract_parameters}
        job_paths = []
        
        def compute_single_result(result_info):
            """Compute transmission for a single result."""
            i, result = result_info
            start_time = time.time()
            
            try:
                log.debug(
                    f"Computing transmission for result {i + 1}/{len(self.results)}: {result.path}"
                )
                
                # Create FFT analyzer
                fft_analyzer = FFT(result, self.mmpp_ref)
                
                # Get transmission interface
                transmission_interface = fft_analyzer.transmission
                
                # Apply dataset/slice context if provided
                if active_dataset_name is not None:
                    transmission_interface = transmission_interface.clone_for_dataset(
                        active_dataset_name, active_slice_info
                    )
                
                # Compute transmission - pass config only, not kwargs
                # (kwargs were already used to create config above)
                trans_result = transmission_interface.compute(
                    config,
                    use_cache=use_cache,
                    save=save,
                    force=force,
                    cache_path=cache_path,
                )
                
                # Extract parameters from job attributes
                extracted_params = {}
                for param_name in extract_parameters:
                    if hasattr(result, 'attributes') and isinstance(result.attributes, dict):
                        param_value = result.attributes.get(param_name, None)
                    else:
                        param_value = None
                    extracted_params[param_name] = param_value
                
                computation_time = time.time() - start_time
                
                return {
                    "success": True,
                    "result": trans_result,
                    "path": result.path,
                    "parameters": extracted_params,
                    "computation_time": computation_time,
                    "error": None,
                }
                
            except Exception as e:
                computation_time = time.time() - start_time
                log.error(f"Failed to compute transmission for {result.path}: {e}")
                # Cleanup after error
                gc.collect()
                return {
                    "success": False,
                    "result": None,
                    "path": result.path,
                    "parameters": {param: None for param in extract_parameters},
                    "computation_time": computation_time,
                    "error": str(e),
                }
            finally:
                # Always run garbage collection after each computation
                gc.collect()
        
        # Execute computations
        if parallel and len(self.results) > 1:
            # Use ThreadPoolExecutor - numpy/scipy FFT releases GIL so this works well
            if max_workers is None:
                # Default: use all CPU cores for batch processing
                max_workers = min(len(self.results), os.cpu_count() or 8)
            
            log.info(f"Using parallel execution with {max_workers} workers")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_result = {
                    executor.submit(compute_single_result, (i, result)): result
                    for i, result in enumerate(self.results)
                }
                
                # Process completed tasks with progress bar
                try:
                    from tqdm import tqdm
                    iterator = tqdm(
                        as_completed(future_to_result),
                        total=len(self.results),
                        desc="Computing transmission",
                        unit="result",
                    )
                except ImportError:
                    iterator = as_completed(future_to_result)
                
                for future in iterator:
                    result_data = future.result()
                    computation_times.append(result_data["computation_time"])
                    
                    if result_data["success"]:
                        successful += 1
                        computed_results.append(result_data["result"])
                        job_paths.append(result_data["path"])
                        
                        # Collect parameters
                        for param_name, param_value in result_data["parameters"].items():
                            parameters[param_name].append(param_value)
                    else:
                        failed += 1
                        errors.append({
                            "path": result_data["path"],
                            "error": result_data["error"]
                        })
        else:
            # Sequential execution
            log.info("Using sequential execution")
            
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    enumerate(self.results),
                    total=len(self.results),
                    desc="Computing transmission",
                    unit="result",
                )
            except ImportError:
                iterator = enumerate(self.results)
            
            for i, result in iterator:
                result_data = compute_single_result((i, result))
                computation_times.append(result_data["computation_time"])
                
                if result_data["success"]:
                    successful += 1
                    computed_results.append(result_data["result"])
                    job_paths.append(result_data["path"])
                    
                    # Collect parameters
                    for param_name, param_value in result_data["parameters"].items():
                        parameters[param_name].append(param_value)
                else:
                    failed += 1
                    errors.append({
                        "path": result_data["path"],
                        "error": result_data["error"]
                    })
        
        # Compute statistics
        avg_time = np.mean(computation_times) if computation_times else 0
        total_time = sum(computation_times)
        
        log.info(
            f"Batch transmission computation completed: {successful} successful, {failed} failed"
        )
        log.info(f"Total time: {total_time:.2f}s, Average per result: {avg_time:.2f}s")
        
        if errors:
            log.warning(f"Errors occurred in {len(errors)} computations:")
            for error in errors[:5]:  # Show first 5 errors
                log.warning(f"  {error['path']}: {error['error']}")
        
        # Clean up parameters - remove those with all None values
        parameters = {
            param: values
            for param, values in parameters.items()
            if any(v is not None for v in values)
        }
        
        if not computed_results:
            raise RuntimeError(
                f"All {len(self.results)} transmission computations failed. "
                "Check logs for details."
            )
        
        # Create batch result container
        batch_result = BatchTransmissionResult(
            results=computed_results,
            parameters=parameters,
            job_paths=job_paths,
        )
        
        # Save batch result if requested
        if save_batch:
            try:
                batch_result.save(batch_cache_file)
                log.info(f"✅ Saved batch result to {batch_cache_file}")
            except Exception as e:
                log.warning(f"Failed to save batch result: {e}")
        
        return batch_result
    
    def _generate_batch_cache_hash(
        self,
        config: TransmissionConfig,
        dataset_name: Optional[str],
        slice_info: Optional[Any],
        extract_parameters: List[str],
    ) -> str:
        """Generate a unique hash for batch cache identification.
        
        The hash is based on:
        - All job paths (sorted for consistency)
        - TransmissionConfig parameters
        - Dataset name and slice info
        - Extract parameters list
        
        Parameters
        ----------
        config : TransmissionConfig
            Transmission configuration
        dataset_name : str, optional
            Dataset name
        slice_info : Any, optional
            Slice information
        extract_parameters : List[str]
            Parameters to extract
            
        Returns
        -------
        str
            16-character hex hash
        """
        # Collect all hashable data
        hash_data = {
            # Job paths (sorted for consistency)
            "job_paths": sorted([str(r.path) for r in self.results]),
            "n_jobs": len(self.results),
            
            # Config parameters (convert dataclass to dict)
            "config": {
                "filter_type": config.filter_type,
                "window_function": config.window_function,
                "spatial_window": config.spatial_window,
                "spatial_step": config.spatial_step,
                "spatial_window_mode": config.spatial_window_mode,
                "average_mode": config.average_mode,
                "y_integration_mode": config.y_integration_mode,
                "component_weights": config.component_weights,
                "normalize": config.normalize,
                "engine": config.engine,
                "tmax": config.tmax,
                "z_layer": config.z_layer,
                "method": config.method,
                "reference_window": config.reference_window,
                "reference_statistic": config.reference_statistic,
                "edge_taper_power": config.edge_taper_power,
                "enable_circular_components": config.enable_circular_components,
                "keep_complex_fft": config.keep_complex_fft,
                "store_component_maps": config.store_component_maps,
                "raw_fft_output": config.raw_fft_output,
            },
            
            # Dataset context
            "dataset_name": dataset_name,
            "slice_info": str(slice_info) if slice_info is not None else None,
            
            # Extract parameters
            "extract_parameters": sorted(extract_parameters),
        }
        
        # Convert to JSON string for hashing (sorted keys for consistency)
        hash_string = json.dumps(hash_data, sort_keys=True, default=str)
        
        # Generate MD5 hash (16 chars = 64 bits, enough for cache identification)
        hash_digest = hashlib.md5(hash_string.encode()).hexdigest()[:16]
        
        return hash_digest


def stack_results(
    results: Union[List[TransmissionResult], BatchTransmissionResult],
    field: str = "transmission"
) -> np.ndarray:
    """Stack transmission results into a single ndarray.
    
    Validates that all results have compatible dimensions and stacks
    the specified field into a 3D array.
    
    Parameters
    ----------
    results : List[TransmissionResult] or BatchTransmissionResult
        List of transmission results to stack
    field : str, default="transmission"
        Field name to stack ("transmission", "power_map", etc.)
        
    Returns
    -------
    np.ndarray
        Stacked array with shape (n_results, n_frequencies, n_x_positions)
        
    Raises
    ------
    ValueError
        If results have incompatible dimensions or missing fields
    """
    # Handle BatchTransmissionResult input
    if isinstance(results, BatchTransmissionResult):
        results = results.results
    
    if not results:
        raise ValueError("Cannot stack empty results list")
    
    # Get reference dimensions
    ref_result = results[0]
    if not hasattr(ref_result, field):
        raise ValueError(f"Field '{field}' not found in TransmissionResult")
    
    ref_data = getattr(ref_result, field)
    ref_shape = ref_data.shape
    ref_freq = ref_result.frequencies
    ref_x = ref_result.x_positions
    
    # Validate all results
    for i, result in enumerate(results):
        if not hasattr(result, field):
            raise ValueError(
                f"Result {i} missing field '{field}'"
            )
        
        data = getattr(result, field)
        if data.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch at result {i}: "
                f"expected {ref_shape}, got {data.shape}"
            )
        
        # Check frequency/position consistency
        if not np.allclose(result.frequencies, ref_freq):
            raise ValueError(
                f"Frequency mismatch at result {i}. "
                "All results must have identical frequency arrays."
            )
        
        if not np.allclose(result.x_positions, ref_x):
            raise ValueError(
                f"X-position mismatch at result {i}. "
                "All results must have identical x-position arrays."
            )
    
    # Stack results
    stacked = np.stack([getattr(r, field) for r in results], axis=0)
    
    log.info(
        f"Stacked {len(results)} results into array with shape {stacked.shape}"
    )
    
    return stacked


__all__ = [
    "BatchTransmission",
    "BatchTransmissionResult",
    "stack_results",
]
