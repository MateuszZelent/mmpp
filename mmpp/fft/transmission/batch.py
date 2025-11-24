"""Batch transmission processing for multiple simulation results.

This module provides batch processing capabilities for FFT transmission analysis,
enabling parallel computation across multiple jobs and visualization of results
as parametric heatmaps.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np

from ...cli.logging_config import get_mmpp_logger
from .compute import TransmissionConfig, TransmissionResult
from .interface import FFTTransmissionInterface

log = get_mmpp_logger("mmpp.fft.transmission.batch")

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
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
    
    def plot_transmission_crosssection_heatmap(
        self,
        swapping_parameter: str,
        x: float,
        x_width: Optional[float] = None,
        freq_unit: str = "GHz",
        trim_0f: int = 0,
        fmax: Optional[float] = None,
        normalize: bool = True,
        cmap: str = "inferno",
        ax: Optional[Axes] = None,
        mark_on_ax: Optional[Axes] = None,
        flip: bool = False,
        log_scale: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        disable_averaging: bool = False,
        **kwargs,
    ):
        """Plot 2D heatmap of transmission cross-sections vs parameter.
        
        Creates a 2D visualization showing how transmission cross-section
        at a specific x position varies with the swapping parameter.
        
        Parameters
        ----------
        swapping_parameter : str
            Name of the parameter to use for the x-axis (e.g., "B0", "d", "p")
        x : float
            X position for cross-section (in meters if > 1, else as index)
        x_width : float, optional
            Width for averaging around x position
        freq_unit : str, default="GHz"
            Frequency unit for display
        trim_0f : int, default=0
            Number of lowest frequency points to trim
        fmax : float, optional
            Maximum frequency to display
        normalize : bool, default=True
            Whether to normalize each cross-section
        cmap : str, default="inferno"
            Colormap name
        ax : Axes, optional
            Matplotlib axes to plot on
        mark_on_ax : Axes, optional
            Additional axes to mark the x position on
        flip : bool, default=False
            Flip the cross-section data
        log_scale : bool, default=False
            Use logarithmic color scale
        vmin, vmax : float, optional
            Color scale limits
        disable_averaging : bool, default=False
            Disable x-width averaging
        **kwargs
            Additional plotting arguments
            
        Returns
        -------
        fig : Figure
            Matplotlib figure
        ax : Axes
            Matplotlib axes
        img : AxesImage
            The image object
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for plotting")
        
        # Get parameter values
        param_values = self.get_parameter_values(swapping_parameter)
        
        # Extract cross-sections from all results
        cross_sections = []
        frequencies = None
        
        for i, result in enumerate(self.results):
            # Use the existing cross-section extraction logic
            # This ensures consistency with single-result plotting
            try:
                # Extract cross-section using internal method
                freq, cross_section = self._extract_crosssection(
                    result, x, x_width, trim_0f, fmax, 
                    freq_unit, normalize, flip, disable_averaging
                )
                
                if frequencies is None:
                    frequencies = freq
                else:
                    # Verify frequency consistency
                    if not np.allclose(frequencies, freq):
                        log.warning(
                            f"Frequency mismatch at result {i}. "
                            "Results may have different configurations."
                        )
                
                cross_sections.append(cross_section)
                
            except Exception as e:
                log.error(f"Failed to extract cross-section from result {i}: {e}")
                # Use NaN array as placeholder
                if frequencies is not None:
                    cross_sections.append(np.full_like(frequencies, np.nan))
                else:
                    raise
        
        # Stack into 2D array: (n_params, n_frequencies)
        heatmap_data = np.array(cross_sections)
        
        # Create plot
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure
        
        # Prepare for imshow (transpose so frequency is on y-axis)
        plot_data = heatmap_data.T  # Now (n_frequencies, n_params)
        
        # Apply log scale if requested
        if log_scale:
            plot_data = np.log10(plot_data + 1e-10)  # Add small value to avoid log(0)
        
        # Create imshow
        extent = [
            param_values.min(), param_values.max(),
            frequencies.min(), frequencies.max()
        ]
        
        img = ax.imshow(
            plot_data,
            aspect='auto',
            origin='lower',
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest',
        )
        
        # Labels and formatting
        ax.set_xlabel(f"{swapping_parameter}", fontsize=12)
        ax.set_ylabel(f"Frequency ({freq_unit})", fontsize=12)
        ax.set_title(
            f"Transmission Cross-section at x={x*1e9:.1f} nm",
            fontsize=14
        )
        
        # Colorbar
        cbar = plt.colorbar(img, ax=ax)
        if log_scale:
            cbar.set_label("log₁₀(Transmission)", fontsize=11)
        else:
            cbar.set_label("Transmission", fontsize=11)
        
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
        
        return fig, ax, img
    
    def _extract_crosssection(
        self,
        result: TransmissionResult,
        x: float,
        x_width: Optional[float],
        trim_0f: int,
        fmax: Optional[float],
        freq_unit: str,
        normalize: bool,
        flip: bool,
        disable_averaging: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract and process cross-section from a single result.
        
        This method replicates the logic from TransmissionResult.plot_transmission_crosssection
        to ensure consistency.
        """
        from .plot import FREQ_SCALE
        
        # Get data
        transmission = result.transmission
        frequencies = result.frequencies
        x_positions = result.x_positions
        
        # Determine if x is in physical units or index
        dx = result.config.metadata.get("dx", None)
        if dx is not None and x > 1:
            # x is in physical units (meters)
            x_index = int(x / dx)
        else:
            # x is an index
            x_index = int(x)
        
        # Handle x_width averaging
        if x_width is not None and not disable_averaging:
            if dx is not None:
                width_cells = int(x_width / dx)
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
            cross_section = transmission[:, x_index]
        
        # Trim low frequencies
        if trim_0f > 0:
            frequencies = frequencies[trim_0f:]
            cross_section = cross_section[trim_0f:]
        
        # Apply frequency limit
        if fmax is not None:
            scale = FREQ_SCALE.get(freq_unit, 1.0)
            freq_displayed = frequencies / scale
            mask = freq_displayed <= fmax
            frequencies = frequencies[mask]
            cross_section = cross_section[mask]
        
        # Normalize
        if normalize:
            max_val = cross_section.max()
            if max_val > 0:
                cross_section = cross_section / max_val
        
        # Flip if requested
        if flip:
            cross_section = cross_section[::-1]
            frequencies = frequencies[::-1]
        
        # Convert frequencies to display unit
        scale = FREQ_SCALE.get(freq_unit, 1.0)
        freq_displayed = frequencies / scale
        
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
        parallel: bool = True,
        max_workers: Optional[int] = None,
        use_cache: bool = True,
        save: bool = False,
        force: bool = False,
        cache_path: Optional[Union[str, Path]] = None,
        extract_parameters: Optional[List[str]] = None,
        **kwargs,
    ) -> BatchTransmissionResult:
        """Compute transmission for all results in batch.
        
        Parameters
        ----------
        config : TransmissionConfig, optional
            Transmission configuration. If None, created from kwargs
        parallel : bool, default=True
            Whether to use parallel processing
        max_workers : int, optional
            Maximum number of worker threads (None for auto)
        use_cache : bool, default=True
            Whether to use cached results
        save : bool, default=False
            Whether to save results to cache
        force : bool, default=False
            Force recomputation even if cached
        cache_path : str or Path, optional
            Path for cache storage
        extract_parameters : List[str], optional
            List of parameter names to extract from job attributes.
            If None, attempts to extract common parameters like B0, d, p
        **kwargs
            Additional arguments for TransmissionConfig if config is None
            
        Returns
        -------
        BatchTransmissionResult
            Container with all computed results and parameters
        """
        from ...fft import FFT
        
        log.info(f"Starting batch transmission computation for {len(self.results)} results")
        log.info(f"Parallel: {parallel}, Use cache: {use_cache}, Save: {save}")
        
        successful = 0
        failed = 0
        errors = []
        computed_results = []
        computation_times = []
        
        # Determine parameters to extract
        if extract_parameters is None:
            # Common simulation parameters
            extract_parameters = ["B0", "d", "p", "thickness", "period", "bias_field"]
        
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
                if self.dataset_name is not None:
                    transmission_interface = transmission_interface.clone_for_dataset(
                        self.dataset_name, self.slice_info
                    )
                
                # Compute transmission
                trans_result = transmission_interface.compute(
                    config,
                    use_cache=use_cache,
                    save=save,
                    force=force,
                    cache_path=cache_path,
                    **kwargs,
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
                return {
                    "success": False,
                    "result": None,
                    "path": result.path,
                    "parameters": {param: None for param in extract_parameters},
                    "computation_time": computation_time,
                    "error": str(e),
                }
        
        # Execute computations
        if parallel and len(self.results) > 1:
            # Parallel execution
            if max_workers is None:
                max_workers = min(len(self.results), 4)  # Reasonable default
            
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
        return BatchTransmissionResult(
            results=computed_results,
            parameters=parameters,
            job_paths=job_paths,
        )


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
