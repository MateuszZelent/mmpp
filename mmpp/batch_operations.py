"""
Batch operations module for MMPP - enables serial computation across multiple simulation results.

This module provides the BatchOperations class that allows for executing FFT computations,
mode analysis, and other operations across entire directories of simulation results using
slice notation like `op[:].fft.modes.compute_modes()` (auto-selects optimal dataset).
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import numpy as np
import zarr

from .cli.logging_config import get_mmpp_logger

# Get logger for batch operations
log = get_mmpp_logger("mmpp.batch")

try:
    from .fft import FFT

    FFT_AVAILABLE = True
except ImportError:
    FFT_AVAILABLE = False
    log.warning("FFT module not available for batch operations")


class BatchFFT:
    """Batch FFT operations handler."""

    def __init__(self, results: list[Any], mmpp_ref: Any):
        """
        Initialize batch FFT operations.

        Parameters:
        -----------
        results : List[Any]
            List of ZarrJobResult objects
        mmpp_ref : Any
            Reference to MMPP instance
        """
        self.results = results
        self.mmpp_ref = mmpp_ref

    @property
    def modes(self) -> "BatchModeAnalyzer":
        """Get batch mode analyzer."""
        return BatchModeAnalyzer(self.results, self.mmpp_ref)

    @property
    def transmission(self) -> "BatchTransmission":
        """Get batch transmission analyzer."""
        from .fft.transmission.batch import BatchTransmission
        
        # Check if dataset context was set (from BatchDatasetWrapper)
        dataset_name = getattr(self, '_dataset_name', None)
        slice_info = getattr(self, '_slice_info', None)
        
        return BatchTransmission(
            self.results, 
            self.mmpp_ref,
            dataset_name=dataset_name,
            slice_info=slice_info
        )
    
    @property
    def spectrum(self) -> "BatchSpectrum":
        """Get batch spectrum analyzer.
        
        Returns batch spectrum processor for computing FFT spectra
        across multiple simulation results.
        
        Returns
        -------
        BatchSpectrum
            Batch spectrum processor
            
        Examples
        --------
        >>> # Compute batch spectrum
        >>> batch = job[:].fft.spectrum.compute_all(
        ...     extract_parameters=["B0", "d"],
        ...     fmin=5e9,
        ...     fmax=25e9,
        ... )
        >>> batch.plot_heatmap("B0")
        """
        from .fft.spectrum_batch import BatchSpectrum
        
        dataset_name = getattr(self, '_dataset_name', None)
        slice_info = getattr(self, '_slice_info', None)
        
        return BatchSpectrum(
            self.results,
            self.mmpp_ref,
            dataset_name=dataset_name,
            slice_info=slice_info,
        )

    def compute_all(self, **kwargs) -> dict[str, Any]:
        """
        Compute FFT for all results in batch.

        Parameters:
        -----------
        **kwargs : dict
            Arguments to pass to FFT computation

        Returns:
        --------
        Dict[str, Any]
            Summary of batch FFT computation results
        """
        if not FFT_AVAILABLE:
            raise ImportError("FFT functionality not available")

        log.info(f"Starting batch FFT computation for {len(self.results)} results")

        successful = 0
        failed = 0
        errors = []

        for i, result in enumerate(self.results):
            try:
                log.debug(
                    f"Computing FFT for result {i + 1}/{len(self.results)}: {result.path}"
                )
                fft_analyzer = FFT(result, self.mmpp_ref)
                fft_analyzer._compute_fft(**kwargs)
                successful += 1

            except Exception as e:
                log.error(f"Failed to compute FFT for {result.path}: {e}")
                failed += 1
                errors.append({"path": result.path, "error": str(e)})

        summary = {
            "total": len(self.results),
            "successful": successful,
            "failed": failed,
            "errors": errors,
        }

        log.info(
            f"Batch FFT computation completed: {successful} successful, {failed} failed"
        )
        return summary


class BatchModeAnalyzer:
    """Batch mode analysis operations handler."""

    def __init__(self, results: list[Any], mmpp_ref: Any):
        """
        Initialize batch mode analyzer.

        Parameters:
        -----------
        results : List[Any]
            List of ZarrJobResult objects
        mmpp_ref : Any
            Reference to MMPP instance
        """
        self.results = results
        self.mmpp_ref = mmpp_ref

    def compute_modes(
        self,
        dset: Optional[str] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Compute FMR modes for all results in batch.

        Parameters:
        -----------
        dset : str, default=None
            Dataset name to analyze (default: auto-select largest m dataset)
        parallel : bool, default=True
            Whether to use parallel processing
        max_workers : Optional[int]
            Maximum number of worker threads (None for auto)
        **kwargs : dict
            Additional arguments to pass to mode computation

        Returns:
        --------
        Dict[str, Any]
            Summary of batch mode computation results
        """
        if not FFT_AVAILABLE:
            raise ImportError("FFT functionality not available for mode analysis")

        # Auto-select largest m dataset if none specified
        if dset is None and self.results:
            dset = self.results[0].get_largest_m_dataset()

        log.info(f"Starting batch mode computation for {len(self.results)} results")
        log.info(f"Dataset: {dset}, Parallel: {parallel}")

        successful = 0
        failed = 0
        errors = []
        computation_times = []

        def compute_single_result(result_info):
            """Compute modes for a single result."""
            i, result = result_info
            start_time = time.time()

            try:
                log.debug(
                    f"Computing modes for result {i + 1}/{len(self.results)}: {result.path}"
                )

                # Get FFT analyzer for this result
                fft_analyzer = FFT(result, self.mmpp_ref)

                # Check if modes already exist
                modes_analyzer = fft_analyzer.modes
                if not modes_analyzer.modes_available:
                    # Compute modes if they don't exist
                    modes_analyzer.compute_modes(dset=dset, **kwargs)
                    log.debug(f"Computed new modes for {result.path}")
                else:
                    log.debug(f"Modes already available for {result.path}")

                computation_time = time.time() - start_time
                return {
                    "success": True,
                    "path": result.path,
                    "computation_time": computation_time,
                    "error": None,
                }

            except Exception as e:
                computation_time = time.time() - start_time
                log.error(f"Failed to compute modes for {result.path}: {e}")
                return {
                    "success": False,
                    "path": result.path,
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
                        desc="Computing modes",
                        unit="result",
                    )
                except ImportError:
                    iterator = as_completed(future_to_result)

                for future in iterator:
                    result_data = future.result()
                    computation_times.append(result_data["computation_time"])

                    if result_data["success"]:
                        successful += 1
                    else:
                        failed += 1
                        errors.append(
                            {"path": result_data["path"], "error": result_data["error"]}
                        )
        else:
            # Sequential execution
            log.info("Using sequential execution")

            try:
                from tqdm import tqdm

                iterator = tqdm(
                    enumerate(self.results),
                    total=len(self.results),
                    desc="Computing modes",
                    unit="result",
                )
            except ImportError:
                iterator = enumerate(self.results)

            for i, result in iterator:
                result_data = compute_single_result((i, result))
                computation_times.append(result_data["computation_time"])

                if result_data["success"]:
                    successful += 1
                else:
                    failed += 1
                    errors.append(
                        {"path": result_data["path"], "error": result_data["error"]}
                    )

        # Compute statistics
        avg_time = np.mean(computation_times) if computation_times else 0
        total_time = sum(computation_times)

        summary = {
            "total": len(self.results),
            "successful": successful,
            "failed": failed,
            "errors": errors,
            "dataset": dset,
            "parallel": parallel,
            "max_workers": max_workers if parallel else 1,
            "total_time": total_time,
            "average_time_per_result": avg_time,
            "computation_times": computation_times,
        }

        log.info(
            f"Batch mode computation completed: {successful} successful, {failed} failed"
        )
        log.info(f"Total time: {total_time:.2f}s, Average per result: {avg_time:.2f}s")

        return summary

    def analyze_all(self, **kwargs) -> dict[str, Any]:
        """
        Analyze all modes for all results in batch.

        Parameters:
        -----------
        **kwargs : dict
            Arguments to pass to mode analysis

        Returns:
        --------
        Dict[str, Any]
            Summary of batch mode analysis results
        """
        if not FFT_AVAILABLE:
            raise ImportError("FFT functionality not available for mode analysis")

        log.info(f"Starting batch mode analysis for {len(self.results)} results")

        successful = 0
        failed = 0
        errors = []
        all_peaks = []

        for i, result in enumerate(self.results):
            try:
                log.debug(
                    f"Analyzing modes for result {i + 1}/{len(self.results)}: {result.path}"
                )

                fft_analyzer = FFT(result, self.mmpp_ref)
                modes_analyzer = fft_analyzer.modes

                # Ensure modes are computed
                if not modes_analyzer.modes_available:
                    modes_analyzer.compute_modes()

                # Analyze peaks
                peaks = modes_analyzer.analyze_all(**kwargs)
                all_peaks.append({"path": result.path, "peaks": peaks})
                successful += 1

            except Exception as e:
                log.error(f"Failed to analyze modes for {result.path}: {e}")
                failed += 1
                errors.append({"path": result.path, "error": str(e)})

        summary = {
            "total": len(self.results),
            "successful": successful,
            "failed": failed,
            "errors": errors,
            "all_peaks": all_peaks,
        }

        log.info(
            f"Batch mode analysis completed: {successful} successful, {failed} failed"
        )
        return summary


class BatchDatasetWrapper:
    """Wrapper for dataset-aware batch operations.
    
    Allows syntax like: job[:].m_layer13[:,...,0].fft.transmission(...)
    """
    
    def __init__(self, results: list[Any], mmpp_ref: Any, dataset_name: str):
        """Initialize dataset wrapper.
        
        Parameters
        ----------
        results : List[Any]
            List of ZarrJobResult objects
        mmpp_ref : Any
            Reference to MMPP instance
        dataset_name : str
            Name of the dataset (e.g., 'm_layer13')
        """
        self.results = results
        self.mmpp_ref = mmpp_ref
        self.dataset_name = dataset_name
        self.slice_info = None
    
    def __getitem__(self, key):
        """Capture slice information, returning a new wrapper (immutable pattern)."""
        new = BatchDatasetWrapper(self.results, self.mmpp_ref, self.dataset_name)
        new.slice_info = key
        return new
    
    @property
    def fft(self) -> BatchFFT:
        """Get batch FFT operations with dataset context."""
        # Create a BatchFFT instance with dataset context
        # The transmission property will use this context
        batch_fft = BatchFFT(self.results, self.mmpp_ref)
        # Store dataset context on the batch_fft instance
        batch_fft._dataset_name = self.dataset_name
        batch_fft._slice_info = self.slice_info
        return batch_fft


class BatchNumpyDatasetWrapper:
    """Wrapper that returns stacked numpy arrays from multiple jobs.
    
    Used by job[:].get.dataset_name[slice] to return 6D numpy arrays
    with shape [n_jobs, t, z, y, x, c] (or appropriate dimensions based on data).
    
    Example
    -------
    >>> arr = job[:].get.m[:]  # Returns 6D array [n_jobs, t, z, y, x, c]
    >>> arr = job[:].get.m[0:100, ...]  # Sliced stacked array
    """
    
    def __init__(self, results: list, mmpp_ref, dataset_name: str):
        self._results = results
        self._mmpp_ref = mmpp_ref
        self._dataset_name = dataset_name
    
    def __getitem__(self, key) -> np.ndarray:
        """Return stacked sliced data as numpy array from all jobs.
        
        Returns array with shape [n_jobs, ...original_dims...]
        """
        arrays = []
        for result in self._results:
            result._ensure_zarr_loaded()
            try:
                member = result._get_zarr_member(self._dataset_name)
                arrays.append(np.asarray(member[key]))
            except (NameError, KeyError) as e:
                log.warning(f"Dataset '{self._dataset_name}' not found in {result.path}: {e}")
                continue
        
        if not arrays:
            raise ValueError(f"Dataset '{self._dataset_name}' not found in any job")
        
        # Stack all arrays along new first axis [n_jobs, ...]
        return np.stack(arrays, axis=0)
    
    @property
    def shape(self):
        """Shape of dataset from first result (all should match)."""
        if self._results:
            self._results[0]._ensure_zarr_loaded()
            try:
                member = self._results[0]._get_zarr_member(self._dataset_name)
                return (len(self._results),) + member.shape
            except (NameError, KeyError):
                pass
        return None
    
    def __repr__(self):
        return f"BatchNumpyDatasetWrapper({self._dataset_name}, n_jobs={len(self._results)}, shape={self.shape})"


class BatchNumpyGetter:
    """Helper providing direct numpy access for batch operations.
    
    Returns stacked numpy arrays from all jobs with shape [n_jobs, t, z, y, x, c].
    
    Example
    -------
    >>> # Batch access - returns 6D stacked array
    >>> arr = job[:].get.m[:]  # shape: [n_jobs, t, z, y, x, c]
    >>> arr = job[:].get.m[0:100, :, :, :, 0]  # sliced stack
    >>> 
    >>> # Works with any dataset name
    >>> arr = job[:].get.m_layer13[:]
    >>> arr = job[:].get["m_layer13"][:]
    """
    
    def __init__(self, results: list, mmpp_ref):
        self._results = results
        self._mmpp_ref = mmpp_ref
    
    def __getattr__(self, name: str) -> BatchNumpyDatasetWrapper:
        """Get BatchNumpyDatasetWrapper for dataset by attribute access."""
        # Check that at least one result has this dataset
        for result in self._results:
            result._ensure_zarr_loaded()
            if name in result._z:
                return BatchNumpyDatasetWrapper(self._results, self._mmpp_ref, name)
        raise AttributeError(f"Dataset '{name}' not found in any job")
    
    def __getitem__(self, key: str) -> BatchNumpyDatasetWrapper:
        """Get BatchNumpyDatasetWrapper for dataset by item access."""
        return self.__getattr__(key)
    
    def __repr__(self):
        if self._results:
            self._results[0]._ensure_zarr_loaded()
            datasets = list(self._results[0]._z.array_keys())
            return f"BatchNumpyGetter(n_jobs={len(self._results)}, datasets={datasets[:3]}{'...' if len(datasets) > 3 else ''})"
        return "BatchNumpyGetter(empty)"


class BatchOperations:
    """
    Main batch operations class that provides access to batch FFT and mode operations.

    This class is returned when using slice notation on MMPP objects, e.g., `op[:]`.
    It provides access to batch operations like:
    - `op[:].fft.modes.compute_modes()` (auto-selects optimal dataset)
    - `op[:].fft.compute_all()`
    - `op[:].m_layer13[:,...,0].fft.transmission(...)` (dataset-aware)
    - `op[:].get.m[:]` (direct numpy access, returns stacked array)
    """

    def __init__(self, results: list[Any], mmpp_ref: Any):
        """
        Initialize batch operations.

        Parameters:
        -----------
        results : List[Any]
            List of ZarrJobResult objects to operate on
        mmpp_ref : Any
            Reference to parent MMPP instance
        """
        self.results = results
        self.mmpp_ref = mmpp_ref

        log.info(f"Initialized batch operations for {len(results)} results")

    @property
    def fft(self) -> BatchFFT:
        """Get batch FFT operations handler."""
        return BatchFFT(self.results, self.mmpp_ref)

    @property
    def get(self) -> BatchNumpyGetter:
        """Access datasets with direct numpy output (batch version).
        
        Returns a BatchNumpyGetter that provides direct stacked numpy array
        access when slicing datasets. Returns arrays with shape [n_jobs, ...]
        where the first dimension corresponds to the number of jobs in batch.
        
        Returns
        -------
        BatchNumpyGetter
            Helper object for numpy-direct batch dataset access
        
        Example
        -------
        >>> # Batch numpy access - returns 6D array [n_jobs, t, z, y, x, c]
        >>> arr = job[:].get.m[:]
        >>> arr = job[:].get.m[0:100, :, :, :, 0]
        >>> 
        >>> # Shape will be [n_jobs, ...original_dims...]
        >>> arr.shape  # e.g. (10, 443, 1, 94, 7520, 3) for 10 jobs
        """
        return BatchNumpyGetter(self.results, self.mmpp_ref)
    
    def __getattr__(self, name: str):
        """Intercept dataset names to enable dataset-aware batch operations.
        
        This allows syntax like: job[:].m_layer13.fft.transmission(...)
        
        Parameters
        ----------
        name : str
            Attribute name (potentially a dataset name)
            
        Returns
        -------
        BatchDatasetWrapper
            Wrapper for dataset-aware operations
        """
        # Check if any result has a dataset with this name.
        for result in self.results:
            try:
                result._ensure_zarr_loaded()
                member = result._z[name]
                if isinstance(member, zarr.Array):
                    log.debug(f"Creating dataset wrapper for: {name}")
                    return BatchDatasetWrapper(self.results, self.mmpp_ref, name)
            except (KeyError, Exception):
                continue
        
        # Not a dataset — raise standard error
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __len__(self) -> int:
        """Return number of results in batch."""
        return len(self.results)

    def __repr__(self) -> str:
        """String representation of batch operations."""
        return f"BatchOperations({len(self.results)} results)"

    def __iter__(self):
        """Make batch operations iterable."""
        return iter(self.results)

    def process(
        self,
        dset: Optional[str] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Process all results in batch with comprehensive analysis.

        This method performs complete analysis including FFT computation
        and mode analysis for all results in the batch.

        Parameters:
        -----------
        dset : str, default=None
            Dataset name to analyze (default: auto-select largest m dataset)
        parallel : bool, default=True
            Whether to use parallel processing
        max_workers : Optional[int]
            Maximum number of worker threads (None for auto)
        **kwargs : dict
            Additional arguments to pass to analysis

        Returns:
        --------
        Dict[str, Any]
            Comprehensive analysis results
        """
        log.info(f"Processing {len(self.results)} results with comprehensive analysis")

        if not self.results:
            return {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "errors": [],
                "mode_results": None,
                "computation_time": 0.0,
            }

        start_time = time.time()

        try:
            # Perform mode computation and analysis
            mode_results = self.fft.modes.compute_modes(
                dset=dset, parallel=parallel, max_workers=max_workers, **kwargs
            )

            total_time = time.time() - start_time

            results = {
                "total": len(self.results),
                "successful": mode_results.get("successful", 0),
                "failed": mode_results.get("failed", 0),
                "errors": mode_results.get("errors", []),
                "mode_results": mode_results,
                "computation_time": total_time,
            }

            log.info(
                f"Batch processing completed in {total_time:.2f}s: "
                f"{results['successful']} successful, {results['failed']} failed"
            )

            return results

        except Exception as e:
            total_time = time.time() - start_time
            log.error(f"Batch processing failed after {total_time:.2f}s: {e}")
            return {
                "total": len(self.results),
                "successful": 0,
                "failed": len(self.results),
                "errors": [{"error": str(e), "context": "batch_processing"}],
                "mode_results": None,
                "computation_time": total_time,
            }

    def prepare_report(
        self, spectrum: bool = True, modes: bool = True, parallel: bool = True, **kwargs
    ) -> dict[str, Any]:
        """
        Prepare comprehensive report for all results (future functionality).

        Parameters:
        -----------
        spectrum : bool, default=True
            Whether to include spectrum analysis
        modes : bool, default=True
            Whether to include mode analysis
        parallel : bool, default=True
            Whether to use parallel processing
        **kwargs : dict
            Additional arguments for analysis

        Returns:
        --------
        Dict[str, Any]
            Comprehensive report summary
        """
        log.info(f"Preparing comprehensive report for {len(self.results)} results")
        log.info(f"Spectrum: {spectrum}, Modes: {modes}, Parallel: {parallel}")

        report = {
            "total_results": len(self.results),
            "spectrum_analysis": None,
            "mode_analysis": None,
            "errors": [],
        }

        try:
            if spectrum:
                log.info("Running spectrum analysis...")
                # This would include FFT spectrum analysis
                spectrum_summary = self.fft.compute_all(**kwargs)
                report["spectrum_analysis"] = spectrum_summary

            if modes:
                log.info("Running mode analysis...")
                # This would include mode computation and analysis
                mode_summary = self.fft.modes.compute_modes(parallel=parallel, **kwargs)
                report["mode_analysis"] = mode_summary

            log.info("Report preparation completed successfully")

        except Exception as e:
            log.error(f"Error during report preparation: {e}")
            report["errors"].append(str(e))

        return report

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary of all results in batch.

        Returns:
        --------
        Dict[str, Any]
            Summary information about the batch
        """
        paths = [result.path for result in self.results]

        # Collect attributes
        all_attributes = set()
        for result in self.results:
            if hasattr(result, "attributes") and isinstance(result.attributes, dict):
                all_attributes.update(result.attributes.keys())

        summary = {
            "count": len(self.results),
            "paths": paths,
            "common_attributes": list(all_attributes),
            "first_result": self.results[0].path if self.results else None,
            "last_result": self.results[-1].path if self.results else None,
        }

        return summary
