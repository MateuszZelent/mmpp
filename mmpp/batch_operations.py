"""
Batch operations module for MMPP - enables serial computation across multiple simulation results.

This module provides the BatchOperations class that allows for executing FFT computations,
mode analysis, and other operations across entire directories of simulation results using
slice notation like `op[:].fft.modes.compute_modes()` (auto-selects optimal dataset).
"""

from typing import List, Any, Optional, Dict, Union
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

from .logging_config import get_mmpp_logger

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

    def __init__(self, results: List[Any], mmpp_ref: Any):
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

    def compute_all(self, **kwargs) -> Dict[str, Any]:
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

    def __init__(self, results: List[Any], mmpp_ref: Any):
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
    ) -> Dict[str, Any]:
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

    def analyze_all(self, **kwargs) -> Dict[str, Any]:
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


class BatchOperations:
    """
    Main batch operations class that provides access to batch FFT and mode operations.

    This class is returned when using slice notation on MMPP objects, e.g., `op[:]`.
    It provides access to batch operations like:
    - `op[:].fft.modes.compute_modes()` (auto-selects optimal dataset)
    - `op[:].fft.compute_all()`
    """

    def __init__(self, results: List[Any], mmpp_ref: Any):
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
    ) -> Dict[str, Any]:
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
    ) -> Dict[str, Any]:
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

    def get_summary(self) -> Dict[str, Any]:
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
