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

    def __repr__(self) -> str:
        return f"BatchFFT({len(self.results)} results)"

    def _repr_html_(self) -> str:
        """Return rich HTML representation for Jupyter notebooks."""
        import uuid as _uuid

        n = len(self.results)
        uid = str(_uuid.uuid4())[:8]

        # ── header ──────────────────────────────────────────────
        html = (
            '<div style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;'
            'border:2px solid #334155;border-radius:12px;padding:18px;margin:10px 0;'
            'background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);'
            'color:#e2e8f0;box-shadow:0 10px 25px rgba(0,0,0,0.3),'
            '0 0 0 1px rgba(148,163,184,0.1) inset;">'
        )
        html += (
            '<h3 style="margin:0 0 12px 0;color:#f1f5f9;font-weight:600;'
            'letter-spacing:0.5px;text-shadow:0 2px 4px rgba(0,0,0,0.3);">'
            f"🔬 Batch FFT Interface &nbsp;"
            f'<span style="color:#60a5fa;font-weight:700;">{n}</span>'
            f'<span style="color:#cbd5e1;font-weight:400;font-size:0.9em;"> result{"s" if n != 1 else ""}</span>'
            "</h3>"
        )

        # ── info section ────────────────────────────────────────
        html += (
            '<div style="background:linear-gradient(135deg,rgba(51,65,85,0.4) 0%,'
            "rgba(30,41,59,0.4) 100%);padding:12px;border-radius:8px;margin-bottom:12px;"
            'border:1px solid rgba(148,163,184,0.15);backdrop-filter:blur(10px);">'
        )
        if n > 0:
            first_name = self.results[0].path.split("/")[-1]
            last_name = self.results[-1].path.split("/")[-1]
            html += (
                f'<span style="color:#94a3b8;">Results:</span> '
                f'<code style="background:rgba(15,23,42,0.6);padding:3px 8px;border-radius:4px;'
                f"font-family:'Courier New',monospace;font-size:0.9em;color:#cbd5e1;"
                f'border:1px solid rgba(71,85,105,0.3);">{first_name}</code>'
            )
            if n > 1:
                html += (
                    f' <span style="color:#64748b;">→</span> '
                    f'<code style="background:rgba(15,23,42,0.6);padding:3px 8px;border-radius:4px;'
                    f"font-family:'Courier New',monospace;font-size:0.9em;color:#cbd5e1;"
                    f'border:1px solid rgba(71,85,105,0.3);">{last_name}</code>'
                )
        else:
            html += '<span style="color:#fbbf24;">⚠️ Empty batch – no results.</span>'
        html += "</div>"

        # ── available operations ────────────────────────────────
        groups = [
            ("Spectrum", [
                ("job[:].fft.spectrum.compute_all(…)", "Compute spectra for all jobs → BatchSpectrumResult"),
                ("batch.plot.heatmap('B0')", "2D heatmap vs swept parameter"),
            ]),
            ("Modes", [
                ("job[:].fft.modes.compute_modes()", "Batch FMR mode detection"),
                ("job[:].fft.modes.analyze_all()", "Analyze peaks across all jobs"),
            ]),
            ("Transmission", [
                ("job[:].m[…].fft.transmission.compute_all(…)", "Batch transmission analysis"),
            ]),
            ("General", [
                ("job[:].fft.compute_all()", "Run raw FFT on all jobs"),
            ]),
        ]

        section_style = (
            "padding:5px 8px;font-weight:600;color:#f1f5f9;"
            "background:rgba(51,65,85,0.8);text-align:left;"
        )

        html += (
            '<div style="background:linear-gradient(135deg,rgba(51,65,85,0.4) 0%,'
            "rgba(30,41,59,0.4) 100%);padding:12px;border-radius:8px;margin-bottom:12px;"
            'border:1px solid rgba(148,163,184,0.15);backdrop-filter:blur(10px);">'
            '<b style="color:#94a3b8;">🔧 Available Operations:</b>'
            '<table style="width:100%;margin-top:8px;border-collapse:collapse;font-size:0.9em;">'
        )
        for group_name, methods in groups:
            html += (
                f'<tr><td colspan="2" style="{section_style}">{group_name}</td></tr>'
            )
            for code, desc in methods:
                html += (
                    '<tr style="border-bottom:1px solid rgba(71,85,105,0.3);">'
                    f'<td style="padding:5px 8px 5px 16px;">'
                    f'<code style="background:rgba(15,23,42,0.6);padding:3px 8px;border-radius:4px;'
                    f'color:#93c5fd;border:1px solid rgba(71,85,105,0.3);font-weight:500;">{code}</code></td>'
                    f'<td style="padding:5px 8px;color:#cbd5e1;">{desc}</td></tr>'
                )
        html += "</table></div>"

        # ── quick-start examples (collapsible) ──────────────────
        example_id = f"batch-fft-ex-{uid}"
        html += (
            f'<div style="margin-top:4px;">'
            f'<span onclick="var e=document.getElementById(\'{example_id}\');'
            f"e.style.display=e.style.display==='none'?'block':'none';\""
            f' style="cursor:pointer;color:#60a5fa;font-size:0.9em;">'
            f"▶ Quick-start examples</span>"
            f'<div id="{example_id}" style="display:none;margin-top:8px;">'
            f'<pre style="background:rgba(15,23,42,0.8);padding:10px;border-radius:6px;'
            f"font-family:'Courier New',monospace;font-size:0.85em;color:#10b981;"
            f'border:1px solid rgba(71,85,105,0.4);overflow-x:auto;">'
            "# Batch spectrum computation\n"
            "batch = job[:].fft.spectrum.compute_all(\n"
            '    extract_parameters=["B0", "d"],\n'
            "    fmin=5e9, fmax=25e9,\n"
            ")\n"
            'batch.plot.heatmap("B0")\n\n'
            "# Batch mode computation\n"
            "job[:].fft.modes.compute_modes()\n\n"
            "# Dataset-aware batch transmission\n"
            "job[:].m_layer13[:200,...,0].fft.transmission.compute_all()"
            "</pre></div></div>"
        )

        html += "</div>"
        return html


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
        # Compose with any previous slice to support chaining: m[0:100][::2]
        if self.slice_info is not None:
            if isinstance(self.slice_info, tuple):
                new.slice_info = self.slice_info + (key,) if not isinstance(key, tuple) else self.slice_info + key
            else:
                new.slice_info = (self.slice_info, key)
        else:
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

    def __init__(self, results: list[Any], mmpp_ref: Any, _filter_kwargs: dict | None = None):
        """
        Initialize batch operations.

        Parameters:
        -----------
        results : List[Any]
            List of ZarrJobResult objects to operate on
        mmpp_ref : Any
            Reference to parent MMPP instance
        _filter_kwargs : dict, optional
            Filter criteria used to produce this batch (from find())
        """
        self.results = results
        self.mmpp_ref = mmpp_ref
        self._filter_kwargs = _filter_kwargs or {}

        log.info(f"Initialized batch operations for {len(results)} results")

    @property
    def mpl(self):
        """Return a plotting helper for the filtered batch results."""
        from .plotting import MMPPlotter

        return MMPPlotter(self.results, self.mmpp_ref)

    @property
    def matplotlib(self):
        """Alias for :attr:`mpl`."""
        return self.mpl

    def __getitem__(self, index):
        """Return one result or a sliced batch."""
        if isinstance(index, slice):
            sliced_results = self.results[index]
            if self.mmpp_ref is not None:
                for res in sliced_results:
                    setter = getattr(res, "_set_mmpp_ref", None)
                    if callable(setter):
                        setter(self.mmpp_ref)
            return BatchOperations(sliced_results, self.mmpp_ref, _filter_kwargs=self._filter_kwargs)

        result = self.results[index]
        if self.mmpp_ref is not None:
            setter = getattr(result, "_set_mmpp_ref", None)
            if callable(setter):
                setter(self.mmpp_ref)
        return result

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

    @property
    def solitons(self):
        """Batch soliton analysis namespace."""
        from .solitons import BatchSolitonsInterface

        return BatchSolitonsInterface(self.results, self.mmpp_ref)

    @property
    def vortex(self):
        """Shortcut alias for ``self.solitons.vortex``."""
        return self.solitons.vortex
    
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

        try:
            plotter = self.mpl
        except ImportError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        try:
            return getattr(plotter, name)
        except AttributeError as exc:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'") from exc

        # Not a dataset — raise standard error
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __len__(self) -> int:
        """Return number of results in batch."""
        return len(self.results)

    def __repr__(self) -> str:
        """String representation of batch operations."""
        return f"BatchOperations({len(self.results)} results)"

    @staticmethod
    def _fmt_num(value: float) -> str:
        """Format a number with SI prefix for human readability.

        Examples: 3.89e10 → '38.9G', 1.13e6 → '1.13M', 0.005 → '5m', 42 → '42'
        """
        if value == 0:
            return "0"
        abs_val = abs(value)
        si = [
            (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"),
            (1, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"), (1e-12, "p"),
        ]
        for threshold, prefix in si:
            if abs_val >= threshold * 0.999:
                scaled = value / threshold
                # pick precision: integer-like → no decimals
                if abs(scaled - round(scaled)) < 1e-9:
                    return f"{int(round(scaled))}{prefix}"
                return f"{scaled:.4g}{prefix}"
        return f"{value:.4g}"

    def _repr_html_(self) -> str:
        """Return rich HTML representation for Jupyter notebooks."""
        import uuid as _uuid
        import html as _html

        n = len(self.results)
        uid = str(_uuid.uuid4())[:8]
        _fmt = self._fmt_num  # local shortcut

        # ── styles ──────────────────────────────────────────────
        _card = (
            'background:linear-gradient(135deg,rgba(51,65,85,0.4) 0%,'
            'rgba(30,41,59,0.4) 100%);padding:12px;border-radius:8px;'
            'margin-bottom:12px;border:1px solid rgba(148,163,184,0.15);'
        )
        _code = (
            "background:rgba(15,23,42,0.6);padding:3px 8px;border-radius:4px;"
            "color:#60a5fa;border:1px solid rgba(71,85,105,0.3);font-weight:500;"
            "font-family:'Courier New',monospace;font-size:0.88em;"
        )
        _bdr = 'border-bottom:1px solid rgba(71,85,105,0.3);'

        # ── outer container ─────────────────────────────────────
        html = (
            '<div style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;'
            'border:2px solid #334155;border-radius:12px;padding:18px;margin:10px 0;'
            'background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);'
            'color:#e2e8f0;box-shadow:0 10px 25px rgba(0,0,0,0.3),'
            '0 0 0 1px rgba(148,163,184,0.1) inset;">'
        )

        # ── header ──────────────────────────────────────────────
        html += (
            '<h3 style="margin:0 0 12px 0;color:#f1f5f9;font-weight:600;'
            'letter-spacing:0.5px;text-shadow:0 2px 4px rgba(0,0,0,0.3);">'
            f'📦 Batch Operations &nbsp;'
            f'<span style="color:#60a5fa;font-weight:700;">{n}</span>'
            f'<span style="color:#cbd5e1;font-weight:400;font-size:0.9em;"> result{"s" if n != 1 else ""}</span>'
            '</h3>'
        )

        if n == 0:
            html += f'<div style="{_card}"><span style="color:#fbbf24;">⚠️ Empty batch – no results.</span></div></div>'
            return html

        # ── 1. applied filter criteria (pill badges) ────────────
        if self._filter_kwargs:
            html += f'<div style="{_card}">'
            html += '<b style="color:#94a3b8;">🔍 Applied Filters:</b>'
            html += '<div style="margin-top:6px;display:flex;flex-wrap:wrap;gap:6px;">'
            for k, v in self._filter_kwargs.items():
                fmt_v = _fmt(v) if isinstance(v, (int, float)) else str(v)
                html += (
                    f'<span style="background:rgba(96,165,250,0.15);border:1px solid rgba(96,165,250,0.3);'
                    f'border-radius:20px;padding:3px 12px;font-size:0.85em;">'
                    f'<b style="color:#93c5fd;">{_html.escape(str(k))}</b>'
                    f'<span style="color:#cbd5e1;"> = </span>'
                    f'<span style="color:#10b981;font-family:monospace;">{_html.escape(fmt_v)}</span></span>'
                )
            html += '</div></div>'

        # ── detect varying parameters ───────────────────────────
        try:
            from .core.metadata_diff import find_differing_parameters, extract_job_metadata
            diff = find_differing_parameters(self.results)
            varying = diff.differing_params
            all_meta = [extract_job_metadata(r) for r in self.results]
        except Exception:
            varying = {}
            all_meta = [{} for _ in self.results]

        # columns for the table — only params that truly vary, cap at 8
        varying_keys = list(varying.keys())[:8]

        # ── 2. varying parameters summary ───────────────────────
        if varying:
            html += f'<div style="{_card}">'
            html += f'<b style="color:#94a3b8;">📊 Varying Parameters ({len(varying)}):</b>'
            html += (
                '<table style="width:100%;margin-top:8px;border-collapse:collapse;font-size:0.88em;">'
                f'<tr style="{_bdr}">'
                '<th style="padding:5px 8px;text-align:left;color:#94a3b8;">Parameter</th>'
                '<th style="padding:5px 8px;text-align:center;color:#94a3b8;">Unique</th>'
                '<th style="padding:5px 8px;text-align:left;color:#94a3b8;">Range</th></tr>'
            )
            for pname in varying_keys:
                vals = varying[pname]
                nums = [v for v in vals if isinstance(v, (int, float)) and v is not None]
                if nums:
                    uv = sorted(set(nums))
                    n_uniq = len(uv)
                    rng = f"{_fmt(uv[0])} → {_fmt(uv[-1])}"
                else:
                    uv = sorted(set(str(v) for v in vals if v is not None))
                    n_uniq = len(uv)
                    rng = ", ".join(uv[:5]) + (" …" if len(uv) > 5 else "")

                html += (
                    f'<tr style="{_bdr}">'
                    f'<td style="padding:5px 8px;"><code style="{_code}">{_html.escape(pname)}</code></td>'
                    f'<td style="padding:5px 8px;text-align:center;color:#a5b4fc;font-weight:600;">{n_uniq}</td>'
                    f'<td style="padding:5px 8px;font-family:monospace;color:#cbd5e1;">{rng}</td></tr>'
                )
            html += '</table></div>'

        # ── 3. full results table (open by default) ─────────────
        html += f'<div style="{_card}">'
        html += f'<b style="color:#94a3b8;">📋 Results ({n}):</b>'
        html += (
            '<div style="margin-top:8px;max-height:400px;overflow:auto;'
            'border:1px solid rgba(71,85,105,0.3);border-radius:6px;">'
            '<table style="width:100%;border-collapse:collapse;font-size:0.84em;white-space:nowrap;">'
        )

        # header
        html += (
            '<thead style="position:sticky;top:0;z-index:1;">'
            f'<tr style="background:#1e293b;{_bdr}">'
            '<th style="padding:6px 8px;text-align:center;color:#94a3b8;">#</th>'
            '<th style="padding:6px 8px;text-align:left;color:#94a3b8;">Name</th>'
        )
        for vk in varying_keys:
            html += f'<th style="padding:6px 8px;text-align:center;color:#a5b4fc;">{_html.escape(vk)}</th>'
        html += (
            '<th style="padding:6px 8px;text-align:left;color:#94a3b8;">Path</th>'
            '</tr></thead><tbody>'
        )

        # rows
        for idx, result in enumerate(self.results):
            name = result.path.split("/")[-1]
            # show only relative-ish path: last 2-3 dirs + filename
            parts = result.path.rstrip("/").split("/")
            if len(parts) > 3:
                short_path = "/".join(parts[-3:])
            else:
                short_path = result.path

            bg = 'background:rgba(15,23,42,0.3);' if idx % 2 == 0 else ''
            html += f'<tr style="{_bdr}{bg}">'
            html += f'<td style="padding:4px 8px;text-align:center;color:#64748b;">{idx}</td>'
            html += f'<td style="padding:4px 8px;"><code style="color:#e2e8f0;">{_html.escape(name)}</code></td>'

            meta = all_meta[idx] if idx < len(all_meta) else {}
            for vk in varying_keys:
                val = meta.get(vk)
                if val is not None and isinstance(val, (int, float)):
                    cell = _fmt(val)
                elif val is not None:
                    cell = str(val)
                else:
                    cell = "–"
                html += f'<td style="padding:4px 8px;text-align:center;color:#cbd5e1;font-family:monospace;">{_html.escape(cell)}</td>'

            html += (
                f'<td style="padding:4px 8px;color:#64748b;font-size:0.85em;" '
                f'title="{_html.escape(result.path)}">{_html.escape(short_path)}</td>'
            )
            html += '</tr>'

        html += '</tbody></table></div></div>'

        # ── 4. available operations table ────────────────────────
        ops = [
            ("job[:].fft.spectrum", "Batch FFT spectrum computation"),
            ("job[:].fft.modes", "Batch FMR mode analysis"),
            ("job[:].fft.transmission", "Batch transmission analysis"),
            ("job[:].get.&lt;dset&gt;[:]", "Stacked numpy arrays [n_jobs, …]"),
            ("job[:].process(…)", "Full analysis pipeline"),
        ]

        html += f'<div style="{_card}">'
        html += '<b style="color:#94a3b8;">🔧 Available Operations:</b>'
        html += '<table style="width:100%;margin-top:8px;border-collapse:collapse;font-size:0.9em;">'
        for code, desc in ops:
            html += (
                f'<tr style="{_bdr}">'
                f'<td style="padding:6px 8px;">'
                f'<code style="{_code}">{code}</code></td>'
                f'<td style="padding:6px 8px;color:#cbd5e1;">{desc}</td></tr>'
            )
        html += '</table></div>'

        # ── 5. quick-start examples (collapsible) ───────────────
        example_id = f"batch-examples-{uid}"
        html += (
            f'<div style="margin-top:4px;">'
            f'<span onclick="var e=document.getElementById(\'{example_id}\');'
            f"e.style.display=e.style.display==='none'?'block':'none';\""
            f' style="cursor:pointer;color:#60a5fa;font-size:0.9em;">'
            f'▶ Quick-start examples</span>'
            f'<div id="{example_id}" style="display:none;margin-top:8px;">'
            f'<pre style="background:rgba(15,23,42,0.8);padding:10px;border-radius:6px;'
            f"font-family:'Courier New',monospace;font-size:0.85em;color:#10b981;"
            f'border:1px solid rgba(71,85,105,0.4);overflow-x:auto;">'
            '# Batch spectrum\n'
            'batch = job[:].fft.spectrum.compute_all(fmin=5e9, fmax=25e9)\n'
            'batch.plot.heatmap("B0")\n\n'
            '# Batch numpy access\n'
            'arr = job[:].get.m[:]          # shape: (n_jobs, t, z, y, x, c)\n\n'
            '# Full pipeline\n'
            'job[:].process(dset="m")'
            '</pre></div></div>'
        )

        html += '</div>'
        return html

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
