"""Batch spectrum computation orchestration."""

from __future__ import annotations

import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ....cache import CacheKey
from ....cli.logging_config import get_mmpp_logger
from .result import BatchSpectrumResult, SpectrumEntry

log = get_mmpp_logger("mmpp.fft.spectrum_batch")


class BatchSpectrum:
    """Batch spectrum computation handler."""

    def __init__(
        self,
        results: List[Any],
        mmpp_ref: Any,
        dataset_name: Optional[str] = None,
        slice_info: Optional[Any] = None,
    ):
        self.results = results
        self.mmpp_ref = mmpp_ref
        self.dataset_name = dataset_name
        self.slice_info = slice_info

    def __call__(self, **kwargs) -> BatchSpectrumResult:
        return self.compute_all(**kwargs)

    def overlay(
        self,
        force: bool = False,
        filter_type: Optional[List[str]] = None,
        find_peaks: Optional[dict] = None,
        **kwargs,
    ) -> "MultiSpectrumResult":
        from .. import MultiSpectrumResult

        spectra = []

        for result in self.results:
            try:
                if self.dataset_name:
                    data_wrapper = result[self.dataset_name]
                    if self.slice_info is not None:
                        data_wrapper = data_wrapper[self.slice_info]
                    fft_obj = data_wrapper.fft
                else:
                    from ...core import FFT

                    fft_obj = FFT(result, self.mmpp_ref)

                spectrum_result = fft_obj._spectrum_impl(
                    force=force,
                    filter_type=filter_type,
                    find_peaks=find_peaks,
                    **kwargs,
                )
                spectrum_result._source_job = result
                spectra.append(spectrum_result)
            except Exception as exc:
                log.warning(
                    "Failed to compute spectrum for %s: %s",
                    getattr(result, "path", str(result)),
                    exc,
                )

        return MultiSpectrumResult(spectra)

    def compute_all(
        self,
        dataset_name: Optional[str] = None,
        z_layer: int = -1,
        method: int = 1,
        slice_info: Optional[Any] = None,
        filter_type: Optional[List[str]] = None,
        window_function: str = "none",
        component_weights: tuple = (1, 0, 0),
        normalize: str = "none",
        engine: str = "auto",
        find_peaks: Optional[dict] = None,
        tmin: Optional[int] = None,
        tmax: Optional[int] = None,
        fmin: Optional[float] = None,
        fmax: Optional[float] = None,
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
        from ...core import FFT

        active_dataset = dataset_name or self.dataset_name
        active_slice = slice_info if slice_info is not None else self.slice_info

        if extract_parameters is None:
            extract_parameters = [
                "B0",
                "Bext",
                "bex",
                "bias_field",
                "applied_field",
                "d",
                "p",
                "thickness",
                "period",
                "latticeconst",
                "phi",
                "theta",
                "angle",
            ]

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

        batch_key = CacheKey.for_batch(
            analysis_type="batch_spectrum",
            job_paths=[r.path for r in self.results],
            dataset_name=active_dataset or "m",
            config=config_for_cache,
            slice_info=active_slice,
            extract_parameters=extract_parameters,
        )

        if batch_cache_dir is None:
            if self.results:
                first_path = Path(self.results[0].path)
                batch_cache_dir = first_path.parent / ".mmpp_batch_cache"
            else:
                batch_cache_dir = Path(".mmpp_batch_cache")
        else:
            batch_cache_dir = Path(batch_cache_dir)

        batch_cache_file = batch_cache_dir / f"{batch_key.to_entry_name()}.pkl"

        if not force and save_batch and batch_cache_file.exists():
            try:
                log.info("Found cached batch result: %s", batch_cache_file)
                cached = BatchSpectrumResult.load(batch_cache_file)
                if len(cached) == len(self.results):
                    log.info("Loaded %s spectra from cache", len(cached))
                    return cached
                log.warning(
                    "Cache mismatch: %s cached vs %s expected. Recomputing...",
                    len(cached),
                    len(self.results),
                )
            except Exception as exc:
                log.warning("Failed to load cache: %s. Recomputing...", exc)

        log.info("Starting batch spectrum computation for %s results", len(self.results))

        computed_spectra = []
        computed_powers = []
        computed_frequencies = None
        parameters: Dict[str, List[Any]] = {p: [] for p in extract_parameters}
        job_paths = []
        errors = []
        computation_times = []

        def compute_single(result_info):
            i, result = result_info
            start_time = time.time()
            try:
                log.debug(
                    "Computing spectrum %s/%s: %s",
                    i + 1,
                    len(self.results),
                    result.path,
                )
                fft_analyzer = FFT(result, self.mmpp_ref)
                freqs, spectrum = fft_analyzer.spectrum(
                    dset=active_dataset,
                    z_layer=z_layer,
                    method=method,
                    slice_info=active_slice,
                    save=save,
                    force=force,
                    filter_type=filter_type,
                    window=window_function,
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
            except Exception as exc:
                log.error("Failed for %s: %s", result.path, exc)
                gc.collect()
                return {
                    "success": False,
                    "path": str(result.path),
                    "error": str(exc),
                    "time": time.time() - start_time,
                }
            finally:
                gc.collect()

        if parallel and len(self.results) > 1:
            if max_workers is None:
                max_workers = min(len(self.results), os.cpu_count() or 8)
            log.info("Using parallel execution with %s workers", max_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(compute_single, (i, r)): r
                    for i, r in enumerate(self.results)
                }
                try:
                    from mmpp.core.mmpp import _running_in_ipython_kernel
                    from tqdm import tqdm

                    if _running_in_ipython_kernel():
                        iterator = as_completed(futures)
                    else:
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
                        errors.append(
                            {
                                "path": result_data["path"],
                                "error": result_data.get("error", "Unknown"),
                            }
                        )
        else:
            log.info("Using sequential execution")
            try:
                from mmpp.core.mmpp import _running_in_ipython_kernel
                from tqdm import tqdm

                if _running_in_ipython_kernel():
                    iterator = enumerate(self.results)
                else:
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
                    errors.append(
                        {
                            "path": result_data["path"],
                            "error": result_data.get("error", "Unknown"),
                        }
                    )

        successful = len(computed_spectra)
        failed = len(errors)
        avg_time = np.mean(computation_times) if computation_times else 0
        total_time = sum(computation_times)
        log.info("Batch spectrum: %s successful, %s failed", successful, failed)
        log.info("Total: %.2fs, Average: %.2fs per result", total_time, avg_time)

        if errors:
            log.warning("Errors in %s computations:", len(errors))
            for err in errors[:3]:
                log.warning("  %s: %s", err["path"], err["error"])

        parameters = {
            k: v for k, v in parameters.items() if any(val is not None for val in v)
        }

        if not computed_spectra:
            raise RuntimeError(f"All {len(self.results)} spectrum computations failed.")

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

        if save_batch:
            try:
                batch_result.save(batch_cache_file)
                log.info("Saved batch result to %s", batch_cache_file)
            except Exception as exc:
                log.warning("Failed to save batch: %s", exc)

        return batch_result


__all__ = ["BatchSpectrum", "BatchSpectrumResult", "SpectrumEntry"]
