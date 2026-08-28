"""Batch spectrum computation orchestration."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ....cache import CacheKey
from ....cli.logging_config import get_mmpp_logger
from .result import BatchSpectrumResult, SpectrumEntry

if TYPE_CHECKING:
    from ..multi import MultiSpectrumResult

log = get_mmpp_logger("mmpp.fft.spectrum_batch")


def _batch_trace(
    frequencies: Any,
    spectrum: Any,
    power: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse one FFT result to deterministic frequency-aligned 1D traces."""
    freqs = np.asarray(frequencies, dtype=float)
    complex_spectrum = np.asarray(spectrum)
    spectral_power = np.asarray(power, dtype=float)
    if freqs.ndim != 1 or freqs.size == 0 or not np.isfinite(freqs).all():
        raise ValueError("Batch frequency axis must be a non-empty finite 1D array")
    if np.any(np.diff(freqs) <= 0):
        raise ValueError("Batch frequency axis must be strictly increasing")
    if complex_spectrum.ndim == 0 or complex_spectrum.shape[0] != freqs.size:
        raise ValueError("Spectrum first axis must match the frequency axis")
    if spectral_power.shape != complex_spectrum.shape:
        raise ValueError("Spectrum and spectral power must have identical shapes")
    if not np.isfinite(complex_spectrum).all() or not np.isfinite(spectral_power).all():
        raise ValueError("Batch spectrum and power must contain only finite values")
    if np.any(spectral_power < 0):
        raise ValueError("Batch spectral power must be non-negative")

    if complex_spectrum.ndim > 1:
        reduction_axes = tuple(range(1, complex_spectrum.ndim))
        complex_spectrum = np.mean(complex_spectrum, axis=reduction_axes)
        # Reduce power directly: abs(mean(F))**2 would introduce phase cancellation.
        spectral_power = np.mean(spectral_power, axis=reduction_axes)
    return freqs, np.asarray(complex_spectrum), np.asarray(spectral_power, dtype=float)


def _compatible_frequency_mask(frequency_axes: list[np.ndarray]) -> np.ndarray:
    """Compare all grids with the first grid in input order."""
    if not frequency_axes:
        return np.array([], dtype=bool)
    reference = np.asarray(frequency_axes[0])
    return np.asarray(
        [
            np.asarray(axis).shape == reference.shape
            and np.allclose(axis, reference, rtol=1e-10, atol=0.0)
            for axis in frequency_axes
        ],
        dtype=bool,
    )


class BatchSpectrum:
    """Batch spectrum computation handler."""

    def __init__(
        self,
        results: list[Any],
        mmpp_ref: Any,
        dataset_name: str | None = None,
        slice_info: Any | None = None,
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
        filter_type: list[str] | None = None,
        find_peaks: dict | None = None,
        **kwargs,
    ) -> MultiSpectrumResult:
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

        if not spectra:
            raise RuntimeError("All overlay spectrum computations failed")
        return MultiSpectrumResult(spectra)

    def compute_all(
        self,
        dataset_name: str | None = None,
        z_layer: int = -1,
        method: int = 1,
        slice_info: Any | None = None,
        filter_type: list[str] | None = None,
        window_function: str = "none",
        component_weights: tuple = (1, 0, 0),
        normalize: str = "none",
        engine: str = "auto",
        find_peaks: dict | None = None,
        tmin: int | None = None,
        tmax: int | None = None,
        fmin: float | None = None,
        fmax: float | None = None,
        parallel: bool = True,
        max_workers: int | None = None,
        use_cache: bool = True,
        save: bool = True,
        force: bool = False,
        extract_parameters: list[str] | None = None,
        save_batch: bool = True,
        batch_cache_dir: str | Path | None = None,
        **kwargs,
    ) -> BatchSpectrumResult:
        from ...core import FFT

        active_dataset = dataset_name or self.dataset_name
        active_slice = slice_info if slice_info is not None else self.slice_info
        if not self.results:
            raise ValueError("Batch spectrum requires at least one result")
        if not isinstance(parallel, (bool, np.bool_)):
            raise TypeError("parallel must be boolean")
        if max_workers is not None:
            if isinstance(max_workers, (bool, np.bool_)) or not isinstance(
                max_workers, (int, np.integer)
            ):
                raise TypeError("max_workers must be a positive integer or None")
            if max_workers <= 0:
                raise ValueError("max_workers must be a positive integer or None")
            max_workers = int(max_workers)

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
                expected_paths = [str(result.path) for result in self.results]
                if (
                    len(cached) == len(self.results)
                    and cached.job_paths == expected_paths
                ):
                    log.info("Loaded %s spectra from cache", len(cached))
                    return cached
                log.warning(
                    "Batch cache entries or job order differ from the request. Recomputing..."
                )
            except Exception as exc:
                log.warning("Failed to load cache: %s. Recomputing...", exc)

        log.info(
            "Starting batch spectrum computation for %s results", len(self.results)
        )

        computed_spectra = []
        computed_powers = []
        computed_frequency_axes = []
        parameters: dict[str, list[Any]] = {p: [] for p in extract_parameters}
        job_paths = []
        errors = []
        computation_times = []
        computed_indices: list[int] = []

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
                spectrum_result = fft_analyzer.spectrum(
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
                freqs = np.asarray(spectrum_result.frequencies)
                spectrum = np.asarray(spectrum_result.spectrum)
                power = np.asarray(spectrum_result.spectral_quantity)
                freqs, spectrum, power = _batch_trace(freqs, spectrum, power)

                extracted = {}
                for param in extract_parameters:
                    if hasattr(result, "attributes") and isinstance(
                        result.attributes, dict
                    ):
                        extracted[param] = result.attributes.get(param)
                    else:
                        extracted[param] = None

                return {
                    "success": True,
                    "index": i,
                    "frequencies": freqs,
                    "spectrum": spectrum,
                    "power": power,
                    "path": str(result.path),
                    "parameters": extracted,
                    "time": time.time() - start_time,
                }
            except Exception as exc:
                log.error("Failed for %s: %s", result.path, exc)
                return {
                    "success": False,
                    "index": i,
                    "path": str(result.path),
                    "error": str(exc),
                    "time": time.time() - start_time,
                }

        iterator: Any
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
                    from tqdm import tqdm

                    from mmpp.core.mmpp import _running_in_ipython_kernel

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
                        computed_frequency_axes.append(result_data["frequencies"])
                        computed_spectra.append(result_data["spectrum"])
                        computed_powers.append(result_data["power"])
                        job_paths.append(result_data["path"])
                        computed_indices.append(int(result_data["index"]))
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
                from tqdm import tqdm

                from mmpp.core.mmpp import _running_in_ipython_kernel

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
                    computed_frequency_axes.append(result_data["frequencies"])
                    computed_spectra.append(result_data["spectrum"])
                    computed_powers.append(result_data["power"])
                    job_paths.append(result_data["path"])
                    computed_indices.append(int(result_data["index"]))
                    for param, value in result_data["parameters"].items():
                        parameters[param].append(value)
                else:
                    errors.append(
                        {
                            "path": result_data["path"],
                            "error": result_data.get("error", "Unknown"),
                        }
                    )

        avg_time = np.mean(computation_times) if computation_times else 0
        total_time = sum(computation_times)

        parameters = {
            k: v for k, v in parameters.items() if any(val is not None for val in v)
        }

        if computed_indices:
            order = np.argsort(np.asarray(computed_indices), kind="stable")
            computed_spectra = [computed_spectra[int(i)] for i in order]
            computed_powers = [computed_powers[int(i)] for i in order]
            computed_frequency_axes = [computed_frequency_axes[int(i)] for i in order]
            job_paths = [job_paths[int(i)] for i in order]
            parameters = {
                key: [values[int(i)] for i in order]
                for key, values in parameters.items()
            }

        if not computed_spectra:
            raise RuntimeError(f"All {len(self.results)} spectrum computations failed.")

        # The first successful job in input order, not the fastest thread, owns
        # the canonical grid. This makes parallel and sequential results identical.
        computed_frequencies = np.asarray(computed_frequency_axes[0])
        compatible = _compatible_frequency_mask(computed_frequency_axes)
        for idx, is_compatible in enumerate(compatible):
            if not bool(is_compatible):
                errors.append(
                    {
                        "path": job_paths[idx],
                        "error": (
                            "Frequency grid differs from the first successful input "
                            "entry; use overlay() for heterogeneous grids or harmonize "
                            "dt/nfft."
                        ),
                    }
                )

        if not np.all(compatible):
            computed_spectra = [
                value
                for value, keep in zip(computed_spectra, compatible, strict=False)
                if keep
            ]
            computed_powers = [
                value
                for value, keep in zip(computed_powers, compatible, strict=False)
                if keep
            ]
            job_paths = [
                value
                for value, keep in zip(job_paths, compatible, strict=False)
                if keep
            ]
            parameters = {
                key: [
                    value
                    for value, keep in zip(values, compatible, strict=False)
                    if keep
                ]
                for key, values in parameters.items()
            }

        successful = len(computed_spectra)
        failed = len(errors)
        log.info("Batch spectrum: %s successful, %s failed", successful, failed)
        log.info("Total: %.2fs, Average: %.2fs per result", total_time, avg_time)
        if errors:
            log.warning("Errors in %s computations:", len(errors))
            for err in errors[:3]:
                log.warning("  %s: %s", err["path"], err["error"])

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
