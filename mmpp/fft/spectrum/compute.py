"""Spectrum compute helpers used by :mod:`mmpp.fft.core`."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..compute_fft import FFTCompute, FFTComputeResult


def format_slice_identifier(slice_info: Optional[Any]) -> str:
    """Create a deterministic cache/save identifier from ``slice_info``."""
    if slice_info is None:
        return "slice=None"

    def format_item(item: Any) -> str:
        if isinstance(item, slice):
            return f"{item.start}:{item.stop}:{item.step}"
        if item is Ellipsis:
            return "..."
        if isinstance(item, tuple):
            return "(" + ",".join(format_item(sub) for sub in item) + ")"
        if isinstance(item, (int, np.integer)):
            return str(int(item))
        return repr(item)

    slice_tuple = slice_info if isinstance(slice_info, tuple) else (slice_info,)
    formatted = ",".join(format_item(part) for part in slice_tuple)
    return f"slice={formatted}"


def build_cache_key(
    dataset_name: str,
    z_layer: int,
    method: int,
    slice_identifier: Optional[str] = None,
    **kwargs,
) -> str:
    """Generate a stable cache key for FFT computations."""
    key_parts = [dataset_name, str(z_layer), str(method)]
    if slice_identifier:
        key_parts.append(slice_identifier)
    for key, value in sorted(kwargs.items()):
        key_parts.append(f"{key}={value}")
    return "|".join(key_parts)


def compute_fft_cached(
    *,
    compute_engine: FFTCompute,
    job_result: Any,
    cache: dict[str, FFTComputeResult],
    dataset_name: Optional[str] = None,
    z_layer: int = -1,
    method: int = 1,
    use_cache: bool = True,
    save: bool = False,
    force: bool = False,
    save_dataset_name: Optional[str] = None,
    slice_info: Optional[Any] = None,
    **kwargs,
) -> FFTComputeResult:
    """Compute FFT with cache-aware delegation to ``FFTCompute``."""
    if dataset_name is None:
        dataset_name = job_result.get_largest_m_dataset()
    if not isinstance(dataset_name, str):
        dataset_name = str(dataset_name)

    slice_identifier = format_slice_identifier(slice_info)
    cache_key = build_cache_key(
        dataset_name,
        z_layer,
        method,
        slice_identifier=slice_identifier,
        **kwargs,
    )

    if use_cache and not force and not save and cache_key in cache:
        return cache[cache_key]

    try:
        result = compute_engine.calculate_fft_data(
            job_result.path,
            dataset_name,
            z_layer,
            method,
            save=save,
            force=force,
            save_dataset_name=save_dataset_name,
            slice_info=slice_info,
            slice_identifier=(None if slice_identifier == "slice=None" else slice_identifier),
            **kwargs,
        )
    except OSError as exc:
        if "directory not empty" in str(exc).lower():
            print(
                "Warning: FFT directory already exists and is not empty. "
                "Use force=True to overwrite."
            )
        raise

    if use_cache and not force:
        cache[cache_key] = result

    return result
