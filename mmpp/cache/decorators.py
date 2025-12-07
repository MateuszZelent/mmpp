"""Caching decorators for post-processing functions."""

from __future__ import annotations

import functools
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union
import logging

import numpy as np

from .key import CacheKey
from .serializers import serialize_for_json

# Try to import zarr
try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False

log = logging.getLogger("mmpp.cache")

# Type variable for decorated function return type
T = TypeVar("T")


def cached_result(
    analysis_type: str,
    key_params: tuple[str, ...] = ("zarr_path", "dataset_name", "z_layer", "method"),
    config_param: str = "config",
    slice_param: str = "slice_info",
    zarr_path_param: str = "zarr_path",
    enabled: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for caching FFT/post-processing results in zarr.
    
    Automatically caches function results based on input parameters.
    Results are stored in the source zarr file under /fft/{analysis_type}/.
    
    Parameters
    ----------
    analysis_type : str
        Type of analysis for cache organization (e.g., "spectrum", "transmission")
    key_params : Tuple[str, ...]
        Parameter names to include in cache key
    config_param : str
        Name of configuration parameter
    slice_param : str
        Name of slice_info parameter
    zarr_path_param : str
        Name of parameter containing zarr file path
    enabled : bool
        Whether caching is enabled (default: True)
        
    Returns
    -------
    Callable
        Decorated function with automatic caching
        
    Examples
    --------
    >>> @cached_result("spectrum", key_params=("zarr_path", "dataset", "z_layer"))
    ... def compute_spectrum(zarr_path, dataset, z_layer, config=None):
    ...     # ... expensive computation
    ...     return result
    
    Notes
    -----
    The decorated function should return a result that can be stored
    as numpy arrays. The decorator automatically handles:
    - Cache key generation
    - Loading from cache if available
    - Saving to cache after computation
    - Metadata storage (timestamps, config, etc.)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if not enabled or not ZARR_AVAILABLE:
                return func(*args, **kwargs)
            
            # Extract parameters from args/kwargs
            all_params = _extract_params(func, args, kwargs)
            
            # Check for force recompute
            force = all_params.get("force", False)
            use_cache = all_params.get("use_cache", True)
            save = all_params.get("save", True)
            
            # Get zarr path
            zarr_path = all_params.get(zarr_path_param)
            if not zarr_path:
                # No path to store cache, just compute
                return func(*args, **kwargs)
            
            # Generate cache key
            cache_key = CacheKey.create(
                analysis_type=analysis_type,
                job_path=str(zarr_path),
                dataset_name=all_params.get("dataset_name", "m"),
                z_layer=all_params.get("z_layer", -1),
                method=all_params.get("method", 1),
                config=all_params.get(config_param),
                slice_info=all_params.get(slice_param),
            )
            
            # Try to load from cache
            if use_cache and not force:
                cached = _load_from_zarr_cache(zarr_path, cache_key)
                if cached is not None:
                    log.debug(f"Cache hit for {cache_key.to_string()}")
                    return cached
            
            # Compute result
            start_time = time.time()
            result = func(*args, **kwargs)
            computation_time = time.time() - start_time
            
            # Save to cache
            if save and not force:
                _save_to_zarr_cache(
                    zarr_path, 
                    cache_key, 
                    result, 
                    all_params,
                    computation_time,
                )
            
            return result
        
        return wrapper
    return decorator


def _extract_params(func: Callable, args: tuple, kwargs: dict) -> dict[str, Any]:
    """Extract all parameters from function call."""
    import inspect
    sig = inspect.signature(func)
    bound = sig.bind_partial(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def _load_from_zarr_cache(
    zarr_path: Union[str, Path], 
    cache_key: CacheKey,
) -> Optional[Any]:
    """Load result from zarr cache if available."""
    try:
        z = zarr.open(str(zarr_path), mode="r")
        
        cache_path = cache_key.to_zarr_path()
        if cache_path not in z:
            # Try legacy format (direct entry name)
            legacy_path = f"fft/{cache_key.to_entry_name()}"
            if legacy_path not in z:
                return None
            cache_path = legacy_path
        
        entry = z[cache_path]
        
        # Load arrays and reconstruct result
        result_data = {}
        for key in entry.keys():
            if not key.startswith("_"):
                result_data[key] = np.array(entry[key])
        
        # If there's a frequencies + spectrum, return as FFTComputeResult-like
        if "frequencies" in result_data and "spectrum" in result_data:
            from ..fft.compute_fft import FFTComputeResult, FFTComputeConfig
            
            # Load metadata
            metadata = {}
            if hasattr(entry, "attrs"):
                metadata = dict(entry.attrs)
            
            # Reconstruct config from metadata
            config_json = metadata.get("config_json", "{}")
            if isinstance(config_json, bytes):
                config_json = config_json.decode("utf-8")
            
            try:
                config_dict = json.loads(config_json)
                config = FFTComputeConfig(**config_dict)
            except Exception:
                config = FFTComputeConfig()
            
            return FFTComputeResult(
                frequencies=result_data["frequencies"],
                spectrum=result_data["spectrum"],
                metadata=metadata,
                config=config,
            )
        
        # Return raw data dict for other types
        return result_data
        
    except Exception as e:
        log.debug(f"Failed to load from cache: {e}")
        return None


def _save_to_zarr_cache(
    zarr_path: Union[str, Path],
    cache_key: CacheKey,
    result: Any,
    params: dict[str, Any],
    computation_time: float,
) -> None:
    """Save result to zarr cache."""
    try:
        z = zarr.open(str(zarr_path), mode="a")
        
        cache_path = cache_key.to_zarr_path()
        
        # Create nested groups if needed
        current = z
        path_parts = cache_path.split("/")
        for part in path_parts[:-1]:
            if part not in current:
                current = current.create_group(part)
            else:
                current = current[part]
        
        entry_name = path_parts[-1]
        
        # Remove existing entry if present
        if entry_name in current:
            del current[entry_name]
        
        entry = current.create_group(entry_name)
        
        # Save arrays from result
        if hasattr(result, "frequencies") and hasattr(result, "spectrum"):
            # FFTComputeResult-like object
            _create_dataset(entry, "frequencies", result.frequencies)
            _create_dataset(entry, "spectrum", result.spectrum)
        elif isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    _create_dataset(entry, key, value)
        
        # Save metadata
        entry.attrs["cache_key"] = cache_key.to_string()
        entry.attrs["analysis_type"] = cache_key.analysis_type
        entry.attrs["dataset_name"] = cache_key.dataset_name
        entry.attrs["z_layer"] = cache_key.z_layer
        entry.attrs["method"] = cache_key.method
        entry.attrs["cached_at"] = datetime.utcnow().isoformat() + "Z"
        entry.attrs["computation_time_s"] = computation_time
        
        # Save config
        config = params.get("config")
        if config is not None:
            entry.attrs["config_json"] = json.dumps(
                serialize_for_json(config), 
                default=str
            )
        
        # Save slice info
        slice_info = params.get("slice_info")
        if slice_info is not None:
            entry.attrs["slice_info"] = json.dumps(
                serialize_for_json(slice_info)
            )
        
        log.debug(f"Saved to cache: {cache_path}")
        
    except Exception as e:
        log.debug(f"Failed to save to cache: {e}")


def _create_dataset(group: Any, name: str, data: np.ndarray) -> None:
    """Create dataset with compatibility for different zarr versions."""
    data_array = np.asarray(data)
    
    # Try different zarr API patterns
    try:
        group.create_dataset(name, data=data_array, overwrite=True)
    except TypeError:
        try:
            group.create_dataset(name, data=data_array)
        except Exception:
            group.create(name, data=data_array)
