"""Serialization utilities for cache key generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any


def serialize_slice(slice_info: Any) -> str:
    """Serialize slice_info to hashable string.
    
    Handles slice objects, Ellipsis, tuples of slices, and nested structures.
    Returns a 16-character hex hash for consistent cache keys.
    
    Parameters
    ----------
    slice_info : Any
        Slice information (slice, Ellipsis, tuple, etc.)
        
    Returns
    -------
    str
        16-character hex hash of serialized slice
        
    Examples
    --------
    >>> serialize_slice(slice(0, 100))
    'a1b2c3d4e5f6g7h8'
    >>> serialize_slice((slice(None), ..., 0))
    'h8g7f6e5d4c3b2a1'
    """
    if slice_info is None:
        return "none"
    
    def _convert(obj: Any) -> Any:
        """Recursively convert slice objects to JSON-serializable form."""
        if obj is None:
            return None
        if obj is Ellipsis:
            return {"__ellipsis__": True}
        if isinstance(obj, slice):
            return {
                "__slice__": True,
                "start": _convert(obj.start),
                "stop": _convert(obj.stop),
                "step": _convert(obj.step),
            }
        if isinstance(obj, (list, tuple)):
            return [_convert(x) for x in obj]
        if isinstance(obj, (int, float, str, bool)):
            return obj
        # Fallback for other types
        return str(obj)
    
    converted = _convert(slice_info)
    json_str = json.dumps(converted, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def serialize_config(config: Any) -> str:
    """Serialize configuration object to hashable string.
    
    Handles dataclasses, dicts, and objects with __dict__.
    Removes non-serializable fields (callbacks, etc.).
    
    Parameters
    ----------
    config : Any
        Configuration object (dataclass, dict, or object)
        
    Returns
    -------
    str
        16-character hex hash of serialized config
    """
    if config is None:
        return "none"
    
    # Convert to dictionary
    if is_dataclass(config):
        config_dict = asdict(config)
    elif hasattr(config, "__dict__"):
        config_dict = dict(vars(config))
    elif isinstance(config, dict):
        config_dict = dict(config)
    else:
        # Fallback: hash string representation
        return hashlib.sha256(str(config).encode()).hexdigest()[:16]
    
    # Remove non-serializable fields
    exclude_fields = {
        "progress_callback",
        "callback",
        "_internal",
        "_cache",
    }
    
    clean_dict = {}
    for k, v in config_dict.items():
        if k in exclude_fields:
            continue
        if callable(v):
            continue
        clean_dict[k] = v
    
    # Serialize to JSON with sorted keys for consistency
    try:
        json_str = json.dumps(clean_dict, sort_keys=True, default=str)
    except Exception:
        json_str = str(clean_dict)
    
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def serialize_for_json(obj: Any) -> Any:
    """Recursively prepare object for JSON serialization.
    
    Used for storing metadata in zarr attributes.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8")
    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    if isinstance(obj, dict):
        return {key: serialize_for_json(value) for key, value in obj.items()}
    if obj is Ellipsis:
        return "..."
    if isinstance(obj, slice):
        return {
            "__slice__": True,
            "start": serialize_for_json(obj.start),
            "stop": serialize_for_json(obj.stop),
            "step": serialize_for_json(obj.step),
        }
    if is_dataclass(obj):
        return serialize_for_json(asdict(obj))
    return str(obj)
