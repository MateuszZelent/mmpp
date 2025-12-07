"""
MMPP Cache Module - Unified caching for post-processing results.

This module provides:
- CacheKey: Consistent key generation for all cache types
- Serializers: Standardized slice/config serialization
- @cached_result: Decorator for automatic caching
- BatchCacheEntry: Zarr storage for batch results
"""

from .key import CacheKey
from .serializers import serialize_slice, serialize_config, serialize_for_json
from .decorators import cached_result
from .batch import BatchCacheEntry, get_batch_cache_path

__all__ = [
    "CacheKey",
    "serialize_slice", 
    "serialize_config",
    "serialize_for_json",
    "cached_result",
    "BatchCacheEntry",
    "get_batch_cache_path",
]
