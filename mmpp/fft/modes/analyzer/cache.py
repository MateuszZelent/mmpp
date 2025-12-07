"""
Mode Cache Management

LRU cache for mode data to avoid recomputing from zarr.
"""

from typing import Dict, Optional, Tuple
import logging

log = logging.getLogger("mmpp.fft.modes")


class ModeCache:
    """LRU cache for mode data.
    
    Caches mode data by (frequency, z_layer) to avoid repeated
    zarr file access for the same mode.
    
    Parameters
    ----------
    maxsize : int
        Maximum number of cached modes (default: 128)
    
    Examples
    --------
    >>> cache = ModeCache(maxsize=64)
    >>> cache.put(9.5, 0, mode_data)
    >>> cached = cache.get(9.5, 0)
    """
    
    def __init__(self, maxsize: int = 128):
        """Initialize cache with maximum size."""
        self._cache: Dict[Tuple[float, int], any] = {}
        self._maxsize = maxsize
        self._access_order = []
        log.debug(f"Initialized mode cache with maxsize={maxsize}")
    
    def get(self, frequency: float, z_layer: int) -> Optional[any]:
        """Get cached mode data.
        
        Parameters
        ----------
        frequency : float
            Frequency in GHz
        z_layer : int
            Z-layer index
            
        Returns
        -------
        Optional[FMRModeData]
            Cached mode data or None if not found
        """
        key = (frequency, z_layer)
        
        if key in self._cache:
            # Update access order (move to end = most recent)
            self._access_order.remove(key)
            self._access_order.append(key)
            log.debug(f"Cache hit: f={frequency:.3f} GHz, z={z_layer}")
            return self._cache[key]
        
        log.debug(f"Cache miss: f={frequency:.3f} GHz, z={z_layer}")
        return None
    
    def put(self, frequency: float, z_layer: int, data: any):
        """Store mode data in cache with LRU eviction.
        
        Parameters
        ----------
        frequency : float
            Frequency in GHz
        z_layer : int
            Z-layer index
        data : FMRModeData
            Mode data to cache
        """
        key = (frequency, z_layer)
        
        # If already exists, update access order
        if key in self._cache:
            self._access_order.remove(key)
        else:
            # Evict least recently used if cache is full
            if len(self._cache) >= self._maxsize:
                lru_key = self._access_order.pop(0)
                del self._cache[lru_key]
                log.debug(f"Evicted LRU entry: {lru_key}")
        
        # Add new entry
        self._cache[key] = data
        self._access_order.append(key)
        log.debug(f"Cached: f={frequency:.3f} GHz, z={z_layer} (size={len(self._cache)})")
    
    def clear(self):
        """Clear all cached entries."""
        old_size = len(self._cache)
        self._cache.clear()
        self._access_order.clear()
        log.debug(f"Cache cleared: removed {old_size} entries")
    
    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._cache)
    
    def __contains__(self, key: Tuple[float, int]) -> bool:
        """Check if key is in cache."""
        return key in self._cache
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ModeCache(size={len(self)}/{self._maxsize})"
