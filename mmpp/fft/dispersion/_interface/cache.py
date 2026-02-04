"""
Cache management for dispersion analysis results.

Handles zarr-based caching, both in job files and external directories.
Provides methods for loading, saving, and context-based cache invalidation.
"""

from __future__ import annotations
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING, cast

import numpy as np
import zarr

if TYPE_CHECKING:
    from ..models import DispersionResult1D
    from ..interface import FFTDispersionInterface

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages cache operations for dispersion results.
    
    Supports two cache locations:
    1. Internal: Within job zarr file (<job>.zarr/fft/dispersion/<dataset>)
    2. External: Separate directory (/tmp/mmpp_cache_<hash>/fft/dispersion/<dataset>)
    
    Cache is keyed by computation context (config, filters, etc.).
    """
    
    def __init__(self, interface: FFTDispersionInterface):
        """
        Initialize cache manager.
        
        Parameters
        ----------
        interface : FFTDispersionInterface
            Parent interface for context access.
        """
        self.interface = interface
        self._cache_dir = interface._cache_dir
        
    def get_cache_hash(self) -> str:
        """
        Generate unique hash for external cache.
        
        Based on: job path + dataset + slice
        
        Returns
        -------
        str
            12-character hexadecimal hash
        """
        components = [
            str(self.interface.parent_fft.job_result.path),
            str(self.interface.dataset_name or "__global__"),
            str(self.interface.slice_info or "__full__"),
        ]
        combined = "|".join(components)
        return hashlib.md5(combined.encode()).hexdigest()[:12]
    
    def build_cache_context(
        self,
        config: dict[str, Any],
        filters: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Build cache context dictionary.
        
        Parameters
        ----------
        config : dict
            Computation configuration
        filters : dict
            Filter configuration
        extra : dict, optional
            Extra context fields
            
        Returns
        -------
        dict
            Complete cache context
        """
        context = {
            "config": config,
            "filters": filters,
        }
        if extra:
            context.update(extra)
        return context
    
    def context_signature(self, context: dict[str, Any]) -> tuple[str, str]:
        """
        Generate cache signature from context.
        
        Parameters
        ----------
        context : dict
            Cache context
            
        Returns
        -------
        signature : str
            JSON signature (pretty)
        context_hash : str
            8-character hex hash
        """
        # Serialize context to canonical JSON
        serialized = json.dumps(
            context,
            sort_keys=True,
            indent=2,
            default=self._serialize_for_json,
        )
        
        # Generate hash
        context_hash = hashlib.md5(serialized.encode()).hexdigest()[:8]
        
        return serialized, context_hash
    
    def load_cached_dispersion(
        self,
        context_hash: str,
    ) -> Optional[DispersionResult1D]:
        """
        Load dispersion result from cache.
        
        Parameters
        ----------
        context_hash : str
            Context hash for cache lookup
            
        Returns
        -------
        DispersionResult1D or None
            Cached result if found, None otherwise
        """
        group = self._get_dispersion_dataset_group(write=False)
        if group is None:
            return None
        
        # Look for cached result with this hash
        result_node = group.get(context_hash)
        if result_node is None or not hasattr(result_node, "get"):
            return None
        
        result_group = result_node
        
        # Load arrays
        k_axis = self._load_group_array(result_group, "k_axis")
        f_axis = self._load_group_array(result_group, "f_axis")
        S = self._load_group_array(result_group, "S")
        S_complex = self._load_group_array(result_group, "S_complex")
        
        if k_axis is None or f_axis is None or S is None:
            logger.warning("Incomplete cached dispersion data for %s", context_hash)
            return None
        
        # Load metadata
        attrs = dict(result_group.attrs)
        
        # Reconstruct DispersionResult1D
        from ..models import DispersionResult1D
        
        result = DispersionResult1D(
            k_axis=k_axis,
            f_axis=f_axis,
            S=S,
            S_complex=S_complex,
            axis=attrs.get("axis", "x"),
            dx=attrs.get("dx", 0.0),
            orth_axis=self._load_group_array(result_group, "orth_axis"),
            metadata=attrs.get("metadata", {}),
        )
        
        logger.info("Loaded dispersion from cache: %s", context_hash)
        return result
    
    def save_dispersion_result(
        self,
        result: DispersionResult1D,
        context_hash: str,
        context_signature: str,
    ) -> bool:
        """
        Save dispersion result to cache.
        
        Parameters
        ----------
        result : DispersionResult1D
            Result to cache
        context_hash : str
            Context hash for cache key
        context_signature : str
            Full context signature (for metadata)
            
        Returns
        -------
        bool
            True if saved successfully
        """
        group = self._get_dispersion_dataset_group(write=True)
        if group is None:
            return False
        
        # Create result group
        if context_hash in group:
            del group[context_hash]
        
        result_group = group.create_group(context_hash)
        
        # Save arrays
        self._create_dataset(result_group, "k_axis", result.k_axis)
        self._create_dataset(result_group, "f_axis", result.f_axis)
        self._create_dataset(result_group, "S", result.S)
        
        # Save S_complex if available
        if result.S_complex is not None:
            self._create_dataset(result_group, "S_complex", result.S_complex)
        
        if result.orth_axis is not None:
            self._create_dataset(result_group, "orth_axis", result.orth_axis)
        
        # Save metadata
        result_group.attrs["axis"] = result.axis
        result_group.attrs["dx"] = result.dx
        result_group.attrs["context_signature"] = context_signature
        result_group.attrs["saved_at"] = datetime.utcnow().isoformat()
        
        logger.info("Saved dispersion to cache: %s", context_hash)
        return True
    
    def _get_dispersion_dataset_group(self, write: bool = False) -> Optional[zarr.Group]:
        """
        Get zarr group for dispersion cache.
        
        Priority: external cache > job zarr
        
        Parameters
        ----------
        write : bool
            If True, create group if needed
            
        Returns
        -------
        zarr.Group or None
        """
        # Priority: external cache if configured
        if self._cache_dir is not None:
            return self._get_external_cache_group(write=write)
        
        # Fallback: job zarr
        return self._get_job_cache_group(write=write)
    
    def _get_external_cache_group(self, write: bool) -> Optional[zarr.Group]:
        """Get external cache group (<cache_dir>/mmpp_cache_<hash>/)."""
        if self._cache_dir is None:
            return None
        
        cache_base = Path(self._cache_dir)
        if not cache_base.exists():
            if not write:
                return None
            cache_base.mkdir(parents=True, exist_ok=True)
        
        cache_hash = self.get_cache_hash()
        cache_path = cache_base / f"mmpp_cache_{cache_hash}"
        
        mode = "a" if write else "r"
        try:
            root = zarr.open(str(cache_path), mode=mode)
        except (OSError, PermissionError, FileNotFoundError) as exc:
            if write:
                logger.warning("Cannot create external cache: %s", exc)
            return None
        
        # Navigate to /fft/dispersion/<dataset>/
        # ... (implementation details)
        
        # This is a skeleton - full implementation would create groups recursively
        return None  # Placeholder
    
    def _get_job_cache_group(self, write: bool) -> Optional[zarr.Group]:
        """Get job zarr cache group (<job>.zarr/fft/dispersion/<dataset>/)."""
        mode = "a" if write else "r"
        try:
            root = zarr.open(self.interface.parent_fft.job_result.path, mode=mode)
        except (OSError, PermissionError, FileNotFoundError) as exc:
            if write:
                raise
            logger.debug("Job cache not available: %s", exc)
            return None
        
        # Navigate to /fft/dispersion/<dataset>/
        # ... (implementation details)
        
        return None  # Placeholder
    
    def _load_group_array(self, group: Any, name: str) -> Optional[np.ndarray]:
        """Load numpy array from zarr group."""
        node = group.get(name)
        if node is None:
            return None
        
        try:
            arr = np.asarray(node, dtype=node.dtype)
            return arr
        except Exception as exc:
            logger.debug("Failed to load array %s: %s", name, exc)
            return None
    
    def _create_dataset(self, group: Any, name: str, data: Any) -> None:
        """Create dataset in zarr group with appropriate chunking."""
        arr = np.asarray(data)
        
        # Determine chunks
        if arr.ndim == 1:
            chunks = min(arr.shape[0], 1000)
        elif arr.ndim == 2:
            chunks = (min(arr.shape[0], 100), min(arr.shape[1], 100))
        else:
            chunks = True
        
        # Create dataset
        group.create_dataset(
            name,
            data=arr,
            chunks=chunks,
            compression="blosc",
            overwrite=True,
        )
    
    def _serialize_for_json(self, value: Any) -> Any:
        """Serialize values for JSON context."""
        if isinstance(value, np.ndarray):
            return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "__dict__"):
            return value.__dict__
        return str(value)
