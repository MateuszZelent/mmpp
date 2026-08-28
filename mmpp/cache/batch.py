"""Batch cache storage utilities using zarr format."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .key import CacheKey
from .serializers import serialize_for_json

# Try to import zarr
try:
    import zarr

    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False

log = logging.getLogger("mmpp.cache.batch")


class BatchCacheEntry:
    """Container for batch cache data in zarr format.

    Stores batch results in a structured zarr group with:
    - Arrays for each job's results
    - Parameter values
    - Metadata (config, timestamps, etc.)
    """

    def __init__(
        self,
        results: list[Any],
        parameters: dict[str, list[Any]],
        job_paths: list[str],
        config: Any | None = None,
        frequencies: np.ndarray | None = None,
    ):
        self.results = results
        self.parameters = parameters
        self.job_paths = job_paths
        self.config = config
        self.frequencies = frequencies

    def save_to_zarr(
        self,
        zarr_path: str | Path,
        cache_key: CacheKey,
    ) -> None:
        """Save batch result to zarr.

        Parameters
        ----------
        zarr_path : str or Path
            Path to zarr store (can be parent directory of any job)
        cache_key : CacheKey
            Cache key for this batch
        """
        if not ZARR_AVAILABLE:
            raise ImportError("zarr is required for batch caching")

        zarr_path = Path(zarr_path)

        # Create .mmpp_batch_cache directory alongside first job
        cache_dir = zarr_path / ".mmpp_batch_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_file = cache_dir / f"{cache_key.to_entry_name()}.zarr"

        z = zarr.open(str(cache_file), mode="w")

        # Save job paths
        z.attrs["job_paths"] = self.job_paths
        z.attrs["n_jobs"] = len(self.job_paths)
        z.attrs["cache_key"] = cache_key.to_string()
        z.attrs["cached_at"] = datetime.now(timezone.utc).isoformat() + "Z"

        # Save config
        if self.config is not None:
            z.attrs["config_json"] = json.dumps(
                serialize_for_json(self.config),
                default=str,
            )

        # Save parameters
        params_group = z.create_group("parameters")
        for param_name, param_values in self.parameters.items():
            # Convert to numpy array if possible
            try:
                arr = np.array(param_values)
                params_group.create_dataset(param_name, data=arr)
            except Exception:
                # Store as JSON for non-numeric types
                params_group.attrs[param_name] = json.dumps(
                    serialize_for_json(param_values)
                )

        # Save frequencies if available
        if self.frequencies is not None:
            z.create_dataset("frequencies", data=self.frequencies)

        # Save results as stacked arrays
        results_group = z.create_group("results")

        # Determine what arrays to save based on first result
        if self.results:
            first = self.results[0]

            # Check for common result types
            if hasattr(first, "spectrum") and hasattr(first, "frequencies"):
                # FFTComputeResult-like
                spectra = np.stack([r.spectrum for r in self.results], axis=0)
                results_group.create_dataset("spectra", data=spectra)

                if self.frequencies is None:
                    z.create_dataset("frequencies", data=first.frequencies)

            elif hasattr(first, "transmission") and hasattr(first, "frequencies"):
                # TransmissionResult-like
                trans = np.stack([r.transmission for r in self.results], axis=0)
                results_group.create_dataset("transmission", data=trans)

                if hasattr(first, "power_map"):
                    power = np.stack([r.power_map for r in self.results], axis=0)
                    results_group.create_dataset("power_map", data=power)

        log.info(f"Saved batch cache to {cache_file}")

    @classmethod
    def load_from_zarr(
        cls,
        zarr_path: str | Path,
        cache_key: CacheKey,
    ) -> BatchCacheEntry | None:
        """Load batch result from zarr cache.

        Parameters
        ----------
        zarr_path : str or Path
            Path to parent directory containing .mmpp_batch_cache
        cache_key : CacheKey
            Cache key to look for

        Returns
        -------
        Optional[BatchCacheEntry]
            Loaded batch entry or None if not found
        """
        if not ZARR_AVAILABLE:
            return None

        zarr_path = Path(zarr_path)
        cache_dir = zarr_path / ".mmpp_batch_cache"
        cache_file = cache_dir / f"{cache_key.to_entry_name()}.zarr"

        if not cache_file.exists():
            return None

        try:
            z = zarr.open(str(cache_file), mode="r")

            job_paths = z.attrs.get("job_paths", [])

            # Load parameters
            parameters = {}
            if "parameters" in z:
                params_group = z["parameters"]
                for key in params_group.keys():
                    parameters[key] = np.array(params_group[key]).tolist()
                for key, value in params_group.attrs.items():
                    parameters[key] = json.loads(value)

            # Load frequencies
            frequencies = None
            if "frequencies" in z:
                frequencies = np.array(z["frequencies"])

            # Create entry (results will be reconstructed by caller)
            entry = cls(
                results=[],  # Caller reconstructs from stacked arrays
                parameters=parameters,
                job_paths=job_paths,
                frequencies=frequencies,
            )

            # Attach raw zarr group for caller to extract results
            entry._zarr_group = z  # type: ignore[attr-defined]

            log.info(f"Loaded batch cache from {cache_file}")
            return entry

        except Exception as e:
            log.debug(f"Failed to load batch cache: {e}")
            return None


def get_batch_cache_path(
    job_paths: list[str],
    cache_key: CacheKey,
) -> Path:
    """Determine cache file path for batch operation.

    Uses first job's parent directory for cache location.
    """
    if not job_paths:
        return Path(".mmpp_batch_cache") / f"{cache_key.to_entry_name()}.zarr"

    first_job = Path(job_paths[0])
    cache_dir = first_job.parent / ".mmpp_batch_cache"
    return cache_dir / f"{cache_key.to_entry_name()}.zarr"
