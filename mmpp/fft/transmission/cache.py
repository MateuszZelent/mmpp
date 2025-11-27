"""Disk cache utilities for transmission analysis results."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import zarr

from .compute import TransmissionConfig, TransmissionResult
from ...cli.logging_config import get_mmpp_logger


log = get_mmpp_logger("mmpp.fft.transmission.cache")


class TransmissionCache:
    """Handles on-disk caching of transmission results using zarr."""

    def __init__(self, job_result: Any, dataset_name: Optional[str] = None):
        self.job_result = job_result
        self.dataset_name = dataset_name

    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for use as zarr key."""
        return "".join(c if c.isalnum() or c in "_-" else "_" for c in name)

    def _ensure_text(self, value: Any) -> Optional[str]:
        """Convert value to text if it's bytes or bytearray."""
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8")
        return value if isinstance(value, str) else None

    def _serialize_for_json(self, obj: Any) -> Any:
        """Recursively prepare object for JSON serialization."""
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, (bytes, bytearray)):
            return obj.decode("utf-8")
        if isinstance(obj, (list, tuple)):
            return [self._serialize_for_json(item) for item in obj]
        if isinstance(obj, dict):
            return {key: self._serialize_for_json(value) for key, value in obj.items()}
        if obj is Ellipsis:
            return "..."
        if isinstance(obj, slice):
            return {
                "__slice__": True,
                "start": self._serialize_for_json(obj.start),
                "stop": self._serialize_for_json(obj.stop),
                "step": self._serialize_for_json(obj.step),
            }
        return str(obj)

    def _load_group_array(self, group: Any, name: str) -> Optional[np.ndarray]:
        """Safely load array from zarr group."""
        try:
            node = group.get(name)
            if node is None:
                return None
            try:
                return np.array(node)
            except Exception:
                log.debug("Failed to load array '%s' from transmission cache", name)
                return None
        except Exception:
            return None

    def _create_dataset(self, group: Any, name: str, data: Any) -> None:
        """Create a dataset under ``group`` with compatibility for zarr API variants."""
        create = getattr(group, "create_dataset")

        # Convert data to numpy array to get shape and dtype
        data_array = np.asarray(data)

        # Standard parameters all zarr versions should accept
        base_kwargs = {
            "data": data_array,
            "shape": data_array.shape,
            "dtype": data_array.dtype,
        }

        # Try most common patterns with shape and dtype
        call_attempts = [
            # Standard zarr 2.x/3.x: name as positional, with shape/dtype
            lambda: create(name, **base_kwargs, overwrite=True),
            lambda: create(name, **base_kwargs),
            # Alternative: name as keyword with shape/dtype
            lambda: create(name=name, **base_kwargs, overwrite=True),
            lambda: create(name=name, **base_kwargs),
            # Zarr 3.x async style: path as keyword-only with shape/dtype
            lambda: create(path=name, **base_kwargs, overwrite=True),
            lambda: create(path=name, **base_kwargs),
            # Fallback without explicit shape/dtype (legacy zarr 2.x)
            lambda: create(name, data=data_array, overwrite=True),
            lambda: create(name, data=data_array),
        ]

        errors: list[Exception] = []

        for attempt in call_attempts:
            try:
                result = attempt()
                if inspect.isawaitable(result):
                    self._await_in_thread(result)
                return
            except TypeError as exc:
                errors.append(exc)
                continue

        if errors:
            raise errors[-1]

    def _await_in_thread(self, awaitable: Any) -> None:
        """Run an awaitable to completion even when an event loop is active."""
        if not inspect.isawaitable(awaitable):
            return

        async def _coro() -> None:
            await awaitable

        def runner() -> None:
            asyncio.run(_coro())

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join()

    def _get_cache_group(
        self,
        cache_path: Optional[Path] = None,
        write: bool = False,
    ) -> Optional[zarr.Group]:
        """Get or create the zarr group for transmission cache.

        Parameters
        ----------
        cache_path : Optional[Path]
            If provided, use this path as cache directory.
            If None, use zarr file directory (default behavior).
        write : bool
            Whether to open for writing (create if needed).

        Returns
        -------
        Optional[zarr.Group]
            The transmission cache group or None if unavailable.
        """
        mode = "a" if write else "r"

        # Determine cache location
        if cache_path is not None:
            # User specified custom cache directory
            cache_dir = Path(cache_path)
            cache_dir.mkdir(parents=True, exist_ok=True)
            zarr_cache_path = cache_dir / "transmission_cache.zarr"
            log.debug("Using custom cache path: %s (mode=%s)", zarr_cache_path, mode)
        else:
            # Default: use same directory as source zarr file
            zarr_cache_path = Path(self.job_result.path)
            log.debug("Using source zarr as cache: %s (mode=%s)", zarr_cache_path, mode)

        try:
            root = zarr.open(str(zarr_cache_path), mode=mode)
        except (OSError, PermissionError, FileNotFoundError) as exc:
            if write:
                raise
            log.debug("Transmission cache not available: %s", exc)
            return None

        if not hasattr(root, "get"):
            if write:
                raise TypeError("Expected Zarr group at cache path")
            log.debug("Transmission cache root is not a group; skipping")
            return None

        root_group = cast(Any, root)
        store_obj = getattr(root_group, "store", None)
        read_only = bool(getattr(store_obj, "read_only", False))
        if write and read_only:
            log.warning(
                "Transmission cache skipped: store is read-only (%s)",
                getattr(store_obj, "path", zarr_cache_path),
            )
            return None

        # Navigate to /fft/transmission/<dataset_name>
        fft_node = root_group.get("fft")
        if fft_node is None:
            if not write:
                return None
            fft_group = root_group.create_group("fft")
        elif hasattr(fft_node, "get"):
            fft_group = fft_node
        else:
            if write:
                raise TypeError("Expected Zarr group at /fft in cache")
            log.debug("Transmission cache /fft node is not a group; skipping")
            return None

        transmission_node = fft_group.get("transmission")
        if transmission_node is None:
            if not write:
                return None
            transmission_group = fft_group.create_group("transmission")
        elif hasattr(transmission_node, "get"):
            transmission_group = transmission_node
        else:
            if write:
                raise TypeError("Expected Zarr group at /fft/transmission in cache")
            log.debug(
                "Transmission cache /fft/transmission node is not a group; skipping"
            )
            return None

        dataset_key = self._sanitize_name(self.dataset_name or "__global__")
        dataset_node = transmission_group.get(dataset_key)
        if dataset_node is None:
            if not write:
                return None
            dataset_group = transmission_group.create_group(dataset_key)
        elif hasattr(dataset_node, "get"):
            dataset_group = dataset_node
        else:
            if write:
                raise TypeError("Expected Zarr group for cached dataset entry")
            log.debug("Transmission cache dataset node is not a group; skipping")
            return None

        return dataset_group

    def generate_cache_key(
        self, config: TransmissionConfig, slice_info: Any = None
    ) -> str:
        """Generate unique cache key from configuration and slice info."""
        from dataclasses import asdict

        config_dict = asdict(config)
        # Remove non-serializable fields (like progress_callback)
        config_dict.pop("progress_callback", None)

        # Add slice info to cache key
        if slice_info is not None:
            config_dict["slice_info"] = self._serialize_for_json(slice_info)

        config_json = json.dumps(config_dict, sort_keys=True)
        hash_obj = hashlib.sha256(config_json.encode())
        return hash_obj.hexdigest()[:16]

    def load_result(
        self,
        config: TransmissionConfig,
        slice_info: Any = None,
        cache_path: Optional[Path] = None,
    ) -> Optional[TransmissionResult]:
        """Load transmission result from cache if available.

        Parameters
        ----------
        config : TransmissionConfig
            Configuration used for computation.
        slice_info : Any
            Slice information if applicable.
        cache_path : Optional[Path]
            Custom cache directory (if None, uses zarr directory).

        Returns
        -------
        Optional[TransmissionResult]
            Cached result or None if not found.
        """
        log.debug(
            "Attempting to load transmission from cache (cache_path=%s)", cache_path
        )

        cache_group = self._get_cache_group(cache_path=cache_path, write=False)
        if cache_group is None:
            log.debug("Cache group not available - cannot load from cache")
            return None

        cache_key = self.generate_cache_key(config, slice_info)
        entry_name = f"transmission_{cache_key}"
        log.debug("Looking for cache entry: %s (key=%s)", entry_name, cache_key[:16])

        entry_node = cache_group.get(entry_name)
        if entry_node is None or not hasattr(entry_node, "get"):
            log.debug("Cache entry not found: %s", entry_name)
            # List available entries for debugging
            try:
                available = (
                    list(cache_group.keys()) if hasattr(cache_group, "keys") else []
                )
                if available:
                    log.debug(
                        "Available cache entries: %s", available[:5]
                    )  # Show first 5
                else:
                    log.debug("No cache entries found in group")
            except Exception:
                pass
            return None

        entry = cast(Any, entry_node)

        # Verify configuration matches
        stored_config_json = entry.attrs.get("config_json")
        if stored_config_json is None:
            log.debug("Cache entry %s missing config_json", entry_name)
            return None

        stored_config_json = self._ensure_text(stored_config_json)
        if stored_config_json is None:
            log.debug("Cache entry %s has invalid config_json", entry_name)
            return None

        try:
            stored_config = TransmissionConfig(**json.loads(stored_config_json))
            # Basic validation - check key parameters match
            if (
                stored_config.method != config.method
                or stored_config.dataset_name != config.dataset_name
                or stored_config.z_layer != config.z_layer
            ):
                log.debug("Cache config mismatch for %s", entry_name)
                return None
        except Exception as exc:
            log.warning("Failed to deserialize transmission config from cache: %s", exc)
            return None

        # Load arrays
        frequencies = self._load_group_array(entry, "frequencies")
        x_positions = self._load_group_array(entry, "x_positions")
        transmission = self._load_group_array(entry, "transmission")
        power_map = self._load_group_array(entry, "power_map")
        reference_power = self._load_group_array(entry, "reference_power")

        if any(
            arr is None
            for arr in [
                frequencies,
                x_positions,
                transmission,
                power_map,
                reference_power,
            ]
        ):
            log.warning("Cache entry %s missing required arrays", entry_name)
            return None

        # Type check to satisfy mypy
        assert frequencies is not None
        assert x_positions is not None
        assert transmission is not None
        assert power_map is not None
        assert reference_power is not None

        # Load optional arrays
        power_plus = self._load_group_array(entry, "power_plus")
        power_minus = self._load_group_array(entry, "power_minus")
        transverse_power = self._load_group_array(entry, "transverse_power")
        longitudinal_power = self._load_group_array(entry, "longitudinal_power")
        complex_spectra_summary = self._load_group_array(
            entry, "complex_spectra_summary"
        )

        # Load metadata
        metadata_json = entry.attrs.get("metadata_json", "{}")
        metadata_json = self._ensure_text(metadata_json) or "{}"
        try:
            metadata = json.loads(metadata_json)
        except Exception:
            metadata = {}

        log.info("Loaded transmission result from cache: %s", entry_name)

        return TransmissionResult(
            frequencies=frequencies,
            x_positions=x_positions,
            transmission=transmission,
            power_map=power_map,
            reference_power=reference_power,
            config=stored_config,
            metadata=metadata,
            power_plus=power_plus,
            power_minus=power_minus,
            transverse_power=transverse_power,
            longitudinal_power=longitudinal_power,
            complex_spectra_summary=complex_spectra_summary,
        )

    def save_result(
        self,
        result: TransmissionResult,
        slice_info: Any = None,
        cache_path: Optional[Path] = None,
        overwrite: bool = False,
    ) -> None:
        """Save transmission result to cache.

        Parameters
        ----------
        result : TransmissionResult
            Result to save.
        slice_info : Any
            Slice information if applicable.
        cache_path : Optional[Path]
            Custom cache directory (if None, uses zarr directory).
        overwrite : bool
            Whether to overwrite existing cache entry.
        """
        log.debug(
            "Attempting to save transmission to cache (cache_path=%s, overwrite=%s)",
            cache_path,
            overwrite,
        )

        cache_group = self._get_cache_group(cache_path=cache_path, write=True)
        if cache_group is None:
            log.debug("Skipping transmission cache save; cache group unavailable")
            return

        cache_key = self.generate_cache_key(result.config, slice_info)
        entry_name = f"transmission_{cache_key}"
        log.debug("Saving cache entry: %s (key=%s)", entry_name, cache_key[:16])

        if entry_name in cache_group:
            if not overwrite:
                log.info(
                    "Transmission cache %s already exists (use overwrite=True to replace)",
                    entry_name,
                )
                return
            log.debug("Overwriting existing cache entry: %s", entry_name)
            del cache_group[entry_name]

        try:
            entry = cache_group.create_group(entry_name)
        except ValueError as exc:
            message = str(exc).lower()
            if "read-only" in message or "read only" in message:
                log.warning("Transmission cache skipped: %s", exc)
                return
            raise

        # Save required arrays
        self._create_dataset(entry, "frequencies", result.frequencies)
        self._create_dataset(entry, "x_positions", result.x_positions)
        self._create_dataset(entry, "transmission", result.transmission)
        self._create_dataset(entry, "power_map", result.power_map)
        self._create_dataset(entry, "reference_power", result.reference_power)

        # Save optional arrays
        if result.power_plus is not None:
            self._create_dataset(entry, "power_plus", result.power_plus)
        if result.power_minus is not None:
            self._create_dataset(entry, "power_minus", result.power_minus)
        if result.transverse_power is not None:
            self._create_dataset(entry, "transverse_power", result.transverse_power)
        if result.longitudinal_power is not None:
            self._create_dataset(entry, "longitudinal_power", result.longitudinal_power)
        if result.complex_spectra_summary is not None:
            self._create_dataset(
                entry, "complex_spectra_summary", result.complex_spectra_summary
            )

        # Save configuration and metadata
        from dataclasses import asdict

        entry.attrs["config_json"] = json.dumps(asdict(result.config))
        entry.attrs["metadata_json"] = json.dumps(result.metadata)
        entry.attrs["dataset_name"] = self.dataset_name
        entry.attrs["slice_info"] = json.dumps(self._serialize_for_json(slice_info))
        entry.attrs["cached_at"] = datetime.utcnow().isoformat() + "Z"
        entry.attrs["job_name"] = getattr(self.job_result, "name", "")
        entry.attrs["zarr_path"] = str(self.job_result.path)

        store = getattr(cache_group, "store", None)
        store_desc = (
            getattr(store, "path", None)
            or getattr(store, "dir_path", None)
            or getattr(store, "filename", None)
        )
        log.info(
            "Transmission result saved: entry=%s store=%s",
            entry_name,
            (
                store_desc or store.__class__.__name__
                if store is not None
                else "<unknown>"
            ),
        )
