"""
FFT Dispersion Interface

Provides user-friendly interface for dispersion analysis integrated with MMPP job results.
Similar to FFTModeInterface but focused on spin-wave dispersion relations.
"""

import asyncio
import copy
import hashlib
import inspect
import json
import logging
import math
import re
import threading
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.colors import CenteredNorm, FuncNorm, LogNorm, Normalize, PowerNorm, SymLogNorm, TwoSlopeNorm

try:
    from scipy.signal import savgol_filter
    from scipy.stats import median_abs_deviation
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

from .core import SpinWaveAnalyzer, DispersionConfig
from .models import DispersionResult1D, DispersionResult2D, DispersionBranch

logger = logging.getLogger(__name__)


class FFTDispersionInterface:
    """
    Enhanced FFT interface with dispersion analysis capabilities.
    
    Provides elegant syntax like: job[0].fft.dispersion.plot_dispersion()
    or job[0].m_layer.fft.dispersion.compute_1d()
    """
    
    def __init__(
        self,
        parent_fft,
        dataset_name: Optional[str] = None,
        slice_info: Optional[Any] = None,
    ):
        """Initialize dispersion interface for FFT result."""
        self.parent_fft = parent_fft
        self.dataset_name = dataset_name
        self.slice_info = slice_info
        self._analyzer = None
        self._config = None
        self._tmax: Optional[int] = 100
        self._memory_cache: dict[str, DispersionResult1D] = {}
        self._filters_config: Optional[dict[str, bool]] = None
        self._last_plot_result: Optional[DispersionResult1D] = None

    def clone_for_dataset(
        self,
        dataset_name: Optional[str],
        slice_info: Optional[Any] = None,
    ) -> "FFTDispersionInterface":
        """Return a dataset-aware clone sharing analyzer/config state."""
        clone = FFTDispersionInterface(
            self.parent_fft,
            dataset_name=dataset_name,
            slice_info=slice_info,
        )
        clone._config = self._config
        clone._tmax = self._tmax
        clone._memory_cache = self._memory_cache
        clone._filters_config = copy.deepcopy(self._filters_config)
        if (self.dataset_name == dataset_name) and (self.slice_info == slice_info):
            clone._analyzer = self._analyzer
        else:
            clone._analyzer = None
        return clone
    
    @property
    def analyzer(self) -> SpinWaveAnalyzer:
        """Get or create SpinWaveAnalyzer instance."""
        if self._analyzer is None:
            zarr_path = self.parent_fft.job_result.path
            config = self._config or DispersionConfig()
            effective_tmax = self._determine_tmax(default=100)
            self._analyzer = SpinWaveAnalyzer(
                zarr_path,
                config=config,
                tmax=effective_tmax,
                slice_info=self.slice_info,
                dataset_name=self.dataset_name,
            )
        return self._analyzer

    @property
    def last_plot_result(self) -> Optional[DispersionResult1D]:
        """
        Get the result from the most recent plot_dispersion call.
        
        Returns
        -------
        DispersionResult1D or None
            The dispersion result from the last plot, or None if no plot has been made yet.
        """
        return self._last_plot_result

    def _determine_tmax(self, default: int = 100) -> Optional[int]:
        """
        Determine number of time steps to load based on config and slicing.
        
        Priority order:
        1. Explicit slice from user (e.g., [:1000,...,2]) - ALWAYS respected
        2. Configured tmax via .configure(tmax=X)
        3. Default tmax=100 (only if no slice and no config)
        
        Returns
        -------
        int or None
            Number of timesteps, or None to use ALL available timesteps
        """
        # Check if user provided explicit time slice
        slice_length = self._infer_time_length_from_slice()
        
        if slice_length is not None:
            # User explicitly specified number of timesteps (e.g., [:1000])
            logger.debug("Using EXPLICIT time slice from user: %d timesteps", slice_length)
            return slice_length
        
        # slice_length is None - could be two cases:
        # A) User used [:] (slice with no stop) → wants ALL timesteps → return None
        # B) No slice at all (slice_info is None) → wants default optimization → use tmax
        
        if self.slice_info is not None:
            # Case A: User DID provide a slice, but it's [:] (no stop)
            # This means "use ALL available timesteps"
            logger.debug("User provided [:] slice - using ALL available timesteps (no tmax limit)")
            return None  # None means "don't limit timesteps"
        
        # Case B: No slice at all - use configured tmax or default
        if self._tmax is not None:
            logger.debug("No user slice - using configured tmax: %d timesteps", self._tmax)
            return int(self._tmax)
        
        # No slice, no config - use default for optimization
        logger.debug("No slice or config - using default tmax: %d timesteps", default)
        return default

    def _infer_time_length_from_slice(self) -> Optional[int]:
        """
        Infer desired time window length from dataset slice info.
        
        Extracts the time dimension specification from user's slice.
        For 5D data (t,z,y,x,c): data[:1000,...,2] → returns 1000
        
        Returns
        -------
        Optional[int]
            - None if no slice info, or slice is [:] (meaning "all timesteps")
            - Positive int if explicit time range specified (e.g., [:1000] → 1000)
        
        Examples
        --------
        [:1000, ..., 2]  → 1000 (explicit: use 1000 timesteps)
        [:, ..., 2]      → None (implicit: use all available)
        [100:200, ...]   → 100 (explicit: use 100 timesteps)
        """
        if self.slice_info is None:
            return None

        candidate = self.slice_info
        if isinstance(candidate, tuple) and candidate:
            # For 5D data: (t_slice, z_slice, y_slice, x_slice, c_index)
            # First element is the time dimension
            for item in candidate:
                if item is Ellipsis:
                    continue
                candidate = item
                break

        if isinstance(candidate, slice):
            start = 0 if candidate.start is None else candidate.start
            stop = candidate.stop
            
            # If stop is None → [:] or [start:] → user wants ALL timesteps
            if stop is None:
                logger.debug("Time slice has no stop (e.g., [:] or [%s:]) - will use all available timesteps", start)
                return None
            
            # If stop is specified → [:1000] → user wants EXACTLY that many
            step = 1 if candidate.step is None else candidate.step
            if step == 0:
                return None
            length = math.ceil((stop - start) / step)
            logger.debug("Explicit time slice detected: [%s:%s:%s] → %d timesteps", start, stop, step, length)
            return max(0, length)

        return None

    # ------------------------------------------------------------------
    # Caching helpers
    # ------------------------------------------------------------------

    def _memory_key(self, mode: str, context_hash: str) -> str:
        dataset_key = self.dataset_name or "__global__"
        return f"{mode}:{dataset_key}:{context_hash}"

    def _build_cache_context(
        self,
        mode: str,
        axis: Optional[str],
        component: Optional[str],
        extra_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        config_obj = self._config or DispersionConfig()
        context = {
            "mode": mode,
            "axis": axis,
            "component": component,
            "dataset": self.dataset_name,
            "slice": self._serialize_for_json(self.slice_info),
            "config": self._serialize_for_json(asdict(config_obj)),
            "kwargs": self._serialize_for_json(extra_kwargs),
            "job_name": getattr(self.parent_fft.job_result, "name", None),
            "zarr_path": str(self.parent_fft.job_result.path),
            "tmax": self._determine_tmax(default=100),
        }
        return context

    def _context_signature(self, context: dict[str, Any]) -> tuple[str, str]:
        normalized = self._serialize_for_json(context)
        context_json = json.dumps(normalized, sort_keys=True)
        context_hash = hashlib.sha1(context_json.encode("utf-8")).hexdigest()
        return context_json, context_hash

    def _serialize_for_json(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (np.generic,)):
            return value.item()
        if isinstance(value, slice):
            return {
                "type": "slice",
                "start": self._serialize_for_json(value.start),
                "stop": self._serialize_for_json(value.stop),
                "step": self._serialize_for_json(value.step),
            }
        if value is Ellipsis:
            return {"type": "ellipsis"}
        if isinstance(value, dict):
            return {str(k): self._serialize_for_json(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._serialize_for_json(v) for v in value]
        if isinstance(value, np.ndarray):
            return {
                "type": "ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        try:
            return self._serialize_for_json(asdict(value))  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001 - fallback to string
            return repr(value)

    def _sanitize_name(self, value: str) -> str:
        return re.sub(r"[^0-9A-Za-z_.-]+", "_", value)

    def _ensure_text(self, value: Any) -> Optional[str]:
        if isinstance(value, str):
            return value
        if isinstance(value, (bytes, bytearray)):
            try:
                return value.decode("utf-8")
            except Exception:  # noqa: BLE001
                return None
        return None

    def _ensure_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _load_group_array(self, group: Any, name: str) -> Optional[np.ndarray]:
        try:
            node = group.get(name)
        except AttributeError:
            return None
        if node is None:
            return None
        try:
            return np.array(node[...] )
        except Exception:  # noqa: BLE001
            try:
                return np.array(node)
            except Exception:  # noqa: BLE001
                logger.debug("Failed to load array '%s' from dispersion cache", name)
                return None

    def _create_dataset(self, group: Any, name: str, data: Any) -> None:
        """Create a dataset under ``group`` with compatibility for zarr API variants."""

        create = getattr(group, "create_dataset")
        
        # Convert data to numpy array to get shape and dtype
        import numpy as np
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

    def _get_dispersion_dataset_group(self, write: bool = False) -> Optional[zarr.Group]:
        mode = "a" if write else "r"
        try:
            root = zarr.open(self.parent_fft.job_result.path, mode=mode)
        except (OSError, PermissionError, FileNotFoundError) as exc:
            if write:
                raise
            logger.debug("Dispersion cache not available: %s", exc)
            return None

        if not hasattr(root, "get"):
            if write:
                raise TypeError("Expected Zarr group at job result path")
            logger.debug("Dispersion cache root is not a group; skipping")
            return None

        root_group = cast(Any, root)
        store_obj = getattr(root_group, "store", None)
        read_only = bool(getattr(store_obj, "read_only", False))
        if write and read_only:
            logger.warning("Dispersion cache skipped: store is read-only (%s)", getattr(store_obj, "path", self.parent_fft.job_result.path))
            return None

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
            logger.debug("Dispersion cache /fft node is not a group; skipping")
            return None

        dispersion_node = fft_group.get("dispersion")
        if dispersion_node is None:
            if not write:
                return None
            dispersion_group = fft_group.create_group("dispersion")
        elif hasattr(dispersion_node, "get"):
            dispersion_group = dispersion_node
        else:
            if write:
                raise TypeError("Expected Zarr group at /fft/dispersion in cache")
            logger.debug("Dispersion cache /fft/dispersion node is not a group; skipping")
            return None

        dataset_key = self._sanitize_name(self.dataset_name or "__global__")
        dataset_node = dispersion_group.get(dataset_key)
        if dataset_node is None:
            if not write:
                return None
            dataset_group = dispersion_group.create_group(dataset_key)
        elif hasattr(dataset_node, "get"):
            dataset_group = dataset_node
        else:
            if write:
                raise TypeError("Expected Zarr group for cached dataset entry")
            logger.debug("Dispersion cache dataset node is not a group; skipping")
            return None

        return dataset_group

    def _load_cached_dispersion_result(
        self,
        dataset_group: Optional[zarr.Group],
        entry_name: str,
        context_hash: str,
    ) -> Optional[DispersionResult1D]:
        if dataset_group is None:
            return None

        entry_node = dataset_group.get(entry_name)
        if entry_node is None or not hasattr(entry_node, "get"):
            return None

        entry = cast(Any, entry_node)
        stored_hash = entry.attrs.get("context_hash")
        if not isinstance(stored_hash, str) or stored_hash != context_hash:
            logger.debug(
                "Dispersion cache mismatch for %s (stored=%s, expected=%s)",
                entry_name,
                stored_hash,
                context_hash,
            )
            return None

        try:
            config_json_raw = entry.attrs.get("config_json")
            config_json = self._ensure_text(config_json_raw)
            config = (
                DispersionConfig(**json.loads(config_json))
                if config_json is not None
                else DispersionConfig()
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to deserialize dispersion config from cache: %s", exc)
            config = DispersionConfig()

        notes_json_raw = entry.attrs.get("notes_json", "[]")
        notes_json = self._ensure_text(notes_json_raw) or "[]"
        try:
            notes = json.loads(notes_json)
        except Exception as exc:  # pragma: no cover - robust parsing fallback
            logger.warning(
                "Failed to parse notes_json for %s: %s — falling back to empty list",
                entry_name,
                exc,
            )
            # If JSONDecodeError has .pos attribute, log a small snippet to help debugging
            pos = getattr(exc, 'pos', None)
            try:
                if isinstance(pos, int):
                    start = max(0, pos - 80)
                    end = pos + 80
                    logger.debug("notes_json snippet around error: %s", notes_json[start:end])
            except Exception:
                # Best-effort only
                pass
            notes = []

        fold_period = entry.attrs.get("fold_period")
        if isinstance(fold_period, float) and math.isnan(fold_period):
            fold_period = None

        axis = self._ensure_text(entry.attrs.get("axis")) or "x"
        component = self._ensure_text(entry.attrs.get("component")) or "perp"
        orth_axis_label = self._ensure_text(entry.attrs.get("orth_axis_label"))

        S = self._load_group_array(entry, "S")
        k_axis = self._load_group_array(entry, "k_axis")
        f_axis = self._load_group_array(entry, "f_axis")
        if S is None or k_axis is None or f_axis is None:
            logger.debug("Dispersion cache entry %s missing required arrays", entry_name)
            return None

        flipx_attr = entry.attrs.get("flipx")
        if isinstance(flipx_attr, (bool, np.bool_)):
            flipx_flag = bool(flipx_attr)
        else:
            flipx_flag = False if flipx_attr is None else bool(flipx_attr)

        result = DispersionResult1D(
            S=S,
            k_axis=k_axis,
            f_axis=f_axis,
            axis=axis,
            component=component,
            config=config,
            S_folded=self._load_group_array(entry, "S_folded"),
            k_folded=self._load_group_array(entry, "k_folded"),
            fold_period=float(fold_period) if fold_period is not None else None,
            S_local=self._load_group_array(entry, "S_local"),
            orth_axis=self._load_group_array(entry, "orth_axis"),
            orth_axis_label=orth_axis_label,
            dt=self._ensure_float(entry.attrs.get("dt")) or 0.0,
            dx=self._ensure_float(entry.attrs.get("dx")) or 0.0,
            flipx=flipx_flag,
            notes=notes,
        )
        return result

    def _trim_dispersion_kmax(self, result: DispersionResult1D, kmax: Any) -> DispersionResult1D:
        try:
            limit = float(kmax)
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid kmax value: %s", kmax)
            return result

        if limit <= 0:
            logger.debug("Non-positive kmax=%s ignored", limit)
            return result

        mask = np.abs(result.k_axis) <= limit
        if not np.any(mask):
            logger.warning("No k-values within ±%g rad/m; skipping kmax trim", limit)
            return result

        S_trim = result.S[mask, :]
        k_trim = result.k_axis[mask]

        S_local_trim = None
        if result.S_local is not None:
            S_local_trim = result.S_local[:, mask, :]

        S_folded_trim = result.S_folded
        k_folded_trim = result.k_folded
        if result.k_folded is not None and result.S_folded is not None:
            mask_folded = np.abs(result.k_folded) <= limit
            if np.any(mask_folded):
                S_folded_trim = result.S_folded[mask_folded, :]
                k_folded_trim = result.k_folded[mask_folded]
            else:
                S_folded_trim = None
                k_folded_trim = None

        notes = list(result.notes or [])
        notes.append(f"Trimmed to |k| ≤ {limit:.3g} rad/m")

        trimmed = DispersionResult1D(
            S=S_trim,
            k_axis=k_trim,
            f_axis=result.f_axis,
            axis=result.axis,
            component=result.component,
            config=result.config,
            S_folded=S_folded_trim,
            k_folded=k_folded_trim,
            fold_period=result.fold_period,
            S_local=S_local_trim,
            orth_axis=result.orth_axis,
            orth_axis_label=result.orth_axis_label,
            dt=result.dt,
            dx=result.dx,
            notes=notes,
        )

        return trimmed

    def _save_dispersion_result(
        self,
        dataset_group: Optional[zarr.Group],
        entry_name: str,
        result: DispersionResult1D,
        context_json: str,
        context_hash: str,
        overwrite: bool,
    ) -> None:
        if dataset_group is None:
            logger.debug("Skipping dispersion cache save; dataset group unavailable")
            return

        if entry_name in dataset_group:
            if not overwrite:
                existing = dataset_group[entry_name]
                stored_hash = existing.attrs.get("context_hash")
                if stored_hash == context_hash:
                    logger.info("Dispersion cache %s already up to date", entry_name)
                else:
                    logger.warning(
                        "Dispersion cache %s exists with different parameters; use force=True to overwrite",
                        entry_name,
                    )
                return
            del dataset_group[entry_name]

        try:
            entry = dataset_group.create_group(entry_name)
        except ValueError as exc:
            message = str(exc).lower()
            if "read-only" in message or "read only" in message:
                logger.warning("Dispersion cache skipped: %s", exc)
                return
            raise
        self._create_dataset(entry, "S", result.S)
        self._create_dataset(entry, "k_axis", result.k_axis)
        self._create_dataset(entry, "f_axis", result.f_axis)

        if result.S_local is not None:
            self._create_dataset(entry, "S_local", result.S_local)
        if result.orth_axis is not None:
            self._create_dataset(entry, "orth_axis", result.orth_axis)
        if result.S_folded is not None:
            self._create_dataset(entry, "S_folded", result.S_folded)
        if result.k_folded is not None:
            self._create_dataset(entry, "k_folded", result.k_folded)

        entry.attrs["axis"] = result.axis
        entry.attrs["component"] = result.component
        entry.attrs["dt"] = float(result.dt)
        entry.attrs["dx"] = float(result.dx)
        entry.attrs["flipx"] = bool(result.flipx)
        if result.fold_period is not None:
            entry.attrs["fold_period"] = float(result.fold_period)
        entry.attrs["orth_axis_label"] = result.orth_axis_label or ""
        entry.attrs["notes_json"] = json.dumps(result.notes or [])
        entry.attrs["config_json"] = json.dumps(asdict(result.config))
        entry.attrs["context_json"] = context_json
        entry.attrs["context_hash"] = context_hash
        entry.attrs["dataset_name"] = self.dataset_name
        entry.attrs["slice_info"] = json.dumps(self._serialize_for_json(self.slice_info))
        entry.attrs["cached_at"] = datetime.utcnow().isoformat() + "Z"
        entry.attrs["job_name"] = getattr(self.parent_fft.job_result, "name", "")
        entry.attrs["zarr_path"] = str(self.parent_fft.job_result.path)
        store = getattr(dataset_group, "store", None)
        store_desc = getattr(store, "path", None) or getattr(store, "dir_path", None) or getattr(store, "filename", None)
        logger.info(
            "Dispersion result saved: group=%s entry=%s store=%s",
            getattr(dataset_group, "path", dataset_group.name),
            entry_name,
            store_desc or store.__class__.__name__ if store is not None else "<unknown>",
        )

    def _resolve_plot_save_path(
        self,
        save: Union[str, Path, bool],
        axis: str,
        result: DispersionResult1D,
    ) -> Optional[Path]:
        if isinstance(save, bool):
            if not save:
                return None
            base = getattr(self.parent_fft.job_result, "path", None)
            base_dir = Path(base) if base is not None else Path.cwd()
            if base_dir.is_file():
                base_dir = base_dir.parent
            job_name = self._sanitize_name(str(getattr(self.parent_fft.job_result, "name", "job")))
            dataset_part = self._sanitize_name(self.dataset_name or "global")
            component_part = self._sanitize_name(result.component or "perp")
            timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            filename = f"{job_name}_{dataset_part}_dispersion_{axis}_{component_part}_{timestamp}.png"
            return base_dir / filename

        if save is None:
            return None

        if isinstance(save, Path):
            return save

        if isinstance(save, (str, bytes, bytearray)):
            path = Path(save)
            return path

        raise TypeError("save must be bool or path-like string")
    
    def configure(
        self,
        dt: Optional[float] = None,
        dx: Optional[float] = None,
        dy: Optional[float] = None,
        component: str = "perp",
        time_window: str = "hann",
        detrend: str = "mean",
        tmax: int = 100,
        **kwargs
    ) -> "FFTDispersionInterface":
        """
        Configure dispersion analysis parameters.
        
        Parameters
        ----------
        dt : float, optional
            Time step [s] (auto-detected from zarr if not provided)
        dx, dy : float, optional  
            Grid spacings [m] (auto-detected from zarr if not provided)
        component : str, default="perp"
            Magnetization component to analyze ('perp', 'mx', 'my', 'mz')
        time_window : str, default="hann"
            Time-domain window function
        detrend : str, default="mean"
            Detrending method ('mean', 'initial', None)
        tmax : int, default=100
            Maximum number of time steps to load for speed
            
        Returns
        -------
        FFTDispersionInterface
            Self for method chaining
        """
        config_params: dict[str, Any] = {
            'component': component,
            'time_window': time_window, 
            'detrend': detrend,
        }
        
        # Add optional parameters if provided
        if dt is not None:
            config_params['dt'] = dt
        if dx is not None:
            config_params['dx'] = dx  
        if dy is not None:
            config_params['dy'] = dy
            
        config_params.update(kwargs)
        self._config = DispersionConfig(**config_params)
        self._tmax = tmax

        # Reset analyzer to use new config
        self._analyzer = None

        return self

    def filters(
        self,
        *,
        remove_static: bool = False,
        average: bool = False,
        window: Optional[Union[str, Sequence[str]]] = None,
    ) -> "FFTDispersionInterface":
        """Return new interface with preprocessing filters applied to data.

        Parameters
        ----------
        remove_static : bool, default=False
            Subtract the initial time sample from every time step.
        average : bool, default=False
            Subtract the temporal average at each spatial point.
        window : str | Sequence[str] | None, optional
            Apply Hann windows: ``'time'`` for temporal, ``'space'`` (or ``'2d'``)
            for spatial, and ``'both'``/``'hann'`` for both domains. Multiple
            entries can be provided via a sequence.

        Returns
        -------
        FFTDispersionInterface
            Cloned interface carrying the requested filters.
        """

        config = self._normalize_filters_config(
            remove_static=remove_static,
            average=average,
            window=window,
        )

        clone = self.clone_for_dataset(self.dataset_name, self.slice_info)
        clone._filters_config = copy.deepcopy(config)
        return clone

    def _normalize_filters_config(
        self,
        *,
        remove_static: bool,
        average: bool,
        window: Optional[Union[str, Sequence[str]]],
    ) -> Optional[dict[str, bool]]:
        config: dict[str, bool] = {}

        if remove_static:
            config["remove_static"] = True
        if average:
            config["remove_average"] = True

        time_flag, space_flag = self._interpret_filter_window(window)
        if time_flag:
            config["hann_time"] = True
        if space_flag:
            config["hann_space"] = True

        return config or None

    def _interpret_filter_window(
        self,
        window: Optional[Union[str, Sequence[str]]],
    ) -> tuple[bool, bool]:
        if window is None:
            return False, False

        entries: list[str]
        if isinstance(window, str):
            entries = [window]
        elif isinstance(window, Sequence):
            entries = list(window)
        else:
            raise TypeError("window must be a string, a sequence of strings, or None")

        time_flag = False
        space_flag = False
        for raw in entries:
            token = str(raw).strip().lower()
            if not token or token in {"none", "null"}:
                continue
            if token in {"time", "hann_time", "temporal"}:
                time_flag = True
            elif token in {"space", "spatial", "hann_space", "2d", "2d_space"}:
                space_flag = True
            elif token in {"both", "hann", "all", "time_space", "space_time"}:
                time_flag = True
                space_flag = True
            else:
                raise ValueError(f"Unknown window option: {raw!r}")

        return time_flag, space_flag

    def _describe_filter_flags(self) -> list[str]:
        if not self._filters_config:
            return []

        labels: list[str] = []
        if self._filters_config.get("remove_static"):
            labels.append("remove_static")
        if self._filters_config.get("remove_average"):
            labels.append("average")
        if self._filters_config.get("hann_time"):
            labels.append("hann_time")
        if self._filters_config.get("hann_space"):
            labels.append("hann_space")
        return labels
    
    def compute_1d(
        self,
        axis: str = "x",
        component: Optional[str] = None,
        **kwargs
    ) -> DispersionResult1D:
        """
        Compute 1D dispersion relation S(k, f).
        
        Parameters
        ----------
        axis : str, default="x"
            Spatial axis for analysis ('x' or 'y')
        component : str, optional
            Override default component setting
        force : bool, optional (via kwargs)
            Force recomputation and overwrite cache entries.
        save : bool, optional (via kwargs)
            Persist result to on-disk cache (alias for ``save_result``).
        save_result : bool, optional (via kwargs)
            Explicit flag to persist the result to the zarr cache.
        use_cache : bool, optional (via kwargs)
            Use in-memory cache when available (default True).
        disk_cache : bool, optional (via kwargs)
            Check on-disk cache (default True). Alias ``use_disk_cache`` is accepted.
        kmax : float, optional (via kwargs)
            Trim returned data to |k| ≤ kmax (rad/m) without affecting cached data.
            Note: Input in rad/m regardless of display units (rad_um, meter, etc.)
        flipx : bool, optional (via kwargs), default True
            When True (default), mirror the dispersion result along the k-axis so that
            positive and negative wave-vectors are swapped. Set to False to preserve the
            raw FFT ordering for diagnostic purposes.
        **kwargs
            Additional parameters passed to compute_dispersion_1d
            
        Returns
        -------
        DispersionResult1D
            1D dispersion analysis result
        """
        compute_kwargs = dict(kwargs)

        kmax = compute_kwargs.pop("kmax", None)
        force = bool(compute_kwargs.pop("force", False))


        filters_config = compute_kwargs.pop("filters", None)
        if filters_config is None:
            filters_config = copy.deepcopy(self._filters_config)

        sanitized_filters = None
        if filters_config:
            sanitized_filters = {key: bool(value) for key, value in filters_config.items() if bool(value)}

        effective_config = self._config or DispersionConfig()
        effective_detrend = compute_kwargs.get("detrend", effective_config.detrend)

        if sanitized_filters and sanitized_filters.get("remove_average") and effective_detrend == "mean":
            logger.debug(
                "remove_average filter requested together with detrend='mean'; keeping flag for reproducibility",
            )

        if sanitized_filters:
            filter_labels = ", ".join(sorted(sanitized_filters))
            logger.info(
                "Dispersion filters active for %s axis=%s component=%s → %s",
                self.dataset_name or "global",
                axis,
                component or effective_config.component,
                filter_labels,
            )
        else:
            sanitized_filters = None

        filters_config = sanitized_filters
        
        # Extract flipx from kwargs (default True)
        flipx = compute_kwargs.pop("flipx", True)

        save_result_flag = compute_kwargs.pop("save_result", None)
        save_alias = compute_kwargs.pop("save", None)
        if save_result_flag is not None:
            persist_result = bool(save_result_flag)
        elif save_alias is not None:
            persist_result = bool(save_alias)
        else:
            persist_result = False

        use_cache = bool(compute_kwargs.pop("use_cache", True))
        disk_cache_setting = compute_kwargs.pop("disk_cache", None)
        if disk_cache_setting is None:
            disk_cache_setting = compute_kwargs.pop("use_disk_cache", True)
        disk_cache_flag = bool(disk_cache_setting)

        context_payload = dict(compute_kwargs)
        context_payload["flipx"] = bool(flipx)
        if filters_config is not None:
            context_payload["filters"] = filters_config
        context = self._build_cache_context(
            mode="dispersion_1d",
            axis=axis,
            component=component,
            extra_kwargs=context_payload,
        )
        context_json, context_hash = self._context_signature(context)
        memory_key = self._memory_key("dispersion_1d", context_hash)

        base_result: Optional[DispersionResult1D] = None

        if force:
            logger.info(
                "Force recompute requested; skipping caches for %s",
                memory_key,
            )
        elif use_cache:
            cached = self._memory_cache.get(memory_key)
            if cached is not None:
                logger.info("Using in-memory dispersion cache for %s", memory_key)
                base_result = cached

        entry_name = f"dispersion1d_{context_hash}"
        disk_group: Optional[zarr.Group] = None

        if base_result is None and disk_cache_flag and not force:
            disk_group = self._get_dispersion_dataset_group(write=False)
            cached_disk = self._load_cached_dispersion_result(
                disk_group,
                entry_name,
                context_hash,
            )
            if cached_disk is not None:
                logger.info("Loaded dispersion1d result from on-disk cache (%s)", entry_name)
                base_result = cached_disk
                if use_cache:
                    self._memory_cache[memory_key] = cached_disk

        if base_result is not None:
            cached_flipx = bool(getattr(base_result, "flipx", True))
            if cached_flipx != bool(flipx):
                logger.info(
                    "Cached dispersion result flipx=%s but requested %s; recomputing",
                    cached_flipx,
                    flipx,
                )
                # Invalidate stale cache entry
                if memory_key in self._memory_cache:
                    self._memory_cache.pop(memory_key, None)
                base_result = None

        if base_result is None:
            logger.info("Computing dispersion1d from scratch (force=%s)", force)
            base_result = self.analyzer.compute_dispersion_1d(
                axis=axis,
                component=component,
                filters=filters_config,
                flipx=flipx,
                **compute_kwargs,
            )
            if use_cache:
                self._memory_cache[memory_key] = base_result
            if persist_result:
                disk_group = disk_group or self._get_dispersion_dataset_group(write=True)
                self._save_dispersion_result(
                    disk_group,
                    entry_name,
                    base_result,
                    context_json,
                    context_hash,
                    overwrite=force,
                )
        else:
            if persist_result:
                disk_group = disk_group or self._get_dispersion_dataset_group(write=True)
                self._save_dispersion_result(
                    disk_group,
                    entry_name,
                    base_result,
                    context_json,
                    context_hash,
                    overwrite=force,
                )

        result = base_result

        if kmax is not None and result is not None:
            result = self._trim_dispersion_kmax(result, kmax)

        return cast(DispersionResult1D, result)
    
    def compute_2d(
        self,
        component: Optional[str] = None,
        **kwargs
    ) -> DispersionResult2D:
        """
        Compute 2D dispersion relation S(kx, ky, f).
        
        Parameters
        ----------
        component : str, optional
            Override default component setting
        **kwargs
            Additional parameters passed to compute_dispersion_2d
            
        Returns
        -------
        DispersionResult2D
            2D dispersion analysis result
        """
        return self.analyzer.compute_dispersion_2d(
            component=component,
            **kwargs
        )
    
    def track_branch(
        self,
        dispersion_result: DispersionResult1D,
        k_path: np.ndarray,
        f_seed: float,
        **kwargs
    ) -> DispersionBranch:
        """
        Track dispersion branch along k-path.
        
        Parameters
        ----------
        dispersion_result : DispersionResult1D
            1D dispersion result to track
        k_path : array-like
            k-values to track along [rad/m]
        f_seed : float
            Initial frequency guess [Hz]
        **kwargs
            Additional parameters passed to track_branch
            
        Returns
        -------
        DispersionBranch
            Tracked dispersion branch
        """
        return self.analyzer.track_branch(
            dispersion_result,
            k_path,
            f_seed,
            **kwargs
        )
    
    def find_peaks(
        self,
        dispersion_result: DispersionResult1D,
        **kwargs
    ) -> list:
        """
        Find spectral peaks in dispersion relation.
        
        Parameters
        ----------
        dispersion_result : DispersionResult1D
            1D dispersion result to analyze
        **kwargs
            Additional parameters passed to find_all_peaks
        Returns
        -------
        list
            List of (k, f, amplitude) tuples for detected peaks
        """
        return self.analyzer.find_all_peaks(dispersion_result, **kwargs)


    def _resolve_colornorm(
        self,
        spectrum: np.ndarray,
        *,
        lognorm_flag: bool,
        vmin: Optional[float],
        vmax: Optional[float],
        colornorm: Union[str, Normalize, None],
        colornorm_kwargs: Optional[dict[str, Any]],
        context: str,
    ) -> Optional[Normalize]:
        """Return a matplotlib Normalize instance based on user settings."""
        kwargs = dict(colornorm_kwargs or {})

        if isinstance(colornorm, Normalize):
            if lognorm_flag:
                logger.info("%s: custom Normalize provided; ignoring lognorm flag", context)
            return colornorm

        norm_name = None
        if colornorm is not None:
            norm_name = str(colornorm).strip().lower()
            if lognorm_flag:
                logger.info("%s: colornorm=%s overrides lognorm flag", context, norm_name)

        normalized_key = None
        if norm_name:
            normalized_key = re.sub(r"[^a-z]", "", norm_name)

        builder_map = {
            "linear": self._build_linear_norm,
            "norm": self._build_linear_norm,
            "normalize": self._build_linear_norm,
            "log": self._build_lognorm,
            "lognorm": self._build_lognorm,
            "power": self._build_power_norm,
            "powernorm": self._build_power_norm,
            "pow": self._build_power_norm,
            "symlog": self._build_symlog_norm,
            "symlognorm": self._build_symlog_norm,
            "centered": self._build_centered_norm,
            "centerednorm": self._build_centered_norm,
            "twoslope": self._build_two_slope_norm,
            "twoslopenorm": self._build_two_slope_norm,
            "func": self._build_func_norm,
            "funcnorm": self._build_func_norm,
        }

        if normalized_key and normalized_key not in builder_map:
            logger.warning(
                "%s: Unknown colornorm '%s'; falling back to default normalization",
                context,
                colornorm,
            )

        if normalized_key in builder_map:
            return builder_map[normalized_key](spectrum, vmin, vmax, kwargs, context)

        if lognorm_flag:
            return self._build_lognorm(spectrum, vmin, vmax, kwargs, context)
        if vmin is not None or vmax is not None or kwargs:
            return self._build_linear_norm(spectrum, vmin, vmax, kwargs, context)
        return None

    def _build_linear_norm(
        self,
        data: np.ndarray,
        vmin: Optional[float],
        vmax: Optional[float],
        kwargs: dict[str, Any],
        context: str,
    ) -> Optional[Normalize]:
        """Construct a standard Normalize."""
        norm_vmin = kwargs.pop("vmin", None)
        norm_vmax = kwargs.pop("vmax", None)
        if vmin is not None:
            norm_vmin = vmin
        if vmax is not None:
            norm_vmax = vmax

        limits = None
        if norm_vmin is None or norm_vmax is None:
            limits = self._auto_limits(data)

        if norm_vmin is None and limits is not None:
            norm_vmin = limits[0]
        if norm_vmax is None and limits is not None:
            norm_vmax = limits[1]

        if norm_vmin is None and norm_vmax is None and not kwargs:
            return None

        if norm_vmax is not None and norm_vmin is not None and norm_vmax <= norm_vmin:
            logger.warning(
                "%s: vmax=%s ≤ vmin=%s; using automatic limits",
                context,
                norm_vmax,
                norm_vmin,
            )
            limits = limits or self._auto_limits(data)
            if limits is None:
                return None
            norm_vmin, norm_vmax = limits

        logger.info(
            "%s: Applying linear normalization (vmin=%s, vmax=%s, extra=%s)",
            context,
            f"{norm_vmin:.2e}" if norm_vmin is not None else None,
            f"{norm_vmax:.2e}" if norm_vmax is not None else None,
            kwargs or {},
        )
        return Normalize(vmin=norm_vmin, vmax=norm_vmax, **kwargs)

    def _build_lognorm(
        self,
        data: np.ndarray,
        vmin: Optional[float],
        vmax: Optional[float],
        kwargs: dict[str, Any],
        context: str,
    ) -> Optional[Normalize]:
        """Construct a LogNorm based on spectrum values."""
        norm_vmin = kwargs.pop("vmin", None)
        norm_vmax = kwargs.pop("vmax", None)
        if vmin is not None:
            norm_vmin = vmin
        if vmax is not None:
            norm_vmax = vmax

        limits = None
        if norm_vmin is None or norm_vmax is None:
            limits = self._auto_limits(data, positive_only=True)
            if limits is None:
                logger.warning("%s: Cannot apply lognorm; spectrum has no positive values", context)
                return None

        if norm_vmin is None:
            norm_vmin = limits[0]
        if norm_vmax is None:
            norm_vmax = limits[1]

        if norm_vmin <= 0:
            logger.warning(
                "%s: vmin=%s ≤ 0 for log scale; using automatic positive bound",
                context,
                norm_vmin,
            )
            norm_vmin = max(limits[0], np.finfo(float).tiny)
        if norm_vmax <= norm_vmin:
            logger.warning(
                "%s: vmax=%s ≤ vmin=%s for log scale; using automatic limits",
                context,
                norm_vmax,
                norm_vmin,
            )
            norm_vmin, norm_vmax = limits

        logger.info(
            "%s: Applying log normalization (vmin=%s, vmax=%s, extra=%s)",
            context,
            f"{norm_vmin:.2e}",
            f"{norm_vmax:.2e}",
            kwargs or {},
        )
        return LogNorm(vmin=norm_vmin, vmax=norm_vmax, **kwargs)

    def _build_power_norm(
        self,
        data: np.ndarray,
        vmin: Optional[float],
        vmax: Optional[float],
        kwargs: dict[str, Any],
        context: str,
    ) -> Optional[Normalize]:
        gamma = kwargs.pop("gamma", kwargs.pop("power", 0.5))
        if gamma is None:
            gamma = 0.5
        norm_vmin = kwargs.pop("vmin", None)
        norm_vmax = kwargs.pop("vmax", None)
        if vmin is not None:
            norm_vmin = vmin
        if vmax is not None:
            norm_vmax = vmax
        limits = None
        if norm_vmin is None or norm_vmax is None:
            limits = self._auto_limits(data)
            if limits is None:
                return None
        if norm_vmin is None:
            norm_vmin = limits[0]
        if norm_vmax is None:
            norm_vmax = limits[1]
        if norm_vmax <= norm_vmin:
            norm_vmin, norm_vmax = limits
        logger.info(
            "%s: Applying power normalization (gamma=%s, vmin=%s, vmax=%s)",
            context,
            gamma,
            f"{norm_vmin:.2e}",
            f"{norm_vmax:.2e}",
        )
        return PowerNorm(gamma=gamma, vmin=norm_vmin, vmax=norm_vmax, **kwargs)

    def _build_symlog_norm(
        self,
        data: np.ndarray,
        vmin: Optional[float],
        vmax: Optional[float],
        kwargs: dict[str, Any],
        context: str,
    ) -> Optional[Normalize]:
        norm_vmin = kwargs.pop("vmin", None)
        norm_vmax = kwargs.pop("vmax", None)
        if vmin is not None:
            norm_vmin = vmin
        if vmax is not None:
            norm_vmax = vmax
        limits = None
        if norm_vmin is None or norm_vmax is None:
            limits = self._auto_limits(data)
        if norm_vmin is None and limits is not None:
            norm_vmin = limits[0]
        if norm_vmax is None and limits is not None:
            norm_vmax = limits[1]
        linthresh = kwargs.pop("linthresh", None)
        if linthresh is None:
            linthresh = self._default_linthresh(data)
        logger.info(
            "%s: Applying symlog normalization (linthresh=%s, vmin=%s, vmax=%s, extra=%s)",
            context,
            linthresh,
            norm_vmin,
            norm_vmax,
            kwargs,
        )
        return SymLogNorm(linthresh=linthresh, vmin=norm_vmin, vmax=norm_vmax, **kwargs)

    def _build_centered_norm(
        self,
        data: np.ndarray,
        vmin: Optional[float],
        vmax: Optional[float],
        kwargs: dict[str, Any],
        context: str,
    ) -> Normalize:
        vcenter = kwargs.pop("vcenter", 0.0)
        logger.info("%s: Applying centered normalization (vcenter=%s, extra=%s)", context, vcenter, kwargs)
        return CenteredNorm(vcenter=vcenter, **kwargs)

    def _build_two_slope_norm(
        self,
        data: np.ndarray,
        vmin: Optional[float],
        vmax: Optional[float],
        kwargs: dict[str, Any],
        context: str,
    ) -> Normalize:
        vcenter = kwargs.pop("vcenter", 0.0)
        norm_vmin = kwargs.pop("vmin", None)
        norm_vmax = kwargs.pop("vmax", None)
        if vmin is not None:
            norm_vmin = vmin
        if vmax is not None:
            norm_vmax = vmax
        limits = None
        if norm_vmin is None or norm_vmax is None:
            limits = self._auto_limits(data)
        if norm_vmin is None and limits is not None:
            norm_vmin = limits[0]
        if norm_vmax is None and limits is not None:
            norm_vmax = limits[1]
        logger.info(
            "%s: Applying TwoSlope normalization (vcenter=%s, vmin=%s, vmax=%s)",
            context,
            vcenter,
            norm_vmin,
            norm_vmax,
        )
        return TwoSlopeNorm(vcenter=vcenter, vmin=norm_vmin, vmax=norm_vmax, **kwargs)

    def _build_func_norm(
        self,
        data: np.ndarray,
        vmin: Optional[float],
        vmax: Optional[float],
        kwargs: dict[str, Any],
        context: str,
    ) -> Normalize:
        functions = kwargs.pop("functions", None)
        if functions is None:
            raise ValueError(
                f"{context}: colornorm='funcnorm' requires 'functions=(forward, inverse)' in colornorm_kwargs",
            )
        norm_vmin = kwargs.pop("vmin", None)
        norm_vmax = kwargs.pop("vmax", None)
        if vmin is not None:
            norm_vmin = vmin
        if vmax is not None:
            norm_vmax = vmax
        logger.info("%s: Applying FuncNorm", context)
        return FuncNorm(functions=functions, vmin=norm_vmin, vmax=norm_vmax, **kwargs)

    def _auto_limits(self, data: np.ndarray, positive_only: bool = False) -> Optional[tuple[float, float]]:
        """Infer data limits while ignoring NaNs and infs."""
        arr = np.asarray(data, dtype=float)
        mask = np.isfinite(arr)
        if positive_only:
            mask &= arr > 0
        if not np.any(mask):
            return None
        subset = arr[mask]
        return float(np.nanmin(subset)), float(np.nanmax(subset))

    def _default_linthresh(self, data: np.ndarray) -> float:
        arr = np.asarray(data, dtype=float)
        arr = np.abs(arr[np.isfinite(arr)])
        arr = arr[arr > 0]
        if arr.size == 0:
            return 1e-9
        return float(np.nanpercentile(arr, 5))


    def _apply_k0_normalization(
        self,
        spectrum: np.ndarray,
        k_axis: np.ndarray,
        strength: Union[int, float] = 1.0,
        compression_mode: str = "adaptive",
        k0_normalization_width: int = 1,
    ) -> np.ndarray:
        """Apply dynamic k≈0 compression to the supplied spectrum."""
        if strength <= 1e-6:
            return spectrum.copy()

        linear_strength = max(0.0, min(1.0, float(strength) / 10.0))

        if compression_mode == "gentle":
            beta = 4.5 - 1.0 * linear_strength
            A_max = 10.0 + 90.0 * linear_strength
            knee_db = 8.0 - 2.0 * linear_strength
            slope_db = 5.0 - 1.0 * linear_strength
        elif compression_mode == "aggressive":
            beta = 4.0 - 1.5 * linear_strength
            A_max = 100.0 + 4900.0 * linear_strength
            knee_db = 6.0 - 2.0 * linear_strength
            slope_db = 3.0 - 1.5 * linear_strength
        elif compression_mode == "preserve_peaks":
            beta = 4.0 - 0.5 * linear_strength
            A_max = 20.0 + 180.0 * linear_strength
            knee_db = 7.0 - 1.0 * linear_strength
            slope_db = 4.0 - 0.5 * linear_strength
        else:
            beta = 1.0 - 0.5 * linear_strength
            A_max = 500.0 + 9500.0 * linear_strength
            knee_db = 6.0
            slope_db = 2.5 - 0.5 * linear_strength

        logger.info(
            "k≈0 dynamic compression: mode=%s, strength=%s (linear=%.2f) → β=%.2f, A_max=%.0f, knee=%.1fdB, slope=%.1fdB",
            compression_mode,
            strength,
            linear_strength,
            beta,
            A_max,
            knee_db,
            slope_db,
        )

        return self._k0_dynamic_filter(
            spectrum,
            k_axis,
            strength=linear_strength,
            beta=beta,
            knee_db=knee_db,
            slope_db=slope_db,
            A_max=A_max,
            k0_normalization_width=k0_normalization_width,
        )


    def _k0_dynamic_filter(
        self,
        PSD_fk: np.ndarray,
        k_vals: np.ndarray,
        k_halfwidth: Optional[float] = None,
        beta: Optional[float] = None,
        knee: Optional[float] = None,
        ratio: Optional[float] = None,
        smooth_win: Optional[int] = 11,
        smooth_poly: int = 2,
        strength: float = 1.0,
        k0_normalization_width: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Internal helper that handles parameter normalization and logging."""
        beta_val = float(kwargs.pop("beta", beta if beta is not None else 3.5))
        knee_db_val = float(kwargs.pop("knee_db", 6.0))
        slope_db_val = float(kwargs.pop("slope_db", 3.0))
        A_max_val = float(kwargs.pop("A_max", 1000.0))

        if knee is not None:
            knee_db_val = max(1e-6, knee_db_val * float(knee))

        ratio_val = ratio if ratio is not None else kwargs.pop("ratio", None)
        if ratio_val is not None:
            ratio_val = max(1.0, float(ratio_val))
            logger.debug("k≈0 dynamic filter ratio cap set to %.3f", ratio_val)

        if kwargs:
            logger.debug("Unused k0 dynamic filter kwargs: %s", ", ".join(sorted(kwargs)))

        if PSD_fk.ndim == 1:
            PSD_compressed, _, _ = self._k0_dynamic_filter_linear(
                PSD_fk,
                k_vals,
                strength=strength,
                A_max=A_max_val,
                beta=beta_val,
                ratio=ratio_val,
                knee_db=knee_db_val,
                slope_db=slope_db_val,
                k0_normalization_width=k0_normalization_width,
                k_halfwidth=k_halfwidth,
                smooth_win=smooth_win,
                smooth_poly=smooth_poly,
            )
            return PSD_compressed

        PSD_compressed, _, _ = self._k0_dynamic_filter_linear(
            PSD_fk.T,
            k_vals,
            strength=strength,
            A_max=A_max_val,
            beta=beta_val,
            ratio=ratio_val,
            knee_db=knee_db_val,
            slope_db=slope_db_val,
            k0_normalization_width=k0_normalization_width,
            k_halfwidth=k_halfwidth,
            smooth_win=smooth_win,
            smooth_poly=smooth_poly,
        )
        return PSD_compressed.T


    def _k0_dynamic_filter_linear(
        self,
        PSD_fk: np.ndarray,
        k_vals: np.ndarray,
        strength: float = 1.0,
        A_max: float = 1000.0,
        beta: float = 3.5,
        ratio: Optional[float] = None,
        knee_db: float = 6.0,
        slope_db: float = 3.0,
        k0_normalization_width: int = 1,
        k_halfwidth: Optional[float] = None,
        smooth_win: Optional[int] = 11,
        smooth_poly: int = 2,
        eps: float = 1e-18,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply the linearized compressor to PSD data."""

        def _odd(n: int) -> int:
            return int(n) + 1 - (int(n) % 2 == 0)

        PSD = np.asarray(PSD_fk).copy()
        k = np.asarray(k_vals)

        if PSD.ndim == 1:
            F, K = 1, PSD.shape[0]
            PSD = PSD[np.newaxis, :]
            is_1d = True
        elif PSD.ndim == 2:
            F, K = PSD.shape
            is_1d = False
        else:
            raise ValueError(f"PSD must be 1D or 2D array, got {PSD.ndim}D")

        center_idx = np.argmin(np.abs(k))
        half_width = max(0, (k0_normalization_width - 1) // 2)
        if k_halfwidth is not None and K > 1:
            dk_values = np.diff(np.sort(np.abs(k)))
            dk = float(np.median(dk_values)) if dk_values.size else 0.0
            if dk > 0:
                half_width = max(half_width, int(np.ceil(abs(k_halfwidth) / dk)))

        idx0 = np.array(
            [
                center_idx + offset
                for offset in range(-half_width, half_width + 1)
                if 0 <= center_idx + offset < K
            ]
        )

        logger.info(
            "k≈0 region: width=%s, center_idx=%s, indices=%s, total_bins=%s",
            k0_normalization_width,
            center_idx,
            idx0.tolist(),
            len(idx0),
        )

        other = np.setdiff1d(np.arange(K), idx0)
        if other.size == 0:
            return PSD, idx0, np.ones((F, idx0.size))

        base = np.median(PSD[:, other], axis=1)
        if _SCIPY_AVAILABLE:
            from scipy.stats import median_abs_deviation as mad
            scale = mad(PSD[:, other], axis=1, scale="normal") + eps
        else:
            q75 = np.percentile(PSD[:, other], 75, axis=1)
            q25 = np.percentile(PSD[:, other], 25, axis=1)
            scale = 0.7413 * (q75 - q25) + eps

        T = base + beta * scale

        if _SCIPY_AVAILABLE and smooth_win is not None and 5 <= smooth_win < F:
            from scipy.signal import savgol_filter
            T = savgol_filter(T, _odd(smooth_win), smooth_poly, mode="interp")

        T = T[:, None]

        strength = float(np.clip(strength, 0.0, 1.0))
        ratio_cap = None if ratio is None else max(1.0, ratio)
        attenuation_cap = max(1.0, A_max)
        if ratio_cap is not None:
            attenuation_cap = min(attenuation_cap, ratio_cap)
        A = 1.0 + (attenuation_cap - 1.0) * strength
        invA = 1.0 / max(A, 1.0)

        V = PSD[:, idx0]
        x_db = 10.0 * np.log10((V + eps) / (T + eps))
        gain_shape = max(1e-6, slope_db)
        w = 1.0 / (1.0 + np.exp(-(x_db - knee_db) / gain_shape))
        local_gain = 1.0 - w * (1.0 - invA)
        PSD[:, idx0] = V * local_gain

        PSD = np.maximum(PSD, 0.0)

        full_gain = np.ones((F, K))
        full_gain[:, idx0] = local_gain

        if is_1d:
            return PSD[0, :], idx0, full_gain[0, :]
        return PSD, idx0, full_gain


    def _compress_above_threshold(
        self,
        V: np.ndarray,
        T: np.ndarray,
        knee: float = 1.0,
        ratio: float = 6.0,
        eps: float = 1e-18,
    ) -> np.ndarray:
        """Soft compressor helper retained for backwards compatibility."""
        s = knee * np.maximum(T, eps)
        over = np.maximum(V - T, 0.0)
        Y = T + s * np.arcsinh(over / s)
        Y = np.minimum(Y, ratio * T)
        return np.where(V > T, Y, V)
    
    def plot_dispersion(
        self,
        ax: Optional[plt.Axes] = None,
        axis: str = "x",
        component: Optional[str] = None,
        figsize: tuple = (12, 8),
        cmap: str = "cmc.davos",
        kscale: str = "rad_um",
        f_units: str = "GHz",
        title: Optional[str] = None,
        save: Union[str, Path, bool, None] = None,
        orth_index: Optional[int] = None,
        dpi: Optional[int] = None,
        k_xlim: Optional[tuple[float, float]] = None,
        lognorm: bool = False,
        k0_normalization: Union[int, float] = 0,
        k0_normalization_width: int = 1,
        compression_mode: str = "adaptive",
        add_comsol_points: str | Path | None = None,
        comsol_k_col: int = 0,
        comsol_f_col: int = 1,
        comsol_extra_cols: tuple[int, ...] | None = None,
        comsol_style: dict[str, object] | None = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        trim_0f: Optional[int] = None,
        fmax: Optional[float] = None,
        colornorm: Union[str, Normalize, None] = None,
        colornorm_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> tuple:
        """
        Plot 1D dispersion relation S(k, f).
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure and axes.
        axis : str, default="x"
            Spatial axis for analysis ('x' or 'y')
        component : str, optional
            Override default component setting
        figsize : tuple, default=(12, 8)
            Figure size (width, height). Only used when ax=None.
        dpi : int, optional
            Figure resolution (passes through to matplotlib). Only used when ax=None.
        cmap : str, default="cmc.davos"
            Colormap for dispersion plot (uses crameri davos colormap by default)
        kscale : str, default="rad_um"
            Wave-vector units: "rad_um" for rad/μm (default), "rad" for rad/m, "meter" for 1/m (cycles per meter)
        f_units : str, default="GHz" 
            Units for frequency axis ('Hz', 'GHz')
        title : str, optional
            Plot title (auto-generated if None)
        save : str | pathlib.Path | bool, optional
            When a path is provided, save the plot there. Pass True to auto-generate
            a filename next to the job's zarr data. False or None disables saving.
        orth_index : int, optional
            Select specific orthogonal slice when avg_over_orthogonal=False
        k_xlim : tuple, optional
            Limits for wave-vector axis after unit conversion (default ±10 rad/μm for rad_um, ±20 m⁻¹ for meter)
        lognorm : bool, default=False
            Use logarithmic normalization for color scale
        colornorm : str | matplotlib.colors.Normalize | None, optional
            Advanced Matplotlib normalization selector. Supported string values:
            "lognorm", "symlognorm", "powernorm", "twoslopenorm",
            "centerednorm", "funcnorm". You can also pass a pre-built
            Normalize instance. When provided, overrides the legacy ``lognorm``
            flag. ``colornorm="lognorm"`` is equivalent to ``lognorm=True``.
        colornorm_kwargs : dict, optional
            Extra keyword arguments forwarded to the normalization constructor
            (e.g. ``{"linthresh": 1e-5}`` for ``symlognorm``). Ignored when
            ``colornorm`` is a Normalize instance.
        k0_normalization : int or float, default=0
            Adaptive k≈0 mode suppression intensity. 0=disabled, 1-10=increasing suppression strength.
            Uses advanced dynamic thresholding based on k≠0 statistical reference with soft audio-style
            compression. Higher values apply stronger compression ratios and lower thresholds.
            Professional algorithm preserves relative peak structures while enabling linear visualization.
        k0_normalization_width : int, default=1
            Number of k-space cells around k≈0 to apply compression. Default=1 compresses only the
            central k≈0 bin. Width=3 compresses center±1, width=5 compresses center±2, etc.
            Controls the spatial extent of the k≈0 normalization region.
        compression_mode : str, default="adaptive"
            Compression strategy for k≈0 normalization. Options:
            - "gentle": Conservative compression with smooth transitions
            - "adaptive": Balanced approach adapting to signal characteristics (default)
            - "aggressive": Strong suppression for very prominent k≈0 modes
            - "preserve_peaks": Maintains relative peak intensities while reducing background
        vmin : float, optional
            Minimum value for color scale normalization. When provided, overrides automatic 
            scaling for the colorbar lower bound.
        vmax : float, optional
            Maximum value for color scale normalization. When provided, overrides automatic 
            scaling for the colorbar upper bound.
        trim_0f : int, optional
            Remove N lowest frequency points from plot (useful when f≈0 has strong artifacts)
        fmax : float, optional
            Maximum frequency to display (in f_units). Frequencies above this will be trimmed.
            Useful for synchronizing Y-axis range when using sharey=True with other plots.
        add_comsol_points : str | Path | None, optional
            Path to COMSOL dispersion export file. When provided, the selected column pair is overlayed
            as scatter points on top of the heatmap.
        comsol_k_col, comsol_f_col : int, optional
            Zero-based column indices used for k-vector and frequency values inside the COMSOL file.
        comsol_extra_cols : tuple[int, ...] | None, optional
            Additional columns to parse and keep available via the COMSOL data container (not plotted).
        comsol_style : dict, optional
            Keyword arguments forwarded to ``ax.scatter`` for the COMSOL overlay (e.g. color, size).
        **kwargs
            Additional parameters for compute_1d and plotting
            
        Returns
        -------
        tuple
            (figure, axis) matplotlib objects. The dispersion result is stored
            in self._last_plot_result for programmatic access if needed.
        """
        # Extract figure-related kwargs if provided dynamically
        if "figsize" in kwargs:
            figsize_override = kwargs.pop("figsize")
            if isinstance(figsize_override, tuple):
                figsize = cast(tuple[float, float], figsize_override)
            elif isinstance(figsize_override, list) and len(figsize_override) == 2:
                figsize = (float(figsize_override[0]), float(figsize_override[1]))
            else:
                raise TypeError("figsize kwarg must be a tuple or list of length 2")

        if "dpi" in kwargs:
            dpi_override = kwargs.pop("dpi")
            dpi = int(dpi_override) if dpi_override is not None else None

        # Backwards compatibility for legacy k_units parameter
        if "k_units" in kwargs:
            legacy_units = kwargs.pop("k_units")
            warnings.warn(
                "k_units is deprecated; use kscale='rad_um', 'rad' or 'meter'",
                DeprecationWarning,
                stacklevel=2,
            )
            if legacy_units in {"rad/m", "rad"}:
                kscale = "rad"
            elif legacy_units in {"1/m", "m^-1", "meter", "per_meter"}:
                kscale = "meter"
            elif legacy_units in {"rad/um", "rad/μm", "rad_um"}:
                kscale = "rad_um"

        kscale = kscale.lower()
        if kscale not in {"rad", "meter", "rad_um"}:
            raise ValueError("kscale must be 'rad_um', 'rad', or 'meter'")

        # Remove plot-specific parameters from compute kwargs
        compute_kwargs = dict(kwargs)
        # k0_normalization_width is handled in plot_dispersion, not compute_dispersion_1d
        compute_kwargs.pop("k0_normalization_width", None)  # Safe removal
        
        if save is True and "save_result" not in compute_kwargs and "save" not in compute_kwargs:
            compute_kwargs["save_result"] = True
        result = self.compute_1d(axis=axis, component=component, **compute_kwargs)


        comsol_style = comsol_style or {}
        colornorm_kwargs = dict(colornorm_kwargs or {})
        comsol_data = None
        if add_comsol_points is not None:
            from .comsol import read_data_from_comsol

            comsol_data = read_data_from_comsol(
                add_comsol_points,
                k_col=comsol_k_col,
                f_col=comsol_f_col,
                extra_cols=comsol_extra_cols,
            )

        # Create plot or use provided axes
        if ax is None:
            if dpi is not None:
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            else:
                fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Prepare axes
        k_axis = result.k_axis
        f_axis = result.f_axis

        spectrum = result.S
        orth_label = result.orth_axis_label or ("y" if axis == "x" else "x")

        if orth_index is not None:
            if result.S_local is None:
                raise ValueError("Result does not contain local spectra; recompute with avg_over_orthogonal=False")
            if orth_index < 0 or orth_index >= result.S_local.shape[0]:
                raise ValueError(f"orth_index {orth_index} out of range (0..{result.S_local.shape[0]-1})")
            spectrum = result.S_local[orth_index]
            if title is None:
                if result.orth_axis is not None and orth_index < len(result.orth_axis):
                    axis_value = result.orth_axis[orth_index]
                    title = f"Spin-Wave Dispersion {orth_label}={axis_value:g}"
                else:
                    title = f"Spin-Wave Dispersion ({orth_label} index {orth_index})"

        # Remove negative frequencies from visualization
        if f_axis.ndim == 1 and spectrum.shape[1] == f_axis.shape[0]:
            positive_mask = f_axis >= 0
            if np.any(positive_mask) and positive_mask.sum() < f_axis.size:
                spectrum = spectrum[:, positive_mask]
                f_axis = f_axis[positive_mask]

        # Trim lowest frequency points if requested
        if trim_0f is not None and trim_0f > 0:
            if f_axis.ndim == 1 and trim_0f < f_axis.shape[0]:
                logger.info(f"Trimming {trim_0f} lowest frequency points from dispersion plot")
                spectrum = spectrum[:, trim_0f:]
                f_axis = f_axis[trim_0f:]
            else:
                logger.warning(f"trim_0f={trim_0f} exceeds available frequency points ({f_axis.shape[0]}), ignoring")

        # Trim frequencies above fmax if requested (applied BEFORE unit conversion)
        if fmax is not None and fmax > 0:
            if f_axis.ndim == 1:
                # f_axis is still in Hz at this point, convert fmax to Hz
                if f_units == "GHz":
                    fmax_hz = fmax * 1e9
                else:  # Hz
                    fmax_hz = fmax
                
                fmax_mask = f_axis <= fmax_hz
                n_above = (~fmax_mask).sum()
                if np.any(fmax_mask):
                    spectrum = spectrum[:, fmax_mask]
                    f_axis = f_axis[fmax_mask]
                    logger.info(f"Trimmed {n_above} frequency points above fmax={fmax} {f_units}")
                else:
                    logger.warning(f"fmax={fmax} {f_units} is below all frequencies, ignoring")

        # Convert units if requested
        if kscale == "meter":
            k_axis = k_axis / (2 * np.pi)
            k_label = r"$k$ [m$^{-1}$]"
            default_k_xlim = (-20.0, 20.0)
        elif kscale == "rad_um":
            k_axis = k_axis / 1e6  # Convert rad/m to rad/μm
            k_label = r"$k$ [rad/μm]"
            default_k_xlim = (-10.0, 10.0)  # Nice clean limits in rad/μm
        else:  # kscale == "rad"
            k_label = r"$k$ [rad/m]"
            default_k_xlim = None

        if f_units == "GHz":
            f_axis = f_axis / 1e9
            f_label = "Frequency [GHz]"
        else:
            f_label = "Frequency [Hz]"

        if k0_normalization and k0_normalization > 0:
            logger.info(
                "Applying k≈0 dynamic compression: strength=%s, mode=%s, width=%s",
                k0_normalization,
                compression_mode,
                k0_normalization_width,
            )
            original_k_axis = result.k_axis
            spectrum = self._apply_k0_normalization(
                spectrum,
                original_k_axis,
                strength=k0_normalization,
                compression_mode=compression_mode,
                k0_normalization_width=k0_normalization_width,
            )

        # Plot dispersion
        norm = self._resolve_colornorm(
            spectrum,
            lognorm_flag=lognorm,
            vmin=vmin,
            vmax=vmax,
            colornorm=colornorm,
            colornorm_kwargs=colornorm_kwargs,
            context="plot_dispersion",
        )

        extent = (
            float(k_axis[0]),
            float(k_axis[-1]),
            float(f_axis[0]),
            float(f_axis[-1]),
        )

        im = ax.imshow(
            spectrum.T,
            cmap=cmap,
            norm=norm,
            aspect="auto",
            origin="lower",
            extent=extent,
        )

        # Formatting
        ax.set_xlabel(k_label)
        ax.set_ylabel(f_label)

        if fmax is not None and fmax > 0:
            y_min = float(np.min(f_axis))
            y_max = float(np.max(f_axis))
            ax.set_ylim(y_min, y_max)

        if k_xlim is not None:
            ax.set_xlim(*k_xlim)
        elif default_k_xlim is not None:
            ax.set_xlim(*default_k_xlim)

        if title is None:
            title = f"Spin-Wave Dispersion S(k{axis}, f)"
            if hasattr(result, "component"):
                title += f" - {result.component} component"
        ax.set_title(title)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"Power Spectral Density [arb. units]")
        
        
        
        if comsol_data is not None:
            k_points = np.asarray(comsol_data.k_values, dtype=float).copy()
            f_points = np.asarray(comsol_data.f_values, dtype=float).copy()
            
            # Apply flipx if it was used in dispersion computation
            if getattr(result, 'flipx', True):
                k_points = -k_points
                logger.info("COMSOL k-values flipped (flipx=True in dispersion result)")

            if kscale == "rad_um":
                k_points = k_points / 1e6
            elif kscale == "meter":
                k_points = k_points / (2 * np.pi)

            if f_units.lower() == "ghz":
                f_points = f_points / 1e9
            elif f_units.lower() != "hz":
                raise ValueError(f"Unsupported frequency units: {f_units}")

            mask = np.isfinite(k_points) & np.isfinite(f_points)
            if not np.any(mask):
                logger.warning(
                    "COMSOL overlay skipped: no finite data points (%s)",
                    add_comsol_points,
                )
            else:
                if not np.all(mask):
                    logger.warning(
                        "COMSOL overlay dropping %d invalid points",
                        int((~mask).sum()),
                    )
                k_points = k_points[mask]
                f_points = f_points[mask]

                scatter_kwargs = {
                    "s": 40,
                    "facecolors": "none",
                    "edgecolors": "white",
                    "linewidths": 1.5,
                    "alpha": 0.9,
                }
                scatter_kwargs.update(comsol_style)

                logger.info(
                    "Overlaying %d COMSOL points on dispersion plot",
                    k_points.size,
                )
                sample_size = min(k_points.size, 5)
                if sample_size:
                    sample_k = " ".join(f"{val:.3f}" for val in k_points[:sample_size])
                    sample_f = " ".join(f"{val:.3f}" for val in f_points[:sample_size])
                    logger.debug(
                        "COMSOL overlay sample k (plot units): %s",
                        sample_k,
                    )
                    logger.debug(
                        "COMSOL overlay sample f (plot units): %s",
                        sample_f,
                    )

                ax.scatter(k_points, f_points, label="COMSOL", **scatter_kwargs)
                

        try:
            fig.tight_layout()
        except Exception:
            pass  # Skip tight_layout if it conflicts with existing colorbar

        # Save if requested
        save_path = None
        if save not in (None, False):
            save_path = self._resolve_plot_save_path(save, axis, result)

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi or 300, bbox_inches="tight")
            logger.info("Dispersion plot saved to %s", save_path)

        # Store result for access but don't return it to avoid verbose output
        self._last_plot_result = result
        return fig, ax

    def plot_branch(
        self,
        branch: DispersionBranch,
        figsize: tuple = (10, 6),
        k_units: str = "rad/m",
        f_units: str = "GHz",
        title: Optional[str] = None,
    save: Optional[str] = None,
    ) -> tuple:
        """
        Plot dispersion branch with group velocity.
        
        Parameters
        ----------
        branch : DispersionBranch
            Branch to plot
        figsize : tuple, default=(10, 6)
            Figure size
        k_units, f_units : str
            Axis units 
        title : str, optional
            Plot title
        save : str, optional
            Save path
            
        Returns
        -------
        tuple
            (figure, (ax1, ax2)) with dispersion and group velocity plots
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
        
        # Prepare data
        k_data = np.asarray(branch.k_path)
        f_data = np.asarray(branch.f_values)
        if branch.group_velocity is None:
            branch.compute_group_velocity()
        vg_data = np.asarray(branch.group_velocity)
        
        # Convert units
        if k_units == "1/m":
            k_data = k_data / (2 * np.pi)
            k_label = r"$k$ [m$^{-1}$]"
        else:
            k_label = r"$k$ [rad/m]"
            
        if f_units == "GHz":
            f_data = f_data / 1e9
            f_label = "Frequency [GHz]"
        else:
            f_label = "Frequency [Hz]"
        
        # Plot dispersion branch
        ax1.plot(k_data, f_data, 'o-', linewidth=2, markersize=4)
        ax1.set_ylabel(f_label)
        ax1.grid(True, alpha=0.3)
        if title:
            ax1.set_title(title)
        else:
            ax1.set_title("Dispersion Branch")
        
        # Plot group velocity  
        ax2.plot(k_data, vg_data / 1e3, 's-', color='red', linewidth=2, markersize=4)
        ax2.set_xlabel(k_label)
        ax2.set_ylabel(r"Group Velocity [km/s]")
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        
        try:
            fig.tight_layout()
        except Exception:
            pass  # Skip tight_layout if it conflicts with existing elements
        
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
            
        return fig, (ax1, ax2)
    
    def plot_result(
        self,
        result: DispersionResult1D,
        ax: Optional[plt.Axes] = None,
        figsize: tuple = (12, 8),
        cmap: str = "cmc.davos",
        kscale: str = "rad_um",
        f_units: str = "GHz",
        title: Optional[str] = None,
        save: Union[str, Path, bool, None] = None,
        orth_index: Optional[int] = None,
        dpi: Optional[int] = None,
        k_xlim: Optional[tuple[float, float]] = None,
        lognorm: bool = False,
        k0_normalization: Union[int, float] = 0,
        k0_normalization_width: int = 1,
        compression_mode: str = "adaptive",
        add_comsol_points: str | Path | None = None,
        comsol_k_col: int = 0,
        comsol_f_col: int = 1,
        comsol_extra_cols: tuple[int, ...] | None = None,
        comsol_style: dict[str, object] | None = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        trim_0f: Optional[int] = None,
        fmax: Optional[float] = None,
        colornorm: Union[str, Normalize, None] = None,
        colornorm_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple:
        """
        Plot a pre-computed dispersion result without recomputation.
        
        This method separates plotting from computation, allowing you to:
        - Compute once with compute_1d(), then plot multiple times with different settings
        - Plot the same result on different axes or in different styles
        - Avoid expensive recomputation when only visual parameters change
        
        Parameters
        ----------
        result : DispersionResult1D
            Pre-computed dispersion result from compute_1d()
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure and axes.
        figsize : tuple, default=(12, 8)
            Figure size (width, height). Only used when ax=None.
        dpi : int, optional
            Figure resolution. Only used when ax=None.
        cmap : str, default="cmc.davos"
            Colormap for dispersion plot
        kscale : str, default="rad_um"
            Wave-vector units: "rad_um" for rad/μm, "rad" for rad/m, "meter" for 1/m
        f_units : str, default="GHz" 
            Frequency axis units ('Hz', 'GHz')
        title : str, optional
            Plot title (auto-generated if None)
        save : str | pathlib.Path | bool, optional
            Save path for the plot
        orth_index : int, optional
            Select specific orthogonal slice when result contains local spectra
        k_xlim : tuple, optional
            Limits for wave-vector axis after unit conversion
        lognorm : bool, default=False
            Use logarithmic color scale normalization
        colornorm : str | matplotlib.colors.Normalize | None, optional
            Select Matplotlib normalization: \"lognorm\", \"symlognorm\", \"powernorm\",
            \"twoslopenorm\", \"centerednorm\", \"funcnorm\" or provide a Normalize instance.
            Overrides the legacy ``lognorm`` flag.
        colornorm_kwargs : dict, optional
            Extra keyword arguments forwarded to the selected normalization constructor.
        k0_normalization : int or float, default=0
            k≈0 mode suppression intensity (0=disabled, 1-10=increasing strength)
        k0_normalization_width : int, default=1
            Number of k-bins around k≈0 to compress
        compression_mode : str, default="adaptive"
            Compression strategy: "gentle", "adaptive", "aggressive", "preserve_peaks"
        vmin, vmax : float, optional
            Manual color scale limits
        trim_0f : int, optional
            Remove N lowest frequency points from plot (useful when f≈0 has strong artifacts)
        fmax : float, optional
            Maximum frequency to display (in f_units). Frequencies above this will be trimmed.
            Useful for synchronizing Y-axis range when using sharey=True with other plots.
        add_comsol_points : str | Path | None, optional
            Path to COMSOL data file for overlay
        comsol_k_col, comsol_f_col : int
            Column indices for k and f in COMSOL file
        comsol_extra_cols : tuple[int, ...] | None
            Additional COMSOL columns to parse
        comsol_style : dict, optional
            Scatter plot style for COMSOL overlay
            
        Returns
        -------
        tuple
            (figure, axis) matplotlib objects
            
        Examples
        --------
        >>> # Compute once
        >>> result = job[0].fft.dispersion.compute_1d(axis="x", save_result=True)
        >>> 
        >>> # Plot multiple times with different settings
        >>> fig1, ax1 = job[0].fft.dispersion.plot_result(result, lognorm=False)
        >>> fig2, ax2 = job[0].fft.dispersion.plot_result(result, lognorm=True, vmax=0.01)
        >>> 
        >>> # Or plot on custom axes
        >>> fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))
        >>> job[0].fft.dispersion.plot_result(result, ax=ax_left, k0_normalization=10)
        >>> job[0].fft.dispersion.plot_result(result, ax=ax_right, k0_normalization=0)
        """
        # This method uses the same plotting logic as plot_dispersion but skips compute_1d
        # We'll extract and reuse the plotting code from plot_dispersion
        
        comsol_style = comsol_style or {}
        colornorm_kwargs = dict(colornorm_kwargs or {})
        comsol_data = None
        if add_comsol_points is not None:
            from .comsol import read_data_from_comsol

            comsol_data = read_data_from_comsol(
                add_comsol_points,
                k_col=comsol_k_col,
                f_col=comsol_f_col,
                extra_cols=comsol_extra_cols,
            )

        # Create plot or use provided axes
        if ax is None:
            if dpi is not None:
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            else:
                fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # Prepare axes
        k_axis = result.k_axis
        f_axis = result.f_axis

        spectrum = result.S
        axis_name = result.axis
        orth_label = result.orth_axis_label or ("y" if axis_name == "x" else "x")

        if orth_index is not None:
            if result.S_local is None:
                raise ValueError("Result does not contain local spectra; recompute with avg_over_orthogonal=False")
            if orth_index < 0 or orth_index >= result.S_local.shape[0]:
                raise ValueError(f"orth_index {orth_index} out of range (0..{result.S_local.shape[0]-1})")
            spectrum = result.S_local[orth_index]
            if title is None:
                if result.orth_axis is not None and orth_index < len(result.orth_axis):
                    axis_value = result.orth_axis[orth_index]
                    title = f"Spin-Wave Dispersion {orth_label}={axis_value:g}"
                else:
                    title = f"Spin-Wave Dispersion ({orth_label} index {orth_index})"

        # Remove negative frequencies from visualization
        if f_axis.ndim == 1 and spectrum.shape[1] == f_axis.shape[0]:
            positive_mask = f_axis >= 0
            if np.any(positive_mask) and positive_mask.sum() < f_axis.size:
                spectrum = spectrum[:, positive_mask]
                f_axis = f_axis[positive_mask]

        # Trim lowest frequency points if requested
        if trim_0f is not None and trim_0f > 0:
            if f_axis.ndim == 1 and trim_0f < f_axis.shape[0]:
                logger.info(f"Trimming {trim_0f} lowest frequency points from dispersion plot")
                spectrum = spectrum[:, trim_0f:]
                f_axis = f_axis[trim_0f:]
            else:
                logger.warning(f"trim_0f={trim_0f} exceeds available frequency points ({f_axis.shape[0]}), ignoring")

        # Trim frequencies above fmax if requested (applied BEFORE unit conversion)
        if fmax is not None and fmax > 0:
            if f_axis.ndim == 1:
                # f_axis is still in Hz at this point, convert fmax to Hz
                if f_units == "GHz":
                    fmax_hz = fmax * 1e9
                else:  # Hz
                    fmax_hz = fmax
                
                fmax_mask = f_axis <= fmax_hz
                n_above = (~fmax_mask).sum()
                if np.any(fmax_mask):
                    spectrum = spectrum[:, fmax_mask]
                    f_axis = f_axis[fmax_mask]
                    logger.info(f"Trimmed {n_above} frequency points above fmax={fmax} {f_units}")
                else:
                    logger.warning(f"fmax={fmax} {f_units} is below all frequencies, ignoring")

        # Convert units if requested
        kscale = kscale.lower()
        if kscale == "meter":
            k_axis = k_axis / (2 * np.pi)
            k_label = r"$k$ [m$^{-1}$]"
            default_k_xlim = (-20.0, 20.0)
        elif kscale == "rad_um":
            k_axis = k_axis / 1e6  # Convert rad/m to rad/μm
            k_label = r"$k$ [rad/μm]"
            default_k_xlim = (-10.0, 10.0)
        else:  # kscale == "rad"
            k_label = r"$k$ [rad/m]"
            default_k_xlim = None

        if f_units == "GHz":
            f_axis = f_axis / 1e9
            f_label = "Frequency [GHz]"
        else:
            f_label = "Frequency [Hz]"

        if k0_normalization and k0_normalization > 0:
            logger.info(
                "Applying k≈0 dynamic compression: strength=%s, mode=%s, width=%s",
                k0_normalization,
                compression_mode,
                k0_normalization_width,
            )
            original_k_axis = result.k_axis
            spectrum = self._apply_k0_normalization(
                spectrum,
                original_k_axis,
                strength=k0_normalization,
                compression_mode=compression_mode,
                k0_normalization_width=k0_normalization_width,
            )

        # Plot dispersion
        norm = self._resolve_colornorm(
            spectrum,
            lognorm_flag=lognorm,
            vmin=vmin,
            vmax=vmax,
            colornorm=colornorm,
            colornorm_kwargs=colornorm_kwargs,
            context="plot_result",
        )

        extent = (
            float(k_axis[0]),
            float(k_axis[-1]),
            float(f_axis[0]),
            float(f_axis[-1]),
        )

        im = ax.imshow(
            spectrum.T,
            cmap=cmap,
            norm=norm,
            aspect="auto",
            origin="lower",
            extent=extent,
        )

        # Formatting
        ax.set_xlabel(k_label)
        ax.set_ylabel(f_label)

        if fmax is not None and fmax > 0:
            y_min = float(np.min(f_axis))
            y_max = float(np.max(f_axis))
            ax.set_ylim(y_min, y_max)

        if k_xlim is not None:
            ax.set_xlim(*k_xlim)
        elif default_k_xlim is not None:
            ax.set_xlim(*default_k_xlim)

        if title is None:
            title = f"Spin-Wave Dispersion S(k{axis_name}, f)"
            if hasattr(result, "component"):
                title += f" - {result.component} component"
        ax.set_title(title)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(r"Power Spectral Density [arb. units]")
        
        if comsol_data is not None:
            k_points = np.asarray(comsol_data.k_values, dtype=float).copy()
            f_points = np.asarray(comsol_data.f_values, dtype=float).copy()

            if getattr(result, "flipx", True):
                k_points = -k_points
                logger.info("COMSOL k-values flipped (flipx=True in dispersion result)")

            if kscale == "rad_um":
                k_points = k_points / 1e6
            elif kscale == "meter":
                k_points = k_points / (2 * np.pi)

            if f_units.lower() == "ghz":
                f_points = f_points / 1e9
            elif f_units.lower() != "hz":
                raise ValueError(f"Unsupported frequency units: {f_units}")

            mask = np.isfinite(k_points) & np.isfinite(f_points)
            if not np.any(mask):
                logger.warning(
                    "COMSOL overlay skipped: no finite data points (%s)",
                    add_comsol_points,
                )
            else:
                if not np.all(mask):
                    logger.warning(
                        "COMSOL overlay dropping %d invalid points",
                        int((~mask).sum()),
                    )
                k_points = k_points[mask]
                f_points = f_points[mask]

                scatter_kwargs = {
                    "s": 40,
                    "facecolors": "none",
                    "edgecolors": "white",
                    "linewidths": 1.5,
                    "alpha": 0.9,
                }
                scatter_kwargs.update(comsol_style)

                logger.info(
                    "Overlaying %d COMSOL points on dispersion plot",
                    k_points.size,
                )

                ax.scatter(k_points[::-1], f_points, label="COMSOL", **scatter_kwargs)

        try:
            fig.tight_layout()
        except Exception:
            pass  # Skip tight_layout if it conflicts with existing colorbar

        # Save if requested
        save_path = None
        if save not in (None, False):
            save_path = self._resolve_plot_save_path(save, axis_name, result)

        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi or 300, bbox_inches="tight")
            logger.info("Dispersion plot saved to %s", save_path)

        return fig, ax
    
    def interactive_analysis(self):
        """
        Launch interactive dispersion analysis (future implementation).
        
        Similar to interactive_spectrum() for modes.
        """
        raise NotImplementedError(
            "Interactive dispersion analysis not yet implemented. "
            "Use plot_dispersion() for now."
        )
    
    def __repr__(self) -> str:
        """Rich documentation display for dispersion interface."""
        try:
            return self._rich_dispersion_display()
        except Exception:
            return self._basic_dispersion_display()

    def _rich_dispersion_display(self) -> str:
        """Render contextual dispersion help using rich panels."""
        import io

        from rich.columns import Columns
        from rich.console import Console
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.table import Table
        from rich.text import Text

        console = Console(file=io.StringIO(), width=120, force_terminal=True)

        job_result = self.parent_fft.job_result
        dataset = self.dataset_name
        slice_str = f"[{self.slice_info}]" if self.slice_info else ""
        config_state = "custom" if self._config else "default"
        analyzer_state = "initialized" if self._analyzer else "lazy"

        summary = Text()
        summary.append("🌊 FFT Dispersion Interface\n", style="bold cyan")
        summary.append(f"📁 Job: {getattr(job_result, 'name', 'unknown')}\n", style="dim")
        if dataset:
            summary.append(f"📊 Dataset: {dataset}{slice_str}\n", style="bold green")
        if self.slice_info:
            summary.append("✂️  Slicing active\n", style="yellow")
        summary.append(f"⚙️  Config: {config_state}\n", style="dim")
        summary.append(f"🧮 Analyzer: {analyzer_state}\n", style="dim")
        if self._filters_config:
            filters_label = ", ".join(self._describe_filter_flags())
            summary.append(f"🧽 Filters: {filters_label}\n", style="yellow")

        core_methods = Text()
        core_methods.append("🔧 Core Methods:\n", style="bold yellow")
        methods = [
            (
                "compute_1d(axis='x', component=None, kmax=None, **opts)",
                "1D dispersion S(k,f) (opts → averaging, windows, folding)",
            ),
            ("compute_2d(component=None, **opts)", "2D dispersion S(kx, ky, f)"),
            (
                "plot_dispersion(axis='x', lognorm=False, kscale='meter', **opts)",
                "Heatmap + auto compute (opts forwarded to compute_1d)",
            ),
            (
                "filters(remove_static=False, average=False, window=None)",
                "Clone interface with preprocessing filters (Hann/time averages)",
            ),
            ("track_branch(result, k_path, f_seed, **opts)", "Follow dispersion branch"),
            ("find_peaks(result, min_prominence=0.0, **opts)", "Detect spectral peaks"),
            ("configure(dt=..., dx=..., dy=..., component='perp', **opts)", "Set defaults once"),
        ]
        for signature, desc in methods:
            core_methods.append("  • ", style="dim")
            core_methods.append(signature, style="code")
            core_methods.append(f" - {desc}\n", style="dim")

        analyzer_info = Text()
        analyzer_info.append("🧪 Analyzer Settings:\n", style="bold magenta")
        analyzer_info.append("  • analyzer property → SpinWaveAnalyzer\n", style="dim")
        analyzer_info.append("  • configure(...) overrides dt/dx/dy/component\n", style="dim")
        analyzer_info.append("  • avg_over_orthogonal toggle via kwargs\n", style="dim")
        if dataset:
            analyzer_info.append(
                f"  • dataset pre-selected: {dataset}\n",
                style="green",
            )
        if self.slice_info is not None:
            analyzer_info.append(
                "  • local spectra available via result.S_local\n",
                style="yellow",
            )

        compute_table = Table(
            show_header=True,
            header_style="bold cyan",
            box=None,
            pad_edge=False,
        )
        compute_table.add_column("argument", style="cyan", no_wrap=True)
        compute_table.add_column("default", style="green")
        compute_table.add_column("description", style="white")
        compute_table.add_row("axis", "'x'", "Propagation axis ('x' | 'y')")
        compute_table.add_row("component", "config.component", "Magnetization component ('perp', 'mx', ...)")
        compute_table.add_row("avg_over_orthogonal", "config.avg_over_orthogonal", "Average orthogonal plane (False keeps slices)")
        compute_table.add_row(
            "orthogonal_avg_mode",
            "config.orthogonal_avg_mode",
            "Collapse strategy: 'magnetization', 'fft_power', 'fft_abs', ...",
        )
        compute_table.add_row("time_window", "config.time_window", "Time-domain window ('hann', None, ...)")
        compute_table.add_row("space_window", "config.space_window", "Spatial window before FFT")
        compute_table.add_row("detrend", "config.detrend", "Detrend strategy ('mean', 'initial', None)")
        compute_table.add_row("fold_period", "config.fold_period", "Real-space period for Brillouin folding")
        compute_table.add_row("fold_agg", "config.fold_agg", "Folding aggregation ('sum' | 'max')")
        compute_table.add_row("force", "False", "Force recomputation and overwrite caches")
        compute_table.add_row("save_result", "False", "Persist dispersion to zarr (alias: save)")
        compute_table.add_row("use_cache", "True", "Use in-memory cache when available")
        compute_table.add_row("disk_cache", "True", "Allow loading from on-disk cache")
        compute_table.add_row("kmax", "None", "Trim |k| beyond limit (rad/m) after compute")
        compute_table.add_row("**kwargs", "", "Forward extra options to SpinWaveAnalyzer")

        filters_table = Table(
            show_header=True,
            header_style="bold yellow",
            box=None,
            pad_edge=False,
        )
        filters_table.add_column("argument", style="yellow", no_wrap=True)
        filters_table.add_column("default", style="green")
        filters_table.add_column("description", style="white")
        filters_table.add_row("remove_static", "False", "Subtract first time frame from all samples")
        filters_table.add_row("average", "False", "Remove temporal mean per spatial point")
        filters_table.add_row(
            "window",
            "None",
            "Hann window selection: 'time', 'space'/'2d', 'both'/'hann'",
        )

        plot_table = Table(
            show_header=True,
            header_style="bold magenta",
            box=None,
            pad_edge=False,
        )
        plot_table.add_column("argument", style="magenta", no_wrap=True)
        plot_table.add_column("default", style="green")
        plot_table.add_column("description", style="white")
        plot_table.add_row("axis", "'x'", "Propagation axis for compute_1d fallback")
        plot_table.add_row("component", "None", "Overrides component when auto-computing")
        plot_table.add_row("result", "None", "Use existing DispersionResult1D when provided")
        plot_table.add_row("figsize", "(12, 8)", "Matplotlib figure size (inches)")
        plot_table.add_row("dpi", "None", "Figure resolution override")
        plot_table.add_row("cmap", "'cmc.davos'", "Colormap for heatmap plot (crameri davos)")
        plot_table.add_row("lognorm", "False", "Use logarithmic color scaling")
        plot_table.add_row("colornorm", "None", "Advanced normalization: lognorm, symlognorm, powernorm, ...")
        plot_table.add_row("colornorm_kwargs", "None", "Extra kwargs for selected normalization")
        plot_table.add_row("vmin", "None", "Manual minimum for color scale normalization")
        plot_table.add_row("vmax", "None", "Manual maximum for color scale normalization")
        plot_table.add_row("k0_normalization", "0", "k≈0 compression strength: 0=off, 1-10=increasing suppression")
        plot_table.add_row("kscale", "'rad_um'", "Wave-vector units: 'rad_um' (rad/μm) | 'rad' | 'meter'")
        plot_table.add_row("f_units", "'GHz'", "Frequency axis units ('Hz' | 'GHz')")
        plot_table.add_row("orth_index", "None", "Select orthogonal slice when available")
        plot_table.add_row("k_xlim", "None", "Manual x-limit tuple (k_min, k_max)")
        plot_table.add_row("save", "None", "Path/PathLike or True for auto-named PNG, False disables")
        plot_table.add_row("save_result", "None", "Forwarded to compute_1d → persist dispersion cache")
        plot_table.add_row("title", "None", "Custom plot title")
        plot_table.add_row("**kwargs", "", "Forwarded to compute_1d (supports kmax, force, save_result, ...)")

        obj_name = "job[0].m_layer.fft" if dataset else "job[0].fft"
        example_lines = [
            "# Quick start",
            f"disp = {obj_name}.dispersion",
            "disp.configure(dt=1e-12, dx=5e-9)",
            "result = disp.compute_1d(axis='x', avg_over_orthogonal=False)",
            "disp.plot_dispersion(result=result, orth_index=0)",
            "",
            "# Branch tracking",
            "k_path = np.linspace(-1e7, 1e7, 41)",
            "branch = disp.track_branch(result, k_path, f_seed=5e9)",
            "disp.plot_branch(branch)",
        ]
        example_code = "\n".join(example_lines)
        syntax = Syntax(example_code, "python", theme="monokai", background_color="default")

        with console.capture() as capture:
            console.print(
                Panel.fit(
                    summary,
                    title="[bold cyan]Dispersion Overview[/bold cyan]",
                    border_style="cyan",
                )
            )
            console.print("")
            console.print(
                Columns(
                    [
                        Panel.fit(
                            core_methods,
                            title="[bold yellow]Toolkit[/bold yellow]",
                            border_style="yellow",
                        ),
                        Panel.fit(
                            analyzer_info,
                            title="[bold magenta]Analyzer[/bold magenta]",
                            border_style="magenta",
                        ),
                    ]
                )
            )
            console.print("")
            console.print(
                Panel(
                    syntax,
                    title="[bold green]Usage Examples[/bold green]",
                    border_style="green",
                )
            )
            console.print("")
            console.print(
                Columns(
                    [
                        Panel(
                            compute_table,
                            title="[bold cyan]compute_1d options[/bold cyan]",
                            border_style="cyan",
                        ),
                        Panel(
                            plot_table,
                            title="[bold magenta]plot_dispersion options[/bold magenta]",
                            border_style="magenta",
                        ),
                        Panel(
                            filters_table,
                            title="[bold yellow]filters() options[/bold yellow]",
                            border_style="yellow",
                        ),
                    ],
                    equal=True,
                )
            )

        return capture.get()

    def _basic_dispersion_display(self) -> str:
        """Fallback plain-text description."""
        dataset_line = (
            f"Dataset: {self.dataset_name}\n" if self.dataset_name else "Dataset: default\n"
        )
        slice_line = (
            f"Slice: {self.slice_info}\n" if self.slice_info is not None else ""
        )
        config_state = "custom" if self._config else "default"

        return (
            "FFTDispersionInterface\n"
            f"{dataset_line}{slice_line}"
            f"Config: {config_state}\n"
            "Available methods:\n"
            "  • compute_1d(axis='x'|'y', component=None, kmax=None, **opts)\n"
            "  • compute_2d()\n"
            "  • plot_dispersion(axis='x', lognorm=False, kscale='meter', **opts)\n"
            "  • filters(remove_static=False, average=False, window=None)\n"
            "  • track_branch(result, k_path, f_seed)\n"
            "  • find_peaks(result, min_prominence=0.1)\n"
            "  • configure(dt=..., dx=..., dy=..., component='perp')\n"
            "compute_1d opts: avg_over_orthogonal, time_window, space_window, detrend, fold_period, fold_agg\n"
            "plot_dispersion opts: result, figsize, dpi, cmap, kscale, f_units, lognorm, orth_index, k_xlim, save(True→auto path), title\n"
            "filters opts: remove_static, average (time mean), window=('time'|'space'/'2d'|'both') Hann\n"
            "Extra kwargs forward to compute_1d (supports kmax, window controls, folding, etc.)\n"
            "Examples:\n"
            f"  disp = {('job[0].m_layer.fft' if self.dataset_name else 'job[0].fft')}.dispersion\n"
            "  result = disp.compute_1d(axis='x', avg_over_orthogonal=False)\n"
            "  branch = disp.track_branch(result, k_path, f_seed=5e9)\n"
            "  disp.plot_branch(branch)\n"
        )
