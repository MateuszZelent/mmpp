"""Internal helpers for FFT input loading and z-layer normalization."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InputLoadMetrics:
    """Profiling metrics for FFT input loading."""

    load_time: float
    data_size_mb: float
    data_size_gb: float
    loading_speed_mbps: float | None
    memory_before_mb: float | None
    memory_after_mb: float | None
    memory_used_mb: float | None


def _resolve_dataset(
    *,
    job: Any,
    zarr_path: str,
    dataset: str,
    logger: Any,
) -> Any:
    """Resolve dataset from Pyzfn job or direct zarr access."""
    data_set = None
    if hasattr(job, dataset):
        data_set = getattr(job, dataset)
    else:
        try:
            import zarr

            z_root = zarr.open(zarr_path, mode="r")
            if dataset in z_root:
                data_set = z_root[dataset]
            else:
                logger.debug(
                    "Dataset %s not found in zarr root, checking if it's an attribute of Pyzfn job",
                    dataset,
                )
        except Exception as exc:
            logger.debug("Could not access zarr directly: %s", exc)

    if data_set is None:
        available = []
        try:
            import zarr

            z_root = zarr.open(zarr_path, mode="r")
            available.extend(list(z_root.group_keys()))
            available.extend(list(z_root.array_keys()))
            available = sorted({key.split("/")[0] for key in available})
        except Exception as exc:
            logger.debug("Unable to enumerate datasets in %s: %s", zarr_path, exc)

        suggestion = (
            f" Available datasets: {', '.join(available)}" if available else ""
        )
        raise ValueError(
            f"Dataset '{dataset}' not found in zarr file '{zarr_path}'.{suggestion}"
        )

    return data_set


def _apply_slice_with_time_policy(
    *,
    data_set: Any,
    slice_info: Any | None,
    tmax: int | None,
    logger: Any,
) -> tuple[np.ndarray, bool]:
    """Apply user slicing and decide whether tmax should still be applied."""
    apply_tmax = tmax is not None and tmax > 0

    if slice_info is not None:
        logger.info("Applying slice_info: %s", slice_info)
        data = data_set[slice_info]

        # Restore dropped axis when integer indexing removed last dimension.
        if isinstance(slice_info, tuple) and len(slice_info) > 0:
            if isinstance(slice_info[-1], int):
                data = data[..., np.newaxis]
                logger.debug("Restored dropped dimension: new shape %s", data.shape)

        if isinstance(slice_info, tuple) and len(slice_info) > 0:
            first_slice = slice_info[0]
            if first_slice is not Ellipsis:
                if isinstance(first_slice, slice):
                    apply_tmax = False
                    if first_slice.stop is not None:
                        logger.debug(
                            "User provided explicit time slice %s - tmax parameter will be ignored",
                            first_slice,
                        )
                    else:
                        logger.debug(
                            "User provided [:] slice - using ALL timesteps (ignoring tmax)"
                        )
                elif isinstance(first_slice, int):
                    apply_tmax = False
    else:
        data = data_set[...]

    return data, apply_tmax


def _apply_tmax(
    *,
    data: np.ndarray,
    tmax: int | None,
    apply_tmax: bool,
    logger: Any,
) -> np.ndarray:
    """Apply tmax truncation if allowed by slice policy."""
    if not apply_tmax or tmax is None:
        return data

    original_time_steps = data.shape[0] if len(data.shape) > 0 else 0
    if tmax < original_time_steps:
        data = data[:tmax]
        logger.info(
            "Applied tmax=%s: reduced from %s to %s time steps",
            tmax,
            original_time_steps,
            tmax,
        )
    else:
        logger.info(
            "tmax=%s >= data length (%s), no truncation applied",
            tmax,
            original_time_steps,
        )

    return data


def _select_z_layer(
    *,
    data: np.ndarray,
    z_layer: int,
    slice_info: Any | None,
    logger: Any,
) -> np.ndarray:
    """Select z-layer while handling ambiguous 4D cases."""
    original_ndim = len(data.shape)

    if original_ndim == 5:  # (t, z, y, x, comp)
        if z_layer == -1:
            data = data[:, -1, :, :, :]
            logger.debug("Selected last z-layer from 5D data")
        else:
            data = data[:, z_layer, :, :, :]
            logger.debug("Selected z-layer %s from 5D data", z_layer)
        return data

    if original_ndim == 4:
        component_was_selected = False
        if slice_info is not None and isinstance(slice_info, tuple):
            non_ellipsis_slices = [s for s in slice_info if s is not Ellipsis]
            if non_ellipsis_slices and isinstance(
                non_ellipsis_slices[-1], (int, np.integer)
            ):
                component_was_selected = True
                logger.debug(
                    "Detected component selection in slice - treating 4D as (t,z,y,x)"
                )

        if component_was_selected:
            if z_layer == -1:
                data = data[:, -1, :, :]
                logger.debug(
                    "Selected last z-layer from 4D data (component pre-selected)"
                )
            else:
                data = data[:, z_layer, :, :]
                logger.debug(
                    "Selected z-layer %s from 4D data (component pre-selected)",
                    z_layer,
                )
        else:
            logger.debug("No z-dimension in 4D data (assuming t,y,x,comp)")
        return data

    if original_ndim == 3:
        logger.debug(
            "3D dataset detected - using provided dimensions without z-layer selection"
        )
        return data
    if original_ndim == 2:
        logger.debug("2D dataset detected - interpreting first axis as time")
        return data
    if original_ndim == 1:
        logger.debug("1D time series detected")
        return data

    raise ValueError(f"Unsupported data shape: {data.shape}")


def _resolve_dt(*, data_set: Any, job: Any, logger: Any) -> float:
    """Resolve timestep with dataset-specific attributes first."""
    dt = None
    try:
        if hasattr(data_set, "attrs") and "t" in data_set.attrs:
            t_attr = data_set.attrs["t"]
            if hasattr(t_attr, "__len__") and len(t_attr) >= 2:
                dt = float(t_attr[1] - t_attr[0])
                logger.debug("Using dt from data_set.attrs['t']: %s", dt)

        if dt is None and hasattr(data_set, "dt"):
            dt = data_set.dt
            logger.debug("Using dt from data_set.dt property: %s", dt)

        if dt is None and hasattr(data_set, "attrs") and "t_sampl" in data_set.attrs:
            dt = data_set.attrs["t_sampl"]
            logger.debug("Using dt from data_set.attrs['t_sampl']: %s", dt)

        if dt is None and hasattr(job, "attrs") and "t_sampl" in job.attrs:
            dt = job.attrs["t_sampl"]
            logger.warning(
                "Using dt from job.attrs['t_sampl']: %s (dataset-specific dt not found)",
                dt,
            )

        if dt is None:
            dt = 1e-12
            logger.warning("t_sampl not found in attrs, using default: %s", dt)
    except (AttributeError, TypeError, IndexError) as exc:
        logger.warning("Could not determine dt: %s, using default", exc)
        dt = 1e-12

    return dt


def load_fft_input_data(
    *,
    zarr_path: str,
    dataset: str,
    z_layer: int,
    tmax: int | None,
    slice_info: Any | None,
    pyzfn_available: bool,
    pyzfn_cls: Any,
    psutil_module: Any | None,
    logger: Any,
) -> tuple[np.ndarray, float]:
    """Load FFT input data from zarr with slicing, z-layer handling, and dt detection."""
    start_time = time.time()
    process = None
    initial_memory = None

    if psutil_module is not None:
        process = psutil_module.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024

    if not pyzfn_available:
        raise ImportError(
            "pyzfn is required to load FFT input data. Install pyzfn before running FFT analysis."
        )

    logger.info("Loading data from zarr: %s", zarr_path)
    logger.debug("Dataset: %s, z_layer: %s, tmax: %s", dataset, z_layer, tmax)

    try:
        job = pyzfn_cls(zarr_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to open zarr job at {zarr_path}: {exc}") from exc

    data_set = _resolve_dataset(job=job, zarr_path=zarr_path, dataset=dataset, logger=logger)

    data_load_start = time.time()
    data, apply_tmax = _apply_slice_with_time_policy(
        data_set=data_set,
        slice_info=slice_info,
        tmax=tmax,
        logger=logger,
    )
    data_load_time = time.time() - data_load_start
    logger.debug("Data loading time: %.3fs", data_load_time)

    data = _apply_tmax(data=data, tmax=tmax, apply_tmax=apply_tmax, logger=logger)

    data_size_mb = data.nbytes / 1024 / 1024
    loading_speed = data_size_mb / data_load_time if data_load_time > 0 else 0
    logger.debug("Data size: %.1f MB", data_size_mb)
    logger.debug("Loading speed: %.1f MB/s", loading_speed)

    layer_select_start = time.time()
    data = _select_z_layer(
        data=data,
        z_layer=z_layer,
        slice_info=slice_info,
        logger=logger,
    )
    layer_select_time = time.time() - layer_select_start
    logger.debug("Layer selection time: %.3fs", layer_select_time)

    dt = _resolve_dt(data_set=data_set, job=job, logger=logger)

    total_time = time.time() - start_time
    if process is not None and initial_memory is not None:
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        logger.debug("Memory increase: %.1f MB", memory_increase)

    logger.info("Data loaded successfully in %.3fs, shape: %s", total_time, data.shape)

    return data, dt


def load_fft_input_data_profiled(
    *,
    zarr_path: str,
    dataset: str,
    z_layer: int,
    tmax: int | None,
    slice_info: Any | None,
    pyzfn_available: bool,
    pyzfn_cls: Any,
    psutil_module: Any | None,
    logger: Any,
) -> tuple[np.ndarray, float, InputLoadMetrics]:
    """Load FFT input data and collect timing/memory metrics."""
    process = None
    memory_before = None
    if psutil_module is not None:
        try:
            process = psutil_module.Process()
            memory_before = process.memory_info().rss / 1024 / 1024
        except Exception:
            process = None
            memory_before = None

    load_start_time = time.time()
    data, dt = load_fft_input_data(
        zarr_path=zarr_path,
        dataset=dataset,
        z_layer=z_layer,
        tmax=tmax,
        slice_info=slice_info,
        pyzfn_available=pyzfn_available,
        pyzfn_cls=pyzfn_cls,
        psutil_module=psutil_module,
        logger=logger,
    )
    load_time = time.time() - load_start_time

    data_size_mb = data.nbytes / 1024 / 1024
    data_size_gb = data_size_mb / 1024
    loading_speed_mbps = data_size_mb / load_time if load_time > 0 else None

    memory_after = None
    memory_used = None
    if process is not None and memory_before is not None:
        try:
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_used = memory_after - memory_before
        except Exception:
            memory_after = None
            memory_used = None

    metrics = InputLoadMetrics(
        load_time=float(load_time),
        data_size_mb=float(data_size_mb),
        data_size_gb=float(data_size_gb),
        loading_speed_mbps=(
            float(loading_speed_mbps) if loading_speed_mbps is not None else None
        ),
        memory_before_mb=(float(memory_before) if memory_before is not None else None),
        memory_after_mb=(float(memory_after) if memory_after is not None else None),
        memory_used_mb=(float(memory_used) if memory_used is not None else None),
    )
    return data, dt, metrics


def log_input_load_metrics(
    *,
    logger: Any,
    data: np.ndarray,
    dt: float,
    metrics: InputLoadMetrics,
) -> None:
    """Emit consistent logging for profiled FFT input loading."""
    logger.info("Data shape: %s, dt: %s", data.shape, dt)
    logger.debug("⏱️  Data loading time: %.3fs", metrics.load_time)
    logger.debug(
        "💾 Data size: %.1f MB (%.2f GB)",
        metrics.data_size_mb,
        metrics.data_size_gb,
    )

    if (
        metrics.memory_before_mb is not None
        and metrics.memory_after_mb is not None
        and metrics.memory_used_mb is not None
    ):
        logger.debug(
            "🧠 Memory usage change: %+.1f MB (before: %.1f MB, after: %.1f MB)",
            metrics.memory_used_mb,
            metrics.memory_before_mb,
            metrics.memory_after_mb,
        )
    else:
        logger.debug("🧠 Memory monitoring unavailable (install psutil for memory stats)")

    if metrics.loading_speed_mbps is not None:
        logger.debug("🚀 Loading speed: %.1f MB/s", metrics.loading_speed_mbps)


def normalize_z_layer_index(
    *,
    zarr_path: str,
    dataset: str,
    z_layer: int,
    pyzfn_available: bool,
    pyzfn_cls: Any,
    logger: Any,
) -> int:
    """Normalize z-layer index to a concrete positive index when possible."""
    try:
        if not pyzfn_available:
            raise ImportError("pyzfn required for data shape inspection")

        temp_job = pyzfn_cls(zarr_path)
        temp_data_set = None
        if hasattr(temp_job, dataset):
            temp_data_set = getattr(temp_job, dataset)
        else:
            z_group = getattr(temp_job, "z", None)
            if z_group is not None and dataset in z_group:
                temp_data_set = z_group[dataset]

        if temp_data_set is None:
            try:
                import zarr

                z_root = zarr.open(zarr_path, mode="r")
                if dataset in z_root:
                    temp_data_set = z_root[dataset]
                    logger.debug("Found dataset '%s' via direct zarr access", dataset)
            except Exception:
                pass

        if temp_data_set is not None:
            data_shape = temp_data_set.shape
            if len(data_shape) == 5 and z_layer == -1:
                normalized = data_shape[1] - 1
                logger.debug(
                    "Normalized z_layer=%s to %s (shape: %s)",
                    z_layer,
                    normalized,
                    data_shape,
                )
                return normalized
            if len(data_shape) == 5 and z_layer < -1:
                normalized = data_shape[1] + z_layer
                logger.debug(
                    "Normalized negative z_layer=%s to %s (shape: %s)",
                    z_layer,
                    normalized,
                    data_shape,
                )
                return normalized
            return z_layer

        logger.debug("Dataset '%s' not found for shape inspection, using z_layer as-is", dataset)
        return z_layer
    except Exception as exc:
        logger.warning("Failed to normalize z_layer: %s, using z_layer as-is", exc)
        return z_layer
