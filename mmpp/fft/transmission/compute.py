"""Transmission analysis core utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from ..compute_fft import FFTCompute, FILTER_TYPES, WINDOW_TYPES

from ...cli.logging_config import get_mmpp_logger


log = get_mmpp_logger("mmpp.fft.transmission")


TransmissionMethod = Literal["power_ratio", "circular", "cpsd"]
AverageMode = Literal["mean", "median", "edge_taper"]
NormalizeMode = Literal["reference", "max", "none"]
ReferenceStatistic = Literal["mean", "median", "max"]


@dataclass(slots=True)
class TransmissionConfig:
    """Configuration parameters for transmission analysis."""

    dataset_name: Optional[str] = None
    z_layer: int = -1
    method: TransmissionMethod = "power_ratio"
    window_function: WINDOW_TYPES = "hann"
    filter_type: FILTER_TYPES = "remove_mean"
    spatial_window: int = 5
    spatial_step: int = 1
    reference_window: Optional[Tuple[int, int]] = None
    reference_statistic: ReferenceStatistic = "mean"
    average_mode: AverageMode = "mean"
    edge_taper_power: float = 1.5
    component_weights: Tuple[float, float, float] = (1.0, 1.0, 0.1)
    enable_circular_components: bool = False
    normalize: NormalizeMode = "reference"
    tmax: Optional[int] = None
    keep_complex_fft: bool = False
    store_component_maps: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def ensure_valid(self) -> None:
        """Validate configuration values."""
        if self.spatial_window <= 0:
            raise ValueError("spatial_window must be > 0")
        if self.spatial_step <= 0:
            raise ValueError("spatial_step must be > 0")
        if self.reference_window is not None:
            start, stop = self.reference_window
            if stop < start:
                raise ValueError("reference_window stop must be >= start")
        if len(self.component_weights) != 3:
            raise ValueError("component_weights must contain three entries for (mx, my, mz)")


@dataclass(slots=True)
class TransmissionResult:
    """Result of a transmission analysis."""

    frequencies: np.ndarray
    x_positions: np.ndarray
    transmission: np.ndarray
    power_map: np.ndarray
    reference_power: np.ndarray
    config: TransmissionConfig
    metadata: Dict[str, Any] = field(default_factory=dict)
    power_plus: Optional[np.ndarray] = None
    power_minus: Optional[np.ndarray] = None
    transverse_power: Optional[np.ndarray] = None
    longitudinal_power: Optional[np.ndarray] = None

    def plot_transmission(self, plot_config=None, **kwargs):
        """Render a frequency-position transmission map.

        Accepts a `plot_config` which may be a mapping (dict) or a
        :class:`TransmissionPlotConfig`. Any additional plotting kwargs
        (e.g., dpi, ax) are forwarded to the plotter.
        """
        # Import here to avoid circular imports at module import time
        from .plot import TransmissionPlotter, TransmissionPlotConfig

        # Convert dict -> TransmissionPlotConfig when needed (same behaviour as FFT interface)
        if plot_config is not None and isinstance(plot_config, dict):
            plot_config = TransmissionPlotConfig(**plot_config)

        plotter = TransmissionPlotter(self)
        return plotter.plot(config=plot_config, **kwargs)


def _compute_hann_weights(length: int, power: float) -> np.ndarray:
    """Create taper weights for averaging across the Y direction."""

    if length <= 1:
        return np.ones((length,), dtype=float)

    window = np.hanning(length)
    window = np.clip(window, 1e-6, None)
    if power != 1.0:
        window = window**power
    window /= window.sum()
    return window


def _aggregate_spatial(
    power: np.ndarray,
    mode: AverageMode,
    edge_taper_power: float,
) -> np.ndarray:
    """Reduce spatial dimensions (z, y, window_x) of the local power map."""

    if power.ndim != 4:
        raise ValueError("Expected power array with shape (freq, z, y, window)")

    freq_axis = 0
    z_axis = 1
    y_axis = 2
    x_axis = 3

    if mode == "mean":
        return power.mean(axis=(z_axis, y_axis, x_axis))

    if mode == "median":
        return np.median(power, axis=(z_axis, y_axis, x_axis))

    if mode == "edge_taper":
        n_z, n_y, n_w = power.shape[1:]
        weights_y = _compute_hann_weights(n_y, edge_taper_power)
        weights_z = np.ones((n_z,), dtype=float)
        weights_z /= weights_z.sum() if weights_z.sum() > 0 else 1.0
        weights_w = np.ones((n_w,), dtype=float)
        weights_w /= weights_w.sum() if weights_w.sum() > 0 else 1.0

        combined = (
            weights_z[:, None, None]
            * weights_y[None, :, None]
            * weights_w[None, None, :]
        )
        weighted = power * combined[None, ...]
        normalization = combined.sum()
        if normalization <= 0:
            normalization = 1.0
        return weighted.sum(axis=(z_axis, y_axis, x_axis)) / normalization

    raise ValueError(f"Unsupported averaging mode: {mode}")


class TransmissionCompute:
    """Compute transmission profiles for FFT datasets."""

    def __init__(self, fft_compute: FFTCompute, job_result: Any):
        self._fft_compute = fft_compute
        self._job_result = job_result

    def _prepare_data(self, config: TransmissionConfig) -> tuple[np.ndarray, float]:
        dataset = config.dataset_name or self._job_result.get_largest_m_dataset()

        data, dt = self._fft_compute.load_data_from_zarr(
            self._job_result.path,
            dataset,
            z_layer=config.z_layer,
            tmax=config.tmax,
        )

        if data.ndim == 4:  # (t, y, x, comp)
            data = data[:, np.newaxis, ...]
        elif data.ndim == 5:
            pass
        else:
            raise ValueError(
                "Transmission analysis requires 4D (t,y,x,c) or 5D (t,z,y,x,c) datasets"
            )

        if data.shape[-1] < 2:
            raise ValueError(
                "Expected vector magnetization components (>=2) in the last dimension"
            )

        return data, dt

    def compute(self, config: TransmissionConfig) -> TransmissionResult:
        config.ensure_valid()

        dataset = config.dataset_name or self._job_result.get_largest_m_dataset()
        data, dt = self._prepare_data(config)

        n_time, n_z, n_y, n_x, n_comp = data.shape

        filtered = self._fft_compute.apply_filter(data, config.filter_type)
        windowed = self._fft_compute.apply_window(filtered, config.window_function)

        window_size = min(config.spatial_window, n_x)
        step = config.spatial_step

        window_starts = list(range(0, max(n_x - window_size + 1, 1), step))
        if not window_starts:
            window_starts = [0]

        x_centers = np.array(
            [start + (window_size - 1) / 2.0 for start in window_starts],
            dtype=float,
        )

        n_windows = len(window_starts)
        n_freq = n_time // 2 + 1

        freqs = np.fft.rfftfreq(n_time, d=dt)

        power_map = np.zeros((n_freq, n_windows), dtype=float)
        transverse_map = (
            np.zeros((n_freq, n_windows), dtype=float)
            if config.store_component_maps
            else None
        )
        longitudinal_map = (
            np.zeros((n_freq, n_windows), dtype=float)
            if config.store_component_maps and n_comp > 2
            else None
        )

        power_plus = (
            np.zeros((n_freq, n_windows), dtype=float)
            if config.enable_circular_components
            else None
        )
        power_minus = (
            np.zeros((n_freq, n_windows), dtype=float)
            if config.enable_circular_components
            else None
        )

        component_weights = np.asarray(config.component_weights, dtype=float)
        component_weights = component_weights[:n_comp]

        for win_idx, start in enumerate(window_starts):
            end = min(start + window_size, n_x)
            window_slice = slice(start, end)
            block = windowed[:, :, :, window_slice, :]

            spectrum = np.fft.rfft(block, axis=0)

            mx_fft = spectrum[..., 0]
            my_fft = spectrum[..., 1]
            power_components = np.abs(mx_fft) ** 2 * component_weights[0]
            power_components += np.abs(my_fft) ** 2 * component_weights[1]

            if n_comp > 2:
                mz_fft = spectrum[..., 2]
                power_components += np.abs(mz_fft) ** 2 * component_weights[2]
                if longitudinal_map is not None:
                    longitudinal_map[:, win_idx] = _aggregate_spatial(
                        np.abs(mz_fft) ** 2,
                        config.average_mode,
                        config.edge_taper_power,
                    )

            aggregated = _aggregate_spatial(
                power_components,
                config.average_mode,
                config.edge_taper_power,
            )

            power_map[:, win_idx] = aggregated

            if transverse_map is not None:
                transverse_map[:, win_idx] = _aggregate_spatial(
                    np.abs(mx_fft) ** 2 + np.abs(my_fft) ** 2,
                    config.average_mode,
                    config.edge_taper_power,
                )

            if config.enable_circular_components and power_plus is not None and power_minus is not None:
                m_plus = (mx_fft + 1j * my_fft) / np.sqrt(2.0)
                m_minus = (mx_fft - 1j * my_fft) / np.sqrt(2.0)
                power_plus[:, win_idx] = _aggregate_spatial(
                    np.abs(m_plus) ** 2,
                    config.average_mode,
                    config.edge_taper_power,
                )
                power_minus[:, win_idx] = _aggregate_spatial(
                    np.abs(m_minus) ** 2,
                    config.average_mode,
                    config.edge_taper_power,
                )

        reference_mask = self._select_reference_windows(
            x_centers,
            window_size,
            config.reference_window,
        )

        if not np.any(reference_mask):
            reference_mask[0] = True

        reference_values = self._compute_reference(
            power_map,
            reference_mask,
            config.reference_statistic,
        )

        if config.normalize == "reference":
            denom = np.where(reference_values <= 0, 1.0, reference_values)
            transmission = power_map / denom[:, None]
        elif config.normalize == "max":
            denom = np.max(power_map, axis=1, keepdims=True)
            denom = np.where(denom <= 0, 1.0, denom)
            transmission = power_map / denom
        else:
            transmission = power_map.copy()

        metadata = {
            "dataset": dataset,
            "z_layer": config.z_layer,
            "window_size": window_size,
            "window_step": step,
            "time_step": dt,
            "method": config.method,
        }
        metadata.update(config.metadata)

        result = TransmissionResult(
            frequencies=freqs,
            x_positions=x_centers,
            transmission=transmission,
            power_map=power_map,
            reference_power=reference_values,
            config=config,
            metadata=metadata,
            power_plus=power_plus,
            power_minus=power_minus,
            transverse_power=transverse_map,
            longitudinal_power=longitudinal_map,
        )

        return result

    @staticmethod
    def _select_reference_windows(
        x_centers: np.ndarray,
        window_size: int,
        reference_window: Optional[Tuple[int, int]],
    ) -> np.ndarray:
        mask = np.zeros_like(x_centers, dtype=bool)
        if reference_window is None:
            if x_centers.size:
                mask[0] = True
            return mask

        start, stop = reference_window
        mask = (x_centers >= start) & (x_centers <= stop)
        if not np.any(mask) and x_centers.size:
            mask[0] = True
        return mask

    @staticmethod
    def _compute_reference(
        power_map: np.ndarray,
        reference_mask: np.ndarray,
        statistic: ReferenceStatistic,
    ) -> np.ndarray:
        ref_columns = power_map[:, reference_mask]
        if ref_columns.ndim == 1:
            ref_columns = ref_columns[:, None]
        if ref_columns.size == 0:
            return np.ones((power_map.shape[0],), dtype=float)

        if statistic == "mean":
            return np.mean(ref_columns, axis=1)
        if statistic == "median":
            return np.median(ref_columns, axis=1)
        if statistic == "max":
            return np.max(ref_columns, axis=1)

        raise ValueError(f"Unsupported reference statistic: {statistic}")


__all__ = [
    "TransmissionConfig",
    "TransmissionCompute",
    "TransmissionResult",
]
