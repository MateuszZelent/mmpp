"""Batch soliton analysis helpers."""

from __future__ import annotations

import resource
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from mmpp._shared.repr_html import make_simple_card

from .vortex._plotting import (
    apply_axes_style,
    ensure_axis,
    pop_axes_style_kwargs,
    pop_figure_kwargs,
)


def _coerce_numeric(value: Any, default: float = float("nan")) -> float:
    """Convert value to float when possible."""
    try:
        return float(value)
    except Exception:
        return float(default)


def _disk_radius_from_attrs(attrs: dict[str, Any]) -> float:
    """Infer disk radius from common metadata keys."""
    for key in ("D", "diameter", "disk_diameter", "pillar_diameter"):
        if key in attrs:
            diameter = _coerce_numeric(attrs[key])
            if np.isfinite(diameter) and diameter > 0.0:
                return 0.5 * diameter
    return float("nan")


def _coordinate_label(name: str) -> str:
    """Human-readable label for batch coordinate axes."""
    labels = {
        "i_pillar_ma": "Current [mA]",
        "ma": "Current [mA]",
        "Jdc": "Jdc [A/m^2]",
        "ni": "ni",
        "addoe": "addoe",
    }
    return labels.get(name, name)


def _resolved_spectrum_kwargs(
    trajectory,
    *,
    method: str,
    nperseg: int | None,
    noverlap: int | None,
) -> dict[str, Any]:
    """Adapt spectral parameters to the available trajectory length."""
    kwargs: dict[str, Any] = {"method": method}
    sample_count = int(len(getattr(trajectory, "time", [])))

    effective_nperseg = (
        None if nperseg is None else min(int(nperseg), max(sample_count, 1))
    )
    if effective_nperseg is not None:
        kwargs["nperseg"] = effective_nperseg

    if noverlap is not None:
        if effective_nperseg is None:
            kwargs["noverlap"] = int(max(noverlap, 0))
        else:
            kwargs["noverlap"] = int(
                min(max(noverlap, 0), max(effective_nperseg - 1, 0))
            )

    return kwargs


def _progress_iter(iterable, *, total: int, desc: str, enabled: bool):
    """Wrap iterable with tqdm when available and requested."""
    if not enabled or total <= 1:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, unit="result")
    except ImportError:
        return iterable


def _memory_mb() -> float:
    try:
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        return float("nan")


_REGIME_ORDER = [
    "stable_gyro",
    "damped",
    "intermittent",
    "collision",
    "remagnetization",
    "error",
]

_REGIME_COLORS = {
    "stable_gyro": "tab:green",
    "damped": "tab:gray",
    "intermittent": "tab:orange",
    "collision": "tab:red",
    "remagnetization": "tab:purple",
    "error": "black",
}


@dataclass
class BatchVortexSpectrumMapResult:
    """Spectrum matrix across a filtered batch of vortex simulations."""

    coordinate: np.ndarray
    frequencies: np.ndarray
    power: np.ndarray
    component: str
    coordinate_name: str = "i_pillar_ma"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frequencies_ghz(self) -> np.ndarray:
        """Frequency axis in GHz."""
        return np.asarray(self.frequencies, dtype=float) * 1e-9

    @property
    def power_db(self) -> np.ndarray:
        """Power expressed as 10*log10(power)."""
        return 10.0 * np.log10(
            np.clip(np.asarray(self.power, dtype=float), 1e-30, None)
        )

    @property
    def plt(self):
        """Plotting accessor."""
        return BatchVortexSpectrumMapPlotAccessor(self)

    def _repr_html_(self) -> str:
        return make_simple_card(
            title=self.__class__.__name__,
            subtitle=f"{self.component} spectrum map",
            rows=[
                ("n_runs", str(int(np.asarray(self.coordinate).size))),
                ("n_freq", str(int(np.asarray(self.frequencies).size))),
                ("coordinate", str(self.coordinate_name)),
            ],
        )


class BatchVortexSpectrumMapPlotAccessor:
    """Plot helpers for :class:`BatchVortexSpectrumMapResult`."""

    def __init__(self, result: BatchVortexSpectrumMapResult):
        self._result = result

    def heatmap(self, *, ax=None, as_ghz: bool = True, db_scale: bool = True, **kwargs):
        """Plot batch spectrum heatmap."""
        mesh_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(mesh_kwargs)
        figure_kwargs = pop_figure_kwargs(mesh_kwargs)
        colorbar = bool(mesh_kwargs.pop("colorbar", True))
        colorbar_options = mesh_kwargs.pop("colorbar_kwargs", {})
        colorbar_kwargs = {} if colorbar_options is None else dict(colorbar_options)

        ax = ensure_axis(ax, default_figsize=(6.5, 4.0), figure_kwargs=figure_kwargs)

        freqs = (
            self._result.frequencies_ghz
            if as_ghz
            else np.asarray(self._result.frequencies, dtype=float)
        )
        power = (
            self._result.power_db
            if db_scale
            else np.asarray(self._result.power, dtype=float)
        )

        mesh = ax.pcolormesh(
            np.asarray(self._result.coordinate, dtype=float),
            freqs,
            power.T,
            shading="auto",
            **mesh_kwargs,
        )
        ax.set_xlabel(_coordinate_label(self._result.coordinate_name))
        ax.set_ylabel("Frequency [GHz]" if as_ghz else "Frequency [Hz]")
        ax.set_title(f"Batch vortex {self._result.component} spectrum")

        if colorbar:
            label = "Power [dB]" if db_scale else "Power [a.u.]"
            colorbar_kwargs.setdefault("label", label)
            ax.figure.colorbar(mesh, ax=ax, **colorbar_kwargs)

        apply_axes_style(ax, style_kwargs)
        return ax

    def _repr_html_(self) -> str:
        return make_simple_card(
            title=self.__class__.__name__,
            subtitle="plot accessor",
            rows=[("methods", ".heatmap()")],
        )


class BatchSolitonsInterface:
    """Batch entry point for soliton analysis namespaces."""

    def __init__(self, results: list[Any], mmpp_instance: Any | None = None):
        self._results = list(results)
        self._mmpp = mmpp_instance
        self._vortex = None

    @property
    def vortex(self):
        """Batch vortex analysis namespace."""
        if self._vortex is None:
            self._vortex = BatchVortexInterface(self._results, self._mmpp)
        return self._vortex

    def __repr__(self) -> str:
        return f"BatchSolitonsInterface({len(self._results)} results)"

    def _repr_html_(self) -> str:
        return make_simple_card(
            title=self.__class__.__name__,
            subtitle="batch soliton namespace",
            rows=[("n_results", str(len(self._results))), ("namespaces", "vortex")],
        )


class BatchVortexSpectrumAccessor:
    """Batch spectrum namespace for vortex runs."""

    def __init__(self, interface: BatchVortexInterface):
        self._interface = interface

    def map(self, **kwargs) -> BatchVortexSpectrumMapResult:
        return self._interface.spectrum_map(**kwargs)

    def _repr_html_(self) -> str:
        return make_simple_card(
            title=self.__class__.__name__,
            subtitle="batch vortex spectrum namespace",
            rows=[("methods", ".map(), .heatmap()")],
        )


class BatchVortexInterface:
    """Batch helpers for vortex trajectory, spectrum and regime analysis."""

    def __init__(self, results: list[Any], mmpp_instance: Any | None = None):
        self._results = list(results)
        self._mmpp = mmpp_instance
        self._spectrum = None

    @property
    def plt(self):
        """Plotting accessor for batch vortex summaries."""
        return BatchVortexPlotAccessor(self)

    @property
    def spectrum(self):
        """Batch spectrum namespace."""
        if self._spectrum is None:
            self._spectrum = BatchVortexSpectrumAccessor(self)
        return self._spectrum

    def _ordered_results(self, sort_by: str | None = "i_pillar_ma") -> list[Any]:
        if sort_by is None:
            return list(self._results)

        def key(result):
            attrs = getattr(result, "attrs", {}) or {}
            if sort_by in attrs:
                return (0, _coerce_numeric(attrs[sort_by], default=float("inf")))
            return (1, str(getattr(result, "path", "")))

        return sorted(self._results, key=key)

    @staticmethod
    def _classify_regime(
        row: pd.Series,
        *,
        power_floor_rel: float,
        radius_floor_nm: float,
        expulsion_ratio: float,
    ) -> str:
        if row.get("status") == "error":
            return "error"
        if int(row.get("n_p_switch", 0)) > 0:
            return "remagnetization"
        if int(row.get("n_expulsion", 0)) > 0:
            return "collision"
        radius_max_rel = _coerce_numeric(row.get("r_max_rel", np.nan))
        if np.isfinite(radius_max_rel) and radius_max_rel >= float(expulsion_ratio):
            return "collision"
        if int(row.get("n_gc_switch", 0)) > 0:
            return "intermittent"
        peak_power_rel = _coerce_numeric(row.get("peak_power_rel", np.nan))
        radius_mean_nm = _coerce_numeric(row.get("r_mean_nm", np.nan))
        if (
            np.isfinite(peak_power_rel) and peak_power_rel < float(power_floor_rel)
        ) or (np.isfinite(radius_mean_nm) and radius_mean_nm < float(radius_floor_nm)):
            return "damped"
        return "stable_gyro"

    def summary(
        self,
        *,
        sort_by: str | None = "i_pillar_ma",
        steady_state: bool = True,
        spectrum_method: str = "welch",
        nperseg: int | None = 512,
        noverlap: int | None = 256,
        radius_threshold: float = 0.6,
        expulsion_ratio: float = 0.95,
        power_floor_rel: float = 0.02,
        radius_floor_nm: float = 0.2,
        show_progress: bool = True,
        parallel: bool | int = False,
        max_workers: int | None = None,
        profile_memory: bool = False,
    ) -> pd.DataFrame:
        """Summarize vortex dynamics across the batch."""
        ordered_results = self._ordered_results(sort_by)
        mem_start = _memory_mb() if profile_memory else float("nan")

        def _summarize_one(index: int, result: Any) -> dict[str, Any]:
            attrs = getattr(result, "attrs", {}) or {}
            row: dict[str, Any] = {
                "index": index,
                "path": getattr(result, "path", None),
                "status": "ok",
                "error": None,
            }
            row.update(
                {
                    key: attrs.get(key)
                    for key in (
                        "i_pillar_ma",
                        "ma",
                        "Jdc",
                        "ni",
                        "addoe",
                        "EnableOersted",
                    )
                }
            )

            try:
                vortex = result.solitons.vortex
                trajectory = (
                    vortex.trajectory.steady_state()
                    if steady_state
                    else vortex.trajectory.raw
                )
                spectrum_kwargs = _resolved_spectrum_kwargs(
                    trajectory,
                    method=spectrum_method,
                    nperseg=nperseg,
                    noverlap=noverlap,
                )
                gyration = vortex.spectrum.gyration(
                    trajectory=trajectory,
                    **spectrum_kwargs,
                )
                breathing = vortex.spectrum.breathing(
                    trajectory=trajectory,
                    **spectrum_kwargs,
                )
                p_switches = vortex.events.polarity_switches(trajectory=trajectory)
                gc_switches = vortex.events.state_switches(
                    trajectory=trajectory,
                    radius_threshold=radius_threshold,
                )
                expulsions = vortex.events.core_expulsions(
                    trajectory=trajectory,
                    expulsion_ratio=expulsion_ratio,
                )

                peak_power = (
                    float(np.max(gyration.power))
                    if getattr(gyration.power, "size", 0)
                    else 0.0
                )
                disk_radius = _disk_radius_from_attrs(attrs)

                row.update(
                    {
                        "n_samples": int(len(trajectory.time)),
                        "peak_gyr_ghz": float(gyration.peak_frequency_ghz),
                        "peak_breath_ghz": float(breathing.peak_frequency_ghz),
                        "peak_power": peak_power,
                        "r_mean_nm": float(np.mean(trajectory.r) * 1e9),
                        "r_max_nm": float(np.max(trajectory.r) * 1e9),
                        "r_max_rel": float(np.max(trajectory.r) / disk_radius)
                        if np.isfinite(disk_radius) and disk_radius > 0.0
                        else float("nan"),
                        "n_p_switch": int(len(p_switches)),
                        "n_gc_switch": int(len(gc_switches)),
                        "n_expulsion": int(len(expulsions)),
                    }
                )
            except Exception as exc:
                row.update(
                    {
                        "status": "error",
                        "error": str(exc),
                        "n_samples": 0,
                        "peak_gyr_ghz": float("nan"),
                        "peak_breath_ghz": float("nan"),
                        "peak_power": float("nan"),
                        "r_mean_nm": float("nan"),
                        "r_max_nm": float("nan"),
                        "r_max_rel": float("nan"),
                        "n_p_switch": 0,
                        "n_gc_switch": 0,
                        "n_expulsion": 0,
                    }
                )

            return row

        if parallel:
            workers = int(
                max_workers or (parallel if isinstance(parallel, int) else 0) or 4
            )
            workers = max(1, workers)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                rows = list(
                    executor.map(
                        lambda item: _summarize_one(*item), enumerate(ordered_results)
                    )
                )
        else:
            iterator = _progress_iter(
                enumerate(ordered_results),
                total=len(ordered_results),
                desc="Summarizing vortex batch",
                enabled=show_progress,
            )
            rows = [_summarize_one(index, result) for index, result in iterator]

        if not rows:
            return pd.DataFrame(
                columns=[
                    "index",
                    "path",
                    "status",
                    "error",
                    "i_pillar_ma",
                    "ma",
                    "Jdc",
                    "ni",
                    "addoe",
                    "EnableOersted",
                    "n_samples",
                    "peak_gyr_ghz",
                    "peak_breath_ghz",
                    "peak_power",
                    "peak_power_rel",
                    "r_mean_nm",
                    "r_max_nm",
                    "r_max_rel",
                    "n_p_switch",
                    "n_gc_switch",
                    "n_expulsion",
                    "regime",
                ]
            )

        frame = pd.DataFrame(rows)
        peak_power_values = frame["peak_power"].to_numpy(dtype=float)
        finite_peak_power = peak_power_values[np.isfinite(peak_power_values)]
        peak_power_max = (
            float(np.max(finite_peak_power)) if finite_peak_power.size else 1.0
        )
        if peak_power_max <= 0.0:
            peak_power_max = 1.0
        frame["peak_power_rel"] = frame["peak_power"].astype(float) / float(
            peak_power_max
        )
        frame["regime"] = frame.apply(
            self._classify_regime,
            axis=1,
            power_floor_rel=power_floor_rel,
            radius_floor_nm=radius_floor_nm,
            expulsion_ratio=expulsion_ratio,
        )
        if profile_memory:
            mem_end = _memory_mb()
            frame.attrs["memory_profile"] = {
                "memory_start_mb": mem_start,
                "memory_end_mb": mem_end,
                "memory_delta_mb": mem_end - mem_start
                if np.isfinite(mem_start) and np.isfinite(mem_end)
                else float("nan"),
            }
        return frame

    def regimes(self, **kwargs) -> pd.DataFrame:
        """Alias for :meth:`summary` to emphasize regime classification."""
        return self.summary(**kwargs)

    def spectrum_map(
        self,
        *,
        component: str = "gyration",
        sort_by: str = "i_pillar_ma",
        steady_state: bool = True,
        spectrum_method: str = "welch",
        nperseg: int | None = 512,
        noverlap: int | None = 256,
        show_progress: bool = True,
        parallel: bool | int = False,
        max_workers: int | None = None,
        profile_memory: bool = False,
    ) -> BatchVortexSpectrumMapResult:
        """Return batch spectrum matrix across the filtered simulations."""
        coordinate: list[float] = []
        power_rows: list[np.ndarray] = []
        frequency_ref: np.ndarray | None = None
        errors: list[dict[str, Any]] = []

        if component not in {"gyration", "breathing"}:
            raise ValueError("component must be 'gyration' or 'breathing'")

        ordered_results = self._ordered_results(sort_by)
        mem_start = _memory_mb() if profile_memory else float("nan")

        def _map_one(
            index: int, result: Any
        ) -> tuple[float, np.ndarray | None, np.ndarray | None, dict[str, Any] | None]:
            attrs = getattr(result, "attrs", {}) or {}
            coord_value = _coerce_numeric(
                attrs.get(sort_by, index), default=float(index)
            )

            try:
                vortex = result.solitons.vortex
                trajectory = (
                    vortex.trajectory.steady_state()
                    if steady_state
                    else vortex.trajectory.raw
                )
                spectrum_kwargs = _resolved_spectrum_kwargs(
                    trajectory,
                    method=spectrum_method,
                    nperseg=nperseg,
                    noverlap=noverlap,
                )
                if component == "gyration":
                    spectrum = vortex.spectrum.gyration(
                        trajectory=trajectory,
                        **spectrum_kwargs,
                    )
                else:
                    spectrum = vortex.spectrum.breathing(
                        trajectory=trajectory,
                        **spectrum_kwargs,
                    )

                frequencies = np.asarray(spectrum.frequencies, dtype=float)
                power = np.asarray(spectrum.power, dtype=float)
                return coord_value, frequencies, power, None
            except Exception as exc:
                return (
                    coord_value,
                    None,
                    None,
                    {
                        "index": index,
                        "path": getattr(result, "path", None),
                        "coordinate": coord_value,
                        "error": str(exc),
                    },
                )

        if parallel:
            workers = int(
                max_workers or (parallel if isinstance(parallel, int) else 0) or 4
            )
            workers = max(1, workers)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                mapped = list(
                    executor.map(
                        lambda item: _map_one(*item), enumerate(ordered_results)
                    )
                )
        else:
            iterator = _progress_iter(
                enumerate(ordered_results),
                total=len(ordered_results),
                desc=f"Computing {component} spectrum map",
                enabled=show_progress,
            )
            mapped = [_map_one(index, result) for index, result in iterator]

        for coord_value, frequencies, power, error in mapped:
            if error is not None:
                errors.append(error)
                continue
            assert frequencies is not None and power is not None
            if frequency_ref is None:
                frequency_ref = frequencies
            elif frequencies.shape != frequency_ref.shape or not np.allclose(
                frequencies, frequency_ref
            ):
                power = np.interp(
                    frequency_ref, frequencies, power, left=0.0, right=0.0
                )
            coordinate.append(coord_value)
            power_rows.append(power)

        if frequency_ref is None or not power_rows:
            metadata = {"errors": errors, "steady_state": steady_state}
            if profile_memory:
                mem_end = _memory_mb()
                metadata["memory_profile"] = {
                    "memory_start_mb": mem_start,
                    "memory_end_mb": mem_end,
                    "memory_delta_mb": mem_end - mem_start
                    if np.isfinite(mem_start) and np.isfinite(mem_end)
                    else float("nan"),
                }
            return BatchVortexSpectrumMapResult(
                coordinate=np.asarray([], dtype=float),
                frequencies=np.asarray([], dtype=float),
                power=np.zeros((0, 0), dtype=float),
                component=component,
                coordinate_name=sort_by,
                metadata=metadata,
            )

        metadata = {"errors": errors, "steady_state": steady_state}
        if profile_memory:
            mem_end = _memory_mb()
            metadata["memory_profile"] = {
                "memory_start_mb": mem_start,
                "memory_end_mb": mem_end,
                "memory_delta_mb": mem_end - mem_start
                if np.isfinite(mem_start) and np.isfinite(mem_end)
                else float("nan"),
            }
        return BatchVortexSpectrumMapResult(
            coordinate=np.asarray(coordinate, dtype=float),
            frequencies=np.asarray(frequency_ref, dtype=float),
            power=np.vstack(power_rows),
            component=component,
            coordinate_name=sort_by,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        return f"BatchVortexInterface({len(self._results)} results)"

    def _repr_html_(self) -> str:
        return make_simple_card(
            title=self.__class__.__name__,
            subtitle="batch vortex namespace",
            rows=[
                ("n_results", str(len(self._results))),
                ("methods", "summary, spectrum_map"),
            ],
        )


class BatchVortexPlotAccessor:
    """Plot helpers for :class:`BatchVortexInterface`."""

    def __init__(self, interface: BatchVortexInterface):
        self._interface = interface

    def spectrum_map(self, **kwargs):
        """Compute and plot batch spectrum map."""
        map_result = self._interface.spectrum_map(**kwargs)
        return map_result.plt.heatmap()

    def orbit_radius(
        self,
        *,
        sort_by: str = "i_pillar_ma",
        ax=None,
        show_progress: bool = True,
        save=None,
        **summary_kwargs,
    ):
        style_kwargs = pop_axes_style_kwargs(summary_kwargs)
        figure_kwargs = pop_figure_kwargs(summary_kwargs)
        axis = ensure_axis(ax, default_figsize=(7.0, 3.2), figure_kwargs=figure_kwargs)
        frame = self._interface.summary(
            sort_by=sort_by,
            show_progress=show_progress,
            **summary_kwargs,
        )
        if not frame.empty:
            x = frame[sort_by].astype(float).to_numpy()
            axis.plot(x, frame["r_mean_nm"], marker="o", label="mean")
            axis.plot(x, frame["r_max_nm"], marker="s", label="max")
            axis.legend(frameon=False)
        axis.set_xlabel(_coordinate_label(sort_by))
        axis.set_ylabel("Orbit radius [nm]")
        axis.set_title("Vortex orbit radius")
        apply_axes_style(axis, style_kwargs)
        if save is not None:
            axis.figure.savefig(save)
        return axis

    def regimes(
        self,
        *,
        sort_by: str = "i_pillar_ma",
        ax=None,
        show_progress: bool = True,
        **summary_kwargs,
    ):
        """Plot regime classification along the selected batch coordinate."""
        style_kwargs = pop_axes_style_kwargs(summary_kwargs)
        figure_kwargs = pop_figure_kwargs(summary_kwargs)
        axis = ensure_axis(ax, default_figsize=(7.0, 2.8), figure_kwargs=figure_kwargs)

        frame = self._interface.summary(
            sort_by=sort_by,
            show_progress=show_progress,
            **summary_kwargs,
        )
        if frame.empty:
            axis.set_title("No vortex results")
            axis.set_xlabel(_coordinate_label(sort_by))
            axis.set_ylabel("Regime")
            apply_axes_style(axis, style_kwargs)
            return axis

        mapping = {name: idx for idx, name in enumerate(_REGIME_ORDER)}
        x = frame[sort_by].astype(float).to_numpy()
        y = np.array(
            [mapping.get(item, mapping["error"]) for item in frame["regime"]],
            dtype=float,
        )
        colors = [_REGIME_COLORS.get(item, "black") for item in frame["regime"]]

        axis.scatter(x, y, c=colors, s=70, zorder=3)
        axis.plot(x, y, color="0.75", linewidth=1.0, zorder=1)
        axis.set_yticks(np.arange(len(_REGIME_ORDER), dtype=float))
        axis.set_yticklabels(_REGIME_ORDER)
        axis.set_xlabel(_coordinate_label(sort_by))
        axis.set_ylabel("Regime")
        axis.set_title("Vortex regime map")
        axis.grid(True, axis="x", alpha=0.25)
        apply_axes_style(axis, style_kwargs)
        return axis

    def dashboard(
        self,
        *,
        sort_by: str = "i_pillar_ma",
        show_progress: bool = True,
        figsize: tuple[float, float] = (11.5, 8.0),
        dpi: int | None = None,
        save=None,
        **summary_kwargs,
    ):
        """Plot a compact set of vortex batch diagnostics."""
        import matplotlib.pyplot as plt

        frame = self._interface.summary(
            sort_by=sort_by,
            show_progress=show_progress,
            **summary_kwargs,
        )
        fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi, sharex=True)
        flat_axes = axes.ravel()

        if frame.empty:
            for axis in flat_axes:
                axis.set_axis_off()
            fig.suptitle("No vortex results")
            return fig, axes, frame

        x = frame[sort_by].astype(float).to_numpy()
        regime_colors = [_REGIME_COLORS.get(item, "black") for item in frame["regime"]]

        flat_axes[0].plot(x, frame["peak_gyr_ghz"], marker="o", label="gyration")
        flat_axes[0].plot(x, frame["peak_breath_ghz"], marker="s", label="breathing")
        flat_axes[0].set_ylabel("Frequency [GHz]")
        flat_axes[0].set_title("Dominant frequencies")
        flat_axes[0].legend(frameon=False)
        flat_axes[0].grid(True, alpha=0.25)

        flat_axes[1].plot(x, frame["r_mean_nm"], marker="o", label="mean radius")
        flat_axes[1].plot(x, frame["r_max_nm"], marker="s", label="max radius")
        flat_axes[1].set_ylabel("Radius [nm]")
        flat_axes[1].set_title("Orbit size")
        flat_axes[1].legend(frameon=False)
        flat_axes[1].grid(True, alpha=0.25)

        flat_axes[2].plot(x, frame["n_p_switch"], marker="o", label="polarity")
        flat_axes[2].plot(x, frame["n_gc_switch"], marker="s", label="G/C")
        flat_axes[2].plot(x, frame["n_expulsion"], marker="^", label="expulsion")
        flat_axes[2].set_xlabel(_coordinate_label(sort_by))
        flat_axes[2].set_ylabel("Count")
        flat_axes[2].set_title("Detected events")
        flat_axes[2].legend(frameon=False)
        flat_axes[2].grid(True, alpha=0.25)

        flat_axes[3].scatter(x, frame["peak_power_rel"], c=regime_colors, s=70)
        flat_axes[3].plot(x, frame["peak_power_rel"], color="0.75", linewidth=1.0)
        flat_axes[3].set_xlabel(_coordinate_label(sort_by))
        flat_axes[3].set_ylabel("Relative power")
        flat_axes[3].set_title("Mode strength / regime")
        flat_axes[3].grid(True, alpha=0.25)

        fig.tight_layout()
        if save is not None:
            fig.savefig(save)
        return fig, axes, frame

    def orbits(
        self,
        *,
        sort_by: str = "i_pillar_ma",
        ax=None,
        show_progress: bool = True,
        colorbar: bool = True,
        grid_alpha: float = 0.2,
        show_disk_boundary: bool = True,
        disk_radius: float | None = None,
        save=None,
        **kwargs,
    ):
        from matplotlib.patches import Circle

        del colorbar
        style_kwargs = pop_axes_style_kwargs(kwargs)
        figure_kwargs = pop_figure_kwargs(kwargs)
        axis = ensure_axis(ax, default_figsize=(5.0, 5.0), figure_kwargs=figure_kwargs)
        ordered = self._interface._ordered_results(sort_by)
        for result in ordered:
            trajectory = result.solitons.vortex.trajectory.steady_state()
            axis.plot(
                np.asarray(trajectory.x) * 1e9,
                np.asarray(trajectory.y) * 1e9,
                alpha=0.8,
            )
        radius = disk_radius
        if radius is None and ordered:
            radius = _disk_radius_from_attrs(getattr(ordered[0], "attrs", {}) or {})
        if show_disk_boundary and radius is not None and np.isfinite(radius):
            axis.add_patch(
                Circle(
                    (0.0, 0.0),
                    float(radius) * 1e9,
                    fill=False,
                    linestyle="--",
                    color="0.4",
                )
            )
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x [nm]")
        axis.set_ylabel("y [nm]")
        axis.grid(True, alpha=float(grid_alpha))
        apply_axes_style(axis, style_kwargs)
        if save is not None:
            axis.figure.savefig(save)
        return axis

    def orbits_grid(
        self,
        *,
        sort_by: str = "i_pillar_ma",
        show_progress: bool = True,
        colorbar: bool = True,
        grid_alpha: float = 0.2,
        show_disk_boundary: bool = True,
        disk_radius: float | None = None,
        ncols: int = 3,
        save=None,
        **kwargs,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        del show_progress, colorbar
        ordered = self._interface._ordered_results(sort_by)
        n = max(len(ordered), 1)
        cols = max(int(ncols), 1)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(
            rows, cols, squeeze=False, figsize=(3.2 * cols, 3.2 * rows)
        )
        flat = axes.ravel()
        for axis, result in zip(flat, ordered):
            trajectory = result.solitons.vortex.trajectory.steady_state()
            axis.plot(np.asarray(trajectory.x) * 1e9, np.asarray(trajectory.y) * 1e9)
            radius = disk_radius
            if radius is None:
                radius = _disk_radius_from_attrs(getattr(result, "attrs", {}) or {})
            if show_disk_boundary and radius is not None and np.isfinite(radius):
                axis.add_patch(
                    Circle(
                        (0.0, 0.0),
                        float(radius) * 1e9,
                        fill=False,
                        linestyle="--",
                        color="0.4",
                    )
                )
            axis.set_aspect("equal", adjustable="box")
            axis.grid(True, alpha=float(grid_alpha))
        for axis in flat[len(ordered) :]:
            axis.set_axis_off()
        fig.tight_layout()
        if save is not None:
            fig.savefig(save)
        return fig, axes

    def _repr_html_(self) -> str:
        return make_simple_card(
            title=self.__class__.__name__,
            subtitle="batch vortex plot accessor",
            rows=[("methods", "regimes, dashboard, orbits")],
        )
