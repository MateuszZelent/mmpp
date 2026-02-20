"""Mode-plot rendering helpers for :mod:`mmpp.fft.modes.interactive`."""

from __future__ import annotations

from typing import Any

import numpy as np

from .filters import component_plot_label, resolve_mode_components
from .mode_layout import (
    apply_mode_colorbars,
    finalize_mode_figure,
    reset_mode_colorbars,
)


def _render_mode_load_error(explorer: Any, exc: Exception) -> None:
    """Render load failure text on all mode axes."""
    if explorer._mode_axes is None:
        return

    for ax in explorer._mode_axes.flatten():
        ax.clear()
        ax.text(
            0.5,
            0.5,
            f"Mode load error:\n{exc}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="crimson",
        )

    if explorer._fig is not None:
        explorer._fig.canvas.draw_idle()


def _resolve_mode_cmaps(explorer: Any) -> tuple[str, str, str]:
    """Resolve current colormap settings for mode rows."""
    cmap_mag = explorer._controls.get("cmap_mag", None)
    cmap_phase = explorer._controls.get("cmap_phase", None)
    cmap_combined = explorer._controls.get("cmap_combined", None)

    cmap_mag_name = str(cmap_mag.value) if cmap_mag is not None else "viridis"
    cmap_phase_name = str(cmap_phase.value) if cmap_phase is not None else "twilight"
    cmap_combined_name = str(cmap_combined.value) if cmap_combined is not None else "RdBu_r"
    return cmap_mag_name, cmap_phase_name, cmap_combined_name


def _resolve_plot_data(
    row_type: str,
    magnitude: np.ndarray,
    phase: np.ndarray,
    cmap_mag_name: str,
    cmap_phase_name: str,
    cmap_combined_name: str,
) -> tuple[np.ndarray, str, float | None, float | None, str]:
    """Build plot data and scaling for one row type."""
    if row_type == "magnitude":
        return magnitude, cmap_mag_name, None, None, "|m|"

    if row_type == "phase":
        return phase, cmap_phase_name, -np.pi, np.pi, "phase"

    plot_data = magnitude * np.cos(phase)
    vmax_val = float(np.nanmax(np.abs(plot_data))) if plot_data.size else 1.0
    if vmax_val <= 0:
        vmax_val = 1.0
    return plot_data, cmap_combined_name, -vmax_val, vmax_val, "combined"


def _render_holography(comp_data: np.ndarray) -> np.ndarray:
    """Render complex data as HSV domain-coloring RGB image."""
    try:
        from ..vortex_optics import VortexOptics

        return VortexOptics.complex_holography(comp_data)
    except Exception:
        # Fallback: visualize phase only if vortex module is unavailable.
        phase = np.angle(comp_data)
        rgb = np.zeros(phase.shape + (3,), dtype=float)
        rgb[..., 0] = (phase + np.pi) / (2 * np.pi)
        rgb[..., 1] = 1.0
        rgb[..., 2] = 1.0
        return rgb


def _apply_mode_axes_style(
    explorer: Any,
    *,
    ax: Any,
    row_idx: int,
    col_idx: int,
    row_title: str,
    comp: str,
    actual_freq: float,
) -> None:
    """Apply labels/grid/title for one mode subplot."""
    component_label = component_plot_label(comp)
    single_component_layout = getattr(explorer, "_layout_variant", "") == "single_component"
    if single_component_layout:
        ax.set_title(
            f"{row_title} ({component_label})",
            fontsize=10,
        )
        ax.set_ylabel("")
        ax.set_xlabel("x [μm]", fontsize=9)
    else:
        if row_idx == 0:
            ax.set_title(f"{component_label} @ {actual_freq:.3f} GHz", fontsize=10)
        if col_idx == 0:
            ax.set_ylabel(row_title, fontsize=9)
        if row_idx == len(explorer._mode_row_types) - 1:
            ax.set_xlabel("x [μm]", fontsize=9)
        else:
            ax.set_xlabel("")

    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2, linestyle=":")


def _overlay_geometry_contour(
    explorer: Any,
    *,
    ax: Any,
    extent: tuple[float, float, float, float],
) -> None:
    """Overlay geometry contour when available."""
    if explorer._geometry_contour is None:
        return

    try:
        geom = explorer._geometry_contour
        geom_y = np.linspace(extent[2], extent[3], geom.shape[0])
        geom_x = np.linspace(extent[0], extent[1], geom.shape[1])
        ax.contour(geom_x, geom_y, geom, levels=[0.5], colors=["white"], linewidths=[1.5])
        ax.contour(geom_x, geom_y, geom, levels=[0.5], colors=["black"], linewidths=[0.5])
    except Exception:
        pass


def update_mode_plots(explorer: Any) -> None:
    """Render mode maps for currently selected frequency."""
    if explorer._mode_axes is None or explorer._current_frequency_ghz is None:
        return

    reset_mode_colorbars(explorer)

    try:
        mode_array, actual_freq, extent = explorer._load_mode(
            explorer._current_frequency_ghz,
            explorer._current_z_layer,
        )
        explorer._loaded_frequency_ghz = float(actual_freq)
    except Exception as exc:
        _render_mode_load_error(explorer, exc)
        return

    if mode_array.ndim == 2:
        mode_array = mode_array[:, :, np.newaxis]

    cmap_mag_name, cmap_phase_name, cmap_combined_name = _resolve_mode_cmaps(explorer)
    resolved_components = resolve_mode_components(mode_array, explorer._current_components)

    row_images: list[Any] = [None] * len(explorer._mode_row_types)

    for row_idx, row_type in enumerate(explorer._mode_row_types):
        for col_idx, comp in enumerate(explorer._current_components):
            ax = explorer._mode_axes[row_idx, col_idx]
            ax.clear()

            comp_data = resolved_components.get(comp)
            if comp_data is None:
                ax.text(
                    0.5,
                    0.5,
                    f"No m_{comp}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            magnitude = np.abs(comp_data)
            phase = np.angle(comp_data)

            use_holography = bool(
                getattr(explorer, "_use_holography", False) and row_type == "phase"
            )
            if use_holography:
                plot_data = _render_holography(comp_data)
                cmap_name = None
                vmin = None
                vmax = None
                row_title = "holography"
            else:
                plot_data, cmap_name, vmin, vmax, row_title = _resolve_plot_data(
                    row_type,
                    magnitude,
                    phase,
                    cmap_mag_name,
                    cmap_phase_name,
                    cmap_combined_name,
                )

            img = ax.imshow(
                plot_data,
                origin="lower",
                extent=extent,
                aspect=explorer._mode_aspect,
                cmap=cmap_name,
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
            )

            if explorer._xlim:
                ax.set_xlim(*explorer._xlim)
            if explorer._ylim:
                ax.set_ylim(*explorer._ylim)

            if row_images[row_idx] is None and not use_holography:
                row_images[row_idx] = img

            _apply_mode_axes_style(
                explorer,
                ax=ax,
                row_idx=row_idx,
                col_idx=col_idx,
                row_title=row_title,
                comp=comp,
                actual_freq=actual_freq,
            )
            _overlay_geometry_contour(explorer, ax=ax, extent=extent)

    apply_mode_colorbars(explorer, row_images)
    finalize_mode_figure(explorer)
    explorer._update_status_text()


__all__ = ["update_mode_plots"]
