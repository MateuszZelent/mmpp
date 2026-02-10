"""Figure-rendering helpers for :mod:`mmpp.fft.modes.interactive`."""

from __future__ import annotations

from typing import Any

import numpy as np

from .filters import _COMPONENT_INDEX, COMPONENT_LABELS


def render_figure(
    explorer: Any,
    *,
    clear_output_fn: Any,
    plt_module: Any,
    grid_spec_cls: Any,
) -> None:
    """Render spectrum + mode figure in widget output or directly."""
    n_components = max(len(explorer._current_components), 1)
    n_rows = max(len(explorer._mode_row_types), 1)

    if explorer._toolbar_enabled and explorer._widget_output is not None:
        with explorer._widget_output:
            clear_output_fn(wait=True)
            create_figure(
                explorer,
                n_rows=n_rows,
                n_components=n_components,
                plt_module=plt_module,
                grid_spec_cls=grid_spec_cls,
            )
            draw_spectrum(explorer)
            explorer._update_mode_plots()
            # Figure is displayed inside widget output, not notebook return value.
            plt_module.show()
    else:
        create_figure(
            explorer,
            n_rows=n_rows,
            n_components=n_components,
            plt_module=plt_module,
            grid_spec_cls=grid_spec_cls,
        )
        draw_spectrum(explorer)
        explorer._update_mode_plots()


def create_figure(
    explorer: Any,
    n_rows: int,
    n_components: int,
    *,
    plt_module: Any,
    grid_spec_cls: Any,
) -> None:
    """Create matplotlib figure and axes layout."""
    explorer._cleanup_figure_connections()

    plt_module.ioff()

    try:
        if explorer._layout_mode == "vertical":
            total_rows = 1 + n_rows
            explorer._fig = plt_module.figure(
                figsize=explorer.figsize,
                dpi=explorer.dpi,
                constrained_layout=False,
            )
            gs = grid_spec_cls(
                total_rows,
                n_components,
                figure=explorer._fig,
                height_ratios=[1.2] + [1.0] * n_rows,
            )

            explorer._ax_spectrum = explorer._fig.add_subplot(gs[0, :])

            axes = []
            for row in range(n_rows):
                row_axes = []
                for col in range(n_components):
                    row_axes.append(explorer._fig.add_subplot(gs[row + 1, col]))
                axes.append(row_axes)
            explorer._mode_axes = np.asarray(axes, dtype=object)
        else:
            explorer._fig = plt_module.figure(
                figsize=explorer.figsize,
                dpi=explorer.dpi,
                constrained_layout=False,
            )
            gs = grid_spec_cls(
                n_rows,
                n_components + 1,
                figure=explorer._fig,
                width_ratios=[1.6] + [1.0] * n_components,
            )

            explorer._ax_spectrum = explorer._fig.add_subplot(gs[:, 0])

            axes = []
            for row in range(n_rows):
                row_axes = []
                for col in range(n_components):
                    row_axes.append(explorer._fig.add_subplot(gs[row, col + 1]))
                axes.append(row_axes)
            explorer._mode_axes = np.asarray(axes, dtype=object)

        if explorer._fig is not None:
            explorer._fig.canvas.mpl_connect("button_press_event", explorer._on_click)
    finally:
        plt_module.ion()


def draw_spectrum(explorer: Any) -> None:
    """Draw filtered spectrum traces and optional peak markers."""
    if explorer._ax_spectrum is None:
        return

    ax = explorer._ax_spectrum
    ax.clear()

    if explorer._filtered_frequencies_ghz.size == 0:
        ax.text(
            0.5,
            0.5,
            "No spectrum data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    freq_scale = explorer._get_freq_scale(explorer._freq_unit)
    freqs_plot = explorer._filtered_frequencies_ghz * freq_scale

    color_map = {"x": "#E76F51", "y": "#2A9D8F", "z": "#457B9D"}

    for comp in explorer._current_components:
        trace = explorer._filtered_component_power.get(comp)
        if trace is None or trace.size == 0:
            continue
        ax.plot(
            freqs_plot,
            trace,
            color=color_map.get(comp, "#4C78A8"),
            linewidth=1.8,
            alpha=0.95,
            label=COMPONENT_LABELS[_COMPONENT_INDEX.get(comp, 2)],
        )

    if explorer._show_peaks and explorer._peaks:
        for freq_ghz, amp in explorer._peaks:
            x_val = freq_ghz * freq_scale
            ax.plot(
                [x_val],
                [amp],
                marker="o",
                markersize=5,
                color="#D62828",
                markeredgecolor="white",
                markeredgewidth=1.0,
                zorder=6,
            )

    draw_frequency_line(explorer)

    label = explorer._title or "FMR Spectrum"
    ax.set_title(f"{label} (click to select frequency)")
    ax.set_xlabel(f"Frequency ({explorer._freq_unit})")
    ax.set_ylabel("log10(Power)" if explorer._filter_state.log_scale else "Power")
    ax.grid(True, alpha=0.25, linestyle="--")
    if len(explorer._current_components) > 1:
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    ax.text(
        0.02,
        0.02,
        "left click: select, right click: snap to peak",
        transform=ax.transAxes,
        fontsize=8,
        alpha=0.75,
        va="bottom",
    )


def draw_frequency_line(explorer: Any) -> None:
    """Draw or update current frequency indicator line."""
    if explorer._ax_spectrum is None or explorer._current_frequency_ghz is None:
        return

    scale = explorer._get_freq_scale(explorer._freq_unit)
    x_value = explorer._current_frequency_ghz * scale

    if explorer._frequency_line is not None:
        try:
            explorer._frequency_line.remove()
        except Exception:
            pass

    explorer._frequency_line = explorer._ax_spectrum.axvline(
        x_value,
        color="#D62828",
        linestyle="--",
        linewidth=1.8,
        alpha=0.85,
    )


__all__ = [
    "render_figure",
    "create_figure",
    "draw_spectrum",
    "draw_frequency_line",
]
