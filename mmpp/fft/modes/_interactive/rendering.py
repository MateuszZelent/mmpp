"""Figure-rendering helpers for :mod:`mmpp.fft.modes.interactive`."""

from __future__ import annotations

from typing import Any

import numpy as np

from .filters import component_plot_label

_DEFAULT_FIGSIZE = (16.0, 10.0)


def _resolve_layout_variant(
    explorer: Any,
    *,
    n_rows: int,
    n_components: int,
) -> str:
    """Resolve effective figure layout variant from user preference."""
    requested = str(getattr(explorer, "_layout_mode", "auto")).strip().lower()
    if requested not in {"auto", "vertical", "horizontal"}:
        requested = "auto"

    if requested == "auto":
        if n_components == 1 and n_rows > 1:
            return "single_component"
        return "vertical"

    return requested


def _resolve_figsize(
    explorer: Any,
    *,
    variant: str,
    n_rows: int,
    n_components: int,
) -> tuple[float, float]:
    """Compute adaptive figure size unless user explicitly overrode it."""
    current: tuple[float, float] = tuple(float(v) for v in explorer.figsize)  # type: ignore[assignment]
    if current != _DEFAULT_FIGSIZE:
        return current

    if variant == "single_component":
        width = max(11.0, 5.6 + 3.6 * n_rows)
        height = 7.0
    elif variant == "horizontal":
        width = max(11.0, 6.2 + 3.0 * n_components)
        height = max(6.8, 1.8 + 2.3 * n_rows)
    else:
        width = max(11.0, 5.2 + 2.9 * n_components)
        height = max(6.8, 2.2 + 2.2 * n_rows)

    return (float(width), float(height))


def render_figure(
    explorer: Any,
    *,
    clear_output_fn: Any,
    plt_module: Any,
    grid_spec_cls: Any,
) -> None:
    """Render spectrum + mode figure in widget output or directly."""
    n_components = max(len(explorer._mode_components), 1)
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
    explorer._mode_cbar_axes = []

    plt_module.ioff()

    try:
        variant = _resolve_layout_variant(
            explorer,
            n_rows=n_rows,
            n_components=n_components,
        )
        explorer._layout_variant = variant
        figsize = _resolve_figsize(
            explorer,
            variant=variant,
            n_rows=n_rows,
            n_components=n_components,
        )

        if variant == "single_component":
            explorer._fig = plt_module.figure(
                figsize=figsize,
                dpi=explorer.dpi,
                constrained_layout=False,
            )
            gs = grid_spec_cls(
                2,
                n_rows,
                figure=explorer._fig,
                height_ratios=[1.35, 1.0],
                hspace=0.23,
                wspace=0.24,
                left=0.06,
                right=0.98,
                top=0.90,
                bottom=0.10,
            )

            explorer._ax_spectrum = explorer._fig.add_subplot(gs[0, :])

            axes: list[list[Any]] = []
            cbar_axes: list[Any] = []
            for row in range(n_rows):
                mode_ax = explorer._fig.add_subplot(gs[1, row])
                axes.append([mode_ax])
                cax = mode_ax.inset_axes([0.10, -0.18, 0.80, 0.08])
                cbar_axes.append(cax)
            explorer._mode_axes = np.asarray(axes, dtype=object)
            explorer._mode_cbar_axes = cbar_axes
        elif variant == "vertical":
            total_rows = 1 + n_rows
            explorer._fig = plt_module.figure(
                figsize=figsize,
                dpi=explorer.dpi,
                constrained_layout=False,
            )
            gs = grid_spec_cls(
                total_rows,
                n_components + 1,
                figure=explorer._fig,
                height_ratios=[1.2] + [1.0] * n_rows,
                width_ratios=[1.0] * n_components + [0.075],
                hspace=0.22,
                wspace=0.20,
                left=0.06,
                right=0.98,
                top=0.92,
                bottom=0.08,
            )

            explorer._ax_spectrum = explorer._fig.add_subplot(gs[0, :-1])

            axes = []
            cbar_axes = []
            for row in range(n_rows):
                row_axes = []
                for col in range(n_components):
                    row_axes.append(explorer._fig.add_subplot(gs[row + 1, col]))
                axes.append(row_axes)
                cbar_axes.append(explorer._fig.add_subplot(gs[row + 1, -1]))
            explorer._mode_axes = np.asarray(axes, dtype=object)
            explorer._mode_cbar_axes = cbar_axes
        else:
            explorer._fig = plt_module.figure(
                figsize=figsize,
                dpi=explorer.dpi,
                constrained_layout=False,
            )
            gs = grid_spec_cls(
                n_rows,
                n_components + 2,
                figure=explorer._fig,
                width_ratios=[1.55] + [1.0] * n_components + [0.075],
                hspace=0.24,
                wspace=0.22,
                left=0.05,
                right=0.98,
                top=0.92,
                bottom=0.08,
            )

            explorer._ax_spectrum = explorer._fig.add_subplot(gs[:, 0])

            axes = []
            cbar_axes = []
            for row in range(n_rows):
                row_axes = []
                for col in range(n_components):
                    row_axes.append(explorer._fig.add_subplot(gs[row, col + 1]))
                axes.append(row_axes)
                cbar_axes.append(explorer._fig.add_subplot(gs[row, -1]))
            explorer._mode_axes = np.asarray(axes, dtype=object)
            explorer._mode_cbar_axes = cbar_axes

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
    # Axes clear invalidates previously cached line handles.
    explorer._frequency_line = None

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

    color_map = {
        "x": "#E76F51",
        "y": "#2A9D8F",
        "z": "#457B9D",
        "+": "#8D5A97",
        "-": "#F4A261",
        "rho": "#5E60CE",
        "phi": "#8338EC",
    }

    plot_components = [
        comp
        for comp in explorer._spectrum_components
        if comp in explorer._filtered_component_power
    ]
    if not plot_components:
        plot_components = list(explorer._filtered_component_power.keys())

    for comp in plot_components:
        trace = explorer._filtered_component_power.get(comp)
        if trace is None or trace.size == 0:
            continue
        ax.plot(
            freqs_plot,
            trace,
            color=color_map.get(comp, "#4C78A8"),
            linewidth=1.8,
            alpha=0.95,
            label=component_plot_label(comp),
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
    if len(plot_components) > 1:
        ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    ax.text(
        0.02,
        0.02,
        "left click: exact frequency, right click or Shift+click: nearest peak",
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
    line = getattr(explorer, "_frequency_line", None)
    if line is not None:
        try:
            if getattr(line, "axes", None) is explorer._ax_spectrum:
                line.set_xdata([x_value, x_value])
                line.set_visible(True)
                return
        except Exception:
            pass
        explorer._frequency_line = None

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
