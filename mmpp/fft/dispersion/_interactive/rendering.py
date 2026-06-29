"""Rendering helpers for interactive dispersion explorer."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _scaled_k_axis(k_axis: Any, kscale: str) -> tuple[Any, str]:
    if kscale == "rad_um":
        return k_axis / 1e6, "k [rad/um]"
    if kscale in {"cycles_m", "meter"}:
        return k_axis / (2 * np.pi), "k [1/m]"
    return k_axis, "k [rad/m]"


def _default_k_window_for_scale(kscale: str) -> tuple[float, float] | None:
    """Return a notebook-friendly default k-window in display units."""
    if kscale == "rad_um":
        return -10.0, 10.0
    if kscale == "rad":
        return -10.0e6, 10.0e6
    if kscale in {"cycles_m", "meter"}:
        return -10.0e6 / (2 * np.pi), 10.0e6 / (2 * np.pi)
    return None


def _display_k_xlim(explorer: Any, k_plot: Any) -> tuple[float, float] | None:
    """Return display-space k limits for the interactive heatmap."""
    explicit = getattr(explorer, "options", {}).get("k_xlim")
    if explicit is not None:
        try:
            lo, hi = explicit
            return float(lo), float(hi)
        except (TypeError, ValueError):
            logger.debug("Ignoring invalid k_xlim=%r", explicit)

    default_window = _default_k_window_for_scale(str(explorer.state.kscale))
    if default_window is None:
        return None

    try:
        data_lo = float(np.nanmin(k_plot))
        data_hi = float(np.nanmax(k_plot))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(data_lo) or not np.isfinite(data_hi) or data_lo >= data_hi:
        return None

    default_lo, default_hi = default_window
    lo = max(data_lo, default_lo)
    hi = min(data_hi, default_hi)
    if lo >= hi:
        return None
    return lo, hi


def _norm(explorer: Any, spectrum: Any) -> Any:
    if not bool(explorer.state.lognorm):
        return None
    try:
        from matplotlib.colors import LogNorm

        positive_values = spectrum[spectrum > 0]
        if positive_values.size:
            return LogNorm(
                vmin=float(np.min(positive_values)),
                vmax=float(np.max(positive_values)),
            )
    except Exception:
        return None
    return None


def _normalise_analytical_setting(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return "DE" if raw_value else None
    if isinstance(raw_value, str):
        value = raw_value.strip()
        if not value:
            return None
        canonical = value.upper()
        if canonical in {"OFF", "NONE", "FALSE", "0"}:
            return None
        return canonical
    return None


def _analytical_k_range(ax: Any, kscale: str) -> tuple[float, float] | None:
    try:
        k_min_disp, k_max_disp = ax.get_xlim()
    except Exception:
        return None
    if k_min_disp is None or k_max_disp is None:
        return None
    try:
        lo = float(k_min_disp)
        hi = float(k_max_disp)
    except (TypeError, ValueError):
        return None
    if lo == hi:
        return None

    if kscale == "rad_um":
        lo *= 1e6
        hi *= 1e6
    elif kscale in {"cycles_m", "meter"}:
        lo *= 2 * np.pi
        hi *= 2 * np.pi

    lo_plot = float(min(lo, hi))
    hi_plot = float(max(lo, hi))
    if np.isnan(lo_plot) or np.isnan(hi_plot):
        return None
    return lo_plot, hi_plot


def _draw_analytical_overlay(explorer: Any, ax: Any, kscale: str) -> None:
    options = explorer.options
    raw_request = options.get("analitical", options.get("analytical"))
    request = _normalise_analytical_setting(raw_request)
    if request is None:
        return
    if not hasattr(ax, "scatter"):
        logger.debug("Analytical overlay skipped: axis does not support scatter")
        return

    try:
        k_range = _analytical_k_range(ax, kscale)
    except Exception:
        k_range = None
    if k_range is None:
        logger.debug("Analytical overlay skipped: invalid k-range")
        return

    try:
        from .._plotting._analytics_overlay import (
            compute_analytical_dispersion,
            extract_material_params,
        )
    except Exception as exc:
        logger.warning("Analytical overlay unavailable: %s", exc)
        return

    auto_params = extract_material_params(explorer.result)
    effective = {
        "B": options.get("B", auto_params.get("B")),
        "Ms": options.get("Ms", auto_params.get("Ms")),
        "Aex": options.get("Aex", auto_params.get("Aex")),
        "d": options.get("d", auto_params.get("d")),
        "Ku": options.get("Ku", auto_params.get("Ku", 0.0)),
        "Kc1": options.get("Kc1", auto_params.get("Kc1", 0.0)),
        "Kc2": options.get("Kc2", auto_params.get("Kc2", 0.0)),
        "phi_ani": options.get("phi_ani", auto_params.get("phi_ani", 0.0)),
        "g": options.get("g", auto_params.get("g", 2.0)),
    }
    missing = [key for key in ("B", "Ms", "Aex", "d") if effective[key] is None]
    if missing:
        logger.warning(
            "Analytical overlay skipped for config=%s: missing material params %s",
            request,
            ", ".join(missing),
        )
        return

    try:
        n_modes = max(1, int(options.get("analytical_n_modes", 1)))
    except Exception:
        n_modes = 1
    try:
        k_points = max(50, min(5000, int(options.get("analytical_k_points", 500))))
    except Exception:
        k_points = 500

    try:
        curves = compute_analytical_dispersion(
            k_range=k_range,
            model=str(options.get("analytical_model", "kalinikos")),
            sw_config=request,
            n_modes=n_modes,
            k_points=k_points,
            phi=options.get("analytical_phi", auto_params.get("phi")),
            D=options.get("analytical_D"),
            B=effective["B"],
            Ms=effective["Ms"],
            d=effective["d"],
            Aex=effective["Aex"],
            Ku=effective["Ku"],
            Kc1=effective["Kc1"],
            Kc2=effective["Kc2"],
            phi_ani=effective["phi_ani"],
            g=effective["g"],
        )
    except Exception as exc:
        logger.warning(
            "Analytical overlay computation failed for config=%s: %s",
            request,
            exc,
        )
        return

    base_scatter = {
        "s": 30,
        "marker": "x",
        "color": "white",
        "alpha": 0.85,
        "linewidths": 1.0,
        "zorder": 20,
    }
    style_override = options.get("analytical_style")
    if isinstance(style_override, dict):
        base_scatter.update(style_override)

    plotted_any = False
    for idx, (k_values, f_ghz, mode_label) in enumerate(curves):
        if k_values is None or f_ghz is None:
            continue

        if kscale == "rad_um":
            k_plot = np.asarray(k_values, dtype=float) / 1e6
        elif kscale in {"cycles_m", "meter"}:
            k_plot = np.asarray(k_values, dtype=float) / (2 * np.pi)
        else:
            k_plot = np.asarray(k_values, dtype=float)
        f_plot = np.asarray(f_ghz, dtype=float)

        finite = np.isfinite(k_plot) & np.isfinite(f_plot)
        if not bool(np.any(finite)):
            continue
        k_plot = k_plot[finite]
        f_plot = f_plot[finite]
        if k_plot.size == 0 or f_plot.size == 0:
            continue

        scatter_kwargs = dict(base_scatter)
        scatter_kwargs["label"] = (
            mode_label if (n_modes == 1 or idx == 0) else f"{mode_label} (n={idx})"
        )
        ax.scatter(k_plot, f_plot, **scatter_kwargs)
        plotted_any = True

    if plotted_any:
        try:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.75)
        except Exception:
            pass


def draw_dispersion_panel(explorer: Any) -> None:
    """Render ``S(k, f)`` into the explorer axes."""
    if explorer.axes is None or explorer.figure is None:
        return

    ax = explorer.axes
    ax.clear()

    spectrum, k_axis, f_axis = explorer.result.frequency_view(
        positive_frequencies=bool(explorer.state.positive_frequencies),
        analysis_source=str(explorer.state.source),
    )
    spectrum = np.asarray(spectrum, dtype=float)
    k_axis = np.asarray(k_axis, dtype=float)
    f_axis = np.asarray(f_axis, dtype=float)

    live_filters = getattr(explorer.state, "live_filters", None)
    if live_filters:
        try:
            from ..utils import apply_dispersion_post_filters

            spectrum = apply_dispersion_post_filters(
                spectrum,
                k_axis=k_axis,
                f_axis=f_axis,
                filters={"live": live_filters},
                include_live=True,
            )
            explorer._last_filter_error = ""
        except Exception as exc:
            explorer._last_filter_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Interactive live filters failed: %s", exc)
    else:
        explorer._last_filter_error = ""

    f_mask = np.ones_like(f_axis, dtype=bool)
    if explorer.state.positive_frequencies:
        f_mask &= f_axis >= 0
    f_mask &= f_axis >= float(explorer.state.fmin_ghz) * 1e9
    f_mask &= f_axis <= float(explorer.state.fmax_ghz) * 1e9
    if np.any(f_mask):
        spectrum = spectrum[:, f_mask]
        f_axis = f_axis[f_mask]
    else:
        spectrum = spectrum[:, :0]
        f_axis = f_axis[:0]

    k_plot, k_label = _scaled_k_axis(k_axis, str(explorer.state.kscale))
    f_plot = f_axis / 1e9
    if k_plot.size == 0 or f_plot.size == 0:
        ax.text(0.5, 0.5, "No dispersion data in selected range", ha="center")
        return

    image = ax.imshow(
        spectrum.T,
        aspect="auto",
        origin="lower",
        extent=(
            float(k_plot[0]),
            float(k_plot[-1]),
            float(f_plot[0]),
            float(f_plot[-1]),
        ),
        cmap=str(explorer.state.cmap),
        norm=_norm(explorer, spectrum),
    )
    explorer._image = image

    k_xlim = _display_k_xlim(explorer, k_plot)
    if k_xlim is not None:
        ax.set_xlim(*k_xlim)

    if explorer.state.show_flags and explorer.state.show_flags.get("grid", True):
        if hasattr(ax, "grid"):
            ax.grid(True, alpha=0.22, linestyle=":")
    if (
        explorer.state.show_flags
        and explorer.state.show_flags.get("selection", True)
        and explorer.state.selected_k is not None
        and explorer.state.selected_f is not None
    ):
        k_sel = float(explorer.state.selected_k)
        f_sel = float(explorer.state.selected_f) / 1e9
        if explorer.state.kscale == "rad_um":
            k_sel /= 1e6
        elif explorer.state.kscale in {"cycles_m", "meter"}:
            k_sel /= 2 * np.pi
        if hasattr(ax, "plot"):
            ax.plot(
                [k_sel],
                [f_sel],
                "o",
                color="#ef4444",
                markerfacecolor="none",
                markeredgewidth=1.8,
                markersize=8,
            )

    try:
        _draw_analytical_overlay(explorer, ax, str(explorer.state.kscale))
    except Exception as exc:
        logger.debug("Analytical overlay rendering failed: %s", exc)

    ax.set_title(f"Dispersion S(k, f) - {explorer.state.source}")
    ax.set_xlabel(k_label)
    ax.set_ylabel("Frequency [GHz]")
    explorer.figure.canvas.draw_idle()


def refresh_output_widget(explorer: Any) -> None:
    """Render current figure into output widget, matching hysteresis explorer.

    For non-interactive backends (e.g. ``inline``), falls back to embedding a
    PNG snapshot so the widget output is always populated.
    """
    output = explorer.controls.get("output") if explorer.controls else None
    if output is None:
        return
    if not hasattr(output, "clear_output") or not hasattr(output, "append_display_data"):
        return
    output.clear_output(wait=False)
    if explorer.figure is None:
        return

    # Check if we have an interactive backend that supports canvas display
    interactive_backend = False
    try:
        import matplotlib

        backend = str(matplotlib.get_backend()).lower()
        interactive_backend = any(
            kw in backend for kw in ("widget", "ipympl", "nbagg", "notebook")
        )
    except Exception:
        pass

    if interactive_backend:
        # Interactive backend: keep the live ipympl/nbagg canvas inside the
        # output area and avoid a blocking synchronous draw during widget
        # startup.  Displaying the figure object itself can fall back to a
        # static repr in some notebook frontends.
        try:
            from IPython.display import display

            canvas = explorer.figure.canvas
            if hasattr(canvas, "draw_idle"):
                canvas.draw_idle()
            with output:
                display(canvas)
        except Exception:
            try:
                if hasattr(explorer.figure.canvas, "draw_idle"):
                    explorer.figure.canvas.draw_idle()
                output.append_display_data(explorer.figure)
            except Exception:
                pass
    else:
        # Inline / non-interactive backend — render to PNG and embed as Image
        try:
            import io

            from IPython.display import Image

            buf = io.BytesIO()
            explorer.figure.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            output.append_display_data(Image(data=buf.getvalue(), format="png"))
        except Exception:
            # Last resort — try direct display
            try:
                explorer.figure.canvas.draw()
                output.append_display_data(explorer.figure)
            except Exception:
                pass
