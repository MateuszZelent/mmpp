"""Mode-figure layout helpers for :mod:`mmpp.fft.modes.interactive`."""

from __future__ import annotations

from typing import Any


def reset_mode_colorbars(explorer: Any) -> None:
    """Remove existing mode colorbars before redraw."""
    has_dedicated_axes = bool(getattr(explorer, "_mode_cbar_axes", []))
    for cbar in explorer._mode_colorbars:
        try:
            if has_dedicated_axes:
                cbar.ax.clear()
            else:
                cbar.remove()
        except Exception:
            pass
    explorer._mode_colorbars = []
    for cax in getattr(explorer, "_mode_cbar_axes", []):
        try:
            cax.clear()
        except Exception:
            pass


def _colorbar_label(row_type: str) -> str:
    mapping = {
        "magnitude": "|m|",
        "phase": "phase [rad]",
        "combined": "Re[m]",
    }
    return mapping.get(str(row_type), str(row_type))


def apply_mode_colorbars(explorer: Any, row_images: list[Any]) -> None:
    """Attach one colorbar per mode-row image."""
    if explorer._fig is None or explorer._mode_axes is None:
        return

    for row_idx, img in enumerate(row_images):
        if img is None:
            continue
        try:
            cbar_axes = getattr(explorer, "_mode_cbar_axes", [])
            row_type = (
                explorer._mode_row_types[row_idx]
                if row_idx < len(explorer._mode_row_types)
                else "mode"
            )
            orientation = (
                "horizontal"
                if getattr(explorer, "_layout_variant", "") == "single_component"
                else "vertical"
            )

            if row_idx < len(cbar_axes):
                cbar = explorer._fig.colorbar(
                    img,
                    cax=cbar_axes[row_idx],
                    orientation=orientation,
                )
            else:
                cbar = explorer._fig.colorbar(
                    img,
                    ax=list(explorer._mode_axes[row_idx, :]),
                    fraction=0.040,
                    pad=0.03,
                    orientation=orientation,
                )

            cbar.set_label(_colorbar_label(row_type), fontsize=8)
            cbar.ax.tick_params(labelsize=7, length=2, width=0.6)
            explorer._mode_colorbars.append(cbar)
        except Exception:
            continue


def finalize_mode_figure(explorer: Any) -> None:
    """Apply suptitle/layout and schedule canvas redraw."""
    if explorer._fig is None:
        return

    explorer._fig.suptitle(
        f"FMR modes at {explorer._current_frequency_ghz:.3f} GHz (z={explorer._current_z_layer})",
        fontsize=12,
    )

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*not compatible with tight_layout.*",
        )
        try:
            if getattr(explorer, "_mode_cbar_axes", []):
                explorer._fig.subplots_adjust(top=0.90)
            else:
                explorer._fig.tight_layout(rect=[0, 0, 1, 0.97])
        except Exception:
            pass

    explorer._fig.canvas.draw_idle()


__all__ = [
    "reset_mode_colorbars",
    "apply_mode_colorbars",
    "finalize_mode_figure",
]
