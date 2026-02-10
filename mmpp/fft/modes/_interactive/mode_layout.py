"""Mode-figure layout helpers for :mod:`mmpp.fft.modes.interactive`."""

from __future__ import annotations

from typing import Any


def reset_mode_colorbars(explorer: Any) -> None:
    """Remove existing mode colorbars before redraw."""
    for cbar in explorer._mode_colorbars:
        try:
            cbar.remove()
        except Exception:
            pass
    explorer._mode_colorbars = []


def apply_mode_colorbars(explorer: Any, row_images: list[Any]) -> None:
    """Attach one colorbar per mode-row image."""
    if explorer._fig is None or explorer._mode_axes is None:
        return

    for row_idx, img in enumerate(row_images):
        if img is None:
            continue
        try:
            cbar = explorer._fig.colorbar(
                img,
                ax=list(explorer._mode_axes[row_idx, :]),
                fraction=0.035,
                pad=0.02,
            )
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
            explorer._fig.tight_layout(rect=[0, 0, 1, 0.97])
        except Exception:
            pass

    explorer._fig.canvas.draw_idle()


__all__ = [
    "reset_mode_colorbars",
    "apply_mode_colorbars",
    "finalize_mode_figure",
]
