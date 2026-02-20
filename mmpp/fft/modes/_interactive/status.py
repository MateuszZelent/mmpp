"""Status message helpers for :mod:`mmpp.fft.modes.interactive`."""

from __future__ import annotations

from typing import Any


def set_status(explorer: Any, message: str, color: str = "#334155", logger: Any = None) -> None:
    """Set status message in toolbar or fallback logger."""
    if explorer._controls and "status" in explorer._controls:
        explorer._controls["status"].value = f"<small style='color:{color}'>{message}</small>"
        return

    if logger is not None:
        logger.info(message)


def update_status_text(explorer: Any, logger: Any = None) -> None:
    """Refresh compact status summary for current interactive state."""
    if not explorer._controls:
        return

    n_peaks = len(explorer._peaks)
    if explorer._current_frequency_ghz is None:
        freq_text = "n/a"
    else:
        freq_text = f"{explorer._current_frequency_ghz:.3f} GHz"

    loaded_freq = getattr(explorer, "_loaded_frequency_ghz", None)
    if (
        loaded_freq is not None
        and explorer._current_frequency_ghz is not None
        and abs(float(loaded_freq) - float(explorer._current_frequency_ghz)) > 1e-6
    ):
        freq_text = f"{freq_text} (mode@{float(loaded_freq):.3f} GHz)"

    set_status(
        explorer,
        f"f={freq_text}, components={','.join(explorer._current_components)}, peaks={n_peaks}",
        color="#334155",
        logger=logger,
    )


__all__ = ["set_status", "update_status_text"]
