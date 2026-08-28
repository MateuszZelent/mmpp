"""Status message helpers for :mod:`mmpp.fft.modes.interactive`."""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any


def _append_status_history(explorer: Any, message: str, color: str) -> None:
    """Append one message to in-widget status history."""
    history = getattr(explorer, "_status_history", None)
    if history is None:
        history = []
        explorer._status_history = history

    timestamp = datetime.now().strftime("%H:%M:%S")
    safe_message = html.escape(str(message))
    history.append(
        f"<div style='color:{color}'><code>{timestamp}</code> {safe_message}</div>"
    )
    if len(history) > 120:
        del history[:-120]

    if explorer._controls and "status_log" in explorer._controls:
        explorer._controls["status_log"].value = (
            "<div style='max-height:150px;overflow:auto;"
            "font-family:monospace;font-size:11px;line-height:1.25;"
            "border:1px solid #e5e7eb;padding:4px;background:#f8fafc;'>"
            + "".join(history[-20:])
            + "</div>"
        )


def set_status(
    explorer: Any,
    message: str,
    color: str = "#334155",
    logger: Any = None,
    persist: bool = True,
) -> None:
    """Set status message in toolbar or fallback logger."""
    safe_message = html.escape(str(message))
    if explorer._controls and "status" in explorer._controls:
        explorer._controls[
            "status"
        ].value = f"<small style='color:{color}'>{safe_message}</small>"
        if persist:
            _append_status_history(explorer, message, color=color)
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
        (
            f"f={freq_text}, "
            f"mode={','.join(explorer._mode_components)}, "
            f"spectrum={','.join(explorer._spectrum_components)}, "
            f"peaks={n_peaks}"
        ),
        color="#334155",
        logger=logger,
        persist=False,
    )


__all__ = ["set_status", "update_status_text"]
