"""Status helpers for interactive dispersion explorer."""

from __future__ import annotations

import html
from datetime import datetime


def set_status(explorer, message: str, *, color: str = "#334155") -> None:
    """Push status line to widget log or fallback history."""
    if not hasattr(explorer, "_status_history"):
        explorer._status_history = []

    timestamp = datetime.now().strftime("%H:%M:%S")
    line = (
        f"<div style='color:{color}'><code>{timestamp}</code> "
        f"{html.escape(str(message))}</div>"
    )
    explorer._status_history.append(line)
    if len(explorer._status_history) > 200:
        del explorer._status_history[:-200]

    controls = getattr(explorer, "controls", {})
    if controls and "status" in controls:
        controls["status"].value = line
    if controls and "status_log" in controls:
        controls["status_log"].value = (
            "<div style='max-height:120px;overflow:auto;"
            "font-family:monospace;font-size:11px;line-height:1.2;"
            "border:1px solid #e5e7eb;padding:4px;background:#f8fafc;'>"
            + "".join(explorer._status_history[-15:])
            + "</div>"
        )
