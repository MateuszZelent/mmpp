"""Status-line helpers for interactive modules."""

from __future__ import annotations


def format_status(*, frame_index: int, n_frames: int, extra: str | None = None) -> str:
    """Build compact status text for toolbars or figure subtitles."""
    total = max(0, int(n_frames))
    index = int(frame_index)
    base = f"frame {index + 1}/{total}" if total else f"frame {index + 1}"
    if extra:
        return f"{base} | {extra}"
    return base
