"""Small helpers for HTML repr cards in notebook displays."""

from __future__ import annotations

from html import escape as _esc


def make_simple_card(
    *,
    title: str,
    subtitle: str | None = None,
    rows: list[tuple[str, str]] | None = None,
) -> str:
    """Build a compact info card used by lightweight _repr_html_ methods."""
    subtitle_html = (
        f"<div style='color:#94a3b8;font-size:0.85em;margin-top:4px;'>{_esc(subtitle)}</div>"
        if subtitle
        else ""
    )

    table_html = ""
    if rows:
        body = "".join(
            "<tr>"
            f"<td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(key)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(value)}</td>"
            "</tr>"
            for key, value in rows
        )
        table_html = (
            "<table style='width:100%;margin-top:10px;border-collapse:collapse;font-size:0.9em;'>"
            f"{body}</table>"
        )

    return (
        "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "border:2px solid #334155;border-radius:12px;padding:14px;margin:8px 0;"
        "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
        "color:#e2e8f0;\">"
        f"<div style='font-size:1.03em;font-weight:600;color:#f1f5f9;'>{_esc(title)}</div>"
        f"{subtitle_html}{table_html}</div>"
    )


__all__ = ["make_simple_card"]
