"""Shared HTML-card generator for PlotAccessor ``_repr_html_`` methods.

Every ``PlotAccessor`` can produce a rich Jupyter card by calling::

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html
        return plot_accessor_html("MyPlotAccessor", [
            (".method(a, b)", "Short description", "Longer tooltip"),
        ])
"""

from __future__ import annotations

from html import escape as _esc
from typing import Sequence

__all__ = ["plot_accessor_html"]


def plot_accessor_html(
    title: str,
    methods: Sequence[tuple[str, str, str]],
    *,
    footer: str | None = None,
    accent: str = "#1d4ed8",
    title_color: str = "#60a5fa",
) -> str:
    """Return a styled HTML card listing *methods* of a PlotAccessor.

    Parameters
    ----------
    title : str
        Class name shown in the card header.
    methods : list of (signature, description, tooltip)
        Each tuple describes one public method.  The *tooltip* is shown
        on hover and should document key parameters.
    footer : str, optional
        Extra text below the table (HTML allowed).
    accent : str
        CSS border colour (default: blue).
    title_color : str
        CSS colour for the title text.
    """
    HV = (
        "onmouseover=\"this.style.background='#1e293b'\" "
        "onmouseout=\"this.style.background='transparent'\""
    )

    rows = "".join(
        f"<tr {HV} title=\"{_esc(tip)}\" style='cursor:pointer;'>"
        f"<td style='padding:4px 10px;font-family:monospace;color:#93c5fd;"
        f"font-size:.88em;white-space:nowrap;'>{_esc(sig)}</td>"
        f"<td style='padding:4px 10px;color:#cbd5e1;font-size:.85em;'>"
        f"{_esc(desc)}</td></tr>"
        for sig, desc, tip in methods
    )

    footer_html = ""
    if footer:
        footer_html = (
            f"<div style='margin-top:8px;font-size:.78em;color:#475569;'>"
            f"{footer}</div>"
        )
    else:
        footer_html = (
            "<div style='margin-top:8px;font-size:.78em;color:#475569;'>"
            "All methods return <code style='color:#bae6fd;'>(fig, ax)</code> "
            "and accept <code style='color:#bae6fd;'>save=</code> path."
            "</div>"
        )

    return (
        f"<div style='font-family:-apple-system,sans-serif;border:2px solid {accent};"
        f"border-radius:10px;padding:12px;margin:6px 0;background:#0f172a;"
        f"color:#e2e8f0;max-width:720px;'>"
        f"<div style='font-weight:700;color:{title_color};margin-bottom:8px;'>"
        f"{_esc(title)}"
        f"<span style='font-size:.75em;color:#475569;font-weight:400;"
        f"margin-left:8px;'>(hover rows for parameter details)</span></div>"
        f"<table style='width:100%;border-collapse:collapse;'>{rows}</table>"
        f"{footer_html}</div>"
    )
