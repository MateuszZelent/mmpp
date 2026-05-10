"""Shared HTML-card generator for PlotAccessor ``_repr_html_`` methods.

Every ``PlotAccessor`` can produce a rich Jupyter card by calling::

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html
        return plot_accessor_html("MyPlotAccessor", [
            (".method(a, b)", "Short description", "Longer tooltip"),
        ])
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from html import escape as _esc

__all__ = ["api_help_html", "plot_accessor_html"]


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
            f"<div style='margin-top:8px;font-size:.78em;color:#475569;'>{footer}</div>"
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


def _public_callables(
    obj: object, names: Sequence[str] | None = None
) -> list[tuple[str, object]]:
    selected = names if names is not None else dir(obj)
    out: list[tuple[str, object]] = []
    for name in selected:
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            out.append((name, value))
    return out


def _signature_text(func: object) -> str:
    try:
        sig = str(inspect.signature(func))
    except Exception:
        return "(...)"
    if len(sig) > 90:
        sig = sig[:87] + "..."
    return sig


def _summary_text(func: object) -> str:
    doc = inspect.getdoc(func) or ""
    if not doc:
        return "No docstring summary available."
    first = doc.strip().splitlines()[0].strip()
    return first or "No docstring summary available."


def _example_for(prefix: str, name: str, func: object) -> str:
    call_args = ""
    params = []
    try:
        signature = inspect.signature(func)
        for param in signature.parameters.values():
            if param.name in {"self", "cls"}:
                continue
            if param.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            if param.default is inspect.Parameter.empty:
                params.append(f"{param.name}=...")
        call_args = ", ".join(params[:3])
    except Exception:
        call_args = ""
    return f"{prefix}.{name}({call_args})"


def api_help_html(
    obj: object,
    *,
    title: str | None = None,
    prefix: str = "obj",
    methods: Sequence[str] | None = None,
    properties: Sequence[tuple[str, str]] | None = None,
    max_methods: int | None = None,
    subtitle: str | None = None,
) -> str:
    """Return an API card with methods, signatures, parameter details and examples.

    The card is generated from the live object, so it stays aligned with method
    signatures as the API evolves.
    """
    title = title or obj.__class__.__name__
    callables = _public_callables(obj, methods)
    if max_methods is not None:
        callables = callables[: int(max_methods)]

    prop_rows = ""
    if properties:
        prop_rows = "".join(
            "<tr>"
            f"<td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(name)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(desc)}</td>"
            f"<td style='padding:4px 8px;font-family:monospace;color:#a7f3d0;'>{_esc(prefix + '.' + name)}</td>"
            "</tr>"
            for name, desc in properties
        )

    method_rows = "".join(
        "<tr style='border-top:1px solid rgba(71,85,105,.35);'>"
        f"<td style='padding:5px 8px;font-family:monospace;color:#93c5fd;white-space:nowrap;'>"
        f"{_esc('.' + name + _signature_text(func))}</td>"
        f"<td style='padding:5px 8px;color:#cbd5e1;'>{_esc(_summary_text(func))}</td>"
        f"<td style='padding:5px 8px;font-family:monospace;color:#a7f3d0;'>{_esc(_example_for(prefix, name, func))}</td>"
        "</tr>"
        for name, func in callables
    )
    if not method_rows:
        method_rows = (
            "<tr><td colspan='3' style='padding:6px 8px;color:#94a3b8;'>"
            "No public callable methods detected.</td></tr>"
        )

    props_block = ""
    if prop_rows:
        props_block = (
            "<div style='font-weight:600;color:#f1f5f9;margin:10px 0 4px;'>Namespaces / properties</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:.88em;'>"
            "<thead><tr style='text-align:left;color:#94a3b8;'>"
            "<th style='padding:4px 8px;'>Accessor</th><th style='padding:4px 8px;'>Description</th>"
            "<th style='padding:4px 8px;'>Example</th></tr></thead>"
            f"<tbody>{prop_rows}</tbody></table>"
        )

    subtitle_html = (
        f"<div style='color:#94a3b8;font-size:.85em;margin-top:3px;'>{_esc(subtitle)}</div>"
        if subtitle
        else ""
    )
    return (
        "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "border:2px solid #334155;border-radius:12px;padding:14px;margin:8px 0;"
        "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
        'color:#e2e8f0;max-width:980px;">'
        f"<div style='font-size:1.04em;font-weight:700;color:#f1f5f9;'>{_esc(title)}</div>"
        f"{subtitle_html}{props_block}"
        "<div style='font-weight:600;color:#f1f5f9;margin:10px 0 4px;'>Methods</div>"
        "<table style='width:100%;border-collapse:collapse;font-size:.86em;'>"
        "<thead><tr style='text-align:left;color:#94a3b8;'>"
        "<th style='padding:4px 8px;'>Signature</th><th style='padding:4px 8px;'>Description</th>"
        "<th style='padding:4px 8px;'>Example</th></tr></thead>"
        f"<tbody>{method_rows}</tbody></table>"
        "</div>"
    )
