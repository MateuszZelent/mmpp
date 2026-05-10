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
import re
from collections.abc import Sequence
from html import escape as _esc

__all__ = [
    "NODE_COLOR_COMPUTE",
    "NODE_COLOR_ANALYSIS",
    "NODE_COLOR_PLOT",
    "NODE_COLOR_UTIL",
    "NODE_COLOR_ADVANCED",
    "accessors_section_html",
    "api_help_html",
    "examples_section_html",
    "helper_badge_html",
    "helper_card_html",
    "helper_code_chip_html",
    "helper_code_grid_html",
    "helper_table_html",
    "html_tabs",
    "metrics_section_html",
    "node_card_html",
    "plot_accessor_html",
]


_HELPER_CARD_FONT = (
    "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; "
)
_HELPER_CARD_CHROME = (
    "border: 2px solid #6272a4; border-radius: 12px; padding: 18px; "
    "margin: 10px 0; background: linear-gradient(135deg, #282a36 0%, "
    "#21222c 50%, #44475a 100%); color: #f8f8f2; "
    "box-shadow: 0 10px 28px rgba(0,0,0,0.45), "
    "0 0 0 1px rgba(98,114,164,0.15) inset;"
)
_HELPER_SECTION_CHROME = (
    "background: linear-gradient(135deg, rgba(68,71,90,0.55) 0%, "
    "rgba(40,42,54,0.55) 100%); padding: 12px; border-radius: 8px; "
    "margin-bottom: 14px; border: 1px solid rgba(98,114,164,0.25); "
    "backdrop-filter: blur(10px);"
)
_HELPER_CODE_CHIP = (
    "background: rgba(40,42,54,0.9); padding: 5px 10px; "
    "border-radius: 5px; display: inline-block; margin: 4px; "
    "font-family: 'Courier New', monospace; font-size: 0.85em; "
    "border: 1px solid rgba(98,114,164,0.4); font-weight: 500;"
)
# Inner-panel style: same gradient as the card but no outer chrome.
# Used for the Overview tab body inside node_card_html().
_HELPER_CARD_INNER = (
    "padding: 4px 0 0 0;"
)

# ── Node-card chip colour palette ──────────────────────────────────────────
# Use these consistently across all analysis-namespace helpers so that
# "Compute" is always blue, "Analysis" always green, etc.
NODE_COLOR_COMPUTE = "#8be9fd"    # Dracula cyan   — compute / primary actions
NODE_COLOR_ANALYSIS = "#50fa7b"  # Dracula green  — analysis sub-interfaces
NODE_COLOR_PLOT = "#bd93f9"      # Dracula purple — plotting
NODE_COLOR_UTIL = "#ffb86c"      # Dracula orange — utilities / cache
NODE_COLOR_ADVANCED = "#ff79c6" # Dracula pink   — experimental / advanced


def metrics_section_html(
    rows: Sequence[tuple[str, str | object, str | None]],
) -> str:
    """Return the context/status section used inside node-card Overview panels.

    Parameters
    ----------
    rows : [(label, value, color_or_None), ...]
        *color_or_None* overrides the default value text colour (#cbd5e1).
        Pass ``None`` to use the default.  Values are HTML-escaped
        automatically.
    """
    if not rows:
        return ""
    parts = []
    for label, value, color in rows:
        vc = _esc(color) if color else "#cbd5e1"
        parts.append(
            f"<b style='color:#94a3b8'>{_esc(str(label))}:</b> "
            "<code style='background:rgba(15,23,42,0.6);padding:4px 10px;"
            f"border-radius:5px;font-size:0.9em;color:{vc};"
            f"border:1px solid rgba(71,85,105,0.3);'>{_esc(str(value))}</code>"
        )
    inner = "<br>".join(parts)
    return f"<div style='{_HELPER_SECTION_CHROME}'>{inner}</div>"


def accessors_section_html(
    groups: Sequence[tuple[str, Sequence[tuple[str, str]]]],
) -> str:
    """Return the ACCESSORS & METHODS section used inside node-card Overview panels.

    Parameters
    ----------
    groups : [(group_label, [(code, chip_color), ...]), ...]
        *group_label* is shown as small muted text (e.g. ``"Compute:"``).
        *chip_color* should be one of the ``NODE_COLOR_*`` constants.
    """
    if not groups:
        return ""
    chip_style = _HELPER_CODE_CHIP  # no extra colour — caller appends it
    rows = ""
    for label, items in groups:
        chips = "".join(
            f"<code style='{chip_style} color:{_esc(color)};'>{_esc(code)}</code>"
            for code, color in items
        )
        rows += (
            f"<small style='color:#64748b;margin-right:6px;'>{_esc(label)}</small>"
            f"{chips}<br>"
        )
    return (
        f"<div style='{_HELPER_SECTION_CHROME}'>"
        "<b style='color:#94a3b8;'>ACCESSORS &amp; METHODS</b><br>"
        f"{rows}</div>"
    )


def examples_section_html(code: str, *, title: str = "Examples") -> str:
    """Return the Examples section with a dark pre/code block.

    Parameters
    ----------
    code : str
        Raw Python source string.  HTML-escaped automatically.
    title : str
        Section heading (default ``"Examples"``).
    """
    return (
        f"<div style='{_HELPER_SECTION_CHROME}'>"
        f"<b style='color:#94a3b8;'>{_esc(title)}</b><br>"
        "<pre style='margin:6px 0 0 0;background:rgba(15,23,42,0.85);"
        "padding:10px;border-radius:6px;color:#e2e8f0;overflow-x:auto;"
        f"font-size:0.85em;'><code>{_esc(code)}</code></pre>"
        "</div>"
    )


def node_card_html(
    title: str,
    *,
    icon: str = "",
    subtitle: str | None = None,
    badge: tuple[str, str] | None = None,
    sections: Sequence[str] | None = None,
    api: str,
    uid: str,
    extra_tabs: Sequence[tuple[str, str]] | None = None,
) -> str:
    """Return the canonical MMPP analysis-node card.

    This is the **single source of truth** for the interactive helper card
    style.  All analysis-namespace helpers (``job[0].fft``,
    ``job[0].solitons``, ``job[0].m.fft``, ``job[:].fft``, …) must call
    this function instead of hand-writing card HTML.

    Structure
    ---------
    ::

        outer div (gradient background + border + box-shadow)
          title <div>  (always visible, above tabs — NOT <h3>!)
          subtitle <div>
          └─ html_tabs
               ├─ Overview tab
               │    section₁  (from metrics_section_html / accessors_section_html / …)
               │    section₂
               │    …
               └─ API tab
                    api_help_html(..., chrome=False)

    Parameters
    ----------
    title : str
        Card title, e.g. ``"FFT Analysis Interface"``.
    icon : str
        Emoji prefix for the title (e.g. ``"🔬"``).
    subtitle : str, optional
        Muted one-liner below the title.
    badge : (label, bg_color), optional
        Small coloured badge appended to the title, e.g.
        ``("ready", "#22c55e")``.
    sections : list[str], optional
        Pre-built HTML section strings produced by
        ``metrics_section_html()``, ``accessors_section_html()``,
        ``examples_section_html()``, or any custom HTML.
    api : str
        Full API tab body — typically ``api_help_html(..., chrome=False)``.
    uid : str
        Short unique suffix for tab element IDs.  Use a fixed
        domain-specific prefix + short ``uuid4()[:8]`` fragment to avoid
        collisions across multiple notebook cells.
    extra_tabs : list[(label, body)], optional
        Additional tabs inserted after "Overview" and before "API".
    """
    # ── badge ──────────────────────────────────────────────────────────────
    badge_html = ""
    if badge:
        blabel, bcolor = badge
        badge_html = (
            f"<span style='background:{_esc(bcolor)};color:#0f172a;"
            "padding:1px 6px;border-radius:10px;font-size:10px;"
            f"margin-left:8px;'>{_esc(blabel)}</span>"
        )

    # ── title / subtitle ───────────────────────────────────────────────────
    prefix = f"{icon} " if icon else ""
    title_html = "<div style=\"font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; border: 2px solid #334155; border-radius: 12px; padding: 18px; margin: 10px 0; background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%); color: #e2e8f0; box-shadow: 0 10px 25px rgba(0,0,0,0.3), 0 0 0 1px rgba(148,163,184,0.1) inset;\">"
    title_html += (
        "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;"
        "margin:0 0 6px 0;letter-spacing:0.5px;text-shadow:0 2px 4px rgba(0,0,0,0.3);'>"
        f"{prefix}{_esc(title)}{badge_html}</div>"
    )
    subtitle_html = (
        f"<div style='font-size:0.85em;color:#94a3b8;margin-bottom:12px;'>"
        f"{subtitle}</div>"  # subtitle is trusted caller HTML (already escaped/built)
        if subtitle
        else ""
    )

    # ── overview panel (sections only, no title — title lives above tabs) ──
    sections_html = "".join(sections or [])
    overview_html = (
        f"<div style='{_HELPER_CARD_INNER}'>"
        f"{sections_html}"
        "</div>"
    )

    # ── assemble tabs ─────────────────────────────────────────────────────
    tabs: list[tuple[str, str]] = [("Overview", overview_html)]
    for label, body in (extra_tabs or []):
        tabs.append((label, body))
    tabs.append(("API", api))

    # ── outer card: title above tabs (matches MMPP Job Manager pattern) ───
    return (
        f"<div style='{_HELPER_CARD_FONT}{_HELPER_CARD_CHROME}'>"
        f"{title_html}{subtitle_html}"
        + html_tabs(tabs, uid=uid)
        + "</div>"
    )


def _html_id(value: str) -> str:
    """Return a safe HTML id fragment for inline notebook controls."""
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    return safe or "mmpp-helper"


def helper_badge_html(
    label: str,
    *,
    color: str = "#38bdf8",
    text_color: str = "#0f172a",
) -> str:
    """Return the compact status badge used by MMPP helper cards."""
    return (
        f"<span style='background:{_esc(color)};color:{_esc(text_color)};"
        "padding:1px 6px;border-radius:10px;font-size:10px'>"
        f"{_esc(label)}</span>"
    )


def helper_table_html(rows: Sequence[tuple[str, object]]) -> str:
    """Return a compact key/value table for helper card summaries."""
    if not rows:
        return ""
    body = "".join(
        "<tr>"
        f"<td style='color:#94a3b8;padding-right:14px'>{_esc(str(key))}</td>"
        f"<td><code>{_esc(str(value))}</code></td>"
        "</tr>"
        for key, value in rows
    )
    return (
        "<table style='border-collapse:collapse;font-size:12px;margin-bottom:4px'>"
        f"{body}</table>"
    )


def helper_code_grid_html(
    groups: Sequence[tuple[str, Sequence[tuple[str, str, str | None]]]],
) -> str:
    """Return the responsive grouped accessor/action grid used by helper cards.

    ``groups`` is ``[(heading, [(code, description, color), ...]), ...]``.
    The color may be ``None`` to use the default cyan.
    """
    if not groups:
        return ""
    cards = []
    for heading, items in groups:
        item_html = "".join(
            f"<code title='{_esc(desc)}' style=\"{_HELPER_CODE_CHIP}"
            f' color: {_esc(color or "#60a5fa")};">{_esc(code)}</code>'
            for code, desc, color in items
        )
        cards.append(
            "<div style='margin-bottom: 6px;'>"
            f"<small style='color: #64748b; margin-right: 6px;'>{_esc(heading)}:</small>"
            f"{item_html}</div>"
        )
    return "".join(cards)


def _helper_section_html(title: str, body: str) -> str:
    return (
        f"<div style='{_HELPER_SECTION_CHROME}'>"
        f'<b style="color: #94a3b8;">{_esc(title)}</b><br>'
        f"{body}"
        "</div>"
    )


def _helper_metrics_html(rows: Sequence[tuple[str, object]]) -> str:
    if not rows:
        return ""
    parts = []
    for key, value in rows:
        parts.append(
            f'<b style="color: #94a3b8;">{_esc(str(key))}:</b> '
            '<code style="background: rgba(15,23,42,0.6); padding: 4px 10px; '
            "border-radius: 5px; font-family: 'Courier New', monospace; "
            "font-size: 0.9em; color: #cbd5e1; "
            f'border: 1px solid rgba(71,85,105,0.3);">{_esc(str(value))}</code>'
        )
    return "<br>".join(parts)


def helper_code_chip_html(code: str, *, color: str = "#60a5fa") -> str:
    """Return the exact inline code-chip style used by the top-level job helper."""
    return (
        f'<code style="{_HELPER_CODE_CHIP} color: {_esc(color)};">{_esc(code)}</code>'
    )


def helper_card_html(
    title: str,
    *,
    subtitle: str | None = None,
    status: tuple[str, str] | None = None,
    metrics: Sequence[tuple[str, object]] | None = None,
    details: Sequence[tuple[str, str]] | None = None,
    tabs: Sequence[tuple[str, str]] | None = None,
    action_groups: Sequence[tuple[str, Sequence[tuple[str, str, str | None]]]]
    | None = None,
    uid: str | None = None,
    max_width: str | None = None,
    accent: str = "#334155",
) -> str:
    """Return the canonical rich HTML template for MMPP notebook helpers.

    The template is based on ``MMPP._repr_html_``: responsive dark gradient
    card, stat/action sections, and optional JavaScript-backed Overview/API
    tabs.
    """
    badge = ""
    if status:
        label, color = status
        badge = f"&nbsp;&nbsp;{helper_badge_html(label, color=color)}"

    subtitle_html = (
        f'<div style="color: #94a3b8; margin-top: 4px;">{_esc(subtitle)}</div>'
        if subtitle
        else ""
    )
    overview_parts: list[str] = []
    if metrics:
        overview_parts.append(
            _helper_section_html("Status", _helper_metrics_html(metrics))
        )
    if action_groups:
        overview_parts.append(
            _helper_section_html(
                "ACCESSORS & METHODS",
                helper_code_grid_html(action_groups),
            )
        )
    overview_parts.extend(
        _helper_section_html(label, body) for label, body in (details or [])
    )

    body_html = "".join(overview_parts)
    if tabs:
        tabs_uid = uid or _html_id(title)
        tab_items = list(tabs)
        if overview_parts:
            first_label, first_body = tab_items[0]
            if first_label.lower() == "overview":
                tab_items[0] = (first_label, body_html + first_body)
            else:
                tab_items.insert(0, ("Overview", body_html))
        body_html = html_tabs(tab_items, uid=tabs_uid)

    return (
        f"<div style='{_HELPER_CARD_FONT}{_HELPER_CARD_CHROME}"
        f"{f'max-width: {_esc(max_width)}; ' if max_width else ''}"
        f"border-color: {_esc(accent)};'>"
        '<h3 style="margin: 0 0 12px 0; color: #f1f5f9; font-weight: 600; '
        'letter-spacing: 0.5px; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">'
        f"{_esc(title)}{badge}</h3>"
        f"{subtitle_html}"
        f"{body_html}"
        "</div>"
    )


def html_tabs(tabs: Sequence[tuple[str, str]], *, uid: str) -> str:
    """Return inline HTML tabs for notebook cards."""
    if not tabs:
        return ""
    tab_button_style = (
        "padding:7px 12px;border:1px solid rgba(96,165,250,0.35);"
        "border-radius:6px;background:rgba(15,23,42,0.65);color:#93c5fd;"
        "cursor:pointer;font-size:0.85em;font-weight:600;margin-right:6px;"
    )
    active_tab_style = (
        "padding:7px 12px;border:1px solid rgba(96,165,250,0.65);"
        "border-radius:6px;background:rgba(96,165,250,0.22);color:#dbeafe;"
        "cursor:pointer;font-size:0.85em;font-weight:700;margin-right:6px;"
    )
    button_html = ""
    panel_html = ""
    panel_ids = [f"{uid}-panel-{idx}" for idx, _ in enumerate(tabs)]
    button_ids = [f"{uid}-tab-{idx}" for idx, _ in enumerate(tabs)]
    for idx, ((label, body), panel_id, button_id) in enumerate(
        zip(tabs, panel_ids, button_ids)
    ):
        active = idx == 0
        show_panels = ";".join(
            f"document.getElementById('{pid}').style.display="
            f"'{('block' if pid == panel_id else 'none')}';"
            for pid in panel_ids
        )
        style_buttons = ";".join(
            f"document.getElementById('{bid}').style.cssText="
            f"'{(active_tab_style if bid == button_id else tab_button_style)}';"
            for bid in button_ids
        )
        button_html += (
            f"<button id='{button_id}' style='{active_tab_style if active else tab_button_style}' "
            f'onclick="{show_panels}{style_buttons}">{_esc(label)}</button>'
        )
        panel_html += (
            f"<div id='{panel_id}' style='display:{'block' if active else 'none'};'>"
            f"{body}</div>"
        )
    return f"<div style='margin-bottom:12px;'>{button_html}</div>{panel_html}"


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
    chrome: bool = True,
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
    inner = (
        f"<div style='font-size:1.04em;font-weight:700;color:#f1f5f9;'>{_esc(title)}</div>"
        f"{subtitle_html}{props_block}"
        "<div style='font-weight:600;color:#f1f5f9;margin:10px 0 4px;'>Methods</div>"
        "<table style='width:100%;border-collapse:collapse;font-size:.86em;'>"
        "<thead><tr style='text-align:left;color:#94a3b8;'>"
        "<th style='padding:4px 8px;'>Signature</th><th style='padding:4px 8px;'>Description</th>"
        "<th style='padding:4px 8px;'>Example</th></tr></thead>"
        f"<tbody>{method_rows}</tbody></table>"
    )
    if not chrome:
        return (
            "<div style='background:rgba(15,23,42,0.6);padding:12px;"
            "border-radius:8px;margin-top:10px;border:1px solid rgba(148,163,184,0.2);'>"
            f"{inner}</div>"
        )

    return (
        "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "border:2px solid #334155;border-radius:12px;padding:14px;margin:8px 0;"
        "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
        'color:#e2e8f0;max-width:980px;">'
        f"{inner}</div>"
    )
