"""Hysteresis plotting namespace."""

from __future__ import annotations

from html import escape as _esc

from .accessor import HysteresisPlotAccessor


def interactive(
    job_result,
    *,
    field: str | None = None,
    magnetization: str | None = None,
    **kwargs,
):
    """Functional entrypoint for interactive hysteresis explorer.

    Examples
    --------
    >>> mmpp.analyze.hysteresis.plot.interactive(job[0], field="B_extx", magnetization="mx")
    """
    from .. import HysteresisInterface

    interface = HysteresisInterface(job_result)
    return interface.plot.interactive(
        field=field,
        magnetization=magnetization,
        **kwargs,
    )


def loop(
    job_result,
    *,
    field: str | None = None,
    magnetization: str | None = None,
    **kwargs,
):
    """Functional entrypoint for static hysteresis loop plotting."""
    from .. import HysteresisInterface

    interface = HysteresisInterface(job_result)
    return interface.plot.loop(
        field=field,
        magnetization=magnetization,
        **kwargs,
    )

def animation(
    job_result,
    *,
    field: str | None = None,
    magnetization: str | None = None,
    **kwargs,
):
    """Functional entrypoint for hysteresis loop animation."""
    from .. import HysteresisInterface

    interface = HysteresisInterface(job_result)
    return interface.plot.animation(
        field=field,
        magnetization=magnetization,
        **kwargs,
    )


__all__ = ["HysteresisPlotAccessor", "interactive", "loop", "animation"]


def _repr_html_() -> str:
    methods = [
        ("interactive(job_result, ...)", "Functional entrypoint for 2-panel interactive explorer"),
        ("loop(job_result, ...)", "Functional entrypoint for static hysteresis loop"),
        ("animation(job_result, ...)", "Functional entrypoint for MP4/GIF or online animation"),
    ]
    rows = "".join(
        f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(name)}</td>"
        f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(desc)}</td></tr>"
        for name, desc in methods
    )
    example = "\n".join(
        [
            "import mmpp",
            "job = mmpp.open('/path/to/sim')[0]",
            "mmpp.analyze.hysteresis.plot.interactive(job, field='B_extx', magnetization='mx')",
        ]
    )
    return (
        "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "border:2px solid #334155;border-radius:12px;padding:14px;margin:8px 0;"
        "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
        "color:#e2e8f0;\">"
        "<div style='font-size:1.03em;font-weight:600;color:#f1f5f9;'>"
        "Hysteresis Plot Module</div>"
        "<table style='width:100%;margin-top:8px;border-collapse:collapse;font-size:0.9em;'>"
        "<thead><tr style='text-align:left;background:rgba(51,65,85,0.6);'>"
        "<th style='padding:6px 8px;color:#e2e8f0;'>Function</th>"
        "<th style='padding:6px 8px;color:#e2e8f0;'>Description</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        "<div style='font-weight:600;color:#e2e8f0;margin-top:10px;margin-bottom:6px;'>Example</div>"
        "<pre style='margin:0;background:rgba(15,23,42,0.85);padding:10px;border-radius:6px;"
        f"color:#e2e8f0;overflow-x:auto;font-size:0.85em;'><code>{_esc(example)}</code></pre>"
        "</div>"
    )


def _repr_mimebundle_(include=None, exclude=None):
    html = _repr_html_()
    text = "<module 'mmpp.analyze.hysteresis.plot'>"
    if html:
        return {"text/html": html, "text/plain": text}
    return {"text/plain": text}
