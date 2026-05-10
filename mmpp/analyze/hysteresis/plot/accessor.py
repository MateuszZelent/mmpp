"""Fluent plot accessor for hysteresis results."""

from __future__ import annotations

import uuid

from mmpp._repr_helpers import api_help_html, html_tabs

from .interactive import HysteresisInteractiveExplorer
from .static import plot_loop


class HysteresisPlotAccessor:
    """Plotting namespace exposed as ``result.plot``."""

    def __init__(self, result):
        self._result = result

    def loop(self, **kwargs):
        """Plot static M(B) loop view."""
        return plot_loop(self._result, **kwargs)

    def interactive(self, **kwargs):
        """Open interactive loop + snapshot explorer."""
        # Backward compatibility: some live notebook sessions may still hold
        # an older explorer implementation where ``debug_clicks`` is required.
        kwargs.setdefault("debug_clicks", None)
        kwargs.setdefault("loop_width", None)
        kwargs.setdefault("snapshot_width", None)
        explorer = HysteresisInteractiveExplorer(self._result)
        return explorer.show(**kwargs)

    def animation(self, **kwargs):
        """Create or export animation walkthrough over the loop."""
        from ..animation import create_animation

        return create_animation(self._result, **kwargs)

    def __repr__(self) -> str:
        return "<HysteresisPlotAccessor: .loop(), .interactive(), .animation()>"

    def _repr_html_(self) -> str:
        methods = [
            (".loop(**kwargs)", "Static hysteresis loop plot"),
            (".interactive(**kwargs)", "Interactive loop + snapshot explorer"),
            (".animation(**kwargs)", "Online or exported MP4/GIF loop animation"),
        ]
        rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{m}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{d}</td></tr>"
            for m, d in methods
        )
        overview = (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            'color:#e2e8f0;">'
            "<div style='font-size:1.05em;font-weight:600;color:#f1f5f9;'>"
            "Hysteresis Plot Accessor</div>"
            "<table style='width:100%;margin-top:8px;border-collapse:collapse;font-size:0.9em;'>"
            f"{rows}</table></div>"
        )
        api = api_help_html(
            self,
            title="Hysteresis plot API help",
            prefix="result.plot",
            methods=["loop", "interactive", "animation"],
            subtitle="Plotting methods exposed by HysteresisResult.plot.",
            chrome=False,
        )
        return (
            '<div style=\'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;'
            "border:2px solid #334155;border-radius:12px;padding:14px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;'>"
            + html_tabs(
                [("Overview", overview), ("API", api)],
                uid=f"mmpp-hysteresis-plot-{uuid.uuid4().hex}",
            )
            + "</div>"
        )

    def _repr_mimebundle_(self, include=None, exclude=None):
        html = self._repr_html_()
        text = self.__repr__()
        if html:
            return {"text/html": html, "text/plain": text}
        return {"text/plain": text}
