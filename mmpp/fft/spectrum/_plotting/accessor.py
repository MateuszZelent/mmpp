"""Fluent plotting accessor for spectrum results."""

from __future__ import annotations

from .static import plot_spectrum


class SpectrumPlotAccessor:
    """Plotting namespace available via ``spec.plot``."""

    def __init__(self, result):
        self._result = result

    def spectrum(self, **kwargs):
        """Plot spectrum using static matplotlib view."""
        return plot_spectrum(self._result, **kwargs)

    def interactive(self, **kwargs):
        """Open interactive explorer routed through modes interface."""
        return self._result.modes.interactive(**kwargs)

    def __repr__(self) -> str:
        return "<SpectrumPlotAccessor: .spectrum(), .interactive()>"

    def _repr_html_(self) -> str:
        import uuid as _uuid

        from mmpp._repr_helpers import api_help_html, html_tabs

        methods = [
            (".spectrum(**kwargs)", "Static matplotlib spectrum plot"),
            (
                ".interactive(**kwargs)",
                "Interactive spectrum explorer (Jupyter widget)",
            ),
        ]
        method_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{m}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{d}</td></tr>"
            for m, d in methods
        )
        params = [
            ("figsize", "(10, 4)", "Figure size (width, height) in inches"),
            ("dpi", "None", "Figure resolution override"),
            ("show_peaks", "True", "Mark detected resonance peaks"),
            ("peak_prominence", "0.01", "Minimum peak prominence for detection"),
            ("xlim", "None", "Frequency axis limits (f_min, f_max) in GHz"),
            ("component", "None", "Magnetization component to plot"),
            ("title", "None", "Custom plot title"),
            ("save", "None", "Path to save figure or True for auto-name"),
        ]
        param_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{n}</td>"
            f"<td style='padding:4px 8px;color:#a5b4fc;'>{d}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{desc}</td></tr>"
            for n, d, desc in params
        )
        example = (
            "# Static spectrum plot\n"
            "spec.plot.spectrum(show_peaks=True, xlim=(0, 30))\n"
            "\n"
            "# Interactive explorer\n"
            "spec.plot.interactive(component='z')"
        )
        html = (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            'color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);">'
            "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;margin-bottom:4px;'>"
            "Spectrum Plot Accessor</div>"
            "<div style='font-size:0.85em;color:#94a3b8;margin-bottom:10px;'>"
            'Plotting namespace available via <code style="color:#a5b4fc;">spec.plot</code></div>'
            # Methods
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Methods</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            f"{method_rows}</table></div>"
            # Parameters
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>"
            "Parameters <span style='color:#94a3b8;font-weight:400;'>(spectrum)</span></div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            "<thead><tr style='text-align:left;background:rgba(51,65,85,0.6);'>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Arg</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Default</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Description</th></tr></thead>"
            f"<tbody>{param_rows}</tbody></table></div>"
            # Examples
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Examples</div>"
            "<pre style='margin:0;background:rgba(15,23,42,0.85);padding:10px;"
            "border-radius:6px;color:#e2e8f0;overflow-x:auto;font-size:0.85em;'>"
            f"<code>{example}</code></pre></div>"
            "</div>"
        )
        api_card = api_help_html(
            self,
            title="Spectrum plot accessor API help",
            prefix="spec.plot",
            methods=["spectrum", "interactive"],
            subtitle=(
                "Live signatures for the plotting namespace returned by "
                "SpectrumResult.plot."
            ),
            chrome=False,
        )
        return (
            f"<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:14px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            'color:#e2e8f0;">'
            + html_tabs(
                [("Overview", html), ("API", api_card)],
                uid=f"spectrum-plot-{str(_uuid.uuid4())[:8]}",
            )
            + "</div>"
        )
