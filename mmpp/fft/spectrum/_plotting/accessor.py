"""Fluent plotting accessor for spectrum results."""

from __future__ import annotations

from .static import plot_spectrum


class SpectrumPlotAccessor:
    """Plotting namespace available via ``spec.plt``."""

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
        return (
            "<div style='font-family:sans-serif;border:1px solid #475569;border-radius:8px;"
            "padding:12px;background:#1e293b;color:#e2e8f0;'>"
            "<b>SpectrumPlotAccessor</b><br/>"
            "<code>spec.plt.spectrum(...)</code> - static spectrum plot<br/>"
            "<code>spec.plt.interactive(...)</code> - interactive spectrum explorer"
            "</div>"
        )

