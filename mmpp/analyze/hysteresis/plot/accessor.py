"""Fluent plot accessor for hysteresis results."""

from __future__ import annotations

from mmpp._repr_helpers import (
    NODE_COLOR_ANALYSIS,
    NODE_COLOR_COMPUTE,
    NODE_COLOR_PLOT,
    accessors_section_html,
    api_help_html,
    examples_section_html,
    metrics_section_html,
    node_card_html,
)

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
        example = "\n".join(
            [
                "plot = result.plot",
                "plot.loop(show_hc=True)",
                "plot.interactive(toolbar='auto')",
                "plot.animation(writer='pillow')",
            ]
        )
        api = api_help_html(
            self,
            title="Hysteresis plot API help",
            prefix="result.plot",
            methods=["loop", "interactive", "animation"],
            subtitle="Plotting methods exposed by HysteresisResult.plot.",
            chrome=False,
        )
        return node_card_html(
            "Hysteresis Plot Accessor",
            icon="📉",
            subtitle="Static, interactive and animated views exposed by HysteresisResult.plot.",
            sections=[
                metrics_section_html(
                    [
                        ("owner", "HysteresisResult.plot", NODE_COLOR_COMPUTE),
                        (
                            "modes",
                            "static / interactive / animation",
                            NODE_COLOR_ANALYSIS,
                        ),
                    ]
                ),
                accessors_section_html(
                    [
                        (
                            "Plot:",
                            [
                                (".loop(**kwargs)", NODE_COLOR_COMPUTE),
                                (".interactive(**kwargs)", NODE_COLOR_ANALYSIS),
                                (".animation(**kwargs)", NODE_COLOR_PLOT),
                            ],
                        )
                    ]
                ),
                examples_section_html(example),
            ],
            api=api,
            uid="mmpp-hysteresis-plot",
        )

    def _repr_mimebundle_(self, include=None, exclude=None):
        html = self._repr_html_()
        text = self.__repr__()
        if html:
            return {"text/html": html, "text/plain": text}
        return {"text/plain": text}
