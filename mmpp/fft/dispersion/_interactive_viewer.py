"""Lightweight interactive-dispersion controller.

This module intentionally has no Matplotlib/IPython/ipywidgets imports at module
import time. It is the stable, testable object returned by fluent interactive
APIs; notebook rendering can be layered on top by calling :meth:`show`.
"""

import json
from html import escape
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ._json import json_safe

if TYPE_CHECKING:
    from .models import DispersionResult1D


def _normalize_interactive_options(
    *,
    components: Optional[list[str]] = None,
    mode_components: Optional[list[str]] = None,
    spectrum_components: Optional[list[str]] = None,
    animate: Optional[bool] = None,
    auto_animate: Optional[bool] = None,
    **kwargs: Any,
) -> tuple[Optional[list[str]], Optional[list[str]], dict[str, Any]]:
    """Normalize compatibility aliases used by spectrum/modes notebooks."""
    if components is not None:
        if mode_components is None:
            mode_components = list(components)
        if spectrum_components is None:
            spectrum_components = list(components)

    options = dict(kwargs)
    options.setdefault("positive_frequencies", True)
    if auto_animate is None and animate is not None:
        auto_animate = bool(animate)
    if auto_animate is not None:
        options["auto_animate"] = bool(auto_animate)

    return mode_components, spectrum_components, options


@dataclass
class DispersionInteractiveViewer:
    """Stable controller returned by dispersion interactive APIs."""

    result: "DispersionResult1D"
    show_requested: bool = True
    mode_components: Optional[list[str]] = None
    spectrum_components: Optional[list[str]] = None
    can_reconstruct_modes: bool = False
    mode_unavailable_reason: str = ""
    options: dict[str, Any] = field(default_factory=dict)
    _display_handle: Any = None
    _widget: Any = None
    _figure: Any = None
    _axes: Any = None
    _controls: dict[str, Any] = field(default_factory=dict)
    _widget_engine: Any = None
    _widget_status: str = "not-shown"
    _widget_error: str = ""

    @classmethod
    def from_result(
        cls,
        result: "DispersionResult1D",
        *,
        show: bool = True,
        can_reconstruct_modes: Optional[bool] = None,
        mode_unavailable_reason: str = "",
        components: Optional[list[str]] = None,
        mode_components: Optional[list[str]] = None,
        spectrum_components: Optional[list[str]] = None,
        animate: Optional[bool] = None,
        auto_animate: Optional[bool] = None,
        **kwargs: Any,
    ) -> "DispersionInteractiveViewer":
        mode_components, spectrum_components, options = _normalize_interactive_options(
            components=components,
            mode_components=mode_components,
            spectrum_components=spectrum_components,
            animate=animate,
            auto_animate=auto_animate,
            **kwargs,
        )
        if can_reconstruct_modes is None:
            can_reconstruct_modes = result.S_complex is not None
        if not can_reconstruct_modes and not mode_unavailable_reason:
            mode_unavailable_reason = (
                "Mode reconstruction requires S_complex and source FFT context."
            )

        viewer = cls(
            result=result,
            show_requested=bool(show),
            mode_components=mode_components,
            spectrum_components=spectrum_components,
            can_reconstruct_modes=bool(can_reconstruct_modes),
            mode_unavailable_reason=mode_unavailable_reason,
            options=options,
        )
        if show:
            viewer.show()
        return viewer

    def show(self) -> "DispersionInteractiveViewer":
        """Display an interactive dispersion heatmap when notebook deps exist."""
        self.show_requested = True
        try:
            from IPython.display import display
        except ImportError:
            return self

        try:
            widget = self._build_widget(display)
        except Exception as exc:
            self._widget_status = "fallback"
            self._widget_error = f"{type(exc).__name__}: {exc}"
            self._close_widget_state()
            widget = None
        self._display_handle = display(widget if widget is not None else self, display_id=True)
        return self

    def close(self) -> None:
        """Best-effort close/update hook for notebook integrations."""
        if self._display_handle is not None and hasattr(self._display_handle, "update"):
            self._display_handle.update(None)
        self._close_widget_state()
        self._display_handle = None
        self._widget_status = "closed"
        self.show_requested = False

    def _close_widget_state(self) -> None:
        """Release widget and Matplotlib resources owned by this viewer."""
        if self._widget_engine is not None:
            try:
                self._widget_engine.close()
            except Exception:
                pass
        for control in list(self._controls.values()):
            if hasattr(control, "close"):
                try:
                    control.close()
                except Exception:
                    pass
        if self._widget is not None and hasattr(self._widget, "close"):
            try:
                self._widget.close()
            except Exception:
                pass
        if self._figure is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self._figure)
            except Exception:
                pass
        self._widget = None
        self._figure = None
        self._axes = None
        self._controls = {}
        self._widget_engine = None

    @property
    def state(self) -> dict[str, Any]:
        """Serializable viewer state for tests, presets, and notebooks."""
        result_notes = list(getattr(self.result, "notes", None) or [])
        return {
            "show": self.show_requested,
            "mode_components": self.mode_components,
            "spectrum_components": self.spectrum_components,
            "can_reconstruct_modes": self.can_reconstruct_modes,
            "mode_unavailable_reason": self.mode_unavailable_reason,
            "result_notes": result_notes,
            "widget_status": self._widget_status,
            "widget_error": self._widget_error,
            "options": json_safe(self.options),
        }

    def export_selection(self, **selection: Any) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of viewer state and selection."""
        return {
            "viewer": self.state,
            "selection": json_safe(selection),
        }

    def save_preset(self, path: str | Path) -> Path:
        """Persist lightweight viewer state to a JSON preset file."""
        preset_path = Path(path)
        preset_path.parent.mkdir(parents=True, exist_ok=True)
        preset_path.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
        return preset_path

    def load_preset(self, path: str | Path) -> "DispersionInteractiveViewer":
        """Load lightweight viewer state from a JSON preset file."""
        preset_path = Path(path)
        payload = json.loads(preset_path.read_text())
        self.show_requested = bool(payload.get("show", self.show_requested))
        self.mode_components = payload.get("mode_components")
        self.spectrum_components = payload.get("spectrum_components")
        self.can_reconstruct_modes = bool(
            payload.get("can_reconstruct_modes", self.can_reconstruct_modes)
        )
        self.mode_unavailable_reason = str(
            payload.get("mode_unavailable_reason", self.mode_unavailable_reason)
        )
        options = payload.get("options", {})
        self.options = dict(options) if isinstance(options, dict) else {}
        self.options.setdefault("positive_frequencies", True)
        return self

    def _repr_html_(self) -> str:
        import uuid as _uuid

        from mmpp._repr_helpers import (
            NODE_COLOR_ANALYSIS,
            NODE_COLOR_PLOT,
            NODE_COLOR_UTIL,
            helper_card_html,
        )

        status = "mode-ready" if self.can_reconstruct_modes else "spectrum-only"
        notes = list(getattr(self.result, "notes", None) or [])
        mode_text = (
            "Complex spectrum is available for mode workflows."
            if self.can_reconstruct_modes
            else "For mode reconstruction, call with store_complex=True."
        )
        metrics = [
            ("status", status),
            ("axis", getattr(self.result, "axis", "?")),
            ("component", getattr(self.result, "component", "?")),
            ("widget", self._widget_status),
        ]
        actions = [
            (
                "Display:",
                [
                    (".show()", "Render ipywidgets heatmap", NODE_COLOR_PLOT),
                    (".close()", "Release display resources", NODE_COLOR_UTIL),
                ],
            ),
            (
                "State:",
                [
                    (".state", "Serializable viewer state", NODE_COLOR_ANALYSIS),
                    (".export_selection(...)", "Export current selection", NODE_COLOR_UTIL),
                    (".save_preset(path)", "Persist viewer preset", NODE_COLOR_UTIL),
                ],
            ),
        ]
        notes_html = (
            "<div style='color:#cbd5e1;font-size:0.9em;'>"
            "Lightweight status view. Full widget controls are initialized by "
            "<code>.show()</code>. "
            f"{escape(mode_text)}</div>"
        )
        if notes:
            rows = "".join(f"<li>{escape(str(note))}</li>" for note in notes[:8])
            if len(notes) > 8:
                rows += f"<li>... {len(notes) - 8} more notes</li>"
            notes_html += f"<ul style='margin:6px 0 0 18px;padding:0;'>{rows}</ul>"
        return helper_card_html(
            f"DispersionInteractiveViewer: {status}",
            subtitle="Notebook controller for plotting spin-wave dispersion S(k, f).",
            status=(status, NODE_COLOR_PLOT),
            metrics=metrics,
            details=[("Notes", notes_html)],
            action_groups=actions,
            uid=f"mmpp-dispersion-interactive-{str(_uuid.uuid4())[:8]}",
        )

    def _build_widget(self, display_func: Any) -> Any:
        """Create the notebook widget lazily, falling back to HTML status."""
        if self._widget is not None:
            return self._widget
        try:
            from ._interactive import DispersionHeatmapWidget
        except ImportError as exc:
            self._widget_status = "fallback"
            self._widget_error = f"{type(exc).__name__}: {exc}"
            return None

        self._widget_engine = DispersionHeatmapWidget(self.result, self.options)
        self._widget = self._widget_engine.build(display_func)
        self._figure = self._widget_engine.figure
        self._axes = self._widget_engine.axes
        self._controls = self._widget_engine.controls
        self._widget_status = "ready"
        self._widget_error = ""
        return self._widget

    def _status_html(self) -> str:
        if self._widget_engine is not None:
            return self._widget_engine.status_html()
        return ""

    def _control_value(self, name: str, default: Any) -> Any:
        control = self._controls.get(name)
        return getattr(control, "value", default)

    def _render_heatmap(self) -> None:
        """Render S(k, f) into the managed Matplotlib axes."""
        if self._widget_engine is not None:
            self._widget_engine.render()
