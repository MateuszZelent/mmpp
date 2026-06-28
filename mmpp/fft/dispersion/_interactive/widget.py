"""Interactive dispersion explorer engine."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Any

from .callbacks import on_canvas_click, sync_analytical_options
from .presets import apply_preset_state, collect_preset_state
from .rendering import draw_dispersion_panel, refresh_output_widget
from .state import DispersionExplorerState
from .status import set_status
from .widgets import build_toolbar

if TYPE_CHECKING:
    from mmpp.fft.dispersion.models import DispersionResult1D


class DispersionHeatmapWidget:
    """Notebook explorer for ``DispersionResult1D`` using the shared toolbar pattern."""

    def __init__(self, result: "DispersionResult1D", options: dict[str, Any]):
        self.result = result
        self.options = dict(options)
        self.state = self._initial_state()
        self.widget: Any = None
        self.figure: Any = None
        self.axes: Any = None
        self.controls: dict[str, Any] = {}
        self._status_history: list[str] = []
        self._presets_dir = None
        self._click_connection = None
        self._image = None

    def build(self, display_func: Any, *, toolbar: bool | str = "auto") -> Any:
        """Create figure, controls, callbacks, and initial heatmap."""
        import matplotlib.pyplot as plt

        toolbar_enabled = self._resolve_toolbar(toolbar)
        figsize = tuple(self.options.get("figsize", (8.0, 5.2)))
        dpi = self.options.get("dpi", 100)
        if hasattr(plt, "ioff"):
            plt.ioff()
        self.figure, self.axes = plt.subplots(figsize=figsize, dpi=dpi)
        if hasattr(self.figure, "canvas") and hasattr(self.figure.canvas, "mpl_connect"):
            self._click_connection = self.figure.canvas.mpl_connect(
                "button_press_event",
                lambda event: on_canvas_click(self, event),
            )

        if toolbar_enabled:
            import ipywidgets as widgets

            build_toolbar(self, widgets)
            return self.widget

        draw_dispersion_panel(self)
        display_func(self.figure)
        set_status(self, "Matplotlib dispersion figure ready", color="#0F766E")
        return self.figure

    def close(self) -> None:
        """Release owned widgets and Matplotlib figure."""
        if self.figure is not None and self._click_connection is not None:
            try:
                self.figure.canvas.mpl_disconnect(self._click_connection)
            except Exception:
                pass
        for control in list(self.controls.values()):
            if hasattr(control, "close"):
                try:
                    control.close()
                except Exception:
                    pass
        if self.widget is not None and hasattr(self.widget, "close"):
            try:
                self.widget.close()
            except Exception:
                pass
        if self.figure is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self.figure)
            except Exception:
                pass
        self.widget = None
        self.figure = None
        self.axes = None
        self.controls = {}
        self._click_connection = None

    def diagnostics(self) -> dict[str, Any]:
        """Return runtime diagnostics for notebook/backend troubleshooting."""
        backend = "unknown"
        interactive_backend = False
        try:
            import matplotlib

            backend = str(matplotlib.get_backend())
            interactive_backend = any(
                kw in backend.lower() for kw in ("widget", "ipympl", "nbagg", "notebook")
            )
        except Exception:
            pass
        return {
            "backend": backend,
            "interactive_backend": interactive_backend,
            "click_connected": bool(self._click_connection is not None),
            "toolbar_enabled": bool(self.widget is not None and self.controls),
            "selected": {
                "k_rad_per_m": self.state.selected_k,
                "f_hz": self.state.selected_f,
                "power": self.state.selected_power,
            },
        }

    def apply_state_to_controls(self) -> None:
        """Synchronize state into toolbar controls."""
        if not self.controls:
            return
        mapping = {
            "fmin": self.state.fmin_ghz,
            "fmax": self.state.fmax_ghz,
            "source": self.state.source,
            "kscale": self.state.kscale,
            "cmap": self.state.cmap,
            "positive": self.state.positive_frequencies,
            "lognorm": self.state.lognorm,
        }
        for key, value in mapping.items():
            if key in self.controls:
                self.controls[key].value = value
        for key, value in (self.state.show_flags or {}).items():
            if key in self.controls:
                self.controls[key].value = bool(value)
        analytical = self.state.analytical or {}
        analytical_mapping = {
            "analytical_enabled": bool(analytical.get("enabled", False)),
            "analytical_model": analytical.get("model", "kalinikos"),
            "analytical_sw_config": analytical.get("sw_config", "DE"),
            "analytical_n_modes": analytical.get("n_modes", 1),
            "analytical_k_points": analytical.get("k_points", 500),
        }
        for key, value in analytical_mapping.items():
            if key in self.controls:
                self.controls[key].value = value
        sync_analytical_options(self)

    def collect_preset(self) -> dict[str, Any]:
        """Return serializable preset state."""
        return collect_preset_state(self)

    def apply_preset(self, payload: dict[str, Any]) -> None:
        """Apply serializable preset state and redraw."""
        apply_preset_state(self, payload)
        self.apply_state_to_controls()
        draw_dispersion_panel(self)
        refresh_output_widget(self)

    def status_html(self) -> str:
        """Return compact note block for fallback/status displays."""
        notes = list(getattr(self.result, "notes", None) or [])
        rows = "".join(f"<li>{escape(str(note))}</li>" for note in notes[:6])
        notes_html = (
            f"<ul style='margin:4px 0 0 16px;padding:0;'>{rows}</ul>"
            if rows
            else ""
        )
        return (
            "<div style='font-family:monospace;color:#cbd5e1;'>"
            "<b>Dispersion interactive</b>"
            f"{notes_html}"
            "</div>"
        )

    def render(self) -> None:
        """Redraw the heatmap and output widget."""
        draw_dispersion_panel(self)
        refresh_output_widget(self)

    def _resolve_toolbar(self, toolbar: bool | str) -> bool:
        if isinstance(toolbar, str):
            if toolbar.lower() == "auto":
                return True
            return toolbar.lower() in {"1", "true", "yes", "on"}
        return bool(toolbar)

    def _initial_state(self) -> DispersionExplorerState:
        f_axis = getattr(self.result, "f_axis", None)
        if f_axis is not None and len(f_axis):
            positives = [float(v) / 1e9 for v in f_axis if float(v) >= 0.0]
            fmin = min(positives) if positives else 0.0
            fmax = max(positives) if positives else 1.0
        else:
            fmin, fmax = 0.0, 1.0
        return DispersionExplorerState(
            fmin_ghz=float(self.options.get("fmin", fmin)),
            fmax_ghz=float(self.options.get("fmax") or fmax or 1.0),
            source=str(self.options.get("source", "display")),
            kscale=str(self.options.get("kscale", "rad_um")),
            cmap=str(self.options.get("cmap", "viridis")),
            positive_frequencies=bool(self.options.get("positive_frequencies", True)),
            lognorm=bool(self.options.get("lognorm", False)),
            analytical=self._initial_analytical_state(),
        )

    def _initial_analytical_state(self) -> dict[str, Any]:
        raw = self.options.get("analytical", False)
        raw_options = dict(raw) if isinstance(raw, dict) else {}
        enabled = bool(raw)
        sw_config = raw_options.get("sw_config") or self.options.get("analytical_sw_config")
        if sw_config is None and isinstance(raw, str):
            sw_config = raw
        if sw_config is None:
            sw_config = "DE"
        return {
            "enabled": enabled,
            "model": str(
                raw_options.get("model")
                or self.options.get("analytical_model")
                or "kalinikos"
            ),
            "sw_config": str(sw_config),
            "n_modes": int(
                raw_options.get("n_modes")
                or self.options.get("analytical_n_modes")
                or 1
            ),
            "k_points": int(
                raw_options.get("k_points")
                or self.options.get("analytical_k_points")
                or 500
            ),
        }
