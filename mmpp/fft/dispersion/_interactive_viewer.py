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


_ANALYTICAL_OPTION_KEYS = {
    "model",
    "sw_config",
    "n_modes",
    "B",
    "Ms",
    "Aex",
    "d",
    "Ku",
    "Kc1",
    "Kc2",
    "phi",
    "phi_ani",
    "D",
    "g",
    "k_points",
    "color",
    "linestyle",
    "linewidth",
    "alpha",
}
_ANALYTICAL_STYLE_KEYS = {"color", "linestyle", "linewidth", "alpha"}
INTERACTIVE_VIEWER_KEYS = {
    "show",
    "toolbar",
    "figsize",
    "dpi",
    "kscale",
    "f_units",
    "fmin",
    "fmax",
    "initial_render",
    "lognorm",
    "source",
    "cmap",
    "components",
    "mode_components",
    "spectrum_components",
    "modes",
    "animate",
    "auto_animate",
    "lattice_constant_nm",
    "use_holography",
    "z_layer",
    "mode_type",
    "n_bz",
    "positive_frequencies",
    "analitical",
    "analytical",
    *_ANALYTICAL_OPTION_KEYS,
}


def split_dispersion_interactive_kwargs(
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split compute-then-viewer kwargs using the canonical viewer option map."""
    viewer_kwargs: dict[str, Any] = {}
    compute_kwargs: dict[str, Any] = {}
    for key, value in dict(kwargs).items():
        if key in INTERACTIVE_VIEWER_KEYS:
            viewer_kwargs[key] = value
        else:
            compute_kwargs[key] = value
    return compute_kwargs, viewer_kwargs


def _normalize_analytical_options(
    analytical: Any = None,
    analitical: Any = None,
    options: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Collect analytical-overlay options into one serializable state block."""
    overlay: dict[str, Any] = {}
    requested = analytical if analytical is not None else analitical

    if isinstance(requested, dict):
        overlay.update(requested)
        overlay.setdefault("enabled", True)
    elif requested is not None:
        overlay["enabled"] = bool(requested)

    if options is not None:
        non_style_keys = _ANALYTICAL_OPTION_KEYS - _ANALYTICAL_STYLE_KEYS
        has_explicit_overlay_option = any(key in options for key in non_style_keys)
        if requested is None and not has_explicit_overlay_option:
            return overlay
        for key in list(options):
            if key in _ANALYTICAL_OPTION_KEYS:
                overlay[key] = options.pop(key)

    if overlay and "enabled" not in overlay:
        overlay["enabled"] = True
    return overlay


def _normalize_interactive_options(
    *,
    components: Optional[list[str]] = None,
    mode_components: Optional[list[str]] = None,
    spectrum_components: Optional[list[str]] = None,
    animate: Optional[bool] = None,
    auto_animate: Optional[bool] = None,
    modes: Any = "auto",
    analytical: Any = None,
    analitical: Any = None,
    **kwargs: Any,
) -> tuple[Optional[list[str]], Optional[list[str]], Any, dict[str, Any], dict[str, Any]]:
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

    analytical_options = _normalize_analytical_options(
        analytical=analytical,
        analitical=analitical,
        options=options,
    )
    if analytical is not None:
        options["analytical"] = analytical
    if analitical is not None:
        options["analitical"] = analitical

    return mode_components, spectrum_components, modes, options, analytical_options


def normalize_dispersion_interactive_options(
    **kwargs: Any,
) -> tuple[Optional[list[str]], Optional[list[str]], Any, dict[str, Any], dict[str, Any]]:
    """Public wrapper for shared dispersion interactive option normalization."""
    return _normalize_interactive_options(**kwargs)



@dataclass
class DispersionInteractiveViewer:
    """Stable controller returned by dispersion interactive APIs."""

    result: "DispersionResult1D"
    show_requested: bool = True
    modes: Any = "auto"
    mode_components: Optional[list[str]] = None
    spectrum_components: Optional[list[str]] = None
    can_reconstruct_modes: bool = False
    mode_unavailable_reason: str = ""
    analytical: dict[str, Any] = field(default_factory=dict)
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
        modes: Any = "auto",
        analytical: Any = None,
        analitical: Any = None,
        **kwargs: Any,
    ) -> "DispersionInteractiveViewer":
        (
            mode_components,
            spectrum_components,
            modes,
            options,
            analytical_options,
        ) = _normalize_interactive_options(
            components=components,
            mode_components=mode_components,
            spectrum_components=spectrum_components,
            animate=animate,
            auto_animate=auto_animate,
            modes=modes,
            analytical=analytical,
            analitical=analitical,
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
            modes=modes,
            mode_components=mode_components,
            spectrum_components=spectrum_components,
            can_reconstruct_modes=bool(can_reconstruct_modes),
            mode_unavailable_reason=mode_unavailable_reason,
            analytical=analytical_options,
            options=options,
        )
        if show:
            viewer.show()
        return viewer

    def show(self, *, toolbar: bool | str = "auto") -> "DispersionInteractiveViewer":
        """Display an interactive dispersion heatmap when notebook deps exist."""
        self.show_requested = True
        self.options["toolbar"] = toolbar
        try:
            from IPython.display import display
        except ImportError:
            return self

        try:
            initial_render = bool(self.options.get("initial_render", False))
            widget = self._build_widget(
                display,
                defer_initial_render=not initial_render,
            )
        except Exception as exc:
            self._widget_status = "fallback"
            self._widget_error = f"{type(exc).__name__}: {exc}"
            self._close_widget_state()
            widget = None
        self._display_handle = display(widget if widget is not None else self, display_id=True)
        if widget is not None and bool(self.options.get("initial_render", False)):
            self._render_widget_after_display()
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
            "modes": json_safe(self.modes),
            "mode_components": self.mode_components,
            "spectrum_components": self.spectrum_components,
            "can_reconstruct_modes": self.can_reconstruct_modes,
            "mode_unavailable_reason": self.mode_unavailable_reason,
            "analytical": json_safe(self.analytical),
            "result_notes": result_notes,
            "widget_status": self._widget_status,
            "widget_error": self._widget_error,
            "diagnostics": self.diagnostics(),
            "options": json_safe(self.options),
        }

    def diagnostics(self) -> dict[str, Any]:
        """Return runtime diagnostics for notebook/backend troubleshooting."""
        if self._widget_engine is not None and hasattr(self._widget_engine, "diagnostics"):
            return json_safe(self._widget_engine.diagnostics())
        return {
            "toolbar_enabled": False,
            "click_connected": False,
            "backend": "not-shown",
            "interactive_backend": False,
        }

    def export_selection(self, **selection: Any) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of viewer state and selection."""
        selection_payload = (
            self._widget_selection_payload() if not selection else dict(selection)
        )
        normalized_selection = self._normalized_selection(selection_payload)
        return {
            "viewer": self.state,
            "selection": json_safe(selection_payload),
            "mode_request": json_safe(self._mode_request(normalized_selection)),
        }

    def mode_at_selection(self, **selection: Any) -> Any:
        """Extract the dispersion mode at the selected ``(k, f)`` point."""
        selection_payload = self._normalized_selection(selection)
        mode_request = self._mode_request(selection_payload)
        if not mode_request.get("available", False):
            raise ValueError(str(mode_request.get("reason") or "Mode selection unavailable."))
        return self.result.modes.at(
            k_rad_um=float(mode_request["k_rad_um"]),
            f_ghz=float(mode_request["f_ghz"]),
            z_layer=int(mode_request.get("z_layer", 0)),
            component=mode_request.get("component"),
        )

    def _normalized_selection(self, selection: dict[str, Any]) -> dict[str, Any]:
        """Merge explicit selection kwargs with the live widget selection."""
        payload = dict(selection or self._widget_selection_payload())
        if "k_rad_um" not in payload and "k_rad_per_m" in payload:
            payload["k_rad_um"] = float(payload["k_rad_per_m"]) / 1e6
        if "k_rad_per_m" not in payload and "k_rad_um" in payload:
            payload["k_rad_per_m"] = float(payload["k_rad_um"]) * 1e6
        if "f_ghz" not in payload and "f_hz" in payload:
            payload["f_ghz"] = float(payload["f_hz"]) / 1e9
        if "f_hz" not in payload and "f_ghz" in payload:
            payload["f_hz"] = float(payload["f_ghz"]) * 1e9
        return payload

    def _widget_selection_payload(self) -> dict[str, Any]:
        """Return the live widget selection in both renderer and mode units."""
        payload: dict[str, Any] = {}
        if self._widget_engine is None:
            return payload
        state = getattr(self._widget_engine, "state", None)
        selected_k = getattr(state, "selected_k", None)
        selected_f = getattr(state, "selected_f", None)
        selected_power = getattr(state, "selected_power", None)
        if selected_k is not None or selected_f is not None:
            payload["source"] = "widget"
        if selected_k is not None:
            k_rad_per_m = float(selected_k)
            payload["k_rad_per_m"] = k_rad_per_m
            payload["k_rad_um"] = k_rad_per_m / 1e6
        if selected_f is not None:
            f_hz = float(selected_f)
            payload["f_hz"] = f_hz
            payload["f_ghz"] = f_hz / 1e9
        if selected_power is not None:
            payload["power"] = float(selected_power)
        return payload

    def _mode_request(self, selection: dict[str, Any]) -> dict[str, Any]:
        """Build a serializable request for mode reconstruction."""
        k_rad_um = selection.get("k_rad_um")
        f_ghz = selection.get("f_ghz")
        component = selection.get("component", getattr(self.result, "component", None))
        request = {
            "available": False,
            "k_rad_um": None,
            "f_ghz": None,
            "z_layer": int(selection.get("z_layer", 0)),
            "component": component,
            "reason": "",
        }
        if k_rad_um is None or f_ghz is None:
            request["reason"] = "Select a dispersion point with k and f first."
            return request
        request["k_rad_um"] = float(k_rad_um)
        request["f_ghz"] = float(f_ghz)
        if not self.can_reconstruct_modes:
            request["reason"] = self.mode_unavailable_reason or (
                "Mode reconstruction requires S_complex."
            )
            return request
        if getattr(self.result, "S_complex", None) is None:
            request["reason"] = "Mode reconstruction requires S_complex."
            return request
        request["available"] = True
        return request

    def save_preset(self, path: str | Path) -> Path:
        """Persist lightweight viewer state to a JSON preset file."""
        preset_path = Path(path)
        preset_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.state
        if self._widget_engine is not None:
            payload = dict(payload)
            payload["explorer"] = json_safe(self._widget_engine.collect_preset())
        preset_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return preset_path

    def load_preset(self, path: str | Path) -> "DispersionInteractiveViewer":
        """Load lightweight viewer state from a JSON preset file."""
        preset_path = Path(path)
        payload = json.loads(preset_path.read_text())
        self.show_requested = bool(payload.get("show", self.show_requested))
        self.modes = payload.get("modes", self.modes)
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
        analytical = payload.get("analytical", {})
        self.analytical = dict(analytical) if isinstance(analytical, dict) else {}
        explorer_payload = payload.get("explorer")
        if isinstance(explorer_payload, dict) and self._widget_engine is not None:
            self._widget_engine.apply_preset(explorer_payload)
        return self

    def _repr_html_(self) -> str:
        import uuid as _uuid

        from mmpp._repr_helpers import (
            NODE_COLOR_ANALYSIS,
            NODE_COLOR_PLOT,
            NODE_COLOR_UTIL,
            accessors_section_html,
            api_help_html,
            metrics_section_html,
            node_card_html,
        )

        status = "mode-ready" if self.can_reconstruct_modes else "spectrum-only"
        notes = list(getattr(self.result, "notes", None) or [])
        mode_text = (
            "Complex spectrum is available for mode workflows."
            if self.can_reconstruct_modes
            else "For mode reconstruction, call with store_complex=True."
        )
        metrics = metrics_section_html(
            [
                ("status", status, NODE_COLOR_PLOT),
                ("axis", getattr(self.result, "axis", "?"), NODE_COLOR_ANALYSIS),
                ("component", getattr(self.result, "component", "?"), NODE_COLOR_ANALYSIS),
                ("widget", self._widget_status, NODE_COLOR_UTIL),
            ]
        )
        actions = accessors_section_html(
            (
                (
                    "Display:",
                    [
                        (".show(toolbar='auto')", NODE_COLOR_PLOT),
                        (".close()", NODE_COLOR_UTIL),
                        (".diagnostics()", NODE_COLOR_UTIL),
                    ],
                ),
                (
                    "State:",
                    [
                        (".state", NODE_COLOR_ANALYSIS),
                        (".export_selection(...)", NODE_COLOR_UTIL),
                        (".save_preset(path)", NODE_COLOR_UTIL),
                    ],
                ),
            )
        )
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
        api = api_help_html(
            self,
            title="Dispersion interactive viewer API help",
            prefix="viewer",
            methods=["show", "close", "diagnostics", "export_selection", "save_preset", "load_preset"],
            subtitle="Live controller returned by dispersion.plot.interactive().",
            chrome=False,
        )
        return node_card_html(
            f"DispersionInteractiveViewer: {status}",
            icon="📈",
            subtitle="Notebook controller for plotting spin-wave dispersion S(k, f).",
            badge=(status, NODE_COLOR_PLOT),
            sections=[metrics, actions, f"<div>{notes_html}</div>"],
            api=api,
            uid=f"mmpp-dispersion-interactive-{str(_uuid.uuid4())[:8]}",
        )

    def _build_widget(
        self,
        display_func: Any,
        *,
        defer_initial_render: bool = False,
    ) -> Any:
        """Create the notebook widget lazily, falling back to HTML status."""
        if self._widget is not None:
            return self._widget
        try:
            from ._interactive import DispersionHeatmapWidget
        except ImportError as exc:
            self._widget_status = "fallback"
            self._widget_error = f"{type(exc).__name__}: {exc}"
            return None

        widget_options = self._widget_options()
        toolbar = widget_options.get("toolbar", "auto")
        self._widget_engine = DispersionHeatmapWidget(self.result, widget_options)
        self._widget = self._widget_engine.build(
            display_func,
            toolbar=toolbar,
            defer_initial_render=defer_initial_render,
        )
        self._figure = self._widget_engine.figure
        self._axes = self._widget_engine.axes
        self._controls = self._widget_engine.controls
        self._widget_status = "ready"
        self._widget_error = ""
        return self._widget

    def _render_widget_after_display(self) -> None:
        """Render the initial heatmap after the widget shell has been displayed."""
        if self._widget_engine is None:
            return
        try:
            from ._interactive.status import set_status

            set_status(
                self._widget_engine,
                "Rendering initial dispersion heatmap...",
                color="#334155",
            )
            self._widget_engine.render()
            self._widget_engine._warn_for_inline_backend()
            if self._widget_engine.diagnostics().get("interactive_backend", False):
                set_status(
                    self._widget_engine,
                    "Interactive dispersion ready",
                    color="#0F766E",
                )
        except Exception as exc:
            self._widget_status = "render-error"
            self._widget_error = f"{type(exc).__name__}: {exc}"
            try:
                from ._interactive.status import set_status

                set_status(
                    self._widget_engine,
                    f"Initial dispersion render failed: {self._widget_error}",
                    color="crimson",
                )
            except Exception:
                pass

    def _widget_options(self) -> dict[str, Any]:
        """Translate stable viewer state into options consumed by the widget engine."""
        options = dict(self.options)
        analytical = dict(self.analytical or {})
        if not analytical:
            return options

        enabled = bool(analytical.get("enabled", True))
        options["analytical"] = analytical.get("sw_config", True) if enabled else False

        mapped_keys = {
            "model": "analytical_model",
            "n_modes": "analytical_n_modes",
            "k_points": "analytical_k_points",
            "phi": "analytical_phi",
            "D": "analytical_D",
        }
        for source_key, target_key in mapped_keys.items():
            if source_key in analytical:
                options[target_key] = analytical[source_key]

        for key in ("B", "Ms", "Aex", "d", "Ku", "Kc1", "Kc2", "phi_ani", "g"):
            if key in analytical:
                options[key] = analytical[key]

        style = {
            key: analytical[key]
            for key in ("color", "linestyle", "linewidth", "alpha")
            if key in analytical
        }
        if style:
            options["analytical_style"] = style
        return options

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
