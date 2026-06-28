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

        widget = self._build_widget(display)
        self._display_handle = display(widget if widget is not None else self, display_id=True)
        return self

    def close(self) -> None:
        """Best-effort close/update hook for notebook integrations."""
        if self._display_handle is not None and hasattr(self._display_handle, "update"):
            self._display_handle.update(None)
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
        self._display_handle = None
        self._widget = None
        self._figure = None
        self._axes = None
        self._controls = {}
        self._widget_status = "closed"
        self.show_requested = False

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
        status = "mode-ready" if self.can_reconstruct_modes else "spectrum-only"
        notes = list(getattr(self.result, "notes", None) or [])
        help_text = (
            "Lightweight status view. Full widget controls are not initialized here. "
            "For mode reconstruction, call with store_complex=True."
            if not self.can_reconstruct_modes
            else "Lightweight status view. Complex spectrum is available for mode workflows."
        )
        notes_html = ""
        if notes:
            rows = "".join(f"<li>{escape(str(note))}</li>" for note in notes[:8])
            if len(notes) > 8:
                rows += f"<li>... {len(notes) - 8} more notes</li>"
            notes_html = f"<ul style='margin:6px 0 0 18px;padding:0;'>{rows}</ul>"
        return (
            "<div style='font-family:monospace;padding:8px;border:1px solid #334155;"
            "border-radius:6px;background:#0f172a;color:#e2e8f0;'>"
            f"DispersionInteractiveViewer: {status}"
            f"<div style='margin-top:6px;color:#cbd5e1;'>{escape(help_text)}</div>"
            f"{notes_html}"
            "</div>"
        )

    def _build_widget(self, display_func: Any) -> Any:
        """Create the notebook widget lazily, falling back to HTML status."""
        if self._widget is not None:
            return self._widget
        try:
            import ipywidgets as widgets
            import matplotlib.pyplot as plt
        except ImportError as exc:
            self._widget_status = "fallback"
            self._widget_error = f"{type(exc).__name__}: {exc}"
            return None

        figsize = tuple(self.options.get("figsize", (7.0, 4.5)))
        dpi = self.options.get("dpi", 100)
        self._figure, self._axes = plt.subplots(figsize=figsize, dpi=dpi)

        f_axis = getattr(self.result, "f_axis", None)
        if f_axis is not None and len(f_axis):
            fmax_default = float(self.options.get("fmax") or max(0.0, max(f_axis) / 1e9))
            fmax_max = max(float(max(f_axis) / 1e9), fmax_default, 0.1)
        else:
            fmax_default = float(self.options.get("fmax") or 1.0)
            fmax_max = max(fmax_default, 0.1)

        controls = {
            "fmax": widgets.FloatText(
                value=fmax_default,
                description="fmax GHz",
            ),
            "lognorm": widgets.Checkbox(
                value=bool(self.options.get("lognorm", False)),
                description="log",
            ),
            "source": widgets.Dropdown(
                options=["display", "raw"],
                value=str(self.options.get("source", "display")),
                description="source",
            ),
            "positive": widgets.Checkbox(
                value=bool(self.options.get("positive_frequencies", True)),
                description="f >= 0",
            ),
            "kscale": widgets.Dropdown(
                options=["rad_um", "rad", "meter"],
                value=str(self.options.get("kscale", "rad_um")),
                description="k",
            ),
        }
        self._controls = controls

        def _on_change(_change: Any = None) -> None:
            self._render_heatmap()

        for control in controls.values():
            if hasattr(control, "observe"):
                control.observe(_on_change, names="value")

        plot_output = widgets.Output()
        info = widgets.HTML(value=self._status_html())
        toolbar = widgets.HBox(
            [
                controls["fmax"],
                controls["lognorm"],
                controls["source"],
                controls["positive"],
                controls["kscale"],
            ]
        )
        self._widget = widgets.VBox([toolbar, plot_output, info])
        self._render_heatmap()

        try:
            with plot_output:
                display_func(self._figure)
        except Exception:
            display_func(self._figure)

        self._widget_status = "ready"
        self._widget_error = ""
        return self._widget

    def _status_html(self) -> str:
        notes = list(getattr(self.result, "notes", None) or [])
        rows = "".join(f"<li>{escape(str(note))}</li>" for note in notes[:6])
        return (
            "<div style='font-family:monospace;color:#cbd5e1;'>"
            "<b>Dispersion interactive</b>"
            f"<ul style='margin:4px 0 0 16px;padding:0;'>{rows}</ul>"
            "</div>"
        )

    def _control_value(self, name: str, default: Any) -> Any:
        control = self._controls.get(name)
        return getattr(control, "value", default)

    def _render_heatmap(self) -> None:
        """Render S(k, f) into the managed Matplotlib axes."""
        if self._axes is None or self._figure is None:
            return

        import numpy as np

        ax = self._axes
        if hasattr(ax, "clear"):
            ax.clear()

        source = str(self._control_value("source", self.options.get("source", "display")))
        positive = bool(
            self._control_value(
                "positive",
                self.options.get("positive_frequencies", True),
            )
        )
        spectrum, k_axis, f_axis = self.result.frequency_view(
            positive_frequencies=positive,
            analysis_source=source,
        )
        spectrum = np.asarray(spectrum, dtype=float)
        k_axis = np.asarray(k_axis, dtype=float)
        f_axis = np.asarray(f_axis, dtype=float)

        fmax = self._control_value("fmax", self.options.get("fmax"))
        if fmax:
            mask = f_axis <= float(fmax) * 1e9
            if np.any(mask):
                spectrum = spectrum[:, mask]
                f_axis = f_axis[mask]

        kscale = str(self._control_value("kscale", self.options.get("kscale", "rad_um")))
        if kscale == "rad_um":
            k_plot = k_axis / 1e6
            k_label = "k [rad/um]"
        elif kscale == "meter":
            k_plot = k_axis / (2 * np.pi)
            k_label = "k [1/m]"
        else:
            k_plot = k_axis
            k_label = "k [rad/m]"
        f_plot = f_axis / 1e9

        norm = None
        if bool(self._control_value("lognorm", self.options.get("lognorm", False))):
            try:
                from matplotlib.colors import LogNorm

                positive_values = spectrum[spectrum > 0]
                if positive_values.size:
                    norm = LogNorm(
                        vmin=float(np.min(positive_values)),
                        vmax=float(np.max(positive_values)),
                    )
            except Exception:
                norm = None

        extent = (
            float(k_plot[0]),
            float(k_plot[-1]),
            float(f_plot[0]),
            float(f_plot[-1]),
        )
        ax.imshow(
            spectrum.T,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=str(self.options.get("cmap", "viridis")),
            norm=norm,
        )
        ax.set_title(f"Dispersion S(k, f) - {source}")
        ax.set_xlabel(k_label)
        ax.set_ylabel("Frequency [GHz]")
        if hasattr(self._figure, "canvas") and hasattr(self._figure.canvas, "draw_idle"):
            self._figure.canvas.draw_idle()
