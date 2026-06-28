"""ipywidgets-based heatmap widget for dispersion results."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mmpp.fft.dispersion.models import DispersionResult1D


class DispersionHeatmapWidget:
    """Build and render the notebook heatmap for ``DispersionResult1D``."""

    def __init__(self, result: "DispersionResult1D", options: dict[str, Any]):
        self.result = result
        self.options = dict(options)
        self.widget: Any = None
        self.figure: Any = None
        self.axes: Any = None
        self.controls: dict[str, Any] = {}

    def build(self, display_func: Any) -> Any:
        """Create controls and render the initial ``S(k, f)`` heatmap."""
        import ipywidgets as widgets
        import matplotlib.pyplot as plt

        if self.widget is not None:
            return self.widget

        figsize = tuple(self.options.get("figsize", (7.0, 4.5)))
        dpi = self.options.get("dpi", 100)
        self.figure, self.axes = plt.subplots(figsize=figsize, dpi=dpi)

        self.controls = self._create_controls(widgets)
        for control in self.controls.values():
            if hasattr(control, "observe"):
                control.observe(lambda _change=None: self.render(), names="value")

        plot_output = widgets.Output()
        toolbar = widgets.HBox(
            [
                self.controls["fmax"],
                self.controls["lognorm"],
                self.controls["source"],
                self.controls["positive"],
                self.controls["kscale"],
            ]
        )
        info = widgets.HTML(value=self.status_html())
        self.widget = widgets.VBox([toolbar, plot_output, info])

        self.render()
        try:
            with plot_output:
                display_func(self.figure)
        except Exception:
            display_func(self.figure)

        return self.widget

    def close(self) -> None:
        """Release owned widgets and Matplotlib figure."""
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

    def status_html(self) -> str:
        """Return a compact status area for notes under the widget."""
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
        """Render ``S(k, f)`` into the managed Matplotlib axes."""
        if self.axes is None or self.figure is None:
            return

        import numpy as np

        ax = self.axes
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

        k_plot, k_label = self._scaled_k_axis(k_axis)
        f_plot = f_axis / 1e9
        if k_plot.size == 0 or f_plot.size == 0:
            if hasattr(ax, "text"):
                ax.text(0.5, 0.5, "No dispersion data", ha="center")
            return

        ax.imshow(
            spectrum.T,
            aspect="auto",
            origin="lower",
            extent=(
                float(k_plot[0]),
                float(k_plot[-1]),
                float(f_plot[0]),
                float(f_plot[-1]),
            ),
            cmap=str(self.options.get("cmap", "viridis")),
            norm=self._norm(spectrum),
        )
        ax.set_title(f"Dispersion S(k, f) - {source}")
        ax.set_xlabel(k_label)
        ax.set_ylabel("Frequency [GHz]")
        if hasattr(self.figure, "canvas") and hasattr(self.figure.canvas, "draw_idle"):
            self.figure.canvas.draw_idle()

    def _create_controls(self, widgets: Any) -> dict[str, Any]:
        f_axis = getattr(self.result, "f_axis", None)
        if f_axis is not None and len(f_axis):
            fmax_default = float(self.options.get("fmax") or max(0.0, max(f_axis) / 1e9))
        else:
            fmax_default = float(self.options.get("fmax") or 1.0)

        return {
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
                options=[
                    ("rad/um", "rad_um"),
                    ("rad/m", "rad"),
                    ("1/m", "cycles_m"),
                ],
                value=str(self.options.get("kscale", "rad_um")),
                description="k",
            ),
        }

    def _control_value(self, name: str, default: Any) -> Any:
        control = self.controls.get(name)
        return getattr(control, "value", default)

    def _scaled_k_axis(self, k_axis: Any) -> tuple[Any, str]:
        import numpy as np

        kscale = str(self._control_value("kscale", self.options.get("kscale", "rad_um")))
        if kscale == "rad_um":
            return k_axis / 1e6, "k [rad/um]"
        if kscale in {"cycles_m", "meter"}:
            return k_axis / (2 * np.pi), "k [1/m]"
        return k_axis, "k [rad/m]"

    def _norm(self, spectrum: Any) -> Any:
        if not bool(self._control_value("lognorm", self.options.get("lognorm", False))):
            return None
        try:
            import numpy as np
            from matplotlib.colors import LogNorm

            positive_values = spectrum[spectrum > 0]
            if positive_values.size:
                return LogNorm(
                    vmin=float(np.min(positive_values)),
                    vmax=float(np.max(positive_values)),
                )
        except Exception:
            return None
        return None
