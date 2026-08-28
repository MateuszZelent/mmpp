"""Interactive topology and size dashboard for isolated skyrmions."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np

from ..._field import select_snapshot, valid_magnetization_mask


class SkyrmionInteractiveDashboard:
    """ipywidgets dashboard for one dataset-bound skyrmion workflow."""

    def __init__(
        self,
        interface: Any,
        *,
        initial_frame: int = 0,
        z_layer: int = -1,
        topology_method: str | None = None,
        size_method: str | None = None,
        initial_module: str = "analysis",
        figsize: tuple[float, float] = (13, 4),
        dpi: int = 110,
    ):
        try:
            import ipywidgets as widgets
        except ImportError as exc:
            raise ImportError(
                "Skyrmion interactive mode requires ipywidgets. "
                "Install MMPP with the 'interactive' extra."
            ) from exc

        self._widgets = widgets
        self._interface = interface
        self.figsize = figsize
        self.dpi = int(dpi)
        self._display_handle: Any = None
        self._last_topology: Any = None
        self._last_size: Any = None
        self._last_spectral_viewer: Any = None

        data = np.asarray(interface._resolve_data())
        n_frames = int(data.shape[0]) if data.ndim in {4, 5} else 1
        n_layers = int(data.shape[1]) if data.ndim == 5 else 1
        frame_value = min(max(int(initial_frame), 0), max(n_frames - 1, 0))
        layer_value = int(z_layer)
        layer_value = min(max(layer_value, -n_layers), n_layers - 1)

        self.frame = widgets.IntSlider(
            description="Frame",
            value=frame_value,
            min=0,
            max=max(n_frames - 1, 0),
            continuous_update=False,
        )
        self.z_layer = widgets.IntSlider(
            description="Z layer",
            value=layer_value,
            min=-n_layers,
            max=n_layers - 1,
            continuous_update=False,
        )
        self.topology_method = widgets.Dropdown(
            description="Topology",
            options=["berg_luscher", "finite_diff"],
            value=topology_method or interface.config.topology.method,
        )
        self.size_method = widgets.Dropdown(
            description="Size",
            options=["auto", "threshold", "domain_wall", "ansatz", "gaussian"],
            value=size_method or interface.config.size.method,
        )
        module = str(initial_module).strip().lower()
        if module not in {"analysis", "spectrum", "modes"}:
            raise ValueError(
                "initial_module must be 'analysis', 'spectrum', or 'modes'."
            )
        self.module = widgets.ToggleButtons(
            description="View",
            options=[
                ("Analysis", "analysis"),
                ("Spectrum", "spectrum"),
                ("Modes", "modes"),
            ],
            value=module,
        )
        self.run_button = widgets.Button(
            description="Open selected view",
            icon="play",
            button_style="success",
        )
        self.run_button.on_click(self._on_run)
        self.status = widgets.HTML(
            "<span style='color:#64748b'>Ready — choose frame and methods.</span>"
        )
        self.image = widgets.Image(
            format="png",
            layout=widgets.Layout(width="100%"),
        )
        controls = widgets.VBox(
            [
                widgets.HTML(
                    "<b>Skyrmion topology &amp; size</b><br>"
                    "<small>m<sub>z</sub>, topological density and radial fit</small>"
                ),
                self.frame,
                self.z_layer,
                self.module,
                self.topology_method,
                self.size_method,
                self.run_button,
                self.status,
            ],
            layout=widgets.Layout(width="300px", min_width="300px"),
        )
        self.root = widgets.HBox(
            [controls, self.image],
            layout=widgets.Layout(width="100%", align_items="flex-start"),
        )

    @property
    def last_topology(self):
        """Most recent topology result."""
        return self._last_topology

    @property
    def last_size(self):
        """Most recent size result."""
        return self._last_size

    def show(self, *, run: bool = True):
        """Display the dashboard and optionally render the initial analysis."""
        try:
            from IPython.display import display
        except ImportError as exc:
            raise ImportError("Skyrmion interactive mode requires IPython.") from exc

        if self._display_handle is None:
            self._display_handle = display(self.root, display_id=True)
        else:
            self._display_handle.update(self.root)
        if run:
            self.run_selected()
        return self

    def _on_run(self, _button: Any) -> None:
        self.run_selected()

    @property
    def last_spectral_viewer(self):
        """Most recently opened FFT spectrum/mode viewer."""
        return self._last_spectral_viewer

    def run_selected(self):
        """Run the analysis or open the selected spectral view."""
        selected = str(self.module.value)
        if selected == "analysis":
            return self.run()
        if selected == "spectrum":
            return self.open_spectrum()
        return self.open_modes()

    def open_spectrum(self):
        """Open the interactive FFT spectrum with spatial mode inspection."""
        self.status.value = "<span style='color:#38bdf8'>Opening spectrum…</span>"
        try:
            viewer = self._interface.interactive_spectrum(
                z_layer=int(self.z_layer.value), dpi=self.dpi
            )
            self._last_spectral_viewer = viewer
            self.status.value = (
                "<span style='color:#34d399'>Spectrum and modes opened</span>"
            )
            return viewer
        except Exception as exc:
            self.status.value = f"<span style='color:#f87171'>Failed: {exc}</span>"
            raise

    def open_modes(self):
        """Open the spatial FFT modes together with the source spectrum."""
        self.status.value = "<span style='color:#38bdf8'>Opening modes…</span>"
        try:
            viewer = self._interface.interactive_modes(
                z_layer=int(self.z_layer.value), dpi=self.dpi
            )
            self._last_spectral_viewer = viewer
            self.status.value = (
                "<span style='color:#34d399'>Spatial modes opened</span>"
            )
            return viewer
        except Exception as exc:
            self.status.value = f"<span style='color:#f87171'>Failed: {exc}</span>"
            raise

    def run(self):
        """Compute and render the currently selected snapshot."""
        import matplotlib.pyplot as plt

        self.run_button.disabled = True
        self.status.value = "<span style='color:#38bdf8'>Analyzing…</span>"
        try:
            frame = int(self.frame.value)
            z_layer = int(self.z_layer.value)
            topology = self._interface.detect(
                frame=frame,
                z_layer=z_layer,
                method=str(self.topology_method.value),
            )
            size = self._interface.fit_size(
                frame=frame,
                z_layer=z_layer,
                method=str(self.size_method.value),
                topology=topology,
            )
            self._last_topology = topology
            self._last_size = size

            snapshot = select_snapshot(
                self._interface._resolve_data(), frame=frame, z_layer=z_layer
            )
            valid = valid_magnetization_mask(snapshot)
            mz = np.ma.masked_where(~valid, np.asarray(snapshot[..., 2], dtype=float))
            dx, dy = self._interface._resolve_spacing()

            fig, axes = plt.subplots(1, 3, figsize=self.figsize, dpi=self.dpi)
            image = axes[0].imshow(mz, origin="upper", cmap="RdBu_r", vmin=-1, vmax=1)
            center_x, center_y = topology.center_xy_m
            center_col = center_x / dx
            center_row = center_y / dy
            if str(topology.convention).lower() == "up":
                center_row = snapshot.shape[0] - 1 - center_row
            axes[0].scatter([center_col], [center_row], marker="+", s=90, c="black")
            axes[0].set_title("m$_z$ snapshot")
            fig.colorbar(image, ax=axes[0], shrink=0.78)

            q_cell = np.asarray(topology.q_density, dtype=float) * float(dx) * float(dy)
            q_cell = np.ma.masked_where(~valid, q_cell)
            q_limit = float(np.nanmax(np.abs(q_cell))) if np.ma.count(q_cell) else 1.0
            q_limit = max(q_limit, 1e-15)
            q_image = axes[1].imshow(
                q_cell,
                origin="upper",
                cmap="coolwarm",
                vmin=-q_limit,
                vmax=q_limit,
            )
            axes[1].set_title(f"Topological density · Q={topology.Q:.4g}")
            fig.colorbar(q_image, ax=axes[1], shrink=0.78, label="charge / cell")

            radius_nm = np.asarray(size.radial_r_m, dtype=float) * 1e9
            axes[2].plot(radius_nm, size.radial_mz, ".", ms=3, label="radial m$_z$")
            if np.asarray(size.model_mz).size == radius_nm.size:
                axes[2].plot(radius_nm, size.model_mz, lw=2, label=size.model)
            if size.radius_nm is not None:
                axes[2].axvline(size.radius_nm, ls="--", color="tab:red", label="R50")
            axes[2].set_xlabel("r [nm]")
            axes[2].set_ylabel("m$_z$")
            axes[2].set_title(f"Size · R={size.radius_nm!s} nm · {size.quality}")
            axes[2].grid(alpha=0.25)
            axes[2].legend(fontsize=8)

            fig.tight_layout()
            buffer = BytesIO()
            fig.savefig(buffer, format="png", bbox_inches="tight")
            plt.close(fig)
            self.image.value = buffer.getvalue()
            flags = ", ".join((*topology.flags, *size.flags)) or "none"
            self.status.value = (
                "<span style='color:#34d399'>Done</span> · "
                f"Q={topology.Q:.5g} · model={size.model} · flags={flags}"
            )
            return topology, size
        except Exception as exc:
            self.status.value = f"<span style='color:#f87171'>Failed: {exc}</span>"
            raise
        finally:
            self.run_button.disabled = False


__all__ = ["SkyrmionInteractiveDashboard"]
