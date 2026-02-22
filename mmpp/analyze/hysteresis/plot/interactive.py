"""Interactive hysteresis loop + snapshot explorer."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    _HAS_MPL = True
except Exception:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]
    GridSpec = Any  # type: ignore[assignment]
    _HAS_MPL = False

try:
    import ipywidgets as widgets
    from IPython import get_ipython
    from IPython.display import clear_output, display

    _HAS_WIDGETS = True
except Exception:  # pragma: no cover - optional dependency
    widgets = None  # type: ignore[assignment]
    clear_output = display = get_ipython = None  # type: ignore[assignment]
    _HAS_WIDGETS = False

from ._interactive import (
    HysteresisExplorerState,
    SnapshotCache,
    build_toolbar,
    draw_loop_panel,
    on_loop_click,
    render_snapshot,
    set_status,
    stop_animation,
    update_loop_cursor,
)


class HysteresisInteractiveExplorer:
    """Two-panel interactive viewer for hysteresis loops and snapshots."""

    def __init__(self, result):
        if not _HAS_MPL:
            raise ImportError("Matplotlib is required for interactive plotting")

        self.result = result
        self.state = HysteresisExplorerState(
            show_flags={
                "hc": bool(result.config.show_hc),
                "mr": bool(result.config.show_mr),
                "ms": bool(result.config.show_ms),
                "arrow": bool(result.config.show_arrow),
                "branch_colors": bool(result.config.show_branch_colors),
                "trail": True,
            }
        )

        self._fig = None
        self._ax_loop = None
        self._ax_snapshot = None
        self._loop_marker = None
        self._loop_trail = None
        self._loop_arrow = None
        self._click_connection = None
        self._animation = None

        self._controls: dict[str, Any] = {}
        self._widget_root = None
        self._status_history: list[str] = []
        self._presets_dir = None

        self._snapshot_job = self.result.metadata.get("job_result")
        self._snapshot_dset = str(self.result.metadata.get("dataset", "m"))
        self._snapshot_slice = self.result.metadata.get("slice_info")
        self._z_bounds = self._infer_z_bounds()

        if self._snapshot_job is not None:
            self._snapshot_cache = SnapshotCache(
                self._snapshot_job,
                dset=self._snapshot_dset,
                slice_info=self._snapshot_slice,
                max_cached=50,
            )
        else:
            self._snapshot_cache = None

    def _infer_z_bounds(self) -> tuple[int, int]:
        if self._snapshot_job is None:
            return (0, 0)
        try:
            dset_obj = self._snapshot_job.get_raw(self._snapshot_dset)
            shape = tuple(getattr(dset_obj, "shape", ()))
            if len(shape) == 5:
                return (0, max(0, int(shape[1]) - 1))
            return (0, 0)
        except Exception:
            return (0, 0)

    @staticmethod
    def _running_in_notebook() -> bool:
        if not _HAS_WIDGETS:
            return False
        ip = get_ipython()
        if ip is None:
            return False
        return bool(getattr(ip, "kernel", None) is not None)

    def _resolve_toolbar(self, toolbar: bool | str) -> bool:
        if isinstance(toolbar, str):
            if toolbar.lower() == "auto":
                return bool(_HAS_WIDGETS and self._running_in_notebook())
            return toolbar.lower() in {"1", "true", "yes", "on"}
        return bool(toolbar)

    def _create_figure(self, figsize: tuple[float, float] | None = None, dpi: int | None = None) -> None:
        cfg = self.result.config
        self._fig = plt.figure(
            figsize=figsize or cfg.figsize,
            dpi=cfg.dpi if dpi is None else dpi,
            constrained_layout=False,
        )
        gs = GridSpec(1, 2, figure=self._fig, width_ratios=[1.15, 1.0], wspace=0.2)
        self._ax_loop = self._fig.add_subplot(gs[0, 0])
        self._ax_snapshot = self._fig.add_subplot(gs[0, 1])

        self._click_connection = self._fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _on_click(self, event) -> None:
        on_loop_click(self, event)

    def _redraw_loop(self) -> None:
        draw_loop_panel(self)
        update_loop_cursor(self, redraw=False)
        if self._fig is not None:
            self._fig.canvas.draw_idle()

    def _frame_for_index(self, idx: int) -> int:
        if self.result.frame_index is None or self.result.frame_index.size == 0:
            return int(idx)
        safe_idx = int(np.clip(idx, 0, self.result.frame_index.size - 1))
        return int(self.result.frame_index[safe_idx])

    def _snapshot_spacing(self) -> tuple[float, float]:
        attrs = getattr(self._snapshot_job, "attrs", {}) if self._snapshot_job is not None else {}
        dx = float(attrs.get("dx", 1e-9)) if hasattr(attrs, "get") else 1e-9
        dy = float(attrs.get("dy", 1e-9)) if hasattr(attrs, "get") else 1e-9
        return dx, dy

    def _update_snapshot(self, *, redraw: bool = True) -> None:
        if self._ax_snapshot is None:
            return
        if self._snapshot_cache is None:
            self._ax_snapshot.clear()
            self._ax_snapshot.text(
                0.5,
                0.5,
                "Snapshot unavailable\n(use from_magnetization source)",
                transform=self._ax_snapshot.transAxes,
                ha="center",
                va="center",
            )
            if redraw and self._fig is not None:
                self._fig.canvas.draw_idle()
            return

        idx = int(self.state.current_idx)
        frame_idx = self._frame_for_index(idx)

        try:
            frame = self._snapshot_cache.get_frame(
                frame_idx,
                component=self.state.snapshot_component,
                z_layer=self.state.z_layer,
                roi=self.state.roi,
            )
            dx, dy = self._snapshot_spacing()
            render_snapshot(
                self._ax_snapshot,
                frame,
                component=self.state.snapshot_component,
                dx=dx,
                dy=dy,
                cmap=self.result.config.colormap_magnitude,
            )
        except Exception as exc:
            self._ax_snapshot.clear()
            self._ax_snapshot.text(
                0.5,
                0.5,
                f"Snapshot error:\n{exc}",
                transform=self._ax_snapshot.transAxes,
                ha="center",
                va="center",
            )

        if redraw and self._fig is not None:
            self._fig.canvas.draw_idle()

    def _set_index(self, idx: int, *, redraw: bool = True) -> None:
        n_points = int(self.result.field.size)
        if n_points == 0:
            return

        self.state.current_idx = int(np.clip(idx, 0, n_points - 1))

        update_loop_cursor(self, redraw=False)
        self._update_snapshot(redraw=False)

        if self._controls and "index" in self._controls:
            slider = self._controls["index"]
            if int(slider.value) != int(self.state.current_idx):
                slider.value = int(self.state.current_idx)

        if redraw and self._fig is not None:
            self._fig.canvas.draw_idle()

        set_status(
            self,
            (
                f"idx={self.state.current_idx}, "
                f"field={self.state.field_value:.5g}, "
                f"M={self.state.magnetization_value:.5g}"
            ),
            color="#0F766E",
        )

    def _apply_state_to_controls(self) -> None:
        if not self._controls:
            return

        self._controls["index"].value = int(self.state.current_idx)
        self._controls["component"].value = str(self.state.snapshot_component)
        if str(self.state.z_layer) != "all":
            self._controls["z_layer"].value = int(self.state.z_layer)
        self._controls["roi"].value = (
            "" if self.state.roi is None else ",".join(str(v) for v in self.state.roi)
        )
        for key in ["hc", "mr", "ms", "arrow", "branch_colors", "trail"]:
            if key in self._controls:
                self._controls[key].value = bool(self.state.show_flags.get(key, False))

    def show(
        self,
        *,
        toolbar: bool | str = "auto",
        show: bool = True,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = None,
        show_hc: bool | None = None,
        show_mr: bool | None = None,
        show_ms: bool | None = None,
        show_arrow: bool | None = None,
        show_branch_colors: bool | None = None,
        trail: bool | None = None,
    ):
        """Open interactive viewer.

        Parameters
        ----------
        toolbar
            ``True`` for ipywidgets toolbar, ``False`` for matplotlib-only mode,
            ``"auto"`` to enable widgets only in notebook environments.
        show
            If ``True``, render immediately (`display`/`plt.show`).
        """
        toolbar_enabled = self._resolve_toolbar(toolbar)

        if show_hc is not None:
            self.state.show_flags["hc"] = bool(show_hc)
        if show_mr is not None:
            self.state.show_flags["mr"] = bool(show_mr)
        if show_ms is not None:
            self.state.show_flags["ms"] = bool(show_ms)
        if show_arrow is not None:
            self.state.show_flags["arrow"] = bool(show_arrow)
        if show_branch_colors is not None:
            self.state.show_flags["branch_colors"] = bool(show_branch_colors)
        if trail is not None:
            self.state.show_flags["trail"] = bool(trail)

        self._create_figure(figsize=figsize, dpi=dpi)
        self._redraw_loop()
        self._update_snapshot(redraw=False)
        self._set_index(self.state.current_idx, redraw=False)

        if toolbar_enabled:
            if not _HAS_WIDGETS:
                raise ImportError("ipywidgets is required for toolbar mode")

            build_toolbar(self, widgets)

            with self._controls["output"]:
                clear_output(wait=True)
                display(self._fig)

            self._apply_state_to_controls()
            set_status(self, "Interactive toolbar ready", color="#0F766E")

            if show:
                display(self._widget_root)
                return None
            return self._widget_root

        set_status(self, "Interactive click mode ready", color="#0F766E")

        if show:
            plt.show()
            return None
        return self._fig

    def close(self) -> None:
        """Close figure and stop animation."""
        stop_animation(self)
        if self._fig is not None and self._click_connection is not None:
            self._fig.canvas.mpl_disconnect(self._click_connection)
        if self._fig is not None:
            plt.close(self._fig)
        self._fig = None
