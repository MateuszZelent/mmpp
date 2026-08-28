"""Matplotlib and animation mixin for dataset plotting."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from .dataset_plotting_core import DatasetPlotCoreMixin


class DatasetPlotMplMixin(DatasetPlotCoreMixin):
    _DEFAULT_FIGSIZE = (8.0, 5.0)

    @staticmethod
    def _default_quiver_scale(
        *,
        stepx: int,
        stepy: int,
        axis_multiplier: float,
        legacy_axis_multiplier: float = 1e-9,
    ) -> float:
        step = max(int(stepx), int(stepy), 1)
        m = float(axis_multiplier)
        if not np.isfinite(m) or m <= 0.0:
            m = float(legacy_axis_multiplier)
        return float((1.0 / float(step)) * (m / float(legacy_axis_multiplier)))

    def _mpl_auto_impl(
        self,
        *,
        z: int = 0,
        t: int = -1,
        ax=None,
        figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
        dpi: int = 100,
        multiplier: float | None = None,
        zero: int | None = None,
        scalar_kw: dict[str, Any] | None = None,
        vector_kw: dict[str, Any] | None = None,
        filename: str | None = None,
    ):
        import matplotlib.pyplot as plt

        scalar_kw = {} if scalar_kw is None else dict(scalar_kw)
        vector_kw = {} if vector_kw is None else dict(vector_kw)

        frame = self._extract_frame(z=z, t=t, zero=zero)
        is_vector = frame.ndim == 3 and frame.shape[-1] >= 2

        if not is_vector:
            scalar_kw.setdefault("figsize", figsize)
            scalar_kw.setdefault("dpi", dpi)
            scalar_kw.setdefault("multiplier", multiplier)
            ax = self._mpl_scalar_impl(z=z, t=t, ax=ax, zero=zero, **scalar_kw)
        else:
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

            scalar_kw.setdefault("component", "norm")
            scalar_kw.setdefault("colorbar", True)
            scalar_kw.setdefault("figsize", figsize)
            scalar_kw.setdefault("dpi", dpi)
            scalar_kw.setdefault("multiplier", multiplier)
            self._mpl_scalar_impl(z=z, t=t, ax=ax, zero=zero, **scalar_kw)

            vector_kw.setdefault("use_color", False)
            vector_kw.setdefault("colorbar", False)
            vector_kw.setdefault("title", None)
            vector_kw.setdefault("figsize", figsize)
            vector_kw.setdefault("dpi", dpi)
            vector_kw.setdefault("multiplier", multiplier)
            ax = self._mpl_vector_impl(z=z, t=t, ax=ax, zero=zero, **vector_kw)

        if filename is not None:
            plt.savefig(str(filename), bbox_inches="tight", pad_inches=0.02)
        return ax

    def _mpl_magnetization_impl(
        self,
        *,
        ax=None,
        figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
        dpi: int = 100,
        multiplier: float | None = None,
        z: int = 0,
        t: int = -1,
        zero: int | None = None,
        scalar_component: int | str | None = "mz",
        vector_vdims: tuple[int | str | None, int | str | None] = (
            "mx",
            "my",
        ),
        filter_field: Any = "norm",
        cmap: str = "viridis",
        colorbar: bool = True,
        colorbar_label: str = "z-component",
        quiver_density: int = 20,
        vector_color: Any = "black",
        vector_width: float = 0.003,
        headwidth: float = 3.5,
        headlength: float = 4.5,
        headaxislength: float = 4.0,
        background_color: str | None = "#e9e9ef",
        cell_grid: bool = False,
        cell_grid_color: str = "white",
        cell_grid_alpha: float = 0.12,
        cell_grid_linewidth: float = 0.35,
        title: str | None = None,
        scalar_kwargs: dict[str, Any] | None = None,
        vector_kwargs: dict[str, Any] | None = None,
        filename: str | None = None,
    ):
        import matplotlib.pyplot as plt

        scalar_kw = {} if scalar_kwargs is None else dict(scalar_kwargs)
        vector_kw = {} if vector_kwargs is None else dict(vector_kwargs)

        filter_ref = filter_field
        if isinstance(filter_field, (int, np.integer, str)):
            filter_ref = (self._dataset, filter_field)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        scalar_kw.setdefault("component", scalar_component)
        scalar_kw.setdefault("cmap", cmap)
        scalar_kw.setdefault("colorbar", bool(colorbar))
        scalar_kw.setdefault("colorbar_label", colorbar_label)
        scalar_kw.setdefault("filter_field", filter_ref)
        scalar_kw.setdefault("interpolation", "none")
        scalar_kw.setdefault("title", None if title is None else str(title))

        self._mpl_scalar_impl(
            z=z,
            t=t,
            ax=ax,
            figsize=figsize,
            dpi=dpi,
            multiplier=multiplier,
            zero=zero,
            **scalar_kw,
        )

        vector_kw.setdefault("vdims", vector_vdims)
        vector_kw.setdefault("use_color", False)
        vector_kw.setdefault("color", vector_color)
        vector_kw.setdefault("colorbar", False)
        vector_kw.setdefault("quiver_density", quiver_density)
        vector_kw.setdefault("filter_field", filter_ref)
        vector_kw.setdefault("pivot", "mid")
        vector_kw.setdefault("width", float(vector_width))
        vector_kw.setdefault("headwidth", float(headwidth))
        vector_kw.setdefault("headlength", float(headlength))
        vector_kw.setdefault("headaxislength", float(headaxislength))
        vector_kw.setdefault("title", None if title is None else str(title))

        self._mpl_vector_impl(
            z=z,
            t=t,
            ax=ax,
            figsize=figsize,
            dpi=dpi,
            multiplier=multiplier,
            zero=zero,
            **vector_kw,
        )

        if background_color is not None:
            ax.set_facecolor(str(background_color))

        if cell_grid:
            frame = self._extract_frame(z=z, t=t, zero=zero)
            base = self._component_image(frame, scalar_component, default="norm")
            self._mpl_add_cell_grid(
                ax,
                base.shape,
                multiplier=multiplier,
                color=cell_grid_color,
                alpha=float(cell_grid_alpha),
                linewidth=float(cell_grid_linewidth),
            )

        if title is not None:
            ax.set_title(str(title))

        if filename is not None:
            plt.savefig(str(filename), bbox_inches="tight", pad_inches=0.02)
        return ax

    def _mpl_scalar_impl(
        self,
        *,
        z: int = 0,
        t: int = -1,
        ax=None,
        figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
        dpi: int = 100,
        multiplier: float | None = None,
        repeat: int = 1,
        zero: int | None = None,
        component: int | str | None = None,
        filter_field: Any = None,
        cmap: str = "viridis",
        colorbar: bool = True,
        colorbar_label: str = "",
        symmetric_clim: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        title: str | None = None,
        filename: str | None = None,
        **imshow_kwargs,
    ):
        import matplotlib.pyplot as plt

        frame = self._extract_frame(z=z, t=t, zero=zero)
        image = self._component_image(frame, component, default="norm")
        image = np.tile(image, (max(int(repeat), 1), max(int(repeat), 1)))
        image = np.asarray(image, dtype=np.float32)

        if filter_field is not None:
            mask = self._coerce_mask(filter_field, image.shape, t=t, z=z, zero=zero)
            image = np.where(mask, image, np.nan)

        if "clim" in imshow_kwargs and (vmin is None and vmax is None):
            clim = imshow_kwargs.pop("clim")
            if isinstance(clim, (tuple, list)) and len(clim) == 2:
                vmin, vmax = float(clim[0]), float(clim[1])

        if symmetric_clim and vmin is None and vmax is None:
            local_min = float(np.nanmin(image)) if np.isfinite(image).any() else 0.0
            local_max = float(np.nanmax(image)) if np.isfinite(image).any() else 0.0
            vmax_abs = max(abs(local_min), abs(local_max))
            vmin = -vmax_abs
            vmax = vmax_abs

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        _, _, extent, _, unit_label = self._resolve_plot_geometry(
            image.shape,
            multiplier=multiplier,
        )
        im = ax.imshow(
            image,
            origin="lower",
            interpolation=imshow_kwargs.pop("interpolation", "none"),
            aspect=imshow_kwargs.pop("aspect", "equal"),
            extent=extent,
            cmap=imshow_kwargs.pop("cmap", cmap),
            vmin=vmin,
            vmax=vmax,
            **imshow_kwargs,
        )

        if colorbar:
            cb = self._mpl_add_colorbar(ax, im, colorbar_label or None)
            if colorbar_label:
                cb.set_label(str(colorbar_label))

        if title is None:
            comp_label = "norm" if component is None else str(component)
            title = f"{self._dataset.dataset_name} [{comp_label}]"
        ax.set(title=title)
        ax.set_aspect("equal")
        self._set_axis_labels(ax, unit_label)

        if filename is not None:
            plt.savefig(str(filename), bbox_inches="tight", pad_inches=0.02)
        return ax

    def _mpl_vector_impl(
        self,
        *,
        z: int = 0,
        t: int = -1,
        ax=None,
        figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
        dpi: int = 100,
        multiplier: float | None = None,
        repeat: int = 1,
        zero: int | None = None,
        filter_field: Any = None,
        vdims: tuple[int | str | None, int | str | None] | None = None,
        vdim_mapping: dict[Any, Any] | None = None,
        color_field: int | str | np.ndarray | None = None,
        cmap: str = "viridis",
        use_color: bool = True,
        colorbar: bool = True,
        colorbar_label: str = "",
        quiver_density: int = 20,
        vector_scale: float | None = None,
        pivot: str = "mid",
        title: str | None = None,
        filename: str | None = None,
        **quiver_kwargs,
    ):
        import matplotlib.pyplot as plt

        frame = self._extract_frame(z=z, t=t, zero=zero)
        if frame.ndim != 3 or frame.shape[-1] < 2:
            raise ValueError(
                f"Vector plotting expects frame shape (y, x, c>=2), got {frame.shape}"
            )

        src_n_comp = int(frame.shape[-1])
        vec = np.asarray(frame, dtype=np.float32)
        if vec.shape[-1] < 3:
            padded = np.zeros(vec.shape[:-1] + (3,), dtype=np.float32)
            padded[..., : vec.shape[-1]] = vec
            vec = padded
        vec = np.tile(vec, (max(int(repeat), 1), max(int(repeat), 1), 1))
        n_comp = int(src_n_comp)
        comp_mapping = self._resolve_vdim_mapping(n_comp, vdim_mapping)

        if vdims is None:
            arrow_x = comp_mapping.get("x", 0 if n_comp >= 1 else None)
            arrow_y = comp_mapping.get("y", 1 if n_comp >= 2 else None)
        else:
            if len(vdims) != 2:
                raise ValueError(f"{vdims=} must contain exactly 2 elements")
            arrow_x = self._resolve_component_index(
                vdims[0],
                n_comp,
                mapping=comp_mapping,
                allow_none=True,
            )
            arrow_y = self._resolve_component_index(
                vdims[1],
                n_comp,
                mapping=comp_mapping,
                allow_none=True,
            )
            if arrow_x is None and arrow_y is None:
                raise ValueError(f"At least one element in {vdims=} must not be None")

        u = (
            np.asarray(vec[:, :, arrow_x], dtype=np.float32)
            if arrow_x is not None
            else np.zeros(vec.shape[:2], dtype=np.float32)
        )
        v = (
            np.asarray(vec[:, :, arrow_y], dtype=np.float32)
            if arrow_y is not None
            else np.zeros(vec.shape[:2], dtype=np.float32)
        )
        (
            np.asarray(vec[:, :, 2], dtype=np.float32)
            if n_comp >= 3
            else np.zeros_like(u)
        )

        if filter_field is not None:
            mask = self._coerce_mask(filter_field, u.shape, t=t, z=z, zero=zero)
        else:
            mask = np.ones_like(u, dtype=bool)

        dens = max(int(quiver_density), 1)
        stepx = max(int(u.shape[1] / dens), 1)
        stepy = max(int(u.shape[0] / dens), 1)

        u_ds = np.asarray(u[::stepy, ::stepx], dtype=np.float32)
        v_ds = np.asarray(v[::stepy, ::stepx], dtype=np.float32)
        mask_ds = np.asarray(mask[::stepy, ::stepx], dtype=bool)
        u_ds = np.where(mask_ds, u_ds, np.nan)
        v_ds = np.where(mask_ds, v_ds, np.nan)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        dx_u, dy_u, extent, axis_multiplier, unit_label = self._resolve_plot_geometry(
            u.shape,
            multiplier=multiplier,
        )
        x0_u, _, y0_u, _ = extent
        x, y = np.meshgrid(
            x0_u + (np.arange(0, u.shape[1], stepx) + 0.5) * dx_u,
            y0_u + (np.arange(0, u.shape[0], stepy) + 0.5) * dy_u,
        )

        c_ds = None
        if use_color:
            if color_field is None:
                if n_comp == 3:
                    preferred = [0, 1, 2]
                    for used in (arrow_x, arrow_y):
                        if used in preferred:
                            preferred.remove(used)
                    color_idx = preferred[0] if preferred else 2
                    c_full = np.asarray(vec[:, :, color_idx], dtype=np.float32)
                else:
                    warnings.warn(
                        "Automatic coloring is only supported for 3-component vectors. "
                        f"Ignoring '{use_color=}'.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    use_color = False
            elif isinstance(color_field, (int, np.integer, str)):
                if isinstance(color_field, str):
                    idx = self._resolve_component_index(
                        color_field,
                        n_comp,
                        mapping=comp_mapping,
                        allow_none=False,
                    )
                    c_full = np.asarray(vec[:, :, idx], dtype=np.float32)
                else:
                    c_full = self._component_image(vec, color_field, default="norm")
            else:
                c_full = self._coerce_scalar_field(
                    color_field,
                    u.shape,
                    t=t,
                    z=z,
                    zero=zero,
                    component=None,
                    default="norm",
                )
            if use_color:
                c_ds = np.asarray(c_full[::stepy, ::stepx], dtype=np.float32)
                c_ds = np.where(mask_ds, c_ds, np.nan)

        quiver_kw = dict(quiver_kwargs)
        passed_scale = quiver_kw.pop("scale", None)
        if passed_scale is not None:
            scale_value = float(passed_scale)
        elif vector_scale is not None:
            scale_value = float(vector_scale)
        else:
            scale_value = type(self)._default_quiver_scale(
                stepx=stepx,
                stepy=stepy,
                axis_multiplier=axis_multiplier,
            )
        quiver_kw.setdefault("angles", "xy")
        quiver_kw.setdefault("scale_units", "xy")
        quiver_kw.setdefault("pivot", pivot)
        quiver_kw["scale"] = float(scale_value)

        if c_ds is None:
            quiver = ax.quiver(
                x,
                y,
                u_ds,
                v_ds,
                **quiver_kw,
            )
        else:
            quiver_kw.setdefault("cmap", cmap)
            quiver = ax.quiver(
                x,
                y,
                u_ds,
                v_ds,
                c_ds,
                **quiver_kw,
            )

        if colorbar and c_ds is not None:
            cb = self._mpl_add_colorbar(ax, quiver, colorbar_label or None)
            if colorbar_label:
                cb.set_label(str(colorbar_label))

        if title is None:
            title = f"{self._dataset.dataset_name} [vector]"
        ax.set(title=title)
        ax.set_aspect("equal")
        self._set_axis_labels(ax, unit_label)

        if filename is not None:
            plt.savefig(str(filename), bbox_inches="tight", pad_inches=0.02)
        return ax

    def _mpl_contour_impl(
        self,
        *,
        z: int = 0,
        t: int = -1,
        ax=None,
        figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
        dpi: int = 100,
        multiplier: float | None = None,
        repeat: int = 1,
        zero: int | None = None,
        component: int | str | None = None,
        filter_field: Any = None,
        levels: int = 12,
        filled: bool = True,
        cmap: str = "viridis",
        colorbar: bool = True,
        colorbar_label: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        title: str | None = None,
        filename: str | None = None,
        **contour_kwargs,
    ):
        import matplotlib.pyplot as plt

        frame = self._extract_frame(z=z, t=t, zero=zero)
        image = self._component_image(frame, component, default="norm")
        image = np.tile(image, (max(int(repeat), 1), max(int(repeat), 1)))
        image = np.asarray(image, dtype=np.float32)

        if filter_field is not None:
            mask = self._coerce_mask(filter_field, image.shape, t=t, z=z, zero=zero)
            image = np.where(mask, image, np.nan)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        dx_u, dy_u, extent, _, unit_label = self._resolve_plot_geometry(
            image.shape,
            multiplier=multiplier,
        )
        x0_u, _, y0_u, _ = extent
        x = x0_u + (np.arange(image.shape[1], dtype=np.float32) + 0.5) * dx_u
        y = y0_u + (np.arange(image.shape[0], dtype=np.float32) + 0.5) * dy_u

        if filled:
            cp = ax.contourf(
                x,
                y,
                image,
                levels=max(int(levels), 2),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                **contour_kwargs,
            )
        else:
            cp = ax.contour(
                x,
                y,
                image,
                levels=max(int(levels), 2),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                **contour_kwargs,
            )

        if colorbar:
            self._mpl_add_colorbar(ax, cp, colorbar_label)

        if title is None:
            comp_label = "norm" if component is None else str(component)
            title = f"{self._dataset.dataset_name} [contour:{comp_label}]"
        ax.set(title=title)
        ax.set_aspect("equal")
        self._set_axis_labels(ax, unit_label)

        if filename is not None:
            plt.savefig(str(filename), bbox_inches="tight", pad_inches=0.02)
        return ax

    @staticmethod
    def _mpl_add_colorwheel(
        ax,
        *,
        width=1.0,
        height=1.0,
        loc: str = "lower right",
        **kwargs,
    ):
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        from ..plotting import hsl2rgb

        n = 200
        x = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        theta = np.mod(np.arctan2(yy, xx) + 2.0 * np.pi, 2.0 * np.pi)
        radius = np.sqrt(xx**2 + yy**2)

        hsl = np.ones((n, n, 3), dtype=np.float32)
        hsl[:, :, 0] = theta / (2.0 * np.pi)
        hsl[:, :, 1] = 1.0
        hsl[:, :, 2] = np.clip(radius / np.sqrt(2.0), 0.0, 1.0)
        rgb = hsl2rgb(hsl)

        rgba = np.zeros((n, n, 4), dtype=np.float32)
        inside = radius <= 1.0
        rgba[inside, :3] = rgb[inside]
        rgba[inside, 3] = 1.0

        cw_ax = inset_axes(ax, width=width, height=height, loc=loc, **kwargs)
        cw_ax.imshow(rgba, origin="lower")
        cw_ax.axis("off")
        return cw_ax

    @staticmethod
    def _mpl_add_colorbar(
        ax,
        mappable,
        colorbar_label: str | None = None,
        *,
        min_height_inches: float = 2.0,
        min_width_inches: float = 0.35,
        min_pad_inches: float = 0.1,
    ):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import Size, make_axes_locatable

        fig = ax.figure
        fig_w, fig_h = fig.get_size_inches()
        pos = ax.get_position()

        min_height_norm = min_height_inches / (fig_h * max(pos.y1 - pos.y0, 1e-12))
        min_width_norm = min_width_inches / max(fig_w, 1e-12)
        min_pad_norm = min_pad_inches / max(fig_w, 1e-12)

        if min_pad_norm > 0.05:
            pad_h = Size.Fixed(min_pad_inches)
        else:
            pad_h = Size.AxesX(ax, aspect=0.05)

        if min_width_norm > 0.05:
            width_h = Size.Fixed(min_width_inches)
        else:
            width_h = Size.AxesX(ax, aspect=0.05)

        v_aspect = min_height_norm if min_height_norm > 1 else 1
        existing_cbs = [a for a in fig.get_axes() if f"cb_{id(ax)}" in a.get_label()]
        divider = make_axes_locatable(ax)
        cax = fig.add_axes(
            divider.get_position(),
            label=f"cb_{id(ax)}_{len(existing_cbs)}",
        )

        if len(existing_cbs) == 0:
            divider.set_horizontal([Size.AxesX(ax), pad_h, width_h])
        else:
            divider.new_horizontal(pad_h, pack_start=False)
            divider.new_horizontal(width_h, pack_start=False)

        divider.set_vertical([Size.AxesY(ax, aspect=v_aspect)])
        ax.set_axes_locator(divider.new_locator(nx=0, ny=0))

        for i, cb in enumerate(existing_cbs, start=1):
            cb.set_axes_locator(divider.new_locator(nx=2 * i, ny=0))
        cax.set_axes_locator(divider.new_locator(nx=2 * (len(existing_cbs) + 1), ny=0))

        cbar = plt.colorbar(mappable, cax=cax)
        if colorbar_label is not None and colorbar_label != "":
            cbar.ax.set_ylabel(str(colorbar_label))
        return cbar

    def _mpl_add_cell_grid(
        self,
        ax,
        shape_xy: tuple[int, int],
        *,
        multiplier: float | None = None,
        color: str = "white",
        alpha: float = 0.12,
        linewidth: float = 0.35,
    ) -> None:
        dx_u, dy_u, extent, _, _ = self._resolve_plot_geometry(
            shape_xy,
            multiplier=multiplier,
        )
        x0_u, x1_u, y0_u, y1_u = extent
        x_edges = x0_u + np.arange(int(shape_xy[1]) + 1, dtype=np.float32) * dx_u
        y_edges = y0_u + np.arange(int(shape_xy[0]) + 1, dtype=np.float32) * dy_u

        ax.set_xticks(x_edges, minor=True)
        ax.set_yticks(y_edges, minor=True)
        ax.grid(
            which="minor",
            color=str(color),
            linewidth=float(linewidth),
            alpha=float(alpha),
        )
        ax.tick_params(which="minor", length=0)

    def _mpl_lightness_impl(
        self,
        *,
        z: int = 0,
        t: int = -1,
        ax=None,
        figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
        dpi: int = 100,
        multiplier: float | None = None,
        repeat: int = 1,
        zero: int | None = None,
        filter_field: Any = None,
        lightness_field: int | str | np.ndarray | None = None,
        clim: tuple[float, float] | None = None,
        colorwheel: bool = True,
        colorwheel_xlabel: str | None = None,
        colorwheel_ylabel: str | None = None,
        colorwheel_args: dict[str, Any] | None = None,
        title: str | None = None,
        filename: str | None = None,
        **imshow_kwargs,
    ):
        import matplotlib.pyplot as plt

        from ..plotting import hsl2rgb

        frame = self._extract_frame(z=z, t=t, zero=zero)

        if frame.ndim == 2:
            hue = np.asarray(frame, dtype=np.float32)
            if lightness_field is None:
                lightness = np.ones_like(hue, dtype=np.float32)
            elif isinstance(lightness_field, (int, np.integer, str)):
                lightness = self._component_image(
                    frame, lightness_field, default="norm"
                )
            else:
                lightness = np.asarray(lightness_field, dtype=np.float32)
                lightness = np.squeeze(lightness)
                if lightness.shape != hue.shape:
                    lightness = np.broadcast_to(lightness, hue.shape)
        elif frame.ndim == 3 and frame.shape[-1] >= 2:
            vec = np.asarray(frame, dtype=np.float32)
            if vec.shape[-1] < 3:
                padded = np.zeros(vec.shape[:-1] + (3,), dtype=np.float32)
                padded[..., : vec.shape[-1]] = vec
                vec = padded
            u = vec[:, :, 0]
            v = vec[:, :, 1]
            hue = np.mod(np.arctan2(v, u) + 2.0 * np.pi, 2.0 * np.pi)

            if lightness_field is None:
                lightness = np.asarray(vec[:, :, 2], dtype=np.float32)
            elif isinstance(lightness_field, (int, np.integer, str)):
                lightness = self._component_image(vec, lightness_field, default="norm")
            else:
                lightness = np.asarray(lightness_field, dtype=np.float32)
                lightness = np.squeeze(lightness)
                if lightness.shape != hue.shape:
                    lightness = np.broadcast_to(lightness, hue.shape)
        else:
            raise ValueError(
                "lightness plot expects 2d scalar or 2d vector frame, "
                f"got shape {frame.shape}"
            )

        hue = np.tile(hue, (max(int(repeat), 1), max(int(repeat), 1))).astype(
            np.float32, copy=False
        )
        lightness = np.tile(
            np.asarray(lightness, dtype=np.float32),
            (max(int(repeat), 1), max(int(repeat), 1)),
        )

        if filter_field is not None:
            mask = self._coerce_mask(filter_field, hue.shape, t=t, z=z, zero=zero)
        else:
            mask = np.ones_like(hue, dtype=bool)

        if clim is not None:
            lo, hi = float(clim[0]), float(clim[1])
        else:
            lo = float(np.nanmin(lightness)) if np.isfinite(lightness).any() else 0.0
            hi = float(np.nanmax(lightness)) if np.isfinite(lightness).any() else 1.0
        if hi <= lo:
            hi = lo + 1e-12

        hue_norm = np.mod(hue, 2.0 * np.pi) / (2.0 * np.pi)
        lightness_norm = np.clip((lightness - lo) / (hi - lo), 0.0, 1.0)

        hsl = np.ones(hue.shape + (3,), dtype=np.float32)
        hsl[:, :, 0] = hue_norm
        hsl[:, :, 1] = 1.0
        hsl[:, :, 2] = lightness_norm
        rgb = hsl2rgb(hsl)

        rgba = np.zeros(hue.shape + (4,), dtype=np.float32)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = 1.0
        rgba[~mask] = 0.0

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        _, _, extent, _, unit_label = self._resolve_plot_geometry(
            hue.shape,
            multiplier=multiplier,
        )
        ax.imshow(
            rgba,
            origin="lower",
            interpolation=imshow_kwargs.pop("interpolation", "none"),
            aspect=imshow_kwargs.pop("aspect", "equal"),
            extent=extent,
            **imshow_kwargs,
        )

        if colorwheel:
            kw = {} if colorwheel_args is None else dict(colorwheel_args)
            cw_ax = self._mpl_add_colorwheel(ax, **kw)
            if colorwheel_xlabel is not None:
                cw_ax.arrow(100, 100, 60, 0, width=5, fc="w", ec="w")
                cw_ax.annotate(str(colorwheel_xlabel), (115, 140), c="w")
            if colorwheel_ylabel is not None:
                cw_ax.arrow(100, 100, 0, -60, width=5, fc="w", ec="w")
                cw_ax.annotate(str(colorwheel_ylabel), (40, 80), c="w")

        if title is None:
            title = f"{self._dataset.dataset_name} [lightness]"
        ax.set(title=title)
        ax.set_aspect("equal")
        self._set_axis_labels(ax, unit_label)

        if filename is not None:
            plt.savefig(str(filename), bbox_inches="tight", pad_inches=0.02)
        return ax

    def _render_frame(
        self,
        frame: np.ndarray,
        *,
        ax,
        mode: str,
        multiplier: float | None = None,
        repeat: int = 1,
        cmap: str | None = None,
        component: int | str | None = None,
        quiver_density: int = 20,
        colorbar: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        title: str | None = None,
    ):
        from ..plotting import hsl2rgb

        draw_mode = self._normalize_mode(mode)
        ax.clear()
        repeat_value = max(int(repeat), 1)
        if frame.ndim >= 2:
            shape_xy = (
                int(frame.shape[0]) * repeat_value,
                int(frame.shape[1]) * repeat_value,
            )
        else:
            shape_xy = (repeat_value, repeat_value)
        dx_u, dy_u, extent, axis_multiplier, unit_label = self._resolve_plot_geometry(
            shape_xy,
            multiplier=multiplier,
        )

        if draw_mode == "snapshot":
            is_vector = frame.ndim == 3 and frame.shape[-1] >= 2 and component is None
            if is_vector:
                vec = np.asarray(frame, dtype=np.float32)
                if vec.shape[-1] < 3:
                    padded = np.zeros(vec.shape[:-1] + (3,), dtype=np.float32)
                    padded[..., : vec.shape[-1]] = vec
                    vec = padded

                vector = np.tile(vec, (repeat_value, repeat_value, 1))
                u = vector[:, :, 0]
                v = vector[:, :, 1]
                w = vector[:, :, 2]

                alphas = np.clip(-np.abs(w) + 1, 0.0, 1.0)
                hsl = np.ones((u.shape[0], u.shape[1], 3), dtype=np.float32)
                hsl[:, :, 0] = np.angle(u + 1j * v) / np.pi / 2
                hsl[:, :, 1] = np.clip(np.sqrt(u**2 + v**2 + w**2), 0.0, 1.0)
                hsl[:, :, 2] = (w + 1) / 2
                rgb = hsl2rgb(hsl)

                dens = max(int(quiver_density), 1)
                stepx = max(int(u.shape[1] / dens), 1)
                stepy = max(int(u.shape[0] / dens), 1)
                scale = type(self)._default_quiver_scale(
                    stepx=stepx,
                    stepy=stepy,
                    axis_multiplier=axis_multiplier,
                )
                x0_u, _, y0_u, _ = extent
                x, y = np.meshgrid(
                    x0_u + (np.arange(0, u.shape[1], stepx) + 0.5) * dx_u,
                    y0_u + (np.arange(0, u.shape[0], stepy) + 0.5) * dy_u,
                )

                ax.quiver(
                    x,
                    y,
                    u[::stepy, ::stepx],
                    v[::stepy, ::stepx],
                    alpha=alphas[::stepy, ::stepx],
                    angles="xy",
                    scale_units="xy",
                    scale=scale,
                )
                ax.imshow(
                    rgb,
                    interpolation="none",
                    origin="lower",
                    aspect="equal",
                    extent=extent,
                )
            else:
                image = self._component_image(frame, component, default="norm")
                image = np.tile(image, (repeat_value, repeat_value))
                im = ax.imshow(
                    image,
                    interpolation="none",
                    origin="lower",
                    aspect="equal",
                    cmap=cmap or "viridis",
                    vmin=vmin,
                    vmax=vmax,
                    extent=extent,
                )
                if colorbar:
                    self._mpl_add_colorbar(ax, im)
        else:
            image = self._component_image(frame, component, default="norm")
            image = np.tile(image, (repeat_value, repeat_value))
            im = ax.imshow(
                image,
                interpolation="none",
                origin="lower",
                aspect="equal",
                cmap=cmap or "viridis",
                vmin=vmin,
                vmax=vmax,
                extent=extent,
            )
            if colorbar:
                self._mpl_add_colorbar(ax, im)

        job_name = getattr(self._dataset.job_result, "name", "job")
        dset = self._dataset.dataset_name
        if title is None:
            if draw_mode == "heatmap":
                comp_label = "norm" if component is None else str(component)
                title = f"{job_name} — {dset} [{comp_label}]"
            else:
                title = f"{job_name} — {dset}"
        ax.set(title=title)
        self._set_axis_labels(ax, unit_label)
        return ax

    def _snapshot_impl(
        self,
        *,
        z: int = 0,
        t: int = -1,
        ax=None,
        figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
        dpi: int = 100,
        multiplier: float | None = None,
        repeat: int = 1,
        zero: int | None = None,
        cmap: str | None = None,
        component: int | str | None = None,
        quiver_density: int = 20,
        colorbar: bool = True,
    ):
        import matplotlib.pyplot as plt

        frame = self._extract_frame(z=z, t=t, zero=zero)
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax = self._render_frame(
            frame,
            ax=ax,
            mode="snapshot",
            multiplier=multiplier,
            repeat=repeat,
            cmap=cmap,
            component=component,
            quiver_density=quiver_density,
            colorbar=colorbar,
        )
        return ax

    def _heatmap_impl(
        self,
        *,
        z: int = 0,
        t: int = -1,
        ax=None,
        figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
        dpi: int = 100,
        multiplier: float | None = None,
        repeat: int = 1,
        zero: int | None = None,
        component: int | str | None = None,
        cmap: str = "viridis",
        colorbar: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
    ):
        import matplotlib.pyplot as plt

        frame = self._extract_frame(z=z, t=t, zero=zero)
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax = self._render_frame(
            frame,
            ax=ax,
            mode="heatmap",
            multiplier=multiplier,
            repeat=repeat,
            cmap=cmap,
            component=component,
            colorbar=colorbar,
            vmin=vmin,
            vmax=vmax,
        )
        return ax

    def interactive(
        self,
        *,
        mode: str = "snapshot",
        component: int | str | None = None,
        z: int = 0,
        figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
        dpi: int = 100,
        multiplier: float | None = None,
        repeat: int = 1,
        zero: int | None = None,
        remove_static: bool = False,
        static_reference: int = 0,
        cmap: str = "viridis",
        quiver_density: int = 20,
        fps: int = 20,
        toolbar: bool = True,
    ):
        import matplotlib.pyplot as plt
        from matplotlib import animation as mpl_animation
        from matplotlib.widgets import Button, CheckButtons, Slider

        sequence = self._extract_sequence(z=z, zero=None)
        n_frames = int(sequence.shape[0])
        reference_index = self._normalize_index(
            static_reference if zero is None else zero,
            n_frames,
        )
        static_frame = np.asarray(sequence[reference_index], dtype=np.float32)
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        if toolbar:
            plt.subplots_adjust(bottom=0.18)
            slider_ax = fig.add_axes((0.15, 0.07, 0.55, 0.04))
            button_ax = fig.add_axes((0.74, 0.065, 0.12, 0.05))
            static_ax = fig.add_axes((0.88, 0.065, 0.08, 0.05))
            frame_slider = Slider(
                slider_ax,
                "Frame",
                valmin=0,
                valmax=max(n_frames - 1, 0),
                valinit=0,
                valstep=1,
            )
            play_btn = Button(button_ax, "Play", color="#e5e7eb", hovercolor="#d1d5db")
            static_check = CheckButtons(
                static_ax,
                [r"$\Delta m_0$"],
                [bool(remove_static or zero is not None)],
            )
        else:
            frame_slider = None
            play_btn = None
            static_check = None

        state = {
            "index": 0,
            "playing": False,
            "remove_static": bool(remove_static or zero is not None),
            "static_reference": reference_index,
        }
        render_mode = self._normalize_mode(mode)

        def _draw(index: int) -> None:
            idx = int(np.clip(int(index), 0, max(n_frames - 1, 0)))
            state["index"] = idx
            frame = np.asarray(sequence[idx], dtype=np.float32)
            if state["remove_static"]:
                frame = frame - static_frame
            title = f"{self._dataset.dataset_name} [{idx + 1}/{n_frames}]"
            if state["remove_static"]:
                title += f" - static removed t={reference_index}"
            self._render_frame(
                frame,
                ax=ax,
                mode=render_mode,
                multiplier=multiplier,
                repeat=repeat,
                cmap=cmap,
                component=component,
                quiver_density=quiver_density,
                colorbar=False,
                title=title,
            )
            if frame_slider is not None and int(round(frame_slider.val)) != idx:
                frame_slider.eventson = False
                frame_slider.set_val(idx)
                frame_slider.eventson = True
            fig.canvas.draw_idle()

        def _on_slider(val):
            _draw(int(round(float(val))))

        def _on_toggle(_event):
            state["playing"] = not state["playing"]
            if play_btn is not None:
                play_btn.label.set_text("Pause" if state["playing"] else "Play")
                fig.canvas.draw_idle()

        def _on_static_toggle(_label):
            state["remove_static"] = not bool(state["remove_static"])
            _draw(state["index"])

        def _tick(_frame):
            if not state["playing"]:
                return ()
            _draw((state["index"] + 1) % max(n_frames, 1))
            return ()

        if frame_slider is not None:
            frame_slider.on_changed(_on_slider)
        if play_btn is not None:
            play_btn.on_clicked(_on_toggle)
        if static_check is not None:
            static_check.on_clicked(_on_static_toggle)

        anim = mpl_animation.FuncAnimation(
            fig,
            _tick,
            interval=1000.0 / max(int(fps), 1),
            blit=False,
            cache_frame_data=False,
        )
        fig.__dict__["_mmpp_interactive"] = {
            "slider": frame_slider,
            "play_button": play_btn,
            "remove_static_check": static_check,
            "animation": anim,
            "state": state,
            "draw": _draw,
        }
        _draw(0)
        return fig

    def animate(
        self,
        *,
        mode: str = "snapshot",
        component: int | str | None = None,
        z: int = 0,
        multiplier: float | None = None,
        repeat: int = 1,
        zero: int | None = None,
        cmap: str = "viridis",
        quiver_density: int = 20,
        fps: int = 20,
        save_path: str | None = None,
        figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
        dpi: int = 100,
    ):
        import matplotlib.pyplot as plt
        from matplotlib import animation as mpl_animation

        sequence = self._extract_sequence(z=z, zero=zero)
        n_frames = int(sequence.shape[0])
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        render_mode = self._normalize_mode(mode)

        def _update(frame_idx: int):
            idx = int(frame_idx) % max(n_frames, 1)
            frame = np.asarray(sequence[idx], dtype=np.float32)
            title = f"{self._dataset.dataset_name} [{idx + 1}/{n_frames}]"
            self._render_frame(
                frame,
                ax=ax,
                mode=render_mode,
                multiplier=multiplier,
                repeat=repeat,
                cmap=cmap,
                component=component,
                quiver_density=quiver_density,
                colorbar=False,
                title=title,
            )
            return []

        anim = mpl_animation.FuncAnimation(
            fig,
            _update,
            frames=max(n_frames, 1),
            interval=1000.0 / max(int(fps), 1),
            repeat=True,
            blit=False,
        )

        if save_path is None:
            return anim

        path = str(save_path)
        suffix = path.lower().rsplit(".", 1)[-1] if "." in path else ""
        writer: Any
        if suffix == "mp4":
            writer = mpl_animation.FFMpegWriter(fps=max(int(fps), 1), bitrate=2000)
        elif suffix == "gif":
            writer = mpl_animation.PillowWriter(fps=max(int(fps), 1))
        else:
            raise ValueError("save_path extension must be .mp4 or .gif")

        anim.save(path, writer=writer, dpi=dpi)
        return path
