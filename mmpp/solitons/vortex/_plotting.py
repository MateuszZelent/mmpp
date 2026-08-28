"""Shared matplotlib plotting helpers for vortex soliton modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_AXES_STYLE_KEYS = {
    "title",
    "xlabel",
    "ylabel",
    "xlim",
    "ylim",
    "xscale",
    "yscale",
    "aspect",
    "xmargin",
    "ymargin",
    "grid",
    "grid_kwargs",
    "legend",
    "legend_kwargs",
    "tight_layout",
    "ax_set",
}

_FIGURE_KEYS = {"figsize", "dpi", "subplot_kw", "gridspec_kw", "layout"}


def pop_axes_style_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pop axis-style options from plotting kwargs."""
    style: dict[str, Any] = {}
    for key in list(kwargs.keys()):
        if key in _AXES_STYLE_KEYS:
            style[key] = kwargs.pop(key)
    return style


def pop_figure_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pop figure-creation options from plotting kwargs."""
    figure: dict[str, Any] = {}
    for key in list(kwargs.keys()):
        if key in _FIGURE_KEYS:
            figure[key] = kwargs.pop(key)
    return figure


def ensure_axis(
    ax=None, *, default_figsize=None, figure_kwargs: dict[str, Any] | None = None
):
    """Return axis, creating a new figure if ``ax`` is None."""
    if ax is not None:
        return ax

    import matplotlib.pyplot as plt

    options = {} if figure_kwargs is None else dict(figure_kwargs)
    if default_figsize is not None and "figsize" not in options:
        options["figsize"] = default_figsize
    _, axis = plt.subplots(**options)
    return axis


def apply_axes_style(ax, style_kwargs: dict[str, Any] | None) -> None:
    """Apply axis-level style options (labels, limits, scales, grid, legend)."""
    if not style_kwargs:
        return

    style = dict(style_kwargs)

    ax_set = style.pop("ax_set", None)
    if isinstance(ax_set, Mapping):
        ax.set(**dict(ax_set))

    if "title" in style:
        ax.set_title(style.pop("title"))
    if "xlabel" in style:
        ax.set_xlabel(style.pop("xlabel"))
    if "ylabel" in style:
        ax.set_ylabel(style.pop("ylabel"))
    if "xlim" in style:
        ax.set_xlim(style.pop("xlim"))
    if "ylim" in style:
        ax.set_ylim(style.pop("ylim"))
    if "xscale" in style:
        ax.set_xscale(style.pop("xscale"))
    if "yscale" in style:
        ax.set_yscale(style.pop("yscale"))
    if "aspect" in style:
        ax.set_aspect(style.pop("aspect"))
    if "xmargin" in style:
        ax.margins(x=style.pop("xmargin"))
    if "ymargin" in style:
        ax.margins(y=style.pop("ymargin"))

    grid_kwargs = style.pop("grid_kwargs", None)
    if "grid" in style:
        grid = style.pop("grid")
        if isinstance(grid, Mapping):
            ax.grid(**dict(grid))
        elif grid is not None:
            if isinstance(grid_kwargs, Mapping):
                ax.grid(bool(grid), **dict(grid_kwargs))
            else:
                ax.grid(bool(grid))

    legend_kwargs = style.pop("legend_kwargs", None)
    if "legend" in style:
        legend = style.pop("legend")
        if isinstance(legend, Mapping):
            ax.legend(**dict(legend))
        elif legend is True:
            if isinstance(legend_kwargs, Mapping):
                ax.legend(**dict(legend_kwargs))
            else:
                ax.legend()
        elif legend is False:
            handle = ax.get_legend()
            if handle is not None:
                handle.remove()
        elif legend is not None:
            if isinstance(legend_kwargs, Mapping):
                ax.legend(loc=legend, **dict(legend_kwargs))
            else:
                ax.legend(loc=legend)

    if style.pop("tight_layout", False):
        ax.figure.tight_layout()
