"""DispersionFilterChain – filter-configuration stage in the dispersion pipeline.

This is the object returned by ``FFTDispersionInterface.filters(...)`` and
mirrors the role of ``SpectrumFilterChain`` in the FMR/spectrum module.

Usage::

    result = (
        job[0].m[:, ..., 0:1]
        .fft.dispersion
        .filters(remove_static=True, live={"gaussian_morph": {"enabled": True}})
        .compute_1d(axis='x', save=True)
    )
    # result is DispersionResult1D with .plot, .analyze, .modes

    lowest = result.analyze.find_lowest_possible_frequency()
    lowest.plot.heatmap()
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from mmpp._repr_helpers import (
    NODE_COLOR_ANALYSIS,
    NODE_COLOR_COMPUTE,
    NODE_COLOR_PLOT,
    NODE_COLOR_UTIL,
    accessors_section_html,
    api_help_html,
    examples_section_html,
    metrics_section_html,
    node_card_html,
)

if TYPE_CHECKING:
    from ..interface import FFTDispersionInterface
    from ..models import DispersionResult1D, DispersionResult2D


class DispersionFilterChain:
    """Filter-configuration proxy returned by ``FFTDispersionInterface.filters(...)``.

    Wraps a cloned :class:`FFTDispersionInterface` carrying the requested
    filter configuration and provides the new fluent compute API.  All
    attributes not found on this class are transparently delegated to the
    wrapped interface (backwards compatibility).

    Attributes
    ----------
    _iface : FFTDispersionInterface
        The cloned interface carrying filter state.

    Examples
    --------
    >>> chain = job[0].m.fft.dispersion.filters(remove_static=True,
    ...     live={"gaussian_morph": {"enabled": True, "sigma_f": 1.0}})
    >>> result = chain.compute_1d(axis='x', save=True)
    >>> lowest = result.analyze.find_lowest_possible_frequency()
    >>> lowest.plot.heatmap()
    """

    def __init__(self, iface: FFTDispersionInterface) -> None:
        # Use object.__setattr__ to avoid triggering __setattr__ recursion
        object.__setattr__(self, "_iface", iface)

    # ------------------------------------------------------------------
    # primary compute methods
    # ------------------------------------------------------------------

    def compute_1d(
        self,
        axis: str = "x",
        *,
        save: bool = False,
        force: bool = False,
        component: str | None = None,
        avg_over_orthogonal: bool = True,
        orthogonal_avg_mode: str = "fft_power",
        **kwargs: Any,
    ) -> DispersionResult1D:
        """Compute the 1D dispersion relation S(k, f).

        Parameters
        ----------
        axis : ``"x"`` | ``"y"``
            Propagation direction.
        save : bool
            Cache the result inside the zarr store / cache directory.
        force : bool
            Recompute even when a cached result is available.
        component : str, optional
            Magnetization component to analyze (``None`` = auto).
        avg_over_orthogonal : bool
            Average over the direction orthogonal to *axis*.
        orthogonal_avg_mode : str
            Method for the orthogonal average when
            *avg_over_orthogonal* is ``True``.
        **kwargs
            Forwarded to the underlying :meth:`~FFTDispersionInterface.compute_1d`.

        Returns
        -------
        DispersionResult1D
            Result with ``.plot``, ``.analyze``, and ``.modes`` accessors.
        """
        iface = object.__getattribute__(self, "_iface")
        result: DispersionResult1D = iface.compute_1d(
            axis=axis,
            save=save,
            force=force,
            component=component,
            avg_over_orthogonal=avg_over_orthogonal,
            orthogonal_avg_mode=orthogonal_avg_mode,
            **kwargs,
        )
        # Attach back-reference so .modes.interactive() can delegate
        object.__setattr__(result, "_interface", iface)
        return result

    def compute_2d(
        self,
        *,
        save: bool = False,
        force: bool = False,
        component: str | None = None,
        **kwargs: Any,
    ) -> DispersionResult2D:
        """Compute the 2D dispersion relation S(kx, ky, f).

        Parameters
        ----------
        save, force
            Caching flags.
        component : str, optional
            Magnetization component.
        **kwargs
            Forwarded to :meth:`~FFTDispersionInterface.compute_2d`.
        """
        if save:
            raise NotImplementedError(
                "Saving/caching compute_2d results is not implemented. "
                "Use save=False and retain the returned DispersionResult2D."
            )
        if force:
            raise ValueError(
                "force=True has no effect when compute_2d caching is unavailable"
            )
        iface = object.__getattribute__(self, "_iface")
        return iface.compute_2d(
            component=component,
            **kwargs,
        )

    # callable shortcut: chain(axis='x') == chain.compute_1d(axis='x')
    def __call__(self, axis: str = "x", **kwargs: Any) -> DispersionResult1D:
        return self.compute_1d(axis=axis, **kwargs)

    # ------------------------------------------------------------------
    # re-filtering: chain more filters non-destructively
    # ------------------------------------------------------------------

    def filters(
        self,
        *,
        remove_static: bool = False,
        average: bool = False,
        window: str | Sequence[str] | None = None,
        pre: dict[str, Any] | None = None,
        post: dict[str, Any] | None = None,
        live: dict[str, Any] | None = None,
        advanced: dict[str, Any] | None = None,
    ) -> DispersionFilterChain:
        """Return a new :class:`DispersionFilterChain` with merged filter config.

        Lets you chain filter calls::

            chain = iface.filters(remove_static=True).filters(live={...})
        """
        iface = object.__getattribute__(self, "_iface")
        new_clone = iface._filters_impl(
            remove_static=remove_static,
            average=average,
            window=window,
            pre=pre,
            post=post,
            live=live,
            advanced=advanced,
        )
        # _filters_impl returns DispersionFilterChain already (after our patch)
        if isinstance(new_clone, DispersionFilterChain):
            return new_clone
        return DispersionFilterChain(new_clone)

    # ------------------------------------------------------------------
    # transparency – delegate everything else to wrapped interface
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        iface = object.__getattribute__(self, "_iface")
        return getattr(iface, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_iface":
            object.__setattr__(self, name, value)
        else:
            iface = object.__getattribute__(self, "_iface")
            setattr(iface, name, value)

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        iface = object.__getattribute__(self, "_iface")
        fc = getattr(iface, "_filters_config", None) or {}
        parts = []
        if fc.get("remove_static"):
            parts.append("remove_static=True")
        if fc.get("hann_time"):
            parts.append("window='time'")
        if fc.get("hann_space"):
            parts.append("window='space'")
        if fc.get("live"):
            parts.append(f"live={list(fc['live'])}")
        summary = ", ".join(parts) or "(no filters)"
        return f"<DispersionFilterChain [{summary}] → .compute_1d(axis='x') → DispersionResult1D>"

    def _repr_html_(self) -> str:
        iface = object.__getattribute__(self, "_iface")
        fc = getattr(iface, "_filters_config", None) or {}
        filters_keys = list(fc.keys()) if fc else []
        fc_summary = ", ".join(filters_keys) if filters_keys else "none"
        fc_detail = examples_section_html(
            str(fc) if fc else "{}",
            title=f"Active Filters: {fc_summary}",
        )
        api = api_help_html(
            self,
            title="Dispersion filter-chain API help",
            prefix="job[0].fft.dispersion.filters(...)",
            methods=["compute_1d", "compute_2d", "filters"],
            subtitle="Live signatures for the fluent dispersion filter-chain stage.",
            chrome=False,
        )
        return node_card_html(
            "Dispersion Filter Chain",
            icon="🧵",
            subtitle="Fluent filter configuration stage before dispersion computation.",
            sections=[
                metrics_section_html(
                    [
                        ("active filters", fc_summary, NODE_COLOR_UTIL),
                        ("result 1D", "compute_1d(...)", NODE_COLOR_COMPUTE),
                        ("result 2D", "compute_2d(...)", NODE_COLOR_ANALYSIS),
                    ]
                ),
                accessors_section_html(
                    [
                        (
                            "Compute:",
                            [
                                (
                                    ".compute_1d(axis='x', save=True)",
                                    NODE_COLOR_COMPUTE,
                                ),
                                (
                                    ".compute_1d(axis='x', avg_over_orthogonal=False)",
                                    NODE_COLOR_COMPUTE,
                                ),
                                (".compute_2d()", NODE_COLOR_ANALYSIS),
                            ],
                        ),
                        (
                            "Chain:",
                            [
                                (".filters(live={...})", NODE_COLOR_PLOT),
                                ("__call__(axis='x')", NODE_COLOR_UTIL),
                            ],
                        ),
                    ]
                ),
                fc_detail,
            ],
            api=api,
            uid="dispersion-filter-chain",
        )
