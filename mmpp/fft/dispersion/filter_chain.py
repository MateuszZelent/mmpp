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

from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

if TYPE_CHECKING:
    from ..models import DispersionResult1D, DispersionResult2D
    from ..interface import FFTDispersionInterface


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

    def __init__(self, iface: "FFTDispersionInterface") -> None:
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
        component: Optional[str] = None,
        avg_over_orthogonal: bool = True,
        orthogonal_avg_mode: str = "fft_power",
        **kwargs: Any,
    ) -> "DispersionResult1D":
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
        result: "DispersionResult1D" = iface.compute_1d(
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
        component: Optional[str] = None,
        **kwargs: Any,
    ) -> "DispersionResult2D":
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
        iface = object.__getattribute__(self, "_iface")
        return iface.compute_2d(
            save=save,
            force=force,
            component=component,
            **kwargs,
        )

    # callable shortcut: chain(axis='x') == chain.compute_1d(axis='x')
    def __call__(self, axis: str = "x", **kwargs: Any) -> "DispersionResult1D":
        return self.compute_1d(axis=axis, **kwargs)

    # ------------------------------------------------------------------
    # re-filtering: chain more filters non-destructively
    # ------------------------------------------------------------------

    def filters(
        self,
        *,
        remove_static: bool = False,
        average: bool = False,
        window: Optional[Union[str, Sequence[str]]] = None,
        pre: Optional[dict[str, Any]] = None,
        post: Optional[dict[str, Any]] = None,
        live: Optional[dict[str, Any]] = None,
        advanced: Optional[dict[str, Any]] = None,
    ) -> "DispersionFilterChain":
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
        from html import escape as _esc

        iface = object.__getattribute__(self, "_iface")
        fc = getattr(iface, "_filters_config", None) or {}

        HV = "onmouseover=\"this.style.background='#1e293b'\" onmouseout=\"this.style.background='transparent'\""

        methods = [
            (".compute_1d(axis='x', save=True)",
             "→ DispersionResult1D",
             "Compute S(k,f) along propagation axis. Returns DispersionResult1D with .plot, .analyze, .modes. "
             "save=True caches to zarr; force=True bypasses cache."),
            (".compute_1d(axis='x', avg_over_orthogonal=False)",
             "→ DispersionResult1D with S_local",
             "avg_over_orthogonal=False preserves all y-slices in S_local — required for mode inspection per y position."),
            (".compute_2d(save=True)",
             "→ DispersionResult2D",
             "Compute full 2D dispersion S(kx, ky, f)."),
            (".filters(live={...})",
             "→ new DispersionFilterChain",
             "Chain additional filters non-destructively. Uses deepcopy — original chain is unchanged."),
        ]
        row_html = "".join(
            f"<tr {HV} title=\"{_esc(tip)}\" style='cursor:pointer;'>"
            f"<td style='padding:4px 10px;font-family:monospace;color:#93c5fd;font-size:.88em;width:55%;'>{_esc(sig)}</td>"
            f"<td style='padding:4px 10px;color:#94a3b8;font-size:.85em;'>{_esc(desc)}</td>"
            f"</tr>"
            for sig, desc, tip in methods
        )

        # Render active filters as expandable section
        filters_keys = list(fc.keys()) if fc else []
        fc_summary = ", ".join(filters_keys) if filters_keys else "none"
        fc_detail = (
            "<details style='margin:6px 0;'>"
            "<summary style='cursor:pointer;font-size:.8em;color:#94a3b8;list-style:none;' "
            f"title='Expand to see full filter configuration'>"
            f"&#9654; active filters: <code style='color:#a5b4fc;'>{_esc(fc_summary)}</code></summary>"
            "<pre style='margin:4px 0 0 12px;font-size:.78em;color:#a5b4fc;"
            "background:#1e293b;padding:6px;border-radius:6px;'>"
            + _esc(str(fc) if fc else "{}")
            + "</pre></details>"
        )

        breadcrumb = (
            "<div style='font-size:.78em;color:#475569;margin-bottom:8px;font-family:monospace;'>"
            "fft.dispersion "
            "<span style='color:#334155;'>›</span> "
            "<span style='color:#7dd3fc;font-weight:600;'>.filters()</span>"
            "</div>"
        )

        return (
            "<div style='font-family:-apple-system,sans-serif;border:2px solid #334155;"
            "border-radius:10px;padding:12px;margin:6px 0;"
            "background:linear-gradient(135deg,#0f172a,#1e293b);"
            "color:#e2e8f0;max-width:680px;'>"
            + breadcrumb
            + "<div style='font-weight:700;color:#7dd3fc;margin-bottom:6px;'>"
            + "DispersionFilterChain"
            + "<span style='font-size:.75em;color:#475569;font-weight:400;margin-left:8px;'>"
            + "(hover rows for parameter details)</span></div>"
            + fc_detail
            + f"<table style='margin-top:4px;width:100%;border-collapse:collapse;'>{row_html}</table>"
            + "</div>"
        )
