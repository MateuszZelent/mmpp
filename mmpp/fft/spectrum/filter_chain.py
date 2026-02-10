"""Fluent spectrum filter chain for ``fft.filters(...).spectrum()``."""

from __future__ import annotations

from typing import Any, Callable

from ..filters import normalize_filter_config, split_filter_stages


def _map_window_from_pre(pre_filters: dict[str, Any]) -> str | None:
    option = pre_filters.get("hann_time")
    if option is None:
        return None
    if isinstance(option, str):
        return option or None
    if isinstance(option, dict):
        value = option.get("window", "hann")
        return str(value) if value else None
    if bool(option):
        return "hann"
    return None


def _map_filter_type_from_pre(pre_filters: dict[str, Any]) -> str | list[str] | None:
    mapped: list[str] = []

    for key, option in pre_filters.items():
        if key == "hann_time":
            continue
        if key == "remove_average":
            mapped.append("remove_mean")
            continue
        if key == "detrend":
            mode = str(option.get("mode", "none")).lower() if isinstance(option, dict) else str(option).lower()
            if mode == "linear":
                mapped.append("detrend_linear")
            continue
        mapped.append(str(key))

    if not mapped:
        return None
    if len(mapped) == 1:
        return mapped[0]
    return mapped


def _post_to_result_kwargs(
    post_filters: dict[str, Any],
    live_filters: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(post_filters)
    merged.update(live_filters)
    kwargs: dict[str, Any] = {}

    if "normalize" in merged:
        kwargs["normalize"] = True
    if "log_transform" in merged:
        kwargs["log_scale"] = True

    gamma = merged.get("gamma")
    if isinstance(gamma, dict):
        kwargs["gamma"] = float(gamma.get("gamma", 1.0))
    elif gamma is not None:
        kwargs["gamma"] = float(gamma)

    percentile = merged.get("percentile_clip")
    if isinstance(percentile, dict):
        kwargs["percentile_clip"] = (
            float(percentile.get("low", 0.0)),
            float(percentile.get("high", 100.0)),
        )

    soft = merged.get("soft_threshold")
    if isinstance(soft, dict):
        kwargs["soft_threshold"] = float(soft.get("percentile", 0.0))
    elif soft is not None:
        kwargs["soft_threshold"] = float(soft)

    baseline = merged.get("baseline_correction")
    if isinstance(baseline, dict):
        kwargs["baseline"] = str(baseline.get("mode", "none"))
    elif baseline is not None:
        kwargs["baseline"] = str(baseline)

    smooth_order = (
        ("gaussian_smooth", "gaussian"),
        ("savgol_smooth", "savgol"),
        ("moving_average", "moving_average"),
    )
    for key, label in smooth_order:
        if key not in merged:
            continue
        option = merged[key]
        kwargs["smooth"] = label
        if isinstance(option, dict):
            kwargs["smooth_window"] = int(option.get("smooth_window", 7))
            kwargs["smooth_sigma"] = float(option.get("smooth_sigma", 1.0))
        break

    return kwargs


class SpectrumFilterChain:
    """Fluent chain applying pre/post FFT filters for spectrum workflows."""

    def __init__(self, spectrum_callable: Callable[..., Any], filters: dict[str, Any] | None = None):
        self._spectrum_callable = spectrum_callable
        self._filters = dict(filters or {})

    def filters(self, **filters: Any) -> SpectrumFilterChain:
        """Return a cloned chain with merged filter configuration."""
        merged = dict(self._filters)
        merged.update(filters)
        return SpectrumFilterChain(self._spectrum_callable, merged)

    def spectrum(self, *args: Any, **kwargs: Any) -> Any:
        """Compute spectrum with pre-filters and apply post/live filters."""
        normalized = normalize_filter_config(self._filters) or {}
        pre_filters, post_filters, live_filters = split_filter_stages(normalized)

        spectrum_kwargs = dict(kwargs)
        if "window" not in spectrum_kwargs:
            window = _map_window_from_pre(pre_filters)
            if window:
                spectrum_kwargs["window"] = window
        if "filter_type" not in spectrum_kwargs:
            filter_type = _map_filter_type_from_pre(pre_filters)
            if filter_type is not None:
                spectrum_kwargs["filter_type"] = filter_type

        result = self._spectrum_callable(*args, **spectrum_kwargs)
        if not (post_filters or live_filters) or not hasattr(result, "filtered"):
            return result

        post_kwargs = _post_to_result_kwargs(post_filters, live_filters)
        if not post_kwargs:
            return result
        return result.filtered(**post_kwargs)

    __call__ = spectrum

    def __repr__(self) -> str:
        return f"<SpectrumFilterChain filters={self._filters!r}>"

    def _repr_html_(self) -> str:
        from html import escape as _esc

        # Active filters
        if self._filters:
            filter_items = ", ".join(
                f"<code style='color:#a5b4fc;'>{_esc(str(k))}</code>"
                for k in self._filters
            )
            filters_display = filter_items
        else:
            filters_display = "<span style='color:#94a3b8;'>none configured</span>"

        methods = [
            (".filters(**kw)", "Clone chain with merged filter configuration"),
            (".spectrum(*args, **kw)", "Compute spectrum with pre-filters, apply post/live filters"),
        ]
        method_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(m)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(d)}</td></tr>"
            for m, d in methods
        )
        options = [
            ("remove_average", "False", "Pre: subtract temporal mean per spatial point"),
            ("hann_time", "None", "Pre: apply Hann window in time domain"),
            ("detrend", "None", "Pre: detrend strategy ('linear')"),
            ("normalize", "False", "Post: normalize spectrum amplitudes"),
            ("log_transform", "False", "Post: apply logarithmic transform"),
            ("gamma", "None", "Post: gamma correction (float or dict)"),
            ("percentile_clip", "None", "Post: clip to percentile range (low, high)"),
            ("soft_threshold", "None", "Post: soft threshold by percentile"),
            ("baseline_correction", "None", "Post: baseline removal ('none', 'median', ...)"),
            ("gaussian_smooth", "None", "Post: Gaussian smoothing (dict with sigma)"),
            ("savgol_smooth", "None", "Post: Savitzky-Golay smoothing (dict)"),
        ]
        opt_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(n)}</td>"
            f"<td style='padding:4px 8px;color:#a5b4fc;'>{_esc(d)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(desc)}</td></tr>"
            for n, d, desc in options
        )
        example = (
            "# Chain filters fluently\n"
            "result = fft.filters(\n"
            "    remove_average=True,\n"
            "    hann_time=True,\n"
            "    normalize=True,\n"
            ").spectrum(component='z')\n"
            "\n"
            "# Stack multiple filters\n"
            "chain = fft.filters(hann_time=True)\n"
            "chain = chain.filters(gamma={'gamma': 0.5})\n"
            "result = chain.spectrum()"
        )
        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);\">"
            "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;margin-bottom:4px;'>"
            "Spectrum Filter Chain</div>"
            "<div style='font-size:0.85em;color:#94a3b8;margin-bottom:10px;'>"
            "Fluent chain for pre/post FFT filter application via "
            "<code style='color:#a5b4fc;'>fft.filters(...).spectrum(...)</code></div>"
            # Active filters
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Active Filters</div>"
            f"<div style='font-size:0.9em;'>{filters_display}</div></div>"
            # Methods
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Methods</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            f"{method_rows}</table></div>"
            # Filter options
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Filter Options</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            "<thead><tr style='text-align:left;background:rgba(51,65,85,0.6);'>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Filter</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Default</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Description</th></tr></thead>"
            f"<tbody>{opt_rows}</tbody></table></div>"
            # Examples
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Examples</div>"
            "<pre style='margin:0;background:rgba(15,23,42,0.85);padding:10px;"
            "border-radius:6px;color:#e2e8f0;overflow-x:auto;font-size:0.85em;'>"
            f"<code>{example}</code></pre></div>"
            "</div>"
        )
