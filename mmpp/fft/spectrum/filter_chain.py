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
        return (
            "<div style='font-family:sans-serif;border:1px solid #475569;border-radius:8px;"
            "padding:12px;background:#1e293b;color:#e2e8f0;'>"
            "<b>SpectrumFilterChain</b><br/>"
            "<code>fft.filters(...).spectrum(...)</code> - fluent filtered spectrum computation"
            "</div>"
        )
