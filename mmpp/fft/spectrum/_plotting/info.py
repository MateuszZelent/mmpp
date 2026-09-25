"""Formatting and figure annotations for FFT computation provenance."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from typing import Any


def validate_info_option(info: str | None) -> None:
    """Validate the public plotting ``info`` option."""
    if info is not None and (not isinstance(info, str) or info != "full"):
        raise ValueError("info must be None or 'full'")


def _short_value(value: Any, limit: int = 140) -> str:
    rendered = str(value)
    if len(rendered) > limit:
        return rendered[: limit - 3] + "..."
    return rendered


def _unique_metadata_value(
    metadata: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    config: Mapping[str, Any],
    config_keys: Sequence[str] = (),
) -> Any:
    values = [
        entry[key] for entry in metadata for key in keys if entry.get(key) is not None
    ]
    if values:
        unique = list(dict.fromkeys(_short_value(value) for value in values))
        return unique[0] if len(unique) == 1 else "varies across plotted jobs"
    for key in config_keys or keys:
        if config.get(key) is not None:
            return config[key]
    return None


def _format_component(component: Any, config: Mapping[str, Any]) -> str | None:
    if component:
        return str(component)
    weights = config.get("component_weights")
    if weights is not None:
        return f"weighted components (weights={_short_value(weights)})"
    return None


def format_fft_info(
    *,
    metadata: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    source_paths: Sequence[str] = (),
    dataset: str | None = None,
    z_layer: Any = None,
    slice_info: Any = None,
    component: Any = None,
    config: Mapping[str, Any] | None = None,
) -> str:
    """Build a compact methodology caption from recorded FFT metadata."""
    if isinstance(metadata, Mapping):
        entries = [metadata]
    elif metadata is None:
        entries = []
    else:
        entries = [entry for entry in metadata if isinstance(entry, Mapping)]
    options = dict(config or {})
    paths = [str(path) for path in source_paths if path]
    lines: list[str] = []

    if paths:
        lines.append(f"Input files ({len(paths)}):")
        for path in paths[:8]:
            lines.append(f"  {path}")
        if len(paths) > 8:
            lines.append(f"  ... and {len(paths) - 8} more")
    else:
        lines.append("Input files: source path not recorded")

    actual_dataset = dataset or _unique_metadata_value(
        entries, ("source_dataset", "dataset"), options
    )
    actual_component = _format_component(component, options)
    requested_z_layer = (
        z_layer
        if z_layer is not None
        else _unique_metadata_value(entries, ("z_layer",), options)
    )
    resolved_z_layer = _unique_metadata_value(entries, ("resolved_z_layer",), {})
    actual_z_layer = resolved_z_layer or requested_z_layer
    selected_data = []
    if actual_dataset is not None:
        selected_data.append(f"dataset={actual_dataset}")
    if actual_component is not None:
        selected_data.append(f"component={actual_component}")
    else:
        selected_data.append("component=all retained components")
    if actual_z_layer is not None:
        if (
            resolved_z_layer is not None
            and requested_z_layer is not None
            and str(resolved_z_layer) != str(requested_z_layer)
        ):
            selected_data.append(
                f"z-layer={resolved_z_layer} (requested {requested_z_layer})"
            )
        elif str(actual_z_layer) == "-1":
            selected_data.append("z-layer=-1 (last layer; resolved index not retained)")
        else:
            selected_data.append(f"z-layer={actual_z_layer}")
    if slice_info is not None:
        selected_data.append(f"slice={_short_value(slice_info)}")
    lines.append("Selected input: " + "; ".join(selected_data))

    method = _unique_metadata_value(entries, ("method",), options)
    if method is None:
        method_line = "FFT method: method was not retained in result metadata"
    elif str(method) == "1":
        method_line = "FFT method 1: spatially average magnetization, then FFT"
    elif str(method) == "2":
        method_line = "FFT method 2: FFT each cell, then spatially average |FFT|²"
    else:
        method_line = f"FFT method: {method}"
    lines.append(method_line)

    sample_counts = []
    time_steps = []
    fft_lengths = []
    frequency_steps = []
    for entry in entries:
        shape = entry.get("data_shape")
        if isinstance(shape, (tuple, list)) and shape:
            sample_counts.append(int(shape[0]))
        dt = entry.get("dt")
        if dt is not None:
            time_steps.append(float(dt))
        fft_length = entry.get("fft_length")
        if fft_length is not None:
            fft_lengths.append(int(fft_length))
        df = entry.get("frequency_resolution")
        if df is not None:
            frequency_steps.append(float(df))

    sampling = []
    if sample_counts:
        unique_samples = list(dict.fromkeys(sample_counts))
        sampling.append(
            f"N={unique_samples[0]}"
            if len(unique_samples) == 1
            else "N varies across plotted jobs"
        )
    elif options.get("tmin") is not None or options.get("tmax") is not None:
        sampling.append(f"time range={options.get('tmin')}:{options.get('tmax')}")
    if time_steps:
        unique_dt = list(dict.fromkeys(time_steps))
        if len(unique_dt) == 1 and unique_dt[0] != 0:
            sampling.extend(
                (f"dt={unique_dt[0]:.6g} s", f"fs={1.0 / unique_dt[0]:.6g} Hz")
            )
        elif len(unique_dt) > 1:
            sampling.append("dt varies across plotted jobs")
    if fft_lengths:
        unique_nfft = list(dict.fromkeys(fft_lengths))
        sampling.append(
            f"FFT length={unique_nfft[0]}"
            if len(unique_nfft) == 1
            else "FFT length varies across plotted jobs"
        )
    if frequency_steps:
        unique_df = list(dict.fromkeys(frequency_steps))
        sampling.append(
            f"Δf={unique_df[0]:.6g} Hz"
            if len(unique_df) == 1
            else "frequency resolution varies across plotted jobs"
        )
    if sampling:
        lines.append("Sampling: " + "; ".join(sampling))

    settings = []
    for label, metadata_keys, config_keys in (
        ("window", ("window",), ("window", "window_function")),
        ("filter", ("filter_type",), ("filter_type",)),
        ("engine", ("engine_selected", "engine"), ("engine",)),
        ("scaling", ("scaling",), ("scaling",)),
        ("zero_padding", ("zero_padding",), ("zero_padding",)),
        ("resample_nonuniform", ("resample_nonuniform",), ("resample_nonuniform",)),
    ):
        value = _unique_metadata_value(entries, metadata_keys, options, config_keys)
        if value is not None:
            requested_engine = None
            if label == "engine":
                requested_engine = _unique_metadata_value(
                    entries, ("engine_requested",), options, ("engine",)
                )
            if requested_engine is not None and str(requested_engine) != str(value):
                settings.append(f"{label}={value} (requested {requested_engine})")
            elif label == "engine" and not entries:
                settings.append(f"engine={value} (resolved backend not retained)")
            else:
                settings.append(f"{label}={value}")
    requested_nfft = _unique_metadata_value(
        entries, ("nfft_requested",), options, ("nfft",)
    )
    if requested_nfft is not None:
        settings.append(f"requested nfft={requested_nfft}")
    if settings:
        lines.append("FFT settings: " + "; ".join(settings))

    if not entries and options:
        lines.append(
            "Sampling metadata: per-job sample count, dt, and resolved engine were not retained by this batch result"
        )
    elif not entries:
        lines.append("Detailed FFT settings are unavailable for this legacy result")

    return "\n".join(
        textwrap.fill(
            line,
            width=108,
            subsequent_indent="    ",
            break_long_words=False,
            break_on_hyphens=False,
        )
        for line in lines
    )


def spectrum_result_info(result: Any) -> str:
    """Format provenance for a single :class:`SpectrumResult`."""
    mode_context = getattr(result, "_mode_context", {}) or {}
    source_job = getattr(result, "_source_job", None)
    source_path = getattr(source_job, "path", None)
    if source_path is None:
        metadata_path = getattr(result, "compute_metadata", {}).get("zarr_path")
        source_path = metadata_path
    return format_fft_info(
        metadata=getattr(result, "compute_metadata", None),
        source_paths=[source_path] if source_path else (),
        dataset=mode_context.get("dset"),
        z_layer=mode_context.get("z_layer"),
        slice_info=mode_context.get("slice_info"),
        component=getattr(result, "component_label", None),
    )


def batch_spectrum_info(result: Any) -> str:
    """Format retained provenance for a batch spectrum result."""
    component = None
    weights = getattr(result, "config_dict", {}).get("component_weights")
    if weights is not None:
        component = f"weighted components (weights={weights})"
    return format_fft_info(
        source_paths=getattr(result, "job_paths", ()),
        dataset=getattr(result, "dataset_name", None),
        z_layer=getattr(result, "z_layer", None),
        component=component,
        config=getattr(result, "config_dict", None),
    )


def add_fft_info(figure: Any, text: str) -> None:
    """Place a full-width monospace provenance caption below a Matplotlib figure."""
    for existing in figure.texts:
        if existing.get_gid() == "mmpp-fft-info":
            existing.set_text(text)
            return

    fontsize = 7.5
    line_count = len(text.splitlines())
    figure_height = max(float(figure.get_size_inches()[1]), 1.0)
    reserved_bottom = min(
        0.56,
        max(0.18, (line_count * fontsize * 1.35 / 72.0 + 0.06) / figure_height),
    )
    layout_engine = figure.get_layout_engine()
    if layout_engine is None:
        if figure.subplotpars.bottom < reserved_bottom:
            figure.subplots_adjust(bottom=reserved_bottom)
    else:
        figure.canvas.draw()
        axes_positions = [(axis, axis.get_position()) for axis in figure.axes]
        # ``None`` re-applies Matplotlib's global default, which can re-enable
        # constrained layout and fail when this figure already has colorbars.
        # ``none`` disables layout while preserving compatibility with the
        # active engine.
        figure.set_layout_engine("none")
        available_height = 1.0 - reserved_bottom
        for axis, position in axes_positions:
            axis.set_position(
                [
                    position.x0,
                    reserved_bottom + position.y0 * available_height,
                    position.width,
                    position.height * available_height,
                ]
            )

    annotation = figure.text(
        0.01,
        0.012,
        text,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        family="monospace",
        linespacing=1.2,
    )
    annotation.set_gid("mmpp-fft-info")


__all__ = [
    "add_fft_info",
    "batch_spectrum_info",
    "format_fft_info",
    "spectrum_result_info",
    "validate_info_option",
]
