"""Sweep parameter discovery and faceted batch-spectrum plotting."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .._plotting.info import add_fft_info, batch_spectrum_info, validate_info_option

_PATH_PARAMETER = re.compile(
    r"^(?P<name>[A-Za-z][A-Za-z0-9_]*)_"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"
)
_VOLATILE_PARAMETERS = {
    "start_time",
    "end_time",
    "total_time",
    "runtime",
    "elapsed_time",
    "steps",
}
_LEGACY_PARAMETER_NAMES = (
    "B0",
    "Bext",
    "bex",
    "bias_field",
    "applied_field",
    "d",
    "p",
    "thickness",
    "period",
    "latticeconst",
    "phi",
    "theta",
    "angle",
)


def _scalar(value: Any) -> Any:
    """Return a Python scalar for common NumPy-backed metadata values."""
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return None


def _path_parameters(path: str | Path) -> dict[str, float]:
    """Read numeric ``name_value`` components from a result path."""
    values: dict[str, float] = {}
    for component in Path(path).parts:
        if component.endswith(".zarr"):
            component = component[: -len(".zarr")]
        match = _PATH_PARAMETER.fullmatch(component)
        if match is None:
            continue
        try:
            values[match.group("name")] = float(match.group("value"))
        except ValueError:
            continue
    return values


def _declared_parameter_names(mmpp_ref: Any) -> list[str]:
    """Read parameter names declared by a sweep metadata JSON, when present."""
    root_value = getattr(mmpp_ref, "base_path", None)
    if not root_value:
        return []
    root = Path(root_value)
    candidates = [root / "metadata.json", *sorted(root.glob("*_metadata.json"))]
    for metadata_path in candidates:
        if not metadata_path.is_file():
            continue
        try:
            with metadata_path.open(encoding="utf-8") as stream:
                metadata = json.load(stream)
        except (OSError, json.JSONDecodeError):
            continue
        ranges = (
            metadata.get("parameter_ranges") if isinstance(metadata, dict) else None
        )
        if isinstance(ranges, dict):
            return [str(name) for name in ranges]
    return []


def get_sweep_parameter_value(result: Any, name: str) -> Any:
    """Get a parameter from result attributes or its conventional path component."""
    attributes = getattr(result, "attributes", None)
    if isinstance(attributes, Mapping):
        if name in attributes:
            value = _scalar(attributes[name])
            if value is not None:
                return value
        folded_name = name.casefold()
        for key, raw_value in attributes.items():
            if str(key).casefold() == folded_name:
                value = _scalar(raw_value)
                if value is not None:
                    return value

    path_value = getattr(result, "path", None)
    if path_value:
        path_values = _path_parameters(path_value)
        if name in path_values:
            return path_values[name]
        folded_name = name.casefold()
        for key, value in path_values.items():
            if key.casefold() == folded_name:
                return value
    return None


def _unique_count(values: Sequence[Any]) -> int:
    unique: list[Any] = []
    for raw_value in values:
        value = _scalar(raw_value)
        if value is None:
            continue
        try:
            is_new = not any(value == existing for existing in unique)
        except Exception:
            is_new = not any(repr(value) == repr(existing) for existing in unique)
        if is_new:
            unique.append(value)
    return len(unique)


def _varying_attribute_names(results: Sequence[Any]) -> list[str]:
    """Infer axes from shared scalar attributes if no sweep declaration exists."""
    if not results:
        return []
    first_attributes = getattr(results[0], "attributes", None)
    if not isinstance(first_attributes, Mapping):
        return []

    names: list[str] = []
    for raw_name in first_attributes:
        name = str(raw_name)
        if name.casefold() in _VOLATILE_PARAMETERS:
            continue
        values = [get_sweep_parameter_value(result, name) for result in results]
        if all(value is not None for value in values) and _unique_count(values) > 1:
            names.append(name)
    return names


def discover_sweep_parameters(results: Sequence[Any], mmpp_ref: Any) -> list[str]:
    """Discover sweep axes from project metadata, result paths, or attributes.

    Project-level ``*_metadata.json`` declarations take precedence. If none is
    available, shared numeric ``name_value`` path components are used. The last
    fallback considers varying scalar result attributes and excludes common
    per-run timing fields.
    """
    declared = _declared_parameter_names(mmpp_ref)
    if declared:
        return list(dict.fromkeys(declared))

    path_values = [_path_parameters(getattr(result, "path", "")) for result in results]
    if path_values:
        shared_names = list(path_values[0])
        for values in path_values[1:]:
            shared_names = [name for name in shared_names if name in values]
        varying_paths = [
            name
            for name in shared_names
            if _unique_count([values[name] for values in path_values]) > 1
        ]
        if varying_paths:
            return varying_paths

    inferred = _varying_attribute_names(results)
    return inferred or list(_LEGACY_PARAMETER_NAMES)


def _numeric_values(result: Any, parameter: str) -> np.ndarray:
    """Get finite numeric values for a plot axis or raise a useful error."""
    raw_values = result.parameters.get(parameter)
    if raw_values is None or len(raw_values) != len(result):
        raise ValueError(
            f"Parameter {parameter!r} is unavailable or does not align with spectra"
        )
    try:
        values = np.asarray(raw_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Parameter {parameter!r} must contain numeric values to plot a sweep"
        ) from exc
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"Parameter {parameter!r} must contain finite scalar values")
    return values


def _sort_value(value: Any) -> tuple[int, Any]:
    """Stable mixed-type ordering for facet labels."""
    value = _scalar(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, float(value))
    return (1, str(value))


def _format_value(value: Any) -> str:
    value = _scalar(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:g}"
    return str(value)


def plot_sweeps(
    result: Any,
    parameters: Sequence[str] | str | None = None,
    *,
    freq_unit: str = "GHz",
    fmin: float | None = None,
    fmax: float | None = None,
    log_scale: bool = True,
    normalize: str = "per_row",
    cmap: str = "viridis",
    colorbar: bool = True,
    max_columns: int = 4,
    figsize: tuple[float, float] | None = None,
    info: str | None = None,
    **kwargs: Any,
) -> dict[str, tuple[Any, np.ndarray]]:
    """Plot one frequency-versus-parameter map per sweep axis.

    Every other varying parameter becomes a facet, so spectra from different
    sweep coordinates are never mixed in the same panel. Repeated runs at the
    same parameter coordinates are averaged in power.
    """
    from .plotting import plot_heatmap
    from .result import BatchSpectrumResult

    validate_info_option(info)

    if isinstance(max_columns, bool) or not isinstance(max_columns, int):
        raise TypeError("max_columns must be a positive integer")
    if max_columns <= 0:
        raise ValueError("max_columns must be a positive integer")

    varying_parameters = []
    normalized_values: dict[str, np.ndarray] = {}
    for name, raw_values in result.parameters.items():
        if len(raw_values) != len(result) or any(value is None for value in raw_values):
            continue
        scalar_values = [_scalar(value) for value in raw_values]
        if any(value is None for value in scalar_values):
            continue
        if _unique_count(scalar_values) <= 1:
            continue
        varying_parameters.append(name)
        try:
            values = np.asarray(scalar_values, dtype=float)
        except (TypeError, ValueError):
            continue
        if values.ndim != 1 or not np.isfinite(values).all():
            continue
        normalized_values[name] = values

    if parameters is None:
        selected_parameters = [
            name for name in varying_parameters if name in normalized_values
        ]
    elif isinstance(parameters, str):
        selected_parameters = [parameters]
    else:
        selected_parameters = list(parameters)
    if not selected_parameters:
        raise ValueError("No varying numeric sweep parameters are available to plot")

    unknown = [name for name in selected_parameters if name not in result.parameters]
    if unknown:
        raise KeyError(
            f"Unknown sweep parameter(s) {unknown}; "
            f"available: {list(result.parameters)}"
        )
    for name in selected_parameters:
        if name not in normalized_values:
            _numeric_values(result, name)
            raise ValueError(
                f"Parameter {name!r} must be numeric and vary across the batch"
            )

    plots: dict[str, tuple[Any, np.ndarray]] = {}
    for parameter in selected_parameters:
        axis_values = normalized_values[parameter]
        facet_parameters = [name for name in varying_parameters if name != parameter]

        grouped_indices: dict[tuple[Any, ...], list[int]] = {}
        for index in range(len(result)):
            key = tuple(
                _scalar(result.parameters[name][index]) for name in facet_parameters
            )
            grouped_indices.setdefault(key, []).append(index)
        group_keys = sorted(
            grouped_indices,
            key=lambda key: tuple(_sort_value(value) for value in key),
        )

        columns = min(max_columns, len(group_keys))
        rows = math.ceil(len(group_keys) / columns)
        if figsize is None:
            current_figsize = (5.0 * columns, 3.6 * rows)
        else:
            current_figsize = figsize
        import matplotlib.pyplot as plt

        # Keep figure creation independent of rc settings left behind by other
        # notebook plots. This routine adds colorbars and then applies
        # tight_layout(), which cannot replace an active constrained-layout
        # engine once colorbars exist (Matplotlib 3.10+).
        with plt.rc_context(
            {
                "figure.autolayout": False,
                "figure.constrained_layout.use": False,
            }
        ):
            figure, axes = plt.subplots(
                rows,
                columns,
                squeeze=False,
                figsize=current_figsize,
            )
        flat_axes = axes.ravel()

        for panel_index, group_key in enumerate(group_keys):
            indices = grouped_indices[group_key]
            indices = sorted(indices, key=lambda index: axis_values[index])
            grouped_by_value: dict[float, list[int]] = {}
            for index in indices:
                grouped_by_value.setdefault(float(axis_values[index]), []).append(index)
            sweep_values = sorted(grouped_by_value)

            powers: list[np.ndarray] = []
            spectra: list[np.ndarray] = []
            representative_indices: list[int] = []
            for sweep_value in sweep_values:
                repeated_indices = grouped_by_value[sweep_value]
                powers.append(
                    np.mean(
                        np.stack([result.powers[index] for index in repeated_indices]),
                        axis=0,
                    )
                )
                spectra.append(
                    np.mean(
                        np.stack([result.spectra[index] for index in repeated_indices]),
                        axis=0,
                    )
                )
                representative_indices.append(repeated_indices[0])

            subset_parameters = {
                name: [
                    result.parameters[name][index] for index in representative_indices
                ]
                for name in result.parameters
            }
            subset_parameters[parameter] = sweep_values
            subset = BatchSpectrumResult(
                frequencies=result.frequencies,
                spectra=spectra,
                powers=powers,
                parameters=subset_parameters,
                job_paths=[result.job_paths[index] for index in representative_indices],
                config_dict=result.config_dict,
                dataset_name=result.dataset_name,
                z_layer=result.z_layer,
            )
            facet_title = ", ".join(
                f"{name}={_format_value(value)}"
                for name, value in zip(facet_parameters, group_key, strict=False)
            )
            plot_heatmap(
                subset,
                parameter=parameter,
                ax=flat_axes[panel_index],
                freq_unit=freq_unit,
                fmin=fmin,
                fmax=fmax,
                log_scale=log_scale,
                normalize=normalize,
                cmap=cmap,
                colorbar=colorbar,
                title=facet_title or parameter,
                **kwargs,
            )

        for unused_axis in flat_axes[len(group_keys) :]:
            unused_axis.set_visible(False)
        figure.suptitle(f"Spectrum across {parameter}")
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        if info == "full":
            add_fft_info(figure, batch_spectrum_info(result))
        plots[parameter] = (figure, axes)

    return plots


__all__ = [
    "discover_sweep_parameters",
    "get_sweep_parameter_value",
    "plot_sweeps",
]
