"""Plotting helpers for batch spectrum results."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def apply_folding(
    param_values: np.ndarray,
    sort_idx: np.ndarray,
    folding_period: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply folding to angular parameter values with mirroring."""
    sorted_params = param_values[sort_idx]
    param_min = sorted_params.min()
    param_max = sorted_params.max()
    param_range = param_max - param_min

    if param_range >= 0.95 * folding_period:
        return param_values, sort_idx

    n_replications = int(np.ceil(folding_period / param_range))
    folded_params = []
    folded_indices = []

    for i in range(n_replications):
        if i % 2 == 0:
            offset = i * param_range
            for val, idx in zip(sorted_params, sort_idx):
                new_val = val - param_min + offset
                if new_val < folding_period:
                    folded_params.append(new_val)
                    folded_indices.append(idx)
        else:
            offset = (i + 1) * param_range
            for val, idx in zip(sorted_params[::-1], sort_idx[::-1]):
                new_val = offset - (val - param_min)
                if new_val < folding_period:
                    folded_params.append(new_val)
                    folded_indices.append(idx)

    folded_params = np.array(folded_params)
    folded_indices = np.array(folded_indices)
    new_sort_idx = np.argsort(folded_params)
    return folded_params[new_sort_idx], folded_indices[new_sort_idx]


def plot_heatmap(
    result: Any,
    parameter: Optional[str] = None,
    ax: Optional[Any] = None,
    freq_unit: str = "GHz",
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    log_scale: bool = True,
    normalize: str = "per_row",
    cmap: str = "viridis",
    colorbar: bool = True,
    title: Optional[str] = None,
    folding: Optional[Union[float, str]] = None,
    verbose: bool = False,
    dpi: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> Tuple[Any, Any]:
    """Plot 2D heatmap of power spectrum vs parameter."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for plotting")

    if parameter is None:
        varying_params = []
        for param_name, values in result.parameters.items():
            unique_values = np.unique([v for v in values if v is not None])
            if len(unique_values) > 1:
                varying_params.append((param_name, len(unique_values)))

        if not varying_params:
            raise ValueError(
                "No varying parameters found! All extracted parameters have constant values.\n"
                f"Available parameters: {list(result.parameters.keys())}\n"
                "Hint: Check if parameters were correctly extracted during compute_all()"
            )

        varying_params.sort(key=lambda x: x[1], reverse=True)
        parameter = varying_params[0][0]

        if verbose:
            print(f"🔍 Auto-detected swapping parameter: '{parameter}'")
            print("\n📊 Available varying parameters:")
            for param_name, n_unique in varying_params:
                values = result.get_parameter_values(param_name)
                print(
                    f"   - {param_name}: {n_unique} unique values "
                    f"(range: {values.min():.3g} to {values.max():.3g})"
                )
            print(f"\nUsing '{parameter}' for heatmap Y-axis.")
            print("To use a different parameter, call: result.plot_heatmap(parameter='...')\n")

    freq_scales = {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12}
    freq_scale = freq_scales.get(freq_unit, 1e9)
    frequencies_scaled = result.frequencies / freq_scale

    freq_mask = np.ones(len(frequencies_scaled), dtype=bool)
    if fmin is not None:
        freq_mask &= frequencies_scaled >= fmin
    if fmax is not None:
        freq_mask &= frequencies_scaled <= fmax

    frequencies_display = frequencies_scaled[freq_mask]
    param_values = result.get_parameter_values(parameter)
    sort_idx = np.argsort(param_values)

    param_unit = ""
    angular_params = ["phi", "theta", "angle", "psi", "alpha", "beta", "gamma"]
    is_angular = parameter.lower() in angular_params

    if folding is not None and is_angular:
        if isinstance(folding, str) and folding.lower() == "auto":
            max_val = param_values.max()
            if max_val <= 7:
                folding_period = 2 * np.pi
                param_unit = " (rad)"
            else:
                folding_period = 360.0
                param_unit = " (°)"
        else:
            folding_period = float(folding)
            if folding_period <= 7:
                param_unit = " (rad)"
            else:
                param_unit = " (°)"

        param_values, sort_idx = apply_folding(param_values, sort_idx, folding_period)
    elif is_angular:
        max_val = param_values.max()
        if max_val <= 7:
            param_unit = " (rad)"
        else:
            param_unit = " (°)"

    data_matrix = []
    for idx in sort_idx:
        power = result.powers[idx][freq_mask]
        if power.ndim > 1:
            power = power.squeeze()
        data_matrix.append(power)

    data_matrix = np.array(data_matrix)
    param_sorted = (
        param_values if folding is not None and is_angular else param_values[sort_idx]
    )

    if data_matrix.ndim != 2:
        raise ValueError(
            f"Expected 2D data matrix, got shape {data_matrix.shape}. "
            "Power spectra should be 1D arrays."
        )

    if normalize == "per_row":
        row_max = np.max(data_matrix, axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        data_matrix = data_matrix / row_max
    elif normalize == "global":
        global_max = np.max(data_matrix)
        if global_max > 0:
            data_matrix = data_matrix / global_max

    if log_scale:
        data_matrix = np.log10(data_matrix + 1e-10)

    if ax is None:
        fig_kwargs = {"figsize": figsize if figsize is not None else (10, 6)}
        if dpi is not None:
            fig_kwargs["dpi"] = dpi
        fig, ax = plt.subplots(**fig_kwargs)
    else:
        fig = ax.figure

    extent = [
        param_sorted[0],
        param_sorted[-1],
        frequencies_display[0],
        frequencies_display[-1],
    ]

    im = ax.imshow(
        data_matrix.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap,
        **kwargs,
    )

    ax.set_xlabel(f"{parameter}{param_unit}")
    ax.set_ylabel(f"Frequency ({freq_unit})")

    if title:
        ax.set_title(title)

    if colorbar:
        label = "log₁₀(Power)" if log_scale else "Power"
        if normalize != "none":
            label += f" ({normalize})"
        cbar = fig.colorbar(im, ax=ax, label=label)
        cbar.outline.set_visible(False)

    return fig, ax


def replicate_experimental_points(
    angles: np.ndarray,
    fres: np.ndarray,
    fres_err: np.ndarray,
    folding: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicate experimental points to fill folding period with mirroring."""
    angle_min = angles.min()
    angle_max = angles.max()
    original_span = angle_max - angle_min
    n_copies = int(np.ceil(folding / original_span))

    angles_list = []
    fres_list = []
    fres_err_list = []

    for i in range(n_copies):
        if i % 2 == 0:
            new_angles = angles + i * original_span
        else:
            new_angles = 2 * (i * original_span + angle_min) - angles + original_span

        mask = (new_angles >= 0) & (new_angles < folding)
        if np.any(mask):
            angles_list.append(new_angles[mask])
            fres_list.append(fres[mask])
            fres_err_list.append(fres_err[mask])

    if len(angles_list) > 0:
        angles_rep = np.concatenate(angles_list)
        fres_rep = np.concatenate(fres_list)
        fres_err_rep = np.concatenate(fres_err_list)
    else:
        angles_rep = angles
        fres_rep = fres
        fres_err_rep = fres_err

    return angles_rep, fres_rep, fres_err_rep


def plot_experimental_data(
    result: Any,
    peaks: str,
    errors: str,
    shift: float = 0.0,
    target_field: Optional[float] = None,
    field_tolerance: float = 0.01,
    marker: str = "o",
    color: str = "cyan",
    s: float = 36,
    alpha: float = 1.0,
    error_color: Optional[str] = None,
    error_linewidth: float = 1.5,
    label: str = "Experimental",
    ax: Optional[Any] = None,
    **heatmap_kwargs,
) -> Tuple[Any, Any]:
    """Plot heatmap with experimental peak positions overlaid."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for plotting")

    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Pandas required for loading experimental data")

    peaks_df = pd.read_csv(peaks)
    errors_df = pd.read_csv(errors)

    exp_data = {
        "angles_deg": peaks_df["Angle (°)"].values,
        "angles_rad": peaks_df["phi (rad)"].values,
        "fields": peaks_df["Field (T)"].values,
        "fres": peaks_df["fres (GHz)"].values,
        "fres_err": errors_df["fres (GHz)"].values,
        "fwhm": peaks_df["FWHM (GHz)"].values,
    }

    if ax is None:
        fig, ax = plot_heatmap(result, **heatmap_kwargs)
    else:
        fig = ax.figure

    if target_field is not None:
        mask = np.abs(exp_data["fields"] - target_field) < field_tolerance
        angles_deg = exp_data["angles_deg"][mask]
        fres = exp_data["fres"][mask]
        fres_err = exp_data["fres_err"][mask]
        field_info = f" @ {target_field} T"
    else:
        angles_deg = exp_data["angles_deg"]
        fres = exp_data["fres"]
        fres_err = exp_data["fres_err"]
        field_info = ""

    angles_deg_shifted = np.where(angles_deg < 0, angles_deg + 360, angles_deg)
    angles_rad = np.deg2rad(angles_deg_shifted)
    angles = angles_rad + shift

    if "folding" in heatmap_kwargs:
        folding = heatmap_kwargs["folding"]
        if folding == "auto":
            varying = getattr(result, "varying_parameters", [])
            varying_param = varying[0] if varying else None
            if varying_param and varying_param["name"] in ["phi", "theta", "angle"]:
                values = varying_param["values"]
                if np.max(values) > 10:
                    folding = 360
                else:
                    folding = 2 * np.pi

        if isinstance(folding, (int, float)):
            angles = angles % folding
            angles, fres, fres_err = replicate_experimental_points(
                angles, fres, fres_err, folding
            )

    if error_color is None:
        error_color = color

    markersize = np.sqrt(s / np.pi)
    ax.errorbar(
        angles,
        fres,
        yerr=fres_err,
        fmt=marker,
        color=color,
        markersize=markersize,
        markeredgecolor="black",
        markeredgewidth=0.5,
        alpha=alpha,
        ecolor=error_color,
        elinewidth=error_linewidth,
        capsize=3,
        label=label + field_info,
        zorder=10,
    )
    ax.legend(loc="best", framealpha=0.9)
    return fig, ax


def overlay_experimental(
    result: Any,
    exp_frequencies: np.ndarray,
    exp_data: np.ndarray,
    parameter_value: Optional[float] = None,
    ax: Optional[Any] = None,
    label: str = "Experimental",
    color: str = "red",
    **plot_kwargs,
) -> Tuple[Any, Any]:
    """Overlay experimental data on spectrum plot."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("Matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if parameter_value is not None and result.parameters:
        param_name = None
        for name, values in result.parameters.items():
            if len(np.unique(values)) > 1:
                param_name = name
                break
        if param_name:
            param_values = result.get_parameter_values(param_name)
            idx = np.argmin(np.abs(param_values - parameter_value))
        else:
            idx = 0
    else:
        idx = 0
        param_name = None

    sim_freq = result.frequencies / 1e9
    sim_power = result.powers[idx]
    if sim_power.ndim > 1:
        sim_power = sim_power.squeeze()

    ax.plot(sim_freq, sim_power, label="Simulation", color="blue", linewidth=2)

    exp_freq_ghz = exp_frequencies / 1e9
    default_kwargs = {"linewidth": 1.5, "linestyle": "--", "alpha": 0.8}
    default_kwargs.update(plot_kwargs)
    ax.plot(exp_freq_ghz, exp_data, label=label, color=color, **default_kwargs)

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Power (a.u.)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if parameter_value is not None and param_name:
        ax.set_title(f"{param_name} = {parameter_value:.3f}")

    return fig, ax
