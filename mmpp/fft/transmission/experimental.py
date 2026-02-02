"""Utilities for overlaying experimental transmission measurements."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

try:  # pragma: no cover - matplotlib is an optional dependency at runtime
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from ..compute import TransmissionResult
except ImportError:  # pragma: no cover
    Axes = Any  # type: ignore
    Line2D = Any  # type: ignore

_FREQUENCY_UNITS = {
    "Hz": 1.0,
    "kHz": 1e3,
    "MHz": 1e6,
    "GHz": 1e9,
}

_STYLE_LOADED = False


def _load_mmpp_style(verbose: bool = False) -> bool:
    """Load mmpp paper.mplstyle if available.
    
    Parameters
    ----------
    verbose : bool
        Whether to print informational messages.
    
    Returns
    -------
    bool
        True if style was successfully loaded, False otherwise.
    """
    global _STYLE_LOADED
    
    if _STYLE_LOADED:
        return True
    
    try:
        import matplotlib.pyplot as plt
        
        # Try to load from package directory
        package_dir = os.path.dirname(os.path.dirname(__file__))
        style_path = os.path.join(package_dir, "paper.mplstyle")
        
        if os.path.exists(style_path):
            plt.style.use(style_path)
            if verbose:
                print(f"✓ Loaded mmpp paper style from: {style_path}")
            _STYLE_LOADED = True
            return True
        else:
            # Fallback paths
            fallback_paths = [
                "./paper.mplstyle",
                os.path.expanduser("~/.mmpp/paper.mplstyle"),
            ]
            
            for path in fallback_paths:
                if os.path.exists(path):
                    plt.style.use(path)
                    if verbose:
                        print(f"✓ Loaded mmpp paper style from: {path}")
                    _STYLE_LOADED = True
                    return True
        
        if verbose:
            print(f"⚠ paper.mplstyle not found, using default matplotlib style")
        _STYLE_LOADED = True
        return False
        
    except Exception as e:
        if verbose:
            print(f"Warning: Could not load paper style: {e}")
        _STYLE_LOADED = True
        return False


def _resolve_frequency_unit(unit: str) -> float:
    try:
        return _FREQUENCY_UNITS[unit]
    except KeyError as exc:  # pragma: no cover - defensive branch
        available = ", ".join(sorted(_FREQUENCY_UNITS))
        raise ValueError(
            f"Unsupported frequency unit '{unit}'. Available units: {available}."
        ) from exc


def _resolve_data_column(
    dataframe: pd.DataFrame,
    *,
    column: Union[int, str, None] = None,
    bias_index: Optional[float] = None,
) -> str:
    """Select a column from a measurement DataFrame.

    Parameters
    ----------
    dataframe:
        The S-parameter measurement data with one column per bias point.
    column:
        Column index or label to select. When ``None`` the column is inferred
        either from ``bias_index`` or, as a fallback, the last numeric column.
    bias_index:
        Optional magnetic bias value. When provided the function looks for a
        column labelled ``2 * bias_index - 1`` (mirrors the experimental
        spreadsheet convention).
    """

    if column is not None:
        if isinstance(column, int):
            try:
                return dataframe.columns[column]
            except IndexError as exc:
                raise ValueError(
                    f"Column index {column} is out of range for experimental data."
                ) from exc
        if column not in dataframe.columns:
            raise ValueError(
                f"Column '{column}' not found in experimental data. Available columns: {list(dataframe.columns)!r}"
            )
        return column

    if bias_index is not None:
        label = f"{int(2 * bias_index) - 1}"
        if label in dataframe.columns:
            return label
        # Fall back to string without decimal even for integer-like floats
        label = f"{int(bias_index)}"
        if label in dataframe.columns:
            return label
        raise ValueError(
            f"Could not resolve column for bias_index={bias_index}. Available labels: {list(dataframe.columns)!r}"
        )

    numeric_columns = [col for col in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[col])]
    if numeric_columns:
        return numeric_columns[-1]
    # fallback to the final column regardless of dtype
    return dataframe.columns[-1]


def overlay_transmission(
    ax: Axes,
    *,
    d: Union[int, str],
    p: Union[int, str],
    base_path: Union[str, Path] = "experiment",
    width_tag: str = "w5",
    sim_result: Optional[TransmissionResult] = None,
    normalize_to: Optional[str] = None,
    normalize: bool = False,
    freq_filename: str = "freq.txt",
    freq_file_unit: str = "GHz",
    target_freq_unit: Optional[str] = None,
    bias_index: Optional[float] = None,
    column: Union[int, str, None] = None,
    reverse_frequency: bool = True,
    flip: bool = True,
    color: str = "tab:red",
    linewidth: float = 0.8,
    label: Optional[str] = None,
    **plot_kwargs: Any,
) -> Line2D:
    """Overlay an experimental transmission trace on a plot.

    Parameters
    ----------
    ax:
        Matplotlib axes returned by :meth:`TransmissionResult.plot_transmission_crosssection`.
    d, p:
        Thickness ``d`` and period ``p`` values used to locate the experimental file.
        **Special case**: If ``d=0`` and ``p=0``, loads reference data from ``ref.txt``
        instead of the standard ``d{d}p{p}_{width_tag}.txt`` file.
    base_path:
        Directory that contains the experimental spectra (default ``"experiment"``).
    width_tag:
        Suffix that distinguishes measurement geometry (default ``"w5"``).
    sim_result:
        Optional `TransmissionResult` object containing simulation data.
        Required when using `normalize_to`.
    normalize_to:
        Normalization strategy. If ``"sim_max"``, scales experimental data to
        match the maximum of the corresponding simulation cross-section.
    freq_filename:
        File with the experimental frequency vector (default ``"freq.txt"``).
    freq_file_unit:
        Unit stored in ``freq_filename`` (``"Hz"``, ``"kHz"``, ``"MHz"`` or ``"GHz"``).
    target_freq_unit:
        Desired unit for plotting. When ``None`` the original unit is used.
    bias_index:
        Optional magnetic bias index used to infer the measurement column.
    column:
        Explicit column index or label from the measurement file. Overrides ``bias_index``.
    reverse_frequency:
        Reverse the frequency axis to match typical experimental formatting.
    flip:
        If True (default), plot frequency on the Y-axis and transmission on the X-axis
        to match the historical overlay behaviour. Set to False to mirror the default
        orientation of :meth:`TransmissionResult.plot_transmission_crosssection`
        (frequency on X-axis, transmission on Y-axis).
    normalize:
        If True, normalizes the experimental transmission so that its maximum value is 1.
        This normalization is applied independently after any `normalize_to` scaling.
        Default is False.
    color, linewidth, label, plot_kwargs:
        Forwarded to :func:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.lines.Line2D
        Handle of the added experimental trace.
    """

    if target_freq_unit is None:
        target_freq_unit = freq_file_unit

    base_path = Path(base_path)
    
    # Special case: d=0 and p=0 means load reference file
    if d == 0 and p == 0:
        spectra_path = base_path / "ref.txt"
    else:
        spectra_path = base_path / f"d{d}p{p}_{width_tag}.txt"
    
    freq_path = base_path / freq_filename

    if not spectra_path.exists():
        raise FileNotFoundError(
            f"Experimental spectrum '{spectra_path}' not found."
        )
    if not freq_path.exists():
        raise FileNotFoundError(
            f"Experimental frequency file '{freq_path}' not found."
        )

    spectrum_df = pd.read_csv(spectra_path, sep="\t")
    selected_column = _resolve_data_column(
        spectrum_df,
        column=column,
        bias_index=bias_index,
    )
    amplitudes = spectrum_df[selected_column].to_numpy(dtype=float)

    # --- Intelligent Normalization ---
    if normalize_to:
        if normalize_to.lower() == "sim_max":
            if sim_result is None:
                raise ValueError(
                    "'sim_result' object must be provided to use 'normalize_to' feature."
                )

            # Find the corresponding simulation cross-section from the axes' lines.
            # This is robust, as it finds what is already plotted on the target axes.
            sim_line = next((line for line in ax.get_lines() if line.get_label() != label), None)
            if sim_line is None:
                raise RuntimeError(
                    "Could not find a simulation line on the provided axes to normalize against."
                )

            # Select simulation transmission data based on orientation
            sim_transmission_data = (
                sim_line.get_xdata() if flip else sim_line.get_ydata()
            )
            sim_max = np.max(sim_transmission_data)

            # Scale experimental data
            exp_max = np.max(amplitudes)
            if exp_max > 0:
                scale_factor = sim_max / exp_max
                amplitudes *= scale_factor
        else:
            raise ValueError(
                f"Unsupported normalization strategy: '{normalize_to}'. Available: 'sim_max'."
            )

    # Apply independent normalization if requested
    if normalize:
        exp_max = np.max(amplitudes)
        if exp_max > 0:
            amplitudes = amplitudes / exp_max

    # Load frequency data, allowing pandas to auto-detect the header.
    # This is more robust if the file contains a header like 'freq (GHz)'.
    freq_df = pd.read_csv(freq_path)
    if freq_df.shape[1] != 1:
        raise ValueError(f"Frequency file '{freq_path}' should contain exactly one column.")
    # Use the values from the first column, regardless of its name.
    frequencies = freq_df.iloc[:, 0].to_numpy(dtype=float)

    if reverse_frequency:
        frequencies = frequencies[::-1]

    from_scale = _resolve_frequency_unit(freq_file_unit)
    to_scale = _resolve_frequency_unit(target_freq_unit)
    frequencies = frequencies * from_scale / to_scale

    if amplitudes.shape[0] != frequencies.shape[0]:
        raise ValueError(
            "Experimental amplitude and frequency arrays have different lengths: "
            f"{amplitudes.shape[0]} vs {frequencies.shape[0]}."
        )

    if label is None:
        label = f"exp d={d} p={p}"

    if flip:
        plot_x, plot_y = amplitudes, frequencies
    else:
        plot_x, plot_y = frequencies, amplitudes

    line, = ax.plot(
        plot_x,
        plot_y,
        color=color,
        linewidth=linewidth,
        label=label,
        **plot_kwargs,
    )
    return line


def _extract_bias_values_from_columns(columns: list[str]) -> list[tuple[str, Optional[float]]]:
    """Extract bias values from numeric column headers.
    
    Parameters
    ----------
    columns : list[str]
        Column names from the experimental data.
    
    Returns
    -------
    list[tuple[str, Optional[float]]]
        List of tuples (column_name, bias_value). Bias value is None if the column
        header is not numeric or cannot be converted using the formula.
        The formula used is: bias_value = (header_value + 1) / 2 (inverse of 2*B - 1)
    """
    bias_values = []
    for col in columns:
        try:
            # Try to convert column name to float
            header_val = float(col)
            # Reverse the formula: if header = 2*B - 1, then B = (header + 1) / 2
            bias_val = (header_val + 1) / 2
            bias_values.append((col, bias_val))
        except (ValueError, TypeError):
            # Column header is not numeric
            bias_values.append((col, None))
    
    return bias_values


def load_experimental_transmission_data(
    *,
    d: Union[int, str],
    p: Union[int, str],
    base_path: Union[str, Path] = "experiment",
    width_tag: str = "w5",
    freq_filename: str = "freq.txt",
    freq_file_unit: str = "GHz",
    target_freq_unit: Optional[str] = None,
    reverse_frequency: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Load experimental transmission data and frequency information.
    
    Parameters
    ----------
    d, p:
        Thickness and period values to locate the experimental file.
    base_path:
        Directory containing the experimental spectra.
    width_tag:
        Suffix that distinguishes measurement geometry (default "w5").
    freq_filename:
        File with the experimental frequency vector (default "freq.txt").
    freq_file_unit:
        Unit stored in freq_filename ("Hz", "kHz", "MHz" or "GHz").
    target_freq_unit:
        Desired unit for plotting. When None, the original unit is used.
    reverse_frequency:
        Whether to reverse the frequency axis.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[float]]
        - frequencies: 1D array of frequency values (MHz, GHz, etc.)
        - transmission_data: 2D array of shape (n_frequencies, n_bias_points)
        - bias_values: 1D list of magnetic field values for each column
    """
    if target_freq_unit is None:
        target_freq_unit = freq_file_unit
    
    base_path = Path(base_path)
    
    # Construct spectra file path
    if d == 0 and p == 0:
        spectra_path = base_path / "ref.txt"
    else:
        spectra_path = base_path / f"d{d}p{p}_{width_tag}.txt"
    
    freq_path = base_path / freq_filename
    
    if not spectra_path.exists():
        raise FileNotFoundError(f"Experimental spectrum '{spectra_path}' not found.")
    if not freq_path.exists():
        raise FileNotFoundError(f"Experimental frequency file '{freq_path}' not found.")
    
    # Load spectrum data
    spectrum_df = pd.read_csv(spectra_path, sep="\t")
    
    # Extract bias values from column headers
    col_bias_pairs = _extract_bias_values_from_columns(spectrum_df.columns.tolist())
    
    # Separate numeric columns and their corresponding bias values
    numeric_cols = []
    bias_vals = []
    for col, bias in col_bias_pairs:
        if bias is not None:
            numeric_cols.append(col)
            bias_vals.append(bias)
    
    if not numeric_cols:
        raise ValueError(
            f"No numeric columns found in experimental data. Available columns: {list(spectrum_df.columns)!r}"
        )
    
    # Sort by bias value for consistent ordering
    sorted_pairs = sorted(zip(numeric_cols, bias_vals), key=lambda x: x[1])
    numeric_cols, bias_vals = zip(*sorted_pairs)
    numeric_cols = list(numeric_cols)
    bias_vals = list(bias_vals)
    
    # Extract transmission data (all numeric columns as 2D array)
    transmission_data = spectrum_df[numeric_cols].to_numpy(dtype=float)
    
    # Load frequency data
    freq_df = pd.read_csv(freq_path)
    if freq_df.shape[1] != 1:
        raise ValueError(f"Frequency file '{freq_path}' should contain exactly one column.")
    frequencies = freq_df.iloc[:, 0].to_numpy(dtype=float)
    
    # Apply frequency reversal if requested
    if reverse_frequency:
        frequencies = frequencies[::-1]
        transmission_data = transmission_data[::-1, :]
    
    # Convert frequency units
    from_scale = _resolve_frequency_unit(freq_file_unit)
    to_scale = _resolve_frequency_unit(target_freq_unit)
    frequencies = frequencies * from_scale / to_scale
    
    # Validate array dimensions
    if transmission_data.shape[0] != frequencies.shape[0]:
        raise ValueError(
            f"Transmission and frequency arrays have different lengths: "
            f"{transmission_data.shape[0]} vs {frequencies.shape[0]}."
        )
    
    return frequencies, transmission_data, bias_vals


def plot_experimental_transmission_heatmap(
    *,
    d: Union[int, str],
    p: Union[int, str],
    base_path: Union[str, Path] = "experiment",
    width_tag: str = "w5",
    freq_filename: str = "freq.txt",
    freq_file_unit: str = "GHz",
    target_freq_unit: Optional[str] = None,
    reverse_frequency: bool = True,
    normalize: bool = False,
    cmap: str = "viridis",
    ax: Optional[Axes] = None,
    inset_colorbar: bool = False,
    colorbar_kwargs: Optional[dict] = None,
    apply_style: bool = True,
    **imshow_kwargs: Any,
) -> tuple[Axes, Any]:
    """Plot experimental transmission data as a 2D heatmap over frequency and magnetic field.
    
    Parameters
    ----------
    d, p:
        Thickness and period values to locate the experimental file.
    base_path:
        Directory containing the experimental spectra (default "experiment").
    width_tag:
        Suffix that distinguishes measurement geometry (default "w5").
    freq_filename:
        File with the experimental frequency vector (default "freq.txt").
    freq_file_unit:
        Unit stored in freq_filename ("Hz", "kHz", "MHz" or "GHz").
    target_freq_unit:
        Desired unit for plotting. When None, the original unit is used.
    reverse_frequency:
        Whether to reverse the frequency axis (default True).
    normalize:
        If True, normalizes the transmission data to [0, 1] range.
    cmap:
        Colormap name for the heatmap (default "viridis").
    ax:
        Matplotlib axes to plot on. If None, creates a new figure.
    inset_colorbar:
        If True, places the colorbar inside the plot area (default False).
    colorbar_kwargs:
        Additional keyword arguments for colorbar creation.
    apply_style:
        If True, automatically loads mmpp paper.mplstyle (default True).
    imshow_kwargs:
        Additional keyword arguments forwarded to imshow.
    
    Returns
    -------
    tuple[Axes, Any]
        - ax: The matplotlib axes object
        - im: The AxesImage object from imshow
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    except ImportError:  # pragma: no cover
        raise RuntimeError("Matplotlib is required for plotting.")
    
    # Load mmpp style if requested
    if apply_style:
        _load_mmpp_style(verbose=False)
    
    # Load experimental data
    frequencies, transmission_data, bias_vals = load_experimental_transmission_data(
        d=d,
        p=p,
        base_path=base_path,
        width_tag=width_tag,
        freq_filename=freq_filename,
        freq_file_unit=freq_file_unit,
        target_freq_unit=target_freq_unit,
        reverse_frequency=reverse_frequency,
    )
    
    # Apply normalization if requested
    plot_data = transmission_data.copy()
    if normalize:
        data_max = np.max(np.abs(plot_data))
        if data_max > 0:
            plot_data = plot_data / data_max
    
    # Create axes if not provided
    creating_new_figure = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    else:
        fig = ax.get_figure()
    
    # Create heatmap
    # extent: [left, right, bottom, top] in data coordinates
    extent = [bias_vals[0], bias_vals[-1], frequencies[0], frequencies[-1]]
    
    im = ax.imshow(
        plot_data,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap,
        interpolation="bilinear",
        **imshow_kwargs,
    )
    
    # Labels
    ax.set_xlabel("Magnetic Field B (arbitrary units)")
    ax.set_ylabel(f"Frequency ({target_freq_unit or freq_file_unit})")
    ax.set_title(f"Experimental Transmission Heatmap (d={d}, p={p})")
    
    # Prepare colorbar kwargs
    if colorbar_kwargs is None:
        colorbar_kwargs = {}
    
    # Create colorbar (inset or standard)
    if inset_colorbar:
        # Create inset axes for colorbar inside the plot
        cax = inset_axes(
            ax,
            width="3%",
            height="40%",
            loc="upper right",
            borderpad=2.0,
        )
        cbar = plt.colorbar(im, cax=cax, **colorbar_kwargs)
    else:
        # Standard colorbar outside the plot
        cbar = plt.colorbar(im, ax=ax, **colorbar_kwargs)
    
    cbar.set_label("Transmission" + (" (normalized)" if normalize else ""))
    
    return ax, im


# Alias for backward compatibility
overlay_experimental_transmission = overlay_transmission

__all__ = [
    "overlay_transmission",
    "overlay_experimental_transmission",
    "load_experimental_transmission_data",
    "plot_experimental_transmission_heatmap",
    "_extract_bias_values_from_columns",
]
