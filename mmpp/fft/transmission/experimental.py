"""Utilities for overlaying experimental transmission measurements."""

from __future__ import annotations

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
    freq_filename: str = "freq.txt",
    freq_file_unit: str = "GHz",
    target_freq_unit: Optional[str] = None,
    bias_index: Optional[float] = None,
    column: Union[int, str, None] = None,
    reverse_frequency: bool = True,
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

            # Get the simulation data (transmission is on the x-axis when flip=True)
            sim_transmission_data = sim_line.get_xdata()
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

    line, = ax.plot(
        amplitudes,
        frequencies,
        color=color,
        linewidth=linewidth,
        label=label,
        **plot_kwargs,
    )
    return line


# Alias for backward compatibility
overlay_experimental_transmission = overlay_transmission

__all__ = [
    "overlay_transmission",
    "overlay_experimental_transmission",
]
