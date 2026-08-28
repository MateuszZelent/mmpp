"""Standalone function to overlay experimental transmission measurements."""

from __future__ import annotations

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency

    class _MissingPandas:
        def __getattr__(self, _name: str):
            raise ImportError("Experimental transmission table loading requires pandas")

    pd = _MissingPandas()  # type: ignore[assignment]
from pathlib import Path
from typing import Any

try:  # pragma: no cover - matplotlib is an optional dependency at runtime
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
except ImportError:  # pragma: no cover
    Axes = Any  # type: ignore
    Line2D = Any  # type: ignore


def overlay_experimental_transmission(
    ax: Axes,
    *,
    d: int | str,
    p: int | str,
    base_path: str | Path = "experiment",
    width_tag: str = "w5",
    freq_filename: str = "freq.txt",
    freq_file_unit: str = "GHz",
    target_freq_unit: str | None = None,
    bias_index: float | None = None,
    column: int | str | None = None,
    reverse_frequency: bool = True,
    color: str = "tab:red",
    linewidth: float = 0.8,
    label: str | None = None,
    **plot_kwargs,
) -> Line2D:
    """Overlay experimental transmission trace on a transmission cross-section.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axes returned by :meth:`TransmissionResult.plot_transmission_crosssection`.
    d, p : int or str
        Thickness ``d`` and period ``p`` values used to locate the experimental file.
    base_path : str or pathlib.Path, default="experiment"
        Directory that contains the experimental spectra.
    width_tag : str, default="w5"
        Suffix that distinguishes measurement geometry.
    freq_filename : str, default="freq.txt"
        File with the experimental frequency vector.
    freq_file_unit : str, default="GHz"
        Unit stored in ``freq_filename`` ("Hz", "kHz", "MHz", "GHz").
    target_freq_unit : str, optional
        Desired unit for plotting. When ``None`` the original unit is used.
    bias_index : float, optional
        Optional magnetic bias index used to infer the measurement column.
    column : int, str, or None, optional
        Explicit column index or label from the measurement file. Overrides ``bias_index``.
    reverse_frequency : bool, default=True
        Reverse the frequency axis to match typical experimental formatting.
    color, linewidth, label, plot_kwargs
        Forwarded to :func:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.lines.Line2D
        Handle of the added experimental trace.

    Examples
    --------
    >>> figa, ax_cross, minima_freqs = result_raw.plot_transmission_crosssection(
    ...     ax=ax_third,
    ...     x=x*1e-9,
    ...     freq_unit="GHz",
    ...     trim_0f=0,
    ...     color="red",
    ...     fmax=2,
    ...     flip=True,
    ...     linewidth=3.3,
    ...     find_minima={'distance': 10, 'freq_range': (0.8, 2.8)},
    ...     mark_on_ax=ax_right,
    ... )
    >>> # Add experimental data
    >>> overlay_experimental_transmission(ax_cross, d=120, p=470)
    """
    if target_freq_unit is None:
        target_freq_unit = freq_file_unit

    base_path = Path(base_path)
    spectra_path = base_path / f"d{d}p{p}_{width_tag}.txt"
    freq_path = base_path / freq_filename

    if not spectra_path.exists():
        raise FileNotFoundError(f"Experimental spectrum '{spectra_path}' not found.")
    if not freq_path.exists():
        raise FileNotFoundError(f"Experimental frequency file '{freq_path}' not found.")

    # Load experimental data
    spectrum_df = pd.read_csv(spectra_path, sep="\t")
    freq_df = pd.read_csv(freq_path, header=None, names=["freq"])

    # Select column
    if column is not None:
        if isinstance(column, int):
            try:
                selected_column = spectrum_df.columns[column]
            except IndexError as exc:
                raise ValueError(
                    f"Column index {column} is out of range for experimental data."
                ) from exc
        else:
            if column not in spectrum_df.columns:
                raise ValueError(
                    f"Column '{column}' not found in experimental data. Available columns: {list(spectrum_df.columns)!r}"
                )
            selected_column = column
    elif bias_index is not None:
        label = f"{int(2 * bias_index) - 1}"
        if label in spectrum_df.columns:
            selected_column = label
        else:
            raise ValueError(
                f"Could not resolve column for bias_index={bias_index}. Available labels: {list(spectrum_df.columns)!r}"
            )
    else:
        # Default to first numeric column
        numeric_columns = [
            col
            for col in spectrum_df.columns
            if pd.api.types.is_numeric_dtype(spectrum_df[col])
        ]
        if numeric_columns:
            selected_column = numeric_columns[0]
        else:
            selected_column = spectrum_df.columns[0]

    amplitudes = spectrum_df[selected_column].to_numpy(dtype=float)
    frequencies = freq_df["freq"].to_numpy(dtype=float)

    if reverse_frequency:
        frequencies = frequencies[::-1]

    # Convert frequency units if needed
    freq_unit_factors = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
    from_scale = freq_unit_factors[freq_file_unit]
    to_scale = freq_unit_factors[target_freq_unit]
    frequencies = frequencies * from_scale / to_scale

    if amplitudes.shape[0] != frequencies.shape[0]:
        raise ValueError(
            "Experimental amplitude and frequency arrays have different lengths: "
            f"{amplitudes.shape[0]} vs {frequencies.shape[0]}."
        )

    if label is None:
        label = f"exp d={d} p={p}"

    (line,) = ax.plot(
        amplitudes,
        frequencies,
        color=color,
        linewidth=linewidth,
        label=label,
        **plot_kwargs,
    )
    return line
