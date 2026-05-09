"""Static plotting helpers for :class:`SpectrumResult`."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    _HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover - optional dependency
    plt = None  # type: ignore[assignment]
    to_rgba = None  # type: ignore[assignment]
    _HAS_MATPLOTLIB = False


def _try_enable_widget_backend() -> None:
    """Try enabling widget backend in Jupyter/IPython."""
    if not _HAS_MATPLOTLIB:
        return
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None:
            return
        current = str(plt.get_backend()).lower()
        if "widget" in current or "ipympl" in current:
            return
        try:
            ipython.run_line_magic("matplotlib", "widget")
        except Exception:
            pass
    except Exception:
        pass


def _generate_pastel_colors(n: int) -> list[Any]:
    """Generate distinct pastel-ish colors for component overlays."""
    if not _HAS_MATPLOTLIB:
        return [(0.4, 0.6, 0.8, 1.0)] * max(1, int(n))
    colors = plt.cm.Accent(np.linspace(0, 1, max(int(n), 3)))
    return [to_rgba(c) for c in colors[: int(n)]]


def plot_spectrum(
    result: Any,
    ax: Any | None = None,
    freq_unit: str = "GHz",
    log_scale: bool = True,
    normalize: bool = False,
    show_peaks: bool = True,
    title: str | None = None,
    dpi: int | None = None,
    **kwargs,
):
    """Plot spectrum in a way compatible with legacy ``SpectrumResult.plot_spectrum``."""
    if not _HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required for plotting")

    _try_enable_widget_backend()

    freq_scales = {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12}
    freq_scale = freq_scales.get(freq_unit, 1e9)
    freqs_display = np.asarray(result.frequencies, dtype=float) / float(freq_scale)

    power = np.asarray(result.power, dtype=float)
    if normalize and power.size:
        vmax = float(np.nanmax(power))
        if vmax > 0:
            power = power / vmax

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    else:
        fig = ax.figure
        if dpi is not None:
            fig.set_dpi(dpi)

    if power.ndim > 1 and power.shape[-1] == 3 and not getattr(result, "_single_component", False):
        if power.ndim > 2:
            spatial_axes = tuple(range(1, power.ndim - 1))
            power_to_plot = np.mean(power, axis=spatial_axes)
        else:
            power_to_plot = power
        labels = [r"$m_x$", r"$m_y$", r"$m_z$"]
        colors = _generate_pastel_colors(3)
        for idx in range(3):
            ax.plot(freqs_display, power_to_plot[:, idx], label=labels[idx], color=colors[idx], **kwargs)
        ax.legend()
    elif power.ndim > 1:
        spatial_axes = tuple(range(1, power.ndim))
        power_to_plot = np.mean(power, axis=spatial_axes)
        if "label" not in kwargs and getattr(result, "component_label", None):
            kwargs["label"] = result.component_label
        elif "label" not in kwargs:
            kwargs["label"] = "Average Power"
        ax.plot(freqs_display, power_to_plot, **kwargs)
        ax.legend()
    else:
        if "label" not in kwargs and getattr(result, "component_label", None):
            kwargs["label"] = result.component_label
        ax.plot(freqs_display, power, **kwargs)
        if "label" in kwargs:
            ax.legend()

    ax.set_xlabel(f"Frequency ({freq_unit})")
    quantity_label = getattr(result, "spectral_quantity_label", None) or (
        "PSD" if getattr(result, "power_quantity", "") == "psd" else "Power"
    )

    if not normalize and not log_scale:
        max_val = float(np.nanmax(power)) if power.size else 0.0
        if max_val > 0:
            exponent = int(np.floor(np.log10(max_val)))
            if abs(exponent) >= 2:
                scale_factor = 10**exponent
                for line in ax.get_lines():
                    ydata = line.get_ydata()
                    line.set_ydata(ydata / scale_factor)
                ax.set_ylabel(f"{quantity_label} (×10$^{{{exponent}}}$ arb. u.)")
                ax.relim()
                ax.autoscale_view()
            else:
                ax.set_ylabel(f"{quantity_label} (arb. u.)")
        else:
            ax.set_ylabel(f"{quantity_label} (arb. u.)")
    elif normalize:
        ax.set_ylabel(f"{quantity_label} (normalized)")
    else:
        ax.set_ylabel(f"{quantity_label} (arb. u.)")

    if log_scale:
        ax.set_yscale("log")

    ax.set_title(title or "FFT Power Spectrum")

    peaks_info = getattr(result, "peaks_info", None)
    if (
        show_peaks
        and isinstance(peaks_info, dict)
        and len(peaks_info.get("indices", [])) > 0
    ):
        peak_freqs = np.asarray(peaks_info.get("frequencies", []), dtype=float) / float(freq_scale)
        peak_amps = np.asarray(peaks_info.get("amplitudes", []), dtype=float)
        peak_powers = peak_amps**2
        if normalize and np.asarray(result.power).size:
            denom = float(np.nanmax(result.power))
            if denom > 0:
                peak_powers = peak_powers / denom

        ax.plot(
            peak_freqs,
            peak_powers,
            "o",
            color="#E74C3C",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=5,
            label="Peaks",
        )

        if peak_freqs.size:
            order = np.argsort(peak_powers)[::-1]
            for rank, idx in enumerate(order[: min(3, peak_freqs.size)]):
                freq = float(peak_freqs[idx])
                power_val = float(peak_powers[idx])
                if rank == 0:
                    ax.vlines(x=freq, ymin=0, ymax=power_val, color="#E74C3C", linestyle=":", alpha=0.6, linewidth=1.2)

                freq_text = f"{freq:.2f}" if 0.01 < freq < 100 else f"{freq:.2e}"
                ax.annotate(
                    f"{freq_text} {freq_unit}",
                    xy=(freq, power_val),
                    xytext=(8, 8 + rank * 12),
                    textcoords="offset points",
                    fontsize=9,
                    color="#2C3E50",
                    fontweight="medium",
                    arrowprops=(
                        {"arrowstyle": "-", "color": "#E74C3C", "alpha": 0.6, "lw": 0.8}
                        if rank == 0
                        else None
                    ),
                    bbox=(
                        {
                            "boxstyle": "round,pad=0.3",
                            "facecolor": "white",
                            "edgecolor": "#E74C3C",
                            "alpha": 0.9,
                            "linewidth": 0.8,
                        }
                        if rank == 0
                        else None
                    ),
                    zorder=10,
                )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.legend(frameon=True, fancybox=True, shadow=False, framealpha=0.9, edgecolor="lightgray", fontsize=9)

    fig.tight_layout()
    return fig, ax, peaks_info
