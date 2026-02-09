"""
FFT Plotting Module

Specialized plotting functionality for FFT analysis results.
"""

from typing import Any, Optional, Union

import numpy as np

# Import shared logging configuration
from ..cli.logging_config import get_mmpp_logger, setup_mmpp_logging

# Get logger for FFT plotting
log = get_mmpp_logger("mmpp.fft.plot")

# Import dependencies with error handling
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Import from our own modules
from .compute_fft import FFTCompute
from .metrics import (
    compute_half_width_at_half_max,
    format_width_value,
    normalize_peak_width_option,
)


class FFTPlotter:
    """
    Specialized plotter for FFT analysis results.

    Provides FFT-specific plotting capabilities.
    """

    def __init__(
        self, results: Union[list[Any], Any], mmpp_instance: Optional[Any] = None
    ):
        """
        Initialize FFT plotter.

        Parameters:
        -----------
        results : List or single result
            ZarrJobResult objects to plot
        mmpp_instance : MMPP, optional
            Reference to parent MMPP instance
        """
        if not isinstance(results, list):
            self.results = [results]
        else:
            self.results = results

        # Set up logging level based on parent debug mode
        debug_mode = getattr(mmpp_instance, "debug", False) if mmpp_instance else False
        setup_mmpp_logging(debug=debug_mode, logger_name="mmpp.fft.plot")

        log.debug(
            f"FFTPlotter.__init__: Received {type(results)} with {len(self.results) if hasattr(self, 'results') else 'unknown'} results"
        )
        for i, result in enumerate(self.results):
            log.debug(
                f"FFTPlotter.__init__: Result {i}: {type(result)} - {getattr(result, 'path', 'no path')}"
            )

        self.mmpp = mmpp_instance
        self.fft_compute = FFTCompute(debug=debug_mode)

        # Basic plot configuration
        self.config = {
            "figsize": (10, 6),
            "dpi": 100,
            "line_alpha": 0.8,
            "line_width": 2,
            "label_fontsize": 12,
            "title_fontsize": 14,
            "tick_fontsize": 10,
            "grid": True,
            "legend": True,
        }

    def _format_result_label(self, result) -> str:
        """Format result label for plotting."""
        import os

        return os.path.basename(result.path)

    @staticmethod
    def _format_slice_identifier(slice_info: Optional[Any]) -> Optional[str]:
        """Create deterministic identifier for slice-aware FFT save/cache names."""
        if slice_info is None:
            return None

        def _format_item(item: Any) -> str:
            if isinstance(item, slice):
                return f"{item.start}:{item.stop}:{item.step}"
            if item is Ellipsis:
                return "..."
            if isinstance(item, tuple):
                return "(" + ",".join(_format_item(sub) for sub in item) + ")"
            if isinstance(item, np.integer):
                return str(int(item))
            return repr(item)

        slice_tuple = slice_info if isinstance(slice_info, tuple) else (slice_info,)
        return "slice=" + ",".join(_format_item(part) for part in slice_tuple)

    def power_spectrum(
        self,
        dataset_name: Optional[str] = None,
        method: int = 1,
        z_layer: int = -1,
        log_scale: bool = True,
        normalize: bool = False,
        save: bool = True,
        ax: Optional[Any] = None,
        force: bool = False,
        save_dataset_name: Optional[str] = None,
        figsize: Optional[tuple[float, float]] = None,
        save_path: Optional[str] = None,
        tmax: Optional[int] = None,
        slice_info: Optional[Any] = None,
        slice_identifier: Optional[str] = None,
        **kwargs,
    ) -> tuple[Any, Any]:
        """
        Plot power spectrum for all results.

        Parameters:
        -----------
        dataset_name : str, optional
            Dataset name (default: auto-select largest m dataset)
        method : int, optional
            FFT method (default: 1)
        z_layer : int, optional
            Z-layer (default: -1)
        log_scale : bool, optional
            Use logarithmic scale for power (default: True)
        normalize : bool, optional
            Normalize power spectra (default: False)
        save : bool, optional
            Save FFT result to zarr file (default: True)
        force : bool, optional
            Force recalculation and overwrite existing (default: False)
        save_dataset_name : str, optional
            Custom name for saved dataset (default: auto-generated)
        figsize : tuple, optional
            Figure size
        save_path : str, optional
            Path to save figure
        tmax : int, optional
            Maximum number of time steps to use for FFT calculation (default: None, use all)
        slice_info : Any, optional
            Optional slicing applied before FFT data loading.
        slice_identifier : str, optional
            Optional deterministic identifier for slice-aware save/cache naming.
        \\*\\*kwargs : Any
            Additional FFT configuration options. Recognised keys include
            ``peak_width``/``fwhh``/``fwhm``/``hwfh`` (bool or str) to add a
            half-width at half-maximum annotation for the dominant peak.

        Returns:
        --------
        tuple
            (figure, axes) matplotlib objects
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for plotting")

        # Auto-select largest m dataset if none specified
        if dataset_name is None and self.results:
            dataset_name = self.results[0].get_largest_m_dataset()

        # Setup figure
        figsize = figsize or self.config["figsize"]
        fig, ax = plt.subplots(figsize=figsize, dpi=self.config["dpi"])

        # Extract FWHM/FWHH request before forwarding kwargs downstream
        peak_width_option = kwargs.pop("peak_width", None)
        for alias in ("fwhh", "fwhm", "hwfh"):
            alias_value = kwargs.pop(alias, None)
            if alias_value is not None:
                peak_width_option = alias_value

        show_peak_width, peak_width_label = normalize_peak_width_option(
            peak_width_option
        )

        if slice_identifier is None and slice_info is not None:
            slice_identifier = self._format_slice_identifier(slice_info)

        # Initialize scale tracking
        global_scale_text = ""

        # Debug: Check number of results
        log.debug(f"Processing {len(self.results)} result(s)")
        for i, result in enumerate(self.results):
            log.debug(
                f"Result {i}: {type(result)} - {getattr(result, 'path', 'no path')}"
            )

        # Analyze all results
        for i, result in enumerate(self.results):
            try:
                fft_result = self.fft_compute.calculate_fft_data(
                    result.path,
                    dataset_name,
                    z_layer,
                    method,
                    save=save,
                    force=force,
                    save_dataset_name=save_dataset_name,
                    slice_info=slice_info,
                    slice_identifier=slice_identifier,
                    tmax=tmax,
                    **kwargs,
                )

                power = np.abs(fft_result.spectrum) ** 2

                # Debug: Check array shapes
                log.debug(f"FFT result shapes: spectrum={fft_result.spectrum.shape}, frequencies={fft_result.frequencies.shape}")

                # Handle multi-dimensional spectrum - average over components if needed
                if power.ndim > 1:
                    log.debug(f"Spectrum is {power.ndim}D, averaging over non-frequency dimensions")
                    # Average over all dimensions except the first (frequency)
                    power = np.mean(power, axis=tuple(range(1, power.ndim)))
                    log.debug(f"After averaging: power shape={power.shape}")

                # Verify array lengths match
                if len(fft_result.frequencies) != len(power):
                    log.error(f"Length mismatch after processing: frequencies={len(fft_result.frequencies)}, power={len(power)}")
                    continue

                # Normalize if requested
                if normalize and np.max(power) > 0:
                    power = power / np.max(power)

                # Create label
                label = self._format_result_label(result)

                # Determine scale factor for amplitude normalization
                power_max = np.max(power) if power.size else 0.0
                scale_factor = 1.0
                power_scaled = power
                global_scale_text = ""
                if power_max > 0 and not log_scale and not normalize:
                    scale_factor_candidate = 10 ** np.floor(np.log10(power_max))
                    if scale_factor_candidate > 0:
                        scale_factor = scale_factor_candidate
                        power_scaled = power / scale_factor

                        # Format the scale factor for the label
                        exponent = int(np.log10(scale_factor))
                        if exponent != 0:
                            global_scale_text = f"$10^{{{exponent}}}$"
                        else:
                            scale_factor = 1.0
                            power_scaled = power
                    else:
                        scale_factor = 1.0
                        power_scaled = power

                freqs_ghz = fft_result.frequencies / 1e9

                # Plot
                if log_scale:
                    lines = ax.semilogy(
                        freqs_ghz,
                        power_scaled,
                        alpha=self.config["line_alpha"],
                        linewidth=self.config["line_width"],
                        label=label,
                    )
                    line = lines[0] if isinstance(lines, list) else lines
                else:
                    lines = ax.plot(
                        freqs_ghz,
                        power_scaled,
                        alpha=self.config["line_alpha"],
                        linewidth=self.config["line_width"],
                        label=label,
                    )
                    line = lines[0] if isinstance(lines, list) else lines

                if show_peak_width:
                    # Ensure arrays have same length before computing FWHM
                    if len(freqs_ghz) != len(power):
                        log.warning(f"Array length mismatch: frequencies={len(freqs_ghz)}, power={len(power)}. Skipping FWHM calculation.")
                        width_info = None
                    else:
                        width_info = compute_half_width_at_half_max(freqs_ghz, power)
                    if width_info is None:
                        log.debug(
                            "Skipping peak width annotation for %s: unable to determine half-width",
                            label,
                        )
                    else:
                        half_level_plot = width_info.half_level / scale_factor
                        if half_level_plot <= 0:
                            log.debug(
                                "Skipping peak width annotation for %s: non-positive half level",
                                label,
                            )
                        else:
                            color = line.get_color()
                            ax.hlines(
                                half_level_plot,
                                width_info.left_frequency,
                                width_info.right_frequency,
                                colors=color,
                                linewidth=1.5,
                                linestyles="-",
                                alpha=0.8,
                            )
                            delta = 0.05
                            ymin = half_level_plot * (1.0 - delta)
                            ymax = half_level_plot * (1.0 + delta)
                            ax.vlines(
                                [width_info.left_frequency, width_info.right_frequency],
                                ymin=ymin,
                                ymax=ymax,
                                colors=color,
                                linewidth=1.2,
                                alpha=0.8,
                            )

                            text = f"{peak_width_label}: {format_width_value(width_info.width)}"
                            ax.annotate(
                                text,
                                xy=(
                                    (width_info.left_frequency + width_info.right_frequency) / 2.0,
                                    half_level_plot,
                                ),
                                xytext=(0, 8),
                                textcoords="offset points",
                                ha="center",
                                va="bottom",
                                fontsize=9,
                                color=color,
                                bbox={
                                    "boxstyle": "round,pad=0.2",
                                    "facecolor": "white",
                                    "edgecolor": color,
                                    "linewidth": 0.8,
                                    "alpha": 0.8,
                                },
                            )

            except Exception as e:
                log.error(f"Error analyzing result {i}: {e}")
                continue

        # Customize plot
        ax.set_xlabel("Frequency (GHz)", fontsize=self.config["label_fontsize"])

        # Format Y-axis label with scale factor
        ylabel = "Normalized FFT Amplitude" if normalize else "FFT Amplitude"
        if global_scale_text and not log_scale:
            ylabel += f"({global_scale_text} arb. units)"
        elif not normalize:
            ylabel += " (arb. units)"

        if log_scale:
            ylabel += " (log scale)"
        ax.set_ylabel(ylabel, fontsize=self.config["label_fontsize"])

        # Handle axis formatting based on scale type
        try:
            # Only apply plain formatting if not using log scale
            if not log_scale:
                ax.ticklabel_format(style="plain", axis="y", useOffset=False)
                # Try to disable scientific notation for linear scale
                y_formatter = ax.yaxis.get_major_formatter()
                if hasattr(y_formatter, "set_useOffset"):
                    y_formatter.set_useOffset(False)
                if hasattr(y_formatter, "set_scientific"):
                    y_formatter.set_scientific(False)

            # X-axis formatting (frequency axis is always linear)
            ax.ticklabel_format(style="plain", axis="x", useOffset=False)
            x_formatter = ax.xaxis.get_major_formatter()
            if hasattr(x_formatter, "set_useOffset"):
                x_formatter.set_useOffset(False)
            if hasattr(x_formatter, "set_scientific"):
                x_formatter.set_scientific(False)

        except AttributeError:
            # If formatter doesn't support these methods, skip formatting
            pass

        title = f"Power Spectrum - {dataset_name} (Method {method})"
        ax.set_title(title, fontsize=self.config["title_fontsize"])

        if self.config["grid"]:
            ax.grid(True, alpha=0.3)

        if self.config["legend"] and len(self.results) > 1:
            ax.legend(fontsize=self.config["label_fontsize"])

        ax.tick_params(labelsize=self.config["tick_fontsize"])
        plt.tight_layout()

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.config["dpi"], bbox_inches="tight")
            log.info(f"Figure saved to: {save_path}")

        return fig, ax

    def __repr__(self) -> str:
        return f"FFTPlotter({len(self.results)} results)"
