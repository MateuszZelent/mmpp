"""
FFT Plotting Module

Specialized plotting functionality for FFT analysis results.
"""

from typing import Optional, Dict, List, Union, Any, Tuple
import numpy as np

# Import shared logging configuration
from ..logging_config import setup_mmpp_logging, get_mmpp_logger

# Get logger for FFT plotting
log = get_mmpp_logger("mmpp.fft.plot")

# Import dependencies with error handling
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Import from our own modules
from .compute_fft import FFTCompute, FFTComputeResult


class FFTPlotter:
    """
    Specialized plotter for FFT analysis results.

    Provides FFT-specific plotting capabilities.
    """

    def __init__(
        self, results: Union[List[Any], Any], mmpp_instance: Optional[Any] = None
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

    def power_spectrum(
        self,
        dataset_name: str = "m_z11",
        method: int = 1,
        z_layer: int = -1,
        log_scale: bool = True,
        normalize: bool = False,
        save: bool = True,
        force: bool = False,
        save_dataset_name: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """
        Plot power spectrum for all results.

        Parameters:
        -----------
        dataset_name : str, optional
            Dataset name (default: "m_z11")
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
        \\*\\*kwargs : Any
            Additional FFT configuration options

        Returns:
        --------
        tuple
            (figure, axes) matplotlib objects
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for plotting")

        # Setup figure
        figsize = figsize or self.config["figsize"]
        fig, ax = plt.subplots(figsize=figsize, dpi=self.config["dpi"])

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
                    **kwargs,
                )

                power = np.abs(fft_result.spectrum) ** 2

                # Normalize if requested
                if normalize:
                    power = power / np.max(power)

                # Create label
                label = self._format_result_label(result)

                # Determine scale factor for amplitude normalization
                power_max = np.max(power)
                if power_max > 0 and not log_scale and not normalize:
                    scale_factor = 10 ** np.floor(np.log10(power_max))
                    power_scaled = power / scale_factor

                    # Format the scale factor for the label
                    exponent = int(np.log10(scale_factor))
                    if exponent != 0:
                        global_scale_text = f"$10^{{{exponent}}}$"
                    else:
                        global_scale_text = ""
                        power_scaled = power  # No scaling needed
                else:
                    power_scaled = power
                    global_scale_text = ""

                # Plot
                if log_scale:
                    ax.semilogy(
                        fft_result.frequencies / 1e9,
                        power_scaled,
                        alpha=self.config["line_alpha"],
                        linewidth=self.config["line_width"],
                        label=label,
                    )
                else:
                    ax.plot(
                        fft_result.frequencies / 1e9,
                        power_scaled,
                        alpha=self.config["line_alpha"],
                        linewidth=self.config["line_width"],
                        label=label,
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
