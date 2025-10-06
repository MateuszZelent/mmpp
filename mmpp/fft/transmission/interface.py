"""User-facing transmission interface for FFT analysis."""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any, Optional

from ...cli.logging_config import get_mmpp_logger

from .compute import TransmissionCompute, TransmissionConfig, TransmissionResult
from .plot import TransmissionPlotConfig, TransmissionPlotter


log = get_mmpp_logger("mmpp.fft.transmission.interface")


class FFTTransmissionInterface:
    """Convenience wrapper accessible via ``job.fft.transmission``."""

    def __init__(self, fft_instance: Any, fft_compute, job_result: Any):
        self._fft = fft_instance
        self._job_result = job_result
        self._compute = TransmissionCompute(fft_compute, job_result)

    def __call__(
        self,
        config: Optional[TransmissionConfig] = None,
        /,
        **kwargs,
    ) -> TransmissionResult:
        """Compute transmission map for provided configuration.

        Parameters
        ----------
        config:
            Optional :class:`TransmissionConfig`. If omitted, keyword arguments are
            used to construct one.
        **kwargs:
            Parameters used when instantiating a new configuration object.
        """

        if config is not None and kwargs:
            raise ValueError("Provide either a TransmissionConfig or keyword arguments, not both")

        if config is None:
            config = TransmissionConfig(**kwargs)

        log.debug("Computing transmission with configuration: %s", asdict(config))
        return self._compute.compute(config)

    def compute(
        self,
        config: Optional[TransmissionConfig] = None,
        /,
        **kwargs,
    ) -> TransmissionResult:
        """Alias for :meth:`__call__` to mirror other interfaces."""

        return self.__call__(config, **kwargs)

    def plot_transmission(
        self,
        config: Optional[TransmissionConfig] = None,
        plot_config: Optional[TransmissionPlotConfig] = None,
        **kwargs,
    ):
        """Compute and immediately plot the transmission map."""

        result = self.__call__(config, **kwargs)
        plotter = TransmissionPlotter(result)
        
        # Convert dict to TransmissionPlotConfig if needed
        if plot_config is not None and isinstance(plot_config, dict):
            plot_config = TransmissionPlotConfig(**plot_config)
        
        return plotter.plot(config=plot_config)

    # ------------------------------------------------------------------
    # Rich / basic representation helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - formatting logic
        try:
            return self._rich_display()
        except Exception:
            return self._basic_display()

    # Internal helpers -------------------------------------------------
    def _basic_display(self) -> str:
        """Fallback basic display matching FFT core style."""
        default_cfg = TransmissionConfig()
        dataset_hint = self._job_result.get_largest_m_dataset()
        
        lines = [
            "=" * 70,
            "🔊 MMPP Transmission Analysis Interface",
            "=" * 70,
            f"📁 Job Dataset: {dataset_hint}",
            "",
            "🔧 CORE METHODS:",
            "─" * 50,
            "  • transmission()       - Compute transmission analysis",
            "  • compute()           - Alias for transmission()",
            "  • plot_transmission() - Compute and plot results",
            "",
            "📋 USAGE EXAMPLES:",
            "─" * 50,
            "  # Basic usage",
            "  result = job[0].fft.transmission(spatial_window=5)",
            "  ",
            "  # With configuration",
            "  result = job[0].fft.transmission(",
            "      spatial_window=10,",
            "      reference_window=(0, 3),",
            "      normalize='reference'",
            "  )",
            "  ",
            "  # Plot results",
            "  fig, ax, m = result.plot_transmission(",
            "      plot_config=dict(freq_unit='GHz', log_scale=True)",
            "  )",
            "",
            "⚙️ CONFIGURATION PARAMETERS (with available options):",
            "─" * 50,
        ]
        
        # Detailed parameter documentation with options
        param_docs = [
            ("spatial_window", "5", "int > 0 - Size of spatial analysis window"),
            ("spatial_step", "1", "int ≥ 1 - Step for sliding window"),
            ("reference_window", "None", "(int, int) or None - Reference region [start, end]"),
            ("normalize", "'reference'", "'reference' | 'max' | 'none'"),
            ("method", "'power_ratio'", "'power_ratio' | 'circular' | 'cpsd'"),
            ("window_function", "'hann'", "'hann' | 'hamming' | 'blackman' | 'bartlett' | 'none'"),
            ("filter_type", "'remove_mean'", "'remove_mean' | 'remove_linear' | 'highpass' | 'none'"),
            ("reference_statistic", "'mean'", "'mean' | 'median' | 'max'"),
            ("average_mode", "'mean'", "'mean' | 'median' | 'edge_taper'"),
            ("component_weights", "(1.0, 1.0, 0.1)", "(mx, my, mz) - Tuple[float, float, float]"),
            ("enable_circular_components", "False", "bool - Enable m+ and m- components"),
            ("store_component_maps", "False", "bool - Store individual component maps"),
            ("dataset_name", "None", "str or None - Dataset name (auto-detect if None)"),
            ("z_layer", "-1", "int - Z layer index (-1 = last)"),
            ("tmax", "None", "int or None - Max time steps"),
        ]
        
        for param, default, description in param_docs:
            lines.append(f"  • {param}")
            lines.append(f"    Default: {default}")
            lines.append(f"    Options: {description}")
            lines.append("")
        
        lines.append("=" * 70)
        return "\n".join(lines)

    def _rich_display(self) -> str:  # pragma: no cover - formatting logic
        import io

        from rich.console import Console
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.table import Table

        buffer = io.StringIO()
        console = Console(file=buffer, width=120, force_terminal=True)

        default_cfg = TransmissionConfig()
        dataset_hint = self._job_result.get_largest_m_dataset()

        # Summary Panel ------------------------------------------------
        summary_table = Table.grid(expand=True)
        summary_table.add_row("🔊", "[bold cyan]Transmission Analysis Interface[/bold cyan]")
        summary_table.add_row("📁", f"Job Dataset: [bold]{dataset_hint}[/bold]")
        summary_table.add_row("", "")
        summary_table.add_row("🔧", "[bold yellow]Core Methods:[/bold yellow]")
        summary_table.add_row("", "  • [code]transmission()[/code] - Compute transmission")
        summary_table.add_row("", "  • [code]compute()[/code] - Alias for transmission()")
        summary_table.add_row("", "  • [code]plot_transmission()[/code] - Compute + plot")

        summary_panel = Panel(
            summary_table,
            title="[bold]MMPP Transmission Interface[/bold]",
            title_align="left",
            border_style="cyan",
        )

        # Usage Panel --------------------------------------------------
        usage_code = """\
result = job[0].fft.transmission(
    spatial_window=5,
    reference_window=(0, 2),
    normalize="reference",
)

fig, ax, m = result.plot_transmission(
    plot_config=dict(freq_unit="GHz", log_scale=False)
)
"""

        usage_panel = Panel(
            Syntax(usage_code, "python", theme="monokai", line_numbers=False),
            title="Quick Start",
            border_style="magenta",
        )

        # Config Panel -------------------------------------------------
        config_table = Table(
            title="TransmissionConfig Parameters",
            show_header=True,
            header_style="bold blue",
        )
        config_table.add_column("Parameter", justify="left", style="cyan")
        config_table.add_column("Default", justify="left", style="green")
        config_table.add_column("Available Options", justify="left", style="yellow")
        
        # Detailed parameter documentation
        param_info = [
            ("spatial_window", "5", "int > 0"),
            ("spatial_step", "1", "int ≥ 1"),
            ("reference_window", "None", "(int, int) or None"),
            ("normalize", "'reference'", "'reference' | 'max' | 'none'"),
            ("method", "'power_ratio'", "'power_ratio' | 'circular' | 'cpsd'"),
            ("window_function", "'hann'", "'hann' | 'hamming' | 'blackman' | 'bartlett' | 'none'"),
            ("filter_type", "'remove_mean'", "'remove_mean' | 'remove_linear' | 'highpass' | 'none'"),
            ("reference_statistic", "'mean'", "'mean' | 'median' | 'max'"),
            ("average_mode", "'mean'", "'mean' | 'median' | 'edge_taper'"),
            ("component_weights", "(1.0, 1.0, 0.1)", "(mx, my, mz) weights"),
            ("enable_circular_components", "False", "bool - m±"),
            ("store_component_maps", "False", "bool"),
            ("dataset_name", "None", "str or None"),
            ("z_layer", "-1", "int (negative = from end)"),
            ("tmax", "None", "int or None"),
        ]
        
        for param, default, options in param_info:
            config_table.add_row(param, default, options)

        config_panel = Panel(config_table, border_style="green", title="[bold]Configuration Options[/bold]")

        console.print(summary_panel)
        console.print(usage_panel)
        console.print(config_panel)

        return buffer.getvalue()


__all__ = ["FFTTransmissionInterface"]
