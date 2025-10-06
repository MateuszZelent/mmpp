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
        default_cfg = TransmissionConfig()
        lines = [
            "🔊 Transmission Analysis Interface",
            "",
            "Key entry points:",
            "  • transmission() / compute()  – calculate TransmissionResult",
            "  • plot_transmission()         – calculate + render heatmap",
            "",
            "Example:",
            "  result = job[0].fft.transmission(spatial_window=5, reference_window=(0, 2))",
            "  fig, ax, _ = result.plot_transmission()",
            "",
            "Selected defaults:",
        ]
        for field in fields(default_cfg):
            value = getattr(default_cfg, field.name)
            lines.append(f"  • {field.name}: {value}")
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
        summary_table.add_row("🔊", "Transmission Analysis Interface")
        summary_table.add_row("📁", f"Job dataset hint: [bold]{dataset_hint}[/bold]")

        summary_panel = Panel(
            summary_table,
            title="MMPP",
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
            title="TransmissionConfig Defaults",
            show_header=True,
            header_style="bold blue",
        )
        config_table.add_column("Parameter", justify="left")
        config_table.add_column("Default", justify="left")
        for field in fields(default_cfg):
            value = getattr(default_cfg, field.name)
            config_table.add_row(field.name, repr(value))

        config_panel = Panel(config_table, border_style="green")

        console.print(summary_panel)
        console.print(usage_panel)
        console.print(config_panel)

        return buffer.getvalue()


__all__ = ["FFTTransmissionInterface"]
