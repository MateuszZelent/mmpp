"""User-facing transmission interface for FFT analysis."""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Optional, Union

from ...cli.logging_config import get_mmpp_logger
from ..method_helpers import CallableMethodHelper

from .cache import TransmissionCache
from .compute import TransmissionCompute, TransmissionConfig, TransmissionResult
from .plot import TransmissionPlotConfig, TransmissionPlotter


log = get_mmpp_logger("mmpp.fft.transmission.interface")


class FFTTransmissionHelpAccessor:
    """Callable helper namespace for transmission workflows."""

    def __init__(self, transmission: "FFTTransmissionInterface", owner: str = "fft.transmission"):
        self._transmission = transmission
        self._owner = owner

    def _method(
        self,
        name: str,
        description: str,
        examples: list[str] | None = None,
    ) -> CallableMethodHelper:
        target = getattr(self._transmission, name)
        return CallableMethodHelper(
            owner=self._owner,
            name=name,
            target=target,
            description=description,
            examples=examples or [],
        )

    @property
    def compute(self) -> CallableMethodHelper:
        return self._method(
            "compute",
            "Compute transmission map (cache-aware alias of call).",
            ["job[0].fft.transmission.help.compute(save=True)"],
        )

    @property
    def call(self) -> CallableMethodHelper:
        return CallableMethodHelper(
            owner=self._owner,
            name="__call__",
            target=self._transmission,
            description="Compute transmission map via callable interface.",
            examples=["job[0].fft.transmission.help.call(save=True)"],
        )

    @property
    def plot_transmission(self) -> CallableMethodHelper:
        return self._method(
            "plot_transmission",
            "Compute and immediately plot transmission map.",
            ["job[0].fft.transmission.help.plot_transmission(save=False)"],
        )

    @property
    def visualize_mode(self) -> CallableMethodHelper:
        return self._method(
            "visualize_mode",
            "Reconstruct and plot 2D spin-wave mode for selected frequency.",
            ["job[0].fft.transmission.help.visualize_mode(f=2.5)"],
        )

    @property
    def visualize_modes(self) -> CallableMethodHelper:
        return self._method(
            "visualize_modes",
            "Batch mode visualization for multiple selected frequencies.",
            ["job[0].fft.transmission.help.visualize_modes([2.2, 2.5, 2.8])"],
        )

    @property
    def save_mode_visualizations(self) -> CallableMethodHelper:
        return self._method(
            "save_mode_visualizations",
            "Save one mode image per selected frequency/bin to a folder.",
            ["job[0].fft.transmission.help.save_mode_visualizations('mode_frames')"],
        )

    def __repr__(self) -> str:
        return (
            "<FFTTransmissionHelpAccessor: compute, call, plot_transmission, "
            "visualize_mode, visualize_modes, save_mode_visualizations>"
        )


class FFTTransmissionInterface:
    """Convenience wrapper accessible via ``job.fft.transmission``."""

    def __init__(
        self,
        fft_instance: Any,
        fft_compute,
        job_result: Any,
        dataset_name: Optional[str] = None,
        slice_info: Optional[Any] = None,
    ):
        self._fft = fft_instance
        self._job_result = job_result
        self._compute = TransmissionCompute(fft_compute, job_result)
        self.dataset_name = dataset_name
        self.slice_info = slice_info
        self._cache = TransmissionCache(job_result, dataset_name)

    def clone_for_dataset(
        self,
        dataset_name: Optional[str],
        slice_info: Optional[Any] = None,
    ) -> "FFTTransmissionInterface":
        """Return a dataset-aware clone for transmission analysis."""
        return FFTTransmissionInterface(
            self._fft,
            self._compute._fft_compute,  # Access the underlying fft_compute
            self._job_result,
            dataset_name=dataset_name,
            slice_info=slice_info,
        )

    @property
    def helpers(self) -> FFTTransmissionHelpAccessor:
        """Helper namespace for major transmission methods."""
        return FFTTransmissionHelpAccessor(self, owner="fft.transmission")

    @property
    def help(self) -> FFTTransmissionHelpAccessor:
        """Alias for :attr:`helpers`."""
        return self.helpers

    def __call__(
        self,
        config: Optional[TransmissionConfig] = None,
        /,
        save: bool = False,
        cache_path: Optional[Union[str, Path]] = None,
        force: bool = False,
        use_cache: bool = True,
        **kwargs,
    ) -> TransmissionResult:
        """Compute transmission map for provided configuration.

        Parameters
        ----------
        config:
            Optional :class:`TransmissionConfig`. If omitted, keyword arguments are
            used to construct one.
        save : bool, optional
            If True, save result to cache (default: False)
        cache_path : str or Path, optional
            Custom cache directory. If None, uses source zarr directory.
            Example: "/tmp/fft_cache"
        force : bool, optional
            If True, force recomputation even if cached (default: False)
        use_cache : bool, optional
            If True, use cache for loading/saving (default: True)
        **kwargs:
            Parameters used when instantiating a new configuration object.

        Returns
        -------
        TransmissionResult
            Computed or cached transmission result.

        Examples
        --------
        >>> # Compute and save to default cache location (zarr directory)
        >>> result = job[0].fft.transmission(save=True)

        >>> # Use custom cache directory
        >>> result = job[0].fft.transmission(save=True, cache_path="/tmp/fft_cache")

        >>> # Force recomputation and update cache
        >>> result = job[0].fft.transmission(save=True, force=True)
        """

        if config is not None and kwargs:
            raise ValueError(
                "Provide either a TransmissionConfig or keyword arguments, not both"
            )

        if config is None:
            # Inject dataset_name from interface if not provided in kwargs
            if self.dataset_name is not None and "dataset_name" not in kwargs:
                kwargs["dataset_name"] = self.dataset_name
            config = TransmissionConfig(**kwargs)
        else:
            # If config provided but dataset_name is None, inject from interface
            if config.dataset_name is None and self.dataset_name is not None:
                from dataclasses import replace

                config = replace(config, dataset_name=self.dataset_name)

        # Normalize/validate config early so cache keys are consistent for aliases
        # like spatial_window=0/False (treated as 1).
        config.ensure_valid()

        # Convert cache_path to Path if string
        cache_dir = Path(cache_path) if cache_path is not None else None

        # Try loading from cache first (unless force is True)
        result = None
        if use_cache and not force:
            result = self._cache.load_result(
                config, self.slice_info, cache_path=cache_dir
            )
            if result is not None:
                log.info("Loaded transmission result from cache")
                return result

        # Compute from scratch
        # Create safe dict for logging (exclude non-serializable fields)
        config_dict = asdict(config)
        config_dict.pop("progress_callback", None)
        log.debug("Computing transmission with configuration: %s", config_dict)
        result = self._compute.compute(config, slice_info=self.slice_info)

        # Save to cache if requested
        if save and use_cache:
            self._cache.save_result(
                result, self.slice_info, cache_path=cache_dir, overwrite=force
            )

        return result

    def compute(
        self,
        config: Optional[TransmissionConfig] = None,
        /,
        save: bool = False,
        cache_path: Optional[Union[str, Path]] = None,
        force: bool = False,
        use_cache: bool = True,
        **kwargs,
    ) -> TransmissionResult:
        """Alias for :meth:`__call__` to mirror other interfaces.

        Parameters
        ----------
        config:
            Optional :class:`TransmissionConfig`. If omitted, keyword arguments are
            used to construct one.
        save : bool, optional
            If True, save result to cache (default: False)
        cache_path : str or Path, optional
            Custom cache directory. If None, uses source zarr directory.
        force : bool, optional
            If True, force recomputation even if cached (default: False)
        use_cache : bool, optional
            If True, use cache for loading/saving (default: True)
        **kwargs:
            Parameters used when instantiating a new configuration object.

        Returns
        -------
        TransmissionResult
            Computed or cached transmission result.
        """

        return self.__call__(
            config,
            save=save,
            cache_path=cache_path,
            force=force,
            use_cache=use_cache,
            **kwargs,
        )

    def plot_transmission(
        self,
        config: Optional[TransmissionConfig] = None,
        plot_config: Optional[TransmissionPlotConfig] = None,
        save: bool = False,
        cache_path: Optional[Union[str, Path]] = None,
        force: bool = False,
        use_cache: bool = True,
        **kwargs,
    ):
        """Compute and immediately plot the transmission map.

        Parameters
        ----------
        config : TransmissionConfig, optional
            Transmission computation config
        plot_config : TransmissionPlotConfig or dict, optional
            Plotting configuration
        save : bool, optional
            If True, save result to cache (default: False)
        cache_path : str or Path, optional
            Custom cache directory. If None, uses source zarr directory.
        force : bool, optional
            If True, force recomputation even if cached (default: False)
        use_cache : bool, optional
            If True, use cache for loading/saving (default: True)
        **kwargs
            Forwarded to computation config

        Returns
        -------
        fig, ax
            Matplotlib figure and axes
        """

        result = self.__call__(
            config,
            save=save,
            cache_path=cache_path,
            force=force,
            use_cache=use_cache,
            **kwargs,
        )
        return result.plot_transmission(plot_config=plot_config)

    def visualize_mode(
        self,
        f: Optional[float] = None,
        *,
        k: Optional[int] = None,
        result: Optional[TransmissionResult] = None,
        compute_kwargs: Optional[dict[str, Any]] = None,
        save: bool = False,
        cache_path: Optional[Union[str, Path]] = None,
        force: bool = False,
        use_cache: bool = True,
        **visualize_kwargs,
    ):
        """Compute (if needed) and visualize single-frequency spin-wave mode.

        Parameters
        ----------
        f, k
            Frequency selector forwarded to :meth:`TransmissionResult.visualize_mode`.
        result : TransmissionResult, optional
            Existing raw transmission result. If omitted, this method computes one.
        compute_kwargs : dict, optional
            Extra arguments passed to :meth:`__call__` when ``result`` is not provided.
            Useful for overriding defaults (filters, datasets, cache options, etc.).
        save, cache_path, force, use_cache
            Cache controls used only when auto-computing ``result``.
        **visualize_kwargs
            Extra arguments forwarded to :meth:`TransmissionResult.visualize_mode`.
        """
        if result is None:
            compute_args = dict(compute_kwargs or {})
            compute_args.setdefault("raw_fft_output", True)
            compute_args.setdefault("y_integration_mode", "none")
            compute_args.setdefault("normalize", "none")
            compute_args.setdefault("average_mode", "none")
            compute_args.setdefault("spatial_window", 1)
            compute_args.setdefault("spatial_step", 1)
            result = self.__call__(
                save=save,
                cache_path=cache_path,
                force=force,
                use_cache=use_cache,
                **compute_args,
            )
        return result.visualize_mode(f=f, k=k, **visualize_kwargs)

    def visualize_modes(
        self,
        frequencies,
        *,
        result: Optional[TransmissionResult] = None,
        compute_kwargs: Optional[dict[str, Any]] = None,
        save: bool = False,
        cache_path: Optional[Union[str, Path]] = None,
        force: bool = False,
        use_cache: bool = True,
        **visualize_kwargs,
    ):
        """Compute (if needed) and visualize multiple selected frequencies."""
        if result is None:
            compute_args = dict(compute_kwargs or {})
            compute_args.setdefault("raw_fft_output", True)
            compute_args.setdefault("y_integration_mode", "none")
            compute_args.setdefault("normalize", "none")
            compute_args.setdefault("average_mode", "none")
            compute_args.setdefault("spatial_window", 1)
            compute_args.setdefault("spatial_step", 1)
            result = self.__call__(
                save=save,
                cache_path=cache_path,
                force=force,
                use_cache=use_cache,
                **compute_args,
            )
        return result.visualize_modes(frequencies, **visualize_kwargs)

    def save_mode_visualizations(
        self,
        output_dir: Union[str, Path],
        *,
        result: Optional[TransmissionResult] = None,
        compute_kwargs: Optional[dict[str, Any]] = None,
        save: bool = False,
        cache_path: Optional[Union[str, Path]] = None,
        force: bool = False,
        use_cache: bool = True,
        **save_kwargs,
    ):
        """Compute (if needed) and save per-frequency mode visualizations."""
        if result is None:
            compute_args = dict(compute_kwargs or {})
            compute_args.setdefault("raw_fft_output", True)
            compute_args.setdefault("y_integration_mode", "none")
            compute_args.setdefault("normalize", "none")
            compute_args.setdefault("average_mode", "none")
            compute_args.setdefault("spatial_window", 1)
            compute_args.setdefault("spatial_step", 1)
            result = self.__call__(
                save=save,
                cache_path=cache_path,
                force=force,
                use_cache=use_cache,
                **compute_args,
            )
        return result.save_mode_visualizations(output_dir=output_dir, **save_kwargs)

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
            "  • visualize_mode()    - Reconstruct and plot single-frequency mode",
            "  • visualize_modes()   - Plot multiple selected frequencies",
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
            "  ",
            "  # Spin-wave mode visualization (auto raw FFT compute)",
            "  fig, ax, meta = job[0].fft.transmission.visualize_mode(f=2.5)",
            "",
            "⚙️ CONFIGURATION PARAMETERS (with available options):",
            "─" * 50,
        ]

        # Detailed parameter documentation with options
        param_docs = [
            (
                "spatial_window",
                "5",
                "int >= 0 (0/False => 1) - Size of spatial analysis window",
            ),
            ("spatial_step", "1", "int ≥ 1 - Step for sliding window"),
            (
                "reference_window",
                "None",
                "(int, int) or None - Reference region [start, end]",
            ),
            ("normalize", "'reference'", "'reference' | 'max' | 'none'"),
            ("method", "'power_ratio'", "'power_ratio' | 'circular' | 'cpsd'"),
            (
                "window_function",
                "'hann'",
                "'hann' | 'hamming' | 'blackman' | 'bartlett' | 'kaiser' | 'tukey' | 'gaussian' | 'none'",
            ),
            (
                "filter_type",
                "'remove_mean'",
                "'remove_mean' | 'remove_static' | 'detrend_linear' | 'remove_mean_and_static' | 'none'",
            ),
            ("reference_statistic", "'mean'", "'mean' | 'median' | 'max'"),
            ("average_mode", "'mean'", "'mean' | 'median' | 'edge_taper' | 'none'"),
            (
                "component_weights",
                "(1.0, 1.0, 0.1)",
                "(mx, my, mz) - Tuple[float, float, float]",
            ),
            (
                "enable_circular_components",
                "False",
                "bool - Enable m+ and m- components",
            ),
            ("store_component_maps", "False", "bool - Store individual component maps"),
            (
                "dataset_name",
                "None",
                "str or None - Dataset name (auto-detect if None)",
            ),
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
        summary_table.add_row(
            "🔊", "[bold cyan]Transmission Analysis Interface[/bold cyan]"
        )
        summary_table.add_row("📁", f"Job Dataset: [bold]{dataset_hint}[/bold]")
        summary_table.add_row("", "")
        summary_table.add_row("🔧", "[bold yellow]Core Methods:[/bold yellow]")
        summary_table.add_row(
            "", "  • [code]transmission()[/code] - Compute transmission"
        )
        summary_table.add_row(
            "", "  • [code]compute()[/code] - Alias for transmission()"
        )
        summary_table.add_row(
            "", "  • [code]plot_transmission()[/code] - Compute + plot"
        )

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
            ("spatial_window", "5", "int >= 0 (0/False => 1)"),
            ("spatial_step", "1", "int ≥ 1"),
            ("reference_window", "None", "(int, int) or None"),
            ("normalize", "'reference'", "'reference' | 'max' | 'none'"),
            ("method", "'power_ratio'", "'power_ratio' | 'circular' | 'cpsd'"),
            (
                "window_function",
                "'hann'",
                "'hann' | 'hamming' | 'blackman' | 'bartlett' | 'kaiser' | 'tukey' | 'gaussian' | 'none'",
            ),
            (
                "filter_type",
                "'remove_mean'",
                "'remove_mean' | 'remove_static' | 'detrend_linear' | 'remove_mean_and_static' | 'none'",
            ),
            ("reference_statistic", "'mean'", "'mean' | 'median' | 'max'"),
            ("average_mode", "'mean'", "'mean' | 'median' | 'edge_taper' | 'none'"),
            ("component_weights", "(1.0, 1.0, 0.1)", "(mx, my, mz) weights"),
            ("enable_circular_components", "False", "bool - m±"),
            ("store_component_maps", "False", "bool"),
            ("dataset_name", "None", "str or None"),
            ("z_layer", "-1", "int (negative = from end)"),
            ("tmax", "None", "int or None"),
        ]

        for param, default, options in param_info:
            config_table.add_row(param, default, options)

        config_panel = Panel(
            config_table,
            border_style="green",
            title="[bold]Configuration Options[/bold]",
        )

        console.print(summary_panel)
        console.print(usage_panel)
        console.print(config_panel)

        return buffer.getvalue()


__all__ = ["FFTTransmissionInterface"]
