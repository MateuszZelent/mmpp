"""
FFT Core Module

Main FFT class providing unified interface for FFT analysis.
"""

from typing import Optional, Dict, List, Union, Any, Tuple
import numpy as np

# Import from our own modules
from .compute_fft import FFTCompute, FFTComputeResult
from .plot import FFTPlotter

# Import mode visualization capabilities
try:
    from .modes import FFTModeInterface, FMRModeAnalyzer, ModeVisualizationConfig

    MODES_AVAILABLE = True
except ImportError:
    MODES_AVAILABLE = False


class FFT:
    """
    Main FFT analysis class providing numpy.fft-like interface.

    This class aggregates FFT computation and plotting capabilities
    for MMPP job results.
    """

    def __init__(self, job_result, mmpp_instance: Optional[Any] = None):
        """
        Initialize FFT analyzer for a job result.

        Parameters:
        -----------
        job_result : ZarrJobResult
            Job result to analyze
        mmpp_instance : MMPP, optional
            Reference to parent MMPP instance
        """
        self.job_result = job_result
        self.mmpp = mmpp_instance

        # Initialize compute engine with debug mode from parent MMPP if available
        debug_mode = getattr(mmpp_instance, "debug", False) if mmpp_instance else False
        self._compute = FFTCompute(debug=debug_mode)

        # Initialize plotter (lazy loaded)
        self._plotter = None

        # Cache for FFT results
        self._cache = {}

    @property
    def plotter(self) -> FFTPlotter:
        """Get plotter instance (lazy initialization)."""
        if self._plotter is None:
            self._plotter = FFTPlotter([self.job_result], self.mmpp)
        return self._plotter

    def _get_cache_key(
        self, dataset_name: str, z_layer: int, method: int, **kwargs
    ) -> str:
        """Generate cache key for FFT results."""
        key_parts = [dataset_name, str(z_layer), str(method)]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        return "|".join(key_parts)

    def _compute_fft(
        self,
        dataset_name: str = "m_z11",
        z_layer: int = -1,
        method: int = 1,
        use_cache: bool = True,
        save: bool = False,
        force: bool = False,
        save_dataset_name: Optional[str] = None,
        **kwargs,
    ) -> FFTComputeResult:
        """
        Compute FFT with caching and optional saving.

        Parameters:
        -----------
        dataset_name : str, optional
            Dataset name (default: "m_z11")
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        use_cache : bool, optional
            Use memory cache (default: True)
        save : bool, optional
            Save result to zarr file (default: False)
        force : bool, optional
            Force recalculation and overwrite existing (default: False)
        save_dataset_name : str, optional
            Custom name for saved dataset (default: auto-generated)
        **kwargs : Any
            Additional FFT configuration options

        Returns:
        --------
        FFTComputeResult
            FFT computation result
        """
        cache_key = self._get_cache_key(dataset_name, z_layer, method, **kwargs)

        # Check memory cache only if not forcing and not saving
        if use_cache and not force and not save and cache_key in self._cache:
            return self._cache[cache_key]

        try:
            result = self._compute.calculate_fft_data(
                self.job_result.path,
                dataset_name,
                z_layer,
                method,
                save=save,
                force=force,
                save_dataset_name=save_dataset_name,
                **kwargs,
            )
        except OSError as e:
            if "directory not empty" in str(e).lower():
                print(
                    "Warning: FFT directory already exists and is not empty. Use force=True to overwrite."
                )
            raise

        # Cache result only if not forcing
        if use_cache and not force:
            self._cache[cache_key] = result

        return result

    def spectrum(
        self,
        dset: str = "m_z11",
        z_layer: int = -1,
        method: int = 1,
        save: bool = False,
        force: bool = False,
        save_dataset_name: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute FFT spectrum.

        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m_z11")
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        save : bool, optional
            Save result to zarr file (default: False)
        force : bool, optional
            Force recalculation and overwrite existing (default: False)
        save_dataset_name : str, optional
            Custom name for saved dataset (default: auto-generated)
        **kwargs : Any
            Additional FFT configuration options

        Returns:
        --------
        np.ndarray
            Complex FFT spectrum
        """
        result = self._compute_fft(
            dset,
            z_layer,
            method,
            save=save,
            force=force,
            save_dataset_name=save_dataset_name,
            **kwargs,
        )
        return result.spectrum

    def frequencies(
        self,
        dset: str = "m_z11",
        z_layer: int = -1,
        method: int = 1,
        save: bool = False,
        force: bool = False,
        save_dataset_name: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Get frequency array for FFT.

        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m_z11")
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        save : bool, optional
            Save result to zarr file (default: False)
        force : bool, optional
            Force recalculation and overwrite existing (default: False)
        save_dataset_name : str, optional
            Custom name for saved dataset (default: auto-generated)
        **kwargs : Any
            Additional FFT configuration options

        Returns:
        --------
        np.ndarray
            Frequency array
        """
        result = self._compute_fft(
            dset,
            z_layer,
            method,
            save=save,
            force=force,
            save_dataset_name=save_dataset_name,
            **kwargs,
        )
        return result.frequencies

    def power(
        self,
        dset: str = "m_z11",
        z_layer: int = -1,
        method: int = 1,
        save: bool = False,
        force: bool = False,
        save_dataset_name: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute power spectrum.

        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m_z11")
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        save : bool, optional
            Save result to zarr file (default: False)
        force : bool, optional
            Force recalculation and overwrite existing (default: False)
        save_dataset_name : str, optional
            Custom name for saved dataset (default: auto-generated)
        **kwargs : Any
            Additional FFT configuration options

        Returns:
        --------
        np.ndarray
            Power spectrum (|FFT|^2)
        """
        spectrum = self.spectrum(
            dset,
            z_layer,
            method,
            save=save,
            force=force,
            save_dataset_name=save_dataset_name,
            **kwargs,
        )
        return np.abs(spectrum) ** 2

    def phase(
        self, dset: str = "m_z11", z_layer: int = -1, method: int = 1, **kwargs
    ) -> np.ndarray:
        """
        Compute phase spectrum.

        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m_z11")
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        **kwargs : Any
            Additional FFT configuration options

        Returns:
        --------
        np.ndarray
            Phase spectrum
        """
        spectrum = self.spectrum(dset, z_layer, method, **kwargs)
        return np.angle(spectrum)

    def magnitude(
        self, dset: str = "m_z11", z_layer: int = -1, method: int = 1, **kwargs
    ) -> np.ndarray:
        """
        Compute magnitude spectrum.

        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m_z11")
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        \\*\\*kwargs : Any
            Additional FFT configuration options

        Returns:
        --------
        np.ndarray
            Magnitude spectrum (\\|FFT\\|)
        """
        spectrum = self.spectrum(dset, z_layer, method, **kwargs)
        return np.abs(spectrum)

    def plot_spectrum(
        self,
        dset: str = "m_z11",
        method: int = 1,
        z_layer: int = -1,
        log_scale: bool = True,
        normalize: bool = False,
        save: bool = True,
        force: bool = False,
        save_dataset_name: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """
        Plot power spectrum.

        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m_z11")
        method : int, optional
            FFT method (default: 1)
        z_layer : int, optional
            Z-layer (default: -1)
        log_scale : bool, optional
            Use logarithmic scale (default: True)
        normalize : bool, optional
            Normalize spectrum (default: False)
        save : bool, optional
            Save FFT result to zarr file (default: True)
        force : bool, optional
            Force recalculation and overwrite existing (default: False)
        save_dataset_name : str, optional
            Custom name for saved dataset (default: auto-generated)
        **kwargs : Any
            Additional plotting options

        Returns:
        --------
        tuple
            (figure, axes) matplotlib objects
        """
        return self.plotter.power_spectrum(
            dataset_name=dset,
            method=method,
            z_layer=z_layer,
            log_scale=log_scale,
            normalize=normalize,
            save=save,
            force=force,
            save_dataset_name=save_dataset_name,
            **kwargs,
        )

    def clear_cache(self):
        """Clear FFT computation cache."""
        self._cache.clear()

    def __repr__(self) -> str:
        """Rich documentation display for FFT interface."""
        try:
            return self._rich_fft_display()
        except Exception:
            return self._basic_fft_display()

    def _rich_fft_display(self) -> str:
        """Create rich documentation display with panels and proper styling."""
        try:
            from rich.console import Console
            from rich.text import Text
            from rich.panel import Panel
            from rich.columns import Columns
            from rich.table import Table
            from rich.syntax import Syntax
            import io

            console = Console(file=io.StringIO(), width=120, force_terminal=True)

            # Get basic info
            path = self.job_result.path
            cache_size = len(self._cache)
            has_modes = MODES_AVAILABLE

            # Summary panel content
            summary_text = Text()
            summary_text.append("🔬 MMPP FFT Analysis Interface\n", style="bold cyan")
            summary_text.append(f"📁 Job Path: {path}\n", style="dim")
            summary_text.append(f"💾 Cache Entries: {cache_size}\n", style="dim")
            summary_text.append(
                f"🎯 Mode Analysis: {'✓ Available' if has_modes else '✗ Unavailable'}\n",
                style="green" if has_modes else "red",
            )

            # Core methods panel content
            core_methods_text = Text()
            core_methods_text.append("🔧 Core FFT Methods:\n", style="bold yellow")
            methods = [
                ("spectrum()", "Get complex FFT spectrum"),
                ("frequencies()", "Get frequency array"),
                ("power()", "Get power spectrum |FFT|²"),
                ("magnitude()", "Get magnitude |FFT|"),
                ("phase()", "Get phase spectrum"),
                ("plot_spectrum()", "Plot power spectrum"),
                ("clear_cache()", "Clear computation cache"),
            ]

            for method, desc in methods:
                core_methods_text.append("  • ", style="dim")
                core_methods_text.append(method, style="code")
                core_methods_text.append(f" - {desc}\n", style="dim")

            # Mode methods panel content (if available)
            if has_modes:
                mode_methods_text = Text()
                mode_methods_text.append(
                    "🌊 Mode Analysis Methods:\n", style="bold blue"
                )
                mode_methods = [
                    ("modes", "Access mode interface"),
                    ("[index]", "Index-based mode access"),
                    ("plot_modes(frequency)", "Plot modes at frequency"),
                    ("interactive_spectrum()", "Interactive spectrum+modes"),
                ]

                for method, desc in mode_methods:
                    mode_methods_text.append("  • ", style="dim")
                    mode_methods_text.append(method, style="code")
                    mode_methods_text.append(f" - {desc}\n", style="dim")
            else:
                mode_methods_text = Text()
                mode_methods_text.append(
                    "🌊 Mode Analysis: Not Available\n", style="bold red"
                )
                mode_methods_text.append(
                    "Install mode visualization dependencies to enable", style="dim"
                )

            # Parameters table
            params_table = Table(show_header=False, box=None, padding=(0, 1))
            params_table.add_column("Parameter", style="bold yellow")
            params_table.add_column("Description", style="white")
            params_table.add_column("Values", style="cyan")

            params = [
                ("dset", "Dataset name", "'m_z11', 'm_x11', 'm_y11'"),
                ("z_layer", "Z-layer index", "-1 (top), 0 (bottom), 1, 2, ..."),
                ("method", "FFT method", "1 (default), 2, 3"),
                ("save", "Save to zarr", "True/False"),
                ("force", "Force recalculation", "True/False"),
            ]

            for param, desc, values in params:
                params_table.add_row(param, desc, values)

            # Usage examples
            example_code = """# Basic FFT operations
power = op[0].fft.power('m_z11')
freqs = op[0].fft.frequencies()
spectrum = op[0].fft.spectrum(save=True, force=True)

# Plotting
fig, ax = op[0].fft.plot_spectrum(log_scale=True)

# Mode analysis (if available)
op[0].fft.modes.interactive_spectrum()
op[0].fft[0][200].plot_modes()  # Elegant syntax
op[0].fft.plot_modes(frequency=1.5)

# Advanced usage
op[0].fft.plotter.power_spectrum(normalize=True)
help(op[0].fft.spectrum)  # Detailed documentation"""

            syntax = Syntax(
                example_code, "python", theme="monokai", background_color="default"
            )

            # Build panels
            with console.capture() as capture:
                # Main summary panel
                console.print(
                    Panel.fit(
                        summary_text,
                        title="[bold cyan]MMPP FFT Interface[/bold cyan]",
                        border_style="cyan",
                    )
                )
                console.print("")

                # Method panels side by side
                console.print(
                    Columns(
                        [
                            Panel.fit(
                                core_methods_text,
                                title="[bold yellow]Core Methods[/bold yellow]",
                                border_style="yellow",
                            ),
                            Panel.fit(
                                mode_methods_text,
                                title="[bold blue]Mode Methods[/bold blue]",
                                border_style="blue" if has_modes else "red",
                            ),
                        ]
                    )
                )
                console.print("")

                # Parameters panel
                console.print(
                    Panel.fit(
                        params_table,
                        title="[bold green]Common Parameters[/bold green]",
                        border_style="green",
                    )
                )
                console.print("")

                # Examples panel
                console.print(
                    Panel.fit(
                        syntax,
                        title="[bold magenta]Usage Examples[/bold magenta]",
                        border_style="magenta",
                    )
                )

            return capture.get()

        except Exception:
            # Fallback to basic text display if rich fails
            return self._basic_fft_display_enhanced()

    def _basic_fft_display(self) -> str:
        """Fallback basic display if rich display fails."""
        return f"FFT(path='{self.job_result.path}', cache_entries={len(self._cache)})"

    def _basic_fft_display_enhanced(self) -> str:
        """Enhanced fallback display with more details if rich display fails."""
        path = self.job_result.path
        cache_size = len(self._cache)
        has_modes = MODES_AVAILABLE

        output = []
        output.append("=" * 70)
        output.append("🔬 MMPP FFT Analysis Interface")
        output.append("=" * 70)
        output.append(f"📁 Job Path: {path}")
        output.append(f"💾 Cache Entries: {cache_size}")
        output.append(
            f"🎯 Mode Analysis: {'✓ Available' if has_modes else '✗ Unavailable'}"
        )
        output.append("")

        # Core FFT Methods
        output.append("🔧 CORE FFT METHODS:")
        output.append("─" * 50)
        methods = [
            (
                "spectrum()",
                "Get complex FFT spectrum",
                "op[0].fft.spectrum('m_z11', z_layer=-1)",
            ),
            ("frequencies()", "Get frequency array", "op[0].fft.frequencies()"),
            ("power()", "Get power spectrum |FFT|²", "op[0].fft.power()"),
            ("magnitude()", "Get magnitude |FFT|", "op[0].fft.magnitude()"),
            ("phase()", "Get phase spectrum", "op[0].fft.phase()"),
            (
                "plot_spectrum()",
                "Plot power spectrum",
                "fig, ax = op[0].fft.plot_spectrum()",
            ),
            ("clear_cache()", "Clear computation cache", "op[0].fft.clear_cache()"),
        ]

        for method, desc, example in methods:
            output.append(f"  • {method:<15} {desc}")
            output.append(f"    └─ {example}")

        output.append("")

        # Mode Analysis (if available)
        if has_modes:
            output.append("🌊 MODE ANALYSIS METHODS:")
            output.append("─" * 50)
            mode_methods = [
                (
                    "modes",
                    "Access mode interface",
                    "op[0].fft.modes.interactive_spectrum()",
                ),
                (
                    "[index]",
                    "Index-based mode access",
                    "op[0].fft[0][200].plot_modes()",
                ),
                (
                    "plot_modes()",
                    "Plot modes at frequency",
                    "op[0].fft.plot_modes(frequency=1.5)",
                ),
                (
                    "interactive_spectrum()",
                    "Interactive spectrum+modes",
                    "op[0].fft.interactive_spectrum()",
                ),
            ]

            for method, desc, example in mode_methods:
                output.append(f"  • {method:<20} {desc}")
                output.append(f"    └─ {example}")
        else:
            output.append("🌊 MODE ANALYSIS: Not Available")
            output.append("   Install mode visualization dependencies to enable")

        output.append("")

        # Common Parameters
        output.append("⚙️  COMMON PARAMETERS:")
        output.append("─" * 50)
        params = [
            ("dset", "Dataset name", "'m_z11', 'm_x11', 'm_y11'"),
            ("z_layer", "Z-layer index", "-1 (top), 0 (bottom), 1, 2, ..."),
            ("method", "FFT method", "1 (default), 2, 3"),
            ("save", "Save to zarr", "True/False"),
            ("force", "Force recalculation", "True/False"),
        ]

        for param, desc, values in params:
            output.append(f"  • {param:<12} {desc:<20} {values}")

        output.append("")

        # Quick Examples
        output.append("🚀 QUICK START EXAMPLES:")
        output.append("─" * 50)
        examples = [
            "# Basic FFT operations",
            "power = op[0].fft.power('m_z11')",
            "freqs = op[0].fft.frequencies()",
            "spectrum = op[0].fft.spectrum(save=True, force=True)",
            "",
            "# Plotting",
            "fig, ax = op[0].fft.plot_spectrum(log_scale=True)",
            "",
            "# Mode analysis (if available)",
            "op[0].fft.modes.interactive_spectrum()",
            "op[0].fft[0][200].plot_modes()  # Elegant syntax",
            "op[0].fft.plot_modes(frequency=1.5)",
            "",
            "# Advanced usage",
            "op[0].fft.plotter.power_spectrum(normalize=True)",
            "help(op[0].fft.spectrum)  # Detailed documentation",
        ]

        for example in examples:
            output.append(f"  {example}")

        output.append("")
        output.append("=" * 70)
        output.append("📖 For detailed docs: help(op[0].fft.spectrum)")
        output.append("🔧 Clear cache: op[0].fft.clear_cache()")
        output.append("=" * 70)

        return "\n".join(output)

    @property
    def modes(self) -> "FFTModeInterface":
        """
        Get mode visualization interface.

        Returns:
        --------
        FFTModeInterface
            Interface for mode operations

        Examples:
        ---------
        >>> op[0].fft.modes.interactive_spectrum()
        >>> op[0].fft.modes.plot_modes(frequency=1.5)
        >>> op[0].fft[0][200].plot_modes()  # Elegant syntax
        """
        if not MODES_AVAILABLE:
            raise ImportError(
                "Mode visualization not available. Check modes module import."
            )

        if not hasattr(self, "_mode_interface"):
            self._mode_interface = FFTModeInterface(0, self)
        return self._mode_interface

    def __getitem__(self, index: int) -> "FFTModeInterface":
        """
        Get FFT result by index for mode operations.

        Parameters:
        -----------
        index : int
            FFT result index (usually 0 for latest)

        Returns:
        --------
        FFTModeInterface
            Interface for mode operations at specific FFT result

        Examples:
        ---------
        >>> op[0].fft[0].interactive_spectrum()
        >>> op[0].fft[0][200].plot_modes()
        """
        if not MODES_AVAILABLE:
            raise ImportError(
                "Mode visualization not available. Check modes module import."
            )

        return FFTModeInterface(index, self)

    def plot_modes(
        self, frequency: float, dset: str = "m_z11", z_layer: int = 0, **kwargs
    ) -> Tuple[Any, Any]:
        """
        Plot FMR modes at specific frequency.

        Parameters:
        -----------
        frequency : float
            Frequency in GHz
        dset : str
            Dataset name
        z_layer : int
            Z-layer index
        **kwargs
            Additional arguments for mode plotting

        Returns:
        --------
        Tuple[Figure, np.ndarray]
            Matplotlib figure and axes
        """
        if not MODES_AVAILABLE:
            raise ImportError(
                "Mode visualization not available. Check modes module import."
            )

        # Create temporary mode analyzer
        debug_mode = getattr(self.mmpp, "debug", False) if self.mmpp else False
        analyzer = FMRModeAnalyzer(
            self.job_result.path, dataset_name=dset, debug=debug_mode
        )
        return analyzer.plot_modes(frequency=frequency, z_layer=z_layer, **kwargs)

    def interactive_spectrum(self, dset: str = "m_z11", **kwargs) -> Any:
        """
        Create interactive spectrum plot with mode visualization.

        Parameters:
        -----------
        dset : str
            Dataset name
        **kwargs
            Additional arguments for interactive plotting

        Returns:
        --------
        Figure
            Interactive matplotlib figure
        """
        if not MODES_AVAILABLE:
            raise ImportError(
                "Mode visualization not available. Check modes module import."
            )

        # Create temporary mode analyzer
        debug_mode = getattr(self.mmpp, "debug", False) if self.mmpp else False
        analyzer = FMRModeAnalyzer(
            self.job_result.path, dataset_name=dset, debug=debug_mode
        )
        return analyzer.interactive_spectrum(**kwargs)
