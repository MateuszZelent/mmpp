"""
FFT Core Module

Main FFT class providing unified interface for FFT analysis.
"""

from typing import Any, Optional

import numpy as np

# Import from our own modules
from .compute_fft import FFTCompute, FFTComputeResult
from .plot import FFTPlotter
from .transmission.interface import FFTTransmissionInterface

# Import mode visualization capabilities
try:
    from .modes import FFTModeInterface, FMRModeAnalyzer, ModeVisualizationConfig

    MODES_AVAILABLE = True
except ImportError:
    MODES_AVAILABLE = False

# Import dispersion analysis capabilities
try:
    from .dispersion import SpinWaveAnalyzer, DispersionConfig, FFTDispersionInterface
    
    DISPERSION_AVAILABLE = True
except ImportError:
    DISPERSION_AVAILABLE = False


class FFT:
    """
    Main FFT analysis class providing numpy.fft-like interface.

    This class aggregates FFT computation and plotting capabilities
    for MMPP job results.
    """
    
    # Feature availability flags
    MODES_AVAILABLE = MODES_AVAILABLE
    DISPERSION_AVAILABLE = DISPERSION_AVAILABLE

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

        # Transmission interface (lazy)
        self._transmission_interface = None

        # Cache for FFT results
        self._cache = {}

    @property
    def plotter(self) -> FFTPlotter:
        """Get plotter instance (lazy initialization)."""
        if self._plotter is None:
            self._plotter = FFTPlotter([self.job_result], self.mmpp)
        return self._plotter

    @property
    def transmission(self) -> FFTTransmissionInterface:
        """Transmission analysis helper."""

        if self._transmission_interface is None:
            self._transmission_interface = FFTTransmissionInterface(
                self,
                self._compute,
                self.job_result,
            )
        return self._transmission_interface

    def _get_cache_key(
        self, dataset_name: str, z_layer: int, method: int, **kwargs
    ) -> str:
        """Generate cache key for FFT results."""
        # Normalize z_layer for consistent cache keys
        # For cache purposes, we use the raw z_layer value since the actual normalization
        # happens in calculate_fft_data and we want consistent caching behavior
        key_parts = [dataset_name, str(z_layer), str(method)]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        return "|".join(key_parts)

    def _compute_fft(
        self,
        dataset_name: Optional[str] = None,
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
            Dataset name (default: auto-select largest m dataset)
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
        # Auto-select largest m dataset if none specified
        if dataset_name is None:
            dataset_name = self.job_result.get_largest_m_dataset()

        if not isinstance(dataset_name, str):
            dataset_name = str(dataset_name)

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
        # Try to compute frequencies efficiently without loading data
        try:
            return self._compute_frequencies_fast(dset, **kwargs)
        except Exception:
            # Fallback to full FFT computation
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

    def _compute_frequencies_fast(
        self,
        dataset_name: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute frequency array without loading full dataset.
        Only reads metadata (dt and shape) from zarr.
        """
        # Auto-select largest m dataset if none specified
        if dataset_name is None:
            dataset_name = self.job_result.get_largest_m_dataset()

        if not isinstance(dataset_name, str):
            dataset_name = str(dataset_name)

        # Get dt and shape without loading data
        try:
            from .compute_fft import PYZFN_AVAILABLE
            if not PYZFN_AVAILABLE:
                raise ImportError("pyzfn required")
            
            from pyzfn import Pyzfn
            job = Pyzfn(self.job_result.path)
            
            # Get dataset
            data_set = None
            if hasattr(job, dataset_name):
                data_set = getattr(job, dataset_name)
            else:
                z_group = getattr(job, "z", None)
                if z_group is not None and dataset_name in z_group:
                    data_set = z_group[dataset_name]
            
            if data_set is None:
                raise ValueError(f"Dataset {dataset_name} not found")
            
            # Get shape (without loading data)
            data_shape = data_set.shape
            n_timesteps = data_shape[0]
            
            # Get dt from job metadata
            dt = float(job.dt)
            
            # Determine FFT length (same logic as in compute_fft)
            fft_length = n_timesteps
            
            zero_padding = kwargs.get("zero_padding", self._compute.config.zero_padding)
            nfft = kwargs.get("nfft", self._compute.config.nfft)
            
            if nfft is not None:
                fft_length = nfft
            elif zero_padding:
                next_power_two = 1 << (n_timesteps - 1).bit_length()
                if next_power_two > n_timesteps:
                    fft_length = next_power_two
            
            # Compute frequencies
            frequencies = np.fft.rfftfreq(fft_length, dt)
            return frequencies
            
        except Exception as e:
            raise RuntimeError(f"Failed to compute frequencies from metadata: {e}") from e

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
    ) -> tuple[Any, Any]:
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
            import io

            from rich.columns import Columns
            from rich.console import Console
            from rich.panel import Panel
            from rich.syntax import Syntax
            from rich.table import Table
            from rich.text import Text

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

            # Plotting methods panel content
            plotting_methods_text = Text()
            plotting_methods_text.append("📈 Plotting Toolkit:\n", style="bold magenta")
            plotting_methods = [
                ("plot_spectrum(log_scale=True)", "Quick-look power spectrum"),
                ("plotter.power_spectrum(normalize=True)", "Overlay multiple results"),
                (
                    "plotter.power_spectrum(save_path='fft.png')",
                    "Export publication figure",
                ),
                ("plot_modes(frequency=..., z_layer=-1)", "Static mode grid"),
                ("modes.save_modes_animation()", "Animated mode evolution"),
            ]

            for method, desc in plotting_methods:
                plotting_methods_text.append("  • ", style="dim")
                plotting_methods_text.append(method, style="code")
                plotting_methods_text.append(f" - {desc}\n", style="dim")

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
                (
                    "dset",
                    "Dataset name",
                    "Auto-selected or explicit: 'm_z11', 'm_x11', 'm_y11'",
                ),
                ("z_layer", "Z-layer index", "-1 (top), 0 (bottom), 1, 2, ..."),
                ("method", "FFT method", "1 (default), 2"),
                ("save", "Save to zarr", "True/False"),
                ("force", "Force recalculation", "True/False"),
                ("zero_padding", "Pad to power-of-two length", "True/False"),
                ("nfft", "Manual FFT length", "int or None"),
            ]

            for param, desc, values in params:
                params_table.add_row(param, desc, values)

            # Usage examples
            example_code = """# Basic FFT operations (auto-selects optimal dataset)
power = job[0].fft.power()
freqs = job[0].fft.frequencies()
spectrum = job[0].fft.spectrum(save=True, force=True)

# Or specify dataset explicitly
power = job[0].fft.power('m_z11')

# Plotting
fig, ax = job[0].fft.plot_spectrum(log_scale=True)
job[0].fft.plotter.power_spectrum(save_path='fft_publication.png')

# Mode analysis (if available)
job[0].fft.modes.interactive_spectrum()
job[0].fft[0][200].plot_modes()  # Elegant syntax
job[0].fft.plot_modes(frequency=1.5)

# Advanced usage
job[0].fft.plotter.power_spectrum(normalize=True)
job[0].fft.modes.save_modes_animation(frequency=10.4, save_path='mode.gif')
help(job[0].fft.spectrum)  # Detailed documentation"""

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
                                plotting_methods_text,
                                title="[bold magenta]Plotting[/bold magenta]",
                                border_style="magenta",
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
                "job[0].fft.spectrum('m_z11', z_layer=-1)",
            ),
            ("frequencies()", "Get frequency array", "job[0].fft.frequencies()"),
            ("power()", "Get power spectrum |FFT|²", "job[0].fft.power()"),
            ("magnitude()", "Get magnitude |FFT|", "job[0].fft.magnitude()"),
            ("phase()", "Get phase spectrum", "job[0].fft.phase()"),
            (
                "plot_spectrum()",
                "Plot power spectrum",
                "fig, ax = job[0].fft.plot_spectrum()",
            ),
            ("clear_cache()", "Clear computation cache", "job[0].fft.clear_cache()"),
        ]

        for method, desc, example in methods:
            output.append(f"  • {method:<15} {desc}")
            output.append(f"    └─ {example}")

        output.append("")

        # Plotting toolkit
        output.append("📈 PLOTTING TOOLKIT:")
        output.append("─" * 50)
        plot_methods = [
            (
                "plot_spectrum(log_scale=True)",
                "Quick-look spectrum",
                "job[0].fft.plot_spectrum(log_scale=True)",
            ),
            (
                "plotter.power_spectrum(normalize=True)",
                "Overlay multiple jobs",
                "job[0].fft.plotter.power_spectrum(normalize=True)",
            ),
            (
                "plotter.power_spectrum(save_path='fft.png')",
                "Export PNG/ publication",
                "job[0].fft.plotter.power_spectrum(save_path='fft.png')",
            ),
            (
                "plot_modes(frequency, z_layer)",
                "Static mode panels",
                "job[0].fft.plot_modes(frequency=10.4, z_layer=-1)",
            ),
            (
                "modes.save_modes_animation()",
                "Animated mode evolution",
                "job[0].fft.modes.save_modes_animation(frequency=10.4)",
            ),
        ]

        for method, desc, example in plot_methods:
            output.append(f"  • {method:<40} {desc}")
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
                    "job[0].fft.modes.interactive_spectrum()",
                ),
                (
                    "[index]",
                    "Index-based mode access",
                    "job[0].fft[0][200].plot_modes()",
                ),
                (
                    "plot_modes()",
                    "Plot modes at frequency",
                    "job[0].fft.plot_modes(frequency=1.5)",
                ),
                (
                    "interactive_spectrum()",
                    "Interactive spectrum+modes",
                    "job[0].fft.interactive_spectrum()",
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
            ("method", "FFT method", "1 (default), 2"),
            ("save", "Save to zarr", "True/False"),
            ("force", "Force recalculation", "True/False"),
            ("zero_padding", "Pad to power-of-two length", "True/False"),
            ("nfft", "Manual FFT length", "int or None"),
        ]

        for param, desc, values in params:
            output.append(f"  • {param:<12} {desc:<20} {values}")

        output.append("")

        # Quick Examples
        output.append("🚀 QUICK START EXAMPLES:")
        output.append("─" * 50)
        examples = [
            "# Basic FFT operations",
            "power = job[0].fft.power('m_z11')",
            "freqs = job[0].fft.frequencies()",
            "spectrum = job[0].fft.spectrum(save=True, force=True)",
            "",
            "# Plotting",
            "fig, ax = job[0].fft.plot_spectrum(log_scale=True)",
            "job[0].fft.plotter.power_spectrum(save_path='fft_publication.png')",
            "",
            "# Mode analysis (if available)",
            "job[0].fft.modes.interactive_spectrum()",
            "job[0].fft[0][200].plot_modes()  # Elegant syntax",
            "job[0].fft.plot_modes(frequency=1.5)",
            "",
            "# Advanced usage",
            "job[0].fft.plotter.power_spectrum(normalize=True)",
            "job[0].fft.modes.save_modes_animation(frequency=10.4, save_path='mode.gif')",
            "help(job[0].fft.spectrum)  # Detailed documentation",
        ]

        for example in examples:
            output.append(f"  {example}")

        output.append("")
        output.append("=" * 70)
        output.append("📖 For detailed docs: help(job[0].fft.spectrum)")
        output.append("🔧 Clear cache: job[0].fft.clear_cache()")
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
        >>> job[0].fft.modes.interactive_spectrum()
        >>> job[0].fft.modes.plot_modes(frequency=1.5)
        >>> job[0].fft[0][200].plot_modes()  # Elegant syntax
        """
        if not MODES_AVAILABLE:
            raise ImportError(
                "Mode visualization not available. Check modes module import."
            )

        if not hasattr(self, "_mode_interface"):
            self._mode_interface = FFTModeInterface(0, self)
        return self._mode_interface

    @property
    def dispersion(self) -> "FFTDispersionInterface":
        """
        Get spin-wave dispersion analysis interface.

        Returns:
        --------
        FFTDispersionInterface
            Interface for dispersion operations

        Examples:
        ---------
        >>> job[0].fft.dispersion.plot_dispersion()
        >>> job[0].fft.dispersion.compute_1d(axis="x")
        >>> job[0].m_layer.fft.dispersion.plot_branch()
        """
        if not DISPERSION_AVAILABLE:
            raise ImportError(
                "Dispersion analysis not available. Check dispersion module import."
            )

        if not hasattr(self, "_dispersion_interface"):
            self._dispersion_interface = FFTDispersionInterface(self)
        return self._dispersion_interface

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
        >>> job[0].fft[0].interactive_spectrum()
        >>> job[0].fft[0][200].plot_modes()
        """
        if not MODES_AVAILABLE:
            raise ImportError(
                "Mode visualization not available. Check modes module import."
            )

        return FFTModeInterface(index, self)

    def plot_modes(
        self, frequency: float, dset: str = "m_z11", z_layer: int = 0, **kwargs
    ) -> tuple[Any, Any]:
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
        log_level = getattr(self.mmpp, "log_level", None) if self.mmpp else None
        analyzer = FMRModeAnalyzer(
            self.job_result.path, dataset_name=dset, debug=debug_mode, log_level=log_level
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
        log_level = getattr(self.mmpp, "log_level", None) if self.mmpp else None
        analyzer = FMRModeAnalyzer(
            self.job_result.path, dataset_name=dset, debug=debug_mode, log_level=log_level
        )
        return analyzer.interactive_spectrum(**kwargs)
