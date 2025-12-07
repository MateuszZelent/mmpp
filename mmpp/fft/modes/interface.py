"""
FFT Mode Interface Module

Provides FFTModeInterface for elegant job[0].fft.modes syntax.
Integrates with DatasetAwareWrapper for slice propagation.
"""

from typing import Any, Optional, Union
import logging

log = logging.getLogger("mmpp.fft.modes")

# Lazy imports to avoid circular dependencies
def _get_data_loader():
    from .data_loader import ModeDataLoader, ModeDataContext
    return ModeDataLoader, ModeDataContext

def _get_interactive():
    from .interactive import InteractiveSpectrum
    return InteractiveSpectrum


class InteractiveSpectrumHelper:
    """Callable helper that shows documentation when accessed as property.
    
    When accessed (job.fft.modes.interactive_spectrum), displays helpful usage.
    When called (job.fft.modes.interactive_spectrum(...)), runs the method.
    """
    
    def __init__(self, modes_interface):
        self._modes = modes_interface
        self._method = modes_interface._interactive_spectrum_impl
    
    def __call__(self, **kwargs):
        """Delegate to actual interactive_spectrum method."""
        return self._method(**kwargs)
    
    def __repr__(self):
        return self._rich_display()
    
    def _rich_display(self) -> str:
        """Generate rich help display for interactive_spectrum."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
            from rich.syntax import Syntax
            from io import StringIO
            
            capture = StringIO()
            console = Console(file=capture, force_terminal=True, width=100)
            
            # Title
            title = Text()
            title.append("🎯 Interactive Spectrum Visualization\n", style="bold cyan")
            title.append(f"📁 Dataset: {self._modes.dataset_name}\n", style="dim")
            if self._modes.component_label:
                title.append(f"📊 Component: {self._modes.component_label}", style="green")
            
            console.print(Panel(title, border_style="cyan"))
            
            # Features
            features = Text()
            features.append("✨ Interactive Features:\n\n", style="bold yellow")
            feature_list = [
                ("Click", "Select frequency → update mode plots"),
                ("Right-click", "Snap to nearest peak"),
                ("Double-click", "Toggle mode animation on subplot"),
                ("'c' key", "Characterize current mode"),
                ("'s' key", "Save animated view"),
                ("'h' key", "Show help dialog"),
            ]
            for key, desc in feature_list:
                features.append(f"  • ", style="dim")
                features.append(f"{key:15}", style="bold green")
                features.append(f" {desc}\n", style="white")
            
            console.print(features)
            
            # Parameters table
            params = Table(show_header=True, header_style="bold magenta")
            params.add_column("Parameter", style="yellow")
            params.add_column("Type", style="cyan")
            params.add_column("Default", style="green")
            params.add_column("Description", style="white")
            
            param_data = [
                ("components", "list", "auto", "['x','y','z'] or [0,1,2]"),
                ("z_layer", "int", "-1", "Z-layer for modes (top layer)"),
                ("dpi", "int", "100", "Figure resolution"),
                ("figsize", "tuple", "(16,10)", "Figure size (width, height)"),
                ("log_scale", "bool", "False", "Logarithmic Y-scale"),
                ("normalize", "bool", "True", "Normalize power to max"),
                ("show_peaks", "bool", "True", "Detect and mark peaks"),
                ("saveanim", "str/bool", "None", "Path to save animation"),
                ("auto_animate", "bool", "False", "Auto-start all animations"),
            ]
            for p, t, d, desc in param_data:
                params.add_row(p, t, d, desc)
            
            console.print(params)
            console.print("")
            
            # Examples
            example = '''# Basic usage with component selection:
job[0].m[:200,...,1].fft.modes.interactive_spectrum(dpi=150)

# Full control:
job[0].fft.modes.interactive_spectrum(
    components=['x', 'y'],
    z_layer=-1,
    dpi=200,
    log_scale=True,
    show_peaks=True,
)

# With auto-animation:
job[0].fft.modes.interactive_spectrum(auto_animate=True)

# Save animation directly:
job[0].fft.modes.interactive_spectrum(saveanim="modes.mp4", auto_save=True)'''
            
            syntax = Syntax(example, "python", theme="monokai", line_numbers=False)
            console.print(Panel(syntax, title="[bold green]Examples", border_style="green"))
            
            return capture.getvalue()
        except ImportError:
            return "interactive_spectrum(...) - Interactive spectrum with mode visualization. Call with () to execute."


class FFTModeInterfaceNew:
    """Enhanced FFT interface with mode visualization capabilities.
    
    Supports slice propagation from DatasetAwareWrapper.
    Provides elegant syntax like: job[0].m[:200,...,1].fft.modes.interactive_spectrum()
    
    Attributes
    ----------
    fft_result_index : int
        Index of FFT result
    parent_fft : FFT
        Parent FFT instance
    _dataset_context : str, optional
        Dataset name from DatasetSpecificFFT
    _slice_context : tuple, optional
        Slice info from DatasetAwareWrapper
    """
    
    def __init__(self, fft_result_index: int, parent_fft: Any):
        """Initialize mode interface.
        
        Parameters
        ----------
        fft_result_index : int
            Index into parent FFT results
        parent_fft : FFT
            Parent FFT instance
        """
        self.fft_result_index = fft_result_index
        self.parent_fft = parent_fft
        
        # Context from DatasetSpecificFFT (set externally)
        self._dataset_context: Optional[str] = None
        self._slice_context: Optional[tuple] = None
        
        # Lazy-loaded instances
        self._data_loader = None
        self._mode_analyzer = None
    
    @property
    def zarr_path(self) -> str:
        """Get zarr path from parent FFT."""
        return self.parent_fft.job_result.path
    
    @property
    def dataset_name(self) -> str:
        """Get dataset name (from context or auto-detect)."""
        if self._dataset_context:
            return self._dataset_context
        # Auto-detect
        try:
            from ...plotting import _find_largest_m_dataset
            return _find_largest_m_dataset(self.zarr_path)
        except Exception:
            return "m"  # Fallback
    
    @property
    def component_index(self) -> Optional[int]:
        """Extract component index from slice_context."""
        if self._slice_context and isinstance(self._slice_context, tuple):
            last = self._slice_context[-1]
            if isinstance(last, int) and 0 <= last <= 2:
                return last
        return None
    
    @property
    def component_label(self) -> Optional[str]:
        """Get label for selected component."""
        labels = [r"$m_x$", r"$m_y$", r"$m_z$"]
        idx = self.component_index
        if idx is not None:
            return labels[idx]
        return None
    
    @property
    def spectrum_result(self):
        """Get spectrum using parent FFT with propagated slice context.
        
        This ensures consistency with job[0].m[...].fft.spectrum() calls.
        The slice_context (time slicing, component selection) is passed to
        the FFT spectrum calculation.
        
        Returns
        -------
        SpectrumResult
            Spectrum result with frequencies, power, peaks_info, component_label
        """
        return self.parent_fft._spectrum_impl(
            dset=self.dataset_name,
            slice_info=self._slice_context,
        )
    
    @property
    def frequencies(self):
        """Get frequencies from spectrum result (in GHz)."""
        return self.spectrum_result.frequencies
    
    @property
    def power_spectrum(self):
        """Get power spectrum (2D or 1D depending on component selection)."""
        return self.spectrum_result.power
    
    @property
    def data_loader(self):
        """Get or create data loader (lazy init)."""
        if self._data_loader is None:
            ModeDataLoader, ModeDataContext = _get_data_loader()
            
            context = ModeDataContext(
                zarr_path=self.zarr_path,
                dataset_name=self.dataset_name,
                slice_info=self._slice_context,
                component_index=self.component_index,
            )
            self._data_loader = ModeDataLoader(context)
            log.debug(f"Created data loader with dataset={self.dataset_name}, component={self.component_index}")
        
        return self._data_loader
    
    def _interactive_spectrum_impl(
        self,
        components: list = None,
        z_layer: int = -1,
        dpi: int = 100,
        figsize: tuple = (16, 10),
        log_scale: bool = False,
        normalize: bool = True,
        freq_unit: str = "GHz",
        show_peaks: bool = True,
        title: Optional[str] = None,
        initial_frequency: Optional[float] = None,
        **kwargs,
    ):
        """Create interactive spectrum with mode visualization panels.
        
        **Key feature:** Uses FFT.spectrum() for spectrum data, ensuring
        consistency with `job[0].m[:200,...,1].fft.spectrum()` calls.
        Slice context (time range, component) is automatically propagated.
        
        Split layout:
        - Left: FFT power spectrum with clickable peaks
        - Right: 3x3 mode grid (magnitude, phase, combined) for each component
        
        Parameters
        ----------
        components : list, optional
            Components to show: ['x', 'y', 'z'] or [0, 1, 2]
            If component was selected via slicing, defaults to that component.
        z_layer : int
            Z-layer for mode visualization (default: -1 = top)
        dpi : int
            Figure resolution (default: 100)
        figsize : tuple
            Figure size (width, height) in inches
        log_scale : bool
            Use logarithmic Y-scale
        normalize : bool
            Normalize power to maximum
        freq_unit : str
            Frequency unit: Hz, kHz, MHz, GHz, THz
        show_peaks : bool
            Detect and show peaks
        title : str, optional
            Custom plot title
        initial_frequency : float, optional
            Start with this frequency selected
        **kwargs
            Additional arguments (find_peaks params, etc.)
        
        Returns
        -------
        Figure
            Interactive matplotlib figure
        
        Examples
        --------
        >>> # Full interactive view with mode panels
        >>> job[0].fft.modes.interactive_spectrum()
        
        >>> # Single component (my) with slice propagation
        >>> job[0].m[:200,...,1].fft.modes.interactive_spectrum(dpi=150)
        
        >>> # Start at specific frequency
        >>> job[0].fft.modes.interactive_spectrum(initial_frequency=9.5)
        """
        # Get spectrum through parent FFT (respects slice_context!)
        # This ensures identical data to job[0].m[...].fft.spectrum()
        find_peaks_params = kwargs.pop('find_peaks', {'min_prominence': 0.01})
        spectrum_result = self.parent_fft._spectrum_impl(
            dset=self.dataset_name,
            slice_info=self._slice_context,
            find_peaks=find_peaks_params,
        )
        
        log.info(
            f"interactive_spectrum: using FFT spectrum with "
            f"dataset={self.dataset_name}, slice={self._slice_context}, "
            f"component={self.component_index}"
        )
        
        # Auto-select components based on slice context
        if components is None and self.component_index is not None:
            # User selected specific component via slicing
            component_names = ['x', 'y', 'z']
            components = [component_names[self.component_index]]
            log.info(f"Auto-selected component: {components[0]} (from slice context)")
        
        # Use FULL legacy FMRModeAnalyzer.interactive_spectrum with all features:
        # - Click to select frequency
        # - Right-click to snap to peak  
        # - Double-click to toggle animations
        # - Press 'c' to characterize mode
        # - Press 's' to save animation
        # - Press 'h' for help
        return self._legacy_analyzer.interactive_spectrum(
            components=components,
            z_layer=z_layer,
            spectrum_result=spectrum_result,  # Inject FFT spectrum!
            **kwargs
        )
    
    @property
    def interactive_spectrum(self):
        """Interactive spectrum with mode visualization (access for help, call to run).
        
        Access without () to see documentation. Call with () to execute.
        
        Examples
        --------
        >>> job[0].fft.modes.interactive_spectrum  # Show help
        >>> job[0].fft.modes.interactive_spectrum(dpi=150)  # Run
        """
        return InteractiveSpectrumHelper(self)
    
    # Alias for backward compatibility
    interactive_spectrum_old = interactive_spectrum
    
    # =========================================================================
    # Methods delegating to legacy FMRModeAnalyzer for features not yet migrated
    # =========================================================================
    
    @property
    def _legacy_analyzer(self):
        """Get legacy FMRModeAnalyzer for features not yet migrated."""
        if self._mode_analyzer is None:
            from . import FMRModeAnalyzer
            
            dataset = self._dataset_context or self.dataset_name
            self._mode_analyzer = FMRModeAnalyzer(
                zarr_path=self.zarr_path,
                dataset_name=dataset,
            )
        return self._mode_analyzer
    
    def plot_modes(
        self,
        frequency: float,
        z_layer: int = -1,
        component: str = "mz",
        show_phase: bool = True,
        show_magnitude: bool = True,
        dpi: int = 100,
        **kwargs,
    ):
        """Plot mode visualization at specified frequency.
        
        Parameters
        ----------
        frequency : float
            Frequency in GHz
        z_layer : int
            Z-layer index (default: -1 for top)
        component : str
            Component to visualize: 'mx', 'my', 'mz' (default: 'mz')
        show_phase : bool
            Show phase plot
        show_magnitude : bool
            Show magnitude plot
        dpi : int
            Figure resolution
        **kwargs
            Additional arguments for plot
        
        Returns
        -------
        Figure
            Matplotlib figure with mode visualization
        """
        # Use component from context if available
        if self.component_index is not None:
            component = ["mx", "my", "mz"][self.component_index]
        
        return self._legacy_analyzer.plot_modes(
            frequency=frequency,
            z_layer=z_layer,
            component=component,
            show_phase=show_phase,
            show_magnitude=show_magnitude,
            **kwargs,
        )
    
    def characterize_mode(
        self,
        frequency: float,
        z_layer: int = 0,
        verbose: bool = False,
        **kwargs,
    ):
        """Characterize mode at frequency.
        
        Parameters
        ----------
        frequency : float
            Frequency to analyze [GHz]
        z_layer : int
            Layer index
        verbose : bool
            Show detailed output
        **kwargs
            Additional arguments
        
        Returns
        -------
        ModeCharacterizationResult
            Classification result with metrics
        """
        return self._legacy_analyzer.characterize_mode(
            frequency=frequency,
            z_layer=z_layer,
            verbose=verbose,
            **kwargs,
        )
    
    def save_modes_animation(
        self,
        frequency: float = None,
        frequency_range: tuple = None,
        animation_type: str = "temporal",
        save_path: str = None,
        dpi: int = 100,
        **kwargs,
    ):
        """Create and save mode animation.
        
        Parameters
        ----------
        frequency : float, optional
            Single frequency for temporal animation
        frequency_range : tuple, optional
            (f_min, f_max) for frequency sweep
        animation_type : str
            'temporal', 'frequency', or 'phase'
        save_path : str, optional
            Output file path (.mp4 or .gif)
        dpi : int
            Animation resolution
        **kwargs
            Additional arguments
        
        Returns
        -------
        Animation or path
            Animation object or saved file path
        """
        return self._legacy_analyzer.save_modes_animation(
            frequency=frequency,
            frequency_range=frequency_range,
            animation_type=animation_type,
            save_path=save_path,
            **kwargs,
        )
    
    def compute_modes(self, dset: str = None, **kwargs):
        """Compute/recompute modes for dataset.
        
        Parameters
        ----------
        dset : str, optional
            Dataset name (uses context if not specified)
        **kwargs
            Additional arguments for mode computation
        """
        dataset = dset or self._dataset_context or self.dataset_name
        return self._legacy_analyzer.compute_modes(dset=dataset, **kwargs)
    
    def __repr__(self) -> str:
        """Rich representation of modes interface."""
        try:
            return self._rich_display()
        except Exception:
            return self._basic_display()
    
    def _rich_display(self) -> str:
        """Generate rich help display."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
            from rich.syntax import Syntax
            from io import StringIO
            
            capture = StringIO()
            console = Console(file=capture, force_terminal=True, width=100)
            
            # Title
            title = Text()
            title.append("🎯 FFT Mode Analyzer\n", style="bold cyan")
            title.append(f"📁 Dataset: {self.dataset_name}\n", style="dim")
            if self.component_label:
                title.append(f"📊 Component: {self.component_label}\n", style="green")
            title.append(f"📂 Path: {self.zarr_path}", style="dim")
            
            console.print(Panel(title, border_style="cyan"))
            
            # Methods table
            methods = Table(show_header=True, header_style="bold yellow")
            methods.add_column("Method", style="cyan")
            methods.add_column("Description", style="white")
            
            methods.add_row("interactive_spectrum(dpi=100)", "Plot spectrum with legends")
            methods.add_row("plot_modes(frequency)", "Visualize mode at frequency")
            methods.add_row("characterize_mode(frequency)", "Classify mode type")
            methods.add_row("save_modes_animation(...)", "Create mode animations")
            methods.add_row("compute_modes()", "Compute/recompute modes")
            
            console.print(methods)
            console.print("")
            
            # Examples
            example = '''# With component selection (my) and DPI:
job[0].m[:200,...,1].fft.modes.interactive_spectrum(dpi=150)

# All components:
job[0].fft.modes.interactive_spectrum(log_scale=True)

# Mode visualization:
job[0].fft.modes.plot_modes(frequency=9.5)

# Mode animation:
job[0].fft.modes.save_modes_animation(frequency=9.5, save_path="mode.mp4")'''
            
            syntax = Syntax(example, "python", theme="monokai", line_numbers=False)
            console.print(Panel(syntax, title="[bold green]Examples", border_style="green"))
            
            return capture.getvalue()
        except ImportError:
            return self._basic_display()
    
    def _basic_display(self) -> str:
        """Basic text display."""
        return (
            f"FFTModeInterface(dataset={self.dataset_name}, "
            f"component={self.component_label or 'all'})"
        )

