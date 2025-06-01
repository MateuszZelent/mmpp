import colorsys
import os
import platform
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

# Import shared logging configuration
from .logging_config import get_mmpp_logger, setup_mmpp_logging

# Get logger for plotting
log = get_mmpp_logger("mmpp.plot")


def _find_largest_m_dataset(zarr_path: str) -> str:
    """
    Automatically find the m dataset with the largest time dimension.

    Parameters:
    -----------
    zarr_path : str
        Path to zarr file

    Returns:
    --------
    str
        Name of the largest m dataset (e.g., "m_z5-8", "m_z11-12", or fallback "m")
    """
    try:
        from pyzfn import Pyzfn

        job = Pyzfn(zarr_path)

        # Get all available datasets that start with "m"
        m_datasets = []
        for key in job.z.keys():
            if key.startswith("m") and not key.startswith("m_"):
                # Include base "m" dataset
                m_datasets.append(key)
            elif key.startswith("m_"):
                # Include cropped datasets like "m_z5-8", "m_z11-12"
                m_datasets.append(key)

        if not m_datasets:
            log.warning(f"No m datasets found in {zarr_path}, using fallback 'm'")
            return "m"

        # Find dataset with largest time dimension
        largest_dataset = "m"
        largest_time_size = 0

        for dataset_name in m_datasets:
            try:
                dataset = job.z[dataset_name]
                if hasattr(dataset, "shape") and len(dataset.shape) >= 1:
                    time_size = dataset.shape[0]  # First dimension is usually time
                    log.debug(f"Dataset {dataset_name}: time size = {time_size}")

                    if time_size > largest_time_size:
                        largest_time_size = time_size
                        largest_dataset = dataset_name
            except Exception as e:
                log.debug(f"Could not check dataset {dataset_name}: {e}")
                continue

        log.info(
            f"Auto-selected dataset '{largest_dataset}' with {largest_time_size} time steps"
        )
        return largest_dataset

    except Exception as e:
        log.warning(
            f"Error finding largest m dataset in {zarr_path}: {e}, using fallback 'm'"
        )
        return "m"


if TYPE_CHECKING:
    from matplotlib.axes import Axes
else:
    Axes = Any

# Import for plotting
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    try:
        import cmocean

        CMOCEAN_AVAILABLE = True
    except ImportError:
        CMOCEAN_AVAILABLE = False
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from pyzfn import Pyzfn

# Import type aliases from core
try:
    from .core import np3d
except ImportError:
    np3d = np.ndarray


@dataclass
class PlotConfig:
    """Configuration for plotting operations."""

    figsize: tuple = (12, 8)
    dpi: int = 100
    style: str = "paper"  # Changed default to custom paper style
    colormap: str = "viridis"
    line_alpha: float = 0.7
    line_width: float = 1.5
    grid: bool = True
    legend: bool = True
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10
    use_custom_fonts: bool = True
    font_family: str = "Arial"
    colors: Optional[Dict[str, str]] = None
    max_legend_params: int = 4  # Maximum number of parameters to show in legend
    sort_results: bool = True  # Whether to sort results by parameters

    def __post_init__(self) -> None:
        """Initialize default colors."""
        if self.colors is None:
            self.colors = {"text": "#808080", "axes": "#808080", "grid": "#cccccc"}


class FontManager:
    """Font management utilities for plotting."""

    @staticmethod
    def get_available_fonts() -> List[str]:
        """Get list of available fonts on the system."""
        if not MATPLOTLIB_AVAILABLE:
            return []

        try:
            # Get fonts from matplotlib font manager
            font_list = font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
            font_names = []
            for font_path in font_list:
                try:
                    font_prop = font_manager.FontProperties(fname=font_path)
                    font_names.append(font_prop.get_name())
                except Exception:
                    continue
            return sorted(list(set(font_names)))
        except Exception:
            return []

    @staticmethod
    def setup_custom_fonts(verbose: bool = True) -> bool:
        """Setup custom fonts from the fonts directory."""
        if not MATPLOTLIB_AVAILABLE:
            return False

        try:
            # Get the fonts directory path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            fonts_dir = os.path.join(current_dir, "fonts")

            if not os.path.exists(fonts_dir):
                if verbose:
                    log.warning(f"Fonts directory not found: {fonts_dir}")
                return False

            # Add custom fonts
            font_files = []
            for root, dirs, files in os.walk(fonts_dir):
                for file in files:
                    if file.endswith((".ttf", ".otf")):
                        font_path = os.path.join(root, file)
                        font_files.append(font_path)
                        font_manager.fontManager.addfont(font_path)

            if verbose and font_files:
                log.info(f"Loaded {len(font_files)} custom fonts from {fonts_dir}")

            return len(font_files) > 0

        except Exception as e:
            if verbose:
                log.error(f"Error setting up custom fonts: {e}")
            return False


class PlotterProxy:
    """Proxy class to provide plotting functionality to search results."""

    def __init__(self, results: List[Any], mmpp_instance: Optional[Any] = None):
        """
        Initialize the plotter proxy.

        Parameters:
        -----------
        results : List[Any]
            List of results to plot
        mmpp_instance : Optional[Any]
            Reference to MMPP instance
        """
        self.results = results
        self.mmpp_instance = mmpp_instance

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to MMPPlotter."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "Plotting functionality not available. Install matplotlib to use plotting features."
            )

        # Create MMPPlotter instance and delegate
        plotter = MMPPlotter(self.results, self.mmpp_instance)
        return getattr(plotter, name)

    def __repr__(self) -> str:
        """String representation of the proxy."""
        if MATPLOTLIB_AVAILABLE:
            return f"PlotterProxy({len(self.results)} results)"
        else:
            return "PlotterProxy(plotting not available - install matplotlib)"


# Global font and style setup cache
_FONTS_INITIALIZED = False
_STYLE_INITIALIZED = False
_STYLING_SETUP_COMPLETED = False


def setup_custom_fonts(verbose: bool = False) -> bool:
    """Setup custom fonts including Arial."""
    global _FONTS_INITIALIZED

    # Skip if already initialized
    if _FONTS_INITIALIZED:
        return True

    try:
        # Import fonts from package directory
        package_dir = os.path.dirname(__file__)
        font_dirs = [
            os.path.join(package_dir, "fonts"),  # Package fonts
            "./fonts",  # Local fonts (development)
            os.path.expanduser("~/.fonts"),  # User fonts
        ]

        fonts_loaded = False
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                if verbose:
                    log.debug(f"🔍 Checking font directory: {font_dir}")
                font_files = font_manager.findSystemFonts(fontpaths=[font_dir])
                for font_file in font_files:
                    try:
                        font_manager.fontManager.addfont(font_file)
                        fonts_loaded = True
                        if verbose:
                            log.debug(f"✓ Added font: {os.path.basename(font_file)}")
                    except Exception as e:
                        if verbose:
                            log.warning(f"Warning: Could not add font {font_file}: {e}")

        # Rebuild font cache if fonts were loaded
        if fonts_loaded:
            font_manager.fontManager.findfont("Arial", rebuild_if_missing=True)

        # Set Arial as default font
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.sans-serif"] = ["Arial"] + plt.rcParams["font.sans-serif"]

        # Check if Arial is available
        available_fonts = set(f.name for f in font_manager.fontManager.ttflist)
        if "Arial" in available_fonts:
            if verbose:
                log.debug("✓ Arial font loaded successfully")
        else:
            if verbose:
                log.warning("⚠ Arial font not found, using default fonts")

        _FONTS_INITIALIZED = True
        return True

    except Exception as e:
        if verbose:
            log.warning(f"Warning: Font setup failed: {e}")
        return False


def load_paper_style(verbose: bool = False) -> bool:
    """Load custom paper style."""
    global _STYLE_INITIALIZED

    # Skip if already initialized
    if _STYLE_INITIALIZED:
        return True

    try:
        # Try to find paper.mplstyle in current directory or relative to this file
        style_paths = [
            "./paper.mplstyle",
            os.path.join(os.path.dirname(__file__), "paper.mplstyle"),
            "/mnt/storage_2/scratch/pl0095-01/zelent/mannga/bowtie/mateusz/sinc/solver_test/paper.mplstyle",
        ]

        for style_path in style_paths:
            if os.path.exists(style_path):
                plt.style.use(style_path)
                if verbose:
                    log.debug(f"✓ Loaded paper style from: {style_path}")
                _STYLE_INITIALIZED = True
                return True

        if verbose:
            log.warning("⚠ paper.mplstyle not found, using default style")
        _STYLE_INITIALIZED = True
        return False

    except Exception as e:
        if verbose:
            log.warning(f"Warning: Could not load paper style: {e}")
        _STYLE_INITIALIZED = True
        return False


def apply_custom_colors(colors: Dict[str, str]) -> None:
    """Apply custom colors to matplotlib rcParams."""
    try:
        if "text" in colors:
            plt.rcParams["text.color"] = colors["text"]
            plt.rcParams["axes.labelcolor"] = colors["text"]
            plt.rcParams["xtick.color"] = colors["text"]
            plt.rcParams["ytick.color"] = colors["text"]

        if "axes" in colors:
            plt.rcParams["axes.edgecolor"] = colors["axes"]

        if "grid" in colors:
            plt.rcParams["grid.color"] = colors["grid"]

    except Exception as e:
        log.warning(f"Warning: Could not apply custom colors: {e}")


class MMPPlotter:
    """
    Advanced plotting functionality for MMPP results.

    This class provides comprehensive plotting capabilities including:
    - Time series plotting with averaging
    - Multiple datasets comparison
    - Component selection (x, y, z)
    - Professional styling and customization
    - Custom fonts and paper-ready styling
    """

    def __init__(
        self, results: Union[List[Any], Any], mmpp_instance: Optional[Any] = None
    ) -> None:
        """
        Initialize the plotter.

        Parameters:
        -----------
        results : List or single result
            ZarrJobResult objects to plot
        mmpp_instance : MMPP, optional
            Reference to the parent MMPP instance
        """
        # Handle both single results and lists
        if not isinstance(results, list):
            self.results = [results]
        else:
            self.results = results

        self.mmpp = mmpp_instance
        self.config = PlotConfig()

        # Set up logging level based on parent debug mode
        debug_mode = getattr(mmpp_instance, "debug", False) if mmpp_instance else False
        setup_mmpp_logging(debug=debug_mode, logger_name="mmpp.plot")

        log.debug(f"MMPPlotter initialized with {len(self.results)} results")

        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )

        # Initialize console if available
        if RICH_AVAILABLE:
            self.console = Console()

        # Setup custom styling
        self._setup_styling()

    def _setup_styling(self) -> None:
        """Setup custom fonts and styling."""
        global _STYLING_SETUP_COMPLETED

        # Skip if styling has already been set up globally
        if _STYLING_SETUP_COMPLETED:
            return

        try:
            # Setup custom fonts if enabled
            if self.config.use_custom_fonts:
                setup_custom_fonts(verbose=False)

            # Load paper style
            if self.config.style == "paper":
                if not load_paper_style(verbose=False):
                    # Fallback to a standard style
                    try:
                        plt.style.use("seaborn-v0_8")
                    except (OSError, ImportError):
                        try:
                            plt.style.use("default")
                        except (OSError, ImportError):
                            pass
            else:
                # Load specified style
                try:
                    if self.config.style in plt.style.available:
                        plt.style.use(self.config.style)
                except Exception as e:
                    log.warning(
                        f"Warning: Could not load style '{self.config.style}': {e}"
                    )

            # Apply custom colors
            apply_custom_colors(self.config.colors)

            # Mark styling as completed globally
            _STYLING_SETUP_COMPLETED = True

        except Exception as e:
            log.error(f"Warning: Styling setup failed: {e}")
            # Even if setup failed, mark as completed to avoid retries
            _STYLING_SETUP_COMPLETED = True

    def __repr__(self) -> str:
        """Rich representation of the plotter."""
        if RICH_AVAILABLE and self.mmpp and self.mmpp._interactive_mode:
            return self._rich_plotter_display()
        else:
            return self._basic_plotter_display()

    def _rich_plotter_display(self) -> str:
        """Generate rich display for plotter."""
        summary_text = Text()
        summary_text.append(
            f"📊 MMPP Plotter for {len(self.results)} datasets\n", style="bold cyan"
        )
        summary_text.append(f"🎨 Style: {self.config.style}\n", style="dim")
        summary_text.append(f"🔤 Font: {self.config.font_family}\n", style="dim")
        summary_text.append(f"📏 Default figsize: {self.config.figsize}\n", style="dim")

        methods_text = Text()
        methods_text.append("🔧 Available methods:\n", style="bold yellow")
        methods = [
            ("plot(x_series, y_series, **kwargs)", "Main plotting method"),
            ("plot_time_series(dataset, **kwargs)", "Time series plots"),
            ("plot_components(dataset, **kwargs)", "Component comparison"),
            ("configure(**kwargs)", "Update plot configuration"),
            ("reset_style()", "Reset to paper style"),
            ("set_style(style_name)", "Change matplotlib style"),
        ]

        for method, description in methods:
            methods_text.append("  • ", style="dim")
            methods_text.append(method, style="code")
            methods_text.append(f" - {description}\n", style="dim")

        examples_text = Text()
        examples_text.append("💡 Usage examples:\n", style="bold green")
        examples = [
            "plotter.plot('t', 'm_z11', comp=2, average=(1,2,3))",
            "plotter.plot_time_series('m_z11', comp='z')",
            "plotter.configure(style='dark_background')",
            "plotter.reset_style()  # Reset to paper style",
        ]

        for example in examples:
            examples_text.append(f"  {example}\n", style="code")

        if RICH_AVAILABLE:
            try:
                with self.console.capture() as capture:
                    self.console.print(
                        Panel.fit(
                            summary_text,
                            title="[bold blue]MMPP Plotter[/bold blue]",
                            border_style="blue",
                        )
                    )
                    self.console.print("")
                    self.console.print(
                        Columns(
                            [
                                Panel.fit(
                                    methods_text,
                                    title="[bold yellow]Methods[/bold yellow]",
                                    border_style="yellow",
                                ),
                                Panel.fit(
                                    examples_text,
                                    title="[bold green]Examples[/bold green]",
                                    border_style="green",
                                ),
                            ]
                        )
                    )
                return capture.get()
            except Exception:
                pass

        return str(summary_text) + "\n" + str(methods_text) + "\n" + str(examples_text)

    def _basic_plotter_display(self) -> str:
        """Generate basic display for plotter."""
        return f"""
MMPP Plotter:
============
📊 Datasets: {len(self.results)}
🎨 Style: {self.config.style}
🔤 Font: {self.config.font_family}

🔧 Main methods:
  • plot(x_series, y_series, **kwargs) - Main plotting method
  • plot_time_series(dataset, **kwargs) - Time series plots
  • plot_components(dataset, **kwargs) - Component comparison
  • configure(**kwargs) - Update configuration
  • reset_style() - Reset to paper style

💡 Example: plotter.plot('t', 'm_z11', comp=2, average=(1,2,3))
"""

    def configure(self, **kwargs) -> "MMPPlotter":
        """
        Configure plot settings.

        Parameters:
        -----------
        \\*\\*kwargs : Any
            Configuration options:
            - figsize : tuple - Figure size (width, height), default (12, 8)
            - dpi : int - Figure DPI, default 100
            - style : str - Matplotlib style, default "paper"
            - colormap : str - Colormap name, default "viridis"
            - line_alpha : float - Line transparency, default 0.7
            - line_width : float - Line width, default 1.5
            - grid : bool - Show grid, default True
            - legend : bool - Show legend, default True
            - title_fontsize : int - Title font size, default 14
            - label_fontsize : int - Label font size, default 12
            - tick_fontsize : int - Tick font size, default 10
            - use_custom_fonts : bool - Use custom fonts, default True
            - font_family : str - Font family, default "Arial"
            - colors : dict - Custom colors for text/axes/grid
            - max_legend_params : int - Max parameters in legend, default 4
            - sort_results : bool - Sort results by parameters, default True

        Returns:
        --------
        MMPPlotter
            Self for method chaining

        Examples:
        ---------
        >>> plotter.configure(sort_results=False, max_legend_params=6)
        >>> plotter.configure(style='dark_background', grid=False)
        """
        style_changed = False

        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                if key == "style":
                    style_changed = True
            else:
                log.warning(f"Warning: Unknown configuration option '{key}'")

        # Apply style if changed
        if style_changed:
            self._setup_styling()

        # Apply color changes if provided
        if "colors" in kwargs:
            apply_custom_colors(self.config.colors)

        return self

    def reset_style(self) -> "MMPPlotter":
        """
        Reset to paper style with Arial font.

        Returns:
        --------
        MMPPlotter
            Self for method chaining
        """
        self.config.style = "paper"
        self.config.use_custom_fonts = True
        self.config.font_family = "Arial"
        self._setup_styling()
        return self

    def set_style(self, style_name: str) -> "MMPPlotter":
        """
        Set matplotlib style.

        Parameters:
        -----------
        style_name : str
            Name of the matplotlib style

        Returns:
        --------
        MMPPlotter
            Self for method chaining
        """
        try:
            if style_name == "paper":
                self.reset_style()
            else:
                plt.style.use(style_name)
                self.config.style = style_name
                log.debug(f"✓ Applied style: {style_name}")
        except Exception as e:
            log.warning(f"Warning: Could not apply style '{style_name}': {e}")

        return self

    def get_available_styles(self) -> List[str]:
        """
        Get list of available matplotlib styles.

        Returns:
        --------
        List[str]
            List of available style names
        """
        styles = list(plt.style.available) + ["paper"]
        return sorted(styles)

    def _parse_component(self, comp: Union[str, int]) -> int:
        """Parse component specification."""
        if isinstance(comp, str):
            comp_map = {"x": 0, "y": 1, "z": 2}
            return comp_map.get(comp.lower(), 2)
        return int(comp)

    def _load_pyzfn_job(self, result) -> Pyzfn:
        """Load a Pyzfn job from a result."""
        return Pyzfn(result.path)

    def _extract_data(
        self,
        job: Pyzfn,
        dataset_name: str,
        x_series: Optional[str] = None,
        comp: Optional[Union[str, int]] = None,
        average: Optional[Tuple[Any, ...]] = None,
    ) -> tuple:
        """
        Extract data from a Pyzfn job.

        Parameters:
        -----------
        job : Pyzfn
            The Pyzfn job instance
        dataset_name : str
            Name of the dataset (e.g., 'm_z11')
        x_series : str, optional
            Name of x-axis data (e.g., 't')
        comp : Union[str, int], optional
            Component to extract (x/0, y/1, z/2)
        average : tuple, optional
            Axes to average over

        Returns:
        --------
        tuple
            (x_data, y_data, metadata)
        """
        try:
            # Get the dataset
            dataset = getattr(job, dataset_name)

            # Extract x-axis data
            x_data = None
            if x_series:
                if x_series in dataset.attrs:
                    x_data = dataset.attrs[x_series]
                elif hasattr(job, x_series):
                    x_data = getattr(job, x_series)
                else:
                    # Try to get from dataset attributes
                    x_data = np.arange(len(dataset))
                    log.warning(f"Warning: '{x_series}' not found, using indices")

            # Extract y-axis data
            y_data = dataset[...]

            # Select component if specified
            if comp is not None:
                comp_idx = self._parse_component(comp)
                if y_data.ndim > comp_idx:
                    y_data = y_data[..., comp_idx]
                else:
                    log.warning(
                        f"Warning: Component {comp} not available, using full data"
                    )

            # Apply averaging if specified
            if average is not None:
                if isinstance(average, (list, tuple)):
                    avg_axes = tuple(ax for ax in average if ax < y_data.ndim)
                    if avg_axes:
                        y_data = np.average(y_data, axis=avg_axes)
                else:
                    if average < y_data.ndim:
                        y_data = np.average(y_data, axis=average)

            # Metadata
            metadata = {
                "path": job.path,
                "dataset": dataset_name,
                "component": comp,
                "averaged_axes": average,
                "shape": y_data.shape,
                "attrs": dict(dataset.attrs) if hasattr(dataset, "attrs") else {},
            }

            return x_data, y_data, metadata

        except Exception as e:
            log.error(f"Error extracting data from {job.path}: {e}")
            return None, None, None

    def plot(
        self,
        x_series: str,
        y_series: str,
        comp: Optional[Union[str, int]] = None,
        average: Optional[Tuple[Any, ...]] = None,
        figsize: Optional[Tuple[Any, ...]] = None,
        title: Optional[Union[str, List[str]]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        legend_labels: Optional[List[str]] = None,
        legend_variables: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        paper_ready: bool = False,
        **kwargs: Any,
    ) -> tuple:
        """
        Create a plot for the specified data series.

        Parameters:
        -----------
        x_series : str
            Name of x-axis data (e.g., 't' for time)
        y_series : str
            Name of y-axis dataset (e.g., 'm_z11')
        comp : Union[str, int], optional
            Component to plot ('x'/'y'/'z' or 0/1/2)
        average : tuple, optional
            Axes to average over (e.g., (1,2,3) for spatial averaging)
        figsize : tuple, optional
            Figure size (width, height)
        title : Union[str, List[str]], optional
            Plot title. Can be:
            - str: Custom title text
            - List[str]: Parameter names to show in title (e.g., ['amp', 'f0'] -> "Amp = 0.1, F0 = 1e9")
        xlabel : str, optional
            X-axis label
        ylabel : str, optional
            Y-axis label
        legend_labels : List[str], optional
            Custom legend labels
        legend_variables : List[str], optional
            Specific variables to show in legend (e.g., ['maxerr', 'f0']).
            If provided, overrides automatic varying parameter detection.
        colors : List[str], optional
            Custom colors for each line
        save_path : str, optional
            Path to save the figure
        paper_ready : bool, optional
            If True, apply paper-ready styling (default: False)
        \\*\\*kwargs : Any
            Additional matplotlib plot arguments

        Returns:
        --------
        tuple
            (figure, axes) matplotlib objects
        """
        if not self.results:
            log.warning("No results to plot")
            return None, None

        # Apply paper-ready styling if requested
        if paper_ready:
            original_style = self.config.style
            self.reset_style()

        # Setup figure
        figsize = figsize or self.config.figsize
        fig, ax = plt.subplots(figsize=figsize, dpi=self.config.dpi)

        # Setup colors
        if colors is None:
            if CMOCEAN_AVAILABLE and hasattr(cmocean.cm, self.config.colormap):
                cmap = getattr(cmocean.cm, self.config.colormap)
            else:
                cmap = plt.cm.get_cmap(self.config.colormap)
            colors = [
                cmap(i / max(1, len(self.results) - 1))
                for i in range(len(self.results))
            ]

        # Progress bar if available
        iterator = (
            tqdm(self.results, desc="Processing datasets")
            if TQDM_AVAILABLE
            else self.results
        )

        plotted_data = []

        # Sort results by all available parameters for consistent ordering
        if self.config.sort_results:
            sorted_results = self._sort_results_by_parameters(self.results)
        else:
            sorted_results = self.results

        # Get varying parameters for smart legend (only show parameters that differ)
        if legend_variables is not None:
            # User specified which variables to show in legend
            varying_params = legend_variables
        else:
            # Auto-detect varying parameters
            varying_params = (
                self._get_varying_parameters(sorted_results)
                if len(sorted_results) > 1
                else []
            )

        # Update iterator to use sorted results
        iterator = (
            tqdm(sorted_results, desc="Processing datasets")
            if TQDM_AVAILABLE
            else sorted_results
        )

        # Plot each result
        for i, result in enumerate(iterator):
            try:
                # Load Pyzfn job
                job = self._load_pyzfn_job(result)

                # Extract data
                x_data, y_data, metadata = self._extract_data(
                    job, y_series, x_series, comp, average
                )

                if x_data is not None and y_data is not None:
                    # Determine label
                    if legend_labels and i < len(legend_labels):
                        label = legend_labels[i]
                    else:
                        # Create informative label showing only varying parameters
                        label = self._format_result_label(result, varying_params)

                    # Plot
                    line = ax.plot(
                        x_data,
                        y_data,
                        color=colors[i % len(colors)],
                        alpha=self.config.line_alpha,
                        linewidth=self.config.line_width,
                        label=label,
                        **kwargs,
                    )

                    plotted_data.append(
                        {
                            "result": result,
                            "x_data": x_data,
                            "y_data": y_data,
                            "metadata": metadata,
                            "line": line[0],
                        }
                    )

            except Exception as e:
                log.error(f"Error plotting {result.path}: {e}")
                continue

        # Customize plot
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.config.label_fontsize)
        elif x_series:
            ax.set_xlabel(x_series, fontsize=self.config.label_fontsize)

        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.config.label_fontsize)
        else:
            ylabel_parts = [y_series]
            if comp is not None:
                comp_name = ["x", "y", "z"][self._parse_component(comp)]
                ylabel_parts.append(f"({comp_name})")
            if average:
                ylabel_parts.append("averaged")
            ax.set_ylabel(" ".join(ylabel_parts), fontsize=self.config.label_fontsize)

        # Set title
        if title:
            if isinstance(title, list):
                # Dynamic title based on parameter values
                dynamic_title = self._format_dynamic_title(title, sorted_results)
                if dynamic_title:
                    ax.set_title(dynamic_title, fontsize=self.config.title_fontsize)
                else:
                    # Fallback if no parameters found
                    title_parts = [f"{y_series} vs {x_series}"]
                    if comp is not None:
                        title_parts.append(f"component {comp}")
                    ax.set_title(
                        " - ".join(title_parts), fontsize=self.config.title_fontsize
                    )
            else:
                # Static title string
                ax.set_title(title, fontsize=self.config.title_fontsize)
        else:
            # Default title
            title_parts = [f"{y_series} vs {x_series}"]
            if comp is not None:
                title_parts.append(f"component {comp}")
            ax.set_title(" - ".join(title_parts), fontsize=self.config.title_fontsize)

        if self.config.grid:
            ax.grid(True, alpha=0.3)

        if self.config.legend and len(plotted_data) > 1:
            ax.legend(fontsize=self.config.label_fontsize)

        ax.tick_params(labelsize=self.config.tick_fontsize)

        plt.tight_layout()

        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")
            log.info(f"Figure saved to: {save_path}")

        # Store plot data for further analysis
        self._last_plot_data = plotted_data

        # Restore original style if paper_ready was used
        if paper_ready and "original_style" in locals():
            self.config.style = original_style

        return fig, ax

    def plot_time_series(
        self,
        dataset: str,
        comp: Union[str, int] = "z",
        average: tuple = (1, 2, 3),
        **kwargs: Any,
    ) -> tuple:
        """
        Convenience method for time series plotting.

        Parameters:
        -----------
        dataset : str
            Dataset name (e.g., 'm_z11')
        comp : Union[str, int], optional
            Component ('x'/'y'/'z' or 0/1/2, default: 'z')
        average : tuple, optional
            Spatial axes to average over (default: (1,2,3))
        \\*\\*kwargs : Any
            Additional arguments passed to plot()

        Returns:
        --------
        tuple
            (figure, axes) matplotlib objects
        """
        return self.plot(
            "t", dataset, comp=comp, average=average, xlabel="Time", **kwargs
        )

    def plot_components(
        self,
        dataset: str,
        time_slice: int = -1,
        average: tuple = (1, 2, 3),
        **kwargs: Any,
    ) -> tuple:
        """
        Plot all three components of a dataset.

        Parameters:
        -----------
        dataset : str
            Dataset name
        time_slice : int, optional
            Time slice to plot (default: -1 for last)
        average : tuple, optional
            Axes to average over
        \\*\\*kwargs : Any
            Additional plot arguments

        Returns:
        --------
        tuple
            (figure, axes) matplotlib objects
        """
        if not self.results:
            log.warning("No results to plot")
            return None, None

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=self.config.dpi)
        components = ["x", "y", "z"]

        for i, comp in enumerate(components):
            ax = axes[i]

            for j, result in enumerate(self.results):
                try:
                    job = self._load_pyzfn_job(result)
                    _, y_data, metadata = self._extract_data(
                        job, dataset, comp=comp, average=average
                    )

                    if y_data is not None:
                        if time_slice is not None and y_data.ndim > 0:
                            value = y_data[time_slice]
                        else:
                            value = y_data

                        ax.bar(
                            j,
                            value,
                            alpha=0.7,
                            label=f"Result {j + 1}" if i == 0 else "",
                        )

                except Exception as e:
                    log.error(f"Error plotting component {comp} for {result.path}: {e}")
                    continue

            ax.set_title(f"{dataset} - {comp.upper()} component")
            ax.set_xlabel("Dataset")
            ax.set_ylabel(f"{dataset}_{comp}")

            if self.config.grid:
                ax.grid(True, alpha=0.3)

        if self.config.legend:
            axes[0].legend()

        plt.tight_layout()
        return fig, axes

    def get_plot_data(self) -> List[Dict]:
        """
        Get data from the last plot for further analysis.

        Returns:
        --------
        List[Dict]
            List of dictionaries containing plot data and metadata
        """
        if hasattr(self, "_last_plot_data"):
            return self._last_plot_data
        else:
            log.warning("No plot data available. Run a plot method first.")
            return []

    def save_all_data(self, filename: str, format: str = "npz") -> None:
        """
        Save all plotted data to file.

        Parameters:
        -----------
        filename : str
            Output filename
        format : str, optional
            Format ('npz', 'csv', 'json')
        """
        if not hasattr(self, "_last_plot_data"):
            log.warning("No plot data to save. Run a plot method first.")
            return

        if format == "npz":
            data_dict = {}
            for i, item in enumerate(self._last_plot_data):
                data_dict[f"x_data_{i}"] = item["x_data"]
                data_dict[f"y_data_{i}"] = item["y_data"]
            np.savez(filename, **data_dict)

        elif format == "csv":
            import pandas as pd

            # Create DataFrame with all data
            df_dict = {}
            for i, item in enumerate(self._last_plot_data):
                df_dict[f"x_{i}"] = item["x_data"]
                df_dict[f"y_{i}"] = item["y_data"]

            df = pd.DataFrame(df_dict)
            df.to_csv(filename, index=False)

        log.info(f"Data saved to: {filename}")

    def _sort_results_by_parameters(self, results: List[Any]) -> List[Any]:
        """
        Sort results by all available parameters for consistent ordering.

        Parameters:
        -----------
        results : List[Any]
            List of result objects to sort

        Returns:
        --------
        List[Any]
            Sorted list of results
        """

        def sort_key(result):
            # Collect all sortable attributes
            sort_values = []

            # Common parameters in order of importance
            important_params = [
                "solver",
                "f0",
                "maxerr",
                "Nx",
                "Ny",
                "Nz",
                "PBCx",
                "PBCy",
                "PBCz",
            ]

            for param in important_params:
                if hasattr(result, param):
                    value = getattr(result, param)
                    # Convert to sortable format
                    if isinstance(value, (int, float)):
                        sort_values.append(value)
                    elif isinstance(value, str):
                        sort_values.append(value)
                    else:
                        sort_values.append(str(value))
                else:
                    sort_values.append(0)  # Default value for missing parameters

            # Add any other attributes not in the important list
            for attr_name in sorted(dir(result)):
                if (
                    not attr_name.startswith("_")
                    and attr_name not in important_params
                    and attr_name not in ["path", "attributes"]
                ):
                    try:
                        value = getattr(result, attr_name)
                        if isinstance(value, (int, float, str)):
                            sort_values.append(value)
                    except Exception:
                        pass

            return tuple(sort_values)

        try:
            return sorted(results, key=sort_key)
        except Exception as e:
            print(f"Warning: Could not sort results: {e}")
            return results

    def _get_varying_parameters(self, results: List[Any]) -> List[str]:
        """
        Identify which parameters vary between results.

        Parameters:
        -----------
        results : List[Any]
            List of result objects to analyze

        Returns:
        --------
        List[str]
            List of parameter names that vary between results
        """
        if len(results) <= 1:
            return []

        # Define parameters to exclude from legend (memory addresses, internal vars, etc.)
        excluded_params = {
            # Memory addresses and pointers
            "Aex",
            "Bext",
            "Ms",
            "alpha",
            "gamma",  # Common MUMAX3 pointers
            # User requested exclusions
            "end_time",
            "maxerr_path",
            "port",
            # Internal/technical parameters
            "path",
            "attributes",
            "job_id",
            "timestamp",
            "uuid",
            # Time-related parameters
            "start_time",
            "runtime",
            # Very long or non-informative parameters
            "command_line",
            "full_path",
            "working_directory",
        }

        def should_exclude_param(name: str, value: Any) -> bool:
            """Check if parameter should be excluded from legend."""
            # Exclude if in explicit exclusion list
            if name in excluded_params:
                return True

            # Exclude memory addresses (hex values starting with 0x)
            if isinstance(value, str) and value.startswith("0x"):
                return True

            # Exclude very long strings (likely file paths or commands)
            if isinstance(value, str) and len(value) > 50:
                return True

            # Exclude parameters with underscores at start/end (internal vars)
            if name.startswith("_") or name.endswith("_"):
                return True

            return False

        # Collect all potential parameters - primarily from attributes dict for ZarrJobResult
        all_params = set()
        for result in results:
            # Primary method: Check attributes dict (ZarrJobResult pattern)
            if hasattr(result, "attributes") and isinstance(result.attributes, dict):
                for attr_name, value in result.attributes.items():
                    # Only consider simple types that can vary meaningfully
                    if isinstance(
                        value, (int, float, str, bool)
                    ) and not should_exclude_param(attr_name, value):
                        all_params.add(attr_name)

            # Fallback: check direct attributes for non-ZarrJobResult objects
            elif not hasattr(result, "attributes"):
                for attr_name in dir(result):
                    if not attr_name.startswith("_") and not callable(
                        getattr(result, attr_name, None)
                    ):
                        try:
                            value = getattr(result, attr_name)
                            if isinstance(
                                value, (int, float, str, bool)
                            ) and not should_exclude_param(attr_name, value):
                                all_params.add(attr_name)
                        except Exception:
                            pass

        # Check which parameters actually vary
        varying_params = []
        for param in all_params:
            values = []
            for result in results:
                # Get value - prefer attributes dict, fall back to direct attribute
                value = None
                try:
                    if hasattr(result, "attributes") and param in result.attributes:
                        value = result.attributes[param]
                    elif hasattr(result, param):
                        value = getattr(result, param)

                    if value is not None:
                        values.append(value)
                except Exception:
                    pass

            # Check if values are different (accounting for floating point precision)
            if len(values) > 1:
                unique_values = set()
                for val in values:
                    if isinstance(val, float):
                        # Round to reasonable precision for comparison
                        unique_values.add(round(val, 10))
                    else:
                        unique_values.add(val)

                if len(unique_values) > 1:
                    varying_params.append(param)

        # Sort by priority (important parameters first)
        priority_params = [
            "solver",
            "f0",
            "maxerr",
            "dt",
            "Nx",
            "Ny",
            "Nz",
            "PBCx",
            "PBCy",
            "PBCz",
            "amp_values",
        ]

        # Sort varying parameters by priority
        sorted_varying = []
        for param in priority_params:
            if param in varying_params:
                sorted_varying.append(param)

        # Add remaining varying parameters alphabetically
        for param in sorted(varying_params):
            if param not in sorted_varying:
                sorted_varying.append(param)

        return sorted_varying

    def _format_dynamic_title(self, title_params: List[str], results: List[Any]) -> str:
        """
        Format dynamic title based on parameter values.

        Parameters:
        -----------
        title_params : List[str]
            List of parameter names to include in title
        results : List[Any]
            List of result objects

        Returns:
        --------
        str
            Formatted title string
        """
        if not title_params or not results:
            return ""

        # Use first result for parameter values (assuming they're constant across plots)
        result = results[0]
        title_parts = []

        # Define formatting rules for title parameters
        format_rules = {
            "maxerr": ".2e",  # Scientific notation
            "f0": ".2e",
            "dt": ".2e",
            "amp_values": ".3e",
            "amp": ".3e",  # Common alias for amp_values
            "solver": "d",  # Integer
            "Nx": "d",
            "Ny": "d",
            "Nz": "d",  # Grid sizes
        }

        def get_value(result, param):
            """Get parameter value from result."""
            try:
                if hasattr(result, "attributes") and param in result.attributes:
                    return result.attributes[param]
                elif hasattr(result, param):
                    return getattr(result, param)
            except Exception:
                pass
            return None

        for param in title_params:
            value = get_value(result, param)
            if value is not None:
                # Format parameter name (capitalize)
                param_name = param.replace("_", " ").title()

                # Format value according to rules
                if param in format_rules:
                    if format_rules[param] == "d":
                        formatted_value = f"{value:d}"
                    else:
                        formatted_value = f"{value:{format_rules[param]}}"
                else:
                    # Default formatting
                    if isinstance(value, float):
                        if abs(value) >= 1000 or abs(value) < 0.001:
                            formatted_value = f"{value:.2e}"
                        else:
                            formatted_value = f"{value:.3f}"
                    else:
                        formatted_value = str(value)

                title_parts.append(f"{param_name} = {formatted_value}")

        return ", ".join(title_parts)

    def _format_result_label(
        self, result: Any, varying_params: Optional[List[str]] = None
    ) -> str:
        """
        Format result label showing only varying parameters with proper precision.

        Parameters:
        -----------
        result : Any
            Result object with attributes
        varying_params : List[str], optional
            List of parameters that vary between results. If None, uses default behavior.

        Returns:
        --------
        str
            Formatted label string
        """
        label_parts = []

        # Define formatting rules for different parameters
        format_rules = {
            "maxerr": ".2e",  # Scientific notation with 2 decimal places
            "f0": ".2e",  # Scientific notation for frequency
            "dt": ".2e",  # Scientific notation for time step
            "amp_values": ".3e",  # Scientific notation for amplitude
            "solver": "d",  # Integer for solver
            "Nx": "d",  # Integer for grid size
            "Ny": "d",  # Integer for grid size
            "Nz": "d",  # Integer for grid size
            "PBCx": "d",  # Integer for PBC
            "PBCy": "d",  # Integer for PBC
            "PBCz": "d",  # Integer for PBC
        }

        def get_value(result, param):
            """Get parameter value from result object, handling both attributes dict and direct attributes."""
            try:
                # Try attributes dict first (ZarrJobResult pattern)
                if hasattr(result, "attributes") and param in result.attributes:
                    return result.attributes[param]
                # Try direct attribute access
                elif hasattr(result, param):
                    return getattr(result, param)
                else:
                    return None
            except Exception:
                return None

        # If varying parameters are provided, use only those
        if varying_params is not None:
            params_to_show = varying_params[: self.config.max_legend_params]

            for param in params_to_show:
                value = get_value(result, param)
                if value is not None:
                    format_spec = format_rules.get(param, "g")

                    try:
                        if format_spec == "d":
                            formatted_value = f"{int(value)}"
                        elif format_spec.endswith("e"):
                            formatted_value = f"{float(value):{format_spec}}"
                        else:
                            formatted_value = f"{value:{format_spec}}"

                        label_parts.append(f"{param}={formatted_value}")
                    except (ValueError, TypeError):
                        label_parts.append(f"{param}={value}")

        else:
            # Fallback to original behavior if no varying parameters specified
            priority_params = ["solver", "f0", "maxerr", "Nx", "Ny", "Nz"]

            # Add priority parameters first
            for param in priority_params:
                value = get_value(result, param)
                if value is not None:
                    format_spec = format_rules.get(param, "g")

                    try:
                        if format_spec == "d":
                            formatted_value = f"{int(value)}"
                        elif format_spec.endswith("e"):
                            formatted_value = f"{float(value):{format_spec}}"
                        else:
                            formatted_value = f"{value:{format_spec}}"

                        label_parts.append(f"{param}={formatted_value}")
                    except (ValueError, TypeError):
                        label_parts.append(f"{param}={value}")

            # Add other interesting parameters (limited by max_legend_params to avoid clutter)
            max_additional = max(
                0, self.config.max_legend_params - len(priority_params)
            )
            other_params_added = 0

            # Get all available parameters
            available_params = set()
            if hasattr(result, "attributes") and isinstance(result.attributes, dict):
                available_params.update(result.attributes.keys())

            for attr_name in dir(result):
                if (
                    not attr_name.startswith("_")
                    and attr_name not in ["path", "attributes"]
                    and not callable(getattr(result, attr_name, None))
                ):
                    available_params.add(attr_name)

            for attr_name in sorted(available_params):
                if other_params_added >= max_additional or attr_name in priority_params:
                    continue

                value = get_value(result, attr_name)
                if (
                    value is not None
                    and isinstance(value, (int, float))
                    and not callable(value)
                ):
                    format_spec = format_rules.get(attr_name, ".2g")

                    try:
                        if format_spec == "d":
                            formatted_value = f"{int(value)}"
                        elif format_spec.endswith("e"):
                            formatted_value = f"{float(value):{format_spec}}"
                        else:
                            formatted_value = f"{value:{format_spec}}"

                        label_parts.append(f"{attr_name}={formatted_value}")
                        other_params_added += 1
                    except Exception:
                        pass

        return ", ".join(label_parts) if label_parts else "Dataset"

    def snapshot(
        self,
        dset: Optional[str] = None,
        z: int = 0,
        t: int = -1,
        ax: Optional[Axes] = None,
        repeat: int = 1,
        zero: Optional[bool] = None,
    ) -> Axes:
        """
        Create a snapshot visualization of magnetization data.

        Parameters:
        -----------
        dset : str, optional
            Dataset name. If None, automatically selects the largest m dataset.
            Dataset name to visualize
        z : int, default 0
            Z-slice to display
        t : int, default -1
            Time step to display (-1 for last)
        ax : Optional[Axes], default None
            Matplotlib axes to plot on (creates new if None)
        repeat : int, default 1
            Number of times to tile the image
        zero : Optional[bool], default None
            Reference time step to subtract (for difference plots)

        Returns:
        --------
        Axes
            Matplotlib axes object with the plot
        """
        if not self.results:
            raise ValueError("No results available for plotting")

        # Use the first result for now
        result = self.results[0]

        # Auto-select dataset if not provided
        if dset is None:
            dset = result.get_largest_m_dataset()
            log.info(f"Auto-selected dataset: {dset}")

        # Get the magnetization data
        arr = result.get_np3d(dset, (t, z, slice(None), slice(None), slice(None)))
        if ax is None:
            shape_ratio = arr.shape[1] / arr.shape[0]
            _, ax = plt.subplots(1, 1, figsize=(4 * shape_ratio, 4), dpi=100)
        if zero is not None:
            arr -= result.get_np3d(
                dset, (zero, z, slice(None), slice(None), slice(None))
            )

        arr = np.tile(arr, (repeat, repeat, 1))
        u = arr[:, :, 0]
        v = arr[:, :, 1]
        w = arr[:, :, 2]

        alphas = -np.abs(w) + 1
        hsl = np.ones((u.shape[0], u.shape[1], 3), dtype=np.float32)
        hsl[:, :, 0] = np.angle(u + 1j * v) / np.pi / 2  # normalization
        hsl[:, :, 1] = np.sqrt(u**2 + v**2 + w**2)
        hsl[:, :, 2] = (w + 1) / 2
        rgb = hsl2rgb(hsl)

        stepx = max(int(u.shape[1] / 20), 1)
        stepy = max(int(u.shape[0] / 20), 1)
        scale = 1 / max(stepx, stepy)
        x, y = np.meshgrid(
            np.arange(0, u.shape[1], stepx) * float(result.z.attrs["dx"]) * 1e9,
            np.arange(0, u.shape[0], stepy) * float(result.z.attrs["dy"]) * 1e9,
        )

        antidots = np.ma.masked_not_equal(result[dset][0, 0, :, :, 2], 0)
        antidots = np.tile(antidots, (repeat, repeat))

        ax.quiver(
            x,
            y,
            u[::stepy, ::stepx],
            v[::stepy, ::stepx],
            alpha=alphas[::stepy, ::stepx],
            angles="xy",
            scale_units="xy",
            scale=scale,
        )

        ax.imshow(
            rgb,
            interpolation="None",
            origin="lower",
            aspect="equal",
            cmap="hsv",
            vmin=-np.pi,
            vmax=np.pi,
            extent=(
                0,
                rgb.shape[1] * float(result.z.attrs["dx"]) * 1e9,
                0,
                rgb.shape[0] * float(result.z.attrs["dy"]) * 1e9,
            ),
        )

        ax.set(title=result.name, xlabel="x (nm)", ylabel="y (nm)")

        # if not isinstance(ax, Axes):
        #     raise ValueError("ax must be a matplotlib Axes object")

        return ax


def hsl2rgb(hsl: np.ndarray) -> np.ndarray:
    """
    Convert HSL color space to RGB.

    Parameters:
    -----------
    hsl : np.ndarray
        HSL color array with shape (..., 3)

    Returns:
    --------
    np.ndarray
        RGB color array with same shape
    """
    h = hsl[..., 0] * 360
    s = hsl[..., 1]
    l = hsl[..., 2]  # noqa: E741

    rgb = np.zeros_like(hsl)
    for i, n in enumerate([0, 8, 4]):
        k = (n + h / 30) % 12
        a = s * np.minimum(l, 1 - l)
        k = np.minimum(k - 3, 9 - k)
        k = np.clip(k, -1, 1)
        rgb[..., i] = l - a * k
    rgb = np.clip(rgb, 0, 1)
    return rgb


def rgb2hsl(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB color space to HSL.

    Parameters:
    -----------
    rgb : np.ndarray
        RGB color array with shape (..., 3)

    Returns:
    --------
    np.ndarray
        HSL color array with same shape
    """
    hsl = np.ones_like(rgb)
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            r, g, b = rgb[i, j]
            h, l, s = colorsys.rgb_to_hls(r, g, b)  # noqa: E741
            hsl[i, j, 0] = h
            hsl[i, j, 1] = s
            hsl[i, j, 2] = l
    return hsl


def load_paper_style(verbose: bool = True) -> bool:
    """Load paper-ready plotting style."""
    if not MATPLOTLIB_AVAILABLE:
        return False

    try:
        # Get the style file path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        style_file = os.path.join(current_dir, "paper.mplstyle")

        if os.path.exists(style_file):
            plt.style.use(style_file)
            if verbose:
                print(f"Loaded paper style from: {style_file}")
            return True
        else:
            if verbose:
                print(f"Paper style file not found: {style_file}")
            return False

    except Exception as e:
        if verbose:
            print(f"Error loading paper style: {e}")
        return False


def setup_custom_fonts(verbose: bool = True) -> bool:
    """Setup custom fonts - convenience function."""
    return FontManager.setup_custom_fonts(verbose)


def apply_custom_colors(colors: Dict[str, str]) -> None:
    """Apply custom color scheme to matplotlib."""
    if not MATPLOTLIB_AVAILABLE:
        return

    try:
        # Apply text colors
        if "text" in colors:
            plt.rcParams["text.color"] = colors["text"]
            plt.rcParams["axes.labelcolor"] = colors["text"]
            plt.rcParams["xtick.color"] = colors["text"]
            plt.rcParams["ytick.color"] = colors["text"]

        # Apply axes colors
        if "axes" in colors:
            plt.rcParams["axes.edgecolor"] = colors["axes"]
            plt.rcParams["axes.spines.left"] = True
            plt.rcParams["axes.spines.bottom"] = True
            plt.rcParams["axes.spines.top"] = False
            plt.rcParams["axes.spines.right"] = False

        # Apply grid colors
        if "grid" in colors:
            plt.rcParams["grid.color"] = colors["grid"]

    except Exception:
        pass


# Convenience object for font operations
class FontUtils:
    """Utility object for font management."""

    @staticmethod
    def get_available() -> List[str]:
        """Get available fonts."""
        return FontManager.get_available_fonts()

    @staticmethod
    def setup_custom(verbose: bool = True) -> bool:
        """Setup custom fonts."""
        return FontManager.setup_custom_fonts(verbose)


# Create a fonts instance for backward compatibility
fonts = FontUtils()
