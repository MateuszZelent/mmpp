from typing import Optional, Dict, List, Union, Any, Tuple, Iterator
import os
import numpy as np
from dataclasses import dataclass
import platform

# Import for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
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
    from rich.console import Console
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from pyzfn import Pyzfn


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
    sort_results: bool = True   # Whether to sort results by parameters

    def __post_init__(self) -> None:
        """Initialize default colors."""
        if self.colors is None:
            self.colors = {"text": "#808080", "axes": "#808080", "grid": "#cccccc"}


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
                    print(f"🔍 Checking font directory: {font_dir}")
                font_files = font_manager.findSystemFonts(fontpaths=[font_dir])
                for font_file in font_files:
                    try:
                        font_manager.fontManager.addfont(font_file)
                        fonts_loaded = True
                        if verbose:
                            print(f"✓ Added font: {os.path.basename(font_file)}")
                    except Exception as e:
                        if verbose:
                            print(f"Warning: Could not add font {font_file}: {e}")

        # Rebuild font cache if fonts were loaded
        if fonts_loaded:
            font_manager.fontManager.findfont('Arial', rebuild_if_missing=True)

        # Set Arial as default font
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.sans-serif"] = ["Arial"] + plt.rcParams["font.sans-serif"]

        # Check if Arial is available
        available_fonts = set(f.name for f in font_manager.fontManager.ttflist)
        if "Arial" in available_fonts:
            if verbose:
                print("✓ Arial font loaded successfully")
        else:
            if verbose:
                print("⚠ Arial font not found, using default fonts")

        _FONTS_INITIALIZED = True
        return True

    except Exception as e:
        if verbose:
            print(f"Warning: Font setup failed: {e}")
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
                    print(f"✓ Loaded paper style from: {style_path}")
                _STYLE_INITIALIZED = True
                return True

        if verbose:
            print("⚠ paper.mplstyle not found, using default style")
        _STYLE_INITIALIZED = True
        return False

    except Exception as e:
        if verbose:
            print(f"Warning: Could not load paper style: {e}")
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
        print(f"Warning: Could not apply custom colors: {e}")


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
                    print(f"Warning: Could not load style '{self.config.style}': {e}")

            # Apply custom colors
            apply_custom_colors(self.config.colors)

            # Mark styling as completed globally
            _STYLING_SETUP_COMPLETED = True

        except Exception as e:
            print(f"Warning: Styling setup failed: {e}")
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
        **kwargs : Any
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
                print(f"Warning: Unknown configuration option '{key}'")

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
                print(f"✓ Applied style: {style_name}")
        except Exception as e:
            print(f"Warning: Could not apply style '{style_name}': {e}")

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
                    print(f"Warning: '{x_series}' not found, using indices")

            # Extract y-axis data
            y_data = dataset[...]

            # Select component if specified
            if comp is not None:
                comp_idx = self._parse_component(comp)
                if y_data.ndim > comp_idx:
                    y_data = y_data[..., comp_idx]
                else:
                    print(f"Warning: Component {comp} not available, using full data")

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
            print(f"Error extracting data from {job.path}: {e}")
            return None, None, None

    def plot(
        self,
        x_series: str,
        y_series: str,
        comp: Optional[Union[str, int]] = None,
        average: Optional[Tuple[Any, ...]] = None,
        figsize: Optional[Tuple[Any, ...]] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        legend_labels: Optional[List[str]] = None,
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
        title : str, optional
            Plot title
        xlabel : str, optional
            X-axis label
        ylabel : str, optional
            Y-axis label
        legend_labels : List[str], optional
            Custom legend labels
        colors : List[str], optional
            Custom colors for each line
        save_path : str, optional
            Path to save the figure
        paper_ready : bool, optional
            If True, apply paper-ready styling (default: False)
        **kwargs : Any
            Additional matplotlib plot arguments

        Returns:
        --------
        tuple
            (figure, axes) matplotlib objects
        """
        if not self.results:
            print("No results to plot")
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
        varying_params = self._get_varying_parameters(sorted_results) if len(sorted_results) > 1 else []
        
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
                print(f"Error plotting {result.path}: {e}")
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

        if title:
            ax.set_title(title, fontsize=self.config.title_fontsize)
        else:
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
            print(f"Figure saved to: {save_path}")

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
        **kwargs : Any
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
        **kwargs : Any
            Additional plot arguments

        Returns:
        --------
        tuple
            (figure, axes) matplotlib objects
        """
        if not self.results:
            print("No results to plot")
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
                            j, value, alpha=0.7, label=f"Result {j+1}" if i == 0 else ""
                        )

                except Exception as e:
                    print(f"Error plotting component {comp} for {result.path}: {e}")
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
            print("No plot data available. Run a plot method first.")
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
            print("No plot data to save. Run a plot method first.")
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

        print(f"Data saved to: {filename}")

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
            important_params = ['solver', 'f0', 'maxerr', 'Nx', 'Ny', 'Nz', 'PBCx', 'PBCy', 'PBCz']
            
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
                if (not attr_name.startswith('_') and 
                    attr_name not in important_params and
                    attr_name not in ['path', 'attributes']):
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
        
        # Collect all potential parameters
        all_params = set()
        for result in results:
            for attr_name in dir(result):
                if (not attr_name.startswith('_') and 
                    attr_name not in ['path', 'attributes'] and
                    not callable(getattr(result, attr_name, None))):
                    try:
                        value = getattr(result, attr_name)
                        if isinstance(value, (int, float, str, bool)):
                            all_params.add(attr_name)
                    except Exception:
                        pass
        
        # Check which parameters actually vary
        varying_params = []
        for param in all_params:
            values = []
            for result in results:
                if hasattr(result, param):
                    try:
                        value = getattr(result, param)
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
        priority_params = ['solver', 'f0', 'maxerr', 'dt', 'Nx', 'Ny', 'Nz', 'PBCx', 'PBCy', 'PBCz', 'amp_values']
        
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

    def _format_result_label(self, result: Any, varying_params: Optional[List[str]] = None) -> str:
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
            'maxerr': '.2e',      # Scientific notation with 2 decimal places
            'f0': '.2e',          # Scientific notation for frequency
            'dt': '.2e',          # Scientific notation for time step
            'amp_values': '.3e',  # Scientific notation for amplitude
            'solver': 'd',        # Integer for solver
            'Nx': 'd',           # Integer for grid size
            'Ny': 'd',           # Integer for grid size  
            'Nz': 'd',           # Integer for grid size
            'PBCx': 'd',         # Integer for PBC
            'PBCy': 'd',         # Integer for PBC
            'PBCz': 'd',         # Integer for PBC
        }
        
        # If varying parameters are provided, use only those
        if varying_params is not None:
            params_to_show = varying_params[:self.config.max_legend_params]
            
            for param in params_to_show:
                if hasattr(result, param):
                    value = getattr(result, param)
                    format_spec = format_rules.get(param, 'g')
                    
                    try:
                        if format_spec == 'd':
                            formatted_value = f"{int(value)}"
                        elif format_spec.endswith('e'):
                            formatted_value = f"{float(value):{format_spec}}"
                        else:
                            formatted_value = f"{value:{format_spec}}"
                        
                        label_parts.append(f"{param}={formatted_value}")
                    except (ValueError, TypeError):
                        label_parts.append(f"{param}={value}")
        
        else:
            # Fallback to original behavior if no varying parameters specified
            priority_params = ['solver', 'f0', 'maxerr', 'Nx', 'Ny', 'Nz']
            
            # Add priority parameters first
            for param in priority_params:
                if hasattr(result, param):
                    value = getattr(result, param)
                    format_spec = format_rules.get(param, 'g')
                    
                    try:
                        if format_spec == 'd':
                            formatted_value = f"{int(value)}"
                        elif format_spec.endswith('e'):
                            formatted_value = f"{float(value):{format_spec}}"
                        else:
                            formatted_value = f"{value:{format_spec}}"
                        
                        label_parts.append(f"{param}={formatted_value}")
                    except (ValueError, TypeError):
                        label_parts.append(f"{param}={value}")
            
            # Add other interesting parameters (limited by max_legend_params to avoid clutter)
            max_additional = max(0, self.config.max_legend_params - len(priority_params))
            other_params_added = 0
            for attr_name in sorted(dir(result)):
                if (other_params_added >= max_additional or 
                    attr_name.startswith('_') or 
                    attr_name in priority_params or
                    attr_name in ['path', 'attributes']):
                    continue
                    
                try:
                    value = getattr(result, attr_name)
                    if isinstance(value, (int, float)) and not callable(value):
                        format_spec = format_rules.get(attr_name, '.2g')
                        
                        if format_spec == 'd':
                            formatted_value = f"{int(value)}"
                        elif format_spec.endswith('e'):
                            formatted_value = f"{float(value):{format_spec}}"
                        else:
                            formatted_value = f"{value:{format_spec}}"
                        
                        label_parts.append(f"{attr_name}={formatted_value}")
                        other_params_added += 1
                except Exception:
                    pass
        
        return ", ".join(label_parts) if label_parts else "Dataset"

class FontManager:
    """
    Comprehensive font management system for mmpp library.
    
    Provides methods to discover, manage, and configure fonts across different platforms.
    """
    
    def __init__(self):
        self._custom_paths: List[str] = []
        self._default_font: Optional[str] = None
        self._font_cache: Dict[str, List[str]] = {}
        self._initialize_system_paths()
    
    def _initialize_system_paths(self) -> None:
        """Initialize system font paths based on the operating system."""
        self._system_paths = []
        
        system = platform.system().lower()
        
        if system == "windows":
            # Windows font paths
            windows_fonts = [
                "C:/Windows/Fonts",
                os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts"),
                os.path.expanduser("~/AppData/Roaming/Microsoft/Windows/Fonts")
            ]
            self._system_paths.extend([p for p in windows_fonts if os.path.exists(p)])
            
        elif system == "darwin":  # macOS
            macos_fonts = [
                "/System/Library/Fonts",
                "/Library/Fonts",
                os.path.expanduser("~/Library/Fonts"),
                "/System/Library/Fonts/Supplemental"
            ]
            self._system_paths.extend([p for p in macos_fonts if os.path.exists(p)])
            
        else:  # Linux and other Unix-like systems
            linux_fonts = [
                "/usr/share/fonts",
                "/usr/local/share/fonts",
                os.path.expanduser("~/.local/share/fonts"),
                os.path.expanduser("~/.fonts"),
                "/usr/share/fonts/TTF",
                "/usr/share/fonts/truetype"
            ]
            self._system_paths.extend([p for p in linux_fonts if os.path.exists(p)])
        
        # Add mmpp package fonts
        package_dir = os.path.dirname(os.path.abspath(__file__))
        mmpp_fonts_path = os.path.join(package_dir, "fonts")
        if os.path.exists(mmpp_fonts_path):
            self._system_paths.append(mmpp_fonts_path)
    
    @property
    def available(self) -> List[str]:
        """
        Get list of all available font families.
        
        Returns:
            List of font family names available on the system
        """
        if not MATPLOTLIB_AVAILABLE:
            return []
        
        # Get fonts from matplotlib font manager
        font_families = set()
        
        # Add system fonts detected by matplotlib
        for font in font_manager.fontManager.ttflist:
            font_families.add(font.name)
        
        # Add fonts from custom paths
        for path in self.paths:
            for font_file in self._scan_font_directory(path):
                try:
                    font_prop = font_manager.FontProperties(fname=font_file)
                    font_families.add(font_prop.get_name())
                except Exception:
                    continue
        
        return sorted(list(font_families))
    
    @property
    def paths(self) -> List[str]:
        """
        Get list of all font search paths.
        
        Returns:
            List of absolute paths where fonts are searched
        """
        all_paths = self._system_paths.copy()
        all_paths.extend(self._custom_paths)
        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in all_paths:
            abs_path = os.path.abspath(path)
            if abs_path not in seen:
                seen.add(abs_path)
                unique_paths.append(abs_path)
        return unique_paths
    
    @property
    def default_font(self) -> Optional[str]:
        """
        Get the current default font family.
        
        Returns:
            Name of the default font family, or None if not set
        """
        return self._default_font
    
    def add_path(self, font_path: str) -> bool:
        """
        Add a custom font directory to the search paths.
        
        Args:
            font_path: Absolute or relative path to font directory
            
        Returns:
            True if path was added successfully, False otherwise
        """
        abs_path = os.path.abspath(font_path)
        
        if not os.path.exists(abs_path):
            print(f"Warning: Font path does not exist: {abs_path}")
            return False
        
        if not os.path.isdir(abs_path):
            print(f"Warning: Font path is not a directory: {abs_path}")
            return False
        
        if abs_path not in self._custom_paths:
            self._custom_paths.append(abs_path)
            self._font_cache.clear()  # Clear cache to force refresh
            
            # Refresh matplotlib font cache if available
            if MATPLOTLIB_AVAILABLE:
                font_manager.fontManager.addfont(abs_path)
                
            return True
        
        return False
    
    def set_default_font(self, font_family: str) -> bool:
        """
        Set the default font family for mmpp plots.
        
        Args:
            font_family: Name of the font family to set as default
            
        Returns:
            True if font was set successfully, False if font not found
        """
        available_fonts = self.available
        
        # Check if font is available
        if font_family not in available_fonts:
            # Try case-insensitive match
            font_lower = font_family.lower()
            matches = [f for f in available_fonts if f.lower() == font_lower]
            
            if matches:
                font_family = matches[0]
            else:
                print(f"Warning: Font '{font_family}' not found in available fonts")
                print(f"Available fonts: {', '.join(available_fonts[:10])}...")
                return False
        
        self._default_font = font_family
        
        # Set as matplotlib default if available
        if MATPLOTLIB_AVAILABLE:
            plt.rcParams['font.family'] = font_family
            
        return True
    
    def find_font(self, font_name: str) -> Optional[str]:
        """
        Find the full path to a specific font file.
        
        Args:
            font_name: Name of the font to find (with or without extension)
            
        Returns:
            Full path to the font file, or None if not found
        """
        # If it's already a filename with extension, search for it directly
        if font_name.endswith(('.ttf', '.otf', '.woff', '.woff2', '.TTF', '.OTF')):
            for path in self.paths:
                # First try direct path
                font_path = os.path.join(path, font_name)
                if os.path.exists(font_path):
                    return os.path.abspath(font_path)
                
                # Then search recursively in subdirectories
                for root, dirs, files in os.walk(path):
                    if font_name in files:
                        return os.path.abspath(os.path.join(root, font_name))
        
        # If no extension, try common extensions
        for ext in ['.ttf', '.otf', '.TTF', '.OTF', '.woff', '.woff2']:
            result = self.find_font(font_name + ext)
            if result:
                return result
        
        # Also try to find by font family name using matplotlib
        if MATPLOTLIB_AVAILABLE:
            try:
                # Search through all font files to find matching family name
                for path in self.paths:
                    for font_file in self._scan_font_directory(path):
                        try:
                            font_prop = font_manager.FontProperties(fname=font_file)
                            if font_prop.get_name().lower() == font_name.lower():
                                return font_file
                        except Exception:
                            continue
            except Exception:
                pass
        
        return None
    
    def refresh(self) -> None:
        """
        Refresh the font cache and reload font information.
        """
        self._font_cache.clear()
        
        if MATPLOTLIB_AVAILABLE:
            # Rebuild matplotlib font cache
            font_manager.fontManager = font_manager.FontManager()
            
            # Re-add custom paths
            for path in self._custom_paths:
                font_manager.fontManager.addfont(path)
    
    def _scan_font_directory(self, directory: str) -> List[str]:
        """
        Scan a directory for font files.
        
        Args:
            directory: Path to directory to scan
            
        Returns:
            List of full paths to font files found
        """
        if directory in self._font_cache:
            return self._font_cache[directory]
        
        font_extensions = {'.ttf', '.otf', '.TTF', '.OTF', '.woff', '.woff2'}
        font_files = []
        
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if any(file.endswith(ext) for ext in font_extensions):
                        font_files.append(os.path.join(root, file))
        except (OSError, PermissionError):
            pass
        
        self._font_cache[directory] = font_files
        return font_files
    
    def show_font_structure(self, max_depth: int = 3) -> None:
        """
        Display the font directory structure for debugging.
        
        Args:
            max_depth: Maximum depth to traverse (default: 3)
        """
        print("Font Directory Structure:")
        print("=" * 50)
        
        for i, path in enumerate(self.paths):
            print(f"\n{i+1}. {path}")
            if not os.path.exists(path):
                print("   [PATH DOES NOT EXIST]")
                continue
                
            try:
                font_count = 0
                for root, dirs, files in os.walk(path):
                    # Calculate current depth
                    depth = root.replace(path, '').count(os.sep)
                    if depth > max_depth:
                        continue
                    
                    # Show directory structure
                    indent = "   " + "  " * depth
                    folder_name = os.path.basename(root) if root != path else "[ROOT]"
                    
                    # Count font files in this directory
                    font_files = [f for f in files if any(f.lower().endswith(ext) for ext in ['.ttf', '.otf', '.woff', '.woff2'])]
                    
                    if font_files:
                        print(f"{indent}📁 {folder_name}/ ({len(font_files)} fonts)")
                        if depth < 2:  # Show font names for shallow directories
                            for font_file in font_files[:5]:  # Show max 5 fonts
                                print(f"{indent}  └─ {font_file}")
                            if len(font_files) > 5:
                                print(f"{indent}  └─ ... and {len(font_files) - 5} more")
                        font_count += len(font_files)
                    elif depth <= 1:  # Show empty directories at shallow levels
                        print(f"{indent}📂 {folder_name}/")
                
                print(f"   Total fonts found: {font_count}")
                
            except (OSError, PermissionError) as e:
                print(f"   [ERROR: {e}]")
    
    def __repr__(self) -> str:
        """String representation showing available fonts."""
        available = self.available
        count = len(available)
        
        if count == 0:
            return "FontManager(no fonts found)"
        
        preview = available[:5]
        if count > 5:
            preview_str = ", ".join(preview) + f", ... ({count - 5} more)"
        else:
            preview_str = ", ".join(preview)
        
        return f"FontManager({count} fonts: {preview_str})"
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        return self.__repr__()


# Create global font manager instance
fonts = FontManager()

class PlotterProxy:
    """Proxy class to provide plotting functionality to search results."""

    def __init__(
        self, results: Union[List[Any], Any], mmpp_instance: Optional[Any] = None
    ) -> None:
        self._results = results
        self._mmpp = mmpp_instance
        self._plotter: Optional[MMPPlotter] = None

    @property
    def matplotlib(self) -> MMPPlotter:
        """Get the matplotlib plotter instance."""
        if self._plotter is None:
            self._plotter = MMPPlotter(self._results, self._mmpp)
        return self._plotter

    @property
    def mpl(self) -> MMPPlotter:
        """Get the matplotlib plotter instance (alias for matplotlib)."""
        return self.matplotlib

    def __len__(self) -> int:
        return len(self._results)

    def __getitem__(self, index: int) -> Any:
        return self._results[index]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._results)
