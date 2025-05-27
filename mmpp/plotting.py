from typing import Optional, Dict, List, Union, Any, Tuple
import os
import numpy as np
from dataclasses import dataclass

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
    colors: Dict[str, str] = None
    
    def __post_init__(self):
        """Initialize default colors."""
        if self.colors is None:
            self.colors = {
                "text": '#808080',
                "axes": '#808080',
                "grid": '#cccccc'
            }


def setup_custom_fonts():
    """Setup custom fonts including Arial."""
    try:
        # Import fonts from custom directory
        font_dirs = ['./fonts']
        
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                font_files = font_manager.findSystemFonts(fontpaths=[font_dir])
                for font_file in font_files:
                    try:
                        font_manager.fontManager.addfont(font_file)
                    except Exception as e:
                        print(f"Warning: Could not add font {font_file}: {e}")
        
        # Set Arial as default font
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.sans-serif'] = ['Arial'] + plt.rcParams['font.sans-serif']
        
        # Check if Arial is available
        available_fonts = set(f.name for f in font_manager.fontManager.ttflist)
        if 'Arial' in available_fonts:
            print("✓ Arial font loaded successfully")
        else:
            print("⚠ Arial font not found, using default fonts")
            
        return True
        
    except Exception as e:
        print(f"Warning: Font setup failed: {e}")
        return False


def load_paper_style():
    """Load custom paper style."""
    try:
        # Try to find paper.mplstyle in current directory or relative to this file
        style_paths = [
            './paper.mplstyle',
            os.path.join(os.path.dirname(__file__), 'paper.mplstyle'),
            '/mnt/storage_2/scratch/pl0095-01/zelent/mannga/bowtie/mateusz/sinc/solver_test/paper.mplstyle'
        ]
        
        for style_path in style_paths:
            if os.path.exists(style_path):
                plt.style.use(style_path)
                print(f"✓ Loaded paper style from: {style_path}")
                return True
        
        print("⚠ paper.mplstyle not found, using default style")
        return False
        
    except Exception as e:
        print(f"Warning: Could not load paper style: {e}")
        return False


def apply_custom_colors(colors: Dict[str, str]):
    """Apply custom colors to matplotlib rcParams."""
    try:
        if "text" in colors:
            plt.rcParams['text.color'] = colors["text"]
            plt.rcParams['axes.labelcolor'] = colors["text"]
            plt.rcParams['xtick.color'] = colors["text"]
            plt.rcParams['ytick.color'] = colors["text"]
        
        if "axes" in colors:
            plt.rcParams['axes.edgecolor'] = colors["axes"]
        
        if "grid" in colors:
            plt.rcParams['grid.color'] = colors["grid"]
            
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
    
    def __init__(self, results, mmpp_instance=None):
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
            raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")
        
        # Initialize console if available
        if RICH_AVAILABLE:
            self.console = Console()
        
        # Setup custom styling
        self._setup_styling()
    
    def _setup_styling(self):
        """Setup custom fonts and styling."""
        try:
            # Setup custom fonts if enabled
            if self.config.use_custom_fonts:
                setup_custom_fonts()
            
            # Load paper style
            if self.config.style == "paper":
                if not load_paper_style():
                    # Fallback to a standard style
                    try:
                        plt.style.use('seaborn-v0_8')
                    except:
                        try:
                            plt.style.use('default')
                        except:
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
            
        except Exception as e:
            print(f"Warning: Styling setup failed: {e}")
    
    def __repr__(self) -> str:
        """Rich representation of the plotter."""
        if RICH_AVAILABLE and self.mmpp and self.mmpp._interactive_mode:
            return self._rich_plotter_display()
        else:
            return self._basic_plotter_display()
    
    def _rich_plotter_display(self) -> str:
        """Generate rich display for plotter."""
        summary_text = Text()
        summary_text.append(f"📊 MMPP Plotter for {len(self.results)} datasets\n", style="bold cyan")
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
            ("set_style(style_name)", "Change matplotlib style")
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
            "plotter.reset_style()  # Reset to paper style"
        ]
        
        for example in examples:
            examples_text.append(f"  {example}\n", style="code")
        
        if RICH_AVAILABLE:
            try:
                with self.console.capture() as capture:
                    self.console.print(Panel.fit(summary_text, title="[bold blue]MMPP Plotter[/bold blue]", border_style="blue"))
                    self.console.print("")
                    self.console.print(Columns([
                        Panel.fit(methods_text, title="[bold yellow]Methods[/bold yellow]", border_style="yellow"),
                        Panel.fit(examples_text, title="[bold green]Examples[/bold green]", border_style="green")
                    ]))
                return capture.get()
            except:
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
    
    def configure(self, **kwargs) -> 'MMPPlotter':
        """
        Configure plot settings.
        
        Parameters:
        -----------
        **kwargs : Any
            Configuration options (figsize, dpi, style, colormap, colors, etc.)
            
        Returns:
        --------
        MMPPlotter
            Self for method chaining
        """
        style_changed = False
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                if key == 'style':
                    style_changed = True
            else:
                print(f"Warning: Unknown configuration option '{key}'")
        
        # Apply style if changed
        if style_changed:
            self._setup_styling()
        
        # Apply color changes if provided
        if 'colors' in kwargs:
            apply_custom_colors(self.config.colors)
        
        return self
    
    def reset_style(self) -> 'MMPPlotter':
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
    
    def set_style(self, style_name: str) -> 'MMPPlotter':
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
            comp_map = {'x': 0, 'y': 1, 'z': 2}
            return comp_map.get(comp.lower(), 2)
        return int(comp)
    
    def _load_pyzfn_job(self, result) -> Pyzfn:
        """Load a Pyzfn job from a result."""
        return Pyzfn(result.path)
    
    def _extract_data(self, job: Pyzfn, dataset_name: str, x_series: str = None, 
                     comp: Union[str, int] = None, average: tuple = None) -> tuple:
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
                'path': job.path,
                'dataset': dataset_name,
                'component': comp,
                'averaged_axes': average,
                'shape': y_data.shape,
                'attrs': dict(dataset.attrs) if hasattr(dataset, 'attrs') else {}
            }
            
            return x_data, y_data, metadata
            
        except Exception as e:
            print(f"Error extracting data from {job.path}: {e}")
            return None, None, None
    
    def plot(self, x_series: str, y_series: str, comp: Union[str, int] = None, 
             average: tuple = None, figsize: tuple = None, title: str = None,
             xlabel: str = None, ylabel: str = None, legend_labels: List[str] = None,
             colors: List[str] = None, save_path: str = None, 
             paper_ready: bool = False, **kwargs) -> tuple:
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
            colors = [cmap(i / max(1, len(self.results) - 1)) for i in range(len(self.results))]
        
        # Progress bar if available
        iterator = tqdm(self.results, desc="Processing datasets") if TQDM_AVAILABLE else self.results
        
        plotted_data = []
        
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
                        # Create informative label from result attributes
                        label_parts = []
                        for key in ['solver', 'f0', 'maxerr']:
                            if hasattr(result, key):
                                value = getattr(result, key)
                                label_parts.append(f"{key}={value}")
                        label = ", ".join(label_parts) if label_parts else f"Dataset {i+1}"
                    
                    # Plot
                    line = ax.plot(x_data, y_data, 
                                  color=colors[i % len(colors)],
                                  alpha=self.config.line_alpha,
                                  linewidth=self.config.line_width,
                                  label=label,
                                  **kwargs)
                    
                    plotted_data.append({
                        'result': result,
                        'x_data': x_data,
                        'y_data': y_data,
                        'metadata': metadata,
                        'line': line[0]
                    })
                
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
                comp_name = ['x', 'y', 'z'][self._parse_component(comp)]
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
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        # Store plot data for further analysis
        self._last_plot_data = plotted_data
        
        # Restore original style if paper_ready was used
        if paper_ready and 'original_style' in locals():
            self.config.style = original_style
        
        return fig, ax
    
    def plot_time_series(self, dataset: str, comp: Union[str, int] = 'z', 
                        average: tuple = (1, 2, 3), **kwargs) -> tuple:
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
        return self.plot('t', dataset, comp=comp, average=average, 
                        xlabel='Time', **kwargs)
    
    def plot_components(self, dataset: str, time_slice: int = -1, 
                       average: tuple = (1, 2, 3), **kwargs) -> tuple:
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
        components = ['x', 'y', 'z']
        
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
                        
                        ax.bar(j, value, alpha=0.7, 
                              label=f"Result {j+1}" if i == 0 else "")
                
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
        if hasattr(self, '_last_plot_data'):
            return self._last_plot_data
        else:
            print("No plot data available. Run a plot method first.")
            return []
    
    def save_all_data(self, filename: str, format: str = 'npz') -> None:
        """
        Save all plotted data to file.
        
        Parameters:
        -----------
        filename : str
            Output filename
        format : str, optional
            Format ('npz', 'csv', 'json')
        """
        if not hasattr(self, '_last_plot_data'):
            print("No plot data to save. Run a plot method first.")
            return
        
        if format == 'npz':
            data_dict = {}
            for i, item in enumerate(self._last_plot_data):
                data_dict[f'x_data_{i}'] = item['x_data']
                data_dict[f'y_data_{i}'] = item['y_data']
            np.savez(filename, **data_dict)
        
        elif format == 'csv':
            import pandas as pd
            # Create DataFrame with all data
            df_dict = {}
            for i, item in enumerate(self._last_plot_data):
                df_dict[f'x_{i}'] = item['x_data']
                df_dict[f'y_{i}'] = item['y_data']
            
            df = pd.DataFrame(df_dict)
            df.to_csv(filename, index=False)
        
        print(f"Data saved to: {filename}")


class PlotterProxy:
    """Proxy class to provide plotting functionality to search results."""
    
    def __init__(self, results, mmpp_instance=None):
        self._results = results
        self._mmpp = mmpp_instance
        self._plotter = None
    
    @property
    def mpl(self) -> MMPPlotter:
        """Get the matplotlib plotter instance."""
        if self._plotter is None:
            self._plotter = MMPPlotter(self._results, self._mmpp)
        return self._plotter
    
    def __len__(self) -> int:
        return len(self._results)
    
    def __getitem__(self, index):
        return self._results[index]
    
    def __iter__(self):
        return iter(self._results)
