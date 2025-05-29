"""
FFT Plotting Module

Specialized plotting functionality for FFT analysis results.
"""

from typing import Optional, Dict, List, Union, Any, Tuple
import numpy as np

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
    
    def __init__(self, results: Union[List[Any], Any], mmpp_instance: Optional[Any] = None):
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
            
        self.mmpp = mmpp_instance
        self.fft_compute = FFTCompute()
        
        # Basic plot configuration
        self.config = {
            'figsize': (10, 6),
            'dpi': 100,
            'line_alpha': 0.8,
            'line_width': 2,
            'label_fontsize': 12,
            'title_fontsize': 14,
            'tick_fontsize': 10,
            'grid': True,
            'legend': True
        }
    
    def _format_result_label(self, result) -> str:
        """Format result label for plotting."""
        import os
        return os.path.basename(result.path)
    
    def power_spectrum(self, dataset_name: str = "m_z11", 
                      method: int = 1,
                      z_layer: int = -1,
                      log_scale: bool = True,
                      normalize: bool = False,
                      figsize: Optional[Tuple[float, float]] = None,
                      save_path: Optional[str] = None,
                      **kwargs) -> Tuple[Any, Any]:
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
        figsize : tuple, optional
            Figure size
        save_path : str, optional
            Path to save figure
        **kwargs : Any
            Additional FFT configuration options
            
        Returns:
        --------
        tuple
            (figure, axes) matplotlib objects
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for plotting")
        
        # Setup figure
        figsize = figsize or self.config['figsize']
        fig, ax = plt.subplots(figsize=figsize, dpi=self.config['dpi'])
        
        # Analyze all results
        for i, result in enumerate(self.results):
            try:
                fft_result = self.fft_compute.calculate_fft_data(
                    result.path, dataset_name, z_layer, method, **kwargs
                )
                
                power = np.abs(fft_result.spectrum)**2
                
                # Normalize if requested
                if normalize:
                    power = power / np.max(power)
                
                # Create label
                label = self._format_result_label(result)
                
                # Plot
                if log_scale:
                    ax.semilogy(fft_result.frequencies, power, 
                               alpha=self.config['line_alpha'],
                               linewidth=self.config['line_width'],
                               label=label)
                else:
                    ax.plot(fft_result.frequencies, power,
                           alpha=self.config['line_alpha'], 
                           linewidth=self.config['line_width'],
                           label=label)
                
            except Exception as e:
                print(f"Error analyzing result {i}: {e}")
                continue
        
        # Customize plot
        ax.set_xlabel('Frequency (Hz)', fontsize=self.config['label_fontsize'])
        ylabel = 'Normalized Power' if normalize else 'Power'
        if log_scale:
            ylabel += ' (log scale)'
        ax.set_ylabel(ylabel, fontsize=self.config['label_fontsize'])
        
        title = f"Power Spectrum - {dataset_name} (Method {method})"
        ax.set_title(title, fontsize=self.config['title_fontsize'])
        
        if self.config['grid']:
            ax.grid(True, alpha=0.3)
        
        if self.config['legend'] and len(self.results) > 1:
            ax.legend(fontsize=self.config['label_fontsize'])
        
        ax.tick_params(labelsize=self.config['tick_fontsize'])
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.config['dpi'], bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        return fig, ax
    
    def __repr__(self) -> str:
        return f"FFTPlotter({len(self.results)} results)"
