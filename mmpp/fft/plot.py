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

try:
    from ..plotting import MMPPlotter, PlotConfig
    MMPP_PLOTTING_AVAILABLE = True
except ImportError:
    MMPP_PLOTTING_AVAILABLE = False

from .main import FFTAnalyzer, FFTResult


class FFTPlotter(MMPPlotter):
    """
    Specialized plotter for FFT analysis results.
    
    Inherits from MMPPlotter and extends it with FFT-specific plotting capabilities.
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
        if not MMPP_PLOTTING_AVAILABLE:
            raise ImportError("MMPP plotting module not available")
        
        super().__init__(results, mmpp_instance)
        self.fft_analyzer = FFTAnalyzer(results, mmpp_instance)
    
    def power_spectrum(self, dataset_name: str = "m_z11", 
                      comp: Optional[Union[str, int]] = None,
                      average: Optional[Tuple[Any, ...]] = None,
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
        comp : Union[str, int], optional
            Component to analyze
        average : tuple, optional
            Axes to average over
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
        figsize = figsize or self.config.figsize
        fig, ax = plt.subplots(figsize=figsize, dpi=self.config.dpi)
        
        # Analyze all results
        fft_results = []
        for i in range(len(self.results)):
            try:
                fft_result = self.fft_analyzer.analyze_single(
                    i, dataset_name, comp=comp, average=average, **kwargs
                )
                fft_results.append(fft_result)
            except Exception as e:
                print(f"Error analyzing result {i}: {e}")
                continue
        
        if not fft_results:
            print("No FFT results to plot")
            return fig, ax
        
        # Plot power spectra
        for i, fft_result in enumerate(fft_results):
            power = fft_result.power_spectrum.copy()
            
            # Normalize if requested
            if normalize:
                power = power / np.max(power)
            
            # Create label
            result = self.results[i]
            label = self._format_result_label(result)
            
            # Plot
            if log_scale:
                ax.semilogy(fft_result.frequencies, power, 
                           alpha=self.config.line_alpha,
                           linewidth=self.config.line_width,
                           label=label)
            else:
                ax.plot(fft_result.frequencies, power,
                       alpha=self.config.line_alpha, 
                       linewidth=self.config.line_width,
                       label=label)
        
        # Customize plot
        ax.set_xlabel('Frequency (Hz)', fontsize=self.config.label_fontsize)
        ylabel = 'Normalized Power' if normalize else 'Power'
        if log_scale:
            ylabel += ' (log scale)'
        ax.set_ylabel(ylabel, fontsize=self.config.label_fontsize)
        
        title_parts = [f"Power Spectrum - {dataset_name}"]
        if comp is not None:
            title_parts.append(f"component {comp}")
        ax.set_title(" - ".join(title_parts), fontsize=self.config.title_fontsize)
        
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        
        if self.config.legend and len(fft_results) > 1:
            ax.legend(fontsize=self.config.label_fontsize)
        
        ax.tick_params(labelsize=self.config.tick_fontsize)
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        return fig, ax
    
    def phase_spectrum(self, dataset_name: str = "m_z11",
                      comp: Optional[Union[str, int]] = None,
                      average: Optional[Tuple[Any, ...]] = None,
                      unwrap: bool = True,
                      figsize: Optional[Tuple[float, float]] = None,
                      save_path: Optional[str] = None,
                      **kwargs) -> Tuple[Any, Any]:
        """
        Plot phase spectrum for all results.
        
        Parameters:
        -----------
        dataset_name : str, optional
            Dataset name (default: "m_z11")
        comp : Union[str, int], optional  
            Component to analyze
        average : tuple, optional
            Axes to average over
        unwrap : bool, optional
            Unwrap phase (default: True)
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
        figsize = figsize or self.config.figsize
        fig, ax = plt.subplots(figsize=figsize, dpi=self.config.dpi)
        
        # Analyze all results
        fft_results = []
        for i in range(len(self.results)):
            try:
                fft_result = self.fft_analyzer.analyze_single(
                    i, dataset_name, comp=comp, average=average, **kwargs
                )
                fft_results.append(fft_result)
            except Exception as e:
                print(f"Error analyzing result {i}: {e}")
                continue
        
        if not fft_results:
            print("No FFT results to plot")
            return fig, ax
        
        # Plot phase spectra
        for i, fft_result in enumerate(fft_results):
            phase = fft_result.phase_spectrum.copy()
            
            # Unwrap phase if requested
            if unwrap:
                phase = np.unwrap(phase)
            
            # Create label
            result = self.results[i]
            label = self._format_result_label(result)
            
            # Plot
            ax.plot(fft_result.frequencies, phase,
                   alpha=self.config.line_alpha,
                   linewidth=self.config.line_width,
                   label=label)
        
        # Customize plot
        ax.set_xlabel('Frequency (Hz)', fontsize=self.config.label_fontsize)
        ylabel = 'Unwrapped Phase (rad)' if unwrap else 'Phase (rad)'
        ax.set_ylabel(ylabel, fontsize=self.config.label_fontsize)
        
        title_parts = [f"Phase Spectrum - {dataset_name}"]
        if comp is not None:
            title_parts.append(f"component {comp}")
        ax.set_title(" - ".join(title_parts), fontsize=self.config.title_fontsize)
        
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        
        if self.config.legend and len(fft_results) > 1:
            ax.legend(fontsize=self.config.label_fontsize)
        
        ax.tick_params(labelsize=self.config.tick_fontsize)
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        return fig, ax
    
    def spectrogram(self, result_index: int = 0, dataset_name: str = "m_z11",
                   comp: Optional[Union[str, int]] = None,
                   average: Optional[Tuple[Any, ...]] = None,
                   window_length: Optional[int] = None,
                   overlap: float = 0.5,
                   figsize: Optional[Tuple[float, float]] = None,
                   save_path: Optional[str] = None,
                   **kwargs) -> Tuple[Any, Any]:
        """
        Plot time-frequency spectrogram for a single result.
        
        Parameters:
        -----------
        result_index : int, optional
            Index of result to analyze (default: 0)
        dataset_name : str, optional
            Dataset name (default: "m_z11")
        comp : Union[str, int], optional
            Component to analyze
        average : tuple, optional
            Axes to average over
        window_length : int, optional
            Length of analysis window (default: auto)
        overlap : float, optional
            Window overlap fraction (default: 0.5)
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
        
        try:
            import scipy.signal
        except ImportError:
            raise ImportError("Scipy required for spectrogram analysis")
        
        # Setup figure
        figsize = figsize or (12, 8)
        fig, ax = plt.subplots(figsize=figsize, dpi=self.config.dpi)
        
        # Extract time series data
        result = self.results[result_index]
        job = self.fft_analyzer._load_pyzfn_job(result)
        time_data, signal_data, metadata = self.fft_analyzer._extract_time_series(
            job, dataset_name, comp, average
        )
        
        if time_data is None or signal_data is None:
            print(f"Could not extract time series data from {result.path}")
            return fig, ax
        
        # Ensure 1D signal
        if signal_data.ndim > 1:
            signal_data = signal_data.flatten()
        
        # Compute spectrogram
        window_length = window_length or min(len(signal_data) // 8, 1024)
        overlap_samples = int(window_length * overlap)
        
        frequencies, times, Sxx = scipy.signal.spectrogram(
            signal_data,
            fs=metadata['sampling_rate'],
            window=self.fft_analyzer.config.window_function,
            nperseg=window_length,
            noverlap=overlap_samples,
            scaling='density'
        )
        
        # Convert to dB scale
        Sxx_db = 10 * np.log10(Sxx + 1e-12)  # Add small value to avoid log(0)
        
        # Plot spectrogram
        im = ax.pcolormesh(times, frequencies, Sxx_db, shading='gouraud', cmap='viridis')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power Spectral Density (dB)', fontsize=self.config.label_fontsize)
        
        # Customize plot
        ax.set_xlabel('Time (s)', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Frequency (Hz)', fontsize=self.config.label_fontsize)
        
        title_parts = [f"Spectrogram - {dataset_name}"]
        if comp is not None:
            title_parts.append(f"component {comp}")
        ax.set_title(" - ".join(title_parts), fontsize=self.config.title_fontsize)
        
        ax.tick_params(labelsize=self.config.tick_fontsize)
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        return fig, ax
    
    def peak_analysis(self, dataset_name: str = "m_z11",
                     comp: Optional[Union[str, int]] = None,
                     average: Optional[Tuple[Any, ...]] = None,
                     prominence: float = 0.1,
                     distance: int = 10,
                     figsize: Optional[Tuple[float, float]] = None,
                     save_path: Optional[str] = None,
                     **kwargs) -> Tuple[Any, Any]:
        """
        Plot peak analysis for power spectra.
        
        Parameters:
        -----------
        dataset_name : str, optional
            Dataset name (default: "m_z11")
        comp : Union[str, int], optional
            Component to analyze
        average : tuple, optional
            Axes to average over
        prominence : float, optional
            Minimum peak prominence (default: 0.1)
        distance : int, optional
            Minimum distance between peaks (default: 10)
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
        
        try:
            import scipy.signal
        except ImportError:
            raise ImportError("Scipy required for peak analysis")
        
        # Setup figure
        figsize = figsize or self.config.figsize
        fig, ax = plt.subplots(figsize=figsize, dpi=self.config.dpi)
        
        # Analyze all results
        for i in range(len(self.results)):
            try:
                fft_result = self.fft_analyzer.analyze_single(
                    i, dataset_name, comp=comp, average=average, **kwargs
                )
                
                # Find peaks
                peaks, properties = scipy.signal.find_peaks(
                    fft_result.power_spectrum,
                    prominence=prominence * np.max(fft_result.power_spectrum),
                    distance=distance
                )
                
                # Create label
                result = self.results[i]
                label = self._format_result_label(result)
                
                # Plot power spectrum
                ax.plot(fft_result.frequencies, fft_result.power_spectrum,
                       alpha=self.config.line_alpha,
                       linewidth=self.config.line_width,
                       label=label)
                
                # Mark peaks
                if len(peaks) > 0:
                    peak_freqs = fft_result.frequencies[peaks]
                    peak_powers = fft_result.power_spectrum[peaks]
                    ax.scatter(peak_freqs, peak_powers, color='red', s=50, zorder=5)
                    
                    # Annotate significant peaks
                    for j, (freq, power) in enumerate(zip(peak_freqs, peak_powers)):
                        if j < 3:  # Annotate only first 3 peaks
                            ax.annotate(f'{freq:.2e} Hz', 
                                       xy=(freq, power),
                                       xytext=(10, 10), 
                                       textcoords='offset points',
                                       fontsize=8,
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                
            except Exception as e:
                print(f"Error analyzing result {i}: {e}")
                continue
        
        # Customize plot
        ax.set_xlabel('Frequency (Hz)', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Power', fontsize=self.config.label_fontsize)
        
        title_parts = [f"Peak Analysis - {dataset_name}"]
        if comp is not None:
            title_parts.append(f"component {comp}")
        ax.set_title(" - ".join(title_parts), fontsize=self.config.title_fontsize)
        
        if self.config.grid:
            ax.grid(True, alpha=0.3)
        
        if self.config.legend and len(self.results) > 1:
            ax.legend(fontsize=self.config.label_fontsize)
        
        ax.tick_params(labelsize=self.config.tick_fontsize)
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        return fig, ax
    
    def __repr__(self) -> str:
        return f"FFTPlotter({len(self.results)} results)"
