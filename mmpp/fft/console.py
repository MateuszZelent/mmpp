"""
FFT Console Module

Rich console interface for beautiful FFT data presentation and analysis.
"""

from typing import Optional, Dict, List, Union, Any
import numpy as np

# Import dependencies with error handling
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    from rich.tree import Tree
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .main import FFTAnalyzer, FFTResult


class FFTConsole:
    """
    Rich console interface for FFT analysis results.
    
    Provides beautiful, interactive presentation of FFT analysis data
    using the rich library.
    """
    
    def __init__(self, results: Union[List[Any], Any], mmpp_instance: Optional[Any] = None):
        """
        Initialize FFT console.
        
        Parameters:
        -----------
        results : List or single result
            ZarrJobResult objects to display
        mmpp_instance : MMPP, optional
            Reference to parent MMPP instance
        """
        if not RICH_AVAILABLE:
            raise ImportError("Rich library required for console display. Install with: pip install rich")
        
        self.results = results if isinstance(results, list) else [results]
        self.mmpp = mmpp_instance
        self.console = Console()
        self.fft_analyzer = FFTAnalyzer(results, mmpp_instance)
        self._fft_cache = {}  # Cache for calculated FFT data
    
    def calculate_fft_data(self, dset: str = "m_z11", method: int = 1, z: int = -1, 
                          fft_engine: str = "numpy", save: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Calculate FFT data with specified parameters.
        
        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m_z11")
        method : int, optional
            Analysis method:
            1 - Per-cell FFT with averaging (default)
            0 - Global FFT analysis
        z : int, optional
            Layer selection (-1 for all layers, specific index for single layer)
        fft_engine : str, optional
            FFT engine to use ("numpy", "scipy", "fftw")
        save : bool, optional
            Save results to file (default: False)
        **kwargs : Any
            Additional FFT parameters
            
        Returns:
        --------
        Dict[str, Any]
            FFT analysis results
        """
        # Create cache key
        cache_key = f"{dset}_{method}_{z}_{fft_engine}_{hash(str(sorted(kwargs.items())))}"
        
        # Check cache first
        if cache_key in self._fft_cache:
            self.console.print(f"[dim]Using cached FFT data for {dset}[/dim]")
            fft_data = self._fft_cache[cache_key]
        else:
            # Display progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task(f"Calculating FFT for {dset}...", total=None)
                
                try:
                    # Update FFT analyzer configuration
                    if fft_engine != "numpy":
                        self.fft_analyzer.config.engine = fft_engine
                    
                    # Calculate FFT based on method
                    if method == 1:
                        fft_data = self._calculate_per_cell_fft(dset, z, **kwargs)
                    else:
                        fft_data = self._calculate_global_fft(dset, z, **kwargs)
                    
                    # Cache results
                    self._fft_cache[cache_key] = fft_data
                    
                    progress.update(task, description="✓ FFT calculation complete")
                    
                except Exception as e:
                    progress.update(task, description=f"✗ FFT calculation failed: {e}")
                    raise
        
        # Display summary
        self._display_fft_summary(fft_data, dset, method, z)
        
        # Save if requested
        if save:
            self._save_fft_data(fft_data, dset, method, z)
        
        return fft_data
    
    def _calculate_per_cell_fft(self, dset: str, z: int, **kwargs) -> Dict[str, Any]:
        """
        Calculate FFT for each cell individually, then average.
        
        Parameters:
        -----------
        dset : str
            Dataset name
        z : int
            Layer selection
        **kwargs : Any
            Additional FFT parameters
            
        Returns:
        --------
        Dict[str, Any]
            Averaged FFT results
        """
        all_fft_results = []
        cell_count = 0
        
        for result_idx, result in enumerate(self.results):
            try:
                # Get dataset
                if hasattr(result, 'get_dataset'):
                    dataset = result.get_dataset(dset)
                else:
                    # Fallback for different result types
                    dataset = getattr(result, dset, None)
                    if dataset is None:
                        continue
                
                # Handle layer selection
                if z == -1:
                    # All layers
                    data = np.array(dataset)
                else:
                    # Specific layer
                    if len(dataset.shape) > 2:
                        data = np.array(dataset[:, :, z])
                    else:
                        data = np.array(dataset)
                
                # Calculate FFT for each cell
                if len(data.shape) >= 2:
                    for i in range(data.shape[0]):
                        for j in range(data.shape[1]):
                            cell_data = data[i, j] if len(data.shape) == 2 else data[i, j, :]
                            
                            if len(cell_data) > 1:  # Skip empty cells
                                fft_result = self.fft_analyzer._calculate_fft_for_data(
                                    cell_data, f"{dset}_cell_{i}_{j}", **kwargs
                                )
                                if fft_result:
                                    all_fft_results.append(fft_result)
                                    cell_count += 1
                else:
                    # 1D data
                    fft_result = self.fft_analyzer._calculate_fft_for_data(
                        data, f"{dset}_result_{result_idx}", **kwargs
                    )
                    if fft_result:
                        all_fft_results.append(fft_result)
                        cell_count += 1
                        
            except Exception as e:
                self.console.print(f"[yellow]Warning: Skipping result {result_idx}: {e}[/yellow]")
                continue
        
        if not all_fft_results:
            raise ValueError(f"No valid FFT data found for dataset {dset}")
        
        # Average the FFT results
        return self._average_fft_results(all_fft_results, cell_count, dset, z)
    
    def _calculate_global_fft(self, dset: str, z: int, **kwargs) -> Dict[str, Any]:
        """
        Calculate global FFT analysis.
        
        Parameters:
        -----------
        dset : str
            Dataset name
        z : int
            Layer selection
        **kwargs : Any
            Additional FFT parameters
            
        Returns:
        --------
        Dict[str, Any]
            Global FFT results
        """
        # Use existing analyzer for global analysis
        fft_result = self.fft_analyzer.analyze_single(0, dset, **kwargs)
        
        return {
            'method': 0,
            'dataset': dset,
            'layer': z,
            'frequencies': fft_result.frequencies,
            'power_spectrum': fft_result.power_spectrum,
            'peak_frequency': fft_result.peak_frequency,
            'peak_power': fft_result.peak_power,
            'metadata': fft_result.metadata,
            'cell_count': 1
        }
    
    def _average_fft_results(self, fft_results: List[Any], cell_count: int, 
                           dset: str, z: int) -> Dict[str, Any]:
        """
        Average multiple FFT results.
        
        Parameters:
        -----------
        fft_results : List[Any]
            List of FFT results to average
        cell_count : int
            Number of cells processed
        dset : str
            Dataset name
        z : int
            Layer selection
            
        Returns:
        --------
        Dict[str, Any]
            Averaged FFT results
        """
        # Get common frequency grid
        min_length = min(len(result.frequencies) for result in fft_results)
        frequencies = fft_results[0].frequencies[:min_length]
        
        # Average power spectra
        power_spectra = []
        for result in fft_results:
            power_spectra.append(result.power_spectrum[:min_length])
        
        averaged_power = np.mean(power_spectra, axis=0)
        
        # Find peak in averaged spectrum
        peak_idx = np.argmax(averaged_power)
        peak_frequency = frequencies[peak_idx]
        peak_power = averaged_power[peak_idx]
        
        # Calculate statistics
        power_std = np.std(power_spectra, axis=0)
        
        return {
            'method': 1,
            'dataset': dset,
            'layer': z,
            'frequencies': frequencies,
            'power_spectrum': averaged_power,
            'power_spectrum_std': power_std,
            'peak_frequency': peak_frequency,
            'peak_power': peak_power,
            'cell_count': cell_count,
            'individual_results': fft_results,
            'metadata': {
                'n_cells': cell_count,
                'frequency_resolution': frequencies[1] - frequencies[0] if len(frequencies) > 1 else 0,
                'max_frequency': frequencies[-1] if len(frequencies) > 0 else 0,
                'fft_length': len(frequencies),
                'window_function': self.fft_analyzer.config.window_function,
                'engine': self.fft_analyzer.config.engine
            }
        }
    
    def _display_fft_summary(self, fft_data: Dict[str, Any], dset: str, method: int, z: int) -> None:
        """
        Display summary of FFT calculation results.
        
        Parameters:
        -----------
        fft_data : Dict[str, Any]
            FFT calculation results
        dset : str
            Dataset name
        method : int
            Analysis method used
        z : int
            Layer selection
        """
        self.console.print(f"\n[bold green]✓ FFT Analysis Complete[/bold green]")
        
        # Create summary table
        summary_table = Table(title=f"FFT Results - {dset}", show_header=False, border_style="green")
        summary_table.add_column("Parameter", style="cyan")
        summary_table.add_column("Value", style="white")
        
        method_name = "Per-cell averaging" if method == 1 else "Global analysis"
        layer_info = "All layers" if z == -1 else f"Layer {z}"
        
        summary_table.add_row("Method", method_name)
        summary_table.add_row("Dataset", dset)
        summary_table.add_row("Layer", layer_info)
        summary_table.add_row("Peak Frequency", f"{fft_data['peak_frequency']:.4e} Hz")
        summary_table.add_row("Peak Power", f"{fft_data['peak_power']:.4e}")
        
        if method == 1:
            summary_table.add_row("Cells Processed", str(fft_data['cell_count']))
        
        summary_table.add_row("Frequency Points", str(len(fft_data['frequencies'])))
        summary_table.add_row("Max Frequency", f"{fft_data['metadata']['max_frequency']:.2e} Hz")
        
        self.console.print(summary_table)
    
    def _save_fft_data(self, fft_data: Dict[str, Any], dset: str, method: int, z: int) -> None:
        """
        Save FFT data to file.
        
        Parameters:
        -----------
        fft_data : Dict[str, Any]
            FFT data to save
        dset : str
            Dataset name
        method : int
            Analysis method
        z : int
            Layer selection
        """
        import pickle
        import os
        from datetime import datetime
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        layer_str = "all" if z == -1 else f"z{z}"
        filename = f"fft_{dset}_method{method}_{layer_str}_{timestamp}.pkl"
        
        # Ensure output directory exists
        output_dir = "fft_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(fft_data, f)
            
            self.console.print(f"[green]✓ FFT data saved to: {filepath}[/green]")
            
            # Also save as text summary
            txt_filename = filepath.replace('.pkl', '_summary.txt')
            with open(txt_filename, 'w') as f:
                f.write(f"FFT Analysis Summary\n")
                f.write(f"===================\n\n")
                f.write(f"Dataset: {dset}\n")
                f.write(f"Method: {method} ({'Per-cell averaging' if method == 1 else 'Global analysis'})\n")
                f.write(f"Layer: {z} ({'All layers' if z == -1 else f'Layer {z}'})\n")
                f.write(f"Peak Frequency: {fft_data['peak_frequency']:.6e} Hz\n")
                f.write(f"Peak Power: {fft_data['peak_power']:.6e}\n")
                if method == 1:
                    f.write(f"Cells Processed: {fft_data['cell_count']}\n")
                f.write(f"Frequency Points: {len(fft_data['frequencies'])}\n")
                f.write(f"Analysis Engine: {fft_data['metadata']['engine']}\n")
            
            self.console.print(f"[dim]Summary saved to: {txt_filename}[/dim]")
            
        except Exception as e:
            self.console.print(f"[red]✗ Failed to save FFT data: {e}[/red]")

    def show_summary(self, dataset_name: str = "m_z11", **kwargs) -> None:
        """
        Display FFT analysis summary for all results.
        
        Parameters:
        -----------
        dataset_name : str, optional
            Dataset name (default: "m_z11")
        **kwargs : Any
            Additional FFT analysis parameters
        """
        self.console.print("\n[bold cyan]🔊 FFT Analysis Summary[/bold cyan]", justify="center")
        self.console.print("=" * 60, style="cyan")
        
        # Analyze all results
        fft_results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Analyzing results...", total=len(self.results))
            
            for i in range(len(self.results)):
                try:
                    fft_result = self.fft_analyzer.analyze_single(i, dataset_name, **kwargs)
                    fft_results.append(fft_result)
                except Exception as e:
                    self.console.print(f"[red]Error analyzing result {i}: {e}[/red]")
                finally:
                    progress.update(task, advance=1)
        
        if not fft_results:
            self.console.print("[red]No FFT results to display[/red]")
            return
        
        # Create summary table
        table = Table(title=f"FFT Analysis Results - {dataset_name}", show_header=True, header_style="bold magenta")
        table.add_column("Result", style="cyan", no_wrap=True)
        table.add_column("Peak Freq (Hz)", style="green")
        table.add_column("Peak Power", style="yellow")
        table.add_column("Bandwidth", style="blue")
        table.add_column("Duration (s)", style="white")
        table.add_column("Samples", style="dim")
        
        for i, fft_result in enumerate(fft_results):
            # Calculate bandwidth (frequency range with >10% of peak power)
            peak_power = fft_result.peak_power
            bandwidth_mask = fft_result.power_spectrum > 0.1 * peak_power
            if np.any(bandwidth_mask):
                freq_range = fft_result.frequencies[bandwidth_mask]
                bandwidth = freq_range[-1] - freq_range[0] if len(freq_range) > 1 else 0
            else:
                bandwidth = 0
            
            table.add_row(
                f"#{i+1}",
                f"{fft_result.peak_frequency:.2e}",
                f"{fft_result.peak_power:.2e}",
                f"{bandwidth:.2e}",
                f"{fft_result.metadata['duration']:.2e}",
                f"{fft_result.metadata['n_samples']}"
            )
        
        self.console.print(table)
        
        # Show configuration
        config_text = Text()
        config_text.append("🔧 Analysis Configuration:\n", style="bold yellow")
        config_text.append(f"  • Window: {self.fft_analyzer.config.window_function}\n", style="dim")
        config_text.append(f"  • Engine: {self.fft_analyzer.config.engine}\n", style="dim")
        config_text.append(f"  • Detrend: {self.fft_analyzer.config.detrend}\n", style="dim")
        
        self.console.print("\n")
        self.console.print(Panel.fit(config_text, title="Configuration", border_style="yellow"))
    
    def show_detailed_result(self, result_index: int = 0, dataset_name: str = "m_z11", **kwargs) -> None:
        """
        Display detailed analysis for a single result.
        
        Parameters:
        -----------
        result_index : int, optional
            Index of result to display (default: 0)
        dataset_name : str, optional
            Dataset name (default: "m_z11")
        **kwargs : Any
            Additional FFT analysis parameters
        """
        if result_index >= len(self.results):
            self.console.print(f"[red]Result index {result_index} out of range[/red]")
            return
        
        self.console.print(f"\n[bold cyan]📊 Detailed FFT Analysis - Result #{result_index + 1}[/bold cyan]", justify="center")
        self.console.print("=" * 70, style="cyan")
        
        # Analyze result
        try:
            fft_result = self.fft_analyzer.analyze_single(result_index, dataset_name, **kwargs)
        except Exception as e:
            self.console.print(f"[red]Error analyzing result: {e}[/red]")
            return
        
        # Create detailed panels
        # Basic info panel
        basic_info = Text()
        basic_info.append("📁 Dataset Information:\n", style="bold green")
        basic_info.append(f"  • Path: {fft_result.metadata['path']}\n", style="dim")
        basic_info.append(f"  • Dataset: {fft_result.metadata['dataset']}\n", style="dim")
        basic_info.append(f"  • Component: {fft_result.metadata['component']}\n", style="dim")
        basic_info.append(f"  • Duration: {fft_result.metadata['duration']:.2e} s\n", style="dim")
        basic_info.append(f"  • Samples: {fft_result.metadata['n_samples']}\n", style="dim")
        basic_info.append(f"  • Sampling Rate: {fft_result.metadata['sampling_rate']:.2e} Hz\n", style="dim")
        
        # FFT info panel
        fft_info = Text()
        fft_info.append("🔊 FFT Results:\n", style="bold blue")
        fft_info.append(f"  • Peak Frequency: {fft_result.peak_frequency:.4e} Hz\n", style="green")
        fft_info.append(f"  • Peak Power: {fft_result.peak_power:.4e}\n", style="green")
        fft_info.append(f"  • Frequency Resolution: {fft_result.metadata['frequency_resolution']:.2e} Hz\n", style="dim")
        fft_info.append(f"  • Max Frequency: {fft_result.metadata['max_frequency']:.2e} Hz\n", style="dim")
        fft_info.append(f"  • FFT Length: {fft_result.metadata['fft_length']}\n", style="dim")
        fft_info.append(f"  • Window: {fft_result.metadata['window_function']}\n", style="dim")
        
        # Statistics panel
        stats_info = Text()
        stats_info.append("📈 Statistics:\n", style="bold yellow")
        stats_info.append(f"  • Mean Power: {np.mean(fft_result.power_spectrum):.4e}\n", style="dim")
        stats_info.append(f"  • Std Power: {np.std(fft_result.power_spectrum):.4e}\n", style="dim")
        stats_info.append(f"  • Total Power: {np.sum(fft_result.power_spectrum):.4e}\n", style="dim")
        
        # Find top frequencies
        top_indices = np.argsort(fft_result.power_spectrum)[-5:][::-1]
        stats_info.append("  • Top 5 Frequencies:\n", style="cyan")
        for i, idx in enumerate(top_indices):
            freq = fft_result.frequencies[idx]
            power = fft_result.power_spectrum[idx]
            stats_info.append(f"    {i+1}. {freq:.3e} Hz ({power:.2e})\n", style="dim")
        
        # Display panels
        self.console.print(Columns([
            Panel.fit(basic_info, title="[bold green]Dataset Info[/bold green]", border_style="green"),
            Panel.fit(fft_info, title="[bold blue]FFT Results[/bold blue]", border_style="blue"),
            Panel.fit(stats_info, title="[bold yellow]Statistics[/bold yellow]", border_style="yellow")
        ]))
    
    def show_comparison(self, dataset_name: str = "m_z11", **kwargs) -> None:
        """
        Display comparison of FFT results across all datasets.
        
        Parameters:
        -----------
        dataset_name : str, optional
            Dataset name (default: "m_z11")
        **kwargs : Any
            Additional FFT analysis parameters
        """
        self.console.print("\n[bold cyan]🔍 FFT Results Comparison[/bold cyan]", justify="center")
        self.console.print("=" * 60, style="cyan")
        
        # Analyze all results
        fft_results = []
        for i in range(len(self.results)):
            try:
                fft_result = self.fft_analyzer.analyze_single(i, dataset_name, **kwargs)
                fft_results.append((i, fft_result))
            except Exception as e:
                self.console.print(f"[red]Error analyzing result {i}: {e}[/red]")
        
        if not fft_results:
            self.console.print("[red]No FFT results to compare[/red]")
            return
        
        # Create comparison tree
        tree = Tree("[bold cyan]FFT Comparison Tree[/bold cyan]")
        
        # Group by similar peak frequencies (within 10%)
        frequency_groups = {}
        for i, fft_result in fft_results:
            peak_freq = fft_result.peak_frequency
            
            # Find existing group
            group_found = False
            for group_freq in frequency_groups:
                if abs(peak_freq - group_freq) / group_freq < 0.1:  # Within 10%
                    frequency_groups[group_freq].append((i, fft_result))
                    group_found = True
                    break
            
            if not group_found:
                frequency_groups[peak_freq] = [(i, fft_result)]
        
        # Build tree
        for group_freq, group_results in frequency_groups.items():
            group_node = tree.add(f"[bold green]~{group_freq:.2e} Hz[/bold green] ({len(group_results)} results)")
            
            for i, fft_result in group_results:
                result_text = f"Result #{i+1}: {fft_result.peak_frequency:.3e} Hz, Power: {fft_result.peak_power:.2e}"
                group_node.add(f"[dim]{result_text}[/dim]")
        
        self.console.print(tree)
        
        # Statistics summary
        all_peak_freqs = [fft_result.peak_frequency for _, fft_result in fft_results]
        all_peak_powers = [fft_result.peak_power for _, fft_result in fft_results]
        
        stats_text = Text()
        stats_text.append("📊 Overall Statistics:\n", style="bold magenta")
        stats_text.append(f"  • Frequency Range: {np.min(all_peak_freqs):.2e} - {np.max(all_peak_freqs):.2e} Hz\n", style="dim")
        stats_text.append(f"  • Mean Peak Frequency: {np.mean(all_peak_freqs):.2e} Hz\n", style="dim")
        stats_text.append(f"  • Frequency Std: {np.std(all_peak_freqs):.2e} Hz\n", style="dim")
        stats_text.append(f"  • Power Range: {np.min(all_peak_powers):.2e} - {np.max(all_peak_powers):.2e}\n", style="dim")
        stats_text.append(f"  • Number of Groups: {len(frequency_groups)}\n", style="cyan")
        
        self.console.print("\n")
        self.console.print(Panel.fit(stats_text, title="[bold magenta]Summary Statistics[/bold magenta]", border_style="magenta"))
    
    def show_config(self) -> None:
        """Display current FFT configuration."""
        config = self.fft_analyzer.config
        
        config_table = Table(title="FFT Configuration", show_header=True, header_style="bold cyan")
        config_table.add_column("Parameter", style="yellow", no_wrap=True)
        config_table.add_column("Value", style="green")
        config_table.add_column("Description", style="dim")
        
        config_items = [
            ("window_function", config.window_function, "Window function for spectral analysis"),
            ("overlap", f"{config.overlap:.2f}", "Overlap for windowed analysis"),
            ("nfft", str(config.nfft), "FFT length (None for auto)"),
            ("detrend", config.detrend, "Detrending method"),
            ("scaling", config.scaling, "Scaling for PSD"),
            ("engine", config.engine, "FFT computation engine"),
            ("cache_results", str(config.cache_results), "Enable result caching"),
            ("zero_padding", str(config.zero_padding), "Apply zero padding"),
        ]
        
        if config.frequency_range:
            config_items.append(
                ("frequency_range", f"{config.frequency_range[0]:.1e} - {config.frequency_range[1]:.1e}", "Analysis frequency range")
            )
        
        for param, value, description in config_items:
            config_table.add_row(param, value, description)
        
        self.console.print("\n")
        self.console.print(config_table)
    
    def export_summary(self, filename: str, dataset_name: str = "m_z11", **kwargs) -> None:
        """
        Export FFT summary to file.
        
        Parameters:
        -----------
        filename : str
            Output filename
        dataset_name : str, optional
            Dataset name (default: "m_z11")
        **kwargs : Any
            Additional FFT analysis parameters
        """
        # Capture console output
        with self.console.capture() as capture:
            self.show_summary(dataset_name, **kwargs)
        
        # Write to file
        with open(filename, 'w') as f:
            f.write(capture.get())
        
        self.console.print(f"[green]✓ Summary exported to: {filename}[/green]")
    
    def __repr__(self) -> str:
        return f"FFTConsole({len(self.results)} results)"
