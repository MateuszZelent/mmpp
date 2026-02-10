"""Helper objects for spectrum-facing fluent API."""

from __future__ import annotations

from io import StringIO


class SpectrumHelper:
    """Callable wrapper exposed as ``FFT.spectrum`` with rich method docs."""

    def __init__(self, fft_instance):
        self._fft = fft_instance
        self._spectrum_method = fft_instance._spectrum_impl

    def __call__(self, *args, **kwargs):
        """Delegate to ``FFT._spectrum_impl``."""
        return self._spectrum_method(*args, **kwargs)

    def __repr__(self):
        return self._rich_display()

    def _repr_html_(self):
        """Defer to text representation in Jupyter for now."""
        return None

    def _rich_display(self) -> str:
        """Generate rich help panel for ``FFT.spectrum`` usage."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.syntax import Syntax
            from rich.table import Table
            from rich.text import Text

            capture = StringIO()
            console = Console(file=capture, force_terminal=True, width=100)

            title = Text()
            title.append("📊 FFT Spectrum Analysis\n", style="bold blue")
            title.append(f"Path: {self._fft.job_result.path}", style="dim")
            console.print(Panel(title, border_style="blue"))

            params_table = Table(show_header=True, header_style="bold green")
            params_table.add_column("Parameter", style="yellow")
            params_table.add_column("Description", style="white")
            params_table.add_column("Default", style="cyan")

            params = [
                ("dset", "Dataset name", "'m'"),
                ("z_layer", "Z-layer index", "-1"),
                ("tmin/tmax", "Time range (indices)", "None"),
                ("fmin/fmax", "Frequency filter (Hz)", "None"),
                ("find_peaks", "Peak detection config", "None"),
                ("force", "Force recalculation", "False"),
                ("save", "Save to zarr", "False"),
            ]
            for param_name, desc, default in params:
                params_table.add_row(param_name, desc, default)

            console.print(params_table)
            console.print("")

            example_code = """# Basic spectrum
result = job[0].fft.spectrum()
freqs, spec = result  # Tuple unpacking

# With time slicing using slice notation
result = job[0].m[:200,...,1].fft.spectrum()

# Or with tmin/tmax parameters
result = job[0].fft.spectrum(tmin=0, tmax=200)

# Fluent plotting API
job[0].fft.spectrum(find_peaks={'min_prominence': 0.1}).plot_spectrum(
    freq_unit="GHz",
    log_scale=True,
    dpi=150
)

# Access properties
result.power       # |FFT|²
result.magnitude   # |FFT|
result.frequencies
result.peaks_info  # If find_peaks was used"""

            syntax = Syntax(example_code, "python", theme="monokai", line_numbers=False)
            console.print(
                Panel(
                    syntax,
                    title="[bold magenta]Usage Examples[/bold magenta]",
                    border_style="magenta",
                )
            )
            return capture.getvalue()
        except ImportError:
            return (
                "FFT.spectrum(...) - Call with parameters to compute FFT spectrum. "
                "Use help(job[0].fft.spectrum) for details."
            )
