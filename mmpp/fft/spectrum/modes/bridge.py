"""Spectrum→modes bridge for FMR analysis."""

from __future__ import annotations

from typing import Any


class SpectrumModes:
    """FMR mode bridge accessible as ``spec.modes``."""

    def __init__(self, spectrum_result: Any):
        self._spectrum = spectrum_result
        self._interface = None

    def _resolve_interface(self):
        """Resolve existing FFT mode interface from source context."""
        if self._interface is not None:
            return self._interface

        interface = None
        source_fft = getattr(self._spectrum, "_source_fft", None)
        source_job = getattr(self._spectrum, "_source_job", None)

        if source_fft is not None:
            interface = source_fft.modes
        elif source_job is not None and hasattr(source_job, "fft"):
            interface = source_job.fft.modes

        if interface is None:
            raise RuntimeError(
                "Mode analysis requires source FFT/job context. "
                "Use job[0].fft.modes or compute spectrum from job[0].fft.spectrum()."
            )

        mode_ctx = getattr(self._spectrum, "_mode_context", {}) or {}
        dataset = mode_ctx.get("dset")
        slice_info = mode_ctx.get("slice_info")
        if dataset is not None:
            interface._dataset_context = dataset
        if slice_info is not None:
            interface._slice_context = slice_info

        self._interface = interface
        return self._interface

    def at(self, f: float, z_layer: int = -1):
        """Return mode at frequency ``f`` [GHz]."""
        return self._resolve_interface().mode(f=f, z_layer=z_layer)

    def at_peak(self, peak_index: int, z_layer: int = -1):
        """Return mode at detected peak index."""
        peaks = self.peak_frequencies
        idx = int(peak_index)
        if idx < 0 or idx >= len(peaks):
            raise IndexError(f"Peak index {idx} out of range for {len(peaks)} peaks")
        return self.at(f=float(peaks[idx]), z_layer=z_layer)

    def __call__(self, f: float, z_layer: int = -1):
        """Callable alias for :meth:`at`."""
        return self.at(f=f, z_layer=z_layer)

    def interactive(self, **kwargs):
        """Launch interactive spectrum + mode explorer."""
        return self._resolve_interface().interactive_spectrum(**kwargs)

    def interactive_spectrum(self, **kwargs):
        """Alias for interactive mode explorer."""
        return self.interactive(**kwargs)

    @property
    def plot(self):
        """Plotting namespace for mode visualizations."""
        from .accessor import SpectrumModesPlotAccessor

        return SpectrumModesPlotAccessor(self)

    @property
    def plt(self):
        """Deprecated alias for :attr:`plot`."""
        return self.plot

    @property
    def peak_frequencies(self) -> list[float]:
        """Peak frequencies converted to GHz (if peaks are available)."""
        peaks_info = getattr(self._spectrum, "peaks_info", None)
        if isinstance(peaks_info, dict):
            freqs = peaks_info.get("frequencies")
            if freqs is not None:
                return [float(f) * 1e-9 for f in freqs]

        peaks = getattr(self._spectrum, "peaks", None)
        if isinstance(peaks, list):
            out: list[float] = []
            for peak in peaks:
                if isinstance(peak, dict) and "frequency_ghz" in peak:
                    out.append(float(peak["frequency_ghz"]))
            return out

        return []

    def __repr__(self) -> str:
        peaks = self.peak_frequencies
        if peaks:
            preview = ", ".join(f"{f:.2f}" for f in peaks[:5])
            return f"<SpectrumModes: {len(peaks)} peaks at {preview} GHz>"
        return "<SpectrumModes: no detected peaks>"

    def _repr_html_(self) -> str:
        from html import escape as _esc

        peaks = self.peak_frequencies
        if peaks:
            peak_rows = "".join(
                f"<tr><td style='padding:2px 8px;color:#93c5fd;font-family:monospace;'>Peak {i}</td>"
                f"<td style='padding:2px 8px;color:#e2e8f0;'>{f:.2f} GHz</td></tr>"
                for i, f in enumerate(peaks[:8])
            )
            example_freq = f"{peaks[0]:.2f}"
        else:
            peak_rows = (
                "<tr><td style='color:#94a3b8;' colspan='2'>"
                "Run spec.plot.spectrum(show_peaks=True) first"
                "</td></tr>"
            )
            example_freq = "5.20"

        methods = [
            (".at(f=...)", "Mode profile at frequency [GHz]"),
            (".at_peak(i)", "Mode profile at detected peak index"),
            (".interactive()", "Full interactive mode explorer"),
            (".plot.imshow(f=...)", "Quick mode plot at frequency"),
            (".plot.animation(...)", "Animate mode profiles"),
        ]
        method_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(m)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(d)}</td></tr>"
            for m, d in methods
        )
        at_params = [
            ("f", "required", "Frequency in GHz"),
            ("component", "None", "Magnetization component ('x', 'y', 'z')"),
        ]
        interactive_params = [
            ("component", "'z'", "Default magnetization component to display"),
            ("cmap", "None", "Colormap for mode visualization"),
            ("dpi", "None", "Figure resolution override"),
            ("save_path", "None", "Path for saving snapshots"),
        ]
        def _param_rows(params):
            return "".join(
                f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(n)}</td>"
                f"<td style='padding:4px 8px;color:#a5b4fc;'>{_esc(d)}</td>"
                f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(desc)}</td></tr>"
                for n, d, desc in params
            )
        example = (
            f"# Get mode at frequency\n"
            f"mode = spec.modes.at(f={example_freq})\n"
            f"mode.plot.imshow(component='z')\n"
            f"\n"
            f"# Get mode at detected peak\n"
            f"mode = spec.modes.at_peak(0)\n"
            f"\n"
            f"# Interactive explorer\n"
            f"spec.modes.interactive()"
        )
        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);\">"
            "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;margin-bottom:4px;'>"
            "FMR Mode Analysis</div>"
            "<div style='font-size:0.85em;color:#94a3b8;margin-bottom:10px;'>"
            "Spatial eigenmode profiles m(x,y) at resonance frequencies</div>"
            # Peaks
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Detected Peaks</div>"
            "<table style='border-collapse:collapse;font-size:0.9em;'>"
            f"{peak_rows}</table></div>"
            # Methods
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Methods</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            f"{method_rows}</table></div>"
            # at() params
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>"
            "Parameters <span style='color:#94a3b8;font-weight:400;'>(.at)</span></div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            "<thead><tr style='text-align:left;background:rgba(51,65,85,0.6);'>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Arg</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Default</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Description</th></tr></thead>"
            f"<tbody>{_param_rows(at_params)}</tbody></table></div>"
            # interactive() params
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>"
            "Parameters <span style='color:#94a3b8;font-weight:400;'>(.interactive)</span></div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            "<thead><tr style='text-align:left;background:rgba(51,65,85,0.6);'>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Arg</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Default</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Description</th></tr></thead>"
            f"<tbody>{_param_rows(interactive_params)}</tbody></table></div>"
            # Examples
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Examples</div>"
            "<pre style='margin:0;background:rgba(15,23,42,0.85);padding:10px;"
            "border-radius:6px;color:#e2e8f0;overflow-x:auto;font-size:0.85em;'>"
            f"<code>{example}</code></pre></div>"
            "</div>"
        )


__all__ = ["SpectrumModes"]
