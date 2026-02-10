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
    def plt(self):
        """Plotting namespace for mode visualizations."""
        from .accessor import SpectrumModesPlotAccessor

        return SpectrumModesPlotAccessor(self)

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
        peaks = self.peak_frequencies
        if peaks:
            rows = "".join(
                f"<tr><td style='padding:2px 8px;color:#93c5fd;font-family:monospace;'>Peak {i}</td>"
                f"<td style='padding:2px 8px;color:#e2e8f0;'>{f:.2f} GHz</td></tr>"
                for i, f in enumerate(peaks[:8])
            )
            example = f"{peaks[0]:.2f}"
        else:
            rows = (
                "<tr><td style='color:#94a3b8;' colspan='2'>"
                "Run spec.plt.spectrum(show_peaks=True) first"
                "</td></tr>"
            )
            example = "5.20"

        return f"""
        <div style="font-family:sans-serif;border:2px solid #334155;border-radius:12px;
                    padding:16px;background:linear-gradient(135deg,#0f172a,#1e293b,#334155);
                    color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);">
          <div style="font-size:1.1em;font-weight:600;margin-bottom:8px;">FMR Mode Analysis</div>
          <div style="font-size:0.9em;color:#94a3b8;margin-bottom:8px;">
            Spatial eigenmode profiles m(x,y) at resonance frequencies
          </div>
          <table style="border-collapse:collapse;margin:6px 0;">{rows}</table>
          <div style="margin-top:10px;font-weight:600;color:#94a3b8;font-size:0.85em;">Methods:</div>
          <table style="margin:4px 0;">
            <tr><td style="padding:3px 8px;font-family:monospace;color:#93c5fd;">.at(f=...)</td>
                <td style="padding:3px 8px;color:#cbd5e1;">Mode profile at frequency</td></tr>
            <tr><td style="padding:3px 8px;font-family:monospace;color:#93c5fd;">.at_peak(0)</td>
                <td style="padding:3px 8px;color:#cbd5e1;">Mode profile at detected peak</td></tr>
            <tr><td style="padding:3px 8px;font-family:monospace;color:#93c5fd;">.interactive()</td>
                <td style="padding:3px 8px;color:#cbd5e1;">Full interactive explorer</td></tr>
            <tr><td style="padding:3px 8px;font-family:monospace;color:#93c5fd;">.plt.animation(...)</td>
                <td style="padding:3px 8px;color:#cbd5e1;">Animate mode profiles</td></tr>
          </table>
          <div style="margin-top:8px;padding:8px;background:#1e293b;border-radius:6px;
                      font-family:monospace;font-size:0.85em;color:#a5b4fc;">
            mode = spec.modes.at(f={example})<br/>
            mode.plt.imshow(component=\"z\")
          </div>
        </div>
        """


__all__ = ["SpectrumModes"]
