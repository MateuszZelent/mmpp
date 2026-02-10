"""Plotting accessor for :class:`~mmpp.fft.spectrum.modes.SpectrumModes`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .bridge import SpectrumModes


class SpectrumModesPlotAccessor:
    """Plot helpers for :class:`SpectrumModes`."""

    def __init__(self, modes: SpectrumModes):
        self._modes = modes

    def imshow(self, f: float, component: str = "z", **kwargs):
        """Plot single mode at frequency ``f`` [GHz]."""
        mode = self._modes.at(f=f)
        return mode.plt.imshow(component=component, **kwargs)

    def animation(
        self,
        frequencies: list[float] | None = None,
        peaks: list[int] | None = None,
        save_path: str | None = None,
        **kwargs,
    ) -> Any:
        """Create frequency-sweep animation via existing mode interface."""
        interface = self._modes._resolve_interface()
        freq_values = list(frequencies or [])

        if peaks:
            peak_freqs = self._modes.peak_frequencies
            for idx in peaks:
                if 0 <= int(idx) < len(peak_freqs):
                    freq_values.append(float(peak_freqs[int(idx)]))

        if freq_values:
            if len(freq_values) == 1:
                return interface.plot_modes(frequency=float(freq_values[0]), **kwargs)
            return interface.save_modes_animation(
                frequency_range=(float(min(freq_values)), float(max(freq_values))),
                animation_type="frequency",
                save_path=save_path,
                **kwargs,
            )

        return interface.save_modes_animation(save_path=save_path, **kwargs)

    def __repr__(self) -> str:
        return "<SpectrumModesPlotAccessor: .imshow(f=...), .animation(...)>"

    def _repr_html_(self) -> str:
        return """
        <div style="font-family:sans-serif;border:1px solid #475569;border-radius:8px;
                    padding:12px;background:#1e293b;color:#e2e8f0;">
          <b>SpectrumModesPlotAccessor</b>
          <table style="margin:6px 0;">
            <tr><td style="padding:3px 8px;font-family:monospace;color:#93c5fd;">.imshow(f=5.2)</td>
                <td style="padding:3px 8px;color:#cbd5e1;">Single mode map at frequency</td></tr>
            <tr><td style="padding:3px 8px;font-family:monospace;color:#93c5fd;">.animation(peaks=[0,1])</td>
                <td style="padding:3px 8px;color:#cbd5e1;">Frequency sweep animation</td></tr>
          </table>
        </div>
        """


__all__ = ["SpectrumModesPlotAccessor"]
