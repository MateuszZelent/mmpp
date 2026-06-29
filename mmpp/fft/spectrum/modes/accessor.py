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
        return mode.plot.imshow(component=component, **kwargs)

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
        import uuid as _uuid

        from mmpp._repr_helpers import api_help_html, html_tabs

        methods = [
            (
                ".imshow(f=..., component='z', **kw)",
                "Plot spatial mode profile m(x,y) at frequency",
            ),
            (
                ".animation(frequencies=..., peaks=..., save_path=...)",
                "Create frequency-sweep animation",
            ),
        ]
        method_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{m}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{d}</td></tr>"
            for m, d in methods
        )
        imshow_params = [
            ("f", "required", "Frequency in GHz"),
            ("component", "'z'", "Magnetization component ('x', 'y', 'z')"),
            ("**kwargs", "", "Forwarded to mode.plot.imshow()"),
        ]
        anim_params = [
            ("frequencies", "None", "List of frequencies [GHz] to sweep"),
            ("peaks", "None", "List of peak indices to animate"),
            ("save_path", "None", "Path to save animation file"),
            ("**kwargs", "", "Forwarded to save_modes_animation()"),
        ]

        def _param_rows(params):
            return "".join(
                f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{n}</td>"
                f"<td style='padding:4px 8px;color:#a5b4fc;'>{d}</td>"
                f"<td style='padding:4px 8px;color:#cbd5e1;'>{desc}</td></tr>"
                for n, d, desc in params
            )

        example = (
            "# Plot single mode at 5.2 GHz\n"
            "spec.modes.plot.imshow(f=5.2, component='z')\n"
            "\n"
            "# Animate detected peaks\n"
            "spec.modes.plot.animation(peaks=[0, 1, 2])\n"
            "\n"
            "# Animate custom frequency range\n"
            "spec.modes.plot.animation(\n"
            "    frequencies=[3.0, 5.0, 7.5],\n"
            "    save_path='modes_sweep.gif'\n"
            ")"
        )
        html = (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            'color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);">'
            "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;margin-bottom:4px;'>"
            "Spectrum Modes Plot Accessor</div>"
            "<div style='font-size:0.85em;color:#94a3b8;margin-bottom:10px;'>"
            "Plotting helpers via <code style='color:#a5b4fc;'>spec.modes.plot</code></div>"
            # Methods
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Methods</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            f"{method_rows}</table></div>"
            # imshow params
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>"
            "Parameters <span style='color:#94a3b8;font-weight:400;'>(imshow)</span></div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            "<thead><tr style='text-align:left;background:rgba(51,65,85,0.6);'>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Arg</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Default</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Description</th></tr></thead>"
            f"<tbody>{_param_rows(imshow_params)}</tbody></table></div>"
            # animation params
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>"
            "Parameters <span style='color:#94a3b8;font-weight:400;'>(animation)</span></div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            "<thead><tr style='text-align:left;background:rgba(51,65,85,0.6);'>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Arg</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Default</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Description</th></tr></thead>"
            f"<tbody>{_param_rows(anim_params)}</tbody></table></div>"
            # Examples
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Examples</div>"
            "<pre style='margin:0;background:rgba(15,23,42,0.85);padding:10px;"
            "border-radius:6px;color:#e2e8f0;overflow-x:auto;font-size:0.85em;'>"
            f"<code>{example}</code></pre></div>"
            "</div>"
        )
        api_card = api_help_html(
            self,
            title="Spectrum modes plot API help",
            prefix="spec.modes.plot",
            methods=["imshow", "animation"],
            subtitle="Live signatures for plots available from SpectrumModes.plot.",
            chrome=False,
        )
        return (
            f"<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:14px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            'color:#e2e8f0;">'
            + html_tabs(
                [("Overview", html), ("API", api_card)],
                uid=f"spectrum-modes-plot-{str(_uuid.uuid4())[:8]}",
            )
            + "</div>"
        )


__all__ = ["SpectrumModesPlotAccessor"]
