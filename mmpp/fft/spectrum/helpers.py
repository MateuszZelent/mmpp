"""Helper objects for spectrum-facing fluent API."""

from __future__ import annotations

from html import escape as _html_escape


class _SpectrumQuickPlot:
    """Proxy returned by ``SpectrumHelper.plot``.

    Calling e.g. ``.spectrum(**kw)`` will first compute the FFT via
    ``SpectrumHelper()`` and then delegate to ``result.plot.spectrum(**kw)``.
    """

    def __init__(self, helper):
        self._helper = helper

    def spectrum(self, **compute_kw):
        """Compute spectrum and plot it immediately.

        All keyword arguments are forwarded to ``spectrum()`` except those
        recognised by the plot accessor (``log_scale``, ``freq_unit``,
        ``show_peaks``, ``dpi``, ``ax``), which are forwarded to
        ``result.plot.spectrum()``.
        """
        plot_keys = {"log_scale", "freq_unit", "show_peaks", "dpi", "ax",
                     "normalize", "figsize", "title", "save_path"}
        plot_kw = {k: compute_kw.pop(k) for k in list(compute_kw) if k in plot_keys}
        result = self._helper(**compute_kw)
        return result.plot.spectrum(**plot_kw)

    def interactive(self, **compute_kw):
        """Compute spectrum and open interactive viewer."""
        plot_keys = {"dpi", "figsize"}
        plot_kw = {k: compute_kw.pop(k) for k in list(compute_kw) if k in plot_keys}
        result = self._helper(**compute_kw)
        return result.plot.interactive(**plot_kw)

    def __repr__(self):
        return "<SpectrumQuickPlot: .spectrum(), .interactive()>"

    def _repr_html_(self) -> str:
        return (
            "<div style='font-family:sans-serif;border:1px solid #475569;border-radius:8px;"
            "padding:12px;background:#1e293b;color:#e2e8f0;'>"
            "<b>Spectrum → Plot</b> (compute + plot in one step)<br/>"
            "<code>data.fft.spectrum.plot.spectrum(log_scale=True)</code><br/>"
            "<code>data.fft.spectrum.plot.interactive(dpi=150)</code>"
            "</div>"
        )


class SpectrumHelper:
    """Callable wrapper exposed as ``FFT.spectrum`` with rich method docs."""

    def __init__(self, fft_instance):
        self._fft = fft_instance
        self._spectrum_method = fft_instance._spectrum_impl

    def __call__(self, *args, **kwargs):
        """Delegate to ``FFT._spectrum_impl``."""
        return self._spectrum_method(*args, **kwargs)

    @property
    def plot(self):
        """Quick-plot proxy: ``data.fft.spectrum.plot.spectrum(...)``."""
        return _SpectrumQuickPlot(self)

    def __repr__(self):
        job_path = getattr(self._fft.job_result, "path", "")
        return f"<SpectrumHelper: call spectrum() to compute | {job_path}>"

    def _repr_html_(self) -> str:
        """HTML card for ``data.fft.spectrum`` accessor."""
        try:
            return self._html_spectrum_display()
        except Exception:
            return ""

    def _html_spectrum_display(self) -> str:
        job_path = getattr(self._fft.job_result, "path", "")
        job_name = getattr(self._fft.job_result, "name", "unknown")

        # ── section builder ─────────────────────────────────────
        section_style = (
            "padding:4px 8px; font-weight:600; color:#f1f5f9; "
            "background:rgba(51,65,85,0.8); text-align:left;"
        )
        row_html = ""

        groups: list[tuple[str, list[tuple[str, str]]]] = [
            ("Parameters", [
                ("tmin / tmax", "Time range (indices)"),
                ("fmin / fmax", "Frequency range filter (Hz)"),
                ("find_peaks={'min_prominence': …}", "Peak detection"),
                ("dset", "Dataset name (default: 'm')"),
                ("z_layer", "Z-layer index (default: -1)"),
                ("save / force", "Save result to zarr / force recalc"),
            ]),
            ("SpectrumResult properties", [
                (".frequencies", "Frequency axis (Hz)"),
                (".power", "Power spectrum |FFT|²"),
                (".magnitude", "Magnitude spectrum |FFT|"),
                (".phase", "Phase spectrum (radians)"),
                (".peaks_info", "Detected peaks (if find_peaks used)"),
            ]),
            ("SpectrumResult actions", [
                (".plot.spectrum()", "Plot power spectrum"),
                (".modes", "Mode analysis from this spectrum"),
            ]),
        ]

        for group_name, items in groups:
            row_html += (
                f"<tr><td colspan='2' style='{section_style}'>"
                f"{_html_escape(group_name)}</td></tr>"
            )
            for name, desc in items:
                row_html += (
                    "<tr>"
                    f"<td style='padding:5px 8px 5px 16px; font-family:monospace; "
                    f"color:#93c5fd; white-space:nowrap;'>{_html_escape(name)}</td>"
                    f"<td style='padding:5px 8px; color:#cbd5e1;'>{_html_escape(desc)}</td>"
                    "</tr>"
                )

        # ── examples ────────────────────────────────────────────
        example_code = "\n".join([
            "# Compute spectrum",
            "result = data.fft.spectrum()",
            "",
            "# Plot power spectrum",
            "result.plot.spectrum(log_scale=True, freq_unit='GHz')",
            "",
            "# Tuple unpacking (backward compat)",
            "freqs, spec = result",
            "",
            "# Frequency range & peak detection",
            "result = data.fft.spectrum(",
            "    fmin=1e9, fmax=20e9,",
            "    find_peaks={'min_prominence': 0.1}",
            ")",
            "",
            "# Time slicing (two ways)",
            "result = data.fft.spectrum(tmin=0, tmax=200)",
            "result = job[0].m[:200, ...].fft.spectrum()",
            "",
            "# Fluent filter chain",
            "data.fft.filters(remove_static=True).spectrum()",
            "",
            "# Access result properties",
            "result.power          # |FFT|²",
            "result.magnitude      # |FFT|",
            "result.frequencies    # Hz",
            "result.peaks_info     # detected peaks",
        ])

        html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; border: 2px solid #334155; border-radius: 12px; padding: 16px; margin: 10px 0; background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%); color: #e2e8f0; box-shadow: 0 10px 22px rgba(0,0,0,0.28);">
          <div style="margin-bottom: 12px;">
            <div style="font-size: 1.1em; font-weight: 600; color: #f1f5f9;">FFT Spectrum — call <code style="color:#93c5fd;">spectrum()</code> to compute</div>
            <div style="color: #94a3b8; margin-top: 4px;">Job: {_html_escape(job_name)}</div>
            <div style="color: #94a3b8; margin-top: 2px;">Path: <code style="color:#cbd5e1;">{_html_escape(job_path)}</code></div>
          </div>

          <div style="background: rgba(15,23,42,0.6); padding: 10px; border-radius: 8px; margin-bottom: 12px; border: 1px solid rgba(148,163,184,0.2);">
            <table style="width:100%; border-collapse: collapse; font-size:0.9em;">
              <thead>
                <tr style="text-align:left; background: rgba(51,65,85,0.6);">
                  <th style="padding:6px 8px; color:#e2e8f0;">Name</th>
                  <th style="padding:6px 8px; color:#e2e8f0;">Description</th>
                </tr>
              </thead>
              <tbody>
                {row_html}
              </tbody>
            </table>
          </div>

          <div style="background: rgba(15,23,42,0.6); padding: 10px; border-radius: 8px; border: 1px solid rgba(148,163,184,0.2);">
            <div style="font-weight: 600; color: #e2e8f0; margin-bottom: 6px;">Examples</div>
            <pre style="margin:0; background: rgba(15,23,42,0.85); padding: 10px; border-radius: 6px; color:#e2e8f0; overflow-x:auto; font-size:0.85em;"><code>{_html_escape(example_code)}</code></pre>
          </div>
        </div>
        """
        return html

