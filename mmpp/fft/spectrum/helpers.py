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
        plot_keys = {
            "log_scale",
            "freq_unit",
            "show_peaks",
            "dpi",
            "ax",
            "normalize",
            "figsize",
            "title",
            "save_path",
            "label",
            "xlim",
            "component",
        }
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
        import uuid as _uuid

        from mmpp._repr_helpers import api_help_html, html_tabs

        # ── methods table ────────────────────────────────────────
        methods = [
            (".spectrum(**kw)", "Compute + plot power spectrum in one step"),
            (".interactive(**kw)", "Compute + open interactive explorer"),
        ]
        method_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{m}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{d}</td></tr>"
            for m, d in methods
        )

        # ── compute params (forwarded to spectrum()) ─────────────
        compute_params = [
            ("method", "1", "1: avg signal → FFT; 2: per-pixel FFT → avg |FFT|²"),
            ("dset", "'m'", "Dataset name"),
            ("z_layer", "-1", "Z-layer index"),
            ("tmin / tmax", "None", "Time range (indices)"),
            ("fmin / fmax", "None", "Frequency range filter (Hz)"),
            ("window", "'hann'", "Window function"),
            ("filter_type", "'remove_mean'", "Pre-FFT filter"),
            ("scaling", "'raw'", "raw, continuous_ft, amplitude, power, psd"),
            ("force", "False", "Force recalculation (bypass cache)"),
        ]
        compute_rows = "".join(
            f"<tr><td style='padding:3px 8px;font-family:monospace;color:#93c5fd;'>{n}</td>"
            f"<td style='padding:3px 8px;color:#a5b4fc;'>{d}</td>"
            f"<td style='padding:3px 8px;color:#cbd5e1;'>{desc}</td></tr>"
            for n, d, desc in compute_params
        )

        # ── plot params (forwarded to result.plot.spectrum()) ────
        plot_params = [
            ("log_scale", "True", "Logarithmic y-axis"),
            ("freq_unit", "'GHz'", "Frequency unit (Hz, kHz, MHz, GHz, THz)"),
            ("show_peaks", "True", "Mark detected peaks"),
            ("normalize", "False", "Normalize to max"),
            ("figsize", "(10, 5)", "Figure size"),
            ("dpi", "None", "Resolution override"),
            ("title", "None", "Custom title"),
        ]
        plot_rows = "".join(
            f"<tr><td style='padding:3px 8px;font-family:monospace;color:#93c5fd;'>{n}</td>"
            f"<td style='padding:3px 8px;color:#a5b4fc;'>{d}</td>"
            f"<td style='padding:3px 8px;color:#cbd5e1;'>{desc}</td></tr>"
            for n, d, desc in plot_params
        )

        examples = (
            "# Quick power spectrum plot\n"
            "data.fft.spectrum.plot.spectrum()\n"
            "\n"
            "# Compare methods on same axes\n"
            "import matplotlib.pyplot as plt\n"
            "fig, ax = plt.subplots()\n"
            "data.fft.spectrum.plot.spectrum(method=1, force=True, ax=ax, label='avg signal → FFT')\n"
            "data.fft.spectrum.plot.spectrum(method=2, force=True, ax=ax, label='per-pixel FFT → avg |FFT|²')\n"
            "ax.legend()\n"
            "\n"
            "# Interactive explorer\n"
            "data.fft.spectrum.plot.interactive()"
        )

        section_style = (
            "background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);"
        )
        table_head = (
            "<thead><tr style='text-align:left;background:rgba(51,65,85,0.6);'>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Arg</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Default</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Description</th></tr></thead>"
        )
        tbl = "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"

        html = (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            'color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);">'
            # Title
            "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;margin-bottom:8px;'>"
            "Spectrum → Quick Plot &amp; Interactive</div>"
            # Methods
            f"<div style='{section_style}'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Methods</div>"
            f"{tbl}{method_rows}</table></div>"
            # Compute params
            f"<div style='{section_style}'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:4px;'>"
            "Compute Parameters <span style='color:#94a3b8;font-weight:400;'>"
            "(forwarded to spectrum())</span></div>"
            f"{tbl}{table_head}<tbody>{compute_rows}</tbody></table></div>"
            # Plot params
            f"<div style='{section_style}'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:4px;'>"
            "Plot Parameters <span style='color:#94a3b8;font-weight:400;'>"
            "(forwarded to result.plot.spectrum())</span></div>"
            f"{tbl}{table_head}<tbody>{plot_rows}</tbody></table></div>"
            # Examples
            f"<div style='{section_style}'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Examples</div>"
            "<pre style='margin:0;background:rgba(15,23,42,0.85);padding:10px;"
            "border-radius:6px;color:#e2e8f0;overflow-x:auto;font-size:0.85em;'>"
            f"<code>{_html_escape(examples)}</code></pre></div>"
            "</div>"
        )
        api_card = api_help_html(
            self,
            title="Spectrum quick-plot API help",
            prefix="job[0].fft.spectrum.plot",
            methods=["spectrum", "interactive"],
            subtitle=(
                "Live signatures for quick one-step compute+plot methods. "
                "Compute arguments are forwarded to spectrum(); plotting arguments "
                "are split and forwarded to the result plot accessor."
            ),
            chrome=False,
        )
        return (
            f"<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:14px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            'color:#e2e8f0;">'
            + html_tabs(
                [("Overview", html), ("API", api_card)],
                uid=f"spectrum-quick-{str(_uuid.uuid4())[:8]}",
            )
            + "</div>"
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
        import uuid as _uuid

        from mmpp._repr_helpers import api_help_html, html_tabs

        job_path = getattr(self._fft.job_result, "path", "")
        job_name = getattr(self._fft.job_result, "name", "unknown")

        # ── section builder ─────────────────────────────────────
        section_style = (
            "padding:4px 8px; font-weight:600; color:#f1f5f9; "
            "background:rgba(51,65,85,0.8); text-align:left;"
        )
        row_html = ""

        groups: list[tuple[str, list[tuple[str, str]]]] = [
            (
                "Compute Parameters",
                [
                    (
                        "method",
                        "1: avg signal → FFT;  2: per-pixel FFT → avg |FFT|² (default: 1)",
                    ),
                    ("dset", "Dataset name (default: 'm')"),
                    ("z_layer", "Z-layer index (default: -1, last layer)"),
                    ("tmin / tmax", "Time range as indices"),
                    ("fmin / fmax", "Frequency range filter (Hz)"),
                    (
                        "window",
                        "Window function: hann, hamming, blackman, tukey, … (default: 'hann')",
                    ),
                    (
                        "filter_type",
                        "Pre-FFT filter: remove_mean, remove_static, detrend_linear, … (default: 'remove_mean')",
                    ),
                    (
                        "scaling",
                        "Spectrum scaling: raw, continuous_ft, amplitude, power, psd (default: 'raw')",
                    ),
                    (
                        "engine",
                        "FFT backend: numpy, scipy, pyfftw, auto (default: 'auto')",
                    ),
                    ("zero_padding", "Pad to next power of 2 (default: True)"),
                    ("nfft", "Explicit FFT length (default: None)"),
                    ("save / force", "Save result to zarr / force recalculation"),
                    ("find_peaks={'min_prominence': …}", "Peak detection"),
                ],
            ),
            (
                "SpectrumResult Properties",
                [
                    (".frequencies / .frequencies_ghz", "Frequency axis (Hz / GHz)"),
                    (".power", "Power spectrum |FFT|²"),
                    (".magnitude", "Magnitude |FFT|"),
                    (".phase", "Phase (radians)"),
                    (".complex", "Raw complex FFT data"),
                    (".peaks_info", "Detected peaks (if find_peaks used)"),
                    (".component_label", "Magnetization component label"),
                ],
            ),
            (
                "SpectrumResult Actions",
                [
                    (".plot.spectrum(**kw)", "Static matplotlib plot"),
                    (".plot.interactive(**kw)", "Jupyter interactive explorer"),
                    (".modes", "Mode analysis from this spectrum"),
                    (".filtered(**kw)", "Post-process: normalize, log, smooth, …"),
                ],
            ),
            (
                "Quick-Plot Shortcut",
                [
                    (
                        "data.fft.spectrum.plot",
                        "Returns plot proxy (call .spectrum() or .interactive())",
                    ),
                    (
                        "data.fft.spectrum.plot.spectrum(**kw)",
                        "Compute + plot in one step",
                    ),
                ],
            ),
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
        example_code = "\n".join(
            [
                "# Basic spectrum",
                "result = data.fft.spectrum()",
                "",
                "# Method comparison (per-pixel power avg vs avg-then-FFT)",
                "r1 = data.fft.spectrum(method=1, force=True)",
                "r2 = data.fft.spectrum(method=2, force=True)",
                "",
                "# Plot both on same axes",
                "import matplotlib.pyplot as plt",
                "fig, ax = plt.subplots()",
                "r1.plot.spectrum(ax=ax, log_scale=True, label='method 1 (avg signal → FFT)')",
                "r2.plot.spectrum(ax=ax, log_scale=True, label='method 2 (per-pixel FFT → avg |FFT|²)')",
                "ax.legend()",
                "",
                "# Quick one-liner plot",
                "data.fft.spectrum.plot.spectrum(log_scale=True, freq_unit='GHz')",
                "",
                "# With FFT options",
                "result = data.fft.spectrum(",
                "    method=1, window='blackman', scaling='amplitude',",
                "    filter_type='remove_static',",
                "    fmin=1e9, fmax=20e9,",
                "    find_peaks={'min_prominence': 0.1}",
                ")",
                "",
                "# Tuple unpacking (backward compat)",
                "freqs, spec = result",
                "",
                "# Time slicing (two equivalent ways)",
                "result = data.fft.spectrum(tmin=0, tmax=200)",
                "result = job[0].m[:200, ...].fft.spectrum()",
                "",
                "# Post-process result",
                "result.filtered(normalize=True, smooth=True)",
            ]
        )

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
        quick_plot_help = api_help_html(
            self.plot,
            title="FFT spectrum quick-plot API help",
            prefix="job[0].fft.spectrum.plot",
            methods=["spectrum", "interactive"],
            subtitle=(
                "Methods available under job[0].fft.spectrum.plot. "
                "They compute the spectrum and immediately render the selected view."
            ),
            chrome=False,
        )
        callable_help = api_help_html(
            self,
            title="FFT spectrum namespace API help",
            prefix="job[0].fft.spectrum",
            properties=[
                ("plot", "Quick plot proxy for compute+plot and interactive views")
            ],
            subtitle=(
                "The spectrum namespace is callable: use job[0].fft.spectrum(...) "
                "to compute a SpectrumResult. The manual card above lists the "
                "forwarded compute parameters."
            ),
            chrome=False,
        )
        return (
            f"<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:14px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            'color:#e2e8f0;">'
            + html_tabs(
                [
                    ("Overview", html),
                    ("Quick Plot API", quick_plot_help),
                    ("Namespace API", callable_help),
                ],
                uid=f"spectrum-{str(_uuid.uuid4())[:8]}",
            )
            + "</div>"
        )
