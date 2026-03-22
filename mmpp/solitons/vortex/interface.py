"""Main entry point for vortex analysis."""

from __future__ import annotations

from typing import Any

from .config import VortexConfig


class VortexInterface:
    """Vortex dynamics analysis namespace."""

    def __init__(
        self,
        job_result,
        dataset_name: str | None = None,
        mmpp_instance: Any | None = None,
        slice_info: Any | None = None,
        config: VortexConfig | None = None,
    ):
        self._job = job_result
        self._dataset = dataset_name
        self._mmpp = mmpp_instance
        self._slice_info = slice_info
        self._config = config or VortexConfig()

        self._topology = None
        self._core = None
        self._trajectory = None
        self._spectrum = None
        self._modes = None
        self._nonlinear = None
        self._events = None
        self._signals = None
        self._energy = None
        self._model = None
        self._bridge = None
        self._plot = None
        self._autofit = None

    @property
    def dataset_name(self) -> str | None:
        """Dataset name used for analysis."""
        if self._dataset is None:
            candidate = self._job.get_largest_m_dataset()
            try:
                self._job._ensure_zarr_loaded()
                if candidate in self._job._z:
                    self._dataset = candidate
            except Exception:
                self._dataset = candidate
        return self._dataset

    @property
    def config(self) -> VortexConfig:
        """Mutable vortex config."""
        return self._config

    @property
    def topology(self):
        """Topology analysis namespace."""
        if self._topology is None:
            from .numerical.topology import TopologyInterface

            self._topology = TopologyInterface(
                self._job,
                dataset_name=self._dataset,
                slice_info=self._slice_info,
                config=self._config,
            )
        return self._topology

    @property
    def core(self):
        """Core-tracking namespace."""
        if self._core is None:
            from .numerical.core import CoreInterface

            self._core = CoreInterface(
                self._job,
                dataset_name=self._dataset,
                slice_info=self._slice_info,
                config=self._config,
            )
        return self._core

    @property
    def trajectory(self):
        """Trajectory analysis namespace."""
        if self._trajectory is None:
            from .trajectory import TrajectoryInterface

            self._trajectory = TrajectoryInterface(
                self._job,
                dataset_name=self._dataset,
                slice_info=self._slice_info,
                config=self._config,
                core_interface=self.core,
            )
        return self._trajectory

    @property
    def spectrum(self):
        """Vortex-specific spectrum analysis namespace."""
        if self._spectrum is None:
            from .spectrum import VortexSpectrumInterface

            self._spectrum = VortexSpectrumInterface(
                self._job,
                dataset_name=self._dataset,
                slice_info=self._slice_info,
                config=self._config,
                core_interface=self.core,
            )
        return self._spectrum

    @property
    def modes(self):
        """Mode-classification namespace."""
        if self._modes is None:
            from .numerical.modes import VortexModesInterface

            self._modes = VortexModesInterface(
                self._job,
                dataset_name=self._dataset,
                slice_info=self._slice_info,
                config=self._config,
                core_interface=self.core,
                spectrum_interface=self.spectrum,
            )
        return self._modes

    @property
    def nonlinear(self):
        """Nonlinear dynamics namespace."""
        if self._nonlinear is None:
            from .numerical.nonlinear import NonlinearInterface

            self._nonlinear = NonlinearInterface(
                self._job,
                dataset_name=self._dataset,
                slice_info=self._slice_info,
                config=self._config,
                core_interface=self.core,
                trajectory_interface=self.trajectory,
                spectrum_interface=self.spectrum,
            )
        return self._nonlinear

    @property
    def events(self):
        """Event detection namespace."""
        if self._events is None:
            from .numerical.events import EventsInterface

            self._events = EventsInterface(
                self._job,
                dataset_name=self._dataset,
                slice_info=self._slice_info,
                config=self._config,
                core_interface=self.core,
                trajectory_interface=self.trajectory,
            )
        return self._events

    @property
    def signals(self):
        """Synthetic electrical-signal namespace (MR/TMR, voltage, PSD)."""
        if self._signals is None:
            from .numerical.signals import SignalsInterface

            self._signals = SignalsInterface(
                self._job,
                dataset_name=self._dataset,
                slice_info=self._slice_info,
                config=self._config,
                core_interface=self.core,
            )
        return self._signals

    @property
    def energy(self):
        """Energy-analysis namespace sourced from table channels."""
        if self._energy is None:
            from .numerical.energy import EnergyInterface

            self._energy = EnergyInterface(
                self._job,
                dataset_name=self._dataset,
                slice_info=self._slice_info,
                config=self._config,
                core_interface=self.core,
            )
        return self._energy

    @property
    def model(self):
        """Analytical-model namespace (Thiele and future adapters)."""
        if self._model is None:
            from .model import VortexModelInterface

            self._model = VortexModelInterface(
                self._job,
                dataset_name=self._dataset,
                slice_info=self._slice_info,
            )
        return self._model

    @property
    def bridge(self):
        """Numerical <-> analytical bridge namespace."""
        if self._bridge is None:
            from .bridge import BridgeInterface

            self._bridge = BridgeInterface(
                vortex_interface=self,
                job_result=self._job,
                dataset_name=self._dataset,
                slice_info=self._slice_info,
            )
        return self._bridge

    @property
    def autofit(self):
        """Physics-informed autofit namespace."""
        if self._autofit is None:
            from .autofit import AutofitInterface

            self._autofit = AutofitInterface(self)
        return self._autofit

    @property
    def plot(self):
        """High-level vortex plotting namespace."""
        if self._plot is None:
            from .plotting import VortexPlotInterface

            self._plot = VortexPlotInterface(self)
        return self._plot

    @property
    def plt(self):
        """Alias for :attr:`plot`."""
        return self.plot

    def track(self, method: str = "auto", **kwargs):
        """Shortcut alias for ``self.core.track``."""
        return self.core.track(method=method, **kwargs)

    def detect(self, **kwargs):
        """Shortcut alias for ``self.topology.detect``."""
        return self.topology.detect(**kwargs)

    def show_simulation_params(
        self,
        *,
        params: str | dict = "auto",
        model: str = "auto",
        current: str | float | None = None,
        param_keys: dict | None = None,
    ):
        """Show resolved simulation parameters as an interactive Jupyter display.

        Calls the same parameter resolution logic used by ``.autofit.thiele()``,
        so you can verify that all physical constants are read correctly before
        running a fit.

        Parameters
        ----------
        params : str or dict
            ``"auto"`` to resolve from job metadata, or explicit dict override.
        model : str
            Model type: ``"auto"``, ``"cpp"``, ``"cip"``.
        current : str, float, or None
            Current specification (attribute key or value in A/m²).
        param_keys : dict, optional
            Custom alias mapping for parameter names.

        Returns
        -------
        IPython.display.HTML
            Rich interactive display of all resolved parameters.
        """
        from .bridge.extract import extract_model_defaults

        try:
            resolution = extract_model_defaults(
                vortex_interface=self,
                params=params,
                model=model,
                current=current,
                param_keys=param_keys,
            )
            return self._render_params_html(resolution)
        except Exception as e:
            return self._render_params_error_html(e)

    def _render_params_html(self, resolution):
        """Render resolved parameters as an interactive HTML card."""
        from html import escape as _esc

        try:
            from IPython.display import HTML
        except ImportError:
            HTML = None

        resolved = resolution.resolved_params
        sources = resolution.param_sources
        model_kind = resolution.model_kind

        card = (
            "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:18px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);"
        )
        section = (
            "background:rgba(15,23,42,0.6);padding:12px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);"
        )
        th_style = "padding:5px 8px;text-align:left;color:#e2e8f0;font-weight:600;"
        td_mono = "padding:4px 8px;font-family:monospace;color:#93c5fd;font-size:0.85em;"
        td_val = "padding:4px 8px;color:#e2e8f0;font-size:0.85em;font-family:monospace;"
        td_src = "padding:4px 8px;color:#94a3b8;font-size:0.8em;"
        badge = (
            "background:rgba(96,165,250,0.2);color:#93c5fd;font-size:0.7em;"
            "padding:2px 6px;border-radius:4px;font-weight:600;margin-left:8px;"
        )

        model_badge_color = "#22c55e" if model_kind == "cpp" else "#f59e0b"

        html = f"<div style='{card}'>"
        html += (
            "<div style='display:flex;align-items:center;gap:10px;margin-bottom:4px;'>"
            "<div style='font-size:1.15em;font-weight:600;color:#f1f5f9;'>"
            "🔬 Simulation Parameters</div>"
            f"<span style='{badge}background:rgba({('34,197,94' if model_kind == 'cpp' else '245,158,11')},0.2);"
            f"color:{model_badge_color};'>{_esc(model_kind.upper())} model</span>"
            "</div>"
            "<div style='font-size:0.85em;color:#94a3b8;margin-bottom:12px;'>"
            f"Resolved from: {_esc(' → '.join(resolution.search_locations))}</div>"
        )

        # Group parameters
        groups = [
            ("🧲 Material", ["Ms", "alpha", "P", "P_raw", "P_eff", "P_model", "A",
                             "Lambda", "epsilonprime"]),
            ("📐 Geometry", ["R", "L", "D", "Area", "L_stt", "dx", "dy", "dz",
                             "Nx", "Ny", "Nz"]),
            ("⚡ Current & STT", ["current_density", "current", "current_A", "current_mA",
                                  "current_dir", "polarizer", "p_z",
                                  "FixedLayerPosition", "fixed_layer_position",
                                  "slonczewski_current_sign"]),
            ("🧭 Field", ["field", "Bx_T", "By_T", "Bz_T",
                          "Bx_mT", "By_mT", "Bz_mT"]),
            ("🎯 Model / Vortex", ["omega0", "N", "chi_scale", "polarity",
                                   "domega0_dJ", "domega0_dJ_user", "domega0_dJ_stt",
                                   "phase_polarization", "d0_scale"]),
        ]

        for title, keys in groups:
            present = [(k, resolved[k], sources.get(k, "?")) for k in keys if k in resolved]
            if not present:
                continue

            html += f"<div style='{section}'>"
            html += (
                f"<div style='font-weight:600;color:#e2e8f0;margin-bottom:8px;"
                f"font-size:1.0em;'>{title}</div>"
            )
            html += (
                "<table style='width:100%;border-collapse:collapse;'>"
                f"<thead><tr style='background:rgba(51,65,85,0.6);'>"
                f"<th style='{th_style}'>Parameter</th>"
                f"<th style='{th_style}'>Value</th>"
                f"<th style='{th_style}'>Source</th></tr></thead><tbody>"
            )
            for name, value, source in present:
                val_str = self._format_param_value(name, value)
                # Color source
                if source.startswith("attrs:"):
                    src_color = "#22c55e"
                elif source.startswith("mx3:"):
                    src_color = "#60a5fa"
                elif source.startswith("computed:") or source.startswith("derived"):
                    src_color = "#f59e0b"
                elif source.startswith("default"):
                    src_color = "#64748b"
                elif source.startswith("override"):
                    src_color = "#c084fc"
                else:
                    src_color = "#94a3b8"
                html += (
                    f"<tr style='border-bottom:1px solid rgba(71,85,105,0.3);'>"
                    f"<td style='{td_mono}font-weight:600;'>{_esc(name)}</td>"
                    f"<td style='{td_val}'>{_esc(val_str)}</td>"
                    f"<td style='{td_src}color:{src_color};'>{_esc(source)}</td></tr>"
                )
            html += "</tbody></table></div>"

        # Catch any remaining params not in the groups above
        all_grouped = set()
        for _, keys in groups:
            all_grouped.update(keys)
        remaining = [(k, v) for k, v in resolved.items() if k not in all_grouped]
        if remaining:
            html += f"<div style='{section}'>"
            html += (
                "<div style='font-weight:600;color:#e2e8f0;margin-bottom:8px;'>"
                "📦 Other</div>"
            )
            html += "<table style='width:100%;border-collapse:collapse;'>"
            for name, value in remaining:
                val_str = self._format_param_value(name, value)
                source = sources.get(name, "?")
                html += (
                    f"<tr style='border-bottom:1px solid rgba(71,85,105,0.3);'>"
                    f"<td style='{td_mono}'>{_esc(name)}</td>"
                    f"<td style='{td_val}'>{_esc(val_str)}</td>"
                    f"<td style='{td_src}'>{_esc(source)}</td></tr>"
                )
            html += "</table></div>"

        html += "</div>"
        return HTML(html) if HTML else html

    def _render_params_error_html(self, error):
        """Render a parameter resolution error in a styled card."""
        from html import escape as _esc

        try:
            from IPython.display import HTML
        except ImportError:
            HTML = None

        html = (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #dc2626;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#1c1917 0%,#292524 50%,#44403c 100%);"
            "color:#fef2f2;box-shadow:0 8px 20px rgba(0,0,0,0.25);\">"
            "<div style='font-size:1.1em;font-weight:600;color:#fca5a5;margin-bottom:8px;'>"
            "❌ Parameter Resolution Failed</div>"
            f"<div style='font-size:0.9em;color:#fef2f2;margin-bottom:8px;'>"
            f"{_esc(type(error).__name__)}: {_esc(str(error))}</div>"
            "<div style='font-size:0.85em;color:#a8a29e;'>"
            "Try providing missing parameters explicitly:<br>"
            "<code style='color:#fca5a5;'>job[0].vortex.show_simulation_params("
            "params={'Ms': 8e5, 'R': 50e-9, ...})</code></div></div>"
        )
        return HTML(html) if HTML else html

    def _format_param_value(self, name: str, value) -> str:
        """Smart formatting for parameter values with SI-aware display."""
        import numpy as np

        if value is None:
            return "None"
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, str):
            return value
        if isinstance(value, (tuple, list)):
            parts = []
            for v in value:
                if isinstance(v, float):
                    parts.append(f"{v:.4g}")
                else:
                    parts.append(str(v))
            return f"({', '.join(parts)})"
        if callable(value):
            return "<callable>"
        if isinstance(value, (int, float, np.integer, np.floating)):
            fval = float(value)
            abs_val = abs(fval)
            if abs_val == 0:
                return "0"
            # SI formatting for common physics ranges
            if abs_val >= 1e12:
                return f"{fval:.4g} ({fval/1e12:.4g} T·unit)"
            if abs_val >= 1e9:
                return f"{fval:.4g} ({fval/1e9:.4g} G)"
            if abs_val >= 1e6:
                return f"{fval:.4g} ({fval/1e6:.4g} M)"
            if abs_val >= 1e3:
                return f"{fval:.4g} ({fval/1e3:.4g} k)"
            if abs_val >= 1:
                return f"{fval:.4g}"
            if abs_val >= 1e-3:
                return f"{fval:.4g} ({fval*1e3:.4g} m)"
            if abs_val >= 1e-6:
                return f"{fval:.4g} ({fval*1e6:.4g} µ)"
            if abs_val >= 1e-9:
                return f"{fval:.4g} ({fval*1e9:.4g} n)"
            if abs_val >= 1e-12:
                return f"{fval:.4g} ({fval*1e12:.4g} p)"
            return f"{fval:.4g}"
        return str(value)

    def __repr__(self) -> str:
        dataset_label = self._dataset if self._dataset is not None else "auto"
        return (
            f"VortexInterface(dataset={dataset_label!r}, "
            f"slice={self._slice_info!r})"
        )

    def _repr_html_(self) -> str:
        """Compact notebook card with available vortex submodules."""
        from html import escape as _esc

        dataset = _esc(str(self._dataset if self._dataset is not None else "auto"))
        slice_label = _esc(str(self._slice_info)) if self._slice_info is not None else "full"
        namespaces = [
            (".topology", "Topological charge & winding number detection"),
            (".core", "Vortex core position tracking (auto, table, Gaussian, CoM, ...)"),
            (".trajectory", "Core trajectory analysis & statistics"),
            (".spectrum", "Gyration frequency spectrum (FFT of core motion)"),
            (".modes", "Mode classification & identification"),
            (".nonlinear", "Nonlinear dynamics analysis"),
            (".events", "Event detection (switching, nucleation, ...)"),
            (".signals", "MR/TMR, voltage and signal spectra"),
            (".energy", "Energy channels from table (E_ex, E_demag, ...)"),
            (".model", "Analytical models (Thiele adapters)"),
            (".bridge", "Numerical ↔ analytical comparison/fit glue"),
            (".autofit", "Physics-informed autofit of analytical models"),
        ]
        ns_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(n)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(d)}</td></tr>"
            for n, d in namespaces
        )
        shortcuts = [
            (".track(method='gaussian', **kw)", "Shortcut → core.track()"),
            (".detect(**kw)", "Shortcut → topology.detect()"),
            (".show_simulation_params()", "Show resolved simulation parameters"),
        ]
        sc_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(m)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(d)}</td></tr>"
            for m, d in shortcuts
        )
        example = (
            "# Quick vortex core tracking\n"
            "vortex = job[0].vortex\n"
            "trajectory = vortex.track(method='gaussian')\n"
            "trajectory.plt.trajectory()\n"
            "\n"
            "# Topology analysis\n"
            "topo = vortex.topology.detect()\n"
            "\n"
            "# Gyration spectrum\n"
            "vortex.spectrum.compute()\n"
            "vortex.spectrum.plot()"
        )
        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);\">"
            "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;margin-bottom:4px;'>"
            "Vortex Dynamics Interface</div>"
            "<div style='font-size:0.85em;color:#94a3b8;margin-bottom:10px;'>"
            "Comprehensive vortex analysis namespace</div>"
            # Context
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='display:flex;flex-wrap:wrap;gap:12px;font-size:0.9em;'>"
            f"<div><span style='color:#94a3b8;'>Dataset:</span> "
            f"<code style='color:#cbd5e1;'>{dataset}</code></div>"
            f"<div><span style='color:#94a3b8;'>Slice:</span> "
            f"<code style='color:#cbd5e1;'>{slice_label}</code></div>"
            "</div></div>"
            # Namespaces
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Namespaces</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            "<thead><tr style='text-align:left;background:rgba(51,65,85,0.6);'>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Accessor</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Description</th></tr></thead>"
            f"<tbody>{ns_rows}</tbody></table></div>"
            # Shortcuts
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Shortcuts</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            f"{sc_rows}</table></div>"
            # Examples
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Examples</div>"
            "<pre style='margin:0;background:rgba(15,23,42,0.85);padding:10px;"
            "border-radius:6px;color:#e2e8f0;overflow-x:auto;font-size:0.85em;'>"
            f"<code>{example}</code></pre></div>"
            "</div>"
        )
