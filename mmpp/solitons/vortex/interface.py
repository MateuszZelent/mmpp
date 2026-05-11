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
        self._health_cache: Any | None = None  # CoreHealthStatus, cached

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

    def check_health(
        self,
        *,
        trajectory=None,
        disk_radius: float | None = None,
        mz_annihilation_threshold: float = 0.05,
        boundary_fraction: float = 0.85,
        core_fraction: float = 0.25,
        force: bool = False,
    ):
        """Check for vortex core annihilation or boundary collision.

        Detects two common failure modes when the applied current is too large:

        * **Annihilation / re-magnetisation** — the vortex core is expelled to
          the disk edge and the sample transitions to a uniform magnetisation
          state.  Detected by comparing the sign and magnitude of the average
          ``m_z`` component between the first and last frame.
        * **Polarity reversal** — the core polarity flips under strong
          out-of-plane STT, changing the gyration handedness.
        * **Boundary collision** — the tracked core trajectory comes too close
          to the disk edge (> ``boundary_fraction`` of R).

        Parameters
        ----------
        trajectory : TrajectoryResult or None
            Pre-computed core trajectory (used for boundary-distance estimate).
            When ``None`` the core is tracked automatically.
        disk_radius : float or None
            Physical disk radius in metres.  Auto-inferred from job attributes.
        mz_annihilation_threshold : float
            ``|mz_final|`` below this value signals annihilation (default 0.05).
        boundary_fraction : float
            Core is flagged as near-boundary when it reaches more than this
            fraction of the disk radius from the centre (default 0.85).
        core_fraction : float
            Fraction of the grid radius used to average ``m_z`` (default 0.25).
        force : bool
            Re-run even if a cached result exists.

        Returns
        -------
        CoreHealthStatus
        """
        from .health import check_core_health

        if not force and self._health_cache is not None:
            return self._health_cache

        traj = trajectory
        if traj is None:
            try:
                traj = self.core.track()
            except Exception:
                pass

        status = check_core_health(
            self._job,
            dataset_name=self.dataset_name,
            trajectory=traj,
            disk_radius=disk_radius,
            mz_annihilation_threshold=mz_annihilation_threshold,
            boundary_fraction=boundary_fraction,
            core_fraction=core_fraction,
            slice_info=self._slice_info,
        )
        self._health_cache = status

        if not status.is_healthy:
            status.issue_python_warnings()

        return status

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
        td_mono = (
            "padding:4px 8px;font-family:monospace;color:#93c5fd;font-size:0.85em;"
        )
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
            (
                "🧲 Material",
                [
                    "Ms",
                    "alpha",
                    "P",
                    "P_raw",
                    "P_eff",
                    "P_model",
                    "A",
                    "Lambda",
                    "epsilonprime",
                ],
            ),
            (
                "📐 Geometry",
                ["R", "L", "D", "Area", "L_stt", "dx", "dy", "dz", "Nx", "Ny", "Nz"],
            ),
            (
                "⚡ Current & STT",
                [
                    "current_density",
                    "current",
                    "current_A",
                    "current_mA",
                    "current_dir",
                    "polarizer",
                    "p_z",
                    "FixedLayerPosition",
                    "fixed_layer_position",
                    "slonczewski_current_sign",
                ],
            ),
            ("🧭 Field", ["field", "Bx_T", "By_T", "Bz_T", "Bx_mT", "By_mT", "Bz_mT"]),
            (
                "🎯 Model / Vortex",
                [
                    "omega0",
                    "N",
                    "chi_scale",
                    "polarity",
                    "domega0_dJ",
                    "domega0_dJ_user",
                    "domega0_dJ_stt",
                    "phase_polarization",
                    "d0_scale",
                ],
            ),
        ]

        for title, keys in groups:
            present = [
                (k, resolved[k], sources.get(k, "?")) for k in keys if k in resolved
            ]
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
            'color:#fef2f2;box-shadow:0 8px 20px rgba(0,0,0,0.25);">'
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
                return f"{fval:.4g} ({fval / 1e12:.4g} T·unit)"
            if abs_val >= 1e9:
                return f"{fval:.4g} ({fval / 1e9:.4g} G)"
            if abs_val >= 1e6:
                return f"{fval:.4g} ({fval / 1e6:.4g} M)"
            if abs_val >= 1e3:
                return f"{fval:.4g} ({fval / 1e3:.4g} k)"
            if abs_val >= 1:
                return f"{fval:.4g}"
            if abs_val >= 1e-3:
                return f"{fval:.4g} ({fval * 1e3:.4g} m)"
            if abs_val >= 1e-6:
                return f"{fval:.4g} ({fval * 1e6:.4g} µ)"
            if abs_val >= 1e-9:
                return f"{fval:.4g} ({fval * 1e9:.4g} n)"
            if abs_val >= 1e-12:
                return f"{fval:.4g} ({fval * 1e12:.4g} p)"
            return f"{fval:.4g}"
        return str(value)

    def __repr__(self) -> str:
        dataset_label = self._dataset if self._dataset is not None else "auto"
        return f"VortexInterface(dataset={dataset_label!r}, slice={self._slice_info!r})"

    def _repr_html_(self) -> str:
        """Compact notebook card with available vortex submodules."""
        from html import escape as _esc

        from mmpp._repr_helpers import (
            NODE_COLOR_ADVANCED,
            NODE_COLOR_ANALYSIS,
            NODE_COLOR_COMPUTE,
            NODE_COLOR_PLOT,
            NODE_COLOR_UTIL,
            accessors_section_html,
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        dataset = _esc(str(self._dataset if self._dataset is not None else "auto"))
        slice_label = (
            _esc(str(self._slice_info)) if self._slice_info is not None else "full"
        )
        namespace_rows = "".join(
            [
                "<tr><td style='padding:5px 8px;font-family:monospace;color:#8be9fd;'>.topology</td><td style='padding:5px 8px;color:#f8f8f2;'>Topological charge, winding number and defect detection.</td></tr>",
                "<tr><td style='padding:5px 8px;font-family:monospace;color:#8be9fd;'>.core</td><td style='padding:5px 8px;color:#f8f8f2;'>Core tracking from magnetization or table data.</td></tr>",
                "<tr><td style='padding:5px 8px;font-family:monospace;color:#8be9fd;'>.trajectory</td><td style='padding:5px 8px;color:#f8f8f2;'>Trajectory fitting, orbit statistics and steady-state windows.</td></tr>",
                "<tr><td style='padding:5px 8px;font-family:monospace;color:#8be9fd;'>.spectrum</td><td style='padding:5px 8px;color:#f8f8f2;'>Gyration / breathing spectra and spectrogram workflows.</td></tr>",
                "<tr><td style='padding:5px 8px;font-family:monospace;color:#8be9fd;'>.modes</td><td style='padding:5px 8px;color:#f8f8f2;'>Mode classification, labeling and branch interpretation.</td></tr>",
                "<tr><td style='padding:5px 8px;font-family:monospace;color:#8be9fd;'>.nonlinear</td><td style='padding:5px 8px;color:#f8f8f2;'>Nonlinear coefficients, amplitude equations and Thiele-style reductions.</td></tr>",
                "<tr><td style='padding:5px 8px;font-family:monospace;color:#8be9fd;'>.events</td><td style='padding:5px 8px;color:#f8f8f2;'>Switching, expulsion, intermittency and regime-change detection.</td></tr>",
                "<tr><td style='padding:5px 8px;font-family:monospace;color:#8be9fd;'>.signals</td><td style='padding:5px 8px;color:#f8f8f2;'>MR/TMR, voltage traces and signal-domain power spectra.</td></tr>",
                "<tr><td style='padding:5px 8px;font-family:monospace;color:#8be9fd;'>.energy</td><td style='padding:5px 8px;color:#f8f8f2;'>Energy-channel analysis from table columns.</td></tr>",
                "<tr><td style='padding:5px 8px;font-family:monospace;color:#8be9fd;'>.model / .bridge / .autofit</td><td style='padding:5px 8px;color:#f8f8f2;'>Analytical models, numerical-analytical comparison and parameter autofit.</td></tr>",
                "<tr><td style='padding:5px 8px;font-family:monospace;color:#8be9fd;'>.plot / .plt</td><td style='padding:5px 8px;color:#f8f8f2;'>High-level plotting shortcuts for the vortex workflow.</td></tr>",
            ]
        )
        namespace_table = (
            "<div style='background:linear-gradient(135deg,rgba(68,71,90,0.55) 0%,rgba(40,42,54,0.55) 100%);"
            "padding:12px;border-radius:8px;margin-bottom:12px;border:1px solid rgba(98,114,164,0.35);'>"
            "<b style='color:#bd93f9;'>Namespace Catalog</b><br>"
            "<table style='width:100%;border-collapse:collapse;margin-top:6px;'>"
            "<thead><tr style='text-align:left;background:rgba(68,71,90,0.4);'>"
            "<th style='padding:4px 8px;color:#f8f8f2;'>Accessor</th>"
            "<th style='padding:4px 8px;color:#f8f8f2;'>What it gives you</th>"
            "</tr></thead>"
            f"<tbody>{namespace_rows}</tbody></table></div>"
        )
        entry_rows = "".join(
            [
                "<tr><td style='padding:5px 8px;font-family:monospace;color:#ffb86c;'>Need the core trajectory first</td><td style='padding:5px 8px;color:#f8f8f2;'><code>.track(...)</code> or <code>.core.track(...)</code></td></tr>",
                "<tr><td style='padding:5px 8px;font-family:monospace;color:#ffb86c;'>Need orbit metrics / steady-state window</td><td style='padding:5px 8px;color:#f8f8f2;'><code>.trajectory</code></td></tr>",
                "<tr><td style='padding:5px 8px;font-family:monospace;color:#ffb86c;'>Need frequency / PSD / spectrogram</td><td style='padding:5px 8px;color:#f8f8f2;'><code>.spectrum</code></td></tr>",
                "<tr><td style='padding:5px 8px;font-family:monospace;color:#ffb86c;'>Need mode identity or branch labeling</td><td style='padding:5px 8px;color:#f8f8f2;'><code>.modes</code></td></tr>",
                "<tr><td style='padding:5px 8px;font-family:monospace;color:#ffb86c;'>Need switching / expulsion diagnostics</td><td style='padding:5px 8px;color:#f8f8f2;'><code>.events</code> and <code>.check_health(...)</code></td></tr>",
                "<tr><td style='padding:5px 8px;font-family:monospace;color:#ffb86c;'>Need analytical fit or comparison</td><td style='padding:5px 8px;color:#f8f8f2;'><code>.model</code>, <code>.bridge</code>, <code>.autofit</code></td></tr>",
            ]
        )
        entry_table = (
            "<div style='background:linear-gradient(135deg,rgba(68,71,90,0.55) 0%,rgba(40,42,54,0.55) 100%);"
            "padding:12px;border-radius:8px;margin-bottom:12px;border:1px solid rgba(98,114,164,0.35);'>"
            "<b style='color:#bd93f9;'>Common Entrypoints</b><br>"
            "<table style='width:100%;border-collapse:collapse;margin-top:6px;'>"
            "<thead><tr style='text-align:left;background:rgba(68,71,90,0.4);'>"
            "<th style='padding:4px 8px;color:#f8f8f2;'>If you want to...</th>"
            "<th style='padding:4px 8px;color:#f8f8f2;'>Start here</th>"
            "</tr></thead>"
            f"<tbody>{entry_rows}</tbody></table></div>"
        )
        method_rows = "".join(
            [
                "<tr><td style='padding:5px 8px;font-family:monospace;color:#50fa7b;'>track(method='auto', **kwargs)</td><td style='padding:5px 8px;color:#f8f8f2;'>Top-level shortcut to <code>core.track()</code>. Typical methods: <code>'auto'</code>, <code>'gaussian'</code>, <code>'centroid'</code>, <code>'table'</code>.</td></tr>",
                "<tr><td style='padding:5px 8px;font-family:monospace;color:#50fa7b;'>detect(**kwargs)</td><td style='padding:5px 8px;color:#f8f8f2;'>Top-level shortcut to <code>topology.detect()</code> for charge / winding analysis.</td></tr>",
                "<tr><td style='padding:5px 8px;font-family:monospace;color:#50fa7b;'>check_health(...)</td><td style='padding:5px 8px;color:#f8f8f2;'>Inspect orbit radius, annihilation risk and boundary/core thresholds. Key args: <code>trajectory=</code>, <code>disk_radius=</code>, <code>force=</code>.</td></tr>",
                "<tr><td style='padding:5px 8px;font-family:monospace;color:#50fa7b;'>show_simulation_params(...)</td><td style='padding:5px 8px;color:#f8f8f2;'>Resolve and render simulation parameters. Key args: <code>params=</code>, <code>model=</code>, <code>current=</code>.</td></tr>",
                "<tr><td style='padding:5px 8px;font-family:monospace;color:#50fa7b;'>interactive(figsize=(10, 7), dpi=100)</td><td style='padding:5px 8px;color:#f8f8f2;'>Open the ipywidgets vortex dashboard that bundles tracking, topology, trajectory, spectrum and events.</td></tr>",
            ]
        )
        method_table = (
            "<div style='background:linear-gradient(135deg,rgba(68,71,90,0.55) 0%,rgba(40,42,54,0.55) 100%);"
            "padding:12px;border-radius:8px;margin-bottom:12px;border:1px solid rgba(98,114,164,0.35);'>"
            "<b style='color:#bd93f9;'>Top-level Methods</b><br>"
            "<table style='width:100%;border-collapse:collapse;margin-top:6px;'>"
            "<thead><tr style='text-align:left;background:rgba(68,71,90,0.4);'>"
            "<th style='padding:4px 8px;color:#f8f8f2;'>Call</th>"
            "<th style='padding:4px 8px;color:#f8f8f2;'>Use it when</th>"
            "</tr></thead>"
            f"<tbody>{method_rows}</tbody></table></div>"
        )
        workflow = examples_section_html(
            "\n".join(
                [
                    "# Workflow 1: trajectory-first analysis",
                    "vortex = job[-1].solitons.vortex",
                    "traj = vortex.track(method='gaussian', core_threshold=0.35)",
                    "traj.plt.trajectory()",
                    "vortex.events.detect(trajectory=traj)",
                    "",
                    "# Workflow 2: spectrum and mode interpretation",
                    "spec = vortex.spectrum.gyration(method='welch', nperseg=1024)",
                    "spec.plt.power_spectrum()",
                    "vortex.modes.classify()",
                    "",
                    "# Workflow 3: sanity / failure diagnostics",
                    "health = vortex.check_health(boundary_fraction=0.85, force=True)",
                    "vortex.show_simulation_params(model='thiele')",
                ]
            ),
            title="Recommended Workflows",
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
        api_card = api_help_html(
            self,
            title="Vortex API help",
            prefix="job[0].vortex",
            properties=[
                ("config", "Mutable vortex configuration"),
                ("topology", "Topological charge and winding number detection"),
                ("core", "Vortex core tracking namespace"),
                ("trajectory", "Trajectory analysis namespace"),
                ("spectrum", "Vortex-specific spectrum analysis namespace"),
                ("modes", "Mode classification namespace"),
                ("nonlinear", "Nonlinear dynamics namespace"),
                ("events", "Event detection namespace"),
                ("signals", "MR/TMR, voltage and signal spectra namespace"),
                ("energy", "Energy-analysis namespace"),
                ("model", "Analytical model namespace"),
                ("bridge", "Numerical-to-analytical comparison/fit namespace"),
                ("autofit", "Physics-informed autofit namespace"),
                ("plot", "High-level vortex plotting namespace"),
                ("plt", "Alias for plot"),
            ],
            methods=[
                "track",
                "detect",
                "check_health",
                "show_simulation_params",
                "interactive",
            ],
            subtitle="Live signatures for top-level vortex shortcuts and namespace map.",
            chrome=False,
        )
        return node_card_html(
            "Vortex Dynamics Interface",
            icon="🌀",
            subtitle="Comprehensive vortex analysis namespace.",
            sections=[
                metrics_section_html(
                    [
                        ("dataset", dataset, NODE_COLOR_COMPUTE),
                        ("slice", slice_label, NODE_COLOR_PLOT),
                    ]
                ),
                accessors_section_html(
                    [
                        (
                            "Core:",
                            [
                                (".core", NODE_COLOR_COMPUTE),
                                (".trajectory", NODE_COLOR_ANALYSIS),
                                (".spectrum", NODE_COLOR_PLOT),
                            ],
                        ),
                        (
                            "Physics:",
                            [
                                (".topology", NODE_COLOR_ANALYSIS),
                                (".modes", NODE_COLOR_ANALYSIS),
                                (".nonlinear", NODE_COLOR_ADVANCED),
                                (".events", NODE_COLOR_UTIL),
                                (".signals", NODE_COLOR_UTIL),
                                (".energy", NODE_COLOR_UTIL),
                            ],
                        ),
                        (
                            "Models:",
                            [
                                (".model", NODE_COLOR_PLOT),
                                (".bridge", NODE_COLOR_PLOT),
                                (".autofit", NODE_COLOR_ADVANCED),
                            ],
                        ),
                        (
                            "Shortcuts:",
                            [
                                (".track(method='gaussian', **kw)", NODE_COLOR_COMPUTE),
                                (".detect(**kw)", NODE_COLOR_ANALYSIS),
                                (".show_simulation_params()", NODE_COLOR_UTIL),
                            ],
                        ),
                    ]
                ),
                entry_table,
                namespace_table,
                method_table,
                workflow,
                examples_section_html(example),
            ],
            api=api_card,
            uid="vortex-interface",
        )

    def interactive(
        self,
        figsize=(10, 7),
        dpi=100,
        *,
        trajectory_source: str = "magnetization",
        center_mode: str = "auto",
    ):
        """Open the interactive vortex dynamics dashboard.

        The dashboard integrates all analysis modules (core tracking, topology,
        trajectory, spectrum, spectrogram, modes, events, signals, Thiele model)
        in a single ipywidgets-based UI.

        Parameters
        ----------
        figsize : tuple
            Figure size for plot panels (width, height in inches).
        dpi : int
            Plot resolution.
        trajectory_source : {"magnetization", "table", "compare"}
            Default source selected in the Trajectory tab. ``table`` uses the
            scalar table core position, which often starts at simulation time 0,
            while ``magnetization`` tracks saved magnetization frames.
        center_mode : {"auto", "orbit", "disk", "raw"}
            Default centering mode used by Core tracking and Trajectory plots.

        Returns
        -------
        VortexInteractiveDashboard
            Dashboard instance (already displayed).
        """
        from .ui import VortexInteractiveDashboard

        dashboard = VortexInteractiveDashboard(
            self,
            figsize=figsize,
            dpi=dpi,
            trajectory_source=trajectory_source,
            center_mode=center_mode,
        )
        dashboard.show()
        return dashboard
