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

    @property
    def dataset_name(self) -> str:
        """Dataset name used for analysis."""
        if self._dataset is None:
            self._dataset = self._job.get_largest_m_dataset()
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
                dataset_name=self.dataset_name,
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
                dataset_name=self.dataset_name,
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
                dataset_name=self.dataset_name,
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
                dataset_name=self.dataset_name,
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
                dataset_name=self.dataset_name,
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
                dataset_name=self.dataset_name,
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
                dataset_name=self.dataset_name,
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
                dataset_name=self.dataset_name,
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
                dataset_name=self.dataset_name,
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
                dataset_name=self.dataset_name,
                slice_info=self._slice_info,
            )
        return self._model

    @property
    def bridge(self):
        """Numerical <-> analytical bridge namespace."""
        if self._bridge is None:
            from .bridge import BridgeInterface

            self._bridge = BridgeInterface()
        return self._bridge

    def track(self, method: str = "gaussian", **kwargs):
        """Shortcut alias for ``self.core.track``."""
        return self.core.track(method=method, **kwargs)

    def detect(self, **kwargs):
        """Shortcut alias for ``self.topology.detect``."""
        return self.topology.detect(**kwargs)

    def __repr__(self) -> str:
        return (
            f"VortexInterface(dataset={self.dataset_name!r}, "
            f"slice={self._slice_info!r})"
        )

    def _repr_html_(self) -> str:
        """Compact notebook card with available vortex submodules."""
        from html import escape as _esc

        dataset = _esc(str(self.dataset_name))
        slice_label = _esc(str(self._slice_info)) if self._slice_info is not None else "full"
        namespaces = [
            (".topology", "Topological charge & winding number detection"),
            (".core", "Vortex core position tracking (Gaussian, CoM, ...)"),
            (".trajectory", "Core trajectory analysis & statistics"),
            (".spectrum", "Gyration frequency spectrum (FFT of core motion)"),
            (".modes", "Mode classification & identification"),
            (".nonlinear", "Nonlinear dynamics analysis"),
            (".events", "Event detection (switching, nucleation, ...)"),
            (".signals", "MR/TMR, voltage and signal spectra"),
            (".energy", "Energy channels from table (E_ex, E_demag, ...)"),
            (".model", "Analytical models (Thiele adapters)"),
            (".bridge", "Numerical ↔ analytical comparison/fit glue"),
        ]
        ns_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(n)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(d)}</td></tr>"
            for n, d in namespaces
        )
        shortcuts = [
            (".track(method='gaussian', **kw)", "Shortcut → core.track()"),
            (".detect(**kw)", "Shortcut → topology.detect()"),
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
