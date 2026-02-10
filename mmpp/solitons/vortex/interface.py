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
            from .topology import TopologyInterface

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
            from .core import CoreInterface

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
            from .modes import VortexModesInterface

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
            from .nonlinear import NonlinearInterface

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
            from .events import EventsInterface

            self._events = EventsInterface(
                self._job,
                dataset_name=self.dataset_name,
                slice_info=self._slice_info,
                config=self._config,
                core_interface=self.core,
                trajectory_interface=self.trajectory,
            )
        return self._events

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
        dataset = self.dataset_name
        slice_label = self._slice_info if self._slice_info is not None else "full"
        modules = [
            "topology",
            "core",
            "trajectory",
            "spectrum",
            "modes",
            "nonlinear",
            "events",
        ]
        module_list = "".join(f"<li><code>{item}</code></li>" for item in modules)
        return (
            "<div style='border:1px solid #d0d7de;padding:10px;border-radius:8px;'>"
            "<b>VortexInterface</b><br>"
            f"<span>dataset=<code>{dataset}</code>, slice=<code>{slice_label}</code></span>"
            "<br><span>Namespaces:</span><ul style='margin:6px 0 0 16px;'>"
            f"{module_list}</ul></div>"
        )
