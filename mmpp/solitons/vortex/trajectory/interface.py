"""High-level trajectory analysis interface."""

from __future__ import annotations

from typing import Any

from ..._method_helpers import InteractiveNodeMixin
from ..config import VortexConfig
from .filtering import filter_trajectory
from .orbit import OrbitInterface
from .phase import PhaseAnalyzer
from .steady_state import extract_steady_state


class TrajectoryInterface(InteractiveNodeMixin):
    """Orbit and phase analysis namespace."""

    _interactive_owner = "job[0].vortex.trajectory"
    _interactive_nodes = frozenset({"filtered", "steady_state"})

    def __init__(
        self,
        job_result,
        dataset_name: str | None,
        slice_info: Any | None,
        config: VortexConfig,
        core_interface,
        vortex_interface=None,
    ):
        self._job = job_result
        self._dataset_name = dataset_name
        self._slice_info = slice_info
        self._config = config
        self._core = core_interface
        self._orbit: OrbitInterface | None = None

    @property
    def raw(self):
        """Raw tracked trajectory (computed lazily)."""
        return self._core.track()

    def filtered(self, method: str | None = None, **kwargs):
        """Return filtered trajectory."""
        selected_method = method or self._config.trajectory.filter_method
        if "window" not in kwargs:
            kwargs["window"] = self._config.trajectory.filter_window
        return filter_trajectory(self.raw, method=selected_method, **kwargs)

    def steady_state(self, threshold: float | None = None, **kwargs):
        """Return steady-state portion of the trajectory."""
        selected_threshold = (
            self._config.trajectory.steady_state_threshold
            if threshold is None
            else float(threshold)
        )
        if "window" not in kwargs:
            kwargs["window"] = self._config.trajectory.steady_state_window
        return extract_steady_state(self.raw, threshold=selected_threshold, **kwargs)

    @property
    def orbit(self) -> OrbitInterface:
        """Orbit fitting namespace."""
        if self._orbit is None:
            self._orbit = OrbitInterface(self)
        return self._orbit

    @property
    def phase(self) -> PhaseAnalyzer:
        """Phase analysis namespace."""
        return PhaseAnalyzer(self.raw)

    def _repr_html_(self) -> str:
        import uuid as _uuid
        from html import escape as _esc

        from mmpp._repr_helpers import (
            NODE_COLOR_ANALYSIS,
            NODE_COLOR_COMPUTE,
            NODE_COLOR_PLOT,
            accessors_section_html,
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        context_rows = [
            ("dataset", self._dataset_name or "auto-detect", None),
            (
                "slice",
                "custom" if self._slice_info is not None else "full geometry",
                None,
            ),
            (
                "filter method",
                self._config.trajectory.filter_method,
                NODE_COLOR_COMPUTE,
            ),
            (
                "steady threshold",
                self._config.trajectory.steady_state_threshold,
                NODE_COLOR_ANALYSIS,
            ),
            (
                "steady window",
                self._config.trajectory.steady_state_window,
                None,
            ),
        ]
        accessors = [
            (
                "Data:",
                [
                    (".raw", NODE_COLOR_COMPUTE),
                    (".filtered(method='savgol', ...)", NODE_COLOR_COMPUTE),
                    (".steady_state(threshold=..., ...)", NODE_COLOR_COMPUTE),
                ],
            ),
            (
                "Analysis:",
                [
                    (".orbit.fit(...)", NODE_COLOR_ANALYSIS),
                    (".phase.frequency_hz", NODE_COLOR_ANALYSIS),
                ],
            ),
            (
                "Plots:",
                [
                    (".raw.plt.trajectory()", NODE_COLOR_PLOT),
                    (".raw.plt.orbit()", NODE_COLOR_PLOT),
                    (".phase.plt.frequency_vs_time()", NODE_COLOR_PLOT),
                ],
            ),
        ]
        namespace_rows = [
            (
                "raw",
                "Tracked core trajectory result. This is the source object for most downstream orbit and phase diagnostics.",
            ),
            (
                "filtered(...)",
                "Smoothed / filtered trajectory using config defaults or explicit method/window overrides.",
            ),
            (
                "steady_state(...)",
                "Returns the portion of the trajectory after transient decay using amplitude thresholding.",
            ),
            (
                "orbit",
                "Orbit fitting namespace for ellipse geometry, radius extraction, and residual analysis.",
            ),
            (
                "phase",
                "Instantaneous phase and angular frequency diagnostics derived from the tracked orbit.",
            ),
        ]
        namespace_body = "".join(
            "<tr>"
            f"<td style='padding:6px 8px;font-family:monospace;color:{NODE_COLOR_COMPUTE};vertical-align:top;'>{_esc(name)}</td>"
            f"<td style='padding:6px 8px;color:#f8f8f2;'>{_esc(desc)}</td>"
            "</tr>"
            for name, desc in namespace_rows
        )
        example = (
            "# Get tracked core trajectory\n"
            "traj = jobs[-1].solitons.vortex.trajectory.raw\n"
            "traj.plt.trajectory()  # x(t), y(t)\n"
            "traj.plt.orbit()       # x vs y\n"
            "\n"
            "# Filtered trajectory\n"
            "smooth = jobs[-1].solitons.vortex.trajectory.filtered(method='savgol')\n"
            "\n"
            "# Steady-state extraction\n"
            "ss = jobs[-1].solitons.vortex.trajectory.steady_state()\n"
            "\n"
            "# Orbit fitting\n"
            "orbit = jobs[-1].solitons.vortex.trajectory.orbit\n"
            "fit = orbit.fit()\n"
            "\n"
            "# Phase analysis\n"
            "phase = jobs[-1].solitons.vortex.trajectory.phase\n"
            "phase.plt.frequency_vs_time(unit='ghz')"
        )
        namespace_html = (
            "<div style='background:linear-gradient(135deg,rgba(68,71,90,0.55) 0%,rgba(40,42,54,0.55) 100%);"
            "padding:12px;border-radius:8px;margin-bottom:12px;border:1px solid rgba(98,114,164,0.35);'>"
            "<b style='color:#bd93f9;'>Namespace Catalog</b>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;margin-top:8px;'>"
            f"{namespace_body}</table></div>"
        )
        api_card = api_help_html(
            self,
            title="Vortex trajectory API help",
            prefix="jobs[-1].solitons.vortex.trajectory",
            properties=[
                ("raw", "Raw tracked trajectory"),
                ("orbit", "Orbit fitting namespace"),
                ("phase", "Phase analysis namespace"),
            ],
            methods=["filtered", "steady_state"],
            subtitle="Live signatures for trajectory filtering and steady-state extraction.",
            chrome=False,
        )
        return node_card_html(
            "Vortex Trajectory Interface",
            icon="🧭",
            subtitle="Tracking-derived orbit, filtering, steady-state, and phase diagnostics for a single vortex run.",
            sections=[
                metrics_section_html(context_rows),
                accessors_section_html(accessors),
                namespace_html,
                examples_section_html(example, title="Trajectory Workflows"),
            ],
            api=api_card,
            uid=f"trajectory-{str(_uuid.uuid4())[:8]}",
        )
