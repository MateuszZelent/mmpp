"""High-level trajectory analysis interface."""

from __future__ import annotations

from typing import Any

from ..config import VortexConfig
from .filtering import filter_trajectory
from .orbit import OrbitInterface
from .phase import PhaseAnalyzer
from .steady_state import extract_steady_state


class TrajectoryInterface:
    """Orbit and phase analysis namespace."""

    def __init__(
        self,
        job_result,
        dataset_name: str | None,
        slice_info: Any | None,
        config: VortexConfig,
        core_interface,
    ):
        self._job = job_result
        self._dataset_name = dataset_name
        self._slice_info = slice_info
        self._config = config
        self._core = core_interface
        self._orbit = None

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

        from mmpp._repr_helpers import api_help_html, html_tabs
        from html import escape as _esc

        methods = [
            (".raw", "Raw tracked trajectory (TrajectoryResult)"),
            (".filtered(method=..., window=...)", "Low-pass / smoothed trajectory"),
            (".steady_state(threshold=...)", "Extract steady-state portion"),
            (".orbit", "Orbit fitting namespace (ellipse, radius, etc.)"),
            (".phase", "Phase analysis namespace (instantaneous φ(t))"),
        ]
        method_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(m)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(d)}</td></tr>"
            for m, d in methods
        )
        example = (
            "# Get raw trajectory\n"
            "traj = vortex.trajectory.raw\n"
            "traj.plt.trajectory()  # x(t), y(t)\n"
            "traj.plt.orbit()       # x vs y\n"
            "\n"
            "# Filtered trajectory\n"
            "smooth = vortex.trajectory.filtered(method='savgol')\n"
            "\n"
            "# Steady-state extraction\n"
            "ss = vortex.trajectory.steady_state()\n"
            "\n"
            "# Orbit fitting\n"
            "orbit = vortex.trajectory.orbit\n"
            "orbit.fit()  # fit elliptical orbit\n"
            "\n"
            "# Phase analysis\n"
            "phase = vortex.trajectory.phase"
        )
        html = (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            'color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);">'
            "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;margin-bottom:4px;'>"
            "Trajectory Interface</div>"
            "<div style='font-size:0.85em;color:#94a3b8;margin-bottom:10px;'>"
            "Orbit, phase, and filtering tools for tracked core trajectory</div>"
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Methods &amp; Properties</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            f"{method_rows}</table></div>"
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
            title="Vortex trajectory API help",
            prefix="vortex.trajectory",
            properties=[
                ("raw", "Raw tracked trajectory"),
                ("orbit", "Orbit fitting namespace"),
                ("phase", "Phase analysis namespace"),
            ],
            methods=["filtered", "steady_state"],
            subtitle="Live signatures for trajectory filtering and steady-state extraction.",
            chrome=False,
        )
        return (
            f"<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:14px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            'color:#e2e8f0;">'
            + html_tabs(
                [("Overview", html), ("API", api_card)],
                uid=f"trajectory-{str(_uuid.uuid4())[:8]}",
            )
            + "</div>"
        )
