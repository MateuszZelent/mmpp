"""Bridge interface scaffolding for numerical <-> analytical workflows."""

from __future__ import annotations

import uuid

from mmpp._repr_helpers import api_help_html, html_tabs
from mmpp._shared.repr_html import make_simple_card

from .compare import compare_trajectories
from .extract import extract_model_defaults
from .fit import fit_thiele_from_trajectory


class _BridgeCompareAccessor:
    def with_(self, lhs, rhs, *, label=("numerical", "analytical")):
        return compare_trajectories(lhs, rhs, label=label)


class _BridgeFitAccessor:
    def thiele_from_trajectory(self, trajectory, **kwargs):
        return fit_thiele_from_trajectory(trajectory, **kwargs)


class _BridgeExtractAccessor:
    def __init__(self, bridge):
        self._bridge = bridge

    def model_defaults(self, **kwargs):
        return extract_model_defaults(
            vortex_interface=self._bridge._vortex_interface,
            job_result=self._bridge._job_result,
            dataset_name=self._bridge._dataset_name,
            slice_info=self._bridge._slice_info,
            **kwargs,
        )


class BridgeInterface:
    """Vortex bridge namespace with compare/fit sub-accessors."""

    def __init__(
        self,
        *,
        vortex_interface=None,
        job_result=None,
        dataset_name: str | None = None,
        slice_info=None,
    ):
        self._vortex_interface = vortex_interface
        self._job_result = job_result
        self._dataset_name = dataset_name
        self._slice_info = slice_info
        self.compare = _BridgeCompareAccessor()
        self.fit = _BridgeFitAccessor()
        self.extract = _BridgeExtractAccessor(self)

    def _repr_html_(self) -> str:
        rows = [
            (".compare.with_(lhs, rhs)", "Overlay/metric comparison of trajectories"),
            (
                ".fit.thiele_from_trajectory(traj)",
                "Fit Thiele-like proxy from trajectory",
            ),
            (
                ".extract.model_defaults(...)",
                "Resolve analytical parameters from attrs/.mx3/manual overrides",
            ),
        ]
        overview = make_simple_card(
            title="Vortex Bridge Interface",
            subtitle="Numerical <-> analytical glue utilities",
            rows=rows,
        )
        api = api_help_html(
            self,
            title="Vortex bridge API help",
            prefix="vortex.bridge",
            properties=[
                ("compare", "Trajectory comparison accessor"),
                ("fit", "Analytical fitting accessor"),
                ("extract", "Model-default extraction accessor"),
            ],
            subtitle="Live public API for numerical-to-analytical bridge helpers.",
            chrome=False,
        )
        return (
            '<div style=\'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;'
            "border:2px solid #334155;border-radius:12px;padding:14px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;'>"
            + html_tabs(
                [("Overview", overview), ("API", api)],
                uid=f"mmpp-vortex-bridge-{uuid.uuid4().hex}",
            )
            + "</div>"
        )


__all__ = ["BridgeInterface"]
