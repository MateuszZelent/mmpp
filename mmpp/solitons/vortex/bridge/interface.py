"""Bridge interface scaffolding for numerical <-> analytical workflows."""

from __future__ import annotations

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
        import uuid as _uuid

        from html import escape as _esc

        from mmpp._repr_helpers import (
            NODE_COLOR_ANALYSIS,
            NODE_COLOR_COMPUTE,
            accessors_section_html,
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        workflow_rows = [
            (
                "compare.with_(lhs, rhs)",
                "Overlay numerical and analytical trajectories, or compare two reduced-order outputs on the same axes/metrics path.",
            ),
            (
                "fit.thiele_from_trajectory(traj, ...)",
                "Fit effective Thiele-like parameters directly from a tracked trajectory result.",
            ),
            (
                "extract.model_defaults(...)",
                "Resolve analytical defaults from attrs, .mx3 metadata, or explicit overrides before building a reduced model.",
            ),
        ]
        workflow_body = "".join(
            "<tr>"
            f"<td style='padding:6px 8px;font-family:monospace;color:{NODE_COLOR_COMPUTE};vertical-align:top;'>{_esc(name)}</td>"
            f"<td style='padding:6px 8px;color:#f8f8f2;'>{_esc(desc)}</td>"
            "</tr>"
            for name, desc in workflow_rows
        )
        workflow_html = (
            "<div style='background:linear-gradient(135deg,rgba(68,71,90,0.55) 0%,rgba(40,42,54,0.55) 100%);"
            "padding:12px;border-radius:8px;margin-bottom:12px;border:1px solid rgba(98,114,164,0.35);'>"
            "<b style='color:#bd93f9;'>Bridge Workflows</b>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;margin-top:8px;'>"
            f"{workflow_body}</table></div>"
        )
        api = api_help_html(
            self,
            title="Vortex bridge API help",
            prefix="jobs[-1].solitons.vortex.bridge",
            properties=[
                ("compare", "Trajectory comparison accessor"),
                ("fit", "Analytical fitting accessor"),
                ("extract", "Model-default extraction accessor"),
            ],
            subtitle="Live public API for numerical-to-analytical bridge helpers.",
            chrome=False,
        )
        return node_card_html(
            "Vortex Bridge Interface",
            icon="🔗",
            subtitle="Glue layer between tracked numerical trajectories and reduced analytical vortex models.",
            sections=[
                metrics_section_html(
                    [
                        (
                            "dataset",
                            self._dataset_name or "auto-detect",
                            NODE_COLOR_COMPUTE,
                        ),
                        (
                            "slice",
                            "custom"
                            if self._slice_info is not None
                            else "full geometry",
                            None,
                        ),
                    ]
                ),
                accessors_section_html(
                    [
                        (
                            "Bridge:",
                            [
                                (".compare.with_(lhs, rhs)", NODE_COLOR_ANALYSIS),
                                (
                                    ".fit.thiele_from_trajectory(traj)",
                                    NODE_COLOR_COMPUTE,
                                ),
                                (".extract.model_defaults(...)", NODE_COLOR_COMPUTE),
                            ],
                        ),
                    ]
                ),
                workflow_html,
                examples_section_html(
                    "traj = jobs[-1].solitons.vortex.trajectory.raw\n"
                    "params = jobs[-1].solitons.vortex.bridge.extract.model_defaults()\n"
                    "fit = jobs[-1].solitons.vortex.bridge.fit.thiele_from_trajectory(traj)\n"
                    "jobs[-1].solitons.vortex.bridge.compare.with_(traj, fit)",
                    title="Bridge Workflows",
                ),
            ],
            api=api,
            uid=f"mmpp-vortex-bridge-{str(_uuid.uuid4())[:8]}",
        )


__all__ = ["BridgeInterface"]
