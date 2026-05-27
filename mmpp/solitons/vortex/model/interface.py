"""Dataset-aware analytical-model interface."""

from __future__ import annotations

from typing import Any


class VortexModelInterface:
    """Entry-point for analytical models attached to a vortex context."""

    def __init__(
        self,
        job_result,
        *,
        dataset_name: str | None = None,
        slice_info: Any | None = None,
    ):
        self._job = job_result
        self._dataset_name = dataset_name
        self._slice_info = slice_info
        self._thiele = None

    @property
    def thiele(self):
        """Thiele-equation model namespace."""
        if self._thiele is None:
            from .thiele.interface import ThieleModelNamespace

            self._thiele = ThieleModelNamespace(
                job_result=self._job,
                dataset_name=self._dataset_name,
                slice_info=self._slice_info,
            )
        return self._thiele

    def _repr_html_(self) -> str:
        import uuid as _uuid

        from mmpp._repr_helpers import (
            NODE_COLOR_ANALYSIS,
            NODE_COLOR_COMPUTE,
            accessors_section_html,
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        api = api_help_html(
            self,
            title="Vortex model API help",
            prefix="jobs[-1].solitons.vortex.model",
            properties=[("thiele", "Thiele-equation model namespace")],
            subtitle="Live public API for dataset-aware analytical model namespaces.",
            chrome=False,
        )
        return node_card_html(
            "Vortex Model Interface",
            icon="📐",
            subtitle="Dataset-aware analytical model entrypoint for reduced vortex dynamics and parameter extraction.",
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
                            "Models:",
                            [
                                (".thiele", NODE_COLOR_ANALYSIS),
                                (".thiele.cpp(...)", NODE_COLOR_COMPUTE),
                                (".thiele.cip(...)", NODE_COLOR_COMPUTE),
                            ],
                        ),
                    ]
                ),
                examples_section_html(
                    "thiele = jobs[-1].solitons.vortex.model.thiele\n"
                    "thiele.cpp()\n"
                    "thiele.cip()",
                    title="Model Workflows",
                ),
            ],
            api=api,
            uid=f"mmpp-vortex-model-{str(_uuid.uuid4())[:8]}",
        )


__all__ = ["VortexModelInterface"]
