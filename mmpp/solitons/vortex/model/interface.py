"""Dataset-aware analytical-model interface."""

from __future__ import annotations

from typing import Any

from mmpp._shared.repr_html import make_simple_card


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
        rows = [
            (".thiele", "Thiele-equation model namespace"),
            (".thiele.cpp(...)", "Build CPP Thiele adapter"),
            (".thiele.cip(...)", "Build CIP Thiele adapter"),
        ]
        return make_simple_card(
            title="Vortex Model Interface",
            subtitle="Dataset-aware analytical models for vortex dynamics",
            rows=rows,
        )


__all__ = ["VortexModelInterface"]
