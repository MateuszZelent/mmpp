"""Dataset-aware analytical-model interface."""

from __future__ import annotations

import uuid
from typing import Any

from mmpp._repr_helpers import api_help_html, html_tabs
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
        overview = make_simple_card(
            title="Vortex Model Interface",
            subtitle="Dataset-aware analytical models for vortex dynamics",
            rows=rows,
        )
        api = api_help_html(
            self,
            title="Vortex model API help",
            prefix="vortex.model",
            properties=[("thiele", "Thiele-equation model namespace")],
            subtitle="Live public API for dataset-aware analytical model namespaces.",
            chrome=False,
        )
        return (
            '<div style=\'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;'
            "border:2px solid #334155;border-radius:12px;padding:14px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;'>"
            + html_tabs(
                [("Overview", overview), ("API", api)],
                uid=f"mmpp-vortex-model-{uuid.uuid4().hex}",
            )
            + "</div>"
        )


__all__ = ["VortexModelInterface"]
