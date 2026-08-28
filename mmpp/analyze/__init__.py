"""Analysis namespace for job-level post-processing modules."""

from __future__ import annotations

import importlib
from html import escape as _esc
from typing import Any

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


class AnalyzeInterface:
    """Entry-point namespace for analysis modules."""

    def __init__(
        self,
        job_result,
        mmpp_instance: Any | None = None,
        dataset_name: str | None = None,
        slice_info: Any | None = None,
    ):
        self._job = job_result
        self._mmpp = mmpp_instance
        self._dataset_name = dataset_name
        self._slice_info = slice_info
        self._hysteresis: Any | None = None

    @property
    def hysteresis(self):
        """Hysteresis analysis namespace."""
        if self._hysteresis is None:
            from .hysteresis import HysteresisInterface

            self._hysteresis = HysteresisInterface(
                self._job,
                mmpp_instance=self._mmpp,
                dataset_name=self._dataset_name,
                slice_info=self._slice_info,
            )
        return self._hysteresis

    def __repr__(self) -> str:
        return (
            "AnalyzeInterface("
            f"dataset={self._dataset_name!r}, slice={self._slice_info!r})"
        )

    def _repr_html_(self) -> str:
        job_name = _esc(str(getattr(self._job, "name", "unknown")))
        job_path = _esc(str(getattr(self._job, "path", "")))
        dataset = (
            _esc(str(self._dataset_name)) if self._dataset_name is not None else "auto"
        )
        slice_label = (
            _esc(str(self._slice_info)) if self._slice_info is not None else "full"
        )

        example = "\n".join(
            [
                "analysis = job[0].analyze",
                "res = analysis.hysteresis.from_table(field='B_extx', magnetization='mx')",
                "res.plot.loop(show_hc=True)",
                "res.plot.interactive(toolbar='auto')",
            ]
        )
        api = api_help_html(
            self,
            title="Analyze API help",
            prefix="job[0].analyze",
            properties=[("hysteresis", "Hysteresis loop analysis namespace")],
            subtitle="Live public API for job-level analysis helpers.",
            chrome=False,
        )
        return node_card_html(
            "Analyze Interface",
            icon="🧪",
            subtitle=f"Job <code>{job_name}</code> at <code>{job_path}</code>",
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
                            "Namespaces:",
                            [(".hysteresis", NODE_COLOR_ANALYSIS)],
                        )
                    ]
                ),
                examples_section_html(example),
            ],
            api=api,
            uid="mmpp-analyze",
        )

    def _repr_mimebundle_(self, include=None, exclude=None):
        html = self._repr_html_()
        text = self.__repr__()
        if html:
            return {"text/html": html, "text/plain": text}
        return {"text/plain": text}


class DatasetSpecificAnalyze(AnalyzeInterface):
    """Analyze namespace bound to a dataset and optional slice context."""

    def __init__(
        self,
        job_result,
        dataset_name: str,
        mmpp_instance: Any | None = None,
        slice_info: Any | None = None,
    ):
        super().__init__(
            job_result,
            mmpp_instance=mmpp_instance,
            dataset_name=dataset_name,
            slice_info=slice_info,
        )


__all__ = ["AnalyzeInterface", "DatasetSpecificAnalyze"]


def __getattr__(name: str):
    if name == "hysteresis":
        return importlib.import_module(".hysteresis", __name__)
    raise AttributeError(name)
