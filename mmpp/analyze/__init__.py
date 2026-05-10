"""Analysis namespace for job-level post-processing modules."""

from __future__ import annotations

import importlib
import uuid
from html import escape as _esc
from typing import Any

from mmpp._repr_helpers import api_help_html, html_tabs


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
        self._hysteresis = None

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

        methods = [
            (".hysteresis", "Hysteresis loop analysis namespace"),
        ]
        rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(name)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(desc)}</td></tr>"
            for name, desc in methods
        )

        example = "\n".join(
            [
                "analysis = job[0].analyze",
                "res = analysis.hysteresis.from_table(field='B_extx', magnetization='mx')",
                "res.plot.loop(show_hc=True)",
                "res.plot.interactive(toolbar='auto')",
            ]
        )

        overview = (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:10px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            'color:#e2e8f0;box-shadow:0 10px 22px rgba(0,0,0,0.28);">'
            "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;'>Analyze Interface</div>"
            f"<div style='color:#94a3b8;margin-top:4px;'>Job: {job_name}</div>"
            f"<div style='color:#94a3b8;margin-top:2px;'>Path: <code style='color:#cbd5e1;'>{job_path}</code></div>"
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-top:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='display:flex;gap:16px;flex-wrap:wrap;font-size:0.9em;'>"
            f"<div><span style='color:#94a3b8;'>Dataset:</span> <code style='color:#cbd5e1;'>{dataset}</code></div>"
            f"<div><span style='color:#94a3b8;'>Slice:</span> <code style='color:#cbd5e1;'>{slice_label}</code></div>"
            "</div></div>"
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-top:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            "<thead><tr style='text-align:left;background:rgba(51,65,85,0.6);'>"
            "<th style='padding:6px 8px;color:#e2e8f0;'>Namespace</th>"
            "<th style='padding:6px 8px;color:#e2e8f0;'>Description</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-top:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Examples</div>"
            "<pre style='margin:0;background:rgba(15,23,42,0.85);padding:10px;border-radius:6px;"
            f"color:#e2e8f0;overflow-x:auto;font-size:0.85em;'><code>{_esc(example)}</code></pre>"
            "</div></div>"
        )
        api = api_help_html(
            self,
            title="Analyze API help",
            prefix="job[0].analyze",
            properties=[("hysteresis", "Hysteresis loop analysis namespace")],
            subtitle="Live public API for job-level analysis helpers.",
            chrome=False,
        )
        return (
            '<div style=\'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;'
            "border:2px solid #334155;border-radius:12px;padding:14px;margin:10px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;box-shadow:0 10px 22px rgba(0,0,0,0.28);'>"
            + html_tabs(
                [("Overview", overview), ("API", api)],
                uid=f"mmpp-analyze-{uuid.uuid4().hex}",
            )
            + "</div>"
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
