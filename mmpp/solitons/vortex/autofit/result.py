"""Result models for vortex autofit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AutofitDiagnostics:
    """Diagnostics from an autofit run."""

    n_evaluations: int = 0
    n_global_evaluations: int = 0
    n_local_evaluations: int = 0
    time_total_s: float = 0.0
    time_global_s: float = 0.0
    time_local_s: float = 0.0

    optimizer_message: str = ""
    optimizer_nit: int = 0
    loss_history: list[float] = field(default_factory=list)
    evaluation_records: list[dict[str, Any]] = field(default_factory=list)

    hessian_approx: np.ndarray | None = None
    param_correlations: np.ndarray | None = None
    param_uncertainties: dict[str, float] | None = None
    poorly_identified: list[str] = field(default_factory=list)

    active_bounds: dict[str, str] = field(default_factory=dict)

    def _repr_html_(self) -> str:
        from html import escape as _esc

        card = (
            "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);"
        )
        section = (
            "background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:8px;border:1px solid rgba(148,163,184,0.2);"
        )
        lbl = "padding:3px 8px;color:#94a3b8;font-size:0.85em;font-weight:600;"
        val = "padding:3px 8px;color:#e2e8f0;font-size:0.85em;font-family:monospace;"

        html = f"<div style='{card}'>"
        html += (
            "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;margin-bottom:10px;'>"
            "📊 Autofit Diagnostics</div>"
        )

        # Timing
        html += f"<div style='{section}'>"
        html += "<div style='font-weight:600;color:#e2e8f0;margin-bottom:4px;font-size:0.9em;'>⏱️ Timing</div>"
        html += "<table style='width:100%;border-collapse:collapse;'>"
        for label, t in [
            ("Total", self.time_total_s),
            ("Global stage", self.time_global_s),
            ("Local stage", self.time_local_s),
        ]:
            html += (
                f"<tr><td style='{lbl}'>{label}</td>"
                f"<td style='{val}'>{t:.2f} s</td></tr>"
            )
        html += "</table></div>"

        # Evaluations
        html += f"<div style='{section}'>"
        html += "<div style='font-weight:600;color:#e2e8f0;margin-bottom:4px;font-size:0.9em;'>🔢 Evaluations</div>"
        html += "<table style='width:100%;border-collapse:collapse;'>"
        for label, n in [
            ("Total", self.n_evaluations),
            ("Global", self.n_global_evaluations),
            ("Local", self.n_local_evaluations),
            ("Optimizer iterations", self.optimizer_nit),
        ]:
            html += f"<tr><td style='{lbl}'>{label}</td><td style='{val}'>{n}</td></tr>"
        if self.optimizer_message:
            html += (
                f"<tr><td style='{lbl}'>Message</td>"
                f"<td style='{val}'>{_esc(self.optimizer_message)}</td></tr>"
            )
        html += "</table></div>"

        # Uncertainties
        if self.param_uncertainties:
            html += f"<div style='{section}'>"
            html += "<div style='font-weight:600;color:#e2e8f0;margin-bottom:4px;font-size:0.9em;'>📐 Parameter Uncertainties</div>"
            html += "<table style='width:100%;border-collapse:collapse;'>"
            for name, unc in self.param_uncertainties.items():
                html += (
                    f"<tr><td style='{lbl}font-family:monospace;color:#93c5fd;'>{_esc(name)}</td>"
                    f"<td style='{val}'>± {unc:.4g}</td></tr>"
                )
            html += "</table></div>"

        # Poorly identified
        if self.poorly_identified:
            html += (
                f"<div style='{section}border-color:rgba(245,158,11,0.3);'>"
                "<div style='font-weight:600;color:#f59e0b;margin-bottom:4px;font-size:0.9em;'>"
                "⚠️ Poorly Identified Parameters</div>"
                "<div style='font-size:0.85em;color:#fbbf24;'>"
                + ", ".join(f"<code>{_esc(p)}</code>" for p in self.poorly_identified)
                + "</div></div>"
            )

        # Active bounds
        if self.active_bounds:
            html += f"<div style='{section}'>"
            html += "<div style='font-weight:600;color:#e2e8f0;margin-bottom:4px;font-size:0.9em;'>🔒 Active Bounds</div>"
            html += "<table style='width:100%;border-collapse:collapse;'>"
            for name, bound in self.active_bounds.items():
                html += (
                    f"<tr><td style='{lbl}font-family:monospace;color:#93c5fd;'>{_esc(name)}</td>"
                    f"<td style='{val}color:#f59e0b;'>{_esc(bound)}</td></tr>"
                )
            html += "</table></div>"

        html += "</div>"
        return html


@dataclass
class VortexAutofitResult:
    """Complete result of a vortex autofit run."""

    best_params: dict[str, float]
    initial_params: dict[str, float]
    param_sources: dict[str, str]
    frozen_params: dict[str, float]
    fitted_params: tuple[str, ...]

    loss_total: float
    loss_breakdown: dict[str, float]
    baseline_loss: float

    comparison: Any  # VortexAnalyticalComparison (lazy import avoids circular)
    diagnostics: AutofitDiagnostics

    success: bool
    warnings: list[str] = field(default_factory=list)
    config: Any = None  # AutofitConfig

    @property
    def improvement_ratio(self) -> float:
        """``loss_total / baseline_loss`` — values < 1 indicate improvement."""
        if self.baseline_loss <= 0.0:
            return float("inf")
        return self.loss_total / self.baseline_loss

    @property
    def plt(self):
        """Plotting accessor for autofit diagnostics."""
        from ._plotting import AutofitPlotAccessor

        return AutofitPlotAccessor(self)

    def __repr__(self) -> str:
        status = "OK" if self.success else "FAILED"
        ratio = self.improvement_ratio
        return (
            f"VortexAutofitResult({status}, "
            f"loss={self.loss_total:.4g}, "
            f"improvement={ratio:.3f}, "
            f"fitted={self.fitted_params})"
        )

    def _repr_html_(self) -> str:
        from html import escape as _esc

        status_color = "#22c55e" if self.success else "#ef4444"
        status_text = "Success" if self.success else "Failed"
        ratio = self.improvement_ratio
        improvement_color = "#22c55e" if ratio < 1.0 else "#ef4444"

        param_rows = "".join(
            f"<tr>"
            f"<td style='padding:3px 8px;font-family:monospace;color:#93c5fd;'>{_esc(k)}</td>"
            f"<td style='padding:3px 8px;color:#e2e8f0;text-align:right;'>{v:.6g}</td>"
            f"<td style='padding:3px 8px;color:#94a3b8;'>"
            f"{'fitted' if k in self.fitted_params else 'frozen'}</td>"
            f"</tr>"
            for k, v in self.best_params.items()
        )

        loss_rows = "".join(
            f"<tr>"
            f"<td style='padding:3px 8px;font-family:monospace;color:#93c5fd;'>{_esc(k)}</td>"
            f"<td style='padding:3px 8px;color:#e2e8f0;text-align:right;'>{v:.4g}</td>"
            f"</tr>"
            for k, v in self.loss_breakdown.items()
        )

        warnings_html = ""
        if self.warnings:
            items = "".join(
                f"<li style='color:#fbbf24;'>{_esc(w)}</li>" for w in self.warnings
            )
            warnings_html = (
                "<div style='margin-top:8px;background:rgba(251,191,36,0.1);"
                "padding:8px;border-radius:6px;border:1px solid rgba(251,191,36,0.3);'>"
                f"<ul style='margin:0;padding-left:20px;'>{items}</ul></div>"
            )

        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            'color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);">'
            # Title
            "<div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;'>"
            "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;'>"
            "Vortex Autofit Result</div>"
            f"<span style='background:{status_color};color:white;font-size:0.75em;"
            f"padding:2px 8px;border-radius:4px;font-weight:600;'>{status_text}</span></div>"
            # Summary
            "<div style='display:flex;gap:16px;margin-bottom:10px;font-size:0.9em;'>"
            f"<div><span style='color:#94a3b8;'>Loss:</span> "
            f"<span style='color:#e2e8f0;font-weight:600;'>{self.loss_total:.4g}</span></div>"
            f"<div><span style='color:#94a3b8;'>Baseline:</span> "
            f"<span style='color:#e2e8f0;'>{self.baseline_loss:.4g}</span></div>"
            f"<div><span style='color:#94a3b8;'>Ratio:</span> "
            f"<span style='color:{improvement_color};font-weight:600;'>{ratio:.3f}</span></div>"
            "</div>"
            # Parameters
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:8px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:4px;'>Parameters</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.85em;'>"
            "<thead><tr style='background:rgba(51,65,85,0.6);'>"
            "<th style='padding:3px 8px;text-align:left;color:#e2e8f0;'>Name</th>"
            "<th style='padding:3px 8px;text-align:right;color:#e2e8f0;'>Value</th>"
            "<th style='padding:3px 8px;text-align:left;color:#e2e8f0;'>Status</th>"
            f"</tr></thead><tbody>{param_rows}</tbody></table></div>"
            # Loss breakdown
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:8px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:4px;'>Loss breakdown</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.85em;'>"
            f"<tbody>{loss_rows}</tbody></table></div>" + warnings_html + "</div>"
        )


__all__ = ["VortexAutofitResult", "AutofitDiagnostics"]
