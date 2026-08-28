# ruff: noqa: UP007
"""Result contracts for skyrmion analysis."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from html import escape
from typing import Any, Optional

import numpy as np


def _tabbed_result_html(
    obj: Any,
    *,
    title: str,
    icon: str,
    prefix: str,
    metrics: tuple[tuple[str, object], ...],
    properties: tuple[tuple[str, str], ...],
) -> str:
    """Render a canonical MMPP analysis-result card."""
    import uuid

    from mmpp._repr_helpers import api_help_html, helper_table_html, html_tabs

    uid = f"mmpp-skyrmion-result-{uuid.uuid4().hex[:8]}"
    rows = [(escape(str(key)), escape(str(value))) for key, value in metrics]
    overview = (
        "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',"
        "Arial,sans-serif;background:linear-gradient(135deg,#0f172a 0%,"
        '#1e293b 50%,#334155 100%);color:#e2e8f0;padding:4px 0 0 0;">'
        "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;"
        f"margin-bottom:4px;'>{escape(icon)} {escape(title)}</div>"
        "<div style='font-size:0.85em;color:#94a3b8;margin-bottom:12px;'>"
        "Physical values use SI units; inspect flags and diagnostics before "
        "interpreting the fit.</div>" + helper_table_html(rows) + "</div>"
    )
    api = api_help_html(
        obj,
        title=f"{title} API help",
        prefix=prefix,
        properties=list(properties),
        methods=[],
        subtitle="Live result attributes and derived convenience properties.",
        chrome=False,
    )
    return (
        "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',"
        "Arial,sans-serif;border:2px solid #334155;border-radius:12px;"
        "padding:18px;margin:10px 0;background:linear-gradient(135deg,#0f172a "
        "0%,#1e293b 50%,#334155 100%);color:#e2e8f0;box-shadow:0 10px 25px "
        'rgba(0,0,0,0.3),0 0 0 1px rgba(148,163,184,0.1) inset;">'
        + html_tabs([("Overview", overview), ("API", api)], uid=uid)
        + "</div>"
    )


@dataclass
class SkyrmionTopologyResult:
    """Topology observables for one two-dimensional magnetisation snapshot."""

    Q: float
    center_xy_m: tuple[float, float]
    polarity: int
    background_sign: int
    core_mz: float
    background_mz: float
    contrast_mz: float
    q_density: np.ndarray
    q_abs_integral: float
    q_purity: float
    q_localized_fraction: float
    confidence: float
    valid: bool
    method: str = "berg_luscher"
    convention: str = "up"
    state: str = "unknown"
    flags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.Q = float(self.Q)
        self.q_density = np.asarray(self.q_density, dtype=float)
        self.center_xy_m = (float(self.center_xy_m[0]), float(self.center_xy_m[1]))
        self.method = str(self.method)
        self.convention = str(self.convention)
        self.state = str(self.state)

    @property
    def topological_density(self) -> np.ndarray:
        """Alias for the local topological-charge density map."""
        return self.q_density

    @property
    def center(self) -> tuple[float, float]:
        return self.center_xy_m

    @property
    def core_position(self) -> tuple[float, float]:
        return self.center_xy_m

    @property
    def core_polarity(self) -> int:
        return int(self.polarity)

    @property
    def background_polarity(self) -> int:
        return int(self.background_sign)

    @property
    def is_reversed(self) -> bool:
        """Whether the core points opposite to the estimated background."""
        return (
            self.polarity != 0
            and self.background_sign != 0
            and self.polarity != self.background_sign
        )

    @property
    def is_skyrmion(self) -> bool:
        return self.state == "skyrmion" and bool(self.valid)

    def _repr_html_(self) -> str:
        return _tabbed_result_html(
            self,
            title="SkyrmionTopologyResult",
            icon="🧭",
            prefix="job.solitons.skyrmion.detect()",
            metrics=(
                ("state", self.state),
                ("Q", f"{self.Q:.6g}"),
                ("center_xy_m", self.center_xy_m),
                ("core_polarity", self.core_polarity),
                ("background_polarity", self.background_polarity),
                ("confidence", f"{self.confidence:.4f}"),
                ("method", self.method),
                ("valid", self.valid),
                ("flags", ", ".join(self.flags) or "none"),
            ),
            properties=(
                ("Q", "Integrated topological charge"),
                ("topological_density", "Local charge-density map"),
                ("center_xy_m", "Detected physical centre in metres"),
                ("core_polarity", "Sign of core m_z"),
                ("background_polarity", "Sign of background m_z"),
                ("state", "Skyrmion-specific classification"),
                ("is_skyrmion", "Classification and validity shortcut"),
                ("confidence", "Topology confidence in [0, 1]"),
                ("flags", "Reason-coded diagnostic flags"),
            ),
        )


@dataclass
class SkyrmionSizeResult:
    """Physical size estimates and radial-fit diagnostics."""

    center_xy_m: tuple[float, float]
    radius_m: Optional[float]
    diameter_m: Optional[float]
    wall_width_m: Optional[float]
    scale_m: Optional[float]
    radius_90_m: Optional[float]
    radius_50_m: Optional[float]
    radius_10_m: Optional[float]
    sigma_m: Optional[float]
    gaussian_fwhm_m: Optional[float]
    model: str
    fit_method: str
    fit_success: bool
    background_mz: float
    core_mz: float
    contrast_mz: float
    normalized_rmse: float
    aicc: float
    quality: str
    requested_method: str = "auto"
    flags: tuple[str, ...] = ()
    radial_r_m: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    radial_mz: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    model_mz: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    candidate_diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.center_xy_m = (float(self.center_xy_m[0]), float(self.center_xy_m[1]))
        self.radial_r_m = np.asarray(self.radial_r_m, dtype=float)
        self.radial_mz = np.asarray(self.radial_mz, dtype=float)
        self.model_mz = np.asarray(self.model_mz, dtype=float)

    @property
    def radius(self) -> Optional[float]:
        return self.radius_m

    @property
    def diameter(self) -> Optional[float]:
        return self.diameter_m

    @property
    def radius_nm(self) -> Optional[float]:
        return None if self.radius_m is None else float(self.radius_m) * 1e9

    @property
    def diameter_nm(self) -> Optional[float]:
        return None if self.diameter_m is None else float(self.diameter_m) * 1e9

    @property
    def wall_width_10_90_m(self) -> Optional[float]:
        return self.wall_width_m

    @property
    def wall_width_10_90(self) -> Optional[float]:
        return self.wall_width_m

    @property
    def residual(self) -> float:
        return float(self.normalized_rmse)

    @property
    def success(self) -> bool:
        return bool(self.fit_success)

    @property
    def method(self) -> str:
        return self.model

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "quality": self.quality,
            "flags": self.flags,
            "normalized_rmse": self.normalized_rmse,
            "aicc": self.aicc,
            "candidates": self.candidate_diagnostics,
            **self.metadata,
        }

    def __iter__(self) -> Iterator[Optional[float]]:
        """Allow ``radius, diameter = result`` in notebook workflows."""
        yield self.radius_m
        yield self.diameter_m

    def _repr_html_(self) -> str:
        return _tabbed_result_html(
            self,
            title="SkyrmionSizeResult",
            icon="📏",
            prefix="job.solitons.skyrmion.size.fit()",
            metrics=(
                ("requested", self.requested_method),
                ("selected model", self.model),
                ("radius [nm]", self.radius_nm),
                ("diameter [nm]", self.diameter_nm),
                ("wall width 10-90 [m]", self.wall_width_m),
                ("sigma [m]", self.sigma_m),
                ("normalized RMSE", f"{self.normalized_rmse:.5g}"),
                ("quality", self.quality),
                ("success", self.success),
                ("flags", ", ".join(self.flags) or "none"),
            ),
            properties=(
                ("radius_m", "Contrast-50 radius in metres"),
                ("diameter_m", "Twice the contrast-50 radius"),
                ("wall_width_10_90_m", "10-90 contrast wall width"),
                ("scale_m", "Domain-wall/ansatz length scale"),
                ("sigma_m", "Gaussian sigma when applicable"),
                ("model", "Selected effective model"),
                ("requested_method", "User-requested model selection"),
                ("diagnostics", "Quality, flags, AICc, and candidate fits"),
                ("radial_r_m", "Radial sample coordinates"),
                ("radial_mz", "Measured radial m_z profile"),
                ("model_mz", "Selected fitted profile"),
            ),
        )


@dataclass
class SkyrmionAnalysisResult:
    """Combined topology and size result."""

    topology: SkyrmionTopologyResult
    size: SkyrmionSizeResult
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def center_xy_m(self) -> tuple[float, float]:
        return self.size.center_xy_m

    @property
    def Q(self) -> float:
        return self.topology.Q

    @property
    def radius_m(self) -> Optional[float]:
        return self.size.radius_m

    @property
    def diameter_m(self) -> Optional[float]:
        return self.size.diameter_m

    def _repr_html_(self) -> str:
        return _tabbed_result_html(
            self,
            title="SkyrmionAnalysisResult",
            icon="🧲",
            prefix="job.solitons.skyrmion.analyze()",
            metrics=(
                ("state", self.topology.state),
                ("Q", f"{self.Q:.6g}"),
                ("radius [nm]", self.size.radius_nm),
                ("diameter [nm]", self.size.diameter_nm),
                ("model", self.size.model),
                ("quality", self.size.quality),
            ),
            properties=(
                ("topology", "SkyrmionTopologyResult"),
                ("size", "SkyrmionSizeResult"),
                ("Q", "Topological-charge shortcut"),
                ("radius_m", "Radius shortcut"),
                ("diameter_m", "Diameter shortcut"),
                ("center_xy_m", "Physical centre shortcut"),
            ),
        )


__all__ = [
    "SkyrmionTopologyResult",
    "SkyrmionSizeResult",
    "SkyrmionAnalysisResult",
]
