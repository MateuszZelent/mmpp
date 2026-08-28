"""Load-bearing data contracts shared by numerical and analytical vortex paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mmpp._shared.repr_html import make_simple_card


@dataclass
class TrajectoryResult:
    """Canonical trajectory contract for vortex analysis workflows."""

    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    polarity: np.ndarray
    method: str
    confidence: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.time = np.asarray(self.time, dtype=float).reshape(-1)
        self.x = np.asarray(self.x, dtype=float).reshape(-1)
        self.y = np.asarray(self.y, dtype=float).reshape(-1)
        self.polarity = np.asarray(self.polarity, dtype=int).reshape(-1)
        self.confidence = np.asarray(self.confidence, dtype=float).reshape(-1)

        n = int(self.time.size)
        for name in ("x", "y", "polarity", "confidence"):
            value = getattr(self, name)
            if value.size != n:
                raise ValueError(
                    f"TrajectoryResult.{name} size mismatch: expected {n}, got {value.size}"
                )

    @property
    def z(self) -> np.ndarray:
        """Complex trajectory signal ``z(t) = (x-x0) + i(y-y0)``."""
        x_center = float(np.mean(self.x)) if self.x.size else 0.0
        y_center = float(np.mean(self.y)) if self.y.size else 0.0
        return (self.x - x_center) + 1j * (self.y - y_center)

    @property
    def r(self) -> np.ndarray:
        """Orbit radius versus time."""
        return np.abs(self.z)

    @property
    def phi(self) -> np.ndarray:
        """Instantaneous orbital angle."""
        return np.angle(self.z)

    @property
    def phi_unwrapped(self) -> np.ndarray:
        """Unwrapped orbital angle."""
        return np.unwrap(self.phi)

    @property
    def velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Numerical velocity components ``(vx, vy)``."""
        if self.time.size < 2:
            zeros = np.zeros_like(self.x, dtype=float)
            return zeros, zeros

        vx = np.gradient(self.x, self.time)
        vy = np.gradient(self.y, self.time)
        return np.asarray(vx, dtype=float), np.asarray(vy, dtype=float)

    @property
    def instantaneous_frequency(self) -> np.ndarray:
        """Angular frequency estimated as ``d(phi_unwrapped)/dt``."""
        if self.time.size < 2:
            return np.zeros_like(self.time, dtype=float)
        return np.asarray(np.gradient(self.phi_unwrapped, self.time), dtype=float)

    @property
    def rotation_sense(self) -> str:
        """Rotation direction inferred from mean angular frequency."""
        omega = self.instantaneous_frequency
        return "CCW" if float(np.mean(omega)) >= 0.0 else "CW"

    @property
    def analysis(self):
        """Source-agnostic trajectory analysis accessor."""
        from .analysis import TrajectoryAnalysisAccessor

        return TrajectoryAnalysisAccessor(self)

    @property
    def compare(self):
        """Trajectory comparison accessor."""
        from .compare import TrajectoryComparisonAccessor

        return TrajectoryComparisonAccessor(self)

    @property
    def plt(self):
        """Static plotting accessor."""
        from .plot.static import TrajectoryPlotAccessor

        return TrajectoryPlotAccessor(self)

    def __repr__(self) -> str:
        return (
            "TrajectoryResult("
            f"n={self.time.size}, method={self.method!r}, "
            f"rotation={self.rotation_sense})"
        )

    def _repr_html_(self) -> str:
        n = int(self.time.size)
        radius_mean_nm = float(np.mean(self.r) * 1e9) if n else float("nan")
        freq_mean_ghz = (
            float(np.mean(self.instantaneous_frequency) * 1e-9)
            if n >= 2
            else float("nan")
        )
        rows = [
            ("samples", str(n)),
            ("method", str(self.method)),
            ("rotation_sense", self.rotation_sense),
            ("radius_mean_nm", f"{radius_mean_nm:.6g}"),
            ("mean_frequency_ghz", f"{freq_mean_ghz:.6g}"),
            (".analysis", "Trajectory analysis accessor"),
            (".compare", "Trajectory comparison accessor"),
            (".plt", "Trajectory plotting accessor"),
        ]
        return make_simple_card(
            title="TrajectoryResult",
            subtitle="Canonical vortex-core trajectory contract",
            rows=rows,
        )


__all__ = ["TrajectoryResult"]
