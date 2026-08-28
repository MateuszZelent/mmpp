"""Phase analysis helpers for vortex trajectories."""

from __future__ import annotations

import warnings

import numpy as np

from ..._method_helpers import InteractiveNodeMixin
from ..core.models import TrajectoryResult
from .models import PhaseResult

try:
    from scipy.signal import hilbert

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback tested
    hilbert = None  # type: ignore[assignment]
    SCIPY_AVAILABLE = False


def _phase_from_complex(trajectory: TrajectoryResult) -> tuple[np.ndarray, str]:
    z = np.asarray(trajectory.z, dtype=np.complex128)
    return np.angle(z), "complex"


def _phase_from_hilbert(trajectory: TrajectoryResult) -> tuple[np.ndarray, str]:
    if not SCIPY_AVAILABLE or hilbert is None:
        warnings.warn(
            "SciPy is unavailable; falling back from Hilbert to complex phase.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _phase_from_complex(trajectory)

    x = np.asarray(trajectory.x, dtype=float)
    analytic = hilbert(x - np.mean(x))
    return np.angle(analytic), "hilbert"


class PhaseAnalyzer(InteractiveNodeMixin):
    """Phase analysis facade for a trajectory."""

    _interactive_owner = "job[0].vortex.trajectory.phase"
    _interactive_nodes = frozenset(
        {"instantaneous", "unwrapped", "frequency", "mean_frequency"}
    )

    def __init__(self, trajectory: TrajectoryResult):
        self._trajectory = trajectory
        self._cache: dict[str, PhaseResult] = {}

    def _build(self, method: str) -> PhaseResult:
        method_norm = method.lower()
        if method_norm in self._cache:
            return self._cache[method_norm]

        if method_norm == "complex":
            phase, effective_method = _phase_from_complex(self._trajectory)
        elif method_norm == "hilbert":
            phase, effective_method = _phase_from_hilbert(self._trajectory)
        else:
            raise ValueError("method must be 'complex' or 'hilbert'")

        phase_unwrapped = np.unwrap(phase)
        if self._trajectory.time.size >= 2:
            omega = np.gradient(phase_unwrapped, self._trajectory.time)
        else:
            omega = np.zeros_like(self._trajectory.time, dtype=float)

        result = PhaseResult(
            time=np.asarray(self._trajectory.time, dtype=float),
            phase=np.asarray(phase, dtype=float),
            phase_unwrapped=np.asarray(phase_unwrapped, dtype=float),
            omega=np.asarray(omega, dtype=float),
            method=effective_method,
            metadata={"requested_method": method_norm},
        )
        self._cache[method_norm] = result
        return result

    def instantaneous(self, method: str = "complex") -> np.ndarray:
        """Return wrapped phase ``phi(t)``."""
        return np.asarray(self._build(method).phase, dtype=float)

    def unwrapped(self, method: str = "complex") -> np.ndarray:
        """Return unwrapped phase."""
        return np.asarray(self._build(method).phase_unwrapped, dtype=float)

    def frequency(self, method: str = "complex", unit: str = "rad/s") -> np.ndarray:
        """Return instantaneous frequency."""
        result = self._build(method)
        unit_norm = unit.lower()

        if unit_norm in {"rad/s", "omega", "w"}:
            return np.asarray(result.omega, dtype=float)
        if unit_norm in {"hz", "f"}:
            return np.asarray(result.frequency_hz, dtype=float)
        if unit_norm in {"ghz"}:
            return np.asarray(result.frequency_hz * 1e-9, dtype=float)
        raise ValueError("unit must be 'rad/s', 'hz', or 'ghz'")

    def mean_frequency(
        self,
        *,
        center: str | tuple[float, float] | np.ndarray = "mean",
        t_min: float | None = None,
        transient_fraction: float | None = None,
        signed: bool = False,
        unit: str = "hz",
    ) -> float:
        """Return mean orbital frequency around an explicit or inferred center."""
        t = np.asarray(self._trajectory.time, dtype=float)
        if t.size < 3:
            return float("nan")
        if t_min is None and transient_fraction is not None:
            frac = min(max(float(transient_fraction), 0.0), 0.95)
            t_min = float(t[0] + frac * (t[-1] - t[0]))
        mask = np.ones_like(t, dtype=bool) if t_min is None else t >= float(t_min)
        if np.count_nonzero(mask) < 3:
            return float("nan")

        x = np.asarray(self._trajectory.x, dtype=float)
        y = np.asarray(self._trajectory.y, dtype=float)
        if isinstance(center, str):
            center_norm = center.lower()
            if center_norm == "mean":
                cx = float(np.mean(x[mask]))
                cy = float(np.mean(y[mask]))
            elif center_norm in {"disk", "origin"}:
                cx = 0.0
                cy = 0.0
            else:
                raise ValueError("center must be 'mean', 'disk', or a 2-tuple")
        else:
            c = np.asarray(center, dtype=float).reshape(2)
            cx, cy = float(c[0]), float(c[1])

        z = (x[mask] - cx) + 1j * (y[mask] - cy)
        phase = np.unwrap(np.angle(z))
        omega = np.gradient(phase, t[mask])
        hz = float(np.mean(omega) / (2.0 * np.pi))
        value = hz if signed else abs(hz)
        unit_norm = unit.lower()
        if unit_norm in {"hz", "f"}:
            return value
        if unit_norm == "ghz":
            return value * 1e-9
        if unit_norm in {"rad/s", "omega", "w"}:
            return value * 2.0 * np.pi
        raise ValueError("unit must be 'hz', 'ghz', or 'rad/s'")

    @property
    def plt(self):
        """Plot accessor for phase using complex method by default."""
        return self._build("complex").plt
