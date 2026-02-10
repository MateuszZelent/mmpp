"""Phase analysis helpers for vortex trajectories."""

from __future__ import annotations

import warnings

import numpy as np

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


class PhaseAnalyzer:
    """Phase analysis facade for a trajectory."""

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

    @property
    def plt(self):
        """Plot accessor for phase using complex method by default."""
        return self._build("complex").plt
