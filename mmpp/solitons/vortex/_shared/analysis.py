"""Source-agnostic analysis accessors for vortex trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mmpp._shared.spectral import compute_psd, infer_dt


@dataclass
class DirectionalSpectrumResult:
    """Directional PSD split into CCW/CW contributions."""

    frequencies: np.ndarray
    power_ccw: np.ndarray
    power_cw: np.ndarray
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_power(self) -> np.ndarray:
        return np.asarray(self.power_ccw, dtype=float) + np.asarray(
            self.power_cw, dtype=float
        )

    @property
    def plt(self):
        return _DirectionalSpectrumPlotAccessor(self)


class _DirectionalSpectrumPlotAccessor:
    def __init__(self, result: DirectionalSpectrumResult):
        self._result = result

    def power_spectrum(self, *, ax=None, unit: str = "hz", **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 3.5), dpi=110)

        unit_norm = str(unit).lower()
        if unit_norm == "ghz":
            freq = self._result.frequencies * 1e-9
            xlabel = "Frequency [GHz]"
        else:
            freq = self._result.frequencies
            xlabel = "Frequency [Hz]"

        ax.plot(freq, self._result.power_ccw, label="CCW (+)", **kwargs)
        ax.plot(freq, self._result.power_cw, label="CW (-)", linestyle="--", **kwargs)
        ax.plot(freq, self._result.total_power, label="total", linestyle=":", alpha=0.9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Power [a.u.]")
        ax.set_title("Directional spectrum")
        ax.grid(True, alpha=0.25)
        ax.legend()
        return ax

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html

        return plot_accessor_html(
            "DirectionalSpectrumPlotAccessor",
            [
                (
                    ".power_spectrum(unit='hz')",
                    "CCW/CW/total directional power spectrum",
                    "unit: 'hz' or 'ghz'. Accepts matplotlib kwargs.",
                ),
            ],
        )


class TrajectoryOrbitAccessor:
    """Orbit fit methods for trajectory analysis."""

    def __init__(self, trajectory):
        self._trajectory = trajectory

    def fit(self, model: str = "ellipse"):
        from ..trajectory.orbit import fit_orbit_ellipse

        model_norm = str(model).lower()
        if model_norm != "ellipse":
            raise ValueError("Only model='ellipse' is supported")
        return fit_orbit_ellipse(self._trajectory)


class TrajectoryPhaseAccessor:
    """Phase/frequency methods for trajectory analysis."""

    def __init__(self, trajectory):
        self._trajectory = trajectory

    def frequency(self, method: str = "complex", unit: str = "rad/s") -> np.ndarray:
        from ..trajectory.phase import PhaseAnalyzer

        analyzer = PhaseAnalyzer(self._trajectory)
        return analyzer.frequency(method=method, unit=unit)

    def instantaneous(self, method: str = "complex") -> np.ndarray:
        from ..trajectory.phase import PhaseAnalyzer

        analyzer = PhaseAnalyzer(self._trajectory)
        return analyzer.instantaneous(method=method)


class TrajectorySpectrumAccessor:
    """Spectrum methods for trajectory analysis."""

    def __init__(self, trajectory):
        self._trajectory = trajectory

    def directional(
        self,
        *,
        method: str = "welch",
        nperseg: int | None = None,
        noverlap: int | None = None,
    ) -> DirectionalSpectrumResult:
        """Compute directional CCW/CW spectra from complex trajectory signal."""
        signal = np.asarray(self._trajectory.z, dtype=np.complex128)
        time = np.asarray(self._trajectory.time, dtype=float)
        if signal.size < 2 or time.size < 2:
            return DirectionalSpectrumResult(
                frequencies=np.array([], dtype=float),
                power_ccw=np.array([], dtype=float),
                power_cw=np.array([], dtype=float),
                method=str(method),
                metadata={"status": "insufficient_samples"},
            )

        dt = infer_dt(time)
        if dt <= 0.0:
            raise ValueError("time axis must be strictly increasing")

        method_norm = str(method).lower()
        if method_norm == "fft":
            method_norm = "periodogram"
        if method_norm not in {"welch", "periodogram"}:
            raise ValueError("method must be 'welch' or 'periodogram'")

        freq, power_ccw, used_method, metadata = compute_psd(
            signal,
            dt=dt,
            method=method_norm,
            nperseg=nperseg,
            noverlap=noverlap,
        )
        _, power_cw, used_method_cw, _ = compute_psd(
            np.conjugate(signal),
            dt=dt,
            method=method_norm,
            nperseg=nperseg,
            noverlap=noverlap,
        )
        used = used_method if used_method == used_method_cw else "mixed"
        metadata["method"] = used

        return DirectionalSpectrumResult(
            frequencies=np.asarray(freq, dtype=float),
            power_ccw=np.asarray(np.real(power_ccw), dtype=float),
            power_cw=np.asarray(np.real(power_cw), dtype=float),
            method=used,
            metadata=metadata,
        )


class TrajectoryAnalysisAccessor:
    """Main analysis namespace attached to canonical trajectories."""

    def __init__(self, trajectory):
        self._trajectory = trajectory
        self._orbit = None
        self._phase = None
        self._spectrum = None

    @property
    def orbit(self) -> TrajectoryOrbitAccessor:
        if self._orbit is None:
            self._orbit = TrajectoryOrbitAccessor(self._trajectory)
        return self._orbit

    @property
    def phase(self) -> TrajectoryPhaseAccessor:
        if self._phase is None:
            self._phase = TrajectoryPhaseAccessor(self._trajectory)
        return self._phase

    @property
    def spectrum(self) -> TrajectorySpectrumAccessor:
        if self._spectrum is None:
            self._spectrum = TrajectorySpectrumAccessor(self._trajectory)
        return self._spectrum


__all__ = [
    "DirectionalSpectrumResult",
    "TrajectoryAnalysisAccessor",
]
