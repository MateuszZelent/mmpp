# ruff: noqa: N802, N803, N806, PLR0913
"""
Thiele equation models for vortex core dynamics in nanodisks.

Implements the collective-coordinate (Thiele) approach for magnetic vortex
core trajectory calculation in two geometries:

- **CIP** (Current-In-Plane): adiabatic + non-adiabatic STT (Zhang–Li),
  following Moon et al., arXiv:0809.0952.
- **CPP** (Current-Perpendicular-to-Plane): Slonczewski STT for vortex STNO,
  following Guslienko et al., Phys. Rev. B 89, 044412 (2014) / PMC 4134337.

Both models reduce the full micromagnetic dynamics to an ODE for the vortex
core position **r**(t) = (X(t), Y(t)), which can be integrated numerically
with ``scipy.integrate.solve_ivp``.

Units
-----
All quantities are in **SI units** throughout:

- Lengths in meters [m]
- Time in seconds [s]
- Magnetization in A/m
- Current density in A/m²
- Magnetic field in Tesla [T]
- Frequencies returned in Hz and GHz

References
----------
1. A.A. Thiele, Phys. Rev. Lett. 30, 230 (1973).
2. J.-H. Moon et al., arXiv:0809.0952 — CIP Thiele + STT (Zhang–Li).
3. K.Y. Guslienko et al., Phys. Rev. B 89 (2014) / PMC 4134337 —
   CPP nonlinear Thiele / vortex STNO auto-oscillator.
4. V. Novosad et al., arXiv:cond-mat/0503632 — gyrotropic eigenfrequency.
5. K.Y. Guslienko et al., J. Appl. Phys. 91, 8037 (2002) — eigenfrequencies.
6. V.S. Pribiag et al., Nature Physics 3, 498 (2007) / arXiv:cond-mat/0702253.
7. R. Dussaux et al., Nature Commun. 1, 8 (2010) / arXiv:1001.4933.
8. A.V. Khvalkovskiy et al., arXiv:0904.1751 — limitations of Thiele approach.

Warning
-------
The Thiele equation is a *rigid-vortex* approximation. It can fail
quantitatively (and sometimes qualitatively) for certain geometries,
large core displacements, or near polarity-switching events.
See Khvalkovskiy et al. (arXiv:0904.1751) for known limitations.
Always validate against full micromagnetic (LLGS) simulations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, TypeAlias

import numpy as np

from .base import AnalyticalResult
from .constants import GAMMA_E, MU0

# ---------------------------------------------------------------------------
# Physical helpers
# ---------------------------------------------------------------------------

# Bohr magneton [J/T]
_MU_B: float = 9.2740100783e-24

# Elementary charge [C]
_E_CHARGE: float = 1.602176634e-19

# Reduced Planck constant [J·s]
_HBAR: float = 1.054571817e-34

# Boltzmann constant [J/K]
_K_B: float = 1.380649e-23


# ---------------------------------------------------------------------------
# Data classes for parameters
# ---------------------------------------------------------------------------


@dataclass
class MaterialParams:
    """
    Material parameters for a ferromagnetic layer.

    Parameters
    ----------
    Ms : float
        Saturation magnetization [A/m].  Typical Permalloy: 8×10⁵ A/m.
    alpha : float
        Gilbert damping constant (dimensionless).  Typical Py: 0.005–0.01.
    P : float
        Spin polarization of the current (dimensionless, 0 < P ≤ 1).
    A : float, optional
        Exchange stiffness [J/m].  Default 1.3×10⁻¹¹ (Permalloy).
        Used only for estimating the exchange length / core radius.
    beta_nonadiabatic : float, optional
        Non-adiabatic STT parameter β (Zhang–Li).  Relevant only for CIP.
        If None, defaults to α (adiabatic limit).
    gamma : float, optional
        Gyromagnetic ratio [rad/(s·T)].  Default: free-electron value.
    """

    Ms: float
    alpha: float
    P: float
    A: float = 1.3e-11
    beta_nonadiabatic: float | None = None
    gamma: float = GAMMA_E  # noqa: RUF009

    @property
    def beta(self) -> float:
        """Non-adiabatic parameter β.  Falls back to α if not set."""
        return self.beta_nonadiabatic if self.beta_nonadiabatic is not None else self.alpha

    @property
    def exchange_length(self) -> float:
        r"""Exchange length :math:`\ell_{\rm ex} = \sqrt{2A/(\mu_0 M_s^2)}` [m]."""
        return math.sqrt(2.0 * self.A / (MU0 * self.Ms**2))


@dataclass
class DiskGeometry:
    """
    Geometry of a magnetic nanodisk.

    Parameters
    ----------
    R : float
        Disk radius [m].
    L : float
        Disk thickness [m].
    core_diameter : float, optional
        Vortex-core diameter [m].  If None, estimated as 2 × exchange_length
        from an associated ``MaterialParams``.
    """

    R: float
    L: float
    core_diameter: float | None = None

    def Rc(self, mat: MaterialParams | None = None) -> float:
        """Core radius [m].  Uses ``core_diameter/2`` if set, else ``exchange_length``."""
        if self.core_diameter is not None:
            return self.core_diameter / 2.0
        if mat is not None:
            return mat.exchange_length
        return 5e-9  # safe fallback ≈ Py exchange length


@dataclass(frozen=True)
class ExternalField:
    """
    External magnetic field applied to the sample.

    All components are in Tesla.  Use ``Bz_T`` for an out-of-plane field
    and ``Bx_T`` / ``By_T`` for in-plane components.
    """

    Bx_T: float = 0.0
    By_T: float = 0.0
    Bz_T: float = 0.0

    @staticmethod
    def from_any(value: object) -> ExternalField:
        """Coerce float / tuple / list / ndarray / ExternalField → ExternalField.

        Rules
        -----
        - ``None``          → zero field
        - ``float`` / ``int`` → interpreted as Bz [T]
        - length-3 sequence → ``(Bx, By, Bz)`` [T]
        - ``ExternalField``  → returned as-is
        """
        if value is None:
            return ExternalField()
        if isinstance(value, ExternalField):
            return value
        if isinstance(value, (int, float)):
            return ExternalField(0.0, 0.0, float(value))
        if isinstance(value, (tuple, list, np.ndarray)):
            arr = np.asarray(value, dtype=float).ravel()
            if arr.size == 3:
                return ExternalField(float(arr[0]), float(arr[1]), float(arr[2]))
        raise TypeError(
            "B_ext must be float (Bz), a length-3 (Bx, By, Bz), or ExternalField"
        )


@dataclass(frozen=True)
class FieldCalibration:
    """
    Phenomenological calibration for external-field effects on vortex dynamics.

    Parameters are meant to be **fitted from micromagnetic simulations** (e.g.
    MuMax3) rather than calculated from first principles and capture how the
    gyrotropic mode responds to the applied field.

    Parameters
    ----------
    domega0_dBz : float
        Linear slope dω₀/dBz [rad/(s·T)].  The effective shift is
        ``p · domega0_dBz · Bz`` (polarity-dependent).  Fitted from
        ω₀(Bz) sweeps in MuMax3.
    seq_per_T : float
        Equilibrium-position shift of the normalized core coordinate per unit
        in-plane field [1/T].  Maps |B_∥| to |s_eq| via
        ``s_eq = chirality · seq_per_T · ẑ × B_∥``.
    chirality : int
        Sign convention for the in-plane equilibrium shift direction (±1).
        Depends on vortex chirality and polarity convention used in the
        simulation.
    """

    domega0_dBz: float = 0.0
    seq_per_T: float = 0.0
    chirality: int = 1

    def omega0_shift(
        self, *, field_state: ExternalField, polarity: int
    ) -> float:
        """Polarity-dependent Bz → ω₀ shift:  Δω₀ = p · (dω₀/dBz) · Bz."""
        p = 1 if int(polarity) >= 0 else -1
        return p * self.domega0_dBz * field_state.Bz_T

    def s_eq(
        self, *, field_state: ExternalField
    ) -> tuple[float, float]:
        """In-plane equilibrium core shift (normalised coords).

        Returns ``(sx_eq, sy_eq) = c · χ_ip · (ẑ × B_∥)``.
        """
        c = 1 if int(self.chirality) >= 0 else -1
        chi = float(self.seq_per_T)
        return (
            c * chi * (-field_state.By_T),
            c * chi * field_state.Bx_T,
        )


# ---------------------------------------------------------------------------
# External field waveform helpers
# ---------------------------------------------------------------------------

#: Callable ``(t) → ExternalField`` or coercible value.
ExternalFieldLike: TypeAlias = float | tuple[float, float, float] | ExternalField | np.ndarray

#: A function returning an :class:`ExternalField` (or coercible) at time *t*.
FieldFunc: TypeAlias = Callable[[float], ExternalFieldLike]


def field_dc(B_ext: ExternalFieldLike = 0.0) -> FieldFunc:
    """Constant external field ``B(t) = const``."""
    B = ExternalField.from_any(B_ext)

    def _b(t: float) -> ExternalField:  # noqa: ARG001
        return B

    return _b


def field_ac(
    B_amp: ExternalFieldLike,
    f_hz: float,
    *,
    B_offset: ExternalFieldLike = 0.0,
    phase: float = 0.0,
) -> FieldFunc:
    """Sinusoidal field:  ``B(t) = B_offset + B_amp · sin(2π f t + φ)``."""
    amp = ExternalField.from_any(B_amp)
    off = ExternalField.from_any(B_offset)
    omega = 2.0 * math.pi * float(f_hz)

    def _b(t: float) -> ExternalField:
        s = math.sin(omega * t + float(phase))
        return ExternalField(
            off.Bx_T + amp.Bx_T * s,
            off.By_T + amp.By_T * s,
            off.Bz_T + amp.Bz_T * s,
        )

    return _b


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class ThieleTrajectoryResult(AnalyticalResult):
    """
    Result of Thiele-equation trajectory integration.

    Attributes
    ----------
    t : np.ndarray
        Time array [s].
    x : np.ndarray
        Core X position [m].
    y : np.ndarray
        Core Y position [m].
    sx : np.ndarray
        Normalized core X position (``x / R``).
    sy : np.ndarray
        Normalized core Y position (``y / R``).
    disk_radius : float
        Disk radius [m] (for context).
    """

    t: np.ndarray = field(default_factory=lambda: np.array([]))
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    sx: np.ndarray = field(default_factory=lambda: np.array([]))
    sy: np.ndarray = field(default_factory=lambda: np.array([]))
    disk_radius: float = 0.0

    # ── derived properties ─────────────────────────────────────

    @property
    def z(self) -> np.ndarray:
        """Complex trajectory z(t) = x + i·y [m]."""
        return self.x + 1j * self.y

    @property
    def r(self) -> np.ndarray:
        """Radial distance from disk centre [m]."""
        return np.sqrt(self.x**2 + self.y**2)

    @property
    def u(self) -> np.ndarray:
        """Normalized radius u = r/R ∈ [0, 1)."""
        return self.r / self.disk_radius if self.disk_radius > 0 else self.r

    @property
    def phi(self) -> np.ndarray:
        """Azimuthal angle φ(t) = arg(z) [rad]."""
        return np.angle(self.z)

    @property
    def phi_unwrapped(self) -> np.ndarray:
        """Unwrapped azimuthal angle [rad]."""
        return np.unwrap(self.phi)

    @property
    def instantaneous_frequency(self) -> np.ndarray:
        """Instantaneous angular frequency ω(t) = dφ/dt [rad/s]."""
        return np.gradient(self.phi_unwrapped, self.t)

    @property
    def instantaneous_frequency_ghz(self) -> np.ndarray:
        """Instantaneous frequency in GHz."""
        return self.instantaneous_frequency / (2.0 * math.pi * 1e9)

    @property
    def velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Velocity (vx, vy) [m/s] via numeric differentiation."""
        vx = np.gradient(self.x, self.t)
        vy = np.gradient(self.y, self.t)
        return vx, vy

    @property
    def speed(self) -> np.ndarray:
        """Speed |v| [m/s]."""
        vx, vy = self.velocity
        return np.sqrt(vx**2 + vy**2)

    @property
    def rotation_sense(self) -> str:
        """``'CCW'`` or ``'CW'`` from mean ω(t)."""
        return "CCW" if np.mean(self.instantaneous_frequency) > 0 else "CW"

    @property
    def steady_state_radius_m(self) -> float:
        """Mean orbital radius in the last 20% of the trajectory [m]."""
        n = max(1, len(self.t) // 5)
        return float(np.mean(self.r[-n:]))

    @property
    def steady_state_frequency_ghz(self) -> float:
        """Mean frequency in the last 20% of the trajectory [GHz]."""
        n = max(1, len(self.t) // 5)
        return float(np.mean(self.instantaneous_frequency_ghz[-n:]))

    # ── spectrum properties ────────────────────────────────────

    @property
    def _spectrum_cache(self) -> tuple[np.ndarray, np.ndarray]:
        """Cached (frequencies_hz, power) from windowed FFT of x(t)."""
        cache_attr = "__spectrum_cache"
        if not hasattr(self, cache_attr) or getattr(self, cache_attr) is None:
            n = len(self.t)
            if n < 4:
                result = (np.array([]), np.array([]))
            else:
                dt = float(self.t[1] - self.t[0]) if n > 1 else 1e-11
                # Use Hann window for spectral leakage reduction
                window = np.hanning(n)
                sig = self.x - np.mean(self.x)
                sig_windowed = sig * window
                fft_vals = np.fft.rfft(sig_windowed)
                freqs = np.fft.rfftfreq(n, d=dt)
                # Normalised one-sided power spectrum
                power = (np.abs(fft_vals) ** 2) / max(float(n), 1.0)
                # Compensate for Hann window energy loss
                power *= 2.0 / max(float(np.mean(window**2)), 1e-30)
                result = (freqs, power)
            object.__setattr__(self, cache_attr, result)
        return getattr(self, cache_attr)

    @property
    def spectrum_frequencies_ghz(self) -> np.ndarray:
        """Frequency axis of the power spectrum [GHz]."""
        freqs, _ = self._spectrum_cache
        return freqs * 1e-9

    @property
    def power_spectrum(self) -> np.ndarray:
        """Power spectrum |FFT(x(t))|² (Hann-windowed, normalised)."""
        _, power = self._spectrum_cache
        return power

    @property
    def dominant_frequency_ghz(self) -> float:
        """Peak frequency from the power spectrum [GHz]."""
        freqs, power = self._spectrum_cache
        if power.size == 0:
            return 0.0
        # Ignore DC bin
        start = 1 if power.size > 1 else 0
        idx = int(np.argmax(power[start:])) + start
        return float(freqs[idx]) * 1e-9

    @property
    def linewidth_ghz(self) -> float:
        """Estimated linewidth (FWHM) from the power spectrum [GHz]."""
        freqs, power = self._spectrum_cache
        if power.size < 3:
            return 0.0
        start = 1 if power.size > 1 else 0
        peak_idx = int(np.argmax(power[start:])) + start
        half_max = power[peak_idx] / 2.0
        # Find half-power crossings around the peak
        above = power >= half_max
        # left edge
        left = peak_idx
        while left > start and above[left]:
            left -= 1
        # right edge
        right = peak_idx
        while right < len(power) - 1 and above[right]:
            right += 1
        df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
        fwhm = float(right - left) * df
        return fwhm * 1e-9

    # ── plotting ───────────────────────────────────────────────

    @property
    def plt(self) -> ThielePlotAccessor:
        """Access plotting methods via fluent API."""
        return ThielePlotAccessor(self)

    # ── repr ───────────────────────────────────────────────────

    def __repr__(self) -> str:
        n = len(self.t)
        r_ss = self.steady_state_radius_m * 1e9 if n > 0 else 0.0
        f_ss = self.steady_state_frequency_ghz if n > 0 else 0.0
        return (
            f"ThieleTrajectoryResult(model={self.model_name!r}, "
            f"n_points={n}, "
            f"R_ss≈{r_ss:.1f} nm, "
            f"f_ss≈{f_ss:.2f} GHz)"
        )

    def _repr_html_(self) -> str:
        from html import escape as _esc

        n = len(self.t)
        r_ss = self.steady_state_radius_m * 1e9 if n > 0 else 0.0
        f_ss = self.steady_state_frequency_ghz if n > 0 else 0.0
        t_ns = self.t[-1] * 1e9 if n > 0 else 0.0
        rot = self.rotation_sense if n > 0 else "N/A"
        model = _esc(str(self.model_name))

        summary_items = [
            ("Points", str(n)),
            ("Duration", f"{t_ns:.1f} ns"),
            ("Steady-state radius", f"{r_ss:.1f} nm"),
            ("Steady-state freq", f"{f_ss:.3f} GHz"),
            ("Rotation", str(rot)),
        ]
        summary_row = "".join(
            f"<div><span style='color:#94a3b8;'>{k}:</span> "
            f"<span style='color:#cbd5e1;'>{v}</span></div>"
            for k, v in summary_items
        )

        properties = [
            (".z", "Complex trajectory z(t) = x + i·y [m]"),
            (".r", "Radial distance from disk centre [m]"),
            (".u", "Normalized radius u = r/R ∈ [0, 1)"),
            (".phi", "Azimuthal angle φ(t) [rad]"),
            (".phi_unwrapped", "Unwrapped azimuthal angle [rad]"),
            (".velocity", "Velocity (vx, vy) [m/s]"),
            (".speed", "Speed |v| [m/s]"),
            (".instantaneous_frequency_ghz", "Instantaneous frequency [GHz]"),
            (".dominant_frequency_ghz", "Peak frequency from power spectrum [GHz]"),
            (".linewidth_ghz", "Estimated linewidth (FWHM) [GHz]"),
            (".power_spectrum", "Power spectrum |FFT(x(t))|²"),
        ]
        prop_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(p)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(d)}</td></tr>"
            for p, d in properties
        )
        plot_methods = [
            (".plt.trajectory()", "Plot x,y trajectory"),
            (".plt.spectrum()", "Plot power spectrum"),
            (".plt.radius()", "Plot radius vs time"),
            (".plt.phase()", "Plot azimuthal angle vs time"),
        ]
        plot_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(m)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(d)}</td></tr>"
            for m, d in plot_methods
        )
        example = (
            "# Inspect trajectory\n"
            "result.plt.trajectory()\n"
            "result.plt.spectrum()\n"
            "\n"
            "# Access properties\n"
            "print(f'Frequency: {result.dominant_frequency_ghz:.3f} GHz')\n"
            "print(f'Linewidth: {result.linewidth_ghz:.3f} GHz')\n"
            "print(f'Rotation: {result.rotation_sense}')"
        )
        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);\">"
            f"<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;margin-bottom:4px;'>"
            f"🌀 Thiele Trajectory — {model}</div>"
            "<div style='font-size:0.85em;color:#94a3b8;margin-bottom:10px;'>"
            "Vortex core trajectory from Thiele equation integration</div>"
            # Summary
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Results</div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:12px;font-size:0.9em;'>"
            f"{summary_row}</div></div>"
            # Properties
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Properties</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            "<thead><tr style='text-align:left;background:rgba(51,65,85,0.6);'>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Accessor</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Description</th></tr></thead>"
            f"<tbody>{prop_rows}</tbody></table></div>"
            # Plotting
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Plotting</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            f"{plot_rows}</table></div>"
            # Examples
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Examples</div>"
            "<pre style='margin:0;background:rgba(15,23,42,0.85);padding:10px;"
            "border-radius:6px;color:#e2e8f0;overflow-x:auto;font-size:0.85em;'>"
            f"<code>{example}</code></pre></div>"
            "</div>"
        )


@dataclass
class ThieleFJFitResult(AnalyticalResult):
    """Result of fitting ``omega0`` and ``N`` to measured ``f(J)`` data."""

    omega0: float = float("nan")
    N: float = float("nan")
    omega0_Oe_per_J: float = 0.0
    chi_scale: float = 1.0
    J_data: np.ndarray = field(default_factory=lambda: np.array([]))
    f_data_hz: np.ndarray = field(default_factory=lambda: np.array([]))
    f_fit_hz: np.ndarray = field(default_factory=lambda: np.array([]))
    valid_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    rmse_hz: float = float("nan")
    success: bool = False
    status: str = "unknown"

    @property
    def f_data_ghz(self) -> np.ndarray:
        """Measured frequencies in GHz."""
        return np.asarray(self.f_data_hz, dtype=float) * 1e-9

    @property
    def f_fit_ghz(self) -> np.ndarray:
        """Model-fit frequencies in GHz."""
        return np.asarray(self.f_fit_hz, dtype=float) * 1e-9

    @property
    def plt(self) -> ThieleFJFitPlotAccessor:
        """Access plotting methods for ``f(J)`` fit."""
        return ThieleFJFitPlotAccessor(self)


@dataclass
class ThieleOptimizationResult(AnalyticalResult):
    """Result of single-objective current optimization for CPP Thiele model."""

    target_frequency_hz: float = float("nan")
    current_density_a_per_m2: float = float("nan")
    predicted_frequency_hz: float = float("nan")
    objective_value_hz: float = float("nan")
    success: bool = False
    status: str = "unknown"
    J_bounds: tuple[float, float] = (float("nan"), float("nan"))

    @property
    def current_density_ga_per_m2(self) -> float:
        """Current density in GA/m²."""
        return float(self.current_density_a_per_m2) * 1e-9

    @property
    def predicted_frequency_ghz(self) -> float:
        """Predicted frequency in GHz."""
        return float(self.predicted_frequency_hz) * 1e-9


class ThieleFJFitPlotAccessor:
    """Plot helpers for :class:`ThieleFJFitResult`."""

    def __init__(self, result: ThieleFJFitResult):
        self._result = result

    def frequency_vs_current(
        self,
        ax=None,
        *,
        current_unit: Literal["A/m2", "GA/m2"] = "GA/m2",
        frequency_unit: Literal["Hz", "GHz"] = "GHz",
        show: bool = False,
        **kwargs,
    ):
        """Plot measured and fitted ``f(J)`` data."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))

        current = np.asarray(self._result.J_data, dtype=float)
        if current_unit == "GA/m2":
            current = current * 1e-9
            xlabel = "Current density [GA/m²]"
        else:
            xlabel = "Current density [A/m²]"

        f_data = np.asarray(self._result.f_data_hz, dtype=float)
        f_fit = np.asarray(self._result.f_fit_hz, dtype=float)
        if frequency_unit == "GHz":
            f_data = f_data * 1e-9
            f_fit = f_fit * 1e-9
            ylabel = "Frequency [GHz]"
        else:
            ylabel = "Frequency [Hz]"

        ax.scatter(current, f_data, label="data", color="tab:blue", s=28)
        ax.plot(current, f_fit, label="fit", color="tab:orange", **kwargs)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        title = (
            f"Thiele fit: ω0={self._result.omega0:.3e} rad/s, "
            f"N={self._result.N:.3f}"
        )
        if abs(self._result.chi_scale - 1.0) > 0.01:
            title += f", χ_scale={self._result.chi_scale:.2f}"
        title += f", RMSE={self._result.rmse_hz:.3e} Hz"
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

        if show:
            plt.show()
        return ax


# ---------------------------------------------------------------------------
# Current waveform helpers
# ---------------------------------------------------------------------------


def ellipse_area(size_x: float, size_y: float) -> float:
    """
    Elliptic pillar area ``A = π·size_x·size_y/4``.

    Parameters
    ----------
    size_x : float
        Full ellipse size along x [m].
    size_y : float
        Full ellipse size along y [m].
    """
    if size_x <= 0.0 or size_y <= 0.0:
        raise ValueError("size_x and size_y must be positive")
    return math.pi * float(size_x) * float(size_y) / 4.0


def slonczewski_mtj_efficiency(
    Pol: float,
    Lambda: float,
    cos_theta: float,
) -> float:
    """
    Effective Slonczewski MTJ efficiency ``epsilon(theta)``.

    Uses:
    ``epsilon = Pol * Lambda^2 / ((Lambda^2 + 1) + (Lambda^2 - 1) * cos_theta)``.
    """
    lam = float(Lambda)
    if lam <= 0.0:
        raise ValueError("Lambda must be positive")
    cth = float(np.clip(cos_theta, -1.0, 1.0))
    lam2 = lam * lam
    denom = (lam2 + 1.0) + (lam2 - 1.0) * cth
    if abs(denom) < 1e-30:
        raise ValueError("Degenerate denominator in Slonczewski efficiency")
    return float(Pol) * lam2 / denom


def current_dc(J_dc: float) -> Callable[[float], float]:
    """
    Constant (DC) current density.

    Parameters
    ----------
    J_dc : float
        Current density [A/m²].

    Returns
    -------
    callable
        ``J_func(t) -> J_dc``
    """

    def _j(t: float) -> float:  # noqa: ARG001
        return J_dc

    return _j


def current_ac(
    J_amp: float,
    f_hz: float,
    J_offset: float = 0.0,
    phase: float = 0.0,
) -> Callable[[float], float]:
    """
    Sinusoidal (AC) current density, optionally with DC offset.

    Parameters
    ----------
    J_amp : float
        AC amplitude [A/m²].
    f_hz : float
        Frequency [Hz].
    J_offset : float, optional
        DC offset [A/m²].
    phase : float, optional
        Phase offset [rad].

    Returns
    -------
    callable
        ``J_func(t) -> J_offset + J_amp · sin(2πf·t + phase)``
    """
    omega = 2.0 * math.pi * f_hz

    def _j(t: float) -> float:
        return J_offset + J_amp * math.sin(omega * t + phase)

    return _j


def current_pulse(
    J_on: float,
    t_on: float,
    t_off: float,
    J_base: float = 0.0,
) -> Callable[[float], float]:
    """
    Rectangular current pulse.

    Parameters
    ----------
    J_on : float
        Pulse current density [A/m²].
    t_on : float
        Pulse start time [s].
    t_off : float
        Pulse end time [s].
    J_base : float, optional
        Baseline current density [A/m²].

    Returns
    -------
    callable
        ``J_func(t) -> J_on`` if ``t_on ≤ t < t_off``, else ``J_base``
    """

    def _j(t: float) -> float:
        return J_on if t_on <= t < t_off else J_base

    return _j


# ---------------------------------------------------------------------------
# Gyrotropic frequency estimate
# ---------------------------------------------------------------------------


def omega0_novosad(mat: MaterialParams, geo: DiskGeometry) -> float:
    r"""
    Estimate the gyrotropic eigenfrequency ω₀ for a vortex in a thin disk.

    Uses the two-vortex (rigid-vortex) analytical result of
    Guslienko & Metlov / Novosad et al. for the translational mode:

    .. math::

        \omega_0 = \frac{10}{9} \gamma_0 M_s \frac{L}{R}
                   F_1\!\left(\frac{L}{R}\right)

    where :math:`F_1(\beta) \approx 1 - 3\beta/(8\pi)` for thin disks
    and :math:`\gamma_0 = \gamma \mu_0`.

    Parameters
    ----------
    mat : MaterialParams
        Material parameters (needs ``Ms``, ``gamma``).
    geo : DiskGeometry
        Disk geometry (needs ``R``, ``L``).

    Returns
    -------
    float
        Angular frequency ω₀ [rad/s].

    References
    ----------
    V. Novosad et al., arXiv:cond-mat/0503632;
    K.Y. Guslienko et al., J. Appl. Phys. 91, 8037 (2002).
    """
    beta = geo.L / geo.R
    # F_1(β) ≈ 1 − (3/8π)β  for thin disks  (leading-order)
    F1 = 1.0 - 3.0 * beta / (8.0 * math.pi)
    if F1 < 0.1:
        F1 = 0.1  # guard for thick disks where expansion breaks down

    gamma0 = mat.gamma * MU0  # rad/(s·T) → rad·m/(s·A)
    omega0 = (10.0 / 9.0) * gamma0 * mat.Ms * (geo.L / geo.R) * F1
    return omega0


def f0_novosad_ghz(mat: MaterialParams, geo: DiskGeometry) -> float:
    """Gyrotropic eigenfrequency in GHz.  Convenience wrapper around :func:`omega0_novosad`."""
    return omega0_novosad(mat, geo) / (2.0 * math.pi * 1e9)


# ---------------------------------------------------------------------------
# MODEL A:  CIP — Current-In-Plane Thiele + Zhang–Li STT
# ---------------------------------------------------------------------------


class CIPThieleModel:
    r"""
    Thiele equation integrator for a vortex driven by **in-plane current** (CIP).

    The ODE follows Moon et al. (arXiv:0809.0952):

    .. math::

        \mathbf{G}(p)\times(\mathbf{u} - \dot{\mathbf{r}})
        = -\nabla U(\mathbf{r})
          - \alpha\,\mathbf{D}\,\dot{\mathbf{r}}
          + \beta\,\mathbf{D}\,\mathbf{u}

    where **u** is the spin-drift velocity proportional to current density.

    Parameters
    ----------
    material : MaterialParams
        Material constants.
    geom : DiskGeometry
        Disk geometry.
    omega0 : float
        Gyrotropic eigenfrequency ω₀ [rad/s].
        Can be obtained from :func:`omega0_novosad` or micromagnetic calibration.
    polarity : int
        Core polarity *p* = +1 or −1.
    current_dir : tuple[float, float]
        Unit vector for in-plane current direction (e.g. ``(1, 0)`` for x).

    Notes
    -----
    The potential is harmonic: :math:`U(\mathbf{r}) = \frac12 \kappa |\mathbf{r}|^2`,
    with :math:`\kappa = \omega_0 G_0` giving the Thiele spring constant.

    The dissipation ratio is approximated as:

    .. math::

        \frac{D}{G_0} \approx \frac12 \ln\!\left(\frac{R}{\delta}\right)

    where δ is the core diameter.  This enters as the effective damping for the
    linear ODE.

    References
    ----------
    J.-H. Moon et al., arXiv:0809.0952.
    """

    def __init__(
        self,
        material: MaterialParams,
        geom: DiskGeometry,
        omega0: float,
        polarity: int = 1,
        current_dir: tuple[float, float] = (1.0, 0.0),
        field: ExternalField | None = None,
        field_cal: FieldCalibration | None = None,
    ) -> None:
        self.material = material
        self.geom = geom
        self.omega0 = omega0
        self.polarity = int(polarity)
        self.field = field if field is not None else ExternalField()
        self.field_cal = field_cal if field_cal is not None else FieldCalibration()
        assert self.polarity in (1, -1), "polarity must be +1 or -1"

        # normalise current direction
        cx, cy = current_dir
        norm = math.sqrt(cx**2 + cy**2)
        self.current_dir = (cx / norm, cy / norm)

        # derived quantities
        self._setup()

    def _setup(self) -> None:
        mat = self.material
        geo = self.geom
        p = self.polarity

        # Spin-drift velocity prefactor: u₀ = μ_B P / (e Ms)  [m³/(A·s)]
        self._u0_prefactor = _MU_B * mat.P / (_E_CHARGE * mat.Ms)

        # Core diameter δ (≈ core_diameter or 2·exchange_length)
        delta = geo.core_diameter if geo.core_diameter else 2.0 * mat.exchange_length

        # D/G_0 ≈ (1/2) ln(R/δ)  (Moon et al.)
        ratio = geo.R / max(delta, 1e-10)
        self._d_over_G0 = 0.5 * math.log(max(ratio, 1.1))

        # Effective constants for the linear ODE
        # ω₀ already given; κ = ω₀·G₀ (but G₀ cancels in the ODE)
        # After dividing through by G₀ the ODE becomes:
        #   p ẑ×(u − ṙ) = −ω₀ r − α (D/G₀) ṙ + β (D/G₀) u
        # Rearranging for ṙ gives a 2×2 linear system per time-step.
        self._alpha = mat.alpha
        self._beta = mat.beta
        self._dG = self._d_over_G0
        self._p = p
        # Base ω₀ stored; Bz shift applied dynamically in _rhs via field_cal
        self._omega0_base = float(self.omega0)

    def _field_at(self, t: float, B_func: FieldFunc | None) -> ExternalField:
        """Resolve field at time *t* (static fallback when ``B_func`` is None)."""
        if B_func is None:
            return self.field
        return ExternalField.from_any(B_func(float(t)))

    def _rhs(
        self,
        t: float,
        state: np.ndarray,
        J_func: Callable,
        B_func: FieldFunc | None,
    ) -> np.ndarray:
        """Right-hand side of the CIP Thiele ODE for solve_ivp."""
        X, Y = state
        J = J_func(t)
        u0 = self._u0_prefactor * J
        ux = u0 * self.current_dir[0]
        uy = u0 * self.current_dir[1]

        # Resolve field (possibly time-dependent)
        B = self._field_at(t, B_func)
        w0 = self._omega0_base + self.field_cal.omega0_shift(
            field_state=B, polarity=self._p
        )
        # In-plane equilibrium shift (in real-space coords)
        sx_eq, sy_eq = self.field_cal.s_eq(field_state=B)
        X_eq = sx_eq * self.geom.R
        Y_eq = sy_eq * self.geom.R
        Xr = X - X_eq
        Yr = Y - Y_eq

        p = self._p
        alpha = self._alpha
        beta = self._beta
        dG = self._dG

        # Moon ODE (see docstring algebra) using relative coordinates
        det = (alpha * dG) ** 2 + p**2  # always > 0
        rhs_I = -w0 * Xr + p * uy + beta * dG * ux
        rhs_II = -w0 * Yr - p * ux + beta * dG * uy

        dXdt = (alpha * dG * rhs_I + p * rhs_II) / det
        dYdt = (-p * rhs_I + alpha * dG * rhs_II) / det

        return np.array([dXdt, dYdt])

    def simulate(
        self,
        t_span: tuple[float, float],
        r0: tuple[float, float] = (1e-9, 0.0),
        J_func: Callable[[float], float] | None = None,
        B_func: FieldFunc | None = None,
        dt: float = 1e-12,
        method: str = "RK45",
        **ivp_kwargs: Any,
    ) -> ThieleTrajectoryResult:
        """
        Integrate the CIP Thiele equation.

        Parameters
        ----------
        t_span : (t_start, t_end)
            Integration window [s].
        r0 : (X₀, Y₀)
            Initial core position [m].
        J_func : callable(t) -> float
            Current-density waveform J(t) [A/m²].  If None, J = 0.
        B_func : FieldFunc, optional
            Time-dependent external field waveform ``B(t)``.  If None the
            static ``self.field`` is used.  See :func:`field_dc`, :func:`field_ac`.
        dt : float
            Maximum time-step (and output sampling) [s].
        method : str
            ``scipy.integrate.solve_ivp`` method.  Default ``'RK45'``.
        **ivp_kwargs
            Extra keyword arguments forwarded to ``solve_ivp``.

        Returns
        -------
        ThieleTrajectoryResult
            Trajectory with all derived properties.
        """
        from scipy.integrate import solve_ivp

        if J_func is None:
            J_func = current_dc(0.0)

        t_eval = np.arange(t_span[0], t_span[1], dt)

        sol = solve_ivp(
            fun=lambda t, y: self._rhs(t, y, J_func, B_func),
            t_span=t_span,
            y0=np.array(r0, dtype=float),
            t_eval=t_eval,
            method=method,
            rtol=ivp_kwargs.pop("rtol", 1e-9),
            atol=ivp_kwargs.pop("atol", 1e-12),
            **ivp_kwargs,
        )

        if not sol.success:
            raise RuntimeError(f"CIP Thiele integration failed: {sol.message}")

        t_out = sol.t
        X = sol.y[0]
        Y = sol.y[1]
        R = self.geom.R

        return ThieleTrajectoryResult(
            model_name=f"CIP Thiele (p={self.polarity:+d}, Moon et al.)",
            t=t_out,
            x=X,
            y=Y,
            sx=X / R,
            sy=Y / R,
            disk_radius=R,
            params={
                "Ms": self.material.Ms,
                "alpha": self.material.alpha,
                "P": self.material.P,
                "beta": self.material.beta,
                "omega0": self.omega0,
                "polarity": self.polarity,
                "current_dir": self.current_dir,
                "field": self.field,
                "field_cal": self.field_cal,
            },
            metadata={
                "mode": "CIP",
                "reference": "Moon et al., arXiv:0809.0952",
            },
        )


# ---------------------------------------------------------------------------
# MODEL B:  CPP — Current-Perpendicular-to-Plane (vortex STNO)
# ---------------------------------------------------------------------------


class CPPThieleModel:
    r"""
    Nonlinear Thiele / auto-oscillator model for a **vortex STNO** (CPP geometry).

    Following Guslienko et al. (2014), the normalised core position
    **s** = **r**/R obeys:

    .. math::

        \dot{\mathbf{s}} = \bigl[\chi(J) - d(u)\,\omega(u)\bigr]\,\mathbf{s}
                           + \omega(u)\,(\hat{\mathbf{z}} \times \mathbf{s})

    where *u* = |**s**| ∈ [0, 1).

    Components:

    - **Nonlinear frequency:**  ω(u) = ω₀(1 + N u²)
    - **Nonlinear damping:**    d(u) = d₀ + d₁ u²
    - **STT pumping:**          χ(J) = γ σ J / 2,  σ = ℏP/(2eL Mₛ)

    Parameters
    ----------
    material : MaterialParams
        Material constants.
    geom : DiskGeometry
        Disk geometry.
    omega0 : float
        Linear gyrotropic eigenfrequency ω₀ [rad/s].
    N : float
        Nonlinear frequency shift coefficient (dimensionless).
        Typical range 0.1–0.5; Guslienko reports ~0.2–0.25 for common geometries.
    polarity : int
        Core polarity *p* = ±1.  Determines rotation sense.

    Notes
    -----
    - **d₀** and **d₁** are computed analytically from α and R/Rc
      (Guslienko 2014, Eq. 4–5).
    - The steady-state orbit radius satisfies χ(J) = d(u₀)·ω(u₀).
    - The initial condition **s₀** must be non-zero (the model has no
      thermal noise; the origin is an unstable fixed point above threshold).
    - **Threshold current** J_th is defined by χ(J_th) = d₀·ω₀.

    References
    ----------
    K.Y. Guslienko et al., Phys. Rev. B 89, 044412 (2014) / PMC 4134337.
    """

    def __init__(
        self,
        material: MaterialParams,
        geom: DiskGeometry,
        omega0: float,
        N: float = 0.25,
        polarity: int = 1,
        omega0_Oe_per_J: float = 0.0,
        field: ExternalField | None = None,
        field_cal: FieldCalibration | None = None,
        chi_scale: float = 1.0,
    ) -> None:
        self.material = material
        self.geom = geom
        self.omega0 = omega0
        self.N = N
        self.polarity = int(polarity)
        self.omega0_Oe_per_J = float(omega0_Oe_per_J)
        self.field = field if field is not None else ExternalField()
        self.field_cal = field_cal if field_cal is not None else FieldCalibration()
        self.chi_scale = float(chi_scale)
        assert self.polarity in (1, -1), "polarity must be +1 or -1"

        self._setup()

    def _setup(self) -> None:
        mat = self.material
        geo = self.geom
        Rc = geo.Rc(mat)

        # Slonczewski coefficient:  σ = ℏ P / (2 e L Ms)  [m²/(A·s)… sort of]
        self._sigma = _HBAR * mat.P / (2.0 * _E_CHARGE * geo.L * mat.Ms)

        # d₀ = α · [5 + 4 ln(R/Rc)] / 8
        ratio = geo.R / max(Rc, 1e-10)
        self._d0 = mat.alpha * (5.0 + 4.0 * math.log(max(ratio, 1.1))) / 8.0

        # d₁ = (11/6) α
        self._d1 = (11.0 / 6.0) * mat.alpha

        # χ(J) = γ σ J / 2
        self._chi_prefactor = mat.gamma * self._sigma / 2.0

    # ── public helpers ─────────────────────────────────────────

    def chi(self, J: float) -> float:
        """STT pumping rate χ(J) [rad/s], scaled by chi_scale."""
        return self.chi_scale * self._chi_prefactor * J

    def d(self, u: float) -> float:
        """Nonlinear damping d(u) [dimensionless]."""
        return self._d0 + self._d1 * u**2

    def omega0_eff(self, J: float, field_state: ExternalField | None = None) -> float:
        """
        Effective linear frequency [rad/s].

        ω₀_eff(J, Bz) = ω₀ + (dω₀/dJ)·J + p·(dω₀/dBz)·Bz

        The Bz dependence is **phenomenological** and must be calibrated
        from micromagnetic simulations (``field_cal.domega0_dBz``).
        The polarity factor ``p`` is applied automatically.
        """
        B = field_state if field_state is not None else self.field
        return (
            self.omega0
            + self.omega0_Oe_per_J * float(J)
            + self.field_cal.omega0_shift(field_state=B, polarity=self.polarity)
        )

    def _field_at(self, t: float, B_func: FieldFunc | None) -> ExternalField:
        """Resolve field at time *t* (static fallback when ``B_func`` is None)."""
        if B_func is None:
            return self.field
        return ExternalField.from_any(B_func(float(t)))

    def s_eq(self, field_state: ExternalField | None = None) -> np.ndarray:
        """
        Equilibrium position of the vortex core in normalized coordinates.

        An in-plane field (Bx, By) shifts the equilibrium via:
            s_eq = chirality · seq_per_T · (ẑ × B_∥) = chirality · seq_per_T · (-By, Bx)

        The coefficient ``seq_per_T`` [1/T] should be calibrated from
        micromagnetic simulations.
        """
        B = field_state if field_state is not None else self.field
        sx, sy = self.field_cal.s_eq(field_state=B)
        return np.array([sx, sy], dtype=float)

    def omega(self, u: float, J: float = 0.0) -> float:
        """Nonlinear gyrotropic frequency ω(u, J) [rad/s]."""
        return self.omega0_eff(J) * (1.0 + self.N * u**2)

    @property
    def J_threshold(self) -> float:
        """Threshold current density for self-oscillation [A/m²]."""
        # χ(J_th) = d₀ · ω₀_eff(0)  →  J_th = 2 d₀ ω₀_eff / (γ σ chi_scale)
        return self._d0 * self.omega0_eff(0.0) / (self.chi_scale * self._chi_prefactor)

    def threshold_current_dc(self) -> float:
        """Threshold DC current density for auto-oscillation [A/m²]."""
        return self.J_threshold

    def predict_frequency_dc(
        self,
        J_dc: float,
        *,
        allow_edge: bool = False,
        omega0_Oe_per_J: float | None = None,
    ) -> float | None:
        """
        Predict steady-state gyration frequency for DC current.

        Parameters
        ----------
        J_dc : float
            DC current density [A/m²].
        allow_edge : bool
            If True, return edge-clamped frequency when u₀ ≥ u_stop.
        omega0_Oe_per_J : float, optional
            Override the model's ``omega0_Oe_per_J`` for this call.

        Returns
        -------
        float | None
            Frequency [Hz], or ``None`` below threshold / edge-clamped (if disallowed).
        """
        old = self.omega0_Oe_per_J
        if omega0_Oe_per_J is not None:
            self.omega0_Oe_per_J = float(omega0_Oe_per_J)
        try:
            u0 = self.steady_state_u(float(J_dc), allow_edge=allow_edge)
            if u0 is None:
                return None
            omega_eff = self.omega(u0, float(J_dc))
            return float(omega_eff / (2.0 * math.pi))
        finally:
            self.omega0_Oe_per_J = old

    def optimize_current_for_target_frequency(
        self,
        target_frequency_hz: float,
        *,
        J_bounds: tuple[float, float] | None = None,
        allow_edge: bool = False,
        n_grid: int = 300,
    ) -> ThieleOptimizationResult:
        """
        Optimize DC current density to match target gyration frequency.

        Uses the model's ``omega0_Oe_per_J`` attribute for Oersted correction.
        """
        target = float(target_frequency_hz)
        if not np.isfinite(target) or target <= 0.0:
            raise ValueError("target_frequency_hz must be a positive finite value")

        if J_bounds is None:
            j_min = 1.01 * self.threshold_current_dc()
            j_max = 6.0 * self.threshold_current_dc()
        else:
            j_min, j_max = float(J_bounds[0]), float(J_bounds[1])
        if not (np.isfinite(j_min) and np.isfinite(j_max) and j_max > j_min):
            raise ValueError("J_bounds must satisfy finite values with j_max > j_min")

        def _loss(jval: float) -> float:
            freq = self.predict_frequency_dc(
                float(jval),
                allow_edge=allow_edge,
            )
            if freq is None or not np.isfinite(freq):
                return 1e30
            return abs(float(freq) - target)

        best_j = j_min
        best_loss = _loss(best_j)
        success = False
        status = "grid"

        try:
            from scipy.optimize import minimize_scalar

            opt = minimize_scalar(_loss, bounds=(j_min, j_max), method="bounded")
            if opt.success and np.isfinite(opt.fun):
                best_j = float(opt.x)
                best_loss = float(opt.fun)
                success = True
                status = str(opt.message)
            else:
                status = f"scipy_failed: {opt.message}"
        except Exception:
            grid = np.linspace(j_min, j_max, max(int(n_grid), 32))
            for jval in grid:
                loss = _loss(float(jval))
                if loss < best_loss:
                    best_loss = loss
                    best_j = float(jval)

        best_freq = self.predict_frequency_dc(
            best_j,
            allow_edge=allow_edge,
        )
        pred_freq = float(best_freq) if best_freq is not None else float("nan")

        return ThieleOptimizationResult(
            model_name="CPP Thiele J optimization",
            target_frequency_hz=target,
            current_density_a_per_m2=best_j,
            predicted_frequency_hz=pred_freq,
            objective_value_hz=best_loss,
            success=bool(success),
            status=status,
            J_bounds=(j_min, j_max),
            params={
                "allow_edge": bool(allow_edge),
                "omega0_Oe_per_J": self.omega0_Oe_per_J,
                "n_grid": int(max(int(n_grid), 32)),
            },
        )

    def steady_state_u(
        self, J: float, *, allow_edge: bool = False, u_stop: float = 0.98,
    ) -> float | None:
        """
        Analytical steady-state normalised radius u₀ for DC current J.

        Solves χ(J) = d(u₀)·ω(u₀, J).  Returns ``None`` if J < J_th.
        If ``allow_edge=True`` and the solution exceeds ``u_stop``, returns
        ``u_stop`` (edge-limited orbit) instead of ``None``.
        """
        J = float(J)
        chi_val = float(self.chi(J))
        w0 = float(self.omega0_eff(J))
        if not np.isfinite(chi_val) or not np.isfinite(w0) or w0 <= 0:
            return None

        N = float(self.N)
        d0 = float(self._d0)
        d1 = float(self._d1)

        # χ = (d₀ + d₁ u²) · w₀(1 + N u²)
        # Let x = u²:  d₁·N·x² + (d₁ + d₀·N)·x + (d₀ − χ/w₀) = 0
        c2 = d1 * N
        c1 = d1 + d0 * N
        c0 = d0 - chi_val / w0

        xs: list[float] = []
        if abs(c2) < 1e-30:
            if abs(c1) < 1e-30:
                # No solution from quadratic
                pass
            else:
                xs = [-c0 / c1]
        else:
            disc = c1 * c1 - 4.0 * c2 * c0
            if disc >= 0:
                sdisc = math.sqrt(disc)
                xs = [(-c1 + sdisc) / (2.0 * c2), (-c1 - sdisc) / (2.0 * c2)]

        u_candidates: list[float] = []
        for x in xs:
            if not np.isfinite(x) or x < 0:
                continue
            u = math.sqrt(x)
            if u <= 0:
                continue
            # Stability check: d/du[d·ω] > 0 at u₀
            # For d(u)=d₀+d₁u² and ω(u)=w₀(1+Nu²):
            #   d/du[dω] ∝ (d₁ + N·d₀ + 2·N·d₁·u²)
            if (d1 + N * d0 + 2.0 * N * d1 * u * u) <= 0.0:
                continue
            u_candidates.append(u)

        if not u_candidates:
            # No interior solution — check if pumping wins at edge
            if allow_edge and chi_val > self.d(u_stop) * self.omega(u_stop, J):
                return float(u_stop)
            return None

        u0 = float(min(u_candidates))
        if u0 >= u_stop:
            return float(u_stop) if allow_edge else None
        return u0

    # ── ODE integration ────────────────────────────────────────

    def _rhs(
        self,
        t: float,
        state: np.ndarray,
        J_func: Callable,
        B_func: FieldFunc | None,
    ) -> np.ndarray:
        """Right-hand side for solve_ivp (with in-plane field equilibrium shift)."""
        s = np.asarray(state, dtype=float)

        # Resolve (possibly time-dependent) field
        B = self._field_at(t, B_func)
        s_eq = self.s_eq(field_state=B)
        s_rel = s - s_eq
        u = float(np.linalg.norm(s_rel))
        u = max(u, 1e-15)  # avoid division by zero

        J = float(J_func(t))
        chi_val = self.chi(J)
        omega_val = self.omega0_eff(J, field_state=B) * (1.0 + self.N * u**2)
        radial = chi_val - self.d(u) * omega_val
        p = self.polarity

        dsx = radial * s_rel[0] - p * omega_val * s_rel[1]
        dsy = radial * s_rel[1] + p * omega_val * s_rel[0]

        return np.array([dsx, dsy])

    def simulate(
        self,
        t_span: tuple[float, float],
        s0: tuple[float, float] = (1e-3, 0.0),
        J_func: Callable[[float], float] | None = None,
        B_func: FieldFunc | None = None,
        dt: float = 1e-11,
        method: str = "RK45",
        **ivp_kwargs: Any,
    ) -> ThieleTrajectoryResult:
        """
        Integrate the CPP nonlinear Thiele equation.

        Parameters
        ----------
        t_span : (t_start, t_end)
            Integration window [s].
        s0 : (sₓ, sᵧ)
            Initial normalised core position r₀/R.  Must be > 0
            (the model has no noise to kick it out of the origin).
        J_func : callable(t) -> float
            Current-density waveform J(t) [A/m²].  Default: J = 0.
        B_func : FieldFunc, optional
            Time-dependent external field waveform ``B(t)``.  If None the
            static ``self.field`` is used.  See :func:`field_dc`, :func:`field_ac`.
        dt : float
            Maximum time-step / output sampling [s].
        method : str
            ``solve_ivp`` method.  Default ``'RK45'``.
        **ivp_kwargs
            Extra keyword arguments forwarded to ``solve_ivp``.

        Returns
        -------
        ThieleTrajectoryResult
            Trajectory (in both physical [m] and normalized units).
        """
        from scipy.integrate import solve_ivp

        if J_func is None:
            J_func = current_dc(0.0)

        t_eval = np.arange(t_span[0], t_span[1] + 0.5 * dt, dt)

        sol = solve_ivp(
            fun=lambda t, y: self._rhs(t, y, J_func, B_func),
            t_span=t_span,
            y0=np.array(s0, dtype=float),
            t_eval=t_eval,
            method=method,
            max_step=ivp_kwargs.pop("max_step", dt),
            rtol=ivp_kwargs.pop("rtol", 1e-9),
            atol=ivp_kwargs.pop("atol", 1e-14),
            **ivp_kwargs,
        )

        if not sol.success:
            raise RuntimeError(f"CPP Thiele integration failed: {sol.message}")

        t_out = sol.t
        SX = sol.y[0]
        SY = sol.y[1]
        R = self.geom.R

        return ThieleTrajectoryResult(
            model_name=f"CPP Thiele STNO (p={self.polarity:+d}, Guslienko 2014)",
            t=t_out,
            x=SX * R,
            y=SY * R,
            sx=SX,
            sy=SY,
            disk_radius=R,
            params={
                "Ms": self.material.Ms,
                "alpha": self.material.alpha,
                "P": self.material.P,
                "omega0": self.omega0,
                "N": self.N,
                "d0": self._d0,
                "d1": self._d1,
                "sigma": self._sigma,
                "J_threshold": self.J_threshold,
                "polarity": self.polarity,
                "field": self.field,
                "field_cal": self.field_cal,
            },
            metadata={
                "mode": "CPP",
                "reference": "Guslienko et al., Phys. Rev. B 89 (2014) / PMC 4134337",
            },
        )

    def simulate_sde(
        self,
        t_span: tuple[float, float],
        s0: tuple[float, float] = (0.0, 0.0),
        J_func: Callable[[float], float] | None = None,
        B_func: FieldFunc | None = None,
        dt: float = 1e-11,
        *,
        temperature_k: float = 300.0,
        diffusion: float | None = None,
        noise_scale: float = 1.0,
        seed: int | None = None,
        clamp_u: float = 0.999,
    ) -> ThieleTrajectoryResult:
        """
        Integrate stochastic CPP Thiele equation using Euler-Maruyama.

        The noise term is isotropic in ``(s_x, s_y)`` and can be controlled by:
        - explicit ``diffusion`` [1/s], or
        - heuristic temperature scaling (``temperature_k``, ``noise_scale``).

        Parameters
        ----------
        B_func : FieldFunc, optional
            Time-dependent external field waveform ``B(t)``.  If None the
            static ``self.field`` is used.
        """
        if J_func is None:
            J_func = current_dc(0.0)

        t0, t1 = float(t_span[0]), float(t_span[1])
        if t1 <= t0:
            raise ValueError("t_span must satisfy t_end > t_start")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        t_eval = np.arange(t0, t1, dt, dtype=float)
        if t_eval.size == 0 or t_eval[-1] < t1:
            t_eval = np.append(t_eval, t1)

        if diffusion is None:
            thermal_factor = max(float(temperature_k), 0.0) / 300.0
            base = abs(self.omega0) * max(float(self.material.alpha), 1e-9) * 1e-4
            diffusion_eff = max(float(noise_scale), 0.0) * thermal_factor * base
        else:
            diffusion_eff = max(float(diffusion), 0.0)

        rng = np.random.default_rng(seed)
        sigma = math.sqrt(max(2.0 * diffusion_eff * dt, 0.0))

        state = np.zeros((t_eval.size, 2), dtype=float)
        state[0, :] = np.asarray(s0, dtype=float)

        for idx in range(1, t_eval.size):
            t_prev = float(t_eval[idx - 1])
            prev = state[idx - 1, :]

            # Resolve field at this time-step
            B = self._field_at(t_prev, B_func)
            s_eq = self.s_eq(field_state=B)
            s_eq_norm = float(np.linalg.norm(s_eq))

            # Relative coordinates for gyrotropic dynamics
            s_rel = prev - s_eq
            u_prev = float(np.linalg.norm(s_rel))
            u_eff = max(u_prev, 1e-15)

            j_prev = float(J_func(t_prev))
            chi_val = self.chi(j_prev)
            w0_eff = self.omega0_eff(j_prev, field_state=B)
            omega_val = w0_eff * (1.0 + self.N * u_eff**2)
            radial = chi_val - self.d(u_eff) * omega_val

            theta = float(self.polarity) * omega_val * dt
            grow = math.exp(radial * dt)
            cth = math.cos(theta)
            sth = math.sin(theta)

            # Rotate relative coordinates, then shift back
            x_rot = cth * s_rel[0] - sth * s_rel[1]
            y_rot = sth * s_rel[0] + cth * s_rel[1]
            deterministic = s_eq + grow * np.array([x_rot, y_rot], dtype=float)

            noise = sigma * rng.standard_normal(2)
            proposal = deterministic + noise

            # Clamp: max radius accounting for shifted equilibrium
            clamp_eff = max(0.0, float(clamp_u) - s_eq_norm)
            u_rel = float(np.linalg.norm(proposal - s_eq))
            if u_rel >= clamp_eff > 0.0:
                s_rel_prop = proposal - s_eq
                proposal = s_eq + s_rel_prop * (clamp_eff / max(u_rel, 1e-30))

            state[idx, :] = proposal

        SX = state[:, 0]
        SY = state[:, 1]
        R = self.geom.R

        return ThieleTrajectoryResult(
            model_name=f"CPP Thiele STNO SDE (p={self.polarity:+d}, Guslienko 2014)",
            t=t_eval,
            x=SX * R,
            y=SY * R,
            sx=SX,
            sy=SY,
            disk_radius=R,
            params={
                "Ms": self.material.Ms,
                "alpha": self.material.alpha,
                "P": self.material.P,
                "omega0": self.omega0,
                "N": self.N,
                "d0": self._d0,
                "d1": self._d1,
                "sigma": self._sigma,
                "J_threshold": self.J_threshold,
                "polarity": self.polarity,
                "field": self.field,
                "field_cal": self.field_cal,
            },
            metadata={
                "mode": "CPP-SDE",
                "reference": "Guslienko et al. + Langevin reduction",
                "diffusion": float(diffusion_eff),
                "temperature_k": float(temperature_k),
                "noise_scale": float(noise_scale),
                "seed": seed,
                "dt": float(dt),
            },
        )


# ---------------------------------------------------------------------------
# Parameter fitting helpers
# ---------------------------------------------------------------------------


def _predict_fj_curve(
    J_data: np.ndarray,
    material: MaterialParams,
    geom: DiskGeometry,
    polarity: int,
    omega0: float,
    N: float,
    omega0_Oe_per_J: float,
    chi_scale: float,
    *,
    allow_edge: bool,
) -> np.ndarray:
    model = CPPThieleModel(
        material=material,
        geom=geom,
        omega0=float(omega0),
        N=float(N),
        polarity=int(polarity),
        omega0_Oe_per_J=float(omega0_Oe_per_J),
        chi_scale=float(chi_scale),
    )
    out = np.full(J_data.shape, np.nan, dtype=float)
    for idx, jval in enumerate(J_data):
        freq = model.predict_frequency_dc(
            float(jval),
            allow_edge=allow_edge,
        )
        if freq is not None and np.isfinite(freq):
            out[idx] = float(freq)
    return out


def fit_omega0_N_to_fJ(
    J_data: np.ndarray,
    f_data_hz: np.ndarray,
    *,
    material: MaterialParams,
    geom: DiskGeometry,
    polarity: int = 1,
    initial_omega0: float | None = None,
    initial_N: float = 0.25,
    fit_omega0_Oe_per_J: bool = False,
    initial_omega0_Oe_per_J: float = 0.0,
    fit_chi_scale: bool = False,
    initial_chi_scale: float = 1.0,
    allow_edge: bool = False,
) -> ThieleFJFitResult:
    """
    Fit ``omega0``, ``N`` (optionally ``chi_scale``) of CPP Thiele model to measured ``f(J)`` points.
    """
    j = np.asarray(J_data, dtype=float).ravel()
    f = np.asarray(f_data_hz, dtype=float).ravel()
    if j.size != f.size:
        raise ValueError("J_data and f_data_hz must have the same length")
    if j.size < 3:
        raise ValueError("At least 3 points are required for fitting")

    finite = np.isfinite(j) & np.isfinite(f)
    j = j[finite]
    f = f[finite]
    if j.size < 3:
        raise ValueError("At least 3 finite points are required for fitting")

    omega0_init = float(omega0_novosad(material, geom) if initial_omega0 is None else initial_omega0)
    n_init = float(initial_N)
    oe_init = float(initial_omega0_Oe_per_J)
    chi_init = float(initial_chi_scale)

    def _objective(params: np.ndarray) -> float:
        omega0_val = max(float(params[0]), 1e6)
        n_val = float(params[1])
        oe_val = oe_init
        chi_val = chi_init
        
        idx = 2
        if fit_omega0_Oe_per_J:
            oe_val = float(params[idx])
            idx += 1
        if fit_chi_scale:
            chi_val = float(params[idx])

        f_pred = _predict_fj_curve(
            j,
            material,
            geom,
            polarity,
            omega0_val,
            n_val,
            oe_val,
            chi_val,
            allow_edge=allow_edge,
        )
        mask = np.isfinite(f_pred)
        if np.count_nonzero(mask) < 2:
            return 1e30
        residual = f_pred[mask] - f[mask]
        return float(np.mean(residual**2))

    x0 = np.array([omega0_init, n_init], dtype=float)
    bounds = [(1e6, 1e14), (-5.0, 5.0)]
    if fit_omega0_Oe_per_J:
        x0 = np.append(x0, oe_init)
        bounds.append((-1e-6, 1e-6))
    if fit_chi_scale:
        x0 = np.append(x0, chi_init)
        bounds.append((0.1, 20.0))

    # Coarse global scan to avoid poor local minima.
    omega_low = max(1e6, 0.2 * omega0_init)
    omega_high = max(omega_low * 1.01, 2.5 * omega0_init)
    omega_grid = np.geomspace(omega_low, omega_high, 55)
    n_grid = np.linspace(-2.0, 2.0, 81)
    oe_grid = np.array([oe_init], dtype=float)
    chi_grid = np.array([chi_init], dtype=float)
    if fit_omega0_Oe_per_J:
        oe_grid = np.linspace(-3e-7, 3e-7, 13)
    if fit_chi_scale:
        chi_grid = np.linspace(0.5, 10.0, 25)

    best = x0.copy()
    best_cost = float(_objective(best))
    for omega_val in omega_grid:
        for n_val in n_grid:
            for oe_val in oe_grid:
                for chi_val in chi_grid:
                    candidate = np.array([omega_val, n_val], dtype=float)
                    if fit_omega0_Oe_per_J:
                        candidate = np.append(candidate, oe_val)
                    if fit_chi_scale:
                        candidate = np.append(candidate, chi_val)
                    score = float(_objective(candidate))
                    if score < best_cost:
                        best_cost = score
                        best = candidate

    success = False
    status = "coarse_grid"
    try:
        from scipy.optimize import minimize

        opt = minimize(_objective, best, method="L-BFGS-B", bounds=bounds)
        if opt.success and np.isfinite(opt.fun):
            best = np.asarray(opt.x, dtype=float)
            best_cost = float(opt.fun)
            success = True
            status = str(opt.message)
        else:
            status = f"scipy_local_failed: {opt.message}"
    except Exception:
        status = "scipy_unavailable"

    omega0_fit = max(float(best[0]), 1e6)
    n_fit = float(best[1])
    idx = 2
    oe_fit = oe_init
    chi_fit = chi_init
    if fit_omega0_Oe_per_J:
        oe_fit = float(best[idx])
        idx += 1
    if fit_chi_scale:
        chi_fit = float(best[idx])
    
    f_fit = _predict_fj_curve(
        j,
        material,
        geom,
        polarity,
        omega0_fit,
        n_fit,
        oe_fit,
        chi_fit,
        allow_edge=allow_edge,
    )

    valid_mask = np.isfinite(f_fit)
    if np.count_nonzero(valid_mask) >= 1:
        rmse = float(np.sqrt(np.mean((f_fit[valid_mask] - f[valid_mask]) ** 2)))
    else:
        rmse = float("nan")

    return ThieleFJFitResult(
        model_name="CPP Thiele f(J) fit",
        omega0=omega0_fit,
        N=n_fit,
        omega0_Oe_per_J=oe_fit,
        chi_scale=chi_fit,
        J_data=j,
        f_data_hz=f,
        f_fit_hz=f_fit,
        valid_mask=valid_mask,
        rmse_hz=rmse,
        success=bool(success),
        status=status,
        params={
            "polarity": int(polarity),
            "initial_omega0": omega0_init,
            "initial_N": n_init,
            "fit_omega0_Oe_per_J": bool(fit_omega0_Oe_per_J),
            "fit_chi_scale": bool(fit_chi_scale),
        },
        metadata={
            "allow_edge": bool(allow_edge),
            "n_points": int(j.size),
            "best_cost": float(best_cost),
        },
    )


# ---------------------------------------------------------------------------
# Plotting accessor
# ---------------------------------------------------------------------------


class ThielePlotAccessor:
    """
    Fluent plotting API for :class:`ThieleTrajectoryResult`.

    Accessed via ``result.plt.xy()``, ``result.plt.orbit()``, etc.
    """

    def __init__(self, result: ThieleTrajectoryResult) -> None:
        self._r = result

    @staticmethod
    def _ensure_axes(ax=None, figsize=(7, 5)):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        return ax

    # ── xy(t) ──────────────────────────────────────────────────

    def xy(
        self,
        ax=None,
        units: Literal["nm", "m", "normalized"] = "nm",
        show: bool = False,
        **kwargs,
    ):
        """
        Plot X(t) and Y(t) vs time.

        Parameters
        ----------
        ax : Axes, optional
        units : str
            ``'nm'``, ``'m'``, or ``'normalized'`` (s = r/R).
        show : bool
        **kwargs
            Forwarded to ``ax.plot()``.
        """
        import matplotlib.pyplot as plt

        ax = self._ensure_axes(ax)
        r = self._r
        t_ns = r.t * 1e9

        if units == "nm":
            ax.plot(t_ns, r.x * 1e9, label="X", **kwargs)
            ax.plot(t_ns, r.y * 1e9, label="Y", **kwargs)
            ax.set_ylabel("Position [nm]")
        elif units == "normalized":
            ax.plot(t_ns, r.sx, label="$s_x$", **kwargs)
            ax.plot(t_ns, r.sy, label="$s_y$", **kwargs)
            ax.set_ylabel("Normalised position $s = r/R$")
        else:
            ax.plot(t_ns, r.x, label="X", **kwargs)
            ax.plot(t_ns, r.y, label="Y", **kwargs)
            ax.set_ylabel("Position [m]")

        ax.set_xlabel("Time [ns]")
        ax.set_title(r.model_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if show:
            plt.show()
        return ax

    # ── orbit 2D ───────────────────────────────────────────────

    def orbit(
        self,
        ax=None,
        units: Literal["nm", "m", "normalized"] = "nm",
        disk_outline: bool = True,
        show: bool = False,
        **kwargs,
    ):
        """
        Plot 2D orbit X vs Y with optional disk outline.

        Parameters
        ----------
        ax : Axes, optional
        units : str
        disk_outline : bool
            Draw a dashed circle at the disk boundary.
        show : bool
        **kwargs
            Forwarded to ``ax.plot()``.
        """
        import matplotlib.pyplot as plt

        ax = self._ensure_axes(ax, figsize=(6, 6))
        r = self._r

        if units == "nm":
            ax.plot(r.x * 1e9, r.y * 1e9, **kwargs)
            ax.set_xlabel("X [nm]")
            ax.set_ylabel("Y [nm]")
            if disk_outline:
                theta = np.linspace(0, 2 * np.pi, 200)
                R_nm = r.disk_radius * 1e9
                ax.plot(R_nm * np.cos(theta), R_nm * np.sin(theta), "k--", alpha=0.3, lw=0.8)
        elif units == "normalized":
            ax.plot(r.sx, r.sy, **kwargs)
            ax.set_xlabel("$s_x$")
            ax.set_ylabel("$s_y$")
            if disk_outline:
                theta = np.linspace(0, 2 * np.pi, 200)
                ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, lw=0.8)
        else:
            ax.plot(r.x, r.y, **kwargs)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")

        ax.set_aspect("equal")
        ax.set_title(r.model_name)
        ax.grid(True, alpha=0.3)
        if show:
            plt.show()
        return ax

    # ── radius vs time ─────────────────────────────────────────

    def radius(
        self,
        ax=None,
        units: Literal["nm", "m", "normalized"] = "nm",
        show: bool = False,
        **kwargs,
    ):
        """Plot orbital radius r(t) vs time."""
        import matplotlib.pyplot as plt

        ax = self._ensure_axes(ax)
        r = self._r
        t_ns = r.t * 1e9

        if units == "nm":
            ax.plot(t_ns, r.r * 1e9, **kwargs)
            ax.set_ylabel("Orbit radius [nm]")
        elif units == "normalized":
            ax.plot(t_ns, r.u, **kwargs)
            ax.set_ylabel("Normalised radius $u = r/R$")
        else:
            ax.plot(t_ns, r.r, **kwargs)
            ax.set_ylabel("Orbit radius [m]")

        ax.set_xlabel("Time [ns]")
        ax.set_title(r.model_name)
        ax.grid(True, alpha=0.3)
        if show:
            plt.show()
        return ax

    # ── frequency vs time ──────────────────────────────────────

    def frequency(self, ax=None, show: bool = False, **kwargs):
        """Plot instantaneous frequency f(t) in GHz."""
        import matplotlib.pyplot as plt

        ax = self._ensure_axes(ax)
        r = self._r
        t_ns = r.t * 1e9

        ax.plot(t_ns, r.instantaneous_frequency_ghz, **kwargs)
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Frequency [GHz]")
        ax.set_title(r.model_name)
        ax.grid(True, alpha=0.3)
        if show:
            plt.show()
        return ax

    # ── power spectrum ──────────────────────────────────────────

    def spectrum(
        self,
        ax=None,
        db: bool = True,
        f_max_ghz: float | None = None,
        show: bool = False,
        **kwargs,
    ):
        """
        Plot power spectrum of the trajectory.

        Parameters
        ----------
        ax : Axes, optional
        db : bool
            If True, plot in dB scale (10·log10).
        f_max_ghz : float, optional
            Upper frequency limit [GHz].  Default: auto.
        show : bool
        **kwargs
            Forwarded to ``ax.plot()``.
        """
        import matplotlib.pyplot as plt

        ax = self._ensure_axes(ax)
        r = self._r
        f_ghz = r.spectrum_frequencies_ghz
        power = r.power_spectrum

        if power.size == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            return ax

        if db:
            power_plot = 10.0 * np.log10(np.maximum(power, 1e-30))
            ylabel = "Power [dB]"
        else:
            power_plot = power
            ylabel = "Power [a.u.]"

        ax.plot(f_ghz, power_plot, **kwargs)

        # Mark dominant frequency
        f_dom = r.dominant_frequency_ghz
        if f_dom > 0:
            idx = int(np.argmin(np.abs(f_ghz - f_dom)))
            ax.axvline(f_dom, color="tab:red", linestyle="--", alpha=0.5, lw=0.8)
            ax.plot(f_dom, power_plot[idx], "rv", markersize=6, alpha=0.8)
            ax.annotate(
                f"{f_dom:.2f} GHz",
                xy=(f_dom, power_plot[idx]),
                xytext=(5, 10),
                textcoords="offset points",
                fontsize=8,
                color="tab:red",
            )

        if f_max_ghz is not None:
            ax.set_xlim(0, f_max_ghz)

        ax.set_xlabel("Frequency [GHz]")
        ax.set_ylabel(ylabel)
        ax.set_title(r.model_name)
        ax.grid(True, alpha=0.3)
        if show:
            plt.show()
        return ax

    # ── overview (6-panel) ─────────────────────────────────────

    def overview(self, figsize=(14, 10), show: bool = False):
        """
        6-panel overview: orbit, X/Y(t), radius(t), frequency(t),
        power spectrum, and info text.

        Returns
        -------
        Figure
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(self._r.model_name, fontsize=13)

        self.orbit(ax=axes[0, 0])
        self.xy(ax=axes[0, 1])
        self.spectrum(ax=axes[0, 2])
        self.radius(ax=axes[1, 0])
        self.frequency(ax=axes[1, 1])

        # Info panel
        r = self._r
        info_ax = axes[1, 2]
        info_ax.axis("off")
        lines = [
            f"R_disk = {r.disk_radius*1e9:.1f} nm",
            f"Duration = {r.t[-1]*1e9:.1f} ns" if len(r.t) > 0 else "",
            f"Points = {len(r.t)}",
            "",
            f"f_ss = {r.steady_state_frequency_ghz:.3f} GHz",
            f"r_ss = {r.steady_state_radius_m*1e9:.1f} nm",
            f"f_dom = {r.dominant_frequency_ghz:.3f} GHz",
            f"Δf = {r.linewidth_ghz*1e3:.1f} MHz",
            f"Rotation: {r.rotation_sense}",
        ]
        info_ax.text(
            0.05, 0.95, "\n".join(lines),
            transform=info_ax.transAxes,
            va="top", ha="left",
            fontfamily="monospace",
            fontsize=10,
        )

        fig.tight_layout()
        if show:
            plt.show()
        return fig
