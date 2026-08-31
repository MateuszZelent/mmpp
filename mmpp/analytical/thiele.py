# ruff: noqa: N802, N803, N806, PLR0913
"""
Thiele equation models for vortex core dynamics in nanodisks.

Implements the collective-coordinate (Thiele) approach for magnetic vortex
core trajectory calculation in two geometries:

- **CIP** (Current-In-Plane): adiabatic + non-adiabatic STT (Zhang–Li),
  following Moon et al., arXiv:0809.0952.
- **CPP** (Current-Perpendicular-to-Plane): Slonczewski STT for vortex STNO,
  following Guslienko et al., Nanoscale Research Letters 9, 386 (2014).

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
3. K.Y. Guslienko et al., Nanoscale Research Letters 9, 386 (2014) —
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
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, cast

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
        Spin polarization of the current (dimensionless).  A signed value is
        accepted because adapter layers use the sign to encode the effective
        CPP pumping convention after reducing a MuMax3 torque.
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

    def __post_init__(self) -> None:
        values = {
            "Ms": self.Ms,
            "alpha": self.alpha,
            "P": self.P,
            "A": self.A,
            "gamma": self.gamma,
        }
        if not all(np.isfinite(float(value)) for value in values.values()):
            raise ValueError("material parameters must be finite")
        if self.Ms <= 0.0:
            raise ValueError("Ms must be positive [A/m]")
        if self.alpha < 0.0:
            raise ValueError("alpha must be non-negative")
        if self.A <= 0.0:
            raise ValueError("A must be positive [J/m]")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive [rad/(s T)]")
        if self.beta_nonadiabatic is not None and not np.isfinite(
            float(self.beta_nonadiabatic)
        ):
            raise ValueError("beta_nonadiabatic must be finite when provided")

    @property
    def beta(self) -> float:
        """Non-adiabatic parameter β.  Falls back to α if not set."""
        return (
            self.beta_nonadiabatic if self.beta_nonadiabatic is not None else self.alpha
        )

    @property
    def exchange_length(self) -> float:
        r"""Exchange length :math:`\ell_{\rm ex} = \sqrt{2A/(\mu_0 M_s^2)}` [m]."""
        return math.sqrt(2.0 * self.A / (MU0 * self.Ms**2))


@dataclass(frozen=True)
class SlonczewskiCPPReduction:
    """
    Reduced CPP Slonczewski coefficients consistent with MuMax3 conventions.

    The MuMax3 Slonczewski implementation combines the damping-like
    ``m x (p x m)`` and field-like ``p x m`` contributions as:

    ``mxpxmFac = (A + alpha * B) / (1 + alpha^2)``
    ``pxmFac   = (B - alpha * A) / (1 + alpha^2)``

    where ``A = beta * epsilon`` and ``B = beta * epsilonprime``.

    In the reduced vortex-CPP model we use:
    - ``pump_polarization`` for the auto-oscillation pumping term ``chi(J)``
    - ``phase_polarization`` for the STT-induced gyrotropic phase shift
    """

    polarizer: tuple[float, float, float]
    p_z: float
    mean_m_dot_p: float
    Lambda: float
    epsilonprime: float
    fixed_layer_position: str
    current_sign: float
    epsilon: float
    gilbert_prefactor: float
    pump_polarization: float
    phase_polarization: float
    torque_thickness: float
    chi_prefactor_per_J: float
    phase_omega_per_J: float


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

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.R)) or self.R <= 0.0:
            raise ValueError("R must be positive and finite [m]")
        if not np.isfinite(float(self.L)) or self.L <= 0.0:
            raise ValueError("L must be positive and finite [m]")
        if self.core_diameter is not None and (
            not np.isfinite(float(self.core_diameter)) or self.core_diameter <= 0.0
        ):
            raise ValueError("core_diameter must be positive and finite [m]")

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

    def __post_init__(self) -> None:
        if not all(
            np.isfinite(float(value)) for value in (self.Bx_T, self.By_T, self.Bz_T)
        ):
            raise ValueError("magnetic-field components must be finite [T]")

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

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.domega0_dBz)):
            raise ValueError("domega0_dBz must be finite")
        if not np.isfinite(float(self.seq_per_T)):
            raise ValueError("seq_per_T must be finite")
        if int(self.chirality) not in (-1, 1):
            raise ValueError("chirality must be +1 or -1")

    def omega0_shift(self, *, field_state: ExternalField, polarity: int) -> float:
        """Polarity-dependent Bz → ω₀ shift:  Δω₀ = p · (dω₀/dBz) · Bz."""
        p = 1 if int(polarity) >= 0 else -1
        return p * self.domega0_dBz * field_state.Bz_T

    def s_eq(self, *, field_state: ExternalField) -> tuple[float, float]:
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
ExternalFieldLike: TypeAlias = (
    float | tuple[float, float, float] | ExternalField | np.ndarray
)

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
    frequency = float(f_hz)
    phase_value = float(phase)
    if not np.isfinite(frequency) or frequency < 0.0:
        raise ValueError("f_hz must be finite and non-negative")
    if not np.isfinite(phase_value):
        raise ValueError("phase must be finite")
    omega = 2.0 * math.pi * frequency

    def _b(t: float) -> ExternalField:
        s = math.sin(omega * t + phase_value)
        return ExternalField(
            off.Bx_T + amp.Bx_T * s,
            off.By_T + amp.By_T * s,
            off.Bz_T + amp.Bz_T * s,
        )

    return _b


def field_ac_vector(
    B_amp: ExternalFieldLike,
    f_hz: float,
    *,
    phase: tuple[float, float, float] = (0.0, 0.0, 0.0),
    offset: ExternalFieldLike = 0.0,
) -> FieldFunc:
    """Sinusoidal vector field with independent phase per component."""
    amp = ExternalField.from_any(B_amp)
    off = ExternalField.from_any(offset)
    ph = np.asarray(phase, dtype=float).reshape(-1)
    if ph.size < 3:
        raise ValueError("phase must provide three components")
    frequency = float(f_hz)
    if not np.all(np.isfinite(ph[:3])):
        raise ValueError("phase components must be finite")
    if not np.isfinite(frequency) or frequency < 0.0:
        raise ValueError("f_hz must be finite and non-negative")
    omega = 2.0 * math.pi * frequency

    def _b(t: float) -> ExternalField:
        arg = omega * float(t)
        return ExternalField(
            off.Bx_T + amp.Bx_T * math.sin(arg + float(ph[0])),
            off.By_T + amp.By_T * math.sin(arg + float(ph[1])),
            off.Bz_T + amp.Bz_T * math.sin(arg + float(ph[2])),
        )

    return _b


def field_rotating_inplane(
    B_amp: float,
    f_hz: float,
    *,
    phase: float = 0.0,
    clockwise: bool = False,
    Bz_offset: float = 0.0,
) -> FieldFunc:
    """Circularly rotating in-plane field ``(Bx, By)`` with optional Bz offset."""
    amp = float(B_amp)
    frequency = float(f_hz)
    phase_value = float(phase)
    bz = float(Bz_offset)
    if not all(np.isfinite(value) for value in (amp, frequency, phase_value, bz)):
        raise ValueError("rotating-field parameters must be finite")
    if frequency < 0.0:
        raise ValueError("f_hz must be non-negative")
    omega = 2.0 * math.pi * frequency
    handedness = -1.0 if clockwise else 1.0

    def _b(t: float) -> ExternalField:
        arg = omega * float(t) + phase_value
        return ExternalField(
            amp * math.cos(arg),
            handedness * amp * math.sin(arg),
            bz,
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
        """Azimuthal angle φ(t) [rad], centered on orbit geometry."""
        x_c = self.x - np.mean(self.x)
        y_c = self.y - np.mean(self.y)
        return np.angle(x_c + 1j * y_c)

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

    def compute_spectrum(
        self,
        *,
        transient_fraction: float = 0.0,
        t_min: float | None = None,
        signal: Literal["x", "y", "radius"] = "x",
        window: Literal["hann", "none"] = "hann",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute one-sided FFT power after optional transient rejection.

        Non-uniform samples are resampled to an equidistant time grid.  The
        returned tuple contains frequency [Hz] and relative FFT power; it is
        not a calibrated electrical PSD in V²/Hz.
        """
        fraction = float(transient_fraction)
        if not np.isfinite(fraction) or not 0.0 <= fraction < 1.0:
            raise ValueError("transient_fraction must lie in [0, 1)")
        if t_min is not None and not np.isfinite(float(t_min)):
            raise ValueError("t_min must be finite when provided")
        if signal not in {"x", "y", "radius"}:
            raise ValueError("signal must be one of {'x', 'y', 'radius'}")
        if window not in {"hann", "none"}:
            raise ValueError("window must be 'hann' or 'none'")

        time = np.asarray(self.t, dtype=float)
        values = {
            "x": np.asarray(self.x, dtype=float),
            "y": np.asarray(self.y, dtype=float),
            "radius": np.asarray(self.r, dtype=float),
        }[signal]
        if time.shape != values.shape or not np.all(np.isfinite(time)):
            raise ValueError(
                "trajectory time and signal arrays must be finite and aligned"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("trajectory signal contains non-finite values")
        if time.size < 4:
            return np.array([], dtype=float), np.array([], dtype=float)
        if np.any(np.diff(time) <= 0.0):
            raise ValueError("trajectory time samples must be strictly increasing")

        start_time = (
            float(t_min)
            if t_min is not None
            else float(time[0] + fraction * (time[-1] - time[0]))
        )
        mask = time >= start_time
        time = time[mask]
        values = values[mask]
        n = int(time.size)
        if n < 4:
            raise ValueError("spectral window must contain at least four samples")

        dt = float((time[-1] - time[0]) / (n - 1))
        if not np.allclose(np.diff(time), dt, rtol=1e-6, atol=1e-15 * dt):
            uniform_time = np.linspace(time[0], time[-1], n)
            values = np.interp(uniform_time, time, values)

        weights = np.hanning(n) if window == "hann" else np.ones(n, dtype=float)
        centered = values - np.mean(values)
        fft_values = np.fft.rfft(centered * weights)
        frequencies = np.fft.rfftfreq(n, d=dt)
        power = (np.abs(fft_values) ** 2) / max(float(n), 1.0)
        power *= 2.0 / max(float(np.mean(weights**2)), 1e-30)
        return frequencies, power

    @property
    def _spectrum_cache(self) -> tuple[np.ndarray, np.ndarray]:
        """Cached (frequencies_hz, power) from windowed FFT of x(t)."""
        # Zwykłe pojedyncze podkreślenie - lintery je ignorują, brak name manglingu.
        cache_attr = "_spectrum_cache_data"

        cached_result = getattr(self, cache_attr, None)
        if cached_result is None:
            result = self.compute_spectrum()

            object.__setattr__(self, cache_attr, result)
            return result

        return cached_result

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
        """Estimated linewidth (FWHM) from the power spectrum [GHz].
        Uses a robust Lorentzian fit for noisy (SDE/micromagnetic) data,
        with a fallback to half-maximum counting for pure ODE trajectories.
        """
        freqs, power = self._spectrum_cache
        if power.size < 3:
            return 0.0

        start = 1 if power.size > 1 else 0
        peak_idx = int(np.argmax(power[start:])) + start
        f_peak = float(freqs[peak_idx])
        p_max = float(power[peak_idx])
        df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0

        # --- 1. Klasyczna metoda progowa (jako fallback i punkt startowy) ---
        half_max = p_max / 2.0
        above = power >= half_max

        left = peak_idx
        while left > start and above[left]:
            left -= 1

        right = peak_idx
        while right < len(power) - 1 and above[right]:
            right += 1

        naive_fwhm = float(right - left) * df

        # Jeśli widmo to absolutna "igła" (np. czyste ODE, brak szumu),
        # metoda progowa jest optymalna - fitowanie nie ma tu sensu.
        if naive_fwhm <= 2.0 * df or power.size < 10:
            return naive_fwhm * 1e-9

        # --- 2. Odporne na szum dopasowanie krzywej Lorentza ---
        try:
            import warnings

            from scipy.optimize import curve_fit

            # Wycinek okna częstotliwości (np. +/- 10 szerokości naiwnych, min. 200 MHz)
            # Uodparnia to fit na artefakty 1/f i asymetrię przy 0 Hz.
            window_hz = max(10.0 * naive_fwhm, 200e6)

            mask = (freqs >= f_peak - window_hz) & (freqs <= f_peak + window_hz)
            mask[:start] = False  # Zignoruj składową DC

            f_win = freqs[mask]
            p_win = power[mask]

            if len(f_win) < 5:
                return naive_fwhm * 1e-9

            # Pracujemy w GHz, aby algorytm optymalizatora
            # nie "zgłupiał" na gigantycznych wartościach f^2 (rzędu 10^18)
            f_win_ghz = f_win * 1e-9
            f_peak_ghz = f_peak * 1e-9

            def lorentzian(f, f0, fwhm, amp, bg):
                gamma = fwhm / 2.0
                return amp * (gamma**2) / ((f - f0) ** 2 + gamma**2) + bg

            bg_guess = float(np.median(p_win))
            amp_guess = p_max - bg_guess
            if amp_guess <= 0:
                amp_guess = p_max

            # Parametry początkowe: f0, fwhm, amplituda, background
            p0 = [f_peak_ghz, naive_fwhm * 1e-9, amp_guess, bg_guess]

            # Ograniczenia zapobiegające "rozjechaniu się" fita na niefizyczne wartości
            bounds = (
                [f_win_ghz[0], 0.0, 0.0, 0.0],
                [f_win_ghz[-1], np.inf, np.inf, np.inf],
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    lorentzian, f_win_ghz, p_win, p0=p0, bounds=bounds, maxfev=2000
                )

            fitted_fwhm_ghz = float(popt[1])
            return fitted_fwhm_ghz

        except Exception:
            # Fallback w przypadku, gdy fit ze SciPy z jakiegoś powodu się nie powiedzie
            return naive_fwhm * 1e-9

    # ── plotting ───────────────────────────────────────────────

    @property
    def plt(self) -> Any:
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
            'color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);">'
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
    domega0_dJ: float = 0.0
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
    def plt(self) -> Any:
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
            f"Thiele fit: ω0={self._result.omega0:.3e} rad/s, N={self._result.N:.3f}"
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
    pol = float(Pol)
    lam = float(Lambda)
    cth = float(cos_theta)
    if not np.isfinite(pol):
        raise ValueError("Pol must be finite")
    if not np.isfinite(lam) or lam <= 0.0:
        raise ValueError("Lambda must be positive and finite")
    if not np.isfinite(cth) or not -1.0 <= cth <= 1.0:
        raise ValueError("cos_theta must be finite and lie in [-1, 1]")
    lam2 = lam * lam
    denom = (lam2 + 1.0) + (lam2 - 1.0) * cth
    if abs(denom) < 1e-30:
        raise ValueError("Degenerate denominator in Slonczewski efficiency")
    return pol * lam2 / denom


def reduce_mumax_slonczewski_cpp(
    *,
    material: MaterialParams,
    torque_thickness: float,
    polarizer: tuple[float, float, float] | tuple[float, float] = (0.0, 0.0, 1.0),
    fixed_layer_position: str = "top",
    Lambda: float = 1.0,
    epsilonprime: float = 0.0,
    mean_m_dot_p: float = 0.0,
) -> SlonczewskiCPPReduction:
    """
    Reduce MuMax3 Slonczewski CPP inputs to vortex-CPP effective coefficients.

    Notes
    -----
    The full MuMax3 cell-wise efficiency depends on ``m·p``.  Its spatial
    projection is not determined by the polarizer's ``p_z`` component.  The
    reduced model therefore takes an explicit representative
    ``mean_m_dot_p`` (zero by default for a centred circular vortex) and uses
    ``p_z`` only for the perpendicular pumping projection.

    MuMax3 defines ``epsilon=P/2`` when ``Lambda=1``.  The Guslienko CPP
    amplitude equation instead uses ``P`` in
    ``chi=-p*gamma*hbar*P*J/(4*e*L*Ms)``.  Consequently the projected
    polarization below contains the required factor of two.
    """
    vec: Any = np.asarray(polarizer, dtype=float).reshape(-1)
    if vec.size == 2:
        vec = np.array([vec[0], vec[1], 0.0], dtype=float)
    if vec.size < 3:
        raise ValueError("polarizer must provide at least 2 or 3 components")
    norm = float(np.linalg.norm(vec[:3]))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("polarizer must be finite and non-zero")
    p = cast(tuple[float, float, float], tuple(float(v) for v in (vec[:3] / norm)))

    pos_token = str(fixed_layer_position).strip().lower()
    if pos_token in {"fixedlayer_top", "top", "+1", "1"}:
        pos_name = "top"
        current_sign = 1.0
    elif pos_token in {"fixedlayer_bottom", "bottom", "-1", "2"}:
        pos_name = "bottom"
        current_sign = -1.0
    else:
        raise ValueError("fixed_layer_position must be 'top' or 'bottom'")

    thickness = float(torque_thickness)
    if not np.isfinite(thickness) or thickness <= 0.0:
        raise ValueError("torque_thickness must be positive and finite")

    mean_dot = float(mean_m_dot_p)
    if not np.isfinite(mean_dot) or not -1.0 <= mean_dot <= 1.0:
        raise ValueError("mean_m_dot_p must be finite and lie in [-1, 1]")
    epsilon = slonczewski_mtj_efficiency(material.P, float(Lambda), mean_dot)
    alpha = float(material.alpha)
    eps_prime = float(epsilonprime)
    if not np.isfinite(eps_prime):
        raise ValueError("epsilonprime must be finite")
    gilb = 1.0 / (1.0 + alpha * alpha)
    p_z = float(p[2])

    pump_p = 2.0 * current_sign * p_z * gilb * (epsilon + alpha * eps_prime)
    phase_p = 2.0 * current_sign * p_z * gilb * (eps_prime - alpha * epsilon)
    prefactor = (
        float(material.gamma)
        * _HBAR
        / (4.0 * _E_CHARGE * thickness * float(material.Ms))
    )

    p_parallel = float(np.hypot(p[0], p[1]))
    if p_parallel > 0.05:
        warnings.warn(
            "Polarizer has a significant in-plane component; the reduced "
            "circular CPP model only maps the out-of-plane p_z component into "
            "vortex auto-oscillation pumping.",
            UserWarning,
            stacklevel=2,
        )

    return SlonczewskiCPPReduction(
        polarizer=p,
        p_z=p_z,
        mean_m_dot_p=mean_dot,
        Lambda=float(Lambda),
        epsilonprime=eps_prime,
        fixed_layer_position=pos_name,
        current_sign=current_sign,
        epsilon=float(epsilon),
        gilbert_prefactor=float(gilb),
        pump_polarization=float(pump_p),
        phase_polarization=float(phase_p),
        torque_thickness=thickness,
        chi_prefactor_per_J=float(prefactor * pump_p),
        phase_omega_per_J=float(prefactor * phase_p),
    )


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

    value = float(J_dc)
    if not np.isfinite(value):
        raise ValueError("J_dc must be finite [A/m^2]")

    def _j(t: float) -> float:  # noqa: ARG001
        return value

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
    amplitude = float(J_amp)
    frequency = float(f_hz)
    offset = float(J_offset)
    phase_value = float(phase)
    if not all(
        np.isfinite(value) for value in (amplitude, frequency, offset, phase_value)
    ):
        raise ValueError("AC-current parameters must be finite")
    if frequency < 0.0:
        raise ValueError("f_hz must be non-negative")
    omega = 2.0 * math.pi * frequency

    def _j(t: float) -> float:
        return offset + amplitude * math.sin(omega * t + phase_value)

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

    on_value = float(J_on)
    start = float(t_on)
    stop = float(t_off)
    base = float(J_base)
    if not all(np.isfinite(value) for value in (on_value, start, stop, base)):
        raise ValueError("pulse parameters must be finite")
    if stop <= start:
        raise ValueError("t_off must be greater than t_on")

    def _j(t: float) -> float:
        return on_value if start <= t < stop else base

    return _j


# ---------------------------------------------------------------------------
# Gyrotropic frequency estimate
# ---------------------------------------------------------------------------


def omega0_novosad(mat: MaterialParams, geo: DiskGeometry) -> float:
    r"""
    Estimate the gyrotropic eigenfrequency ω₀ for a vortex in a thin disk.

    Uses the leading thin-disk asymptote of the side-charge-free vortex
    result used by Novosad et al. for the translational mode:

    .. math::

        \omega_0 = \frac{5}{9\pi} \gamma \mu_0 M_s \frac{L}{R}.

    This is not the full finite-aspect-ratio expression in Eq. (3) of
    Novosad et al., which contains the magnetostatic integral
    :math:`F_v(L/R)` and an exchange correction.  In particular, no ad-hoc
    first-order thickness factor is applied here.  Use a micromagnetically
    calibrated ``omega0`` whenever quantitative finite-thickness accuracy is
    required.

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
    if beta > 0.2:
        warnings.warn(
            "omega0_novosad uses the thin-disk asymptote but L/R "
            f"is {beta:.3g}; supply a calibrated omega0 for quantitative use.",
            UserWarning,
            stacklevel=2,
        )

    gamma0 = mat.gamma * MU0  # rad/(s·T) → rad·m/(s·A)
    omega0 = (5.0 / (9.0 * math.pi)) * gamma0 * mat.Ms * beta
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
        self.omega0 = float(omega0)
        self.polarity = int(polarity)
        self.field = field if field is not None else ExternalField()
        self.field_cal = field_cal if field_cal is not None else FieldCalibration()
        if self.polarity not in (1, -1):
            raise ValueError("polarity must be +1 or -1")
        if not np.isfinite(self.omega0) or self.omega0 <= 0.0:
            raise ValueError("omega0 must be a positive finite angular frequency")

        # normalise current direction
        cx, cy = float(current_dir[0]), float(current_dir[1])
        norm = math.sqrt(cx**2 + cy**2)
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("current_dir must be a finite non-zero vector")
        self.current_dir = (cx / norm, cy / norm)

        # derived quantities
        self._setup()

    def _setup(self) -> None:
        mat = self.material
        geo = self.geom
        p = self.polarity

        # Spin-drift velocity prefactor: u₀ = - μ_B P / (e Ms)  [m³/(A·s)]
        self._u0_prefactor = -_MU_B * mat.P / (_E_CHARGE * mat.Ms)

        # Core diameter δ (≈ core_diameter or 2·exchange_length)
        delta = geo.core_diameter if geo.core_diameter else 2.0 * mat.exchange_length

        # D/G_0 ≈ (1/2) ln(R/δ)  (Moon et al.)
        ratio = geo.R / max(delta, 1e-10)
        self._d_over_G0 = 0.5 * math.log(max(ratio, 1.1))

        # Effective constants for the linear ODE
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
        J = float(J_func(t))
        if not np.isfinite(J):
            raise ValueError("J_func returned a non-finite current density")
        u0 = self._u0_prefactor * J
        ux = u0 * self.current_dir[0]
        uy = u0 * self.current_dir[1]

        # Resolve field (possibly time-dependent)
        B = self._field_at(t, B_func)
        w0 = self._omega0_base + self.field_cal.omega0_shift(
            field_state=B, polarity=self._p
        )
        if not np.isfinite(w0) or w0 <= 0.0:
            raise ValueError(
                "field calibration produced a non-positive gyrotropic frequency"
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

        # Moon ODE using relative coordinates
        det = (alpha * dG) ** 2 + p**2  # always > 0

        # FIZYKA: Prawidłowe znaki sił STT (rotacja CCW dla p=1)
        rhs_I = -w0 * Xr - p * uy + beta * dG * ux
        rhs_II = -w0 * Yr + p * ux + beta * dG * uy

        # MATEMATYKA: Prawidłowa odwrotna macierz sprzężeń
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

        t0, t1 = float(t_span[0]), float(t_span[1])
        step = float(dt)
        initial = np.asarray(r0, dtype=float).reshape(2)
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            raise ValueError("t_span must contain finite values with t_end > t_start")
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("dt must be positive and finite")
        if not np.all(np.isfinite(initial)):
            raise ValueError("r0 must contain finite coordinates")

        t_eval: np.ndarray = np.arange(t0, t1, step, dtype=float)
        if t_eval.size == 0 or t_eval[-1] < t1:
            t_eval = np.append(t_eval, t1)

        sol = solve_ivp(
            fun=lambda t, y: self._rhs(t, y, J_func, B_func),
            t_span=(t0, t1),
            y0=initial,
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
    - **STT pumping:**          χ(J) = -p γ σ J / 2,  σ = ℏP/(2eL Mₛ)

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
    domega0_dJ : float
        Current-induced frequency shift [rad/s / A/m²]. Used for Oersted field correction.

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
    K.Y. Guslienko et al., Nanoscale Research Letters 9, 386 (2014).
    """

    def __init__(
        self,
        material: MaterialParams,
        geom: DiskGeometry,
        omega0: float,
        N: float = 0.25,
        polarity: int = 1,
        domega0_dJ: float = 0.0,
        field: ExternalField | None = None,
        field_cal: FieldCalibration | None = None,
        chi_scale: float = 1.0,
        torque_thickness: float | None = None,
        omega0_Oe_per_J: float | None = None,
    ) -> None:
        self.material = material
        self.geom = geom
        self.omega0 = float(omega0)
        self.N = float(N)
        self.polarity = int(polarity)
        if omega0_Oe_per_J is not None and float(domega0_dJ) != 0.0:
            raise ValueError("provide only one of domega0_dJ or legacy omega0_Oe_per_J")
        self.domega0_dJ = float(
            domega0_dJ if omega0_Oe_per_J is None else omega0_Oe_per_J
        )
        self.field = field if field is not None else ExternalField()
        self.field_cal = field_cal if field_cal is not None else FieldCalibration()
        self.chi_scale = float(chi_scale)
        self.torque_thickness = float(
            geom.L if torque_thickness is None else torque_thickness
        )
        if self.polarity not in (1, -1):
            raise ValueError("polarity must be +1 or -1")
        scalar_values = {
            "omega0": self.omega0,
            "N": self.N,
            "domega0_dJ": self.domega0_dJ,
            "chi_scale": self.chi_scale,
            "torque_thickness": self.torque_thickness,
        }
        if not all(np.isfinite(value) for value in scalar_values.values()):
            raise ValueError("CPP model coefficients must be finite")
        if self.omega0 <= 0.0:
            raise ValueError("omega0 must be positive [rad/s]")
        if self.chi_scale <= 0.0:
            raise ValueError("chi_scale must be positive")
        if self.torque_thickness <= 0.0:
            raise ValueError("torque_thickness must be positive [m]")

        self._setup()

    def _setup(self) -> None:
        mat = self.material
        geo = self.geom
        Rc = geo.Rc(mat)

        # Slonczewski coefficient:  σ = ℏ P / (2 e L Ms)  [m²/(A·s)… sort of]
        thickness = max(float(self.torque_thickness), 1e-30)
        self._sigma = _HBAR * mat.P / (2.0 * _E_CHARGE * thickness * mat.Ms)

        # d₀ = α · [5 + 4 ln(R/Rc)] / 8
        ratio = geo.R / max(Rc, 1e-10)
        self._d0 = mat.alpha * (5.0 + 4.0 * math.log(max(ratio, 1.1))) / 8.0

        # d₁ = (11/6) α
        self._d1 = (11.0 / 6.0) * mat.alpha

        # χ(J) = -p γ σ J / 2
        self._chi_prefactor = mat.gamma * self._sigma / 2.0

    # ── public helpers ─────────────────────────────────────────

    def chi(self, J: float) -> float:
        """
        STT pumping rate χ(J) [rad/s], scaled by ``chi_scale``.

        Notes
        -----
        For vortex CPP auto-oscillations the pumping sign depends on the
        relative orientation of the spin polarization and the vortex-core
        polarity. In the reduced Thiele model this enters as a ``-p`` factor
        multiplying the scalar pumping rate.
        """
        return -float(self.polarity) * self.chi_scale * self._chi_prefactor * J

    def d(self, u: float) -> float:
        """Nonlinear damping d(u) [dimensionless]."""
        return self._d0 + self._d1 * u**2

    def omega0_eff(
        self,
        J: float,
        field_state: ExternalField | None = None,
        *,
        domega0_dJ: float | None = None,
    ) -> float:
        """
        Effective linear frequency [rad/s].

        ω₀_eff(J, Bz) = ω₀ + (dω₀/dJ)·J + p·(dω₀/dBz)·Bz

        The Bz dependence is **phenomenological** and must be calibrated
        from micromagnetic simulations (``field_cal.domega0_dBz``).
        The polarity factor ``p`` is applied automatically.
        """
        B = (
            ExternalField.from_any(field_state)
            if field_state is not None
            else self.field
        )
        # Wybierz podany override lub użyj atrybutu zdefiniowanego w modelu
        slope = self.domega0_dJ if domega0_dJ is None else float(domega0_dJ)
        return (
            self.omega0
            + slope * float(J)
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
        B = (
            ExternalField.from_any(field_state)
            if field_state is not None
            else self.field
        )
        sx, sy = self.field_cal.s_eq(field_state=B)
        out = np.array([sx, sy], dtype=float)
        norm = float(np.linalg.norm(out))
        if norm >= 1.0:
            raise ValueError(
                f"FieldCalibration places s_eq outside the disk (|s_eq|={norm:.3g})"
            )
        if norm > 0.8 and not getattr(self, "_seq_warned", False):
            warnings.warn(
                f"Large equilibrium shift |s_eq|={norm:.3g}; rigid-vortex "
                "field calibration is outside its safe range.",
                UserWarning,
                stacklevel=2,
            )
            self._seq_warned = True
        return out

    def omega(
        self, u: float, J: float = 0.0, *, domega0_dJ: float | None = None
    ) -> float:
        """Nonlinear gyrotropic frequency ω(u, J) [rad/s]."""
        # Przekazujemy parametr do wywołania pod spodem
        return self.omega0_eff(J, domega0_dJ=domega0_dJ) * (1.0 + self.N * u**2)

    @property
    def J_threshold(self) -> float:
        """Threshold current density for self-oscillation [A/m²]."""
        # For omega0_eff(J)=omega0_eff(0)+domega0_dJ*J, solve the complete
        # linear-growth condition chi(J_th)-d0*omega0_eff(J_th)=0.
        pump_slope = -float(self.polarity) * self.chi_scale * self._chi_prefactor
        denom = pump_slope - self._d0 * self.domega0_dJ
        if abs(denom) < 1e-30:
            return float("inf")
        return self._d0 * self.omega0_eff(0.0) / denom

    def threshold_current_dc(self) -> float:
        """Threshold DC current density for auto-oscillation [A/m²]."""
        return self.J_threshold

    def predict_frequency_dc(
        self,
        J_dc: float,
        *,
        allow_edge: bool = False,
        domega0_dJ: float | None = None,
    ) -> float | None:
        """
        Predict steady-state gyration frequency for DC current.

        Parameters
        ----------
        J_dc : float
            DC current density [A/m²].
        allow_edge : bool
            If True, return edge-clamped frequency when u₀ ≥ u_stop.
        domega0_dJ : float, optional
            Override the model's ``domega0_dJ`` for this call.

        Returns
        -------
        float | None
            Frequency [Hz], or ``None`` below threshold / edge-clamped (if disallowed).
        """
        u0 = self.steady_state_u(
            float(J_dc), allow_edge=allow_edge, domega0_dJ=domega0_dJ
        )
        if u0 is None:
            return None
        omega_eff = self.omega(u0, float(J_dc), domega0_dJ=domega0_dJ)
        return float(omega_eff / (2.0 * math.pi))

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
        """
        target = float(target_frequency_hz)
        if not np.isfinite(target) or target <= 0.0:
            raise ValueError("target_frequency_hz must be a positive finite value")

        if J_bounds is None:
            threshold = self.threshold_current_dc()
            candidates = (1.01 * threshold, 6.0 * threshold)
            j_min, j_max = min(candidates), max(candidates)
        else:
            candidates = (float(J_bounds[0]), float(J_bounds[1]))
            j_min, j_max = min(candidates), max(candidates)
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
                "domega0_dJ": self.domega0_dJ,
                "n_grid": int(max(int(n_grid), 32)),
            },
        )

    def steady_state_u(
        self,
        J: float,
        *,
        allow_edge: bool = False,
        u_stop: float = 0.98,
        domega0_dJ: float | None = None,
    ) -> float | None:
        """
        Analytical steady-state normalised radius u₀ for DC current J.

        Solves χ(J) = d(u₀)·ω(u₀, J).  Returns ``None`` if J < J_th.
        If ``allow_edge=True`` and the solution exceeds ``u_stop``, returns
        ``u_stop`` (edge-limited orbit) instead of ``None``.
        """
        J = float(J)
        chi_val = float(self.chi(J))
        # Obliczenia z parametrem thread-safe:
        w0 = float(self.omega0_eff(J, domega0_dJ=domega0_dJ))
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
            if (d1 + N * d0 + 2.0 * N * d1 * u * u) <= 0.0:
                continue
            u_candidates.append(u)

        if not u_candidates:
            if allow_edge and chi_val > self.d(u_stop) * self.omega(
                u_stop, J, domega0_dJ=domega0_dJ
            ):
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
        if not np.isfinite(J):
            raise ValueError("J_func returned a non-finite current density")
        chi_val = self.chi(J)
        omega0_eff = self.omega0_eff(J, field_state=B)
        if omega0_eff <= 0.0 and not getattr(self, "_omega0_eff_warned", False):
            warnings.warn(
                "omega0_eff <= 0; CPP Thiele model is outside its calibrated "
                "valid range for this current or field.",
                UserWarning,
                stacklevel=2,
            )
            self._omega0_eff_warned = True
        omega_val = omega0_eff * (1.0 + self.N * u**2)
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
        clamp_u: float | None = 0.999,
        edge_behavior: str = "freeze",
        **ivp_kwargs: Any,
    ) -> ThieleTrajectoryResult:
        """
        Integrate the CPP nonlinear Thiele equation.
        """
        from scipy.integrate import solve_ivp

        if J_func is None:
            J_func = current_dc(0.0)

        clamp_u_value = None if clamp_u is None else float(clamp_u)
        if clamp_u_value is not None and (
            not np.isfinite(clamp_u_value)
            or clamp_u_value <= 0.0
            or clamp_u_value > 1.0
        ):
            raise ValueError("clamp_u must lie in (0, 1] when provided")
        edge_behavior_token = str(edge_behavior).strip().lower()
        if edge_behavior_token not in {"freeze", "truncate"}:
            raise ValueError("edge_behavior must be one of {'freeze', 'truncate'}")

        t0, t1 = float(t_span[0]), float(t_span[1])
        step = float(dt)
        initial = np.asarray(s0, dtype=float).reshape(2)
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            raise ValueError("t_span must contain finite values with t_end > t_start")
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("dt must be positive and finite")
        if not np.all(np.isfinite(initial)):
            raise ValueError("s0 must contain finite coordinates")
        if float(np.linalg.norm(initial)) >= 1.0:
            raise ValueError("s0 must start inside the physical disk (|s0| < 1)")

        t_eval: np.ndarray = np.arange(t0, t1 + 0.5 * step, step)
        # Guard against floating-point overshoot in np.arange
        if t_eval.size and t_eval[-1] > t1:
            t_eval = t_eval[:-1]
        if t_eval.size == 0 or t_eval[-1] < t1:
            t_eval = np.append(t_eval, t1)

        user_events = ivp_kwargs.pop("events", None)
        event_registry: list[Any] = []
        event_kinds: list[str] = []

        if clamp_u_value is not None:

            def _edge_event(t: float, y: np.ndarray) -> float:
                B = self._field_at(t, B_func)
                s_eq = self.s_eq(field_state=B)
                s_rel = np.asarray(y, dtype=float) - s_eq
                return float(np.hypot(s_rel[0], s_rel[1]) - clamp_u_value)

            setattr(_edge_event, "terminal", True)  # noqa: B010
            setattr(_edge_event, "direction", 1.0)  # noqa: B010
            event_registry.append(_edge_event)
            event_kinds.append("relative")

        def _disk_edge_event(t: float, y: np.ndarray) -> float:
            del t
            return float(1.0 - np.hypot(float(y[0]), float(y[1])))

        setattr(_disk_edge_event, "terminal", True)  # noqa: B010
        setattr(_disk_edge_event, "direction", -1.0)  # noqa: B010
        event_registry.append(_disk_edge_event)
        event_kinds.append("disk")

        if user_events is not None:
            if isinstance(user_events, (list, tuple)):
                event_registry.extend(user_events)
                event_kinds.extend("user" for _ in user_events)
            else:
                event_registry.append(user_events)
                event_kinds.append("user")

        sol = solve_ivp(
            fun=lambda t, y: self._rhs(t, y, J_func, B_func),
            t_span=(t0, t1),
            y0=initial,
            t_eval=t_eval,
            events=event_registry if event_registry else None,
            method=method,
            max_step=ivp_kwargs.pop("max_step", dt),
            rtol=ivp_kwargs.pop("rtol", 1e-9),
            atol=ivp_kwargs.pop("atol", 1e-14),
            **ivp_kwargs,
        )

        if not sol.success:
            raise RuntimeError(f"CPP Thiele integration failed: {sol.message}")

        t_out = np.asarray(sol.t, dtype=float)
        SX = np.asarray(sol.y[0], dtype=float)
        SY = np.asarray(sol.y[1], dtype=float)
        R = self.geom.R
        edge_limited = False
        edge_hit_time = None
        edge_hit_kind = None

        if event_registry:
            t_events = getattr(sol, "t_events", None) or []
            y_events = getattr(sol, "y_events", None) or []
            hit_candidates: list[tuple[float, int]] = []
            for event_index, event_times in enumerate(t_events):
                if event_index >= len(event_kinds):
                    continue
                if event_kinds[event_index] == "user":
                    continue
                if len(event_times) > 0:
                    hit_candidates.append((float(event_times[0]), event_index))

            if hit_candidates:
                _, hit_index = min(hit_candidates, key=lambda item: item[0])
                edge_limited = True
                edge_hit_time = float(t_events[hit_index][0])
                edge_hit_kind = event_kinds[hit_index]
                hit_state = np.asarray(y_events[hit_index][0], dtype=float)
                B_hit = self._field_at(edge_hit_time, B_func)
                s_eq_hit = self.s_eq(field_state=B_hit)
                s_rel_hit = hit_state - s_eq_hit
                if edge_hit_kind == "relative" and clamp_u_value is not None:
                    u_hit = float(np.hypot(s_rel_hit[0], s_rel_hit[1]))
                    if u_hit > 0.0:
                        clamped_rel = s_rel_hit * (clamp_u_value / max(u_hit, 1e-30))
                    else:
                        clamped_rel = np.array([clamp_u_value, 0.0], dtype=float)
                    clamped_state = s_eq_hit + clamped_rel
                else:
                    u_abs = float(np.hypot(hit_state[0], hit_state[1]))
                    if u_abs > 0.0:
                        clamped_state = hit_state / max(u_abs, 1e-30)
                    else:
                        clamped_state = np.array([1.0, 0.0], dtype=float)

                if SX.size:
                    SX[-1] = float(clamped_state[0])
                    SY[-1] = float(clamped_state[1])

                if (
                    edge_behavior_token == "freeze"
                    and t_eval.size
                    and t_out.size < t_eval.size
                ):
                    remaining_mask = np.asarray(t_eval, dtype=float) > edge_hit_time
                    remaining_t = np.asarray(t_eval, dtype=float)[remaining_mask]
                    if remaining_t.size:
                        t_out = np.concatenate([t_out, remaining_t])
                        SX = np.concatenate(
                            [
                                SX,
                                np.full(
                                    remaining_t.shape,
                                    float(clamped_state[0]),
                                    dtype=float,
                                ),
                            ]
                        )
                        SY = np.concatenate(
                            [
                                SY,
                                np.full(
                                    remaining_t.shape,
                                    float(clamped_state[1]),
                                    dtype=float,
                                ),
                            ]
                        )

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
                "domega0_dJ": self.domega0_dJ,
                "field": self.field,
                "field_cal": self.field_cal,
                "clamp_u": clamp_u_value,
            },
            metadata={
                "mode": "CPP",
                "reference": "Guslienko et al., Nanoscale Research Letters 9:386 (2014)",
                "edge_limited": bool(edge_limited),
                "edge_hit_time": edge_hit_time,
                "edge_hit_kind": edge_hit_kind,
                "edge_behavior": edge_behavior_token if edge_limited else None,
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
        """
        if J_func is None:
            J_func = current_dc(0.0)

        t0, t1 = float(t_span[0]), float(t_span[1])
        step = float(dt)
        initial = np.asarray(s0, dtype=float).reshape(2)
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            raise ValueError("t_span must contain finite values with t_end > t_start")
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("dt must be positive and finite")
        if not np.all(np.isfinite(initial)):
            raise ValueError("s0 must contain finite coordinates")
        if float(np.linalg.norm(initial)) >= 1.0:
            raise ValueError("s0 must start inside the physical disk (|s0| < 1)")
        temperature = float(temperature_k)
        amplitude_scale = float(noise_scale)
        clamp = float(clamp_u)
        if not np.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature_k must be finite and non-negative")
        if not np.isfinite(amplitude_scale) or amplitude_scale < 0.0:
            raise ValueError("noise_scale must be finite and non-negative")
        if not np.isfinite(clamp) or not 0.0 < clamp <= 1.0:
            raise ValueError("clamp_u must lie in (0, 1]")

        t_eval: np.ndarray = np.arange(t0, t1, step, dtype=float)
        if t_eval.size == 0 or t_eval[-1] < t1:
            t_eval = np.append(t_eval, t1)

        if diffusion is None:
            gyro = (
                2.0
                * math.pi
                * float(self.material.Ms)
                * float(self.geom.L)
                / float(self.material.gamma)
            )
            # From <xi_i(t)xi_j(t')>=2*kBT*D*delta_ij*delta(t-t')
            # and inversion of D*I+p*G*J2, converted to s=X/R.
            base_diffusion = (
                _K_B
                * temperature
                * self._d0
                / (gyro * (1.0 + self._d0**2) * self.geom.R**2)
            )
            diffusion_eff = amplitude_scale**2 * base_diffusion
            diffusion_model = "thiele_fdt"
        else:
            diffusion_eff = float(diffusion)
            if not np.isfinite(diffusion_eff) or diffusion_eff < 0.0:
                raise ValueError("diffusion must be finite and non-negative")
            gyro = (
                2.0
                * math.pi
                * float(self.material.Ms)
                * float(self.geom.L)
                / float(self.material.gamma)
            )
            diffusion_model = "explicit"

        rng = np.random.default_rng(seed)

        state = np.zeros((t_eval.size, 2), dtype=float)
        state[0, :] = initial

        for idx in range(1, t_eval.size):
            t_prev = float(t_eval[idx - 1])
            h = float(t_eval[idx] - t_eval[idx - 1])
            prev = state[idx - 1, :]

            # Resolve field at this time-step
            B = self._field_at(t_prev, B_func)
            s_eq = self.s_eq(field_state=B)

            # Relative coordinates for gyrotropic dynamics
            s_rel = prev - s_eq
            u_prev = float(np.linalg.norm(s_rel))
            u_eff = max(u_prev, 1e-15)

            j_prev = float(J_func(t_prev))
            chi_val = self.chi(j_prev)
            w0_eff = self.omega0_eff(j_prev, field_state=B)
            omega_val = w0_eff * (1.0 + self.N * u_eff**2)
            radial = chi_val - self.d(u_eff) * omega_val

            theta = float(self.polarity) * omega_val * h
            grow = math.exp(radial * h)
            cth = math.cos(theta)
            sth = math.sin(theta)

            # Rotate relative coordinates, then shift back
            x_rot = cth * s_rel[0] - sth * s_rel[1]
            y_rot = sth * s_rel[0] + cth * s_rel[1]
            deterministic = s_eq + grow * np.array([x_rot, y_rot], dtype=float)

            sigma = math.sqrt(2.0 * diffusion_eff * h)
            noise = sigma * rng.standard_normal(2)
            proposal = deterministic + noise

            # Clamp: w bezwzględnym układzie dysku
            u_abs = float(np.linalg.norm(proposal))
            if u_abs >= clamp:
                proposal = proposal * (clamp / max(u_abs, 1e-30))

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
                "domega0_dJ": self.domega0_dJ,
                "field": self.field,
                "field_cal": self.field_cal,
            },
            metadata={
                "mode": "CPP-SDE",
                "reference": "Guslienko et al. + Langevin reduction",
                "diffusion": float(diffusion_eff),
                "diffusion_units": "normalized_coordinate^2/s",
                "diffusion_model": diffusion_model,
                "gyrocoefficient_kg_per_s": float(gyro),
                "temperature_k": temperature,
                "noise_scale": amplitude_scale,
                "seed": seed,
                "dt": step,
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
    domega0_dJ: float,
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
        domega0_dJ=float(domega0_dJ),
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
    fit_domega0_dJ: bool = False,
    initial_domega0_dJ: float = 0.0,
    fit_chi_scale: bool = False,
    initial_chi_scale: float = 1.0,
    allow_edge: bool = False,
    domega0_dJ_bounds: tuple[float, float] = (-0.1, 0.1),
) -> ThieleFJFitResult:
    """
    Fit ``omega0``, ``N`` (optionally ``chi_scale``) of CPP Thiele model to measured ``f(J)`` points.
    """
    j = np.asarray(J_data, dtype=float).ravel()
    f = np.asarray(f_data_hz, dtype=float).ravel()
    if j.size != f.size:
        raise ValueError("J_data and f_data_hz must have the same length")
    n_parameters = 2 + int(fit_domega0_dJ) + int(fit_chi_scale)
    min_points = max(3, n_parameters + 1)
    if j.size < min_points:
        raise ValueError(
            f"At least {min_points} points are required for {n_parameters} fitted "
            "parameters"
        )

    finite = np.isfinite(j) & np.isfinite(f)
    j = j[finite]
    f = f[finite]
    if j.size < min_points:
        raise ValueError(
            f"At least {min_points} finite points are required for "
            f"{n_parameters} fitted parameters"
        )
    if np.any(f <= 0.0):
        raise ValueError("f_data_hz must contain positive frequencies")
    if np.unique(j).size < min_points:
        raise ValueError(
            f"At least {min_points} distinct current-density values are required"
        )

    omega0_init = float(
        omega0_novosad(material, geom) if initial_omega0 is None else initial_omega0
    )
    n_init = float(initial_N)
    dj_init = float(initial_domega0_dJ)
    chi_init = float(initial_chi_scale)
    if not all(
        np.isfinite(value) for value in (omega0_init, n_init, dj_init, chi_init)
    ):
        raise ValueError("initial fit parameters must be finite")
    if omega0_init <= 0.0 or chi_init <= 0.0:
        raise ValueError("initial_omega0 and initial_chi_scale must be positive")
    frequency_scale = max(float(np.median(np.abs(f))), 1.0)

    def _objective(params: np.ndarray) -> float:
        omega0_val = max(float(params[0]), 1e6)
        n_val = float(params[1])
        dj_val = dj_init
        chi_val = chi_init

        idx = 2
        if fit_domega0_dJ:
            dj_val = float(params[idx])
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
            dj_val,
            chi_val,
            allow_edge=allow_edge,
        )
        if not np.all(np.isfinite(f_pred)):
            return 1e30
        residual = (f_pred - f) / frequency_scale
        return float(np.mean(residual**2))

    x0 = np.array([omega0_init, n_init], dtype=float)
    bounds = [(1e6, 1e14), (-5.0, 5.0)]
    if fit_domega0_dJ:
        dj_lower, dj_upper = sorted(
            (float(domega0_dJ_bounds[0]), float(domega0_dJ_bounds[1]))
        )
        if not (
            np.isfinite(dj_lower) and np.isfinite(dj_upper) and dj_upper > dj_lower
        ):
            raise ValueError("domega0_dJ_bounds must be finite and non-degenerate")
        if not dj_lower <= dj_init <= dj_upper:
            raise ValueError("initial_domega0_dJ must lie within domega0_dJ_bounds")
        x0 = np.append(x0, dj_init)
        bounds.append((dj_lower, dj_upper))
    if fit_chi_scale:
        x0 = np.append(x0, chi_init)
        bounds.append((0.1, 20.0))

    # Deterministic bounded search to avoid poor local minima without the
    # combinatorial 1.4M-evaluation grid used by the historical four-parameter
    # path.  The budget stays bounded when optional parameters are enabled.
    omega_low = max(1e6, 0.2 * omega0_init)
    omega_high = max(omega_low * 1.01, 2.5 * omega0_init)
    omega_grid = np.geomspace(omega_low, omega_high, 25)
    n_grid = np.linspace(-2.0, 2.0, 31)

    best = x0.copy()
    best_cost = float(_objective(best))
    n_evaluations = 1
    for omega_val in omega_grid:
        for n_val in n_grid:
            candidate = x0.copy()
            candidate[0] = omega_val
            candidate[1] = n_val
            score = float(_objective(candidate))
            n_evaluations += 1
            if score < best_cost:
                best_cost = score
                best = candidate

    if fit_domega0_dJ or fit_chi_scale:
        rng = np.random.default_rng(0)
        for _ in range(2048):
            candidate = x0.copy()
            candidate[0] = math.exp(
                rng.uniform(math.log(omega_low), math.log(omega_high))
            )
            candidate[1] = rng.uniform(-2.0, 2.0)
            idx = 2
            if fit_domega0_dJ:
                candidate[idx] = rng.uniform(dj_lower, dj_upper)
                idx += 1
            if fit_chi_scale:
                candidate[idx] = math.exp(rng.uniform(math.log(0.1), math.log(20.0)))
            score = float(_objective(candidate))
            n_evaluations += 1
            if score < best_cost:
                best_cost = score
                best = candidate

    success = bool(np.isfinite(best_cost) and best_cost < 1e30)
    status = "bounded_deterministic_search"
    try:
        from scipy.optimize import minimize

        scales = np.array(
            [
                max(omega0_init, 1e6),
                1.0,
                *([max(abs(dj_lower), abs(dj_upper), 1e-6)] if fit_domega0_dJ else []),
                *([1.0] if fit_chi_scale else []),
            ],
            dtype=float,
        )
        scaled_bounds = [
            (lower / scale, upper / scale)
            for (lower, upper), scale in zip(bounds, scales)
        ]
        opt = minimize(
            lambda scaled: _objective(np.asarray(scaled, dtype=float) * scales),
            best / scales,
            method="L-BFGS-B",
            bounds=scaled_bounds,
        )
        if opt.success and np.isfinite(opt.fun):
            best = np.asarray(opt.x, dtype=float) * scales
            best_cost = float(opt.fun)
            success = True
            status = str(opt.message)
        else:
            status = f"{status}; scipy_local_failed: {opt.message}"
    except Exception:
        status = f"{status}; scipy_unavailable"

    omega0_fit = max(float(best[0]), 1e6)
    n_fit = float(best[1])
    idx = 2
    dj_fit = dj_init
    chi_fit = chi_init
    if fit_domega0_dJ:
        dj_fit = float(best[idx])
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
        dj_fit,
        chi_fit,
        allow_edge=allow_edge,
    )

    valid_mask = np.isfinite(f_fit)
    if np.all(valid_mask):
        rmse = float(np.sqrt(np.mean((f_fit - f) ** 2)))
    else:
        rmse = float("nan")
        success = False
        status = f"{status}; invalid_predictions"

    return ThieleFJFitResult(
        model_name="CPP Thiele f(J) fit",
        omega0=omega0_fit,
        N=n_fit,
        domega0_dJ=dj_fit,
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
            "fit_domega0_dJ": bool(fit_domega0_dJ),
            "fit_chi_scale": bool(fit_chi_scale),
            "domega0_dJ_bounds": tuple(float(value) for value in domega0_dJ_bounds),
        },
        metadata={
            "allow_edge": bool(allow_edge),
            "n_points": int(j.size),
            "n_fitted_parameters": int(n_parameters),
            "normalized_mean_square_cost": float(best_cost),
            "frequency_scale_hz": float(frequency_scale),
            "search_evaluations": int(n_evaluations),
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
                ax.plot(
                    R_nm * np.cos(theta), R_nm * np.sin(theta), "k--", alpha=0.3, lw=0.8
                )
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
            f"R_disk = {r.disk_radius * 1e9:.1f} nm",
            f"Duration = {r.t[-1] * 1e9:.1f} ns" if len(r.t) > 0 else "",
            f"Points = {len(r.t)}",
            "",
            f"f_ss = {r.steady_state_frequency_ghz:.3f} GHz",
            f"r_ss = {r.steady_state_radius_m * 1e9:.1f} nm",
            f"f_dom = {r.dominant_frequency_ghz:.3f} GHz",
            f"Δf = {r.linewidth_ghz * 1e3:.1f} MHz",
            f"Rotation: {r.rotation_sense}",
        ]
        info_ax.text(
            0.05,
            0.95,
            "\n".join(lines),
            transform=info_ax.transAxes,
            va="top",
            ha="left",
            fontfamily="monospace",
            fontsize=10,
        )

        fig.tight_layout()
        if show:
            plt.show()
        return fig
