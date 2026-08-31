# ruff: noqa: N802, N803, N806, PLR0913
"""
Field-resolved Thiele model for vortex STNO/MTJ trajectories.

This module implements a two-dimensional collective-coordinate solver for a
magnetic vortex core in a circular free layer under a vector magnetic field
B = (Bx, By, Bz) and a CPP spin-polarized current.  It is intended as the
field-resolved counterpart of the reduced circular ``CPPThieleModel`` in
``thiele.py``.

Conventions
-----------
All quantities are SI:

* core position X = (X, Y) in metres,
* magnetic flux density B in tesla,
* current density J in A/m^2,
* energy U in joule,
* forces in newton,
* gyro and damping coefficients in kg/s,
* angular frequencies in rad/s.

The implemented equation is

    (D I + p G J2) Xdot = F_ST - grad_X U,

where p is the vortex core polarity (+1 or -1), G > 0 is the magnitude of the
B-convention gyrocoefficient 2*pi*Ms*L/gamma, D > 0 is the damping coefficient,
and J2 is the in-plane operator z_hat x v = (-v_y, v_x).  With J = 0,
B = 0, D << G and U = kappa |X|^2/2, the model gives

    Xdot ~= p * (kappa/G) * J2 X - (D/G) * (kappa/G) * X,

therefore it reduces to the usual gyrotropic motion with polarity-dependent
sense and Gilbert damping.

The model is deliberately semi-analytical.  It includes physically constrained
closed-form terms, but exposes the coefficients that are normally obtained from
micromagnetic calibration sweeps.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import numpy as np

from .base import AnalyticalResult
from .constants import GAMMA_E
from .thiele import DiskGeometry, ExternalField, ExternalFieldLike, MaterialParams

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_HBAR: float = 1.054571817e-34  # J s
_E_CHARGE: float = 1.602176634e-19  # C, positive elementary charge

J2: np.ndarray = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=float)
I2: np.ndarray = np.eye(2, dtype=float)

CurrentFunc: TypeAlias = Callable[[float], float]
FieldFunc: TypeAlias = Callable[[float], ExternalFieldLike]
PolarizerFunc: TypeAlias = Callable[
    [float, ExternalField], tuple[float, float, float] | np.ndarray
]


# ---------------------------------------------------------------------------
# Waveform helpers
# ---------------------------------------------------------------------------


def current_dc(J_dc: float) -> CurrentFunc:
    """Return a constant current-density waveform J(t) = J_dc [A/m^2]."""
    value = float(J_dc)
    if not np.isfinite(value):
        raise ValueError("J_dc must be finite [A/m^2]")

    def _j(_t: float) -> float:
        return value

    return _j


def field_dc(B_ext: ExternalFieldLike = 0.0) -> FieldFunc:
    """Return a constant field waveform B(t) = const [T]."""
    B = ExternalField.from_any(B_ext)

    def _b(_t: float) -> ExternalField:
        return B

    return _b


def normalize_polarizer(
    polarizer: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """Return a normalized three-component polarizer vector."""
    p: Any = np.asarray(polarizer, dtype=float).reshape(-1)
    if p.size == 2:
        p = np.array([p[0], p[1], 0.0], dtype=float)
    if p.size < 3:
        raise ValueError("polarizer must have two or three components")
    p = p[:3]
    n = float(np.linalg.norm(p))
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError("polarizer must be finite and non-zero")
    return p / n


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldResolvedCalibration:
    """
    Calibration coefficients for the vector-field Thiele model.

    Parameters
    ----------
    d0, d1:
        Dimensionless damping coefficients in ``D/G = d0 + d1 u^2``.  If ``d0``
        or ``d1`` is ``None``, analytical rigid-vortex estimates are used.
    domega_dBz, domega_dBz2:
        Out-of-plane-field frequency response.  ``domega_dBz`` has units
        rad/(s T); the model applies it as ``p*domega_dBz*Bz``.  The quadratic
        coefficient has units rad/(s T^2).
    domega_dJ:
        Current/Oersted linear frequency response in rad/(s (A/m^2)).
    N_Bz, N_Bz2, N_ip2:
        Optional field dependence of the nonlinear frequency shift ``N`` with
        units 1/T, 1/T^2 and 1/T^2.
    G_u2, G_Bz, G_Bz2:
        Optional corrections to the gyrocoefficient ``G``.
    D_Bz, D_ip2:
        Optional multiplicative corrections to damping.
    seq_per_T:
        Normalized in-plane-field equilibrium susceptibility.  If supplied, the
        small-field equilibrium at J=0 is
        ``X_eq/R = chirality * seq_per_T * (z_hat x B_parallel)``.
    lambda_H_per_T:
        Direct in-plane-field force coefficient in N/T for
        ``F_H = lambda_H_per_T * (z_hat x B_parallel)``.  Takes precedence over
        ``seq_per_T`` and ``xi_H`` when not ``None``.
    xi_H:
        Dimensionless analytical Zeeman coupling estimate used only if neither
        ``lambda_H_per_T`` nor ``seq_per_T`` is supplied.  The estimate is
        ``lambda_H_per_T = chirality*pi*xi_H*Ms*L*R``.
    k_ip_iso_per_T2, k_ip_aniso_per_T2:
        In-plane-field-induced stiffness changes.  The isotropic term scales
        ``K`` by ``1 + k_ip_iso_per_T2*|B_parallel|^2``.  The anisotropic term
        adds opposite curvature along and perpendicular to ``B_parallel``.
    stt_z_efficiency:
        Dimensionless multiplier of the analytical perpendicular Slonczewski
        force coefficient ``pi*hbar*P/(2e)``.
    stt_z_sign:
        Current-sign convention for the perpendicular Slonczewski force.  Use
        ``+1`` for the Dussaux electron-flow convention in which positive J pumps
        a vortex with ``p*pz > 0``.  Use ``-1`` to reproduce the sign convention
        of the current reduced ``CPPThieleModel``.
    lambda_parallel_dl_per_J:
        Force coefficient for damping-like torque from an in-plane polarizer:
        ``F = lambda_parallel_dl_per_J * J * p_parallel``.  Units N/(A/m^2).
    lambda_parallel_flt_per_J:
        Force coefficient for field-like torque from an in-plane polarizer:
        ``F = lambda_parallel_flt_per_J * J * (z_hat x p_parallel)``.
        Units N/(A/m^2).
    lambda_z_fieldlike_per_J_per_m:
        Optional radial field-like/conservative-like coefficient for a pz
        polarizer: ``F = lambda_z_fieldlike_per_J_per_m * J * pz * X``.
        Units N/((A/m^2) m).  Default zero.
    min_omega_factor:
        Positive floor applied to ``omega0_eff`` as a fraction of the zero-field
        ``omega0`` to avoid invalid vortex-state dynamics outside calibration.
    """

    d0: float | None = None
    d1: float | None = None

    domega_dBz: float = 0.0
    domega_dBz2: float = 0.0
    domega_dJ: float = 0.0

    N_Bz: float = 0.0
    N_Bz2: float = 0.0
    N_ip2: float = 0.0

    G_u2: float = 0.0
    G_Bz: float = 0.0
    G_Bz2: float = 0.0

    D_Bz: float = 0.0
    D_ip2: float = 0.0

    seq_per_T: float = 0.0
    lambda_H_per_T: float | None = None
    xi_H: float | None = None

    k_ip_iso_per_T2: float = 0.0
    k_ip_aniso_per_T2: float = 0.0

    stt_z_efficiency: float = 1.0
    stt_z_sign: float = 1.0
    lambda_parallel_dl_per_J: float = 0.0
    lambda_parallel_flt_per_J: float = 0.0
    lambda_z_fieldlike_per_J_per_m: float = 0.0

    min_omega_factor: float = 0.02
    saturation: SaturationCalibration = field(
        default_factory=lambda: SaturationCalibration()
    )
    oersted: OerstedCalibration = field(default_factory=lambda: OerstedCalibration())
    thermal: ThermalCalibration = field(default_factory=lambda: ThermalCalibration())
    current_drive: CurrentDrive = field(default_factory=lambda: CurrentDrive())
    Bz_saturation_T: float | None = None
    Bz_mode: Literal["polynomial", "saturation_field"] = "polynomial"

    def __post_init__(self) -> None:
        nested = {"saturation", "oersted", "thermal", "current_drive", "Bz_mode"}
        for name, value in vars(self).items():
            if name in nested or value is None:
                continue
            if not np.isfinite(float(value)):
                raise ValueError(f"calibration coefficient {name} must be finite")
        if self.d0 is not None and self.d0 < 0.0:
            raise ValueError("calibration d0 must be non-negative")
        if self.stt_z_efficiency < 0.0:
            raise ValueError("stt_z_efficiency must be non-negative")
        if self.stt_z_sign == 0.0:
            raise ValueError("stt_z_sign must be non-zero")
        if self.min_omega_factor < 0.0:
            raise ValueError("min_omega_factor must be non-negative")
        if self.Bz_saturation_T is not None and self.Bz_saturation_T <= 0.0:
            raise ValueError("Bz_saturation_T must be positive when provided")
        if self.Bz_mode not in {"polynomial", "saturation_field"}:
            raise ValueError("Bz_mode must be 'polynomial' or 'saturation_field'")


@dataclass(frozen=True)
class CurrentDrive:
    """Electrical-current to current-density conversion for a CPP pillar."""

    area_m2: float | None = None
    current_sign: float = 1.0
    name: str = "CPP uniform"

    def __post_init__(self) -> None:
        if self.area_m2 is not None and (
            not np.isfinite(float(self.area_m2)) or self.area_m2 <= 0.0
        ):
            raise ValueError("area_m2 must be finite and positive when provided")
        if not np.isfinite(float(self.current_sign)) or self.current_sign == 0.0:
            raise ValueError("current_sign must be finite and non-zero")
        if not str(self.name).strip():
            raise ValueError("current-drive name must be non-empty")

    def area(self, geom: DiskGeometry) -> float:
        """Return effective current area [m^2]."""
        return float(
            math.pi * geom.R * geom.R if self.area_m2 is None else self.area_m2
        )

    def J_from_I(self, I_A: float, geom: DiskGeometry) -> float:
        """Convert electrical current [A] into signed current density [A/m^2]."""
        area = self.area(geom)
        if not np.isfinite(area) or area <= 0.0:
            raise ValueError("current area must be finite and positive")
        return float(self.current_sign) * float(I_A) / area


@dataclass(frozen=True)
class SaturationCalibration:
    """Amplitude-saturation terms for nonlinear damping and frequency."""

    d2: float | None = None
    d4: float = 0.0
    d_edge: float = 0.0
    u_damp_max: float = 0.85
    edge_epsilon: float = 1e-6
    K_edge: float = 0.0
    u_edge_max: float = 0.90
    N4: float = 0.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value is not None and not np.isfinite(float(value)):
                raise ValueError(f"saturation coefficient {name} must be finite")
        if self.u_damp_max <= 0.0:
            raise ValueError("u_damp_max must be positive")
        if not 0.0 < self.edge_epsilon < 1.0:
            raise ValueError("edge_epsilon must lie in (0, 1)")
        if self.K_edge < 0.0:
            raise ValueError("K_edge must be non-negative")
        if not 0.0 < self.u_edge_max <= 1.0:
            raise ValueError("u_edge_max must lie in (0, 1]")


@dataclass(frozen=True)
class OerstedCalibration:
    """Current-induced Oersted corrections to the vortex potential."""

    K2_per_J: float = 0.0
    K4_per_J: float = 0.0
    K6_per_J: float = 0.0
    direct_omega_per_J: float = 0.0
    direct_omega_sat_J: float | None = None

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value is not None and not np.isfinite(float(value)):
                raise ValueError(f"Oersted coefficient {name} must be finite")
        if self.direct_omega_sat_J is not None and self.direct_omega_sat_J <= 0.0:
            raise ValueError("direct_omega_sat_J must be positive when provided")


@dataclass(frozen=True)
class ThermalCalibration:
    """Optional Joule-heating frequency shift."""

    dT_dI2: float = 0.0
    domega_dT: float = 0.0
    dMs_dT_over_Ms: float = 0.0
    thermal_sat_I: float | None = None

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value is not None and not np.isfinite(float(value)):
                raise ValueError(f"thermal coefficient {name} must be finite")
        if self.dT_dI2 < 0.0:
            raise ValueError("dT_dI2 must be non-negative")
        if self.thermal_sat_I is not None and self.thermal_sat_I <= 0.0:
            raise ValueError("thermal_sat_I must be positive when provided")


@dataclass(frozen=True)
class FrequencyExtractionConfig:
    """Defaults for trajectory frequency extraction."""

    transient_fraction: float = 0.5
    center: Literal["mean", "conservative_equilibrium", "given"] = "mean"
    method: Literal["geometric", "fft_resistance", "fft_x", "fft_y"] = "geometric"
    window: Literal["hann", "none"] = "hann"

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.transient_fraction)) or not (
            0.0 <= self.transient_fraction < 1.0
        ):
            raise ValueError("transient_fraction must lie in [0, 1)")
        if self.center not in {"mean", "conservative_equilibrium", "given"}:
            raise ValueError("unsupported orbit-center mode")
        if self.method not in {"geometric", "fft_resistance", "fft_x", "fft_y"}:
            raise ValueError("unsupported frequency-extraction method")
        if self.window not in {"hann", "none"}:
            raise ValueError("window must be 'hann' or 'none'")


@dataclass
class FieldResolvedTrajectoryResult(AnalyticalResult):
    """Trajectory returned by :class:`FieldResolvedCPPThieleModel`."""

    t: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    x: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    y: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sx: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    sy: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    disk_radius: float = 0.0

    @property
    def X(self) -> np.ndarray:
        """Position array with shape (n, 2) in metres."""
        return np.column_stack([self.x, self.y])

    @property
    def s(self) -> np.ndarray:
        """Normalized position array with shape (n, 2)."""
        return np.column_stack([self.sx, self.sy])

    @property
    def r(self) -> np.ndarray:
        """Radial distance |X| [m]."""
        return np.hypot(self.x, self.y)

    @property
    def u(self) -> np.ndarray:
        """Normalized radius u = |X|/R."""
        return self.r / self.disk_radius if self.disk_radius > 0.0 else self.r

    @property
    def phi(self) -> np.ndarray:
        """Azimuthal angle around the disk centre [rad]."""
        return np.angle(self.x + 1j * self.y)

    @property
    def phi_unwrapped(self) -> np.ndarray:
        """Unwrapped azimuthal angle [rad]."""
        return np.unwrap(self.phi)

    @property
    def omega_inst(self) -> np.ndarray:
        """Instantaneous angular frequency dphi/dt [rad/s]."""
        if self.t.size < 2:
            return np.array([], dtype=float)
        return np.gradient(self.phi_unwrapped, self.t)

    @property
    def frequency_inst_hz(self) -> np.ndarray:
        """Instantaneous frequency [Hz]."""
        return self.omega_inst / (2.0 * math.pi)

    @property
    def velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Numerical velocity components (vx, vy) [m/s]."""
        if self.t.size < 2:
            z = np.zeros_like(self.x)
            return z, z
        return np.gradient(self.x, self.t), np.gradient(self.y, self.t)

    @property
    def speed(self) -> np.ndarray:
        """Speed |dX/dt| [m/s]."""
        vx, vy = self.velocity
        return np.hypot(vx, vy)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class FieldResolvedCPPThieleModel:
    """
    Field-resolved CPP Thiele model for vortex-core trajectories.

    This class is a trajectory model, not only an amplitude model.  It supports
    vector fields ``Bx``, ``By`` and ``Bz`` through an explicit potential and
    explicit force terms.  The model remains a rigid-vortex/semi-analytical
    approximation and must be calibrated against micromagnetics when used
    quantitatively.
    """

    def __init__(
        self,
        material: MaterialParams,
        geom: DiskGeometry,
        omega0: float,
        *,
        N: float = 0.25,
        polarity: int = 1,
        chirality: int = 1,
        polarizer: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 1.0),
        calibration: FieldResolvedCalibration | None = None,
    ) -> None:
        self.material = material
        self.geom = geom
        self.omega0 = float(omega0)
        self.N = float(N)
        self.polarity = 1 if int(polarity) >= 0 else -1
        self.chirality = 1 if int(chirality) >= 0 else -1
        self.polarizer = normalize_polarizer(polarizer)
        self.cal = (
            calibration if calibration is not None else FieldResolvedCalibration()
        )

        if not np.isfinite(self.omega0) or self.omega0 <= 0.0:
            raise ValueError("omega0 must be a positive angular frequency [rad/s]")
        if self.geom.R <= 0.0 or self.geom.L <= 0.0:
            raise ValueError("geom.R and geom.L must be positive")

        self._setup()

    # ------------------------------------------------------------------
    # Base coefficients and unitful derived quantities
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        mat = self.material
        geo = self.geom
        gamma = float(getattr(mat, "gamma", GAMMA_E))
        if gamma <= 0.0:
            raise ValueError("material.gamma must be positive [rad/(s T)]")

        self.G0 = 2.0 * math.pi * float(mat.Ms) * float(geo.L) / gamma
        """Positive gyrocoefficient magnitude [kg/s] in the B-field convention."""

        Rc = float(geo.Rc(mat))
        ratio = float(geo.R) / max(Rc, 1e-12)
        d0_default = float(mat.alpha) * (5.0 + 4.0 * math.log(max(ratio, 1.1))) / 8.0
        d1_default = (11.0 / 6.0) * float(mat.alpha)
        self.d0 = d0_default if self.cal.d0 is None else float(self.cal.d0)
        self.d1 = d1_default if self.cal.d1 is None else float(self.cal.d1)
        self.current_drive = self.cal.current_drive
        if not np.isfinite(self.d0) or self.d0 < 0.0:
            raise ValueError("calibrated d0 must be finite and non-negative")
        if not np.isfinite(self.d1):
            raise ValueError("calibrated d1 must be finite")

        self.kappa0 = self.G0 * self.omega0
        """Zero-field linear stiffness [N/m]."""

        self.kappa4_0 = self.kappa0 * self.N
        """Zero-field quartic stiffness coefficient [N/m]."""

        self.lambda_stt_z = (
            float(self.cal.stt_z_sign)
            * float(self.cal.stt_z_efficiency)
            * math.pi
            * _HBAR
            * float(self.material.P)
            / (2.0 * _E_CHARGE)
        )
        """Perpendicular Slonczewski force coefficient [N/((A/m^2) m)]."""

    @property
    def chi_z_per_J(self) -> float:
        """Perpendicular Slonczewski pumping prefactor ``lambda_stt_z/G0`` [m^2/(A s)]."""
        return self.lambda_stt_z / self.G0

    # ------------------------------------------------------------------
    # Field/current-dependent scalar coefficients
    # ------------------------------------------------------------------

    def _field(self, value: ExternalFieldLike | ExternalField) -> ExternalField:
        return ExternalField.from_any(value)

    def omega0_Bz(self, B: ExternalFieldLike | ExternalField = 0.0) -> float:
        """Out-of-plane-field contribution to the small-signal frequency [rad/s]."""
        bf = self._field(B)
        if (
            self.cal.Bz_mode == "saturation_field"
            and self.cal.Bz_saturation_T is not None
            and self.cal.Bz_saturation_T > 0.0
        ):
            hz = bf.Bz_T / float(self.cal.Bz_saturation_T)
            return float(self.omega0 * (1.0 + self.polarity * hz))
        return float(
            self.omega0
            + self.polarity * float(self.cal.domega_dBz) * bf.Bz_T
            + float(self.cal.domega_dBz2) * bf.Bz_T * bf.Bz_T
        )

    def direct_current_omega_shift(
        self,
        J: float,
        I_A: float | None = None,
        B: ExternalFieldLike | ExternalField = 0.0,  # noqa: ARG002
    ) -> float:
        """Current-dependent frequency shift not represented by stiffness [rad/s]."""
        shift = float(self.cal.domega_dJ) * float(J)
        oe = self.cal.oersted
        if oe.direct_omega_sat_J is not None and oe.direct_omega_sat_J > 0.0:
            js = float(oe.direct_omega_sat_J)
            shift += float(oe.direct_omega_per_J) * js * math.tanh(float(J) / js)
        else:
            shift += float(oe.direct_omega_per_J) * float(J)

        if I_A is not None:
            th = self.cal.thermal
            if th.thermal_sat_I is not None and th.thermal_sat_I > 0.0:
                scale = 1.0 - math.exp(-((float(I_A) / float(th.thermal_sat_I)) ** 2))
                dT = float(th.dT_dI2) * float(th.thermal_sat_I) ** 2 * scale
            else:
                dT = float(th.dT_dI2) * float(I_A) * float(I_A)
            shift += float(th.domega_dT) * dT
        return float(shift)

    def omega0_eff(
        self,
        J: float,
        B: ExternalFieldLike | ExternalField = 0.0,
        I_A: float | None = None,
    ) -> float:
        """Linear angular frequency response ``omega0(J,Bz)`` [rad/s]."""
        omega = self.omega0_Bz(B) + self.direct_current_omega_shift(J, I_A, B)
        floor = max(float(self.cal.min_omega_factor), 0.0) * self.omega0
        if omega < floor:
            if floor <= 0.0:
                raise ValueError(
                    "calibration produced a non-positive gyrotropic frequency"
                )
            if not getattr(self, "_omega_floor_warned", False):
                warnings.warn(
                    "calibration produced omega0 below its configured validity "
                    f"floor; clipping to {self.cal.min_omega_factor:g}*omega0",
                    UserWarning,
                    stacklevel=2,
                )
                self._omega_floor_warned = True
            return float(floor)
        return float(omega)

    def N_eff(self, B: ExternalFieldLike | ExternalField = 0.0) -> float:
        """Effective nonlinear frequency shift ``N(B)`` [dimensionless]."""
        bf = self._field(B)
        b2 = bf.Bx_T * bf.Bx_T + bf.By_T * bf.By_T
        return float(
            self.N
            + self.polarity * float(self.cal.N_Bz) * bf.Bz_T
            + float(self.cal.N_Bz2) * bf.Bz_T * bf.Bz_T
            + float(self.cal.N_ip2) * b2
        )

    def G_mag(
        self, X: np.ndarray, J: float, B: ExternalFieldLike | ExternalField = 0.0
    ) -> float:  # noqa: ARG002
        """Positive gyrocoefficient magnitude ``G`` [kg/s]."""
        bf = self._field(B)
        u2 = float(np.dot(X, X)) / max(self.geom.R * self.geom.R, 1e-30)
        scale = (
            1.0
            + float(self.cal.G_u2) * u2
            + self.polarity * float(self.cal.G_Bz) * bf.Bz_T
            + float(self.cal.G_Bz2) * bf.Bz_T * bf.Bz_T
        )
        value = self.G0 * scale
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                "gyro calibration produced a non-positive G; "
                "the requested state is outside its valid range"
            )
        return float(value)

    def damping_ratio(
        self, X: np.ndarray, J: float, B: ExternalFieldLike | ExternalField = 0.0
    ) -> float:
        """Return dimensionless damping ratio ``d = D/G``."""
        bf = self._field(B)
        x = np.asarray(X, dtype=float).reshape(2)
        u2 = float(np.dot(x, x)) / max(self.geom.R * self.geom.R, 1e-30)
        b2 = bf.Bx_T * bf.Bx_T + bf.By_T * bf.By_T
        sat = self.cal.saturation
        d2 = self.d1 if sat.d2 is None else float(sat.d2)
        d = self.d0 + d2 * u2 + float(sat.d4) * u2 * u2
        if float(sat.d_edge) != 0.0:
            umax2 = max(float(sat.u_damp_max) * float(sat.u_damp_max), 1e-12)
            denom = max(umax2 - u2, float(sat.edge_epsilon))
            d += float(sat.d_edge) * u2 / denom
        field_scale = (
            1.0
            + self.polarity * float(self.cal.D_Bz) * bf.Bz_T
            + float(self.cal.D_ip2) * b2
        )
        value = d * field_scale
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(
                "damping calibration produced D/G < 0; "
                "the requested state is outside its valid range"
            )
        return float(value)

    def D_coeff(
        self, X: np.ndarray, J: float, B: ExternalFieldLike | ExternalField = 0.0
    ) -> float:  # noqa: ARG002
        """Damping coefficient ``D`` [kg/s]."""
        return float(self.G_mag(X, J, B) * self.damping_ratio(X, J, B))

    # ------------------------------------------------------------------
    # Conservative potential and force terms
    # ------------------------------------------------------------------

    def oersted_stiffness_terms(self, J: float) -> tuple[float, float, float]:
        """Return ``C*J*(K2,K4,K6)`` Oersted stiffness additions [N/m]."""
        scale = float(self.chirality) * float(J)
        oe = self.cal.oersted
        return (
            scale * float(oe.K2_per_J),
            scale * float(oe.K4_per_J),
            scale * float(oe.K6_per_J),
        )

    def K2_tensor(
        self,
        X: np.ndarray,
        J: float,
        B: ExternalFieldLike | ExternalField = 0.0,
        I_A: float | None = None,
    ) -> np.ndarray:  # noqa: ARG002
        """Linear stiffness tensor ``K2`` [N/m]."""
        bf = self._field(B)
        # The potential energy is independent of the kinetic gyrocoefficient.
        # G0 converts the calibrated angular-frequency law to stiffness; any
        # G(X,B) correction then acts only in the Thiele kinetic matrix.
        kappa = self.G0 * self.omega0_eff(J, bf, I_A)

        bvec = np.array([bf.Bx_T, bf.By_T], dtype=float)
        b2 = float(np.dot(bvec, bvec))
        K = kappa * (1.0 + float(self.cal.k_ip_iso_per_T2) * b2) * I2.copy()

        if b2 > 0.0 and abs(float(self.cal.k_ip_aniso_per_T2)) > 0.0:
            bhat = bvec / math.sqrt(b2)
            phat = J2 @ bhat
            anis = np.outer(bhat, bhat) - np.outer(phat, phat)
            K = K + kappa * float(self.cal.k_ip_aniso_per_T2) * b2 * anis

        K2_oe, _, _ = self.oersted_stiffness_terms(J)
        if K2_oe != 0.0:
            K = K + K2_oe * I2
        return np.asarray(K, dtype=float)

    def K4_scalar(
        self,
        X: np.ndarray,
        J: float,
        B: ExternalFieldLike | ExternalField = 0.0,
        I_A: float | None = None,
    ) -> float:  # noqa: ARG002
        """Quartic stiffness coefficient ``K4`` [N/m]."""
        _, K4_oe, _ = self.oersted_stiffness_terms(J)
        return float(self.G0 * self.omega0_eff(J, B, I_A) * self.N_eff(B) + K4_oe)

    def K6_scalar(
        self,
        X: np.ndarray,
        J: float,
        B: ExternalFieldLike | ExternalField = 0.0,
        I_A: float | None = None,
    ) -> float:  # noqa: ARG002
        """Sixth-order stiffness coefficient ``K6`` [N/m]."""
        _, _, K6_oe = self.oersted_stiffness_terms(J)
        return float(
            self.G0 * self.omega0_eff(J, B, I_A) * float(self.cal.saturation.N4) + K6_oe
        )

    def lambda_H(self, J: float, B: ExternalFieldLike | ExternalField = 0.0) -> float:  # noqa: ARG002
        """In-plane field force coefficient [N/T]."""
        if self.cal.lambda_H_per_T is not None:
            return float(self.cal.lambda_H_per_T)

        if abs(float(self.cal.seq_per_T)) > 0.0:
            # Enforce X_eq/R = chirality*seq_per_T*(z x B_parallel)
            return float(
                self.kappa0 * self.geom.R * self.chirality * float(self.cal.seq_per_T)
            )

        if self.cal.xi_H is not None:
            return float(
                self.chirality
                * math.pi
                * float(self.cal.xi_H)
                * self.material.Ms
                * self.geom.L
                * self.geom.R
            )

        return 0.0

    def field_force(
        self, J: float, B: ExternalFieldLike | ExternalField = 0.0
    ) -> np.ndarray:
        """Zeeman force from in-plane field, ``F_H`` [N]."""
        bf = self._field(B)
        bvec = np.array([bf.Bx_T, bf.By_T], dtype=float)
        return self.lambda_H(J, bf) * (J2 @ bvec)

    def edge_potential(self, X: np.ndarray) -> float:
        """Conservative edge barrier [J]."""
        sat = self.cal.saturation
        if float(sat.K_edge) == 0.0:
            return 0.0
        x = np.asarray(X, dtype=float).reshape(2)
        r2 = float(np.dot(x, x))
        umax2 = max(float(sat.u_edge_max) * float(sat.u_edge_max), 1e-12)
        u2 = r2 / max(self.geom.R * self.geom.R, 1e-30)
        epsilon = float(sat.edge_epsilon)
        if not np.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
            raise ValueError("saturation.edge_epsilon must lie in (0, 1)")
        raw_denom = 1.0 - u2 / umax2
        if raw_denom >= epsilon:
            return float(0.5 * float(sat.K_edge) * r2 / raw_denom)

        # Continue U linearly in r^2 beyond the regularisation point.  This
        # preserves both U and dU/d(r^2), so grad_edge_potential remains the
        # exact gradient instead of disagreeing by a factor epsilon.
        radius2_limit = self.geom.R**2 * umax2 * (1.0 - epsilon)
        potential_limit = 0.5 * float(sat.K_edge) * radius2_limit / epsilon
        continuation = (
            0.5 * float(sat.K_edge) * (r2 - radius2_limit) / (epsilon * epsilon)
        )
        return float(potential_limit + continuation)

    def grad_edge_potential(self, X: np.ndarray) -> np.ndarray:
        """Gradient of conservative edge barrier [N]."""
        sat = self.cal.saturation
        x = np.asarray(X, dtype=float).reshape(2)
        if float(sat.K_edge) == 0.0:
            return np.zeros(2, dtype=float)
        r2 = float(np.dot(x, x))
        umax2 = max(float(sat.u_edge_max) * float(sat.u_edge_max), 1e-12)
        u2 = r2 / max(self.geom.R * self.geom.R, 1e-30)
        epsilon = float(sat.edge_epsilon)
        if not np.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
            raise ValueError("saturation.edge_epsilon must lie in (0, 1)")
        denom = max(1.0 - u2 / umax2, epsilon)
        return float(sat.K_edge) * x / (denom * denom)

    def grad_potential(
        self,
        X: np.ndarray,
        J: float,
        B: ExternalFieldLike | ExternalField = 0.0,
        I_A: float | None = None,
    ) -> np.ndarray:
        """Gradient of the conservative potential, ``grad_X U`` [N]."""
        x = np.asarray(X, dtype=float).reshape(2)
        K2 = self.K2_tensor(x, J, B, I_A)
        K4 = self.K4_scalar(x, J, B, I_A)
        K6 = self.K6_scalar(x, J, B, I_A)
        r2 = float(np.dot(x, x))
        return (
            K2 @ x
            + (K4 / max(self.geom.R * self.geom.R, 1e-30)) * r2 * x
            + (K6 / max(self.geom.R**4, 1e-60)) * r2 * r2 * x
            + self.grad_edge_potential(x)
            - self.field_force(J, B)
        )

    def potential(
        self,
        X: np.ndarray,
        J: float,
        B: ExternalFieldLike | ExternalField = 0.0,
        I_A: float | None = None,
    ) -> float:
        """Conservative potential ``U(X,J,B)`` [J]."""
        x = np.asarray(X, dtype=float).reshape(2)
        K2 = self.K2_tensor(x, J, B, I_A)
        K4 = self.K4_scalar(x, J, B, I_A)
        K6 = self.K6_scalar(x, J, B, I_A)
        r2 = float(np.dot(x, x))
        return float(
            0.5 * x @ K2 @ x
            + 0.25 * (K4 / max(self.geom.R * self.geom.R, 1e-30)) * r2 * r2
            + (K6 / (6.0 * max(self.geom.R**4, 1e-60))) * r2 * r2 * r2
            + self.edge_potential(x)
            - self.field_force(J, B) @ x
        )

    # ------------------------------------------------------------------
    # Spin-torque force terms
    # ------------------------------------------------------------------

    def stt_force(
        self,
        X: np.ndarray,
        J: float,
        polarizer: tuple[float, float, float] | np.ndarray | None = None,
    ) -> np.ndarray:
        """Spin-transfer force ``F_ST`` [N]."""
        x = np.asarray(X, dtype=float).reshape(2)
        pvec = self.polarizer if polarizer is None else normalize_polarizer(polarizer)
        p_parallel = pvec[:2]
        pz = float(pvec[2])

        force = np.zeros(2, dtype=float)
        # Perpendicular component: anti-/pro-damping tangent force.
        force += self.lambda_stt_z * float(J) * pz * (J2 @ x)
        # Optional pz field-like component; useful for calibration to MuMax.
        if abs(float(self.cal.lambda_z_fieldlike_per_J_per_m)) > 0.0:
            force += float(self.cal.lambda_z_fieldlike_per_J_per_m) * float(J) * pz * x
        # In-plane polarizer terms: static force components.
        if np.dot(p_parallel, p_parallel) > 0.0:
            force += float(self.cal.lambda_parallel_dl_per_J) * float(J) * p_parallel
            force += (
                float(self.cal.lambda_parallel_flt_per_J) * float(J) * (J2 @ p_parallel)
            )
        return force

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def rhs(
        self,
        t: float,
        X: np.ndarray,
        J_func: CurrentFunc | None = None,
        B_func: FieldFunc | None = None,
        polarizer_func: PolarizerFunc | None = None,
        I_func: CurrentFunc | None = None,
    ) -> np.ndarray:
        """Right-hand side ``dX/dt`` [m/s] for ``solve_ivp``."""
        x = np.asarray(X, dtype=float).reshape(2)
        I_A = None if I_func is None else float(I_func(float(t)))
        if J_func is not None:
            J = float(J_func(float(t)))
        elif I_A is not None:
            J = self.current_drive.J_from_I(I_A, self.geom)
        else:
            J = 0.0
            I_A = 0.0
        if not np.isfinite(J) or (I_A is not None and not np.isfinite(I_A)):
            raise ValueError("current waveform returned a non-finite value")
        B = (
            ExternalField()
            if B_func is None
            else ExternalField.from_any(B_func(float(t)))
        )
        pvec = (
            self.polarizer
            if polarizer_func is None
            else normalize_polarizer(polarizer_func(float(t), B))
        )

        G = self.G_mag(x, J, B)
        D = self.D_coeff(x, J, B)
        force = self.stt_force(x, J, pvec) - self.grad_potential(x, J, B, I_A)
        A = D * I2 + float(self.polarity) * G * J2
        return np.linalg.solve(A, force)

    def radial_growth_rate_small_signal(
        self,
        J: float,
        B: ExternalFieldLike = 0.0,
        polarizer: tuple[float, float, float] | np.ndarray | None = None,
    ) -> float:
        """
        Exact small-signal radial growth rate for B_parallel=0 [1/s].

        This diagnostic is valid for the circular case.  It includes the
        ``1/(1+d0^2)`` correction absent from the simplest amplitude equation.
        """
        x = np.array([1e-12, 0.0], dtype=float)
        bf = self._field(B)
        pvec = self.polarizer if polarizer is None else normalize_polarizer(polarizer)
        G = self.G_mag(x, J, bf)
        D = self.D_coeff(x, J, bf)
        kappa = float(self.K2_tensor(x, J, bf)[0, 0])
        pz = float(pvec[2])
        lam = self.lambda_stt_z * float(J) * pz
        return float((-D * kappa + self.polarity * G * lam) / (D * D + G * G))

    def small_signal_omega_exact(
        self,
        J: float,
        B: ExternalFieldLike = 0.0,
        polarizer: tuple[float, float, float] | np.ndarray | None = None,
    ) -> float:
        """Exact small-signal angular velocity for the circular case [rad/s]."""
        x = np.array([1e-12, 0.0], dtype=float)
        bf = self._field(B)
        pvec = self.polarizer if polarizer is None else normalize_polarizer(polarizer)
        G = self.G_mag(x, J, bf)
        D = self.D_coeff(x, J, bf)
        kappa = float(self.K2_tensor(x, J, bf)[0, 0])
        pz = float(pvec[2])
        lam = self.lambda_stt_z * float(J) * pz
        return float((self.polarity * G * kappa + D * lam) / (D * D + G * G))

    def equilibrium_conservative(
        self,
        J: float = 0.0,
        B: ExternalFieldLike = 0.0,
        *,
        include_quartic: bool = True,
    ) -> np.ndarray:
        """
        Conservative equilibrium ``grad U = 0`` [m].

        This ignores nonconservative STT.  It is intended for checking the
        in-plane-field displacement and for initializing trajectories.
        """
        bf = self._field(B)
        fh = self.field_force(J, bf)
        K = self.K2_tensor(np.zeros(2), J, bf)
        if not include_quartic:
            return np.linalg.solve(K, fh)

        try:
            from scipy.optimize import root

            def fun(x: np.ndarray) -> np.ndarray:
                return self.grad_potential(np.asarray(x, dtype=float), J, bf)

            sol = root(fun, np.linalg.solve(K, fh))
            if sol.success and np.all(np.isfinite(sol.x)):
                return np.asarray(sol.x, dtype=float)
        except Exception:
            pass
        return np.linalg.solve(K, fh)

    def linearize_rhs(
        self,
        X0: np.ndarray,
        J: float = 0.0,
        B: ExternalFieldLike = 0.0,
        *,
        eps: float = 1e-12,
    ) -> np.ndarray:
        """Finite-difference Jacobian of ``rhs`` at ``X0`` [1/s]."""
        x0 = np.asarray(X0, dtype=float).reshape(2)
        bf = self._field(B)
        jac = np.zeros((2, 2), dtype=float)
        for j in range(2):
            dx = np.zeros(2, dtype=float)
            dx[j] = float(eps)
            fp = self.rhs(0.0, x0 + dx, current_dc(J), field_dc(bf))
            fm = self.rhs(0.0, x0 - dx, current_dc(J), field_dc(bf))
            jac[:, j] = (fp - fm) / (2.0 * float(eps))
        return jac

    def simulate(
        self,
        t_span: tuple[float, float],
        *,
        X0: tuple[float, float] | np.ndarray | None = None,
        s0: tuple[float, float] | np.ndarray | None = None,
        J_func: CurrentFunc | None = None,
        B_func: FieldFunc | None = None,
        polarizer_func: PolarizerFunc | None = None,
        I_func: CurrentFunc | float | None = None,
        dt: float = 1e-11,
        method: str = "RK45",
        clamp_u: float | None = 0.995,
        **ivp_kwargs: Any,
    ) -> FieldResolvedTrajectoryResult:
        """Integrate the field-resolved Thiele equation."""
        try:
            from scipy.integrate import solve_ivp
        except Exception:  # pragma: no cover - exercised when SciPy is unavailable
            solve_ivp = None

        t0, t1 = float(t_span[0]), float(t_span[1])
        step = float(dt)
        if not np.isfinite(t0) or not np.isfinite(t1) or not t1 > t0:
            raise ValueError("t_span must contain finite values with t_end > t_start")
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("dt must be positive and finite")
        if X0 is not None and s0 is not None:
            raise ValueError("provide only one of X0 or s0")
        if X0 is None:
            s_init = (
                np.array([1e-4, 0.0], dtype=float)
                if s0 is None
                else np.asarray(s0, dtype=float).reshape(2)
            )
            x_init = s_init * self.geom.R
        else:
            x_init = np.asarray(X0, dtype=float).reshape(2)
        if not np.all(np.isfinite(x_init)):
            raise ValueError("initial position must contain finite coordinates")
        if float(np.linalg.norm(x_init)) >= self.geom.R:
            raise ValueError("initial vortex position must lie inside the disk")

        if callable(I_func):
            resolved_I_func = I_func
        elif I_func is None:
            resolved_I_func = None
        else:
            I_value = float(I_func)

            def resolved_I_func(_t: float) -> float:
                return I_value

        t_eval: np.ndarray = np.arange(t0, t1 + 0.5 * step, step, dtype=float)
        if t_eval.size and t_eval[-1] > t1:
            t_eval = t_eval[:-1]
        if t_eval.size == 0 or t_eval[0] != t0:
            t_eval = np.insert(t_eval, 0, t0)
        if t_eval[-1] < t1:
            t_eval = np.append(t_eval, t1)

        events: list[Callable] = []
        if clamp_u is not None:
            clamp = float(clamp_u)
            if not np.isfinite(clamp) or not 0.0 < clamp <= 1.0:
                raise ValueError("clamp_u must lie in (0, 1] when provided")

            def _edge_event(_t: float, y: np.ndarray) -> float:
                return float(np.linalg.norm(y) / self.geom.R - clamp)

            _edge_event.terminal = True  # type: ignore[attr-defined]
            _edge_event.direction = 1.0  # type: ignore[attr-defined]
            events.append(_edge_event)

        edge_limited = False
        if solve_ivp is not None:
            sol = solve_ivp(
                fun=lambda t, y: self.rhs(
                    t, y, J_func, B_func, polarizer_func, resolved_I_func
                ),
                t_span=(t0, t1),
                y0=x_init,
                t_eval=t_eval,
                events=events if events else None,
                method=method,
                max_step=ivp_kwargs.pop("max_step", step),
                rtol=ivp_kwargs.pop("rtol", 1e-8),
                atol=ivp_kwargs.pop("atol", 1e-13),
                **ivp_kwargs,
            )
            if not sol.success:
                raise RuntimeError(
                    f"Field-resolved Thiele integration failed: {sol.message}"
                )
            t_out = np.asarray(sol.t, dtype=float)
            x = np.asarray(sol.y[0], dtype=float)
            y = np.asarray(sol.y[1], dtype=float)
            edge_limited = bool(
                getattr(sol, "t_events", None) and len(sol.t_events[0]) > 0
            )
        else:
            t_out, xy = self._simulate_rk4(
                t_eval,
                x_init,
                J_func=J_func,
                B_func=B_func,
                polarizer_func=polarizer_func,
                I_func=resolved_I_func,
                clamp_u=clamp_u,
            )
            x = xy[:, 0]
            y = xy[:, 1]
            edge_limited = bool(t_out.size < t_eval.size)

        R = float(self.geom.R)
        return FieldResolvedTrajectoryResult(
            model_name=f"FieldResolvedCPPThieleModel(p={self.polarity:+d}, C={self.chirality:+d})",
            t=np.asarray(t_out, dtype=float),
            x=x,
            y=y,
            sx=x / R,
            sy=y / R,
            disk_radius=R,
            params={
                "Ms": self.material.Ms,
                "alpha": self.material.alpha,
                "P": self.material.P,
                "R": self.geom.R,
                "L": self.geom.L,
                "omega0": self.omega0,
                "N": self.N,
                "G0": self.G0,
                "kappa0": self.kappa0,
                "d0": self.d0,
                "d1": self.d1,
                "chi_z_per_J": self.chi_z_per_J,
                "polarity": self.polarity,
                "chirality": self.chirality,
                "polarizer": tuple(float(v) for v in self.polarizer),
                "calibration": self.cal,
            },
            metadata={
                "mode": "field-resolved CPP Thiele",
                "edge_limited": edge_limited,
            },
        )

    def _simulate_rk4(
        self,
        t_eval: np.ndarray,
        x_init: np.ndarray,
        *,
        J_func: CurrentFunc | None,
        B_func: FieldFunc | None,
        polarizer_func: PolarizerFunc | None,
        I_func: CurrentFunc | None,
        clamp_u: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Small fixed-step RK4 fallback used when SciPy is unavailable."""
        values: list[np.ndarray] = [np.asarray(x_init, dtype=float).reshape(2)]
        times = [float(t_eval[0])]
        for idx in range(1, int(t_eval.size)):
            t = float(t_eval[idx - 1])
            h = float(t_eval[idx] - t_eval[idx - 1])
            y0 = values[-1]
            k1 = self.rhs(t, y0, J_func, B_func, polarizer_func, I_func)
            k2 = self.rhs(
                t + 0.5 * h, y0 + 0.5 * h * k1, J_func, B_func, polarizer_func, I_func
            )
            k3 = self.rhs(
                t + 0.5 * h, y0 + 0.5 * h * k2, J_func, B_func, polarizer_func, I_func
            )
            k4 = self.rhs(t + h, y0 + h * k3, J_func, B_func, polarizer_func, I_func)
            y_next = y0 + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if clamp_u is not None and np.linalg.norm(y_next) / self.geom.R >= float(
                clamp_u
            ):
                break
            values.append(np.asarray(y_next, dtype=float))
            times.append(float(t_eval[idx]))
        return np.asarray(times, dtype=float), np.vstack(values)

    @staticmethod
    def _result_time(result: Any) -> np.ndarray:
        if hasattr(result, "t"):
            return np.asarray(result.t, dtype=float)
        return np.asarray(result.time, dtype=float)

    @staticmethod
    def _result_xy(result: Any) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(result.x, dtype=float),
            np.asarray(result.y, dtype=float),
        )

    def _time_mask(
        self,
        result: Any,
        *,
        t_min: float | None = None,
        transient_fraction: float | None = None,
    ) -> np.ndarray:
        t = self._result_time(result)
        if t.size == 0:
            return np.zeros(0, dtype=bool)
        if t_min is None and transient_fraction is not None:
            frac = min(max(float(transient_fraction), 0.0), 0.95)
            t_min = float(t[0] + frac * (t[-1] - t[0]))
        if t_min is None:
            return np.ones_like(t, dtype=bool)
        return t >= float(t_min)

    def orbit_center(
        self,
        result: Any,
        mode: str | tuple[float, float] | np.ndarray = "mean",
        t_min: float | None = None,
    ) -> np.ndarray:
        """Estimate orbit center after transient [m]."""
        if isinstance(mode, str):
            mode_norm = mode.lower()
            if mode_norm == "mean":
                x, y = self._result_xy(result)
                mask = self._time_mask(result, t_min=t_min)
                if not np.any(mask):
                    return np.array([float("nan"), float("nan")], dtype=float)
                return np.array([float(np.mean(x[mask])), float(np.mean(y[mask]))])
            if mode_norm == "conservative_equilibrium":
                return self.equilibrium_conservative()
            if mode_norm == "disk" or mode_norm == "origin":
                return np.zeros(2, dtype=float)
            raise ValueError(
                "center mode must be 'mean', 'conservative_equilibrium', or 'disk'"
            )
        center = np.asarray(mode, dtype=float).reshape(2)
        return center

    def frequency_geometric(
        self,
        result: Any,
        center: str | tuple[float, float] | np.ndarray | None = "mean",
        t_min: float | None = None,
        transient_fraction: float | None = None,
        signed: bool = False,
    ) -> float:
        """Mean orbital frequency around selected center [Hz]."""
        t = self._result_time(result)
        x, y = self._result_xy(result)
        mask = self._time_mask(
            result,
            t_min=t_min,
            transient_fraction=transient_fraction,
        )
        if np.count_nonzero(mask) < 3:
            return float("nan")
        center_value = "mean" if center is None else center
        c = self.orbit_center(result, center_value, t_min=t[mask][0])
        z = (x[mask] - c[0]) + 1j * (y[mask] - c[1])
        if np.nanmax(np.abs(z)) <= 0.0:
            return float("nan")
        phase = np.unwrap(np.angle(z))
        selected_t = np.asarray(t[mask], dtype=float)
        if not np.all(np.isfinite(selected_t)) or np.any(np.diff(selected_t) <= 0.0):
            return float("nan")
        # A least-squares phase slope is less sensitive than averaging a
        # numerical derivative at the two endpoints.
        centered_t = selected_t - float(np.mean(selected_t))
        denom = float(np.dot(centered_t, centered_t))
        if denom <= 0.0:
            return float("nan")
        omega = float(np.dot(centered_t, phase - float(np.mean(phase))) / denom)
        freq = omega / (2.0 * math.pi)
        return freq if signed else abs(freq)

    def _fft_peak_hz(
        self,
        signal: np.ndarray,
        time: np.ndarray,
        *,
        window: str = "hann",
    ) -> float:
        x = np.asarray(signal, dtype=float).reshape(-1)
        t = np.asarray(time, dtype=float).reshape(-1)
        if x.size < 3 or t.size != x.size:
            return float("nan")
        diffs = np.diff(t)
        if not np.all(np.isfinite(diffs)) or np.any(diffs <= 0.0):
            return float("nan")
        dt = float(np.median(diffs))
        if not np.isfinite(dt) or dt <= 0.0:
            return float("nan")
        if float(np.max(np.abs(diffs - dt))) > 1e-3 * dt:
            uniform_t = np.linspace(float(t[0]), float(t[-1]), int(t.size))
            x = np.interp(uniform_t, t, x)
            t = uniform_t
            dt = float(t[1] - t[0])
        centered = x - float(np.mean(x))
        if window == "hann":
            centered = centered * np.hanning(centered.size)
        elif window != "none":
            raise ValueError("window must be 'hann' or 'none'")
        spectrum = np.fft.rfft(centered)
        freqs = np.fft.rfftfreq(centered.size, d=dt)
        power = np.abs(spectrum) ** 2
        if power.size <= 1:
            return float("nan")
        idx = int(np.argmax(power[1:]) + 1)
        return float(freqs[idx])

    def frequency_fft(
        self,
        result: Any,
        signal: str = "resistance",
        t_min: float | None = None,
        transient_fraction: float | None = None,
        window: str = "hann",
    ) -> float:
        """FFT-based frequency proxy matching MTJ measurements [Hz]."""
        t = self._result_time(result)
        x, y = self._result_xy(result)
        mask = self._time_mask(
            result,
            t_min=t_min,
            transient_fraction=transient_fraction,
        )
        if np.count_nonzero(mask) < 3:
            return float("nan")
        signal_norm = str(signal).lower()
        if signal_norm in {"x", "core_x"}:
            values = x[mask]
        elif signal_norm in {"y", "core_y"}:
            values = y[mask]
        elif signal_norm in {"radius", "r"}:
            c = self.orbit_center(result, "mean", t_min=t[mask][0])
            values = np.hypot(x[mask] - c[0], y[mask] - c[1])
        elif signal_norm in {"resistance", "mtj", "voltage"}:
            p = self.polarizer[:2]
            if float(np.dot(p, p)) > 0.0:
                values = p[0] * x[mask] + p[1] * y[mask]
            else:
                values = x[mask]
        else:
            raise ValueError("signal must be 'resistance', 'x', 'y', or 'radius'")
        return self._fft_peak_hz(values, t[mask], window=window)

    def simulate_dc_sweep(
        self,
        I_values_A,
        B: ExternalFieldLike = 0.0,
        *,
        t_total: float,
        dt: float,
        transient_fraction: float = 0.5,
        s0: tuple[float, float] = (1e-3, 0.0),
        frequency_signal: str = "resistance",
        **kwargs: Any,
    ):
        """Return frequency, amplitude, center, and diagnostics for a DC-current sweep."""
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for value in np.asarray(I_values_A, dtype=float).reshape(-1):
            J = self.current_drive.J_from_I(float(value), self.geom)
            result = self.simulate(
                (0.0, float(t_total)),
                I_func=float(value),
                B_func=field_dc(B),
                dt=float(dt),
                s0=s0,
                **kwargs,
            )
            mask = self._time_mask(result, transient_fraction=transient_fraction)
            center = self.orbit_center(
                result, "mean", t_min=result.t[mask][0] if np.any(mask) else None
            )
            u = result.u[mask] if np.any(mask) else np.array([], dtype=float)
            f_geom = self.frequency_geometric(
                result,
                center=center,
                transient_fraction=transient_fraction,
            )
            f_fft = self.frequency_fft(
                result,
                signal=frequency_signal,
                transient_fraction=transient_fraction,
            )
            growth = self.radial_growth_rate_small_signal(J, B)
            edge_limited = bool(result.metadata.get("edge_limited", False))
            if edge_limited:
                regime = "edge_limited"
            elif not np.isfinite(growth) or growth <= 0.0:
                regime = "damped"
            else:
                regime = "stable_gyro"
            rows.append(
                {
                    "I_A": float(value),
                    "I_mA": float(value) * 1e3,
                    "J_Apm2": float(J),
                    "frequency_geom_hz": float(f_geom),
                    "frequency_fft_hz": float(f_fft),
                    "u_mean": float(np.mean(u)) if u.size else float("nan"),
                    "u_max": float(np.max(u)) if u.size else float("nan"),
                    "center_x_m": float(center[0]),
                    "center_y_m": float(center[1]),
                    "regime": regime,
                    "edge_limited": edge_limited,
                }
            )
        return pd.DataFrame(rows)


__all__ = [
    "J2",
    "CurrentFunc",
    "FieldFunc",
    "PolarizerFunc",
    "current_dc",
    "field_dc",
    "normalize_polarizer",
    "CurrentDrive",
    "SaturationCalibration",
    "OerstedCalibration",
    "ThermalCalibration",
    "FrequencyExtractionConfig",
    "FieldResolvedCalibration",
    "FieldResolvedTrajectoryResult",
    "FieldResolvedCPPThieleModel",
]
