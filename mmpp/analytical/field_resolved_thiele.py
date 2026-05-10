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
from dataclasses import dataclass, field
from typing import Any, Callable, TypeAlias

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

    def _j(_t: float) -> float:
        return float(J_dc)

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
    p = np.asarray(polarizer, dtype=float).reshape(-1)
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

    def omega0_eff(self, J: float, B: ExternalFieldLike | ExternalField = 0.0) -> float:
        """Linear angular frequency response ``omega0(J,Bz)`` [rad/s]."""
        bf = self._field(B)
        p = float(self.polarity)
        omega = (
            self.omega0
            + float(self.cal.domega_dJ) * float(J)
            + p * float(self.cal.domega_dBz) * bf.Bz_T
            + float(self.cal.domega_dBz2) * bf.Bz_T * bf.Bz_T
        )
        floor = max(float(self.cal.min_omega_factor), 0.0) * self.omega0
        return float(max(omega, floor))

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
        return float(max(self.G0 * scale, 1e-30))

    def D_coeff(
        self, X: np.ndarray, J: float, B: ExternalFieldLike | ExternalField = 0.0
    ) -> float:  # noqa: ARG002
        """Damping coefficient ``D`` [kg/s]."""
        bf = self._field(B)
        G = self.G_mag(X, J, bf)
        u2 = float(np.dot(X, X)) / max(self.geom.R * self.geom.R, 1e-30)
        b2 = bf.Bx_T * bf.Bx_T + bf.By_T * bf.By_T
        d = (self.d0 + self.d1 * u2) * (
            1.0
            + self.polarity * float(self.cal.D_Bz) * bf.Bz_T
            + float(self.cal.D_ip2) * b2
        )
        return float(max(G * d, 0.0))

    # ------------------------------------------------------------------
    # Conservative potential and force terms
    # ------------------------------------------------------------------

    def K2_tensor(
        self, X: np.ndarray, J: float, B: ExternalFieldLike | ExternalField = 0.0
    ) -> np.ndarray:  # noqa: ARG002
        """Linear stiffness tensor ``K2`` [N/m]."""
        bf = self._field(B)
        G = self.G_mag(X, J, bf)
        kappa = G * self.omega0_eff(J, bf)

        bvec = np.array([bf.Bx_T, bf.By_T], dtype=float)
        b2 = float(np.dot(bvec, bvec))
        K = kappa * (1.0 + float(self.cal.k_ip_iso_per_T2) * b2) * I2.copy()

        if b2 > 0.0 and abs(float(self.cal.k_ip_aniso_per_T2)) > 0.0:
            bhat = bvec / math.sqrt(b2)
            phat = J2 @ bhat
            anis = np.outer(bhat, bhat) - np.outer(phat, phat)
            K = K + kappa * float(self.cal.k_ip_aniso_per_T2) * b2 * anis

        return np.asarray(K, dtype=float)

    def K4_scalar(
        self, X: np.ndarray, J: float, B: ExternalFieldLike | ExternalField = 0.0
    ) -> float:  # noqa: ARG002
        """Quartic stiffness coefficient ``K4`` [N/m]."""
        G = self.G_mag(X, J, B)
        return float(G * self.omega0_eff(J, B) * self.N_eff(B))

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

    def grad_potential(
        self, X: np.ndarray, J: float, B: ExternalFieldLike | ExternalField = 0.0
    ) -> np.ndarray:
        """Gradient of the conservative potential, ``grad_X U`` [N]."""
        x = np.asarray(X, dtype=float).reshape(2)
        K2 = self.K2_tensor(x, J, B)
        K4 = self.K4_scalar(x, J, B)
        r2 = float(np.dot(x, x))
        return (
            K2 @ x
            + (K4 / max(self.geom.R * self.geom.R, 1e-30)) * r2 * x
            - self.field_force(J, B)
        )

    def potential(
        self, X: np.ndarray, J: float, B: ExternalFieldLike | ExternalField = 0.0
    ) -> float:
        """Conservative potential ``U(X,J,B)`` [J]."""
        x = np.asarray(X, dtype=float).reshape(2)
        K2 = self.K2_tensor(x, J, B)
        K4 = self.K4_scalar(x, J, B)
        r2 = float(np.dot(x, x))
        return float(
            0.5 * x @ K2 @ x
            + 0.25 * (K4 / max(self.geom.R * self.geom.R, 1e-30)) * r2 * r2
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
    ) -> np.ndarray:
        """Right-hand side ``dX/dt`` [m/s] for ``solve_ivp``."""
        x = np.asarray(X, dtype=float).reshape(2)
        J = 0.0 if J_func is None else float(J_func(float(t)))
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
        force = self.stt_force(x, J, pvec) - self.grad_potential(x, J, B)
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
        kappa = G * self.omega0_eff(J, bf)
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
        kappa = G * self.omega0_eff(J, bf)
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
        dt: float = 1e-11,
        method: str = "RK45",
        clamp_u: float | None = 0.995,
        **ivp_kwargs: Any,
    ) -> FieldResolvedTrajectoryResult:
        """Integrate the field-resolved Thiele equation."""
        from scipy.integrate import solve_ivp

        t0, t1 = float(t_span[0]), float(t_span[1])
        if not t1 > t0:
            raise ValueError("t_span must satisfy t_end > t_start")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
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

        t_eval = np.arange(t0, t1 + 0.5 * dt, dt, dtype=float)
        if t_eval.size and t_eval[-1] > t1:
            t_eval = t_eval[:-1]

        events: list[Callable] = []
        if clamp_u is not None:
            clamp = float(clamp_u)
            if clamp <= 0.0:
                raise ValueError("clamp_u must be positive when provided")

            def _edge_event(_t: float, y: np.ndarray) -> float:
                return float(np.linalg.norm(y) / self.geom.R - clamp)

            _edge_event.terminal = True  # type: ignore[attr-defined]
            _edge_event.direction = 1.0  # type: ignore[attr-defined]
            events.append(_edge_event)

        sol = solve_ivp(
            fun=lambda t, y: self.rhs(t, y, J_func, B_func, polarizer_func),
            t_span=(t0, t1),
            y0=x_init,
            t_eval=t_eval,
            events=events if events else None,
            method=method,
            max_step=ivp_kwargs.pop("max_step", dt),
            rtol=ivp_kwargs.pop("rtol", 1e-8),
            atol=ivp_kwargs.pop("atol", 1e-13),
            **ivp_kwargs,
        )
        if not sol.success:
            raise RuntimeError(
                f"Field-resolved Thiele integration failed: {sol.message}"
            )

        x = np.asarray(sol.y[0], dtype=float)
        y = np.asarray(sol.y[1], dtype=float)
        R = float(self.geom.R)
        return FieldResolvedTrajectoryResult(
            model_name=f"FieldResolvedCPPThieleModel(p={self.polarity:+d}, C={self.chirality:+d})",
            t=np.asarray(sol.t, dtype=float),
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
                "edge_limited": bool(
                    getattr(sol, "t_events", None) and len(sol.t_events[0]) > 0
                ),
            },
        )


__all__ = [
    "J2",
    "CurrentFunc",
    "FieldFunc",
    "PolarizerFunc",
    "current_dc",
    "field_dc",
    "normalize_polarizer",
    "FieldResolvedCalibration",
    "FieldResolvedTrajectoryResult",
    "FieldResolvedCPPThieleModel",
]
