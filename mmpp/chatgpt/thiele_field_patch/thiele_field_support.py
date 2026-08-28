# ruff: noqa: N802, N803, N806, PLR0913
"""thiele_field_support.py

Drop-in *field-aware* Thiele models for a magnetic vortex core.

This file is meant as a **patch module** you can copy into your package and then
migrate your existing code to use these classes.

Key fixes vs your current implementation
----------------------------------------
1) **Units:** if `B_ext` is in Tesla (μ0·H), then the LLG prefactor is **γ**,
   not `γ*μ0`. Multiplying by μ0 again silently kills the scale by ~1e-6.

2) **Physics:** a uniform external field does **not** enter as `ω0 += p·γ·B`
   in a vortex gyrotropic mode. In Thiele theory, field enters mainly via:
   - a **static force / equilibrium shift** for *in-plane* field,
   - a **stiffness / gyrovector modification** for *out-of-plane* field.

   In a compact reduced model we implement this by two *calibratable* couplings:
   - `domega0_dBz` [rad/s/T] for Bz → ω0 shift,
   - `chi_inplane_per_T` [1/T] for in-plane B → core shift (in reduced coords).

3) **In-plane field support:** we shift the confinement centre by `s_eq(Bx,By)`
   so the orbit occurs around the displaced equilibrium.

You still need to calibrate
---------------------------
- `domega0_dBz`: get ω0(Bz) from micromagnetics / experiment at small Bz.
- `chi_inplane_per_T`: measure static core displacement vs Bx (or By).

For strong fields, large orbits, non-circular pillars, or near core switching,
full micromagnetic validation is required.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np

# Optional integration with your package (AnalyticalResult, constants)
try:  # pragma: no cover
    from .base import AnalyticalResult
except Exception:  # pragma: no cover

    class AnalyticalResult:  # type: ignore[no-redef]  # minimal fallback
        model_name: str = ""
        params: dict[str, Any] = {}
        metadata: dict[str, Any] = {}


try:  # pragma: no cover
    from .constants import GAMMA_E
except Exception:  # pragma: no cover
    GAMMA_E = 1.760_859_630_23e11  # rad/(s·T)

# ---------------------------------------------------------------------------
# Physical helpers
# ---------------------------------------------------------------------------

# Bohr magneton [J/T]
_MU_B: float = 9.2740100783e-24

# Elementary charge [C]
_E_CHARGE: float = 1.602176634e-19

# Reduced Planck constant [J·s]
_HBAR: float = 1.054571817e-34


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class MaterialParams:
    Ms: float
    alpha: float
    P: float
    A: float = 1.3e-11
    beta_nonadiabatic: float | None = None
    gamma: float = GAMMA_E

    @property
    def beta(self) -> float:
        return (
            self.beta_nonadiabatic if self.beta_nonadiabatic is not None else self.alpha
        )


@dataclass
class DiskGeometry:
    R: float
    L: float
    core_diameter: float | None = None

    def Rc(self, mat: MaterialParams | None = None) -> float:
        if self.core_diameter is not None:
            return self.core_diameter / 2.0
        # Reasonable default core radius if mat not provided
        return 5e-9


# ---------------------------------------------------------------------------
# External field API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExternalField:
    """External magnetic induction **B** in Tesla (μ0·H)."""

    Bx: float = 0.0
    By: float = 0.0
    Bz: float = 0.0

    @staticmethod
    def from_any(value: object) -> ExternalField:
        """Coerce float/tuple/list/ndarray/ExternalField → ExternalField.

        Rules:
        - float/int → interpreted as Bz
        - (Bx,By,Bz) sequence length 3
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
        raise TypeError("B_ext must be float (Bz) or a length-3 (Bx,By,Bz)")


ExternalFieldLike: TypeAlias = (
    float | tuple[float, float, float] | ExternalField | np.ndarray
)
FieldFunc: TypeAlias = Callable[[float], ExternalFieldLike]


def field_dc(B_ext: ExternalFieldLike = 0.0) -> FieldFunc:
    """Constant external field B(t)=const in Tesla."""

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
    """Sinusoidal field: B(t) = B_offset + B_amp*sin(2π f t + phase)."""

    amp = ExternalField.from_any(B_amp)
    off = ExternalField.from_any(B_offset)
    omega = 2.0 * math.pi * float(f_hz)

    def _b(t: float) -> ExternalField:
        s = math.sin(omega * t + float(phase))
        return ExternalField(
            off.Bx + amp.Bx * s,
            off.By + amp.By * s,
            off.Bz + amp.Bz * s,
        )

    return _b


@dataclass
class FieldCoupling:
    """How B enters the reduced Thiele model.

    Parameters
    ----------
    domega0_dBz : float
        Linear slope for out-of-plane field influence: Δω0 ≈ p·(dω0/dBz)·Bz.
        Units: [rad/s/T]. Default 0 → no effect.

    chi_inplane_per_T : float
        Static core displacement susceptibility (in *normalised* coords s=r/R):
            s_eq = c·chi_inplane_per_T · ( -By, +Bx )
        Units: [1/T]. Default 0 → no shift.

    chirality : int
        Vortex chirality c = ±1. Controls sign of in-plane shift.
    """

    domega0_dBz: float = 0.0
    chi_inplane_per_T: float = 0.0
    chirality: int = 1

    def omega0_shift(self, *, field: ExternalField, polarity: int) -> float:
        p = 1 if int(polarity) >= 0 else -1
        c = float(self.domega0_dBz)
        return p * c * float(field.Bz)

    def s_eq(self, *, field: ExternalField) -> tuple[float, float]:
        c = 1 if int(self.chirality) >= 0 else -1
        chi = float(self.chi_inplane_per_T)
        # z × B_inplane = (-By, +Bx)
        return (
            c * chi * (-float(field.By)),
            c * chi * (float(field.Bx)),
        )


# ---------------------------------------------------------------------------
# Minimal result container (compatible idea with your ThieleTrajectoryResult)
# ---------------------------------------------------------------------------


@dataclass
class ThieleTrajectoryResult(AnalyticalResult):
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    sx: np.ndarray
    sy: np.ndarray
    disk_radius: float


# ---------------------------------------------------------------------------
# CIP model (Moon et al.) with external field
# ---------------------------------------------------------------------------


class CIPThieleModel:
    def __init__(
        self,
        material: MaterialParams,
        geom: DiskGeometry,
        omega0: float,
        polarity: int = 1,
        current_dir: tuple[float, float] = (1.0, 0.0),
        *,
        B_ext: ExternalFieldLike = 0.0,
        field_coupling: FieldCoupling | None = None,
        chirality: int = 1,
    ) -> None:
        self.material = material
        self.geom = geom
        self.omega0 = float(omega0)
        self.polarity = int(polarity)
        if self.polarity not in (1, -1):
            raise ValueError("polarity must be ±1")

        # normalise current direction
        cx, cy = current_dir
        norm = float(math.hypot(cx, cy))
        if norm <= 0:
            raise ValueError("current_dir must be non-zero")
        self.current_dir = (float(cx) / norm, float(cy) / norm)

        self.field = ExternalField.from_any(B_ext)
        self.field_coupling = field_coupling or FieldCoupling(chirality=int(chirality))

        self._setup()

    def _setup(self) -> None:
        mat = self.material
        geo = self.geom

        # Spin-drift velocity prefactor: u₀ = μ_B P / (e Ms)
        self._u0_prefactor = _MU_B * mat.P / (_E_CHARGE * mat.Ms)

        # Core diameter δ
        if geo.core_diameter is not None:
            delta = float(geo.core_diameter)
        else:
            # fallback order-of-magnitude
            delta = 10e-9

        ratio = geo.R / max(delta, 1e-12)
        self._d_over_G0 = 0.5 * math.log(max(ratio, 1.1))

        self._alpha = float(mat.alpha)
        self._beta = float(mat.beta)
        self._dG = float(self._d_over_G0)
        self._p = int(self.polarity)
        self._omega0_base = float(self.omega0)

    def _field_at(self, t: float, B_func: FieldFunc | None) -> ExternalField:
        if B_func is None:
            return self.field
        return ExternalField.from_any(B_func(float(t)))

    def _rhs(
        self,
        t: float,
        state: np.ndarray,
        J_func: Callable[[float], float],
        B_func: FieldFunc | None,
    ) -> np.ndarray:
        X, Y = float(state[0]), float(state[1])

        # current → drift velocity
        J = float(J_func(t))
        u0 = self._u0_prefactor * J
        ux = u0 * self.current_dir[0]
        uy = u0 * self.current_dir[1]

        # external field
        B = self._field_at(t, B_func)
        w0 = self._omega0_base + self.field_coupling.omega0_shift(
            field=B, polarity=self._p
        )
        sx_eq, sy_eq = self.field_coupling.s_eq(field=B)
        X_eq = sx_eq * self.geom.R
        Y_eq = sy_eq * self.geom.R
        Xr = X - X_eq
        Yr = Y - Y_eq

        p = self._p
        alpha = self._alpha
        beta = self._beta
        dG = self._dG

        det = (alpha * dG) ** 2 + p**2
        rhs_I = -w0 * Xr + p * uy + beta * dG * ux
        rhs_II = -w0 * Yr - p * ux + beta * dG * uy

        dXdt = (alpha * dG * rhs_I + p * rhs_II) / det
        dYdt = (-p * rhs_I + alpha * dG * rhs_II) / det

        return np.array([dXdt, dYdt], dtype=float)

    def simulate(
        self,
        t_span: tuple[float, float],
        r0: tuple[float, float] = (0.0, 0.0),
        *,
        J_func: Callable[[float], float] | None = None,
        B_func: FieldFunc | None = None,
        dt: float = 1e-12,
        method: str = "RK45",
        **ivp_kwargs: Any,
    ) -> ThieleTrajectoryResult:
        from scipy.integrate import solve_ivp

        if J_func is None:
            J_func = lambda t: 0.0  # noqa: E731

        t0, t1 = float(t_span[0]), float(t_span[1])
        t_eval = np.arange(t0, t1 + 0.5 * float(dt), float(dt))

        sol = solve_ivp(
            fun=lambda t, y: self._rhs(t, y, J_func, B_func),
            t_span=(t0, t1),
            y0=np.array(r0, dtype=float),
            t_eval=t_eval,
            method=method,
            max_step=ivp_kwargs.pop("max_step", float(dt)),
            rtol=ivp_kwargs.pop("rtol", 1e-9),
            atol=ivp_kwargs.pop("atol", 1e-12),
            **ivp_kwargs,
        )
        if not sol.success:
            raise RuntimeError(f"CIP Thiele integration failed: {sol.message}")

        R = float(self.geom.R)
        X = sol.y[0]
        Y = sol.y[1]
        return ThieleTrajectoryResult(
            model_name=f"CIP Thiele + B (p={self.polarity:+d})",
            t=sol.t,
            x=X,
            y=Y,
            sx=X / R,
            sy=Y / R,
            disk_radius=R,
            params={
                "omega0": self.omega0,
                "polarity": self.polarity,
                "B_ext": (self.field.Bx, self.field.By, self.field.Bz),
                "domega0_dBz": self.field_coupling.domega0_dBz,
                "chi_inplane_per_T": self.field_coupling.chi_inplane_per_T,
                "chirality": self.field_coupling.chirality,
            },
            metadata={"mode": "CIP"},
        )


# ---------------------------------------------------------------------------
# CPP model (Guslienko 2014) with external field
# ---------------------------------------------------------------------------


class CPPThieleModel:
    def __init__(
        self,
        material: MaterialParams,
        geom: DiskGeometry,
        omega0: float,
        N: float = 0.25,
        polarity: int = 1,
        *,
        omega0_Oe_per_J: float = 0.0,
        B_ext: ExternalFieldLike = 0.0,
        field_coupling: FieldCoupling | None = None,
        chirality: int = 1,
    ) -> None:
        self.material = material
        self.geom = geom
        self.omega0 = float(omega0)
        self.N = float(N)
        self.polarity = int(polarity)
        if self.polarity not in (1, -1):
            raise ValueError("polarity must be ±1")

        self.omega0_Oe_per_J = float(omega0_Oe_per_J)
        self.field = ExternalField.from_any(B_ext)
        self.field_coupling = field_coupling or FieldCoupling(chirality=int(chirality))

        self._setup()

    def _setup(self) -> None:
        mat = self.material
        geo = self.geom

        self._sigma = _HBAR * mat.P / (2.0 * _E_CHARGE * geo.L * mat.Ms)

        # d0, d1 (as in Guslienko 2014, but requires Rc; we fallback safely)
        Rc = geo.Rc(mat)
        ratio = geo.R / max(Rc, 1e-12)
        self._d0 = mat.alpha * (5.0 + 4.0 * math.log(max(ratio, 1.1))) / 8.0
        self._d1 = (11.0 / 6.0) * mat.alpha

        self._chi_prefactor = mat.gamma * self._sigma / 2.0

    def _field_at(self, t: float, B_func: FieldFunc | None) -> ExternalField:
        if B_func is None:
            return self.field
        return ExternalField.from_any(B_func(float(t)))

    def chi(self, J: float) -> float:
        return self._chi_prefactor * float(J)

    def d(self, u: float) -> float:
        u = float(u)
        return float(self._d0 + self._d1 * u * u)

    def omega0_eff(self, J: float, field: ExternalField) -> float:
        return (
            self.omega0
            + self.omega0_Oe_per_J * float(J)
            + self.field_coupling.omega0_shift(field=field, polarity=self.polarity)
        )

    def omega(self, u: float, J: float, field: ExternalField) -> float:
        w0 = float(self.omega0_eff(J, field))
        return w0 * (1.0 + self.N * float(u) * float(u))

    @property
    def J_threshold(self) -> float:
        w0 = self.omega0_eff(0.0, self.field)
        if w0 <= 0 or not np.isfinite(w0):
            return float("nan")
        return float(self._d0 * w0 / self._chi_prefactor)

    def steady_state_u(
        self, J: float, *, allow_edge: bool = False, u_stop: float = 0.98
    ) -> float | None:
        J = float(J)
        field = self.field
        chi_val = float(self.chi(J))
        w0 = float(self.omega0_eff(J, field))
        if not (np.isfinite(chi_val) and np.isfinite(w0) and w0 > 0):
            return None

        N = float(self.N)
        d0 = float(self._d0)
        d1 = float(self._d1)

        # χ = (d0 + d1 u^2) * w0 * (1 + N u^2)
        c2 = d1 * N
        c1 = d1 + d0 * N
        c0 = d0 - chi_val / w0

        xs: list[float] = []
        if abs(c2) < 1e-30:
            if abs(c1) > 1e-30:
                xs = [-c0 / c1]
        else:
            disc = c1 * c1 - 4.0 * c2 * c0
            if disc >= 0:
                sdisc = math.sqrt(disc)
                xs = [(-c1 + sdisc) / (2.0 * c2), (-c1 - sdisc) / (2.0 * c2)]

        u_candidates: list[float] = []
        for x in xs:
            if not np.isfinite(x) or x <= 0:
                continue
            u = math.sqrt(x)
            # stability condition for this reduced model
            if (d1 + N * d0 + 2.0 * N * d1 * u * u) <= 0.0:
                continue
            u_candidates.append(u)

        if not u_candidates:
            if allow_edge and chi_val > self.d(u_stop) * self.omega(u_stop, J, field):
                return float(u_stop)
            return None

        u0 = float(min(u_candidates))
        if u0 >= u_stop:
            return float(u_stop) if allow_edge else None
        return u0

    def predict_frequency_dc(
        self, J_dc: float, *, allow_edge: bool = False
    ) -> float | None:
        u0 = self.steady_state_u(float(J_dc), allow_edge=allow_edge)
        if u0 is None:
            return None
        w = self.omega(u0, float(J_dc), self.field)
        return float(w / (2.0 * math.pi))

    def _rhs(
        self,
        t: float,
        state: np.ndarray,
        J_func: Callable[[float], float],
        B_func: FieldFunc | None,
    ) -> np.ndarray:
        sx, sy = float(state[0]), float(state[1])

        B = self._field_at(t, B_func)
        sx_eq, sy_eq = self.field_coupling.s_eq(field=B)
        srx = sx - sx_eq
        sry = sy - sy_eq
        u = max(float(math.hypot(srx, sry)), 1e-15)

        J = float(J_func(t))
        chi_val = float(self.chi(J))
        omega_val = float(self.omega(u, J, B))
        radial = chi_val - self.d(u) * omega_val

        p = int(self.polarity)
        dsx = radial * srx - p * omega_val * sry
        dsy = radial * sry + p * omega_val * srx
        return np.array([dsx, dsy], dtype=float)

    def simulate(
        self,
        t_span: tuple[float, float],
        s0: tuple[float, float] = (1e-3, 0.0),
        *,
        J_func: Callable[[float], float] | None = None,
        B_func: FieldFunc | None = None,
        dt: float = 1e-11,
        method: str = "RK45",
        **ivp_kwargs: Any,
    ) -> ThieleTrajectoryResult:
        from scipy.integrate import solve_ivp

        if J_func is None:
            J_func = lambda t: 0.0  # noqa: E731

        t0, t1 = float(t_span[0]), float(t_span[1])
        t_eval = np.arange(t0, t1 + 0.5 * float(dt), float(dt))

        sol = solve_ivp(
            fun=lambda t, y: self._rhs(t, y, J_func, B_func),
            t_span=(t0, t1),
            y0=np.array(s0, dtype=float),
            t_eval=t_eval,
            method=method,
            max_step=ivp_kwargs.pop("max_step", float(dt)),
            rtol=ivp_kwargs.pop("rtol", 1e-9),
            atol=ivp_kwargs.pop("atol", 1e-14),
            **ivp_kwargs,
        )
        if not sol.success:
            raise RuntimeError(f"CPP Thiele integration failed: {sol.message}")

        SX = sol.y[0]
        SY = sol.y[1]
        R = float(self.geom.R)
        return ThieleTrajectoryResult(
            model_name=f"CPP Thiele + B (p={self.polarity:+d})",
            t=sol.t,
            x=SX * R,
            y=SY * R,
            sx=SX,
            sy=SY,
            disk_radius=R,
            params={
                "omega0": self.omega0,
                "N": self.N,
                "omega0_Oe_per_J": self.omega0_Oe_per_J,
                "J_threshold": self.J_threshold,
                "B_ext": (self.field.Bx, self.field.By, self.field.Bz),
                "domega0_dBz": self.field_coupling.domega0_dBz,
                "chi_inplane_per_T": self.field_coupling.chi_inplane_per_T,
                "chirality": self.field_coupling.chirality,
            },
            metadata={"mode": "CPP"},
        )
