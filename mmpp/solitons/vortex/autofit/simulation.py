"""Analytical simulation context used by single-job autofit."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class SimulationContext:
    """Pre-built simulation context to avoid per-evaluation overhead.

    Key optimisations:
    - Pre-computes all constants that don't change between evaluations
    - Uses Numba-JIT RK4 integrator (~1000x faster than scipy solve_ivp)
    - Avoids rebuilding MaterialParams/DiskGeometry/Model objects per eval
    """

    def __init__(
        self,
        *,
        vortex_interface,
        numerical,
        resolution,
        base_params: dict[str, Any],
        frozen_params: dict[str, float],
        disk_radius: float,
        tracking_source: str,
        tracking_method: str | None,
        initial_condition: str,
    ):
        from ..plotting import (
            _resolve_analytical_initial_state,
            _trajectory_center,
            _trajectory_dt,
        )

        self._vortex = vortex_interface
        self._numerical = numerical
        self._resolution = resolution
        self._frozen_params = frozen_params
        self._disk_radius = disk_radius
        self._field = base_params.get("field")
        self._initial_condition = str(initial_condition)

        self._center = _trajectory_center(numerical)
        self._dt = _trajectory_dt(numerical)
        time = np.asarray(numerical.time, dtype=float)
        self._time = time
        self._t0 = float(time[0]) if time.size else 0.0
        self._t1 = float(time[-1]) if time.size >= 2 else (self._t0 + self._dt)

        R = max(float(base_params["R"]), 1e-18)
        rel_x, rel_y, source = _resolve_analytical_initial_state(
            vortex_interface,
            numerical,
            base_params,
            tracking_source=tracking_source,
            tracking_method=tracking_method,
            initial_condition=initial_condition,
        )
        self._initial_condition_source = source

        self._s0_x = rel_x / R
        self._s0_y = rel_y / R
        self._r0_x = rel_x
        self._r0_y = rel_y
        self._R = R

        nominal_end = self._t0 + self._dt * max(int(time.size) - 1, 1)
        self._sim_t1 = max(self._t1, nominal_end)

        self._current_density = base_params.get("current_density")
        self._current_density_is_callable = callable(self._current_density)

        self._model_kind = resolution.model_kind
        self._fast_path_enabled = False
        self._fast_path_reason = "unsupported"

        if self._model_kind == "cpp":
            self._precompute_cpp(base_params)
        else:
            self._precompute_cip(base_params)

    def _precompute_cpp(self, params: dict) -> None:
        """Pre-compute CPP model constants from base params."""
        _hbar = 1.054571817e-34
        _e_charge = 1.602176634e-19
        gamma = float(params.get("gamma", 1.76085963023e11))

        Ms = float(params["Ms"])
        alpha = float(params["alpha"])
        A = float(params.get("A", 1.3e-11))
        R = float(params["R"])
        L = float(params.get("L_stt", params["L"]))

        positive = {"Ms": Ms, "A": A, "R": R, "L_stt": L, "gamma": gamma}
        if any(not np.isfinite(value) or value <= 0.0 for value in positive.values()):
            raise ValueError(
                "CPP material and geometry scales must be finite and positive"
            )
        if not np.isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be finite and non-negative")

        self._chi_prefactor_per_p = gamma * _hbar / (4.0 * _e_charge * L * Ms)

        mu0 = 4e-7 * math.pi
        lex = math.sqrt(2.0 * A / (mu0 * Ms * Ms))
        Rc = max(lex, 1e-10)

        ratio = R / max(Rc, 1e-10)
        self._d0 = alpha * (5.0 + 4.0 * math.log(max(ratio, 1.1))) / 8.0
        self._d1 = (11.0 / 6.0) * alpha

        self._polarity = int(np.sign(float(params.get("polarity", 1))) or 1)
        self._p_model_base = float(
            params["P_model"] if "P_model" in params else params["P"]
        )
        self._domega0_dJ_base = float(params.get("domega0_dJ", 0.0))
        if not np.isfinite(self._p_model_base) or not np.isfinite(
            self._domega0_dJ_base
        ):
            raise ValueError("P_model and domega0_dJ must be finite")

        J = self._current_density
        if callable(J):
            try:
                J_val = float(J((self._t0 + self._sim_t1) / 2.0))
            except Exception:
                J_val = 0.0
            self._J_const = J_val
        elif J is not None:
            self._J_const = float(J)
        else:
            self._J_const = 0.0
        if self._current_density_is_callable:
            self._fast_path_enabled = False
            self._fast_path_reason = (
                "time-dependent current requires scipy reference integrator"
            )
        elif field_has_effect(self._field):
            self._fast_path_enabled = False
            self._fast_path_reason = (
                "external field requires scipy reference integrator"
            )
        else:
            self._fast_path_enabled = True
            self._fast_path_reason = "constant-current zero-field CPP"

    def _precompute_cip(self, params: dict) -> None:
        """Pre-compute CIP model constants from base params."""
        _mu_b = 9.2740100783e-24
        _e_charge = 1.602176634e-19

        Ms = float(params["Ms"])
        alpha = float(params["alpha"])
        P = float(params["P"])
        A = float(params.get("A", 1.3e-11))
        R = float(params["R"])

        positive = {"Ms": Ms, "A": A, "R": R}
        if any(not np.isfinite(value) or value <= 0.0 for value in positive.values()):
            raise ValueError(
                "CIP material and geometry scales must be finite and positive"
            )
        if not np.isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be finite and non-negative")
        if not np.isfinite(P):
            raise ValueError("P must be finite")

        self._alpha_cip = alpha
        beta_value = params.get("beta_nonadiabatic", params.get("beta", alpha))
        self._beta_cip = alpha if beta_value is None else float(beta_value)
        if not np.isfinite(self._beta_cip):
            raise ValueError("beta_nonadiabatic must be finite")
        self._polarity = int(np.sign(float(params.get("polarity", 1))) or 1)

        self._u0_prefactor_per_p = -_mu_b / (_e_charge * Ms)

        mu0 = 4e-7 * math.pi
        lex = math.sqrt(2.0 * A / (mu0 * Ms * Ms))
        core_diam = 2.0 * lex
        ratio = R / max(core_diam, 1e-10)
        self._dG_cip = 0.5 * math.log(max(ratio, 1.1))

        current_dir = np.asarray(
            params.get("current_dir", (1.0, 0.0)), dtype=float
        ).reshape(-1)
        if current_dir.size != 2 or not np.all(np.isfinite(current_dir)):
            raise ValueError("current_dir must contain exactly two finite components")
        norm = float(np.linalg.norm(current_dir))
        if norm <= 0.0:
            raise ValueError("current_dir must be non-zero")
        self._current_dir = (
            float(current_dir[0] / norm),
            float(current_dir[1] / norm),
        )

        J = self._current_density
        if callable(J):
            try:
                J_val = float(J((self._t0 + self._sim_t1) / 2.0))
            except Exception:
                J_val = 0.0
            self._J_const = J_val
        elif J is not None:
            self._J_const = float(J)
        else:
            self._J_const = 0.0
        if self._current_density_is_callable:
            self._fast_path_enabled = False
            self._fast_path_reason = (
                "time-dependent current requires scipy reference integrator"
            )
        elif field_has_effect(self._field):
            self._fast_path_enabled = False
            self._fast_path_reason = (
                "external field requires scipy reference integrator"
            )
        else:
            self._fast_path_enabled = True
            self._fast_path_reason = "constant-current zero-field CIP"

    @property
    def fast_path_enabled(self) -> bool:
        return bool(self._fast_path_enabled)

    @property
    def fast_path_reason(self) -> str:
        return str(self._fast_path_reason)

    def simulate(self, params: dict[str, Any]):
        """Simulate analytical trajectory using fast path when available."""
        from ..model.adapters import thiele_to_trajectory_result
        from ..plotting import (
            _resample_trajectory_to_reference,
            _trajectory_center,
            _translate_trajectory,
        )
        from ._numba_kernels import HAS_NUMBA

        use_fast = HAS_NUMBA and self._fast_path_enabled
        if self._model_kind == "cpp":
            raw = (
                self._simulate_cpp_fast(params)
                if use_fast
                else self._simulate_cpp_scipy(params)
            )
        else:
            raw = (
                self._simulate_cip_fast(params)
                if use_fast
                else self._simulate_cip_scipy(params)
            )

        analytical = thiele_to_trajectory_result(
            raw,
            method=f"thiele_{self._model_kind}",
            polarity=self._polarity,
        )

        raw_center = _trajectory_center(analytical)
        alignment_center = (
            (0.0, 0.0) if not field_has_effect(self._field) else raw_center
        )
        shift = (
            self._center[0] - alignment_center[0],
            self._center[1] - alignment_center[1],
        )
        aligned = _translate_trajectory(
            analytical,
            shift=shift,
            method_suffix="+aligned",
            metadata={
                "raw_center": raw_center,
                "alignment_reference_center": alignment_center,
            },
        )
        return _resample_trajectory_to_reference(aligned, self._time)

    def _simulate_cpp_fast(self, params: dict[str, Any]):
        """Ultra-fast CPP simulation via Numba RK4."""
        from mmpp.analytical.thiele import ThieleTrajectoryResult

        from ._numba_kernels import integrate_cpp_rk4

        omega0 = float(params["omega0"])
        N = float(params.get("N", 0.25))
        chi_scale = float(params.get("chi_scale", 1.0))
        p_model = float(params.get("P_model", params.get("P", self._p_model_base)))
        domega0_dJ = float(params.get("domega0_dJ", self._domega0_dJ_base))
        d0_scale = float(params.get("d0_scale", 1.0))
        clamp_u = float(params.get("clamp_u", 0.999))

        coefficients = (omega0, N, chi_scale, p_model, domega0_dJ, d0_scale)
        if not all(np.isfinite(value) for value in coefficients):
            raise ValueError("CPP autofit coefficients must be finite")
        if omega0 <= 0.0:
            raise ValueError("omega0 must be positive")
        if chi_scale <= 0.0 or d0_scale <= 0.0:
            raise ValueError("chi_scale and d0_scale must be positive")
        if not 0.0 < clamp_u <= 1.0:
            raise ValueError("clamp_u must lie in (0, 1]")

        J = self._J_const
        chi_val = (
            chi_scale
            * (-float(self._polarity))
            * self._chi_prefactor_per_p
            * p_model
            * J
        )
        omega0_eff = omega0 + domega0_dJ * J
        if omega0_eff <= 0.0:
            raise ValueError("omega0_eff must remain positive on the fast path")

        t_out, sx_out, sy_out = integrate_cpp_rk4(
            self._t0,
            self._sim_t1,
            self._dt,
            self._s0_x,
            self._s0_y,
            chi_val,
            omega0_eff,
            N,
            self._d0 * d0_scale,
            self._d1 * d0_scale,
            float(self._polarity),
            0.0,
            0.0,
            4,
        )

        if np.isfinite(clamp_u) and clamp_u > 0.0 and sx_out.size:
            u = np.sqrt(sx_out**2 + sy_out**2)
            hit = np.flatnonzero(u >= clamp_u)
            if hit.size:
                idx = int(hit[0])
                u_hit = float(u[idx])
                if u_hit > 0.0:
                    scale = clamp_u / max(u_hit, 1e-30)
                    sx_edge = float(sx_out[idx] * scale)
                    sy_edge = float(sy_out[idx] * scale)
                else:
                    sx_edge = float(clamp_u)
                    sy_edge = 0.0
                sx_out[idx:] = sx_edge
                sy_out[idx:] = sy_edge
                edge_limited = True
                edge_hit_time = float(t_out[idx])
            else:
                edge_limited = False
                edge_hit_time = None
        else:
            edge_limited = False
            edge_hit_time = None

        R = self._R
        return ThieleTrajectoryResult(
            model_name=f"CPP Thiele STNO (p={self._polarity:+d}, fast)",
            t=t_out,
            x=sx_out * R,
            y=sy_out * R,
            sx=sx_out,
            sy=sy_out,
            disk_radius=R,
            params={"clamp_u": clamp_u},
            metadata={
                "mode": "CPP",
                "integrator": "numba_rk4",
                "edge_limited": bool(edge_limited),
                "edge_hit_time": edge_hit_time,
                "edge_behavior": "freeze" if edge_limited else None,
            },
        )

    def _simulate_cip_fast(self, params: dict[str, Any]):
        """Ultra-fast CIP simulation via Numba RK4."""
        from mmpp.analytical.thiele import ThieleTrajectoryResult

        from ._numba_kernels import integrate_cip_rk4

        omega0 = float(params["omega0"])
        if not np.isfinite(omega0) or omega0 <= 0.0:
            raise ValueError("omega0 must be finite and positive")

        J = self._J_const
        polarization = float(params.get("P", 0.0))
        beta_value = params.get("beta_nonadiabatic", params.get("beta", self._beta_cip))
        beta = self._beta_cip if beta_value is None else float(beta_value)
        if not np.isfinite(polarization) or not np.isfinite(beta):
            raise ValueError("P and beta_nonadiabatic must be finite")
        u0 = self._u0_prefactor_per_p * polarization * J
        u0_cx = u0 * self._current_dir[0]
        u0_cy = u0 * self._current_dir[1]

        t_out, X_out, Y_out = integrate_cip_rk4(
            self._t0,
            self._sim_t1,
            self._dt,
            self._r0_x,
            self._r0_y,
            omega0,
            u0_cx,
            u0_cy,
            self._alpha_cip,
            beta,
            self._dG_cip,
            float(self._polarity),
            0.0,
            0.0,
            4,
        )

        R = self._R
        return ThieleTrajectoryResult(
            model_name=f"CIP Thiele (p={self._polarity:+d}, fast)",
            t=t_out,
            x=X_out,
            y=Y_out,
            sx=X_out / R,
            sy=Y_out / R,
            disk_radius=R,
            params={},
            metadata={"mode": "CIP", "integrator": "numba_rk4"},
        )

    def _simulate_cpp_scipy(self, params: dict[str, Any]):
        """Fallback: CPP simulation via scipy solve_ivp."""
        from mmpp.analytical import (
            CPPThieleModel,
            DiskGeometry,
            MaterialParams,
            current_dc,
        )

        mat = MaterialParams(
            Ms=float(params["Ms"]),
            alpha=float(params["alpha"]),
            P=float(params["P_model"] if "P_model" in params else params["P"]),
            A=float(params.get("A", 1.3e-11)),
            gamma=float(params.get("gamma", 1.76085963023e11)),
        )
        geo = DiskGeometry(R=float(params["R"]), L=float(params["L"]))
        polarity = int(np.sign(float(params.get("polarity", 1))) or 1)

        model = CPPThieleModel(
            material=mat,
            geom=geo,
            omega0=float(params["omega0"]),
            N=float(params.get("N", 0.25)),
            polarity=polarity,
            domega0_dJ=float(params.get("domega0_dJ", 0.0)),
            field=coerce_field_for_autofit(self._field),
            chi_scale=float(params.get("chi_scale", 1.0)),
            torque_thickness=float(params.get("L_stt", params["L"])),
        )
        d0_scale = float(params.get("d0_scale", 1.0))
        if d0_scale != 1.0:
            model._d0 *= d0_scale
            model._d1 *= d0_scale
        return model.simulate(
            t_span=(self._t0, self._sim_t1),
            s0=(self._s0_x, self._s0_y),
            J_func=self._current_density
            if callable(self._current_density)
            else current_dc(float(self._J_const)),
            dt=self._dt,
            clamp_u=float(params.get("clamp_u", 0.999)),
            edge_behavior="freeze",
            rtol=1e-6,
            atol=1e-9,
        )

    def _simulate_cip_scipy(self, params: dict[str, Any]):
        """Fallback: CIP simulation via scipy solve_ivp."""
        from mmpp.analytical import (
            CIPThieleModel,
            DiskGeometry,
            MaterialParams,
            current_dc,
        )

        mat = MaterialParams(
            Ms=float(params["Ms"]),
            alpha=float(params["alpha"]),
            P=float(params["P"]),
            A=float(params.get("A", 1.3e-11)),
            beta_nonadiabatic=params.get("beta_nonadiabatic", params.get("beta", None)),
            gamma=float(params.get("gamma", 1.76085963023e11)),
        )
        geo = DiskGeometry(R=float(params["R"]), L=float(params["L"]))
        polarity = int(np.sign(float(params.get("polarity", 1))) or 1)
        current_dir = tuple(params.get("current_dir", (1.0, 0.0)))

        model = CIPThieleModel(
            material=mat,
            geom=geo,
            omega0=float(params["omega0"]),
            polarity=polarity,
            current_dir=current_dir,
            field=coerce_field_for_autofit(self._field),
        )
        return model.simulate(
            t_span=(self._t0, self._sim_t1),
            r0=(self._r0_x, self._r0_y),
            J_func=self._current_density
            if callable(self._current_density)
            else current_dc(float(self._J_const)),
            dt=self._dt,
            rtol=1e-6,
            atol=1e-9,
        )


def coerce_field_for_autofit(value: Any):
    if value is None:
        return None
    from mmpp.analytical.thiele import ExternalField

    return ExternalField.from_any(value)


def field_has_effect(field: Any) -> bool:
    if field is None:
        return False
    try:
        coerced = coerce_field_for_autofit(field)
    except Exception:
        return True
    if coerced is None:
        return False
    return any(
        abs(float(component)) > 1e-30
        for component in (coerced.Bx_T, coerced.By_T, coerced.Bz_T)
    )


__all__ = ["SimulationContext", "coerce_field_for_autofit", "field_has_effect"]
