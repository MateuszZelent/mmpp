"""Thiele-equation analysis helpers for vortex nonlinear dynamics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from mmpp.analytical import (
    CIPThieleModel,
    CPPThieleModel,
    DiskGeometry,
    ExternalField,
    FieldCalibration,
    FieldFunc,
    MaterialParams,
    ThieleFJFitResult,
    ThieleOptimizationResult,
    ThieleTrajectoryResult,
    current_dc,
    fit_omega0_N_to_fJ,
    omega0_novosad,
)
from mmpp.analytical.constants import GAMMA_E, MU0

from ..core.models import TrajectoryResult
from .interactive import (
    build_thiele_dashboard,
    proxy_psd,
    proxy_signal_from_trajectory,
)
from .models import ThieleForceBalanceResult


def _attr_float(attrs: Any, keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        try:
            value = attrs.get(key, None)
        except Exception:  # pragma: no cover - attrs backend-specific
            value = None
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)


def _infer_dataset_nx_ny(job_result: Any, dataset_name: str | None) -> tuple[int, int] | None:
    if dataset_name is None:
        return None
    try:
        dataset = job_result[dataset_name]
    except Exception:
        return None

    shape = tuple(getattr(dataset, "shape", ()))
    if len(shape) < 3:
        return None
    if shape[-1] != 3:
        return None
    return int(shape[-2]), int(shape[-3])


def _coerce_force_series(
    force: Any,
    time: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
) -> np.ndarray:
    n_t = int(time.size)
    zeros = np.zeros((n_t, 2), dtype=float)
    if force is None:
        return zeros

    if callable(force):
        values = np.zeros((n_t, 2), dtype=float)
        for idx in range(n_t):
            out = force(
                float(time[idx]),
                float(x[idx]),
                float(y[idx]),
                float(vx[idx]),
                float(vy[idx]),
            )
            vec = np.asarray(out, dtype=float).reshape(-1)
            if vec.size != 2:
                raise ValueError("force callback must return 2 values: (Fx, Fy)")
            values[idx, :] = vec
        return values

    arr = np.asarray(force, dtype=float)
    if arr.shape == (2,):
        return np.repeat(arr[np.newaxis, :], n_t, axis=0)
    if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] == n_t:
        return arr
    raise ValueError("force must be None, callable, shape (2,), or shape (Nt,2)")


class ThieleAnalyzer:
    """Thiele-force diagnostics and analytical trajectory wrappers."""

    def __init__(self, job_result, dataset_name: str | None, core_interface):
        self._job = job_result
        self._dataset_name = dataset_name
        self._core = core_interface

    def _resolve_trajectory(self, trajectory: TrajectoryResult | None) -> TrajectoryResult:
        if trajectory is not None:
            return trajectory
        return self._core.track()

    def _infer_polarity(self, trajectory: TrajectoryResult | None = None) -> int:
        if trajectory is not None:
            values = np.asarray(trajectory.polarity, dtype=float)
            if values.size:
                mean_val = float(np.mean(values))
                if mean_val > 0.0:
                    return 1
                if mean_val < 0.0:
                    return -1

        attrs = getattr(self._job, "attrs", {})
        val = _attr_float(attrs, ("polarity", "p"), 1.0)
        return 1 if val >= 0.0 else -1

    def _resolve_material(self, material: MaterialParams | dict[str, float] | None) -> MaterialParams:
        if isinstance(material, MaterialParams):
            return material

        attrs = getattr(self._job, "attrs", {})
        payload: dict[str, float] = {
            "Ms": _attr_float(attrs, ("Ms", "ms", "Msat"), 8.0e5),
            "alpha": _attr_float(attrs, ("alpha",), 0.01),
            "P": _attr_float(attrs, ("P", "pol", "polarization"), 0.35),
            "A": _attr_float(attrs, ("Aex", "A"), 1.3e-11),
        }
        if material is not None:
            payload.update({key: float(value) for key, value in material.items()})
        return MaterialParams(**payload)

    def _resolve_geometry(
        self,
        geometry: DiskGeometry | dict[str, float] | None,
    ) -> DiskGeometry:
        if isinstance(geometry, DiskGeometry):
            return geometry

        attrs = getattr(self._job, "attrs", {})
        dx = _attr_float(attrs, ("dx",), 1.0e-9)
        dy = _attr_float(attrs, ("dy",), dx)
        dims = _infer_dataset_nx_ny(self._job, self._dataset_name)
        if dims is None:
            est_r = 50.0e-9
        else:
            nx, ny = dims
            est_r = 0.45 * min(nx * dx, ny * dy)

        dz = _attr_float(attrs, ("dz",), 1.0e-9)
        nz = _attr_float(attrs, ("Nz",), 1.0)
        est_l = _attr_float(attrs, ("thickness", "L", "d"), dz * max(nz, 1.0))

        payload: dict[str, float] = {
            "R": est_r,
            "L": est_l,
        }
        if geometry is not None:
            payload.update({key: float(value) for key, value in geometry.items()})
        return DiskGeometry(**payload)

    def _build_cpp_model(
        self,
        *,
        material: MaterialParams | dict[str, float] | None,
        geometry: DiskGeometry | dict[str, float] | None,
        omega0: float | None,
        N: float,
        polarity: int | None,
        omega0_Oe_per_J: float = 0.0,
        field: ExternalField | None = None,
        field_cal: FieldCalibration | None = None,
        chi_scale: float = 1.0,
    ) -> tuple[CPPThieleModel, MaterialParams, DiskGeometry]:
        mat = self._resolve_material(material)
        geo = self._resolve_geometry(geometry)
        omega0_val = float(omega0_novosad(mat, geo) if omega0 is None else omega0)
        p = self._infer_polarity() if polarity is None else int(np.sign(polarity) or 1)
        model = CPPThieleModel(
            material=mat,
            geom=geo,
            omega0=omega0_val,
            N=float(N),
            polarity=p,
            omega0_Oe_per_J=float(omega0_Oe_per_J),
            field=field,
            field_cal=field_cal,
            chi_scale=float(chi_scale),
        )
        return model, mat, geo

    def force_balance(
        self,
        *,
        trajectory: TrajectoryResult | None = None,
        polarity: int | None = None,
        vorticity: int = 1,
        Ms: float | None = None,
        thickness: float | None = None,
        alpha: float | None = None,
        eta: float = 1.0,
        gamma0: float | None = None,
        kappa: float | None = None,
        center: tuple[float, float] | None = None,
        stt_force: np.ndarray | tuple[float, float] | Callable[..., Any] | None = None,
        oersted_force: np.ndarray | tuple[float, float] | Callable[..., Any] | None = None,
    ) -> ThieleForceBalanceResult:
        """Decompose effective forces from tracked vortex trajectory."""
        traj = self._resolve_trajectory(trajectory)

        time = np.asarray(traj.time, dtype=float)
        x = np.asarray(traj.x, dtype=float)
        y = np.asarray(traj.y, dtype=float)
        vx, vy = traj.velocity
        vx = np.asarray(vx, dtype=float)
        vy = np.asarray(vy, dtype=float)

        attrs = getattr(self._job, "attrs", {})
        p = int(self._infer_polarity(traj) if polarity is None else np.sign(polarity) or 1)
        w = int(np.sign(vorticity) or 1)

        ms = float(Ms) if Ms is not None else _attr_float(attrs, ("Ms", "ms", "Msat"), 8.0e5)
        l_thick = (
            float(thickness)
            if thickness is not None
            else _attr_float(attrs, ("thickness", "L", "d"), 20.0e-9)
        )
        alpha_val = float(alpha) if alpha is not None else _attr_float(attrs, ("alpha",), 0.01)
        gamma0_val = float(gamma0) if gamma0 is not None else float(GAMMA_E * MU0)

        G = float(2.0 * np.pi * p * w * ms * l_thick / max(gamma0_val, 1e-30))
        D = float(abs(alpha_val * eta * G))

        if center is None:
            x0 = float(np.mean(x)) if x.size else 0.0
            y0 = float(np.mean(y)) if y.size else 0.0
        else:
            x0 = float(center[0])
            y0 = float(center[1])

        rx = x - x0
        ry = y - y0
        gyro_force = np.column_stack((-G * vy, G * vx))

        if kappa is None:
            target_x = gyro_force[:, 0] + D * vx
            target_y = gyro_force[:, 1] + D * vy
            denom = float(np.sum(rx * rx + ry * ry))
            if denom > 1e-30:
                num = float(np.sum(rx * target_x + ry * target_y))
                kappa_val = -num / denom
            else:
                kappa_val = 0.0
        else:
            kappa_val = float(kappa)

        conservative_force = np.column_stack((-kappa_val * rx, -kappa_val * ry))
        dissipative_force = np.column_stack((-D * vx, -D * vy))
        stt = _coerce_force_series(stt_force, time, x, y, vx, vy)
        oersted = _coerce_force_series(oersted_force, time, x, y, vx, vy)
        residual = gyro_force - conservative_force - dissipative_force - stt - oersted

        return ThieleForceBalanceResult(
            time=time,
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            gyro_force=gyro_force,
            conservative_force=conservative_force,
            dissipative_force=dissipative_force,
            stt_force=stt,
            oersted_force=oersted,
            residual_force=residual,
            G=G,
            D=D,
            kappa=kappa_val,
            polarity=p,
            vorticity=w,
            metadata={
                "center": (x0, y0),
                "eta": float(eta),
                "alpha": alpha_val,
                "Ms": ms,
                "thickness": l_thick,
                "gamma0": gamma0_val,
                "dataset_name": self._dataset_name,
            },
        )

    def simulate_cpp(
        self,
        *,
        material: MaterialParams | dict[str, float] | None = None,
        geometry: DiskGeometry | dict[str, float] | None = None,
        omega0: float | None = None,
        N: float = 0.25,
        polarity: int | None = None,
        current_density: float = 0.0,
        current_waveform: Callable[[float], float] | None = None,
        t_span: tuple[float, float] = (0.0, 20.0e-9),
        s0: tuple[float, float] = (1.0e-3, 0.0),
        dt: float = 1.0e-11,
        field: ExternalField | None = None,
        field_cal: FieldCalibration | None = None,
        B_func: FieldFunc | None = None,
        chi_scale: float = 1.0,
        **simulate_kwargs,
    ) -> ThieleTrajectoryResult:
        """Run analytical CPP Thiele simulation and return trajectory result."""
        model, _, _ = self._build_cpp_model(
            material=material,
            geometry=geometry,
            omega0=omega0,
            N=N,
            polarity=polarity,
            field=field,
            field_cal=field_cal,
            chi_scale=chi_scale,
        )
        j_func = current_waveform if current_waveform is not None else current_dc(float(current_density))
        result = model.simulate(t_span=t_span, s0=s0, J_func=j_func, B_func=B_func, dt=dt, **simulate_kwargs)
        result.metadata["dataset_name"] = self._dataset_name
        result.metadata["source"] = "mmpp.solitons.vortex.nonlinear.thiele"
        return result

    def simulate_cpp_sde(
        self,
        *,
        material: MaterialParams | dict[str, float] | None = None,
        geometry: DiskGeometry | dict[str, float] | None = None,
        omega0: float | None = None,
        N: float = 0.25,
        polarity: int | None = None,
        current_density: float = 0.0,
        current_waveform: Callable[[float], float] | None = None,
        t_span: tuple[float, float] = (0.0, 20.0e-9),
        s0: tuple[float, float] = (0.0, 0.0),
        dt: float = 1.0e-11,
        temperature_k: float = 300.0,
        diffusion: float | None = None,
        noise_scale: float = 1.0,
        seed: int | None = None,
        clamp_u: float = 0.999,
        field: ExternalField | None = None,
        field_cal: FieldCalibration | None = None,
        B_func: FieldFunc | None = None,
        chi_scale: float = 1.0,
    ) -> ThieleTrajectoryResult:
        """Run stochastic CPP Thiele simulation (Euler-Maruyama)."""
        model, _, _ = self._build_cpp_model(
            material=material,
            geometry=geometry,
            omega0=omega0,
            N=N,
            polarity=polarity,
            field=field,
            field_cal=field_cal,
            chi_scale=chi_scale,
        )
        j_func = current_waveform if current_waveform is not None else current_dc(float(current_density))
        result = model.simulate_sde(
            t_span=t_span,
            s0=s0,
            J_func=j_func,
            B_func=B_func,
            dt=dt,
            temperature_k=temperature_k,
            diffusion=diffusion,
            noise_scale=noise_scale,
            seed=seed,
            clamp_u=clamp_u,
        )
        result.metadata["dataset_name"] = self._dataset_name
        result.metadata["source"] = "mmpp.solitons.vortex.nonlinear.thiele"
        return result

    def threshold_current_dc(
        self,
        *,
        material: MaterialParams | dict[str, float] | None = None,
        geometry: DiskGeometry | dict[str, float] | None = None,
        omega0: float | None = None,
        N: float = 0.25,
        polarity: int | None = None,
        field: ExternalField | None = None,
        field_cal: FieldCalibration | None = None,
        chi_scale: float = 1.0,
    ) -> float:
        """Return threshold DC current density for CPP model [A/m²]."""
        model, _, _ = self._build_cpp_model(
            material=material,
            geometry=geometry,
            omega0=omega0,
            N=N,
            polarity=polarity,
            field=field,
            field_cal=field_cal,
            chi_scale=chi_scale,
        )
        return float(model.threshold_current_dc())

    def predict_frequency_dc(
        self,
        J_dc: float,
        *,
        material: MaterialParams | dict[str, float] | None = None,
        geometry: DiskGeometry | dict[str, float] | None = None,
        omega0: float | None = None,
        N: float = 0.25,
        polarity: int | None = None,
        allow_edge: bool = False,
        omega0_Oe_per_J: float = 0.0,
        field: ExternalField | None = None,
        field_cal: FieldCalibration | None = None,
        chi_scale: float = 1.0,
    ) -> float | None:
        """Predict steady-state frequency for given DC current density [Hz]."""
        model, _, _ = self._build_cpp_model(
            material=material,
            geometry=geometry,
            omega0=omega0,
            N=N,
            polarity=polarity,
            omega0_Oe_per_J=omega0_Oe_per_J,
            field=field,
            field_cal=field_cal,
            chi_scale=chi_scale,
        )
        return model.predict_frequency_dc(
            J_dc,
            allow_edge=allow_edge,
        )

    def fit_omega0_N_to_fJ(
        self,
        J_data: np.ndarray,
        f_data_hz: np.ndarray,
        *,
        material: MaterialParams | dict[str, float] | None = None,
        geometry: DiskGeometry | dict[str, float] | None = None,
        polarity: int | None = None,
        initial_omega0: float | None = None,
        initial_N: float = 0.25,
        fit_omega0_Oe_per_J: bool = False,
        initial_omega0_Oe_per_J: float = 0.0,
        fit_chi_scale: bool = False,
        initial_chi_scale: float = 1.0,
        allow_edge: bool = False,
    ) -> ThieleFJFitResult:
        """Fit CPP-model ``omega0`` and ``N`` (optionally ``chi_scale``) to target ``f(J)`` data."""
        mat = self._resolve_material(material)
        geo = self._resolve_geometry(geometry)
        p = self._infer_polarity() if polarity is None else int(np.sign(polarity) or 1)
        return fit_omega0_N_to_fJ(
            J_data,
            f_data_hz,
            material=mat,
            geom=geo,
            polarity=p,
            initial_omega0=initial_omega0,
            initial_N=initial_N,
            fit_omega0_Oe_per_J=fit_omega0_Oe_per_J,
            initial_omega0_Oe_per_J=initial_omega0_Oe_per_J,
            fit_chi_scale=fit_chi_scale,
            initial_chi_scale=initial_chi_scale,
            allow_edge=allow_edge,
        )

    def optimize_current_for_target_frequency(
        self,
        target_frequency_hz: float,
        *,
        material: MaterialParams | dict[str, float] | None = None,
        geometry: DiskGeometry | dict[str, float] | None = None,
        omega0: float | None = None,
        N: float = 0.25,
        polarity: int | None = None,
        J_bounds: tuple[float, float] | None = None,
        allow_edge: bool = False,
        omega0_Oe_per_J: float = 0.0,
        n_grid: int = 300,
        field: ExternalField | None = None,
        field_cal: FieldCalibration | None = None,
        chi_scale: float = 1.0,
    ) -> ThieleOptimizationResult:
        """Optimize DC current density to match target frequency."""
        model, _, _ = self._build_cpp_model(
            material=material,
            geometry=geometry,
            omega0=omega0,
            N=N,
            polarity=polarity,
            omega0_Oe_per_J=omega0_Oe_per_J,
            field=field,
            field_cal=field_cal,
            chi_scale=chi_scale,
        )
        return model.optimize_current_for_target_frequency(
            target_frequency_hz,
            J_bounds=J_bounds,
            allow_edge=allow_edge,
            n_grid=n_grid,
        )

    def proxy_signal(
        self,
        trajectory: TrajectoryResult,
        *,
        disk_radius: float | None = None,
        polarizer: tuple[float, float, float] = (1.0, 0.0, 0.0),
        center: tuple[float, float] | None = None,
        cubic: float = 0.0,
    ) -> np.ndarray:
        """Compute MTJ readout proxy from trajectory."""
        return proxy_signal_from_trajectory(
            trajectory,
            disk_radius=disk_radius,
            polarizer=polarizer,
            center=center,
            cubic=cubic,
        )

    def proxy_psd(
        self,
        signal: np.ndarray,
        *,
        dt: float,
        method: str = "welch",
        nperseg: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute PSD of proxy readout signal."""
        return proxy_psd(
            signal,
            dt=dt,
            method=method,
            nperseg=nperseg,
        )

    def interactive_dashboard(self, **kwargs):
        """Create interactive ipywidgets dashboard for CPP Thiele tuning."""
        return build_thiele_dashboard(self, **kwargs)

    def simulate_cip(
        self,
        *,
        material: MaterialParams | dict[str, float] | None = None,
        geometry: DiskGeometry | dict[str, float] | None = None,
        omega0: float | None = None,
        polarity: int | None = None,
        current_density: float = 0.0,
        current_waveform: Callable[[float], float] | None = None,
        current_dir: tuple[float, float] = (1.0, 0.0),
        t_span: tuple[float, float] = (0.0, 20.0e-9),
        r0: tuple[float, float] = (1.0e-9, 0.0),
        dt: float = 1.0e-12,
        field: ExternalField | None = None,
        field_cal: FieldCalibration | None = None,
        B_func: FieldFunc | None = None,
        **simulate_kwargs,
    ) -> ThieleTrajectoryResult:
        """Run analytical CIP Thiele simulation and return trajectory result."""
        mat = self._resolve_material(material)
        geo = self._resolve_geometry(geometry)
        omega0_val = float(omega0_novosad(mat, geo) if omega0 is None else omega0)
        p = self._infer_polarity() if polarity is None else int(np.sign(polarity) or 1)

        model = CIPThieleModel(
            material=mat,
            geom=geo,
            omega0=omega0_val,
            polarity=p,
            current_dir=current_dir,
            field=field,
            field_cal=field_cal,
        )
        j_func = current_waveform if current_waveform is not None else current_dc(float(current_density))
        result = model.simulate(t_span=t_span, r0=r0, J_func=j_func, B_func=B_func, dt=dt, **simulate_kwargs)
        result.metadata["dataset_name"] = self._dataset_name
        result.metadata["source"] = "mmpp.solitons.vortex.nonlinear.thiele"
        return result


__all__ = ["ThieleAnalyzer"]
