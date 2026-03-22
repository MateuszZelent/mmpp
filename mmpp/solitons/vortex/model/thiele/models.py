"""Shared builders and context helpers for Thiele model adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from mmpp.analytical import (
    DiskGeometry,
    MaterialParams,
    current_dc,
    omega0_novosad,
    reduce_mumax_slonczewski_cpp,
)


@dataclass
class ThieleBuildContext:
    """Resolved context used to build analytical Thiele model adapters."""

    material: MaterialParams
    geometry: DiskGeometry
    polarity: int
    omega0: float


@dataclass
class CPPSpinTorqueContext:
    """Resolved MuMax-style Slonczewski CPP parameters for the adapter layer."""

    material: MaterialParams
    torque_thickness: float
    domega0_dJ_total: float
    metadata: dict[str, Any]


def _attr_float(attrs: Any, keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        value = None
        try:
            value = attrs.get(key, None)
        except Exception:
            value = None
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)


def _payload_float(source: Any, keys: tuple[str, ...], default: float) -> float:
    if source is None:
        return float(default)
    if isinstance(source, dict):
        for key in keys:
            if key not in source:
                continue
            try:
                return float(source[key])
            except (TypeError, ValueError):
                continue
        return float(default)
    for key in keys:
        value = getattr(source, key, None)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)


def infer_material_params(
    material: MaterialParams | dict[str, float] | None,
    *,
    job_result=None,
) -> MaterialParams:
    if isinstance(material, MaterialParams):
        return material

    attrs = getattr(job_result, "attrs", {}) if job_result is not None else {}
    payload = {
        "Ms": _attr_float(attrs, ("Ms", "ms", "Msat"), 8.0e5),
        "alpha": _attr_float(attrs, ("alpha",), 0.01),
        "P": _attr_float(attrs, ("P", "pol", "polarization"), 0.35),
        "A": _attr_float(attrs, ("Aex", "A"), 1.3e-11),
    }
    if material is not None:
        if isinstance(material, dict):
            for key, value in material.items():
                if key not in {"Ms", "alpha", "P", "A", "beta_nonadiabatic", "gamma"}:
                    continue
                payload[key] = float(value)
        else:
            for key in ("Ms", "alpha", "P", "A", "beta_nonadiabatic", "gamma"):
                value = getattr(material, key, None)
                if value is None:
                    continue
                payload[key] = float(value)
    return MaterialParams(**payload)


def infer_disk_geometry(
    geom: DiskGeometry | dict[str, float] | None,
    *,
    job_result=None,
    dataset_name: str | None = None,
) -> DiskGeometry:
    if isinstance(geom, DiskGeometry):
        return geom

    attrs = getattr(job_result, "attrs", {}) if job_result is not None else {}
    dx = _attr_float(attrs, ("dx",), 1e-9)
    dy = _attr_float(attrs, ("dy",), dx)

    radius_guess = 50e-9
    if job_result is not None and dataset_name:
        try:
            dataset = getattr(job_result, dataset_name)
            shape = tuple(getattr(dataset, "shape", ()))
            if len(shape) >= 4:
                nx = int(shape[-2])
                ny = int(shape[-3])
                radius_guess = 0.45 * min(nx * dx, ny * dy)
        except Exception:
            pass

    dz = _attr_float(attrs, ("dz",), 1e-9)
    nz = _attr_float(attrs, ("Nz",), 1.0)
    thickness_guess = _attr_float(attrs, ("thickness", "L", "d"), dz * max(nz, 1.0))

    payload = {
        "R": radius_guess,
        "L": thickness_guess,
    }
    if geom is not None:
        payload.update({key: float(value) for key, value in geom.items()})
    return DiskGeometry(**payload)


def infer_polarity(
    polarity: int | str | None,
    *,
    job_result=None,
) -> int:
    if isinstance(polarity, str):
        token = polarity.strip().lower()
        if token in {"auto", "from_data"}:
            polarity = None
        elif token in {"+1", "up", "positive", "pos"}:
            return 1
        elif token in {"-1", "down", "negative", "neg"}:
            return -1

    if polarity is not None:
        value = int(np.sign(int(polarity)))
        return value if value != 0 else 1

    attrs = getattr(job_result, "attrs", {}) if job_result is not None else {}
    value = _attr_float(attrs, ("polarity", "p"), 1.0)
    return 1 if value >= 0.0 else -1


def infer_omega0(
    omega0: float | None,
    *,
    material: MaterialParams,
    geometry: DiskGeometry,
) -> float:
    if omega0 is not None:
        return float(omega0)
    return float(omega0_novosad(material, geometry))


def resolve_current_waveform(
    J_func: Callable[[float], float] | float | str | None,
    *,
    job_result=None,
) -> Callable[[float], float]:
    if callable(J_func):
        return J_func
    if isinstance(J_func, (int, float)):
        return current_dc(float(J_func))
    if isinstance(J_func, str):
        token = J_func.strip().lower()
        if token in {"auto_from_table", "auto", "table"}:
            if job_result is not None:
                try:
                    table = job_result["table"]
                    for key in ("J", "j", "Jdc", "J_dc", "current_density"):
                        if key in table:
                            values = np.asarray(table[key][:], dtype=float).reshape(-1)
                            if values.size:
                                return current_dc(float(np.mean(values)))
                except Exception:
                    pass
            return current_dc(0.0)
        raise ValueError(
            "Unsupported J_func string value. Use callable, float, or 'auto_from_table'."
        )
    return current_dc(0.0)


def resolve_cpp_spin_torque_context(
    *,
    material: MaterialParams,
    geometry: DiskGeometry,
    domega0_dJ: float = 0.0,
    torque_thickness: float | None = None,
    polarizer: tuple[float, float, float] | tuple[float, float] | None = None,
    fixed_layer_position: str | None = None,
    Lambda: float | None = None,
    epsilonprime: float | None = None,
) -> CPPSpinTorqueContext:
    """Resolve effective MuMax-style CPP Slonczewski coefficients for the adapter."""
    if polarizer is None and Lambda is None and epsilonprime is None and fixed_layer_position is None:
        return CPPSpinTorqueContext(
            material=material,
            torque_thickness=float(geometry.L if torque_thickness is None else torque_thickness),
            domega0_dJ_total=float(domega0_dJ),
            metadata={},
        )

    reduction = reduce_mumax_slonczewski_cpp(
        material=material,
        torque_thickness=float(geometry.L if torque_thickness is None else torque_thickness),
        polarizer=(0.0, 0.0, 1.0) if polarizer is None else polarizer,
        fixed_layer_position="top" if fixed_layer_position is None else fixed_layer_position,
        Lambda=1.0 if Lambda is None else float(Lambda),
        epsilonprime=0.0 if epsilonprime is None else float(epsilonprime),
    )

    return CPPSpinTorqueContext(
        material=replace(material, P=float(reduction.pump_polarization)),
        torque_thickness=float(reduction.torque_thickness),
        domega0_dJ_total=float(domega0_dJ) + float(reduction.phase_omega_per_J),
        metadata={
            "P_raw": float(material.P),
            "P_eff": float(reduction.epsilon),
            "P_model": float(reduction.pump_polarization),
            "phase_polarization": float(reduction.phase_polarization),
            "p_z": float(reduction.p_z),
            "polarizer": tuple(float(v) for v in reduction.polarizer),
            "fixed_layer_position": reduction.fixed_layer_position,
            "Lambda": float(reduction.Lambda),
            "epsilonprime": float(reduction.epsilonprime),
            "domega0_dJ_stt": float(reduction.phase_omega_per_J),
        },
    )


__all__ = [
    "ThieleBuildContext",
    "CPPSpinTorqueContext",
    "infer_material_params",
    "infer_disk_geometry",
    "infer_polarity",
    "infer_omega0",
    "resolve_current_waveform",
    "resolve_cpp_spin_torque_context",
]
