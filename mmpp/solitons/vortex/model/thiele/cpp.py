"""CPP Thiele adapter returning canonical vortex trajectory results."""

from __future__ import annotations

from typing import Any

from mmpp.analytical import CPPThieleModel

from ..adapters import thiele_to_trajectory_result
from .models import (
    infer_disk_geometry,
    infer_material_params,
    infer_omega0,
    infer_polarity,
    resolve_cpp_spin_torque_context,
    resolve_current_waveform,
)


def _resolve_optional_value(source: Any, *keys: str):
    if source is None:
        return None
    if isinstance(source, dict):
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
        return None
    for key in keys:
        value = getattr(source, key, None)
        if value is not None:
            return value
    return None


class CPPModelAdapter:
    """Thin adapter exposing ``simulate`` -> ``TrajectoryResult`` contract."""

    def __init__(
        self,
        model: CPPThieleModel,
        *,
        polarity: int,
        metadata: dict[str, Any] | None = None,
        job_result=None,
    ):
        self._model = model
        self._polarity = 1 if int(polarity) >= 0 else -1
        self._metadata = dict(metadata or {})
        self._job = job_result

    @property
    def model(self) -> CPPThieleModel:
        """Access underlying analytical model instance."""
        return self._model

    def simulate(
        self,
        *,
        t_span: tuple[float, float],
        J_func=None,
        dt: float = 1e-11,
        s0: tuple[float, float] = (1e-3, 0.0),
        **kwargs,
    ):
        waveform = resolve_current_waveform(J_func, job_result=self._job)
        raw = self._model.simulate(
            t_span=t_span,
            s0=s0,
            J_func=waveform,
            dt=float(dt),
            **kwargs,
        )
        return thiele_to_trajectory_result(
            raw,
            method="thiele_cpp",
            polarity=self._polarity,
            metadata={
                **self._metadata,
                "model_family": "thiele",
                "model_variant": "cpp",
            },
        )


def cpp(
    *,
    material=None,
    geom=None,
    polarity: int | str | None = 1,
    omega0: float | None = None,
    N: float = 0.25,
    domega0_dJ: float = 0.0,
    field=None,
    field_cal=None,
    chi_scale: float = 1.0,
    torque_thickness: float | None = None,
    polarizer: tuple[float, float, float] | tuple[float, float] | None = None,
    fixed_layer_position: str | None = None,
    Lambda: float | None = None,
    epsilonprime: float | None = None,
    job_result=None,
    dataset_name: str | None = None,
    slice_info=None,
) -> CPPModelAdapter:
    """Build CPP Thiele model adapter (dataset-free or dataset-aware)."""
    _ = slice_info  # reserved for stage-3 model/data coupling

    mat_raw = infer_material_params(material, job_result=job_result)
    geo = infer_disk_geometry(geom, job_result=job_result, dataset_name=dataset_name)
    p = infer_polarity(polarity, job_result=job_result)
    omega0_value = infer_omega0(omega0, material=mat_raw, geometry=geo)

    spin_ctx = resolve_cpp_spin_torque_context(
        material=mat_raw,
        geometry=geo,
        domega0_dJ=float(domega0_dJ),
        torque_thickness=(
            torque_thickness
            if torque_thickness is not None
            else _resolve_optional_value(material, "torque_thickness", "L_stt")
        ),
        polarizer=(
            polarizer
            if polarizer is not None
            else _resolve_optional_value(material, "polarizer", "FixedLayer")
        ),
        fixed_layer_position=(
            fixed_layer_position
            if fixed_layer_position is not None
            else _resolve_optional_value(material, "fixed_layer_position", "FixedLayerPosition")
        ),
        Lambda=Lambda if Lambda is not None else _resolve_optional_value(material, "Lambda"),
        epsilonprime=(
            epsilonprime
            if epsilonprime is not None
            else _resolve_optional_value(material, "epsilonprime")
        ),
    )

    model = CPPThieleModel(
        material=spin_ctx.material,
        geom=geo,
        omega0=omega0_value,
        N=float(N),
        polarity=p,
        domega0_dJ=float(spin_ctx.domega0_dJ_total),
        field=field,
        field_cal=field_cal,
        chi_scale=float(chi_scale),
        torque_thickness=float(spin_ctx.torque_thickness),
    )
    return CPPModelAdapter(
        model,
        polarity=p,
        metadata={
            "dataset_name": dataset_name,
            "omega0": float(omega0_value),
            "N": float(N),
            **spin_ctx.metadata,
        },
        job_result=job_result,
    )


__all__ = ["CPPModelAdapter", "cpp"]
