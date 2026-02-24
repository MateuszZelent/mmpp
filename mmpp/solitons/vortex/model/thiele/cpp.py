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
    resolve_current_waveform,
)


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
    job_result=None,
    dataset_name: str | None = None,
    slice_info=None,
) -> CPPModelAdapter:
    """Build CPP Thiele model adapter (dataset-free or dataset-aware)."""
    _ = slice_info  # reserved for stage-3 model/data coupling

    mat = infer_material_params(material, job_result=job_result)
    geo = infer_disk_geometry(geom, job_result=job_result, dataset_name=dataset_name)
    p = infer_polarity(polarity, job_result=job_result)
    omega0_value = infer_omega0(omega0, material=mat, geometry=geo)

    model = CPPThieleModel(
        material=mat,
        geom=geo,
        omega0=omega0_value,
        N=float(N),
        polarity=p,
        domega0_dJ=float(domega0_dJ),
        field=field,
        field_cal=field_cal,
        chi_scale=float(chi_scale),
    )
    return CPPModelAdapter(
        model,
        polarity=p,
        metadata={
            "dataset_name": dataset_name,
            "omega0": float(omega0_value),
            "N": float(N),
        },
        job_result=job_result,
    )


__all__ = ["CPPModelAdapter", "cpp"]
