"""CIP Thiele adapter returning canonical vortex trajectory results."""

from __future__ import annotations

from typing import Any

from mmpp.analytical import CIPThieleModel

from ..adapters import thiele_to_trajectory_result
from .models import (
    infer_disk_geometry,
    infer_material_params,
    infer_omega0,
    infer_polarity,
    resolve_current_waveform,
)


class CIPModelAdapter:
    """Thin adapter exposing ``simulate`` -> ``TrajectoryResult`` contract."""

    def __init__(
        self,
        model: CIPThieleModel,
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
    def model(self) -> CIPThieleModel:
        """Access underlying analytical model instance."""
        return self._model

    def simulate(
        self,
        *,
        t_span: tuple[float, float],
        J_func=None,
        dt: float = 1e-12,
        r0: tuple[float, float] = (1e-9, 0.0),
        **kwargs,
    ):
        waveform = resolve_current_waveform(J_func, job_result=self._job)
        raw = self._model.simulate(
            t_span=t_span,
            r0=r0,
            J_func=waveform,
            dt=float(dt),
            **kwargs,
        )
        return thiele_to_trajectory_result(
            raw,
            method="thiele_cip",
            polarity=self._polarity,
            metadata={
                **self._metadata,
                "model_family": "thiele",
                "model_variant": "cip",
            },
        )


def cip(
    *,
    material=None,
    geom=None,
    polarity: int | str | None = 1,
    omega0: float | None = None,
    current_dir: tuple[float, float] = (1.0, 0.0),
    field=None,
    field_cal=None,
    job_result=None,
    dataset_name: str | None = None,
    slice_info=None,
) -> CIPModelAdapter:
    """Build CIP Thiele model adapter (dataset-free or dataset-aware)."""
    _ = slice_info  # reserved for stage-3 model/data coupling

    mat = infer_material_params(material, job_result=job_result)
    geo = infer_disk_geometry(geom, job_result=job_result, dataset_name=dataset_name)
    p = infer_polarity(polarity, job_result=job_result)
    omega0_value = infer_omega0(omega0, material=mat, geometry=geo)

    model = CIPThieleModel(
        material=mat,
        geom=geo,
        omega0=omega0_value,
        polarity=p,
        current_dir=current_dir,
        field=field,
        field_cal=field_cal,
    )
    return CIPModelAdapter(
        model,
        polarity=p,
        metadata={
            "dataset_name": dataset_name,
            "omega0": float(omega0_value),
        },
        job_result=job_result,
    )


__all__ = ["CIPModelAdapter", "cip"]
