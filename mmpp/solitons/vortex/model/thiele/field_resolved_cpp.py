"""Field-resolved CPP Thiele adapter returning canonical vortex trajectories."""

from __future__ import annotations

from typing import Any, cast

from mmpp.analytical import FieldResolvedCalibration, FieldResolvedCPPThieleModel

from ...._method_helpers import InteractiveNodeMixin
from ..adapters import thiele_to_trajectory_result
from .models import (
    infer_disk_geometry,
    infer_material_params,
    infer_omega0,
    infer_polarity,
    resolve_cpp_spin_torque_context,
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


class FieldResolvedCPPModelAdapter(InteractiveNodeMixin):
    """Adapter exposing field-resolved CPP Thiele simulation as ``TrajectoryResult``."""

    _interactive_owner = "model"
    _interactive_nodes = frozenset({"simulate", "simulate_dc_sweep"})

    def __init__(
        self,
        model: FieldResolvedCPPThieleModel,
        *,
        polarity: int,
        metadata: dict[str, Any] | None = None,
    ):
        self._model = model
        self._polarity = 1 if int(polarity) >= 0 else -1
        self._metadata = dict(metadata or {})

    @property
    def model(self) -> FieldResolvedCPPThieleModel:
        """Access underlying analytical model instance."""
        return self._model

    def simulate(self, *, t_span: tuple[float, float], **kwargs):
        """Simulate and return a canonical vortex trajectory."""
        raw = self._model.simulate(t_span=t_span, **kwargs)
        return thiele_to_trajectory_result(
            raw,
            method="thiele_field_resolved_cpp",
            polarity=self._polarity,
            metadata={
                **self._metadata,
                "model_family": "thiele",
                "model_variant": "field_resolved_cpp",
            },
        )

    def simulate_dc_sweep(self, *args, **kwargs):
        """Delegate to the field-resolved model DC sweep helper."""
        return self._model.simulate_dc_sweep(*args, **kwargs)


def field_resolved_cpp(
    *,
    material=None,
    geom=None,
    polarity: int | str | None = 1,
    chirality: int | str | None = 1,
    omega0: float | None = None,
    N: float = 0.25,
    polarizer: tuple[float, float, float] | tuple[float, float] | None = None,
    calibration: FieldResolvedCalibration | None = None,
    torque_thickness: float | None = None,
    fixed_layer_position: str | None = None,
    Lambda: float | None = None,
    epsilonprime: float | None = None,
    job_result=None,
    dataset_name: str | None = None,
    slice_info=None,
) -> FieldResolvedCPPModelAdapter:
    """Build a dataset-aware field-resolved CPP Thiele model adapter."""
    _ = slice_info
    mat_raw = infer_material_params(material, job_result=job_result)
    geo = infer_disk_geometry(geom, job_result=job_result, dataset_name=dataset_name)
    p = infer_polarity(polarity, job_result=job_result)
    c = 1 if int(cast(Any, chirality)) >= 0 else -1
    omega0_value = infer_omega0(omega0, material=mat_raw, geometry=geo)

    spin_ctx = resolve_cpp_spin_torque_context(
        material=mat_raw,
        geometry=geo,
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
            else _resolve_optional_value(
                material, "fixed_layer_position", "FixedLayerPosition"
            )
        ),
        Lambda=Lambda
        if Lambda is not None
        else _resolve_optional_value(material, "Lambda"),
        epsilonprime=(
            epsilonprime
            if epsilonprime is not None
            else _resolve_optional_value(material, "epsilonprime")
        ),
    )
    resolved_polarizer = spin_ctx.metadata.get(
        "polarizer", polarizer or (0.0, 0.0, 1.0)
    )

    model = FieldResolvedCPPThieleModel(
        material=spin_ctx.material,
        geom=geo,
        omega0=omega0_value,
        N=float(N),
        polarity=p,
        chirality=c,
        polarizer=resolved_polarizer,
        calibration=calibration,
    )
    return FieldResolvedCPPModelAdapter(
        model,
        polarity=p,
        metadata={
            "dataset_name": dataset_name,
            "omega0": float(omega0_value),
            "N": float(N),
            "chirality": c,
            **spin_ctx.metadata,
        },
    )


__all__ = ["FieldResolvedCPPModelAdapter", "field_resolved_cpp"]
