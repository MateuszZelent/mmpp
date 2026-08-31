"""Field-resolved CPP Thiele adapter returning canonical vortex trajectories."""

from __future__ import annotations

from dataclasses import replace
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

    def _repr_html_(self) -> str:
        """Render the canonical tabbed notebook helper for this adapter."""
        from ._html import adapter_repr_html

        return adapter_repr_html(self, variant="field_resolved_cpp")


def field_resolved_cpp(
    *,
    material=None,
    geom=None,
    polarity: int | str | None = 1,
    chirality: int | str | None = 1,
    omega0: float | None = None,
    N: float = 0.25,
    domega0_dJ: float = 0.0,
    polarizer: tuple[float, float, float] | tuple[float, float] | None = None,
    calibration: FieldResolvedCalibration | None = None,
    torque_thickness: float | None = None,
    fixed_layer_position: str | None = None,
    Lambda: float | None = None,
    epsilonprime: float | None = None,
    mean_m_dot_p: float = 0.0,
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
        mean_m_dot_p=float(mean_m_dot_p),
    )
    resolved_polarizer = spin_ctx.metadata.get(
        "polarizer", polarizer or (0.0, 0.0, 1.0)
    )

    resolved_calibration = (
        calibration if calibration is not None else FieldResolvedCalibration()
    )
    resolved_calibration = replace(
        resolved_calibration,
        domega_dJ=float(resolved_calibration.domega_dJ)
        + float(spin_ctx.domega0_dJ_total),
    )

    # FieldResolvedCPPThieleModel projects p_z explicitly and its G uses the
    # physical magnetic thickness.  The reduced CPP material already contains
    # p_z and uses L_stt, so undo that projection once and rescale thickness.
    if spin_ctx.metadata:
        p_z = float(spin_ctx.metadata.get("p_z", 0.0))
        p_model = float(spin_ctx.material.P)
        if abs(p_z) > 1e-15:
            p_field_model = (
                -p_model * float(geo.L) / (float(spin_ctx.torque_thickness) * p_z)
            )
        else:
            p_field_model = 0.0
        convention = "mumax_reduced_to_dussaux"
    else:
        p_field_model = (
            float(mat_raw.P) * float(geo.L) / float(spin_ctx.torque_thickness)
        )
        convention = "direct_dussaux"

    field_material = replace(mat_raw, P=float(p_field_model))
    model = FieldResolvedCPPThieleModel(
        material=field_material,
        geom=geo,
        omega0=omega0_value,
        N=float(N),
        polarity=p,
        chirality=c,
        polarizer=resolved_polarizer,
        calibration=resolved_calibration,
    )
    return FieldResolvedCPPModelAdapter(
        model,
        polarity=p,
        metadata={
            "dataset_name": dataset_name,
            "omega0": float(omega0_value),
            "N": float(N),
            "chirality": c,
            "P_field_model": float(p_field_model),
            "field_cpp_convention": convention,
            "torque_thickness": float(spin_ctx.torque_thickness),
            **spin_ctx.metadata,
        },
    )


__all__ = ["FieldResolvedCPPModelAdapter", "field_resolved_cpp"]
