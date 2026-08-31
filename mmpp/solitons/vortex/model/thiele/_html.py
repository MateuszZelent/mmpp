"""Notebook helper cards for dataset-aware Thiele model adapters."""

from __future__ import annotations

import math
import uuid
from typing import Any

import numpy as np

from mmpp._repr_helpers import (
    NODE_COLOR_ADVANCED,
    NODE_COLOR_ANALYSIS,
    NODE_COLOR_COMPUTE,
    accessors_section_html,
    api_help_html,
    examples_section_html,
    metrics_section_html,
    node_card_html,
)


def _validity_html() -> str:
    return (
        "<div style='padding:4px 0;color:#cbd5e1;line-height:1.55;'>"
        "<p>The adapter resolves material, geometry, polarity, current, and "
        "MuMax-compatible spin-torque metadata before constructing the reduced "
        "model. Inspect <code>.model</code> and the card metrics before running.</p>"
        "<p>Results describe collective vortex-core motion. Spatial spin-wave "
        "modes, core reversal, annihilation, and post-expulsion dynamics require "
        "a micromagnetic solver.</p>"
        "<p>A radius inferred from the simulation box and an uncalibrated "
        "Novosad thin-disk frequency are estimates, not validation data.</p>"
        "</div>"
    )


def namespace_repr_html(namespace: Any) -> str:
    """Render the dataset-aware Thiele factory namespace."""
    api = api_help_html(
        namespace,
        title="Thiele model namespace API help",
        prefix="job[0].vortex.model.thiele",
        methods=["cpp", "cip", "field_resolved_cpp"],
        subtitle="Live signatures for dataset-aware model factories.",
        chrome=False,
    )
    return node_card_html(
        "Current-driven Thiele Models",
        icon="🌀",
        subtitle="Dataset-aware CPP, CIP, and field-resolved vortex-core dynamics.",
        sections=[
            metrics_section_html(
                [
                    (
                        "dataset",
                        namespace._dataset_name or "auto-detect",
                        NODE_COLOR_COMPUTE,
                    ),
                    (
                        "slice",
                        "custom"
                        if namespace._slice_info is not None
                        else "full geometry",
                        None,
                    ),
                    (
                        "parameter source",
                        "job metadata + explicit overrides"
                        if namespace._job is not None
                        else "explicit inputs",
                        NODE_COLOR_ANALYSIS,
                    ),
                ]
            ),
            accessors_section_html(
                [
                    (
                        "Build:",
                        [
                            (".cpp(...)", NODE_COLOR_COMPUTE),
                            (".cip(...)", NODE_COLOR_COMPUTE),
                            (".field_resolved_cpp(...)", NODE_COLOR_ADVANCED),
                        ],
                    )
                ]
            ),
            examples_section_html(
                "thiele = job[0].vortex.model.thiele\n"
                "cpp = thiele.cpp(N=0.30, polarity='auto')\n"
                "trajectory = cpp.simulate(\n"
                "    t_span=(0.0, 80e-9), J_func=current_ac(...), dt=10e-12\n"
                ")",
                title="Dataset-aware workflow",
            ),
        ],
        extra_tabs=[("Validity", _validity_html())],
        api=api,
        uid=f"vortex-thiele-namespace-{uuid.uuid4().hex[:8]}",
    )


def adapter_repr_html(adapter: Any, *, variant: str) -> str:
    """Render a dataset-aware CPP, CIP, or field-resolved adapter."""
    if variant not in {"cpp", "cip", "field_resolved_cpp"}:
        raise ValueError(f"unsupported Thiele adapter variant: {variant!r}")

    model = adapter._model
    metadata = adapter._metadata
    geometry = model.geom
    material = model.material
    f0_ghz = float(model.omega0) / (2.0 * math.pi * 1e9)
    metrics: list[tuple[str, object, str | None]] = [
        ("variant", variant, NODE_COLOR_ANALYSIS),
        ("dataset", metadata.get("dataset_name") or "explicit", NODE_COLOR_COMPUTE),
        ("polarity", int(adapter._polarity), None),
        ("radius", f"{float(geometry.R) * 1e9:.6g} nm", None),
        ("thickness", f"{float(geometry.L) * 1e9:.6g} nm", None),
        ("Ms", f"{float(material.Ms):.6g} A/m", None),
        ("alpha", f"{float(material.alpha):.6g}", None),
        ("f0", f"{f0_ghz:.6g} GHz", NODE_COLOR_ANALYSIS),
    ]

    if variant == "cpp":
        threshold = float(model.threshold_current_dc())
        metrics.extend(
            [
                ("N", f"{float(model.N):.6g}", None),
                (
                    "threshold current",
                    f"{threshold:.6g} A/m²" if np.isfinite(threshold) else "not finite",
                    NODE_COLOR_COMPUTE,
                ),
                (
                    "torque mapping",
                    metadata.get("slonczewski_mapping", "direct reduced P"),
                    None,
                ),
            ]
        )
        methods = ["simulate"]
        actions = [
            (".simulate(...)", NODE_COLOR_COMPUTE),
            (".model.simulate_sde(...)", NODE_COLOR_ADVANCED),
            (".model.predict_frequency_dc(...)", NODE_COLOR_ANALYSIS),
        ]
        example = (
            "result = adapter.simulate(\n"
            "    t_span=(0.0, 80e-9), J_func=current_ac(...),\n"
            "    dt=10e-12, s0=(1e-3, 0.0),\n"
            ")"
        )
        title = "Dataset-aware CPP Thiele Adapter"
    elif variant == "cip":
        metrics.extend(
            [
                ("beta", f"{float(material.beta):.6g}", None),
                ("current direction", str(tuple(model.current_dir)), None),
            ]
        )
        methods = ["simulate"]
        actions = [
            (".simulate(...)", NODE_COLOR_COMPUTE),
            (".model", NODE_COLOR_ANALYSIS),
        ]
        example = (
            "result = adapter.simulate(\n"
            "    t_span=(0.0, 20e-9), J_func=current_pulse(...),\n"
            "    dt=5e-12, r0=(1e-9, 0.0),\n"
            ")"
        )
        title = "Dataset-aware CIP Thiele Adapter"
    else:
        metrics.extend(
            [
                ("N", f"{float(model.N):.6g}", None),
                ("chirality", int(model.chirality), None),
                (
                    "torque convention",
                    metadata.get("field_cpp_convention", "direct"),
                    None,
                ),
            ]
        )
        methods = ["simulate", "simulate_dc_sweep"]
        actions = [
            (".simulate(...)", NODE_COLOR_COMPUTE),
            (".simulate_dc_sweep(...)", NODE_COLOR_COMPUTE),
            (".model.frequency_geometric(...)", NODE_COLOR_ANALYSIS),
        ]
        example = (
            "result = adapter.simulate(\n"
            "    t_span=(0.0, 80e-9), J_func=current_dc(J_dc),\n"
            "    B_func=field_dc((Bx_T, By_T, Bz_T)), dt=10e-12,\n"
            ")"
        )
        title = "Dataset-aware Field-resolved CPP Adapter"

    api = api_help_html(
        adapter,
        title=f"{title} API help",
        prefix="adapter",
        properties=[("model", "Underlying analytical model with resolved inputs")],
        methods=methods,
        subtitle="The adapter converts analytical output to canonical vortex results.",
        chrome=False,
    )
    return node_card_html(
        title,
        icon="🔌",
        subtitle="Resolved simulation metadata and notebook-ready model execution.",
        badge=("inspect calibration", "#fb923c"),
        sections=[
            metrics_section_html(metrics),
            accessors_section_html([("Run:", actions)]),
            examples_section_html(example, title="Simulation workflow"),
        ],
        extra_tabs=[("Validity", _validity_html())],
        api=api,
        uid=f"vortex-thiele-adapter-{variant}-{uuid.uuid4().hex[:8]}",
    )


__all__ = ["adapter_repr_html", "namespace_repr_html"]
