"""Canonical notebook cards for current-driven Thiele models and results."""

from __future__ import annotations

import math
import uuid
from typing import Any

import numpy as np

from mmpp._repr_helpers import (
    NODE_COLOR_ADVANCED,
    NODE_COLOR_ANALYSIS,
    NODE_COLOR_COMPUTE,
    NODE_COLOR_PLOT,
    accessors_section_html,
    api_help_html,
    examples_section_html,
    metrics_section_html,
    node_card_html,
)


def _frequency_ghz(omega_rad_s: object) -> str:
    try:
        value = float(omega_rad_s)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(value):
        return "unknown"
    return f"{value / (2.0 * math.pi * 1e9):.6g} GHz"


def _validity_tab(*, include_electrical_note: bool = True) -> str:
    electrical = (
        "<p><b>Readout:</b> coordinate FFT power is not a calibrated voltage "
        "PSD. A magnetoresistance and circuit model is required for electrical "
        "power or linewidth in measurement units.</p>"
        if include_electrical_note
        else ""
    )
    return (
        "<div style='padding:4px 0;color:#cbd5e1;line-height:1.55;'>"
        "<p><b>Resolved physics:</b> collective vortex-core translation, "
        "gyrotropic frequency, nonlinear orbit growth, damping, current pumping, "
        "and configured field/current shifts.</p>"
        "<p><b>Not resolved:</b> spatial spin-wave eigenfunctions, core "
        "deformation, nucleation, polarity reversal, annihilation, or the "
        "post-expulsion state.</p>"
        f"{electrical}"
        "<p><b>Qualification:</b> calibrate frequency, nonlinear shift, torque "
        "efficiency, and edge criterion against independent micromagnetic data "
        "before quantitative extrapolation.</p>"
        "</div>"
    )


def trajectory_result_html(result: Any) -> str:
    """Render a tabbed helper for a Thiele trajectory result."""
    n = int(np.asarray(result.t).size)
    duration_ns = float(result.t[-1] - result.t[0]) * 1e9 if n > 1 else 0.0
    radius_nm = float(result.steady_state_radius_m) * 1e9 if n else 0.0
    frequency_ghz = float(result.steady_state_frequency_ghz) if n > 1 else 0.0
    rotation = str(result.rotation_sense) if n > 1 else "N/A"
    edge_limited = bool(result.metadata.get("edge_limited", False))

    api = api_help_html(
        result,
        title="Thiele trajectory API help",
        prefix="result",
        properties=[
            ("t", "Time samples [s]"),
            ("x", "Core x coordinate [m]"),
            ("y", "Core y coordinate [m]"),
            ("r", "Orbit radius [m]"),
            ("u", "Normalized orbit radius |X|/R"),
            ("instantaneous_frequency_ghz", "Instantaneous frequency [GHz]"),
            ("dominant_frequency_ghz", "FFT peak frequency [GHz]"),
            ("linewidth_ghz", "Estimated spectral FWHM [GHz]"),
            ("plt", "Trajectory plotting accessor"),
        ],
        methods=["compute_spectrum"],
        subtitle=(
            "Live signatures and derived quantities for a reduced "
            "vortex-core trajectory."
        ),
        chrome=False,
    )
    return node_card_html(
        "Thiele Trajectory Result",
        icon="🌀",
        subtitle="Reduced current-driven vortex-core trajectory and response spectrum.",
        badge=("edge limit", "#fb923c") if edge_limited else ("ready", "#22c55e"),
        sections=[
            metrics_section_html(
                [
                    ("model", result.model_name or "Thiele", NODE_COLOR_ANALYSIS),
                    ("samples", n, NODE_COLOR_COMPUTE),
                    ("duration", f"{duration_ns:.6g} ns", None),
                    ("steady radius", f"{radius_nm:.6g} nm", NODE_COLOR_ANALYSIS),
                    (
                        "steady frequency",
                        f"{frequency_ghz:.6g} GHz",
                        NODE_COLOR_ANALYSIS,
                    ),
                    ("rotation", rotation, None),
                    ("edge limited", str(edge_limited).lower(), None),
                ]
            ),
            accessors_section_html(
                [
                    (
                        "Analyze:",
                        [
                            (".compute_spectrum(...)", NODE_COLOR_COMPUTE),
                            (".dominant_frequency_ghz", NODE_COLOR_ANALYSIS),
                            (".linewidth_ghz", NODE_COLOR_ANALYSIS),
                        ],
                    ),
                    (
                        "Inspect:",
                        [
                            (".x / .y / .r / .u", NODE_COLOR_ANALYSIS),
                            (
                                ".instantaneous_frequency_ghz",
                                NODE_COLOR_ANALYSIS,
                            ),
                        ],
                    ),
                    (
                        "Plot:",
                        [
                            (".plt.xy()", NODE_COLOR_PLOT),
                            (".plt.spectrum()", NODE_COLOR_PLOT),
                            (".plt.overview()", NODE_COLOR_PLOT),
                        ],
                    ),
                ]
            ),
            examples_section_html(
                "frequency_hz, relative_power = result.compute_spectrum(\n"
                "    transient_fraction=0.6, signal='x', window='hann'\n"
                ")\n"
                "result.plt.overview()",
                title="Spectrum workflow",
            ),
        ],
        extra_tabs=[("Validity", _validity_tab())],
        api=api,
        uid=f"thiele-trajectory-{uuid.uuid4().hex[:8]}",
    )


def field_trajectory_result_html(result: Any) -> str:
    """Render a tabbed helper for a field-resolved trajectory."""
    n = int(np.asarray(result.t).size)
    duration_ns = float(result.t[-1] - result.t[0]) * 1e9 if n > 1 else 0.0
    radius_nm = float(np.mean(result.r[-max(n // 5, 1) :])) * 1e9 if n else 0.0
    finite_frequency = np.asarray(result.frequency_inst_hz, dtype=float)
    finite_frequency = finite_frequency[np.isfinite(finite_frequency)]
    frequency_ghz = (
        float(np.median(finite_frequency[-max(finite_frequency.size // 5, 1) :])) * 1e-9
        if finite_frequency.size
        else 0.0
    )
    api = api_help_html(
        result,
        title="Field-resolved trajectory API help",
        prefix="result",
        properties=[
            ("t", "Time samples [s]"),
            ("X", "Core coordinates with shape (n, 2) [m]"),
            ("s", "Normalized core coordinates"),
            ("r", "Orbit radius [m]"),
            ("frequency_inst_hz", "Instantaneous frequency [Hz]"),
            ("velocity", "Velocity components [m/s]"),
            ("speed", "Core speed [m/s]"),
        ],
        subtitle="Derived quantities for a field-resolved CPP trajectory.",
        chrome=False,
    )
    return node_card_html(
        "Field-resolved Thiele Trajectory",
        icon="🧭",
        subtitle="CPP vortex-core motion with calibrated vector-field terms.",
        badge=("ready", "#22c55e"),
        sections=[
            metrics_section_html(
                [
                    (
                        "model",
                        result.model_name or "field-resolved CPP",
                        NODE_COLOR_ANALYSIS,
                    ),
                    ("samples", n, NODE_COLOR_COMPUTE),
                    ("duration", f"{duration_ns:.6g} ns", None),
                    (
                        "late-time radius",
                        f"{radius_nm:.6g} nm",
                        NODE_COLOR_ANALYSIS,
                    ),
                    (
                        "late-time frequency",
                        f"{frequency_ghz:.6g} GHz",
                        NODE_COLOR_ANALYSIS,
                    ),
                ]
            ),
            accessors_section_html(
                [
                    (
                        "Inspect:",
                        [
                            (".X / .s / .r / .u", NODE_COLOR_ANALYSIS),
                            (".frequency_inst_hz", NODE_COLOR_ANALYSIS),
                            (".velocity / .speed", NODE_COLOR_ANALYSIS),
                        ],
                    )
                ]
            ),
        ],
        extra_tabs=[("Validity", _validity_tab(include_electrical_note=False))],
        api=api,
        uid=f"field-thiele-trajectory-{uuid.uuid4().hex[:8]}",
    )


def fj_fit_result_html(result: Any) -> str:
    """Render a tabbed helper for a multi-current frequency fit."""
    api = api_help_html(
        result,
        title="Thiele f(J) fit API help",
        prefix="fit",
        properties=[
            ("omega0", "Fitted zero-current angular frequency [rad/s]"),
            ("N", "Fitted nonlinear frequency coefficient"),
            ("domega0_dJ", "Fitted/direct current frequency slope"),
            ("chi_scale", "Effective spin-torque pumping scale"),
            ("f_data_hz", "Input frequencies [Hz]"),
            ("f_fit_hz", "Model prediction at input currents [Hz]"),
            ("valid_mask", "Points represented by a valid steady orbit"),
            ("rmse_hz", "Root-mean-square frequency residual [Hz]"),
            ("plt", "Fit plotting accessor"),
        ],
        subtitle="Fit diagnostics and arrays for the calibrated CPP f(J) curve.",
        chrome=False,
    )
    return node_card_html(
        "Thiele f(J) Fit Result",
        icon="📈",
        subtitle="Multi-current calibration of gyrotropic frequency and nonlinearity.",
        badge=("fit converged", "#22c55e")
        if bool(result.success)
        else ("inspect fit", "#fb923c"),
        sections=[
            metrics_section_html(
                [
                    ("success", str(bool(result.success)).lower(), NODE_COLOR_COMPUTE),
                    ("status", result.status, None),
                    ("data points", int(np.asarray(result.J_data).size), None),
                    ("valid points", int(np.count_nonzero(result.valid_mask)), None),
                    ("f0", _frequency_ghz(result.omega0), NODE_COLOR_ANALYSIS),
                    ("N", f"{float(result.N):.6g}", NODE_COLOR_ANALYSIS),
                    (
                        "domega0/dJ",
                        f"{float(result.domega0_dJ):.6g} rad s⁻¹/(A m⁻²)",
                        None,
                    ),
                    ("RMSE", f"{float(result.rmse_hz):.6g} Hz", None),
                ]
            ),
            accessors_section_html(
                [
                    (
                        "Inspect:",
                        [
                            (".f_data_hz / .f_fit_hz", NODE_COLOR_ANALYSIS),
                            (".valid_mask", NODE_COLOR_ANALYSIS),
                            (".params / .metadata", NODE_COLOR_ANALYSIS),
                        ],
                    ),
                    (
                        "Plot:",
                        [
                            (".plt.frequency_vs_current()", NODE_COLOR_PLOT),
                        ],
                    ),
                ]
            ),
            examples_section_html(
                "fit.plt.frequency_vs_current()\n"
                "print(fit.omega0, fit.N, fit.rmse_hz, fit.success)",
                title="Inspect the calibration",
            ),
        ],
        extra_tabs=[
            (
                "Validity",
                "<div style='padding:4px 0;color:#cbd5e1;line-height:1.55;'>"
                "<p>A physical fit requires several distinct current points with "
                "resolved steady oscillations. One trajectory cannot separately "
                "identify damping, torque efficiency, stiffness, and N.</p>"
                "<p>Inspect valid_mask, residuals, current range, and edge-limited "
                "points before reusing the parameters.</p></div>",
            )
        ],
        api=api,
        uid=f"thiele-fj-fit-{uuid.uuid4().hex[:8]}",
    )


def optimization_result_html(result: Any) -> str:
    """Render a tabbed helper for target-frequency current optimization."""
    api = api_help_html(
        result,
        title="Thiele current optimization API help",
        prefix="optimum",
        properties=[
            ("target_frequency_hz", "Requested gyrotropic frequency [Hz]"),
            ("current_density_a_per_m2", "Selected current density [A/m²]"),
            ("predicted_frequency_hz", "Model prediction [Hz]"),
            ("objective_value_hz", "Absolute frequency mismatch [Hz]"),
            ("J_bounds", "Current-search interval [A/m²]"),
            ("success", "Optimizer convergence flag"),
            ("status", "Optimizer status"),
        ],
        subtitle="Selected current and residual for a calibrated CPP model.",
        chrome=False,
    )
    return node_card_html(
        "Thiele Current Optimization Result",
        icon="🎯",
        subtitle="Model-predicted DC current for a requested gyrotropic frequency.",
        badge=("converged", "#22c55e")
        if bool(result.success)
        else ("inspect result", "#fb923c"),
        sections=[
            metrics_section_html(
                [
                    ("success", str(bool(result.success)).lower(), NODE_COLOR_COMPUTE),
                    ("status", result.status, None),
                    (
                        "target",
                        f"{float(result.target_frequency_hz) * 1e-9:.6g} GHz",
                        NODE_COLOR_ANALYSIS,
                    ),
                    (
                        "predicted",
                        f"{float(result.predicted_frequency_hz) * 1e-9:.6g} GHz",
                        NODE_COLOR_ANALYSIS,
                    ),
                    (
                        "current",
                        f"{float(result.current_density_a_per_m2):.6g} A/m²",
                        NODE_COLOR_COMPUTE,
                    ),
                    (
                        "mismatch",
                        f"{float(result.objective_value_hz):.6g} Hz",
                        None,
                    ),
                    ("bounds", str(tuple(result.J_bounds)), None),
                ]
            ),
            accessors_section_html(
                [
                    (
                        "Inspect:",
                        [
                            (".current_density_ga_per_m2", NODE_COLOR_ANALYSIS),
                            (".predicted_frequency_ghz", NODE_COLOR_ANALYSIS),
                            (".params / .metadata", NODE_COLOR_ANALYSIS),
                        ],
                    )
                ]
            ),
        ],
        extra_tabs=[
            (
                "Validity",
                "<div style='padding:4px 0;color:#cbd5e1;line-height:1.55;'>"
                "<p>This is an optimization inside the calibrated reduced model, "
                "not proof that the physical device reaches the target. Reject "
                "edge-limited solutions and verify the selected point with "
                "micromagnetics or experiment.</p></div>",
            )
        ],
        api=api,
        uid=f"thiele-current-optimum-{uuid.uuid4().hex[:8]}",
    )


def model_repr_html(model: Any, *, variant: str) -> str:
    """Render a canonical helper for a direct analytical model instance."""
    if variant not in {"cpp", "cip", "field_resolved_cpp"}:
        raise ValueError(f"unsupported Thiele helper variant: {variant!r}")

    material = model.material
    geom = model.geom
    common_metrics: list[tuple[str, object, str | None]] = [
        ("variant", variant, NODE_COLOR_ANALYSIS),
        ("polarity", int(model.polarity), None),
        ("radius", f"{float(geom.R) * 1e9:.6g} nm", None),
        ("thickness", f"{float(geom.L) * 1e9:.6g} nm", None),
        ("Ms", f"{float(material.Ms):.6g} A/m", None),
        ("alpha", f"{float(material.alpha):.6g}", None),
        ("f0", _frequency_ghz(model.omega0), NODE_COLOR_ANALYSIS),
    ]

    if variant == "cpp":
        threshold = float(model.threshold_current_dc())
        threshold_text = (
            f"{threshold:.6g} A/m²" if np.isfinite(threshold) else "not finite"
        )
        common_metrics.extend(
            [
                ("N", f"{float(model.N):.6g}", None),
                ("threshold current", threshold_text, NODE_COLOR_COMPUTE),
                (
                    "torque thickness",
                    f"{float(model.torque_thickness) * 1e9:.6g} nm",
                    None,
                ),
            ]
        )
        methods = [
            "simulate",
            "simulate_sde",
            "threshold_current_dc",
            "predict_frequency_dc",
            "optimize_current_for_target_frequency",
            "steady_state_u",
        ]
        actions = [
            (".simulate(...)", NODE_COLOR_COMPUTE),
            (".simulate_sde(...)", NODE_COLOR_ADVANCED),
            (".predict_frequency_dc(...)", NODE_COLOR_ANALYSIS),
            (".optimize_current_for_target_frequency(...)", NODE_COLOR_ANALYSIS),
        ]
        example = (
            "J_dc = 1.3 * model.threshold_current_dc()\n"
            "result = model.simulate(\n"
            "    t_span=(0.0, 80e-9), s0=(1e-3, 0.0),\n"
            "    J_func=current_dc(J_dc), dt=10e-12,\n"
            ")"
        )
        title = "CPP Thiele Model"
        icon = "⚡"
    elif variant == "cip":
        common_metrics.extend(
            [
                ("beta", f"{float(material.beta):.6g}", None),
                ("current direction", str(tuple(model.current_dir)), None),
            ]
        )
        methods = ["simulate"]
        actions = [(".simulate(...)", NODE_COLOR_COMPUTE)]
        example = (
            "result = model.simulate(\n"
            "    t_span=(0.0, 20e-9), r0=(1e-9, 0.0),\n"
            "    J_func=current_pulse(4e10, t_on=1e-9, t_off=5e-9),\n"
            "    dt=5e-12,\n"
            ")"
        )
        title = "CIP Thiele Model"
        icon = "➡"
    else:
        common_metrics.extend(
            [
                ("N", f"{float(model.N):.6g}", None),
                ("chirality", int(model.chirality), None),
                ("polarizer", str(tuple(float(v) for v in model.polarizer)), None),
            ]
        )
        methods = [
            "simulate",
            "simulate_dc_sweep",
            "frequency_geometric",
            "frequency_fft",
            "small_signal_omega_exact",
        ]
        actions = [
            (".simulate(...)", NODE_COLOR_COMPUTE),
            (".simulate_dc_sweep(...)", NODE_COLOR_COMPUTE),
            (".frequency_geometric(...)", NODE_COLOR_ANALYSIS),
            (".frequency_fft(...)", NODE_COLOR_ANALYSIS),
        ]
        example = (
            "result = model.simulate(\n"
            "    t_span=(0.0, 80e-9), J_func=current_dc(J_dc),\n"
            "    B_func=field_dc((0.0, 0.0, Bz_T)), dt=10e-12,\n"
            ")"
        )
        title = "Field-resolved CPP Thiele Model"
        icon = "🧭"

    api = api_help_html(
        model,
        title=f"{title} API help",
        prefix="model",
        properties=[],
        methods=methods,
        subtitle="Live signatures for the direct analytical model.",
        chrome=False,
    )
    return node_card_html(
        title,
        icon=icon,
        subtitle="Reduced collective-coordinate model for current-driven vortex dynamics.",
        badge=("calibration required", "#fb923c"),
        sections=[
            metrics_section_html(common_metrics),
            accessors_section_html([("Run:", actions)]),
            examples_section_html(example, title="Minimal workflow"),
        ],
        extra_tabs=[("Validity", _validity_tab())],
        api=api,
        uid=f"thiele-model-{variant}-{uuid.uuid4().hex[:8]}",
    )


__all__ = [
    "field_trajectory_result_html",
    "fj_fit_result_html",
    "model_repr_html",
    "optimization_result_html",
    "trajectory_result_html",
]
