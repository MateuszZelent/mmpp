"""Single-job autofit orchestration."""

from __future__ import annotations

import math
import sys
import time as _time
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import AutofitConfig
from .features import TrajectoryFeatures, extract_features
from .losses import compute_loss
from .optimizers import run_optimization
from .result import VortexAutofitResult

if TYPE_CHECKING:
    from ..interface import VortexInterface


def run_single_job_fit(
    vortex_interface: "VortexInterface",
    config: AutofitConfig,
) -> VortexAutofitResult:
    """Run a single-job autofit pipeline.

    Orchestrates: parameter resolution -> trajectory extraction ->
    physics-informed init -> optimisation -> comparison building.
    """
    from ..bridge.extract import AnalyticalParameterResolution, extract_model_defaults
    from ..plotting import (
        VortexAnalyticalComparison,
        _resolve_plot_trajectory,
        _simulate_matching_trajectory,
        _trajectory_center,
        _trajectory_dt,
    )

    warnings_list: list[str] = []
    t0_total = _time.monotonic()
    _log = _make_logger(config.verbose, t0_total)

    # 0. Warm up Numba kernels FIRST (JIT compilation runs once)
    from ._numba_kernels import HAS_NUMBA, warmup

    if HAS_NUMBA:
        _log("Compiling Numba JIT kernels (first run only)...")
        warmup()
        _log("Numba ready — using accelerated RK4 integrator.")
    else:
        _log("Numba not available — using scipy fallback (slower).")

    # 1. Resolve numerical trajectory
    _log("Loading numerical trajectory...")
    numerical, traj_source = _resolve_plot_trajectory(
        vortex_interface,
        config.trajectory,
        tracking_source=config.tracking_source,
        tracking_method=config.tracking_method,
    )
    _log(f"Trajectory loaded: {numerical.time.size} samples, "
         f"source={traj_source}")

    if numerical.time.size < 10:
        warnings_list.append(
            f"Very short trajectory ({numerical.time.size} samples). "
            "Fit quality may be poor."
        )

    # 2. Resolve base parameters from job metadata
    _log("Resolving model parameters...")
    resolution = extract_model_defaults(
        vortex_interface=vortex_interface,
        trajectory=numerical,
        params=config.params,
        model=config.model,
        current=config.current,
    )
    base_params = dict(resolution.resolved_params)
    _log(f"Model: {resolution.model_kind.upper()}, "
         f"omega0={float(base_params.get('omega0', 0)):.4g} rad/s")
    if "current_mA" in base_params:
        _log(f"Resolved current: I={float(base_params['current_mA']):.4g} mA, "
             f"J={float(base_params.get('current_density', 0.0)):.4g} A/m²")
    elif "current_density" in base_params:
        _log(f"Resolved current density: J={float(base_params['current_density']):.4g} A/m²")
    if base_params.get("field") is not None:
        try:
            bx, by, bz = np.asarray(base_params["field"], dtype=float).reshape(-1)[:3]
            _log(f"Resolved field: B=({bx:.4g}, {by:.4g}, {bz:.4g}) T")
        except Exception:
            _log(f"Resolved field could not be coerced for logging: {base_params['field']!r}")
    if resolution.model_kind == "cpp" and "P_model" in base_params:
        _log(
            "Resolved CPP STT: "
            f"Praw={float(base_params.get('P_raw', base_params.get('P', 0.0))):.4g}, "
            f"Peff={float(base_params.get('P_eff', base_params.get('P', 0.0))):.4g}, "
            f"Pmodel={float(base_params['P_model']):.4g}, "
            f"pz={float(base_params.get('p_z', 1.0)):.4g}, "
            f"Lambda={float(base_params.get('Lambda', 1.0)):.4g}, "
            f"eps'={float(base_params.get('epsilonprime', 0.0)):.4g}, "
            f"stack={base_params.get('fixed_layer_position', 'top')}"
        )

    disk_radius = float(base_params.get("R", 1e-7))
    features_num = extract_features(numerical, reference_radius=disk_radius)
    _log(f"Numerical features: f={features_num.dominant_freq_hz*1e-9:.4g} GHz, "
         f"r={features_num.mean_radius*1e9:.4g} nm, "
         f"drift={features_num.radius_drift_ratio:.3f}")

    # 3. Build parameter specs and identify fit/frozen sets
    param_specs = config.get_param_specs()
    active_param_names = [
        name for name in config.fit_params
        if name in param_specs and not param_specs[name].frozen
    ]
    frozen_params: dict[str, float] = {}
    for name in config.fit_params:
        if name not in active_param_names:
            frozen_params[name] = float(
                param_specs[name].initial
                if name in param_specs and param_specs[name].initial is not None
                else base_params.get(name, 0.0)
            )
    _log(f"Fitting: {active_param_names}")
    if frozen_params:
        _log(f"Frozen: {frozen_params}")

    # 4. Physics-informed initialization
    initial_params = _physics_informed_init(
        numerical,
        features_num,
        base_params,
        active_param_names,
        param_specs,
    )
    _enrich_param_specs_with_dynamic_priors(
        param_specs,
        active_names=active_param_names,
        base_params=base_params,
        initial_params=initial_params,
    )
    _clip_params_to_specs(
        initial_params,
        active_param_names,
        param_specs,
        logger=_log,
        warnings_list=warnings_list,
        label="Initial guess",
    )

    # 5. Get loss weights
    weights = config.get_weights()

    # 6. Pre-build simulation context (avoid rebuilding per evaluation)
    sim_ctx = _SimulationContext(
        vortex_interface=vortex_interface,
        numerical=numerical,
        resolution=resolution,
        base_params=base_params,
        frozen_params=frozen_params,
        disk_radius=disk_radius,
        tracking_source=config.tracking_source,
        tracking_method=config.tracking_method,
        initial_condition=config.initial_condition,
    )
    _log(f"Simulation context: s0=({sim_ctx._s0_x:.4f}, {sim_ctx._s0_y:.4f}), "
         f"R={sim_ctx._R*1e9:.1f} nm, J={sim_ctx._J_const:.3e} A/m², "
         f"dt={sim_ctx._dt*1e12:.2f} ps, init={sim_ctx._initial_condition_source}, "
         f"n_steps={int((sim_ctx._sim_t1-sim_ctx._t0)/sim_ctx._dt)}")
    if sim_ctx.fast_path_enabled:
        _log(f"Fast path: Numba RK4 enabled ({sim_ctx.fast_path_reason})")
    else:
        _log(f"Fast path disabled: {sim_ctx.fast_path_reason}")
    if sim_ctx._model_kind == "cpp":
        base_cpp_metrics = _cpp_linear_threshold_metrics_from_params(base_params)
        chi_at_J = float(base_cpp_metrics["chi"]) if base_cpp_metrics is not None else 0.0
        threshold_chi = (
            float(base_cpp_metrics["threshold"])
            if base_cpp_metrics is not None
            else sim_ctx._d0 * float(base_params.get("omega0", 0))
        )
        chi_ratio = (
            float(base_cpp_metrics["chi_ratio"])
            if base_cpp_metrics is not None
            else chi_at_J / max(threshold_chi, 1e-30)
        )
        _log(f"  chi(J)={chi_at_J:.4e}, threshold={threshold_chi:.4e}, "
             f"ratio={chi_ratio:.2f}x")
        if chi_ratio < 1.0 and not any(name in active_param_names for name in ("chi_scale", "P_model", "d0_scale")):
            warnings_list.append(
                "Current is below the current Thiele threshold estimate, but no threshold-control "
                "parameter is being fitted. Consider adding chi_scale, P_model, or d0_scale."
            )
            _log("  WARNING: below threshold and no threshold-control parameter is active "
                 "(chi_scale / P_model / d0_scale).")

        # Auto-correct chi_scale only when the current already pumps in the
        # correct direction. Negative chi means the sign convention is wrong
        # and increasing chi_scale would only amplify antidamping.
        if chi_at_J <= 0.0:
            warnings_list.append(
                "Resolved CPP pumping is antidamping (chi(J) <= 0). "
                "Check current sign, fixed-layer position, polarizer, and core polarity."
            )
            _log("  WARNING: chi(J) <= 0, so this parameter set is antidamping; "
                 "chi_scale auto-boost is disabled.")
        elif chi_ratio < 1.0 and "chi_scale" in initial_params:
            needed_chi_scale = (1.0 / chi_ratio) * 1.3  # 30% above threshold
            old_cs = initial_params["chi_scale"]
            if old_cs < needed_chi_scale:
                initial_params["chi_scale"] = needed_chi_scale
                _log(f"  WARNING: Below threshold! Auto-adjusting chi_scale "
                     f"{old_cs:.3f} -> {needed_chi_scale:.3f} to start above threshold")
                warnings_list.append(
                    f"Current is below Thiele threshold (chi/threshold={chi_ratio:.2f}). "
                    f"chi_scale was auto-adjusted to {needed_chi_scale:.3f}."
                )
        _clip_params_to_specs(
            initial_params,
            active_param_names,
            param_specs,
            logger=_log,
            warnings_list=warnings_list,
            label="Threshold-adjusted guess",
        )

    _log(f"Initial guess: {initial_params}")

    # 7. Build objective function with tqdm progress bar and optional live dashboard
    eval_count = [0]
    best_so_far = [float("inf")]
    evaluation_records: list[dict[str, Any]] = []

    pbar = _make_progress_bar(config.max_eval, config.verbose)
    if config.live_plot:
        from ._plotting import AutofitLiveMonitor

        live_monitor = AutofitLiveMonitor(
            numerical_features=features_num,
            numerical_trajectory=numerical,
            enabled=True,
            update_every=config.live_plot_every,
        )
    else:
        live_monitor = None

    def _evaluate(
        param_values: dict[str, float],
        *,
        count_eval: bool,
        update_visuals: bool,
        enforce_budget: bool,
    ) -> tuple[float, dict[str, float]]:
        full_params = dict(base_params)
        full_params.update(frozen_params)
        full_params.update(param_values)

        try:
            ana_trajectory = sim_ctx.simulate(full_params)
        except Exception:
            return 1e10, {}

        features_ana = extract_features(ana_trajectory, reference_radius=disk_radius)

        loss, breakdown = compute_loss(
            features_num,
            features_ana,
            weights=weights,
            param_values=param_values,
            param_specs=param_specs,
        )
        collapse_guard = _collapse_guard_penalty(features_num, features_ana)
        if collapse_guard > 0.0:
            breakdown = dict(breakdown)
            breakdown["L_collapse_guard"] = collapse_guard
            loss += collapse_guard
        frequency_guard = _frequency_guard_penalty(features_num, features_ana)
        if frequency_guard > 0.0:
            breakdown = dict(breakdown)
            breakdown["L_frequency_guard"] = frequency_guard
            loss += frequency_guard
        edge_guard = _edge_collision_guard_penalty(
            features_num,
            features_ana,
            ana_trajectory=ana_trajectory,
            reference_radius=disk_radius,
        )
        if edge_guard > 0.0:
            breakdown = dict(breakdown)
            breakdown["L_edge_guard"] = edge_guard
            loss += edge_guard
        threshold_guard = _cpp_threshold_guard_penalty(
            features_num,
            features_ana,
            full_params,
        )
        if threshold_guard > 0.0:
            breakdown = dict(breakdown)
            breakdown["L_threshold_guard"] = threshold_guard
            loss += threshold_guard

        next_eval = eval_count[0] + 1 if count_eval else eval_count[0]
        best_candidate = min(best_so_far[0], loss)

        if count_eval:
            eval_count[0] = next_eval
        if loss < best_so_far[0]:
            best_so_far[0] = loss

        record = {
            "eval": next_eval,
            "loss": float(loss),
            "best_loss": float(best_candidate),
            "freq_ghz": float(features_ana.dominant_freq_hz * 1e-9),
            "radius_nm": float(features_ana.mean_radius * 1e9),
            "max_radius_nm": float(features_ana.max_radius * 1e9),
            "core_distance_nm": float(features_ana.mean_core_distance * 1e9),
            "max_core_distance_nm": float(features_ana.max_core_distance * 1e9),
            "drift_ratio": float(features_ana.radius_drift_ratio),
            "edge_limited": bool(ana_trajectory.metadata.get("edge_limited", False)),
            "edge_hit_time_ns": (
                float(ana_trajectory.metadata["edge_hit_time"]) * 1e9
                if ana_trajectory.metadata.get("edge_hit_time") is not None
                else np.nan
            ),
            "params": {name: float(param_values.get(name, np.nan)) for name in active_param_names},
        }
        cpp_metrics = _cpp_linear_threshold_metrics_from_params(full_params)
        if cpp_metrics is not None:
            record["chi_ratio"] = float(cpp_metrics["chi_ratio"])
            record["chi"] = float(cpp_metrics["chi"])
            record["threshold"] = float(cpp_metrics["threshold"])
        if count_eval:
            evaluation_records.append(record)

        # Update progress bar
        if update_visuals and pbar is not None:
            pbar.set_postfix(loss=f"{loss:.4g}", best=f"{best_so_far[0]:.4g}",
                             refresh=False)
            pbar.update(1)
        if update_visuals and live_monitor is not None:
            live_monitor.update(
                record,
                analytical_trajectory=ana_trajectory,
                force=loss <= best_so_far[0],
            )

        # Hard cap on evaluations
        if enforce_budget and eval_count[0] >= config.max_eval:
            raise _MaxEvalReached()

        return loss, breakdown

    def _objective(param_values: dict[str, float]) -> tuple[float, dict[str, float]]:
        return _evaluate(
            param_values,
            count_eval=True,
            update_visuals=True,
            enforce_budget=True,
        )

    # 9. Compute baseline loss (with initial params)
    _log("Computing baseline loss...")
    baseline_loss, baseline_breakdown = _objective(initial_params)
    _log(f"Baseline loss: {baseline_loss:.4g}")

    seed_params, seed_loss = _select_threshold_aware_seed(
        features_num=features_num,
        base_params=base_params,
        initial_params=initial_params,
        active_names=active_param_names,
        param_specs=param_specs,
        evaluator=lambda p: _evaluate(
            p,
            count_eval=False,
            update_visuals=False,
            enforce_budget=False,
        ),
    )
    if seed_loss + 1e-12 < baseline_loss:
        initial_params = seed_params
        _log(
            f"Threshold-aware seed search improved start: "
            f"{baseline_loss:.4g} -> {seed_loss:.4g}"
        )
        _log(f"Optimiser start seed: {initial_params}")

    # 10. Run optimisation
    _log(f"Optimising ({len(active_param_names)} params, "
         f"global={'ON' if config.global_search else 'OFF'}, "
         f"max_eval={config.max_eval})...")
    best_fitted_params, diagnostics = run_optimization(
        _objective,
        param_names=active_param_names,
        param_specs=param_specs,
        initial_values=initial_params,
        config=config,
    )

    if pbar is not None:
        pbar.close()
    if live_monitor is not None:
        live_monitor.close()

    diagnostics.evaluation_records = evaluation_records

    _log(f"Done: {diagnostics.n_evaluations} evals in "
         f"{diagnostics.time_total_s:.1f}s")

    # 11. Compute final loss
    final_loss, final_breakdown = _evaluate(
        best_fitted_params,
        count_eval=False,
        update_visuals=False,
        enforce_budget=False,
    )
    _log(f"Final loss: {final_loss:.4g} "
         f"(improvement: {final_loss/max(baseline_loss, 1e-30):.3f})")

    # 12. Build best_params dict (all params, fitted + frozen + base)
    all_best_params: dict[str, float] = {}
    for name in config.fit_params:
        if name in best_fitted_params:
            all_best_params[name] = best_fitted_params[name]
        elif name in frozen_params:
            all_best_params[name] = frozen_params[name]
        elif name in initial_params:
            all_best_params[name] = initial_params[name]

    _log(f"Best params: {all_best_params}")

    # 13. Build comparison with best params
    full_best = dict(base_params)
    full_best.update(frozen_params)
    full_best.update(best_fitted_params)

    best_resolution = AnalyticalParameterResolution(
        resolved_params=full_best,
        param_sources=dict(resolution.param_sources),
        model_kind=resolution.model_kind,
        search_locations=resolution.search_locations,
    )

    analytical_best, raw_center, alignment_center = _simulate_matching_trajectory(
        vortex_interface,
        numerical,
        best_resolution,
        tracking_source=config.tracking_source,
        tracking_method=config.tracking_method,
        initial_condition=config.initial_condition,
    )
    comparison = VortexAnalyticalComparison(
        vortex_interface=vortex_interface,
        numerical=numerical,
        analytical=analytical_best,
        resolution=best_resolution,
        analytical_raw_center=raw_center,
        analytical_center_reference=alignment_center,
        trajectory_source=traj_source,
    )

    # Orbit comparison summary
    m = comparison.metrics
    _log("─── Orbit comparison ───")
    _log(f"  Numerical:  orbit={m.numerical_radius_nm:.1f} nm, "
         f"core={m.numerical_core_distance_nm:.1f} nm, "
         f"f={m.numerical_freq_ghz:.4f} GHz")
    _log(f"  Analytical: orbit={m.analytical_radius_nm:.1f} nm, "
         f"core={m.analytical_core_distance_nm:.1f} nm, "
         f"f={m.analytical_freq_ghz:.4f} GHz")
    _log(f"  Δorbit={m.delta_radius_mean*1e9:.1f} nm, "
         f"Δcore={m.delta_core_distance_mean*1e9:.1f} nm, "
         f"Δf={m.delta_freq_mean*1e-9:.4f} GHz, "
         f"RMS={m.rms_xy_residual*1e9:.1f} nm")

    # 14. Assess success
    cpp_metrics_best = _cpp_linear_threshold_metrics_from_params(full_best)
    success, success_failures = _assess_fit_success(
        baseline_loss=baseline_loss,
        final_loss=final_loss,
        comparison=comparison,
        diagnostics=diagnostics,
        features_num=features_num,
        cpp_metrics=cpp_metrics_best,
    )
    warnings_list.extend(success_failures)

    if diagnostics.active_bounds:
        bound_info = ", ".join(
            f"{k}@{v}" for k, v in diagnostics.active_bounds.items()
        )
        warnings_list.append(f"Parameters at bounds: {bound_info}")

    if diagnostics.poorly_identified:
        warnings_list.append(
            f"Poorly identified parameters: {diagnostics.poorly_identified}"
        )

    if success_failures:
        _log("Success gate rejected fit:")
        for failure in success_failures:
            _log(f"  - {failure}")

    elapsed = _time.monotonic() - t0_total
    _log(f"Autofit complete in {elapsed:.1f}s  "
         f"(success={success}, ratio={final_loss/max(baseline_loss,1e-30):.3f})")

    return VortexAutofitResult(
        best_params=all_best_params,
        initial_params=initial_params,
        param_sources=dict(resolution.param_sources),
        frozen_params=frozen_params,
        fitted_params=tuple(active_param_names),
        loss_total=final_loss,
        loss_breakdown=final_breakdown,
        baseline_loss=baseline_loss,
        comparison=comparison,
        diagnostics=diagnostics,
        success=success,
        warnings=warnings_list,
        config=config,
    )


class _MaxEvalReached(Exception):
    """Raised when max evaluation count is reached."""


class _SimulationContext:
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
        self._disk_radius = disk_radius
        self._field = base_params.get("field")
        self._initial_condition = str(initial_condition)

        # Pre-compute simulation parameters that don't change between evaluations
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
        self._sim_t1 = max(self._t1, nominal_end) + self._dt

        self._current_density = base_params.get("current_density")
        self._current_density_is_callable = callable(self._current_density)

        # Pre-compute constants that are the same across all evaluations
        self._model_kind = resolution.model_kind
        self._fast_path_enabled = False
        self._fast_path_reason = "unsupported"

        # For CPP: pre-compute sigma, d0/d1 base values, chi_prefactor
        # Most of these depend on material params (Ms, alpha, P, R, L)
        # which are fixed during autofit; only omega0, N, chi_scale vary
        if self._model_kind == "cpp":
            self._precompute_cpp(base_params)
        else:
            self._precompute_cip(base_params)

    def _precompute_cpp(self, params: dict) -> None:
        """Pre-compute CPP model constants from base params."""
        _HBAR = 1.054571817e-34
        _E_CHARGE = 1.602176634e-19
        _GAMMA_E = 1.76085963023e11  # rad/(s·T)

        Ms = float(params["Ms"])
        alpha = float(params["alpha"])
        A = float(params.get("A", 1.3e-11))
        R = float(params["R"])
        L = float(params.get("L_stt", params["L"]))

        # chi prefactor per unit model polarization P_model.
        self._chi_prefactor_per_p = _GAMMA_E * _HBAR / (4.0 * _E_CHARGE * L * Ms)

        # Exchange length for core radius
        MU0 = 4e-7 * math.pi
        lex = math.sqrt(2.0 * A / (MU0 * Ms * Ms))
        Rc = max(lex, 1e-10)

        # d0, d1
        ratio = R / max(Rc, 1e-10)
        self._d0 = alpha * (5.0 + 4.0 * math.log(max(ratio, 1.1))) / 8.0
        self._d1 = (11.0 / 6.0) * alpha

        self._polarity = int(np.sign(float(params.get("polarity", 1))) or 1)
        self._p_model_base = float(params.get("P_model", params["P"]))
        self._domega0_dJ_base = float(params.get("domega0_dJ", 0.0))

        # Current density as constant scalar (for fast path)
        J = self._current_density
        if callable(J):
            # Try to evaluate at midpoint; if it's constant, use the scalar
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
            self._fast_path_reason = "time-dependent current requires scipy reference integrator"
        elif _field_has_effect(self._field):
            self._fast_path_enabled = False
            self._fast_path_reason = "external field requires scipy reference integrator"
        else:
            self._fast_path_enabled = True
            self._fast_path_reason = "constant-current zero-field CPP"

    def _precompute_cip(self, params: dict) -> None:
        """Pre-compute CIP model constants from base params."""
        _MU_B = 9.2740100783e-24
        _E_CHARGE = 1.602176634e-19

        Ms = float(params["Ms"])
        alpha = float(params["alpha"])
        P = float(params["P"])
        A = float(params.get("A", 1.3e-11))
        R = float(params["R"])
        L = float(params.get("L", 10e-9))

        self._alpha_cip = alpha
        self._beta_cip = float(params.get("beta_nonadiabatic", params.get("beta", alpha)))
        self._polarity = int(np.sign(float(params.get("polarity", 1))) or 1)

        # u0 prefactor
        self._u0_prefactor = -_MU_B * P / (_E_CHARGE * Ms)

        # D/G0
        MU0 = 4e-7 * math.pi
        lex = math.sqrt(2.0 * A / (MU0 * Ms * Ms))
        core_diam = 2.0 * lex
        ratio = R / max(core_diam, 1e-10)
        self._dG_cip = 0.5 * math.log(max(ratio, 1.1))

        current_dir = tuple(params.get("current_dir", (1.0, 0.0)))
        norm = math.sqrt(current_dir[0] ** 2 + current_dir[1] ** 2)
        self._current_dir = (current_dir[0] / norm, current_dir[1] / norm)

        # Current density as constant scalar
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
            self._fast_path_reason = "time-dependent current requires scipy reference integrator"
        elif _field_has_effect(self._field):
            self._fast_path_enabled = False
            self._fast_path_reason = "external field requires scipy reference integrator"
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
        """Simulate analytical trajectory — Numba fast path."""
        from .._shared.models import TrajectoryResult
        from ..model.adapters import thiele_to_trajectory_result
        from ..plotting import (
            _resample_trajectory_to_reference,
            _trajectory_center,
            _translate_trajectory,
        )

        from ._numba_kernels import HAS_NUMBA

        use_fast = HAS_NUMBA and self._fast_path_enabled
        if self._model_kind == "cpp":
            raw = self._simulate_cpp_fast(params) if use_fast else self._simulate_cpp_scipy(params)
        else:
            raw = self._simulate_cip_fast(params) if use_fast else self._simulate_cip_scipy(params)

        analytical = thiele_to_trajectory_result(
            raw,
            method=f"thiele_{self._model_kind}",
            polarity=self._polarity,
        )

        raw_center = _trajectory_center(analytical)
        alignment_center = (0.0, 0.0) if not _field_has_effect(self._field) else raw_center
        shift = (self._center[0] - alignment_center[0], self._center[1] - alignment_center[1])
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

        J = self._J_const
        chi_val = (
            chi_scale
            * (-float(self._polarity))
            * self._chi_prefactor_per_p
            * p_model
            * J
        )
        omega0_eff = omega0 + domega0_dJ * J

        # No external field in autofit fast path (s_eq = 0)
        t_out, sx_out, sy_out = integrate_cpp_rk4(
            self._t0, self._sim_t1, self._dt,
            self._s0_x, self._s0_y,
            chi_val, omega0_eff, N, self._d0 * d0_scale, self._d1 * d0_scale,
            float(self._polarity), 0.0, 0.0,
            4,  # substeps
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

        J = self._J_const
        u0 = self._u0_prefactor * J
        u0_cx = u0 * self._current_dir[0]
        u0_cy = u0 * self._current_dir[1]

        t_out, X_out, Y_out = integrate_cip_rk4(
            self._t0, self._sim_t1, self._dt,
            self._r0_x, self._r0_y,
            omega0, u0_cx, u0_cy,
            self._alpha_cip, self._beta_cip, self._dG_cip,
            float(self._polarity), 0.0, 0.0,
            4,  # substeps
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
        """Fallback: CPP simulation via scipy solve_ivp (when Numba unavailable)."""
        from mmpp.analytical import CPPThieleModel, DiskGeometry, MaterialParams, current_dc

        mat = MaterialParams(
            Ms=float(params["Ms"]),
            alpha=float(params["alpha"]),
            P=float(params.get("P_model", params["P"])),
            A=float(params.get("A", 1.3e-11)),
        )
        geo = DiskGeometry(R=float(params["R"]), L=float(params["L"]))
        polarity = int(np.sign(float(params.get("polarity", 1))) or 1)

        model = CPPThieleModel(
            material=mat, geom=geo,
            omega0=float(params["omega0"]),
            N=float(params.get("N", 0.25)),
            polarity=polarity,
            domega0_dJ=float(params.get("domega0_dJ", 0.0)),
            field=_coerce_field_for_autofit(self._field),
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
            J_func=self._current_density if callable(self._current_density) else current_dc(float(self._J_const)),
            dt=self._dt,
            clamp_u=float(params.get("clamp_u", 0.999)),
            edge_behavior="freeze",
            rtol=1e-6, atol=1e-9,  # looser tolerances for autofit
        )

    def _simulate_cip_scipy(self, params: dict[str, Any]):
        """Fallback: CIP simulation via scipy solve_ivp (when Numba unavailable)."""
        from mmpp.analytical import CIPThieleModel, DiskGeometry, MaterialParams, current_dc

        mat = MaterialParams(
            Ms=float(params["Ms"]),
            alpha=float(params["alpha"]),
            P=float(params["P"]),
            A=float(params.get("A", 1.3e-11)),
        )
        geo = DiskGeometry(R=float(params["R"]), L=float(params["L"]))
        polarity = int(np.sign(float(params.get("polarity", 1))) or 1)
        current_dir = tuple(params.get("current_dir", (1.0, 0.0)))

        model = CIPThieleModel(
            material=mat, geom=geo,
            omega0=float(params["omega0"]),
            polarity=polarity,
            field=_coerce_field_for_autofit(self._field),
        )
        return model.simulate(
            t_span=(self._t0, self._sim_t1),
            r0=(self._r0_x, self._r0_y),
            J_func=self._current_density if callable(self._current_density) else current_dc(float(self._J_const)),
            dt=self._dt,
            current_dir=current_dir,
            rtol=1e-6, atol=1e-9,  # looser tolerances for autofit
        )


def _physics_informed_init(
    trajectory,
    features_num: TrajectoryFeatures,
    base_params: dict[str, Any],
    active_names: list[str],
    param_specs: dict,
) -> dict[str, float]:
    """Derive initial values for fit parameters from trajectory and metadata."""
    init: dict[str, float] = {}

    for name in active_names:
        spec = param_specs.get(name)

        if name == "omega0":
            omega0_guess = _estimate_cpp_omega0_from_features(features_num, base_params)
            if omega0_guess is not None:
                init[name] = omega0_guess
            elif "omega0" in base_params:
                init[name] = float(base_params["omega0"])
            elif trajectory.time.size >= 4:
                omega = np.asarray(trajectory.instantaneous_frequency, dtype=float)
                init[name] = float(np.abs(np.median(omega)))
            elif spec and spec.initial is not None:
                init[name] = spec.initial
            else:
                init[name] = 1e10

        elif name == "N":
            if "N" in base_params and base_params["N"] != 0.25:
                init[name] = float(base_params["N"])
            elif spec and spec.initial is not None:
                init[name] = spec.initial
            else:
                init[name] = 0.25

        elif name == "chi_scale":
            chi_guess = _estimate_cpp_chi_scale_from_features(features_num, base_params)
            if chi_guess is not None:
                init[name] = chi_guess
            else:
                init[name] = float(base_params.get("chi_scale", 1.0))

        elif name == "phase0":
            if trajectory.time.size >= 1:
                z = trajectory.z
                init[name] = float(np.angle(z[0])) if z.size else 0.0
            else:
                init[name] = 0.0

        elif name in ("center_x", "center_y"):
            x_arr = np.asarray(trajectory.x, dtype=float)
            y_arr = np.asarray(trajectory.y, dtype=float)
            if name == "center_x":
                init[name] = float(np.mean(x_arr)) if x_arr.size else 0.0
            else:
                init[name] = float(np.mean(y_arr)) if y_arr.size else 0.0

        elif name == "d0_scale":
            init[name] = 1.0

        elif name == "domega0_dJ":
            init[name] = float(base_params.get("domega0_dJ", 0.0))

        else:
            if spec and spec.initial is not None:
                init[name] = spec.initial
            elif name in base_params:
                init[name] = float(base_params[name])
            else:
                init[name] = 0.0

    return init


def _clip_params_to_specs(
    params: dict[str, float],
    active_names: list[str],
    param_specs: dict[str, Any],
    *,
    logger=None,
    warnings_list: list[str] | None = None,
    label: str = "Initial guess",
) -> None:
    """Clamp in-place parameter values to declared optimisation bounds."""
    for name in active_names:
        spec = param_specs.get(name)
        if spec is None or name not in params:
            continue
        value = float(params[name])
        clipped = float(np.clip(value, spec.lower, spec.upper))
        if clipped != value:
            params[name] = clipped
            if logger is not None:
                logger(
                    f"{label}: clipping {name} from {value:.4g} "
                    f"to bounds [{spec.lower:.4g}, {spec.upper:.4g}] -> {clipped:.4g}"
                )
            if warnings_list is not None:
                warnings_list.append(
                    f"{label}: {name} clipped from {value:.4g} "
                    f"to [{spec.lower:.4g}, {spec.upper:.4g}]"
                )


def _estimate_cpp_omega0_from_features(
    features_num: TrajectoryFeatures,
    base_params: dict[str, Any],
) -> float | None:
    if "R" not in base_params:
        return None
    f_num = float(features_num.dominant_freq_hz)
    if not np.isfinite(f_num) or f_num <= 0.0:
        return None

    R = max(float(base_params["R"]), 1e-30)
    u = np.clip(float(features_num.mean_radius) / R, 1e-6, 0.98)
    N = float(base_params.get("N", 0.25))
    J = float(base_params.get("current_density", 0.0) or 0.0)
    domega0_dJ = float(base_params.get("domega0_dJ", 0.0))
    omega_eff = 2.0 * np.pi * f_num
    denom = max(1.0 + N * u * u, 1e-12)
    omega0 = (omega_eff / denom) - domega0_dJ * J
    if not np.isfinite(omega0) or omega0 <= 0.0:
        return None
    return float(omega0)


def _estimate_cpp_chi_scale_from_features(
    features_num: TrajectoryFeatures,
    base_params: dict[str, Any],
) -> float | None:
    required = {"Ms", "alpha", "R", "A", "current_density"}
    if not required.issubset(base_params):
        return None

    J = float(base_params.get("current_density", 0.0) or 0.0)
    if not np.isfinite(J) or abs(J) < 1e-30:
        return None

    R = max(float(base_params["R"]), 1e-30)
    u = np.clip(float(features_num.mean_radius) / R, 1e-6, 0.98)
    if u <= 0.0:
        return None

    omega0 = _estimate_cpp_omega0_from_features(features_num, base_params)
    if omega0 is None:
        omega0 = float(base_params.get("omega0", 0.0))
    if not np.isfinite(omega0) or omega0 <= 0.0:
        return None

    _HBAR = 1.054571817e-34
    _E_CHARGE = 1.602176634e-19
    _GAMMA_E = 1.76085963023e11
    MU0 = 4e-7 * math.pi

    Ms = float(base_params["Ms"])
    alpha = float(base_params["alpha"])
    P = float(base_params.get("P_model", base_params.get("P", 0.0)))
    A = float(base_params.get("A", 1.3e-11))
    L = float(base_params.get("L_stt", base_params.get("L", 0.0)))
    N = float(base_params.get("N", 0.25))
    domega0_dJ = float(base_params.get("domega0_dJ", 0.0))
    d0_scale = float(base_params.get("d0_scale", 1.0))

    sigma = _HBAR * P / (2.0 * _E_CHARGE * L * Ms)
    chi_prefactor = _GAMMA_E * sigma / 2.0
    polarity = int(np.sign(float(base_params.get("polarity", 1))) or 1)
    chi_prefactor_signed = -float(polarity) * chi_prefactor

    lex = math.sqrt(2.0 * A / (MU0 * Ms * Ms))
    Rc = max(lex, 1e-10)
    ratio = R / max(Rc, 1e-10)
    d0 = alpha * (5.0 + 4.0 * math.log(max(ratio, 1.1))) / 8.0
    d1 = (11.0 / 6.0) * alpha
    d0 *= d0_scale
    d1 *= d0_scale

    omega0_eff = omega0 + domega0_dJ * J
    required_chi = (d0 + d1 * u * u) * omega0_eff * (1.0 + N * u * u)
    signed_drive = chi_prefactor_signed * J
    if not np.isfinite(signed_drive) or signed_drive <= 0.0:
        return None
    chi_scale = required_chi / max(signed_drive, 1e-30)
    if not np.isfinite(chi_scale) or chi_scale <= 0.0:
        return None
    return float(np.clip(chi_scale, 0.15, 20.0))


def _cpp_linear_threshold_metrics_from_params(
    params: dict[str, Any],
) -> dict[str, float] | None:
    """Compute linear CPP threshold metrics for the current parameter set."""
    required = {"Ms", "alpha", "A", "R", "current_density", "omega0"}
    if not required.issubset(params):
        return None

    J = float(params.get("current_density", 0.0) or 0.0)
    omega0 = float(params.get("omega0", 0.0) or 0.0)
    if not np.isfinite(J) or not np.isfinite(omega0) or omega0 <= 0.0:
        return None

    _HBAR = 1.054571817e-34
    _E_CHARGE = 1.602176634e-19
    _GAMMA_E = 1.76085963023e11
    MU0 = 4e-7 * math.pi

    Ms = float(params["Ms"])
    alpha = float(params["alpha"])
    P_model = float(params.get("P_model", params.get("P", 0.0)))
    A = float(params.get("A", 1.3e-11))
    R = float(params["R"])
    L = float(params.get("L_stt", params.get("L", 0.0)))
    chi_scale = float(params.get("chi_scale", 1.0))
    domega0_dJ = float(params.get("domega0_dJ", 0.0))
    d0_scale = float(params.get("d0_scale", 1.0))
    polarity = int(np.sign(float(params.get("polarity", 1))) or 1)

    if Ms <= 0.0 or R <= 0.0 or L <= 0.0:
        return None

    sigma_per_p = _HBAR / (2.0 * _E_CHARGE * L * Ms)
    chi_prefactor_per_p = _GAMMA_E * sigma_per_p / 2.0
    chi = chi_scale * (-float(polarity)) * chi_prefactor_per_p * P_model * J

    lex = math.sqrt(2.0 * A / (MU0 * Ms * Ms))
    Rc = max(lex, 1e-10)
    ratio = R / max(Rc, 1e-10)
    d0 = alpha * (5.0 + 4.0 * math.log(max(ratio, 1.1))) / 8.0
    omega0_eff = omega0 + domega0_dJ * J
    threshold = d0 * d0_scale * omega0_eff
    chi_ratio = chi / max(threshold, 1e-30)

    return {
        "chi": float(chi),
        "threshold": float(threshold),
        "chi_ratio": float(chi_ratio),
        "omega0_eff": float(omega0_eff),
        "P_model": float(P_model),
    }


def _build_cpp_threshold_seed_candidates(
    features_num: TrajectoryFeatures,
    base_params: dict[str, Any],
    initial_params: dict[str, float],
    active_names: list[str],
    param_specs: dict[str, Any],
) -> list[dict[str, float]]:
    """Build a small pool of threshold-aware warm starts for CPP fits."""
    if not active_names:
        return [dict(initial_params)]

    candidates: list[dict[str, float]] = [dict(initial_params)]
    stable_target = (
        float(features_num.mean_radius) > 0.0
        and float(features_num.radius_drift_ratio) >= 0.85
    )
    if not stable_target:
        return candidates

    full_initial = dict(base_params)
    full_initial.update(initial_params)
    metrics = _cpp_linear_threshold_metrics_from_params(full_initial)
    if metrics is None:
        return candidates

    ratio = float(metrics["chi_ratio"])
    R = max(float(full_initial.get("R", 0.0) or 0.0), 1e-30)
    target_u = np.clip(float(features_num.tail_mean_radius) / R, 1e-6, 0.98)
    margin = max(1.05, 1.0 + 0.35 * target_u)
    threshold_params = [name for name in ("chi_scale", "P_model", "d0_scale") if name in active_names]
    if not threshold_params:
        return candidates

    candidate = dict(initial_params)
    n_scalers = max(len(threshold_params), 1)

    if ratio <= 0.0:
        if "P_model" in active_names:
            desired_sign = np.sign(-float(full_initial.get("polarity", 1)) * float(full_initial.get("current_density", 0.0)))
            if desired_sign == 0.0:
                desired_sign = 1.0
            current = float(candidate.get("P_model", full_initial.get("P_model", full_initial.get("P", 0.0))))
            candidate["P_model"] = abs(current) * float(desired_sign)
            candidates.append(candidate)
        return _unique_seed_candidates(candidates)

    required_scale = max(margin / max(ratio, 1e-12), 1.0)
    per_param_scale = required_scale ** (1.0 / n_scalers)

    if "chi_scale" in active_names:
        current = float(candidate.get("chi_scale", full_initial.get("chi_scale", 1.0)))
        candidate["chi_scale"] = current * per_param_scale
    if "P_model" in active_names:
        current = float(candidate.get("P_model", full_initial.get("P_model", full_initial.get("P", 0.0))))
        candidate["P_model"] = current * per_param_scale
    if "d0_scale" in active_names:
        current = float(candidate.get("d0_scale", full_initial.get("d0_scale", 1.0)))
        candidate["d0_scale"] = current / per_param_scale

    for factor in (0.9, 1.0, 1.1, 1.25):
        variant = dict(candidate)
        if "chi_scale" in active_names and "chi_scale" in candidate:
            variant["chi_scale"] = candidate["chi_scale"] * factor
        if "P_model" in active_names and "P_model" in candidate:
            variant["P_model"] = candidate["P_model"] * factor
        if "d0_scale" in active_names and "d0_scale" in candidate:
            variant["d0_scale"] = candidate["d0_scale"] / factor
        _clip_params_to_specs(variant, active_names, param_specs)
        candidates.append(variant)

    return _unique_seed_candidates(candidates)


def _unique_seed_candidates(
    candidates: list[dict[str, float]],
) -> list[dict[str, float]]:
    unique: list[dict[str, float]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for candidate in candidates:
        key = tuple(sorted((name, round(float(value), 12)) for name, value in candidate.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(dict(candidate))
    return unique


def _select_threshold_aware_seed(
    *,
    features_num: TrajectoryFeatures,
    base_params: dict[str, Any],
    initial_params: dict[str, float],
    active_names: list[str],
    param_specs: dict[str, Any],
    evaluator,
) -> tuple[dict[str, float], float]:
    """Pick the best warm start from a small threshold-aware candidate pool."""
    best_params = dict(initial_params)
    best_loss = float("inf")
    for candidate in _build_cpp_threshold_seed_candidates(
        features_num,
        base_params,
        initial_params,
        active_names,
        param_specs,
    ):
        loss, _ = evaluator(candidate)
        if loss < best_loss:
            best_loss = float(loss)
            best_params = dict(candidate)
    return best_params, best_loss


def _cpp_threshold_guard_penalty(
    features_num: TrajectoryFeatures,
    features_ana: TrajectoryFeatures,
    params: dict[str, Any],
) -> float:
    """Penalise sub-threshold analytical candidates for stable numerical orbits."""
    if float(features_num.mean_radius) <= 0.0 or float(features_num.radius_drift_ratio) < 0.85:
        return 0.0

    metrics = _cpp_linear_threshold_metrics_from_params(params)
    if metrics is None:
        return 0.0

    ratio = float(metrics["chi_ratio"])
    penalty = 0.0
    if ratio <= 0.0:
        penalty += 6.0 + min(abs(ratio), 4.0)
    elif ratio < 1.0:
        penalty += 4.0 * ((1.0 - ratio) / 0.25) ** 2

    tail_target = max(float(features_num.tail_mean_radius), 1e-30)
    tail_ratio = float(features_ana.tail_mean_radius) / tail_target
    if ratio < 1.05 and tail_ratio < 0.8:
        penalty += ((0.8 - tail_ratio) / 0.8) ** 2 * 2.5

    return float(penalty)


def _collapse_guard_penalty(
    features_num: TrajectoryFeatures,
    features_ana: TrajectoryFeatures,
) -> float:
    target_radius = float(features_num.mean_radius)
    if target_radius <= 0.0:
        return 0.0
    if float(features_num.radius_drift_ratio) < 0.85:
        return 0.0

    ana_radius_ratio = float(features_ana.mean_radius) / target_radius
    ana_drift = float(features_ana.radius_drift_ratio)
    penalty = 0.0
    if ana_radius_ratio < 0.7:
        penalty += ((0.7 - ana_radius_ratio) / 0.7) ** 2 * 3.0
    if ana_drift < 0.8:
        penalty += ((0.8 - ana_drift) / 0.8) ** 2 * 2.0
    return float(penalty)


def _frequency_guard_penalty(
    features_num: TrajectoryFeatures,
    features_ana: TrajectoryFeatures,
) -> float:
    """Strongly penalise near-static analytical candidates for stable targets."""
    f_num = float(features_num.dominant_freq_hz)
    if f_num <= 0.0 or float(features_num.mean_radius) <= 0.0:
        return 0.0
    if float(features_num.radius_drift_ratio) < 0.85:
        return 0.0

    f_ana = max(float(features_ana.dominant_freq_hz), 0.0)
    freq_ratio = f_ana / max(f_num, 1e-30)
    radius_ratio = float(features_ana.mean_radius) / max(float(features_num.mean_radius), 1e-30)
    drift_ratio = float(features_ana.radius_drift_ratio)

    penalty = 0.0
    if freq_ratio < 0.5:
        penalty += ((0.5 - freq_ratio) / 0.5) ** 2 * 6.0
    if freq_ratio < 0.25 and radius_ratio > 0.25:
        penalty += ((0.25 - freq_ratio) / 0.25) ** 2 * 10.0
    if freq_ratio < 0.15 and drift_ratio > 0.2:
        penalty += 12.0
    return float(penalty)


def _edge_collision_guard_penalty(
    features_num: TrajectoryFeatures,
    features_ana: TrajectoryFeatures,
    *,
    ana_trajectory,
    reference_radius: float,
) -> float:
    """Penalise analytical edge hits when the numerical orbit stays well inside the disk."""
    R = float(reference_radius)
    if not np.isfinite(R) or R <= 0.0:
        return 0.0

    ana_edge = bool(getattr(ana_trajectory, "metadata", {}).get("edge_limited", False))
    num_max = float(features_num.max_core_distance)
    ana_max = float(features_ana.max_core_distance)

    if num_max >= 0.95 * R:
        return 0.0

    penalty = 0.0
    if ana_edge:
        penalty += 8.0 * ((0.95 * R - num_max) / max(R, 1e-30)) ** 2

    if ana_max > 0.98 * R and num_max < 0.9 * R:
        excess = (ana_max - 0.98 * R) / max(R, 1e-30)
        margin = (0.9 * R - num_max) / max(R, 1e-30)
        penalty += 6.0 * max(excess, 0.0) ** 2 + 3.0 * max(margin, 0.0) ** 2

    return float(penalty)


def _assess_fit_success(
    *,
    baseline_loss: float,
    final_loss: float,
    comparison,
    diagnostics,
    features_num: TrajectoryFeatures,
    cpp_metrics: dict[str, float] | None,
) -> tuple[bool, list[str]]:
    """Decide whether the fit is physically acceptable, not just lower-loss."""
    failures: list[str] = []

    if diagnostics.n_evaluations <= 0:
        failures.append("Fit did not run any evaluations.")
    if final_loss >= baseline_loss:
        failures.append(
            f"Fit did not improve loss (baseline={baseline_loss:.4g}, fitted={final_loss:.4g})."
        )

    m = comparison.metrics
    stable_target = (
        float(features_num.mean_radius) > 0.0
        and float(features_num.radius_drift_ratio) >= 0.85
    )
    if stable_target:
        num_freq = max(float(m.numerical_freq_ghz), 1e-12)
        num_orbit = max(float(m.numerical_radius_nm), 1e-9)
        freq_rel = abs(float(m.delta_freq_mean) * 1e-9) / num_freq
        orbit_rel = abs(float(m.delta_radius_mean) * 1e9) / num_orbit
        ana_freq_ratio = float(m.analytical_freq_ghz) / num_freq

        if freq_rel > 0.25:
            failures.append(
                f"Frequency mismatch too large for a stable target (Δf/f={freq_rel:.2f})."
            )
        if orbit_rel > 0.35:
            failures.append(
                f"Orbit-radius mismatch too large for a stable target (Δr/r={orbit_rel:.2f})."
            )
        if ana_freq_ratio < 0.2 and float(m.analytical_radius_nm) > 0.2 * num_orbit:
            failures.append(
                "Analytical solution has substantial radius but nearly zero gyrotropic frequency."
            )
        ana_edge = bool(getattr(comparison.analytical, "metadata", {}).get("edge_limited", False))
        if ana_edge and float(m.numerical_core_distance_max_nm) < 0.95 * float(comparison.resolved_params.get("R", 0.0)) * 1e9:
            failures.append(
                "Analytical solution collides with the nanodot edge while numerical orbit remains inside the disk."
            )
        if cpp_metrics is not None:
            chi_ratio = float(cpp_metrics.get("chi_ratio", np.nan))
            if np.isfinite(chi_ratio) and chi_ratio < 0.9:
                failures.append(
                    f"Best-fit CPP state remains sub-threshold or near-threshold (J/Jth={chi_ratio:.2f})."
                )

    return len(failures) == 0, failures


def _enrich_param_specs_with_dynamic_priors(
    param_specs: dict[str, Any],
    *,
    active_names: list[str],
    base_params: dict[str, Any],
    initial_params: dict[str, float],
) -> None:
    """Populate missing prior means/stds from resolved physics defaults.

    This keeps the optimizer near physically plausible scales instead of
    exploiting weakly constrained directions such as absurd ``omega0``.
    """
    for name in active_names:
        spec = param_specs.get(name)
        if spec is None:
            continue

        if name == "omega0":
            base_omega0 = float(base_params.get("omega0", initial_params.get("omega0", np.nan)))
            if np.isfinite(base_omega0) and base_omega0 > 0.0:
                # The default omega0 bounds are intentionally broad, but for
                # single-job fitting they make it too easy to escape into
                # absurd high-frequency minima. Only tighten obviously wide
                # bounds; respect user-provided narrower windows.
                if spec.lower < 0.01 * base_omega0:
                    spec.lower = max(spec.lower, 0.2 * base_omega0)
                if spec.upper > 100.0 * base_omega0:
                    spec.upper = min(spec.upper, 5.0 * base_omega0)
            if spec.prior_mean is None:
                candidate = base_omega0
                if np.isfinite(candidate) and candidate > 0.0:
                    spec.prior_mean = candidate
            if spec.prior_std is None:
                spec.prior_std = 0.35
            if spec.initial is None:
                candidate = initial_params.get("omega0")
                if candidate is not None and np.isfinite(candidate) and candidate > 0.0:
                    spec.initial = float(candidate)
        elif name == "chi_scale":
            if spec.prior_mean is None:
                candidate = float(initial_params.get("chi_scale", base_params.get("chi_scale", 1.0)))
                if np.isfinite(candidate) and candidate > 0.0:
                    spec.prior_mean = candidate
            if spec.prior_std is None:
                spec.prior_std = 0.5
        elif name == "P_model":
            base_candidate = float(base_params.get("P_model", base_params.get("P", np.nan)))
            if np.isfinite(base_candidate):
                if spec.lower < -1.5:
                    spec.lower = max(spec.lower, -1.0)
                if spec.upper > 1.5:
                    spec.upper = min(spec.upper, 1.0)
            if spec.prior_mean is None:
                if np.isfinite(base_candidate):
                    spec.prior_mean = base_candidate
            if spec.prior_std is None:
                spec.prior_std = 0.15
        elif name == "d0_scale":
            if spec.lower < 0.5:
                spec.lower = max(spec.lower, 0.6)
            if spec.upper > 2.0:
                spec.upper = min(spec.upper, 1.6)
            if spec.prior_mean is None:
                spec.prior_mean = 1.0
            if spec.prior_std is None:
                spec.prior_std = 0.15


def _make_logger(verbose: bool, t0: float | None = None):
    """Create a print-based logger with elapsed time."""
    _t0 = t0 if t0 is not None else _time.monotonic()

    def _log(msg: str) -> None:
        if verbose:
            elapsed = _time.monotonic() - _t0
            print(f"[autofit {elapsed:6.1f}s] {msg}", flush=True)

    return _log


def _make_progress_bar(total: int, verbose: bool):
    """Create a tqdm progress bar that works in both Jupyter and terminal."""
    if not verbose:
        return None
    try:
        # Try Jupyter/IPython widget first (renders as interactive bar in notebook)
        from tqdm.auto import tqdm
        return tqdm(
            total=total,
            desc="autofit",
            unit="eval",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {postfix}]",
        )
    except ImportError:
        return None


def _coerce_field_for_autofit(value: Any):
    if value is None:
        return None
    from mmpp.analytical.thiele import ExternalField

    return ExternalField.from_any(value)


def _field_has_effect(field: Any) -> bool:
    if field is None:
        return False
    try:
        coerced = _coerce_field_for_autofit(field)
    except Exception:
        return True
    if coerced is None:
        return False
    return any(
        abs(float(component)) > 1e-30
        for component in (coerced.Bx_T, coerced.By_T, coerced.Bz_T)
    )


__all__ = ["run_single_job_fit"]
