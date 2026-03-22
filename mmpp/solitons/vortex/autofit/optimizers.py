"""Optimization pipeline for vortex autofit."""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy import optimize

from .config import AutofitConfig, ParameterSpec
from .result import AutofitDiagnostics


@dataclass
class _OptimizationState:
    """Internal optimisation tracking."""

    loss_history: list[float] = field(default_factory=list)
    n_evals: int = 0
    best_loss: float = float("inf")
    best_x: np.ndarray | None = None


def _clip_to_bounds(
    values: np.ndarray,
    bounds: list[tuple[float, float]],
) -> np.ndarray:
    """Project a parameter vector into the closed box defined by ``bounds``."""
    if values.size == 0:
        return values.astype(float, copy=True)
    lower = np.asarray([lo for lo, _ in bounds], dtype=float)
    upper = np.asarray([hi for _, hi in bounds], dtype=float)
    return np.clip(np.asarray(values, dtype=float), lower, upper)


def run_optimization(
    objective_fn: Callable[[dict[str, float]], tuple[float, dict[str, float]]],
    *,
    param_names: list[str],
    param_specs: dict[str, ParameterSpec],
    initial_values: dict[str, float],
    config: AutofitConfig,
) -> tuple[dict[str, float], AutofitDiagnostics]:
    """Run the full optimisation pipeline: global search + local refinement.

    Parameters
    ----------
    objective_fn : callable
        Takes ``dict[str, float]`` -> ``(loss, breakdown)``.
    param_names : list[str]
        Names of parameters to optimise (non-frozen only).
    param_specs : dict
        Specifications for each parameter.
    initial_values : dict
        Starting point for each parameter.
    config : AutofitConfig
        Full autofit configuration.

    Returns
    -------
    best_params : dict[str, float]
        Optimal parameter values.
    diagnostics : AutofitDiagnostics
        Run diagnostics.
    """
    from .single import _MaxEvalReached

    state = _OptimizationState()

    bounds = []
    x0 = []
    for name in param_names:
        spec = param_specs[name]
        bounds.append((spec.lower, spec.upper))
        x0.append(initial_values.get(name, spec.initial or 0.0))

    x0 = np.array(x0, dtype=float)
    x0 = _clip_to_bounds(x0, bounds)
    state.best_x = x0.copy()

    def _scalar_objective(x_vec: np.ndarray) -> float:
        params = {name: float(x_vec[i]) for i, name in enumerate(param_names)}
        loss, _ = objective_fn(params)
        state.loss_history.append(loss)
        state.n_evals += 1
        if loss < state.best_loss:
            state.best_loss = loss
            state.best_x = x_vec.copy()
        return loss

    t_start = _time.monotonic()
    n_global = 0
    t_global = 0.0

    # Evaluate initial point
    try:
        init_loss = _scalar_objective(x0)
    except _MaxEvalReached:
        init_loss = state.best_loss

    optimizer_message = ""
    optimizer_nit = 0
    rng_seed = config.random_seed

    # Stage 1: Global search (optional, OFF by default)
    if config.global_search and len(param_names) > 0:
        t_g0 = _time.monotonic()
        n_before = state.n_evals

        try:
            if config.global_method == "differential_evolution":
                de_result = optimize.differential_evolution(
                    _scalar_objective,
                    bounds=bounds,
                    x0=x0,
                    maxiter=config.global_maxiter,
                    popsize=config.global_popsize,
                    seed=rng_seed,
                    tol=1e-4,
                    atol=1e-8,
                    polish=False,
                    init="sobol" if rng_seed is not None else "latinhypercube",
                )
                optimizer_message = str(de_result.message)
                optimizer_nit += int(de_result.nit)
            else:
                # Sobol sampling fallback
                from scipy.stats.qmc import Sobol

                n_sobol = min(config.global_maxiter * 5, 128)
                sampler = Sobol(d=len(param_names), seed=rng_seed)
                samples = sampler.random(n_sobol)
                bounds_arr = np.array(bounds)
                candidates = bounds_arr[:, 0] + samples * (
                    bounds_arr[:, 1] - bounds_arr[:, 0]
                )

                for candidate in candidates:
                    _scalar_objective(candidate)

                optimizer_message = f"Sobol search: {n_sobol} samples"
        except _MaxEvalReached:
            optimizer_message = f"Global search stopped: max_eval={config.max_eval}"

        n_global = state.n_evals - n_before
        t_global = _time.monotonic() - t_g0

    # Stage 2: Local refinement
    t_l0 = _time.monotonic()
    n_before_local = state.n_evals

    if len(param_names) > 0:
        try:
            local_result = optimize.minimize(
                _scalar_objective,
                x0=_clip_to_bounds(state.best_x, bounds),
                method=config.local_method,
                bounds=bounds,
                options={
                    "maxiter": config.local_maxiter,
                    "ftol": 1e-8,
                    "gtol": 1e-6,
                },
            )
            optimizer_message = str(local_result.message)
            optimizer_nit += int(local_result.nit)
        except _MaxEvalReached:
            optimizer_message += " | Local stopped: max_eval reached"

    n_local = state.n_evals - n_before_local
    t_local = _time.monotonic() - t_l0
    t_total = _time.monotonic() - t_start

    best_x = state.best_x
    best_params = {name: float(best_x[i]) for i, name in enumerate(param_names)}

    # Check active bounds
    active_bounds: dict[str, str] = {}
    for i, name in enumerate(param_names):
        lo, hi = bounds[i]
        val = float(best_x[i])
        tol = max(abs(hi - lo) * 1e-6, 1e-15)
        if abs(val - lo) < tol:
            active_bounds[name] = "lower"
        elif abs(val - hi) < tol:
            active_bounds[name] = "upper"

    # Lightweight uncertainty estimation (skip if too many evals already)
    hessian_approx = None
    param_uncertainties: dict[str, float] | None = None
    poorly_identified: list[str] = []

    remaining_budget = config.max_eval - state.n_evals
    hessian_cost = 2 * len(param_names)
    if len(param_names) >= 1 and remaining_budget >= hessian_cost:
        try:
            hessian_approx = _approx_hessian_diag(
                _scalar_objective, best_x, bounds
            )
            param_uncertainties = {}
            for i, name in enumerate(param_names):
                if hessian_approx[i] > 0:
                    sigma = 1.0 / np.sqrt(hessian_approx[i])
                    param_uncertainties[name] = float(sigma)
                    if abs(best_x[i]) > 0 and sigma / abs(best_x[i]) > 1.0:
                        poorly_identified.append(name)
                else:
                    param_uncertainties[name] = float("inf")
                    poorly_identified.append(name)
        except Exception:
            pass

    diagnostics = AutofitDiagnostics(
        n_evaluations=state.n_evals,
        n_global_evaluations=n_global,
        n_local_evaluations=n_local,
        time_total_s=t_total,
        time_global_s=t_global,
        time_local_s=t_local,
        optimizer_message=optimizer_message,
        optimizer_nit=optimizer_nit,
        loss_history=state.loss_history,
        hessian_approx=hessian_approx,
        param_uncertainties=param_uncertainties,
        poorly_identified=poorly_identified,
        active_bounds=active_bounds,
    )

    return best_params, diagnostics


def _approx_hessian_diag(
    func: Callable[[np.ndarray], float],
    x: np.ndarray,
    bounds: list[tuple[float, float]],
) -> np.ndarray:
    """Approximate diagonal of the Hessian via central finite differences."""
    f0 = func(x)
    n = len(x)
    diag = np.zeros(n, dtype=float)

    for i in range(n):
        lo, hi = bounds[i]
        h = max(abs(x[i]) * 1e-4, 1e-10)
        h = min(h, (hi - lo) * 0.1)

        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] = min(x[i] + h, hi)
        x_minus[i] = max(x[i] - h, lo)

        actual_h = (x_plus[i] - x_minus[i]) / 2.0
        if actual_h <= 0:
            continue

        f_plus = func(x_plus)
        f_minus = func(x_minus)
        diag[i] = (f_plus - 2 * f0 + f_minus) / (actual_h ** 2)

    return diag


__all__ = ["run_optimization"]
