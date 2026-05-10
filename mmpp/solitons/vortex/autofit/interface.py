"""Autofit interface accessible as ``job.vortex.autofit``."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from mmpp._repr_helpers import api_help_html, html_tabs

from .config import AutofitConfig, ParameterSpec
from .result import VortexAutofitResult

if TYPE_CHECKING:
    from ..interface import VortexInterface


class _ThieleMethod:
    """Callable wrapper for ``AutofitInterface.thiele`` with rich Jupyter display."""

    def __init__(self, interface: AutofitInterface):
        self._interface = interface

    def __call__(
        self,
        *,
        trajectory: str = "steady_state",
        tracking_source: str = "auto",
        tracking_method: str | None = None,
        initial_condition: str = "auto",
        model: str = "auto",
        params: str | dict = "auto",
        current: str | float | None = None,
        fit_params: tuple[str, ...] = ("omega0", "N", "chi_scale"),
        param_specs: dict[str, ParameterSpec] | None = None,
        objective: str = "hybrid",
        weights: dict[str, float] | None = None,
        global_search: bool = False,
        global_method: str = "differential_evolution",
        global_maxiter: int = 15,
        global_popsize: int = 8,
        local_method: str = "L-BFGS-B",
        local_maxiter: int = 100,
        max_eval: int = 500,
        align_phase: bool = True,
        align_center: bool = True,
        windowing: str = "steady_state",
        random_seed: int | None = None,
        allow_oersted: bool = False,
        verbose: bool = True,
        live_plot: bool = False,
        live_plot_every: int = 5,
        **config_overrides: Any,
    ) -> VortexAutofitResult:
        return self._interface._run_thiele(
            trajectory=trajectory,
            tracking_source=tracking_source,
            tracking_method=tracking_method,
            initial_condition=initial_condition,
            model=model,
            params=params,
            current=current,
            fit_params=fit_params,
            param_specs=param_specs,
            objective=objective,
            weights=weights,
            global_search=global_search,
            global_method=global_method,
            global_maxiter=global_maxiter,
            global_popsize=global_popsize,
            local_method=local_method,
            local_maxiter=local_maxiter,
            max_eval=max_eval,
            align_phase=align_phase,
            align_center=align_center,
            windowing=windowing,
            random_seed=random_seed,
            allow_oersted=allow_oersted,
            verbose=verbose,
            live_plot=live_plot,
            live_plot_every=live_plot_every,
            **config_overrides,
        )

    def __repr__(self) -> str:
        return "ThieleMethod(call with .thiele() or .thiele(objective='spectral', ...))"

    def _repr_html_(self) -> str:
        return self._interface._thiele_repr_html()


class AutofitInterface:
    """Physics-informed autofit of analytical vortex models to numerical trajectories."""

    def __init__(self, vortex_interface: VortexInterface):
        self._interface = vortex_interface
        self._thiele_method: _ThieleMethod | None = None

    @property
    def thiele(self) -> _ThieleMethod:
        """Physics-informed autofit of a Thiele model — callable, with Jupyter helper."""
        if self._thiele_method is None:
            self._thiele_method = _ThieleMethod(self)
        return self._thiele_method

    def _run_thiele(
        self,
        *,
        trajectory: str = "steady_state",
        tracking_source: str = "auto",
        tracking_method: str | None = None,
        initial_condition: str = "auto",
        model: str = "auto",
        params: str | dict = "auto",
        current: str | float | None = None,
        fit_params: tuple[str, ...] = ("omega0", "N", "chi_scale"),
        param_specs: dict[str, ParameterSpec] | None = None,
        objective: str = "hybrid",
        weights: dict[str, float] | None = None,
        global_search: bool = False,
        global_method: str = "differential_evolution",
        global_maxiter: int = 15,
        global_popsize: int = 8,
        local_method: str = "L-BFGS-B",
        local_maxiter: int = 100,
        max_eval: int = 500,
        align_phase: bool = True,
        align_center: bool = True,
        windowing: str = "steady_state",
        random_seed: int | None = None,
        allow_oersted: bool = False,
        verbose: bool = True,
        live_plot: bool = False,
        live_plot_every: int = 5,
        **config_overrides: Any,
    ) -> VortexAutofitResult:
        """Run physics-informed autofit of a Thiele model to the numerical trajectory.

        Parameters
        ----------
        trajectory : str
            Which trajectory slice to use: ``"steady_state"``, ``"full"``, ``"filtered"``.
        tracking_source : str
            Tracking source: ``"auto"``, ``"table"``, ``"magnetization"``.
        initial_condition : str
            Analytical initial state: ``"auto"``, ``"script"``, ``"raw"``,
            ``"trajectory"``, ``"perturbation"``.
        model : str
            Model type: ``"auto"``, ``"cpp"``, ``"cip"``.
        params : str or dict
            ``"auto"`` to resolve from job metadata, or explicit dict.
        current : str, float, or None
            Current specification (attribute key or value in A/m²).
        fit_params : tuple of str
            Parameter names to fit. Default: ``("omega0", "N", "chi_scale")``.
        objective : str
            Loss mode: ``"time"``, ``"spectral"``, ``"hybrid"``.
        weights : dict, optional
            Override default loss weights.
        global_search : bool
            Whether to run global optimisation stage (default: off).
        max_eval : int
            Hard cap on total objective evaluations (default: 500).
        verbose : bool
            Print progress every ~2 seconds (default: True).
        random_seed : int, optional
            For reproducibility.

        Returns
        -------
        VortexAutofitResult
            Full fit result with diagnostics, comparison, and loss breakdown.
        """
        from .single import run_single_job_fit

        config = AutofitConfig(
            trajectory=trajectory,
            tracking_source=tracking_source,
            tracking_method=tracking_method,
            initial_condition=initial_condition,
            model=model,
            params=params,
            current=current,
            fit_params=fit_params,
            param_specs=param_specs,
            objective=objective,
            weights=weights,
            global_search=global_search,
            global_method=global_method,
            global_maxiter=global_maxiter,
            global_popsize=global_popsize,
            local_method=local_method,
            local_maxiter=local_maxiter,
            max_eval=max_eval,
            align_phase=align_phase,
            align_center=align_center,
            windowing=windowing,
            random_seed=random_seed,
            allow_oersted=allow_oersted,
            verbose=verbose,
            live_plot=live_plot,
            live_plot_every=live_plot_every,
        )
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(config, key, value)
        return run_single_job_fit(self._interface, config)

    def __repr__(self) -> str:
        return "AutofitInterface(methods=[.thiele()])"

    def _repr_html_(self) -> str:
        return self._thiele_repr_html()

    def _thiele_repr_html(self) -> str:
        from html import escape as _esc

        from .config import DEFAULT_PARAM_SPECS, LOSS_PRESETS

        # ── Styles ──
        card = (
            "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:18px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);"
        )
        section = (
            "background:rgba(15,23,42,0.6);padding:12px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);"
        )
        th_style = "padding:5px 8px;text-align:left;color:#e2e8f0;font-weight:600;"
        td_mono = (
            "padding:4px 8px;font-family:monospace;color:#93c5fd;font-size:0.85em;"
        )
        td_val = "padding:4px 8px;color:#a5b4fc;font-size:0.85em;text-align:center;"
        td_desc = "padding:4px 8px;color:#cbd5e1;font-size:0.85em;"
        code_style = (
            "background:rgba(15,23,42,0.85);padding:10px;border-radius:6px;"
            "color:#e2e8f0;overflow-x:auto;font-size:0.85em;font-family:monospace;"
            "display:block;white-space:pre;margin:0;"
        )
        badge = (
            "background:rgba(96,165,250,0.2);color:#93c5fd;font-size:0.7em;"
            "padding:2px 6px;border-radius:4px;font-weight:600;margin-left:8px;"
        )

        # ── Title ──
        html = f"<div style='{card}'>"
        html += (
            "<div style='display:flex;align-items:center;gap:10px;margin-bottom:4px;'>"
            "<div style='font-size:1.15em;font-weight:600;color:#f1f5f9;'>"
            "🔧 Vortex Autofit Interface</div>"
            f"<span style='{badge}'>Physics-informed fitting</span>"
            "</div>"
            "<div style='font-size:0.85em;color:#94a3b8;margin-bottom:12px;'>"
            "Fit analytical Thiele models to numerical vortex trajectories</div>"
        )

        # ── 1. Method: .thiele() ──
        thiele_args = [
            (
                "trajectory",
                '"steady_state"',
                "str",
                "Trajectory slice: 'steady_state', 'full', 'filtered'",
            ),
            (
                "tracking_source",
                '"auto"',
                "str",
                "Data source: 'auto', 'table', 'magnetization'",
            ),
            (
                "tracking_method",
                "None",
                "str|None",
                "Override tracking method (e.g. 'gaussian', 'table')",
            ),
            (
                "initial_condition",
                '"auto"',
                "str",
                "Analytical start: 'auto', 'script', 'raw', 'trajectory', 'perturbation'",
            ),
            (
                "model",
                '"auto"',
                "str",
                "Model type: 'auto', 'cpp' (⊥ polarised), 'cip' (in-plane)",
            ),
            (
                "params",
                '"auto"',
                "str|dict",
                "'auto' resolves from job metadata, or explicit dict",
            ),
            (
                "current",
                "None",
                "str|float|None",
                "Current density: attribute key or value [A/m²]",
            ),
            (
                "fit_params",
                "('omega0','N','chi_scale')",
                "tuple[str,...]",
                "Parameter names to optimise",
            ),
            (
                "param_specs",
                "None",
                "dict|None",
                "Custom ParameterSpec overrides per parameter",
            ),
            ("objective", '"hybrid"', "str", "Loss mode: 'time', 'spectral', 'hybrid'"),
            ("weights", "None", "dict|None", "Override default loss component weights"),
            (
                "global_search",
                "False",
                "bool",
                "Run global optimisation (differential evolution)",
            ),
            (
                "global_method",
                '"differential_evolution"',
                "str",
                "Global optimiser method",
            ),
            ("global_maxiter", "15", "int", "Max iterations for global search"),
            (
                "global_popsize",
                "8",
                "int",
                "Population size for differential evolution",
            ),
            ("local_method", '"L-BFGS-B"', "str", "Local optimiser method"),
            ("local_maxiter", "100", "int", "Max iterations for local refinement"),
            ("max_eval", "500", "int", "Hard cap on total objective evaluations"),
            ("align_phase", "True", "bool", "Align orbit phase before comparison"),
            ("align_center", "True", "bool", "Align orbit center before comparison"),
            ("windowing", '"steady_state"', "str", "Time windowing strategy"),
            ("random_seed", "None", "int|None", "Seed for reproducibility"),
            (
                "allow_oersted",
                "False",
                "bool",
                "Include Oersted field contribution in model",
            ),
            ("verbose", "True", "bool", "Print progress messages during fit"),
            ("live_plot", "False", "bool", "Show live dashboard (Jupyter only)"),
            (
                "live_plot_every",
                "5",
                "int",
                "Update live dashboard every N evaluations",
            ),
        ]

        html += f"<div style='{section}'>"
        html += (
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:8px;font-size:1.0em;'>"
            "📋 <code style='color:#60a5fa;font-size:1.0em;'>.thiele(**kwargs)</code>"
            " → <code style='color:#22c55e;font-size:0.85em;'>VortexAutofitResult</code></div>"
        )
        html += (
            "<table style='width:100%;border-collapse:collapse;'>"
            f"<thead><tr style='background:rgba(51,65,85,0.6);'>"
            f"<th style='{th_style}'>Argument</th>"
            f"<th style='{th_style}text-align:center;'>Default</th>"
            f"<th style='{th_style}'>Type</th>"
            f"<th style='{th_style}'>Description</th></tr></thead><tbody>"
        )
        for name, default, typ, desc in thiele_args:
            html += (
                f"<tr style='border-bottom:1px solid rgba(71,85,105,0.3);'>"
                f"<td style='{td_mono}'>{_esc(name)}</td>"
                f"<td style='{td_val}'>{_esc(default)}</td>"
                f"<td style='{td_desc}font-style:italic;color:#94a3b8;'>{_esc(typ)}</td>"
                f"<td style='{td_desc}'>{_esc(desc)}</td></tr>"
            )
        html += "</tbody></table></div>"

        # ── 2. Fittable Parameters ──
        html += f"<div style='{section}'>"
        html += (
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:8px;'>"
            "🎯 Default Fittable Parameters "
            "<span style='color:#94a3b8;font-weight:400;font-size:0.85em;'>"
            "(use in fit_params=...)</span></div>"
        )
        html += (
            "<table style='width:100%;border-collapse:collapse;'>"
            f"<thead><tr style='background:rgba(51,65,85,0.6);'>"
            f"<th style='{th_style}'>Name</th>"
            f"<th style='{th_style}text-align:center;'>Bounds</th>"
            f"<th style='{th_style}text-align:center;'>Initial</th>"
            f"<th style='{th_style}text-align:center;'>Prior</th>"
            f"<th style='{th_style}text-align:center;'>Frozen</th>"
            f"<th style='{th_style}'>Scale</th></tr></thead><tbody>"
        )
        for name, spec in DEFAULT_PARAM_SPECS.items():
            # Format bounds
            lo = f"{spec.lower:.2g}" if abs(spec.lower) < 1e15 else "-∞"
            hi = f"{spec.upper:.2g}" if abs(spec.upper) < 1e15 else "+∞"
            bounds_str = f"[{lo}, {hi}]"
            init_str = f"{spec.initial:.4g}" if spec.initial is not None else "—"
            if spec.prior_mean is not None and spec.prior_std is not None:
                prior_str = (
                    f"{spec.prior_type} μ={spec.prior_mean:.3g} σ={spec.prior_std:.2g}"
                )
            else:
                prior_str = spec.prior_type
            frozen_icon = "❄️" if spec.frozen else "🔓"
            scale_str = f"{spec.scale:.2g}"
            html += (
                f"<tr style='border-bottom:1px solid rgba(71,85,105,0.3);'>"
                f"<td style='{td_mono}font-weight:600;'>{_esc(name)}</td>"
                f"<td style='{td_val}'>{_esc(bounds_str)}</td>"
                f"<td style='{td_val}'>{_esc(init_str)}</td>"
                f"<td style='{td_desc}font-size:0.8em;'>{_esc(prior_str)}</td>"
                f"<td style='{td_val}'>{frozen_icon}</td>"
                f"<td style='{td_val}'>{_esc(scale_str)}</td></tr>"
            )
        html += "</tbody></table>"
        html += (
            "<div style='margin-top:6px;font-size:0.8em;color:#94a3b8;'>"
            "❄️ = frozen by default (include in <code style='color:#93c5fd;'>fit_params</code> to unfreeze), "
            "🔓 = fitted by default</div>"
        )
        html += "</div>"

        # ── 3. Loss Presets ──
        html += f"<div style='{section}'>"
        html += (
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:8px;'>"
            "⚖️ Loss Objective Presets "
            "<span style='color:#94a3b8;font-weight:400;font-size:0.85em;'>"
            "(set via objective=...)</span></div>"
        )
        # Loss component names
        loss_keys = list(next(iter(LOSS_PRESETS.values())).keys())
        loss_labels = {
            "w_xy": "L_xy (x,y MSE)",
            "w_r": "L_r (radius)",
            "w_phi": "L_φ (phase)",
            "w_freq": "L_freq (frequency)",
            "w_psd": "L_psd (spectral)",
            "w_ellip": "L_ellip (eccentricity)",
            "w_stability": "L_stab (stability)",
            "w_reg": "L_reg (prior reg.)",
        }
        html += (
            "<table style='width:100%;border-collapse:collapse;'>"
            f"<thead><tr style='background:rgba(51,65,85,0.6);'>"
            f"<th style='{th_style}'>Component</th>"
        )
        for preset_name in LOSS_PRESETS:
            html += f"<th style='{th_style}text-align:center;'>{_esc(preset_name)}</th>"
        html += "</tr></thead><tbody>"
        for key in loss_keys:
            label = loss_labels.get(key, key)
            html += "<tr style='border-bottom:1px solid rgba(71,85,105,0.3);'>"
            html += f"<td style='{td_desc}'>{_esc(label)}</td>"
            for preset_name, weights in LOSS_PRESETS.items():
                w = weights.get(key, 0.0)
                color = "#22c55e" if w >= 0.5 else ("#a5b4fc" if w > 0 else "#475569")
                html += (
                    f"<td style='padding:4px 8px;text-align:center;"
                    f"color:{color};font-weight:{'600' if w >= 0.5 else '400'};'>"
                    f"{w:.2f}</td>"
                )
            html += "</tr>"
        html += "</tbody></table></div>"

        # ── 4. Result API ──
        html += f"<div style='{section}'>"
        html += (
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:8px;'>"
            "📊 Result API "
            "<span style='color:#94a3b8;font-weight:400;font-size:0.85em;'>"
            "(VortexAutofitResult)</span></div>"
        )
        result_items = [
            (".best_params", "dict[str, float]", "Optimised parameter values"),
            (".loss_total", "float", "Total weighted loss"),
            (".loss_breakdown", "dict[str, float]", "Per-component unweighted losses"),
            (".improvement_ratio", "float", "loss / baseline (< 1 = improvement)"),
            (".success", "bool", "Whether optimisation converged"),
            (
                ".comparison",
                "VortexAnalyticalComparison",
                "Num vs analytical trajectory pair",
            ),
            (
                ".diagnostics",
                "AutofitDiagnostics",
                "Evaluation records, timing, uncertainties",
            ),
            (
                ".warnings",
                "list[str]",
                "Fit warnings (boundary hits, poor identifiability)",
            ),
            (".plt.convergence()", "Axes", "Loss vs evaluation (log scale)"),
            (".plt.parameter_comparison()", "Axes", "Initial vs fitted bar chart"),
            (".plt.loss_breakdown()", "Axes", "Horizontal bars of loss components"),
            (
                ".plt.dashboard()",
                "(Figure, Axes)",
                "2×2 dashboard: convergence + orbit + metrics",
            ),
        ]
        html += (
            "<table style='width:100%;border-collapse:collapse;'>"
            f"<thead><tr style='background:rgba(51,65,85,0.6);'>"
            f"<th style='{th_style}'>Accessor</th>"
            f"<th style='{th_style}'>Returns</th>"
            f"<th style='{th_style}'>Description</th></tr></thead><tbody>"
        )
        for accessor, ret, desc in result_items:
            html += (
                f"<tr style='border-bottom:1px solid rgba(71,85,105,0.3);'>"
                f"<td style='{td_mono}'>{_esc(accessor)}</td>"
                f"<td style='{td_desc}font-style:italic;color:#94a3b8;font-size:0.8em;'>{_esc(ret)}</td>"
                f"<td style='{td_desc}'>{_esc(desc)}</td></tr>"
            )
        html += "</tbody></table></div>"

        # ── 5. Examples ──
        html += f"<div style='{section}'>"
        html += (
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:8px;'>"
            "💡 Examples</div>"
        )
        example = (
            "# Basic usage — fits omega0, N, chi_scale with hybrid loss\n"
            "result = job[0].vortex.autofit.thiele()\n"
            "\n"
            "# Spectral-only fit with custom parameters\n"
            "result = job[0].vortex.autofit.thiele(\n"
            "    objective='spectral',\n"
            "    fit_params=('omega0', 'N', 'chi_scale', 'phase0'),\n"
            "    global_search=True,\n"
            "    global_maxiter=100,\n"
            ")\n"
            "\n"
            "# CIP model with explicit current\n"
            "result = job[0].vortex.autofit.thiele(\n"
            "    model='cip',\n"
            "    current=5e10,  # A/m²\n"
            "    trajectory='full',\n"
            ")\n"
            "\n"
            "# Custom parameter bounds\n"
            "from mmpp.solitons.vortex.autofit import ParameterSpec\n"
            "result = job[0].vortex.autofit.thiele(\n"
            "    param_specs={\n"
            "        'omega0': ParameterSpec(lower=1e8, upper=1e11, prior_type='log_normal'),\n"
            "        'N': ParameterSpec(lower=0.0, upper=1.0, initial=0.2),\n"
            "    }\n"
            ")\n"
            "\n"
            "# Inspect results\n"
            "print(result.best_params)\n"
            "print(f'Improvement: {result.improvement_ratio:.3f}')\n"
            "result.plt.dashboard()        # Full diagnostic plots\n"
            "result.plt.convergence()      # Loss convergence curve\n"
            "result.comparison.plt.orbit() # Orbit overlay: numerical vs fitted"
        )
        html += f"<pre style='{code_style}'><code>{_esc(example)}</code></pre>"
        html += "</div>"

        html += "</div>"
        api = api_help_html(
            self,
            title="Vortex autofit API help",
            prefix="vortex.autofit",
            properties=[
                (
                    "thiele",
                    "Callable physics-informed autofit helper; use vortex.autofit.thiele(...)",
                )
            ],
            subtitle="The thiele callable accepts the parameters listed in Overview.",
            chrome=False,
        )
        return (
            '<div style=\'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;'
            "border:2px solid #334155;border-radius:12px;padding:14px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);'>"
            + html_tabs(
                [("Overview", html), ("API", api)],
                uid=f"mmpp-vortex-autofit-{uuid.uuid4().hex}",
            )
            + "</div>"
        )


__all__ = ["AutofitInterface"]
