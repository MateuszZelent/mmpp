"""Configuration and parameter specification for vortex autofit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ParameterSpec:
    """Specification for a single fittable parameter."""

    lower: float = -np.inf
    upper: float = np.inf
    initial: float | None = None
    prior_mean: float | None = None
    prior_std: float | None = None
    prior_type: str = "gaussian"
    frozen: bool = False
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.lower >= self.upper:
            raise ValueError(
                f"ParameterSpec: lower ({self.lower}) must be < upper ({self.upper})"
            )
        if self.prior_type not in {"gaussian", "log_normal", "uniform"}:
            raise ValueError(
                f"prior_type must be 'gaussian', 'log_normal', or 'uniform', "
                f"got {self.prior_type!r}"
            )

    def _repr_html_(self) -> str:
        from html import escape as _esc

        lo = f"{self.lower:.4g}" if abs(self.lower) < 1e15 else "-∞"
        hi = f"{self.upper:.4g}" if abs(self.upper) < 1e15 else "+∞"
        init = f"{self.initial:.4g}" if self.initial is not None else "—"
        pm = f"{self.prior_mean:.4g}" if self.prior_mean is not None else "—"
        ps = f"{self.prior_std:.4g}" if self.prior_std is not None else "—"
        frozen_icon = "❄️ Frozen" if self.frozen else "🔓 Free"
        frozen_color = "#f59e0b" if self.frozen else "#22c55e"

        td = "padding:4px 10px;color:#cbd5e1;font-size:0.9em;"
        lbl = "padding:4px 10px;color:#94a3b8;font-size:0.9em;font-weight:600;"
        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:10px;padding:14px;margin:6px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;box-shadow:0 6px 15px rgba(0,0,0,0.2);display:inline-block;\">"
            "<div style='font-weight:600;color:#f1f5f9;margin-bottom:8px;'>"
            f"ParameterSpec <span style='color:{frozen_color};font-size:0.85em;'>"
            f"{frozen_icon}</span></div>"
            "<table style='border-collapse:collapse;'>"
            f"<tr><td style='{lbl}'>Bounds</td><td style='{td}'>[{_esc(lo)}, {_esc(hi)}]</td></tr>"
            f"<tr><td style='{lbl}'>Initial</td><td style='{td}'>{_esc(init)}</td></tr>"
            f"<tr><td style='{lbl}'>Prior type</td><td style='{td}'>{_esc(self.prior_type)}</td></tr>"
            f"<tr><td style='{lbl}'>Prior mean</td><td style='{td}'>{_esc(pm)}</td></tr>"
            f"<tr><td style='{lbl}'>Prior std</td><td style='{td}'>{_esc(ps)}</td></tr>"
            f"<tr><td style='{lbl}'>Scale</td><td style='{td}'>{self.scale:.4g}</td></tr>"
            "</table></div>"
        )


# Default parameter specifications for CPP model (no Oersted)
DEFAULT_PARAM_SPECS: dict[str, ParameterSpec] = {
    "omega0": ParameterSpec(
        lower=1e6,
        upper=1e12,
        prior_type="log_normal",
        scale=1e9,
    ),
    "N": ParameterSpec(
        lower=-0.5,
        upper=2.0,
        initial=0.25,
        prior_mean=0.25,
        prior_std=0.3,
        prior_type="gaussian",
        scale=1.0,
    ),
    "chi_scale": ParameterSpec(
        lower=0.1,
        upper=10.0,
        initial=1.0,
        prior_mean=1.0,
        prior_std=0.5,
        prior_type="log_normal",
        scale=1.0,
    ),
    "P_model": ParameterSpec(
        lower=-1.0,
        upper=1.0,
        prior_std=0.15,
        prior_type="gaussian",
        scale=1.0,
    ),
    "phase0": ParameterSpec(
        lower=-np.pi,
        upper=np.pi,
        initial=0.0,
        prior_type="uniform",
        frozen=True,
    ),
    "center_x": ParameterSpec(
        lower=-1e-6,
        upper=1e-6,
        initial=0.0,
        prior_type="gaussian",
        frozen=True,
        scale=1e-9,
    ),
    "center_y": ParameterSpec(
        lower=-1e-6,
        upper=1e-6,
        initial=0.0,
        prior_type="gaussian",
        frozen=True,
        scale=1e-9,
    ),
    "d0_scale": ParameterSpec(
        lower=0.6,
        upper=1.6,
        initial=1.0,
        prior_mean=1.0,
        prior_std=0.15,
        prior_type="log_normal",
        frozen=True,
    ),
    "domega0_dJ": ParameterSpec(
        lower=-1e-4,
        upper=1e-4,
        initial=0.0,
        prior_mean=0.0,
        prior_std=1e-5,
        prior_type="gaussian",
        frozen=True,
    ),
}


# Default loss weights for each objective mode
LOSS_PRESETS: dict[str, dict[str, float]] = {
    "time": {
        "w_xy": 1.0,
        "w_r": 0.5,
        "w_core": 0.35,
        "w_phi": 0.3,
        "w_freq": 0.2,
        "w_psd": 0.0,
        "w_ellip": 0.1,
        "w_stability": 0.4,
        "w_reg": 0.01,
    },
    "spectral": {
        "w_xy": 0.0,
        "w_r": 0.2,
        "w_core": 0.15,
        "w_phi": 0.0,
        "w_freq": 1.0,
        "w_psd": 0.8,
        "w_ellip": 0.1,
        "w_stability": 0.2,
        "w_reg": 0.01,
    },
    "hybrid": {
        "w_xy": 0.5,
        "w_r": 0.3,
        "w_core": 0.3,
        "w_phi": 0.2,
        "w_freq": 0.5,
        "w_psd": 0.3,
        "w_ellip": 0.1,
        "w_stability": 0.35,
        "w_reg": 0.01,
    },
}


@dataclass
class AutofitConfig:
    """Full configuration for a vortex autofit run."""

    # Trajectory resolution
    trajectory: str = "steady_state"
    tracking_source: str = "auto"
    tracking_method: str | None = None
    initial_condition: str = "auto"

    # Model
    model: str = "auto"
    params: str | dict = "auto"
    current: str | float | None = None

    # Fit parameters
    fit_params: tuple[str, ...] = ("omega0", "N", "chi_scale")
    param_specs: dict[str, ParameterSpec] | None = None

    # Objective
    objective: str = "hybrid"
    weights: dict[str, float] | None = None

    # Optimization
    global_search: bool = False
    global_method: str = "differential_evolution"
    global_maxiter: int = 15
    global_popsize: int = 8
    local_method: str = "L-BFGS-B"
    local_maxiter: int = 100
    max_eval: int = 500

    # Phase alignment
    align_phase: bool = True
    align_center: bool = True

    # Windowing
    windowing: str = "steady_state"
    n_periods: int = 10

    # Reproducibility
    random_seed: int | None = None

    # Field handling (Phase 2+)
    allow_oersted: bool = False
    allow_field_fit: bool = False

    # Progress
    verbose: bool = True
    live_plot: bool = False
    live_plot_every: int = 5

    def get_param_specs(self) -> dict[str, ParameterSpec]:
        """Merge user specs with defaults for active fit_params."""
        specs: dict[str, ParameterSpec] = {}
        user_specs = self.param_specs or {}
        for name in self.fit_params:
            if name in user_specs:
                specs[name] = user_specs[name]
            elif name in DEFAULT_PARAM_SPECS:
                spec = DEFAULT_PARAM_SPECS[name]
                specs[name] = ParameterSpec(
                    lower=spec.lower,
                    upper=spec.upper,
                    initial=spec.initial,
                    prior_mean=spec.prior_mean,
                    prior_std=spec.prior_std,
                    prior_type=spec.prior_type,
                    frozen=False if spec.frozen else spec.frozen,
                    scale=spec.scale,
                )
            else:
                specs[name] = ParameterSpec()
        return specs

    def get_weights(self) -> dict[str, float]:
        """Resolve loss weights from preset or user override."""
        if self.objective not in LOSS_PRESETS:
            raise ValueError(
                f"objective must be one of {list(LOSS_PRESETS)}, got {self.objective!r}"
            )
        weights = dict(LOSS_PRESETS[self.objective])
        if self.weights:
            weights.update(self.weights)
        return weights

    def _repr_html_(self) -> str:
        from html import escape as _esc

        card = (
            "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);"
        )
        section = (
            "background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:8px;border:1px solid rgba(148,163,184,0.2);"
        )
        lbl = "padding:3px 8px;color:#94a3b8;font-size:0.85em;font-weight:600;white-space:nowrap;"
        val = "padding:3px 8px;color:#e2e8f0;font-size:0.85em;font-family:monospace;"

        groups = [
            ("🎯 Trajectory", [
                ("trajectory", self.trajectory),
                ("tracking_source", self.tracking_source),
                ("tracking_method", self.tracking_method),
                ("windowing", self.windowing),
                ("n_periods", self.n_periods),
            ]),
            ("🧮 Model", [
                ("model", self.model),
                ("params", self.params if isinstance(self.params, str) else "{...}"),
                ("current", self.current),
                ("allow_oersted", self.allow_oersted),
                ("allow_field_fit", self.allow_field_fit),
            ]),
            ("🔧 Fit Parameters", [
                ("fit_params", ", ".join(self.fit_params)),
                ("param_specs", f"{len(self.param_specs)} overrides" if self.param_specs else "defaults"),
            ]),
            ("⚖️ Objective", [
                ("objective", self.objective),
                ("weights", "custom" if self.weights else f"preset ({self.objective})"),
            ]),
            ("🔍 Optimisation", [
                ("global_search", self.global_search),
                ("global_method", self.global_method),
                ("global_maxiter", self.global_maxiter),
                ("global_popsize", self.global_popsize),
                ("local_method", self.local_method),
                ("local_maxiter", self.local_maxiter),
                ("max_eval", self.max_eval),
            ]),
            ("⚙️ Alignment & Misc", [
                ("align_phase", self.align_phase),
                ("align_center", self.align_center),
                ("random_seed", self.random_seed),
                ("verbose", self.verbose),
                ("live_plot", self.live_plot),
                ("live_plot_every", self.live_plot_every),
            ]),
        ]

        html = f"<div style='{card}'>"
        html += (
            "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;margin-bottom:10px;'>"
            "⚙️ AutofitConfig</div>"
        )

        for title, fields in groups:
            html += f"<div style='{section}'>"
            html += f"<div style='font-weight:600;color:#e2e8f0;margin-bottom:4px;font-size:0.9em;'>{title}</div>"
            html += "<table style='width:100%;border-collapse:collapse;'>"
            for name, value in fields:
                v_str = str(value) if value is not None else "None"
                # Color booleans
                if isinstance(value, bool):
                    v_color = "#22c55e" if value else "#94a3b8"
                    v_str = str(value)
                else:
                    v_color = "#e2e8f0"
                html += (
                    f"<tr style='border-bottom:1px solid rgba(71,85,105,0.15);'>"
                    f"<td style='{lbl}'>{_esc(name)}</td>"
                    f"<td style='{val}color:{v_color};'>{_esc(v_str)}</td></tr>"
                )
            html += "</table></div>"

        # Resolved weights table
        try:
            resolved = self.get_weights()
            loss_labels = {
                "w_xy": "x,y MSE", "w_r": "radius", "w_phi": "phase",
                "w_freq": "frequency", "w_psd": "spectral PSD",
                "w_ellip": "eccentricity", "w_stability": "stability", "w_reg": "regularisation",
            }
            html += f"<div style='{section}'>"
            html += "<div style='font-weight:600;color:#e2e8f0;margin-bottom:4px;font-size:0.9em;'>⚖️ Resolved Weights</div>"
            html += "<div style='display:flex;flex-wrap:wrap;gap:6px;'>"
            for key, w in resolved.items():
                label = loss_labels.get(key, key)
                bg = "rgba(34,197,94,0.15)" if w >= 0.5 else ("rgba(165,180,252,0.1)" if w > 0 else "rgba(71,85,105,0.2)")
                color = "#22c55e" if w >= 0.5 else ("#a5b4fc" if w > 0 else "#475569")
                html += (
                    f"<span style='background:{bg};border:1px solid {color}33;"
                    f"padding:3px 8px;border-radius:5px;font-size:0.8em;color:{color};'>"
                    f"{_esc(label)} <b>{w:.2f}</b></span>"
                )
            html += "</div></div>"
        except Exception:
            pass

        html += "</div>"
        return html


__all__ = [
    "AutofitConfig",
    "ParameterSpec",
    "DEFAULT_PARAM_SPECS",
    "LOSS_PRESETS",
]
