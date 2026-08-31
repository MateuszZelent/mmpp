"""High-level nonlinear analysis interface for vortex dynamics."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..._method_helpers import InteractiveNodeMixin
from .._plotting import (
    apply_axes_style,
    ensure_axis,
    pop_axes_style_kwargs,
    pop_figure_kwargs,
)
from ..config import VortexConfig
from .amplitude_equation import compute_amplitude_equation
from .models import (
    AmplitudeEquationResult,
    STBatchResult,
    STParametersResult,
    ThieleForceBalanceResult,
)
from .nonliniearthiele import ThieleAnalyzer
from .slavin_tiberkevich import extract_st_parameters


class NonlinearInterface(InteractiveNodeMixin):
    """Nonlinear analysis namespace (ST, amplitude equation, and Thiele tools)."""

    _interactive_owner = "job[0].vortex.nonlinear"
    _interactive_nodes = frozenset(
        {
            "amplitude_equation",
            "slavin_tiberkevich",
            "slavin_tiberkevich_batch",
            "force_balance",
            "interactive_dashboard",
        }
    )
    _interactive_examples = {
        "amplitude_equation": [
            "amplitude = job[0].vortex.nonlinear.amplitude_equation()"
        ],
        "slavin_tiberkevich": ["st = job[0].vortex.nonlinear.slavin_tiberkevich()"],
        "slavin_tiberkevich_batch": [
            "batch = job[0].vortex.nonlinear.slavin_tiberkevich_batch(jobs, currents)"
        ],
        "force_balance": ["forces = job[0].vortex.nonlinear.force_balance()"],
        "interactive_dashboard": ["job[0].vortex.nonlinear.interactive_dashboard()"],
    }

    def __init__(
        self,
        job_result,
        dataset_name: str | None,
        slice_info: Any | None,
        config: VortexConfig,
        core_interface,
        trajectory_interface,
        spectrum_interface,
    ):
        self._job = job_result
        self._dataset_name = dataset_name
        self._slice_info = slice_info
        self._config = config
        self._core = core_interface
        self._trajectory = trajectory_interface
        self._spectrum = spectrum_interface

        self._last_amplitude: AmplitudeEquationResult | None = None
        self._last_st: STParametersResult | None = None
        self._last_batch: STBatchResult | None = None
        self._thiele: ThieleAnalyzer | None = None

    def _resolve_trajectory(self, trajectory=None):
        if trajectory is not None:
            return trajectory
        return self._core.track()

    def amplitude_equation(
        self,
        *,
        trajectory=None,
        reference_radius: float | None = None,
        method: str | None = None,
        center: tuple[float, float] | None = None,
        force: bool = False,
    ) -> AmplitudeEquationResult:
        """Compute nonlinear amplitude equation variables ``c(t)``, ``p(t)``, ``omega(t)``."""
        if (
            not force
            and self._last_amplitude is not None
            and trajectory is None
            and reference_radius is None
        ):
            return self._last_amplitude

        cfg = self._config.nonlinear
        selected_method = method or cfg.phase_method
        selected_reference_radius = (
            cfg.reference_radius if reference_radius is None else reference_radius
        )

        result = compute_amplitude_equation(
            self._resolve_trajectory(trajectory),
            reference_radius=selected_reference_radius,
            center=center,
            method=selected_method,
        )
        self._last_amplitude = result
        return result

    def slavin_tiberkevich(
        self,
        *,
        trajectory=None,
        spectrum_method: str | None = None,
        phase_method: str | None = None,
        steady_state_fraction: float | None = None,
        reference_radius: float | None = None,
        current_a: float | None = None,
        force: bool = False,
    ) -> STParametersResult:
        """Extract Slavin-Tiberkevich parameters from tracked trajectory."""
        if (
            not force
            and self._last_st is not None
            and trajectory is None
            and spectrum_method is None
            and phase_method is None
            and steady_state_fraction is None
            and reference_radius is None
            and current_a is None
        ):
            return self._last_st

        cfg = self._config.nonlinear
        result = extract_st_parameters(
            self._resolve_trajectory(trajectory),
            spectrum_method=spectrum_method or cfg.spectrum_method,
            phase_method=phase_method or cfg.phase_method,
            steady_state_fraction=(
                cfg.steady_state_fraction
                if steady_state_fraction is None
                else float(steady_state_fraction)
            ),
            reference_radius=(
                cfg.reference_radius if reference_radius is None else reference_radius
            ),
            current_a=current_a,
        )
        self._last_st = result
        return result

    def slavin_tiberkevich_batch(
        self,
        jobs,
        currents,
        *,
        dataset_name: str | None = None,
        **kwargs,
    ) -> STBatchResult:
        """Run Slavin-Tiberkevich extraction across a list of jobs and currents."""
        job_list = list(jobs)
        current_values = np.asarray(currents, dtype=float)

        if not job_list:
            raise ValueError("jobs cannot be empty")
        if current_values.size != len(job_list):
            raise ValueError("currents length must match jobs length")

        powers: list[float] = []
        linewidths: list[float] = []
        frequencies: list[float] = []

        for job_item, current in zip(job_list, current_values, strict=False):
            if dataset_name is None:
                vortex = job_item.solitons.vortex
            else:
                vortex = getattr(job_item, dataset_name).solitons.vortex

            st = vortex.nonlinear.slavin_tiberkevich(current_a=float(current), **kwargs)
            powers.append(float(st.generation_power))
            linewidths.append(float(st.linewidth_hz))
            frequencies.append(float(st.f_0_ghz) * 1e9)

        power_arr = np.asarray(powers, dtype=float)
        freq_arr = np.asarray(frequencies, dtype=float)

        valid = np.isfinite(power_arr) & np.isfinite(freq_arr)
        if int(np.sum(valid)) >= 2 and float(np.std(power_arr[valid])) > 1e-18:
            slope, _ = np.polyfit(power_arr[valid], 2.0 * np.pi * freq_arr[valid], 1)
            n_global = float(slope)
            fit_status = "ok"
        else:
            n_global = 0.0
            fit_status = "insufficient_variation"

        result = STBatchResult(
            currents=np.asarray(current_values, dtype=float),
            powers=power_arr,
            linewidths=np.asarray(linewidths, dtype=float),
            frequencies_hz=freq_arr,
            N=n_global,
            metadata={
                "n_jobs": len(job_list),
                "fit_status": fit_status,
                "dataset_name": dataset_name,
            },
        )
        self._last_batch = result
        return result

    @property
    def thiele(self) -> ThieleAnalyzer:
        """Thiele-force analysis and analytical simulation helpers."""
        if self._thiele is None:
            self._thiele = ThieleAnalyzer(
                self._job,
                dataset_name=self._dataset_name,
                core_interface=self._core,
            )
        return self._thiele

    def force_balance(self, **kwargs) -> ThieleForceBalanceResult:
        """Shortcut alias for ``self.thiele.force_balance``."""
        return self.thiele.force_balance(**kwargs)

    def interactive_dashboard(self, **kwargs):
        """Shortcut alias for ``self.thiele.interactive_dashboard``."""
        return self.thiele.interactive_dashboard(**kwargs)

    @property
    def plt(self):
        """Convenience plotting namespace."""
        return NonlinearInterfacePlotAccessor(self)

    def _repr_html_(self) -> str:
        import uuid as _uuid
        from html import escape as _esc

        from mmpp._repr_helpers import (
            NODE_COLOR_ANALYSIS,
            NODE_COLOR_COMPUTE,
            NODE_COLOR_PLOT,
            accessors_section_html,
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        context_rows = [
            ("dataset", self._dataset_name or "auto-detect", NODE_COLOR_COMPUTE),
            (
                "slice",
                "custom" if self._slice_info is not None else "full geometry",
                None,
            ),
            (
                "ST spectrum method",
                self._config.nonlinear.spectrum_method,
                NODE_COLOR_ANALYSIS,
            ),
            ("phase method", self._config.nonlinear.phase_method, NODE_COLOR_ANALYSIS),
            ("steady fraction", self._config.nonlinear.steady_state_fraction, None),
        ]
        accessors = [
            (
                "Compute:",
                [
                    (".amplitude_equation(...)", NODE_COLOR_COMPUTE),
                    (".slavin_tiberkevich(...)", NODE_COLOR_COMPUTE),
                    (
                        ".slavin_tiberkevich_batch(jobs, currents, ...)",
                        NODE_COLOR_COMPUTE,
                    ),
                ],
            ),
            (
                "Thiele:",
                [
                    (".thiele", NODE_COLOR_ANALYSIS),
                    (".force_balance(...)", NODE_COLOR_ANALYSIS),
                    (".interactive_dashboard(...)", NODE_COLOR_ANALYSIS),
                ],
            ),
            (
                "Plotting:",
                [
                    (".plt.power_vs_current()", NODE_COLOR_PLOT),
                    (".plt.linewidth_vs_current()", NODE_COLOR_PLOT),
                    (".plt.force_balance(...)", NODE_COLOR_PLOT),
                ],
            ),
        ]
        methods = [
            (".amplitude_equation()", "Compute c(t), p(t), ω(t) from orbit"),
            (".slavin_tiberkevich()", "Extract ST parameters (Q, Γ+, Γ−, N)"),
            (
                ".slavin_tiberkevich_batch(jobs, currents)",
                "ST parameters across current sweep",
            ),
            (".thiele", "Thiele equation analysis namespace"),
            (".force_balance(**kw)", "Shortcut → thiele.force_balance()"),
            (
                ".interactive_dashboard(**kw)",
                "Shortcut → thiele.interactive_dashboard()",
            ),
            (".plt.power_vs_current()", "Plot P(I) from batch results"),
            (".plt.linewidth_vs_current()", "Plot Δf(I) from batch results"),
            (".plt.force_balance()", "Plot Thiele force decomposition"),
        ]
        method_rows = "".join(
            "<tr>"
            f"<td style='padding:6px 8px;font-family:monospace;color:{NODE_COLOR_COMPUTE};vertical-align:top;'>{_esc(m)}</td>"
            f"<td style='padding:6px 8px;color:#f8f8f2;'>{_esc(d)}</td>"
            "</tr>"
            for m, d in methods
        )
        example = (
            "# Slavin-Tiberkevich parameters\n"
            "st = jobs[-1].solitons.vortex.nonlinear.slavin_tiberkevich()\n"
            "print(f'Q={st.Q:.1f}, N={st.N:.2e}')\n"
            "\n"
            "# Batch across current sweep\n"
            "batch = jobs[-1].solitons.vortex.nonlinear.slavin_tiberkevich_batch(\n"
            "    jobs=[job1, job2, job3],\n"
            "    currents=[1e-3, 2e-3, 3e-3]\n"
            ")\n"
            "jobs[-1].solitons.vortex.nonlinear.plt.power_vs_current()\n"
            "\n"
            "# Thiele force balance\n"
            "jobs[-1].solitons.vortex.nonlinear.force_balance()\n"
            "jobs[-1].solitons.vortex.nonlinear.thiele.interactive_dashboard()"
        )
        methods_html = (
            "<div style='background:linear-gradient(135deg,rgba(68,71,90,0.55) 0%,rgba(40,42,54,0.55) 100%);"
            "padding:12px;border-radius:8px;margin-bottom:12px;border:1px solid rgba(98,114,164,0.35);'>"
            "<b style='color:#bd93f9;'>Nonlinear Workflows</b>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;margin-top:8px;'>"
            f"{method_rows}</table></div>"
        )
        api = api_help_html(
            self,
            title="Nonlinear dynamics API help",
            prefix="jobs[-1].solitons.vortex.nonlinear",
            properties=[
                ("thiele", "Thiele-force analysis and analytical simulation helpers"),
                ("plt", "Convenience plotting namespace"),
            ],
            methods=[
                "amplitude_equation",
                "slavin_tiberkevich",
                "slavin_tiberkevich_batch",
                "force_balance",
                "interactive_dashboard",
            ],
            subtitle="Live public API for Slavin-Tiberkevich, Thiele, and amplitude analysis.",
            chrome=False,
        )
        return node_card_html(
            "Nonlinear Dynamics Interface",
            icon="📉",
            subtitle="Slavin-Tiberkevich, Thiele-equation, and amplitude-equation analysis for nonlinear vortex dynamics.",
            sections=[
                metrics_section_html(context_rows),
                accessors_section_html(accessors),
                methods_html,
                examples_section_html(example, title="Nonlinear Examples"),
            ],
            api=api,
            uid=f"mmpp-vortex-nonlinear-{str(_uuid.uuid4())[:8]}",
        )


class NonlinearInterfacePlotAccessor(InteractiveNodeMixin):
    """Plotting facade for :class:`NonlinearInterface`."""

    _interactive_owner = "job[0].vortex.nonlinear.plt"
    _interactive_nodes = frozenset(
        {"power_vs_current", "linewidth_vs_current", "force_balance"}
    )
    _interactive_examples = {
        "power_vs_current": ["job[0].vortex.nonlinear.plt.power_vs_current()"],
        "linewidth_vs_current": ["job[0].vortex.nonlinear.plt.linewidth_vs_current()"],
        "force_balance": ["job[0].vortex.nonlinear.plt.force_balance()"],
    }

    def __init__(self, interface: NonlinearInterface):
        self._interface = interface

    def power_vs_current(self, **kwargs):
        """Plot power-vs-current from latest batch or single-point estimate."""
        if self._interface._last_batch is not None:
            return self._interface._last_batch.plt.power_vs_current(**kwargs)
        st = self._interface.slavin_tiberkevich()
        return st.plt.power_vs_current(**kwargs)

    def linewidth_vs_current(self, **kwargs):
        """Plot linewidth-vs-current from latest batch (or single point at x=0)."""
        if self._interface._last_batch is not None:
            return self._interface._last_batch.plt.linewidth_vs_current(**kwargs)

        st = self._interface.slavin_tiberkevich()
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(plot_kwargs.pop("ax", None), figure_kwargs=figure_kwargs)

        linewidth = float(st.linewidth_hz)
        if plot_kwargs.pop("as_mhz", True):
            linewidth *= 1e-6
            ylabel = "Linewidth [MHz]"
        else:
            ylabel = "Linewidth [Hz]"

        ax.plot([0.0], [linewidth], marker="o", **plot_kwargs)
        ax.set_xlabel("Index")
        ax.set_ylabel(ylabel)
        ax.set_title("Linewidth vs current")
        apply_axes_style(ax, style_kwargs)
        return ax

    def force_balance(self, **kwargs):
        """Plot Thiele force balance from current or provided trajectory."""
        all_kwargs = dict(kwargs)

        nested_compute = all_kwargs.pop("compute_kwargs", None)
        compute_kwargs: dict[str, Any] = {}
        if isinstance(nested_compute, dict):
            compute_kwargs.update(nested_compute)

        compute_keys = {
            "trajectory",
            "polarity",
            "vorticity",
            "Ms",
            "thickness",
            "eta",
            "gamma",
            "gamma0",
            "kappa",
            "center",
            "stt_force",
            "oersted_force",
        }

        for key in list(all_kwargs.keys()):
            if key in compute_keys:
                compute_kwargs[key] = all_kwargs.pop(key)
        if "thiele_alpha" in all_kwargs:
            compute_kwargs["alpha"] = all_kwargs.pop("thiele_alpha")

        plot_kwargs = dict(all_kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        plot_kwargs.update(style_kwargs)
        plot_kwargs.update(figure_kwargs)

        result = self._interface.thiele.force_balance(**compute_kwargs)
        return result.plt.force_balance(**plot_kwargs)

    def _repr_html_(self) -> str:
        import uuid as _uuid

        from mmpp._repr_helpers import api_help_html, node_card_html, plot_accessor_html

        overview = plot_accessor_html(
            "NonlinearInterfacePlotAccessor",
            [
                (
                    ".power_vs_current()",
                    "Generation power P(I) from batch or single ST result",
                    "Uses latest batch if available, else single-point.",
                ),
                (
                    ".linewidth_vs_current(as_mhz=True)",
                    "Linewidth Δf(I) from batch or single-point",
                    "as_mhz: convert to MHz.",
                ),
                (
                    ".force_balance(as_norm=True)",
                    "Thiele force decomposition vs time",
                    "as_norm: True for |F| norms. Accepts compute_kwargs dict.",
                ),
            ],
        )
        api = api_help_html(
            self,
            title="Nonlinear plot API help",
            prefix="jobs[-1].solitons.vortex.nonlinear.plt",
            methods=["power_vs_current", "linewidth_vs_current", "force_balance"],
            subtitle="Plot helpers for latest nonlinear results or on-demand calculations.",
            chrome=False,
        )
        return node_card_html(
            "Nonlinear Plot Accessor",
            icon="🎨",
            subtitle="Plot shortcuts for nonlinear current sweeps and force-balance diagnostics.",
            sections=[overview],
            api=api,
            uid=f"mmpp-vortex-nonlinear-plot-{str(_uuid.uuid4())[:8]}",
        )
