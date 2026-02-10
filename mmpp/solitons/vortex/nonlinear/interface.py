"""High-level nonlinear analysis interface for vortex dynamics."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import VortexConfig
from .amplitude_equation import compute_amplitude_equation
from .models import (
    AmplitudeEquationResult,
    STBatchResult,
    STParametersResult,
    ThieleForceBalanceResult,
)
from .slavin_tiberkevich import extract_st_parameters
from .thiele import ThieleAnalyzer


class NonlinearInterface:
    """Nonlinear analysis namespace (ST, amplitude equation, and Thiele tools)."""

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
        if not force and self._last_amplitude is not None and trajectory is None and reference_radius is None:
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
            reference_radius=(cfg.reference_radius if reference_radius is None else reference_radius),
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

        for job_item, current in zip(job_list, current_values):
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
        from html import escape as _esc

        methods = [
            (".amplitude_equation()", "Compute c(t), p(t), ω(t) from orbit"),
            (".slavin_tiberkevich()", "Extract ST parameters (Q, Γ+, Γ−, N)"),
            (".slavin_tiberkevich_batch(jobs, currents)", "ST parameters across current sweep"),
            (".thiele", "Thiele equation analysis namespace"),
            (".force_balance(**kw)", "Shortcut → thiele.force_balance()"),
            (".interactive_dashboard(**kw)", "Shortcut → thiele.interactive_dashboard()"),
            (".plt.power_vs_current()", "Plot P(I) from batch results"),
            (".plt.linewidth_vs_current()", "Plot Δf(I) from batch results"),
            (".plt.force_balance()", "Plot Thiele force decomposition"),
        ]
        method_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(m)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(d)}</td></tr>"
            for m, d in methods
        )
        example = (
            "# Slavin-Tiberkevich parameters\n"
            "st = vortex.nonlinear.slavin_tiberkevich()\n"
            "print(f'Q={st.Q:.1f}, N={st.N:.2e}')\n"
            "\n"
            "# Batch across current sweep\n"
            "batch = vortex.nonlinear.slavin_tiberkevich_batch(\n"
            "    jobs=[job1, job2, job3],\n"
            "    currents=[1e-3, 2e-3, 3e-3]\n"
            ")\n"
            "vortex.nonlinear.plt.power_vs_current()\n"
            "\n"
            "# Thiele force balance\n"
            "vortex.nonlinear.force_balance()\n"
            "vortex.nonlinear.thiele.interactive_dashboard()"
        )
        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);\">"
            "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;margin-bottom:4px;'>"
            "Nonlinear Dynamics Interface</div>"
            "<div style='font-size:0.85em;color:#94a3b8;margin-bottom:10px;'>"
            "Slavin-Tiberkevich, Thiele equation, and amplitude analysis</div>"
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Methods</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            f"{method_rows}</table></div>"
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Examples</div>"
            "<pre style='margin:0;background:rgba(15,23,42,0.85);padding:10px;"
            "border-radius:6px;color:#e2e8f0;overflow-x:auto;font-size:0.85em;'>"
            f"<code>{example}</code></pre></div>"
            "</div>"
        )


class NonlinearInterfacePlotAccessor:
    """Plotting facade for :class:`NonlinearInterface`."""

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

        import matplotlib.pyplot as plt

        st = self._interface.slavin_tiberkevich()
        ax = kwargs.pop("ax", None)
        if ax is None:
            _, ax = plt.subplots()

        linewidth = float(st.linewidth_hz)
        if kwargs.pop("as_mhz", True):
            linewidth *= 1e-6
            ylabel = "Linewidth [MHz]"
        else:
            ylabel = "Linewidth [Hz]"

        ax.plot([0.0], [linewidth], marker="o", **kwargs)
        ax.set_xlabel("Index")
        ax.set_ylabel(ylabel)
        ax.set_title("Linewidth vs current")
        return ax

    def force_balance(self, **kwargs):
        """Plot Thiele force balance from current or provided trajectory."""
        result = self._interface.thiele.force_balance(**kwargs)
        return result.plt.force_balance()
