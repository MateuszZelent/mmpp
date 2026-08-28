"""Result models for vortex nonlinear analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..._method_helpers import InteractiveNodeMixin
from .._plotting import (
    apply_axes_style,
    ensure_axis,
    pop_axes_style_kwargs,
    pop_figure_kwargs,
)


@dataclass
class AmplitudeEquationResult:
    """Complex-amplitude dynamics derived from tracked vortex orbit."""

    time: np.ndarray
    complex_amplitude: np.ndarray
    power: np.ndarray
    phase: np.ndarray
    omega: np.ndarray
    method: str
    reference_radius: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frequency_hz(self) -> np.ndarray:
        """Instantaneous frequency in Hz."""
        return np.asarray(self.omega, dtype=float) / (2.0 * np.pi)

    @property
    def plt(self) -> AmplitudePlotAccessor:
        """Plotting accessor."""
        return AmplitudePlotAccessor(self)

    def _repr_html_(self) -> str:
        import uuid as _uuid

        from mmpp._repr_helpers import (
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        return node_card_html(
            "Amplitude Equation Result",
            icon="📐",
            subtitle="Complex-amplitude dynamics derived from the tracked vortex orbit.",
            sections=[
                metrics_section_html(
                    [
                        ("method", self.method, None),
                        ("n_samples", int(np.asarray(self.time).size), None),
                        (
                            "reference_radius",
                            f"{float(self.reference_radius):.6g}",
                            None,
                        ),
                    ]
                ),
                examples_section_html(
                    "amp = jobs[-1].solitons.vortex.nonlinear.amplitude_equation()\n"
                    "amp.plt.power_vs_time()\n"
                    "amp.plt.complex_plane()",
                    title="Result Usage",
                ),
            ],
            api=api_help_html(
                self,
                title="Amplitude-equation result API help",
                prefix="jobs[-1].solitons.vortex.nonlinear.amplitude_equation()",
                properties=[
                    ("time", "Time axis"),
                    ("complex_amplitude", "Complex amplitude c(t)"),
                    ("power", "Generation power p(t)"),
                    ("phase", "Phase trajectory"),
                    ("omega", "Instantaneous angular frequency"),
                    ("frequency_hz", "Instantaneous frequency in Hz"),
                    ("plt", "Plotting accessor"),
                ],
                subtitle="Live attributes of the amplitude-equation result.",
                chrome=False,
            ),
            uid=f"amplitude-equation-result-{str(_uuid.uuid4())[:8]}",
        )


@dataclass
class STParametersResult:
    """Slavin-Tiberkevich parameters extracted from a single trajectory."""

    omega_0: float
    f_0_ghz: float
    N: float
    Gamma_G: float
    Q: float
    sigma: float
    I_threshold: float
    generation_power: float
    linewidth_hz: float
    quality_factor: float
    linewidth_resolution_limited: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def power_coefficient_of_variation(self) -> float:
        """Coefficient of variation of generated power, retained as legacy ``Q``."""
        return float(self.Q)

    @property
    def plt(self) -> STPlotAccessor:
        """Plotting accessor for single-point ST parameters."""
        return STPlotAccessor(self)

    def _repr_html_(self) -> str:
        import uuid as _uuid

        from mmpp._repr_helpers import (
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        return node_card_html(
            "ST Parameters Result",
            icon="🧮",
            subtitle="Slavin-Tiberkevich parameters extracted from a single trajectory.",
            sections=[
                metrics_section_html(
                    [
                        ("f_0_ghz", f"{float(self.f_0_ghz):.6g}", None),
                        ("N", f"{float(self.N):.6g}", None),
                        ("Gamma_G", f"{float(self.Gamma_G):.6g}", None),
                        ("linewidth_hz", f"{float(self.linewidth_hz):.6g}", None),
                        ("Q meaning", "power coefficient of variation", None),
                    ]
                ),
                examples_section_html(
                    "st = jobs[-1].solitons.vortex.nonlinear.slavin_tiberkevich()\n"
                    "st.quality_factor\n"
                    "st.plt.power_vs_current()",
                    title="Result Usage",
                ),
            ],
            api=api_help_html(
                self,
                title="ST parameters API help",
                prefix="jobs[-1].solitons.vortex.nonlinear.slavin_tiberkevich()",
                properties=[
                    ("omega_0", "Auto-oscillation angular frequency"),
                    ("f_0_ghz", "Auto-oscillation frequency in GHz"),
                    ("N", "Nonlinear frequency shift coefficient"),
                    ("Gamma_G", "Positive damping"),
                    ("Q", "Power coefficient of variation"),
                    ("sigma", "Spin-torque efficiency"),
                    ("I_threshold", "Threshold current"),
                    ("generation_power", "Generated power"),
                    ("linewidth_hz", "Estimated linewidth"),
                    ("quality_factor", "Quality factor"),
                    ("plt", "Plotting accessor"),
                ],
                subtitle="Live attributes of the extracted Slavin-Tiberkevich parameters.",
                chrome=False,
            ),
            uid=f"st-parameters-result-{str(_uuid.uuid4())[:8]}",
        )


@dataclass
class STBatchResult:
    """Batch Slavin-Tiberkevich summary across current sweep."""

    currents: np.ndarray
    powers: np.ndarray
    linewidths: np.ndarray
    frequencies_hz: np.ndarray
    N: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frequencies_ghz(self) -> np.ndarray:
        """Dominant frequencies in GHz."""
        return np.asarray(self.frequencies_hz, dtype=float) * 1e-9

    @property
    def plt(self) -> STBatchPlotAccessor:
        """Plotting accessor for batch ST results."""
        return STBatchPlotAccessor(self)

    def _repr_html_(self) -> str:
        import uuid as _uuid

        from mmpp._repr_helpers import (
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        return node_card_html(
            "ST Batch Result",
            icon="📦",
            subtitle="Batch Slavin-Tiberkevich summary across a current sweep.",
            sections=[
                metrics_section_html(
                    [
                        ("n_currents", int(np.asarray(self.currents).size), None),
                        ("N", f"{float(self.N):.6g}", None),
                        (
                            "fit_status",
                            self.metadata.get("fit_status", "unknown"),
                            None,
                        ),
                    ]
                ),
                examples_section_html(
                    "batch = jobs[-1].solitons.vortex.nonlinear.slavin_tiberkevich_batch(jobs, currents)\n"
                    "batch.plt.power_vs_current()\n"
                    "batch.plt.frequency_vs_current()",
                    title="Result Usage",
                ),
            ],
            api=api_help_html(
                self,
                title="ST batch result API help",
                prefix="jobs[-1].solitons.vortex.nonlinear.slavin_tiberkevich_batch()",
                properties=[
                    ("currents", "Sweep current values"),
                    ("powers", "Generated powers"),
                    ("linewidths", "Linewidth values"),
                    ("frequencies_hz", "Dominant frequencies in Hz"),
                    ("frequencies_ghz", "Dominant frequencies in GHz"),
                    ("N", "Global nonlinear frequency-shift fit"),
                    ("plt", "Plotting accessor"),
                ],
                subtitle="Live attributes of the batch Slavin-Tiberkevich sweep.",
                chrome=False,
            ),
            uid=f"st-batch-result-{str(_uuid.uuid4())[:8]}",
        )


class AmplitudePlotAccessor(InteractiveNodeMixin):
    """Plot helpers for :class:`AmplitudeEquationResult`."""

    _interactive_owner = "job[0].vortex.nonlinear.amplitude_equation().plt"
    _interactive_nodes = frozenset({"power_vs_time", "phase_vs_time", "complex_plane"})
    _interactive_examples = {
        "power_vs_time": [
            "job[0].vortex.nonlinear.amplitude_equation().plt.power_vs_time()"
        ],
        "phase_vs_time": [
            "job[0].vortex.nonlinear.amplitude_equation().plt.phase_vs_time()"
        ],
        "complex_plane": [
            "job[0].vortex.nonlinear.amplitude_equation().plt.complex_plane()"
        ],
    }

    def __init__(self, result: AmplitudeEquationResult):
        self._result = result

    def power_vs_time(self, *, ax=None, **kwargs):
        """Plot ``p(t)=|c(t)|^2``."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        ax.plot(self._result.time, self._result.power, **plot_kwargs)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Generation power p(t) [a.u.]")
        ax.set_title("Amplitude equation: power")
        apply_axes_style(ax, style_kwargs)
        return ax

    def phase_vs_time(self, *, ax=None, as_unwrapped: bool = True, **kwargs):
        """Plot trajectory phase versus time."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        if as_unwrapped:
            values = np.asarray(self._result.phase, dtype=float)
            ylabel = "Phase [rad]"
        else:
            values = np.angle(self._result.complex_amplitude)
            ylabel = "Wrapped phase [rad]"

        ax.plot(self._result.time, values, **plot_kwargs)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title("Amplitude equation: phase")
        apply_axes_style(ax, style_kwargs)
        return ax

    def complex_plane(self, *, ax=None, **kwargs):
        """Plot complex amplitude trajectory in Re-Im plane."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        c = np.asarray(self._result.complex_amplitude, dtype=np.complex128)
        ax.plot(c.real, c.imag, **plot_kwargs)
        ax.set_xlabel("Re(c)")
        ax.set_ylabel("Im(c)")
        ax.set_title("Complex amplitude c(t)")
        ax.set_aspect("equal")
        apply_axes_style(ax, style_kwargs)
        return ax

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html

        return plot_accessor_html(
            "AmplitudePlotAccessor",
            [
                (
                    ".power_vs_time()",
                    "p(t)=|c(t)|² generation power vs time",
                    "Accepts matplotlib kwargs.",
                ),
                (
                    ".phase_vs_time(as_unwrapped=True)",
                    "Phase vs time",
                    "as_unwrapped: True for cumulative, False for wrapped [-π,π].",
                ),
                (
                    ".complex_plane()",
                    "Complex amplitude c(t) in Re-Im plane",
                    "Equal aspect ratio. Accepts matplotlib kwargs.",
                ),
            ],
        )


class STPlotAccessor(InteractiveNodeMixin):
    """Plot helpers for :class:`STParametersResult`."""

    _interactive_owner = "job[0].vortex.nonlinear.slavin_tiberkevich().plt"
    _interactive_nodes = frozenset({"power_vs_current"})
    _interactive_examples = {
        "power_vs_current": [
            "job[0].vortex.nonlinear.slavin_tiberkevich().plt.power_vs_current()"
        ]
    }

    def __init__(self, result: STParametersResult):
        self._result = result

    def power_vs_current(self, *, ax=None, current_a: float | None = None, **kwargs):
        """Plot single-point generation power against current."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        if current_a is None:
            current_a = self._result.metadata.get("current_a")
        x_val = float(current_a) if current_a is not None else 0.0

        ax.plot([x_val], [self._result.generation_power], marker="o", **plot_kwargs)
        ax.set_xlabel("Current [A]" if current_a is not None else "Index")
        ax.set_ylabel("Generation power p_gen [a.u.]")
        ax.set_title("Slavin-Tiberkevich: power vs current")
        apply_axes_style(ax, style_kwargs)
        return ax

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html

        return plot_accessor_html(
            "STPlotAccessor",
            [
                (
                    ".power_vs_current(current_a=...)",
                    "Single-point generation power at given current",
                    "current_a: current value in Amperes. Falls back to metadata if None.",
                ),
            ],
        )


class STBatchPlotAccessor(InteractiveNodeMixin):
    """Plot helpers for :class:`STBatchResult`."""

    _interactive_owner = "job[0].vortex.nonlinear.slavin_tiberkevich_batch(...).plt"
    _interactive_nodes = frozenset(
        {"power_vs_current", "linewidth_vs_current", "frequency_vs_current"}
    )
    _interactive_examples = {
        "power_vs_current": [
            "job[0].vortex.nonlinear.slavin_tiberkevich_batch(...).plt.power_vs_current()"
        ],
        "linewidth_vs_current": [
            "job[0].vortex.nonlinear.slavin_tiberkevich_batch(...).plt.linewidth_vs_current()"
        ],
        "frequency_vs_current": [
            "job[0].vortex.nonlinear.slavin_tiberkevich_batch(...).plt.frequency_vs_current()"
        ],
    }

    def __init__(self, result: STBatchResult):
        self._result = result

    def power_vs_current(self, *, ax=None, **kwargs):
        """Plot generation power as function of current."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        ax.plot(self._result.currents, self._result.powers, marker="o", **plot_kwargs)
        ax.set_xlabel("Current [A]")
        ax.set_ylabel("Generation power p_gen [a.u.]")
        ax.set_title("Power vs current")
        apply_axes_style(ax, style_kwargs)
        return ax

    def linewidth_vs_current(self, *, ax=None, as_mhz: bool = True, **kwargs):
        """Plot linewidth versus current."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        values = np.asarray(self._result.linewidths, dtype=float)
        ylabel = "Linewidth [Hz]"
        if as_mhz:
            values = values * 1e-6
            ylabel = "Linewidth [MHz]"

        ax.plot(self._result.currents, values, marker="o", **plot_kwargs)
        ax.set_xlabel("Current [A]")
        ax.set_ylabel(ylabel)
        ax.set_title("Linewidth vs current")
        apply_axes_style(ax, style_kwargs)
        return ax

    def frequency_vs_current(self, *, ax=None, as_ghz: bool = True, **kwargs):
        """Plot dominant gyration frequency versus current."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        values = np.asarray(self._result.frequencies_hz, dtype=float)
        ylabel = "Frequency [Hz]"
        if as_ghz:
            values = values * 1e-9
            ylabel = "Frequency [GHz]"

        ax.plot(self._result.currents, values, marker="o", **plot_kwargs)
        ax.set_xlabel("Current [A]")
        ax.set_ylabel(ylabel)
        ax.set_title("Frequency vs current")
        apply_axes_style(ax, style_kwargs)
        return ax

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html

        return plot_accessor_html(
            "STBatchPlotAccessor",
            [
                (
                    ".power_vs_current()",
                    "Generation power p_gen vs current I",
                    "Plots full sweep. Accepts matplotlib kwargs.",
                ),
                (
                    ".linewidth_vs_current(as_mhz=True)",
                    "Linewidth Δf vs current I",
                    "as_mhz: convert to MHz.",
                ),
                (
                    ".frequency_vs_current(as_ghz=True)",
                    "Dominant frequency f₀ vs current I",
                    "as_ghz: convert to GHz.",
                ),
            ],
        )


@dataclass
class ThieleForceBalanceResult:
    """Force decomposition from the Thiele equation on a tracked trajectory."""

    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    gyro_force: np.ndarray
    conservative_force: np.ndarray
    dissipative_force: np.ndarray
    stt_force: np.ndarray
    oersted_force: np.ndarray
    residual_force: np.ndarray
    G: float
    D: float
    kappa: float
    polarity: int
    vorticity: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def residual_norm(self) -> np.ndarray:
        """Residual-force norm over time."""
        return np.linalg.norm(np.asarray(self.residual_force, dtype=float), axis=1)

    @property
    def gyro_norm(self) -> np.ndarray:
        """Gyro-force norm over time."""
        return np.linalg.norm(np.asarray(self.gyro_force, dtype=float), axis=1)

    @property
    def residual_ratio(self) -> np.ndarray:
        """Point-wise residual ratio ``|F_res|/|F_gyro|``."""
        gyro = self.gyro_norm
        return self.residual_norm / np.clip(gyro, 1e-30, None)

    @property
    def plt(self) -> ThieleForcePlotAccessor:
        """Plotting accessor."""
        return ThieleForcePlotAccessor(self)

    def _repr_html_(self) -> str:
        import uuid as _uuid

        from mmpp._repr_helpers import (
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        return node_card_html(
            "Thiele Force Balance Result",
            icon="⚖️",
            subtitle="Force decomposition from the Thiele equation on a tracked vortex trajectory.",
            sections=[
                metrics_section_html(
                    [
                        ("n_samples", int(np.asarray(self.time).size), None),
                        ("G", f"{float(self.G):.6g}", None),
                        ("D", f"{float(self.D):.6g}", None),
                        ("kappa", f"{float(self.kappa):.6g}", None),
                        (
                            "residual_mean",
                            f"{float(np.mean(self.residual_norm)):.6g}",
                            None,
                        ),
                    ]
                ),
                examples_section_html(
                    "fb = jobs[-1].solitons.vortex.nonlinear.force_balance()\n"
                    "fb.residual_ratio\n"
                    "fb.plt.force_balance()",
                    title="Result Usage",
                ),
            ],
            api=api_help_html(
                self,
                title="Thiele force-balance API help",
                prefix="jobs[-1].solitons.vortex.nonlinear.force_balance()",
                properties=[
                    ("time", "Time axis"),
                    ("gyro_force", "Gyro force vectors"),
                    ("conservative_force", "Conservative force vectors"),
                    ("dissipative_force", "Dissipative force vectors"),
                    ("stt_force", "Spin-torque force vectors"),
                    ("oersted_force", "Oersted force vectors"),
                    ("residual_force", "Residual force vectors"),
                    ("residual_norm", "Residual norm over time"),
                    ("residual_ratio", "Residual-to-gyro ratio"),
                    ("plt", "Plotting accessor"),
                ],
                subtitle="Live attributes of the Thiele force decomposition.",
                chrome=False,
            ),
            uid=f"thiele-force-balance-result-{str(_uuid.uuid4())[:8]}",
        )


class ThieleForcePlotAccessor(InteractiveNodeMixin):
    """Plot helpers for :class:`ThieleForceBalanceResult`."""

    _interactive_owner = "job[0].vortex.nonlinear.force_balance().plt"
    _interactive_nodes = frozenset({"force_balance"})
    _interactive_examples = {
        "force_balance": ["job[0].vortex.nonlinear.force_balance().plt.force_balance()"]
    }

    def __init__(self, result: ThieleForceBalanceResult):
        self._result = result

    def force_balance(self, *, ax=None, as_norm: bool = True, **kwargs):
        """Plot force decomposition over time."""
        plot_kwargs = dict(kwargs)
        style_kwargs = pop_axes_style_kwargs(plot_kwargs)
        figure_kwargs = pop_figure_kwargs(plot_kwargs)
        ax = ensure_axis(ax, figure_kwargs=figure_kwargs)

        t = np.asarray(self._result.time, dtype=float)
        shared_kwargs = {k: v for k, v in plot_kwargs.items() if k != "label"}
        if as_norm:
            ax.plot(
                t,
                np.linalg.norm(self._result.gyro_force, axis=1),
                label="|F_gyro|",
                **shared_kwargs,
            )
            ax.plot(
                t,
                np.linalg.norm(self._result.conservative_force, axis=1),
                label="|F_cons|",
                linestyle="--",
                **{k: v for k, v in shared_kwargs.items() if k != "linestyle"},
            )
            ax.plot(
                t,
                np.linalg.norm(self._result.dissipative_force, axis=1),
                label="|F_diss|",
                linestyle=":",
                **{k: v for k, v in shared_kwargs.items() if k != "linestyle"},
            )
            ax.plot(
                t,
                np.linalg.norm(self._result.residual_force, axis=1),
                label="|F_res|",
                linewidth=1.2,
                **{k: v for k, v in shared_kwargs.items() if k != "linewidth"},
            )
            ax.set_ylabel("Force norm [a.u.]")
        else:
            ax.plot(t, self._result.gyro_force[:, 0], label="F_gyro,x", **shared_kwargs)
            ax.plot(
                t,
                self._result.gyro_force[:, 1],
                label="F_gyro,y",
                linestyle="--",
                **{k: v for k, v in shared_kwargs.items() if k != "linestyle"},
            )
            ax.plot(
                t,
                self._result.residual_force[:, 0],
                label="F_res,x",
                linestyle=":",
                **{k: v for k, v in shared_kwargs.items() if k != "linestyle"},
            )
            ax.plot(
                t,
                self._result.residual_force[:, 1],
                label="F_res,y",
                linestyle="-.",
                **{k: v for k, v in shared_kwargs.items() if k != "linestyle"},
            )
            ax.set_ylabel("Force component [a.u.]")

        ax.set_xlabel("Time [s]")
        ax.set_title("Thiele force balance")
        ax.legend()
        apply_axes_style(ax, style_kwargs)
        return ax

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html

        return plot_accessor_html(
            "ThieleForcePlotAccessor",
            [
                (
                    ".force_balance(as_norm=True)",
                    "Force decomposition (gyro, conservative, dissipative, residual) vs time",
                    "as_norm: True for |F| norms, False for x/y components.",
                ),
            ],
        )
