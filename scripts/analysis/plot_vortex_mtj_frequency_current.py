"""Plot the reduced MTJ/CPP vortex frequency and orbit versus total current.

The analytical model consumes current density ``J`` in A/m².  This helper
converts a total circular-pillar current ``I`` to ``J = I/(pi R²)``, samples
both signed core-polarity branches, and adds a dense grid around each signed
threshold.  Solid curves are inside the configured rigid-orbit boundary;
dashed curves are edge-clamped reduced-model values.

Example
-------
python scripts/analysis/plot_vortex_mtj_frequency_current.py \
    --output /tmp/vortex_mtj_frequency_current.png \
    --csv /tmp/vortex_mtj_frequency_current.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# When invoked as ``python scripts/analysis/...py``, Python puts the script
# directory first on ``sys.path``.  Prefer this checkout over an older globally
# installed mmpp package so that the plotted contract is reproducible.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _build_adapter(
    *,
    polarity: int,
    radius_m: float,
    thickness_m: float,
    saturation_a_per_m: float,
    alpha: float,
    polarization: float,
    exchange_j_per_m: float,
    nonlinear_shift: float,
    Lambda: float,
    epsilonprime: float,
):
    from mmpp.analytical import DiskGeometry, MaterialParams, omega0_novosad
    from mmpp.solitons.vortex.model.thiele import cpp

    material = MaterialParams(
        Ms=saturation_a_per_m,
        alpha=alpha,
        P=polarization,
        A=exchange_j_per_m,
    )
    geometry = DiskGeometry(R=radius_m, L=thickness_m)
    omega0 = omega0_novosad(material, geometry)
    return cpp(
        material=material,
        geom=geometry,
        omega0=omega0,
        N=nonlinear_shift,
        polarity=polarity,
        torque_thickness=thickness_m,
        polarizer=(0.0, 0.0, 1.0),
        fixed_layer_position="top",
        Lambda=Lambda,
        epsilonprime=epsilonprime,
        mean_m_dot_p=0.0,
    )


def _current_grid(
    i_min_mA: float,
    i_max_mA: float,
    thresholds_mA: list[float],
    *,
    coarse_points: int,
    dense_points: int,
    dense_half_width_mA: float,
) -> np.ndarray:
    """Build a monotonic current grid with extra points around thresholds."""
    pieces = [np.linspace(i_min_mA, i_max_mA, max(int(coarse_points), 3))]
    half_width = abs(float(dense_half_width_mA))
    for threshold in thresholds_mA:
        lower = max(i_min_mA, threshold - half_width)
        upper = min(i_max_mA, threshold + half_width)
        if upper > lower:
            pieces.append(np.linspace(lower, upper, max(int(dense_points), 3)))
    return np.unique(np.concatenate(pieces))


def _evaluate(adapter, currents_mA: np.ndarray) -> list[dict[str, float | int | str]]:
    model = adapter.model
    area_m2 = math.pi * float(model.geom.R) ** 2
    rows: list[dict[str, float | int | str]] = []
    for current_mA in currents_mA:
        current_a = float(current_mA) * 1e-3
        current_density = current_a / area_m2
        u = model.steady_state_u(current_density, allow_edge=False)
        frequency_hz = (
            None
            if u is None
            else model.omega(float(u), current_density) / (2.0 * math.pi)
        )
        edge_frequency_hz = model.predict_frequency_dc(
            current_density,
            allow_edge=True,
        )
        if frequency_hz is not None:
            regime = "steady_orbit"
        elif edge_frequency_hz is not None:
            regime = "edge_limited"
        else:
            regime = "below_threshold_or_invalid"
        rows.append(
            {
                "polarity": int(model.polarity),
                "current_mA": float(current_mA),
                "current_density_A_per_m2": float(current_density),
                "u": float(u) if u is not None else float("nan"),
                "radius_nm": (
                    float(u) * float(model.geom.R) * 1e9
                    if u is not None
                    else float("nan")
                ),
                "frequency_GHz": (
                    float(frequency_hz) * 1e-9
                    if frequency_hz is not None
                    else float("nan")
                ),
                "edge_frequency_GHz": (
                    float(edge_frequency_hz) * 1e-9
                    if edge_frequency_hz is not None
                    else float("nan")
                ),
                "regime": regime,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "polarity",
        "current_mA",
        "current_density_A_per_m2",
        "u",
        "radius_nm",
        "frequency_GHz",
        "edge_frequency_GHz",
        "regime",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    rows_by_polarity: dict[int, list[dict[str, float | int | str]]],
    thresholds_mA: dict[int, float],
    *,
    output: Path,
    radius_nm: float,
    thickness_nm: float,
    nonlinear_shift: float,
    Lambda: float,
    epsilonprime: float,
    i_min_mA: float,
    i_max_mA: float,
) -> None:
    colors = {-1: "#1769aa", 1: "#d95f02"}
    labels = {-1: r"core polarity $p=-1$", 1: r"core polarity $p=+1$"}

    figure, (frequency_axis, radius_axis) = plt.subplots(
        2,
        1,
        figsize=(11.5, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": (1.45, 1.0), "hspace": 0.08},
    )
    for polarity, rows in rows_by_polarity.items():
        color = colors[polarity]
        current = np.asarray([float(row["current_mA"]) for row in rows])
        frequency = np.asarray([float(row["frequency_GHz"]) for row in rows])
        edge_frequency = np.asarray([float(row["edge_frequency_GHz"]) for row in rows])
        radius = np.asarray([float(row["radius_nm"]) for row in rows])
        valid = np.isfinite(frequency)
        edge_only = np.isfinite(edge_frequency) & ~valid

        frequency_axis.plot(
            current[valid],
            frequency[valid],
            color=color,
            linewidth=2.1,
            label=labels[polarity],
        )
        frequency_axis.plot(
            current[edge_only],
            edge_frequency[edge_only],
            color=color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.72,
            label=f"{labels[polarity]} edge-limited",
        )
        radius_axis.plot(
            current[valid],
            radius[valid],
            color=color,
            linewidth=2.0,
            label=labels[polarity],
        )

        threshold = thresholds_mA[polarity]
        if i_min_mA <= threshold <= i_max_mA:
            frequency_axis.axvline(
                threshold,
                color=color,
                linestyle=":",
                linewidth=1.25,
                alpha=0.9,
            )
            frequency_axis.annotate(
                rf"$I_{{th}}={threshold:.3f}$ mA",
                xy=(threshold, 0.02),
                xycoords=("data", "axes fraction"),
                xytext=(5 if polarity < 0 else -5, 0),
                textcoords="offset points",
                rotation=90,
                ha="left" if polarity < 0 else "right",
                va="bottom",
                color=color,
                fontsize=9,
            )

    frequency_axis.set_ylabel("Gyrotropic frequency $f$ (GHz)")
    frequency_axis.set_title(
        "MTJ/CPP vortex gyrotropic frequency versus total current\n"
        f"circular pillar R={radius_nm:g} nm, free layer L={thickness_nm:g} nm; "
        f"N={nonlinear_shift:g}, $\\Lambda$={Lambda:g}, $\\epsilon'$={epsilonprime:g}"
    )
    frequency_axis.grid(True, which="both", alpha=0.24)
    frequency_axis.legend(loc="upper right", fontsize=9, ncol=2)
    frequency_axis.text(
        0.012,
        0.97,
        "solid: rigid steady orbit   dashed: edge-clamped reduced-model value",
        transform=frequency_axis.transAxes,
        va="top",
        fontsize=9,
        color="#444444",
    )

    radius_axis.set_xlabel("Total MTJ current $I$ (mA)")
    radius_axis.set_ylabel("Orbit radius $r_0$ (nm)")
    radius_axis.grid(True, which="both", alpha=0.24)
    radius_axis.set_xlim(i_min_mA, i_max_mA)
    radius_axis.legend(loc="best", fontsize=9)

    finite_thresholds = [
        value for value in thresholds_mA.values() if i_min_mA <= value <= i_max_mA
    ]
    if finite_thresholds:
        zoom_half_width = max(0.55, 2.0 * abs(finite_thresholds[0]))
        zoom_left = max(i_min_mA, min(finite_thresholds) - zoom_half_width)
        zoom_right = min(i_max_mA, max(finite_thresholds) + zoom_half_width)
        inset = inset_axes(
            frequency_axis,
            width="39%",
            height="47%",
            loc="lower right",
            borderpad=1.4,
        )
        for polarity, rows in rows_by_polarity.items():
            color = colors[polarity]
            current = np.asarray([float(row["current_mA"]) for row in rows])
            frequency = np.asarray([float(row["frequency_GHz"]) for row in rows])
            valid = np.isfinite(frequency)
            inset.plot(current[valid], frequency[valid], color=color, linewidth=1.5)
            threshold = thresholds_mA[polarity]
            if zoom_left <= threshold <= zoom_right:
                inset.axvline(threshold, color=color, linestyle=":", linewidth=0.9)
        inset.set_xlim(zoom_left, zoom_right)
        inset.grid(True, alpha=0.2)
        inset.set_title("threshold detail", fontsize=8)
        inset.tick_params(labelsize=7)

    figure.text(
        0.01,
        0.01,
        "J = I/A, A = pi R²; J in A/m². The Novosad thin-disk omega0 estimate "
        "is illustrative; calibrate omega0, torque efficiency and Oersted shift "
        "against MuMax3/experiment for quantitative use.",
        fontsize=8.5,
        color="#444444",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/vortex_mtj_frequency_current.png")
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--i-min-mA", type=float, default=-10.0)
    parser.add_argument("--i-max-mA", type=float, default=10.0)
    parser.add_argument("--dense-half-width-mA", type=float, default=0.25)
    parser.add_argument("--coarse-points", type=int, default=401)
    parser.add_argument("--dense-points", type=int, default=801)
    parser.add_argument("--radius-nm", type=float, default=128.0)
    parser.add_argument("--thickness-nm", type=float, default=9.0)
    parser.add_argument("--Ms", type=float, default=8.0e5)
    parser.add_argument("--alpha", type=float, default=0.013)
    parser.add_argument("--P", type=float, default=0.45)
    parser.add_argument("--A-pJ-per-m", type=float, default=10.0)
    parser.add_argument("--N", type=float, default=0.30)
    parser.add_argument("--Lambda", type=float, default=1.0)
    parser.add_argument("--epsilonprime", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.i_min_mA) or not np.isfinite(args.i_max_mA):
        raise ValueError("current limits must be finite")
    if args.i_max_mA <= args.i_min_mA:
        raise ValueError("--i-max-mA must be greater than --i-min-mA")

    radius_m = float(args.radius_nm) * 1e-9
    thickness_m = float(args.thickness_nm) * 1e-9
    adapters = {
        polarity: _build_adapter(
            polarity=polarity,
            radius_m=radius_m,
            thickness_m=thickness_m,
            saturation_a_per_m=float(args.Ms),
            alpha=float(args.alpha),
            polarization=float(args.P),
            exchange_j_per_m=float(args.A_pJ_per_m) * 1e-12,
            nonlinear_shift=float(args.N),
            Lambda=float(args.Lambda),
            epsilonprime=float(args.epsilonprime),
        )
        for polarity in (-1, 1)
    }
    area_m2 = math.pi * radius_m**2
    thresholds_mA = {
        polarity: float(adapter.model.J_threshold) * area_m2 * 1e3
        for polarity, adapter in adapters.items()
    }
    currents_mA = _current_grid(
        float(args.i_min_mA),
        float(args.i_max_mA),
        list(thresholds_mA.values()),
        coarse_points=int(args.coarse_points),
        dense_points=int(args.dense_points),
        dense_half_width_mA=float(args.dense_half_width_mA),
    )
    rows_by_polarity = {
        polarity: _evaluate(adapter, currents_mA)
        for polarity, adapter in adapters.items()
    }
    all_rows = [row for rows in rows_by_polarity.values() for row in rows]
    if args.csv is not None:
        _write_csv(args.csv, all_rows)
    _plot(
        rows_by_polarity,
        thresholds_mA,
        output=args.output,
        radius_nm=float(args.radius_nm),
        thickness_nm=float(args.thickness_nm),
        nonlinear_shift=float(args.N),
        Lambda=float(args.Lambda),
        epsilonprime=float(args.epsilonprime),
        i_min_mA=float(args.i_min_mA),
        i_max_mA=float(args.i_max_mA),
    )

    print(f"saved plot: {args.output}")
    if args.csv is not None:
        print(f"saved data: {args.csv}")
    print(f"samples per polarity: {currents_mA.size}")
    for polarity in (-1, 1):
        valid_count = sum(
            row["regime"] == "steady_orbit" for row in rows_by_polarity[polarity]
        )
        print(
            f"p={polarity:+d}: I_threshold={thresholds_mA[polarity]:.6g} mA, "
            f"steady points={valid_count}"
        )


if __name__ == "__main__":
    main()
