"""Vortex core health checks — annihilation and boundary collision detection.

When too strong a current is applied the vortex core can be:

* **Expelled** — pushed to the disk edge (boundary collision).  The core then
  annihilates with the spin-wave halo and the system re-magnetises uniformly.
* **Reversed (polarity flip)** — the core polarity switches under strong
  out-of-plane STT, yielding a damped final state.

Both pathologies manifest as a change in the sign (or magnitude → 0) of the
averaged ``m_z`` component at the core between the first and last frame of the
simulation.

Public API
----------
>>> status = check_core_health(job_result, dataset_name="m")
>>> if status.is_healthy:
...     ...
>>> # or from VortexInterface:
>>> status = vortex.check_health()
>>> status.warn_on_plot(ax)   # attach annotation to a matplotlib Axes
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class CoreHealthStatus:
    """Describes detected vortex core health at the end of a simulation.

    Attributes
    ----------
    is_healthy : bool
        ``True`` if neither annihilation nor polarity reversal was detected.
    polarity_flipped : bool
        ``True`` if the sign of averaged ``m_z`` reversed between first and
        last frames.
    annihilated : bool
        ``True`` if averaged ``|m_z|`` at the core dropped below
        ``mz_annihilation_threshold`` in the last frame.
    boundary_collision : bool
        ``True`` if the trajectory came within ``boundary_fraction`` of the
        estimated disk edge at any point.
    mz_initial : float
        Average ``m_z`` at the core in the first frame.
    mz_final : float
        Average ``m_z`` at the core in the last frame.
    min_wall_distance_frac : float | None
        Minimum distance to the boundary expressed as a fraction of the disk
        radius (1.0 = edge).  ``None`` when trajectory is not available.
    warnings : list[str]
        Human-readable warning strings (empty when healthy).
    excluded : bool
        Set to ``True`` by the caller when the ``exclude_annihilated`` flag
        was passed.  Read-only marker used by downstream code.
    """

    is_healthy: bool
    polarity_flipped: bool
    annihilated: bool
    boundary_collision: bool
    mz_initial: float
    mz_final: float
    min_wall_distance_frac: float | None = None
    warnings: list[str] = field(default_factory=list)
    excluded: bool = False

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def issue_python_warnings(self) -> None:
        """Emit Python :mod:`warnings` for each detected problem."""
        for msg in self.warnings:
            warnings.warn(msg, UserWarning, stacklevel=3)

    def warn_on_plot(self, ax_or_fig, *, color: str = "#f97316") -> None:
        """Attach an annotation to a matplotlib Axes or Figure.

        Parameters
        ----------
        ax_or_fig : matplotlib Axes or Figure
            Target to annotate.  If a Figure is passed the first axes is used.
        color : str
            Text/border colour for the annotation (default: orange).
        """
        if not self.warnings:
            return
        try:
            import matplotlib.pyplot as plt  # noqa: F401
            from matplotlib.figure import Figure

            if isinstance(ax_or_fig, Figure):
                ax = ax_or_fig.axes[0] if ax_or_fig.axes else None
                if ax is None:
                    return
            else:
                ax = ax_or_fig

            msg = " | ".join(self.warnings)
            ax.annotate(
                f"⚠ {msg}",
                xy=(0.01, 0.99),
                xycoords="axes fraction",
                fontsize=8,
                va="top",
                ha="left",
                color=color,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="#1a1a1a",
                    edgecolor=color,
                    alpha=0.85,
                ),
            )
        except Exception:  # never crash the plot
            pass

    def _repr_html_(self) -> str:
        from html import escape as _esc

        color = "#22c55e" if self.is_healthy else "#f97316"
        label = "HEALTHY" if self.is_healthy else "ISSUES DETECTED"
        rows = [
            ("polarity_flipped", str(self.polarity_flipped)),
            ("annihilated", str(self.annihilated)),
            ("boundary_collision", str(self.boundary_collision)),
            ("mz_initial", f"{self.mz_initial:.4f}"),
            ("mz_final", f"{self.mz_final:.4f}"),
        ]
        if self.min_wall_distance_frac is not None:
            rows.append((
                "min_wall_distance",
                f"{self.min_wall_distance_frac:.3f} R",
            ))
        if self.excluded:
            rows.append(("excluded", "True (annihilation excluded by user flag)"))

        warn_html = ""
        if self.warnings:
            warn_html = (
                "<div style='background:rgba(249,115,22,0.15);border:1px solid #f97316;"
                "border-radius:6px;padding:8px;margin-top:8px;font-size:0.85em;"
                "color:#fdba74;font-family:monospace;'>"
                + "<br>".join(_esc(w) for w in self.warnings)
                + "</div>"
            )

        row_html = "".join(
            f"<tr><td style='padding:3px 8px;font-family:monospace;color:#93c5fd;'>"
            f"{_esc(k)}</td>"
            f"<td style='padding:3px 8px;color:#e2e8f0;'>{_esc(v)}</td></tr>"
            for k, v in rows
        )
        return (
            "<div style=\"font-family:-apple-system,sans-serif;"
            "border:1px solid #334155;border-radius:8px;padding:12px;"
            "background:#0f172a;color:#e2e8f0;\">"
            f"<div style='font-weight:600;color:{_esc(color)};margin-bottom:6px;'>"
            f"Core Health: {label}</div>"
            "<table style='border-collapse:collapse;'>"
            f"{row_html}</table>"
            f"{warn_html}"
            "</div>"
        )

    def __repr__(self) -> str:  # noqa: D105
        status = "HEALTHY" if self.is_healthy else "UNHEALTHY"
        return (
            f"CoreHealthStatus({status}, "
            f"annihilated={self.annihilated}, "
            f"polarity_flipped={self.polarity_flipped}, "
            f"boundary_collision={self.boundary_collision})"
        )


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _average_core_mz(data: np.ndarray, frame_idx: int, core_fraction: float) -> float:
    """Return mean ``m_z`` in the central ``core_fraction`` of the disk.

    Parameters
    ----------
    data : ndarray, shape (Nt, Ny, Nx, 3) or (Nt, Nz, Ny, Nx, 3)
        Full magnetisation array.
    frame_idx : int
        Time-step index (0 = first, -1 = last).
    core_fraction : float
        Fraction of the grid radius considered the "core region".
    """
    arr = np.asarray(data, dtype=float)

    # Normalise to (Ny, Nx, 3)
    if arr.ndim == 5:
        # (Nt, Nz, Ny, Nx, 3) → pick middle z layer
        frame = arr[frame_idx, arr.shape[1] // 2, ...]
    elif arr.ndim == 4:
        frame = arr[frame_idx]
    elif arr.ndim == 3:
        frame = arr
    else:
        return float("nan")

    mz = frame[..., 2]  # (Ny, Nx)
    ny, nx = mz.shape

    # Build a circular mask centred on the grid
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    r_max = min(cy, cx) * core_fraction
    yy, xx = np.mgrid[0:ny, 0:nx]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = dist <= r_max

    if not np.any(mask):
        return float(np.mean(mz))

    return float(np.mean(mz[mask]))


def _min_wall_distance(
    trajectory_x: np.ndarray,
    trajectory_y: np.ndarray,
    disk_radius: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
) -> float:
    """Return the minimum (core-position → disk-edge) distance as fraction of R."""
    x = np.asarray(trajectory_x, dtype=float)
    y = np.asarray(trajectory_y, dtype=float)
    r_core = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    min_wall = float(disk_radius) - float(np.max(r_core))
    return min_wall / float(disk_radius)  # > 0 inside, < 0 outside


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def check_core_health(
    job_result,
    dataset_name: str | None = None,
    *,
    trajectory=None,
    disk_radius: float | None = None,
    mz_annihilation_threshold: float = 0.05,
    boundary_fraction: float = 0.85,
    core_fraction: float = 0.25,
    slice_info: Any | None = None,
) -> CoreHealthStatus:
    """Detect vortex core annihilation and boundary collision.

    Parameters
    ----------
    job_result :
        An MMPP job result object (``job[i]``).
    dataset_name : str or None
        Magnetisation dataset name (auto-resolved when ``None``).
    trajectory : TrajectoryResult or None
        Pre-computed trajectory used for boundary-collision detection.
        When ``None`` the check is skipped.
    disk_radius : float or None
        Physical disk radius in metres.  Auto-inferred from ``job_result.attrs``
        when ``None``.
    mz_annihilation_threshold : float
        |mz_final| < this value → annihilation detected (default 0.05).
    boundary_fraction : float
        Core-to-edge distance / R < (1 - boundary_fraction) triggers boundary
        collision warning (default 0.85 → warns when core > 85% of R).
    core_fraction : float
        Fraction of the grid used to average ``m_z`` for the health check
        (default 0.25 → central 25% radius).
    slice_info :
        Optional zarr slice passed when reading the dataset.

    Returns
    -------
    CoreHealthStatus
    """
    # ---- resolve dataset -----------------------------------------------
    if dataset_name is None:
        try:
            dataset_name = job_result.get_largest_m_dataset()
        except Exception:
            dataset_name = "m"

    # ---- load magnetisation data ---------------------------------------
    data: np.ndarray | None = None
    try:
        dset = getattr(job_result, dataset_name)
        if slice_info is not None:
            dset = dset[slice_info]
        data = np.asarray(dset.numpy(copy=False), dtype=float)
    except Exception:
        pass

    # ---- compute mz at start / end ------------------------------------
    mz_initial = float("nan")
    mz_final = float("nan")
    if data is not None and data.ndim >= 3:
        mz_initial = _average_core_mz(data, 0, core_fraction)
        mz_final = _average_core_mz(data, -1, core_fraction)

    # ---- classify problems --------------------------------------------
    polarity_flipped = False
    annihilated = False
    boundary_collision = False
    min_wall_frac: float | None = None
    warn_msgs: list[str] = []

    if np.isfinite(mz_initial) and np.isfinite(mz_final):
        if abs(mz_final) < mz_annihilation_threshold:
            annihilated = True
            warn_msgs.append(
                f"Core annihilated: |mz_final|={abs(mz_final):.3f} < {mz_annihilation_threshold}"
            )
        elif np.sign(mz_initial) != np.sign(mz_final) and mz_initial != 0.0:
            polarity_flipped = True
            warn_msgs.append(
                f"Polarity reversed: mz {mz_initial:+.3f} → {mz_final:+.3f} "
                "(core re-magnetisation)"
            )

    # ---- boundary collision via trajectory ----------------------------
    if trajectory is not None:
        try:
            tx = np.asarray(trajectory.x, dtype=float)
            ty = np.asarray(trajectory.y, dtype=float)

            # Resolve disk radius
            R = disk_radius
            if R is None or not np.isfinite(R) or R <= 0.0:
                attrs = getattr(job_result, "attrs", {}) or {}
                for key in ("R", "radius"):
                    val = attrs.get(key)
                    if val is not None:
                        try:
                            R = float(val)
                            break
                        except Exception:
                            pass
                if R is None:
                    for key in ("D", "diameter"):
                        val = attrs.get(key)
                        if val is not None:
                            try:
                                R = float(val) / 2.0
                                break
                            except Exception:
                                pass

            if R is not None and np.isfinite(R) and R > 0.0:
                cx = float(np.mean(tx))
                cy = float(np.mean(ty))
                frac = _min_wall_distance(tx, ty, R, cx, cy)
                min_wall_frac = frac
                if frac < (1.0 - boundary_fraction):
                    boundary_collision = True
                    r_max_nm = (R - frac * R) * 1e9
                    warn_msgs.append(
                        f"Boundary collision: core reached {r_max_nm:.1f} nm "
                        f"from disk edge ({frac*100:.1f}% R left)"
                    )
        except Exception:
            pass

    is_healthy = not (polarity_flipped or annihilated or boundary_collision)
    return CoreHealthStatus(
        is_healthy=is_healthy,
        polarity_flipped=polarity_flipped,
        annihilated=annihilated,
        boundary_collision=boundary_collision,
        mz_initial=mz_initial,
        mz_final=mz_final,
        min_wall_distance_frac=min_wall_frac,
        warnings=warn_msgs,
    )
