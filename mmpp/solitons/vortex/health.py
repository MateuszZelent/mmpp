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

    is_healthy: bool | None
    polarity_flipped: bool
    annihilated: bool
    boundary_collision: bool
    mz_initial: float
    mz_final: float
    min_wall_distance_frac: float | None = None
    warnings: list[str] = field(default_factory=list)
    excluded: bool = False
    status: str = "unavailable"

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
            if ax is None:
                return

            msg = " | ".join(self.warnings)
            ax.annotate(
                f"⚠ {msg}",
                xy=(0.01, 0.99),
                xycoords="axes fraction",
                fontsize=8,
                va="top",
                ha="left",
                color=color,
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "#1a1a1a",
                    "edgecolor": color,
                    "alpha": 0.85,
                },
            )
        except Exception:  # never crash the plot
            pass

    def _repr_html_(self) -> str:
        from html import escape as _esc

        color = "#22c55e" if self.is_healthy is True else "#f97316"
        label = self.status.upper().replace("_", " ")
        rows = [
            ("status", label),
            ("polarity_flipped", str(self.polarity_flipped)),
            ("annihilated", str(self.annihilated)),
            ("boundary_collision", str(self.boundary_collision)),
            ("mz_initial", f"{self.mz_initial:.4f}"),
            ("mz_final", f"{self.mz_final:.4f}"),
        ]
        if self.min_wall_distance_frac is not None:
            rows.append(
                (
                    "min_wall_distance",
                    f"{self.min_wall_distance_frac:.3f} R",
                )
            )
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
            '<div style="font-family:-apple-system,sans-serif;'
            "border:1px solid #334155;border-radius:8px;padding:12px;"
            'background:#0f172a;color:#e2e8f0;">'
            f"<div style='font-weight:600;color:{_esc(color)};margin-bottom:6px;'>"
            f"Core Health: {label}</div>"
            "<table style='border-collapse:collapse;'>"
            f"{row_html}</table>"
            f"{warn_html}"
            "</div>"
        )

    def __repr__(self) -> str:  # noqa: D105
        status = (
            "HEALTHY"
            if self.is_healthy is True
            else "UNHEALTHY"
            if self.is_healthy is False
            else str(self.status).upper()
        )
        return (
            f"CoreHealthStatus({status}, "
            f"annihilated={self.annihilated}, "
            f"polarity_flipped={self.polarity_flipped}, "
            f"boundary_collision={self.boundary_collision})"
        )


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _frame_mz(data: np.ndarray, frame_idx: int) -> np.ndarray | None:
    """Return one out-of-plane frame from an MMPP vector-field array."""
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 5:
        frame = arr[frame_idx, arr.shape[1] // 2]
    elif arr.ndim == 4:
        frame = arr[frame_idx]
    elif arr.ndim == 3:
        frame = arr
    else:
        return None
    if frame.ndim != 3 or frame.shape[-1] < 3:
        return None
    return np.asarray(frame[..., 2], dtype=float)


def _local_core_mz(
    data: np.ndarray,
    frame_idx: int,
    x_m: float,
    y_m: float,
    *,
    dx: float,
    dy: float,
    y_axis: str,
    radius_pixels: int,
) -> tuple[float, float]:
    """Return signed and absolute local ``m_z`` around a tracked core point."""
    mz = _frame_mz(data, frame_idx)
    if mz is None or not np.isfinite(mz).all():
        return float("nan"), float("nan")
    ny, nx = mz.shape
    xi = int(round(float(x_m) / dx))
    yi = int(round(float(y_m) / dy))
    if y_axis == "up":
        yi = ny - 1 - yi
    if not (0 <= xi < nx and 0 <= yi < ny):
        return float("nan"), float("nan")
    x0, x1 = max(0, xi - radius_pixels), min(nx, xi + radius_pixels + 1)
    y0, y1 = max(0, yi - radius_pixels), min(ny, yi + radius_pixels + 1)
    patch = mz[y0:y1, x0:x1]
    if patch.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(patch)), float(np.mean(np.abs(patch)))


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
    disk_center: tuple[float, float] | None = None,
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
    for name, value, lower, upper in (
        ("mz_annihilation_threshold", mz_annihilation_threshold, 0.0, 1.0),
        ("boundary_fraction", boundary_fraction, 0.0, 1.0),
        ("core_fraction", core_fraction, 0.0, 1.0),
    ):
        if not np.isfinite(value) or not lower < value <= upper:
            raise ValueError(f"{name} must be finite and in ({lower}, {upper}]")
    if disk_center is not None:
        if len(disk_center) != 2 or not np.isfinite(disk_center).all():
            raise ValueError("disk_center must contain two finite coordinates")

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

    # ---- measure local mz at the tracked endpoints ---------------------
    mz_initial = float("nan")
    mz_final = float("nan")
    abs_mz_final = float("nan")
    local_state_available = False
    tx = np.asarray(getattr(trajectory, "x", []), dtype=float).reshape(-1)
    ty = np.asarray(getattr(trajectory, "y", []), dtype=float).reshape(-1)
    if (
        data is not None
        and data.ndim in {4, 5}
        and tx.size == ty.size == data.shape[0]
        and tx.size >= 2
    ):
        attrs = getattr(job_result, "attrs", {}) or {}
        try:
            dx = float(attrs.get("dx", attrs.get("cellsize_x")))
            dy = float(attrs.get("dy", attrs.get("cellsize_y", dx)))
        except (TypeError, ValueError):
            dx = dy = float("nan")
        if np.isfinite(dx) and np.isfinite(dy) and dx > 0.0 and dy > 0.0:
            metadata = getattr(trajectory, "metadata", {}) or {}
            y_axis = str(metadata.get("y_axis", "up")).lower()
            if y_axis not in {"up", "down"}:
                y_axis = "up"
            radius_px = int(
                np.clip(round(min(data.shape[-3:-1]) * float(core_fraction) / 2), 1, 4)
            )
            mz_initial, _ = _local_core_mz(
                data,
                0,
                tx[0],
                ty[0],
                dx=dx,
                dy=dy,
                y_axis=y_axis,
                radius_pixels=radius_px,
            )
            mz_final, abs_mz_final = _local_core_mz(
                data,
                -1,
                tx[-1],
                ty[-1],
                dx=dx,
                dy=dy,
                y_axis=y_axis,
                radius_pixels=radius_px,
            )
            local_state_available = bool(
                np.isfinite(mz_initial) and np.isfinite(mz_final)
            )

    # ---- classify problems --------------------------------------------
    polarity_flipped = False
    annihilated = False
    boundary_collision = False
    min_wall_frac: float | None = None
    warn_msgs: list[str] = []

    if local_state_available:
        if abs_mz_final < mz_annihilation_threshold:
            annihilated = True
            warn_msgs.append(
                "No localized out-of-plane core signal at the tracked final position: "
                f"mean(|mz|)={abs_mz_final:.3f} < {mz_annihilation_threshold}"
            )
        elif np.sign(mz_initial) != np.sign(mz_final) and mz_initial != 0.0:
            polarity_flipped = True
            warn_msgs.append(
                f"Polarity reversed: mz {mz_initial:+.3f} → {mz_final:+.3f} "
                "(core re-magnetisation)"
            )

    # ---- boundary collision via trajectory ----------------------------
    if trajectory is not None and tx.size == ty.size and tx.size > 0:
        try:
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

            if (
                R is not None
                and np.isfinite(R)
                and R > 0.0
                and np.isfinite(tx).all()
                and np.isfinite(ty).all()
            ):
                if disk_center is not None:
                    cx, cy = map(float, disk_center)
                else:
                    attrs = getattr(job_result, "attrs", {}) or {}
                    if data is not None and data.ndim in {4, 5}:
                        ny, nx = data.shape[-3:-1]
                    else:
                        ny = nx = 1
                    center_x = attrs.get("center_x")
                    center_y = attrs.get("center_y")
                    dx = attrs.get("dx", attrs.get("cellsize_x"))
                    dy = attrs.get("dy", attrs.get("cellsize_y", dx))
                    if center_x is None or center_y is None:
                        if dx is None or dy is None or nx < 2 or ny < 2:
                            raise ValueError(
                                "Cannot infer disk center without physical grid spacing"
                            )
                        dx_value, dy_value = float(dx), float(dy)
                        if (
                            not np.isfinite(dx_value)
                            or not np.isfinite(dy_value)
                            or dx_value <= 0.0
                            or dy_value <= 0.0
                        ):
                            raise ValueError("Grid spacing must be finite and positive")
                        if center_x is None:
                            center_x = (nx - 1) * dx_value / 2.0
                        if center_y is None:
                            center_y = (ny - 1) * dy_value / 2.0
                    cx, cy = float(center_x), float(center_y)
                    if not np.isfinite([cx, cy]).all():
                        raise ValueError("Disk center must be finite")
                frac = _min_wall_distance(tx, ty, R, cx, cy)
                if np.isfinite(frac):
                    min_wall_frac = frac
                if frac < (1.0 - boundary_fraction):
                    boundary_collision = True
                    r_max_nm = (R - frac * R) * 1e9
                    warn_msgs.append(
                        f"Boundary collision: core reached {r_max_nm:.1f} nm "
                        f"from disk edge ({frac * 100:.1f}% R left)"
                    )
        except (TypeError, ValueError, OverflowError):
            min_wall_frac = None

    issues_detected = polarity_flipped or annihilated or boundary_collision
    has_boundary_measurement = min_wall_frac is not None
    complete = local_state_available and has_boundary_measurement
    is_healthy = False if issues_detected else (True if complete else None)
    if issues_detected:
        status_label = "unhealthy"
    elif complete:
        status_label = "healthy"
    elif local_state_available or has_boundary_measurement:
        status_label = "partial"
        warn_msgs.append(
            "Core health is partial: local texture and boundary status could not "
            "both be assessed."
        )
    else:
        status_label = "unavailable"
        warn_msgs.append(
            "Core health is unavailable: a time-aligned tracked core, local field "
            "data, and physical cell spacing are required."
        )
    return CoreHealthStatus(
        is_healthy=is_healthy,
        polarity_flipped=polarity_flipped,
        annihilated=annihilated,
        boundary_collision=boundary_collision,
        mz_initial=mz_initial,
        mz_final=mz_final,
        min_wall_distance_frac=min_wall_frac,
        warnings=warn_msgs,
        status=status_label,
    )
