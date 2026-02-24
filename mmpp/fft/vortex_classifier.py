"""Deprecated compatibility wrapper for legacy vortex mode classification.

This module preserves the historical ``AdvancedVortexClassifier`` API while
internally using a lightweight, maintained implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import warnings

import numpy as np


def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _circular_mean(angles: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    if angles.size == 0:
        return float("nan"), 0.0
    if weights is None:
        vec = np.mean(np.exp(1j * angles))
    else:
        w = np.asarray(weights, dtype=float)
        if np.sum(w) <= 0:
            return float("nan"), 0.0
        vec = np.sum(w * np.exp(1j * angles)) / np.sum(w)
    return float(np.angle(vec)), float(np.abs(vec))


@dataclass
class VortexClassificationConfig:
    """Legacy configuration container kept for backward compatibility."""

    tol_phi_quadrature: float = 0.5
    eta_parallel_for_gyr: float = 0.6
    min_core_radius: float = 0.01

    eta_perp_for_breath: float = 0.6
    std_phi_mz_for_breath: float = 0.5

    ring_thickness_factor: float = 0.04
    nbins_radial: int = 96

    node_amplitude_threshold: float = 0.25
    smoothing_kernel_size: int = 3


@dataclass
class VortexModeResult:
    """Legacy vortex-mode result model retained for old callers."""

    m_index: int
    n_index: int
    l_index: int | None = None

    mode_type: str = "azimuthal"
    rotation_sense: str = "CW"
    confidence: float = 0.0

    core_position: tuple[float, float] = (0.0, 0.0)
    r_star: float = 0.0
    frequency: float = 0.0

    E_parallel: float = 0.0
    E_perp: float = 0.0
    E_parallel_frac: float = 0.0

    delta_phi_xy: float = 0.0
    dist_to_quadrature: float = 0.0
    std_phi_mz_on_ring: float = 0.0
    phase_coherence_xy: float = 0.0

    core_orbit_radius: float = 0.0
    gyration_frequency: float | None = None

    radial_nodes: list[float] = field(default_factory=list)
    analysis_radius: float = 0.0
    notes: list[str] = field(default_factory=list)


class AdvancedVortexClassifier:
    """Backward-compatible classifier facade.

    The original monolithic implementation has been retired; this class now
    computes robust low-cost descriptors required by legacy code paths.
    """

    def __init__(self, config: VortexClassificationConfig | None = None):
        warnings.warn(
            "AdvancedVortexClassifier is deprecated; use "
            "job.m.vortex.modes.classify_all() from solitons.vortex.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.config = config or VortexClassificationConfig()

    @staticmethod
    def _estimate_core_position(dmz: np.ndarray) -> tuple[float, float]:
        idx = int(np.argmax(np.abs(dmz)))
        y, x = np.unravel_index(idx, dmz.shape)
        return float(x), float(y)

    @staticmethod
    def _radial_profile(values: np.ndarray, center: tuple[float, float], bins: int) -> tuple[np.ndarray, np.ndarray]:
        ny, nx = values.shape
        yy, xx = np.indices((ny, nx))
        cx, cy = center
        radii = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        edges = np.linspace(0.0, float(np.max(radii)), max(int(bins), 16) + 1)
        ids = np.clip(np.digitize(radii.ravel(), edges) - 1, 0, edges.size - 2)
        sums = np.bincount(ids, weights=values.ravel(), minlength=edges.size - 1)
        counts = np.bincount(ids, minlength=edges.size - 1)
        with np.errstate(invalid="ignore"):
            profile = np.divide(sums, counts, where=counts > 0)
        profile[counts == 0] = np.nan
        centers = 0.5 * (edges[:-1] + edges[1:])
        return np.asarray(centers, dtype=float), np.asarray(profile, dtype=float)

    @staticmethod
    def _estimate_winding(dmx: np.ndarray, dmy: np.ndarray, center: tuple[float, float], radius: float, width: float) -> tuple[int, float]:
        ny, nx = dmx.shape
        yy, xx = np.indices((ny, nx))
        cx, cy = center
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        phi = np.arctan2(yy - cy, xx - cx)
        mask = np.abs(r - float(radius)) <= max(float(width), 1.0)
        if int(np.sum(mask)) < 16:
            return 0, 0.0

        complex_field = np.real(dmx[mask]) + 1j * np.real(dmy[mask])
        phase = np.angle(complex_field)
        order = np.argsort(phi[mask])
        phi_sorted = phi[mask][order]
        phase_sorted = np.unwrap(phase[order])
        dphi = float(phi_sorted[-1] - phi_sorted[0])
        if abs(dphi) < 1e-9:
            return 0, 0.0
        winding = int(np.round((phase_sorted[-1] - phase_sorted[0]) / (2.0 * np.pi)))

        trend = np.polyfit(phi_sorted, phase_sorted, 1)
        residual = phase_sorted - np.polyval(trend, phi_sorted)
        quality = float(np.clip(1.0 / (1.0 + np.std(residual)), 0.0, 1.0))
        return winding, quality

    def classify_mode(
        self,
        mode_data,
        R_dot: float | None = None,
        dx: float | None = None,
        dy: float | None = None,
        dz: float | None = None,
        verbose: bool = False,
    ) -> VortexModeResult:
        """Classify a single FFT mode with legacy-compatible output fields."""
        _ = dz, verbose
        mode_array = np.asarray(getattr(mode_data, "mode_array"))
        if mode_array.ndim != 3 or mode_array.shape[-1] < 3:
            raise ValueError("mode_data.mode_array must have shape (Ny, Nx, 3)")

        ny, nx, _ = mode_array.shape
        frequency = float(getattr(mode_data, "frequency", 0.0))
        metadata = dict(getattr(mode_data, "metadata", {}) or {})
        if dx is None or dy is None:
            dx_meta, dy_meta = metadata.get("spatial_resolution", (1.0, 1.0))
            dx = float(dx if dx is not None else dx_meta)
            dy = float(dy if dy is not None else dy_meta)
        assert dx is not None and dy is not None

        if R_dot is None:
            R_dot = 0.5 * min(float(nx) * float(dx), float(ny) * float(dy))
        R_dot = max(float(R_dot), 1e-30)

        dmx = np.asarray(mode_array[:, :, 0], dtype=np.complex128)
        dmy = np.asarray(mode_array[:, :, 1], dtype=np.complex128)
        dmz = np.asarray(mode_array[:, :, 2], dtype=np.complex128)

        core = self._estimate_core_position(np.real(dmz))
        cx, cy = core

        e_par = float(np.sum(np.abs(dmx) ** 2 + np.abs(dmy) ** 2))
        e_perp = float(np.sum(np.abs(dmz) ** 2))
        e_total = max(e_par + e_perp, 1e-30)
        e_par_frac = e_par / e_total

        total_amp = np.sqrt(np.abs(dmx) ** 2 + np.abs(dmy) ** 2 + np.abs(dmz) ** 2)
        radii_px, radial_profile = self._radial_profile(total_amp, core, bins=self.config.nbins_radial)
        if np.all(np.isnan(radial_profile)):
            r_star_px = 0.35 * min(nx, ny)
            radial_nodes = []
        else:
            max_idx = int(np.nanargmax(radial_profile))
            r_star_px = float(radii_px[max_idx])
            profile_norm = radial_profile / max(float(np.nanmax(radial_profile)), 1e-30)
            minima = np.where(
                (profile_norm[1:-1] < profile_norm[:-2])
                & (profile_norm[1:-1] < profile_norm[2:])
                & (profile_norm[1:-1] < (1.0 - float(self.config.node_amplitude_threshold)))
            )[0] + 1
            radial_nodes = [float(radii_px[i]) for i in minima]

        width_px = max(float(self.config.ring_thickness_factor) * min(nx, ny), 1.0)
        m_idx, m_quality = self._estimate_winding(
            dmx,
            dmy,
            core,
            radius=r_star_px,
            width=width_px,
        )
        rotation = "CCW" if m_idx >= 0 else "CW"

        in_plane_amp = np.sqrt(np.abs(dmx) ** 2 + np.abs(dmy) ** 2)
        amp_thr = float(np.max(in_plane_amp)) * 0.1 if in_plane_amp.size else 0.0
        mask_ip = in_plane_amp >= amp_thr
        if np.any(mask_ip):
            phase_diff = _wrap_to_pi(np.angle(dmy[mask_ip]) - np.angle(dmx[mask_ip]))
            delta_phi_xy, phase_coh = _circular_mean(phase_diff, weights=in_plane_amp[mask_ip])
        else:
            delta_phi_xy, phase_coh = float("nan"), 0.0

        yy, xx = np.indices((ny, nx))
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        ring_mask = np.abs(r - r_star_px) <= width_px
        if np.any(ring_mask):
            phi_mz = np.angle(dmz[ring_mask])
            std_phi_mz = float(np.nanstd(_wrap_to_pi(phi_mz - np.nanmean(phi_mz))))
        else:
            std_phi_mz = float("inf")

        if np.isfinite(delta_phi_xy):
            dist_quad = min(
                abs(float(_wrap_to_pi(np.array([delta_phi_xy - np.pi / 2]))[0])),
                abs(float(_wrap_to_pi(np.array([delta_phi_xy + np.pi / 2]))[0])),
            )
        else:
            dist_quad = float("inf")

        perp_frac = e_perp / e_total
        notes: list[str] = []
        if abs(m_idx) == 1 and e_par_frac >= self.config.eta_parallel_for_gyr and dist_quad <= self.config.tol_phi_quadrature:
            mode_type = "gyration"
        elif perp_frac >= self.config.eta_perp_for_breath and std_phi_mz <= self.config.std_phi_mz_for_breath:
            mode_type = "breathing"
        else:
            mode_type = "azimuthal"
            if abs(m_idx) == 0:
                notes.append("winding_undetermined")

        conf = float(
            np.clip(
                0.35 * min(max(e_par_frac, perp_frac), 1.0)
                + 0.35 * float(phase_coh)
                + 0.30 * float(m_quality),
                0.0,
                1.0,
            )
        )

        r_star_m = float(r_star_px) * float(np.hypot(dx, dy)) / np.sqrt(2.0)
        core_orbit = float(np.clip(r_star_m / R_dot, 0.0, 2.0))

        return VortexModeResult(
            m_index=int(m_idx),
            n_index=int(len(radial_nodes)),
            mode_type=str(mode_type),
            rotation_sense=str(rotation),
            confidence=float(conf),
            core_position=(float(cx), float(cy)),
            r_star=float(r_star_m),
            frequency=float(frequency),
            E_parallel=float(e_par),
            E_perp=float(e_perp),
            E_parallel_frac=float(e_par_frac),
            delta_phi_xy=float(delta_phi_xy) if np.isfinite(delta_phi_xy) else 0.0,
            dist_to_quadrature=float(dist_quad) if np.isfinite(dist_quad) else float("inf"),
            std_phi_mz_on_ring=float(std_phi_mz) if np.isfinite(std_phi_mz) else float("inf"),
            phase_coherence_xy=float(phase_coh),
            core_orbit_radius=float(core_orbit),
            gyration_frequency=float(frequency) if mode_type == "gyration" else None,
            radial_nodes=list(radial_nodes),
            analysis_radius=float(r_star_m),
            notes=notes,
        )


__all__ = [
    "VortexClassificationConfig",
    "VortexModeResult",
    "AdvancedVortexClassifier",
]
