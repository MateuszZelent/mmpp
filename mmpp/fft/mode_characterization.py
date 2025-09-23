from __future__ import annotations

"""Utilities for automated characterisation of FMR modes."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..cli.logging_config import get_mmpp_logger

# Import advanced vortex classifier
try:
    from .vortex_classifier import AdvancedVortexClassifier, VortexClassificationConfig, VortexModeResult
    VORTEX_CLASSIFIER_AVAILABLE = True
except ImportError:
    VORTEX_CLASSIFIER_AVAILABLE = False
    log.debug("Advanced vortex classifier not available")

if TYPE_CHECKING:  # pragma: no cover - only used for typing
    from .modes import FMRModeData

log = get_mmpp_logger("mmpp.fft.mode_characterization")


def _wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    """Wrap angles to the interval [-pi, pi]."""
    return (angles + np.pi) % (2 * np.pi) - np.pi


def _circular_stats(angles: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Return circular mean and concentration (0-1)."""
    if angles.size == 0 or np.all(weights <= 0):
        return float("nan"), 0.0
    complex_vec = np.sum(weights * np.exp(1j * angles))
    total_weight = np.sum(weights)
    if total_weight == 0:
        return float("nan"), 0.0
    mean_angle = float(np.angle(complex_vec))
    concentration = float(np.abs(complex_vec) / total_weight)
    return mean_angle, concentration


def _radial_profile(
    values: np.ndarray,
    center: tuple[float, float],
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute azimuthally averaged radial profile."""
    ny, nx = values.shape
    y_idx, x_idx = np.indices((ny, nx))
    cx, cy = center
    radii = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)

    max_radius = radii.max()
    if max_radius == 0:
        return np.array([0.0]), np.array([values.mean()])

    bin_edges = np.linspace(0, max_radius, bins + 1)
    bin_indices = np.digitize(radii.ravel(), bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, bins - 1)

    values_flat = values.ravel()
    sums = np.bincount(bin_indices, weights=values_flat, minlength=bins)
    counts = np.bincount(bin_indices, minlength=bins)

    with np.errstate(invalid="ignore"):
        radial_mean = np.divide(sums, counts, where=counts > 0)

    radial_mean[counts == 0] = np.nan

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, radial_mean


@dataclass(slots=True)
class ModeCharacteristicConfig:
    """Configuration for automatic mode characterisation."""

    relative_amplitude_threshold: float = 0.1
    min_ring_coverage: float = 0.65
    ring_width_fraction: float = 0.08
    radial_bins: int = 48
    quadrature_tolerance: float = 0.55  # radians
    breathing_phase_uniformity: float = 0.65
    gyration_parallel_ratio: float = 0.55
    breathing_perp_ratio: float = 0.5
    anisotropy_ratio: float = 0.25
    
    # Advanced vortex classifier settings
    use_vortex_classifier: bool = False  # Enable advanced vortex analysis
    vortex_dot_radius: Optional[float] = None  # Auto-estimate if None
    min_points_for_winding: int = 64
    default_radius_fraction: float = 0.35
    min_radius_fraction: float = 0.12

    def __post_init__(self) -> None:
        if not 0 < self.relative_amplitude_threshold < 1:
            raise ValueError("relative_amplitude_threshold should be in (0, 1)")
        if not 0 < self.min_ring_coverage <= 1:
            raise ValueError("min_ring_coverage should be in (0, 1]")
        if not 0 < self.ring_width_fraction <= 0.5:
            raise ValueError("ring_width_fraction should be in (0, 0.5]")
        if self.radial_bins < 8:
            raise ValueError("radial_bins must be >= 8")
        if self.min_points_for_winding < 16:
            raise ValueError("min_points_for_winding must be >= 16")
        if not 0 < self.min_radius_fraction < self.default_radius_fraction < 1:
            raise ValueError("radius fractions must satisfy 0 < min < default < 1")


@dataclass(slots=True)
class ModeCharacterizationResult:
    """Results produced by :class:`ModeCharacterAnalyzer`."""

    frequency: Optional[float]
    m_index: Optional[int]
    m_quality: float
    rotation_sense: Optional[str]
    radial_nodes: int
    energy_parallel: float
    energy_perp: float
    dominant_component: str
    phase_xy_mean: Optional[float]
    phase_xy_coherence: float
    phase_z_uniformity: float
    primary_class: str
    labels: list[str] = field(default_factory=list)
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""
        return {
            "frequency": self.frequency,
            "m_index": self.m_index,
            "m_quality": self.m_quality,
            "rotation_sense": self.rotation_sense,
            "radial_nodes": self.radial_nodes,
            "energy_parallel": self.energy_parallel,
            "energy_perp": self.energy_perp,
            "dominant_component": self.dominant_component,
            "phase_xy_mean": self.phase_xy_mean,
            "phase_xy_coherence": self.phase_xy_coherence,
            "phase_z_uniformity": self.phase_z_uniformity,
            "primary_class": self.primary_class,
            "labels": list(self.labels),
            "confidence": self.confidence,
            "notes": list(self.notes),
            "diagnostics": dict(self.diagnostics),
        }


class ModeCharacterAnalyzer:
    """Analyse spatial mode maps to extract gyro/breathing/azimuthal labels."""

    def __init__(self, config: Optional[ModeCharacteristicConfig] = None) -> None:
        self.config = config or ModeCharacteristicConfig()

    def analyze(
        self,
        mode: "FMRModeData",
        *,
        core_position: Optional[tuple[float, float]] = None,
        analysis_radius: Optional[float] = None,
    ) -> ModeCharacterizationResult:
        """Return classification for a given :class:`FMRModeData` instance."""

        mode_array = mode.mode_array
        ny, nx, _ = mode_array.shape

        dx, dy = mode.metadata.get("spatial_resolution", (1.0, 1.0))
        area_element = float(dx) * float(dy)

        # Determine centre of the texture
        center = (
            core_position
            or mode.metadata.get("core_position_px")
            or mode.metadata.get("core_position")
        )
        if center is None:
            center = self._estimate_core_position(mode_array)
            notes = ["core_position estimated from |mz|"]
        else:
            notes = []
        cx, cy = center

        # Energy partitions
        energy_x = np.sum(np.abs(mode_array[:, :, 0]) ** 2) * area_element
        energy_y = np.sum(np.abs(mode_array[:, :, 1]) ** 2) * area_element
        energy_z = np.sum(np.abs(mode_array[:, :, 2]) ** 2) * area_element
        energy_parallel = float(energy_x + energy_y)
        energy_perp = float(energy_z)
        total_energy = max(energy_parallel + energy_perp, 1e-30)
        parallel_ratio = energy_parallel / total_energy
        perp_ratio = energy_perp / total_energy

        dominant_component = "in_plane" if parallel_ratio >= perp_ratio else "out_of_plane"

        # Phase relations between mx and my
        mx = mode_array[:, :, 0]
        my = mode_array[:, :, 1]
        in_plane_amp = np.sqrt(np.abs(mx) ** 2 + np.abs(my) ** 2)
        amp_threshold = self.config.relative_amplitude_threshold * np.nanmax(in_plane_amp)
        ip_mask = in_plane_amp >= amp_threshold

        if np.any(ip_mask):
            phase_diff = _wrap_to_pi(np.angle(my[ip_mask]) - np.angle(mx[ip_mask]))
            weights = in_plane_amp[ip_mask]
            phase_xy_mean, phase_xy_coherence = _circular_stats(phase_diff, weights)
        else:
            phase_xy_mean, phase_xy_coherence = float("nan"), 0.0
            notes.append("insufficient in-plane amplitude for robust phase analysis")

        # mz phase uniformity
        mz = mode_array[:, :, 2]
        mz_amp = np.abs(mz)
        mz_threshold = self.config.relative_amplitude_threshold * np.nanmax(mz_amp)
        mz_mask = mz_amp >= mz_threshold
        if np.any(mz_mask):
            phase_z = np.angle(mz[mz_mask])
            phase_z_uniformity = _circular_stats(phase_z, mz_amp[mz_mask])[1]
        else:
            phase_z_uniformity = 0.0

        # Radial analysis
        total_magnitude = np.sqrt(np.sum(np.abs(mode_array) ** 2, axis=2))
        radii, radial_profile = _radial_profile(total_magnitude, (cx, cy), self.config.radial_bins)

        if np.all(np.isnan(radial_profile)):
            radial_nodes = 0
        else:
            norm_profile = radial_profile / np.nanmax(radial_profile)
            minima_mask = (
                (norm_profile[1:-1] < norm_profile[:-2])
                & (norm_profile[1:-1] < norm_profile[2:])
                & (norm_profile[1:-1] < 0.3)
            )
            radial_nodes = int(np.sum(minima_mask))

        # Set analysis radius if not provided
        if analysis_radius is None:
            size_scale = min(nx, ny) / 2.0
            fallback_radius = self.config.default_radius_fraction * size_scale
            min_radius = self.config.min_radius_fraction * size_scale

            if np.all(np.isnan(radial_profile)):
                radius_candidate = fallback_radius
            else:
                max_idx = int(np.nanargmax(radial_profile))
                radius_candidate = float(radii[max_idx])

            if radius_candidate < min_radius:
                analysis_radius = fallback_radius
            else:
                analysis_radius = radius_candidate

        m_index, m_quality, coverage = self._estimate_winding_number(
            mx,
            my,
            (cx, cy),
            analysis_radius,
            in_plane_amp,
        )

        if coverage < self.config.min_ring_coverage:
            notes.append(
                f"ring coverage below threshold ({coverage:.2f}) – winding number less reliable"
            )

        rotation = None
        if m_index is not None:
            if m_index > 0:
                rotation = "CCW"
            elif m_index < 0:
                rotation = "CW"

        labels = [f"m={m_index}" if m_index is not None else "m=undetermined", f"n={radial_nodes}"]
        if rotation:
            labels.append(rotation)
        labels.append(f"parallel={parallel_ratio:.2f}")
        labels.append(f"perp={perp_ratio:.2f}")

        result = ModeCharacterizationResult(
            frequency=mode.frequency,
            m_index=m_index,
            m_quality=m_quality,
            rotation_sense=rotation,
            radial_nodes=radial_nodes,
            energy_parallel=energy_parallel,
            energy_perp=energy_perp,
            dominant_component=dominant_component,
            phase_xy_mean=phase_xy_mean,
            phase_xy_coherence=phase_xy_coherence,
            phase_z_uniformity=phase_z_uniformity,
            primary_class="unclassified",
            labels=labels,
            notes=notes,
            diagnostics={
                "ring_coverage": coverage,
                "analysis_radius": analysis_radius,
                "parallel_ratio": parallel_ratio,
                "perp_ratio": perp_ratio,
            },
        )

        self._assign_classification(result)
        return result

    def analyze_vortex(
        self,
        mode: "FMRModeData",
        *,
        core_position: Optional[tuple[float, float]] = None,
        R_dot: Optional[float] = None,
        verbose: bool = False,
    ) -> "VortexModeResult":
        """
        Analyze mode using advanced vortex/skyrmion classifier.
        
        Parameters:
        -----------
        mode : FMRModeData
            Mode data to analyze
        core_position : tuple, optional
            Core position in pixels. If None, estimated automatically
        R_dot : float, optional  
            Dot radius. If None, taken from config or estimated
        verbose : bool
            Print detailed analysis
            
        Returns:
        --------
        VortexModeResult
            Advanced classification result with indices and physics
        """
        
        if not VORTEX_CLASSIFIER_AVAILABLE:
            raise ImportError("Advanced vortex classifier not available. Check vortex_classifier.py")
        
        # Create vortex classifier with compatible config
        vortex_config = VortexClassificationConfig(
            tol_phi_quadrature=self.config.quadrature_tolerance,
            eta_parallel_for_gyr=self.config.gyration_parallel_ratio,
            eta_perp_for_breath=self.config.breathing_perp_ratio,
            std_phi_mz_for_breath=self.config.breathing_phase_uniformity,
            nbins_radial=max(48, self.config.radial_bins),
        )
        
        classifier = AdvancedVortexClassifier(vortex_config)
        
        # Use provided R_dot or get from config
        dot_radius = R_dot or self.config.vortex_dot_radius
        
        # Get spatial resolution
        dx, dy = mode.metadata.get("spatial_resolution", (1.0, 1.0))
        
        # Run advanced vortex analysis
        result = classifier.classify_mode(
            mode, 
            R_dot=dot_radius,
            dx=dx, 
            dy=dy,
            verbose=verbose
        )
        
        return result

    @staticmethod
    def _estimate_core_position(mode_array: np.ndarray) -> tuple[float, float]:
        """Estimate vortex/skyrmion core position from |mz| map."""
        mz_amp = np.abs(mode_array[:, :, 2])
        if np.all(mz_amp == 0):
            total = np.sqrt(np.sum(np.abs(mode_array) ** 2, axis=2))
            flat_idx = int(np.argmax(total))
        else:
            flat_idx = int(np.argmax(mz_amp))
        ny, nx, _ = mode_array.shape
        y_idx, x_idx = np.unravel_index(flat_idx, (ny, nx))
        return float(x_idx), float(y_idx)

    def _estimate_winding_number(
        self,
        mx: np.ndarray,
        my: np.ndarray,
        center: tuple[float, float],
        radius: float,
        in_plane_amp: np.ndarray,
    ) -> tuple[Optional[int], float, float]:
        """Estimate azimuthal winding of mx + i my."""
        ny, nx = mx.shape
        y_idx, x_idx = np.indices((ny, nx))
        cx, cy = center
        radii = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)

        ring_mask = np.abs(radii - radius) <= max(radius * self.config.ring_width_fraction, 1.0)
        amp_threshold = self.config.relative_amplitude_threshold * np.nanmax(in_plane_amp)
        ring_mask &= in_plane_amp >= amp_threshold

        coverage = 0.0
        if np.any(ring_mask):
            phi = np.arctan2(y_idx[ring_mask] - cy, x_idx[ring_mask] - cx)
            weights = in_plane_amp[ring_mask]
            coverage = (np.max(phi) - np.min(phi)) / (2 * np.pi)

            if ring_mask.sum() < self.config.min_points_for_winding or coverage < 0.1:
                return None, 0.0, coverage

            order = np.argsort(phi)
            phi_sorted = phi[order]
            mx_real = np.real(mx)
            my_real = np.real(my)
            complex_field = mx_real[ring_mask] + 1j * my_real[ring_mask]
            phase = np.angle(complex_field)[order]
            phase_unwrapped = np.unwrap(phase)

            try:
                coeffs = np.polyfit(phi_sorted, phase_unwrapped, 1, w=weights[order])
                slope, intercept = coeffs
            except Exception as exc:  # pragma: no cover - numerical failures are rare
                log.debug("polyfit failed for winding number: %s", exc)
                return None, 0.0, coverage

            residuals = phase_unwrapped - (slope * phi_sorted + intercept)
            with np.errstate(invalid="ignore"):
                scatter = float(np.sqrt(np.nanmean(residuals**2)))

            m_float = slope / (1.0)
            m_index = int(np.round(m_float))
            quality = float(max(0.0, 1.0 - scatter / np.pi))
            return m_index, quality, float(coverage)

        return None, 0.0, coverage

    def _assign_classification(self, result: ModeCharacterizationResult) -> None:
        """Fill in classification, labels and confidence based on metrics."""
        parallel_ratio = result.diagnostics["parallel_ratio"]
        perp_ratio = result.diagnostics["perp_ratio"]

        gyration_score = 0.0
        breathing_score = 0.0
        azimuthal_score = 0.0

        if result.phase_xy_mean is not None and not np.isnan(result.phase_xy_mean):
            deviation = min(
                abs(_wrap_to_pi(result.phase_xy_mean - np.pi / 2)),
                abs(_wrap_to_pi(result.phase_xy_mean + np.pi / 2)),
            )
        else:
            deviation = np.pi

        # Scores for key mode families
        if parallel_ratio >= self.config.gyration_parallel_ratio:
            gyration_score += 0.4
        gyration_score += 0.4 * result.phase_xy_coherence
        gyration_score += max(0.0, 0.2 - deviation / np.pi)
        if result.m_index is not None and abs(result.m_index) == 1:
            gyration_score += 0.2

        if perp_ratio >= self.config.breathing_perp_ratio:
            breathing_score += 0.4
        breathing_score += 0.4 * result.phase_z_uniformity
        if result.m_index is None or abs(result.m_index) == 0:
            breathing_score += 0.2
        if result.radial_nodes == 0:
            breathing_score += 0.1

        if result.m_index is not None and abs(result.m_index) >= 1:
            azimuthal_score += 0.4 + 0.1 * min(abs(result.m_index), 4)
            azimuthal_score += 0.3 * result.m_quality
            azimuthal_score += 0.2 * (1.0 - abs(perp_ratio - parallel_ratio))
        if result.radial_nodes >= 1:
            azimuthal_score += 0.1

        scores = {
            "gyration": gyration_score,
            "breathing": breathing_score,
            "azimuthal": azimuthal_score,
        }

        primary = max(scores, key=scores.get)
        confidence = min(1.0, scores[primary])

        # Additional refinements
        if primary == "azimuthal" and result.radial_nodes > 0:
            result.labels.append("radial-structure")
        if primary == "azimuthal" and result.m_index is not None and abs(result.m_index) > 1:
            result.labels.append("|m|>1")
        if primary == "breathing" and result.radial_nodes >= 1:
            result.labels.append("breathing-overtone")
        if primary == "gyration" and result.rotation_sense:
            result.labels.append(f"{result.rotation_sense.lower()}-rotation")

        result.primary_class = primary
        result.confidence = confidence
        result.diagnostics["scores"] = scores

        if confidence < 0.3:
            result.notes.append("classification confidence low; consider manual inspection")
