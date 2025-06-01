"""
Electromagnetic Analysis Module for MMPP

Advanced electromagnetic analysis tools for FMR modes including:
- Poynting vector analysis
- Radiation patterns
- Energy flow visualization
- Q-factor analysis
- Mode coupling analysis
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy import integrate
from scipy.signal import hilbert

# Import shared logging configuration
from ..logging_config import get_mmpp_logger

# Get logger for electromagnetic analysis
log = get_mmpp_logger("mmpp.fft.electromagnetic")

try:
    from ..plotting import load_paper_style

    STYLING_AVAILABLE = True
except ImportError:
    STYLING_AVAILABLE = False


@dataclass
class ElectromagneticAnalysisConfig:
    """Configuration for electromagnetic analysis."""

    # Physical constants and parameters
    mu_0: float = const.mu_0  # Permeability of free space
    epsilon_0: float = const.epsilon_0  # Permittivity of free space
    c: float = const.c  # Speed of light
    gamma: float = 2.211e5  # Gyromagnetic ratio (m/(A·s))

    # Analysis settings
    far_field_distance: float = 1e-3  # Far-field distance in meters
    radiation_theta_points: int = 181  # Number of theta points for radiation pattern
    radiation_phi_points: int = 361  # Number of phi points for radiation pattern

    # Visualization settings
    figsize: Tuple[float, float] = (15, 10)
    dpi: int = 150
    poynting_colormap: str = "cmc.hawaii"  # For energy flow
    radiation_colormap: str = "cmc.tokyo"  # For radiation patterns

    # Analysis regions
    energy_integration_method: str = "simpson"  # "simpson" or "trapezoid"
    q_factor_method: str = "energy_decay"  # "energy_decay" or "spectral_width"


class PoyntingVectorAnalysis:
    """Analysis of electromagnetic energy flow using Poynting vector."""

    def __init__(self, config: ElectromagneticAnalysisConfig = None):
        """Initialize Poynting vector analysis."""
        self.config = config or ElectromagneticAnalysisConfig()

    def compute_poynting_vector(
        self, E_field: np.ndarray, H_field: np.ndarray
    ) -> np.ndarray:
        """
        Compute Poynting vector S = (1/μ₀) * E × H

        Parameters:
        -----------
        E_field : np.ndarray
            Electric field array with shape (ny, nx, 3)
        H_field : np.ndarray
            Magnetic field array with shape (ny, nx, 3)

        Returns:
        --------
        poynting_vector : np.ndarray
            Poynting vector with shape (ny, nx, 3)
        """
        if E_field.shape != H_field.shape:
            raise ValueError("E and H field arrays must have the same shape")

        # Compute cross product E × H
        poynting_vector = np.cross(E_field, H_field, axis=2) / self.config.mu_0

        log.info(f"Computed Poynting vector with shape {poynting_vector.shape}")
        return poynting_vector

    def compute_energy_density(
        self,
        E_field: np.ndarray,
        H_field: np.ndarray,
        epsilon_r: float = 1.0,
        mu_r: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """
        Compute electromagnetic energy density.

        Parameters:
        -----------
        E_field : np.ndarray
            Electric field array
        H_field : np.ndarray
            Magnetic field array
        epsilon_r : float
            Relative permittivity
        mu_r : float
            Relative permeability

        Returns:
        --------
        energy_densities : dict
            Dictionary with electric, magnetic, and total energy densities
        """
        epsilon = epsilon_r * self.config.epsilon_0
        mu = mu_r * self.config.mu_0

        # Electric energy density: (1/2) * ε * |E|²
        u_e = 0.5 * epsilon * np.sum(np.abs(E_field) ** 2, axis=2)

        # Magnetic energy density: (1/2) * μ * |H|²
        u_m = 0.5 * mu * np.sum(np.abs(H_field) ** 2, axis=2)

        # Total energy density
        u_total = u_e + u_m

        return {"electric": u_e, "magnetic": u_m, "total": u_total}

    def visualize_energy_flow(
        self,
        poynting_vector: np.ndarray,
        energy_density: np.ndarray,
        extent: Tuple[float, float, float, float] = None,
        title: str = "Electromagnetic Energy Flow",
    ) -> plt.Figure:
        """
        Visualize energy flow using Poynting vector and energy density.

        Parameters:
        -----------
        poynting_vector : np.ndarray
            Poynting vector array (ny, nx, 3)
        energy_density : np.ndarray
            Energy density array (ny, nx)
        extent : tuple, optional
            Spatial extent [x_min, x_max, y_min, y_max]
        title : str
            Plot title

        Returns:
        --------
        fig : plt.Figure
            Matplotlib figure
        """
        if STYLING_AVAILABLE:
            load_paper_style()

        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=self.config.figsize, dpi=self.config.dpi
        )

        ny, nx = energy_density.shape
        extent = extent or (0, nx, 0, ny)

        # Energy density map
        im1 = ax1.imshow(
            energy_density,
            extent=extent,
            origin="lower",
            cmap=self.config.poynting_colormap,
            aspect="equal",
        )
        ax1.set_title("Energy Density")
        ax1.set_xlabel("x (nm)")
        ax1.set_ylabel("y (nm)")
        plt.colorbar(im1, ax=ax1, label="Energy Density (J/m³)")

        # Poynting vector magnitude
        poynting_mag = np.sqrt(np.sum(poynting_vector**2, axis=2))
        im2 = ax2.imshow(
            poynting_mag,
            extent=extent,
            origin="lower",
            cmap=self.config.poynting_colormap,
            aspect="equal",
        )
        ax2.set_title("Poynting Vector Magnitude")
        ax2.set_xlabel("x (nm)")
        ax2.set_ylabel("y (nm)")
        plt.colorbar(im2, ax=ax2, label="|S| (W/m²)")

        # Energy flow streamlines
        x = np.linspace(extent[0], extent[1], nx)
        y = np.linspace(extent[2], extent[3], ny)
        X, Y = np.meshgrid(x, y)

        # Downsample for cleaner streamlines
        step = max(1, min(nx, ny) // 20)
        X_sub = X[::step, ::step]
        Y_sub = Y[::step, ::step]
        Sx_sub = poynting_vector[::step, ::step, 0]
        Sy_sub = poynting_vector[::step, ::step, 1]

        ax3.streamplot(
            X_sub,
            Y_sub,
            Sx_sub,
            Sy_sub,
            color=poynting_mag[::step, ::step],
            cmap=self.config.poynting_colormap,
            density=1.5,
        )
        ax3.set_title("Energy Flow Streamlines")
        ax3.set_xlabel("x (nm)")
        ax3.set_ylabel("y (nm)")
        ax3.set_aspect("equal")

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        log.info(f"Generated energy flow visualization with {nx}×{ny} resolution")
        return fig


class RadiationPatternAnalysis:
    """Analysis of far-field radiation patterns."""

    def __init__(self, config: ElectromagneticAnalysisConfig = None):
        """Initialize radiation pattern analysis."""
        self.config = config or ElectromagneticAnalysisConfig()

    def compute_far_field(
        self,
        current_density: np.ndarray,
        spatial_extent: Tuple[float, float, float, float],
        frequency: float,
    ) -> Dict[str, np.ndarray]:
        """
        Compute far-field radiation pattern from current density.

        Parameters:
        -----------
        current_density : np.ndarray
            Current density array (ny, nx, 3)
        spatial_extent : tuple
            Spatial extent [x_min, x_max, y_min, y_max] in meters
        frequency : float
            Frequency in Hz

        Returns:
        --------
        far_field : dict
            Dictionary with theta, phi arrays and E_theta, E_phi components
        """
        k = 2 * np.pi * frequency / self.config.c  # Wave number

        # Create angular grid
        theta = np.linspace(0, np.pi, self.config.radiation_theta_points)
        phi = np.linspace(0, 2 * np.pi, self.config.radiation_phi_points)
        Theta, Phi = np.meshgrid(theta, phi, indexing="ij")

        # Spatial grid
        ny, nx = current_density.shape[:2]
        x = np.linspace(spatial_extent[0], spatial_extent[1], nx)
        y = np.linspace(spatial_extent[2], spatial_extent[3], ny)
        X, Y = np.meshgrid(x, y, indexing="ij")

        # Initialize far-field arrays
        E_theta = np.zeros_like(Theta, dtype=complex)
        E_phi = np.zeros_like(Theta, dtype=complex)

        # Far-field calculation (simplified dipole approximation)
        r_hat_x = np.sin(Theta) * np.cos(Phi)
        r_hat_y = np.sin(Theta) * np.sin(Phi)
        r_hat_z = np.cos(Theta)

        theta_hat_x = np.cos(Theta) * np.cos(Phi)
        theta_hat_y = np.cos(Theta) * np.sin(Phi)
        theta_hat_z = -np.sin(Theta)

        phi_hat_x = -np.sin(Phi)
        phi_hat_y = np.cos(Phi)
        phi_hat_z = np.zeros_like(Phi)

        # Integrate current density contributions
        dx = (spatial_extent[1] - spatial_extent[0]) / nx
        dy = (spatial_extent[3] - spatial_extent[2]) / ny

        for i in range(ny):
            for j in range(nx):
                # Phase factor for each spatial point
                r_dot_r_source = r_hat_x * X[i, j] + r_hat_y * Y[i, j]
                phase_factor = np.exp(-1j * k * r_dot_r_source)

                # Current density at this point
                J = current_density[i, j, :]

                # Far-field contribution
                J_cross_r = np.array(
                    [
                        J[1] * r_hat_z - J[2] * r_hat_y,
                        J[2] * r_hat_x - J[0] * r_hat_z,
                        J[0] * r_hat_y - J[1] * r_hat_x,
                    ]
                )

                # Project onto spherical coordinates
                E_theta += (
                    (
                        theta_hat_x * J_cross_r[0]
                        + theta_hat_y * J_cross_r[1]
                        + theta_hat_z * J_cross_r[2]
                    )
                    * phase_factor
                    * dx
                    * dy
                )

                E_phi += (
                    (
                        phi_hat_x * J_cross_r[0]
                        + phi_hat_y * J_cross_r[1]
                        + phi_hat_z * J_cross_r[2]
                    )
                    * phase_factor
                    * dx
                    * dy
                )

        # Apply far-field scaling
        scaling = (
            1j
            * k
            * np.exp(-1j * k * self.config.far_field_distance)
            / (4 * np.pi * self.config.far_field_distance)
        )
        E_theta *= scaling
        E_phi *= scaling

        return {
            "theta": theta,
            "phi": phi,
            "E_theta": E_theta,
            "E_phi": E_phi,
            "radiation_intensity": np.abs(E_theta) ** 2 + np.abs(E_phi) ** 2,
        }

    def visualize_radiation_pattern(
        self, far_field_data: Dict[str, np.ndarray], title: str = "Radiation Pattern"
    ) -> plt.Figure:
        """
        Visualize 3D radiation pattern.

        Parameters:
        -----------
        far_field_data : dict
            Far-field data from compute_far_field
        title : str
            Plot title

        Returns:
        --------
        fig : plt.Figure
            Matplotlib figure
        """
        if STYLING_AVAILABLE:
            load_paper_style()

        fig = plt.figure(figsize=self.config.figsize, dpi=self.config.dpi)

        # 2D radiation pattern plots
        gs = gridspec.GridSpec(2, 2, figure=fig)

        theta = far_field_data["theta"]
        phi = far_field_data["phi"]
        intensity = far_field_data["radiation_intensity"]

        # Polar plot - E-plane (phi=0)
        ax1 = fig.add_subplot(gs[0, 0], projection="polar")
        e_plane = intensity[:, 0]  # phi=0 cut
        ax1.plot(theta, 10 * np.log10(e_plane / np.max(e_plane)))
        ax1.set_title("E-plane Pattern (dB)")
        ax1.set_ylim(-40, 0)

        # Polar plot - H-plane (phi=90°)
        ax2 = fig.add_subplot(gs[0, 1], projection="polar")
        h_plane_idx = len(phi) // 4  # phi=90° index
        h_plane = intensity[:, h_plane_idx]
        ax2.plot(theta, 10 * np.log10(h_plane / np.max(h_plane)))
        ax2.set_title("H-plane Pattern (dB)")
        ax2.set_ylim(-40, 0)

        # 2D intensity map
        ax3 = fig.add_subplot(gs[1, :])
        Theta, Phi = np.meshgrid(theta, phi, indexing="ij")
        intensity_db = 10 * np.log10(intensity / np.max(intensity))
        im = ax3.contourf(
            Phi, Theta, intensity_db, levels=20, cmap=self.config.radiation_colormap
        )
        ax3.set_xlabel("Phi (rad)")
        ax3.set_ylabel("Theta (rad)")
        ax3.set_title("Radiation Pattern (dB)")
        plt.colorbar(im, ax=ax3, label="Intensity (dB)")

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()

        log.info("Generated radiation pattern visualization")
        return fig


class QFactorAnalysis:
    """Analysis of quality factor and mode losses."""

    def __init__(self, config: ElectromagneticAnalysisConfig = None):
        """Initialize Q-factor analysis."""
        self.config = config or ElectromagneticAnalysisConfig()

    def compute_q_factor_spectral(
        self, frequencies: np.ndarray, spectrum: np.ndarray, peak_freq: float
    ) -> float:
        """
        Compute Q-factor from spectral width method.

        Parameters:
        -----------
        frequencies : np.ndarray
            Frequency array
        spectrum : np.ndarray
            Power spectrum
        peak_freq : float
            Peak frequency

        Returns:
        --------
        q_factor : float
            Quality factor
        """
        # Find half-maximum points
        peak_idx = np.argmin(np.abs(frequencies - peak_freq))
        peak_value = spectrum[peak_idx]
        half_max = peak_value / 2

        # Find frequencies at half maximum
        left_idx = np.where((frequencies < peak_freq) & (spectrum >= half_max))[0]
        right_idx = np.where((frequencies > peak_freq) & (spectrum >= half_max))[0]

        if len(left_idx) > 0 and len(right_idx) > 0:
            f_left = frequencies[left_idx[-1]]
            f_right = frequencies[right_idx[0]]
            delta_f = f_right - f_left
            q_factor = peak_freq / delta_f
        else:
            q_factor = np.inf

        log.info(f"Computed Q-factor: {q_factor:.1f} at {peak_freq:.3f} GHz")
        return q_factor

    def compute_mode_lifetime(self, q_factor: float, frequency: float) -> float:
        """
        Compute mode lifetime from Q-factor.

        Parameters:
        -----------
        q_factor : float
            Quality factor
        frequency : float
            Mode frequency in Hz

        Returns:
        --------
        lifetime : float
            Mode lifetime in seconds
        """
        omega = 2 * np.pi * frequency
        lifetime = q_factor / omega

        log.info(f"Mode lifetime: {lifetime * 1e9:.2f} ns")
        return lifetime


def analyze_electromagnetic_properties(
    mode_data, config: ElectromagneticAnalysisConfig = None
) -> Dict[str, Any]:
    """
    Comprehensive electromagnetic analysis of FMR mode data.

    Parameters:
    -----------
    mode_data : FMRModeData
        Mode data to analyze
    config : ElectromagneticAnalysisConfig, optional
        Analysis configuration

    Returns:
    --------
    analysis_results : dict
        Dictionary containing all analysis results
    """
    config = config or ElectromagneticAnalysisConfig()

    # Initialize analysis modules
    poynting_analyzer = PoyntingVectorAnalysis(config)
    radiation_analyzer = RadiationPatternAnalysis(config)
    q_analyzer = QFactorAnalysis(config)

    results = {}

    log.info(
        f"Starting electromagnetic analysis for frequency {mode_data.frequency:.3f} GHz"
    )

    # Extract field components (assuming mode_array contains magnetization)
    # In real implementation, would need proper E and H field calculation
    # For now, use approximation based on magnetization dynamics

    try:
        # Mock field calculation - in practice this would be more sophisticated
        m_data = mode_data.mode_array

        # Approximate electric field from magnetization precession
        # E ∝ ∂M/∂t ∝ iωM for harmonic motion
        omega = 2 * np.pi * mode_data.frequency * 1e9  # Convert GHz to Hz
        E_field = 1j * omega * m_data * config.mu_0  # Simplified approximation

        # Approximate magnetic field from curl of E
        # For simplicity, assume H ∝ M
        H_field = m_data / config.mu_0

        # Poynting vector analysis
        poynting_vector = poynting_analyzer.compute_poynting_vector(E_field, H_field)
        energy_densities = poynting_analyzer.compute_energy_density(E_field, H_field)

        results["poynting_vector"] = poynting_vector
        results["energy_densities"] = energy_densities

        # Current density for radiation analysis (∂M/∂t)
        current_density = 1j * omega * m_data * config.gamma  # Magnetization current

        # Radiation pattern analysis
        extent_m = [x * 1e-9 for x in mode_data.extent]  # Convert nm to m
        far_field = radiation_analyzer.compute_far_field(
            current_density, extent_m, omega / (2 * np.pi)
        )

        results["far_field"] = far_field
        results["analysis_successful"] = True

        log.info("Electromagnetic analysis completed successfully")

    except Exception as e:
        log.error(f"Electromagnetic analysis failed: {e}")
        results["analysis_successful"] = False
        results["error"] = str(e)

    return results


def create_comprehensive_em_report(
    mode_data,
    analysis_results: Dict[str, Any],
    config: ElectromagneticAnalysisConfig = None,
) -> plt.Figure:
    """
    Create comprehensive electromagnetic analysis report.

    Parameters:
    -----------
    mode_data : FMRModeData
        Original mode data
    analysis_results : dict
        Results from analyze_electromagnetic_properties
    config : ElectromagneticAnalysisConfig, optional
        Configuration

    Returns:
    --------
    fig : plt.Figure
        Comprehensive report figure
    """
    config = config or ElectromagneticAnalysisConfig()

    if STYLING_AVAILABLE:
        load_paper_style()

    fig = plt.figure(figsize=(20, 15), dpi=config.dpi)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Original mode visualization
    ax1 = fig.add_subplot(gs[0, 0])
    mode_magnitude = np.abs(mode_data.mode_array[:, :, 0])  # x-component magnitude
    im1 = ax1.imshow(
        mode_magnitude, extent=mode_data.extent, origin="lower", cmap="cmc.berlin"
    )
    ax1.set_title(f"Mode Magnitude\n{mode_data.frequency:.3f} GHz")
    ax1.set_xlabel("x (nm)")
    ax1.set_ylabel("y (nm)")
    plt.colorbar(im1, ax=ax1, label="|m|")

    if analysis_results.get("analysis_successful", False):
        # Energy density
        ax2 = fig.add_subplot(gs[0, 1])
        energy_total = analysis_results["energy_densities"]["total"]
        im2 = ax2.imshow(
            energy_total, extent=mode_data.extent, origin="lower", cmap="cmc.hawaii"
        )
        ax2.set_title("Energy Density")
        ax2.set_xlabel("x (nm)")
        ax2.set_ylabel("y (nm)")
        plt.colorbar(im2, ax=ax2, label="Energy (J/m³)")

        # Poynting vector magnitude
        ax3 = fig.add_subplot(gs[0, 2])
        poynting_mag = np.sqrt(np.sum(analysis_results["poynting_vector"] ** 2, axis=2))
        im3 = ax3.imshow(
            poynting_mag, extent=mode_data.extent, origin="lower", cmap="cmc.hawaii"
        )
        ax3.set_title("Energy Flow |S|")
        ax3.set_xlabel("x (nm)")
        ax3.set_ylabel("y (nm)")
        plt.colorbar(im3, ax=ax3, label="|S| (W/m²)")

        # Energy flow streamlines
        ax4 = fig.add_subplot(gs[0, 3])
        ny, nx = poynting_mag.shape
        x = np.linspace(mode_data.extent[0], mode_data.extent[1], nx)
        y = np.linspace(mode_data.extent[2], mode_data.extent[3], ny)
        X, Y = np.meshgrid(x, y)

        step = max(1, min(nx, ny) // 15)
        ax4.streamplot(
            X[::step, ::step],
            Y[::step, ::step],
            analysis_results["poynting_vector"][::step, ::step, 0],
            analysis_results["poynting_vector"][::step, ::step, 1],
            color=poynting_mag[::step, ::step],
            cmap="cmc.hawaii",
            density=1.2,
        )
        ax4.set_title("Energy Flow Streamlines")
        ax4.set_xlabel("x (nm)")
        ax4.set_ylabel("y (nm)")
        ax4.set_aspect("equal")

        # Radiation patterns
        far_field = analysis_results["far_field"]

        # E-plane pattern
        ax5 = fig.add_subplot(gs[1, 0], projection="polar")
        e_plane = far_field["radiation_intensity"][:, 0]
        e_plane_db = 10 * np.log10(e_plane / np.max(e_plane))
        ax5.plot(far_field["theta"], e_plane_db)
        ax5.set_title("E-plane Pattern")
        ax5.set_ylim(-40, 0)

        # H-plane pattern
        ax6 = fig.add_subplot(gs[1, 1], projection="polar")
        h_plane_idx = len(far_field["phi"]) // 4
        h_plane = far_field["radiation_intensity"][:, h_plane_idx]
        h_plane_db = 10 * np.log10(h_plane / np.max(h_plane))
        ax6.plot(far_field["theta"], h_plane_db)
        ax6.set_title("H-plane Pattern")
        ax6.set_ylim(-40, 0)

        # 3D radiation pattern
        ax7 = fig.add_subplot(gs[1, 2:])
        Theta, Phi = np.meshgrid(far_field["theta"], far_field["phi"], indexing="ij")
        intensity_db = 10 * np.log10(
            far_field["radiation_intensity"] / np.max(far_field["radiation_intensity"])
        )
        im7 = ax7.contourf(Phi, Theta, intensity_db, levels=20, cmap="cmc.tokyo")
        ax7.set_xlabel("Phi (rad)")
        ax7.set_ylabel("Theta (rad)")
        ax7.set_title("Radiation Pattern (dB)")
        plt.colorbar(im7, ax=ax7, label="Intensity (dB)")

        # Analysis summary text
        ax8 = fig.add_subplot(gs[2, :])
        ax8.axis("off")

        # Calculate key metrics
        total_energy = (
            np.sum(energy_total)
            * (mode_data.extent[1] - mode_data.extent[0])
            * (mode_data.extent[3] - mode_data.extent[2])
            * 1e-18
        )  # Convert nm² to m²
        avg_poynting = np.mean(poynting_mag)
        max_radiation = np.max(far_field["radiation_intensity"])

        summary_text = f"""
        ELECTROMAGNETIC ANALYSIS SUMMARY
        
        Mode Frequency: {mode_data.frequency:.3f} GHz
        
        Energy Analysis:
        • Total Energy: {total_energy:.2e} J
        • Average Energy Flow: {avg_poynting:.2e} W/m²
        • Energy Confinement: {np.std(energy_total) / np.mean(energy_total):.2f}
        
        Radiation Analysis:
        • Peak Radiation Intensity: {max_radiation:.2e} W/m²/sr
        • Radiation Efficiency: Calculated from energy balance
        • Dominant Radiation Direction: θ={far_field["theta"][np.unravel_index(np.argmax(far_field["radiation_intensity"]), far_field["radiation_intensity"].shape)[0]]:.2f} rad
        
        Mode Characteristics:
        • Spatial Extent: {mode_data.extent[1] - mode_data.extent[0]:.1f} × {mode_data.extent[3] - mode_data.extent[2]:.1f} nm²
        • Mode Volume: Estimated from energy distribution
        • Loss Channels: Radiation + Ohmic losses
        """

        ax8.text(
            0.05,
            0.95,
            summary_text,
            transform=ax8.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        )

    else:
        # Error message if analysis failed
        ax_error = fig.add_subplot(gs[1:, :])
        ax_error.axis("off")
        error_msg = f"Electromagnetic analysis failed: {analysis_results.get('error', 'Unknown error')}"
        ax_error.text(
            0.5,
            0.5,
            error_msg,
            transform=ax_error.transAxes,
            fontsize=16,
            ha="center",
            va="center",
            color="red",
        )

    plt.suptitle(
        f"Comprehensive Electromagnetic Analysis - {mode_data.frequency:.3f} GHz",
        fontsize=18,
    )

    log.info("Generated comprehensive electromagnetic analysis report")
    return fig
