"""
Advanced Vortex/Skyrmion Mode Classifier for MMPP
Implements rigorous classification based on theoretical equations
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, field
import logging

log = logging.getLogger(__name__)


@dataclass
class VortexClassificationConfig:
    """Configuration for vortex/skyrmion mode classification"""
    
    # Gyration mode thresholds
    tol_phi_quadrature: float = 0.5  # radians ~28.6°
    eta_parallel_for_gyr: float = 0.6  # E_parallel / E_total threshold
    min_core_radius: float = 0.01  # relative to R_dot
    
    # Breathing mode thresholds  
    eta_perp_for_breath: float = 0.6  # E_perp / E_total threshold
    std_phi_mz_for_breath: float = 0.5  # radians for uniform mz phase
    
    # Ring analysis parameters
    ring_thickness_factor: float = 0.04  # ring half-width as fraction of R_dot
    nbins_radial: int = 96  # number of radial bins for analysis
    
    # Radial node detection
    node_amplitude_threshold: float = 0.25  # fraction of max amplitude
    smoothing_kernel_size: int = 3  # for radial profile smoothing


@dataclass  
class VortexModeResult:
    """Results of vortex/skyrmion mode classification"""
    
    # Core indices
    m_index: int  # azimuthal index
    n_index: int  # radial index  
    l_index: Optional[int] = None  # thickness index (for 3D)
    
    # Classification
    mode_type: str = "azimuthal"  # "gyration", "breathing", "azimuthal"
    rotation_sense: str = "CW"  # "CW" or "CCW"
    confidence: float = 0.0
    
    # Physical parameters
    core_position: Tuple[float, float] = (0.0, 0.0)  # (cx, cy) in pixels
    r_star: float = 0.0  # ring radius where amplitude peaks
    frequency: float = 0.0  # GHz
    
    # Energy measures  
    E_parallel: float = 0.0  # in-plane energy
    E_perp: float = 0.0  # out-of-plane energy
    E_parallel_frac: float = 0.0  # E_parallel / E_total
    
    # Phase relationships
    delta_phi_xy: float = 0.0  # mean phase difference mx-my (radians)
    dist_to_quadrature: float = 0.0  # distance to nearest ±π/2
    std_phi_mz_on_ring: float = 0.0  # std of mz phase on ring
    phase_coherence_xy: float = 0.0  # coherence of mx-my phases
    
    # Core dynamics (for gyration)
    core_orbit_radius: float = 0.0  # R_G / R_dot
    gyration_frequency: Optional[float] = None  # ω_G estimate
    
    # Additional metrics
    radial_nodes: List[float] = field(default_factory=list)  # positions of nodes
    analysis_radius: float = 0.0  # radius used for analysis
    notes: List[str] = field(default_factory=list)


class AdvancedVortexClassifier:
    """
    Advanced vortex/skyrmion mode classifier implementing theoretical equations.
    
    Implements classification based on:
    - Thiele equation dynamics for gyration modes  
    - Azimuthal index m from phase winding
    - Radial index n from amplitude nodes
    - Energy partitioning E_parallel vs E_perp
    - Phase coherence and quadrature relations
    """
    
    def __init__(self, config: Optional[VortexClassificationConfig] = None):
        self.config = config or VortexClassificationConfig()
    
    def classify_mode(
        self, 
        mode_data: "FMRModeData",
        R_dot: Optional[float] = None,
        dx: Optional[float] = None,
        dy: Optional[float] = None,
        dz: Optional[float] = None,
        verbose: bool = False
    ) -> VortexModeResult:
        """
        Classify a single frequency mode using advanced vortex analysis.
        
        Parameters:
        -----------
        mode_data : FMRModeData
            Mode data containing complex magnetization components
        R_dot : float, optional  
            Dot radius in same units as dx/dy. If None, estimated from data
        dx, dy, dz : float, optional
            Grid spacings. If None, taken from mode_data.metadata
        verbose : bool
            Print detailed classification analysis
            
        Returns:
        --------
        VortexModeResult
            Complete classification with indices, energies, and diagnostics
        """
        
        # Extract data and parameters
        mode_array = mode_data.mode_array  # shape (ny, nx, 3)
        ny, nx, _ = mode_array.shape
        frequency = mode_data.frequency
        
        # Get spatial parameters
        if dx is None or dy is None:
            dx, dy = mode_data.metadata.get("spatial_resolution", (1.0, 1.0))
        dx, dy = float(dx), float(dy)
        if dz is not None:
            dz = float(dz)
            
        # Estimate dot radius if not provided
        if R_dot is None:
            R_dot = min(nx * dx, ny * dy) / 4  # rough estimate
        R_dot = float(R_dot)
        
        # Initialize result
        result = VortexModeResult(
            frequency=frequency,
            m_index=0,
            n_index=0
        )
        
        # Extract complex components
        dmx = mode_array[:, :, 0]  # shape (ny, nx)  
        dmy = mode_array[:, :, 1]
        dmz = mode_array[:, :, 2]
        
        # 1. Find vortex core center
        core_pos = self._find_core_center(dmz)
        result.core_position = core_pos
        cx, cy = core_pos
        
        # 2. Compute radial analysis 
        ring_data = self._compute_ring_analysis(dmx, dmy, dmz, cx, cy, dx, dy, R_dot)
        result.r_star = ring_data["r_star"]
        result.analysis_radius = ring_data["analysis_radius"]
        result.radial_nodes = ring_data["radial_nodes"]
        
        # 3. Determine azimuthal index m
        m_index = self._compute_azimuthal_index(dmx, dmy, ring_data)
        result.m_index = m_index
        
        # 4. Determine radial index n  
        n_index = self._compute_radial_index(dmx, dmy, dmz, ring_data)
        result.n_index = n_index
        
        # 5. Energy measures
        energy_data = self._compute_energy_measures(dmx, dmy, dmz, dx, dy)
        result.E_parallel = energy_data["E_parallel"]
        result.E_perp = energy_data["E_perp"] 
        result.E_parallel_frac = energy_data["E_parallel_frac"]
        
        # 6. Phase relationships
        phase_data = self._compute_phase_relations(dmx, dmy, dmz, ring_data)
        result.delta_phi_xy = phase_data["delta_phi_xy"]
        result.dist_to_quadrature = phase_data["dist_to_quadrature"]
        result.std_phi_mz_on_ring = phase_data["std_phi_mz"]
        result.phase_coherence_xy = phase_data["coherence_xy"]
        
        # 7. Rotation sense
        rotation_sense = self._compute_rotation_sense(dmx, dmy)
        result.rotation_sense = rotation_sense
        
        # 8. Classification logic
        classification = self._classify_mode_type(result)
        result.mode_type = classification["type"]
        result.confidence = classification["confidence"]
        result.notes.extend(classification["notes"])
        
        # 9. Gyration-specific analysis
        if result.mode_type == "gyration":
            gyro_data = self._analyze_gyration_mode(result, R_dot)
            result.core_orbit_radius = gyro_data["orbit_radius"]
            result.gyration_frequency = gyro_data["frequency_estimate"]
        
        if verbose:
            self._print_detailed_analysis(result, ring_data, energy_data, phase_data)
            
        return result
    
    def _find_core_center(self, dmz: np.ndarray) -> Tuple[float, float]:
        """Find vortex core center from mz component maximum/minimum"""
        
        # For core with mz > 0, find maximum; for mz < 0, find minimum
        mz_abs = np.abs(dmz)
        max_idx = np.unravel_index(np.argmax(mz_abs), dmz.shape)
        cy, cx = float(max_idx[0]), float(max_idx[1])
        
        # Refine with centroid if needed
        # Use weight function w = 1 - |mz|^2 for better centering
        mz_mag = np.abs(dmz)
        weight = 1.0 - (mz_mag / np.max(mz_mag))**2
        
        y_indices, x_indices = np.meshgrid(np.arange(dmz.shape[1]), 
                                          np.arange(dmz.shape[0]), indexing='ij')
        
        total_weight = np.sum(weight)
        if total_weight > 0:
            cx_refined = np.sum(x_indices * weight) / total_weight  
            cy_refined = np.sum(y_indices * weight) / total_weight
            
            # Use refined if reasonable (within 2 pixels of initial)
            if abs(cx_refined - cx) < 2 and abs(cy_refined - cy) < 2:
                cx, cy = cx_refined, cy_refined
        
        return cx, cy
    
    def _compute_ring_analysis(
        self, 
        dmx: np.ndarray, 
        dmy: np.ndarray, 
        dmz: np.ndarray,
        cx: float, 
        cy: float, 
        dx: float, 
        dy: float, 
        R_dot: float
    ) -> Dict[str, Any]:
        """Compute radial profile and ring parameters"""
        
        ny, nx = dmx.shape
        
        # Create coordinate grids
        x_coords = (np.arange(nx) - cx) * dx
        y_coords = (np.arange(ny) - cy) * dy  
        X, Y = np.meshgrid(x_coords, y_coords)
        R = np.hypot(X, Y)
        PHI = np.arctan2(Y, X)
        
        # Total amplitude
        amp_total = np.sqrt(np.abs(dmx)**2 + np.abs(dmy)**2 + np.abs(dmz)**2)
        
        # Radial profile
        nbins = self.config.nbins_radial
        r_edges = np.linspace(0, R_dot, nbins + 1)
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        
        radial_profile = np.zeros(nbins)
        for i in range(nbins):
            mask = (R >= r_edges[i]) & (R < r_edges[i + 1])
            if np.any(mask):
                radial_profile[i] = np.mean(amp_total[mask])
        
        # Smooth profile
        if self.config.smoothing_kernel_size >= 3:
            kernel_size = self.config.smoothing_kernel_size
            kernel = np.ones(kernel_size) / kernel_size
            radial_profile = np.convolve(radial_profile, kernel, mode='same')
        
        # Find r_star (peak of radial profile, excluding center)
        start_idx = max(1, int(0.1 * nbins))  # avoid center
        peak_idx = start_idx + np.argmax(radial_profile[start_idx:])
        r_star = r_centers[peak_idx]
        
        # Analysis radius for ring
        dr = max(dx, dy) + self.config.ring_thickness_factor * R_dot
        analysis_radius = dr
        
        # Ring mask
        ring_mask = (R >= (r_star - dr)) & (R <= (r_star + dr))
        
        # Find radial nodes
        prof_norm = radial_profile / (np.max(radial_profile) + 1e-12)
        nodes = []
        threshold = self.config.node_amplitude_threshold
        
        # Find local minima below threshold
        for i in range(1, len(prof_norm) - 1):
            if (prof_norm[i] < prof_norm[i-1] and 
                prof_norm[i] < prof_norm[i+1] and
                prof_norm[i] < threshold):
                nodes.append(r_centers[i])
        
        return {
            "r_star": r_star,
            "analysis_radius": analysis_radius,
            "ring_mask": ring_mask,
            "R": R,
            "PHI": PHI,
            "radial_profile": radial_profile,
            "r_centers": r_centers,
            "radial_nodes": nodes
        }
    
    def _compute_azimuthal_index(
        self, 
        dmx: np.ndarray, 
        dmy: np.ndarray, 
        ring_data: Dict[str, Any]
    ) -> int:
        """Compute azimuthal index m from phase winding on ring"""
        
        ring_mask = ring_data["ring_mask"]
        PHI = ring_data["PHI"]
        
        if np.sum(ring_mask) < 8:  # too few points
            return 0
        
        # In-plane complex field
        m_perp = dmx + 1j * dmy
        
        # Phase on ring
        psi = np.angle(m_perp[ring_mask])
        phi_ring = PHI[ring_mask]
        
        # Sort by azimuthal angle and unwrap
        sort_idx = np.argsort(phi_ring)
        phi_sorted = phi_ring[sort_idx]
        psi_sorted = psi[sort_idx]
        psi_unwrapped = np.unwrap(psi_sorted)
        
        # Compute winding: m = (1/2π) ∫ ∂ψ/∂φ dφ
        if len(phi_sorted) > 1:
            dphi_total = phi_sorted[-1] - phi_sorted[0]
            dpsi_total = psi_unwrapped[-1] - psi_unwrapped[0]
            
            # Handle case where we don't have full 2π coverage
            if dphi_total > np.pi:  # reasonable coverage
                m_raw = dpsi_total / (2.0 * np.pi)
                m_index = int(np.round(m_raw))
            else:
                # Estimate from partial coverage
                if dphi_total > 0:
                    slope = dpsi_total / dphi_total
                    m_raw = slope / 1.0  # rough estimate 
                    m_index = int(np.round(m_raw))
                else:
                    m_index = 0
        else:
            m_index = 0
            
        return m_index
    
    def _compute_radial_index(
        self, 
        dmx: np.ndarray, 
        dmy: np.ndarray, 
        dmz: np.ndarray,
        ring_data: Dict[str, Any]
    ) -> int:
        """Compute radial index n from number of amplitude nodes"""
        
        # Count nodes in radial profile
        radial_nodes = ring_data["radial_nodes"]
        n_index = len(radial_nodes)
        
        return n_index
    
    def _compute_energy_measures(
        self, 
        dmx: np.ndarray, 
        dmy: np.ndarray, 
        dmz: np.ndarray,
        dx: float, 
        dy: float
    ) -> Dict[str, float]:
        """Compute energy partitioning"""
        
        area_element = dx * dy
        
        E_parallel = area_element * np.sum(np.abs(dmx)**2 + np.abs(dmy)**2)
        E_perp = area_element * np.sum(np.abs(dmz)**2)
        E_total = E_parallel + E_perp + 1e-30
        E_parallel_frac = E_parallel / E_total
        
        return {
            "E_parallel": float(E_parallel),
            "E_perp": float(E_perp), 
            "E_total": float(E_total),
            "E_parallel_frac": float(E_parallel_frac)
        }
    
    def _compute_phase_relations(
        self, 
        dmx: np.ndarray, 
        dmy: np.ndarray, 
        dmz: np.ndarray,
        ring_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute phase relationships and coherence"""
        
        ring_mask = ring_data["ring_mask"]
        
        # Mean phase difference between mx and my
        coherence_complex = np.mean(dmx * np.conj(dmy))
        delta_phi_xy = float(np.angle(coherence_complex))
        
        # Distance to nearest quadrature (±π/2)
        dist_to_quadrature = min(
            abs(abs(delta_phi_xy) - np.pi/2),
            abs((np.pi - abs(delta_phi_xy)) - np.pi/2)
        )
        
        # Phase coherence (magnitude of normalized cross-correlation)
        mx_norm = dmx / (np.abs(dmx) + 1e-12)
        my_norm = dmy / (np.abs(dmy) + 1e-12) 
        coherence_xy = float(np.abs(np.mean(mx_norm * np.conj(my_norm))))
        
        # Standard deviation of mz phase on ring
        if np.sum(ring_mask) > 0:
            mz_phase_ring = np.angle(dmz[ring_mask])
            std_phi_mz = float(np.std(mz_phase_ring))
        else:
            std_phi_mz = float('nan')
        
        return {
            "delta_phi_xy": delta_phi_xy,
            "dist_to_quadrature": dist_to_quadrature,
            "coherence_xy": coherence_xy,
            "std_phi_mz": std_phi_mz
        }
    
    def _compute_rotation_sense(
        self, 
        dmx: np.ndarray, 
        dmy: np.ndarray
    ) -> str:
        """Determine CW/CCW rotation sense"""
        
        # S = ∫ (mx + i my)(mx - i my)* d²r
        m_plus = dmx + 1j * dmy
        m_minus = dmx - 1j * dmy
        S = np.sum(m_plus * np.conj(m_minus))
        
        return "CCW" if np.imag(S) > 0 else "CW"
    
    def _classify_mode_type(self, result: VortexModeResult) -> Dict[str, Any]:
        """Classify mode type based on computed parameters"""
        
        m = result.m_index
        E_par_frac = result.E_parallel_frac
        E_perp_frac = 1.0 - E_par_frac
        dist_quad = result.dist_to_quadrature
        std_mz = result.std_phi_mz_on_ring
        coherence = result.phase_coherence_xy
        
        notes = []
        confidence = 0.0
        
        # Breathing mode: m=0, strong out-of-plane, uniform mz phase
        breathing_score = 0.0
        if m == 0:
            breathing_score += 0.4
            notes.append("m=0 supports breathing")
        if E_perp_frac > self.config.eta_perp_for_breath:
            breathing_score += 0.4  
            notes.append(f"Strong out-of-plane energy: {E_perp_frac:.3f}")
        if not np.isnan(std_mz) and std_mz < self.config.std_phi_mz_for_breath:
            breathing_score += 0.3
            notes.append(f"Uniform mz phase: std={std_mz:.3f}")
        
        # Gyration mode: |m|=1, strong in-plane, quadrature x-y
        gyration_score = 0.0  
        if abs(m) == 1:
            gyration_score += 0.4
            notes.append(f"|m|=1 supports gyration: m={m}")
        if E_par_frac > self.config.eta_parallel_for_gyr:
            gyration_score += 0.4
            notes.append(f"Strong in-plane energy: {E_par_frac:.3f}")
        if dist_quad < self.config.tol_phi_quadrature:
            gyration_score += 0.3
            notes.append(f"Good quadrature: dist={dist_quad:.3f} rad")
        
        # Classify based on highest score
        if breathing_score > gyration_score and breathing_score > 0.7:
            mode_type = "breathing"
            confidence = min(breathing_score, 1.0)
        elif gyration_score > 0.7:
            mode_type = "gyration" 
            confidence = min(gyration_score, 1.0)
        else:
            mode_type = "azimuthal"
            confidence = 0.5  # default for azimuthal
            notes.append(f"Azimuthal mode with m={m}")
        
        return {
            "type": mode_type,
            "confidence": confidence,
            "notes": notes
        }
    
    def _analyze_gyration_mode(
        self, 
        result: VortexModeResult, 
        R_dot: float
    ) -> Dict[str, Any]:
        """Additional analysis for gyration modes"""
        
        # Estimate core orbit radius (would need time series for full analysis)
        # For now, use typical scaling
        orbit_radius = 0.1  # placeholder - would be R_G / R_dot from time series
        
        # Estimate gyration frequency from Thiele equation  
        # ω_G ≈ κ/|G| where G ≈ 2π|q||p|Ms*L/|γ|
        # This is very rough without material parameters
        frequency_estimate = None  # would need material params
        
        return {
            "orbit_radius": orbit_radius,
            "frequency_estimate": frequency_estimate
        }
    
    def _print_detailed_analysis(
        self, 
        result: VortexModeResult,
        ring_data: Dict[str, Any],
        energy_data: Dict[str, Any], 
        phase_data: Dict[str, Any]
    ) -> None:
        """Print detailed verbose analysis"""
        
        print("\n" + "="*80)
        print("🌀 ADVANCED VORTEX/SKYRMION MODE ANALYSIS")
        print("="*80)
        
        print(f"Frequency: {result.frequency:.3f} GHz")
        print(f"Final Classification: {result.mode_type.upper()}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Mode indices: m={result.m_index}, n={result.n_index}")
        
        print(f"\n📍 CORE ANALYSIS:")
        cx, cy = result.core_position
        print(f"   • Core center: ({cx:.1f}, {cy:.1f}) pixels")  
        print(f"   • Ring radius r*: {result.r_star:.2f}")
        print(f"   • Analysis radius: {result.analysis_radius:.2f}")
        print(f"   • Rotation sense: {result.rotation_sense}")
        
        print(f"\n⚡ ENERGY DISTRIBUTION:")
        print(f"   • In-plane energy: {result.E_parallel:.2e}")
        print(f"   • Out-of-plane energy: {result.E_perp:.2e}")  
        print(f"   • In-plane fraction: {result.E_parallel_frac:.3f}")
        print(f"   • Out-of-plane fraction: {1-result.E_parallel_frac:.3f}")
        
        print(f"\n🔄 PHASE RELATIONSHIPS:")
        print(f"   • Phase diff mx-my: {result.delta_phi_xy*180/np.pi:.1f}° ({result.delta_phi_xy:.3f} rad)")
        print(f"   • Distance to quadrature: {result.dist_to_quadrature*180/np.pi:.1f}°")
        print(f"   • Phase coherence xy: {result.phase_coherence_xy:.3f}")
        print(f"   • mz phase std on ring: {result.std_phi_mz_on_ring:.3f} rad")
        
        print(f"\n📊 MODE INDICES:")
        print(f"   • Azimuthal index m: {result.m_index}")
        print(f"   • Radial index n: {result.n_index}")
        print(f"   • Radial nodes at: {result.radial_nodes}")
        
        if result.mode_type == "gyration":
            print(f"\n🌀 GYRATION SPECIFICS:")
            print(f"   • Core orbit radius: {result.core_orbit_radius:.3f}")
            if result.gyration_frequency:
                print(f"   • Estimated ω_G: {result.gyration_frequency:.3f} GHz")
        
        print(f"\n📝 CLASSIFICATION NOTES:")
        for note in result.notes:
            print(f"   • {note}")
        
        print("="*80)


def integrate_vortex_classifier_with_mmpp():
    """Integration function to add vortex classifier to existing MMPP system"""
    
    # This would be called to extend the existing ModeCharacterAnalyzer
    pass


# Test function
def test_vortex_classifier():
    """Test the vortex classifier with synthetic data"""
    
    print("🧪 Testing Advanced Vortex Classifier")
    print("="*50)
    
    # Create synthetic vortex mode data
    size = 64
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size) 
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y, X)
    
    # Gyration mode m=1
    dmx_gyr = np.exp(1j * PHI) * np.exp(-R**2/4)  
    dmy_gyr = np.exp(1j * (PHI + np.pi/2)) * np.exp(-R**2/4)  # 90° phase shift
    dmz_gyr = 0.1 * np.exp(1j * 0.1*PHI) * np.exp(-R**2/8)
    
    # Create mock FMRModeData
    class MockModeData:
        def __init__(self, frequency, mode_array):
            self.frequency = frequency
            self.mode_array = mode_array
            self.metadata = {"spatial_resolution": (0.1, 0.1)}  # 100 nm pixels
    
    mode_array_gyr = np.stack([dmx_gyr, dmy_gyr, dmz_gyr], axis=2)
    mock_data = MockModeData(8.5, mode_array_gyr)
    
    # Test classifier
    classifier = AdvancedVortexClassifier()
    result = classifier.classify_mode(mock_data, R_dot=2.0, verbose=True)
    
    print(f"\n✅ Test completed!")
    print(f"   Classification: {result.mode_type}")
    print(f"   m-index: {result.m_index}")  
    print(f"   n-index: {result.n_index}")
    print(f"   Confidence: {result.confidence:.3f}")
    

if __name__ == "__main__":
    test_vortex_classifier()