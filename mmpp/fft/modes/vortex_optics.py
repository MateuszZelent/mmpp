"""
Vortex Optics Module - Topological Transformations for Magnetization Dynamics

This module provides rigorous tensor transformations from Cartesian basis to
higher-symmetry bases (circular/helical and cylindrical) for analyzing
gyrotropic vortex cores and azimuthal spin wave modes.

Author: Micromag Physics Engine
Integration: mmpp.fft.modes
"""

import logging
from typing import Optional

import matplotlib.colors as mcolors
import numpy as np

log = logging.getLogger("mmpp.fft.modes.vortex_optics")


class VortexOptics:
    """
    Fizyczny silnik (Micromag Singular Optics Engine) do rygorystycznej 
    transformacji tensorów dynamiki magnetyzacji z bazy kartezjańskiej 
    na bazy wynikające z wyższych symetrii topologicznych.
    """
    
    @staticmethod
    def to_circular_basis(m_x: np.ndarray, m_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform to circular polarization basis (Helical Basis).
        
        Isolates chiral gyrotropic core modes and spin waves carrying 
        orbital angular momentum (OAM).
        
        Parameters:
        -----------
        m_x : np.ndarray
            X-component of magnetization (real or complex)
        m_y : np.ndarray
            Y-component of magnetization (real or complex)
            
        Returns:
        --------
        tuple[np.ndarray, np.ndarray]
            (m_plus, m_minus) - Right (RCP) and Left (LCP) circular polarizations
            
        Notes:
        ------
        The 1/√2 factor ensures unitary transformation (energy conservation).
        - m_plus: Right-hand circular polarization (RCP) - counterclockwise
        - m_minus: Left-hand circular polarization (LCP) - clockwise
        
        For gyrotropic vortex core: one of (m+, m-) dominates depending on core polarity.
        """
        m_plus = (m_x + 1j * m_y) / np.sqrt(2.0)
        m_minus = (m_x - 1j * m_y) / np.sqrt(2.0)
        
        log.debug(f"Circular basis: m+ shape={m_plus.shape}, m- shape={m_minus.shape}")
        return m_plus, m_minus

    @staticmethod
    def to_cylindrical_basis(
        m_x: np.ndarray,
        m_y: np.ndarray,
        center: Optional[tuple[float, float]] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform to cylindrical (magnetocentric) basis.
        
        Deconstructs vector field into independent breathing modes (m_rho) 
        and azimuthal/rotating modes (m_phi).
        
        Parameters:
        -----------
        m_x : np.ndarray
            X-component of magnetization
        m_y : np.ndarray  
            Y-component of magnetization
        center : tuple, optional
            (cx, cy) center of symmetry. If None, uses geometric center.
            
        Returns:
        --------
        tuple[np.ndarray, np.ndarray]
            (m_rho, m_phi) - Radial and azimuthal components
            
        Notes:
        ------
        - m_rho: Breathing/radial oscillations (expansion/contraction)
        - m_phi: Azimuthal/tangential oscillations (rotation around center)
        
        Perfect for analyzing modes with cylindrical symmetry (vortices, skyrmions).
        """
        ny, nx = m_x.shape[-2:]
        y_idx, x_idx = np.indices((ny, nx))
        
        # Wyznaczenie środka symetrii układu
        if center is None:
            cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
        else:
            cx, cy = center
            
        # Zgodność z prawoskrętnym fizycznym układem odniesienia
        # (Oś Y w NumPy rośnie w dół, więc ją odwracamy)
        X = x_idx - cx
        Y = -(y_idx - cy)
        
        phi = np.arctan2(Y, X)
        
        # Ochrona dla tensorów o wyższym rzędzie (gdyby pojawił się wymiar np. czasu/częstotliwości)
        while phi.ndim < m_x.ndim:
            phi = np.expand_dims(phi, 0)
            
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        # Orthogonal rotation in-plane: (x, y) → (ρ, φ)
        m_rho = m_x * cos_phi + m_y * sin_phi
        m_phi = -m_x * sin_phi + m_y * cos_phi
        
        log.debug(f"Cylindrical basis: m_rho shape={m_rho.shape}, m_phi shape={m_phi.shape}")
        return m_rho, m_phi

    @staticmethod
    def complex_holography(
        z_array: np.ndarray, 
        gamma: float = 0.6, 
        noise_threshold: float = 1e-4,
        saturation: float = 1.0
    ) -> np.ndarray:
        """
        Complex Holography via Domain Coloring.
        
        Projects complex-plane information onto HSV color space, then converts 
        to RGB for Matplotlib rendering. Encodes both amplitude and phase in 
        single image.
        
        Parameters:
        -----------
        z_array : np.ndarray
            Complex array (mode data: amplitude × exp(i·phase))
        gamma : float, default=0.6
            Nonlinear brightness enhancement (pulls wavefronts from noise floor)
            Values < 1 enhance low amplitudes, > 1 suppress them
        noise_threshold : float, default=1e-4
            Relative threshold below which pixels are set to black (noise suppression)
        saturation : float, default=1.0
            Color saturation (1.0 = full color, 0.0 = grayscale)
            
        Returns:
        --------
        np.ndarray
            RGB image array with shape (*z_array.shape, 3) in range [0, 1]
            
        Notes:
        ------
        Color mapping:
        - Hue (H): Phase angle, -π (red) → 0 (cyan) → +π (red)
        - Saturation (S): Fixed at `saturation` parameter
        - Value (V): Amplitude with gamma correction and noise gating
        
        Topological features:
        - Vortex cores appear as points where all colors meet (phase singularity)
        - Phase gradients → color wheel rotation
        - Amplitude modulation → brightness variation
        """
        amp = np.abs(z_array)
        phase = np.angle(z_array)
        
        max_amp = np.nanmax(amp)
        if max_amp == 0 or np.isnan(max_amp):
            # Return black image if no signal
            return np.zeros(z_array.shape + (3,), dtype=np.float32)
        
        # Nonlinear amplitude enhancement (gamma correction)
        # Pulls weak wavefronts out of noise background
        V = np.clip(amp / max_amp, 0, 1) ** gamma
        
        # Noise floor suppression: set very low amplitudes to black
        V[amp < noise_threshold * max_amp] = 0.0
        
        # Cyclic phase mapping to hue [0, 1]
        # Phase wraps: -π → 0, 0 → 0.5, +π → 1.0
        H = (phase + np.pi) / (2 * np.pi)
        
        # Full color saturation for topological clarity
        S = np.full_like(H, saturation)
        
        # Stack into HSV image
        hsv_img = np.dstack((H, S, V))
        
        # Convert to RGB for Matplotlib
        rgb_img = mcolors.hsv_to_rgb(hsv_img)
        
        log.debug(f"Holography: input shape={z_array.shape}, RGB output shape={rgb_img.shape}")
        return rgb_img
    
    @staticmethod
    def resolve_physical_components(
        m_x: np.ndarray,
        m_y: np.ndarray,
        m_z: np.ndarray,
        components: list[str],
        vortex_center: Optional[tuple[float, float]] = None
    ) -> dict[str, np.ndarray]:
        """
        Intelligent router for physical basis transformations.
        
        Decodes requested component labels and performs necessary transformations
        on-the-fly. Supports Cartesian, helical, and cylindrical bases.
        
        Parameters:
        -----------
        m_x, m_y, m_z : np.ndarray
            Cartesian magnetization components
        components : list[str]
            Requested components. Supported values:
            - 'x', 'y', 'z': Cartesian (standard)
            - '+', '-': Circular/helical (RCP, LCP)
            - 'rho', 'phi': Cylindrical (radial, azimuthal)
        vortex_center : tuple, optional
            Center for cylindrical transformation
            
        Returns:
        --------
        dict[str, np.ndarray]
            Dictionary mapping component labels to transformed data arrays
            
        Examples:
        ---------
        >>> data = resolve_physical_components(mx, my, mz, ['+', '-', 'z'])
        >>> m_plus = data['+']  # Right circular polarization
        >>> m_minus = data['-']  # Left circular polarization
        >>> m_z = data['z']  # Out-of-plane component
        """
        resolved_data = {'x': m_x, 'y': m_y, 'z': m_z}
        
        # Transform only if requested (lazy evaluation for performance)
        needs_circular = any(c in ['+', '-'] for c in components)
        needs_cylindrical = any(c in ['rho', 'phi'] for c in components)
        
        if needs_circular:
            m_plus, m_minus = VortexOptics.to_circular_basis(m_x, m_y)
            resolved_data['+'] = m_plus
            resolved_data['-'] = m_minus
            log.info("Circular basis computed: m+ (RCP), m- (LCP)")
        
        if needs_cylindrical:
            m_rho, m_phi = VortexOptics.to_cylindrical_basis(m_x, m_y, center=vortex_center)
            resolved_data['rho'] = m_rho
            resolved_data['phi'] = m_phi
            log.info("Cylindrical basis computed: m_rho (radial), m_phi (azimuthal)")
        
        return resolved_data
    
    @staticmethod
    def get_component_label(component: str, latex: bool = True) -> str:
        """
        Get formatted label for component visualization.
        
        Parameters:
        -----------
        component : str
            Component key ('x', 'y', 'z', '+', '-', 'rho', 'phi')
        latex : bool, default=True
            Use LaTeX formatting for publication quality
            
        Returns:
        --------
        str
            Formatted label string
        """
        if latex:
            labels = {
                'x': r'$m_x$',
                'y': r'$m_y$',
                'z': r'$m_z$',
                '+': r'$m^+$ (RCP)',
                '-': r'$m^-$ (LCP)',
                'rho': r'$m_\rho$ (Radial)',
                'phi': r'$m_\phi$ (Azimuthal)',
            }
        else:
            labels = {
                'x': 'mx',
                'y': 'my',
                'z': 'mz',
                '+': 'm+ (RCP)',
                '-': 'm- (LCP)',
                'rho': 'm_rho (Radial)',
                'phi': 'm_phi (Azimuthal)',
            }
        return labels.get(component, component)


# Convenience functions for direct module-level access
def to_circular(m_x: np.ndarray, m_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shorthand for VortexOptics.to_circular_basis"""
    return VortexOptics.to_circular_basis(m_x, m_y)


def to_cylindrical(
    m_x: np.ndarray,
    m_y: np.ndarray,
    center: Optional[tuple[float, float]] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Shorthand for VortexOptics.to_cylindrical_basis"""
    return VortexOptics.to_cylindrical_basis(m_x, m_y, center)


def hologram(
    z_array: np.ndarray, 
    gamma: float = 0.6, 
    noise_threshold: float = 1e-4
) -> np.ndarray:
    """Shorthand for VortexOptics.complex_holography"""
    return VortexOptics.complex_holography(z_array, gamma, noise_threshold)