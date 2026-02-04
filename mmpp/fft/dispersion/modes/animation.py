"""
Spin wave mode animation with proper complex amplitude handling.

Implements the methodology from Rychły et al. for visualizing
spin wave propagation with amplitude and phase information.

The key equation:
    m(r,t) = Re[m̃(r) * exp(-iωt)] = A(r) * cos(φ(r) - ωt)

where:
    - A(r) = |m̃(r)| is the amplitude envelope
    - φ(r) = arg(m̃(r)) is the spatial phase
    - ω = 2πf₀ is the angular frequency
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np

logger = logging.getLogger(__name__)

# Check for animation dependencies
_HAS_MATPLOTLIB = False
_HAS_ANIMATION = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    _HAS_MATPLOTLIB = True
except ImportError:
    pass

try:
    from matplotlib.animation import FuncAnimation
    _HAS_ANIMATION = True
except ImportError:
    pass

if TYPE_CHECKING:
    from ..models import DispersionResult1D
    from .models import DispersionMode


def extract_amplitude_phase(M_complex: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract amplitude and phase from complex magnetization field.
    
    From the FFT result M̃ = Re[M] + i·Im[M], compute:
        A = |M̃| = √(Re² + Im²)     amplitude
        φ = arg(M̃) = atan2(Im, Re)  phase
    
    Parameters
    ----------
    M_complex : np.ndarray
        Complex magnetization values from FFT (1D or 2D array)
        
    Returns
    -------
    amplitude : np.ndarray
        Amplitude envelope |M̃| (same shape as input)
    phase : np.ndarray
        Phase in radians from -π to π (same shape as input)
        
    Example
    -------
    >>> M = np.array([1+1j, 2+0j, 1-1j])
    >>> A, phi = extract_amplitude_phase(M)
    >>> print(A)   # [1.414, 2.0, 1.414]
    >>> print(phi) # [π/4, 0, -π/4]
    """
    amplitude = np.abs(M_complex)
    phase = np.angle(M_complex)
    return amplitude, phase


def compute_spinwave_field(
    amplitude: np.ndarray,
    phase: np.ndarray,
    omega: float,
    t: float,
    damping_time: Optional[float] = None,
) -> np.ndarray:
    """
    Compute real magnetization field at time t.
    
    Main equation from Rychły et al.:
        m(y,t) = A(y) · cos(φ(y) - ωt) · exp(-t/τ)
    
    Parameters
    ----------
    amplitude : np.ndarray
        Amplitude envelope A(y)
    phase : np.ndarray
        Spatial phase φ(y) in radians
    omega : float
        Angular frequency ω = 2πf [rad/s]
    t : float
        Time [s]
    damping_time : float, optional
        Damping time constant τ [s]. If None, no damping.
        
    Returns
    -------
    m_field : np.ndarray
        Real magnetization field m(y,t)
    """
    # Total phase at time t
    phase_total = phase - omega * t
    
    # Real field: m = A * cos(φ - ωt)
    m_field = amplitude * np.cos(phase_total)
    
    # Apply damping if specified
    if damping_time is not None and damping_time > 0:
        decay = np.exp(-t / damping_time)
        m_field *= decay
    
    return m_field


def generate_animation_frames(
    amplitude: np.ndarray,
    phase: np.ndarray,
    frequency_hz: float,
    n_frames: int = 120,
    n_periods: float = 3.0,
    damping_time: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate all frames for spin wave animation.
    
    Computes m(y,t) = A(y) · cos(φ(y) - 2πf₀t) for each time step.
    
    Parameters
    ----------
    amplitude : np.ndarray
        Amplitude envelope A(y) from IFFT
    phase : np.ndarray
        Spatial phase φ(y) from IFFT
    frequency_hz : float
        Mode frequency f₀ [Hz]
    n_frames : int
        Number of animation frames
    n_periods : float
        Number of oscillation periods to animate
    damping_time : float, optional
        Damping time constant [s]
        
    Returns
    -------
    frames : np.ndarray
        Array of shape (n_frames, len(amplitude)) with m(y,t) for each t
    time_array : np.ndarray
        Time values [s] for each frame
    """
    # Angular frequency
    omega = 2 * np.pi * frequency_hz
    
    # Total duration: n_periods oscillations
    period = 1.0 / frequency_hz
    total_time = n_periods * period
    
    # Time array
    time_array = np.linspace(0, total_time, n_frames, endpoint=False)
    
    # Generate frames
    frames = np.zeros((n_frames, len(amplitude)))
    
    for i, t in enumerate(time_array):
        frames[i] = compute_spinwave_field(
            amplitude, phase, omega, t, damping_time
        )
    
    logger.debug(
        "Generated %d frames over %.2f ns (%.1f periods at %.2f GHz)",
        n_frames, total_time * 1e9, n_periods, frequency_hz / 1e9
    )
    
    return frames, time_array


class SpinWaveModeAnimator:
    """
    Animator for spin wave mode visualization.
    
    Creates animations showing the oscillating magnetization field
    within the amplitude envelope, based on the Rychły et al. methodology.
    
    Parameters
    ----------
    y_axis : np.ndarray
        Position axis [m]
    amplitude : np.ndarray
        Mode amplitude envelope A(y)
    phase : np.ndarray
        Mode spatial phase φ(y)
    frequency_hz : float
        Mode frequency [Hz]
    k_value : float, optional
        Wave vector [rad/m] for display
        
    Example
    -------
    >>> # From folded dispersion mode
    >>> y, profile_complex, info = folder.extract_mode_profile(
    ...     result, k_0=1e6, f_0=5e9, return_complex=True
    ... )
    >>> A, phi = extract_amplitude_phase(profile_complex)
    >>> animator = SpinWaveModeAnimator(y, A, phi, 5e9)
    >>> animator.animate()
    """
    
    def __init__(
        self,
        y_axis: np.ndarray,
        amplitude: np.ndarray,
        phase: np.ndarray,
        frequency_hz: float,
        k_value: Optional[float] = None,
    ):
        self.y_axis = y_axis
        self.amplitude = amplitude
        self.phase = phase
        self.frequency_hz = frequency_hz
        self.k_value = k_value
        
        # Derived quantities
        self.omega = 2 * np.pi * frequency_hz
        self.period = 1.0 / frequency_hz
        self.wavelength = (2 * np.pi / k_value) if k_value else None
        
        # Animation state
        self._frames: Optional[np.ndarray] = None
        self._time_array: Optional[np.ndarray] = None
        self._fig: Optional[Figure] = None
        self._ax: Optional[Axes] = None
        self._anim = None
    
    def generate_frames(
        self,
        n_frames: int = 120,
        n_periods: float = 3.0,
        damping_time: Optional[float] = None,
    ) -> np.ndarray:
        """
        Pre-generate all animation frames.
        
        Parameters
        ----------
        n_frames : int
            Number of frames
        n_periods : float
            Number of oscillation periods
        damping_time : float, optional
            Damping time constant [s]
            
        Returns
        -------
        frames : np.ndarray
            Animation frames array
        """
        self._frames, self._time_array = generate_animation_frames(
            amplitude=self.amplitude,
            phase=self.phase,
            frequency_hz=self.frequency_hz,
            n_frames=n_frames,
            n_periods=n_periods,
            damping_time=damping_time,
        )
        return self._frames
    
    def plot_snapshot(
        self,
        t: float = 0.0,
        phase_offset: float = 0.0,
        figsize: tuple[float, float] = (12, 5),
        show_envelope: bool = True,
        show_phase: bool = False,
        ax: Optional[Axes] = None,
    ) -> tuple[Figure, Axes]:
        """
        Plot a single snapshot of the spin wave at time t.
        
        Parameters
        ----------
        t : float
            Time [s] or phase offset if phase_offset != 0
        phase_offset : float
            Additional phase offset [rad]
        figsize : tuple
            Figure size
        show_envelope : bool
            Show amplitude envelope
        show_phase : bool
            Show phase profile in secondary axis
        ax : Axes, optional
            Existing axes
            
        Returns
        -------
        fig, ax
        """
        if not _HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")
        
        # Compute field at this instant
        phase_total = self.phase - self.omega * t + phase_offset
        m_field = self.amplitude * np.cos(phase_total)
        
        # Position in μm for display
        y_um = self.y_axis * 1e6
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        # Oscillating field
        ax.plot(y_um, m_field, 'b-', linewidth=2, label=r'$m(y,t)$ - field')
        
        # Envelope
        if show_envelope:
            ax.plot(y_um, self.amplitude, 'r--', linewidth=1.5, 
                    alpha=0.7, label=r'$A(y)$ - envelope')
            ax.plot(y_um, -self.amplitude, 'r--', linewidth=1.5, alpha=0.7)
            ax.fill_between(y_um, -self.amplitude, self.amplitude, 
                           alpha=0.1, color='red')
        
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Position y [μm]')
        ax.set_ylabel('Magnetization [arb. u.]')
        
        # Title with mode info
        title = f'Spin Wave Mode | f₀ = {self.frequency_hz/1e9:.2f} GHz'
        if self.k_value:
            title += f' | k₀ = {self.k_value/1e6:.2f} rad/μm'
        title += f' | t = {t*1e9:.2f} ns'
        ax.set_title(title)
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Phase profile on secondary axis
        if show_phase:
            ax2 = ax.twinx()
            ax2.plot(y_um, np.unwrap(self.phase), 'g-', linewidth=1, 
                    alpha=0.5, label=r'$\phi(y)$')
            ax2.set_ylabel('Phase φ(y) [rad]', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
        
        plt.tight_layout()
        
        return fig, ax
    
    def animate(
        self,
        n_frames: int = 120,
        n_periods: float = 3.0,
        interval: int = 50,
        damping_time: Optional[float] = None,
        figsize: tuple[float, float] = (14, 6),
        show_envelope: bool = True,
        save_path: Optional[str] = None,
        dpi: int = 150,
    ):
        """
        Create and display animated spin wave visualization.
        
        Shows m(y,t) = A(y) · cos(φ(y) - ωt) oscillating within the envelope.
        
        Parameters
        ----------
        n_frames : int
            Number of animation frames
        n_periods : float
            Number of oscillation periods to show
        interval : int
            Delay between frames [ms]
        damping_time : float, optional
            Damping time constant [s]
        figsize : tuple
            Figure size
        show_envelope : bool
            Display amplitude envelope
        save_path : str, optional
            Path to save animation (mp4, gif)
        dpi : int
            Resolution for saved animation
            
        Returns
        -------
        anim : FuncAnimation
            The animation object
        """
        if not _HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for animation")
        if not _HAS_ANIMATION:
            raise ImportError("matplotlib.animation required")
        
        # Generate frames if not already done
        if self._frames is None or len(self._frames) != n_frames:
            self.generate_frames(n_frames, n_periods, damping_time)
        
        # Position in μm
        y_um = self.y_axis * 1e6
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        self._fig = fig
        self._ax = ax
        
        # Initialize plot elements
        line_field, = ax.plot([], [], 'b-', linewidth=2, label=r'$m(y,t)$')
        
        if show_envelope:
            ax.plot(y_um, self.amplitude, 'r--', linewidth=1.5, 
                    alpha=0.6, label=r'$A(y)$ envelope')
            ax.plot(y_um, -self.amplitude, 'r--', linewidth=1.5, alpha=0.6)
            ax.fill_between(y_um, -self.amplitude, self.amplitude, 
                           alpha=0.1, color='red')
        
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        
        # Axis limits
        y_margin = 0.05 * (y_um[-1] - y_um[0])
        ax.set_xlim(y_um[0] - y_margin, y_um[-1] + y_margin)
        ax.set_ylim(-np.max(self.amplitude) * 1.3, np.max(self.amplitude) * 1.3)
        
        ax.set_xlabel('Position y [μm]', fontsize=12)
        ax.set_ylabel('Magnetization [arb. u.]', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Time text
        time_text = ax.text(
            0.02, 0.95, '', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        # Base title
        base_title = f'Spin Wave Mode | f₀ = {self.frequency_hz/1e9:.2f} GHz'
        if self.k_value:
            base_title += f' | k₀ = {self.k_value/1e6:.2f} rad/μm'
        ax.set_title(base_title, fontsize=13)
        
        def init():
            line_field.set_data([], [])
            time_text.set_text('')
            return line_field, time_text
        
        def update(frame_idx):
            m_field = self._frames[frame_idx]
            t = self._time_array[frame_idx]
            
            line_field.set_data(y_um, m_field)
            
            # Time display
            t_ns = t * 1e9
            phase_cycles = t * self.frequency_hz
            time_text.set_text(
                f't = {t_ns:.2f} ns\n'
                f'phase = {phase_cycles:.2f} cycles'
            )
            
            return line_field, time_text
        
        # Create animation
        anim = FuncAnimation(
            fig, update, frames=n_frames,
            init_func=init, interval=interval,
            blit=True, repeat=True
        )
        
        self._anim = anim
        
        # Save if requested
        if save_path:
            logger.info("Saving animation to %s", save_path)
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', dpi=dpi)
            else:
                anim.save(save_path, writer='ffmpeg', fps=1000//interval, dpi=dpi)
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def get_physical_parameters(self) -> dict:
        """
        Get dictionary of physical parameters for this mode.
        
        Returns
        -------
        dict
            Mode parameters with units
        """
        params = {
            'frequency_GHz': self.frequency_hz / 1e9,
            'angular_frequency_rad_per_s': self.omega,
            'period_ns': self.period * 1e9,
            'max_amplitude': float(np.max(self.amplitude)),
            'mean_amplitude': float(np.mean(self.amplitude)),
        }
        
        if self.k_value:
            params['k_rad_per_um'] = self.k_value / 1e6
            params['wavelength_um'] = (2 * np.pi / self.k_value) * 1e6
            params['phase_velocity_m_per_s'] = self.omega / self.k_value
        
        # Estimate group velocity from phase gradient
        if len(self.phase) > 5:
            dy = self.y_axis[1] - self.y_axis[0]
            dphi = np.gradient(np.unwrap(self.phase), dy)
            k_local_mean = np.mean(np.abs(dphi))
            if k_local_mean > 0:
                params['estimated_k_from_phase'] = k_local_mean / 1e6
        
        return params
    
    def __repr__(self) -> str:
        return (
            f"SpinWaveModeAnimator(f={self.frequency_hz/1e9:.2f} GHz, "
            f"N_points={len(self.y_axis)})"
        )


def animate_mode_from_folding(
    folder: "BrillouinZoneFolding",
    result: "DispersionResult1D",
    k_0: float,
    f_0: float,
    n_frames: int = 120,
    n_periods: float = 3.0,
    delta_k: Optional[float] = None,
    delta_f: Optional[float] = None,
    save_path: Optional[str] = None,
    **animate_kwargs,
) -> SpinWaveModeAnimator:
    """
    Convenience function to animate a mode directly from folding results.
    
    Parameters
    ----------
    folder : BrillouinZoneFolding
        The BZ folding object
    result : DispersionResult1D
        Original dispersion result
    k_0 : float
        Mode wave vector [rad/m]
    f_0 : float
        Mode frequency [Hz]
    n_frames : int
        Animation frames
    n_periods : float
        Oscillation periods to show
    delta_k, delta_f : float, optional
        Filter widths
    save_path : str, optional
        Path to save animation
    **animate_kwargs
        Extra kwargs for animate()
        
    Returns
    -------
    animator : SpinWaveModeAnimator
        The animator object
    """
    # Extract mode profile with complex data
    y_axis, profile_complex, info = folder.extract_mode_profile(
        result=result,
        k_0=k_0,
        f_0=f_0,
        delta_k=delta_k,
        delta_f=delta_f,
        return_complex=True,
    )
    
    # Get amplitude and phase
    amplitude, phase = extract_amplitude_phase(profile_complex)
    
    # Create animator
    animator = SpinWaveModeAnimator(
        y_axis=y_axis,
        amplitude=amplitude,
        phase=phase,
        frequency_hz=f_0,
        k_value=k_0,
    )
    
    # Run animation
    animator.animate(
        n_frames=n_frames,
        n_periods=n_periods,
        save_path=save_path,
        **animate_kwargs,
    )
    
    return animator
