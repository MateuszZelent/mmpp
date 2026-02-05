"""
Mode profile visualization and animation.

After extracting a mode using the Rychły et al. algorithm, this module provides
tools for plotting and animating the complex mode profile m(x,y).

Classes
-------
ModeProfile
    Container for mode data with plotting and animation capabilities.
"""

from __future__ import annotations

import logging
from typing import Optional, Literal, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


logger = logging.getLogger(__name__)


class ModeProfile:
    """
    Container for spatial mode profile m(x,y) with visualization tools.
    
    This object is returned by `modes.mode(k, f)` and contains the complex
    spatial mode profile extracted using the Rychły et al. algorithm.
    
    The mode is stored as a complex array M(x,y) = A(x,y) * exp(i*φ(x,y)),
    allowing extraction of:
    - Real part: Re[M] = A*cos(φ) - physical magnetization at t=0
    - Imaginary part: Im[M] = A*sin(φ) - 90° phase-shifted component
    - Amplitude: |M| - envelope showing oscillation strength
    - Phase: arg(M) - spatial phase distribution
    
    Attributes
    ----------
    m_xy : np.ndarray
        Complex mode profile, shape (N_y, N_x)
    x : np.ndarray
        X-axis positions in meters
    y : np.ndarray
        Y-axis positions in meters
    k : float
        Wave vector in m^-1
    f : float
        Frequency in Hz
    info : dict
        Metadata about the mode extraction
    
    Examples
    --------
    >>> mode = modes.mode(k=2.3, f=1.12)
    >>> mode.plot(mode_type='abs', cmap='hot', dpi=150)
    >>> mode.plot(mode_type='phase', cmap='hsv')
    >>> mode.animate(duration_ns=10, n_frames=100, mode_type='real')
    """
    
    def __init__(
        self,
        m_xy: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        k: float,
        f: float,
        info: dict,
    ):
        """
        Initialize mode profile container.
        
        Parameters
        ----------
        m_xy : np.ndarray
            Complex mode profile (N_y, N_x)
        x : np.ndarray
            X-axis in meters
        y : np.ndarray
            Y-axis in meters
        k : float
            Wave vector in m^-1
        f : float
            Frequency in Hz
        info : dict
            Metadata dictionary
        """
        self.m_xy = m_xy
        self.x = x
        self.y = y
        self.k = k
        self.f = f
        self.info = info
    
    def __repr__(self) -> str:
        k_um = self.k / 1e6
        f_ghz = self.f / 1e9
        return (
            f"ModeProfile(k={k_um:.3f} rad/μm, f={f_ghz:.3f} GHz, "
            f"shape={self.m_xy.shape})"
        )
    
    def _get_mode_data(
        self,
        mode_type: Literal['real', 'imag', 'abs', 'phase', 'complex', 'ampl_phase']
    ) -> np.ndarray:
        """Extract requested component from complex mode data."""
        if mode_type == 'real':
            return np.real(self.m_xy)
        elif mode_type == 'imag':
            return np.imag(self.m_xy)
        elif mode_type == 'abs':
            return np.abs(self.m_xy)
        elif mode_type == 'phase':
            return np.angle(self.m_xy)
        elif mode_type == 'complex':
            return self.m_xy
        elif mode_type == 'ampl_phase':
            # Return RGB array
            from ..utils import create_amplitude_phase_colormap
            return create_amplitude_phase_colormap(self.m_xy)
        else:
            raise ValueError(
                f"Unknown mode_type='{mode_type}'. "
                f"Choose from: 'real', 'imag', 'abs', 'phase', 'ampl_phase', 'complex'"
            )
    
    def _get_default_cmap(
        self,
        mode_type: Literal['real', 'imag', 'abs', 'phase', 'complex', 'ampl_phase']
    ) -> str:
        """Get default colormap for each mode type."""
        defaults = {
            'real': 'RdBu_r',
            'imag': 'RdBu_r',
            'abs': 'hot',
            'phase': 'hsv',
            'complex': 'RdBu_r',
            'ampl_phase': None,  # RGB - no colormap
        }
        return defaults.get(mode_type, 'viridis')
    
    def plot(
        self,
        mode_type: Literal['real', 'imag', 'abs', 'phase', 'ampl_phase'] = 'real',
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        dpi: int = 150,
        figsize: Optional[Tuple[float, float]] = None,
        title: Optional[str] = None,
        xlabel: str = 'x [μm]',
        ylabel: str = 'y [μm]',
        colorbar: bool = True,
        colorbar_label: Optional[str] = None,
        aspect: str = 'auto',
        interpolation: str = 'nearest',
        show: bool = True,
    ) -> Tuple[Figure, Axes]:
        """
        Plot 2D spatial mode profile.
        
        Parameters
        ----------
        mode_type : {'real', 'imag', 'abs', 'phase', 'ampl_phase'}
            Which component to plot:
            - 'real': Re[M] - physical magnetization at t=0
            - 'imag': Im[M] - 90° phase-shifted component
            - 'abs': |M| - amplitude envelope
            - 'phase': arg(M) - spatial phase (−π to π)
            - 'ampl_phase': Combined amplitude×phase (hue=phase, brightness=amplitude)
        cmap : str, optional
            Matplotlib colormap. If None, uses defaults:
            - 'RdBu_r' for real/imag
            - 'hot' for abs
            - 'hsv' for phase
            - None for ampl_phase (uses RGB)
        vmin, vmax : float, optional
            Color scale limits. If None, auto-scaled. Not used for 'ampl_phase'.
        dpi : int
            Figure resolution (default: 150)
        figsize : tuple, optional
            Figure size (width, height) in inches. If None, auto-sized.
        title : str, optional
            Plot title. If None, generates from mode_type and (k, f).
        xlabel, ylabel : str
            Axis labels (default: 'x [μm]', 'y [μm]')
        colorbar : bool
            Show colorbar (default: True). Not shown for 'ampl_phase'.
        colorbar_label : str, optional
            Colorbar label. If None, auto-generated.
        aspect : str
            Aspect ratio ('auto', 'equal', or float)
        interpolation : str
            Interpolation method for imshow (default: 'nearest')
        show : bool
            Call plt.show() (default: True)
        
        Returns
        -------
        fig : Figure
            Matplotlib figure
        ax : Axes
            Matplotlib axes
        
        Examples
        --------
        >>> mode = modes.mode(k=2.3, f=1.12)
        >>> mode.plot(mode_type='abs', cmap='hot', dpi=200)
        >>> mode.plot(mode_type='phase', vmin=-np.pi, vmax=np.pi)
        >>> mode.plot(mode_type='real', figsize=(12, 8), colorbar_label='m_z')
        >>> mode.plot(mode_type='ampl_phase')  # Combined view
        """
        import matplotlib.pyplot as plt
        
        # Get data
        data = self._get_mode_data(mode_type)
        is_rgb = (mode_type == 'ampl_phase')
        
        # Auto colormap
        if cmap is None:
            cmap = self._get_default_cmap(mode_type)
        
        # Auto figsize
        if figsize is None:
            aspect_ratio = data.shape[0] / data.shape[1]  # N_y / N_x
            figsize = (10, 10 * aspect_ratio * 0.8)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Extent for proper axis scaling
        x_min, x_max = self.x.min() * 1e6, self.x.max() * 1e6  # μm
        y_min, y_max = self.y.min() * 1e6, self.y.max() * 1e6  # μm
        extent = [x_min, x_max, y_min, y_max]
        
        # Plot
        if is_rgb:
            # RGB image - no colormap
            im = ax.imshow(
                data,
                origin='lower',
                extent=extent,
                aspect=aspect,
                interpolation=interpolation,
            )
        else:
            # Scalar data with colormap
            im = ax.imshow(
                data,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                origin='lower',
                extent=extent,
                aspect=aspect,
                interpolation=interpolation,
            )
        
        # Labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Title
        if title is None:
            k_um = self.k / 1e6
            f_ghz = self.f / 1e9
            mode_labels = {
                'real': 'Re[M]',
                'imag': 'Im[M]',
                'abs': '|M|',
                'phase': 'φ[M]',
                'ampl_phase': 'Ampl×Phase',
            }
            title = f"{mode_labels.get(mode_type, mode_type)} — k={k_um:.3f} rad/μm, f={f_ghz:.3f} GHz"
        ax.set_title(title)
        
        # Colorbar (only for scalar data)
        if colorbar and not is_rgb:
            if colorbar_label is None:
                colorbar_label = {
                    'real': 'Re[M] [a.u.]',
                    'imag': 'Im[M] [a.u.]',
                    'abs': '|M| [a.u.]',
                    'phase': 'Phase [rad]',
                }.get(mode_type, 'Value [a.u.]')
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(colorbar_label)
        
        if show:
            plt.show()
        
        return fig, ax
    
    def animate(
        self,
        duration_ns: Optional[float] = None,
        n_frames: int = 100,
        mode_type: Literal['real', 'imag'] = 'real',
        cmap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        dpi: int = 100,
        figsize: Optional[Tuple[float, float]] = None,
        title_template: Optional[str] = None,
        xlabel: str = 'x [μm]',
        ylabel: str = 'y [μm]',
        colorbar: bool = True,
        fps: int = 30,
        repeat: bool = True,
        interval: Optional[float] = None,
    ):
        """
        Animate mode oscillation in time: m(x,y,t) = Re[M(x,y) * exp(-iωt)].
        
        This shows the temporal evolution of the mode at its resonance frequency.
        The animation displays m(x,y,t) = Re[M * exp(-i*2π*f*t)] or Im[...].
        
        Parameters
        ----------
        duration_ns : float, optional
            Animation duration in nanoseconds. If None, shows 3 periods.
        n_frames : int
            Number of animation frames (default: 100)
        mode_type : {'real', 'imag'}
            Which time-dependent component:
            - 'real': m(t) = Re[M * exp(-iωt)] - physical oscillation
            - 'imag': m(t) = Im[M * exp(-iωt)] - quadrature component
        cmap : str, optional
            Colormap (default: 'RdBu_r')
        vmin, vmax : float, optional
            Color scale limits. If None, uses global min/max across all frames.
        dpi : int
            Figure resolution (default: 100)
        figsize : tuple, optional
            Figure size (width, height) in inches
        title_template : str, optional
            Title with {t_ns} placeholder (e.g., "Mode at t={t_ns:.2f} ns")
        xlabel, ylabel : str
            Axis labels
        colorbar : bool
            Show colorbar (default: True)
        fps : int
            Frames per second for animation (default: 30)
        repeat : bool
            Loop animation (default: True)
        interval : float, optional
            Delay between frames in ms. If None, computed from fps.
        
        Returns
        -------
        anim : matplotlib.animation.FuncAnimation
            Animation object. Use anim.save('file.mp4') or anim.save('file.gif').
        
        Examples
        --------
        >>> mode = modes.mode(k=2.3, f=1.12)
        >>> anim = mode.animate(duration_ns=10, n_frames=120, fps=30)
        >>> anim.save('mode_oscillation.gif', writer='pillow', fps=30)
        >>> 
        >>> # Custom animation
        >>> anim = mode.animate(
        ...     duration_ns=5,
        ...     mode_type='real',
        ...     cmap='seismic',
        ...     vmin=-1, vmax=1,
        ...     dpi=150,
        ...     title_template='t = {t_ns:.2f} ns'
        ... )
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
        # Compute time parameters
        period_s = 1 / self.f
        omega = 2 * np.pi * self.f
        
        if duration_ns is None:
            # Default: 3 periods
            duration_ns = 3 * period_s * 1e9
        
        duration_s = duration_ns * 1e-9
        time_array = np.linspace(0, duration_s, n_frames)
        
        # Pre-compute all frames
        logger.info(f"Generating {n_frames} animation frames for {duration_ns:.1f} ns...")
        
        frames = []
        for t in time_array:
            # m(t) = Re[M * exp(-i*ω*t)] or Im[M * exp(-i*ω*t)]
            m_t_complex = self.m_xy * np.exp(-1j * omega * t)
            
            if mode_type == 'real':
                m_t = np.real(m_t_complex)
            elif mode_type == 'imag':
                m_t = np.imag(m_t_complex)
            else:
                raise ValueError(f"Animation mode_type must be 'real' or 'imag', got '{mode_type}'")
            
            frames.append(m_t)
        
        frames = np.array(frames)  # (n_frames, N_y, N_x)
        
        # Auto color limits (global across all frames)
        if vmin is None:
            vmin = frames.min()
        if vmax is None:
            vmax = frames.max()
        
        # Auto colormap
        if cmap is None:
            cmap = 'RdBu_r'
        
        # Auto figsize
        if figsize is None:
            aspect_ratio = self.m_xy.shape[0] / self.m_xy.shape[1]
            figsize = (10, 10 * aspect_ratio * 0.8)
        
        # Extent
        x_min, x_max = self.x.min() * 1e6, self.x.max() * 1e6
        y_min, y_max = self.y.min() * 1e6, self.y.max() * 1e6
        extent = [x_min, x_max, y_min, y_max]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Initial frame
        im = ax.imshow(
            frames[0],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin='lower',
            extent=extent,
            aspect='auto',
            interpolation='nearest',
        )
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Title
        if title_template is None:
            k_um = self.k / 1e6
            f_ghz = self.f / 1e9
            title_template = (
                f"Mode k={k_um:.3f} rad/μm, f={f_ghz:.3f} GHz — t={{t_ns:.2f}} ns"
            )
        
        title = ax.set_title(title_template.format(t_ns=0.0))
        
        # Colorbar
        if colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(f'{mode_type.capitalize()}[M(t)] [a.u.]')
        
        # Animation update function
        def update(frame_idx):
            im.set_data(frames[frame_idx])
            t_ns = time_array[frame_idx] * 1e9
            title.set_text(title_template.format(t_ns=t_ns))
            return [im, title]
        
        # Interval between frames
        if interval is None:
            interval = 1000 / fps  # ms
        
        # Create animation
        anim = FuncAnimation(
            fig,
            update,
            frames=n_frames,
            interval=interval,
            blit=True,
            repeat=repeat,
        )
        
        logger.info(
            f"Animation created: {n_frames} frames, {duration_ns:.1f} ns, "
            f"{fps} fps ({n_frames/fps:.2f}s real-time)"
        )
        
        return anim
    
    def get_components(self) -> dict:
        """
        Get all mode components as dict.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'real': Re[M]
            - 'imag': Im[M]
            - 'abs': |M|
            - 'phase': arg(M)
            - 'complex': M
        
        Examples
        --------
        >>> mode = modes.mode(k=2.3, f=1.12)
        >>> comp = mode.get_components()
        >>> plt.imshow(comp['abs'])
        """
        return {
            'real': np.real(self.m_xy),
            'imag': np.imag(self.m_xy),
            'abs': np.abs(self.m_xy),
            'phase': np.angle(self.m_xy),
            'complex': self.m_xy,
        }
    
    def to_dict(self) -> dict:
        """
        Export to dictionary (legacy format).
        
        Returns
        -------
        dict
            Dictionary with keys: 'x', 'y', 'm_xy', 'k', 'f', 'info'
        """
        return {
            'x': self.x,
            'y': self.y,
            'm_xy': self.m_xy,
            'k': self.k,
            'f': self.f,
            'info': self.info,
        }
