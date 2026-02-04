"""
Plotting utilities for InteractiveDispersionModes.

Handles matplotlib visualization of dispersion and mode profiles.
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    plt = None  # type: ignore

if TYPE_CHECKING:
    from ...models import DispersionResult1D

logger = logging.getLogger(__name__)


class InteractivePlotter:
    """Handles matplotlib plotting for interactive dispersion visualization."""
    
    def __init__(self, result: Dispersion Result1D):
        """
        Initialize plotter with dispersion result.
        
        Parameters
        ----------
        result : DispersionResult1D
            Dispersion result to visualize.
        """
        if not _HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")
        
        self.result = result
        self.fig: Figure | None = None
        self.ax_disp: Axes | None = None
        self.ax_mode: Axes | None = None
        self.colorbar_disp = None
        self.colorbar_mode = None
    
    def initialize_figure(
        self,
        figsize: tuple[float, float] = (10, 10),
        dpi: int = 150,
    ) -> tuple[Figure, Axes, Axes]:
        """
        Create matplotlib figure with dispersion and mode subplots.
        
        Parameters
        ----------
        figsize : tuple
            Figure size (width, height)
        dpi : int
            Figure DPI
            
        Returns
        -------
        fig : Figure
            Matplotlib figure
        ax_disp : Axes
            Dispersion subplot axes
        ax_mode : Axes
            Mode visualization subplot axes
        """
        plt.ioff()
        
        # Create figure with 2 rows: dispersion on top, mode viz below
        self.fig, (self.ax_disp, self.ax_mode) = plt.subplots(
            2,
            1,
            figsize=figsize,
            dpi=dpi,
            gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.25},
        )
        
        plt.ion()
        
        return self.fig, self.ax_disp, self.ax_mode
    
    def update_dispersion_plot(
        self,
        ax: Axes,
        f_min: float,
        f_max: float,
        lattice_nm: float,
        cmap: str = "viridis",
        selected_k: float | None = None,
        selected_f: float | None = None,
    ) -> None:
        """
        Update the dispersion heatmap.
        
        Parameters
        ----------
        ax : Axes
            Matplotlib axes for dispersion plot
        f_min : float
            Minimum frequency in GHz
        f_max : float
            Maximum frequency in GHz
        lattice_nm : float
            Lattice constant in nm (for FBZ lines)
        cmap : str
            Colormap name
        selected_k : float, optional
            Selected k-value in rad/m for marker
        selected_f : float, optional
            Selected frequency in Hz for marker
        """
        # Clear axes
        ax.clear()
        
        # Remove old colorbar
        if self.colorbar_disp is not None:
            try:
                self.colorbar_disp.remove()
            except Exception:
                pass
            self.colorbar_disp = None
        
        # Get data
        S = self.result.S.T  # (Nf, Nk)
        k_axis = self.result.k_axis / 1e6  # rad/μm
        f_axis = self.result.f_axis / 1e9  # GHz
        
        # Apply frequency limits
        f_mask = (f_axis >= f_min) & (f_axis <= f_max)
        
        if np.sum(f_mask) < 2:
            ax.text(0.5, 0.5, "No data in frequency range", 
                   transform=ax.transAxes, ha="center")
            return
        
        S = S[f_mask, :]
        f_axis_plot = f_axis[f_mask]
        
        extent = [k_axis[0], k_axis[-1], f_axis_plot[0], f_axis_plot[-1]]
        
        # Plot heatmap
        im = ax.imshow(
            np.log10(S + 1e-20),
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=cmap,
            interpolation="bilinear",
        )
        
        self.colorbar_disp = self.fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        self.colorbar_disp.set_label("log₁₀(S)", fontsize=9)
        
        # Add FBZ lines
        a = lattice_nm * 1e-9
        k_bz = np.pi / a / 1e6  # rad/μm
        ax.axvline(-k_bz, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axvline(k_bz, color="red", linestyle="--", linewidth=1.5, alpha=0.7, 
                  label=f"±π/a = ±{k_bz:.1f}")
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5, linewidth=1)
        ax.legend(loc="upper right", fontsize=8)
        
        # Labels
        ax.set_xlabel(r"$k$ [rad/μm]", fontsize=10)
        ax.set_ylabel("f [GHz]", fontsize=10)
        ax.set_title(f"Dispersion S(k, f) | a = {a*1e9:.0f} nm | Click to select mode", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.tick_params(labelsize=9)
        
        # Set k-axis limits to ±2 BZ
        k_limit = 2 * k_bz
        ax.set_xlim(-k_limit, k_limit)
        
        # Draw selection marker if exists
        if selected_k is not None and selected_f is not None:
            self.draw_selection_markers(
                ax=ax,
                k_sel=selected_k,
                f_sel=selected_f,
                lattice_nm=lattice_nm,
                n_bz=3,  # This should be passed as parameter
                k_direction="both",  # This should be passed as parameter
            )
    
    def draw_selection_markers(
        self,
        ax: Axes,
        k_sel: float,
        f_sel: float,
        lattice_nm: float,
        n_bz: int,
        k_direction: str,
    ) -> int:
        """
        Draw markers showing selected (k, f) and BZ mask positions.
        
        Parameters
        ----------
        ax : Axes
            Matplotlib axes
        k_sel : float
            Selected k in rad/m
        f_sel : float
            Selected f in Hz
        lattice_nm : float
            Lattice constant in nm
        n_bz : int
            Number of BZ in mask
        k_direction : str
            Direction filter ('both', 'positive', 'negative')
            
        Returns
        -------
        int
            Number of mask positions drawn
        """
        k_sel_um = k_sel / 1e6  # rad/μm
        f_sel_ghz = f_sel / 1e9  # GHz
        
        a = lattice_nm * 1e-9
        G = 2 * np.pi / a / 1e6  # rad/μm
        
        # Draw main selection marker (red square)
        ax.plot(
            k_sel_um,
            f_sel_ghz,
            "rs",
            markersize=12,
            markerfacecolor="none",
            markeredgewidth=2,
            label="Selected",
        )
        
        # Draw all mask positions as circles
        k_axis = self.result.k_axis / 1e6
        k_min, k_max = k_axis.min(), k_axis.max()
        
        count = 0
        for n in range(-n_bz, n_bz + 1):
            if n == 0:
                continue
            
            k_copy = k_sel_um + n * G
            
            # Check k-direction filter
            if k_direction == "positive" and k_copy < 0:
                continue
            if k_direction == "negative" and k_copy > 0:
                continue
            
            # Check if within k-axis range
            if k_min <= k_copy <= k_max:
                ax.plot(
                    k_copy,
                    f_sel_ghz,
                    "o",
                    markersize=8,
                    markerfacecolor="none",
                    markeredgecolor="lime",
                    markeredgewidth=1.5,
                )
                
                # Draw connecting line
                ax.plot(
                    [k_sel_um, k_copy],
                    [f_sel_ghz, f_sel_ghz],
                    "g--",
                    linewidth=0.8,
                    alpha=0.5,
                )
                count += 1
        
        return count + 1  # Include main selection
    
    def update_mode_visualization(
        self,
        ax: Axes,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        mode_2d: np.ndarray,
        k_sel: float,
        f_sel: float,
        cmap: str = "RdBu_r",
    ) -> None:
        """
        Update the 2D spatial mode visualization.
        
        Parameters
        ----------
        ax : Axes
            Matplotlib axes for mode plot
        x_axis : ndarray
            x-axis in meters
        y_axis : ndarray
            y-axis in meters
        mode_2d : ndarray
            2D mode profile m(x, y)
        k_sel : float
            Selected k in rad/m
        f_sel : float
            Selected f in Hz
        cmap : str
            Colormap for mode visualization
        """
        ax.clear()
        
        # Remove old colorbar
        if self.colorbar_mode is not None:
            try:
                self.colorbar_mode.remove()
            except Exception:
                pass
            self.colorbar_mode = None
        
        # Convert axes to μm
        x_um = x_axis * 1e6
        y_um = y_axis * 1e6
        
        # Plot 2D spatial heatmap
        extent = [x_um[0], x_um[-1], y_um[0], y_um[-1]]
        
        # Use symmetric colormap for real part
        vmax = np.max(np.abs(mode_2d))
        if vmax < 1e-20:
            vmax = 1.0
        vmin = -vmax
        
        im = ax.imshow(
            mode_2d,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
        )
        
        self.colorbar_mode = self.fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        self.colorbar_mode.set_label("Re[m]", fontsize=9)
        
        # Labels
        ax.set_xlabel("x [μm]", fontsize=10)
        ax.set_ylabel("y [μm]", fontsize=10)
        
        k_str = f"k = {k_sel/1e6:.2f} rad/μm"
        f_str = f"f = {f_sel/1e9:.2f} GHz"
        ax.set_title(f"Mode Profile m(x, y) | {k_str}, {f_str}", fontsize=11)
        ax.tick_params(labelsize=9)
