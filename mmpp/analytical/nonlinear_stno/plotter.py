"""
mmpp.analytical.nonlinear_stno.plotter
========================================
Publication-quality rendering for STNO spectroscopy dashboards.

:class:`DashboardPlotter` wraps all Matplotlib logic so that the calling
script only needs to pass pre-computed data arrays and sweep metadata.

Example
-------
>>> from mmpp.analytical.nonlinear_stno import DashboardPlotter
>>> plotter = DashboardPlotter()
>>> plotter.plot_2x2(
...     f_axis, map_Jac, map_fmod, map_Jdc, map_Field,
...     sweeps, theory_lines,
...     output_name="Fig_STNO.png",
... )
"""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Any, Dict, Optional


class DashboardPlotter:
    """Render a 2×2 STNO spectrogram dashboard.

    The four panels show PSD maps as a function of:
        (a) Modulation amplitude  J_AC
        (b) Modulation frequency  f_mod
        (c) DC bias current       J_DC
        (d) External field        µ₀H_z

    Attributes
    ----------
    cmap : str
        Matplotlib colormap name (default ``'magma'``).
    vmin, vmax : float
        PSD colour-scale limits in dB.
    dpi : int
        Figure resolution.
    """

    cmap: str = "magma"
    vmin: float = -75.0
    vmax: float = 5.0
    dpi: int = 150

    # ------------------------------------------------------------------ #
    # Style helpers                                                       #
    # ------------------------------------------------------------------ #

    def setup_publication_style(self) -> None:
        """Apply publication-quality Matplotlib rcParams."""
        plt.rcParams.update(
            {
                "font.family": "serif",
                "axes.labelsize": 12,
                "axes.titlesize": 13,
                "legend.fontsize": 9,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "figure.dpi": self.dpi,
                "savefig.bbox": "tight",
            }
        )

    # ------------------------------------------------------------------ #
    # Main plot                                                           #
    # ------------------------------------------------------------------ #

    def plot_2x2(
        self,
        f_axis: np.ndarray,
        map_Jac: np.ndarray,
        map_fmod: np.ndarray,
        map_Jdc: np.ndarray,
        map_Field: np.ndarray,
        sweeps: Dict[str, np.ndarray],
        theory: Dict[str, Any],
        output_name: str = "Fig_STNO.png",
        show: bool = False,
    ) -> matplotlib.figure.Figure:
        """Compose and save the 4-panel STNO spectrogram figure.

        Parameters
        ----------
        f_axis : np.ndarray, shape (n_freqs,)
            Frequency axis [GHz].
        map_Jac, map_fmod, map_Jdc, map_Field : np.ndarray, shape (n_freqs, n_sweep)
            Transposed PSD arrays (frequency on y-axis) for each sweep.
        sweeps : dict
            Dictionary with keys ``'jac'``, ``'fmod'``, ``'jdc'``, ``'field'``
            holding the 1-D sweep arrays (SI units: A/m², A/m², A/m², T).
        theory : dict
            Overlay curves. Expected keys (all optional):

            * ``'f_sw_base'``  – float, base spin-wave frequency [GHz]
            * ``'fG_static'``  – float, static carrier frequency [GHz]
            * ``'fG_field'``   – 1-D array, field-tunable carrier [GHz]
            * ``'fSW_field'``  – 1-D array, field-tunable spin-wave [GHz]

        output_name : str
            File path for the saved figure.
        show : bool
            If True, call ``plt.show()`` after saving.

        Returns
        -------
        matplotlib.figure.Figure
            The composed figure object.
        """
        self.setup_publication_style()

        fig, axs = plt.subplots(2, 2, figsize=(13, 10))
        vmin, vmax, cmap = self.vmin, self.vmax, self.cmap

        # ---- Panel (a): J_AC sweep ----------------------------------------
        ext1 = [
            sweeps["jac"][0] / 1e10,
            sweeps["jac"][-1] / 1e10,
            f_axis[0],
            f_axis[-1],
        ]
        im1 = axs[0, 0].imshow(
            map_Jac,
            aspect="auto",
            origin="lower",
            extent=ext1,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axs[0, 0].set_title(
            r"(a) Kinematic Mach Shock vs $J_{AC}$", fontweight="bold"
        )
        axs[0, 0].set_xlabel(
            r"Modulation Amplitude $J_{AC}$ [$10^{10}$ A/m$^2$]"
        )
        axs[0, 0].set_ylabel("Frequency [GHz]")
        if "f_sw_base" in theory:
            axs[0, 0].axhline(
                theory["f_sw_base"],
                color="cyan",
                linestyle=":",
                alpha=0.8,
                lw=1.5,
                label="Spin-Wave Band",
            )
            axs[0, 0].legend(loc="lower right", framealpha=0.2)

        # ---- Panel (b): f_mod sweep ----------------------------------------
        ext2 = [
            sweeps["fmod"][0] / 1e6,
            sweeps["fmod"][-1] / 1e6,
            f_axis[0],
            f_axis[-1],
        ]
        axs[0, 1].imshow(
            map_fmod,
            aspect="auto",
            origin="lower",
            extent=ext2,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axs[0, 1].set_title(
            r"(b) Coherent Fractional Phase-Locking vs $f_{mod}$", fontweight="bold"
        )
        axs[0, 1].set_xlabel(r"Modulation Frequency $f_{mod}$ [MHz]")
        axs[0, 1].set_ylabel("Frequency [GHz]")
        if "fG_static" in theory:
            axs[0, 1].axhline(
                theory["fG_static"],
                color="white",
                linestyle="--",
                alpha=0.8,
                lw=1.5,
                label=r"Static $f_G$",
            )
            axs[0, 1].legend(
                loc="upper right", framealpha=0.2, labelcolor="white"
            )

        # ---- Panel (c): J_DC sweep ----------------------------------------
        ext3 = [
            sweeps["jdc"][0] / 1e10,
            sweeps["jdc"][-1] / 1e10,
            f_axis[0],
            f_axis[-1],
        ]
        axs[1, 0].imshow(
            map_Jdc,
            aspect="auto",
            origin="lower",
            extent=ext3,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axs[1, 0].set_title(
            r"(c) Free-Running Core Tunability vs $J_{DC}$", fontweight="bold"
        )
        axs[1, 0].set_xlabel(r"Base Current $J_{DC}$ [$10^{10}$ A/m$^2$]")
        axs[1, 0].set_ylabel("Frequency [GHz]")

        # ---- Panel (d): Field sweep ----------------------------------------
        ext4 = [
            sweeps["field"][0] * 1000,
            sweeps["field"][-1] * 1000,
            f_axis[0],
            f_axis[-1],
        ]
        axs[1, 1].imshow(
            map_Field,
            aspect="auto",
            origin="lower",
            extent=ext4,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axs[1, 1].set_title(
            r"(d) Free-Running Topological Dispersion vs $\mu_0 H_z$",
            fontweight="bold",
        )
        axs[1, 1].set_xlabel(r"External Magnetic Field $\mu_0 H_z$ [mT]")
        axs[1, 1].set_ylabel("Frequency [GHz]")

        if "fG_field" in theory:
            axs[1, 1].plot(
                sweeps["field"] * 1000,
                theory["fG_field"],
                "w--",
                lw=1.5,
                alpha=0.8,
                label=r"$f_G(H)$ FLT Theory",
            )
        if "fSW_field" in theory:
            axs[1, 1].plot(
                sweeps["field"] * 1000,
                theory["fSW_field"],
                "cyan",
                linestyle=":",
                lw=1.5,
                alpha=0.9,
                label=r"$f_{SW}(H)$ Theory",
            )
        if "fG_field" in theory or "fSW_field" in theory:
            axs[1, 1].legend(
                loc="upper left", framealpha=0.2, labelcolor="white"
            )

        # ---- Shared colorbar ----------------------------------------------
        plt.tight_layout()
        cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
        fig.colorbar(im1, cax=cbar_ax, label="Power Spectral Density [dB]")

        plt.savefig(output_name, bbox_inches="tight")
        print(f"[nonlinear_stno] Figure saved → {output_name}")

        if show:
            plt.show()

        return fig

    # ------------------------------------------------------------------ #
    # Convenience: single-panel spectrogram                               #
    # ------------------------------------------------------------------ #

    def plot_spectrogram(
        self,
        f_axis: np.ndarray,
        sweep_axis: np.ndarray,
        psd_map: np.ndarray,
        xlabel: str = "Sweep parameter",
        title: str = "STNO Spectrogram",
        output_name: Optional[str] = None,
        show: bool = True,
        ax: Optional[matplotlib.axes.Axes] = None,
    ) -> matplotlib.axes.Axes:
        """Quick single-panel spectrogram plot.

        Parameters
        ----------
        f_axis : np.ndarray
            Frequency axis [GHz].
        sweep_axis : np.ndarray
            Horizontal sweep parameter (arbitrary units – label with *xlabel*).
        psd_map : np.ndarray, shape (n_freqs, n_sweep)
            Transposed PSD array.
        xlabel : str
        title : str
        output_name : str or None
            Save path (no save if None).
        show : bool
        ax : Axes or None
            Existing axes; a new figure is created when None.

        Returns
        -------
        matplotlib.axes.Axes
        """
        self.setup_publication_style()

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 5))

        ext = [sweep_axis[0], sweep_axis[-1], f_axis[0], f_axis[-1]]
        im = ax.imshow(
            psd_map,
            aspect="auto",
            origin="lower",
            extent=ext,
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency [GHz]")
        ax.set_title(title, fontweight="bold")
        plt.colorbar(im, ax=ax, label="PSD [dB]")

        if output_name is not None:
            plt.savefig(output_name, bbox_inches="tight")
        if show:
            plt.show()

        return ax
