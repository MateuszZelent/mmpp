"""
Core data models for spin-wave dispersion analysis.

Defines result structures and configuration classes for dispersion calculations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Any, Optional, List, Tuple
import numpy as np

if TYPE_CHECKING:
    from ._plotting.accessor import DispersionPlotAccessor
    from .analyze import DispersionAnalyzeAccessor
    from .modes.bridge import DispersionModesBridge


@dataclass
class DispersionConfig:
    """Configuration parameters for dispersion analysis."""
    
    # Time domain processing
    dt: float = 1e-12  # Time step [s] - default value
    time_window: Optional[str] = "hann"  # Window function for time domain
    detrend: str = "mean"  # Detrending method: 'mean', 'initial', None
    
    # Spatial processing  
    dx: Optional[float] = None  # Grid spacing in x [m]
    dy: Optional[float] = None  # Grid spacing in y [m]
    dz: Optional[float] = None  # Grid spacing in z [m] 
    space_window: Optional[str] = None  # Window function for spatial domain
    avg_over_orthogonal: bool = True  # Average over orthogonal directions
    orthogonal_avg_mode: str = "magnetization"  # How to collapse orthogonal axis

    # Component selection
    component: str = "perp"  # 'perp', 'mx', 'my', 'mz', 'sum'
    
    # Brillouin zone folding
    fold_period: Optional[float] = None  # Real-space period [m] for BZ folding
    fold_agg: str = "sum"  # Aggregation method: 'sum' or 'max'

    # Spectral scaling
    scaling: str = "raw_power"  # 'raw_power', 'amplitude_squared', or 'psd'
    
    # Branch tracking
    dk_max: float = 1e5  # Max k-deviation for sampling [rad/m]
    df_max: Optional[float] = None  # Max f-deviation for branch tracking [Hz]
    min_prominence: float = 0.0  # Minimum peak prominence for detection


@dataclass  
class DispersionResult1D:
    """Results from 1D dispersion analysis S(k,f)."""
    
    # Core data
    S: np.ndarray  # Spectral power (Nk, Nf)
    k_axis: np.ndarray  # Wave vector axis [rad/m]
    f_axis: np.ndarray  # Frequency axis [Hz]
    
    # Analysis parameters
    axis: str  # Propagation direction: 'x' or 'y'
    component: str  # Analyzed component
    config: DispersionConfig
    
    # Optional folded data
    S_folded: Optional[np.ndarray] = None  # Folded spectrum
    k_folded: Optional[np.ndarray] = None  # Folded k-axis
    fold_period: Optional[float] = None  # Folding period

    # Optional local spectra when orthogonal averaging is disabled
    # ``S_local`` remains the backward-compatible display alias for local spectra.
    S_local: Optional[np.ndarray] = None  # (N_orthogonal, Nk, Nf)
    S_local_raw: Optional[np.ndarray] = None
    S_local_display: Optional[np.ndarray] = None
    orth_axis: Optional[np.ndarray] = None  # Coordinate values along orthogonal axis
    orth_axis_label: Optional[str] = None  # Name of orthogonal axis ('x' or 'y')
    
    # Complex FFT data for mode reconstruction (avoids re-computing FFT)
    S_complex: Optional[np.ndarray] = None  # Complex spectrum (Nk, Nf) or (N_orth, Nk, Nf)

    # Raw/display separation. ``S`` remains the backward-compatible display alias.
    S_raw: Optional[np.ndarray] = None
    S_display: Optional[np.ndarray] = None
    scaling: str = "raw_power"
    scaling_factors: Optional[Dict[str, float]] = None
    
    # Metadata
    dt: float = 0.0
    dx: float = 0.0
    flipx: bool = True  # Whether k-axis was flipped to correct FFT convention
    notes: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.S_display is None:
            self.S_display = self.S
        else:
            self.S = self.S_display
        if self.S_raw is None:
            self.S_raw = self.S_display
        if self.S_local_display is None:
            self.S_local_display = self.S_local
        else:
            self.S_local = self.S_local_display
        if self.S_local_raw is None:
            self.S_local_raw = self.S_local_display
        self.scaling = str(self.scaling or "raw_power")
        if self.scaling_factors is None:
            self.scaling_factors = {}
        if self.notes is None:
            self.notes = []
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of dispersion array (Nk, Nf)."""
        return self.S.shape
    
    @property 
    def k_range(self) -> Tuple[float, float]:
        """Wave vector range [rad/m]."""
        return (self.k_axis.min(), self.k_axis.max())
        
    @property
    def f_range(self) -> Tuple[float, float]:
        """Frequency range [Hz]."""
        return (self.f_axis.min(), self.f_axis.max())
        
    @property
    def is_folded(self) -> bool:
        """Whether BZ folding was applied."""
        return self.S_folded is not None

    def spectrum_for(self, source: str = "display") -> np.ndarray:
        """Return spectrum data for a named source: ``raw`` or ``display``."""
        source_key = str(source or "display").lower()
        if source_key in {"display", "s", "active"}:
            return self.S_display if self.S_display is not None else self.S
        if source_key == "raw":
            return self.S_raw if self.S_raw is not None else self.S
        raise ValueError("source must be 'raw' or 'display'")
        
    def get_active_data(
        self,
        analysis_source: str = "display",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get currently active S, k, f data (folded if available)."""
        source_key = str(analysis_source or "display").lower()
        if (
            source_key in {"display", "s", "active"}
            and self.is_folded
            and self.S_folded is not None
            and self.k_folded is not None
        ):
            return self.S_folded, self.k_folded, self.f_axis
        return self.spectrum_for(source_key), self.k_axis, self.f_axis

    def frequency_view(
        self,
        *,
        positive_frequencies: bool = True,
        analysis_source: str = "display",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(S, k_axis, f_axis)`` with optional positive-frequency trim."""
        S, k_axis, f_axis = self.get_active_data(analysis_source=analysis_source)
        if not positive_frequencies:
            return S, k_axis, f_axis
        mask = f_axis >= 0
        if mask.sum() == f_axis.size:
            return S, k_axis, f_axis
        return S[:, mask], k_axis, f_axis[mask]
    
    def sample_at_k(
        self,
        k_query: float,
        dk_max: Optional[float] = None,
        *,
        analysis_source: str = "raw",
    ) -> Tuple[float, float]:
        """Sample dispersion at given k, return (k_eff, f_peak)."""
        if dk_max is None:
            dk_max = self.config.dk_max
            
        S, k_axis, f_axis = self.get_active_data(analysis_source=analysis_source)
        
        mask = np.abs(k_axis - k_query) <= dk_max
        if not np.any(mask):
            raise ValueError(f"No k within {dk_max} of {k_query}")
            
        S_slice = S[mask, :].sum(axis=0)
        idx = np.argmax(S_slice)
        f_peak = f_axis[idx]
        k_eff = float(np.average(k_axis[mask], weights=S[mask, idx] + 1e-12))
        
        return k_eff, float(f_peak)

    def select_orthogonal_slice(self, index: int) -> "DispersionResult1D":
        """Create a new result containing a single orthogonal slice."""
        if self.S_local is None:
            raise ValueError("No orthogonal slices stored; recompute with avg_over_orthogonal=False")
        if index < 0 or index >= self.S_local.shape[0]:
            raise IndexError(f"Orthogonal index {index} out of bounds (0..{self.S_local.shape[0]-1})")

        slice_notes = list(self.notes or []) + [f"Orthogonal slice #{index}"]

        # Extract per-slice S_complex if available
        slice_S_complex = None
        if self.S_complex is not None and self.S_complex.ndim == 3:
            slice_S_complex = self.S_complex[index]  # (Nk, Nf)

        local_display = (
            self.S_local_display
            if self.S_local_display is not None
            else self.S_local
        )
        local_raw = self.S_local_raw if self.S_local_raw is not None else local_display

        slice_result = DispersionResult1D(
            S=local_display[index],
            k_axis=self.k_axis,
            f_axis=self.f_axis,
            axis=self.axis,
            component=self.component,
            config=self.config,
            S_folded=self.S_folded,
            k_folded=self.k_folded,
            fold_period=self.fold_period,
            S_complex=slice_S_complex,
            S_raw=local_raw[index] if local_raw is not None else None,
            S_display=local_display[index] if local_display is not None else None,
            orth_axis_label=self.orth_axis_label,
            dt=self.dt,
            dx=self.dx,
            flipx=self.flipx,
            scaling=self.scaling,
            scaling_factors=dict(self.scaling_factors or {}),
            notes=slice_notes,
        )
        return slice_result

    # ------------------------------------------------------------------
    # New accessor namespaces  (spectrum/modes architecture)
    # ------------------------------------------------------------------

    @property
    def plot(self) -> "DispersionPlotAccessor":
        """Plotting namespace: ``.plot.heatmap()``, ``.plot.branch(branch)``."""
        from ._plotting.accessor import DispersionPlotAccessor
        return DispersionPlotAccessor(self)

    @property
    def analyze(self) -> "DispersionAnalyzeAccessor":
        """Analysis namespace: ``.analyze.find_lowest_possible_frequency()``."""
        from .analyze import DispersionAnalyzeAccessor
        return DispersionAnalyzeAccessor(self)

    @property
    def modes(self) -> "DispersionModesBridge":
        """Modes bridge: ``.modes.interactive()``, ``.modes.at(k, f)``."""
        from .modes.bridge import DispersionModesBridge
        return DispersionModesBridge(self)

    def filtered(self, live: Optional[Dict[str, Any]] = None, **kwargs) -> "DispersionResult1D":
        """Return a new :class:`DispersionResult1D` with *live* post-filters applied.

        Non-destructive – original data is never modified.

        Parameters
        ----------
        live : dict, optional
            Live-filter configuration dict (same format as ``.filters(live=...)``).
        **kwargs
            Additional filter keyword arguments forwarded to the filter engine.

        Returns
        -------
        DispersionResult1D
            New instance with filtered ``S``.
        """
        import copy

        if live is None and not kwargs:
            return self

        S_base = self.S_display if self.S_display is not None else self.S
        S_new = S_base.copy()

        if live:
            try:
                from .utils import apply_dispersion_post_filters

                S_new = apply_dispersion_post_filters(
                    S_new,
                    k_axis=self.k_axis,
                    f_axis=self.f_axis,
                    filters={"live": live},
                    include_live=True,
                )
            except Exception:
                pass  # degrade gracefully

        new_result = copy.copy(self)
        object.__setattr__(new_result, "S", S_new)
        object.__setattr__(new_result, "S_display", S_new)
        object.__setattr__(new_result, "S_raw", self.S_raw)
        return new_result

    # ------------------------------------------------------------------
    # Jupyter repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        nk, nf = self.S.shape
        kmin = self.k_axis.min() / 1e6
        kmax = self.k_axis.max() / 1e6
        fmin = self.f_axis[self.f_axis >= 0].min() / 1e9 if (self.f_axis >= 0).any() else 0.0
        fmax = self.f_axis.max() / 1e9
        return (
            f"DispersionResult1D(axis={self.axis!r}, component={self.component!r}, "
            f"shape=({nk}, {nf}), k=[{kmin:.2f}..{kmax:.2f}] rad/\u03bcm, "
            f"f=[{fmin:.2f}..{fmax:.2f}] GHz)"
        )

    def _repr_html_(self) -> str:
        from html import escape as _esc
        import numpy as _np

        nk, nf = self.S.shape
        kmin = float(self.k_axis.min()) / 1e6
        kmax = float(self.k_axis.max()) / 1e6
        f_pos = self.f_axis[self.f_axis >= 0]
        fmin = float(f_pos.min()) / 1e9 if f_pos.size else 0.0
        fmax_val = float(self.f_axis.max()) / 1e9
        has_local = self.S_local is not None
        has_complex = self.S_complex is not None
        has_folded = self.S_folded is not None
        smax = float(self.S.max())

        HV = "onmouseover=\"this.style.background='#1e293b'\" onmouseout=\"this.style.background='transparent'\""

        def stat_row(k, v, tip=""):
            t = f" title=\"{_esc(tip)}\"" if tip else ""
            return (
                f"<tr {HV}{t}>"
                f"<td style='padding:2px 10px;color:#94a3b8;font-size:.85em;'>{_esc(k)}</td>"
                f"<td style='padding:2px 10px;color:#a5b4fc;font-weight:600;font-family:monospace;'>{_esc(str(v))}</td>"
                f"</tr>"
            )

        def method_row(sig, desc, tip=""):
            t = f" title=\"{_esc(tip)}\"" if tip else ""
            return (
                f"<tr {HV}{t} style='cursor:pointer;'>"
                f"<td style='padding:3px 10px;font-family:monospace;color:#93c5fd;font-size:.88em;'>{_esc(sig)}</td>"
                f"<td style='padding:3px 10px;color:#cbd5e1;font-size:.85em;'>{_esc(desc)}</td>"
                f"</tr>"
            )

        flags = []
        if has_local:
            n_orth = self.S_local.shape[0]
            flags.append(f"S_local ({n_orth} slices)")
        if has_complex:
            flags.append("S_complex")
        if has_folded:
            flags.append("S_folded")
        flags_html = "".join(
            f"<span style='background:#1e3a5f;color:#7dd3fc;border-radius:4px;"
            f"padding:1px 6px;font-size:.75em;margin-right:4px;'>{_esc(f)}</span>"
            for f in flags
        ) if flags else "<span style='color:#475569;font-size:.8em;'>—</span>"

        stats_html = (
            "<table style='border-collapse:collapse;width:100%;margin-bottom:8px;'>"
            + stat_row("axis", self.axis, "Propagation direction for k-space decomposition")
            + stat_row("component", self.component, "Magnetization component used in FFT")
            + stat_row("shape", f"({nk} k-bins, {nf} f-bins)", "Size of S(k,f) array")
            + stat_row("k range", f"{kmin:.3f} … {kmax:.3f} rad/\u03bcm", "Wave-vector axis extent")
            + stat_row("f range", f"{fmin:.3f} … {fmax_val:.3f} GHz", "Frequency axis extent (positive half shown)")
            + stat_row("S_max", f"{smax:.4g}", "Maximum spectral density value")
            + "</table>"
        )

        def section(label, color, badge, rows_html, tip="", open_=False):
            op = " open" if open_ else ""
            return (
                f"<details{op} style='margin:4px 0;'>"
                f"<summary style='cursor:pointer;padding:4px 6px;border-radius:6px;"
                f"background:#1e293b;color:{color};font-family:monospace;font-size:.88em;"
                f"list-style:none;display:flex;align-items:center;gap:8px;'"
                f" title=\"{_esc(tip)}\">"
                f"<span style='color:#475569;'>&#9654;</span>"
                f"<span>{_esc(label)}</span>"
                f"<span style='background:{color}22;color:{color};border-radius:4px;"
                f"padding:0px 6px;font-size:.75em;margin-left:auto;'>{_esc(badge)}</span>"
                f"</summary>"
                f"<div style='margin-left:16px;margin-top:4px;'>"
                f"<table style='border-collapse:collapse;width:100%;'>{rows_html}</table>"
                f"</div>"
                f"</details>"
            )

        plot_rows = (
            method_row(".plot.heatmap(fmax=10, lognorm=True)",
                       "S(k,f) power heatmap",
                       "Plot spin-wave dispersion as a 2D heatmap. fmax clips the frequency axis. lognorm uses logarithmic color scale.")
            + method_row(".plot.heatmap(orth_index=0)",
                         "Single orthogonal slice heatmap",
                         "Show S(k,f) for one y-slice only (requires avg_over_orthogonal=False).")
            + method_row(".plot.branch(branch)",
                         "Dispersion branch + group velocity",
                         "Plot a tracked DispersionBranch: frequency vs k on the left, group velocity dω/dk on the right.")
        )

        analyze_rows = (
            method_row(".analyze.find_lowest_possible_frequency()",
                       "→ LowestFrequencyResult",
                       "Find the true minimum frequency on the branch — for backward-volume SW it occurs at k>0, not at k=0.")
            + method_row(".analyze.find_lowest_possible_frequency(side='both', smooth_sigma=2.0)",
                         "search both k halves, with Gaussian smoothing",
                         "side='both' searches full k-axis. smooth_sigma applies Gaussian smoothing to f_peak(k) before argmin.")
        )

        modes_rows = (
            method_row(".modes.interactive(lattice_constant_nm=470)",
                       "Open interactive dispersion-mode widget",
                       "Opens ipywidgets-based interactive explorer. Click on S(k,f) to see the spatial mode profile m(x,y).")
            + method_row(".modes.at(k_rad_um=2.3, f_ghz=5.0)",
                         "→ DispersionModeResult",
                         "Extract mode image at a specific (k*, f*) point. Requires S_complex to be stored.")
            + method_row(".modes.at(...).plot.imshow()",
                         "Mode spatial profile |ψ(x,y)|",
                         "Show the reconstructed spin-wave mode amplitude. mode_type: abs | real | imag | phase.")
        )

        filtered_rows = method_row(
            ".filtered(live={'gaussian_morph': {'enabled': True, 'sigma_f': 1.0}})",
            "→ new DispersionResult1D",
            "Non-destructive: applies live post-filters to S(k,f) and returns a new result. Original data unchanged."
        )

        breadcrumb = (
            "<div style='font-size:.78em;color:#475569;margin-bottom:8px;font-family:monospace;'>"
            "fft.dispersion "
            "<span style='color:#334155;'>›</span> "
            ".filters() "
            "<span style='color:#334155;'>›</span> "
            "<span style='color:#7dd3fc;font-weight:600;'>.compute_1d()</span>"
            "</div>"
        )

        return (
            "<div style='font-family:-apple-system,BlinkMacSystemFont,sans-serif;"
            "border:2px solid #2563eb;border-radius:10px;padding:14px;margin:6px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#0c1a35 100%);color:#e2e8f0;max-width:720px;'>"
            + breadcrumb
            + "<div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;'>"
            + "<span style='font-weight:700;font-size:1.05em;color:#f1f5f9;'>DispersionResult1D</span>"
            + "<span style='background:#1d4ed8;color:#bfdbfe;border-radius:5px;padding:1px 8px;font-size:.75em;'>S(k,f)</span>"
            + f"<span style='margin-left:auto;font-size:.8em;color:#475569;padding:1px 8px;border:1px solid #334155;border-radius:4px;'>axis={_esc(self.axis)}</span>"
            + "</div>"
            + stats_html
            + "<div style='font-size:.78em;color:#64748b;margin:4px 0 6px 2px;'>Optional stored arrays: "
            + flags_html
            + "</div>"
            + section(".plot", "#60a5fa", "DispersionPlotAccessor", plot_rows,
                      "Plotting namespace: S(k,f) heatmap and dispersion branch visualization.",
                      open_=True)
            + section(".analyze", "#34d399", "DispersionAnalyzeAccessor", analyze_rows,
                      "Analysis tools: find the true minimum frequency on the dispersion branch.")
            + section(".modes", "#f59e0b", "DispersionModesBridge", modes_rows,
                      "Mode extraction: interactive widget or single-point mode profile m(x,y).")
            + section(".filtered(...)", "#a78bfa", "non-destructive", filtered_rows,
                      "Apply live/post filters to S(k,f) without recomputing FFT. Returns new DispersionResult1D.")
            + "</div>"
        )


@dataclass
class DispersionResult2D:
    """Results from 2D dispersion analysis S(kx,ky,f)."""
    
    # Core data
    S: np.ndarray  # Spectral power (Nkx, Nky, Nf)
    kx_axis: np.ndarray  # kx wave vector axis [rad/m]
    ky_axis: np.ndarray  # ky wave vector axis [rad/m] 
    f_axis: np.ndarray  # Frequency axis [Hz]
    
    # Analysis parameters
    component: str  # Analyzed component
    config: DispersionConfig
    
    # Metadata
    dt: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    notes: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.notes is None:
            self.notes = []
            
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of dispersion array (Nkx, Nky, Nf)."""
        return self.S.shape
        
    @property
    def kx_range(self) -> Tuple[float, float]:
        """kx range [rad/m]."""
        return (self.kx_axis.min(), self.kx_axis.max())
        
    @property  
    def ky_range(self) -> Tuple[float, float]:
        """ky range [rad/m]."""
        return (self.ky_axis.min(), self.ky_axis.max())
        
    @property
    def f_range(self) -> Tuple[float, float]:
        """Frequency range [Hz]."""
        return (self.f_axis.min(), self.f_axis.max())
        
    def slice_1d(self, direction: str, k_value: float = 0.0, dk_max: Optional[float] = None) -> DispersionResult1D:
        """Extract 1D slice along kx or ky direction."""
        if dk_max is None:
            dk_max = self.config.dk_max
            
        if direction == 'kx':
            # Slice along kx at fixed ky
            mask = np.abs(self.ky_axis - k_value) <= dk_max
            if not np.any(mask):
                raise ValueError(f"No ky within {dk_max} of {k_value}")
            S_1d = self.S[:, mask, :].mean(axis=1)  # Average over ky slice
            k_axis = self.kx_axis
            dx = self.dx
            axis = 'x'
            
        elif direction == 'ky':
            # Slice along ky at fixed kx  
            mask = np.abs(self.kx_axis - k_value) <= dk_max
            if not np.any(mask):
                raise ValueError(f"No kx within {dk_max} of {k_value}")
            S_1d = self.S[mask, :, :].mean(axis=0)  # Average over kx slice
            k_axis = self.ky_axis  
            dx = self.dy
            axis = 'y'
        else:
            raise ValueError("direction must be 'kx' or 'ky'")
            
        return DispersionResult1D(
            S=S_1d,
            k_axis=k_axis,
            f_axis=self.f_axis,
            axis=axis,
            component=self.component,
            config=self.config,
            dt=self.dt,
            dx=dx,
            scaling=getattr(self, "scaling", "raw_power"),
            scaling_factors=dict(getattr(self, "scaling_factors", {}) or {}),
            notes=(self.notes or []) + [f"1D slice from 2D at {direction}={k_value}"]
        )


@dataclass
class DispersionBranch:
    """A tracked dispersion branch f(k)."""
    
    # Branch data
    k_path: np.ndarray  # Wave vector path [rad/m]
    f_values: np.ndarray  # Frequencies [Hz] 
    amplitudes: np.ndarray  # Spectral amplitudes at (k,f) points
    
    # Branch properties  
    branch_id: int = 0  # Branch identifier
    mode_type: Optional[str] = None  # Mode classification
    group_velocity: Optional[np.ndarray] = None  # dω/dk [Hz⋅m]
    
    # Analysis metadata
    tracking_config: Optional[Dict[str, Any]] = None
    notes: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.tracking_config is None:
            self.tracking_config = {}
        if self.notes is None:
            self.notes = []
            
    @property
    def length(self) -> int:
        """Number of points in branch."""
        return len(self.k_path)
        
    @property  
    def k_range(self) -> Tuple[float, float]:
        """Wave vector range [rad/m]."""
        return (self.k_path.min(), self.k_path.max())
        
    @property
    def f_range(self) -> Tuple[float, float]:
        """Frequency range [Hz]."""
        return (self.f_values.min(), self.f_values.max())
        
    def compute_group_velocity(self, smooth: bool = True) -> np.ndarray:
        """Compute group velocity dω/dk = 2π⋅df/dk."""
        if smooth:
            # Use gradient with smoothing
            vg = 2 * np.pi * np.gradient(self.f_values, self.k_path)
        else:
            # Simple finite differences with edge handling
            vg = 2 * np.pi * np.gradient(self.f_values, self.k_path)

        self.group_velocity = vg
        return vg
        
    def interpolate_at_k(self, k_query: np.ndarray) -> np.ndarray:
        """Interpolate branch frequencies at query k values.""" 
        return np.interp(k_query, self.k_path, self.f_values)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'k_path': self.k_path.tolist(),
            'f_values': self.f_values.tolist(), 
            'amplitudes': self.amplitudes.tolist(),
            'branch_id': self.branch_id,
            'mode_type': self.mode_type,
            'group_velocity': self.group_velocity.tolist() if self.group_velocity is not None else None,
            'tracking_config': self.tracking_config or {},
            'notes': self.notes or [],
            'k_range': self.k_range,
            'f_range': self.f_range
        }
