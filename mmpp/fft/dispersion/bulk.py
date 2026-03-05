"""Bulk / sweep analysis for spin-wave dispersion minimum frequency.

The core problem: storing full S(k,f) for dozens of simulations is expensive
(each array can be ~10–100 MB).  This module extracts only what is needed:

* Scalar summary per job: f_min, k*, vg(k*), f(k=0), Δf
* 1-D cross-section  : S(k) evaluated at f ≈ f_min  →  memory tiny
* 1-D branch         : f_peak(k)                     →  memory tiny

These are collected into a :class:`BulkMinimumFrequencyResult` that
supports heatmaps, line plots, and serialisation.

Typical usage::

    from mmpp.fft.dispersion.bulk import scan_minimum_frequency

    jobs = [mmpp.MMPP(path) for path in zarr_paths]

    bulk = scan_minimum_frequency(
        jobs,
        param_values=[0, 10, 20, 30, 40],     # e.g. field [mT]
        param_label="B_ext [mT]",
        filters=dict(remove_static=True, live={"gaussian_morph": {"enabled": True}}),
        find_kwargs=dict(side="positive", smooth_sigma=2.0),
        compute_kwargs=dict(axis="x", component="perp"),
    )

    bulk.plot.heatmap()            # S(k) at f_min vs param
    bulk.plot.f_min_vs_param()     # f_min(param) line plot
    bulk.plot.branches()           # all f_peak(k) curves stacked

Or from a single result via the .analyze accessor::

    bulk = result.analyze.scan([result1, result2, ...], params=[0, 10], ...)
"""

from __future__ import annotations

import gc
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from .models import DispersionResult1D
    from .analyze import LowestFrequencyResult, FindLowestKwargs


# ---------------------------------------------------------------------------
# Low-level extractor (memory-efficient)
# ---------------------------------------------------------------------------

def _extract_compact(
    result: "DispersionResult1D",
    find_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Compute LowestFrequencyResult and extract compact data from it.

    Does NOT store the full S(k,f).  Stores:
    - scalars: f_min, k_star, vg, f_k0
    - 1-D crosssection_at_fmin: S(k) at the f-bin nearest f_min
    - 1-D crosssection_at_fk0:  S(k) at the f-bin nearest f(k=0)
    - 1-D branch_f: f_peak(k) over searched k range
    - 1-D branch_k: k values for branch_f
    - k_axis: full k-axis [rad/m]
    """
    from .analyze import DispersionAnalyzeAccessor

    lowest = DispersionAnalyzeAccessor(result).find_lowest_possible_frequency(**find_kwargs)

    S = result.S          # (Nk, Nf)
    f_axis = result.f_axis
    k_axis = result.k_axis

    # Prefer slice-based frequency selection to avoid duplicating S[:, pos_f].
    if np.all(np.diff(f_axis) >= 0):
        f_selector: slice | np.ndarray = slice(int(np.searchsorted(f_axis, 0.0, side="left")), None)
        f_pos = f_axis[f_selector]
    else:
        pos_idx = np.flatnonzero(f_axis >= 0)
        if pos_idx.size == 0:
            raise ValueError("Frequency axis does not contain non-negative values.")
        if int(pos_idx[-1] - pos_idx[0] + 1) == int(pos_idx.size):
            f_selector = slice(int(pos_idx[0]), int(pos_idx[-1]) + 1)
            f_pos = f_axis[f_selector]
        else:
            f_selector = pos_idx
            f_pos = f_axis[pos_idx]

    if f_pos.size == 0:
        raise ValueError("No positive frequencies available for compact extraction.")
    f_axis_monotonic = bool(np.all(np.diff(f_pos) >= 0))

    # Apply same fmin cutoff as find_lowest_possible_frequency
    fmin_hz = find_kwargs.get("fmin_hz", "auto")
    if fmin_hz == "auto":
        fmin_cutoff = 0.05 * float(f_pos.max()) if f_pos.size else 0.0
    elif fmin_hz is not None and fmin_hz > 0:
        fmin_cutoff = float(fmin_hz)
    else:
        fmin_cutoff = 0.0

    if fmin_cutoff > 0:
        if f_axis_monotonic:
            start_idx = int(np.searchsorted(f_pos, fmin_cutoff, side="left"))
            if start_idx >= f_pos.size:
                raise ValueError(
                    f"fmin_hz={fmin_cutoff:.3e} exceeds available positive-frequency range "
                    f"[{float(f_pos.min()):.3e}, {float(f_pos.max()):.3e}] Hz.",
                )
            f_pos = f_pos[start_idx:]
            if isinstance(f_selector, slice):
                base_start = int(f_selector.start or 0)
                f_selector = slice(base_start + start_idx, f_selector.stop, f_selector.step)
            else:
                f_selector = f_selector[start_idx:]
        else:
            f_keep = f_pos >= fmin_cutoff
            if not np.any(f_keep):
                raise ValueError(
                    f"fmin_hz={fmin_cutoff:.3e} exceeds available positive-frequency values.",
                )
            f_pos = f_pos[f_keep]
            if isinstance(f_selector, slice):
                idx_full = np.arange(f_axis.shape[0], dtype=int)[f_selector]
                f_selector = idx_full[f_keep]
            else:
                f_selector = f_selector[f_keep]

    if isinstance(f_selector, slice):
        abs_start = int(f_selector.start or 0)
        idx_fmin_abs = abs_start + int(np.abs(f_pos - lowest.f_min_hz).argmin())
        idx_fk0_abs = abs_start + int(np.abs(f_pos - lowest.f_at_k0_hz).argmin())
    else:
        idx_fmin_abs = int(f_selector[int(np.abs(f_pos - lowest.f_min_hz).argmin())])
        idx_fk0_abs = int(f_selector[int(np.abs(f_pos - lowest.f_at_k0_hz).argmin())])

    # Cross-sections at requested frequencies
    cs_at_fmin = S[:, idx_fmin_abs].copy()
    cs_at_fk0 = S[:, idx_fk0_abs].copy()

    return {
        "f_min_hz":              lowest.f_min_hz,
        "k_star_rad_m":          lowest.k_at_f_min,
        "vg_at_min_m_s":         lowest.group_velocity_at_min,
        "f_at_k0_hz":            lowest.f_at_k0_hz,
        "crosssection_at_fmin":  cs_at_fmin,
        "crosssection_at_fk0":   cs_at_fk0,
        "branch_f":              lowest.branch_f.copy(),
        "branch_k":              lowest.branch_k.copy(),
        "k_axis":                k_axis.copy(),
        "f_axis_pos":            f_pos.copy(),
    }


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BulkMinimumFrequencyResult:
    """Collection of minimum-frequency analysis results across a parameter sweep.

    Stores only compact data (scalars + 1-D cross-sections) — no full S(k,f).

    Attributes
    ----------
    param_values : np.ndarray
        Scan parameter values (shape: N).
    param_label : str
        Human-readable label for the scan parameter (e.g. ``"B_ext [mT]"``).
    f_min_hz : np.ndarray
        Minimum branch frequency for each scan point [Hz] (shape: N).
    k_star_rad_m : np.ndarray
        Wave-vector at f_min for each scan point [rad/m] (shape: N).
    vg_at_min : np.ndarray
        Group velocity at k* for each scan point [m/s] (shape: N).
    f_at_k0_hz : np.ndarray
        f(k≈0) / FMR frequency for each scan point [Hz] (shape: N).
    crosssections_at_fmin : list[np.ndarray]
        S(k) at f≈f_min for each scan point.  Each array has shape (Nk,).
    crosssections_at_fk0 : list[np.ndarray]
        S(k) at f≈f(k=0) for each scan point.
    branches_f : list[np.ndarray]
        f_peak(k) for each scan point.
    branches_k : list[np.ndarray]
        Corresponding k-axis for branches_f.
    k_axes : list[np.ndarray]
        Full k-axis [rad/m] for each scan point (may differ between jobs).
    errors : dict[int, str]
        ``{index: error_message}`` for jobs that failed to compute.
    meta : dict
        Arbitrary metadata (axis, component, filters_config, …).
    """

    param_values: np.ndarray
    param_label: str
    f_min_hz: np.ndarray
    k_star_rad_m: np.ndarray
    vg_at_min: np.ndarray
    f_at_k0_hz: np.ndarray
    crosssections_at_fmin: list[np.ndarray]
    crosssections_at_fk0: list[np.ndarray]
    branches_f: list[np.ndarray]
    branches_k: list[np.ndarray]
    k_axes: list[np.ndarray]
    errors: dict[int, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # --- Analytical overlay (optional) ---
    analytical_f_min_hz: np.ndarray | None = None
    analytical_k_star_rad_m: np.ndarray | None = None
    analytical_f_k0_hz: np.ndarray | None = None
    analytical_model: str | None = None
    analytical_params: dict[str, Any] | None = None
    # Multiple overlays: list of dicts with keys:
    #   label, f_min_hz, k_star_rad_m, f_k0_hz, model, params
    analytical_overlays: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n(self) -> int:
        """Number of scan points."""
        return len(self.param_values)

    @property
    def f_min_ghz(self) -> np.ndarray:
        return self.f_min_hz / 1e9

    @property
    def f_at_k0_ghz(self) -> np.ndarray:
        return self.f_at_k0_hz / 1e9

    @property
    def k_star_rad_um(self) -> np.ndarray:
        return self.k_star_rad_m / 1e6

    @property
    def delta_f_mhz(self) -> np.ndarray:
        """Δf = f(k=0) − f_min  [MHz]."""
        return (self.f_at_k0_hz - self.f_min_hz) / 1e6

    @property
    def has_analytical(self) -> bool:
        """Whether analytical overlay data is available."""
        return self.analytical_f_min_hz is not None or len(self.analytical_overlays) > 0

    @property
    def analytical_f_min_ghz(self) -> np.ndarray | None:
        return self.analytical_f_min_hz / 1e9 if self.analytical_f_min_hz is not None else None

    @property
    def analytical_f_k0_ghz(self) -> np.ndarray | None:
        return self.analytical_f_k0_hz / 1e9 if self.analytical_f_k0_hz is not None else None

    @property
    def analytical_k_star_rad_um(self) -> np.ndarray | None:
        return self.analytical_k_star_rad_m / 1e6 if self.analytical_k_star_rad_m is not None else None

    @property
    def analytical_delta_f_mhz(self) -> np.ndarray | None:
        """Analytical Δf = f(k=0) − f_min  [MHz]."""
        if self.analytical_f_k0_hz is None or self.analytical_f_min_hz is None:
            return None
        return (self.analytical_f_k0_hz - self.analytical_f_min_hz) / 1e6

    # ------------------------------------------------------------------
    # Analytical overlay (post-hoc)
    # ------------------------------------------------------------------

    def add_analytical(
        self,
        *,
        Ms: float,
        d: float,
        Aex: float,
        Ku: float = 0.0,
        Kc1: float = 0.0,
        Kc2: float = 0.0,
        phi: float = np.pi / 2,
        phi_ani: float = 0.0,
        g: float = 2.0,
        model: str = "kalinikos",
        label: str | None = None,
        param_is_mT: bool = False,
        k_range: tuple[float, float] | None = None,
        n_k: int = 500,
        side: str = "positive",
    ) -> "BulkMinimumFrequencyResult":
        """Compute analytical dispersion overlay on existing results.

        Can be called **multiple times** to stack several overlays on the
        same plot (e.g. with/without anisotropy).

        Parameters
        ----------
        Ms, d, Aex, Ku, Kc1, Kc2, phi, phi_ani, g
            Material parameters (same meaning as in ``mmpp.analytical``).
        model : str
            Analytical model name (default ``"kalinikos"``).
        label : str, optional
            Display label for the overlay legend.  Defaults to the *model*
            name if omitted.
        param_is_mT : bool
            If True, ``param_values`` are in mT and divided by 1000 for T.
        k_range : tuple, optional
            ``(k_min, k_max)`` in rad/m.  Default: from simulation k_axes.
        n_k : int
            Number of k points (default 500).
        side : str
            ``"positive"``, ``"negative"``, or ``"both"``.

        Returns
        -------
        self
            Returns self for chaining.
        """
        from mmpp.analytical import dispersion as _an_disp

        model_func = getattr(_an_disp, model, None)
        if model_func is None:
            raise ValueError(f"Unknown model {model!r}")

        # Resolve k range
        if k_range is not None:
            k_lo, k_hi = k_range
        elif self.k_axes and self.k_axes[0].size > 0:
            k_all = self.k_axes[0]
            if side == "positive":
                k_lo, k_hi = 0.0, float(k_all.max())
            elif side == "negative":
                k_lo, k_hi = float(k_all.min()), 0.0
            else:
                k_lo, k_hi = float(k_all.min()), float(k_all.max())
        else:
            k_lo, k_hi = 0.0, 10e6

        k_an = np.linspace(k_lo, k_hi, n_k)
        mat = dict(Ms=Ms, d=d, Aex=Aex, Ku=Ku, Kc1=Kc1, Kc2=Kc2,
                   phi=phi, phi_ani=phi_ani, g=g)

        an_f_min = np.full(self.n, np.nan)
        an_k_star = np.full(self.n, np.nan)
        an_f_k0 = np.full(self.n, np.nan)

        for i in range(self.n):
            B_val = float(self.param_values[i])
            if param_is_mT:
                B_val /= 1000.0
            B_val = abs(B_val)  # model requires positive B (field magnitude)
            try:
                res = model_func(k=k_an, B=B_val, **mat)
                if side == "positive":
                    mask = k_an > 0
                elif side == "negative":
                    mask = k_an < 0
                else:
                    mask = np.ones(len(k_an), dtype=bool)

                f_masked = res.f[mask]
                k_masked = k_an[mask]
                idx = int(np.nanargmin(f_masked))
                an_f_min[i] = f_masked[idx] * 1e9
                an_k_star[i] = k_masked[idx]
                an_f_k0[i] = res.f[np.argmin(np.abs(k_an))] * 1e9
            except Exception:
                pass

        overlay_label = label or model
        overlay = {
            "label": overlay_label,
            "f_min_hz": an_f_min,
            "k_star_rad_m": an_k_star,
            "f_k0_hz": an_f_k0,
            "model": model,
            "params": {**mat, "model": model, "param_is_mT": param_is_mT},
        }
        self.analytical_overlays.append(overlay)

        # Keep legacy single-overlay fields updated (latest overlay)
        self.analytical_f_min_hz = an_f_min
        self.analytical_k_star_rad_m = an_k_star
        self.analytical_f_k0_hz = an_f_k0
        self.analytical_model = overlay_label
        self.analytical_params = overlay["params"]
        return self

    @property
    def plot(self) -> "BulkMinimumPlotAccessor":
        return BulkMinimumPlotAccessor(self)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> Path:
        """Save to a compressed numpy archive (``.npz``).

        Parameters
        ----------
        path : path-like
            Output file path.  ``.npz`` extension added if absent.

        Returns
        -------
        Path
            Absolute path of the saved file.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, Any] = {
            "param_values":   self.param_values,
            "f_min_hz":       self.f_min_hz,
            "k_star_rad_m":   self.k_star_rad_m,
            "vg_at_min":      self.vg_at_min,
            "f_at_k0_hz":     self.f_at_k0_hz,
            "param_label":    np.array([self.param_label]),
        }
        for i, (cs_fmin, cs_fk0, bf, bk, ka) in enumerate(
            zip(
                self.crosssections_at_fmin,
                self.crosssections_at_fk0,
                self.branches_f,
                self.branches_k,
                self.k_axes,
            )
        ):
            arrays[f"cs_fmin_{i}"]   = cs_fmin
            arrays[f"cs_fk0_{i}"]    = cs_fk0
            arrays[f"branch_f_{i}"]  = bf
            arrays[f"branch_k_{i}"]  = bk
            arrays[f"k_axis_{i}"]    = ka

        arrays["errors_keys"]   = np.array(list(self.errors.keys()), dtype=int)
        arrays["errors_values"] = np.array(list(self.errors.values()), dtype=object)

        # Analytical overlay
        if self.analytical_f_min_hz is not None:
            arrays["an_f_min_hz"]      = self.analytical_f_min_hz
            arrays["an_k_star_rad_m"]  = self.analytical_k_star_rad_m
            arrays["an_f_k0_hz"]       = self.analytical_f_k0_hz
        if self.analytical_model is not None:
            arrays["an_model"] = np.array([self.analytical_model])
        if self.analytical_params is not None:
            import json
            arrays["an_params_json"] = np.array([json.dumps(self.analytical_params)])

        np.savez_compressed(str(path), **arrays)
        return path.resolve()

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BulkMinimumFrequencyResult":
        """Load from a ``.npz`` file saved by :meth:`save`.

        Parameters
        ----------
        path : path-like
            Path to the ``.npz`` file.
        """
        path = Path(path)
        data = np.load(str(path), allow_pickle=True)

        param_values  = data["param_values"]
        n             = len(param_values)
        param_label   = str(data["param_label"][0])
        f_min_hz      = data["f_min_hz"]
        k_star_rad_m  = data["k_star_rad_m"]
        vg_at_min     = data["vg_at_min"]
        f_at_k0_hz    = data["f_at_k0_hz"]

        cs_fmin   = [data[f"cs_fmin_{i}"]  for i in range(n) if f"cs_fmin_{i}"  in data]
        cs_fk0    = [data[f"cs_fk0_{i}"]   for i in range(n) if f"cs_fk0_{i}"   in data]
        branches_f = [data[f"branch_f_{i}"] for i in range(n) if f"branch_f_{i}" in data]
        branches_k = [data[f"branch_k_{i}"] for i in range(n) if f"branch_k_{i}" in data]
        k_axes     = [data[f"k_axis_{i}"]   for i in range(n) if f"k_axis_{i}"   in data]

        keys   = list(data.get("errors_keys",   np.array([], dtype=int)))
        values = list(data.get("errors_values", np.array([], dtype=object)))
        errors = {int(k): str(v) for k, v in zip(keys, values)}

        # Analytical overlay (may be absent in older files)
        an_f_min = data["an_f_min_hz"] if "an_f_min_hz" in data else None
        an_k_star = data["an_k_star_rad_m"] if "an_k_star_rad_m" in data else None
        an_f_k0 = data["an_f_k0_hz"] if "an_f_k0_hz" in data else None
        an_model = str(data["an_model"][0]) if "an_model" in data else None
        an_params = None
        if "an_params_json" in data:
            import json
            try:
                an_params = json.loads(str(data["an_params_json"][0]))
            except Exception:
                pass

        return cls(
            param_values=param_values,
            param_label=param_label,
            f_min_hz=f_min_hz,
            k_star_rad_m=k_star_rad_m,
            vg_at_min=vg_at_min,
            f_at_k0_hz=f_at_k0_hz,
            crosssections_at_fmin=cs_fmin,
            crosssections_at_fk0=cs_fk0,
            branches_f=branches_f,
            branches_k=branches_k,
            k_axes=k_axes,
            errors=errors,
            analytical_f_min_hz=an_f_min,
            analytical_k_star_rad_m=an_k_star,
            analytical_f_k0_hz=an_f_k0,
            analytical_model=an_model,
            analytical_params=an_params,
        )

    # ------------------------------------------------------------------
    # DataFrame export
    # ------------------------------------------------------------------

    def to_dataframe(self):
        """Export scalar results to a :class:`pandas.DataFrame`.

        Columns: ``param``, ``f_min_ghz``, ``k_star_rad_um``,
        ``vg_at_min_km_s``, ``f_at_k0_ghz``, ``delta_f_mhz``.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe().")
        return pd.DataFrame(
            {
                "param":            self.param_values,
                "f_min_ghz":        self.f_min_ghz,
                "k_star_rad_um":    self.k_star_rad_um,
                "vg_at_min_km_s":   self.vg_at_min / 1e3,
                "f_at_k0_ghz":      self.f_at_k0_ghz,
                "delta_f_mhz":      self.delta_f_mhz,
            }
        )

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        ok = self.n - len(self.errors)
        return (
            f"BulkMinimumFrequencyResult("
            f"n={self.n}, ok={ok}, errors={len(self.errors)}, "
            f"param={self.param_label!r}, "
            f"f_min=[{self.f_min_ghz.min():.3f}..{self.f_min_ghz.max():.3f}] GHz)"
        )

    def _repr_html_(self) -> str:
        from html import escape as _esc

        HV = "onmouseover=\"this.style.background='#1e293b'\" onmouseout=\"this.style.background='transparent'\""
        ok = self.n - len(self.errors)

        stat_rows = [
            ("scan points",   f"{self.n}  ({ok} OK, {len(self.errors)} errors)",
             "Total jobs in scan; errors shows jobs that raised exceptions."),
            ("parameter",     _esc(self.param_label),
             "Scan parameter label as provided to scan_minimum_frequency()."),
            ("f_min range",   f"{self.f_min_ghz.min():.4f} … {self.f_min_ghz.max():.4f} GHz",
             "Range of minimum branch frequency across all scan points."),
            ("k* range",      f"{self.k_star_rad_um.min():.3f} … {self.k_star_rad_um.max():.3f} rad/μm",
             "Range of wave-vector at f_min across scan points."),
            ("Δf range",      f"{self.delta_f_mhz.min():.2f} … {self.delta_f_mhz.max():.2f} MHz",
             "Range of Δf = f(k=0) − f_min. Positive = backward-volume character."),
            ("stored per pt", "S(k)|f_min  +  S(k)|f(k=0)  +  f_peak(k)",
             "Compact 1-D arrays only — full S(k,f) is NOT stored."),
        ]
        stat_html = "".join(
            f"<tr {HV} title=\"{_esc(tip)}\" style='cursor:help;'>"
            f"<td style='padding:3px 10px;font-family:monospace;color:#93c5fd;font-size:.86em;'>{_esc(k)}</td>"
            f"<td style='padding:3px 10px;color:#a5b4fc;font-size:.88em;font-weight:600;'>{v}</td>"
            f"</tr>"
            for k, v, tip in stat_rows
        )

        plot_methods = [
            (".plot.heatmap()",
             "2-D heatmap: S(k)|f_min vs param",
             "Color = spectral density at f_min, x = k [rad/μm], y = param. "
             "Shows how the k-profile at f_min changes with the scan parameter."),
            (".plot.f_min_vs_param()",
             "f_min(param) line plot",
             "Scatter/line of the minimum branch frequency vs scan parameter. "
             "Optionally overlays f(k=0)."),
            (".plot.k_star_vs_param()",
             "k*(param) line plot",
             "Wave-vector at f_min as a function of the scan parameter."),
            (".plot.delta_f_vs_param()",
             "Δf(param) = f(k=0) − f_min",
             "Shows how the frequency gap between k=0 and f_min varies with parameter."),
            (".plot.branches()",
             "f_peak(k) for all scan points",
             "Stacked line plot of all extracted f_peak(k) branches, coloured by param."),
            (".plot.vg_vs_param()",
             "v_g(k*) vs param",
             "Group velocity at k* as a function of the scan parameter [km/s]."),
        ]
        plot_html = "".join(
            f"<tr {HV} title=\"{_esc(tip)}\" style='cursor:pointer;'>"
            f"<td style='padding:3px 10px;font-family:monospace;color:#f9a8d4;font-size:.84em;'>{_esc(sig)}</td>"
            f"<td style='padding:3px 10px;color:#cbd5e1;font-size:.82em;'>{_esc(desc)}</td>"
            f"</tr>"
            for sig, desc, tip in plot_methods
        )

        error_section = ""
        if self.errors:
            err_rows = "".join(
                f"<tr><td style='padding:2px 8px;color:#fca5a5;font-family:monospace;font-size:.8em;'>{i}</td>"
                f"<td style='padding:2px 8px;color:#f87171;font-size:.78em;'>{_esc(str(msg)[:100])}</td></tr>"
                for i, msg in self.errors.items()
            )
            error_section = (
                f"<details style='margin-top:6px;'>"
                f"<summary style='cursor:pointer;color:#fca5a5;font-size:.8em;list-style:none;'>"
                f"&#9654; {len(self.errors)} error(s)</summary>"
                f"<table style='margin-top:4px;border-collapse:collapse;'>{err_rows}</table>"
                f"</details>"
            )

        return (
            "<div style='font-family:-apple-system,BlinkMacSystemFont,sans-serif;"
            "border:2px solid #334155;border-left:4px solid #6366f1;"
            "border-radius:10px;padding:14px;margin:6px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#0c1a35 100%);"
            "color:#e2e8f0;max-width:720px;'>"
            "<div style='font-weight:700;font-size:1.05em;color:#f1f5f9;margin-bottom:10px;'>"
            "BulkMinimumFrequencyResult"
            "<span style='background:#312e81;color:#c7d2fe;border-radius:4px;"
            "padding:1px 8px;font-size:.72em;font-weight:600;margin-left:8px;'>"
            f"n={self.n} scan points</span></div>"
            f"<table style='width:100%;border-collapse:collapse;margin-bottom:8px;'>{stat_html}</table>"
            + error_section
            + "<details style='margin-top:8px;'>"
            "<summary style='cursor:pointer;font-size:.83em;color:#f9a8d4;list-style:none;"
            "padding:3px 6px;background:#1e293b;border-radius:5px;'>"
            "▶ <code>.plot</code> — BulkMinimumPlotAccessor</summary>"
            "<div style='margin-left:12px;margin-top:4px;'>"
            f"<table style='width:100%;border-collapse:collapse;'>{plot_html}</table>"
            "</div></details>"
            "<div style='margin-top:8px;font-size:.77em;color:#475569;'>"
            "<code style='color:#a5b4fc;'>.save(path)</code> / "
            "<code style='color:#a5b4fc;'>.load(path)</code> — npz archive &nbsp;|&nbsp; "
            "<code style='color:#a5b4fc;'>.to_dataframe()</code> — pandas export"
            "</div>"
            "</div>"
        )


# ---------------------------------------------------------------------------
# Plot accessor
# ---------------------------------------------------------------------------

class BulkMinimumPlotAccessor:
    """Plotting namespace for :class:`BulkMinimumFrequencyResult`.

    Accessed via ``bulk.plot``.

    Methods
    -------
    heatmap()
        2-D heatmap of S(k) at f_min vs scan parameter.
    f_min_vs_param()
        f_min as a function of the scan parameter.
    k_star_vs_param()
        k* as a function of the scan parameter.
    delta_f_vs_param()
        Δf = f(k=0) − f_min as a function of the scan parameter.
    vg_vs_param()
        Group velocity at k* vs scan parameter.
    branches()
        All extracted f_peak(k) curves stacked by parameter.
    """

    def __init__(self, bulk: BulkMinimumFrequencyResult) -> None:
        self._bulk = bulk
        # Cache sort order for all plot methods
        self._idx = np.argsort(bulk.param_values)

    # ------------------------------------------------------------------

    def heatmap(
        self,
        ax: Optional["Axes"] = None,
        *,
        figsize: tuple[float, float] = (12, 7),
        dpi: Optional[int] = None,
        which: str = "fmin",
        cmap: str = "cmc.davos",
        kscale: str = "rad_um",
        lognorm: bool = False,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        k_xlim: Optional[tuple[float, float]] = None,
        title: Optional[str] = None,
        annotate_fmin: bool = True,
        save: Union[str, Path, bool, None] = None,
    ) -> tuple["Figure", "Axes"]:
        """2-D heatmap of S(k) cross-sections stacked vs scan parameter.

        This is the key bulk plot: each row is S(k) at f≈f_min (or f≈f(k=0))
        for one scan point; rows are ordered by the parameter value.

        Parameters
        ----------
        which : ``"fmin"`` | ``"fk0"``
            Which cross-section to plot.  ``"fmin"`` uses S(k) at f=f_min;
            ``"fk0"`` uses S(k) at f=f(k=0).
        kscale : ``"rad_um"`` | ``"meter"``
            Wave-vector unit.
        lognorm : bool
            Logarithmic color scale.
        annotate_fmin : bool
            Draw scatter of k* positions on top of the heatmap.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        try:
            import cmcrameri  # noqa: F401
        except ImportError:
            cmap = "viridis"

        bulk = self._bulk

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, **({"dpi": dpi} if dpi else {}))
        else:
            fig = ax.get_figure()

        # Build matrix: (N_params, Nk) — interpolate to a common k-grid
        n = bulk.n
        # Use the k-axis of the first valid job as reference
        k_ref = bulk.k_axes[0]
        cs_list = (
            bulk.crosssections_at_fmin if which == "fmin"
            else bulk.crosssections_at_fk0
        )

        rows = []
        for i, (cs, ka) in enumerate(zip(cs_list, bulk.k_axes)):
            if ka.size == k_ref.size:
                rows.append(cs)
            else:
                # Interpolate to reference k-axis
                rows.append(np.interp(k_ref, ka, cs, left=0.0, right=0.0))
        matrix = np.array(rows)   # (N, Nk)

        k_plot = k_ref / 1e6 if kscale == "rad_um" else k_ref
        k_label = r"$k$ [rad/μm]" if kscale == "rad_um" else r"$k$ [rad/m]"
        params = bulk.param_values

        extent = [
            float(k_plot[0]), float(k_plot[-1]),
            float(params[0]),  float(params[-1]),
        ]

        norm = None
        if lognorm:
            s_min = matrix[matrix > 0].min() if np.any(matrix > 0) else 1e-12
            norm = LogNorm(
                vmin=vmin if vmin is not None else s_min,
                vmax=vmax if vmax is not None else matrix.max(),
            )
        elif vmin is not None or vmax is not None:
            from matplotlib.colors import Normalize
            norm = Normalize(vmin=vmin, vmax=vmax)

        im = ax.imshow(
            matrix,
            cmap=cmap,
            norm=norm,
            aspect="auto",
            origin="lower",
            extent=extent,
        )
        fig.colorbar(im, ax=ax, label="S(k)|f_min  [arb. units]")

        if annotate_fmin and which == "fmin":
            ks = bulk.k_star_rad_um if kscale == "rad_um" else bulk.k_star_rad_m
            ax.scatter(
                ks, params,
                s=50, color="red", marker="*", zorder=10,
                label="k* (f_min)",
            )
            ax.legend(fontsize=8)

        ax.set_xlabel(k_label)
        ax.set_ylabel(bulk.param_label)
        ax.set_title(title or f"S(k) at f_min  vs  {bulk.param_label}")

        if k_xlim is not None:
            ax.set_xlim(*k_xlim)

        try:
            fig.tight_layout()
        except Exception:
            pass

        if save not in (None, False):
            self._save(fig, save)
        return fig, ax

    # ------------------------------------------------------------------

    def f_min_vs_param(
        self,
        ax: Optional["Axes"] = None,
        *,
        figsize: tuple[float, float] = (8, 5),
        dpi: Optional[int] = None,
        show_fk0: bool = True,
        show_delta_f: bool = False,
        f_units: str = "GHz",
        title: Optional[str] = None,
        save: Union[str, Path, bool, None] = None,
    ) -> tuple["Figure", "Axes"]:
        """Plot f_min (and optionally f(k=0)) vs scan parameter.

        Parameters
        ----------
        show_fk0 : bool
            Overlay f(k=0) as a dashed cyan line.
        show_delta_f : bool
            Add a second y-axis showing Δf = f(k=0) − f_min.
        """
        import matplotlib.pyplot as plt

        bulk = self._bulk
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, **({"dpi": dpi} if dpi else {}))
        else:
            fig = ax.get_figure()

        p = bulk.param_values[self._idx]
        f_min  = (bulk.f_min_ghz  if f_units == "GHz" else bulk.f_min_hz)[self._idx]
        f_k0   = (bulk.f_at_k0_ghz if f_units == "GHz" else bulk.f_at_k0_hz)[self._idx]
        f_label = "f [GHz]"      if f_units == "GHz" else "f [Hz]"

        ax.plot(p, f_min, "o-", color="#f97316", linewidth=2, markersize=6, label="f_min (sim)")
        if show_fk0:
            ax.plot(p, f_k0, "s--", color="#22d3ee", linewidth=1.5, markersize=5, label="f(k=0) (sim)")

        # Analytical overlays (multiple)
        _an_colors = ["#a3e635", "#f472b6", "#38bdf8", "#fbbf24", "#c084fc", "#34d399"]
        _an_markers_f = ["x", "v", "^", "d", "p", "h"]
        _an_markers_k0 = ["+", "1", "2", "3", "4", "*"]
        for oi, ov in enumerate(bulk.analytical_overlays):
            c = _an_colors[oi % len(_an_colors)]
            mf = _an_markers_f[oi % len(_an_markers_f)]
            mk = _an_markers_k0[oi % len(_an_markers_k0)]
            lbl = ov["label"]
            an_f = (ov["f_min_hz"] / 1e9 if f_units == "GHz" else ov["f_min_hz"])[self._idx]
            ax.plot(p, an_f, f"{mf}--", color=c, linewidth=2, markersize=7,
                    label=f"f_min ({lbl})")
            if show_fk0 and ov["f_k0_hz"] is not None:
                an_fk = (ov["f_k0_hz"] / 1e9 if f_units == "GHz" else ov["f_k0_hz"])[self._idx]
                ax.plot(p, an_fk, f"{mk}:", color=c, linewidth=1.5, markersize=6,
                        label=f"f(k=0) ({lbl})")

        ax.set_xlabel(bulk.param_label)
        ax.set_ylabel(f_label)
        ax.set_title(title or f"f_min  vs  {bulk.param_label}")
        ax.grid(True, alpha=0.25)

        if show_delta_f:
            ax2 = ax.twinx()
            df = bulk.delta_f_mhz[self._idx]
            ax2.plot(p, df, "^:", color="#a78bfa", linewidth=1.5, markersize=4, label="Δf (sim) [MHz]")
            for oi, ov in enumerate(bulk.analytical_overlays):
                c = _an_colors[oi % len(_an_colors)]
                lbl = ov["label"]
                an_df = (ov["f_k0_hz"] - ov["f_min_hz"]) / 1e6
                ax2.plot(p, an_df[self._idx], "v:", color=c,
                         linewidth=1.5, markersize=4, label=f"Δf ({lbl}) [MHz]")
            ax2.set_ylabel("Δf = f(k=0) − f_min  [MHz]", color="#a78bfa")
            ax2.tick_params(axis="y", labelcolor="#a78bfa")

            # Merge legends from both axes
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="best")
        else:
            ax.legend(fontsize=9)

        try:
            fig.tight_layout()
        except Exception:
            pass
        if save not in (None, False):
            self._save(fig, save)
        return fig, ax

    # ------------------------------------------------------------------

    def k_star_vs_param(
        self,
        ax: Optional["Axes"] = None,
        *,
        figsize: tuple[float, float] = (8, 5),
        dpi: Optional[int] = None,
        kscale: str = "rad_um",
        title: Optional[str] = None,
        save: Union[str, Path, bool, None] = None,
    ) -> tuple["Figure", "Axes"]:
        """Plot k* (wave-vector at f_min) vs scan parameter."""
        import matplotlib.pyplot as plt

        bulk = self._bulk
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, **({"dpi": dpi} if dpi else {}))
        else:
            fig = ax.get_figure()

        p = bulk.param_values[self._idx]
        k_data  = (bulk.k_star_rad_um if kscale == "rad_um" else bulk.k_star_rad_m)[self._idx]
        k_label = r"$k^*$ [rad/μm]" if kscale == "rad_um" else r"$k^*$ [rad/m]"

        ax.plot(p, k_data, "D-", color="#4ade80", linewidth=2, markersize=6, label="k* (sim)")
        _an_colors = ["#a3e635", "#f472b6", "#38bdf8", "#fbbf24", "#c084fc", "#34d399"]
        _an_markers = ["x", "v", "^", "d", "p", "h"]
        for oi, ov in enumerate(bulk.analytical_overlays):
            c = _an_colors[oi % len(_an_colors)]
            m = _an_markers[oi % len(_an_markers)]
            an_k_data = ov["k_star_rad_m"]
            if kscale == "rad_um":
                an_k_data = an_k_data / 1e6
            ax.plot(p, an_k_data[self._idx], f"{m}--", color=c, linewidth=2, markersize=7,
                    label=f"k* ({ov['label']})")
        ax.legend(fontsize=9)
        ax.set_xlabel(bulk.param_label)
        ax.set_ylabel(k_label)
        ax.set_title(title or f"k* (wave-vector at f_min)  vs  {bulk.param_label}")
        ax.grid(True, alpha=0.25)

        try:
            fig.tight_layout()
        except Exception:
            pass
        if save not in (None, False):
            self._save(fig, save)
        return fig, ax

    # ------------------------------------------------------------------

    def delta_f_vs_param(
        self,
        ax: Optional["Axes"] = None,
        *,
        figsize: tuple[float, float] = (8, 5),
        dpi: Optional[int] = None,
        title: Optional[str] = None,
        save: Union[str, Path, bool, None] = None,
    ) -> tuple["Figure", "Axes"]:
        """Plot Δf = f(k=0) − f_min  vs scan parameter [MHz]."""
        import matplotlib.pyplot as plt

        bulk = self._bulk
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, **({"dpi": dpi} if dpi else {}))
        else:
            fig = ax.get_figure()

        p = bulk.param_values[self._idx]
        ax.plot(p, bulk.delta_f_mhz[self._idx], "o-", color="#a78bfa", linewidth=2, markersize=6, label="Δf (sim)")
        _an_colors = ["#a3e635", "#f472b6", "#38bdf8", "#fbbf24", "#c084fc", "#34d399"]
        _an_markers = ["x", "v", "^", "d", "p", "h"]
        for oi, ov in enumerate(bulk.analytical_overlays):
            c = _an_colors[oi % len(_an_colors)]
            m = _an_markers[oi % len(_an_markers)]
            an_df = (ov["f_k0_hz"] - ov["f_min_hz"]) / 1e6
            ax.plot(p, an_df[self._idx], f"{m}--", color=c,
                    linewidth=2, markersize=7, label=f"Δf ({ov['label']})")
        ax.legend(fontsize=9)
        ax.axhline(0, color="#475569", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_xlabel(bulk.param_label)
        ax.set_ylabel("Δf = f(k=0) − f_min  [MHz]")
        ax.set_title(title or f"Δf  vs  {bulk.param_label}")
        ax.grid(True, alpha=0.25)

        try:
            fig.tight_layout()
        except Exception:
            pass
        if save not in (None, False):
            self._save(fig, save)
        return fig, ax

    # ------------------------------------------------------------------

    def vg_vs_param(
        self,
        ax: Optional["Axes"] = None,
        *,
        figsize: tuple[float, float] = (8, 5),
        dpi: Optional[int] = None,
        title: Optional[str] = None,
        save: Union[str, Path, bool, None] = None,
    ) -> tuple["Figure", "Axes"]:
        """Plot group velocity at k* vs scan parameter [km/s]."""
        import matplotlib.pyplot as plt

        bulk = self._bulk
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, **({"dpi": dpi} if dpi else {}))
        else:
            fig = ax.get_figure()

        vg_km_s = self._bulk.vg_at_min[self._idx] / 1e3
        p = bulk.param_values[self._idx]

        ax.plot(p, vg_km_s, "s-", color="#fb923c", linewidth=2, markersize=6)
        ax.axhline(0, color="#475569", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_xlabel(bulk.param_label)
        ax.set_ylabel("v_g(k*)  [km/s]")
        ax.set_title(title or f"Group velocity at k*  vs  {bulk.param_label}")
        ax.grid(True, alpha=0.25)

        try:
            fig.tight_layout()
        except Exception:
            pass
        if save not in (None, False):
            self._save(fig, save)
        return fig, ax

    # ------------------------------------------------------------------

    def branches(
        self,
        ax: Optional["Axes"] = None,
        *,
        figsize: tuple[float, float] = (10, 6),
        dpi: Optional[int] = None,
        kscale: str = "rad_um",
        f_units: str = "GHz",
        cmap: str = "viridis",
        alpha: float = 0.7,
        linewidth: float = 1.5,
        title: Optional[str] = None,
        colorbar: bool = True,
        save: Union[str, Path, bool, None] = None,
    ) -> tuple["Figure", "Axes"]:
        """Plot all extracted f_peak(k) branches, coloured by scan parameter.

        Parameters
        ----------
        cmap : str
            Colormap used to colour branches by parameter value.
        colorbar : bool
            Add a colorbar encoding the parameter value.
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        bulk = self._bulk
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, **({"dpi": dpi} if dpi else {}))
        else:
            fig = ax.get_figure()

        params = bulk.param_values
        p_min, p_max = float(params.min()), float(params.max())
        cmap_obj = cm.get_cmap(cmap)
        norm = mcolors.Normalize(vmin=p_min, vmax=p_max)

        for i, (bk, bf, p) in enumerate(zip(bulk.branches_k, bulk.branches_f, params)):
            k_plot = bk / 1e6 if kscale == "rad_um" else bk
            f_plot = bf / 1e9 if f_units == "GHz" else bf
            color = cmap_obj(norm(float(p)))
            ax.plot(k_plot, f_plot, color=color, linewidth=linewidth, alpha=alpha)

        k_label = r"$k$ [rad/μm]" if kscale == "rad_um" else r"$k$ [rad/m]"
        f_label = "f [GHz]"      if f_units == "GHz"     else "f [Hz]"
        ax.set_xlabel(k_label)
        ax.set_ylabel(f_label)
        ax.set_title(title or f"f_peak(k) branches  —  {bulk.param_label}")
        ax.grid(True, alpha=0.2)

        if colorbar:
            sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label=bulk.param_label, fraction=0.03)

        try:
            fig.tight_layout()
        except Exception:
            pass
        if save not in (None, False):
            self._save(fig, save)
        return fig, ax

    # ------------------------------------------------------------------

    def summary(
        self,
        figsize: tuple[float, float] = (14, 10),
        dpi: Optional[int] = None,
        save: Union[str, Path, bool, None] = None,
    ) -> tuple["Figure", Any]:
        """4-panel summary figure: heatmap, f_min, k*, Δf."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=figsize, **({"dpi": dpi} if dpi else {}))
        ax_hm, ax_fmin, ax_kstar, ax_df = axes.ravel()

        self.heatmap(ax=ax_hm)
        self.f_min_vs_param(ax=ax_fmin, show_fk0=True, show_delta_f=False)
        self.k_star_vs_param(ax=ax_kstar)
        self.delta_f_vs_param(ax=ax_df)

        try:
            fig.tight_layout()
        except Exception:
            pass
        if save not in (None, False):
            self._save(fig, save)
        return fig, axes

    # ------------------------------------------------------------------

    def _save(self, fig: "Figure", save: Any) -> None:
        from datetime import datetime, timezone
        if isinstance(save, bool):
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            path = Path(f"bulk_dispersion_{ts}.png")
        else:
            path = Path(save)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches="tight")

    def __repr__(self) -> str:
        return f"<BulkMinimumPlotAccessor(n={self._bulk.n})>"

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import plot_accessor_html
        return plot_accessor_html("BulkMinimumPlotAccessor", [
            (".heatmap(which='fmin', lognorm=True)",
             "S(k) cross-section heatmap vs scan parameter",
             "which: 'fmin' or 'fk0'. lognorm, cmap, vmin/vmax, k_xlim, annotate_fmin, save."),
            (".f_min_vs_param(show_fk0=True, show_delta_f=False)",
             "f_min (and f(k=0)) vs scan parameter",
             "show_fk0: overlay f(k=0). show_delta_f: second y-axis with Δf. f_units: 'GHz'|'Hz'. Supports multiple analytical overlays."),
            (".k_star_vs_param(kscale='rad_um')",
             "k* (wave-vector at f_min) vs scan parameter",
             "kscale: 'rad_um' or 'rad_m'. Includes analytical overlays if present."),
            (".delta_f_vs_param()",
             "Δf = f(k=0) − f_min [MHz] vs scan parameter",
             "Shows sim and all analytical overlay Δf curves."),
            (".vg_vs_param()",
             "Group velocity at k* [km/s] vs scan parameter",
             "Estimated vg = dω/dk at the frequency minimum."),
            (".branches(cmap='viridis')",
             "All f_peak(k) branches coloured by parameter",
             "cmap, alpha, linewidth, colorbar, f_units. Overlays all branches."),
            (".summary()",
             "4-panel summary: heatmap + f_min + k* + Δf",
             "Creates a (14×10) figure with four sub-plots."),
        ])


# ---------------------------------------------------------------------------
# Main entry-point: scan_minimum_frequency
# ---------------------------------------------------------------------------

def scan_minimum_frequency(
    sources: Iterable[Any],
    param_values: Sequence[float] | None = None,
    param_label: str = "parameter",
    *,
    job_indices: Sequence[int] | None = None,
    param_attr: str | None = None,
    param_scale: float = 1.0,
    filters: dict[str, Any] | None = None,
    compute_kwargs: dict[str, Any] | None = None,
    find_kwargs: dict[str, Any] | None = None,
    slice_spec: Any | None = None,
    dataset: str = "m",
    z_slice: Any | None = None,
    analytical_params: dict[str, Any] | None = None,
    n_workers: int = 1,
    verbose: bool = True,
    on_error: str = "warn",
) -> BulkMinimumFrequencyResult:
    """Scan the minimum branch frequency across a parameter sweep.

    For each source the function:

    1. Accesses the magnetisation dataset (``job[0].m[...]``).
    2. Applies ``filters`` via ``fft.dispersion.filters(...)``.
    3. Calls ``compute_1d(...)`` to get :class:`DispersionResult1D`.
    4. Extracts only compact data (scalars + 1-D cross-sections) — the
       full ``S(k,f)`` array is immediately discarded.

    Memory usage per job is therefore O(Nk) instead of O(Nk × Nf).

    Parameters
    ----------
    sources : mmpp.MMPP | list[ZarrJobResult] | list[DispersionResult1D]
        Sweep source — one of:

        * **``mmpp.MMPP(path)``** (most common): a collection object.  Iterated
          automatically; each item is a ``ZarrJobResult`` accessed as
          ``src.m[slice_spec].fft.dispersion``.
        * A plain list / iterable of ``ZarrJobResult`` objects.
        * A list of pre-computed :class:`DispersionResult1D` — used directly.
        * A list of callables ``() -> DispersionResult1D`` — called once per job.
    param_values : sequence of float, optional
        Physical axis values — one entry per *selected* job.  If ``None``
        (default), indices ``0, 1, 2, …`` are used as labels unless
        *param_attr* is set.
    job_indices : sequence of int, optional
        Select a subset of jobs from *sources* by integer index.  When
        provided, only those positions are processed and *param_values* must
        have the same length (or be ``None`` to auto-label with the indices).
        Example: ``job_indices=[0, 5, 10, 15, 20, 25, 30]`` picks 7 jobs from
        a 26-job MMPP container.
    param_attr : str, optional
        Name of an attribute on each job to read automatically as the
        parameter value.  For example ``param_attr="b"`` reads
        ``job.attrs["b"]`` for every job.  Combined with *param_scale* this
        replaces passing *param_values* manually.
    param_scale : float
        Multiply the value read from *param_attr* by this factor before
        storing.  Useful to convert units, e.g. Tesla → mT: ``param_scale=1000``.
    param_label : str
        Human-readable label for the scan parameter (used in plots).
    filters : dict, optional
        Passed to ``fft.dispersion.filters(**filters)``.
    compute_kwargs : dict, optional
        Passed to ``DispersionFilterChain.compute_1d(**compute_kwargs)``.
    find_kwargs : dict, optional
        Passed to ``DispersionAnalyzeAccessor.find_lowest_possible_frequency(**find_kwargs)``.
    slice_spec : slice / index, optional
        Spatial slice applied to ``src.m[slice_spec]``.
        ``None`` (default) means no slicing — the full dataset is passed to
        the FFT pipeline.
    dataset : str
        Dataset name on the simulation object (default ``"m"``).
    z_slice : optional
        Legacy alias for *slice_spec*.
    analytical_params : dict, optional
        Material parameters for the analytical dispersion model.  When
        provided, an analytical f_min(B) curve is computed alongside the
        simulation results and stored in the result for overlay plotting.

        Required keys: ``Ms``, ``d``, ``Aex``.  Optional: ``Ku``, ``Kc1``,
        ``Kc2``, ``phi``, ``phi_ani``, ``g``.

        Special keys:

        * ``model`` – analytical model name (default ``"kalinikos"``).
          Also accepts ``"backward_volume"``, ``"damon_eshbach"``, etc.
        * ``param_is_mT`` – if ``True``, divide ``param_values`` by 1000
          to convert mT → T before passing to the model (default ``False``).
        * ``k_range`` – ``(k_min, k_max)`` in rad/m (default: from simulation).
        * ``n_k`` – number of k points (default: 500).
        * ``side`` – ``"positive"`` or ``"both"`` (default: from ``find_kwargs``).
    n_workers : int
        Number of concurrent threads for processing jobs (default: 1 =
        sequential).  Threading is effective because numpy FFT, zarr I/O,
        and scipy all release the GIL.  Recommended: 2‒4 for I/O-bound
        workloads; avoid very large values to control peak RAM usage.
    verbose : bool
        Print progress to stdout.
    on_error : ``"warn"`` | ``"raise"`` | ``"skip"``
        How to handle per-job exceptions.

    Returns
    -------
    BulkMinimumFrequencyResult
        Compact sweep result with ``.plot``, ``.save()``, ``.to_dataframe()``.

    Examples
    --------
    Sweep over applied field values::

        import mmpp
        from mmpp.fft.dispersion.bulk import scan_minimum_frequency

        # jobs is a single mmpp.MMPP container (may hold more jobs than needed)
        jobs = mmpp.MMPP("/path/to/sweep_dir/", debug=False)

        # Case A: all jobs map 1-to-1 to param_values
        bulk = scan_minimum_frequency(
            jobs,
            param_values=[0, 5, 10, 15, 20, 25, 30],   # 7 jobs in container
            param_label="B_ext [mT]",
        )

        # Case B: select 7 specific jobs from a 26-job container
        bulk = scan_minimum_frequency(
            jobs,
            job_indices=[0, 5, 10, 15, 20, 25, 30],    # which jobs to pick
            param_values=[0, 5, 10, 15, 20, 25, 30],   # their B_ext values
            param_label="B_ext [mT]",
            filters=dict(remove_static=True,
                         live={"gaussian_morph": {"enabled": True, "sigma_f": 1.0}}),
            compute_kwargs=dict(axis="x", save=True, force=False),
            find_kwargs=dict(side="positive", smooth_sigma=2.0),
            slice_spec=(slice(None), Ellipsis, slice(0, 1)),
        )

        bulk.plot.summary()
        bulk.save("sweep_field.npz")
    """
    # Normalise sources -------------------------------------------------------
    # Accept mmpp.MMPP objects (iterable over ZarrJobResult), plain lists, etc.
    if hasattr(sources, "zarr_results"):
        # mmpp.MMPP container — unpack its ZarrJobResult list directly
        all_sources: list[Any] = list(sources.zarr_results)
    else:
        all_sources = list(sources)

    # Optional job selection --------------------------------------------------
    if job_indices is not None:
        idx_list = list(job_indices)
        try:
            sources = [all_sources[i] for i in idx_list]  # type: ignore[assignment]
        except IndexError as exc:
            raise IndexError(
                f"job_indices contains an out-of-range index for a container "
                f"of {len(all_sources)} jobs."
            ) from exc
    else:
        sources = all_sources  # type: ignore[assignment]

    # Param values: read from attr, fall back to indices ----------------------
    if param_values is None and param_attr is not None:
        extracted: list[float] = []
        for src in sources:  # type: ignore[union-attr]
            try:
                val = src.attrs[param_attr]
            except Exception as exc:
                raise AttributeError(
                    f"Cannot read attrs[{param_attr!r}] from {src!r}: {exc}"
                ) from exc
            extracted.append(float(val) * param_scale)
        param_values_arr = np.asarray(extracted, dtype=float)

        # Sort sources by ascending parameter value
        sort_idx = np.argsort(param_values_arr)
        param_values_arr = param_values_arr[sort_idx]
        sources = [sources[i] for i in sort_idx]  # type: ignore[index]
    elif param_values is None:
        param_values_arr = np.asarray(
            job_indices if job_indices is not None else range(len(sources)),  # type: ignore[arg-type]
            dtype=float,
        )
    else:
        param_values_arr = np.asarray(param_values, dtype=float)

    if len(sources) != len(param_values_arr):
        raise ValueError(
            f"sources has {len(sources)} items but param_values has "
            f"{len(param_values_arr)} — must match.  "
            f"Use job_indices to select a subset of the {len(all_sources)}-job container."
        )

    filters        = filters        or {}
    compute_kwargs = compute_kwargs or {}
    find_kwargs    = find_kwargs    or {}

    if z_slice is not None and slice_spec is None:
        slice_spec = z_slice
    # Note: slice_spec=None means no slicing (dataset passed as-is)

    f_min_arr      = np.full(len(sources), np.nan)
    k_star_arr     = np.full(len(sources), np.nan)
    vg_arr_out     = np.full(len(sources), np.nan)
    f_k0_arr       = np.full(len(sources), np.nan)
    cs_fmin_list:  list[np.ndarray] = []
    cs_fk0_list:   list[np.ndarray] = []
    branches_f:    list[np.ndarray] = []
    branches_k:    list[np.ndarray] = []
    k_axes:        list[np.ndarray] = []
    errors: dict[int, str] = {}

    # ── Helper : process a single job (may run in a worker thread) ─────
    def _process_one_job(
        i: int,
        src: Any,
    ) -> dict[str, Any]:
        """Process job *i* and return compact data or an error dict."""
        iface = None
        result = None
        compact = None
        try:
            if callable(src) and not hasattr(src, "fft"):
                result = src()
            elif hasattr(src, "S") and hasattr(src, "k_axis"):
                result = src
            else:
                data_accessor = getattr(src, dataset)
                if slice_spec is not None:
                    data_accessor = data_accessor[slice_spec]

                iface = data_accessor.fft.dispersion
                chain = iface.filters(**filters)

                _ck = dict(compute_kwargs)
                _ck["use_cache"] = False
                _ck["disk_cache"] = False
                _ck["save"] = False
                _ck["save_result"] = False
                _ck["store_complex"] = False
                if _ck.get("avg_over_orthogonal", True) is False:
                    warnings.warn(
                        "scan_minimum_frequency overrides avg_over_orthogonal=False to "
                        "avg_over_orthogonal=True to avoid storing large local spectra.",
                        stacklevel=2,
                    )
                _ck["avg_over_orthogonal"] = True
                result = chain.compute_1d(**_ck)
                del chain, data_accessor, _ck

            if iface is not None:
                try:
                    object.__delattr__(result, "_interface")
                except (AttributeError, TypeError):
                    pass

            compact = _extract_compact(result, find_kwargs)

            out: dict[str, Any] = {
                "ok": True,
                "i": i,
                "f_min_hz": compact["f_min_hz"],
                "k_star_rad_m": compact["k_star_rad_m"],
                "vg_at_min_m_s": compact["vg_at_min_m_s"],
                "f_at_k0_hz": compact["f_at_k0_hz"],
                "crosssection_at_fmin": compact["crosssection_at_fmin"],
                "crosssection_at_fk0": compact["crosssection_at_fk0"],
                "branch_f": compact["branch_f"],
                "branch_k": compact["branch_k"],
                "k_axis": compact["k_axis"],
            }
            del result, compact
            return out

        except Exception as exc:
            return {"ok": False, "i": i, "error": f"{type(exc).__name__}: {exc}", "exc": exc}

        finally:
            if iface is not None:
                try:
                    iface.release_memory(clear_memory_cache=True, unload_raw_data=True)
                except Exception:
                    pass
                iface = None
            result = None
            compact = None
            gc.collect()

    # ── Run jobs (sequential or parallel) ─────────────────────────────
    n_total = len(sources)
    n_workers_eff = max(1, min(n_workers, n_total))

    if n_workers_eff <= 1:
        # Sequential path — keep the simple progress output
        for i, src in enumerate(sources):
            if verbose:
                print(f"[{i+1}/{n_total}]  {param_label}={param_values_arr[i]}", end="  ")
            out = _process_one_job(i, src)
            if out["ok"]:
                f_min_arr[i] = out["f_min_hz"]
                k_star_arr[i] = out["k_star_rad_m"]
                vg_arr_out[i] = out["vg_at_min_m_s"]
                f_k0_arr[i] = out["f_at_k0_hz"]
                cs_fmin_list.append(out["crosssection_at_fmin"])
                cs_fk0_list.append(out["crosssection_at_fk0"])
                branches_f.append(out["branch_f"])
                branches_k.append(out["branch_k"])
                k_axes.append(out["k_axis"])
                if verbose:
                    print(f"  f_min={out['f_min_hz']/1e9:.4f} GHz  k*={out['k_star_rad_m']/1e6:.3f} rad/μm")
            else:
                errors[i] = out["error"]
                cs_fmin_list.append(np.array([]))
                cs_fk0_list.append(np.array([]))
                branches_f.append(np.array([]))
                branches_k.append(np.array([]))
                k_axes.append(np.array([]))
                if on_error == "raise":
                    raise out["exc"]
                elif on_error == "warn":
                    warnings.warn(f"Job {i} ({param_label}={param_values_arr[i]}) failed: {out['error']}", stacklevel=2)
                if verbose:
                    print(f"  ERROR: {out['error']}")
    else:
        # Parallel path — ThreadPoolExecutor
        if verbose:
            print(f"Processing {n_total} jobs with {n_workers_eff} threads...")

        # Pre-allocate placeholder lists (will be filled by index)
        results_by_idx: dict[int, dict[str, Any]] = {}
        _progress_lock = threading.Lock()
        _done_count = [0]  # mutable counter for threads

        def _worker(i_src: tuple[int, Any]) -> dict[str, Any]:
            i, src = i_src
            out = _process_one_job(i, src)
            with _progress_lock:
                _done_count[0] += 1
                if verbose:
                    status = "OK" if out["ok"] else "ERR"
                    pv = param_values_arr[i]
                    extra = ""
                    if out["ok"]:
                        extra = f"  f_min={out['f_min_hz']/1e9:.4f} GHz"
                    print(f"  [{_done_count[0]}/{n_total}] {param_label}={pv}  {status}{extra}")
            return out

        with ThreadPoolExecutor(max_workers=n_workers_eff) as pool:
            futures = {pool.submit(_worker, (i, src)): i for i, src in enumerate(sources)}
            for future in as_completed(futures):
                out = future.result()
                results_by_idx[out["i"]] = out

        # Collect results in order
        for i in range(n_total):
            out = results_by_idx[i]
            if out["ok"]:
                f_min_arr[i] = out["f_min_hz"]
                k_star_arr[i] = out["k_star_rad_m"]
                vg_arr_out[i] = out["vg_at_min_m_s"]
                f_k0_arr[i] = out["f_at_k0_hz"]
                cs_fmin_list.append(out["crosssection_at_fmin"])
                cs_fk0_list.append(out["crosssection_at_fk0"])
                branches_f.append(out["branch_f"])
                branches_k.append(out["branch_k"])
                k_axes.append(out["k_axis"])
            else:
                errors[i] = out["error"]
                cs_fmin_list.append(np.array([]))
                cs_fk0_list.append(np.array([]))
                branches_f.append(np.array([]))
                branches_k.append(np.array([]))
                k_axes.append(np.array([]))
                if on_error == "raise":
                    raise out.get("exc", RuntimeError(out["error"]))
                elif on_error == "warn":
                    warnings.warn(f"Job {i} ({param_label}={param_values_arr[i]}) failed: {out['error']}", stacklevel=2)

        if verbose:
            n_ok = n_total - len(errors)
            print(f"Done: {n_ok}/{n_total} succeeded, {len(errors)} errors")


    # ── Analytical dispersion overlay ─────────────────────────────────
    an_f_min_arr: np.ndarray | None = None
    an_k_star_arr: np.ndarray | None = None
    an_f_k0_arr: np.ndarray | None = None
    an_model_name: str | None = None

    if analytical_params is not None:
        an_p = dict(analytical_params)  # don't mutate caller's dict
        an_model_name = str(an_p.pop("model", "kalinikos"))
        param_is_mT = bool(an_p.pop("param_is_mT", False))
        an_k_range = an_p.pop("k_range", None)
        an_n_k = int(an_p.pop("n_k", 500))
        an_side = an_p.pop("side", find_kwargs.get("side", "positive"))

        # Resolve k range from simulation or user
        if an_k_range is not None:
            k_lo, k_hi = an_k_range
        elif k_axes and k_axes[0].size > 0:
            k_all = k_axes[0]
            if an_side == "positive":
                k_lo, k_hi = 0.0, float(k_all.max())
            elif an_side == "negative":
                k_lo, k_hi = float(k_all.min()), 0.0
            else:
                k_lo, k_hi = float(k_all.min()), float(k_all.max())
        else:
            k_lo, k_hi = 0.0, 10e6  # fallback 0‥10 rad/μm

        k_an = np.linspace(k_lo, k_hi, an_n_k)

        # Import the requested model
        from mmpp.analytical import dispersion as _an_disp
        model_func = getattr(_an_disp, an_model_name, None)
        if model_func is None:
            raise ValueError(
                f"Unknown analytical model {an_model_name!r}. "
                f"Available: kalinikos, backward_volume, damon_eshbach, forward_volume, bottcher"
            )

        an_f_min_arr = np.full(len(sources), np.nan)
        an_k_star_arr = np.full(len(sources), np.nan)
        an_f_k0_arr = np.full(len(sources), np.nan)

        for i in range(len(param_values_arr)):
            B_val = float(param_values_arr[i])
            if param_is_mT:
                B_val = B_val / 1000.0  # mT → T
            B_val = abs(B_val)  # model requires positive B (field magnitude)

            try:
                res_an = model_func(k=k_an, B=B_val, **an_p)
                f_ghz = res_an.f  # GHz

                # Find f_min on the analytical curve
                if an_side == "positive":
                    mask = k_an > 0
                elif an_side == "negative":
                    mask = k_an < 0
                else:
                    mask = np.ones(len(k_an), dtype=bool)

                if np.any(mask) and np.any(np.isfinite(f_ghz[mask])):
                    f_masked = f_ghz[mask]
                    k_masked = k_an[mask]
                    idx_min = int(np.nanargmin(f_masked))
                    an_f_min_arr[i] = f_masked[idx_min] * 1e9  # → Hz
                    an_k_star_arr[i] = k_masked[idx_min]

                    # f at k≈0
                    idx_k0 = int(np.argmin(np.abs(k_an)))
                    an_f_k0_arr[i] = f_ghz[idx_k0] * 1e9  # → Hz
            except Exception as exc:
                if verbose:
                    print(f"  [analytical] B={B_val:.4f}T failed: {exc}")

        if verbose:
            n_ok = int(np.isfinite(an_f_min_arr).sum())
            print(f"\nAnalytical ({an_model_name}): {n_ok}/{len(param_values_arr)} points computed")

    return BulkMinimumFrequencyResult(
        param_values=param_values_arr,
        param_label=param_label,
        f_min_hz=f_min_arr,
        k_star_rad_m=k_star_arr,
        vg_at_min=vg_arr_out,
        f_at_k0_hz=f_k0_arr,
        crosssections_at_fmin=cs_fmin_list,
        crosssections_at_fk0=cs_fk0_list,
        branches_f=branches_f,
        branches_k=branches_k,
        k_axes=k_axes,
        errors=errors,
        meta={
            "filters":         filters,
            "compute_kwargs":  compute_kwargs,
            "find_kwargs":     find_kwargs,
        },
        analytical_f_min_hz=an_f_min_arr,
        analytical_k_star_rad_m=an_k_star_arr,
        analytical_f_k0_hz=an_f_k0_arr,
        analytical_model=an_model_name,
        analytical_params=analytical_params,
    )
