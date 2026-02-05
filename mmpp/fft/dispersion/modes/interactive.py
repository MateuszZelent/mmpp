"""
Interactive Jupyter widget interface for dispersion mode analysis.

Provides an ipywidgets-based interactive exploration of BZ-folded
dispersion relations with real-time parameter adjustment.

Features:
- Stacked layout: dispersion plot on top, mode visualization (2D spatial) below
- Click on dispersion to select mode → shows marker and mask positions
- Slider for N_BZ (number of Brillouin zones in mask)
- Option for +k/-k direction selection
- Real-time 2D spatial mode visualization m(x, y) using pre-computed S_complex

Algorithm for spatial mode reconstruction (following Rychły et al.):
1. Dispersion computation stores both S (power spectrum) and S_complex (phase-preserving)
2. User selects (k, f) on dispersion plot
3. Create BZ mask for k_0 ± n·G (all periodic copies)
4. Extract S_complex at f_0 and apply mask
5. IFFT only over k → propagation axis (FAST - no re-computation!)
6. Result: Spatial mode profile m(x, y) at selected (k_0, f_0)

Performance: Mode visualization is INSTANT because S_complex is pre-computed and cached.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

# Check for interactive dependencies
_HAS_WIDGETS = False
_HAS_MATPLOTLIB = False

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    _HAS_WIDGETS = True
except ImportError:
    pass

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    import matplotlib

    _HAS_MATPLOTLIB = True
except ImportError:
    pass

if TYPE_CHECKING:
    from ..models import DispersionResult1D
    from ..interface import FFTDispersionInterface
    from .models import FoldedDispersionResult
    from .folding import BrillouinZoneFolding
    from .detection import BrillouinZoneDetector

from .mode_profile import ModeProfile


class InteractiveDispersionModes:
    """
    Interactive widget for exploring Brillouin zone folded dispersion.

    Features:
    - Stacked layout: dispersion on top, mode visualization below
    - Click to select mode → visualize m(y, t) heatmap
    - Markers showing k and all k ± n×BZ positions included in mask
    - Slider for number of BZ in mask
    - Option for +k/-k direction selection
    """

    def _base_default_params(self) -> dict[str, object]:
        """Canonical defaults used by fresh and hot-reloaded instances."""
        return {
            "lattice_nm": 470.0,
            "n_bz_mask": 10,
            "k_margin_bins": 0,
            "f_margin_bins": 0,
            "neighbor_reduce": "mean",
            "f_min_ghz": 0.0,
            "f_max_ghz": 10.0,
            "k_direction": "both",
            "cmap_disp": "viridis",
            "cmap_mode": "RdBu_r",
            # Live post-filter defaults
            "live_snr_enabled": False,
            "live_snr_threshold": 3.0,
            "live_gaussian_enabled": False,
            "live_sigma_f": 1.0,
            "live_sigma_k": 1.0,
            "live_gaussian_threshold_std": 1.5,
            "live_wiener_enabled": False,
            "live_wiener_window": 5,
            "live_bandpass_enabled": False,
            "live_kmin_rad_um": -10.0,
            "live_kmax_rad_um": 10.0,
            # Compute-stage recompute filter defaults
            "pre_remove_static": False,
            "pre_remove_average": False,
            "pre_hann_time": False,
            "pre_hann_space": False,
            "pre_envelope_enabled": False,
            "pre_envelope_threshold_std": 2.0,
            "pre_envelope_margin": 10,
            "pre_wavelet_enabled": False,
            "pre_wavelet_level": 3,
            "pre_equalize_enabled": False,
            "pre_compression_enabled": False,
            "pre_compression_alpha": 10.0,
            "pre_welch_enabled": False,
            "pre_welch_segments": 4,
            "pre_welch_overlap": 0.5,
            # Enhancement filters (non-destructive, applied on display)
            "live_log_enabled": False,
            "live_log_method": "log1p",
            "live_gamma_enabled": False,
            "live_gamma_value": 0.5,
            "live_clahe_enabled": False,
            "live_clahe_clip": 0.03,
            "live_clahe_tile": 16,
            "live_lcn_enabled": False,
            "live_lcn_sigma": 10.0,
            "live_unsharp_enabled": False,
            "live_unsharp_sigma": 2.0,
            "live_unsharp_alpha": 1.5,
            "live_percentile_enabled": False,
            "live_percentile_low": 2.0,
            "live_percentile_high": 99.0,
            "live_soft_threshold_enabled": False,
            "live_soft_percentile": 50.0,
            "live_soft_smoothness": 5.0,
        }


    def _ensure_runtime_state(self):
        """Backfill attributes for stale/autoreloaded notebook instances."""
        if not hasattr(self, "_animation"):
            self._animation = None
        if not hasattr(self, "_is_animating"):
            self._is_animating = False
        if not hasattr(self, "_last_compute_kwargs"):
            self._last_compute_kwargs = {}
        if not hasattr(self, "_default_params") or not isinstance(self._default_params, dict):
            self._default_params = {}
        if not hasattr(self, "_presets_dir"):
            self._presets_dir = None
        if not hasattr(self, "_geometry_contour"):
            self._geometry_contour = None
        if not hasattr(self, "_first_dispersion_plot"):
            self._first_dispersion_plot = True
        if not hasattr(self, "_dispersion_xlim"):
            self._dispersion_xlim = None
        if not hasattr(self, "_dispersion_ylim"):
            self._dispersion_ylim = None
        if not hasattr(self, "_first_mode_plot"):
            self._first_mode_plot = True
        if not hasattr(self, "_mode_xlim"):
            self._mode_xlim = None
        if not hasattr(self, "_mode_ylim"):
            self._mode_ylim = None

        base = self._base_default_params()
        for key, value in base.items():
            self._default_params.setdefault(key, value)

    def __init__(self, dispersion_interface: FFTDispersionInterface):
        """Initialize the interactive modes analysis."""
        self.interface = dispersion_interface
        self.result: DispersionResult1D | None = None
        self.folded: FoldedDispersionResult | None = None

        # Lazy import components
        self._detector: BrillouinZoneDetector | None = None
        self._folder: BrillouinZoneFolding | None = None

        # Widget state
        self._widgets_created = False
        self._output: widgets.Output | None = None
        self._fig: Figure | None = None
        self._ax_disp: Axes | None = None
        self._ax_mode: Axes | None = None
        self._colorbar_disp = None
        self._colorbar_mode = None

        # Selection state
        self._selected_k = None
        self._selected_f = None
        self._mask_markers = []  # Artists for mask position markers
        
        # Animation state
        self._animation = None  # FuncAnimation object
        self._is_animating = False
        self._last_compute_kwargs: dict[str, object] = {}

        # Default parameters
        self._default_params = self._base_default_params()

        # Figure settings
        self._dpi = 150
        self._figsize = (10, 10)
        
        # Preset management
        self._presets_dir = None  # Will be set when needed
        
        # Geometry contour overlay (for mode visualization)
        self._geometry_contour: np.ndarray | None = None

    @property
    def detector(self) -> BrillouinZoneDetector:
        """Lazy-load detector."""
        if self._detector is None:
            from .detection import BrillouinZoneDetector

            self._detector = BrillouinZoneDetector()
        return self._detector

    def _get_folder(self, lattice_constant: float, n_periods: int) -> BrillouinZoneFolding:
        """Get or create folder with specified parameters."""
        from .folding import BrillouinZoneFolding

        return BrillouinZoneFolding(lattice_constant, n_periods)
    
    # =========================================================================
    # Preset Management
    # =========================================================================
    
    def _get_presets_dir(self) -> Path:
        """Get or create directory for storing presets.
        
        Presety są zapisywane w podfolderze '.mmpp_presets' w bieżącym
        katalogu roboczym (cwd), co pozwala na osobne presety dla każdego projektu.
        """
        if self._presets_dir is None:
            # Store in current working directory under .mmpp_presets
            import os
            cwd = Path(os.getcwd())
            self._presets_dir = cwd / ".mmpp_presets"
            self._presets_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Preset directory: {self._presets_dir}")
        return self._presets_dir
    
    def _get_current_params(self) -> dict:
        """Extract current parameter values from widgets."""
        if not self._widgets_created:
            return self._default_params.copy()
        
        params = {
            # Basic parameters
            "lattice_nm": float(self.w_lattice.value),
            "n_bz_mask": int(self.w_n_bz_mask.value),
            "k_margin_bins": int(self.w_k_margin.value),
            "f_margin_bins": int(self.w_f_margin.value),
            "neighbor_reduce": str(self.w_neighbor_reduce.value),
            "f_min_ghz": float(self.w_fmin.value),
            "f_max_ghz": float(self.w_fmax.value),
            "k_direction": str(self.w_k_direction.value),
            "cmap_disp": str(self.w_cmap_disp.value),
            "cmap_mode": str(self.w_cmap_mode.value),
            # Live post-filters
            "live_snr_enabled": bool(self.w_live_snr_enabled.value),
            "live_snr_threshold": float(self.w_live_snr_threshold.value),
            "live_gaussian_enabled": bool(self.w_live_gaussian_enabled.value),
            "live_sigma_f": float(self.w_live_sigma_f.value),
            "live_sigma_k": float(self.w_live_sigma_k.value),
            "live_gaussian_threshold_std": float(self.w_live_gaussian_threshold_std.value),
            "live_wiener_enabled": bool(self.w_live_wiener_enabled.value),
            "live_wiener_window": int(self.w_live_wiener_window.value),
            "live_bandpass_enabled": bool(self.w_live_bandpass_enabled.value),
            "live_kmin_rad_um": float(self.w_live_kmin.value),
            "live_kmax_rad_um": float(self.w_live_kmax.value),
            # Compute-stage filters
            "pre_remove_static": bool(self.w_pre_remove_static.value),
            "pre_remove_average": bool(self.w_pre_remove_average.value),
            "pre_hann_time": bool(self.w_pre_hann_time.value),
            "pre_hann_space": bool(self.w_pre_hann_space.value),
            "pre_envelope_enabled": bool(self.w_pre_envelope_enabled.value),
            "pre_envelope_threshold_std": float(self.w_pre_envelope_threshold_std.value),
            "pre_envelope_margin": int(self.w_pre_envelope_margin.value),
            "pre_wavelet_enabled": bool(self.w_pre_wavelet_enabled.value),
            "pre_wavelet_level": int(self.w_pre_wavelet_level.value),
            "pre_equalize_enabled": bool(self.w_pre_equalize_enabled.value),
            "pre_compression_enabled": bool(self.w_pre_compression_enabled.value),
            "pre_compression_alpha": float(self.w_pre_compression_alpha.value),
            "pre_welch_enabled": bool(self.w_pre_welch_enabled.value),
            "pre_welch_segments": int(self.w_pre_welch_segments.value),
            "pre_welch_overlap": float(self.w_pre_welch_overlap.value),
            # Enhancement filters
            "live_log_enabled": bool(self.w_live_log_enabled.value),
            "live_log_method": str(self.w_live_log_method.value),
            "live_gamma_enabled": bool(self.w_live_gamma_enabled.value),
            "live_gamma_value": float(self.w_live_gamma_value.value),
            "live_clahe_enabled": bool(self.w_live_clahe_enabled.value),
            "live_clahe_clip": float(self.w_live_clahe_clip.value),
            "live_clahe_tile": int(self.w_live_clahe_tile.value),
            "live_lcn_enabled": bool(self.w_live_lcn_enabled.value),
            "live_lcn_sigma": float(self.w_live_lcn_sigma.value),
            "live_unsharp_enabled": bool(self.w_live_unsharp_enabled.value),
            "live_unsharp_sigma": float(self.w_live_unsharp_sigma.value),
            "live_unsharp_alpha": float(self.w_live_unsharp_alpha.value),
            "live_percentile_enabled": bool(self.w_live_percentile_enabled.value),
            "live_percentile_low": float(self.w_live_percentile_low.value),
            "live_percentile_high": float(self.w_live_percentile_high.value),
            "live_soft_threshold_enabled": bool(self.w_live_soft_threshold_enabled.value),
            "live_soft_percentile": float(self.w_live_soft_percentile.value),
            "live_soft_smoothness": float(self.w_live_soft_smoothness.value),
        }
        return params
    
    def _apply_params(self, params: dict):
        """Apply parameter values to widgets."""
        if not self._widgets_created:
            self._default_params.update(params)
            return
        
        # Update widgets with new values
        # Basic parameters
        if "lattice_nm" in params:
            self.w_lattice.value = float(params["lattice_nm"])
        if "n_bz_mask" in params:
            self.w_n_bz_mask.value = int(params["n_bz_mask"])
        if "k_margin_bins" in params:
            self.w_k_margin.value = int(params["k_margin_bins"])
        if "f_margin_bins" in params:
            self.w_f_margin.value = int(params["f_margin_bins"])
        if "neighbor_reduce" in params:
            self.w_neighbor_reduce.value = str(params["neighbor_reduce"])
        if "f_min_ghz" in params:
            self.w_fmin.value = float(params["f_min_ghz"])
        if "f_max_ghz" in params:
            self.w_fmax.value = float(params["f_max_ghz"])
        if "k_direction" in params:
            self.w_k_direction.value = str(params["k_direction"])
        if "cmap_disp" in params:
            self.w_cmap_disp.value = str(params["cmap_disp"])
        if "cmap_mode" in params:
            self.w_cmap_mode.value = str(params["cmap_mode"])
        
        # Live post-filters
        if "live_snr_enabled" in params:
            self.w_live_snr_enabled.value = bool(params["live_snr_enabled"])
        if "live_snr_threshold" in params:
            self.w_live_snr_threshold.value = float(params["live_snr_threshold"])
        if "live_gaussian_enabled" in params:
            self.w_live_gaussian_enabled.value = bool(params["live_gaussian_enabled"])
        if "live_sigma_f" in params:
            self.w_live_sigma_f.value = float(params["live_sigma_f"])
        if "live_sigma_k" in params:
            self.w_live_sigma_k.value = float(params["live_sigma_k"])
        if "live_gaussian_threshold_std" in params:
            self.w_live_gaussian_threshold_std.value = float(params["live_gaussian_threshold_std"])
        if "live_wiener_enabled" in params:
            self.w_live_wiener_enabled.value = bool(params["live_wiener_enabled"])
        if "live_wiener_window" in params:
            self.w_live_wiener_window.value = int(params["live_wiener_window"])
        if "live_bandpass_enabled" in params:
            self.w_live_bandpass_enabled.value = bool(params["live_bandpass_enabled"])
        if "live_kmin_rad_um" in params:
            self.w_live_kmin.value = float(params["live_kmin_rad_um"])
        if "live_kmax_rad_um" in params:
            self.w_live_kmax.value = float(params["live_kmax_rad_um"])
        
        # Compute-stage filters
        if "pre_remove_static" in params:
            self.w_pre_remove_static.value = bool(params["pre_remove_static"])
        if "pre_remove_average" in params:
            self.w_pre_remove_average.value = bool(params["pre_remove_average"])
        if "pre_hann_time" in params:
            self.w_pre_hann_time.value = bool(params["pre_hann_time"])
        if "pre_hann_space" in params:
            self.w_pre_hann_space.value = bool(params["pre_hann_space"])
        if "pre_envelope_enabled" in params:
            self.w_pre_envelope_enabled.value = bool(params["pre_envelope_enabled"])
        if "pre_envelope_threshold_std" in params:
            self.w_pre_envelope_threshold_std.value = float(params["pre_envelope_threshold_std"])
        if "pre_envelope_margin" in params:
            self.w_pre_envelope_margin.value = int(params["pre_envelope_margin"])
        if "pre_wavelet_enabled" in params:
            self.w_pre_wavelet_enabled.value = bool(params["pre_wavelet_enabled"])
        if "pre_wavelet_level" in params:
            self.w_pre_wavelet_level.value = int(params["pre_wavelet_level"])
        if "pre_equalize_enabled" in params:
            self.w_pre_equalize_enabled.value = bool(params["pre_equalize_enabled"])
        if "pre_compression_enabled" in params:
            self.w_pre_compression_enabled.value = bool(params["pre_compression_enabled"])
        if "pre_compression_alpha" in params:
            self.w_pre_compression_alpha.value = float(params["pre_compression_alpha"])
        if "pre_welch_enabled" in params:
            self.w_pre_welch_enabled.value = bool(params["pre_welch_enabled"])
        if "pre_welch_segments" in params:
            self.w_pre_welch_segments.value = int(params["pre_welch_segments"])
        if "pre_welch_overlap" in params:
            self.w_pre_welch_overlap.value = float(params["pre_welch_overlap"])
        
        # Enhancement filters
        if "live_log_enabled" in params:
            self.w_live_log_enabled.value = bool(params["live_log_enabled"])
        if "live_log_method" in params:
            self.w_live_log_method.value = str(params["live_log_method"])
        if "live_gamma_enabled" in params:
            self.w_live_gamma_enabled.value = bool(params["live_gamma_enabled"])
        if "live_gamma_value" in params:
            self.w_live_gamma_value.value = float(params["live_gamma_value"])
        if "live_clahe_enabled" in params:
            self.w_live_clahe_enabled.value = bool(params["live_clahe_enabled"])
        if "live_clahe_clip" in params:
            self.w_live_clahe_clip.value = float(params["live_clahe_clip"])
        if "live_clahe_tile" in params:
            self.w_live_clahe_tile.value = int(params["live_clahe_tile"])
        if "live_lcn_enabled" in params:
            self.w_live_lcn_enabled.value = bool(params["live_lcn_enabled"])
        if "live_lcn_sigma" in params:
            self.w_live_lcn_sigma.value = float(params["live_lcn_sigma"])
        if "live_unsharp_enabled" in params:
            self.w_live_unsharp_enabled.value = bool(params["live_unsharp_enabled"])
        if "live_unsharp_sigma" in params:
            self.w_live_unsharp_sigma.value = float(params["live_unsharp_sigma"])
        if "live_unsharp_alpha" in params:
            self.w_live_unsharp_alpha.value = float(params["live_unsharp_alpha"])
        if "live_percentile_enabled" in params:
            self.w_live_percentile_enabled.value = bool(params["live_percentile_enabled"])
        if "live_percentile_low" in params:
            self.w_live_percentile_low.value = float(params["live_percentile_low"])
        if "live_percentile_high" in params:
            self.w_live_percentile_high.value = float(params["live_percentile_high"])
        if "live_soft_threshold_enabled" in params:
            self.w_live_soft_threshold_enabled.value = bool(params["live_soft_threshold_enabled"])
        if "live_soft_percentile" in params:
            self.w_live_soft_percentile.value = float(params["live_soft_percentile"])
        if "live_soft_smoothness" in params:
            self.w_live_soft_smoothness.value = float(params["live_soft_smoothness"])
    
    def save_preset(self, name: str) -> bool:
        """Save current parameters as a preset.
        
        Preset jest zapisywany w podfolderze '.mmpp_presets' w bieżącym
        katalogu roboczym, dzięki czemu każdy projekt może mieć własne presety.
        
        Parameters
        ----------
        name : str
            Name of the preset (without .json extension)
            
        Returns
        -------
        bool
            True if saved successfully, False otherwise
        """
        try:
            # Sanitize name
            name = name.strip().replace("/", "_").replace("\\", "_")
            if not name:
                logger.warning("Preset name cannot be empty")
                return False
            
            presets_dir = self._get_presets_dir()
            preset_file = presets_dir / f"{name}.json"
            
            params = self._get_current_params()
            
            # Add metadata
            from datetime import datetime
            preset_data = {
                "created": datetime.now().isoformat(),
                "params": params
            }
            
            with open(preset_file, 'w') as f:
                json.dump(preset_data, f, indent=2)
            
            logger.info(f"Preset '{name}' saved to {preset_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save preset '{name}': {e}")
            return False
    
    def load_preset(self, name: str) -> bool:
        """Load parameters from a preset.
        
        Parameters
        ----------
        name : str
            Name of the preset (without .json extension)
            
        Returns
        -------
        bool
            True if loaded successfully, False otherwise
        """
        try:
            presets_dir = self._get_presets_dir()
            preset_file = presets_dir / f"{name}.json"
            
            if not preset_file.exists():
                logger.warning(f"Preset '{name}' not found at {preset_file}")
                return False
            
            with open(preset_file, 'r') as f:
                preset_data = json.load(f)
            
            # Handle both old format (direct params) and new format (with metadata)
            if isinstance(preset_data, dict) and "params" in preset_data:
                params = preset_data["params"]
            else:
                params = preset_data
            
            self._apply_params(params)
            logger.info(f"Preset '{name}' loaded from {preset_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to load preset '{name}': {e}")
            return False
    
    def delete_preset(self, name: str) -> bool:
        """Delete a saved preset.
        
        Parameters
        ----------
        name : str
            Name of the preset (without .json extension)
            
        Returns
        -------
        bool
            True if deleted successfully, False otherwise
        """
        try:
            presets_dir = self._get_presets_dir()
            preset_file = presets_dir / f"{name}.json"
            
            if not preset_file.exists():
                logger.warning(f"Preset '{name}' not found")
                return False
            
            preset_file.unlink()
            logger.info(f"Preset '{name}' deleted")
            return True
        except Exception as e:
            logger.error(f"Failed to delete preset '{name}': {e}")
            return False
    
    def list_presets(self) -> list[str]:
        """List all available presets.
        
        Returns
        -------
        list[str]
            List of preset names (without .json extension)
        """
        try:
            presets_dir = self._get_presets_dir()
            preset_files = list(presets_dir.glob("*.json"))
            return sorted([f.stem for f in preset_files])
        except Exception as e:
            logger.error(f"Failed to list presets: {e}")
            return []

    def fold(
        self,
        result: DispersionResult1D | None = None,
        lattice_constant: float | None = None,
        n_periods: int = 3,
        peak_threshold: float = 0.01,
        **compute_kwargs,
    ) -> FoldedDispersionResult:
        """Fold dispersion to first Brillouin zone (non-interactive)."""
        if result is None:
            result = self.interface.compute_1d(**compute_kwargs)
        self.result = result

        if lattice_constant is None:
            lattice_constant = 470e-9
            logger.info("Using default lattice constant: %.1f nm", lattice_constant * 1e9)

        folder = self._get_folder(lattice_constant, n_periods)
        self.folded = folder.fold_dispersion(result, peak_threshold)

        return self.folded

    def plot_interactive(
        self,
        result: DispersionResult1D | None = None,
        figsize: tuple[float, float] = (10, 10),
        dpi: int = 150,
        lattice_constant_nm: float | None = None,
        add_contour: np.ndarray | None = None,
        **compute_kwargs,
    ):
        """
        Launch interactive widget for dispersion mode exploration.

        Parameters
        ----------
        result : DispersionResult1D, optional
            Pre-computed dispersion. If None, uses self.result or computes automatically.
        figsize : tuple
            Figure size (width, height) in inches
        dpi : int
            Figure DPI (default 150)
        lattice_constant_nm : float, optional
            Initial lattice constant in nm. If None, uses default from dispersion_modes().
        add_contour : np.ndarray, optional
            2D geometry array (0/1) to overlay as contour on mode visualization.
            This is useful for showing material boundaries (e.g., oscillators, antidots).
        **compute_kwargs : dict
            Extra kwargs passed to compute_1d if result needs to be computed.
        """

        self._ensure_runtime_state()

        if not _HAS_WIDGETS:
            raise ImportError(
                "ipywidgets is required for interactive mode. "
                "Install with: pip install ipywidgets"
            )
        if not _HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for plotting.")

        self._dpi = dpi
        self._figsize = figsize

        # Priority: explicit result > self.result > compute
        if result is not None:
            self.result = result
            self._last_compute_kwargs = {}
        elif self.result is None:
            logger.info("Computing dispersion result...")
            self._last_compute_kwargs = dict(compute_kwargs)
            self.result = self.interface.compute_1d(**compute_kwargs)

        # Set lattice constant (use provided or keep default from dispersion_modes())
        if lattice_constant_nm is not None:
            self._default_params["lattice_nm"] = lattice_constant_nm

        # Set initial parameters from result
        f_max_ghz = self.result.f_axis.max() / 1e9
        self._default_params["f_max_ghz"] = min(f_max_ghz, 20.0)

        # Store geometry contour for mode overlay
        if add_contour is not None:
            # Squeeze to 2D if needed
            geom = np.asarray(add_contour).squeeze()
            if geom.ndim == 2:
                self._geometry_contour = geom
                logger.info(f"Geometry contour set with shape {geom.shape}")
            else:
                logger.warning(f"add_contour must be 2D after squeeze, got {geom.ndim}D")
                self._geometry_contour = None
        else:
            self._geometry_contour = None

        # Create widgets
        self._create_widgets()

        # Create the main layout
        main_layout = self._create_layout()

        # Display
        display(main_layout)

        # Initial plot
        self._initialize_figure()
        self._update_dispersion_plot()

    def _create_widgets(self):
        """Create all interactive widgets."""
        self._ensure_runtime_state()
        params = self._default_params

        # === Lattice parameters ===
        self.w_lattice = widgets.FloatSlider(
            value=params["lattice_nm"],
            min=50,
            max=20000,
            step=10,
            description="a [nm]:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )

        self.w_n_bz_mask = widgets.IntSlider(
            value=params["n_bz_mask"],
            min=0,
            max=40,
            step=1,
            description="N_BZ mask:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )

        self.w_show_bz_lines = widgets.Checkbox(
            value=True,
            description="Show BZ lines",
            layout=widgets.Layout(width="95%"),
            tooltip="Show/hide Brillouin zone boundary lines on dispersion plot",
        )

        # === Frequency range ===
        self.w_fmin = widgets.FloatSlider(
            value=params["f_min_ghz"],
            min=0,
            max=params["f_max_ghz"],
            step=0.1,
            description="f min [GHz]:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )

        self.w_fmax = widgets.FloatSlider(
            value=params["f_max_ghz"],
            min=0.1,
            max=params["f_max_ghz"] * 1.5,
            step=0.1,
            description="f max [GHz]:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )

        # === k-direction selection ===
        self.w_k_direction = widgets.Dropdown(
            options=[
                ("Both ±k", "both"),
                ("Only +k", "positive"),
                ("Only -k", "negative"),
            ],
            value=params["k_direction"],
            description="k-dirs:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
        )

        self.w_k_margin = widgets.IntSlider(
            value=params["k_margin_bins"],
            min=0,
            max=3,
            step=1,
            description="k margin:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )

        self.w_f_margin = widgets.IntSlider(
            value=params["f_margin_bins"],
            min=0,
            max=3,
            step=1,
            description="f margin:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )

        self.w_neighbor_reduce = widgets.Dropdown(
            options=[
                ("Mean (recommended)", "mean"),
                ("Sum", "sum"),
            ],
            value=params["neighbor_reduce"],
            description="Nbh agg:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
        )

        # === Mode visualization type ===
        self.w_mode_type = widgets.Dropdown(
            options=[
                ("Real part", "real"),
                ("Imaginary part", "imag"),
                ("Amplitude |M|", "abs"),
                ("Phase φ[M]", "phase"),
                ("Ampl×Phase", "ampl_phase"),
            ],
            value="real",
            description="Mode type:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
        )

        self.w_mode_x_periods = widgets.FloatSlider(
            value=3.0,
            min=0.5,
            max=5.0,
            step=0.5,
            description="x width:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            tooltip="Number of lattice periods to show in x direction",
            continuous_update=False,
        )

        # === Colormaps ===
        self.w_cmap_disp = widgets.Dropdown(
            options=["viridis", "plasma", "cividis", "turbo", "inferno"],
            value=params["cmap_disp"],
            description="Cmap disp:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
        )

        self.w_cmap_mode = widgets.Dropdown(
            options=["RdBu_r", "seismic", "coolwarm", "bwr", "PiYG"],
            value="RdBu_r",
            description="Cmap mode:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
        )

        # === Live post-filters (from cached S(k,f)) ===
        self.w_live_snr_enabled = widgets.Checkbox(
            value=params["live_snr_enabled"],
            description="SNR filter",
            layout=widgets.Layout(width="95%"),
        )
        self.w_live_snr_threshold = widgets.FloatSlider(
            value=params["live_snr_threshold"],
            min=0.5,
            max=10.0,
            step=0.1,
            description="SNR thr:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )
        self.w_live_gaussian_enabled = widgets.Checkbox(
            value=params["live_gaussian_enabled"],
            description="Gauss+morph",
            layout=widgets.Layout(width="95%"),
        )
        self.w_live_sigma_f = widgets.FloatSlider(
            value=params["live_sigma_f"],
            min=0.0,
            max=4.0,
            step=0.1,
            description="σf:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )
        self.w_live_sigma_k = widgets.FloatSlider(
            value=params["live_sigma_k"],
            min=0.0,
            max=4.0,
            step=0.1,
            description="σk:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )
        self.w_live_gaussian_threshold_std = widgets.FloatSlider(
            value=params["live_gaussian_threshold_std"],
            min=0.0,
            max=4.0,
            step=0.1,
            description="thr std:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )
        self.w_live_wiener_enabled = widgets.Checkbox(
            value=params["live_wiener_enabled"],
            description="Wiener2D",
            layout=widgets.Layout(width="95%"),
        )
        self.w_live_wiener_window = widgets.IntSlider(
            value=params["live_wiener_window"],
            min=1,
            max=21,
            step=2,
            description="W win:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )
        self.w_live_bandpass_enabled = widgets.Checkbox(
            value=params["live_bandpass_enabled"],
            description="Bandpass (k,f)",
            layout=widgets.Layout(width="95%"),
        )
        self.w_live_kmin = widgets.FloatSlider(
            value=params["live_kmin_rad_um"],
            min=-20.0,
            max=0.0,
            step=0.1,
            description="k min:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )
        self.w_live_kmax = widgets.FloatSlider(
            value=params["live_kmax_rad_um"],
            min=0.0,
            max=20.0,
            step=0.1,
            description="k max:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
        )

        # === Enhancement Filters (Non-destructive, applied on display) ===
        self.w_live_log_enabled = widgets.Checkbox(
            value=params["live_log_enabled"],
            description="Log transform",
            layout=widgets.Layout(width="95%"),
        )
        self.w_live_log_method = widgets.Dropdown(
            options=[
                ("log1p (smooth)", "log1p"),
                ("log10 (classic)", "log10"),
                ("arcsinh (symmetric)", "arcsinh"),
                ("sqrt (mild)", "sqrt"),
            ],
            value=params["live_log_method"],
            description="Method:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
        )
        self.w_live_gamma_enabled = widgets.Checkbox(
            value=params["live_gamma_enabled"],
            description="Gamma correction",
            layout=widgets.Layout(width="95%"),
        )
        self.w_live_gamma_value = widgets.FloatSlider(
            value=params["live_gamma_value"],
            min=0.1,
            max=2.0,
            step=0.05,
            description="γ:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )
        self.w_live_clahe_enabled = widgets.Checkbox(
            value=params["live_clahe_enabled"],
            description="CLAHE (local contrast)",
            layout=widgets.Layout(width="95%"),
        )
        self.w_live_clahe_clip = widgets.FloatSlider(
            value=params["live_clahe_clip"],
            min=0.01,
            max=0.5,
            step=0.01,
            description="Clip limit:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )
        self.w_live_clahe_tile = widgets.IntSlider(
            value=params["live_clahe_tile"],
            min=4,
            max=64,
            step=4,
            description="Tile size:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )
        self.w_live_lcn_enabled = widgets.Checkbox(
            value=params["live_lcn_enabled"],
            description="Local Contrast Norm",
            layout=widgets.Layout(width="95%"),
        )
        self.w_live_lcn_sigma = widgets.FloatSlider(
            value=params["live_lcn_sigma"],
            min=2.0,
            max=50.0,
            step=1.0,
            description="σ (smooth):",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )
        self.w_live_unsharp_enabled = widgets.Checkbox(
            value=params["live_unsharp_enabled"],
            description="Unsharp mask (edges)",
            layout=widgets.Layout(width="95%"),
        )
        self.w_live_unsharp_sigma = widgets.FloatSlider(
            value=params["live_unsharp_sigma"],
            min=0.5,
            max=10.0,
            step=0.5,
            description="σ blur:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )
        self.w_live_unsharp_alpha = widgets.FloatSlider(
            value=params["live_unsharp_alpha"],
            min=0.1,
            max=5.0,
            step=0.1,
            description="α strength:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )
        self.w_live_percentile_enabled = widgets.Checkbox(
            value=params["live_percentile_enabled"],
            description="Percentile autoscale",
            layout=widgets.Layout(width="95%"),
        )
        self.w_live_percentile_low = widgets.FloatSlider(
            value=params["live_percentile_low"],
            min=0.0,
            max=50.0,
            step=0.5,
            description="Low %:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )
        self.w_live_percentile_high = widgets.FloatSlider(
            value=params["live_percentile_high"],
            min=50.0,
            max=100.0,
            step=0.5,
            description="High %:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )
        self.w_live_soft_threshold_enabled = widgets.Checkbox(
            value=params["live_soft_threshold_enabled"],
            description="Soft threshold",
            layout=widgets.Layout(width="95%"),
        )
        self.w_live_soft_percentile = widgets.FloatSlider(
            value=params["live_soft_percentile"],
            min=0.0,
            max=100.0,
            step=5.0,
            description="Thr %:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )
        self.w_live_soft_smoothness = widgets.FloatSlider(
            value=params["live_soft_smoothness"],
            min=1.0,
            max=20.0,
            step=0.5,
            description="Smooth:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )

        # === Compute-stage filters (require FFT recompute) ===
        self.w_pre_remove_static = widgets.Checkbox(
            value=params["pre_remove_static"],
            description="remove_static",
            layout=widgets.Layout(width="95%"),
        )
        self.w_pre_remove_average = widgets.Checkbox(
            value=params["pre_remove_average"],
            description="remove_average",
            layout=widgets.Layout(width="95%"),
        )
        self.w_pre_hann_time = widgets.Checkbox(
            value=params["pre_hann_time"],
            description="hann_time",
            layout=widgets.Layout(width="95%"),
        )
        self.w_pre_hann_space = widgets.Checkbox(
            value=params["pre_hann_space"],
            description="hann_space",
            layout=widgets.Layout(width="95%"),
        )
        self.w_pre_envelope_enabled = widgets.Checkbox(
            value=params["pre_envelope_enabled"],
            description="envelope",
            layout=widgets.Layout(width="95%"),
        )
        self.w_pre_envelope_threshold_std = widgets.FloatSlider(
            value=params["pre_envelope_threshold_std"],
            min=0.0,
            max=5.0,
            step=0.1,
            description="env thr:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )
        self.w_pre_envelope_margin = widgets.IntSlider(
            value=params["pre_envelope_margin"],
            min=0,
            max=50,
            step=1,
            description="env marg:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )
        self.w_pre_wavelet_enabled = widgets.Checkbox(
            value=params["pre_wavelet_enabled"],
            description="wavelet1D",
            layout=widgets.Layout(width="95%"),
        )
        self.w_pre_wavelet_level = widgets.IntSlider(
            value=params["pre_wavelet_level"],
            min=1,
            max=6,
            step=1,
            description="w lvl:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )
        self.w_pre_equalize_enabled = widgets.Checkbox(
            value=params["pre_equalize_enabled"],
            description="amp equalize",
            layout=widgets.Layout(width="95%"),
        )
        self.w_pre_compression_enabled = widgets.Checkbox(
            value=params["pre_compression_enabled"],
            description="compression",
            layout=widgets.Layout(width="95%"),
        )
        self.w_pre_compression_alpha = widgets.FloatSlider(
            value=params["pre_compression_alpha"],
            min=1.0,
            max=50.0,
            step=0.5,
            description="α:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )
        self.w_pre_welch_enabled = widgets.Checkbox(
            value=params["pre_welch_enabled"],
            description="Welch avg",
            layout=widgets.Layout(width="95%"),
        )
        self.w_pre_welch_segments = widgets.IntSlider(
            value=params["pre_welch_segments"],
            min=2,
            max=12,
            step=1,
            description="segments:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )
        self.w_pre_welch_overlap = widgets.FloatSlider(
            value=params["pre_welch_overlap"],
            min=0.0,
            max=0.9,
            step=0.05,
            description="overlap:",
            layout=widgets.Layout(width="95%"),
            style={"description_width": "70px"},
            continuous_update=False,
        )

        # === Buttons ===
        self.w_update = widgets.Button(
            description="🔄 Refresh Plot",
            button_style="success",
            layout=widgets.Layout(width="95%"),
        )

        self.w_reset_zoom = widgets.Button(
            description="🔍 Reset Zoom",
            button_style="",
            layout=widgets.Layout(width="95%"),
            tooltip="Reset zoom for both dispersion and mode plots",
        )

        self.w_auto_detect = widgets.Button(
            description="🔍 Auto-detect a",
            button_style="info",
            layout=widgets.Layout(width="95%"),
        )

        self.w_show_system_info = widgets.Button(
            description="ℹ️ System Info",
            button_style="",
            layout=widgets.Layout(width="95%"),
            tooltip="Show simulation parameters and data dimensions",
        )

        self.w_animate = widgets.Button(
            description="🎬 Animate Mode",
            button_style="warning",
            layout=widgets.Layout(width="95%"),
            tooltip="Toggle mode oscillation animation (full 2π cycle)",
        )

        # === Animation controls ===
        self.w_anim_frames = widgets.IntSlider(
            value=60,
            min=10,
            max=200,
            step=10,
            description="Frames:",
            style={'description_width': '60px'},
            layout=widgets.Layout(width="95%"),
            tooltip="Number of frames per animation cycle (higher = smoother but slower)",
        )

        self.w_anim_fps = widgets.IntSlider(
            value=30,
            min=10,
            max=60,
            step=5,
            description="FPS:",
            style={'description_width': '60px'},
            layout=widgets.Layout(width="95%"),
            tooltip="Frames per second (higher = faster animation)",
        )

        self.w_save_animation = widgets.Button(
            description="💾 Save Animation",
            button_style="",
            layout=widgets.Layout(width="95%"),
            tooltip="Save animation as MP4 or GIF file",
        )

        self.w_anim_save_mode = widgets.Dropdown(
            options=[
                ("Mode only", "mode"),
                ("Full view (dispersion + mode)", "full"),
            ],
            value="mode",
            description="View:",
            style={'description_width': '60px'},
            layout=widgets.Layout(width="95%"),
            tooltip="Choose what to save: mode animation only or full interface",
        )

        self.w_anim_file_format = widgets.Dropdown(
            options=[
                ("GIF (animated)", "gif"),
                ("MP4 (video)", "mp4"),
            ],
            value="gif",
            description="Format:",
            style={'description_width': '60px'},
            layout=widgets.Layout(width="95%"),
            tooltip="Choose file format: GIF or MP4",
        )

        self.w_recompute = widgets.Button(
            description="♻️ Recompute FFT",
            button_style="primary",
            layout=widgets.Layout(width="95%"),
            tooltip="Recompute dispersion with selected compute-stage filters",
        )

        # === Info display ===
        self.w_info = widgets.HTML(
            value="<small>Click on dispersion to select mode (k, f)</small>",
        )

        self.w_mode_info = widgets.HTML(
            value="<small>No mode selected</small>",
        )

        self._output = widgets.Output(layout=widgets.Layout(width="100%", height="auto"))

        # === Connect callbacks ===
        self.w_update.on_click(self._on_update)
        self.w_reset_zoom.on_click(self._on_reset_zoom)
        self.w_auto_detect.on_click(self._on_auto_detect)
        self.w_show_system_info.on_click(self._on_show_system_info)
        self.w_animate.on_click(self._on_animate)
        self.w_save_animation.on_click(self._on_save_animation)
        self.w_recompute.on_click(self._on_recompute_dispersion)

        # Connect changes that should update immediately
        # w_lattice also updates BZ lines on dispersion plot
        for w in [self.w_cmap_disp, self.w_fmin, self.w_fmax, self.w_lattice, self.w_show_bz_lines]:
            w.observe(self._on_display_param_change, names="value")

        # Connect mode visualization params
        # w_lattice affects BZ mask positions for mode extraction
        for w in [
            self.w_n_bz_mask,
            self.w_k_direction,
            self.w_k_margin,
            self.w_f_margin,
            self.w_neighbor_reduce,
            self.w_mode_type,
            self.w_cmap_mode,
            self.w_lattice,
        ]:
            w.observe(self._on_mode_param_change, names="value")
        
        # x_periods slider needs special handling to reset zoom
        self.w_mode_x_periods.observe(self._on_x_periods_change, names="value")
        
        # Watch n_bz to show/hide k-direction widget
        self.w_n_bz_mask.observe(self._on_n_bz_change, names="value")
        self._update_k_direction_visibility()

        # Animation controls - restart animation if active
        for w in [self.w_anim_frames, self.w_anim_fps]:
            w.observe(self._on_anim_param_change, names="value")

        # Live post-filter controls update plot immediately.
        for w in [
            self.w_live_snr_enabled,
            self.w_live_snr_threshold,
            self.w_live_gaussian_enabled,
            self.w_live_sigma_f,
            self.w_live_sigma_k,
            self.w_live_gaussian_threshold_std,
            self.w_live_wiener_enabled,
            self.w_live_wiener_window,
            self.w_live_bandpass_enabled,
            self.w_live_kmin,
            self.w_live_kmax,
            # Enhancement filters (non-destructive)
            self.w_live_log_enabled,
            self.w_live_log_method,
            self.w_live_gamma_enabled,
            self.w_live_gamma_value,
            self.w_live_clahe_enabled,
            self.w_live_clahe_clip,
            self.w_live_clahe_tile,
            self.w_live_lcn_enabled,
            self.w_live_lcn_sigma,
            self.w_live_unsharp_enabled,
            self.w_live_unsharp_sigma,
            self.w_live_unsharp_alpha,
            self.w_live_percentile_enabled,
            self.w_live_percentile_low,
            self.w_live_percentile_high,
            self.w_live_soft_threshold_enabled,
            self.w_live_soft_percentile,
            self.w_live_soft_smoothness,
        ]:
            w.observe(self._on_live_filter_change, names="value")

        # === Preset Management (Top Priority) ===
        available_presets = self.list_presets()
        preset_options = [("-- Load Preset --", "")] + [(name, name) for name in available_presets]
        
        self.w_preset_load = widgets.Dropdown(
            options=preset_options,
            value="",
            description="",
            layout=widgets.Layout(width="calc(100% - 90px)"),
            tooltip="Load saved preset from current folder",
        )
        
        self.w_preset_refresh_btn = widgets.Button(
            description="🔄",
            button_style="",
            layout=widgets.Layout(width="40px"),
            tooltip="Refresh preset list",
        )
        
        self.w_preset_name = widgets.Text(
            value="",
            placeholder="Preset name...",
            description="",
            layout=widgets.Layout(width="calc(100% - 90px)"),
        )
        
        self.w_preset_save_btn = widgets.Button(
            description="💾",
            button_style="success",
            layout=widgets.Layout(width="40px"),
            tooltip="Save current settings as preset",
        )
        
        self.w_preset_delete_btn = widgets.Button(
            description="🗑️",
            button_style="danger",
            layout=widgets.Layout(width="40px"),
            tooltip="Delete selected preset",
        )
        
        # Connect preset callbacks
        self.w_preset_save_btn.on_click(self._on_save_preset)
        self.w_preset_delete_btn.on_click(self._on_delete_preset)
        self.w_preset_refresh_btn.on_click(self._on_refresh_presets)
        self.w_preset_load.observe(self._on_load_preset, names="value")

        self._widgets_created = True

    def _create_layout(self) -> widgets.Widget:
        """Create layout with controls on left, stacked plots on right."""

        live_filters_box = widgets.VBox(
            [
                self.w_live_snr_enabled,
                self.w_live_snr_threshold,
                self.w_live_gaussian_enabled,
                self.w_live_sigma_f,
                self.w_live_sigma_k,
                self.w_live_gaussian_threshold_std,
                self.w_live_wiener_enabled,
                self.w_live_wiener_window,
                self.w_live_bandpass_enabled,
                self.w_live_kmin,
                self.w_live_kmax,
                widgets.HTML("<small>Bandpass uses current f min/f max sliders.</small>"),
            ],
            layout=widgets.Layout(width="100%"),
        )

        compute_filters_box = widgets.VBox(
            [
                self.w_pre_remove_static,
                self.w_pre_remove_average,
                self.w_pre_hann_time,
                self.w_pre_hann_space,
                self.w_pre_envelope_enabled,
                self.w_pre_envelope_threshold_std,
                self.w_pre_envelope_margin,
                self.w_pre_wavelet_enabled,
                self.w_pre_wavelet_level,
                self.w_pre_equalize_enabled,
                self.w_pre_compression_enabled,
                self.w_pre_compression_alpha,
                self.w_pre_welch_enabled,
                self.w_pre_welch_segments,
                self.w_pre_welch_overlap,
                self.w_recompute,
            ],
            layout=widgets.Layout(width="100%"),
        )

        # Enhancement filters (non-destructive, image-like processing)
        enhancement_filters_box = widgets.VBox(
            [
                widgets.HTML("<small><b>Dynamic Range</b></small>"),
                self.w_live_log_enabled,
                self.w_live_log_method,
                self.w_live_gamma_enabled,
                self.w_live_gamma_value,
                self.w_live_percentile_enabled,
                self.w_live_percentile_low,
                self.w_live_percentile_high,
                widgets.HTML("<hr style='margin:5px'>"),
                widgets.HTML("<small><b>Contrast Enhancement</b></small>"),
                self.w_live_clahe_enabled,
                self.w_live_clahe_clip,
                self.w_live_clahe_tile,
                self.w_live_lcn_enabled,
                self.w_live_lcn_sigma,
                widgets.HTML("<hr style='margin:5px'>"),
                widgets.HTML("<small><b>Edge Enhancement</b></small>"),
                self.w_live_unsharp_enabled,
                self.w_live_unsharp_sigma,
                self.w_live_unsharp_alpha,
                widgets.HTML("<hr style='margin:5px'>"),
                widgets.HTML("<small><b>Noise Suppression (soft)</b></small>"),
                self.w_live_soft_threshold_enabled,
                self.w_live_soft_percentile,
                self.w_live_soft_smoothness,
                widgets.HTML("<small style='color:#888'>These filters enhance visibility without destroying data.</small>"),
            ],
            layout=widgets.Layout(width="100%"),
        )

        filters_accordion = widgets.Accordion(
            children=[enhancement_filters_box, live_filters_box, compute_filters_box],
            selected_index=0,  # Enhancement filters open by default
            layout=widgets.Layout(width="95%"),
        )
        filters_accordion.set_title(0, "✨ Enhancement (fast)")
        filters_accordion.set_title(1, "🔧 Classic post-filters")
        filters_accordion.set_title(2, "♻️ Compute filters (recompute)")


        # Preset controls at the very top
        preset_load_box = widgets.HBox(
            [self.w_preset_load, self.w_preset_refresh_btn],
            layout=widgets.Layout(width="100%")
        )
        
        preset_save_box = widgets.HBox(
            [self.w_preset_name, self.w_preset_save_btn],
            layout=widgets.Layout(width="100%")
        )
        
        preset_controls = widgets.VBox(
            [
                widgets.HTML("<small style='color:#666'><b>📁 Presets</b></small>"),
                preset_load_box,
                preset_save_box,
                self.w_preset_delete_btn,
            ],
            layout=widgets.Layout(width="100%", padding="3px")
        )

        # === TAB 1: Dispersion Parameters ===
        tab_dispersion = widgets.VBox(
            [
                widgets.HTML("<small><b>Lattice & BZ</b></small>"),
                self.w_lattice,
                self.w_auto_detect,
                self.w_show_bz_lines,
                self.w_show_system_info,
                widgets.HTML("<hr style='margin:5px'>"),
                widgets.HTML("<small><b>Frequency Range</b></small>"),
                self.w_fmin,
                self.w_fmax,
                widgets.HTML("<hr style='margin:5px'>"),
                widgets.HTML("<small><b>Display</b></small>"),
                self.w_cmap_disp,
            ],
            layout=widgets.Layout(width="100%", padding="5px")
        )

        # === TAB 2: Mode Parameters ===
        tab_modes = widgets.VBox(
            [
                widgets.HTML("<small><b>BZ Mask Settings</b></small>"),
                self.w_n_bz_mask,
                self.w_k_direction,
                self.w_k_margin,
                self.w_f_margin,
                self.w_neighbor_reduce,
                widgets.HTML("<hr style='margin:5px'>"),
                widgets.HTML("<small><b>Visualization</b></small>"),
                self.w_mode_type,
                self.w_mode_x_periods,
                self.w_cmap_mode,
            ],
            layout=widgets.Layout(width="100%", padding="5px")
        )

        # === TAB 3: Actions & Animation ===
        tab_actions = widgets.VBox(
            [
                widgets.HTML("<small><b>Plot Controls</b></small>"),
                self.w_update,
                self.w_reset_zoom,
                widgets.HTML("<hr style='margin:5px'>"),
                widgets.HTML("<small><b>Animation</b></small>"),
                self.w_animate,
                self.w_anim_frames,
                self.w_anim_fps,
                self.w_anim_save_mode,
                self.w_anim_file_format,
                self.w_save_animation,
            ],
            layout=widgets.Layout(width="100%", padding="5px")
        )

        # === TAB 4: Filters ===
        tab_filters = widgets.VBox(
            [
                filters_accordion,
            ],
            layout=widgets.Layout(width="100%", padding="5px")
        )

        # Create tabs
        tabs = widgets.Tab(
            children=[tab_dispersion, tab_modes, tab_actions, tab_filters],
            layout=widgets.Layout(width="100%")
        )
        tabs.set_title(0, "📊 Dispersion")
        tabs.set_title(1, "🎯 Modes")
        tabs.set_title(2, "⚡ Actions")
        tabs.set_title(3, "🔧 Filters")

        # Left panel: controls
        left_panel = widgets.VBox(
            [
                widgets.HTML("<b>🌊 BZ Mode Analysis</b>"),
                widgets.HTML("<hr style='margin:3px'>"),
                preset_controls,
                widgets.HTML("<hr style='margin:3px'>"),
                tabs,
                widgets.HTML("<hr style='margin:5px'>"),
                self.w_info,
                widgets.HTML("<hr style='margin:3px'>"),
                widgets.HTML("<small><b>Selected Mode</b></small>"),
                self.w_mode_info,
            ],
            layout=widgets.Layout(
                width="320px",
                padding="5px",
                border="1px solid #ddd",
            ),
        )

        # Right panel: stacked plots
        right_panel = widgets.VBox(
            [
                self._output,
            ],
            layout=widgets.Layout(
                width="calc(100% - 340px)",
                min_width="760px",
            ),
        )

        # Main layout
        main = widgets.HBox(
            [
                left_panel,
                right_panel,
            ],
            layout=widgets.Layout(
                width="100%",
            ),
        )

        return main

    def _initialize_figure(self):
        """Create the matplotlib figure with two subplots."""
        with self._output:
            clear_output(wait=True)

            plt.ioff()

            # Create figure with 2 rows: dispersion on top, mode viz below
            self._fig, (self._ax_disp, self._ax_mode) = plt.subplots(
                2,
                1,
                figsize=self._figsize,
                dpi=self._dpi,
                gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.25},
            )

            # Connect click events
            self._fig.canvas.mpl_connect("button_press_event", self._on_click)

            plt.ion()
            plt.show()

    def _on_click(self, event):
        """Handle mouse click on the dispersion plot."""
        if event.inaxes != self._ax_disp:
            return

        k_clicked = event.xdata * 1e6  # μm^-1 → m^-1
        f_clicked = event.ydata * 1e9  # GHz → Hz

        self._selected_k = k_clicked
        self._selected_f = f_clicked

        # Update mode info
        self.w_mode_info.value = (
            f"<small><b>k</b> = {k_clicked/1e6:.2f} rad/μm<br>"
            f"<b>f</b> = {f_clicked/1e9:.2f} GHz</small>"
        )

        # Update visualization
        self._update_mode_visualization()

    def _on_update(self, _):
        """Handle update button click."""
        self._update_dispersion_plot()
        self._refresh_mode_or_animation()

    def _on_reset_zoom(self, _):
        """Reset both dispersion and mode plot zoom to defaults."""
        # Set flags to trigger default limits on next update
        self._first_dispersion_plot = True
        self._first_mode_plot = True
        self._update_dispersion_plot()
        self._refresh_mode_or_animation()
        self.w_info.value = "<small style='color:green'>✅ Zoom reset for both plots</small>"

    def _on_recompute_dispersion(self, _):
        """Recompute dispersion with selected compute-stage filters."""
        if self.result is None:
            self.w_info.value = "<small style='color:red'>⚠️ No dispersion result to recompute</small>"
            return

        self.w_recompute.disabled = True
        old_desc = self.w_recompute.description
        self.w_recompute.description = "⏳ Recomputing..."

        try:
            filters_cfg = self._build_compute_filters_config()

            # Preserve interactive mode extraction capability.
            axis = self.result.axis
            component = self.result.component
            compute_kwargs = dict(self._last_compute_kwargs)
            compute_kwargs["axis"] = axis
            compute_kwargs["component"] = component
            compute_kwargs["avg_over_orthogonal"] = False
            compute_kwargs["force"] = True  # Force recompute from original data
            compute_kwargs["save"] = False

            # CRITICAL: Clear any cached filtered results to ensure we start from original raw data
            # This prevents "sticky filters" where unchecked filters still affect results
            if hasattr(self.interface, '_clear_cache'):
                self.interface._clear_cache()

            if filters_cfg:
                compute_kwargs["filters"] = filters_cfg
            else:
                # Explicitly set filters to None/empty to ensure no filters are applied
                compute_kwargs["filters"] = None

            t0 = time.perf_counter()
            self.result = self.interface.compute_1d(**compute_kwargs)
            elapsed = time.perf_counter() - t0
            self._last_compute_kwargs = {
                k: v for k, v in compute_kwargs.items()
                if k not in {"force", "save", "filters"}
            }

            # Refresh frequency slider bounds for new result.
            f_max_ghz = float(self.result.f_axis.max() / 1e9)
            self.w_fmin.max = max(f_max_ghz, 0.1)
            self.w_fmax.max = max(f_max_ghz * 1.5, 0.2)
            if self.w_fmax.value > self.w_fmax.max:
                self.w_fmax.value = self.w_fmax.max

            # Count active filters for user info
            n_filters = 0
            if filters_cfg:
                n_filters += sum(1 for k, v in filters_cfg.items() if k in ["remove_static", "remove_average", "hann_time", "hann_space"] and v)
                if "pre" in filters_cfg:
                    n_filters += len([k for k, v in filters_cfg["pre"].items() if isinstance(v, dict) and v.get("enabled")])
            
            filter_info = f" | {n_filters} filter(s) applied" if n_filters > 0 else " | no filters (original data)"
            self.w_info.value = (
                f"<small style='color:green'>✅ Recomputed from original data in {elapsed:.2f} s{filter_info}</small>"
            )

            self._update_dispersion_plot()
            self._refresh_mode_or_animation()
        except Exception as exc:
            logger.exception("Dispersion recompute failed")
            self.w_info.value = f"<small style='color:red'>❌ Recompute error: {exc}</small>"
        finally:
            self.w_recompute.description = old_desc
            self.w_recompute.disabled = False

    def _on_display_param_change(self, change):
        """Handle display parameter changes."""
        self._update_dispersion_plot()

    def _refresh_mode_or_animation(self):
        """Refresh mode visualization or restart animation if active."""
        if self._selected_k is None:
            return
        
        self._ensure_animation_state()
        if self._is_animating:
            # Restart animation with new parameters
            self._stop_animation()
            self._on_animate(None)
        else:
            # Update static mode visualization
            self._update_mode_visualization()

    def _on_mode_param_change(self, change):
        """Handle mode visualization parameter changes."""
        self._refresh_mode_or_animation()
    
    def _on_x_periods_change(self, change):
        """Handle x_periods slider change - reset zoom to apply new width."""
        # Reset flag so new width from slider is applied
        self._first_mode_plot = True
        self._refresh_mode_or_animation()
    
    def _on_n_bz_change(self, change):
        """Handle N_BZ slider change."""
        self._update_k_direction_visibility()
        self._refresh_mode_or_animation()

    def _on_anim_param_change(self, change):
        """Handle animation parameter changes (frames, fps)."""
        # Only restart if animation is currently active
        self._ensure_animation_state()
        if self._is_animating and self._selected_k is not None:
            self._stop_animation()
            self._on_animate(None)

    def _on_live_filter_change(self, change):
        """Handle live post-filter parameter changes."""
        self._update_dispersion_plot()
    
    def _on_save_preset(self, _):
        """Save current parameters as a preset."""
        preset_name = self.w_preset_name.value.strip()
        if not preset_name:
            self.w_info.value = "<small style='color:orange'>⚠️ Enter preset name</small>"
            return
        
        if self.save_preset(preset_name):
            presets_dir = self._get_presets_dir()
            self.w_info.value = f"<small style='color:green'>✅ Saved to .mmpp_presets/{preset_name}.json</small>"
            self._refresh_preset_dropdown()
            self.w_preset_name.value = ""
        else:
            self.w_info.value = f"<small style='color:red'>❌ Failed to save '{preset_name}'</small>"
    
    def _on_load_preset(self, change):
        """Load a selected preset."""
        preset_name = change['new']
        if not preset_name:
            return
        
        if self.load_preset(preset_name):
            self.w_info.value = f"<small style='color:green'>✅ Preset '{preset_name}' loaded</small>"
            # Update plots with new parameters
            self._update_dispersion_plot()
            self._refresh_mode_or_animation()
        else:
            self.w_info.value = f"<small style='color:red'>❌ Failed to load preset '{preset_name}'</small>"
    
    def _on_delete_preset(self, _):
        """Delete the selected preset."""
        preset_name = self.w_preset_load.value
        if not preset_name:
            self.w_info.value = "<small style='color:orange'>⚠️ Select preset to delete</small>"
            return
        
        if self.delete_preset(preset_name):
            self.w_info.value = f"<small style='color:green'>✅ Deleted '{preset_name}'</small>"
            self._refresh_preset_dropdown()
            self.w_preset_load.value = ""
        else:
            self.w_info.value = f"<small style='color:red'>❌ Failed to delete '{preset_name}'</small>"
    
    def _on_refresh_presets(self, _):
        """Refresh the preset dropdown list."""
        self._refresh_preset_dropdown()
        presets_dir = self._get_presets_dir()
        count = len(self.list_presets())
        self.w_info.value = f"<small style='color:green'>✅ Found {count} preset(s) in {presets_dir.name}/</small>"
    
    def _refresh_preset_dropdown(self):
        """Update the preset dropdown with current list of presets."""
        available_presets = self.list_presets()
        preset_options = [("-- Load Preset --", "")] + [(name, name) for name in available_presets]
        self.w_preset_load.options = preset_options
    
    def _update_k_direction_visibility(self):
        """Show/hide k-direction dropdown based on N_BZ value.
        
        With current mask definition (k0 ± n*G), direction is meaningful even
        for N_BZ=0, so keep it visible in normal operation.
        """
        self.w_k_direction.layout.display = ''

    def _build_live_filters_config(self) -> dict[str, object] | None:
        """Build live-capable post-filter config from widget values."""
        live_cfg: dict[str, object] = {}

        if self.w_live_snr_enabled.value:
            live_cfg["snr_filter"] = {
                "enabled": True,
                "threshold_snr": float(self.w_live_snr_threshold.value),
                "method": "percentile",
                "noise_percentile": 5.0,
            }

        if self.w_live_gaussian_enabled.value:
            live_cfg["gaussian_morph"] = {
                "enabled": True,
                "sigma_f": float(self.w_live_sigma_f.value),
                "sigma_k": float(self.w_live_sigma_k.value),
                "threshold_std": float(self.w_live_gaussian_threshold_std.value),
                "opening_size": 3,
            }

        if self.w_live_wiener_enabled.value:
            live_cfg["wiener2d"] = {
                "enabled": True,
                "window_size": int(self.w_live_wiener_window.value),
            }

        if self.w_live_bandpass_enabled.value:
            k_min = min(self.w_live_kmin.value, self.w_live_kmax.value) * 1e6
            k_max = max(self.w_live_kmin.value, self.w_live_kmax.value) * 1e6
            f_min = float(self.w_fmin.value) * 1e9
            f_max = float(self.w_fmax.value) * 1e9
            live_cfg["fk_bandpass"] = {
                "enabled": True,
                "k_min": k_min,
                "k_max": k_max,
                "f_min": f_min,
                "f_max": f_max,
            }

        # =====================================================================
        # Enhancement filters (non-destructive, applied in order for best results)
        # Order: percentile → soft_threshold → log_transform → gamma → 
        #        local_contrast → clahe → unsharp_mask
        # =====================================================================

        # Percentile autoscale - clip to robust range first
        if self.w_live_percentile_enabled.value:
            live_cfg["percentile_autoscale"] = {
                "enabled": True,
                "low_percentile": float(self.w_live_percentile_low.value),
                "high_percentile": float(self.w_live_percentile_high.value),
            }

        # Soft threshold - non-destructive noise suppression
        if self.w_live_soft_threshold_enabled.value:
            live_cfg["soft_threshold"] = {
                "enabled": True,
                "threshold_percentile": float(self.w_live_soft_percentile.value),
                "smoothness": float(self.w_live_soft_smoothness.value),
            }

        # Log transform - dynamic range compression
        if self.w_live_log_enabled.value:
            live_cfg["log_transform"] = {
                "enabled": True,
                "method": str(self.w_live_log_method.value),
                "scale": 1.0,
                "floor_percentile": 1.0,
            }

        # Gamma correction - reveal weak signals
        if self.w_live_gamma_enabled.value:
            live_cfg["gamma"] = {
                "enabled": True,
                "gamma": float(self.w_live_gamma_value.value),
            }

        # Local contrast normalization
        if self.w_live_lcn_enabled.value:
            live_cfg["local_contrast"] = {
                "enabled": True,
                "sigma": float(self.w_live_lcn_sigma.value),
                "epsilon": 1e-5,
            }

        # CLAHE - adaptive histogram equalization
        if self.w_live_clahe_enabled.value:
            live_cfg["clahe"] = {
                "enabled": True,
                "clip_limit": float(self.w_live_clahe_clip.value),
                "tile_size": int(self.w_live_clahe_tile.value),
            }

        # Unsharp mask - edge enhancement (apply last for sharpening)
        if self.w_live_unsharp_enabled.value:
            live_cfg["unsharp_mask"] = {
                "enabled": True,
                "sigma": float(self.w_live_unsharp_sigma.value),
                "alpha": float(self.w_live_unsharp_alpha.value),
                "threshold": 0.0,
            }

        if not live_cfg:
            return None
        return {"live": live_cfg}


    def _build_compute_filters_config(self) -> dict[str, object] | None:
        """Build compute-stage filter config for recomputation."""
        filters_cfg: dict[str, object] = {}

        if self.w_pre_remove_static.value:
            filters_cfg["remove_static"] = True
        if self.w_pre_remove_average.value:
            filters_cfg["remove_average"] = True
        if self.w_pre_hann_time.value:
            filters_cfg["hann_time"] = True
        if self.w_pre_hann_space.value:
            filters_cfg["hann_space"] = True

        pre_cfg: dict[str, object] = {}

        if self.w_pre_envelope_enabled.value:
            pre_cfg["envelope_extraction"] = {
                "enabled": True,
                "threshold_std": float(self.w_pre_envelope_threshold_std.value),
                "margin_samples": int(self.w_pre_envelope_margin.value),
            }

        if self.w_pre_wavelet_enabled.value:
            pre_cfg["wavelet_denoise"] = {
                "enabled": True,
                "wavelet": "db4",
                "level": int(self.w_pre_wavelet_level.value),
                "method": "visu",
            }

        if self.w_pre_equalize_enabled.value:
            pre_cfg["amplitude_equalization"] = {
                "enabled": True,
                "smoothing_fraction": 0.05,
                "max_gain": 10.0,
                "target": "mean",
            }

        if self.w_pre_compression_enabled.value:
            pre_cfg["dynamic_compression"] = {
                "enabled": True,
                "method": "log",
                "alpha": float(self.w_pre_compression_alpha.value),
                "preserve_scale": True,
            }

        if self.w_pre_welch_enabled.value:
            pre_cfg["welch_average"] = {
                "enabled": True,
                "n_segments": int(self.w_pre_welch_segments.value),
                "overlap": float(self.w_pre_welch_overlap.value),
                "apply_hann": True,
            }

        if pre_cfg:
            filters_cfg["pre"] = pre_cfg

        live_cfg = self._build_live_filters_config()
        if live_cfg is not None and isinstance(live_cfg.get("live"), dict):
            filters_cfg["live"] = live_cfg["live"]

        return filters_cfg or None

    def _on_auto_detect(self, _):
        """Handle auto-detect button click."""
        if self.result is None:
            return

        methods = ["autocorrelation", "fft", "peak_spacing"]
        results = []

        for method in methods:
            try:
                a = self.detector.detect_lattice_constant(self.result, method=method)
                if 50e-9 < a < 5000e-9:
                    results.append(a)
            except Exception:
                pass

        if results:
            detected_a = np.median(results)
            self.w_lattice.value = detected_a * 1e9
            self.w_info.value = f"<small style='color:green'>Detected: {detected_a*1e9:.0f} nm</small>"
        else:
            self.w_info.value = "<small style='color:orange'>Detection failed</small>"

    def _on_show_system_info(self, _):
        """Display detailed system information from simulation data."""
        try:
            analyzer = self.interface.analyzer
            
            # Grid spacings
            dx = analyzer.grid_spacings.get('dx', 0) * 1e9  # m → nm
            dy = analyzer.grid_spacings.get('dy', 0) * 1e9
            dz = analyzer.grid_spacings.get('dz', 0) * 1e9
            dt = analyzer.dt * 1e12  # s → ps
            
            # Data dimensions
            if analyzer.M_data is not None:
                shape = analyzer.M_data.shape
                nt, nz, ny, nx = shape[0], shape[1], shape[2], shape[3]
            else:
                nt = ny = nx = nz = 0
            
            # Physical domain sizes
            Lx = nx * dx  # nm
            Ly = ny * dy  # nm
            Lz = nz * dz  # nm
            T_total = nt * dt / 1000  # ps → ns
            
            # Frequency/k-space info from result
            if self.result is not None:
                fmax = self.result.f_axis.max() / 1e9  # Hz → GHz
                fmin = self.result.f_axis.min() / 1e9
                df = (self.result.f_axis[1] - self.result.f_axis[0]) / 1e9 if len(self.result.f_axis) > 1 else 0
                
                kmax = self.result.k_axis.max() / 1e6  # 1/m → rad/μm
                kmin = self.result.k_axis.min() / 1e6
                dk = (self.result.k_axis[1] - self.result.k_axis[0]) / 1e6 if len(self.result.k_axis) > 1 else 0
                
                nf = len(self.result.f_axis)
                nk = len(self.result.k_axis)
            else:
                fmax = fmin = df = kmax = kmin = dk = nf = nk = 0
            
            # Format HTML info
            info_html = f"""
            <div style='font-family:monospace; font-size:11px; line-height:1.4; background:#f5f5f5; padding:8px; border-radius:4px; margin:5px 0;'>
            <b style='color:#2563eb'>📊 System Parameters</b><br>
            <hr style='margin:4px 0; border:none; border-top:1px solid #ccc'>
            <b>Spatial Grid:</b><br>
            • dx = {dx:.2f} nm, dy = {dy:.2f} nm, dz = {dz:.2f} nm<br>
            • Nx = {nx}, Ny = {ny}, Nz = {nz}<br>
            • Lx = {Lx:.1f} nm ({Lx/1000:.3f} μm)<br>
            • Ly = {Ly:.1f} nm ({Ly/1000:.3f} μm)<br>
            • Lz = {Lz:.1f} nm ({Lz/1000:.3f} μm)<br>
            <br>
            <b>Time Domain:</b><br>
            • dt = {dt:.3f} ps (t_sampl)<br>
            • Nt = {nt} steps<br>
            • T_total = {T_total:.2f} ns<br>
            <br>
            <b>Frequency Space:</b><br>
            • f_range = [{fmin:.2f}, {fmax:.2f}] GHz<br>
            • df = {df:.3f} GHz<br>
            • Nf = {nf} points<br>
            <br>
            <b>k-Space:</b><br>
            • k_range = [{kmin:.2f}, {kmax:.2f}] rad/μm<br>
            • dk = {dk:.3f} rad/μm<br>
            • Nk = {nk} points<br>
            <br>
            <b>Current Lattice:</b><br>
            • a = {self.w_lattice.value:.1f} nm<br>
            • G = 2π/a = {2*np.pi/(self.w_lattice.value/1000):.3f} rad/μm<br>
            </div>
            """
            
            self.w_info.value = info_html
            
        except Exception as e:
            import traceback
            logger.error(f"System info display failed:\n{traceback.format_exc()}")
            self.w_info.value = f"<small style='color:red'>❌ Error: {str(e)[:100]}</small>"

    def _ensure_animation_state(self):
        """Backfill animation attributes for legacy/stale live instances."""
        if not hasattr(self, "_animation"):
            self._animation = None
        if not hasattr(self, "_is_animating"):
            self._is_animating = False

    def _on_animate(self, _):
        """Toggle animation of selected mode in the mode visualization panel."""
        self._ensure_animation_state()

        if self._selected_k is None or self._selected_f is None:
            self.w_info.value = "<small style='color:red'>⚠️ Select a mode first (click on dispersion)</small>"
            return
        
        # Toggle animation on/off
        if self._is_animating:
            # Stop animation
            self._stop_animation()
            self.w_info.value = "<small>Animation stopped</small>"
            self.w_animate.description = "🎬 Animate Mode"
            self.w_animate.button_style = "warning"
            # Restore static view
            self._update_mode_visualization()
            return
        
        try:
            from matplotlib.animation import FuncAnimation
            
            # Get parameters
            a = self.w_lattice.value * 1e-9
            n_bz = self.w_n_bz_mask.value
            k_direction = self.w_k_direction.value
            mode_type = self.w_mode_type.value
            
            # Extract complex mode data
            x_axis, y_axis, mode_2d_complex = self._extract_mode_2d_custom(
                k_0=self._selected_k,
                f_0=self._selected_f,
                lattice_constant=a,
                n_bz=n_bz,
                k_direction=k_direction,
                k_margin_bins=self.w_k_margin.value,
                f_margin_bins=self.w_f_margin.value,
                neighbor_reduce=self.w_neighbor_reduce.value,
            )
            
            # Time parameters for full 2π cycle
            period_s = 1.0 / self._selected_f  # Full period
            omega = 2 * np.pi * self._selected_f
            n_frames = self.w_anim_frames.value  # From widget
            fps = self.w_anim_fps.value  # From widget
            
            # Time array for one complete cycle (0 to T)
            time_array = np.linspace(0, period_s, n_frames, endpoint=False)
            
            # Pre-compute all frames for selected visualization mode
            mode_labels = {
                'real': 'Re[M]',
                'imag': 'Im[M]',
                'abs': '|M|',
                'phase': 'φ[M]',
                'ampl_phase': 'Ampl×Phase',
            }
            mode_label = mode_labels.get(mode_type, mode_type)
            is_rgb = (mode_type == 'ampl_phase')

            if mode_type in ['real', 'imag']:
                frames = []
                for t in time_array:
                    m_t_complex = mode_2d_complex * np.exp(-1j * omega * t)
                    if mode_type == 'real':
                        m_t = np.real(m_t_complex)
                    else:
                        m_t = np.imag(m_t_complex)
                    frames.append(m_t)

                frames = np.array(frames)  # (n_frames, N_y, N_x)
                vmax = np.max(np.abs(frames))
                if vmax < 1e-20:
                    vmax = 1.0
                vmin = -vmax
                cmap = self.w_cmap_mode.value
                cbar_label = "Re[M(t)]" if mode_type == 'real' else "Im[M(t)]"

            elif mode_type == 'abs':
                # |M| is the spatial envelope and stays constant in time for a pure harmonic mode.
                amplitude = np.abs(mode_2d_complex)
                frames = np.repeat(amplitude[np.newaxis, :, :], n_frames, axis=0)
                vmin = 0.0
                vmax = np.max(amplitude)
                if vmax < 1e-20:
                    vmax = 1.0
                cmap = 'hot'
                cbar_label = "|M|"

            elif mode_type == 'phase':
                frames = []
                for t in time_array:
                    m_t_complex = mode_2d_complex * np.exp(-1j * omega * t)
                    frames.append(np.angle(m_t_complex))

                frames = np.array(frames)
                vmin = -np.pi
                vmax = np.pi
                cmap = 'hsv'
                cbar_label = "φ[M(t)] [rad]"

            elif mode_type == 'ampl_phase':
                from ..utils import create_amplitude_phase_colormap

                amplitude_ref = np.abs(mode_2d_complex)
                amp_min = float(amplitude_ref.min())
                amp_max = float(amplitude_ref.max())

                frames = []
                for t in time_array:
                    m_t_complex = mode_2d_complex * np.exp(-1j * omega * t)
                    frames.append(
                        create_amplitude_phase_colormap(
                            m_t_complex,
                            amp_min=amp_min,
                            amp_max=amp_max,
                        )
                    )

                frames = np.array(frames)  # (n_frames, N_y, N_x, 3)
                vmin = None
                vmax = None
                cmap = None
                cbar_label = None

            else:
                raise ValueError(f"Unknown mode_type='{mode_type}'")
            
            # Setup axes
            ax = self._ax_mode
            
            # Save current zoom/pan state BEFORE clearing
            xlim_saved = ax.get_xlim()
            ylim_saved = ax.get_ylim()
            
            ax.clear()
            
            # Remove old mode colorbar
            if self._colorbar_mode is not None:
                try:
                    self._colorbar_mode.remove()
                except Exception:
                    pass
                self._colorbar_mode = None
            
            # Extent
            x_um = x_axis * 1e6
            y_um = y_axis * 1e6
            extent = [x_um[0], x_um[-1], y_um[0], y_um[-1]]
            
            # Initial frame
            if is_rgb:
                im = ax.imshow(
                    frames[0],
                    aspect="auto",
                    origin="lower",
                    extent=extent,
                    interpolation="bilinear",
                )
            else:
                im = ax.imshow(
                    frames[0],
                    aspect="auto",
                    origin="lower",
                    extent=extent,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="bilinear",
                )
            
            # Colorbar only for scalar data
            if not is_rgb:
                self._colorbar_mode = self._fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
                self._colorbar_mode.set_label(cbar_label, fontsize=9)
            
            # Labels
            ax.set_xlabel("x [μm]", fontsize=10)
            ax.set_ylabel("y [μm]", fontsize=10)
            
            # Title (will be updated each frame)
            k_str = f"k = {self._selected_k/1e6:.2f} rad/μm"
            f_str = f"f = {self._selected_f/1e9:.2f} GHz"
            title = ax.set_title(
                f"{mode_label} Mode | {k_str}, {f_str} | t=0.00 ns | φ=0.00°",
                fontsize=11
            )
            ax.tick_params(labelsize=9)
            
            # Restore zoom/pan state (preserve user's zoom during animation)
            # Check if this is first plot (matplotlib default xlim is 0,1)
            is_first_plot = (abs(xlim_saved[0] - 0.0) < 0.01 and abs(xlim_saved[1] - 1.0) < 0.01)
            
            if not is_first_plot and not self._first_mode_plot:
                # Preserve user's zoom/pan
                ax.set_xlim(xlim_saved)
                ax.set_ylim(ylim_saved)
            else:
                # First animation - use configured number of periods from slider
                x_periods = self.w_mode_x_periods.value
                x_center = (x_um[0] + x_um[-1]) / 2
                half_width = (x_periods / 2.0) * (a * 1e6)  # periods to μm
                ax.set_xlim(x_center - half_width, x_center + half_width)
                ax.set_ylim(y_um[0], y_um[-1])
            
            # Overlay geometry contour if provided (draw once, persists through animation)
            if self._geometry_contour is not None:
                try:
                    geom = self._geometry_contour
                    # Create coordinate arrays for the geometry
                    geom_y = np.linspace(y_um[0], y_um[-1], geom.shape[0])
                    geom_x = np.linspace(x_um[0], x_um[-1], geom.shape[1])
                    
                    # Draw contour at level 0.5 (boundary between 0 and 1)
                    ax.contour(
                        geom_x, geom_y, geom,
                        levels=[0.5],
                        colors=['white'],
                        linewidths=[1.5],
                        linestyles=['solid'],
                    )
                    # Add black outline for visibility on light backgrounds
                    ax.contour(
                        geom_x, geom_y, geom,
                        levels=[0.5],
                        colors=['black'],
                        linewidths=[0.5],
                        linestyles=['solid'],
                    )
                except Exception as contour_err:
                    logger.warning(f"Failed to draw geometry contour in animation: {contour_err}")
            
            # Animation update function
            def update(frame_idx):
                im.set_data(frames[frame_idx])
                t_ns = time_array[frame_idx] * 1e9
                phase_deg = (time_array[frame_idx] / period_s) * 360
                title.set_text(
                    f"{mode_label} Mode | {k_str}, {f_str} | t={t_ns:.2f} ns | φ={phase_deg:.0f}°"
                )
                return [im, title]
            
            # Create animation
            interval = 1000 / fps  # ms between frames
            self._animation = FuncAnimation(
                self._fig,
                update,
                frames=n_frames,
                interval=interval,
                blit=True,
                repeat=True,
            )
            
            self._is_animating = True
            self.w_animate.description = "⏸️ Stop Animation"
            self.w_animate.button_style = "danger"
            abs_note = " | |M| is time-invariant for harmonic modes" if mode_type == 'abs' else ""
            self.w_info.value = (
                f"<small style='color:green'>🎬 Animating: {n_frames} frames, "
                f"T={period_s*1e9:.2f} ns (1 period = 2π), mode={mode_label}{abs_note}</small>"
            )
            
            # Redraw
            self._fig.canvas.draw_idle()
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Animation failed:\n{tb}")
            self.w_info.value = f"<small style='color:red'>❌ Animation error: {str(e)[:50]}</small>"
            self._is_animating = False
            self.w_animate.description = "🎬 Animate Mode"
            self.w_animate.button_style = "warning"
    
    def _on_save_animation(self, _):
        """Save the current animation to file.

        Goal: saved file should match what the user currently sees (layout, zoom,
        filters, colormaps, markers).
        """
        if self._selected_k is None or self._selected_f is None:
            self.w_info.value = "<small style='color:red'>⚠️ Select a mode first (click on dispersion)</small>"
            return
        
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
            try:
                from matplotlib.animation import ImageMagickWriter
                _has_imagemagick = True
            except ImportError:
                _has_imagemagick = False
            from pathlib import Path
            from datetime import datetime
            
            # Get parameters
            a = self.w_lattice.value * 1e-9
            n_bz = self.w_n_bz_mask.value
            k_direction = self.w_k_direction.value
            mode_type = self.w_mode_type.value
            n_frames = self.w_anim_frames.value
            fps = self.w_anim_fps.value
            save_mode = self.w_anim_save_mode.value
            file_format = self.w_anim_file_format.value
            
            self.w_info.value = "<small style='color:blue'>⏳ Preparing animation for save...</small>"

            # Capture current view state (zoom/pan) so the saved animation matches the UI.
            disp_xlim = None
            disp_ylim = None
            mode_xlim = None
            mode_ylim = None
            if self._ax_disp is not None:
                disp_xlim = self._ax_disp.get_xlim()
                disp_ylim = self._ax_disp.get_ylim()
            if self._ax_mode is not None:
                mode_xlim = self._ax_mode.get_xlim()
                mode_ylim = self._ax_mode.get_ylim()
            
            # Extract complex mode data
            x_axis, y_axis, mode_2d_complex = self._extract_mode_2d_custom(
                k_0=self._selected_k,
                f_0=self._selected_f,
                lattice_constant=a,
                n_bz=n_bz,
                k_direction=k_direction,
                k_margin_bins=self.w_k_margin.value,
                f_margin_bins=self.w_f_margin.value,
                neighbor_reduce=self.w_neighbor_reduce.value,
            )
            
            # Time parameters
            period_s = 1.0 / self._selected_f
            omega = 2 * np.pi * self._selected_f
            time_array = np.linspace(0, period_s, n_frames, endpoint=False)
            
            # Pre-compute frames (same logic as _on_animate)
            mode_labels = {
                'real': 'Re[M]',
                'imag': 'Im[M]',
                'abs': '|M|',
                'phase': 'φ[M]',
                'ampl_phase': 'Ampl×Phase',
            }
            mode_label = mode_labels.get(mode_type, mode_type)
            is_rgb = (mode_type == 'ampl_phase')

            if mode_type in ['real', 'imag']:
                frames = []
                for t in time_array:
                    m_t_complex = mode_2d_complex * np.exp(-1j * omega * t)
                    frames.append(np.real(m_t_complex) if mode_type == 'real' else np.imag(m_t_complex))
                frames = np.array(frames)
                vmax = np.max(np.abs(frames))
                if vmax < 1e-20:
                    vmax = 1.0
                vmin = -vmax
                cmap = self.w_cmap_mode.value
                cbar_label = "Re[M(t)]" if mode_type == 'real' else "Im[M(t)]"

            elif mode_type == 'abs':
                amplitude = np.abs(mode_2d_complex)
                frames = np.repeat(amplitude[np.newaxis, :, :], n_frames, axis=0)
                vmin, vmax = 0.0, np.max(amplitude)
                if vmax < 1e-20:
                    vmax = 1.0
                cmap = 'hot'
                cbar_label = "|M|"

            elif mode_type == 'phase':
                frames = []
                for t in time_array:
                    m_t_complex = mode_2d_complex * np.exp(-1j * omega * t)
                    frames.append(np.angle(m_t_complex))
                frames = np.array(frames)
                vmin, vmax = -np.pi, np.pi
                cmap = 'hsv'
                cbar_label = "φ[M(t)] [rad]"

            elif mode_type == 'ampl_phase':
                from ..utils import create_amplitude_phase_colormap
                amplitude_ref = np.abs(mode_2d_complex)
                amp_min, amp_max = float(amplitude_ref.min()), float(amplitude_ref.max())
                frames = []
                for t in time_array:
                    m_t_complex = mode_2d_complex * np.exp(-1j * omega * t)
                    frames.append(create_amplitude_phase_colormap(m_t_complex, amp_min=amp_min, amp_max=amp_max))
                frames = np.array(frames)
                vmin, vmax = None, None
                cmap = None
                cbar_label = None

            # Create temporary figure for saving (Full HD quality: 1920px width minimum)
            # Quality settings: figsize * dpi = pixels (e.g., 16 inches * 120 dpi = 1920 px)
            save_dpi = 120  # High quality export
            
            if save_mode == "mode":
                # Save only mode panel: 1920x1200 px (16" x 10" @ 120 dpi)
                fig_save, ax_save = plt.subplots(
                    figsize=(16, 10),
                    dpi=save_dpi,
                    constrained_layout=True,
                )
                ax_disp_save = None
            else:
                # Save full view: 1920x1680 px (16" x 14" @ 120 dpi)
                fig_save, (ax_disp_save, ax_save) = plt.subplots(
                    2,
                    1,
                    figsize=(16, 14),
                    dpi=save_dpi,
                    gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.25},
                    constrained_layout=True,
                )
                
                # Plot dispersion exactly like the interactive view (filters, limits, markers).
                if self.result is None:
                    raise ValueError("No dispersion result available")

                S_map = self.result.S  # (Nk, Nf)
                live_filters = self._build_live_filters_config()
                if live_filters is not None:
                    try:
                        from ..utils import apply_dispersion_post_filters

                        S_map = apply_dispersion_post_filters(
                            S_map,
                            k_axis=self.result.k_axis,
                            f_axis=self.result.f_axis,
                            filters=live_filters,
                            include_live=True,
                        )
                    except Exception:
                        logger.exception("Live post-filter application failed for save; using raw S")
                        S_map = self.result.S

                S = S_map.T  # (Nf, Nk)
                k_axis_plot = self.result.k_axis / 1e6  # rad/μm
                f_axis_plot = self.result.f_axis / 1e9  # GHz

                # Get frequency limits (for viewport only, not data cutting)
                f_min = float(self.w_fmin.value)
                f_max = float(self.w_fmax.value)

                # Check if there's valid data
                if len(f_axis_plot) < 2 or len(k_axis_plot) < 2:
                    raise ValueError("Insufficient data for animation save")

                # Cut data to show only positive frequencies (upper half)
                positive_freq_mask = f_axis_plot >= 0
                if np.sum(positive_freq_mask) > 0:
                    S = S[positive_freq_mask, :]
                    f_axis_positive = f_axis_plot[positive_freq_mask]
                else:
                    f_axis_positive = f_axis_plot

                # Plot data with extent from 0 to fmax
                extent_disp = [k_axis_plot[0], k_axis_plot[-1], 0, f_axis_positive[-1]]
                im_disp = ax_disp_save.imshow(
                    np.log10(S + 1e-20),
                    aspect="auto",
                    origin="lower",
                    extent=extent_disp,
                    cmap=self.w_cmap_disp.value,
                    interpolation="bilinear",
                )
                fig_save.colorbar(im_disp, ax=ax_disp_save, shrink=0.8, pad=0.02).set_label("log₁₀(S)", fontsize=9)

                G = 2 * np.pi / a / 1e6  # rad/μm
                # Restore current zoom/pan if available (match UI)
                is_default_xlim = disp_xlim is None or (abs(disp_xlim[0] - 0.0) < 0.01 and abs(disp_xlim[1] - 1.0) < 0.01)
                is_default_ylim = disp_ylim is None or (abs(disp_ylim[0] - 0.0) < 0.01 and abs(disp_ylim[1] - 1.0) < 0.01)
                if is_default_xlim:
                    ax_disp_save.set_xlim(-1.5 * G, 1.5 * G)
                else:
                    ax_disp_save.set_xlim(disp_xlim)
                if is_default_ylim:
                    # Show all available frequencies on first plot
                    ax_disp_save.set_ylim(0, f_axis_positive[-1])
                else:
                    # Preserve user zoom but constrain to frequency limits and available data
                    view_f_min = max(disp_ylim[0], f_min, 0)
                    view_f_max = min(disp_ylim[1], f_max, f_axis_positive[-1])
                    ax_disp_save.set_ylim(view_f_min, view_f_max)

                # Add BZ boundary lines if enabled (using current xlim)
                if self.w_show_bz_lines.value:
                    k_range = ax_disp_save.get_xlim()
                    k_max = max(abs(k_range[0]), abs(k_range[1]))
                    if G > 0:
                        n_zones = int(np.ceil(k_max / G)) + 1
                    else:
                        n_zones = 1

                    for n in range(-n_zones, n_zones + 1):
                        if n == 0:
                            continue
                        k_line = n * G
                        if abs(k_line) <= k_max * 1.1:
                            alpha = 0.8 if abs(n) == 1 else 0.4
                            ax_disp_save.axvline(
                                k_line,
                                color="red",
                                linestyle="--",
                                linewidth=1.5 if abs(n) == 1 else 1.0,
                                alpha=alpha,
                            )
                    
                    ax_disp_save.legend([f"BZ boundaries (G = {G:.1f} rad/μm)"], loc="upper right", fontsize=8)
                
                # Always add k=0 reference line
                ax_disp_save.axvline(0, color="gray", linestyle=":", alpha=0.5, linewidth=1)
                ax_disp_save.set_xlabel(r"$k$ [rad/μm]", fontsize=10)
                ax_disp_save.set_ylabel("f [GHz]", fontsize=10)
                ax_disp_save.set_title(
                    f"Dispersion S(k, f) | a = {a*1e9:.0f} nm | Click to select mode",
                    fontsize=11,
                )
                ax_disp_save.grid(True, alpha=0.3, linestyle=":")
                ax_disp_save.tick_params(labelsize=9)

                # Mark selection and mask replicas (but don't touch UI info text)
                self._draw_selection_markers(ax_disp_save, update_info=False)
            
            # Setup mode animation axes
            x_um = x_axis * 1e6
            y_um = y_axis * 1e6
            extent = [x_um[0], x_um[-1], y_um[0], y_um[-1]]
            
            if is_rgb:
                im = ax_save.imshow(frames[0], aspect="auto", origin="lower", extent=extent, interpolation="bilinear")
            else:
                im = ax_save.imshow(frames[0], aspect="auto", origin="lower", extent=extent,
                                   cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")
                fig_save.colorbar(im, ax=ax_save, shrink=0.8, pad=0.02).set_label(cbar_label, fontsize=9)
            
            # Add geometry contour if available
            if self._geometry_contour is not None:
                try:
                    geom = self._geometry_contour
                    geom_y = np.linspace(y_um[0], y_um[-1], geom.shape[0])
                    geom_x = np.linspace(x_um[0], x_um[-1], geom.shape[1])
                    ax_save.contour(geom_x, geom_y, geom, levels=[0.5], colors=['white'], linewidths=[1.5])
                    ax_save.contour(geom_x, geom_y, geom, levels=[0.5], colors=['black'], linewidths=[0.5])
                except Exception:
                    pass
            
            ax_save.set_xlabel("x [μm]", fontsize=10)
            ax_save.set_ylabel("y [μm]", fontsize=10)
            k_str = f"k = {self._selected_k/1e6:.2f} rad/μm"
            f_str = f"f = {self._selected_f/1e9:.2f} GHz"
            title = ax_save.set_title(
                f"{mode_label} Mode | {k_str}, {f_str} | t=0.00 ns | φ=0.00°",
                fontsize=11,
            )
            ax_save.tick_params(labelsize=9)

            # Match current mode zoom/pan when available.
            is_default_xlim = mode_xlim is None or (abs(mode_xlim[0] - 0.0) < 0.01 and abs(mode_xlim[1] - 1.0) < 0.01)
            is_default_ylim = mode_ylim is None or (abs(mode_ylim[0] - 0.0) < 0.01 and abs(mode_ylim[1] - 1.0) < 0.01)

            if is_default_xlim:
                # Same default view logic as the UI.
                x_periods = float(self.w_mode_x_periods.value)
                x_center = (x_um[0] + x_um[-1]) / 2
                half_width = (x_periods / 2.0) * (a * 1e6)
                ax_save.set_xlim(x_center - half_width, x_center + half_width)
            else:
                ax_save.set_xlim(mode_xlim)

            if is_default_ylim:
                ax_save.set_ylim(y_um[0], y_um[-1])
            else:
                ax_save.set_ylim(mode_ylim)
            
            # Animation update function
            def update(frame_idx):
                im.set_data(frames[frame_idx])
                t_ns = time_array[frame_idx] * 1e9
                phase_deg = (time_array[frame_idx] / period_s) * 360
                title.set_text(
                    f"{mode_label} Mode | {k_str}, {f_str} | t={t_ns:.2f} ns | φ={phase_deg:.0f}°"
                )
                return [im, title]
            
            # Create animation
            anim = FuncAnimation(
                fig_save,
                update,
                frames=n_frames,
                interval=1000 / fps,
                blit=False,  # more reliable when saving complex multi-axes figures
                repeat=False,
            )
            
            # Generate filename
            k_val = f"{self._selected_k/1e6:.2f}".replace('.', 'p')
            f_val = f"{self._selected_f/1e9:.2f}".replace('.', 'p')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mode_anim_{save_mode}_k{k_val}_f{f_val}_{mode_type}_{timestamp}.{file_format}"
            
            # Save animation with appropriate writer
            output_path = Path.cwd() / filename
            self.w_info.value = f"<small style='color:blue'>💾 Saving animation to {filename}...</small>"
            
            # Use higher DPI for GIF to compensate for color limitation
            export_dpi = 150 if file_format == "gif" else save_dpi
            
            if file_format == "gif":
                # Try ImageMagick first (much better quality and dithering)
                if _has_imagemagick:
                    try:
                        # ImageMagick with optimized settings for quality
                        writer = ImageMagickWriter(
                            fps=fps,
                            metadata={'Author': 'MMPP', 'Title': 'Mode Animation'},
                            bitrate=2000,  # Higher bitrate for better quality
                            extra_args=['-layers', 'Optimize']  # Optimize file size while keeping quality
                        )
                        writer_name = "ImageMagick"
                    except Exception:
                        # Fallback to Pillow if ImageMagick fails
                        writer = PillowWriter(fps=fps, metadata={'Author': 'MMPP', 'Title': 'Mode Animation'})
                        writer_name = "Pillow (256 colors)"
                else:
                    # PillowWriter fallback - limited to 256 colors
                    writer = PillowWriter(fps=fps, metadata={'Author': 'MMPP', 'Title': 'Mode Animation'})
                    writer_name = "Pillow (256 colors)"
            else:  # mp4
                # High quality MP4: increased bitrate for Full HD
                writer = FFMpegWriter(fps=fps, bitrate=4000, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-preset', 'slower', '-crf', '18'])
                writer_name = "FFmpeg"
            
            anim.save(str(output_path), writer=writer, dpi=export_dpi)
            
            plt.close(fig_save)
            
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            
            # Add quality hint for GIF format
            quality_hint = ""
            if file_format == "gif" and writer_name == "Pillow (256 colors)":
                quality_hint = "<br>💡 <i>Tip: Use MP4 format for better color quality</i>"
            
            self.w_info.value = (
                f"<small style='color:green'>✅ Animation saved: {filename} ({file_size_mb:.1f} MB)<br>"
                f"{n_frames} frames @ {fps} fps, {export_dpi} DPI, mode={mode_label}, writer={writer_name}{quality_hint}</small>"
            )
            
        except Exception as e:
            import traceback
            logger.error(f"Animation save failed:\n{traceback.format_exc()}")
            self.w_info.value = f"<small style='color:red'>❌ Save error: {str(e)[:80]}</small>"

    def _stop_animation(self):
        """Stop the current animation."""
        self._ensure_animation_state()

        if self._animation is not None:
            self._animation.event_source.stop()
            self._animation = None
        self._is_animating = False

    def _update_dispersion_plot(self):
        """Update the dispersion heatmap."""
        if self.result is None or self._ax_disp is None:
            return

        ax = self._ax_disp

        # Save current zoom/pan state BEFORE clearing
        xlim_saved = ax.get_xlim()
        ylim_saved = ax.get_ylim()

        # Clear axes
        ax.clear()

        # Remove old colorbar
        if self._colorbar_disp is not None:
            try:
                self._colorbar_disp.remove()
            except Exception:
                pass
            self._colorbar_disp = None

        # Get data (optionally with live post-filters)
        S_map = self.result.S  # (Nk, Nf)
        live_filters = self._build_live_filters_config()
        if live_filters is not None:
            try:
                from ..utils import apply_dispersion_post_filters

                S_map = apply_dispersion_post_filters(
                    S_map,
                    k_axis=self.result.k_axis,
                    f_axis=self.result.f_axis,
                    filters=live_filters,
                    include_live=True,
                )
            except Exception:
                logger.exception("Live post-filter application failed")
                self.w_info.value = "<small style='color:red'>⚠️ Live filter error (showing raw S)</small>"
                S_map = self.result.S

        S = S_map.T  # (Nf, Nk)
        k_axis = self.result.k_axis / 1e6  # rad/μm
        f_axis = self.result.f_axis / 1e9  # GHz

        # Get frequency limits (for viewport only, not data cutting)
        f_min = self.w_fmin.value
        f_max = self.w_fmax.value

        # Check if there's any valid data at all
        if len(f_axis) < 2:
            ax.text(0.5, 0.5, "No data available", transform=ax.transAxes, ha="center")
            self._fig.canvas.draw_idle()
            return

        # Cut data to show only positive frequencies (upper half)
        # If f_axis contains negative frequencies, take only f >= 0
        positive_freq_mask = f_axis >= 0
        if np.sum(positive_freq_mask) > 0:
            S = S[positive_freq_mask, :]
            f_axis_positive = f_axis[positive_freq_mask]
        else:
            # All frequencies are already positive
            f_axis_positive = f_axis

        # Plot data with extent from 0 to fmax
        extent = [k_axis[0], k_axis[-1], 0, f_axis_positive[-1]]

        # Plot heatmap
        im = ax.imshow(
            np.log10(S + 1e-20),
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap=self.w_cmap_disp.value,
            interpolation="bilinear",
        )

        self._colorbar_disp = self._fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        self._colorbar_disp.set_label("log₁₀(S)", fontsize=9)

        # Calculate reciprocal lattice vector (for BZ boundaries and default zoom)
        a = self.w_lattice.value * 1e-9
        G = 2 * np.pi / a / 1e6  # rad/μm (reciprocal lattice vector)
        
        # Restore zoom/pan state (or set default ±1.5G on first plot)
        k_limit = 1.5 * G  # ±1.5 zones
        
        # Check if this is first plot (matplotlib default xlim is 0,1)
        # OR if user explicitly reset zoom via reset button
        is_first_plot = (abs(xlim_saved[0] - 0.0) < 0.01 and abs(xlim_saved[1] - 1.0) < 0.01)
        
        if is_first_plot or self._first_dispersion_plot:
            # First plot or explicit reset - use default ±1.5G for k, full range for f
            ax.set_xlim(-k_limit, k_limit)
            ax.set_ylim(0, f_axis_positive[-1])  # Show all available positive frequencies
            self._first_dispersion_plot = False
        else:
            # Preserve user's zoom/pan for k
            ax.set_xlim(xlim_saved)
            # For frequency: constrain to [f_min, f_max] but stay within available data
            view_f_min = max(ylim_saved[0], f_min, 0)
            view_f_max = min(ylim_saved[1], f_max, f_axis_positive[-1])
            ax.set_ylim(view_f_min, view_f_max)
        
        # Add BZ boundary lines (reciprocal lattice vectors G = 2π/a) if enabled
        if self.w_show_bz_lines.value:
            # Show multiple BZ boundaries within k-range
            k_range = ax.get_xlim()
            k_max = max(abs(k_range[0]), abs(k_range[1]))
            n_zones = int(np.ceil(k_max / G)) + 1
            
            for n in range(-n_zones, n_zones + 1):
                if n == 0:
                    continue
                k_line = n * G
                if abs(k_line) <= k_max * 1.1:  # Show if within range
                    alpha = 0.8 if abs(n) == 1 else 0.4  # Emphasize ±1G
                    ax.axvline(k_line, color="red", linestyle="--",
                              linewidth=1.5 if abs(n) == 1 else 1.0, alpha=alpha)
            
            # Add legend
            ax.legend([f"BZ boundaries (G = {G:.1f} rad/μm)"], loc="upper right", fontsize=8)
        
        # Always add k=0 reference line
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5, linewidth=1)

        # Labels
        ax.set_xlabel(r"$k$ [rad/μm]", fontsize=10)
        ax.set_ylabel("f [GHz]", fontsize=10)
        ax.set_title(f"Dispersion S(k, f) | a = {a*1e9:.0f} nm | Click to select mode", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.tick_params(labelsize=9)

        # Redraw selection marker if exists
        if self._selected_k is not None and self._selected_f is not None:
            self._draw_selection_markers(ax)

        self._fig.canvas.draw_idle()

    def _draw_selection_markers(self, ax: Axes, *, update_info: bool = True):
        """Draw markers showing selected (k, f) and all mask positions."""
        if self._selected_k is None:
            return

        k_sel = self._selected_k / 1e6  # rad/μm
        f_sel = self._selected_f / 1e9  # GHz

        a = self.w_lattice.value * 1e-9
        G = 2 * np.pi / a / 1e6  # rad/μm
        n_bz = self.w_n_bz_mask.value
        k_direction = self.w_k_direction.value

        # Draw main selection marker (red square)
        ax.plot(
            k_sel,
            f_sel,
            "rs",
            markersize=12,
            markerfacecolor="none",
            markeredgewidth=2,
            label="Selected",
        )

        # Draw all mask positions as circles
        k_axis = self.result.k_axis / 1e6
        k_min, k_max = k_axis.min(), k_axis.max()

        for n in range(-n_bz, n_bz + 1):
            if n == 0:
                continue  # Skip main selection

            k_copy = k_sel + n * G

            # Check k-direction filter
            if k_direction == "positive" and k_copy < 0:
                continue
            if k_direction == "negative" and k_copy > 0:
                continue

            # Check if within k-axis range
            if k_min <= k_copy <= k_max:
                ax.plot(
                    k_copy,
                    f_sel,
                    "o",
                    markersize=8,
                    markerfacecolor="none",
                    markeredgecolor="lime",
                    markeredgewidth=1.5,
                )

                # Draw connecting line
                ax.plot(
                    [k_sel, k_copy],
                    [f_sel, f_sel],
                    "g--",
                    linewidth=0.8,
                    alpha=0.5,
                )

        # Update info with mask positions
        mask_positions = []
        for n in range(-n_bz, n_bz + 1):
            k_copy = self._selected_k + n * (2 * np.pi / a)
            if k_direction == "positive" and k_copy < 0:
                continue
            if k_direction == "negative" and k_copy > 0:
                continue
            if self.result.k_axis.min() <= k_copy <= self.result.k_axis.max():
                mask_positions.append(k_copy / 1e6)

        if update_info:
            self.w_info.value = (
                f"<small>Mask includes <b>{len(mask_positions)}</b> k-positions<br>"
                f"k = {', '.join([f'{k:.1f}' for k in mask_positions[:5]])}"
                f"{'...' if len(mask_positions) > 5 else ''} rad/μm<br>"
                f"neighbors: Δk=±{self.w_k_margin.value} bin, Δf=±{self.w_f_margin.value} bin, "
                f"agg={self.w_neighbor_reduce.value}</small>"
            )

    def _update_mode_visualization(self):
        """Update the 2D spatial mode visualization m(x, y)."""
        if self.result is None or self._ax_mode is None:
            return
        if self._selected_k is None or self._selected_f is None:
            return

        ax = self._ax_mode

        # Save current zoom/pan state BEFORE clearing
        xlim_saved = ax.get_xlim()
        ylim_saved = ax.get_ylim()

        ax.clear()

        # Remove old colorbar
        if self._colorbar_mode is not None:
            try:
                self._colorbar_mode.remove()
            except Exception:
                pass
            self._colorbar_mode = None

        try:
            # Get parameters
            a = self.w_lattice.value * 1e-9
            n_bz = self.w_n_bz_mask.value
            k_direction = self.w_k_direction.value
            mode_type = self.w_mode_type.value  # 'real', 'imag', 'abs', 'phase'

            # Extract spatial mode profile m(x, y) - returns COMPLEX data
            x_axis, y_axis, mode_2d_complex = self._extract_mode_2d_custom(
                k_0=self._selected_k,
                f_0=self._selected_f,
                lattice_constant=a,
                n_bz=n_bz,
                k_direction=k_direction,
                k_margin_bins=self.w_k_margin.value,
                f_margin_bins=self.w_f_margin.value,
                neighbor_reduce=self.w_neighbor_reduce.value,
            )
            
            # Extract requested component
            if mode_type == 'real':
                mode_2d = np.real(mode_2d_complex)
                cmap = self.w_cmap_mode.value
                cbar_label = "Re[M]"
                use_rgb = False
            elif mode_type == 'imag':
                mode_2d = np.imag(mode_2d_complex)
                cmap = self.w_cmap_mode.value
                cbar_label = "Im[M]"
                use_rgb = False
            elif mode_type == 'abs':
                mode_2d = np.abs(mode_2d_complex)
                cmap = 'hot'  # Better for amplitude
                cbar_label = "|M|"
                use_rgb = False
            elif mode_type == 'phase':
                mode_2d = np.angle(mode_2d_complex)
                cmap = 'hsv'  # Cyclic colormap for phase
                cbar_label = "φ[M] [rad]"
                use_rgb = False
            elif mode_type == 'ampl_phase':
                # Use RGB colormap: hue=phase, brightness=amplitude
                from ..utils import create_amplitude_phase_colormap
                mode_2d = create_amplitude_phase_colormap(mode_2d_complex)
                cmap = None  # RGB data doesn't use colormap
                cbar_label = "Ampl×Phase"
                use_rgb = True

            # Convert axes to convenient units
            x_um = x_axis * 1e6  # μm
            y_um = y_axis * 1e6  # μm

            # Plot 2D spatial heatmap
            extent = [x_um[0], x_um[-1], y_um[0], y_um[-1]]

            # Auto color limits based on mode type
            if mode_type in ['real', 'imag']:
                # Symmetric for real/imag
                vmax = np.max(np.abs(mode_2d))
                if vmax < 1e-20:
                    vmax = 1.0
                vmin = -vmax
            elif mode_type == 'abs':
                # Always positive
                vmin = 0
                vmax = np.max(mode_2d)
                if vmax < 1e-20:
                    vmax = 1.0
            elif mode_type == 'phase':
                # Phase range
                vmin = -np.pi
                vmax = np.pi
            elif mode_type == 'ampl_phase':
                # RGB image - no vmin/vmax needed
                vmin = None
                vmax = None

            if use_rgb:
                # RGB image - no colormap
                im = ax.imshow(
                    mode_2d,
                    aspect="auto",
                    origin="lower",
                    extent=extent,
                    interpolation="bilinear",
                )
            else:
                # Scalar data with colormap
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

            # Only add colorbar for scalar data
            if not use_rgb:
                self._colorbar_mode = self._fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
                self._colorbar_mode.set_label(cbar_label, fontsize=9)

            # Overlay geometry contour if provided
            if self._geometry_contour is not None:
                try:
                    geom = self._geometry_contour
                    # Create coordinate arrays for the geometry
                    # Assume geometry spans the same spatial domain as mode data
                    geom_y = np.linspace(y_um[0], y_um[-1], geom.shape[0])
                    geom_x = np.linspace(x_um[0], x_um[-1], geom.shape[1])
                    
                    # Draw contour at level 0.5 (boundary between 0 and 1)
                    # Use a distinct color that stands out on any colormap
                    ax.contour(
                        geom_x, geom_y, geom,
                        levels=[0.5],
                        colors=['white'],
                        linewidths=[1.5],
                        linestyles=['solid'],
                    )
                    # Add black outline for visibility on light backgrounds
                    ax.contour(
                        geom_x, geom_y, geom,
                        levels=[0.5],
                        colors=['black'],
                        linewidths=[0.5],
                        linestyles=['solid'],
                    )
                except Exception as contour_err:
                    logger.warning(f"Failed to draw geometry contour: {contour_err}")

            # Labels
            ax.set_xlabel("x [μm]", fontsize=10)
            ax.set_ylabel("y [μm]", fontsize=10)


            # Title with mode type
            mode_type_labels = {
                'real': 'Re[M]',
                'imag': 'Im[M]',
                'abs': '|M|',
                'phase': 'φ[M]',
                'ampl_phase': 'Ampl×Phase',
            }
            mode_label = mode_type_labels.get(mode_type, mode_type)
            k_str = f"k = {self._selected_k/1e6:.2f} rad/μm"
            f_str = f"f = {self._selected_f/1e9:.2f} GHz"
            ax.set_title(f"{mode_label} Mode | {k_str}, {f_str}", fontsize=11)
            ax.tick_params(labelsize=9)

            # Restore zoom/pan state (or use auto limits on first plot)
            # Check if this is first plot (matplotlib default xlim is 0,1)
            is_first_plot = (abs(xlim_saved[0] - 0.0) < 0.01 and abs(xlim_saved[1] - 1.0) < 0.01)
            
            if not is_first_plot and not self._first_mode_plot:
                # Preserve user's zoom/pan
                ax.set_xlim(xlim_saved)
                ax.set_ylim(ylim_saved)
            else:
                # First plot - use configured number of periods from slider
                x_periods = self.w_mode_x_periods.value
                x_center = (x_um[0] + x_um[-1]) / 2
                half_width = (x_periods / 2.0) * (a * 1e6)  # periods to μm
                ax.set_xlim(x_center - half_width, x_center + half_width)
                # Keep full y range
                ax.set_ylim(y_um[0], y_um[-1])
                self._first_mode_plot = False

            # Redraw dispersion with markers
            self._update_dispersion_plot()

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error extracting mode:\n{e}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="red",
            )
            logger.exception("Mode extraction failed")

        self._fig.canvas.draw_idle()

    def _extract_mode_2d_custom(
        self,
        k_0: float,
        f_0: float,
        lattice_constant: float,
        n_bz: int,
        k_direction: str,
        k_margin_bins: int = 0,
        f_margin_bins: int = 0,
        neighbor_reduce: str = "mean",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract 2D spatial mode profile m(x, y) using pre-computed S_complex.
        
        Algorithm (following Rychły et al.):
        1. Use S_complex from dispersion result (already FFT'd!)
        2. Select frequency neighborhood around f_0 and create mask for k_0 ± n·G
           (optionally with extra +/- k-bin margin around each replica)
        3. Aggregate selected frequency bins (mean/sum), then IFFT only over k
           → propagation axis (phase preserved!)
        4. Result: M(x, y) spatial profile of the mode
        
        This is FAST - no re-computation of FFT! Uses cached S_complex.
        
        Returns x_axis, y_axis, mode_2d(x, y).
        """
        from .extraction import extract_mode_2d

        if self.result is None:
            raise ValueError(
                "No dispersion result available. "
                "Run dispersion_modes(..., save=True) or compute dispersion first."
            )

        x_axis, y_axis, mode_2d, info = extract_mode_2d(
            self.result,
            k_0=float(k_0),
            f_0=float(f_0),
            lattice_constant=float(lattice_constant),
            n_bz=int(n_bz),
            k_direction=str(k_direction),
            k_margin_bins=int(k_margin_bins),
            f_margin_bins=int(f_margin_bins),
            neighbor_reduce=str(neighbor_reduce),
        )

        logger.info(
            "Mode profile extracted: shape=%s, k_bins=%s, f_bins=%s",
            getattr(mode_2d, "shape", None),
            info.get("k_bins_selected"),
            info.get("f_bins_selected"),
        )
        return x_axis, y_axis, mode_2d

    # =========================================================================
    # Non-interactive methods
    # =========================================================================

    def mode(
        self,
        k: float,
        f: float,
        lattice_constant_nm: float | None = None,
        n_bz: int = 3,
        k_direction: str = "both",
        k_margin_bins: int = 0,
        f_margin_bins: int = 0,
        neighbor_reduce: str = "mean",
    ) -> ModeProfile:
        """
        Extract 2D spatial mode profile m(x, y) and return visualization object.
        
        Returns a ModeProfile object with plotting and animation capabilities.
        Uses the Rychły et al. algorithm to reconstruct spatial mode from
        pre-computed S_complex data.
        
        Parameters
        ----------
        k : float
            Wave vector in rad/μm
        f : float
            Frequency in GHz
        lattice_constant_nm : float, optional
            Lattice constant in nm. If None, uses default from widget (470 nm).
        n_bz : int, default=3
            Number of Brillouin zones to include in mask (±n_bz around k_0).
        k_direction : str, default='both'
            Direction filter: 'both' (±k), 'positive' (+k only), 'negative' (-k only).
        k_margin_bins : int, default=0
            Number of neighboring k-bins (±) to include around each selected replica.
        f_margin_bins : int, default=0
            Number of neighboring frequency bins (±) to include around selected f.
        neighbor_reduce : {'mean', 'sum'}, default='mean'
            Reduction over the selected frequency neighborhood.
            
        Returns
        -------
        ModeProfile
            Object containing complex mode profile M(x,y) with methods:
            - .plot(mode_type='real'|'imag'|'abs'|'phase', cmap=..., dpi=...)
            - .animate(duration_ns=..., n_frames=..., fps=...)
            - .get_components() → dict with all components
            - .to_dict() → legacy dict format
            
        Examples
        --------
        >>> modes = job[0].m_layer13[...].fft.dispersion.dispersion_modes(save=True)
        >>> 
        >>> # Get mode object
        >>> mode = modes.mode(k=2.30, f=1.12)
        >>> 
        >>> # Plot different components
        >>> mode.plot(mode_type='abs', cmap='hot', dpi=150)
        >>> mode.plot(mode_type='phase', cmap='hsv')
        >>> mode.plot(mode_type='real', figsize=(12, 8))
        >>> 
        >>> # Animate
        >>> anim = mode.animate(duration_ns=10, n_frames=100, fps=30)
        >>> anim.save('mode.gif', writer='pillow')
        >>> 
        >>> # Access data
        >>> print(mode.m_xy.shape)  # Complex array (N_y, N_x)
        >>> components = mode.get_components()  # dict with real, imag, abs, phase
        """
        if self.result is None:
            raise ValueError(
                "No dispersion result available. "
                "Run dispersion_modes() with save=True first."
            )
            
        # Convert to SI units
        k_si = k * 1e6  # rad/μm → rad/m
        f_si = f * 1e9  # GHz → Hz
        
        # Get lattice constant
        if lattice_constant_nm is None:
            lattice_constant_nm = self._default_params.get("lattice_nm", 470.0)
        a = lattice_constant_nm * 1e-9  # nm → m
        
        # Extract mode using internal method (returns COMPLEX data!)
        x_axis, y_axis, mode_2d_complex = self._extract_mode_2d_custom(
            k_0=k_si,
            f_0=f_si,
            lattice_constant=a,
            n_bz=n_bz,
            k_direction=k_direction,
            k_margin_bins=k_margin_bins,
            f_margin_bins=f_margin_bins,
            neighbor_reduce=neighbor_reduce,
        )
        
        # Build metadata
        info = {
            'k_rad_um': k,
            'f_GHz': f,
            'lattice_constant_nm': lattice_constant_nm,
            'n_bz': n_bz,
            'k_direction': k_direction,
            'k_margin_bins': int(k_margin_bins),
            'f_margin_bins': int(f_margin_bins),
            'neighbor_reduce': str(neighbor_reduce),
            'shape': mode_2d_complex.shape,
            'amplitude_max': float(np.abs(mode_2d_complex).max()),
        }
        
        # Return ModeProfile object
        return ModeProfile(
            m_xy=mode_2d_complex,
            x=x_axis,
            y=y_axis,
            k=k_si,
            f=f_si,
            info=info,
        )

    def extract_mode_profile(
        self,
        k: float,
        f: float,
        lattice_constant: float | None = None,
        n_periods: int = 3,
        delta_k: float | None = None,
        delta_f: float | None = None,
        result: DispersionResult1D | None = None,
        **compute_kwargs,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Extract the spatial profile of a specific mode."""
        if result is None:
            if self.result is not None:
                result = self.result
            else:
                result = self.interface.compute_1d(**compute_kwargs)

        if lattice_constant is None:
            lattice_constant = 470e-9

        folder = self._get_folder(lattice_constant, n_periods)
        return folder.extract_mode_profile(
            result=result,
            k_0=k,
            f_0=f,
            delta_k=delta_k,
            delta_f=delta_f,
        )

    def extract_mode_evolution(
        self,
        k: float,
        f: float,
        lattice_constant: float | None = None,
        n_periods: int = 3,
        delta_k: float | None = None,
        delta_f: float | None = None,
        result: DispersionResult1D | None = None,
        **compute_kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract full time-space evolution of a mode."""
        if result is None:
            if self.result is not None:
                result = self.result
            else:
                result = self.interface.compute_1d(**compute_kwargs)

        if lattice_constant is None:
            lattice_constant = 470e-9

        folder = self._get_folder(lattice_constant, n_periods)
        return folder.extract_mode_time_evolution(
            result=result,
            k_0=k,
            f_0=f,
            delta_k=delta_k,
            delta_f=delta_f,
        )

    def animate_mode(
        self,
        k: float,
        f: float,
        lattice_constant: float | None = None,
        n_periods: int = 3,
        n_frames: int = 120,
        delta_k: float | None = None,
        delta_f: float | None = None,
        damping_time: float | None = None,
        save_path: str | None = None,
        result: DispersionResult1D | None = None,
        **animate_kwargs,
    ):
        """Animate a specific spin wave mode."""
        from .animation import SpinWaveModeAnimator, extract_amplitude_phase

        if not _HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for animation")

        if result is None:
            if self.result is not None:
                result = self.result
            else:
                result = self.interface.compute_1d()

        if lattice_constant is None:
            lattice_constant = 470e-9

        folder = self._get_folder(lattice_constant, n_periods)
        y_axis, profile_complex, info = folder.extract_mode_profile(
            result=result,
            k_0=k,
            f_0=f,
            delta_k=delta_k,
            delta_f=delta_f,
            return_complex=True,
        )

        amplitude, phase = extract_amplitude_phase(profile_complex)

        animator = SpinWaveModeAnimator(
            y_axis=y_axis,
            amplitude=amplitude,
            phase=phase,
            frequency_hz=f,
            k_value=k,
        )

        animator.animate(
            n_frames=n_frames,
            n_periods=n_periods,
            damping_time=damping_time,
            save_path=save_path,
            **animate_kwargs,
        )

        return animator

    def plot_static(
        self,
        result: DispersionResult1D | None = None,
        lattice_constant: float | None = None,
        n_periods: int = 3,
        peak_threshold: float = 0.01,
        f_min: float = 0,
        f_max: float = np.inf,
        figsize: tuple[float, float] = (12, 7),
        dpi: int = 150,
        show_heatmap: bool = True,
        show_bz_lines: bool = True,
        cmap: str = "viridis",
        ax: Axes | None = None,
        **compute_kwargs,
    ) -> tuple[Figure, Axes, FoldedDispersionResult]:
        """Create a static (non-interactive) folded dispersion plot."""
        if not _HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for plotting")

        if result is None:
            result = self.interface.compute_1d(**compute_kwargs)
        self.result = result

        if lattice_constant is None:
            lattice_constant = 470e-9

        folder = self._get_folder(lattice_constant, n_periods)
        folded = folder.fold_dispersion(result, peak_threshold)
        folded = folded.filter_frequency(f_min * 1e9, f_max * 1e9)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.get_figure()

        # Plot heatmap
        if show_heatmap:
            S = result.S.T
            k_axis = result.k_axis / 1e6
            f_axis = result.f_axis / 1e9

            # Cut data to show only positive frequencies (upper half)
            positive_freq_mask = f_axis >= 0
            if np.sum(positive_freq_mask) > 0:
                S = S[positive_freq_mask, :]
                f_axis_positive = f_axis[positive_freq_mask]
            else:
                f_axis_positive = f_axis

            # Plot data with extent from 0 to fmax
            extent = [k_axis[0], k_axis[-1], 0, f_axis_positive[-1]]
            ax.imshow(
                np.log10(S + 1e-20),
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap=cmap,
                alpha=0.8,
            )
            # Apply frequency limits to viewport only
            ax.set_ylim(f_min, f_max if f_max < np.inf else f_axis_positive[-1])

        # BZ boundary lines (reciprocal lattice vectors G = 2π/a) if enabled
        if show_bz_lines:
            G = 2 * np.pi / lattice_constant / 1e6  # rad/μm
            ax.axvline(-G, color="red", linestyle="--", linewidth=1.5)
            ax.axvline(G, color="red", linestyle="--", linewidth=1.5, label=f"±G = ±{G:.1f}")

            # Show ±2G as well if in range
            k_range = ax.get_xlim()
            k_max = max(abs(k_range[0]), abs(k_range[1]))
            if 2*G <= k_max:
                ax.axvline(-2*G, color="red", linestyle="--", linewidth=1.0, alpha=0.4)
                ax.axvline(2*G, color="red", linestyle="--", linewidth=1.0, alpha=0.4)

        ax.set_xlabel(r"$k$ [rad/μm]", fontsize=10)
        ax.set_ylabel("f [GHz]", fontsize=10)
        ax.set_title(f"Dispersion | a = {lattice_constant*1e9:.0f} nm", fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3, linestyle=":")

        return fig, ax, folded

    def __repr__(self) -> str:
        status = "result loaded" if self.result is not None else "no result"
        return f"InteractiveDispersionModes({status})"
