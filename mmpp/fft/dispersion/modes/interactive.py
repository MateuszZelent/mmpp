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

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from .._json import json_safe
from .._interactive_viewer import (
    normalize_dispersion_interactive_options,
    split_dispersion_interactive_kwargs,
)
from ._interactive import (
    apply_params as _apply_params_impl,
    base_default_params as _base_default_params_impl,
    build_compute_filters_config as _build_compute_filters_config_impl,
    build_live_filters_config as _build_live_filters_config_impl,
    create_layout as _create_layout_impl,
    on_animate as _on_animate_impl,
    on_save_animation as _on_save_animation_impl,
    stop_animation as _stop_animation_impl,
    delete_preset as _delete_preset_impl,
    ensure_animation_state as _ensure_animation_state_impl,
    ensure_runtime_state as _ensure_runtime_state_impl,
    get_current_params as _get_current_params_impl,
    get_presets_dir as _get_presets_dir_impl,
    list_presets as _list_presets_impl,
    load_preset as _load_preset_impl,
    on_delete_preset as _on_delete_preset_impl,
    on_load_preset as _on_load_preset_impl,
    on_refresh_presets as _on_refresh_presets_impl,
    on_save_preset as _on_save_preset_impl,
    refresh_preset_dropdown as _refresh_preset_dropdown_impl,
    save_preset as _save_preset_impl,
)


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
        return _base_default_params_impl()


    def _ensure_runtime_state(self):
        """Backfill attributes for stale/autoreloaded notebook instances."""
        _ensure_runtime_state_impl(self)

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
        self._display_handle = None
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
        self._interactive_viewer_options: dict[str, object] = {}
        self._mode_components: list[str] | None = None
        self._spectrum_components: list[str] | None = None
        self._analytical_options: dict[str, object] = {}

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
        return _get_presets_dir_impl(self, logger)
    
    def _get_current_params(self) -> dict:
        """Extract current parameter values from widgets."""
        return _get_current_params_impl(self)
    
    def _apply_params(self, params: dict):
        """Apply parameter values to widgets."""
        _apply_params_impl(self, params)
    
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
        return _save_preset_impl(self, name, logger)
    
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
        return _load_preset_impl(self, name, logger)
    
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
        return _delete_preset_impl(self, name, logger)
    
    def list_presets(self) -> list[str]:
        """List all available presets.
        
        Returns
        -------
        list[str]
            List of preset names (without .json extension)
        """
        return _list_presets_impl(self, logger)

    @property
    def state(self) -> dict[str, Any]:
        """Return a lightweight state compatible with the new dispersion viewer."""
        self._ensure_runtime_state()
        return {
            "modes": True,
            "mode_components": self._mode_components,
            "spectrum_components": self._spectrum_components,
            "can_reconstruct_modes": (
                self.result is not None
                and getattr(self.result, "S_complex", None) is not None
            ),
            "options": json_safe(self._interactive_viewer_options),
            "analytical": json_safe(self._analytical_options),
            "selected_k": self._selected_k,
            "selected_f": self._selected_f,
            "default_params": json_safe(self._default_params),
        }

    def _selection_payload(self, selection: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = dict(selection or {})
        if not payload and (self._selected_k is not None or self._selected_f is not None):
            payload["source"] = "legacy_modes"
        if (
            "k_rad_per_m" not in payload
            and "k_rad_um" in payload
            and payload["k_rad_um"] is not None
        ):
            payload["k_rad_per_m"] = float(payload["k_rad_um"]) * 1e6
        if (
            "k_rad_um" not in payload
            and "k_rad_per_m" in payload
            and payload["k_rad_per_m"] is not None
        ):
            payload["k_rad_um"] = float(payload["k_rad_per_m"]) / 1e6
        if (
            "f_hz" not in payload
            and "f_ghz" in payload
            and payload["f_ghz"] is not None
        ):
            payload["f_hz"] = float(payload["f_ghz"]) * 1e9
        if (
            "f_ghz" not in payload
            and "f_hz" in payload
            and payload["f_hz"] is not None
        ):
            payload["f_ghz"] = float(payload["f_hz"]) / 1e9
        if "k_rad_per_m" not in payload and self._selected_k is not None:
            payload["k_rad_per_m"] = float(self._selected_k)
            payload["k_rad_um"] = float(self._selected_k) / 1e6
        if "f_hz" not in payload and self._selected_f is not None:
            payload["f_hz"] = float(self._selected_f)
            payload["f_ghz"] = float(self._selected_f) / 1e9
        return payload

    def _mode_request(self, selection: dict[str, Any]) -> dict[str, Any]:
        k_rad_um = selection.get("k_rad_um")
        f_ghz = selection.get("f_ghz")
        component = selection.get(
            "component",
            getattr(self.result, "component", None) if self.result is not None else None,
        )
        request = {
            "available": False,
            "k_rad_um": None,
            "f_ghz": None,
            "z_layer": int(selection.get("z_layer", 0)),
            "component": component,
            "reason": "",
        }
        if k_rad_um is None or f_ghz is None:
            request["reason"] = "Select a dispersion point with k and f first."
            return request
        request["k_rad_um"] = float(k_rad_um)
        request["f_ghz"] = float(f_ghz)
        if self.result is None:
            request["reason"] = "Mode reconstruction requires a dispersion result."
            return request
        if getattr(self.result, "S_complex", None) is None:
            request["reason"] = "Mode reconstruction requires S_complex."
            return request
        request["available"] = True
        return request

    def export_selection(self, **selection: Any) -> dict[str, Any]:
        """Export selected ``(k, f)`` using the same shape as the new viewer."""
        payload = self._selection_payload(selection)
        return {
            "viewer": json_safe(self.state),
            "selection": json_safe(payload),
            "mode_request": json_safe(self._mode_request(payload)),
        }

    def collect_preset(self) -> dict[str, Any]:
        """Collect a shared interactive preset compatible with the new viewer."""
        self._ensure_runtime_state()
        selection = self._selection_payload()
        return {
            "schema_version": "dispersion-interactive-preset/v1",
            "viewer": json_safe(self.state),
            "selection": json_safe(selection),
            "mode_request": json_safe(self._mode_request(selection)),
            "legacy_modes": {
                "params": json_safe(self._get_current_params()),
            },
        }

    def apply_preset(self, payload: dict[str, Any]) -> "InteractiveDispersionModes":
        """Apply a shared interactive preset from legacy or new viewer state."""
        self._ensure_runtime_state()
        if not isinstance(payload, dict):
            return self

        viewer_state = payload.get("viewer")
        if not isinstance(viewer_state, dict):
            viewer_state = payload

        self._mode_components = viewer_state.get(
            "mode_components",
            self._mode_components,
        )
        self._spectrum_components = viewer_state.get(
            "spectrum_components",
            self._spectrum_components,
        )
        options = viewer_state.get("options")
        if isinstance(options, dict):
            self._interactive_viewer_options = dict(options)
        analytical = viewer_state.get("analytical")
        if isinstance(analytical, dict):
            self._analytical_options = dict(analytical)

        legacy_modes = payload.get("legacy_modes")
        if isinstance(legacy_modes, dict) and isinstance(
            legacy_modes.get("params"),
            dict,
        ):
            self._apply_params(dict(legacy_modes["params"]))
        elif isinstance(viewer_state.get("default_params"), dict):
            self._apply_params(dict(viewer_state["default_params"]))

        if isinstance(payload.get("selection"), dict):
            self.apply_selection(payload)
        elif isinstance(payload.get("explorer"), dict):
            explorer = payload["explorer"]
            self.apply_selection(
                k_rad_per_m=explorer.get("selected_k"),
                f_hz=explorer.get("selected_f"),
            )
        return self

    def apply_selection(
        self,
        payload: dict[str, Any] | None = None,
        **selection: Any,
    ) -> "InteractiveDispersionModes":
        """Apply a selection exported by this object or ``DispersionInteractiveViewer``."""
        merged = dict(selection)
        if payload:
            if isinstance(payload.get("selection"), dict):
                merged.update(payload["selection"])
            else:
                merged.update(payload)
        normalized = self._selection_payload(merged)
        if "k_rad_per_m" in normalized and normalized["k_rad_per_m"] is not None:
            self._selected_k = float(normalized["k_rad_per_m"])
        if "f_hz" in normalized and normalized["f_hz"] is not None:
            self._selected_f = float(normalized["f_hz"])
        return self

    def mode_at_selection(self, **selection: Any) -> Any:
        """Extract a mode at the current or supplied selection."""
        payload = self._selection_payload(selection)
        request = self._mode_request(payload)
        if not request.get("available", False):
            raise ValueError(str(request.get("reason") or "Mode selection unavailable."))
        return self.result.modes.at(
            k_rad_um=float(request["k_rad_um"]),
            f_ghz=float(request["f_ghz"]),
            z_layer=int(request.get("z_layer", 0)),
            component=request.get("component"),
        )

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
        fmax: float | None = None,
        f_units: str = "GHz",
        lognorm: bool | None = None,
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
        fmax : float, optional
            Initial maximum displayed frequency. Interpreted according to *f_units*.
        f_units : {"GHz", "Hz"}
            Units for *fmax*.
        lognorm : bool, optional
            Compatibility option from dispersion heatmaps. Enables the widget's
            non-destructive log display filter when True.
        add_contour : np.ndarray, optional
            2D geometry array (0/1) to overlay as contour on mode visualization.
            This is useful for showing material boundaries (e.g., oscillators, antidots).
        **compute_kwargs : dict
            Extra kwargs passed to compute_1d if result needs to be computed.
        """

        self._ensure_runtime_state()
        compute_kwargs, viewer_kwargs = split_dispersion_interactive_kwargs(
            dict(compute_kwargs)
        )
        (
            mode_components,
            spectrum_components,
            _modes_requested,
            viewer_options,
            analytical_options,
        ) = normalize_dispersion_interactive_options(**viewer_kwargs)
        self._interactive_viewer_options = dict(viewer_options)
        self._mode_components = mode_components
        self._spectrum_components = spectrum_components
        self._analytical_options = dict(analytical_options)

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
        if fmax is not None:
            units = f_units.lower()
            if units == "ghz":
                fmax_ghz = float(fmax)
            elif units == "hz":
                fmax_ghz = float(fmax) / 1e9
            else:
                raise ValueError("f_units must be 'GHz' or 'Hz'")
            if fmax_ghz <= 0.0:
                raise ValueError("fmax must be positive")
            self._default_params["f_max_ghz"] = fmax_ghz
        if lognorm is not None:
            self._default_params["live_log_enabled"] = bool(lognorm)
            self._default_params.setdefault("live_log_method", "log1p")
        if "n_bz" in viewer_options:
            self._default_params["n_bz_mask"] = int(viewer_options["n_bz"])
        if "mode_type" in viewer_options:
            self._default_params["mode_type"] = str(viewer_options["mode_type"])
        if "cmap" in viewer_options:
            self._default_params["cmap_disp"] = str(viewer_options["cmap"])

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

        # Display once and update the same output on repeated calls.
        if self._display_handle is None:
            self._display_handle = display(main_layout, display_id=True)
        else:
            self._display_handle.update(main_layout)

        # Initial plot
        self._initialize_figure()
        self._update_dispersion_plot()

    def close(self) -> None:
        """Best-effort cleanup for notebook display, animation, and figure state."""
        self._ensure_runtime_state()

        animation = getattr(self, "_animation", None)
        event_source = getattr(animation, "event_source", None)
        if event_source is not None and hasattr(event_source, "stop"):
            event_source.stop()
        self._animation = None
        self._is_animating = False

        if self._display_handle is not None and hasattr(self._display_handle, "update"):
            self._display_handle.update(None)
        self._display_handle = None

        fig = getattr(self, "_fig", None)
        plt_module = globals().get("plt")
        if fig is not None and plt_module is not None and hasattr(plt_module, "close"):
            plt_module.close(fig)
        self._fig = None
        self._ax_disp = None
        self._ax_mode = None
        self._colorbar_disp = None
        self._colorbar_mode = None
        self._mask_markers = []

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
            value=params.get("mode_type", "real"),
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
        return _create_layout_impl(self, widgets)

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
        _on_save_preset_impl(self, _, logger)
    
    def _on_load_preset(self, change):
        """Load a selected preset."""
        _on_load_preset_impl(self, change, logger)
    
    def _on_delete_preset(self, _):
        """Delete the selected preset."""
        _on_delete_preset_impl(self, _, logger)
    
    def _on_refresh_presets(self, _):
        """Refresh the preset dropdown list."""
        _on_refresh_presets_impl(self, _, logger)
    
    def _refresh_preset_dropdown(self):
        """Update the preset dropdown with current list of presets."""
        _refresh_preset_dropdown_impl(self, logger)
    
    def _update_k_direction_visibility(self):
        """Show/hide k-direction dropdown based on N_BZ value.
        
        With current mask definition (k0 ± n*G), direction is meaningful even
        for N_BZ=0, so keep it visible in normal operation.
        """
        self.w_k_direction.layout.display = ''

    def _build_live_filters_config(self) -> dict[str, object] | None:
        """Build live-capable post-filter config from widget values."""
        return _build_live_filters_config_impl(self)


    def _build_compute_filters_config(self) -> dict[str, object] | None:
        """Build compute-stage filter config for recomputation."""
        return _build_compute_filters_config_impl(self)

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
        _ensure_animation_state_impl(self)

    def _on_animate(self, _):
        """Toggle animation of selected mode in the mode visualization panel."""
        _on_animate_impl(self, _, logger)
    
    def _on_save_animation(self, _):
        """Save the current animation to file.

        Goal: saved file should match what the user currently sees (layout, zoom,
        filters, colormaps, markers).
        """
        _on_save_animation_impl(self, _, logger)

    def _stop_animation(self):
        """Stop the current animation."""
        _stop_animation_impl(self)

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
