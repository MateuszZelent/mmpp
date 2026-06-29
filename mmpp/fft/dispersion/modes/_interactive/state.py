"""Runtime state helpers for interactive dispersion modes."""

from __future__ import annotations

from typing import Any


def base_default_params() -> dict[str, object]:
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
        "mode_type": "real",
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


def ensure_runtime_state(explorer: Any) -> None:
    """Backfill attributes for stale/autoreloaded notebook instances."""
    if not hasattr(explorer, "_animation"):
        explorer._animation = None
    if not hasattr(explorer, "_is_animating"):
        explorer._is_animating = False
    if not hasattr(explorer, "_last_compute_kwargs"):
        explorer._last_compute_kwargs = {}
    if not hasattr(explorer, "_interactive_viewer_options"):
        explorer._interactive_viewer_options = {}
    if not hasattr(explorer, "_mode_components"):
        explorer._mode_components = None
    if not hasattr(explorer, "_spectrum_components"):
        explorer._spectrum_components = None
    if not hasattr(explorer, "_analytical_options"):
        explorer._analytical_options = {}
    if not hasattr(explorer, "_default_params") or not isinstance(
        explorer._default_params, dict
    ):
        explorer._default_params = {}
    if not hasattr(explorer, "_presets_dir"):
        explorer._presets_dir = None
    if not hasattr(explorer, "_geometry_contour"):
        explorer._geometry_contour = None
    if not hasattr(explorer, "_first_dispersion_plot"):
        explorer._first_dispersion_plot = True
    if not hasattr(explorer, "_dispersion_xlim"):
        explorer._dispersion_xlim = None
    if not hasattr(explorer, "_dispersion_ylim"):
        explorer._dispersion_ylim = None
    if not hasattr(explorer, "_first_mode_plot"):
        explorer._first_mode_plot = True
    if not hasattr(explorer, "_mode_xlim"):
        explorer._mode_xlim = None
    if not hasattr(explorer, "_mode_ylim"):
        explorer._mode_ylim = None

    base = base_default_params()
    for key, value in base.items():
        explorer._default_params.setdefault(key, value)


def ensure_animation_state(explorer: Any) -> None:
    """Backfill animation attributes for legacy/stale live instances."""
    if not hasattr(explorer, "_animation"):
        explorer._animation = None
    if not hasattr(explorer, "_is_animating"):
        explorer._is_animating = False


__all__ = [
    "base_default_params",
    "ensure_runtime_state",
    "ensure_animation_state",
]
