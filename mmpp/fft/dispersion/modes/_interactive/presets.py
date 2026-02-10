"""Preset persistence helpers for interactive dispersion modes."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


def get_presets_dir(explorer: Any, logger: Any) -> Path:
    """Get or create directory for storing presets."""
    if explorer._presets_dir is None:
        cwd = Path(os.getcwd())
        explorer._presets_dir = cwd / ".mmpp_presets"
        explorer._presets_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Preset directory: %s", explorer._presets_dir)
    return explorer._presets_dir


def get_current_params(explorer: Any) -> dict:
    """Extract current parameter values from widgets."""
    if not explorer._widgets_created:
        return explorer._default_params.copy()

    return {
        # Basic parameters
        "lattice_nm": float(explorer.w_lattice.value),
        "n_bz_mask": int(explorer.w_n_bz_mask.value),
        "k_margin_bins": int(explorer.w_k_margin.value),
        "f_margin_bins": int(explorer.w_f_margin.value),
        "neighbor_reduce": str(explorer.w_neighbor_reduce.value),
        "f_min_ghz": float(explorer.w_fmin.value),
        "f_max_ghz": float(explorer.w_fmax.value),
        "k_direction": str(explorer.w_k_direction.value),
        "cmap_disp": str(explorer.w_cmap_disp.value),
        "cmap_mode": str(explorer.w_cmap_mode.value),
        # Live post-filters
        "live_snr_enabled": bool(explorer.w_live_snr_enabled.value),
        "live_snr_threshold": float(explorer.w_live_snr_threshold.value),
        "live_gaussian_enabled": bool(explorer.w_live_gaussian_enabled.value),
        "live_sigma_f": float(explorer.w_live_sigma_f.value),
        "live_sigma_k": float(explorer.w_live_sigma_k.value),
        "live_gaussian_threshold_std": float(
            explorer.w_live_gaussian_threshold_std.value
        ),
        "live_wiener_enabled": bool(explorer.w_live_wiener_enabled.value),
        "live_wiener_window": int(explorer.w_live_wiener_window.value),
        "live_bandpass_enabled": bool(explorer.w_live_bandpass_enabled.value),
        "live_kmin_rad_um": float(explorer.w_live_kmin.value),
        "live_kmax_rad_um": float(explorer.w_live_kmax.value),
        # Compute-stage filters
        "pre_remove_static": bool(explorer.w_pre_remove_static.value),
        "pre_remove_average": bool(explorer.w_pre_remove_average.value),
        "pre_hann_time": bool(explorer.w_pre_hann_time.value),
        "pre_hann_space": bool(explorer.w_pre_hann_space.value),
        "pre_envelope_enabled": bool(explorer.w_pre_envelope_enabled.value),
        "pre_envelope_threshold_std": float(explorer.w_pre_envelope_threshold_std.value),
        "pre_envelope_margin": int(explorer.w_pre_envelope_margin.value),
        "pre_wavelet_enabled": bool(explorer.w_pre_wavelet_enabled.value),
        "pre_wavelet_level": int(explorer.w_pre_wavelet_level.value),
        "pre_equalize_enabled": bool(explorer.w_pre_equalize_enabled.value),
        "pre_compression_enabled": bool(explorer.w_pre_compression_enabled.value),
        "pre_compression_alpha": float(explorer.w_pre_compression_alpha.value),
        "pre_welch_enabled": bool(explorer.w_pre_welch_enabled.value),
        "pre_welch_segments": int(explorer.w_pre_welch_segments.value),
        "pre_welch_overlap": float(explorer.w_pre_welch_overlap.value),
        # Enhancement filters
        "live_log_enabled": bool(explorer.w_live_log_enabled.value),
        "live_log_method": str(explorer.w_live_log_method.value),
        "live_gamma_enabled": bool(explorer.w_live_gamma_enabled.value),
        "live_gamma_value": float(explorer.w_live_gamma_value.value),
        "live_clahe_enabled": bool(explorer.w_live_clahe_enabled.value),
        "live_clahe_clip": float(explorer.w_live_clahe_clip.value),
        "live_clahe_tile": int(explorer.w_live_clahe_tile.value),
        "live_lcn_enabled": bool(explorer.w_live_lcn_enabled.value),
        "live_lcn_sigma": float(explorer.w_live_lcn_sigma.value),
        "live_unsharp_enabled": bool(explorer.w_live_unsharp_enabled.value),
        "live_unsharp_sigma": float(explorer.w_live_unsharp_sigma.value),
        "live_unsharp_alpha": float(explorer.w_live_unsharp_alpha.value),
        "live_percentile_enabled": bool(explorer.w_live_percentile_enabled.value),
        "live_percentile_low": float(explorer.w_live_percentile_low.value),
        "live_percentile_high": float(explorer.w_live_percentile_high.value),
        "live_soft_threshold_enabled": bool(explorer.w_live_soft_threshold_enabled.value),
        "live_soft_percentile": float(explorer.w_live_soft_percentile.value),
        "live_soft_smoothness": float(explorer.w_live_soft_smoothness.value),
    }


def apply_params(explorer: Any, params: dict) -> None:
    """Apply parameter values to widgets."""
    if not explorer._widgets_created:
        explorer._default_params.update(params)
        return

    # Basic parameters
    if "lattice_nm" in params:
        explorer.w_lattice.value = float(params["lattice_nm"])
    if "n_bz_mask" in params:
        explorer.w_n_bz_mask.value = int(params["n_bz_mask"])
    if "k_margin_bins" in params:
        explorer.w_k_margin.value = int(params["k_margin_bins"])
    if "f_margin_bins" in params:
        explorer.w_f_margin.value = int(params["f_margin_bins"])
    if "neighbor_reduce" in params:
        explorer.w_neighbor_reduce.value = str(params["neighbor_reduce"])
    if "f_min_ghz" in params:
        explorer.w_fmin.value = float(params["f_min_ghz"])
    if "f_max_ghz" in params:
        explorer.w_fmax.value = float(params["f_max_ghz"])
    if "k_direction" in params:
        explorer.w_k_direction.value = str(params["k_direction"])
    if "cmap_disp" in params:
        explorer.w_cmap_disp.value = str(params["cmap_disp"])
    if "cmap_mode" in params:
        explorer.w_cmap_mode.value = str(params["cmap_mode"])

    # Live post-filters
    if "live_snr_enabled" in params:
        explorer.w_live_snr_enabled.value = bool(params["live_snr_enabled"])
    if "live_snr_threshold" in params:
        explorer.w_live_snr_threshold.value = float(params["live_snr_threshold"])
    if "live_gaussian_enabled" in params:
        explorer.w_live_gaussian_enabled.value = bool(params["live_gaussian_enabled"])
    if "live_sigma_f" in params:
        explorer.w_live_sigma_f.value = float(params["live_sigma_f"])
    if "live_sigma_k" in params:
        explorer.w_live_sigma_k.value = float(params["live_sigma_k"])
    if "live_gaussian_threshold_std" in params:
        explorer.w_live_gaussian_threshold_std.value = float(
            params["live_gaussian_threshold_std"]
        )
    if "live_wiener_enabled" in params:
        explorer.w_live_wiener_enabled.value = bool(params["live_wiener_enabled"])
    if "live_wiener_window" in params:
        explorer.w_live_wiener_window.value = int(params["live_wiener_window"])
    if "live_bandpass_enabled" in params:
        explorer.w_live_bandpass_enabled.value = bool(params["live_bandpass_enabled"])
    if "live_kmin_rad_um" in params:
        explorer.w_live_kmin.value = float(params["live_kmin_rad_um"])
    if "live_kmax_rad_um" in params:
        explorer.w_live_kmax.value = float(params["live_kmax_rad_um"])

    # Compute-stage filters
    if "pre_remove_static" in params:
        explorer.w_pre_remove_static.value = bool(params["pre_remove_static"])
    if "pre_remove_average" in params:
        explorer.w_pre_remove_average.value = bool(params["pre_remove_average"])
    if "pre_hann_time" in params:
        explorer.w_pre_hann_time.value = bool(params["pre_hann_time"])
    if "pre_hann_space" in params:
        explorer.w_pre_hann_space.value = bool(params["pre_hann_space"])
    if "pre_envelope_enabled" in params:
        explorer.w_pre_envelope_enabled.value = bool(params["pre_envelope_enabled"])
    if "pre_envelope_threshold_std" in params:
        explorer.w_pre_envelope_threshold_std.value = float(
            params["pre_envelope_threshold_std"]
        )
    if "pre_envelope_margin" in params:
        explorer.w_pre_envelope_margin.value = int(params["pre_envelope_margin"])
    if "pre_wavelet_enabled" in params:
        explorer.w_pre_wavelet_enabled.value = bool(params["pre_wavelet_enabled"])
    if "pre_wavelet_level" in params:
        explorer.w_pre_wavelet_level.value = int(params["pre_wavelet_level"])
    if "pre_equalize_enabled" in params:
        explorer.w_pre_equalize_enabled.value = bool(params["pre_equalize_enabled"])
    if "pre_compression_enabled" in params:
        explorer.w_pre_compression_enabled.value = bool(params["pre_compression_enabled"])
    if "pre_compression_alpha" in params:
        explorer.w_pre_compression_alpha.value = float(params["pre_compression_alpha"])
    if "pre_welch_enabled" in params:
        explorer.w_pre_welch_enabled.value = bool(params["pre_welch_enabled"])
    if "pre_welch_segments" in params:
        explorer.w_pre_welch_segments.value = int(params["pre_welch_segments"])
    if "pre_welch_overlap" in params:
        explorer.w_pre_welch_overlap.value = float(params["pre_welch_overlap"])

    # Enhancement filters
    if "live_log_enabled" in params:
        explorer.w_live_log_enabled.value = bool(params["live_log_enabled"])
    if "live_log_method" in params:
        explorer.w_live_log_method.value = str(params["live_log_method"])
    if "live_gamma_enabled" in params:
        explorer.w_live_gamma_enabled.value = bool(params["live_gamma_enabled"])
    if "live_gamma_value" in params:
        explorer.w_live_gamma_value.value = float(params["live_gamma_value"])
    if "live_clahe_enabled" in params:
        explorer.w_live_clahe_enabled.value = bool(params["live_clahe_enabled"])
    if "live_clahe_clip" in params:
        explorer.w_live_clahe_clip.value = float(params["live_clahe_clip"])
    if "live_clahe_tile" in params:
        explorer.w_live_clahe_tile.value = int(params["live_clahe_tile"])
    if "live_lcn_enabled" in params:
        explorer.w_live_lcn_enabled.value = bool(params["live_lcn_enabled"])
    if "live_lcn_sigma" in params:
        explorer.w_live_lcn_sigma.value = float(params["live_lcn_sigma"])
    if "live_unsharp_enabled" in params:
        explorer.w_live_unsharp_enabled.value = bool(params["live_unsharp_enabled"])
    if "live_unsharp_sigma" in params:
        explorer.w_live_unsharp_sigma.value = float(params["live_unsharp_sigma"])
    if "live_unsharp_alpha" in params:
        explorer.w_live_unsharp_alpha.value = float(params["live_unsharp_alpha"])
    if "live_percentile_enabled" in params:
        explorer.w_live_percentile_enabled.value = bool(params["live_percentile_enabled"])
    if "live_percentile_low" in params:
        explorer.w_live_percentile_low.value = float(params["live_percentile_low"])
    if "live_percentile_high" in params:
        explorer.w_live_percentile_high.value = float(params["live_percentile_high"])
    if "live_soft_threshold_enabled" in params:
        explorer.w_live_soft_threshold_enabled.value = bool(params["live_soft_threshold_enabled"])
    if "live_soft_percentile" in params:
        explorer.w_live_soft_percentile.value = float(params["live_soft_percentile"])
    if "live_soft_smoothness" in params:
        explorer.w_live_soft_smoothness.value = float(params["live_soft_smoothness"])


def save_preset(explorer: Any, name: str, logger: Any) -> bool:
    """Save current parameters as a preset."""
    try:
        name = name.strip().replace("/", "_").replace("\\", "_")
        if not name:
            logger.warning("Preset name cannot be empty")
            return False

        presets_dir = get_presets_dir(explorer, logger)
        preset_file = presets_dir / f"{name}.json"

        params = get_current_params(explorer)
        preset_data = {"created": datetime.now().isoformat(), "params": params}

        with open(preset_file, "w", encoding="utf-8") as f:
            json.dump(preset_data, f, indent=2)

        logger.info("Preset '%s' saved to %s", name, preset_file)
        return True
    except Exception as exc:
        logger.error("Failed to save preset '%s': %s", name, exc)
        return False


def load_preset(explorer: Any, name: str, logger: Any) -> bool:
    """Load parameters from a preset."""
    try:
        presets_dir = get_presets_dir(explorer, logger)
        preset_file = presets_dir / f"{name}.json"

        if not preset_file.exists():
            logger.warning("Preset '%s' not found at %s", name, preset_file)
            return False

        with open(preset_file, encoding="utf-8") as f:
            preset_data = json.load(f)

        if isinstance(preset_data, dict) and "params" in preset_data:
            params = preset_data["params"]
        else:
            params = preset_data

        apply_params(explorer, params)
        logger.info("Preset '%s' loaded from %s", name, preset_file)
        return True
    except Exception as exc:
        logger.error("Failed to load preset '%s': %s", name, exc)
        return False


def delete_preset(explorer: Any, name: str, logger: Any) -> bool:
    """Delete a saved preset."""
    try:
        presets_dir = get_presets_dir(explorer, logger)
        preset_file = presets_dir / f"{name}.json"

        if not preset_file.exists():
            logger.warning("Preset '%s' not found", name)
            return False

        preset_file.unlink()
        logger.info("Preset '%s' deleted", name)
        return True
    except Exception as exc:
        logger.error("Failed to delete preset '%s': %s", name, exc)
        return False


def list_presets(explorer: Any, logger: Any) -> list[str]:
    """List all available presets."""
    try:
        presets_dir = get_presets_dir(explorer, logger)
        preset_files = list(presets_dir.glob("*.json"))
        return sorted([f.stem for f in preset_files])
    except Exception as exc:
        logger.error("Failed to list presets: %s", exc)
        return []


def refresh_preset_dropdown(explorer: Any, logger: Any) -> None:
    """Update preset dropdown with current list of presets."""
    available_presets = list_presets(explorer, logger)
    preset_options = [("-- Load Preset --", "")] + [
        (name, name) for name in available_presets
    ]
    explorer.w_preset_load.options = preset_options


def on_save_preset(explorer: Any, _evt: Any, logger: Any) -> None:
    """Save current parameters as a preset from UI callback."""
    preset_name = explorer.w_preset_name.value.strip()
    if not preset_name:
        explorer.w_info.value = "<small style='color:orange'>⚠️ Enter preset name</small>"
        return

    if save_preset(explorer, preset_name, logger):
        explorer.w_info.value = (
            f"<small style='color:green'>✅ Saved to .mmpp_presets/{preset_name}.json</small>"
        )
        refresh_preset_dropdown(explorer, logger)
        explorer.w_preset_name.value = ""
    else:
        explorer.w_info.value = (
            f"<small style='color:red'>❌ Failed to save '{preset_name}'</small>"
        )


def on_load_preset(explorer: Any, change: dict[str, Any], logger: Any) -> None:
    """Load selected preset from UI callback."""
    preset_name = change["new"]
    if not preset_name:
        return

    if load_preset(explorer, preset_name, logger):
        explorer.w_info.value = (
            f"<small style='color:green'>✅ Preset '{preset_name}' loaded</small>"
        )
        explorer._update_dispersion_plot()
        explorer._refresh_mode_or_animation()
    else:
        explorer.w_info.value = (
            f"<small style='color:red'>❌ Failed to load preset '{preset_name}'</small>"
        )


def on_delete_preset(explorer: Any, _evt: Any, logger: Any) -> None:
    """Delete selected preset from UI callback."""
    preset_name = explorer.w_preset_load.value
    if not preset_name:
        explorer.w_info.value = (
            "<small style='color:orange'>⚠️ Select preset to delete</small>"
        )
        return

    if delete_preset(explorer, preset_name, logger):
        explorer.w_info.value = f"<small style='color:green'>✅ Deleted '{preset_name}'</small>"
        refresh_preset_dropdown(explorer, logger)
        explorer.w_preset_load.value = ""
    else:
        explorer.w_info.value = (
            f"<small style='color:red'>❌ Failed to delete '{preset_name}'</small>"
        )


def on_refresh_presets(explorer: Any, _evt: Any, logger: Any) -> None:
    """Refresh preset dropdown from UI callback."""
    refresh_preset_dropdown(explorer, logger)
    presets_dir = get_presets_dir(explorer, logger)
    count = len(list_presets(explorer, logger))
    explorer.w_info.value = (
        f"<small style='color:green'>✅ Found {count} preset(s) in {presets_dir.name}/</small>"
    )


__all__ = [
    "get_presets_dir",
    "get_current_params",
    "apply_params",
    "save_preset",
    "load_preset",
    "delete_preset",
    "list_presets",
    "refresh_preset_dropdown",
    "on_save_preset",
    "on_load_preset",
    "on_delete_preset",
    "on_refresh_presets",
]
