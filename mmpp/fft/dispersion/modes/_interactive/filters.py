"""Filter-config helpers for interactive dispersion modes."""

from __future__ import annotations

from typing import Any


def build_live_filters_config(explorer: Any) -> dict[str, object] | None:
    """Build live-capable post-filter config from widget values."""

    live_cfg: dict[str, object] = {}

    if explorer.w_live_snr_enabled.value:
        live_cfg["snr_filter"] = {
            "enabled": True,
            "threshold_snr": float(explorer.w_live_snr_threshold.value),
            "method": "percentile",
            "noise_percentile": 5.0,
        }

    if explorer.w_live_gaussian_enabled.value:
        live_cfg["gaussian_morph"] = {
            "enabled": True,
            "sigma_f": float(explorer.w_live_sigma_f.value),
            "sigma_k": float(explorer.w_live_sigma_k.value),
            "threshold_std": float(explorer.w_live_gaussian_threshold_std.value),
            "opening_size": 3,
        }

    if explorer.w_live_wiener_enabled.value:
        live_cfg["wiener2d"] = {
            "enabled": True,
            "window_size": int(explorer.w_live_wiener_window.value),
        }

    if explorer.w_live_bandpass_enabled.value:
        k_min = min(explorer.w_live_kmin.value, explorer.w_live_kmax.value) * 1e6
        k_max = max(explorer.w_live_kmin.value, explorer.w_live_kmax.value) * 1e6
        f_min = float(explorer.w_fmin.value) * 1e9
        f_max = float(explorer.w_fmax.value) * 1e9
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
    if explorer.w_live_percentile_enabled.value:
        live_cfg["percentile_autoscale"] = {
            "enabled": True,
            "low_percentile": float(explorer.w_live_percentile_low.value),
            "high_percentile": float(explorer.w_live_percentile_high.value),
        }

    # Soft threshold - non-destructive noise suppression
    if explorer.w_live_soft_threshold_enabled.value:
        live_cfg["soft_threshold"] = {
            "enabled": True,
            "threshold_percentile": float(explorer.w_live_soft_percentile.value),
            "smoothness": float(explorer.w_live_soft_smoothness.value),
        }

    # Log transform - dynamic range compression
    if explorer.w_live_log_enabled.value:
        live_cfg["log_transform"] = {
            "enabled": True,
            "method": str(explorer.w_live_log_method.value),
            "scale": 1.0,
            "floor_percentile": 1.0,
        }

    # Gamma correction - reveal weak signals
    if explorer.w_live_gamma_enabled.value:
        live_cfg["gamma"] = {
            "enabled": True,
            "gamma": float(explorer.w_live_gamma_value.value),
        }

    # Local contrast normalization
    if explorer.w_live_lcn_enabled.value:
        live_cfg["local_contrast"] = {
            "enabled": True,
            "sigma": float(explorer.w_live_lcn_sigma.value),
            "epsilon": 1e-5,
        }

    # CLAHE - adaptive histogram equalization
    if explorer.w_live_clahe_enabled.value:
        live_cfg["clahe"] = {
            "enabled": True,
            "clip_limit": float(explorer.w_live_clahe_clip.value),
            "tile_size": int(explorer.w_live_clahe_tile.value),
        }

    # Unsharp mask - edge enhancement (apply last for sharpening)
    if explorer.w_live_unsharp_enabled.value:
        live_cfg["unsharp_mask"] = {
            "enabled": True,
            "sigma": float(explorer.w_live_unsharp_sigma.value),
            "alpha": float(explorer.w_live_unsharp_alpha.value),
            "threshold": 0.0,
        }

    if not live_cfg:
        return None
    return {"live": live_cfg}


def build_compute_filters_config(explorer: Any) -> dict[str, object] | None:
    """Build compute-stage filter config for recomputation."""

    filters_cfg: dict[str, object] = {}

    if explorer.w_pre_remove_static.value:
        filters_cfg["remove_static"] = True
    if explorer.w_pre_remove_average.value:
        filters_cfg["remove_average"] = True
    if explorer.w_pre_hann_time.value:
        filters_cfg["hann_time"] = True
    if explorer.w_pre_hann_space.value:
        filters_cfg["hann_space"] = True

    pre_cfg: dict[str, object] = {}

    if explorer.w_pre_envelope_enabled.value:
        pre_cfg["envelope_extraction"] = {
            "enabled": True,
            "threshold_std": float(explorer.w_pre_envelope_threshold_std.value),
            "margin_samples": int(explorer.w_pre_envelope_margin.value),
        }

    if explorer.w_pre_wavelet_enabled.value:
        pre_cfg["wavelet_denoise"] = {
            "enabled": True,
            "wavelet": "db4",
            "level": int(explorer.w_pre_wavelet_level.value),
            "method": "visu",
        }

    if explorer.w_pre_equalize_enabled.value:
        pre_cfg["amplitude_equalization"] = {
            "enabled": True,
            "smoothing_fraction": 0.05,
            "max_gain": 10.0,
            "target": "mean",
        }

    if explorer.w_pre_compression_enabled.value:
        pre_cfg["dynamic_compression"] = {
            "enabled": True,
            "method": "log",
            "alpha": float(explorer.w_pre_compression_alpha.value),
            "preserve_scale": True,
        }

    if explorer.w_pre_welch_enabled.value:
        pre_cfg["welch_average"] = {
            "enabled": True,
            "n_segments": int(explorer.w_pre_welch_segments.value),
            "overlap": float(explorer.w_pre_welch_overlap.value),
            "apply_hann": True,
        }

    if pre_cfg:
        filters_cfg["pre"] = pre_cfg

    live_cfg = build_live_filters_config(explorer)
    if live_cfg is not None and isinstance(live_cfg.get("live"), dict):
        filters_cfg["live"] = live_cfg["live"]

    return filters_cfg or None
