"""Composable filter pipeline shared across FFT modules."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config import FilterConfig
from .postprocess import apply_postprocess_filters
from .preprocess import apply_filter as apply_preprocess_filter
from .windows import apply_window

logger = logging.getLogger(__name__)

# Preprocess filters
PREPROCESS_FILTER_KEYS = {
    "remove_static",
    "remove_average",
    "remove_mean",
    "remove_mean_and_static",
    "hann_time",
    "detrend",
    "detrend_linear",
    "savgol_smooth",
    "baseline_correction",
    "high_pass",
    "band_pass",
    "spectral_derivative",
}

# Postprocess filters
POSTPROCESS_FILTER_KEYS = {
    "normalize",
    "log_transform",
    "gamma",
    "percentile_clip",
    "soft_threshold",
    "gaussian_smooth",
    "savgol_smooth",
    "moving_average",
    "baseline_correction",
}

# Filters safe to apply on cached/postcomputed spectrum
LIVE_FILTER_KEYS = {
    "normalize",
    "log_transform",
    "gamma",
    "percentile_clip",
    "soft_threshold",
    "gaussian_smooth",
    "savgol_smooth",
    "moving_average",
    "baseline_correction",
}


def _is_enabled(option: Any) -> bool:
    if isinstance(option, dict):
        return bool(option.get("enabled", True))
    return bool(option)


def _normalize_legacy_entry(key: str, value: Any) -> tuple[str, Any]:
    """Convert legacy root keys into stage-aware entry."""
    if key == "smooth_filter":
        mode = str(value).lower()
        if mode in {"gaussian", "gaussian_smooth"}:
            return "gaussian_smooth", {"enabled": True}
        if mode in {"savgol", "savgol_smooth"}:
            return "savgol_smooth", {"enabled": True}
        if mode in {"moving", "moving_average"}:
            return "moving_average", {"enabled": True}
        return "", None
    if key == "baseline_mode":
        mode = str(value).lower()
        if mode in {"none", "", "false"}:
            return "", None
        return "baseline_correction", {"enabled": True, "mode": mode}
    if key == "log_scale":
        return "log_transform", bool(value)
    return "", None


def normalize_filter_config(
    filters: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Normalize filter config into stable stage blocks.

    Output schema:
    {
      "pre": {...},
      "post": {...},
      "live": {...},
      ...unknown passthrough keys...
    }
    """
    if not filters:
        return None

    normalized: dict[str, Any] = {}

    # Legacy aliases used by interactive spectrum.
    for legacy_key in ("smooth_filter", "baseline_mode", "log_scale"):
        if legacy_key in filters:
            mapped_key, mapped_val = _normalize_legacy_entry(legacy_key, filters[legacy_key])
            if mapped_key:
                normalized.setdefault("post", {})[mapped_key] = mapped_val

    # Explicit stage blocks.
    for stage_name in ("pre", "post", "live"):
        stage_cfg = filters.get(stage_name)
        if isinstance(stage_cfg, dict):
            normalized.setdefault(stage_name, {}).update(
                {str(k): v for k, v in stage_cfg.items()}
            )

    skip_keys = {"pre", "post", "live", "advanced", "smooth_filter", "baseline_mode", "log_scale"}
    for key, value in filters.items():
        if key in skip_keys:
            continue
        if key in PREPROCESS_FILTER_KEYS:
            if _is_enabled(value):
                normalized.setdefault("pre", {})[key] = value
        elif key in POSTPROCESS_FILTER_KEYS:
            if _is_enabled(value):
                normalized.setdefault("post", {})[key] = value
        elif key in LIVE_FILTER_KEYS:
            if _is_enabled(value):
                normalized.setdefault("live", {})[key] = value
        else:
            # Preserve unknown keys to keep cache signatures stable.
            normalized[key] = value

    return normalized or None


def split_filter_stages(
    filters: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Split filter config into pre/post/live stage dictionaries."""
    cfg = normalize_filter_config(filters)
    if not cfg:
        return {}, {}, {}

    pre: dict[str, Any] = {}
    post: dict[str, Any] = {}
    live: dict[str, Any] = {}
    for stage_name, target in (("pre", pre), ("post", post), ("live", live)):
        stage_cfg = cfg.get(stage_name)
        if isinstance(stage_cfg, dict):
            for name, option in stage_cfg.items():
                if _is_enabled(option):
                    target[name] = option
    return pre, post, live


def classify_filter_execution(
    filters: dict[str, Any] | None,
) -> dict[str, list[str]]:
    """Return active filters grouped by compute/post/live applicability."""
    pre, post, live = split_filter_stages(filters)
    compute_stage = sorted(pre.keys())
    post_stage = sorted(post.keys())
    live_capable = sorted(
        name
        for name in set(list(post.keys()) + list(live.keys()))
        if name in LIVE_FILTER_KEYS
    )
    return {
        "compute_stage": compute_stage,
        "post_stage": post_stage,
        "live_capable": live_capable,
    }


class FilterPipeline:
    """Centralized filter pipeline used by FFT modules."""

    def __init__(self, config: FilterConfig | None = None):
        self.config = config or FilterConfig()

    def _config_to_stages(self) -> dict[str, dict[str, Any]]:
        """Convert dataclass configuration into stage dictionaries."""
        pre: dict[str, Any] = {}
        if self.config.pre.remove_static:
            pre["remove_static"] = True
        if self.config.pre.remove_mean:
            pre["remove_mean"] = True
        if self.config.pre.detrend == "linear":
            pre["detrend_linear"] = True
        if self.config.pre.window and self.config.pre.window != "none":
            pre["hann_time"] = self.config.pre.window
        if self.config.pre.high_pass_cutoff is not None:
            pre["high_pass"] = {"cutoff_fraction": float(self.config.pre.high_pass_cutoff)}
        if self.config.pre.band_pass is not None:
            lo, hi = self.config.pre.band_pass
            pre["band_pass"] = {"low_fraction": float(lo), "high_fraction": float(hi)}
        if self.config.pre.savgol_window > 0:
            pre["savgol_smooth"] = {
                "window_length": int(self.config.pre.savgol_window),
                "polyorder": int(self.config.pre.savgol_polyorder),
            }

        post: dict[str, Any] = {}
        if self.config.post.baseline != "none":
            post["baseline_correction"] = {"mode": self.config.post.baseline}
        lo, hi = self.config.post.percentile_clip
        if lo > 0.0 or hi < 100.0:
            post["percentile_clip"] = {"low": float(lo), "high": float(hi)}
        if self.config.post.soft_threshold > 0.0:
            post["soft_threshold"] = {"percentile": float(self.config.post.soft_threshold)}
        if self.config.post.smooth != "none":
            post_key = {
                "gaussian": "gaussian_smooth",
                "savgol": "savgol_smooth",
                "moving_average": "moving_average",
            }.get(self.config.post.smooth, "gaussian_smooth")
            post[post_key] = {
                "smooth_window": int(self.config.post.smooth_window),
                "smooth_sigma": float(self.config.post.smooth_sigma),
            }
        if self.config.post.normalize:
            post["normalize"] = True
        if self.config.post.log_scale:
            post["log_transform"] = True
        if self.config.post.gamma != 1.0:
            post["gamma"] = {"gamma": float(self.config.post.gamma)}

        return {"pre": pre, "post": post, "live": dict(post)}

    def preprocess(
        self,
        data: np.ndarray,
        dt: float = 1.0,
        filters: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Apply preprocessing filters (time-domain)."""
        _ = dt  # reserved for future filters requiring physical frequency scale
        cfg = normalize_filter_config(filters) if filters is not None else self._config_to_stages()
        pre_filters = cfg.get("pre", {}) if cfg else {}
        if not pre_filters:
            return np.asarray(data)

        result = np.array(data, copy=True)
        for name, option in pre_filters.items():
            lname = str(name).lower()
            if lname == "hann_time":
                # Support bool, string alias, and option dict.
                window_type = "hann"
                if isinstance(option, str):
                    window_type = option
                elif isinstance(option, dict):
                    window_type = str(option.get("window", "hann"))
                result = apply_window(result, window_type=window_type)
                continue
            result = apply_preprocess_filter(result, lname)
        return result

    def postprocess(
        self,
        spectrum: np.ndarray,
        frequencies: np.ndarray,
        filters: dict[str, Any] | None = None,
        stage: str = "post",
    ) -> np.ndarray:
        """Apply postprocess or live-stage filters to spectrum arrays."""
        cfg = normalize_filter_config(filters) if filters is not None else self._config_to_stages()
        if not cfg:
            return np.asarray(spectrum)
        stage_filters = cfg.get(stage, {})
        if not stage_filters:
            return np.asarray(spectrum)
        return apply_postprocess_filters(spectrum, frequencies, stage_filters)

    def live(
        self,
        spectrum: np.ndarray,
        frequencies: np.ndarray,
        filters: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Apply live-capable postprocess filters."""
        return self.postprocess(spectrum, frequencies, filters=filters, stage="live")

    @staticmethod
    def is_live(changed_field: str) -> bool:
        """Check whether changing a given filter can avoid FFT recompute."""
        return str(changed_field) in LIVE_FILTER_KEYS

    @staticmethod
    def classify_changes(
        old: dict[str, Any] | None,
        new: dict[str, Any] | None,
    ) -> tuple[bool, bool]:
        """Return ``(needs_recompute, needs_redisplay)``."""
        old_norm = normalize_filter_config(old) or {}
        new_norm = normalize_filter_config(new) or {}
        old_pre, old_post, old_live = split_filter_stages(old_norm)
        new_pre, new_post, new_live = split_filter_stages(new_norm)

        pre_changed = old_pre != new_pre
        display_changed = (old_post != new_post) or (old_live != new_live)
        return pre_changed, pre_changed or display_changed
