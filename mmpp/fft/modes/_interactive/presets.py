"""Preset management helpers for :mod:`mmpp.fft.modes.interactive`."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .filters import (
    COMPONENT_NAMES,
    normalize_component_selection,
    normalize_spectrum_component_selection,
)


def get_presets_dir(explorer: Any) -> Path:
    """Return (and lazily create) project-local presets directory."""
    if explorer._presets_dir is None:
        explorer._presets_dir = Path.cwd() / ".mmpp_presets"
        explorer._presets_dir.mkdir(parents=True, exist_ok=True)
    return explorer._presets_dir


def list_presets(explorer: Any) -> list[str]:
    """List available interactive toolbar presets."""
    preset_dir = get_presets_dir(explorer)
    names = []
    for file_path in sorted(preset_dir.glob("fmr_*.json")):
        name = file_path.stem.removeprefix("fmr_")
        if name:
            names.append(name)
    return names


def collect_preset_state(explorer: Any) -> dict[str, Any]:
    """Collect serializable state from current controls."""
    return {
        # "components" kept for backward-compat with older presets.
        "components": list(explorer._mode_components),
        "mode_components": list(explorer._mode_components),
        "spectrum_components": list(explorer._spectrum_components),
        "z_layer": int(explorer._current_z_layer),
        "freq_min": float(explorer._filter_state.freq_min),
        "freq_max": float(explorer._filter_state.freq_max),
        "smooth_filter": str(explorer._filter_state.smooth_filter),
        "smooth_window": int(explorer._filter_state.smooth_window),
        "smooth_sigma": float(explorer._filter_state.smooth_sigma),
        "baseline_mode": str(explorer._filter_state.baseline_mode),
        "clip_percentile_low": float(explorer._filter_state.clip_percentile_low),
        "clip_percentile_high": float(explorer._filter_state.clip_percentile_high),
        "soft_threshold_percentile": float(
            explorer._filter_state.soft_threshold_percentile
        ),
        "normalize": bool(explorer._filter_state.normalize),
        "log_scale": bool(explorer._filter_state.log_scale),
        "show_peaks": bool(explorer._show_peaks),
        "peak_prominence": float(explorer._peak_prominence),
        "peak_distance": int(explorer._peak_distance),
        "mode_view": (
            "all" if len(explorer._mode_row_types) > 1 else explorer._mode_row_types[0]
        ),
        "cmap_mag": (
            str(explorer._controls.get("cmap_mag").value)
            if explorer._controls.get("cmap_mag")
            else "viridis"
        ),
        "cmap_phase": (
            str(explorer._controls.get("cmap_phase").value)
            if explorer._controls.get("cmap_phase")
            else "twilight"
        ),
        "cmap_combined": (
            str(explorer._controls.get("cmap_combined").value)
            if explorer._controls.get("cmap_combined")
            else "RdBu_r"
        ),
        "aspect": (
            str(explorer._controls.get("aspect").value)
            if explorer._controls.get("aspect")
            else "equal"
        ),
        "layout": (
            str(explorer._controls.get("layout").value)
            if explorer._controls.get("layout")
            else "auto"
        ),
        "freq_unit": str(explorer._freq_unit),
    }


def apply_preset_state(explorer: Any, payload: dict[str, Any]) -> None:
    """Apply preset payload to widgets/state."""
    if not isinstance(payload, dict):
        raise TypeError("Preset payload must be a JSON object")
    if not explorer._controls:
        return

    freq_unit = str(payload.get("freq_unit", explorer._freq_unit)).strip()
    explorer._get_freq_scale(freq_unit)

    integer_keys = {"z_layer", "smooth_window", "peak_distance"}
    numeric_keys = integer_keys | {
        "freq_min",
        "freq_max",
        "smooth_sigma",
        "clip_percentile_low",
        "clip_percentile_high",
        "soft_threshold_percentile",
        "peak_prominence",
    }
    for key in numeric_keys:
        if key not in payload:
            continue
        value = payload[key]
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Preset field {key!r} must be numeric") from exc
        if not np.isfinite(numeric):
            raise ValueError(f"Preset field {key!r} must be finite")
        if key in integer_keys and (
            isinstance(value, (bool, np.bool_)) or int(numeric) != numeric
        ):
            raise ValueError(f"Preset field {key!r} must be an integer")

    def _option_values(control_name: str) -> set[str]:
        values: set[str] = set()
        for option in explorer._controls[control_name].options:
            raw = option[1] if isinstance(option, tuple) else option
            values.add(str(raw))
        return values

    enum_fields = {
        "smooth_filter": "smooth_filter",
        "baseline_mode": "baseline_mode",
        "mode_view": "mode_view",
        "cmap_mag": "cmap_mag",
        "cmap_phase": "cmap_phase",
        "cmap_combined": "cmap_combined",
        "aspect": "aspect",
        "layout": "layout",
    }
    for payload_key, control_name in enum_fields.items():
        if payload_key in payload and str(payload[payload_key]) not in _option_values(
            control_name
        ):
            raise ValueError(
                f"Preset field {payload_key!r} has unsupported value "
                f"{payload[payload_key]!r}"
            )

    clip_low_value = float(
        payload.get("clip_percentile_low", explorer._filter_state.clip_percentile_low)
    )
    clip_high_value = float(
        payload.get("clip_percentile_high", explorer._filter_state.clip_percentile_high)
    )
    if clip_low_value > clip_high_value:
        raise ValueError(
            "Preset clip_percentile_low cannot exceed clip_percentile_high"
        )

    explorer._internal_update = True
    try:
        mode_components = normalize_component_selection(
            payload.get("mode_components", payload.get("components")),
            available=explorer._available_components or COMPONENT_NAMES,
        )
        spectrum_components = normalize_spectrum_component_selection(
            payload.get("spectrum_components", payload.get("components")),
            available=explorer._available_components,
        )

        mode_key = (
            "mode_components"
            if "mode_components" in explorer._controls
            else "components"
        )
        explorer._controls[mode_key].value = tuple(mode_components)
        if "spectrum_components" in explorer._controls:
            explorer._controls["spectrum_components"].value = tuple(spectrum_components)
        z_control = explorer._controls["z_layer"]
        z_val = int(payload.get("z_layer", explorer._current_z_layer))
        explorer._controls["z_layer"].value = int(
            np.clip(z_val, z_control.min, z_control.max)
        )

        fmin_control = explorer._controls["fmin"]
        fmax_control = explorer._controls["fmax"]
        fmin = float(payload.get("freq_min", explorer._filter_state.freq_min))
        fmax = float(payload.get("freq_max", explorer._filter_state.freq_max))
        explorer._controls["fmin"].value = float(
            np.clip(fmin, fmin_control.min, fmin_control.max)
        )
        explorer._controls["fmax"].value = float(
            np.clip(fmax, fmax_control.min, fmax_control.max)
        )

        smooth_filter = str(
            payload.get("smooth_filter", explorer._filter_state.smooth_filter)
        )
        explorer._controls["smooth_filter"].value = smooth_filter

        smooth_window = explorer._controls["smooth_window"]
        explorer._controls["smooth_window"].value = int(
            np.clip(
                int(payload.get("smooth_window", explorer._filter_state.smooth_window)),
                smooth_window.min,
                smooth_window.max,
            )
        )

        smooth_sigma = explorer._controls["smooth_sigma"]
        explorer._controls["smooth_sigma"].value = float(
            np.clip(
                float(payload.get("smooth_sigma", explorer._filter_state.smooth_sigma)),
                smooth_sigma.min,
                smooth_sigma.max,
            )
        )

        baseline_mode = str(
            payload.get("baseline_mode", explorer._filter_state.baseline_mode)
        )
        explorer._controls["baseline_mode"].value = baseline_mode
        clip_low = explorer._controls["clip_low"]
        clip_high = explorer._controls["clip_high"]
        soft_thr = explorer._controls["soft_threshold"]
        explorer._controls["clip_low"].value = float(
            np.clip(
                float(
                    payload.get(
                        "clip_percentile_low",
                        explorer._filter_state.clip_percentile_low,
                    )
                ),
                clip_low.min,
                clip_low.max,
            )
        )
        explorer._controls["clip_high"].value = float(
            np.clip(
                float(
                    payload.get(
                        "clip_percentile_high",
                        explorer._filter_state.clip_percentile_high,
                    )
                ),
                clip_high.min,
                clip_high.max,
            )
        )
        explorer._controls["soft_threshold"].value = float(
            np.clip(
                float(
                    payload.get(
                        "soft_threshold_percentile",
                        explorer._filter_state.soft_threshold_percentile,
                    )
                ),
                soft_thr.min,
                soft_thr.max,
            )
        )

        explorer._controls["normalize"].value = bool(
            payload.get("normalize", explorer._filter_state.normalize)
        )
        explorer._controls["log_scale"].value = bool(
            payload.get("log_scale", explorer._filter_state.log_scale)
        )
        explorer._controls["show_peaks"].value = bool(
            payload.get("show_peaks", explorer._show_peaks)
        )
        peak_prom = explorer._controls["peak_prom"]
        explorer._controls["peak_prom"].value = float(
            np.clip(
                float(payload.get("peak_prominence", explorer._peak_prominence)),
                peak_prom.min,
                peak_prom.max,
            )
        )
        peak_dist = explorer._controls["peak_dist"]
        explorer._controls["peak_dist"].value = int(
            np.clip(
                int(payload.get("peak_distance", explorer._peak_distance)),
                peak_dist.min,
                peak_dist.max,
            )
        )

        mode_view = str(payload.get("mode_view", "all"))
        explorer._controls["mode_view"].value = mode_view

        cmap_mag = str(payload.get("cmap_mag", "viridis"))
        explorer._controls["cmap_mag"].value = cmap_mag

        cmap_phase = str(payload.get("cmap_phase", "twilight"))
        explorer._controls["cmap_phase"].value = cmap_phase

        cmap_combined = str(payload.get("cmap_combined", "RdBu_r"))
        explorer._controls["cmap_combined"].value = cmap_combined

        aspect = str(payload.get("aspect", "equal"))
        explorer._controls["aspect"].value = aspect

        layout = str(payload.get("layout", "auto"))
        explorer._controls["layout"].value = layout
        explorer._freq_unit = freq_unit
    finally:
        explorer._internal_update = False

    explorer._read_controls()
    explorer._recompute_filtered_spectrum()
    explorer._refresh_freq_slider_bounds()
    explorer._render_figure()


def refresh_preset_options(explorer: Any) -> None:
    """Refresh preset dropdown options."""
    if "preset_select" not in explorer._controls:
        return
    options = [("-- load preset --", "")] + [
        (name, name) for name in list_presets(explorer)
    ]
    current = explorer._controls["preset_select"].value
    explorer._controls["preset_select"].options = options
    if current not in [opt[1] for opt in options]:
        explorer._controls["preset_select"].value = ""


def on_save_preset_clicked(explorer: Any, _btn: Any) -> None:
    """Persist current toolbar config as a preset."""
    if not explorer._controls:
        return

    name = str(explorer._controls["preset_name"].value).strip()
    if not name:
        explorer._set_status("Preset name required", color="darkorange")
        return

    safe_name = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_")).strip(
        "_-"
    )
    if not safe_name:
        explorer._set_status("Preset name contains invalid characters", color="crimson")
        return

    payload = collect_preset_state(explorer)
    payload["saved_at"] = datetime.now().isoformat()

    preset_path = get_presets_dir(explorer) / f"fmr_{safe_name}.json"
    try:
        preset_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        explorer._set_status(f"Failed to save preset: {exc}", color="crimson")
        return

    explorer._controls["preset_name"].value = ""
    refresh_preset_options(explorer)
    explorer._controls["preset_select"].value = safe_name
    explorer._set_status(f"Preset saved: {preset_path.name}", color="seagreen")


def on_load_preset_changed(explorer: Any, change: Any) -> None:
    """Load selected preset and apply values to toolbar."""
    if change.get("name") != "value":
        return
    name = str(change.get("new") or "").strip()
    if not name:
        return

    preset_path = get_presets_dir(explorer) / f"fmr_{name}.json"
    if not preset_path.exists():
        explorer._set_status(f"Preset not found: {name}", color="crimson")
        refresh_preset_options(explorer)
        return

    try:
        payload = json.loads(preset_path.read_text(encoding="utf-8"))
    except Exception as exc:
        explorer._set_status(f"Failed to load preset: {exc}", color="crimson")
        return

    if not isinstance(payload, dict):
        explorer._set_status(
            "Failed to load preset: root must be a JSON object", color="crimson"
        )
        return

    previous = collect_preset_state(explorer)
    try:
        apply_preset_state(explorer, payload)
    except Exception as exc:
        try:
            apply_preset_state(explorer, previous)
        except Exception:
            pass
        explorer._set_status(f"Failed to apply preset: {exc}", color="crimson")
        return
    explorer._set_status(f"Preset loaded: {name}", color="seagreen")


def on_delete_preset_clicked(explorer: Any, _btn: Any) -> None:
    """Delete selected preset file."""
    if not explorer._controls:
        return

    name = str(explorer._controls["preset_select"].value or "").strip()
    if not name:
        explorer._set_status("Select preset to delete", color="darkorange")
        return

    preset_path = get_presets_dir(explorer) / f"fmr_{name}.json"
    if not preset_path.exists():
        explorer._set_status(f"Preset not found: {name}", color="crimson")
        refresh_preset_options(explorer)
        return

    try:
        preset_path.unlink()
    except Exception as exc:
        explorer._set_status(f"Failed to delete preset: {exc}", color="crimson")
        return

    refresh_preset_options(explorer)
    explorer._set_status(f"Preset deleted: {name}", color="seagreen")


__all__ = [
    "get_presets_dir",
    "list_presets",
    "collect_preset_state",
    "apply_preset_state",
    "refresh_preset_options",
    "on_save_preset_clicked",
    "on_load_preset_changed",
    "on_delete_preset_clicked",
]
