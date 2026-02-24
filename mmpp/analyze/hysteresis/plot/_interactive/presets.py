"""Preset management for hysteresis interactive explorer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def get_presets_dir(explorer: Any) -> Path:
    """Resolve and create presets directory."""
    if getattr(explorer, "_presets_dir", None) is None:
        explorer._presets_dir = Path.cwd() / ".mmpp_presets"
        explorer._presets_dir.mkdir(parents=True, exist_ok=True)
    return explorer._presets_dir


def list_presets(explorer: Any) -> list[str]:
    """List available hysteresis presets."""
    preset_dir = get_presets_dir(explorer)
    names: list[str] = []
    for file_path in sorted(preset_dir.glob("hysteresis_*.json")):
        name = file_path.stem.removeprefix("hysteresis_")
        if name:
            names.append(name)
    return names


def collect_preset_state(explorer: Any) -> dict[str, Any]:
    """Collect serializable explorer state."""
    state = explorer.state
    return {
        "current_idx": int(state.current_idx),
        "snapshot_component": str(state.snapshot_component),
        "z_layer": state.z_layer,
        "roi": list(state.roi) if state.roi is not None else None,
        "show_flags": dict(state.show_flags),
        "debug_clicks": bool(getattr(explorer, "_debug_clicks", False)),
        "loop_panel_weight": float(getattr(state, "loop_panel_weight", 1.15)),
        "snapshot_panel_weight": float(getattr(state, "snapshot_panel_weight", 1.0)),
        "dset": str(getattr(explorer, "_snapshot_dset", "m")),
    }


def apply_preset_state(explorer: Any, payload: dict[str, Any]) -> None:
    """Apply preset payload to explorer state and refresh controls."""
    state = explorer.state
    state.current_idx = int(payload.get("current_idx", state.current_idx))
    state.snapshot_component = str(
        payload.get("snapshot_component", state.snapshot_component)
    )
    state.z_layer = payload.get("z_layer", state.z_layer)

    roi_value = payload.get("roi", state.roi)
    if roi_value is None:
        state.roi = None
    else:
        state.roi = tuple(int(v) for v in roi_value)

    show_flags = payload.get("show_flags")
    if isinstance(show_flags, dict):
        merged = dict(state.show_flags)
        for key, value in show_flags.items():
            merged[str(key)] = bool(value)
        state.show_flags = merged

    if "debug_clicks" in payload:
        explorer._debug_clicks = bool(payload.get("debug_clicks"))
    if "loop_panel_weight" in payload:
        state.loop_panel_weight = float(payload.get("loop_panel_weight"))
    if "snapshot_panel_weight" in payload:
        state.snapshot_panel_weight = float(payload.get("snapshot_panel_weight"))


def save_preset(explorer: Any, name: str) -> Path:
    """Save current state to a named preset file."""
    preset_name = str(name).strip()
    if not preset_name:
        raise ValueError("Preset name must be non-empty")

    path = get_presets_dir(explorer) / f"hysteresis_{preset_name}.json"
    payload = collect_preset_state(explorer)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_preset(explorer: Any, name: str) -> dict[str, Any]:
    """Load preset payload and apply it to explorer state."""
    path = get_presets_dir(explorer) / f"hysteresis_{str(name).strip()}.json"
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    apply_preset_state(explorer, payload)
    return payload
