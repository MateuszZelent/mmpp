"""Preset management for interactive dispersion explorer."""

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
    """List available dispersion presets."""
    preset_dir = get_presets_dir(explorer)
    names: list[str] = []
    for file_path in sorted(preset_dir.glob("dispersion_*.json")):
        name = file_path.stem.removeprefix("dispersion_")
        if name:
            names.append(name)
    return names


def collect_preset_state(explorer: Any) -> dict[str, Any]:
    """Collect serializable explorer state."""
    state = explorer.state
    return {
        "fmin_ghz": float(state.fmin_ghz),
        "fmax_ghz": float(state.fmax_ghz),
        "source": str(state.source),
        "kscale": str(state.kscale),
        "cmap": str(state.cmap),
        "positive_frequencies": bool(state.positive_frequencies),
        "lognorm": bool(state.lognorm),
        "selected_k": state.selected_k,
        "selected_f": state.selected_f,
        "mode_type": str(getattr(state, "mode_type", "abs") or "abs"),
        "show_flags": dict(state.show_flags or {}),
        "analytical": dict(state.analytical or {}),
        "live_filters": dict(state.live_filters or {}),
    }


def apply_preset_state(explorer: Any, payload: dict[str, Any]) -> None:
    """Apply preset payload to explorer state."""
    state = explorer.state
    state.fmin_ghz = float(payload.get("fmin_ghz", state.fmin_ghz))
    state.fmax_ghz = float(payload.get("fmax_ghz", state.fmax_ghz))
    state.source = str(payload.get("source", state.source))
    state.kscale = str(payload.get("kscale", state.kscale))
    state.cmap = str(payload.get("cmap", state.cmap))
    state.positive_frequencies = bool(
        payload.get("positive_frequencies", state.positive_frequencies)
    )
    state.lognorm = bool(payload.get("lognorm", state.lognorm))
    state.selected_k = payload.get("selected_k", state.selected_k)
    state.selected_f = payload.get("selected_f", state.selected_f)
    state.mode_type = str(payload.get("mode_type", getattr(state, "mode_type", "abs")))
    show_flags = payload.get("show_flags")
    if isinstance(show_flags, dict):
        merged = dict(state.show_flags or {})
        for key, value in show_flags.items():
            merged[str(key)] = bool(value)
        state.show_flags = merged
    analytical = payload.get("analytical")
    if isinstance(analytical, dict):
        merged_analytical = dict(state.analytical or {})
        merged_analytical.update(analytical)
        state.analytical = merged_analytical
    live_filters = payload.get("live_filters")
    if isinstance(live_filters, dict):
        state.live_filters = dict(live_filters) or None


def save_preset(explorer: Any, name: str) -> Path:
    """Save current state to a named preset file."""
    preset_name = str(name).strip()
    if not preset_name:
        raise ValueError("Preset name must be non-empty")
    path = get_presets_dir(explorer) / f"dispersion_{preset_name}.json"
    path.write_text(
        json.dumps(collect_preset_state(explorer), indent=2), encoding="utf-8"
    )
    return path


def load_preset(explorer: Any, name: str) -> dict[str, Any]:
    """Load preset payload and apply it to explorer state."""
    path = get_presets_dir(explorer) / f"dispersion_{str(name).strip()}.json"
    if not path.exists():
        raise FileNotFoundError(f"Preset not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    apply_preset_state(explorer, payload)
    return payload
