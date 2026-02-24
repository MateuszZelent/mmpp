"""JSON preset helpers for interactive viewers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_preset(path: str | Path, payload: dict[str, Any]) -> Path:
    """Save preset dictionary as UTF-8 JSON file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def load_preset(path: str | Path) -> dict[str, Any]:
    """Load preset dictionary from JSON file."""
    source = Path(path)
    return json.loads(source.read_text(encoding="utf-8"))
