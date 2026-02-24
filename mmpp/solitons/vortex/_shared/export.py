"""Export helpers for shared vortex trajectory contracts (phase 1 scaffold)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def export_trajectory_json(trajectory, path: str | Path) -> Path:
    """Export canonical trajectory arrays to JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "time": np.asarray(trajectory.time, dtype=float).tolist(),
        "x": np.asarray(trajectory.x, dtype=float).tolist(),
        "y": np.asarray(trajectory.y, dtype=float).tolist(),
        "polarity": np.asarray(trajectory.polarity, dtype=int).tolist(),
        "confidence": np.asarray(trajectory.confidence, dtype=float).tolist(),
        "method": str(trajectory.method),
        "metadata": dict(getattr(trajectory, "metadata", {})),
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


__all__ = ["export_trajectory_json"]
