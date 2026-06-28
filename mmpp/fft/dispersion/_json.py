"""Internal JSON normalization helpers for dispersion notebook controllers."""

from __future__ import annotations

from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Convert common NumPy values to JSON-safe Python objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
