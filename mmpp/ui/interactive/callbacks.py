"""Tiny callback helpers for shared interactive viewers."""

from __future__ import annotations

from .state import InteractiveState


def call_with_state(state: InteractiveState, callback, **kwargs):
    """Execute callback with state + keyword arguments."""
    if callback is None:
        return None
    return callback(state=state, **kwargs)
