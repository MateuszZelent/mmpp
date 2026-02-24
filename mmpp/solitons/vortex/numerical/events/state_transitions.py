"""Compatibility bridge for G/C state-transition detection."""

from ...events.state_transitions import classify_gc_states, detect_state_switches

__all__ = ["classify_gc_states", "detect_state_switches"]
