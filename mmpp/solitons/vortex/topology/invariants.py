"""Compatibility wrapper for numerical topology invariants."""

from __future__ import annotations

from ..numerical.topology.invariants import (
    chirality_from_ring,
    chirality_ring,
    chirality_ring_with_confidence,
    classify_state,
    polarity,
    polarity_from_core,
    topological_charge,
    winding_number,
    winding_number_from_ring,
)

for _fn in (
    polarity,
    topological_charge,
    winding_number,
    chirality_ring,
    chirality_ring_with_confidence,
    polarity_from_core,
    chirality_from_ring,
    winding_number_from_ring,
    classify_state,
):
    _fn.__module__ = __name__

__all__ = [
    "polarity",
    "topological_charge",
    "winding_number",
    "chirality_ring",
    "chirality_ring_with_confidence",
    "polarity_from_core",
    "chirality_from_ring",
    "winding_number_from_ring",
    "classify_state",
]
