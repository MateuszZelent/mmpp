"""Comparison glue for trajectory sources."""

from __future__ import annotations


def compare_trajectories(lhs, rhs, *, label=("lhs", "rhs")):
    """Return canonical trajectory comparison object."""
    return lhs.compare.with_(rhs, label=label)


__all__ = ["compare_trajectories"]
