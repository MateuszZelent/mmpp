"""Plot helpers for Thiele model adapters (phase 1 scaffold)."""

from __future__ import annotations


def plot_orbit(result, *, ax=None, **kwargs):
    """Plot trajectory returned by Thiele adapter."""
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(5, 4), dpi=110)
    ax.plot(result.x, result.y, **kwargs)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Thiele trajectory")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    return ax


__all__ = ["plot_orbit"]
