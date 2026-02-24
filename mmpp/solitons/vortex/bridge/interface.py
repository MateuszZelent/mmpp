"""Bridge interface scaffolding for numerical <-> analytical workflows."""

from __future__ import annotations

from mmpp._shared.repr_html import make_simple_card

from .compare import compare_trajectories
from .fit import fit_thiele_from_trajectory


class _BridgeCompareAccessor:
    def with_(self, lhs, rhs, *, label=("numerical", "analytical")):
        return compare_trajectories(lhs, rhs, label=label)


class _BridgeFitAccessor:
    def thiele_from_trajectory(self, trajectory, **kwargs):
        return fit_thiele_from_trajectory(trajectory, **kwargs)


class BridgeInterface:
    """Vortex bridge namespace with compare/fit sub-accessors."""

    def __init__(self):
        self.compare = _BridgeCompareAccessor()
        self.fit = _BridgeFitAccessor()

    def _repr_html_(self) -> str:
        rows = [
            (".compare.with_(lhs, rhs)", "Overlay/metric comparison of trajectories"),
            (".fit.thiele_from_trajectory(traj)", "Fit Thiele-like proxy from trajectory"),
        ]
        return make_simple_card(
            title="Vortex Bridge Interface",
            subtitle="Numerical <-> analytical glue utilities",
            rows=rows,
        )


__all__ = ["BridgeInterface"]
