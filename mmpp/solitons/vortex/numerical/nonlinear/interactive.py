"""Compatibility bridge for nonlinear interactive dashboard helpers."""

from ...nonlinear.interactive import (
    ThieleInteractiveDashboard,
    build_thiele_dashboard,
    proxy_psd,
    proxy_signal_from_trajectory,
)

__all__ = [
    "ThieleInteractiveDashboard",
    "build_thiele_dashboard",
    "proxy_signal_from_trajectory",
    "proxy_psd",
]
