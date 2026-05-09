"""Canonical module name for nonlinear Thiele analysis.

The older ``nonliniearthiele`` module remains available for compatibility.
New code should import from this module.
"""

from __future__ import annotations

from .nonliniearthiele import ThieleAnalyzer

__all__ = ["ThieleAnalyzer"]
