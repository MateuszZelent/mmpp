"""Dataset-aware namespace for Thiele model adapters."""

from __future__ import annotations

from ...._method_helpers import InteractiveNodeMixin
from .cip import cip
from .cpp import cpp
from .field_resolved_cpp import field_resolved_cpp


class ThieleModelNamespace(InteractiveNodeMixin):
    """Factory namespace for CIP/CPP Thiele adapters."""

    _interactive_owner = "job[0].vortex.model.thiele"
    _interactive_nodes = frozenset({"cpp", "cip", "field_resolved_cpp"})

    def __init__(
        self, *, job_result=None, dataset_name: str | None = None, slice_info=None
    ):
        self._job = job_result
        self._dataset_name = dataset_name
        self._slice_info = slice_info

    def cpp(self, **kwargs):
        """Build dataset-aware CPP model adapter."""
        return cpp(
            job_result=self._job,
            dataset_name=self._dataset_name,
            slice_info=self._slice_info,
            **kwargs,
        )

    def cip(self, **kwargs):
        """Build dataset-aware CIP model adapter."""
        return cip(
            job_result=self._job,
            dataset_name=self._dataset_name,
            slice_info=self._slice_info,
            **kwargs,
        )

    def field_resolved_cpp(self, **kwargs):
        """Build dataset-aware field-resolved CPP model adapter."""
        return field_resolved_cpp(
            job_result=self._job,
            dataset_name=self._dataset_name,
            slice_info=self._slice_info,
            **kwargs,
        )


__all__ = ["ThieleModelNamespace"]
