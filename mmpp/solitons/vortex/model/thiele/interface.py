"""Dataset-aware namespace for Thiele model adapters."""

from __future__ import annotations

from .cip import cip
from .cpp import cpp


class ThieleModelNamespace:
    """Factory namespace for CIP/CPP Thiele adapters."""

    def __init__(self, *, job_result=None, dataset_name: str | None = None, slice_info=None):
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


__all__ = ["ThieleModelNamespace"]
