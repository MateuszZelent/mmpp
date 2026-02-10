"""Top-level interface for soliton analysis namespaces."""

from __future__ import annotations

from typing import Any


class SolitonInterface:
    """Entry point for soliton analysis on a single job."""

    def __init__(
        self,
        job_result,
        mmpp_instance: Any | None = None,
        dataset_name: str | None = None,
        slice_info: Any | None = None,
    ):
        self._job = job_result
        self._mmpp = mmpp_instance
        self._dataset_name = dataset_name
        self._slice_info = slice_info
        self._vortex = None

    @property
    def dataset_name(self) -> str:
        """Dataset name used by this soliton interface."""
        if self._dataset_name is None:
            self._dataset_name = self._job.get_largest_m_dataset()
        return self._dataset_name

    @property
    def vortex(self):
        """Vortex analysis namespace."""
        if self._vortex is None:
            from .vortex import VortexInterface

            self._vortex = VortexInterface(
                self._job,
                dataset_name=self.dataset_name,
                mmpp_instance=self._mmpp,
                slice_info=self._slice_info,
            )
        return self._vortex

    def __repr__(self) -> str:
        return (
            f"SolitonInterface(dataset={self.dataset_name!r}, "
            f"slice={self._slice_info!r})"
        )


class DatasetSpecificSolitons(SolitonInterface):
    """Soliton interface bound to a specific dataset and optional slice."""

    def __init__(
        self,
        job_result,
        dataset_name: str,
        mmpp_instance: Any | None = None,
        slice_info: Any | None = None,
    ):
        super().__init__(
            job_result,
            mmpp_instance=mmpp_instance,
            dataset_name=dataset_name,
            slice_info=slice_info,
        )
