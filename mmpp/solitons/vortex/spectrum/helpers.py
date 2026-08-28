"""Notebook-facing callable helpers for vortex spectra."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from ..._method_helpers import InteractiveNodeMixin


class GyrationSpectrumHelper(InteractiveNodeMixin):
    """Callable ``vortex.spectrum.gyration`` notebook accessor."""

    _interactive_owner = "job[0].vortex.spectrum.gyration"
    _interactive_nodes = frozenset({"interactive", "interactive_modes", "mode"})

    def __init__(self, interface: Any, target: Callable[..., Any]):
        self._interface = interface
        self._target = target

    def __call__(self, method: str | None = None, **kwargs: Any):
        """Compute the vortex-core gyration spectrum."""
        return self._target(method=method, **kwargs)

    @property
    def __signature__(self):
        """Expose the compute signature to notebook/API introspection."""
        return inspect.signature(self._target)

    def interactive(self, **kwargs: Any):
        """Open the vortex dashboard directly on its Spectrum module."""
        parent = self._interface._vortex_interface
        if parent is None:
            raise RuntimeError(
                "The Spectrum dashboard requires a parent VortexInterface. "
                "Use job[0].vortex.interactive(initial_module='spectrum')."
            )
        kwargs.setdefault("initial_module", "spectrum")
        return parent.interactive(**kwargs)

    def interactive_modes(self, **kwargs: Any):
        """Open the spatial FFT-mode explorer for the same dataset/slice."""
        return self._fft_modes().interactive_spectrum(**kwargs)

    def mode(self, f: float | None = None, **kwargs: Any):
        """Return the spatial FFT mode nearest ``f`` GHz or the gyration peak."""
        if f is None:
            f = float(self().peak_frequency_ghz)
        return self._fft_modes().mode(f=float(f), **kwargs)

    def _fft_modes(self):
        job = self._interface._job
        dataset_name = self._interface._dataset_name
        data = getattr(job, dataset_name) if dataset_name else job
        slice_info = self._interface._slice_info
        if slice_info is not None and hasattr(data, "__getitem__"):
            data = data[slice_info]
        fft = getattr(data, "fft", None)
        if fft is None:
            raise RuntimeError(
                "Spatial mode visualization requires a magnetisation dataset "
                "with the FFT accessor. Select it explicitly, for example "
                "job[0].m.vortex.spectrum.gyration.interactive_modes()."
            )
        return fft.modes

    def __repr__(self) -> str:
        return (
            "<GyrationSpectrumHelper: call (), .interactive(), "
            ".interactive_modes(), .mode(f=...)>"
        )

    def _repr_html_(self) -> str:
        import uuid

        from mmpp._repr_helpers import (
            NODE_COLOR_ANALYSIS,
            NODE_COLOR_COMPUTE,
            NODE_COLOR_PLOT,
            accessors_section_html,
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        prefix = "job[0].vortex.spectrum.gyration"
        api = api_help_html(
            self,
            title="Vortex gyration spectrum API help",
            prefix=prefix,
            methods=["__call__", "interactive", "interactive_modes", "mode"],
            properties=[],
            subtitle=(
                "Callable spectrum computation plus direct interactive and "
                "spatial-mode visualization entrypoints."
            ),
            chrome=False,
        )
        return node_card_html(
            "Vortex Gyration Spectrum",
            icon="📊",
            subtitle=(
                "Core-orbit spectrum. Call the helper, open its dashboard, "
                "or inspect the corresponding spatial FFT mode."
            ),
            sections=[
                metrics_section_html(
                    [
                        (
                            "dataset",
                            self._interface._dataset_name or "auto",
                            NODE_COLOR_COMPUTE,
                        ),
                        (
                            "default method",
                            self._interface._config.spectrum.method,
                            NODE_COLOR_ANALYSIS,
                        ),
                    ]
                ),
                accessors_section_html(
                    [
                        ("Compute:", [("(...)", NODE_COLOR_COMPUTE)]),
                        (
                            "Interactive:",
                            [
                                (".interactive()", NODE_COLOR_ANALYSIS),
                                (".interactive_modes()", NODE_COLOR_PLOT),
                            ],
                        ),
                        (
                            "Spatial mode:",
                            [(".mode(f=None)", NODE_COLOR_PLOT)],
                        ),
                    ]
                ),
                examples_section_html(
                    "# Compute the 1D core-gyration spectrum\n"
                    "spec = job[0].vortex.spectrum.gyration(method='welch')\n"
                    "spec.plt.power_spectrum()\n\n"
                    "# Interactive vortex Spectrum panel\n"
                    "job[0].vortex.spectrum.gyration.interactive()\n\n"
                    "# Spatial FFT-mode explorer\n"
                    "job[0].vortex.spectrum.gyration.interactive_modes()\n\n"
                    "# Mode at the dominant gyration peak\n"
                    "mode = job[0].vortex.spectrum.gyration.mode()\n"
                    "mode.plot.interactive()",
                    title="Spectrum and Mode Workflows",
                ),
            ],
            api=api,
            uid=f"vortex-gyration-helper-{uuid.uuid4().hex[:8]}",
        )


__all__ = ["GyrationSpectrumHelper"]
