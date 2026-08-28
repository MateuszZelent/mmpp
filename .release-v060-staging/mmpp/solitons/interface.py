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
    def dataset_name(self) -> str | None:
        """Dataset name used by this soliton interface."""
        if self._dataset_name is None:
            candidate = self._job.get_largest_m_dataset()
            # Verify the dataset actually exists in the zarr
            try:
                self._job._ensure_zarr_loaded()
                if candidate in self._job._z:
                    self._dataset_name = candidate
                # else: leave as None → table-only mode
            except Exception:
                self._dataset_name = candidate  # best-effort fallback
        return self._dataset_name

    @property
    def vortex(self):
        """Vortex analysis namespace."""
        if self._vortex is None:
            from .vortex import VortexInterface

            self._vortex = VortexInterface(
                self._job,
                dataset_name=self._dataset_name,
                mmpp_instance=self._mmpp,
                slice_info=self._slice_info,
            )
        return self._vortex

    def __repr__(self) -> str:
        dataset_label = self._dataset_name if self._dataset_name is not None else "auto"
        return (
            f"SolitonInterface(dataset={dataset_label!r}, slice={self._slice_info!r})"
        )

    def _repr_html_(self) -> str:
        import uuid as _uuid
        from html import escape as _esc

        from mmpp._repr_helpers import (
            _HELPER_SECTION_CHROME,
            accessors_section_html,
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        dataset = str(self._dataset_name if self._dataset_name is not None else "auto")
        slice_label = (
            str(self._slice_info) if self._slice_info is not None else "full"
        )

        # ── context ───────────────────────────────────────────────
        status = metrics_section_html([
            ("dataset", dataset, "#93c5fd"),
            ("slice", slice_label, None),
        ])

        # ── namespaces (table inside a section block) ─────────────
        namespaces = [
            (".vortex", "Vortex dynamics: topology, core, trajectory, spectrum, modes, nonlinear, events"),
        ]
        ns_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(n)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(d)}</td></tr>"
            for n, d in namespaces
        )
        ns_section = (
            f"<div style='{_HELPER_SECTION_CHROME}'>"
            "<b style='color:#94a3b8;'>Namespaces</b><br>"
            f"<table style='width:100%;border-collapse:collapse;font-size:0.9em;margin-top:6px;'>"
            f"{ns_rows}</table></div>"
        )

        # ── workflow (table inside a section block) ───────────────
        workflow = [
            ("1. Topology",  "vortex.topology.detect()",         "polarity, chirality, winding number"),
            ("2. Track core","vortex.core.track()",               "auto/table/Gaussian core tracking"),
            ("3. Trajectory","vortex.trajectory",                  "filtering, orbit fitting, phase"),
            ("4. Spectrum",  "vortex.spectrum.gyration()",         "gyration power spectrum"),
            ("5. Modes",     "vortex.modes.classify()",            "mode classification"),
            ("6. Nonlinear", "vortex.nonlinear",                   "Slavin-Tiberkevich, Thiele"),
            ("7. Events",    "vortex.events",                      "polarity switches, expulsion"),
        ]
        wf_rows = "".join(
            f"<tr><td style='padding:4px 8px;color:#a5b4fc;font-weight:600;'>{_esc(s)}</td>"
            f"<td style='padding:4px 8px;font-family:monospace;color:#93c5fd;font-size:0.85em;'>{_esc(c)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(d)}</td></tr>"
            for s, c, d in workflow
        )
        wf_section = (
            f"<div style='{_HELPER_SECTION_CHROME}'>"
            "<b style='color:#94a3b8;'>Analysis Workflow</b><br>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;margin-top:6px;'>"
            "<thead><tr style='text-align:left;background:rgba(51,65,85,0.6);'>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Step</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Access</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Description</th></tr></thead>"
            f"<tbody>{wf_rows}</tbody></table></div>"
        )

        # ── examples ──────────────────────────────────────────────
        example_code = (
            "# Shortcut: job[0].vortex is alias for job[0].solitons.vortex\n"
            "vortex = job[0].vortex\n"
            "\n"
            "# 1. Detect vortex topology\n"
            "topo = vortex.topology.detect()\n"
            "print(f'p={topo.polarity}, c={topo.chirality}')\n"
            "\n"
            "# 2. Track vortex core\n"
            "traj = vortex.track()  # shortcut for vortex.core.track()\n"
            "traj.plt.trajectory()  # plot x(t), y(t)\n"
            "\n"
            "# 3. Gyration spectrum\n"
            "spec = vortex.spectrum.gyration()\n"
            "spec.plt.power_spectrum()\n"
            "\n"
            "# 4. Mode classification\n"
            "mode = vortex.modes.classify()  # dominant mode\n"
            "\n"
            "# 5. Nonlinear analysis\n"
            "st = vortex.nonlinear.slavin_tiberkevich()\n"
            "vortex.nonlinear.thiele.force_balance()"
        )
        examples = examples_section_html(example_code)

        # ── api card ──────────────────────────────────────────────
        api_card = api_help_html(
            self,
            title="Soliton API help",
            prefix="job[0].solitons",
            properties=[
                ("vortex", "Vortex dynamics analysis namespace"),
                ("dataset_name", "Dataset name used by this soliton interface"),
            ],
            subtitle="Top-level soliton namespace. Use nested accessors for concrete analysis methods.",
            chrome=False,
        )

        return node_card_html(
            "Soliton Analysis Interface 2",
            icon="🌀",
            subtitle="Comprehensive soliton dynamics analysis",
            sections=[status, ns_section, wf_section, examples],
            api=api_card,
            uid=f"solitons-{str(_uuid.uuid4())[:8]}",
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
