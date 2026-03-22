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
            f"SolitonInterface(dataset={dataset_label!r}, "
            f"slice={self._slice_info!r})"
        )

    def _repr_html_(self) -> str:
        from html import escape as _esc

        dataset = _esc(str(self._dataset_name if self._dataset_name is not None else "auto"))
        slice_label = _esc(str(self._slice_info)) if self._slice_info is not None else "full"

        namespaces = [
            (".vortex", "Vortex dynamics analysis (topology, core, trajectory, spectrum, modes, nonlinear, events)"),
        ]
        ns_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(n)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(d)}</td></tr>"
            for n, d in namespaces
        )
        workflow = [
            ("1. Topology", "data.solitons.vortex.topology.detect()", "Detect polarity, chirality, winding number"),
            ("2. Track core", "data.solitons.vortex.core.track()", "Auto/table/Gaussian core position tracking"),
            ("3. Trajectory", "data.solitons.vortex.trajectory", "Filtering, steady-state, orbit fitting, phase"),
            ("4. Spectrum", "data.solitons.vortex.spectrum.gyration()", "Gyration power spectrum from trajectory"),
            ("5. Modes", "data.solitons.vortex.modes.classify()", "Mode classification (gyration, breathing)"),
            ("6. Nonlinear", "data.solitons.vortex.nonlinear", "Slavin-Tiberkevich, Thiele, amplitude equation"),
            ("7. Events", "data.solitons.vortex.events", "Polarity switches, state transitions, expulsion"),
        ]
        wf_rows = "".join(
            f"<tr><td style='padding:4px 8px;color:#a5b4fc;font-weight:600;'>{_esc(s)}</td>"
            f"<td style='padding:4px 8px;font-family:monospace;color:#93c5fd;font-size:0.85em;'>{_esc(c)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(d)}</td></tr>"
            for s, c, d in workflow
        )
        example = (
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
            "modes = vortex.modes.classify_all()  # all modes\n"
            "\n"
            "# 5. Nonlinear analysis\n"
            "st = vortex.nonlinear.slavin_tiberkevich()\n"
            "vortex.nonlinear.thiele.force_balance()"
        )
        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);\">"
            "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;margin-bottom:4px;'>"
            "🌀 Soliton Analysis Interface</div>"
            "<div style='font-size:0.85em;color:#94a3b8;margin-bottom:10px;'>"
            "Comprehensive soliton dynamics analysis</div>"
            # Context
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='display:flex;flex-wrap:wrap;gap:12px;font-size:0.9em;'>"
            f"<div><span style='color:#94a3b8;'>Dataset:</span> "
            f"<code style='color:#cbd5e1;'>{dataset}</code></div>"
            f"<div><span style='color:#94a3b8;'>Slice:</span> "
            f"<code style='color:#cbd5e1;'>{slice_label}</code></div>"
            "</div></div>"
            # Namespaces
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Namespaces</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            f"{ns_rows}</table></div>"
            # Workflow
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>"
            "Analysis Workflow</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            "<thead><tr style='text-align:left;background:rgba(51,65,85,0.6);'>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Step</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Access</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Description</th></tr></thead>"
            f"<tbody>{wf_rows}</tbody></table></div>"
            # Examples
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Examples</div>"
            "<pre style='margin:0;background:rgba(15,23,42,0.85);padding:10px;"
            "border-radius:6px;color:#e2e8f0;overflow-x:auto;font-size:0.85em;'>"
            f"<code>{example}</code></pre></div>"
            "</div>"
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
