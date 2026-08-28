"""Run a local release-smoke gate for FFT dispersion.

The gate is intentionally small and synthetic. It verifies that the public
dispersion imports, headless interactive controller, dataset-first docs examples,
explicit time-window examples, mode-reconstruction policy, and benchmark path
work from the current checkout. Release workflows should run this after
installing the built wheel as an additional packaging proof. JSON reports include
``docs_example_summary`` for a compact overview of the full-dataset,
sliced-dataset, ``tmin``/``tmax``, mode fallback, and legacy adapter paths.
Top-level ``masterplan_contracts``, ``masterplan_failures``, and
``masterplan_failure_details`` summarize the interactive-dispersion plan status.
``summary`` gives the shortest status view, and ``recommended_next_steps`` maps
failed contract groups to concise repair hints. Use ``--summary-only`` to print
only the compact status payload while still writing the full report to
``--output``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
WIDGET_MODULE_ROOTS = {"IPython", "ipywidgets", "matplotlib"}


def _path_points_to_repo_root(path_entry: str) -> bool:
    if path_entry == "":
        return True
    try:
        return Path(path_entry).resolve() == REPO_ROOT
    except (OSError, RuntimeError):
        return path_entry == str(REPO_ROOT)


def _prepare_import_path(import_mode: str) -> dict[str, str]:
    """Prepare imports for checkout development or installed-package smoke tests."""
    if import_mode not in {"checkout", "installed"}:
        raise ValueError("import_mode must be 'checkout' or 'installed'")

    repo_root = str(REPO_ROOT)
    if import_mode == "checkout":
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
    else:
        sys.path[:] = [
            entry for entry in sys.path if not _path_points_to_repo_root(entry)
        ]
        for module_name in list(sys.modules):
            if module_name == "mmpp" or module_name.startswith("mmpp."):
                del sys.modules[module_name]

    return {"import_mode": import_mode, "repo_root": repo_root}


def _load_benchmark_runner() -> Callable[..., dict[str, Any]]:
    benchmark_path = REPO_ROOT / "scripts" / "analysis" / "benchmark_fft_dispersion.py"
    spec = importlib.util.spec_from_file_location(
        "_mmpp_fft_dispersion_benchmark",
        benchmark_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load benchmark helper from {benchmark_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_benchmark


def _loaded_widget_modules() -> set[str]:
    return {
        name
        for name in sys.modules
        if name.split(".", maxsplit=1)[0] in WIDGET_MODULE_ROOTS
    }


def _headless_import_report(before: set[str]) -> dict[str, Any]:
    new_modules = sorted(_loaded_widget_modules() - before)
    return {
        "no_new_widget_modules": not new_modules,
        "new_widget_modules": new_modules,
    }


def _make_headless_viewer_state() -> dict[str, Any]:
    from mmpp.fft.dispersion._interactive_viewer import DispersionInteractiveViewer
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    result = DispersionResult1D(
        S=np.ones((4, 5), dtype=np.float32),
        k_axis=np.linspace(-1.0, 1.0, 4),
        f_axis=np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
        axis="x",
        component="perp",
        config=DispersionConfig(dt=1.0, dx=1.0),
        dt=1.0,
        dx=1.0,
    )
    viewer = DispersionInteractiveViewer.from_result(result, show=False)
    exported = viewer.export_selection(
        k_rad_per_m=np.float64(1.25e6),
        f_hz=0.0,
        frame=np.int64(2),
        marker=np.array([1.0, 2.0], dtype=np.float32),
        source="release_gate",
    )
    with tempfile.TemporaryDirectory(prefix="mmpp-dispersion-viewer-") as tmp:
        preset_path = Path(tmp) / "viewer-preset.json"
        viewer.save_preset(preset_path)
        reloaded = DispersionInteractiveViewer.from_result(
            result,
            show=False,
            components=["temporary"],
            fmax=1.0,
        )
        reloaded.load_preset(preset_path)
    state = viewer.state
    return {
        **state,
        "positive_frequencies": bool(
            state.get("options", {}).get("positive_frequencies")
        ),
        "preset_roundtrip": reloaded.state == state,
        "export_selection": exported["selection"],
    }


def _run_viewer_display_lifecycle_smoke() -> dict[str, Any]:
    """Exercise explicit show/close outside the headless import measurement."""
    from mmpp.fft.dispersion._interactive_viewer import DispersionInteractiveViewer
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    result = DispersionResult1D(
        S=np.ones((5, 4), dtype=np.float32),
        k_axis=np.array([-600e6, -5e6, 0.0, 5e6, 600e6]),
        f_axis=np.array([0.0, 5e9, 10e9, 500e9]),
        axis="x",
        component="perp",
        config=DispersionConfig(dt=1.0, dx=1.0),
        dt=1.0,
        dx=1.0,
    )
    lifecycle = DispersionInteractiveViewer.from_result(
        result,
        show=False,
        fmax=8,
        f_units="GHz",
        kscale="rad_um",
        live_filters={
            "percentile_autoscale": {
                "enabled": True,
                "low_percentile": 0.0,
                "high_percentile": 100.0,
            }
        },
    )
    lifecycle.show()
    shown = lifecycle.show_requested is True
    figure_after_show = getattr(lifecycle, "_figure", None) is not None
    axes_after_show = getattr(lifecycle, "_axes", None) is not None
    widget_status_after_show = getattr(lifecycle, "_widget_status", "")
    state_after_show = lifecycle.state
    rendered_xlim = None
    rendered_ylim = None
    axes = getattr(lifecycle, "_axes", None)
    if axes is not None:
        try:
            rendered_xlim = [float(value) for value in axes.get_xlim()]
            rendered_ylim = [float(value) for value in axes.get_ylim()]
        except Exception:
            rendered_xlim = None
            rendered_ylim = None
    lifecycle.close()
    return {
        "shown": shown,
        "closed": lifecycle.show_requested is False
        and lifecycle._display_handle is None,
        "figure_after_show": figure_after_show,
        "axes_after_show": axes_after_show,
        "widget_status_after_show": widget_status_after_show,
        "live_filters_after_show": state_after_show.get("live_filters"),
        "live_filter_error": (
            getattr(
                getattr(lifecycle, "_widget_engine", None), "_last_filter_error", ""
            )
        ),
        "rendered_xlim": rendered_xlim,
        "rendered_ylim": rendered_ylim,
    }


def _viewer_status(viewer_state: dict[str, Any]) -> dict[str, Any]:
    selection = viewer_state.get("export_selection", {})
    lifecycle = viewer_state.get("display_lifecycle") or {}
    widget_ready = lifecycle.get("widget_status_after_show") == "ready"
    rendered_xlim = lifecycle.get("rendered_xlim") or []
    rendered_ylim = lifecycle.get("rendered_ylim") or []
    live_filters = lifecycle.get("live_filters_after_show") or {}
    checks = {
        "viewer_headless": viewer_state.get("show") is False,
        "positive_frequencies": viewer_state.get("positive_frequencies") is True,
        "preset_roundtrip": viewer_state.get("preset_roundtrip") is True,
        "display_lifecycle": lifecycle.get("shown") is True
        and lifecycle.get("closed") is True,
        "auto_show_initial_figure": (not widget_ready)
        or (
            lifecycle.get("figure_after_show") is True
            and lifecycle.get("axes_after_show") is True
        ),
        "auto_show_default_k_xlim": (not widget_ready)
        or (
            len(rendered_xlim) == 2
            and rendered_xlim[0] <= -9.99
            and rendered_xlim[1] >= 9.99
            and rendered_xlim[0] >= -10.01
            and rendered_xlim[1] <= 10.01
        ),
        "auto_show_fmax_axis": (not widget_ready)
        or (
            len(rendered_ylim) == 2
            and min(rendered_ylim) >= -1e-9
            and max(rendered_ylim) <= 8.0
        ),
        "live_filters_state": (not widget_ready)
        or (
            isinstance(live_filters, dict)
            and "percentile_autoscale" in live_filters
            and not lifecycle.get("live_filter_error")
        ),
        "export_selection": selection.get("source") == "release_gate"
        and selection.get("marker") == [1.0, 2.0],
    }
    failures = sorted(name for name, ok in checks.items() if not ok)
    return {
        "status": "failed" if failures else "ok",
        "failures": failures,
    }


def _make_headless_mode_viewer_state() -> dict[str, Any]:
    """Exercise mode-specific headless viewers without notebook dependencies."""
    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D

    n_k, n_f = 8, 6
    k_axis = np.linspace(-4.0e6, 3.0e6, n_k)
    f_axis = np.arange(n_f, dtype=float) * 0.5e9
    idx_k, idx_f = n_k // 2 + 1, 2
    S_complex = np.zeros((n_k, n_f), dtype=np.complex128)
    S_complex[idx_k, idx_f] = 1.0 + 0.5j
    result = DispersionResult1D(
        S=np.abs(S_complex).astype(np.float32),
        S_raw=np.abs(S_complex).astype(np.float32),
        S_display=np.abs(S_complex).astype(np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=1.0, dx=5e-9),
        dt=1.0,
        dx=5e-9,
        S_complex=S_complex,
    )

    mode = result.modes.at(
        k_rad_um=np.float64(k_axis[idx_k] / 1e6),
        f_ghz=np.float64(f_axis[idx_f] / 1e9),
    )
    single_viewer = mode.plot.interactive(
        show=False,
        mode_type="phase",
        alpha=np.float32(0.5),
    )
    single_export = single_viewer.export_selection(
        source="release_gate",
        frame=np.int64(3),
        marker=np.array([3.0, 4.0], dtype=np.float32),
    )

    animation_viewer = result.modes.plot.animation(
        peaks=[np.int64(0), np.int64(2)],
        show=False,
        fps=np.int64(12),
    )
    animation_export = animation_viewer.export_selection(
        source="release_gate",
        frame=np.int64(4),
        marker=np.array([5.0, 6.0], dtype=np.float32),
    )

    return {
        "single_mode": {
            **single_viewer.state,
            "export_selection": single_export["selection"],
        },
        "animation": {
            **animation_viewer.state,
            "export_selection": animation_export["selection"],
        },
    }


def _mode_viewers_status(mode_viewers: dict[str, Any]) -> dict[str, Any]:
    single_mode = mode_viewers.get("single_mode", {})
    animation = mode_viewers.get("animation", {})
    single_selection = single_mode.get("export_selection", {})
    animation_selection = animation.get("export_selection", {})
    checks = {
        "single_mode_headless": single_mode.get("show") is False,
        "single_mode_phase": single_mode.get("mode_type") == "phase",
        "single_mode_export": single_selection.get("source") == "release_gate"
        and single_selection.get("marker") == [3.0, 4.0],
        "animation_headless": animation.get("show") is False,
        "animation_peaks": animation.get("peaks") == [0, 2],
        "animation_export": animation_selection.get("source") == "release_gate"
        and animation_selection.get("marker") == [5.0, 6.0],
    }
    failures = sorted(name for name, ok in checks.items() if not ok)
    return {
        "status": "failed" if failures else "ok",
        "failures": failures,
    }


def _write_docs_smoke_zarr(path: Path) -> None:
    import zarr

    n_t, n_x = 8, 8
    t = np.arange(n_t, dtype=float)[:, None]
    x = np.arange(n_x, dtype=float)[None, :]
    wave = 0.2 * np.exp(2j * np.pi * (t / n_t + x / n_x))
    data = np.zeros((n_t, 1, 1, n_x, 3), dtype=np.float32)
    data[:, 0, 0, :, 0] = wave.real
    data[:, 0, 0, :, 1] = wave.imag

    root = zarr.open(str(path), mode="w")
    root.create_dataset("m", data=data, chunks=data.shape)
    root.attrs["t_sampl"] = 1e-12
    root.attrs["dx"] = 5e-9
    root.attrs["dy"] = 5e-9


def _run_docs_example_smoke() -> dict[str, Any]:
    """Execute the public dispersion docs pattern on a synthetic dataset."""
    import mmpp
    from mmpp.fft.dispersion.interface import FFTDispersionInterface

    with tempfile.TemporaryDirectory(prefix="mmpp-dispersion-docs-smoke-") as tmp:
        zarr_path = Path(tmp) / "docs-smoke.zarr"
        _write_docs_smoke_zarr(zarr_path)
        db = mmpp.open(tmp, force=True, max_workers=1)
        full_disp = db[0].m.fft.dispersion
        full_viewer = full_disp.plot.interactive(
            axis="x",
            component="perp",
            fmax=25,
            show=False,
            disk_cache=False,
            progress=False,
        )
        auto_modes_viewer = full_disp.plot.interactive(
            axis="x",
            component="perp",
            fmax=25,
            modes=True,
            show=False,
            disk_cache=False,
            progress=False,
        )
        legacy_alias_viewer = full_disp.interactive_analysis(
            axis="x",
            component="perp",
            fmax=25,
            show=False,
            disk_cache=False,
            progress=False,
        )
        dataset_disp = db[0].m[:4, ...].fft.dispersion
        dataset_progress: list[dict[str, Any]] = []
        dataset_viewer = dataset_disp.plot.interactive(
            axis="x",
            component="perp",
            fmax=25,
            show=False,
            disk_cache=False,
            progress=False,
            progress_callback=dataset_progress.append,
        )

        disp = FFTDispersionInterface(
            SimpleNamespace(
                job_result=SimpleNamespace(path=zarr_path, name="docs-smoke")
            )
        )

        compute_viewer = disp.plot.interactive(
            axis="x",
            component="perp",
            tmin=2,
            tmax=6,
            fmax=25,
            show=False,
            disk_cache=False,
        )
        result = disp.compute_1d(
            axis="x",
            component="perp",
            store_complex=False,
            scaling="amplitude_squared",
            disk_cache=False,
        )
        result_viewer = result.plot.interactive(show=False, fmax=25)
        exported = result_viewer.export_selection(
            k_rad_per_m=0.0,
            f_hz=0.0,
            source="docs_example",
        )
        mode_result = disp.compute_1d(
            axis="x",
            component="perp",
            avg_over_orthogonal=False,
            orthogonal_avg_mode="fft_power",
            store_complex=True,
            scaling="amplitude_squared",
            disk_cache=False,
        )
        modes_viewer = mode_result.modes.interactive(
            show=False,
            lattice_constant_nm=470,
        )
        positive_f = mode_result.f_axis[mode_result.f_axis >= 0]
        target_f_hz = float(positive_f[min(1, positive_f.size - 1)])
        target_k_rad_m = float(mode_result.k_axis[len(mode_result.k_axis) // 2])
        mode = mode_result.modes.at(
            k_rad_um=target_k_rad_m / 1e6,
            f_ghz=target_f_hz / 1e9,
        )
        mode_viewer = mode.plot.interactive(show=False, mode_type="abs")
        animation_viewer = mode_result.modes.plot.animation(peaks=[0], show=False)
        legacy_modes = disp.dispersion_modes(
            result=mode_result,
            lattice_constant_nm=470,
        )
        legacy_modes_export = legacy_modes.export_selection(
            k_rad_um=target_k_rad_m / 1e6,
            f_ghz=target_f_hz / 1e9,
            source="legacy_modes",
        )

    return {
        "full_dataset_viewer_show": bool(full_viewer.state["show"]),
        "full_dataset_shape": list(full_viewer.result.shape),
        "full_dataset_time_window": [
            getattr(full_viewer.result._interface, "_tmin", None),
            getattr(full_viewer.result._interface, "_tmax", None),
        ],
        "auto_modes_viewer_show": bool(auto_modes_viewer.state["show"]),
        "auto_modes_has_complex": auto_modes_viewer.result.S_complex is not None,
        "auto_modes_can_reconstruct": bool(auto_modes_viewer.can_reconstruct_modes),
        "auto_modes_unavailable_reason": auto_modes_viewer.mode_unavailable_reason,
        "legacy_alias_viewer_show": bool(legacy_alias_viewer.state["show"]),
        "legacy_alias_same_type": legacy_alias_viewer.__class__.__name__
        == full_viewer.__class__.__name__,
        "legacy_alias_shape": list(legacy_alias_viewer.result.shape),
        "dataset_first_viewer_show": bool(dataset_viewer.state["show"]),
        "dataset_first_dataset": dataset_disp.dataset_name,
        "dataset_first_slice": str(dataset_disp.slice_info),
        "dataset_first_shape": list(dataset_viewer.result.shape),
        "dataset_first_progress_stages": [
            str(event.get("stage")) for event in dataset_progress
        ],
        "compute_viewer_show": bool(compute_viewer.state["show"]),
        "compute_viewer_shape": list(compute_viewer.result.shape),
        "compute_viewer_time_window": [
            getattr(compute_viewer.result._interface, "_tmin", None),
            getattr(compute_viewer.result._interface, "_tmax", None),
        ],
        "result_viewer_show": bool(result_viewer.state["show"]),
        "positive_frequencies": bool(
            result_viewer.state["options"].get("positive_frequencies")
        ),
        "result_shape": list(result.shape),
        "scaling": result.scaling,
        "s_complex_is_none": result.S_complex is None,
        "result_notes": result_viewer.state["result_notes"],
        "result_viewer_can_reconstruct_modes": bool(
            result_viewer.can_reconstruct_modes
        ),
        "result_viewer_unavailable_reason": result_viewer.mode_unavailable_reason,
        "export_source": exported["selection"]["source"],
        "export_mode_request_reason": exported.get("mode_request", {}).get("reason"),
        "mode_result_has_complex": mode_result.S_complex is not None,
        "modes_viewer_can_reconstruct": bool(modes_viewer.can_reconstruct_modes),
        "mode_viewer_show": bool(mode_viewer.state["show"]),
        "mode_viewer_type": mode_viewer.state["mode_type"],
        "animation_peaks": animation_viewer.state["peaks"],
        "legacy_modes_has_interface": getattr(legacy_modes.result, "_interface", None)
        is disp,
        "legacy_modes_lattice_nm": legacy_modes.state["default_params"].get(
            "lattice_nm"
        ),
        "legacy_modes_can_reconstruct": bool(
            legacy_modes.state["can_reconstruct_modes"]
        ),
        "legacy_modes_export_source": legacy_modes_export["selection"]["source"],
        "legacy_modes_request_available": bool(
            legacy_modes_export["mode_request"]["available"]
        ),
    }


def _docs_example_status(docs_example: dict[str, Any]) -> dict[str, Any]:
    progress_stages = set(docs_example.get("dataset_first_progress_stages") or [])
    checks = {
        "full_dataset_headless": docs_example.get("full_dataset_viewer_show") is False,
        "full_dataset_uses_all_timesteps": (
            docs_example.get("full_dataset_shape") or [None, None]
        )[-1]
        == 8,
        "full_dataset_no_time_window": docs_example.get("full_dataset_time_window")
        == [None, None],
        "auto_modes_headless": docs_example.get("auto_modes_viewer_show") is False,
        "auto_modes_store_complex": docs_example.get("auto_modes_has_complex") is True,
        "auto_modes_can_reconstruct": docs_example.get("auto_modes_can_reconstruct")
        is True,
        "legacy_alias_headless": docs_example.get("legacy_alias_viewer_show") is False,
        "legacy_alias_same_type": docs_example.get("legacy_alias_same_type") is True,
        "legacy_alias_uses_all_timesteps": (
            docs_example.get("legacy_alias_shape") or [None, None]
        )[-1]
        == 8,
        "dataset_first_headless": docs_example.get("dataset_first_viewer_show")
        is False,
        "dataset_first_uses_m": docs_example.get("dataset_first_dataset") == "m",
        "dataset_first_slice": (
            "slice(None, 4" in str(docs_example.get("dataset_first_slice"))
            or "slice(0, 4" in str(docs_example.get("dataset_first_slice"))
        ),
        "dataset_first_shape": (
            docs_example.get("dataset_first_shape") or [None, None]
        )[-1]
        == 4,
        "dataset_first_progress": {
            "prepare",
            "compute",
            "cache",
            "result",
            "viewer",
            "done",
        }.issubset(progress_stages),
        "compute_viewer_headless": docs_example.get("compute_viewer_show") is False,
        "compute_viewer_time_window": docs_example.get("compute_viewer_time_window")
        == [2, 6],
        "compute_viewer_window_shape": (
            docs_example.get("compute_viewer_shape") or [None, None]
        )[-1]
        == 4,
        "result_viewer_headless": docs_example.get("result_viewer_show") is False,
        "positive_frequencies": docs_example.get("positive_frequencies") is True,
        "amplitude_squared_scaling": docs_example.get("scaling") == "amplitude_squared",
        "preview_without_complex": docs_example.get("s_complex_is_none") is True,
        "preview_modes_unavailable": docs_example.get(
            "result_viewer_can_reconstruct_modes"
        )
        is False,
        "preview_fallback_mentions_store_complex": "store_complex=True"
        in str(docs_example.get("result_viewer_unavailable_reason"))
        or "S_complex" in str(docs_example.get("export_mode_request_reason")),
        "mode_result_has_complex": docs_example.get("mode_result_has_complex") is True,
        "modes_viewer_can_reconstruct": docs_example.get("modes_viewer_can_reconstruct")
        is True,
        "mode_viewer_headless": docs_example.get("mode_viewer_show") is False,
        "mode_viewer_abs": docs_example.get("mode_viewer_type") == "abs",
        "animation_peaks": docs_example.get("animation_peaks") == [0],
        "legacy_modes_has_interface": docs_example.get("legacy_modes_has_interface")
        is True,
        "legacy_modes_lattice": docs_example.get("legacy_modes_lattice_nm") == 470,
        "legacy_modes_can_reconstruct": docs_example.get("legacy_modes_can_reconstruct")
        is True,
        "legacy_modes_export": docs_example.get("legacy_modes_export_source")
        == "legacy_modes"
        and docs_example.get("legacy_modes_request_available") is True,
    }
    failures = sorted(name for name, ok in checks.items() if not ok)
    return {
        "status": "failed" if failures else "ok",
        "failures": failures,
        "observed": {
            "full_dataset_shape": docs_example.get("full_dataset_shape"),
            "full_dataset_time_window": docs_example.get("full_dataset_time_window"),
            "auto_modes_has_complex": docs_example.get("auto_modes_has_complex"),
            "auto_modes_can_reconstruct": docs_example.get(
                "auto_modes_can_reconstruct"
            ),
            "legacy_alias_same_type": docs_example.get("legacy_alias_same_type"),
            "legacy_alias_shape": docs_example.get("legacy_alias_shape"),
            "dataset_first_shape": docs_example.get("dataset_first_shape"),
            "dataset_first_slice": docs_example.get("dataset_first_slice"),
            "compute_viewer_shape": docs_example.get("compute_viewer_shape"),
            "compute_viewer_time_window": docs_example.get(
                "compute_viewer_time_window"
            ),
            "fallback_mode_reason": docs_example.get(
                "result_viewer_unavailable_reason"
            ),
            "legacy_modes_lattice_nm": docs_example.get("legacy_modes_lattice_nm"),
            "legacy_modes_can_reconstruct": docs_example.get(
                "legacy_modes_can_reconstruct"
            ),
            "legacy_modes_request_available": docs_example.get(
                "legacy_modes_request_available"
            ),
            "progress_stages": sorted(progress_stages),
        },
    }


def _docs_example_summary(docs_example: dict[str, Any]) -> dict[str, Any]:
    """Return a compact human-readable summary of the public docs smoke paths."""

    def _frequency_bins(key: str) -> Any:
        shape = docs_example.get(key) or []
        if not shape:
            return None
        return shape[-1]

    return {
        "full_dataset": {
            "frequency_bins": _frequency_bins("full_dataset_shape"),
            "time_window": docs_example.get("full_dataset_time_window"),
        },
        "dataset_slice": {
            "dataset": docs_example.get("dataset_first_dataset"),
            "slice": docs_example.get("dataset_first_slice"),
            "frequency_bins": _frequency_bins("dataset_first_shape"),
        },
        "explicit_time_window": {
            "frequency_bins": _frequency_bins("compute_viewer_shape"),
            "time_window": docs_example.get("compute_viewer_time_window"),
        },
        "mode_policy": {
            "modes_true_has_complex": docs_example.get("auto_modes_has_complex"),
            "modes_true_can_reconstruct": docs_example.get(
                "auto_modes_can_reconstruct"
            ),
            "fallback_without_complex": docs_example.get(
                "result_viewer_unavailable_reason"
            ),
        },
        "legacy_adapters": {
            "interactive_analysis_same_type": docs_example.get(
                "legacy_alias_same_type"
            ),
            "interactive_analysis_frequency_bins": _frequency_bins(
                "legacy_alias_shape"
            ),
            "dispersion_modes_has_interface": docs_example.get(
                "legacy_modes_has_interface"
            ),
            "dispersion_modes_can_reconstruct": docs_example.get(
                "legacy_modes_can_reconstruct"
            ),
            "dispersion_modes_request_available": docs_example.get(
                "legacy_modes_request_available"
            ),
        },
    }


def _run_widget_smoke(*, require: bool = False) -> dict[str, Any]:
    """Run the real ipywidgets/Matplotlib mode widget path when deps exist."""
    missing: list[str] = []
    try:
        import ipywidgets  # noqa: F401
    except ImportError:
        missing.append("ipywidgets")
    try:
        import IPython  # noqa: F401
    except ImportError:
        missing.append("IPython")
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        missing.append("matplotlib")
        plt = None

    if missing:
        return {
            "status": "failed" if require else "skipped",
            "missing": missing,
            "reason": "optional notebook/widget dependencies are unavailable",
            "required": bool(require),
        }

    from mmpp.fft.dispersion.models import DispersionConfig, DispersionResult1D
    from mmpp.fft.dispersion.modes.interactive import InteractiveDispersionModes

    n_k, n_f = 8, 6
    k_axis = np.linspace(-4.0e6, 3.0e6, n_k)
    f_axis = np.arange(n_f, dtype=float) * 0.5e9
    S_complex = np.zeros((n_k, n_f), dtype=np.complex128)
    S_complex[n_k // 2 + 1, 2] = 1.0 + 0.0j
    result = DispersionResult1D(
        S=np.abs(S_complex).astype(np.float32),
        S_raw=np.abs(S_complex).astype(np.float32),
        S_display=np.abs(S_complex).astype(np.float32),
        k_axis=k_axis,
        f_axis=f_axis,
        axis="x",
        component="perp",
        config=DispersionConfig(dt=1.0, dx=5e-9),
        dt=1.0,
        dx=5e-9,
        S_complex=S_complex,
    )
    modes = InteractiveDispersionModes(SimpleNamespace(compute_1d=lambda **_: result))

    try:
        modes.plot_interactive(
            result=result,
            figsize=(4.0, 3.0),
            dpi=72,
            lattice_constant_nm=470.0,
            fmax=2.0,
            f_units="GHz",
            lognorm=False,
        )
        fig = getattr(modes, "_fig", None)
        has_widgets = all(
            hasattr(modes, name)
            for name in ("w_lattice", "w_n_bz_mask", "w_fmax", "w_mode_type")
        )
        report = {
            "status": "ok",
            "missing": [],
            "required": bool(require),
            "has_figure": fig is not None,
            "has_axes": getattr(modes, "_ax_disp", None) is not None
            and getattr(modes, "_ax_mode", None) is not None,
            "has_widgets": has_widgets,
            "display_handle_present": getattr(modes, "_display_handle", None)
            is not None,
        }
        modes.close()
        report["close_cleared_display"] = (
            getattr(modes, "_display_handle", None) is None
        )
        report["close_cleared_figure"] = getattr(modes, "_fig", None) is None
    except Exception as exc:  # pragma: no cover - depends on optional widget stack
        report = {
            "status": "failed",
            "missing": [],
            "required": bool(require),
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if plt is not None and getattr(modes, "_fig", None) is not None:
            modes.close()

    return report


def _recommended_masterplan_next_steps(masterplan_failures: list[str]) -> list[str]:
    """Return concise repair hints for failed masterplan contract groups."""

    hints = {
        "headless_viewer": (
            "Inspect viewer_status.failures; show=False should stay lightweight "
            "and preserve preset/export semantics."
        ),
        "mode_viewers": (
            "Inspect mode_viewers_status.failures; mode plot and animation "
            "controllers should remain headless-safe."
        ),
        "dataset_time_modes_and_legacy": (
            "Inspect docs_example_status.failures and observed values; check "
            "dataset slice, tmin/tmax, modes=True, and legacy adapter contracts."
        ),
        "headless_import_boundary": (
            "Inspect headless_imports.new_widget_modules; headless paths should "
            "not import IPython, ipywidgets, or Matplotlib."
        ),
        "optional_widget_smoke": (
            "Inspect widget_smoke; install or repair optional notebook/widget "
            "dependencies, or rerun without --require-widget-smoke when optional."
        ),
        "benchmark_threshold": (
            "Inspect benchmark.threshold_failures, elapsed_s, and peak_memory_mb; "
            "adjust performance or configured thresholds intentionally."
        ),
    }
    if not masterplan_failures:
        return [
            "Run the real notebook smoke for %matplotlib widget, first render, "
            "analytics overlay, mode extraction, Export snapshot, and legacy UI."
        ]
    return [
        hints.get(name, f"Inspect contract group {name}.")
        for name in masterplan_failures
    ]


def run_release_gate(
    output_path: str | Path | None = None,
    *,
    max_elapsed_s: float | None = None,
    max_peak_memory_mb: float | None = None,
    benchmark_backend: str = "numpy",
    import_mode: str = "checkout",
    require_widget_smoke: bool = False,
) -> dict[str, Any]:
    """Run import, viewer, and benchmark smoke checks for FFT dispersion."""
    headless_widget_modules_before = _loaded_widget_modules()
    import_info = _prepare_import_path(import_mode)

    import mmpp
    import mmpp.fft.dispersion as dispersion
    from mmpp.fft.dispersion._interactive_viewer import DispersionInteractiveViewer

    run_benchmark = _load_benchmark_runner()
    viewer_state = _make_headless_viewer_state()
    mode_viewers = _make_headless_mode_viewer_state()
    mode_viewers_status = _mode_viewers_status(mode_viewers)
    docs_example = _run_docs_example_smoke()
    docs_example_status = _docs_example_status(docs_example)
    headless_imports = _headless_import_report(headless_widget_modules_before)
    viewer_state = {
        **viewer_state,
        "display_lifecycle": _run_viewer_display_lifecycle_smoke(),
    }
    viewer_status = _viewer_status(viewer_state)
    widget_smoke = _run_widget_smoke(require=require_widget_smoke)
    benchmark = run_benchmark(
        profile="small-ci",
        backend=benchmark_backend,
        workers=1,
        store_complex=False,
        scaling="amplitude_squared",
        max_elapsed_s=max_elapsed_s,
        max_peak_memory_mb=max_peak_memory_mb,
        import_mode=import_mode,
    )
    status = (
        "failed"
        if benchmark.get("threshold_status") == "failed"
        or not headless_imports.get("no_new_widget_modules", False)
        or viewer_status.get("status") == "failed"
        or mode_viewers_status.get("status") == "failed"
        or docs_example_status.get("status") == "failed"
        or widget_smoke.get("status") == "failed"
        else "ok"
    )
    masterplan_contracts = {
        "headless_viewer": viewer_status.get("status"),
        "mode_viewers": mode_viewers_status.get("status"),
        "dataset_time_modes_and_legacy": docs_example_status.get("status"),
        "headless_import_boundary": (
            "ok" if headless_imports.get("no_new_widget_modules", False) else "failed"
        ),
        "optional_widget_smoke": widget_smoke.get("status"),
        "benchmark_threshold": benchmark.get("threshold_status"),
    }
    masterplan_failures = [
        name
        for name, contract_status in masterplan_contracts.items()
        if contract_status == "failed"
    ]
    recommended_next_steps = _recommended_masterplan_next_steps(masterplan_failures)
    masterplan_failure_details: dict[str, Any] = {}
    if viewer_status.get("status") == "failed":
        masterplan_failure_details["headless_viewer"] = {
            "failures": viewer_status.get("failures", []),
            "display_lifecycle": viewer_state.get("display_lifecycle", {}),
        }
    if mode_viewers_status.get("status") == "failed":
        masterplan_failure_details["mode_viewers"] = {
            "failures": mode_viewers_status.get("failures", []),
        }
    if docs_example_status.get("status") == "failed":
        masterplan_failure_details["dataset_time_modes_and_legacy"] = {
            "failures": docs_example_status.get("failures", []),
            "observed": docs_example_status.get("observed", {}),
        }
    if not headless_imports.get("no_new_widget_modules", False):
        masterplan_failure_details["headless_import_boundary"] = {
            "new_widget_modules": headless_imports.get("new_widget_modules", []),
        }
    if widget_smoke.get("status") == "failed":
        masterplan_failure_details["optional_widget_smoke"] = {
            "missing": widget_smoke.get("missing", []),
            "reason": widget_smoke.get("reason"),
            "error": widget_smoke.get("error"),
        }
    if benchmark.get("threshold_status") == "failed":
        masterplan_failure_details["benchmark_threshold"] = {
            "threshold_status": benchmark.get("threshold_status"),
            "threshold_failures": benchmark.get("threshold_failures"),
            "elapsed_s": benchmark.get("elapsed_s"),
            "peak_memory_mb": benchmark.get("peak_memory_mb"),
        }
    summary = {
        "status": status,
        "failed_contract_count": len(masterplan_failures),
        "failed_contracts": masterplan_failures,
        "first_next_step": (
            recommended_next_steps[0] if recommended_next_steps else None
        ),
    }
    report = {
        "gate": "fft_dispersion_release_smoke",
        "status": status,
        "summary": summary,
        "imports": {
            "mmpp": mmpp.__name__ == "mmpp",
            "mmpp.fft.dispersion": dispersion.__name__ == "mmpp.fft.dispersion",
            "DispersionInteractiveViewer": (
                DispersionInteractiveViewer.__name__ == "DispersionInteractiveViewer"
            ),
        },
        "viewer": viewer_state,
        "viewer_status": viewer_status,
        "mode_viewers": mode_viewers,
        "mode_viewers_status": mode_viewers_status,
        "headless_imports": headless_imports,
        "docs_example": docs_example,
        "docs_example_summary": _docs_example_summary(docs_example),
        "docs_example_status": docs_example_status,
        "masterplan_contracts": masterplan_contracts,
        "masterplan_failures": masterplan_failures,
        "masterplan_failure_details": masterplan_failure_details,
        "recommended_next_steps": recommended_next_steps,
        "widget_smoke": widget_smoke,
        "benchmark": benchmark,
        "import_path": import_info,
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-elapsed-s", type=float, default=None)
    parser.add_argument("--max-peak-memory-mb", type=float, default=None)
    parser.add_argument(
        "--benchmark-backend",
        choices=["numpy", "scipy", "pyfftw"],
        default="numpy",
        help="FFT backend used by the synthetic benchmark part of this gate.",
    )
    parser.add_argument(
        "--import-mode",
        choices=["checkout", "installed"],
        default="checkout",
        help="Use checkout imports for development or installed package imports for smoke tests.",
    )
    parser.add_argument(
        "--require-widget-smoke",
        action="store_true",
        help="Fail when ipywidgets/Matplotlib smoke dependencies are unavailable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional JSON report path. The report includes docs_example_summary "
            "with full-dataset, sliced-dataset, tmin/tmax, and mode-policy "
            "and legacy-adapter smoke details, plus masterplan_contracts, "
            "masterplan_failures, masterplan_failure_details, and "
            "recommended_next_steps. The top-level summary is the quickest "
            "status view."
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help=(
            "Print only summary, masterplan_contracts, masterplan_failures, "
            "and recommended_next_steps to stdout. --output still writes the "
            "full JSON report."
        ),
    )
    args = parser.parse_args(argv)

    report = run_release_gate(
        output_path=args.output,
        max_elapsed_s=args.max_elapsed_s,
        max_peak_memory_mb=args.max_peak_memory_mb,
        benchmark_backend=args.benchmark_backend,
        import_mode=args.import_mode,
        require_widget_smoke=args.require_widget_smoke,
    )
    printed_report = (
        {
            "summary": report["summary"],
            "masterplan_contracts": report["masterplan_contracts"],
            "masterplan_failures": report["masterplan_failures"],
            "recommended_next_steps": report["recommended_next_steps"],
        }
        if args.summary_only
        else report
    )
    print(json.dumps(printed_report, indent=2, sort_keys=True))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
