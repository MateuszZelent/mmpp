"""Run a local release-smoke gate for FFT dispersion.

The gate is intentionally small and synthetic. It verifies that the public
dispersion imports, headless interactive controller, and benchmark path work from
the current checkout. Release workflows should run this after installing the
built wheel as an additional packaging proof.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Callable

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
        "positive_frequencies": bool(state.get("options", {}).get("positive_frequencies")),
        "preset_roundtrip": reloaded.state == state,
        "export_selection": exported["selection"],
    }


def _run_viewer_display_lifecycle_smoke() -> bool:
    """Exercise explicit show/close outside the headless import measurement."""
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
    lifecycle = DispersionInteractiveViewer.from_result(result, show=False)
    lifecycle.show()
    shown = lifecycle.show_requested is True
    lifecycle.close()
    return shown and lifecycle.show_requested is False and lifecycle._display_handle is None


def _viewer_status(viewer_state: dict[str, Any]) -> dict[str, Any]:
    selection = viewer_state.get("export_selection", {})
    checks = {
        "viewer_headless": viewer_state.get("show") is False,
        "positive_frequencies": viewer_state.get("positive_frequencies") is True,
        "preset_roundtrip": viewer_state.get("preset_roundtrip") is True,
        "display_lifecycle": viewer_state.get("display_lifecycle") is True,
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
    from mmpp.fft.dispersion.interface import FFTDispersionInterface

    with tempfile.TemporaryDirectory(prefix="mmpp-dispersion-docs-smoke-") as tmp:
        zarr_path = Path(tmp) / "docs-smoke.zarr"
        _write_docs_smoke_zarr(zarr_path)
        disp = FFTDispersionInterface(
            SimpleNamespace(
                job_result=SimpleNamespace(path=zarr_path, name="docs-smoke")
            )
        )

        compute_viewer = disp.plot.interactive(
            axis="x",
            component="perp",
            fmax=25,
            show=False,
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

    return {
        "compute_viewer_show": bool(compute_viewer.state["show"]),
        "result_viewer_show": bool(result_viewer.state["show"]),
        "positive_frequencies": bool(
            result_viewer.state["options"].get("positive_frequencies")
        ),
        "result_shape": list(result.shape),
        "scaling": result.scaling,
        "s_complex_is_none": result.S_complex is None,
        "result_notes": result_viewer.state["result_notes"],
        "export_source": exported["selection"]["source"],
        "mode_result_has_complex": mode_result.S_complex is not None,
        "modes_viewer_can_reconstruct": bool(modes_viewer.can_reconstruct_modes),
        "mode_viewer_show": bool(mode_viewer.state["show"]),
        "mode_viewer_type": mode_viewer.state["mode_type"],
        "animation_peaks": animation_viewer.state["peaks"],
    }


def _docs_example_status(docs_example: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "compute_viewer_headless": docs_example.get("compute_viewer_show") is False,
        "result_viewer_headless": docs_example.get("result_viewer_show") is False,
        "positive_frequencies": docs_example.get("positive_frequencies") is True,
        "amplitude_squared_scaling": docs_example.get("scaling")
        == "amplitude_squared",
        "preview_without_complex": docs_example.get("s_complex_is_none") is True,
        "mode_result_has_complex": docs_example.get("mode_result_has_complex") is True,
        "modes_viewer_can_reconstruct": docs_example.get(
            "modes_viewer_can_reconstruct"
        )
        is True,
        "mode_viewer_headless": docs_example.get("mode_viewer_show") is False,
        "mode_viewer_abs": docs_example.get("mode_viewer_type") == "abs",
        "animation_peaks": docs_example.get("animation_peaks") == [0],
    }
    failures = sorted(name for name, ok in checks.items() if not ok)
    return {
        "status": "failed" if failures else "ok",
        "failures": failures,
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
            "display_handle_present": getattr(modes, "_display_handle", None) is not None,
        }
        modes.close()
        report["close_cleared_display"] = getattr(modes, "_display_handle", None) is None
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
    report = {
        "gate": "fft_dispersion_release_smoke",
        "status": status,
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
        "docs_example_status": docs_example_status,
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
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run_release_gate(
        output_path=args.output,
        max_elapsed_s=args.max_elapsed_s,
        max_peak_memory_mb=args.max_peak_memory_mb,
        benchmark_backend=args.benchmark_backend,
        import_mode=args.import_mode,
        require_widget_smoke=args.require_widget_smoke,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
