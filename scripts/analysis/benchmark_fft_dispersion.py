"""Benchmark FFT dispersion compute time and peak Python memory.

This script intentionally uses synthetic data so it can run in CI or a clean
developer checkout without external simulation fixtures.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import zarr

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PROFILES: dict[str, tuple[int, int, int, int, int]] = {
    "small-ci": (32, 1, 8, 64, 3),
    "medium-dev": (128, 1, 16, 256, 3),
    "research-reference": (512, 1, 32, 1024, 3),
}


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


def _write_complex_wave_zarr(
    path: Path,
    *,
    shape: tuple[int, int, int, int, int],
    amplitude: float = 0.2,
    f_bin: int = 2,
    k_bin: int = 3,
    dt: float = 2e-12,
    dx: float = 5e-9,
    dy: float = 5e-9,
) -> Path:
    n_t, n_z, n_y, n_x, n_c = shape
    if n_c != 3:
        raise ValueError("shape must be (T, Z, Y, X, 3)")

    t = np.arange(n_t, dtype=float)[:, None, None, None]
    x = np.arange(n_x, dtype=float)[None, None, None, :]
    phase = 2.0 * np.pi * ((f_bin * t / n_t) + (k_bin * x / n_x))
    wave = amplitude * np.exp(1j * phase)

    data = np.zeros(shape, dtype=np.float32)
    data[..., 0] = wave.real
    data[..., 1] = wave.imag

    root = zarr.open(str(path), mode="w")
    root.create_dataset("m", data=data, chunks=shape)
    root.attrs["t_sampl"] = dt
    root.attrs["dx"] = dx
    root.attrs["dy"] = dy
    return path


def _array_mb(value: Any) -> float:
    return float(getattr(value, "nbytes", 0)) / (1024.0 * 1024.0)


def _estimate_pipeline_memory_mb(
    shape: tuple[int, int, int, int, int],
    *,
    store_complex: bool,
    avg_over_orthogonal: bool,
) -> dict[str, float]:
    """Return a lightweight memory preflight for the 1D dispersion pipeline."""
    n_t, n_z, n_y, n_x, n_c = shape
    raw_mb = float(np.prod(shape) * np.dtype(np.float32).itemsize) / (1024.0 * 1024.0)
    signal_shape = (n_t, n_z, n_y, n_x)
    signal_mb = float(np.prod(signal_shape) * np.dtype(np.complex64).itemsize) / (
        1024.0 * 1024.0
    )
    collapsed_orth = 1 if avg_over_orthogonal else n_y
    spectrum_shape = (collapsed_orth, n_x, n_t)
    spectrum_complex_mb = (
        float(np.prod(spectrum_shape) * np.dtype(np.complex64).itemsize)
        / (1024.0 * 1024.0)
    )
    spectrum_power_mb = (
        float(np.prod(spectrum_shape) * np.dtype(np.float32).itemsize)
        / (1024.0 * 1024.0)
    )
    s_complex_mb = spectrum_complex_mb if store_complex else 0.0
    s_local_mb = spectrum_power_mb if not avg_over_orthogonal else 0.0
    estimated_peak_mb = (
        raw_mb
        + signal_mb
        + spectrum_complex_mb
        + spectrum_complex_mb
        + spectrum_power_mb
        + s_complex_mb
        + s_local_mb
    )
    return {
        "raw_data_mb": round(raw_mb, 6),
        "signal_mb": round(signal_mb, 6),
        "spatial_fft_mb": round(spectrum_complex_mb, 6),
        "temporal_fft_mb": round(spectrum_complex_mb, 6),
        "power_mb": round(spectrum_power_mb, 6),
        "complex_cache_mb": round(s_complex_mb, 6),
        "local_spectra_mb": round(s_local_mb, 6),
        "estimated_peak_mb": round(estimated_peak_mb, 6),
    }


def _evaluate_thresholds(
    report: dict[str, Any],
    *,
    max_elapsed_s: float | None = None,
    max_peak_memory_mb: float | None = None,
) -> None:
    thresholds = {
        "max_elapsed_s": max_elapsed_s,
        "max_peak_memory_mb": max_peak_memory_mb,
    }
    failures: list[dict[str, Any]] = []
    if max_elapsed_s is not None and report["elapsed_s"] > max_elapsed_s:
        failures.append(
            {
                "metric": "elapsed_s",
                "actual": report["elapsed_s"],
                "limit": float(max_elapsed_s),
            }
        )
    if (
        max_peak_memory_mb is not None
        and report["peak_memory_mb"] > max_peak_memory_mb
    ):
        failures.append(
            {
                "metric": "peak_memory_mb",
                "actual": report["peak_memory_mb"],
                "limit": float(max_peak_memory_mb),
            }
        )

    report["thresholds"] = {
        key: float(value)
        for key, value in thresholds.items()
        if value is not None
    }
    report["threshold_status"] = "failed" if failures else "ok"
    report["threshold_failures"] = failures


def run_benchmark(
    *,
    output_path: str | Path | None = None,
    shape: tuple[int, int, int, int, int] | None = None,
    profile: str = "small-ci",
    backend: str | None = None,
    workers: int | None = 1,
    store_complex: bool = False,
    scaling: str = "amplitude_squared",
    avg_over_orthogonal: bool = True,
    max_elapsed_s: float | None = None,
    max_peak_memory_mb: float | None = None,
    import_mode: str = "checkout",
    preflight_only: bool = False,
) -> dict[str, Any]:
    """Run a synthetic 1D dispersion benchmark and return a JSON-safe report."""
    import_info = _prepare_import_path(import_mode)
    if profile not in BENCHMARK_PROFILES:
        raise ValueError(
            f"Unknown benchmark profile '{profile}'. "
            f"Available: {', '.join(sorted(BENCHMARK_PROFILES))}"
        )
    if shape is None:
        shape = BENCHMARK_PROFILES[profile]
        profile_name = profile
    else:
        profile_name = profile if shape == BENCHMARK_PROFILES[profile] else "custom"
    memory_preflight = _estimate_pipeline_memory_mb(
        shape,
        store_complex=store_complex,
        avg_over_orthogonal=avg_over_orthogonal,
    )
    if preflight_only:
        n_t, _n_z, n_y, n_x, _n_c = shape
        local_orth = n_y if not avg_over_orthogonal else 1
        spectrum_mb = (
            float(n_x * n_t * np.dtype(np.float32).itemsize)
            / (1024.0 * 1024.0)
        )
        local_mb = (
            float(local_orth * n_x * n_t * np.dtype(np.float32).itemsize)
            / (1024.0 * 1024.0)
            if not avg_over_orthogonal
            else 0.0
        )
        complex_mb = (
            float(local_orth * n_x * n_t * np.dtype(np.complex64).itemsize)
            / (1024.0 * 1024.0)
            if store_complex
            else 0.0
        )
        report = {
            "benchmark": "fft_dispersion_1d",
            "mode": "preflight",
            "profile": profile_name,
            "available_profiles": {
                key: list(value) for key, value in BENCHMARK_PROFILES.items()
            },
            "shape": list(shape),
            "backend": backend or "not-run",
            "workers": workers,
            "elapsed_s": 0.0,
            "peak_memory_mb": memory_preflight["estimated_peak_mb"],
            "peak_memory_kind": "estimated_preflight",
            "result_shape": [n_x, n_t],
            "f_bins": int(n_t),
            "k_bins": int(n_x),
            "s_raw_mb": round(spectrum_mb, 6),
            "s_display_mb": round(spectrum_mb, 6),
            "s_complex_mb": round(complex_mb, 6),
            "s_local_mb": round(local_mb, 6),
            "s_local_raw_mb": round(local_mb, 6),
            "s_local_display_mb": round(local_mb, 6),
            "scaling": str(scaling),
            "store_complex": bool(store_complex),
            "avg_over_orthogonal": bool(avg_over_orthogonal),
            "memory_preflight_mb": memory_preflight,
            "import_path": import_info,
        }
        _evaluate_thresholds(
            report,
            max_elapsed_s=max_elapsed_s,
            max_peak_memory_mb=max_peak_memory_mb,
        )

        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

        return report

    from mmpp.fft.dispersion import _fft_backend
    from mmpp.fft.dispersion.core import SpinWaveAnalyzer
    from mmpp.fft.dispersion.models import DispersionConfig

    original = _fft_backend.get_info()
    with tempfile.TemporaryDirectory(prefix="mmpp-dispersion-bench-") as tmp:
        zarr_path = _write_complex_wave_zarr(Path(tmp) / "wave.zarr", shape=shape)
        try:
            if backend is not None:
                _fft_backend.set_backend(backend)
            if workers is not None:
                _fft_backend.set_workers(workers)

            analyzer = SpinWaveAnalyzer(
                zarr_path,
                config=DispersionConfig(
                    time_window=None,
                    space_window=None,
                    detrend=None,
                ),
                tmax=None,
            )

            tracemalloc.start()
            start = time.perf_counter()
            result = analyzer.compute_dispersion_1d(
                axis="x",
                component="perp",
                avg_over_orthogonal=avg_over_orthogonal,
                orthogonal_avg_mode="fft_power",
                store_complex=store_complex,
                scaling=scaling,
            )
            elapsed = time.perf_counter() - start
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        finally:
            _fft_backend.set_backend(original["backend"])
            _fft_backend.set_workers(original["workers"])
            if tracemalloc.is_tracing():
                tracemalloc.stop()

    report = {
        "benchmark": "fft_dispersion_1d",
        "mode": "execute",
        "profile": profile_name,
        "available_profiles": {key: list(value) for key, value in BENCHMARK_PROFILES.items()},
        "shape": list(shape),
        "backend": backend or original["backend"],
        "workers": workers,
        "elapsed_s": round(float(elapsed), 6),
        "peak_memory_mb": round(float(peak) / (1024.0 * 1024.0), 6),
        "result_shape": list(result.S.shape),
        "f_bins": int(result.f_axis.size),
        "k_bins": int(result.k_axis.size),
        "s_raw_mb": round(_array_mb(result.S_raw), 6),
        "s_display_mb": round(_array_mb(result.S_display), 6),
        "s_complex_mb": round(_array_mb(result.S_complex), 6),
        "s_local_mb": round(_array_mb(result.S_local), 6),
        "s_local_raw_mb": round(_array_mb(result.S_local_raw), 6),
        "s_local_display_mb": round(_array_mb(result.S_local_display), 6),
        "scaling": str(result.scaling),
        "store_complex": bool(store_complex),
        "avg_over_orthogonal": bool(avg_over_orthogonal),
        "memory_preflight_mb": memory_preflight,
        "import_path": import_info,
    }
    _evaluate_thresholds(
        report,
        max_elapsed_s=max_elapsed_s,
        max_peak_memory_mb=max_peak_memory_mb,
    )

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    return report


def _shape_arg(value: str) -> tuple[int, int, int, int, int]:
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 5:
        raise argparse.ArgumentTypeError("shape must have five comma-separated integers")
    return parts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(BENCHMARK_PROFILES),
        default="small-ci",
        help="Named synthetic benchmark shape. Ignored when --shape is supplied.",
    )
    parser.add_argument("--shape", type=_shape_arg, default=None)
    parser.add_argument("--backend", choices=["numpy", "scipy", "pyfftw"], default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--store-complex", action="store_true")
    parser.add_argument(
        "--scaling",
        choices=["raw_power", "amplitude_squared", "psd"],
        default="amplitude_squared",
    )
    parser.add_argument("--no-orthogonal-average", action="store_true")
    parser.add_argument("--max-elapsed-s", type=float, default=None)
    parser.add_argument("--max-peak-memory-mb", type=float, default=None)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Report profile memory preflight without creating data or running FFT.",
    )
    parser.add_argument(
        "--import-mode",
        choices=["checkout", "installed"],
        default="checkout",
        help="Use checkout imports for development or installed package imports for smoke tests.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run_benchmark(
        output_path=args.output,
        shape=args.shape,
        profile=args.profile,
        backend=args.backend,
        workers=args.workers,
        store_complex=args.store_complex,
        scaling=args.scaling,
        avg_over_orthogonal=not args.no_orthogonal_average,
        max_elapsed_s=args.max_elapsed_s,
        max_peak_memory_mb=args.max_peak_memory_mb,
        import_mode=args.import_mode,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["threshold_status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
