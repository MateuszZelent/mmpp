"""Read-only source audit probes; synthetic stores are confined to /tmp.

Run from the repository root with PYTHONPATH=. MPLBACKEND=Agg python3
scripts/analysis/audit_interactive_2026_09_25.py. Prints JSON evidence.
These probes document defects, rather than asserting the defects are desirable.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram

from mmpp._shared.spectral import compute_psd
from mmpp.fft.modes import FMRModeData
from mmpp.fft.modes._interactive.callbacks import on_phase_index_changed
from mmpp.fft.modes._interactive.controls import guess_layer_bounds
from mmpp.fft.modes._interactive.filters import _to_ghz
from mmpp.fft.modes.interactive import InteractiveSpectrum
from mmpp.solitons.skyrmion import (
    SkyrmionInterface,
    SkyrmionTopologyConfig,
    detect_skyrmion,
)
from mmpp.solitons.skyrmion.ui.interactive_dashboard import SkyrmionInteractiveDashboard
from tests.fixtures.synthetic_skyrmion import generate_synthetic_skyrmion
from tests.test_skyrmion_analysis import _create_job


def main():
    evidence = {
        "python": platform.python_version(),
        "backend": matplotlib.get_backend(),
    }
    for package in ("numpy", "scipy", "matplotlib", "ipywidgets", "ipympl", "zarr"):
        try:
            evidence[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            evidence[package] = "not installed"

    # Identical symmetric Hann window, detrend and frequency bins on both sides.
    n, fs, k = 1024, 1024.0, 32
    x = np.sin(2 * np.pi * k * np.arange(n) / n)
    f, p, _, meta = compute_psd(x, dt=1 / fs, method="periodogram", scaling="density")
    ref_f, ref_p = periodogram(x, fs=fs, window=np.hanning(n), scaling="density")
    evidence["periodogram"] = {
        "peak_ratio_to_scipy": float(p[k] / ref_p[k]),
        "integrated_mmpp": float(np.sum(p) * (f[1] - f[0])),
        "integrated_scipy": float(np.sum(ref_p) * (ref_f[1] - ref_f[0])),
        "metadata": meta,
    }
    try:
        compute_psd(np.array([0.0, 1.0]), dt=1.0, method="welch")
        evidence["welch_two_samples"] = "accepted"
    except Exception as exc:
        evidence["welch_two_samples"] = f"{type(exc).__name__}: {exc}"
    evidence["frequency_fallback"] = {
        "input_Hz": [0, 1000],
        "output_GHz": _to_ghz(np.array([0, 1000])).tolist(),
    }
    evidence["memory_layer_bounds"] = guess_layer_bounds(
        SimpleNamespace(
            analyzer=SimpleNamespace(
                modes_path="missing",
                zarr_file={},
                _memory_modes=np.zeros((4, 1, 2, 2, 3)),
            )
        )
    )

    # Execute the actual phase-preview callback on Fourier coefficients of sin(wt).
    coeff = np.fft.rfft(x)[k] * 2 / n
    mode = np.zeros((2, 2, 3), dtype=complex)
    mode[..., 0] = coeff
    fig, ax = plt.subplots()
    im = ax.imshow(np.zeros((2, 2)))
    ex = SimpleNamespace(
        _internal_update=False,
        _fig=fig,
        _current_frequency_ghz=1.0,
        _current_z_layer=0,
        _mode_axes=np.array([[ax]]),
        _controls={"phase_index": SimpleNamespace(max=3)},
        _mode_type="real",
        _mode_components=["x"],
        _mode_row_types=["combined"],
        _load_mode=lambda *_: (mode, 1.0, (0, 1, 0, 1)),
    )
    on_phase_index_changed(ex, {"new": 1})
    evidence["phase_preview_quarter_period"] = {
        "actual_callback": float(im.get_array()[0, 0]),
        "expected_forward_time": 1.0,
        "title": ax.get_title(),
    }
    plt.close(fig)

    mode[..., 0] = np.array([[0, 1], [0, 1]])
    mode[..., 1] = 10 * mode[..., 0]
    viewer = InteractiveSpectrum(
        spectrum_result=SimpleNamespace(
            frequencies_ghz=np.arange(4, dtype=float), power=np.ones((4, 3))
        ),
        analyzer=SimpleNamespace(
            get_mode=lambda *_: FMRModeData(1.0, mode, extent=(0, 100, 0, 100))
        ),
    )
    viewer.show(components=["x", "y"], mode_view="magnitude", toolbar=False, show=False)
    evidence["mode_rendering"] = {
        "component_clims": [ax.images[0].get_clim() for ax in viewer._mode_axes[0]],
        "row_colorbar_count": len(viewer._mode_colorbars),
        "colorbar_range": [
            viewer._mode_colorbars[0].vmin,
            viewer._mode_colorbars[0].vmax,
        ],
        "extent_in_nm": list(viewer._mode_axes[0, 0].images[0].get_extent()),
        "xlabel": viewer._mode_axes[0, 0].get_xlabel(),
    }
    plt.close(viewer._fig)

    with tempfile.TemporaryDirectory(prefix="mmpp-interactive-audit-") as tmp:
        field = generate_synthetic_skyrmion(Nx=64, Ny=64, radius=14e-9)
        data = np.repeat(field[None, None, ...], 4, axis=0)
        job = _create_job(Path(tmp), "skyrmion", data)
        interface = SkyrmionInterface(job, dataset_name="m")
        first = interface.detect()
        interface.config.topology.min_abs_q = 1.5
        cached = interface.detect()
        fresh = interface.detect(force=True)
        evidence["skyrmion_config_cache"] = {
            "same_object_after_config_change": cached is first,
            "cached_state": cached.state,
            "forced_state": fresh.state,
            "cached_valid": bool(cached.valid),
            "forced_valid": bool(fresh.valid),
        }
        sliced = SkyrmionInterface(
            job,
            dataset_name="m",
            slice_info=(
                slice(None),
                slice(None),
                slice(None, None, 2),
                slice(None, None, 2),
                slice(None),
            ),
        )
        evidence["skyrmion_spatial_stride"] = {
            "full_radius_nm": interface.fit_size(method="threshold").radius_nm,
            "stride2_radius_nm": sliced.fit_size(method="threshold").radius_nm,
            "stride2_spacing_m": sliced._resolve_spacing(),
        }
        measured = []
        real_resolve = interface._resolve_data

        def resolve():
            arr = real_resolve()
            measured.append({"shape": list(arr.shape), "bytes": arr.nbytes})
            return arr

        interface._result_cache.clear()
        interface.config.topology.min_abs_q = 0.5
        with patch.object(interface, "_resolve_data", side_effect=resolve):
            start = time.perf_counter()
            dashboard = SkyrmionInteractiveDashboard(interface, size_method="threshold")
            built = time.perf_counter()
            dashboard.run()
            finished = time.perf_counter()
        evidence["skyrmion_dashboard"] = {
            "materializations": measured,
            "build_s": built - start,
            "run_s": finished - built,
            "png_bytes": len(dashboard.image.value),
        }
        # Changed controls leave old results/status visible until explicit Run.
        dashboard.frame.value = 1
        evidence["skyrmion_pending_state"] = dashboard.status.value
        for widget in (
            dashboard.root,
            dashboard.frame,
            dashboard.z_layer,
            dashboard.image,
        ):
            widget.close()

        sparse_job = SimpleNamespace(attrs={}, m=data)
        evidence["missing_spacing_m"] = SkyrmionInterface(
            sparse_job, dataset_name="m"
        )._resolve_spacing()

    timings = []
    for size in (64, 128, 256):
        field = generate_synthetic_skyrmion(Nx=size, Ny=size, radius=size * 0.22e-9)
        row = {"grid": size}
        for method in ("berg_luscher", "finite_diff"):
            start = time.perf_counter()
            result = detect_skyrmion(
                field, 1e-9, 1e-9, config=SkyrmionTopologyConfig(method=method)
            )
            row[method] = {"seconds": time.perf_counter() - start, "Q": result.Q}
        timings.append(row)
    evidence["topology_single_run_timings"] = timings

    print(json.dumps(evidence, indent=2, default=str))


if __name__ == "__main__":
    main()
