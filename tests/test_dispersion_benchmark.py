from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_dispersion_benchmark_reports_time_memory_and_result_metadata(tmp_path):
    from scripts.analysis.benchmark_fft_dispersion import run_benchmark

    output_path = tmp_path / "benchmark.json"
    report = run_benchmark(
        output_path=output_path,
        shape=(8, 1, 2, 8, 3),
        backend="numpy",
        workers=1,
        store_complex=False,
        scaling="amplitude_squared",
    )

    assert report["benchmark"] == "fft_dispersion_1d"
    assert report["profile"] == "custom"
    assert "small-ci" in report["available_profiles"]
    assert report["backend"] == "numpy"
    assert report["shape"] == [8, 1, 2, 8, 3]
    assert report["result_shape"] == [8, 8]
    assert report["elapsed_s"] >= 0.0
    assert report["peak_memory_mb"] > 0.0
    assert report["s_raw_mb"] > 0.0
    assert report["s_complex_mb"] == 0.0
    assert report["s_local_mb"] == 0.0
    assert report["s_local_raw_mb"] == 0.0
    assert report["s_local_display_mb"] == 0.0
    assert report["memory_preflight_mb"]["raw_data_mb"] > 0.0
    assert (
        report["memory_preflight_mb"]["estimated_peak_mb"]
        >= report["memory_preflight_mb"]["raw_data_mb"]
    )
    assert report["scaling"] == "amplitude_squared"
    assert report["store_complex"] is False
    assert output_path.exists()

    persisted = json.loads(output_path.read_text())
    assert persisted["benchmark"] == report["benchmark"]
    assert persisted["peak_memory_mb"] == report["peak_memory_mb"]


def test_dispersion_benchmark_named_profiles_and_memory_preflight(tmp_path):
    from scripts.analysis.benchmark_fft_dispersion import (
        BENCHMARK_PROFILES,
        run_benchmark,
    )

    report = run_benchmark(
        output_path=tmp_path / "profile-benchmark.json",
        profile="small-ci",
        backend="numpy",
        workers=1,
        store_complex=True,
        avg_over_orthogonal=False,
    )

    assert report["profile"] == "small-ci"
    assert report["shape"] == list(BENCHMARK_PROFILES["small-ci"])
    assert set(report["available_profiles"]) >= {
        "small-ci",
        "medium-dev",
        "research-reference",
    }
    preflight = report["memory_preflight_mb"]
    assert preflight["signal_mb"] > 0.0
    assert preflight["spatial_fft_mb"] > 0.0
    assert preflight["temporal_fft_mb"] > 0.0
    assert preflight["power_mb"] > 0.0
    assert preflight["complex_cache_mb"] > 0.0
    assert preflight["local_spectra_mb"] > 0.0
    assert report["s_local_mb"] > 0.0
    assert report["s_local_raw_mb"] > 0.0
    assert report["s_local_display_mb"] > 0.0
    assert preflight["estimated_peak_mb"] >= sum(
        preflight[key]
        for key in (
            "raw_data_mb",
            "signal_mb",
            "power_mb",
            "complex_cache_mb",
            "local_spectra_mb",
        )
    )


def test_dispersion_benchmark_preflight_only_reports_large_profile_without_execution(
    tmp_path,
):
    from scripts.analysis.benchmark_fft_dispersion import (
        BENCHMARK_PROFILES,
        run_benchmark,
    )

    report = run_benchmark(
        output_path=tmp_path / "research-preflight.json",
        profile="research-reference",
        backend="pyfftw",
        workers=1,
        store_complex=True,
        avg_over_orthogonal=False,
        preflight_only=True,
        max_peak_memory_mb=4096.0,
    )

    assert report["mode"] == "preflight"
    assert report["profile"] == "research-reference"
    assert report["shape"] == list(BENCHMARK_PROFILES["research-reference"])
    assert report["backend"] == "pyfftw"
    assert report["peak_memory_kind"] == "estimated_preflight"
    assert report["elapsed_s"] == 0.0
    assert (
        report["peak_memory_mb"] == report["memory_preflight_mb"]["estimated_peak_mb"]
    )
    assert report["s_complex_mb"] > 0.0
    assert report["s_local_mb"] > 0.0
    assert report["s_local_raw_mb"] > 0.0
    assert report["s_local_display_mb"] > 0.0
    assert report["threshold_status"] == "ok"


def test_dispersion_benchmark_reports_threshold_status(tmp_path):
    from scripts.analysis.benchmark_fft_dispersion import run_benchmark

    report = run_benchmark(
        output_path=tmp_path / "threshold-benchmark.json",
        shape=(8, 1, 2, 8, 3),
        backend="numpy",
        workers=1,
        max_elapsed_s=60.0,
        max_peak_memory_mb=256.0,
    )

    assert report["threshold_status"] == "ok"
    assert report["threshold_failures"] == []
    assert report["thresholds"] == {
        "max_elapsed_s": 60.0,
        "max_peak_memory_mb": 256.0,
    }


def test_dispersion_benchmark_cli_fails_when_threshold_is_exceeded(tmp_path):
    output_path = tmp_path / "failed-threshold-benchmark.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/benchmark_fft_dispersion.py",
            "--shape",
            "8,1,2,8,3",
            "--backend",
            "numpy",
            "--workers",
            "1",
            "--max-elapsed-s",
            "0",
            "--output",
            str(output_path),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert output_path.exists()
    report = json.loads(output_path.read_text())
    assert report["threshold_status"] == "failed"
    assert report["threshold_failures"][0]["metric"] == "elapsed_s"


def test_dispersion_benchmark_cli_preflight_only_checks_memory_threshold(tmp_path):
    output_path = tmp_path / "preflight-threshold-benchmark.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/analysis/benchmark_fft_dispersion.py",
            "--profile",
            "medium-dev",
            "--preflight-only",
            "--max-peak-memory-mb",
            "1",
            "--output",
            str(output_path),
        ],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert output_path.exists()
    report = json.loads(output_path.read_text())
    assert report["mode"] == "preflight"
    assert report["threshold_status"] == "failed"
    assert report["threshold_failures"][0]["metric"] == "peak_memory_mb"


def test_fft_dispersion_benchmark_workflow_runs_medium_and_research_profiles():
    workflow = Path(".github/workflows/fft-dispersion-benchmark.yml").read_text()

    assert "workflow_dispatch:" in workflow
    assert "schedule:" in workflow
    assert 'cron: "17 3 * * 1"' in workflow
    assert 'pip install -e ".[fft]"' in workflow
    assert "Run medium benchmark" in workflow
    assert "--profile medium-dev" in workflow
    assert "--store-complex" in workflow
    assert "--no-orthogonal-average" in workflow
    assert "--max-elapsed-s 120" in workflow
    assert "--max-peak-memory-mb 2048" in workflow
    assert "Research profile memory preflight" in workflow
    assert "--profile research-reference" in workflow
    assert "--preflight-only" in workflow
    assert "--max-peak-memory-mb 4096" in workflow
    assert "fft-dispersion-benchmark-reports" in workflow
