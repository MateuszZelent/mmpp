from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path


def test_declared_python_versions_match_ci_matrix():
    pyproject = Path("pyproject.toml").read_text()
    setup_py = Path("setup.py").read_text()
    ci_workflow = Path(".github/workflows/ci.yml").read_text()
    expected = ["3.9", "3.10", "3.11", "3.12"]

    assert 'python-version: ["3.9", "3.10", "3.11", "3.12"]' in ci_workflow
    for version in expected:
        classifier = f"Programming Language :: Python :: {version}"
        assert classifier in pyproject
        assert classifier in setup_py


def test_dev_extra_declares_docs_linkify_dependency():
    pyproject = Path("pyproject.toml").read_text()
    setup_py = Path("setup.py").read_text()
    docs_conf = Path("docs/conf.py").read_text()
    pyproject_dev = re.search(r"dev = \[(.*?)\n\]", pyproject, re.S)
    setup_dev = re.search(r'"dev": \[(.*?)\n        \]', setup_py, re.S)

    assert '"linkify"' in docs_conf
    assert '"linkify-it-py",' in pyproject
    assert '"linkify-it-py",' in setup_py
    assert pyproject_dev is not None
    assert setup_dev is not None
    assert '"scipy",' in pyproject_dev.group(1)
    assert '"scipy",' in setup_dev.group(1)


def test_sphinx_static_path_exists():
    docs_conf = Path("docs/conf.py").read_text()

    assert 'html_static_path = ["_static"]' in docs_conf
    assert docs_conf.count('html_static_path = ["_static"]') == 1
    assert Path("docs/_static").is_dir()


def test_sphinx_theme_options_do_not_use_unsupported_display_version():
    docs_conf = Path("docs/conf.py").read_text()

    assert '"display_version"' not in docs_conf


def test_sphinx_suppresses_intentional_archive_toctree_warnings():
    docs_conf = Path("docs/conf.py").read_text()

    assert '"toc.not_included"' in docs_conf


def test_known_docs_warning_regressions_are_fixed():
    development_readme = Path("docs/development/README.md").read_text()
    refactor_plan = Path("docs/raports/10.05.2025/refacktor/plan.md").read_text()
    hysteresis_result = Path("mmpp/analyze/hysteresis/result.py").read_text()

    assert "../../DEVELOPMENT.md" not in development_readme
    assert "```toml\nzarr>=3\nh5py\n```" not in refactor_plan
    assert "same |B| range" not in hysteresis_result


def test_dispersion_docstrings_do_not_trigger_rst_substitutions():
    dispersion_interface = Path("mmpp/fft/dispersion/interface.py").read_text()
    dispersion_core = Path("mmpp/fft/dispersion/core.py").read_text()

    assert "Trim returned data to |k|" not in dispersion_interface
    assert "mean |FFT|" not in dispersion_core


def test_docs_workflows_use_dev_extra_for_linkify_dependency():
    docs_workflow = Path(".github/workflows/docs.yml").read_text()
    release_workflow = Path(".github/workflows/release.yml").read_text()

    assert 'pip install -e ".[dev]"' in docs_workflow
    assert "pip install -e .[dev]" not in docs_workflow
    assert 'pip install -e ".[dev]"' in release_workflow
    assert "pip install linkify-it-py" not in docs_workflow
    assert "pip install linkify-it-py" not in release_workflow


def test_full_extra_includes_fft_backend_dependencies():
    pyproject = Path("pyproject.toml").read_text()
    setup_py = Path("setup.py").read_text()
    pyproject_full = re.search(r"full = \[(.*?)\n\]", pyproject, re.S)
    setup_full = re.search(r'"full": \[(.*?)\n        \]', setup_py, re.S)

    assert pyproject_full is not None
    assert setup_full is not None
    assert '"pyfftw",' in pyproject_full.group(1)
    assert '"pyfftw",' in setup_full.group(1)


def test_fft_dispersion_release_gate_reports_core_api_and_benchmark(tmp_path):
    from scripts.analysis.verify_fft_dispersion_release_gate import run_release_gate

    output_path = tmp_path / "release-gate.json"
    report = run_release_gate(output_path=output_path)

    assert report["gate"] == "fft_dispersion_release_smoke"
    assert report["status"] == "ok"
    assert report["imports"]["mmpp"] is True
    assert report["imports"]["mmpp.fft.dispersion"] is True
    assert report["imports"]["DispersionInteractiveViewer"] is True
    assert report["viewer"]["show"] is False
    assert report["viewer"]["positive_frequencies"] is True
    assert report["viewer"]["preset_roundtrip"] is True
    assert report["viewer"]["display_lifecycle"] is True
    assert report["headless_imports"] == {
        "no_new_widget_modules": True,
        "new_widget_modules": [],
    }
    assert report["viewer"]["export_selection"] == {
        "k_rad_per_m": 1250000.0,
        "f_hz": 0.0,
        "frame": 2,
        "marker": [1.0, 2.0],
        "source": "release_gate",
    }
    assert report["viewer_status"] == {"status": "ok", "failures": []}
    assert report["mode_viewers"]["single_mode"]["show"] is False
    assert report["mode_viewers"]["single_mode"]["mode_type"] == "phase"
    assert report["mode_viewers"]["single_mode"]["export_selection"] == {
        "source": "release_gate",
        "frame": 3,
        "marker": [3.0, 4.0],
    }
    assert report["mode_viewers"]["animation"]["show"] is False
    assert report["mode_viewers"]["animation"]["peaks"] == [0, 2]
    assert report["mode_viewers"]["animation"]["export_selection"] == {
        "source": "release_gate",
        "frame": 4,
        "marker": [5.0, 6.0],
    }
    assert report["mode_viewers_status"] == {"status": "ok", "failures": []}
    assert report["docs_example"]["dataset_first_viewer_show"] is False
    assert report["docs_example"]["dataset_first_dataset"] == "m"
    assert "slice(None, 4" in report["docs_example"]["dataset_first_slice"]
    assert report["docs_example"]["dataset_first_shape"]
    assert set(report["docs_example"]["dataset_first_progress_stages"]) >= {
        "prepare",
        "compute",
        "cache",
        "result",
        "viewer",
        "done",
    }
    assert report["docs_example"]["compute_viewer_show"] is False
    assert report["docs_example"]["result_viewer_show"] is False
    assert report["docs_example"]["positive_frequencies"] is True
    assert report["docs_example"]["result_shape"] == [8, 8]
    assert report["docs_example"]["scaling"] == "amplitude_squared"
    assert report["docs_example"]["s_complex_is_none"] is True
    assert report["docs_example"]["export_source"] == "docs_example"
    assert report["docs_example"]["mode_result_has_complex"] is True
    assert report["docs_example"]["modes_viewer_can_reconstruct"] is True
    assert report["docs_example"]["mode_viewer_show"] is False
    assert report["docs_example"]["mode_viewer_type"] == "abs"
    assert report["docs_example"]["animation_peaks"] == [0]
    assert report["docs_example_status"] == {"status": "ok", "failures": []}
    assert report["docs_example"]["result_notes"]
    assert report["widget_smoke"]["status"] in {"ok", "skipped"}
    if report["widget_smoke"]["status"] == "ok":
        assert report["widget_smoke"]["has_figure"] is True
        assert report["widget_smoke"]["has_axes"] is True
        assert report["widget_smoke"]["has_widgets"] is True
        assert report["widget_smoke"]["close_cleared_display"] is True
        assert report["widget_smoke"]["close_cleared_figure"] is True
    else:
        assert report["widget_smoke"]["missing"]
        assert report["widget_smoke"]["required"] is False
    assert report["benchmark"]["benchmark"] == "fft_dispersion_1d"
    assert report["benchmark"]["profile"] == "small-ci"
    assert report["benchmark"]["shape"] == [32, 1, 8, 64, 3]
    assert report["benchmark"]["backend"] == "numpy"
    assert report["benchmark"]["store_complex"] is False
    assert report["benchmark"]["s_raw_mb"] > 0.0
    assert report["benchmark"]["s_local_raw_mb"] == 0.0
    assert report["benchmark"]["s_local_display_mb"] == 0.0
    assert output_path.exists()

    persisted = json.loads(output_path.read_text())
    assert persisted == report


def test_fft_dispersion_release_gate_forwards_benchmark_thresholds(tmp_path):
    from scripts.analysis.verify_fft_dispersion_release_gate import run_release_gate

    report = run_release_gate(
        output_path=tmp_path / "threshold-release-gate.json",
        max_elapsed_s=60.0,
        max_peak_memory_mb=256.0,
    )

    assert report["status"] == "ok"
    assert report["benchmark"]["threshold_status"] == "ok"
    assert report["benchmark"]["threshold_failures"] == []
    assert report["benchmark"]["thresholds"] == {
        "max_elapsed_s": 60.0,
        "max_peak_memory_mb": 256.0,
    }


def test_fft_dispersion_release_gate_can_select_benchmark_backend(tmp_path):
    from scripts.analysis.verify_fft_dispersion_release_gate import run_release_gate

    report = run_release_gate(
        output_path=tmp_path / "scipy-backend-release-gate.json",
        benchmark_backend="scipy",
    )

    assert report["status"] == "ok"
    assert report["benchmark"]["backend"] == "scipy"


def test_fft_dispersion_release_gate_can_require_widget_smoke(tmp_path):
    from scripts.analysis.verify_fft_dispersion_release_gate import run_release_gate

    report = run_release_gate(
        output_path=tmp_path / "required-widget-release-gate.json",
        require_widget_smoke=True,
    )

    assert report["widget_smoke"]["required"] is True
    if report["widget_smoke"]["status"] == "failed":
        assert report["status"] == "failed"
        assert report["widget_smoke"].get("missing") or report["widget_smoke"].get("error")
    else:
        assert report["status"] == "ok"
        assert report["widget_smoke"]["status"] == "ok"
        assert report["widget_smoke"]["has_figure"] is True
        assert report["widget_smoke"]["close_cleared_display"] is True
        assert report["widget_smoke"]["close_cleared_figure"] is True


def test_fft_dispersion_release_gate_fails_on_headless_widget_imports(monkeypatch):
    from scripts.analysis import verify_fft_dispersion_release_gate as release_gate

    monkeypatch.setattr(
        release_gate,
        "_headless_import_report",
        lambda _before: {
            "no_new_widget_modules": False,
            "new_widget_modules": ["matplotlib"],
        },
    )

    report = release_gate.run_release_gate()

    assert report["status"] == "failed"
    assert report["headless_imports"] == {
        "no_new_widget_modules": False,
        "new_widget_modules": ["matplotlib"],
    }


def test_fft_dispersion_release_gate_fails_on_early_headless_widget_import(monkeypatch):
    from scripts.analysis import verify_fft_dispersion_release_gate as release_gate

    original_prepare_import_path = release_gate._prepare_import_path

    def prepare_and_inject_widget_module(import_mode):
        info = original_prepare_import_path(import_mode)
        monkeypatch.setitem(sys.modules, "matplotlib", types.ModuleType("matplotlib"))
        return info

    monkeypatch.setattr(
        release_gate,
        "_prepare_import_path",
        prepare_and_inject_widget_module,
    )
    monkeypatch.setattr(
        release_gate,
        "_run_widget_smoke",
        lambda *, require=False: {
            "status": "skipped",
            "missing": ["ipywidgets"],
            "reason": "test stub",
            "required": bool(require),
        },
    )

    report = release_gate.run_release_gate()

    assert report["status"] == "failed"
    assert report["headless_imports"] == {
        "no_new_widget_modules": False,
        "new_widget_modules": ["matplotlib"],
    }


def test_fft_dispersion_release_gate_fails_on_docs_example_regression(monkeypatch):
    from scripts.analysis import verify_fft_dispersion_release_gate as release_gate

    monkeypatch.setattr(
        release_gate,
        "_run_docs_example_smoke",
        lambda: {
            "dataset_first_viewer_show": False,
            "dataset_first_dataset": "m",
            "dataset_first_slice": "(slice(None, 4, None), Ellipsis)",
            "dataset_first_shape": [8, 4],
            "dataset_first_progress_stages": [
                "prepare",
                "compute",
                "cache",
                "result",
                "viewer",
                "done",
            ],
            "compute_viewer_show": False,
            "result_viewer_show": False,
            "positive_frequencies": True,
            "result_shape": [8, 8],
            "scaling": "amplitude_squared",
            "s_complex_is_none": True,
            "export_source": "docs_example",
            "mode_result_has_complex": False,
            "modes_viewer_can_reconstruct": False,
            "mode_viewer_show": False,
            "mode_viewer_type": "abs",
            "animation_peaks": [0],
            "result_notes": ["synthetic"],
        },
    )

    report = release_gate.run_release_gate()

    assert report["status"] == "failed"
    assert report["docs_example_status"] == {
        "status": "failed",
        "failures": [
            "mode_result_has_complex",
            "modes_viewer_can_reconstruct",
        ],
    }
    assert report["docs_example"]["mode_result_has_complex"] is False
    assert report["docs_example"]["modes_viewer_can_reconstruct"] is False


def test_fft_dispersion_release_gate_fails_on_mode_viewers_regression(monkeypatch):
    from scripts.analysis import verify_fft_dispersion_release_gate as release_gate

    monkeypatch.setattr(
        release_gate,
        "_make_headless_mode_viewer_state",
        lambda: {
            "single_mode": {
                "show": True,
                "mode_type": "phase",
                "export_selection": {"source": "release_gate"},
            },
            "animation": {
                "show": False,
                "peaks": [],
                "export_selection": {"source": "release_gate"},
            },
        },
    )

    report = release_gate.run_release_gate()

    assert report["status"] == "failed"
    assert report["mode_viewers_status"] == {
        "status": "failed",
        "failures": [
            "animation_export",
            "animation_peaks",
            "single_mode_export",
            "single_mode_headless",
        ],
    }


def test_fft_dispersion_release_gate_fails_on_main_viewer_regression(monkeypatch):
    from scripts.analysis import verify_fft_dispersion_release_gate as release_gate

    monkeypatch.setattr(
        release_gate,
        "_make_headless_viewer_state",
        lambda: {
            "show": True,
            "positive_frequencies": False,
            "preset_roundtrip": False,
            "export_selection": {
                "source": "release_gate",
                "marker": [9.0],
            },
        },
    )
    monkeypatch.setattr(
        release_gate,
        "_run_viewer_display_lifecycle_smoke",
        lambda: False,
    )

    report = release_gate.run_release_gate()

    assert report["status"] == "failed"
    assert report["viewer_status"] == {
        "status": "failed",
        "failures": [
            "display_lifecycle",
            "export_selection",
            "positive_frequencies",
            "preset_roundtrip",
            "viewer_headless",
        ],
    }


def test_fft_dispersion_release_gate_headless_imports_ignore_display_lifecycle(
    monkeypatch,
):
    from scripts.analysis import verify_fft_dispersion_release_gate as release_gate
    from mmpp.fft.dispersion._interactive_viewer import DispersionInteractiveViewer

    monkeypatch.delitem(sys.modules, "IPython", raising=False)
    monkeypatch.delitem(sys.modules, "IPython.display", raising=False)

    def show_and_load_ipython(self):
        monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
        monkeypatch.setitem(
            sys.modules,
            "IPython.display",
            types.ModuleType("IPython.display"),
        )
        self.show_requested = True
        return self

    monkeypatch.setattr(DispersionInteractiveViewer, "show", show_and_load_ipython)
    monkeypatch.setattr(
        release_gate,
        "_run_widget_smoke",
        lambda *, require=False: {
            "status": "skipped",
            "missing": ["ipywidgets", "matplotlib"],
            "reason": "test stub",
            "required": bool(require),
        },
    )

    report = release_gate.run_release_gate()

    assert report["status"] == "ok"
    assert report["viewer_status"] == {"status": "ok", "failures": []}
    assert report["headless_imports"] == {
        "no_new_widget_modules": True,
        "new_widget_modules": [],
    }


def test_release_gate_installed_import_mode_removes_checkout_from_sys_path(monkeypatch):
    from scripts.analysis.verify_fft_dispersion_release_gate import (
        REPO_ROOT,
        _prepare_import_path,
    )

    repo_root = str(REPO_ROOT)
    monkeypatch.setattr(sys, "path", [repo_root, "", "/tmp/site-packages"])

    info = _prepare_import_path("installed")

    assert info == {"import_mode": "installed", "repo_root": repo_root}
    assert repo_root not in sys.path
    assert "" not in sys.path
    assert sys.path == ["/tmp/site-packages"]


def test_release_workflow_installs_built_artifacts_before_publish():
    workflow = Path(".github/workflows/release.yml").read_text()
    extras_smoke = workflow.split("  extras-smoke:", 1)[1].split("  build:", 1)[0]
    build_job = workflow.split("  build:", 1)[1].split("  publish-pypi:", 1)[0]
    publish_pypi = workflow.split("  publish-pypi:", 1)[1].split(
        "  publish-testpypi:", 1
    )[0]
    publish_testpypi = workflow.split("  publish-testpypi:", 1)[1]

    assert "pip install linkify-it-py" not in workflow
    assert "Build documentation" in workflow
    assert "sphinx-build -b html . _build --keep-going" in workflow
    assert "Benchmark profile preflight" in workflow
    assert "--profile medium-dev --preflight-only --max-peak-memory-mb 1024" in workflow
    assert (
        "--profile research-reference --preflight-only --max-peak-memory-mb 4096"
        in workflow
    )
    assert "extras-smoke:" in workflow
    assert "needs: build" in extras_smoke
    assert 'extra: ["fft", "interactive", "plotting", "full"]' in workflow
    assert 'pip install -e ".[${{ matrix.extra }}]"' not in workflow
    assert "actions/download-artifact@v4" in extras_smoke
    assert "name: dist" in extras_smoke
    assert "Install built wheel extra" in extras_smoke
    assert 'f"{wheel}[${{ matrix.extra }}]"' in extras_smoke
    assert 'python -c "import mmpp; import mmpp.fft; import mmpp.fft.dispersion"' not in extras_smoke
    assert "Required widget smoke" in workflow
    assert "Required pyFFTW backend smoke" in workflow
    assert "matrix.extra == 'fft' || matrix.extra == 'full'" in workflow
    assert "--benchmark-backend pyfftw" in workflow
    assert "matrix.extra == 'interactive' || matrix.extra == 'full'" in workflow
    assert "--require-widget-smoke" in workflow
    assert "needs: verify" in build_job
    assert "Smoke installed wheel" in workflow
    assert "python -m pip install --force-reinstall dist/*.whl" in workflow
    assert "python -m pip install --force-reinstall --no-deps dist/*.whl" not in workflow
    assert (
        "python scripts/analysis/verify_fft_dispersion_release_gate.py "
        "--import-mode installed "
        "--max-elapsed-s 60 --max-peak-memory-mb 256 "
        "--output /tmp/mmpp-fft-dispersion-wheel-smoke.json"
    ) in workflow
    assert (
        "python scripts/analysis/verify_fft_dispersion_release_gate.py "
        "--import-mode installed "
        "--max-elapsed-s 60 --max-peak-memory-mb 256 "
        "--output /tmp/mmpp-fft-dispersion-${{ matrix.extra }}-smoke.json"
    ) in workflow
    assert "Smoke installed sdist" in workflow
    assert "python -m pip wheel --wheel-dir /tmp/mmpp-sdist-wheel dist/*.tar.gz" in workflow
    assert "python -m pip wheel --no-deps --wheel-dir /tmp/mmpp-sdist-wheel" not in workflow
    assert "python -m pip install --force-reinstall /tmp/mmpp-sdist-wheel/*.whl" in workflow
    assert (
        "python -m pip install --force-reinstall --no-deps /tmp/mmpp-sdist-wheel/*.whl"
        not in workflow
    )
    assert (
        "python scripts/analysis/verify_fft_dispersion_release_gate.py "
        "--import-mode installed "
        "--max-elapsed-s 60 --max-peak-memory-mb 256 "
        "--output /tmp/mmpp-fft-dispersion-sdist-smoke.json"
    ) in workflow
    assert "needs: [build, extras-smoke]" in publish_pypi
    assert "needs: [build, extras-smoke]" in publish_testpypi


def test_dispersion_docs_prefer_live_interactive_api_and_label_legacy_path():
    api_doc = Path("docs/api/fft/dispersion.md").read_text()
    tutorial = Path("docs/tutorials/dispersion_analysis.md").read_text()

    assert "disp.plot.interactive(" in api_doc
    assert "res.plot.interactive(show=False" in api_doc
    assert "mode.plot.interactive(show=False" in api_doc
    assert "modes.plot.animation(peaks=[0], show=False" in api_doc
    assert "result_notes" in api_doc
    assert "MMPP_FFT_BACKEND" in api_doc
    assert "MMPP_FFT_WORKERS" in api_doc
    assert "compute_2d()` are experimental" in api_doc
    assert "S_raw" in api_doc
    assert "S_display" in api_doc
    assert "S_local_raw" in api_doc
    assert "S_local_display" in api_doc
    assert "S_local" in api_doc
    assert "result.S" in api_doc
    assert 'analysis_source="display"' in api_doc
    assert "save_complex" not in api_doc
    assert "store_complex" in api_doc
    assert "res1d.plot.interactive(show=False" in tutorial
    assert "mode.plot.interactive(show=False" in tutorial
    assert "modes.plot.animation(peaks=[0], show=False" in tutorial
    assert "result_notes" in tutorial
    assert "MMPP_FFT_BACKEND" in tutorial
    assert "workers=1" in tutorial
    assert "`compute_2d()` is experimental" in tutorial
    assert "S_raw" in tutorial
    assert "S_display" in tutorial
    assert "S_local_raw" in tutorial
    assert "S_local_display" in tutorial
    assert "S_local" in tutorial
    assert "result.S" in tutorial
    assert 'analysis_source="display"' in tutorial
    assert "save_complex" not in tutorial
    assert "store_complex" in tutorial
    assert "## Legacy Folded-Mode Workflow" in tutorial
    legacy_section = tutorial.split("## Legacy Folded-Mode Workflow", maxsplit=1)[1]
    assert "disp.dispersion_modes(" in legacy_section
    assert "modes.plot_interactive()" in legacy_section


def test_dispersion_audit_masterplan_markdown_structure_is_intact():
    report = Path(
        "docs/analysis/RAPORT_AUDYT_FFT_DISPERSION_INTERACTIVE_2026-06-28.md"
    ).read_text()

    assert report.count("```") % 2 == 0
    required_sections = [
        "## Werdykt wykonawczy",
        "## Stan aktualny po naprawach",
        "## Definicja produkcyjności",
        "## Rejestr ryzyk",
        "## Masterplan perfekcyjny",
        "## Macierz akceptacji",
        "## Konkluzja",
    ]
    for section in required_sections:
        assert section in report

    for phase in range(9):
        assert f"### Faza {phase}:" in report

    assert "```bash\n```" not in report
    assert "```python\n```" not in report
