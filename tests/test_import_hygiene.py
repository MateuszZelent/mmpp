"""Import hygiene tests for top-level MMPP APIs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap


def test_import_mmpp_without_matplotlib_reports_fft_dependency_groups() -> None:
    """Top-level import must not require plotting modules."""
    code = textwrap.dedent(
        """
        import json
        import sys

        class _BlockMatplotlib:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "matplotlib" or fullname.startswith("matplotlib."):
                    raise ModuleNotFoundError("blocked matplotlib for import hygiene")
                return None

        sys.meta_path.insert(0, _BlockMatplotlib())

        import mmpp

        status = mmpp.check_dependencies(verbose=False)
        print(json.dumps({key: status[key]["available"] for key in sorted(status)}))
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )

    status = json.loads(completed.stdout)
    assert "fft" in status
    assert "dispersion" in status
    assert "interactive" in status


def test_import_mmpp_does_not_probe_widget_stack() -> None:
    """Top-level import should not import or probe plotting/notebook modules."""
    code = textwrap.dedent(
        """
        import json
        import sys

        roots = {"IPython", "ipywidgets", "matplotlib"}
        attempts = []

        class _RecordWidgetImports:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in roots:
                    attempts.append(fullname)
                    raise ModuleNotFoundError(
                        f"blocked optional UI dependency: {fullname}"
                    )
                return None

        sys.meta_path.insert(0, _RecordWidgetImports())

        import mmpp  # noqa: F401

        print(json.dumps(sorted(set(attempts))))
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )

    assert json.loads(completed.stdout) == []


def test_import_mmpp_degrades_on_pandas_binary_incompatibility() -> None:
    """A broken pandas/numpy ABI should not crash top-level import."""
    code = textwrap.dedent(
        """
        import importlib.abc
        import importlib.machinery
        import json
        import sys
        import types

        rich = types.ModuleType("rich")
        rich.print = print
        rich.__path__ = []
        sys.modules["rich"] = rich

        class _RichDummy:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                return None

        for name in [
            "columns",
            "console",
            "logging",
            "panel",
            "progress",
            "syntax",
            "table",
            "text",
            "theme",
            "tree",
        ]:
            module = types.ModuleType(f"rich.{name}")
            module.Columns = _RichDummy
            module.Console = _RichDummy
            module.Panel = _RichDummy
            module.Progress = _RichDummy
            module.RichHandler = _RichDummy
            module.Syntax = _RichDummy
            module.Table = _RichDummy
            module.Text = _RichDummy
            module.Theme = _RichDummy
            module.Tree = _RichDummy
            sys.modules[f"rich.{name}"] = module

        class _BrokenPandasLoader(importlib.abc.Loader):
            def create_module(self, spec):
                return None

            def exec_module(self, module):
                raise ValueError(
                    "numpy.dtype size changed, may indicate binary incompatibility. "
                    "Expected 96 from C header, got 88 from PyObject"
                )

        class _BrokenPandasFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "pandas":
                    return importlib.machinery.ModuleSpec(
                        fullname,
                        _BrokenPandasLoader(),
                    )
                return None

        sys.meta_path.insert(0, _BrokenPandasFinder())

        import mmpp

        result = {
            "core_available": mmpp._CORE_AVAILABLE,
            "core_error": mmpp._CORE_IMPORT_ERROR,
            "version": mmpp.__version__,
        }
        try:
            mmpp.open("/tmp/not-used")
        except ImportError as exc:
            result["open_error"] = str(exc)

        print(json.dumps(result, sort_keys=True))
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )

    status = json.loads(completed.stdout)
    assert status["core_available"] is False
    assert "ValueError: numpy.dtype size changed" in status["core_error"]
    assert "Core MMPP functionality not available" in status["open_error"]
    assert "numpy.dtype size changed" in status["open_error"]
    assert "Restart the notebook kernel" in status["open_error"]


def test_fft_dispersion_headless_mode_viewers_do_not_probe_widget_stack() -> None:
    """show=False mode viewers must stay independent from UI dependencies."""
    code = textwrap.dedent(
        """
        import json
        import sys

        roots = {"IPython", "ipywidgets", "matplotlib"}
        attempts = []

        class _RecordWidgetImports:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in roots:
                    attempts.append(fullname)
                    raise ModuleNotFoundError(
                        f"blocked optional UI dependency: {fullname}"
                    )
                return None

        sys.meta_path.insert(0, _RecordWidgetImports())

        from scripts.analysis.verify_fft_dispersion_release_gate import (
            _make_headless_mode_viewer_state,
        )

        state = _make_headless_mode_viewer_state()
        print(json.dumps({
            "attempts": sorted(set(attempts)),
            "single_show": state["single_mode"]["show"],
            "animation_show": state["animation"]["show"],
        }, sort_keys=True))
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )

    status = json.loads(completed.stdout)
    assert status == {
        "animation_show": False,
        "attempts": [],
        "single_show": False,
    }


def test_import_analytical_without_matplotlib_keeps_stno_compute_available() -> None:
    """Analytical compute models should not import plotting eagerly."""
    code = textwrap.dedent(
        """
        import json
        import sys

        class _BlockMatplotlib:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "matplotlib" or fullname.startswith("matplotlib."):
                    raise ModuleNotFoundError("blocked matplotlib for import hygiene")
                return None

        sys.meta_path.insert(0, _BlockMatplotlib())

        import mmpp.analytical as analytical
        from mmpp.analytical.nonlinear_stno import (
            STNOParameters,
            SpectrumAnalyzer,
            run_all_sweeps_parallel,
        )

        print(json.dumps({
            "kittel": callable(analytical.kittel),
            "params": STNOParameters.__name__,
            "analyzer": SpectrumAnalyzer.__name__,
            "engine": callable(run_all_sweeps_parallel),
        }))
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )

    status = json.loads(completed.stdout)
    assert status == {
        "kittel": True,
        "params": "STNOParameters",
        "analyzer": "SpectrumAnalyzer",
        "engine": True,
    }


def test_dispersion_fft_backend_respects_environment_configuration() -> None:
    """Documented FFT env vars must be honored on a fresh import."""
    code = textwrap.dedent(
        """
        import json
        from mmpp.fft.dispersion._fft_backend import get_info

        print(json.dumps(get_info(), sort_keys=True))
        """
    )
    env = {
        **os.environ,
        "MMPP_FFT_BACKEND": "numpy",
        "MMPP_FFT_WORKERS": "2",
    }

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    info = json.loads(completed.stdout)
    assert info["backend"] == "numpy"
    assert info["workers"] == 2


def test_dispersion_internal_interface_exports_no_experimental_helpers() -> None:
    """Private dispersion helpers must not be advertised as public API."""
    import mmpp.fft.dispersion._interface as internal_interface
    from mmpp.fft.dispersion._interface.k0_filtering import K0Filter

    assert internal_interface.__all__ == []
    assert "K0Filter" not in internal_interface.__dict__
    assert K0Filter.__name__ == "K0Filter"
