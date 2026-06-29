from __future__ import annotations

import builtins
import importlib

from mmpp.core.mmpp import MMPP, ScanResult

mmpp_module = importlib.import_module("mmpp.core.mmpp")


def test_rich_progress_disabled_inside_ipython_kernel(monkeypatch) -> None:
    class ZMQInteractiveShell:
        pass

    monkeypatch.setattr(
        builtins, "get_ipython", lambda: ZMQInteractiveShell(), raising=False
    )
    monkeypatch.setattr(mmpp_module.sys.stderr, "isatty", lambda: True)

    assert mmpp_module._running_in_ipython_kernel() is True
    assert mmpp_module._should_render_rich_progress() is False


def test_scan_all_zarr_folders_runs_without_rich_progress_when_disabled(
    monkeypatch,
) -> None:
    scanner = MMPP.__new__(MMPP)
    scanner.max_workers = 1

    calls: list[str] = []

    def fake_scan(path: str) -> ScanResult:
        calls.append(path)
        return ScanResult(path=path, attributes={"path": path})

    monkeypatch.setattr(scanner, "_scan_single_zarr", fake_scan)
    monkeypatch.setattr(mmpp_module, "_should_render_rich_progress", lambda: False)

    results = scanner._scan_all_zarr_folders(["a.zarr", "b.zarr"])

    assert sorted(calls) == ["a.zarr", "b.zarr"]
    assert sorted(result.path for result in results) == ["a.zarr", "b.zarr"]
