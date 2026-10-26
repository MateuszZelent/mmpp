"""Atomic publication of standalone Zarr exports on local Linux filesystems."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import zarr
from zarr.storage import DirectoryStore

from .transactions import _atomic_json, _digest, _manifest, _sync_dir, fcntl

_INTEGRITY_FILE = ".mmpp-integrity.json"
_INTEGRITY_ATTR = "mmpp_export_integrity_version"


def _supports_pinned_directories() -> bool:
    return os.path.isdir("/proc/self/fd")


class _PinnedDirectoryStore(DirectoryStore):
    """Keep a reader on the opened generation across a directory exchange."""

    def __init__(self, path: Path) -> None:
        self._directory_fd: int | None = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            proc_path = Path(f"/proc/self/fd/{self._directory_fd}")
            if not proc_path.exists():
                raise RuntimeError("Pinned Zarr exports require /proc/self/fd")
            super().__init__(str(proc_path))
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        fd = getattr(self, "_directory_fd", None)
        if fd is not None:
            self._directory_fd = None
            os.close(fd)

    def __del__(self) -> None:
        self.close()


def _renameat2(source: Path, destination: Path, flags: int) -> None:
    """Rename without a gap or fail closed when the filesystem lacks support."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise OSError(errno.ENOSYS, "atomic Zarr export requires renameat2")
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), flags) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def publish_export(path: Path, populate: Callable[[Any], None]) -> None:
    """Publish an entire directory, retaining any prior export as a sibling."""
    if fcntl is None:
        raise RuntimeError("Atomic Zarr export requires a supported file lock")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    locks = path.parent / ".mmpp-export-locks"
    locks.mkdir(exist_ok=True)
    key = hashlib.sha256(path.name.encode()).hexdigest()
    with (locks / f"{key}.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        temporary = path.parent / f".mmpp-export-{uuid.uuid4().hex}"
        swapped = False
        try:
            if path.is_symlink():
                raise ValueError(f"Refusing to replace symlink {path}")
            root = zarr.open_group(str(temporary), mode="w")
            populate(root)
            root.attrs[_INTEGRITY_ATTR] = 1
            files = _manifest(temporary)
            _atomic_json(temporary / _INTEGRITY_FILE, {"version": 1, "files": files})
            _sync_dir(temporary)
            _sync_dir(path.parent)
            if path.exists():
                if not path.is_dir():
                    raise ValueError(f"Zarr export path is not a directory: {path}")
                _renameat2(temporary, path, 2)  # RENAME_EXCHANGE
                swapped = True  # temporary now contains the previous export
            else:
                _renameat2(temporary, path, 1)  # RENAME_NOREPLACE
            _sync_dir(path.parent)
        finally:
            if not swapped and temporary.exists():
                shutil.rmtree(temporary)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def open_export(path: Path) -> Any:
    """Verify new exports and retain read compatibility with legacy exports."""
    path = Path(path)
    if not _supports_pinned_directories():
        root = zarr.open_group(str(path), mode="r")
        if (path / _INTEGRITY_FILE).exists() or root.attrs.get(
            _INTEGRITY_ATTR
        ) is not None:
            raise RuntimeError("Verified Zarr export reads require /proc/self/fd")
        return root
    store = _PinnedDirectoryStore(path)
    pinned_path = Path(store.path)
    marker = pinned_path / _INTEGRITY_FILE
    root = zarr.open_group(store, mode="r")
    if not marker.exists():
        if root.attrs.get(_INTEGRITY_ATTR) is not None:
            raise ValueError(f"Missing Zarr export integrity manifest: {marker}")
        return root  # Legacy exports did not record a payload manifest.
    try:
        manifest = json.loads(marker.read_bytes())
        files = manifest["files"]
        if (
            manifest["version"] != 1
            or not isinstance(files, dict)
            or ".zgroup" not in files
        ):
            raise ValueError("invalid integrity manifest")
        if root.attrs.get(_INTEGRITY_ATTR) != 1:
            raise ValueError("missing export integrity version")
        for name, record in files.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("invalid file path in integrity manifest")
            target = pinned_path.joinpath(*relative.parts)
            if target.is_symlink():
                raise ValueError("symlink in integrity manifest")
            size, digest = _digest(target)
            if size != record["size"] or digest != record["sha256"]:
                raise ValueError(f"payload mismatch for {name}")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Zarr export {path}: {exc}") from exc
    return root
