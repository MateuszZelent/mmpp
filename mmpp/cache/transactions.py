"""Crash-safe generations for local on-disk analysis caches.

Only the small head file is replaced. Older generations remain available if a
new computation fails or the process is interrupted before publication.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import zarr

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows needs a separate lock backend.
    fcntl = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: Path, payload: Any) -> None:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    fd, filename = tempfile.mkstemp(prefix=".mmpp-write-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(filename, path)
        _sync_dir(path.parent)
    finally:
        try:
            os.unlink(filename)
        except FileNotFoundError:
            pass


def _key(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("Cache name must be a nonempty string")
    return hashlib.sha256(name.encode()).hexdigest()


def group_directory(group: Any) -> Path:
    """Resolve a local Zarr group's directory or reject unsupported stores."""
    store = getattr(group, "store", None)
    root = getattr(store, "path", None)
    if root is None:
        raise RuntimeError("Transactional cache requires a local directory store")
    return Path(root) / str(getattr(group, "path", ""))


def _digest(path: Path, *, sync: bool = False) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        if sync:
            os.fsync(stream.fileno())
    return size, digest.hexdigest()


def _manifest(directory: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    directories = []
    for current, dirs, names in os.walk(directory, followlinks=False):
        root = Path(current)
        if any((root / name).is_symlink() for name in dirs + names):
            raise ValueError("Cache generations may not contain symbolic links")
        directories.append(root)
        for name in names:
            file_path = root / name
            if file_path == directory / ".mmpp-integrity.json":
                continue
            size, digest = _digest(file_path, sync=True)
            files[file_path.relative_to(directory).as_posix()] = {
                "size": size,
                "sha256": digest,
            }
    for item in reversed(directories):
        _sync_dir(item)
    return files


def publish_cache(
    parent_path: Path,
    name: str,
    populate: Callable[[Any], None],
    *,
    source_revision: str | None,
    overwrite: bool,
) -> bool:
    """Build and commit one generation. Return false if already present."""
    if fcntl is None:
        raise RuntimeError("Atomic cache publication requires a supported file lock")
    parent = Path(parent_path)
    parent.mkdir(parents=True, exist_ok=True)
    key = _key(name)
    heads = parent / ".mmpp-heads"
    locks = parent / ".mmpp-locks"
    generations = parent / ".mmpp-generations" / key
    for directory in (heads, locks, generations):
        directory.mkdir(parents=True, exist_ok=True)
    lock_path = locks / f"{key}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            head_path = heads / f"{key}.json"
            if (
                not overwrite
                and open_published_cache(parent, name, source_revision=source_revision)
                is not None
            ):
                return False
            generation = uuid.uuid4().hex
            directory = generations / generation
            group = zarr.open_group(str(directory), mode="w")
            populate(group)
            group.attrs["complete"] = True
            files = _manifest(directory)
            integrity_path = directory / ".mmpp-integrity.json"
            _atomic_json(integrity_path, {"version": 1, "files": files})
            _, integrity_hash = _digest(integrity_path)
            _sync_dir(generations)
            _sync_dir(generations.parent)
            _sync_dir(parent)
            _sync_dir(parent.parent)
            _atomic_json(
                head_path,
                {
                    "version": 1,
                    "name": name,
                    "generation": generation,
                    "integrity_sha256": integrity_hash,
                    "source_revision": source_revision,
                },
            )
            return True
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def open_published_cache(
    parent_path: Path,
    name: str,
    *,
    source_revision: str | None,
    allow_unversioned: bool = False,
) -> Any | None:
    """Open a verified generation pinned by one head; reject unknown sources."""
    if source_revision is None and not allow_unversioned:
        return None
    parent = Path(parent_path)
    key = _key(name)
    head_path = parent / ".mmpp-heads" / f"{key}.json"
    try:
        head = json.loads(head_path.read_bytes())
        generation = head["generation"]
        if (
            head["version"] != 1
            or head["name"] != name
            or head["source_revision"] != source_revision
            or not isinstance(generation, str)
            or len(generation) != 32
            or any(c not in "0123456789abcdef" for c in generation)
        ):
            return None
        directory = parent / ".mmpp-generations" / key / generation
        integrity_bytes = (directory / ".mmpp-integrity.json").read_bytes()
        if hashlib.sha256(integrity_bytes).hexdigest() != head["integrity_sha256"]:
            return None
        integrity = json.loads(integrity_bytes)
        if integrity["version"] != 1 or not isinstance(integrity["files"], dict):
            return None
        expected = integrity["files"]
        if not expected or ".zgroup" not in expected:
            return None
        for name, record in expected.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                return None
            path = directory.joinpath(*relative.parts)
            if path.is_symlink():
                return None
            size, digest = _digest(path)
            if size != record["size"] or digest != record["sha256"]:
                return None
        group = zarr.open_group(str(directory), mode="r")
        if group.attrs.get("complete") is not True:
            return None
        return group
    except (OSError, ValueError, TypeError, KeyError) as exc:
        log.debug("Ignoring incomplete or corrupt cache %s: %s", head_path, exc)
        return None


def peek_cache_revision(parent_path: Path, name: str) -> str | None:
    """Read an untrusted revision hint; callers must verify the source itself."""
    try:
        head = json.loads(
            (Path(parent_path) / ".mmpp-heads" / f"{_key(name)}.json").read_bytes()
        )
        value = head["source_revision"]
        return value if isinstance(value, str) else None
    except (OSError, ValueError, TypeError, KeyError):
        return None
