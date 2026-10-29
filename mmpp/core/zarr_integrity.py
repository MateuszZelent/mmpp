"""Read committed MuMax Zarr snapshots without accepting incomplete chunks.

MuMax stores an immutable record per frame and atomically advances commit.json.
The standard Zarr metadata is only a projection, which may lag behind the
commit while a solver is running or after an interruption.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import threading
import warnings
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from typing import Any

import zarr

try:
    from zarr.storage import DirectoryStore
except ImportError:  # Zarr 3 is outside the package's declared dependency range.
    DirectoryStore = None  # type: ignore[assignment,misc]


class ZarrIntegrityError(ValueError):
    """A committed Zarr result is incomplete, malformed, or inconsistent."""


def _read_json(path: Path) -> Any:
    try:
        with path.open("rb") as stream:
            return json.load(stream)
    except (OSError, ValueError) as exc:
        raise ZarrIntegrityError(
            f"Invalid or unreadable Zarr metadata {path}: {exc}"
        ) from exc


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _recover_zero_projection(directory: Path) -> bytes:
    """Build an in-memory head from a contiguous, unanchored frame prefix.

    Inspect records and physical chunk sizes now; verify each payload SHA-256
    when it is actually read, using the same path as committed snapshots.
    Never infer timestamps from chunk names or rewrite the source metadata.
    """
    try:
        metadata = _read_json(directory / ".zarray")
        shape, chunks = metadata["shape"], metadata["chunks"]
        if (
            len(shape) != 5
            or len(chunks) != 5
            or shape[0] != 0
            or chunks[0] != 1
            or any(not _integer(v) or v <= 0 for v in shape[1:] + chunks)
        ):
            raise ValueError("recovery requires a zero-time, frame-chunked array")
        attrs = _read_json(directory / ".zattrs")
        if not isinstance(attrs, dict):
            raise ValueError("array attributes must be an object")
        axes = list(
            itertools.product(
                *(
                    range((size + chunk - 1) // chunk)
                    for size, chunk in zip(shape[1:], chunks[1:])
                )
            )
        )
        # Freeze the directory listing: records arriving later belong to a new
        # reader. Missing records or chunks delimit an incomplete tail.
        record_names = {entry.name for entry in (directory / ".frames").iterdir()}
        previous = ""
        previous_time = -math.inf
        count = 0
        while f"{count:020d}.json" in record_names:
            raw = (directory / ".frames" / f"{count:020d}.json").read_bytes()
            frame = json.loads(raw)
            timestamp = frame["time"]
            if (
                not _integer(frame["step"])
                or frame["step"] != count
                or frame["previous_sha256"] != previous
                or not isinstance(timestamp, (int, float))
                or isinstance(timestamp, bool)
                or not math.isfinite(timestamp)
                or timestamp <= previous_time
            ):
                raise ValueError(f"invalid frame record or timestamp at step {count}")
            expected = {f"{count}." + ".".join(map(str, axis)) for axis in axes}
            complete = True
            for record in frame["chunks"]:
                key, size, digest = record["key"], record["size"], record["sha256"]
                if (
                    key not in expected
                    or not _integer(size)
                    or size <= 0
                    or not isinstance(digest, str)
                    or len(digest) != 64
                ):
                    raise ValueError(f"invalid chunk record at step {count}")
                bytes.fromhex(digest)
                expected.remove(key)
                try:
                    actual_size = (directory / key).stat().st_size
                except FileNotFoundError:
                    complete = False
                    continue
                if actual_size != size:
                    raise ValueError(f"chunk size mismatch: {key}")
            if expected:
                raise ValueError(f"missing chunk records at step {count}")
            if not complete:
                break
            previous = hashlib.sha256(raw).hexdigest()
            previous_time = float(timestamp)
            count += 1
        if count == 0:
            raise ValueError("no complete frame prefix with trustworthy timestamps")
        metadata["shape"][0] = count
        return json.dumps(
            {
                "schema_version": 1,
                "generation": count,
                "committed_steps": count,
                "metadata": metadata,
                "name": attrs.get("name", directory.name),
                "unit": attrs.get("unit", ""),
                "last_record_sha256": previous,
                "mmpp_recovery": {
                    "reason": "zero_time_projection_without_commit",
                    "anchored_to_commit": False,
                    "frame_count": count,
                    "payload_verification": "sha256_on_read",
                },
                "recovered_attrs": attrs,
            }
        ).encode()
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        AttributeError,
    ) as exc:
        raise ZarrIntegrityError(
            f"Missing Zarr commit in {directory}; zero-time recovery failed: {exc}"
        ) from exc


class _DirectoryReader:
    """Minimal synchronous file reader used behind the Zarr 3 adapter."""

    def __init__(self, path: str):
        self.path = path

    def _normalize_key(self, key: str) -> str:
        location = PurePosixPath(key)
        if location.is_absolute() or ".." in location.parts or "\\" in key:
            raise ZarrIntegrityError(f"Invalid Zarr key {key!r}")
        return str(location)

    def __getitem__(self, key: str) -> bytes:
        try:
            return (Path(self.path) / self._normalize_key(key)).read_bytes()
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            raise KeyError(key) from None

    def __contains__(self, key: object) -> bool:
        return (
            isinstance(key, str)
            and (Path(self.path) / self._normalize_key(key)).is_file()
        )


class CommittedDirectoryStore(DirectoryStore if DirectoryStore else _DirectoryReader):  # type: ignore[misc]
    """A read-only snapshot overlay for committed fields and scalar tables."""

    def __init__(self, path: str):
        super().__init__(path)
        self._root = Path(path)
        self._fields: dict[str, dict[str, Any] | None] = {}
        self._tables: dict[str, dict[str, Any] | None] = {}
        # Records can contain many spatial chunk hashes. Retain only a small
        # number at a time instead of one entry per chunk in the entire run.
        self._record_cache: OrderedDict[tuple[str, int], dict[str, tuple[int, str]]] = (
            OrderedDict()
        )
        self._record_lock = threading.RLock()
        self._snapshot_lock = threading.RLock()
        self._root_attrs: bytes | None
        try:
            self._root_attrs = (self._root / ".zattrs").read_bytes()
        except (FileNotFoundError, NotADirectoryError):
            self._root_attrs = None
        except OSError as exc:
            raise ZarrIntegrityError(
                f"Cannot read root Zarr attributes in {path}: {exc}"
            ) from exc

    def __getstate__(self) -> dict[str, Any]:
        with self._snapshot_lock, self._record_lock:
            state = self.__dict__.copy()
            state.pop("_snapshot_lock", None)
            state.pop("_record_lock", None)
            state["_record_cache"] = OrderedDict()
            return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._record_lock = threading.RLock()
        self._snapshot_lock = threading.RLock()

    def _field(self, dataset: str) -> dict[str, Any] | None:
        with self._snapshot_lock:
            return self._load_field(dataset)

    def _load_field(self, dataset: str) -> dict[str, Any] | None:
        location = PurePosixPath(dataset)
        if location.is_absolute() or ".." in location.parts or "\\" in dataset:
            raise ZarrIntegrityError(f"Invalid Zarr dataset path {dataset!r}")
        if dataset in self._fields:
            return self._fields[dataset]
        directory = self._root / dataset
        commit_path = directory / "commit.json"
        try:
            raw = commit_path.read_bytes()
        except (FileNotFoundError, NotADirectoryError):
            if (directory / ".frames").exists():
                # Only zero-time projections qualify. A nonempty projection or
                # an invalid existing head must never silently downgrade.
                raw = _recover_zero_projection(directory)
            else:
                self._fields[dataset] = None
                return None
        except OSError as exc:
            raise ZarrIntegrityError(
                f"Cannot read Zarr commit {commit_path}: {exc}"
            ) from exc
        try:
            commit = json.loads(raw)
            count = commit["committed_steps"]
            metadata = commit["metadata"]
            shape, chunks = metadata["shape"], metadata["chunks"]
            if (
                commit["schema_version"] != 1
                or not _integer(count)
                or count < 0
                or commit["generation"] != count
                or len(shape) != 5
                or len(chunks) != 5
                or shape[0] != count
                or chunks[0] != 1
                or any(not _integer(v) or v <= 0 for v in shape[1:] + chunks)
                or any(c > s for c, s in zip(chunks[1:], shape[1:]))
                or metadata["zarr_format"] != 2
                or metadata["dtype"] != "<f4"
                or metadata["order"] != "C"
                or metadata["filters"] is not None
                or metadata["compressor"]["id"] != "zstd"
                or not isinstance(commit["name"], str)
                or not isinstance(commit["unit"], str)
            ):
                raise ValueError("invalid committed schema")
            expected_keys = list(
                itertools.product(
                    *(
                        range((shape[i] + chunks[i] - 1) // chunks[i])
                        for i in range(1, 5)
                    )
                )
            )
            previous = ""
            timestamps: list[float] = []
            record_hashes: list[str] = []
            for step in range(count):
                frame_path = directory / ".frames" / f"{step:020d}.json"
                frame_raw = frame_path.read_bytes()
                frame = json.loads(frame_raw)
                timestamp = frame["time"]
                if (
                    frame["step"] != step
                    or frame["previous_sha256"] != previous
                    or not isinstance(timestamp, (int, float))
                    or isinstance(timestamp, bool)
                    or not math.isfinite(timestamp)
                ):
                    raise ValueError(f"invalid frame record {step}")
                expected = {
                    f"{step}." + ".".join(map(str, key)) for key in expected_keys
                }
                for record in frame["chunks"]:
                    key, size, digest = record["key"], record["size"], record["sha256"]
                    if (
                        key not in expected
                        or not _integer(size)
                        or size <= 0
                        or not isinstance(digest, str)
                        or len(digest) != 64
                    ):
                        raise ValueError(f"invalid chunk record in frame {step}")
                    bytes.fromhex(digest)
                    expected.remove(key)
                if expected:
                    raise ValueError(f"missing chunk records in frame {step}")
                previous = hashlib.sha256(frame_raw).hexdigest()
                record_hashes.append(previous)
                timestamps.append(float(timestamp))
            if previous != commit["last_record_sha256"]:
                raise ValueError("frame record chain does not match commit")
        except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
            raise ZarrIntegrityError(
                f"Invalid committed Zarr dataset {directory}: {exc}"
            ) from exc
        attrs = {
            **commit.get("recovered_attrs", {}),
            "t": timestamps,
            "name": commit["name"],
            "unit": commit["unit"],
        }
        recovery = commit.get("mmpp_recovery")
        if recovery is not None:
            attrs["mmpp_recovery"] = recovery
            warnings.warn(
                f"Recovered {count} time samples from {directory} without commit.json; "
                "using a pinned contiguous frame prefix, not a completed run. "
                "Chunk SHA-256 is verified on read; source files are unchanged.",
                RuntimeWarning,
                stacklevel=3,
            )
        snapshot = {
            "metadata": json.dumps(metadata).encode(),
            "attrs": json.dumps(attrs).encode(),
            "record_hashes": record_hashes,
            "revision": hashlib.sha256(
                str(self._root.resolve()).encode()
                + b"\0"
                + (self._root_attrs or b"{}")
                + b"\0"
                + raw
            ).hexdigest(),
        }
        self._fields[dataset] = snapshot
        return snapshot

    def _chunk_record(
        self, dataset: str, leaf: str, field: dict[str, Any]
    ) -> tuple[int, str] | None:
        parts = leaf.split(".")
        if len(parts) != 5 or not all(part.isdecimal() for part in parts):
            return None
        step = int(parts[0])
        hashes = field["record_hashes"]
        if step >= len(hashes):
            return None
        key = (dataset, step)
        with self._record_lock:
            records = self._record_cache.get(key)
            if records is None:
                path = self._root / dataset / ".frames" / f"{step:020d}.json"
                try:
                    raw = path.read_bytes()
                    if hashlib.sha256(raw).hexdigest() != hashes[step]:
                        raise ValueError("record differs from the pinned snapshot")
                    frame = json.loads(raw)
                    records = {
                        item["key"]: (item["size"], item["sha256"])
                        for item in frame["chunks"]
                    }
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    raise ZarrIntegrityError(
                        f"Invalid committed frame record {path}: {exc}"
                    ) from exc
                self._record_cache[key] = records
                if len(self._record_cache) > 8:
                    self._record_cache.popitem(last=False)
            else:
                self._record_cache.move_to_end(key)
            return records.get(leaf)

    def _table(self, table: str) -> dict[str, Any] | None:
        with self._snapshot_lock:
            return self._load_table(table)

    def _load_table(self, table: str) -> dict[str, Any] | None:
        if table in self._tables:
            return self._tables[table]
        path = self._root / table / "manifest.json"
        try:
            path.stat()
        except (FileNotFoundError, NotADirectoryError):
            self._tables[table] = None
            return None
        except OSError as exc:
            raise ZarrIntegrityError(
                f"Cannot inspect table manifest {path}: {exc}"
            ) from exc
        manifest = _read_json(path)
        try:
            columns, units, rows = (
                manifest["columns"],
                manifest["units"],
                manifest["committed_rows"],
            )
            if (
                manifest["version"] != 1
                or manifest["format"] != "zarr"
                or not _integer(rows)
                or rows < 0
                or not _integer(manifest["chunk_rows"])
                or manifest["chunk_rows"] <= 0
                or not isinstance(columns, list)
                or not columns
                or not all(
                    isinstance(c, str) and c and "/" not in c and not c.startswith(".")
                    for c in columns
                )
                or len(columns) != len(set(columns))
                or not isinstance(units, list)
                or len(units) != len(columns)
            ):
                raise ValueError("invalid table manifest")
        except (KeyError, TypeError, ValueError) as exc:
            raise ZarrIntegrityError(f"Invalid table manifest {path}: {exc}") from exc
        self._tables[table] = manifest
        return manifest

    def __getitem__(self, key: str) -> bytes:
        key = self._normalize_key(key)
        if key == ".zattrs":
            if self._root_attrs is None:
                raise KeyError(key)
            return self._root_attrs
        parent, _, leaf = key.rpartition("/")
        if parent and leaf in (".zarray", ".zattrs"):
            field = self._field(parent)
            if field is not None:
                return field["metadata" if leaf == ".zarray" else "attrs"]
            if leaf == ".zarray":
                table, _, column = parent.rpartition("/")
                manifest = self._table(table) if table else None
                if manifest is not None and column in manifest["columns"]:
                    try:
                        metadata = json.loads(super().__getitem__(key))
                        if metadata["shape"][0] < manifest["committed_rows"]:
                            raise ValueError(
                                "column is shorter than its committed prefix"
                            )
                        metadata["shape"][0] = manifest["committed_rows"]
                        return json.dumps(metadata).encode()
                    except (KeyError, ValueError, TypeError, IndexError) as exc:
                        raise ZarrIntegrityError(
                            f"Invalid committed table column {key}: {exc}"
                        ) from exc
        field = self._fields.get(parent)
        if field is not None:
            record = self._chunk_record(parent, leaf, field)
            if record is not None:
                try:
                    data = bytes(super().__getitem__(key))
                except KeyError as exc:
                    raise ZarrIntegrityError(
                        f"Missing committed Zarr chunk {key}"
                    ) from exc
                size, digest = record
                if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
                    raise ZarrIntegrityError(f"Corrupt committed Zarr chunk {key}")
                return data
        table, _, column = parent.rpartition("/")
        manifest = self._tables.get(table)
        if manifest is not None and column in manifest["columns"] and leaf.isdecimal():
            if int(leaf) * manifest["chunk_rows"] < manifest["committed_rows"]:
                try:
                    return super().__getitem__(key)
                except KeyError as exc:
                    raise ZarrIntegrityError(
                        f"Missing committed table chunk {key}"
                    ) from exc
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        normalized_key = str(self._normalize_key(key))
        if normalized_key == ".zattrs":
            return self._root_attrs is not None
        parent, _, leaf = normalized_key.rpartition("/")
        if parent and leaf in (".zarray", ".zattrs"):
            if self._field(parent) is not None:
                return True
            if leaf == ".zarray":
                table, _, column = parent.rpartition("/")
                manifest = self._table(table) if table else None
                if (
                    manifest is not None
                    and column in manifest["columns"]
                    and manifest["committed_rows"] > 0
                    and not super().__contains__(normalized_key)
                ):
                    raise ZarrIntegrityError(
                        f"Missing committed table metadata {normalized_key}"
                    )
        return super().__contains__(normalized_key)

    def getitems(self, keys: Any, *, contexts: Any) -> dict[str, bytes]:
        # DirectoryStore's default checks __contains__ first. That would turn
        # a missing committed chunk into Zarr's fill value instead of an error.
        result = {}
        for key in keys:
            try:
                result[key] = self[key]
            except KeyError:
                pass
        return result


def open_source(path: str) -> Any:
    """Open a local result read-only, pinning each committed dataset lazily."""
    snapshot = CommittedDirectoryStore(path)
    if DirectoryStore is None:
        from .zarr3_store import SnapshotStore

        # Consolidated metadata bypasses per-array validation; use live keys
        # through the pinned overlay for both Zarr 2-format and 3-format stores.
        return zarr.open_group(
            SnapshotStore(snapshot), mode="r", use_consolidated=False
        )
    return zarr.open(snapshot, mode="r")


def source_revision(path: str, dataset: str) -> str | None:
    """Identify a committed source snapshot; legacy stores have no safe ID."""
    snapshot = CommittedDirectoryStore(path)._field(dataset)
    return None if snapshot is None else str(snapshot["revision"])


def job_source_revision(job: Any, dataset: str | None) -> str | None:
    """Return the version pinned by a JobResult, or inspect a detached path."""
    if not dataset:
        return None
    try:
        group = job.z
    except AttributeError:
        return source_revision(str(job.path), dataset)
    array = group[dataset]
    store = getattr(array, "store", None)
    if store is None or not hasattr(store, "_field"):
        return None
    snapshot = store._field(getattr(array, "path", dataset))
    return None if snapshot is None else str(snapshot["revision"])
