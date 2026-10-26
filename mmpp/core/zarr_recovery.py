"""Diagnose interrupted Zarr output and export verified snapshots.

Run ``python -m mmpp.core.zarr_recovery STORE.zarr DATASET``. This tool never
rewrites the source, fills missing chunks, or treats a field as a solver restart.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any


def _digest(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _json(path: Path) -> Any:
    with path.open("rb") as stream:
        return json.load(stream)


def diagnose_dataset(
    zarr_path: str | Path, dataset: str, *, verify_chunks: bool = True
) -> dict[str, Any]:
    """Report a locally verifiable contiguous prefix without modifying files.

    For an incomplete record chain, ``verified_prefix`` describes the prefix
    with internally consistent records and chunks; it is not independently
    anchored to the latest head until ``head_chain_valid`` is true.
    """
    root = Path(zarr_path)
    relative = PurePosixPath(dataset)
    if relative.is_absolute() or ".." in relative.parts or "\\" in dataset:
        raise ValueError(f"Invalid dataset path {dataset!r}")
    directory = root.joinpath(*relative.parts)
    report: dict[str, Any] = {
        "store": str(root),
        "dataset": dataset,
        "format": "unknown",
        "committed_steps": None,
        "verified_prefix": 0,
        "head_chain_valid": False,
        "healthy": False,
        "issues": [],
        "warnings": [],
    }
    issues: list[str] = report["issues"]
    warnings: list[str] = report["warnings"]
    try:
        root_attrs = root / ".zattrs"
        if root_attrs.exists() and not isinstance(_json(root_attrs), dict):
            issues.append(f"Root attributes are not a JSON object: {root_attrs}")
    except (OSError, ValueError) as exc:
        issues.append(f"Root attributes unreadable: {exc}")

    commit_path = directory / "commit.json"
    try:
        head = _json(commit_path)
    except FileNotFoundError:
        report["format"] = "legacy"
        warnings.append(
            "No commit manifest: Zarr fill semantics allow absent chunks; "
            "file counts alone cannot prove integrity."
        )
        if (directory / ".frames").exists():
            issues.append("Committed frame records exist but commit.json is missing")
        try:
            metadata = _json(directory / ".zarray")
            shape, chunks = metadata["shape"], metadata["chunks"]
            if (
                len(shape) != 5
                or len(chunks) != 5
                or any(not isinstance(value, int) or value <= 0 for value in chunks)
            ):
                raise ValueError("invalid shape or chunks")
            expected = math.prod(
                (size + chunk - 1) // chunk
                for size, chunk in zip(shape, chunks, strict=True)
            )
            present = sum(
                1
                for entry in directory.iterdir()
                if entry.is_file() and re.fullmatch(r"\d+(?:\.\d+){4}", entry.name)
            )
            report.update(
                legacy_shape=shape,
                legacy_expected_chunks=expected,
                legacy_chunk_files=present,
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            issues.append(f"Legacy array metadata unreadable: {exc}")
        return report
    except (OSError, ValueError) as exc:
        issues.append(f"Commit unreadable: {exc}")
        return report

    report["format"] = "committed-v1"
    try:
        count = head["committed_steps"]
        metadata = head["metadata"]
        shape, chunks = metadata["shape"], metadata["chunks"]
        if (
            head["schema_version"] != 1
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or head["generation"] != count
            or len(shape) != 5
            or len(chunks) != 5
            or shape[0] != count
            or chunks[0] != 1
            or any(not isinstance(v, int) or v <= 0 for v in shape[1:] + chunks)
        ):
            raise ValueError("invalid committed schema or step count")
        report["committed_steps"] = count
        expected_axes = list(
            itertools.product(
                *(range((shape[i] + chunks[i] - 1) // chunks[i]) for i in range(1, 5))
            )
        )
        try:
            projection = _json(directory / ".zarray")
            report["projection_steps"] = projection["shape"][0]
            report["projection_matches"] = projection["shape"][0] == count
        except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
            warnings.append(f"Compatibility projection unreadable: {exc}")

        previous = ""
        prefix_open = True
        for step in range(count):
            frame_path = directory / ".frames" / f"{step:020d}.json"
            try:
                raw = frame_path.read_bytes()
                frame = json.loads(raw)
                if frame["step"] != step or frame["previous_sha256"] != previous:
                    raise ValueError("invalid record sequence or previous hash")
                if not math.isfinite(frame["time"]):
                    raise ValueError("nonfinite timestamp")
                expected = {
                    f"{step}." + ".".join(map(str, axis)) for axis in expected_axes
                }
                chunk_errors = []
                for record in frame["chunks"]:
                    key = record["key"]
                    if (
                        key not in expected
                        or not isinstance(record["size"], int)
                        or record["size"] <= 0
                    ):
                        raise ValueError("invalid or duplicate chunk record")
                    expected.remove(key)
                    if verify_chunks:
                        try:
                            size, digest = _digest(directory / key)
                            if size != record["size"] or digest != record["sha256"]:
                                chunk_errors.append(
                                    f"Chunk {key} size or SHA-256 mismatch"
                                )
                        except OSError as exc:
                            chunk_errors.append(f"Chunk {key} unreadable: {exc}")
                if expected:
                    raise ValueError("record omits required chunks")
                previous = hashlib.sha256(raw).hexdigest()
                if chunk_errors:
                    issues.extend(chunk_errors)
                    prefix_open = False
                elif prefix_open:
                    report["verified_prefix"] += 1
            except (OSError, ValueError, TypeError, KeyError) as exc:
                issues.append(f"Frame {step} unreadable or invalid: {exc}")
                prefix_open = False
                break
        else:
            report["head_chain_valid"] = previous == head["last_record_sha256"]
            if not report["head_chain_valid"]:
                issues.append("Frame record chain does not match commit head")
        if not verify_chunks:
            warnings.append("Chunk payloads were not verified")
        if not report["head_chain_valid"]:
            warnings.append("The verified prefix is not anchored to the commit head")
        # Keep diagnosis aligned with the reader's complete schema validation,
        # including compressor, dtype, bounds and hash encoding.
        from .zarr_integrity import CommittedDirectoryStore, ZarrIntegrityError

        try:
            CommittedDirectoryStore(str(root))._field(dataset)
        except ZarrIntegrityError as exc:
            issues.append(str(exc))
        report["healthy"] = bool(
            verify_chunks
            and not issues
            and report["head_chain_valid"]
            and report["verified_prefix"] == count
        )
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        issues.append(f"Invalid commit schema: {exc}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("store", help="Path to a local .zarr directory")
    parser.add_argument("dataset", help="Field dataset name")
    parser.add_argument(
        "--metadata-only", action="store_true", help="Do not read chunk payloads"
    )
    parser.add_argument(
        "--export-snapshot",
        metavar="DESTINATION.zarr",
        help="Export this complete verified field to a separate conventional Zarr store",
    )
    args = parser.parse_args()
    report = diagnose_dataset(
        args.store, args.dataset, verify_chunks=not args.metadata_only
    )
    if args.export_snapshot:
        if args.metadata_only or not report["healthy"]:
            parser.error("Snapshot export requires a fully verified committed field")
        from .zarr_snapshot import export_field_snapshot

        export_field_snapshot(args.store, args.dataset, args.export_snapshot)
        report["snapshot_export"] = args.export_snapshot
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
