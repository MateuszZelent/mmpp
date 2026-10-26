"""Export one committed simulation field as an independent Zarr snapshot."""

from __future__ import annotations

import itertools
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import zarr

from ..cache.exports import publish_export
from .zarr_integrity import ZarrIntegrityError, open_source


def export_field_snapshot(
    source: str | Path, dataset: str, destination: str | Path
) -> None:
    """Copy one pinned committed field after verifying every chunk on read.

    The source is never modified. The output is a standalone Zarr v2 group,
    not a solver checkpoint, and does not include other fields or tables.
    """
    source_path = Path(source).resolve()
    destination_path = Path(destination).absolute()
    resolved_destination = destination_path.resolve()
    if (
        source_path == resolved_destination
        or source_path in resolved_destination.parents
        or resolved_destination in source_path.parents
    ):
        raise ValueError("Snapshot destination must be separate from the source store")
    relative = PurePosixPath(dataset)
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or "\\" in dataset
    ):
        raise ValueError(f"Invalid dataset path {dataset!r}")

    root = open_source(str(source_path))
    field = root[dataset]
    if not isinstance(field, zarr.Array):
        raise ValueError(f"Snapshot source is not an array: {dataset!r}")
    store = field.store
    snapshot = store._field(field.path) if hasattr(store, "_field") else None
    if snapshot is None:
        raise ZarrIntegrityError(
            "Cannot verify a legacy source without a committed revision"
        )
    attrs = dict(root.attrs)
    attrs.pop(
        "run_status_file", None
    )  # This export does not contain the source run.json.
    attrs["mmpp_snapshot"] = {
        "source_store": str(source_path),
        "source_dataset": dataset,
        "source_revision": snapshot["revision"],
        "scope": "single_field",
    }

    def populate(destination_root: Any) -> None:
        destination_root.attrs.update(attrs)
        parent = destination_root
        for part in relative.parts[:-1]:
            parent = parent.require_group(part)
        array = parent.create_dataset(
            relative.parts[-1],
            shape=field.shape,
            chunks=field.chunks,
            dtype=field.dtype,
            compressor=field.compressor,
            fill_value=field.fill_value,
            filters=field.filters,
            order=field.order,
        )
        array.attrs.update(dict(field.attrs))
        axes = [
            range(0, size, chunk)
            for size, chunk in zip(field.shape, field.chunks, strict=True)
        ]
        for starts in itertools.product(*axes):
            selection = tuple(
                slice(start, min(start + chunk, size))
                for start, chunk, size in zip(
                    starts, field.chunks, field.shape, strict=True
                )
            )
            array[selection] = np.asarray(field[selection])

    publish_export(destination_path, populate)
