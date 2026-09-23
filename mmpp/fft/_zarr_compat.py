"""Small helpers for writing arrays across supported Zarr group APIs."""

from __future__ import annotations

from typing import Any

import numpy as np


def write_zarr_array(
    group: Any,
    name: str,
    data: Any,
    *,
    chunks: Any = None,
    overwrite: bool = False,
) -> Any:
    """Write ``data`` to a Zarr group using its available array API.

    Zarr 2 exposes ``create_dataset`` while newer Zarr releases expose
    ``create_array``. Keep the compatibility fallback local so cache writers
    do not need to know which API the active store provides.
    """
    array = np.asarray(data)
    if overwrite:
        try:
            if name in group:
                del group[name]
        except (KeyError, TypeError, ValueError):
            pass

    create_array = getattr(group, "create_array", None)
    if callable(create_array):
        kwargs: dict[str, Any] = {"data": array, "overwrite": overwrite}
        if chunks is not None:
            kwargs["chunks"] = chunks
        try:
            return create_array(name, **kwargs)
        except TypeError:
            # Some Zarr-compatible stores accept shape/dtype instead of data.
            kwargs.pop("data", None)
            kwargs.pop("overwrite", None)
            if chunks is not None:
                kwargs["chunks"] = chunks
            return create_array(
                name,
                shape=array.shape,
                dtype=array.dtype,
                **kwargs,
            )

    create_dataset = getattr(group, "create_dataset", None)
    if callable(create_dataset):
        kwargs = {
            "data": array,
            "shape": array.shape,
            "dtype": array.dtype,
            "overwrite": overwrite,
        }
        if chunks is not None:
            kwargs["chunks"] = chunks
        try:
            return create_dataset(name, **kwargs)
        except TypeError:
            kwargs.pop("shape", None)
            kwargs.pop("dtype", None)
            return create_dataset(name, **kwargs)

    raise AttributeError("Zarr group has neither create_array nor create_dataset")


__all__ = ["write_zarr_array"]
