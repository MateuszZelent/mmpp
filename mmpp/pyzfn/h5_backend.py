"""HDF5-backed adapters that quack like zarr objects.

Amumax's H5 storage mode writes per-quantity HDF5 files inside a .zarr
directory.  The adapters in this module present H5 data through the same
interface that downstream mmpp code expects from zarr Arrays and Groups,
so no user-facing API changes are required.

Layout on disk::

    simulation.zarr/
    ├── m/
    │   └── m.h5            # datasets: /0, /1, …, /t
    ├── table/
    │   └── table.h5        # datasets: /step, /t, /mx, /my, /mz, …
    ├── .zattrs
    └── .zgroup
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# H5Array — wraps a single HDF5 dataset
# ---------------------------------------------------------------------------


class H5Array:
    """Thin wrapper around an ``h5py.Dataset`` that exposes the subset of
    the ``zarr.Array`` interface used throughout mmpp."""

    __slots__ = ("_ds",)

    def __init__(self, dataset: h5py.Dataset) -> None:
        self._ds = dataset

    # -- zarr-compatible properties ------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return self._ds.shape

    @property
    def dtype(self) -> np.dtype:
        return self._ds.dtype

    @property
    def ndim(self) -> int:
        return self._ds.ndim

    @property
    def size(self) -> int:
        return int(np.prod(self._ds.shape))

    @property
    def name(self) -> str:
        return self._ds.name

    # -- indexing ------------------------------------------------------------
    def __getitem__(self, key: Any) -> NDArray:
        return np.asarray(self._ds[key])

    def __len__(self) -> int:
        return self.shape[0] if self.ndim > 0 else 0

    def __repr__(self) -> str:
        return f"H5Array(shape={self.shape}, dtype={self.dtype})"


# ---------------------------------------------------------------------------
# H5StackedArray — virtual view over step datasets /0, /1, …
# ---------------------------------------------------------------------------


class H5StackedArray:
    """Presents the numeric step datasets of a quantity HDF5 file as a single
    array with an extra leading time dimension: ``(n_steps, nz, ny, nx, ncomp)``.

    Indexing is *lazy*: only the requested steps are read from disk.
    """

    def __init__(self, h5file: h5py.File, step_keys: list[str]) -> None:
        self._h5 = h5file
        self._step_keys = sorted(step_keys, key=int)
        # Infer shape/dtype from first step
        sample = self._h5[self._step_keys[0]]
        self._step_shape: tuple[int, ...] = sample.shape
        self._dtype: np.dtype = sample.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._step_keys), *self._step_shape)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def ndim(self) -> int:
        return 1 + len(self._step_shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    # -- indexing ------------------------------------------------------------
    def __getitem__(self, key: Any) -> NDArray:
        # Normalise key to a tuple
        if not isinstance(key, tuple):
            key = (key,)

        time_key = key[0]
        rest = key[1:] if len(key) > 1 else ()

        # Determine which steps to read
        if isinstance(time_key, (int, np.integer)):
            idx = int(time_key)
            if idx < 0:
                idx += len(self._step_keys)
            ds_name = self._step_keys[idx]
            data = np.asarray(self._h5[ds_name][:])
            if rest:
                data = data[rest]
            return data

        indices: list[Any]
        if isinstance(time_key, slice):
            indices = list(range(*time_key.indices(len(self._step_keys))))
        elif hasattr(time_key, "__iter__"):
            indices = list(time_key)
        else:
            # Fallback: try reading directly
            indices = [int(time_key)]

        frames = []
        for i in indices:
            idx = int(i)
            if idx < 0:
                idx += len(self._step_keys)
            ds_name = self._step_keys[idx]
            frame = np.asarray(self._h5[ds_name][:])
            if rest:
                frame = frame[rest]
            frames.append(frame)

        return np.stack(frames, axis=0)

    def __len__(self) -> int:
        return len(self._step_keys)

    def __repr__(self) -> str:
        return f"H5StackedArray(shape={self.shape}, dtype={self.dtype})"


# ---------------------------------------------------------------------------
# H5QuantityGroup — wraps a quantity's .h5 file (m.h5, B_eff.h5, …)
# ---------------------------------------------------------------------------


class H5QuantityGroup:
    """Wraps a per-quantity HDF5 file and exposes it as a zarr-Group-like
    object.

    * Accessing the group as an **array** (e.g. ``group.shape``, ``group[:]``)
      delegates to the virtual stacked array.
    * ``group["t"]`` returns the timestamps array.
    * ``group.keys()`` lists all datasets.
    """

    def __init__(self, h5_path: str | Path, quantity_name: str) -> None:
        self._h5_path = str(h5_path)
        self._quantity_name = quantity_name
        self._h5: h5py.File | None = None
        self._stacked: H5StackedArray | None = None
        self._step_keys: list[str] | None = None
        self._non_step_keys: list[str] | None = None

    def _ensure_open(self) -> None:
        if self._h5 is None:
            self._h5 = h5py.File(self._h5_path, "r")
            all_keys = list(self._h5.keys())
            self._step_keys = [k for k in all_keys if k.isdigit()]
            self._non_step_keys = [k for k in all_keys if not k.isdigit()]
            if self._step_keys:
                self._stacked = H5StackedArray(self._h5, self._step_keys)

    # -- zarr.Array-like interface (stacked view) ----------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        self._ensure_open()
        if self._stacked is not None:
            return self._stacked.shape
        return (0,)

    @property
    def dtype(self) -> np.dtype:
        self._ensure_open()
        if self._stacked is not None:
            return self._stacked.dtype
        return np.dtype("float32")

    @property
    def ndim(self) -> int:
        self._ensure_open()
        if self._stacked is not None:
            return self._stacked.ndim
        return 0

    @property
    def size(self) -> int:
        self._ensure_open()
        if self._stacked is not None:
            return self._stacked.size
        return 0

    # -- indexing (delegates to stacked array) -------------------------------
    def __getitem__(self, key: Any) -> Any:
        self._ensure_open()
        # String key → child dataset (e.g. "t" for timestamps)
        if isinstance(key, str):
            if self._h5 is not None and key in self._h5:
                return H5Array(self._h5[key])
            raise KeyError(f"Dataset '{key}' not found in {self._h5_path}")

        # Numeric / slice key → stacked array
        if self._stacked is not None:
            return self._stacked[key]
        raise IndexError("No step data available")

    def __len__(self) -> int:
        self._ensure_open()
        if self._stacked is not None:
            return self._stacked.shape[0]
        return 0

    # -- zarr.Group-like interface -------------------------------------------
    def keys(self) -> list[str]:
        self._ensure_open()
        return list(self._h5.keys()) if self._h5 else []

    def __contains__(self, key: object) -> bool:
        self._ensure_open()
        return key in self._h5 if self._h5 else False

    def __iter__(self) -> Iterator[str]:
        self._ensure_open()
        return iter(self._h5.keys()) if self._h5 else iter([])

    @property
    def attrs(self) -> dict:
        self._ensure_open()
        if self._h5 is not None:
            return dict(self._h5.attrs)
        return {}

    def __repr__(self) -> str:
        return f"H5QuantityGroup('{self._quantity_name}', shape={self.shape})"

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self) -> None:
        self.close()


# ---------------------------------------------------------------------------
# H5TableGroup — wraps table/table.h5
# ---------------------------------------------------------------------------


class H5TableGroup:
    """Wraps ``table/table.h5`` to expose each column as a child dataset,
    matching the zarr table group interface."""

    def __init__(self, h5_path: str | Path) -> None:
        self._h5_path = str(h5_path)
        self._h5: h5py.File | None = None

    def _ensure_open(self) -> None:
        if self._h5 is None:
            self._h5 = h5py.File(self._h5_path, "r")

    def keys(self) -> list[str]:
        self._ensure_open()
        return list(self._h5.keys()) if self._h5 else []

    def __getitem__(self, key: str) -> H5Array:
        self._ensure_open()
        if self._h5 is not None and key in self._h5:
            return H5Array(self._h5[key])
        raise KeyError(f"Column '{key}' not found in table")

    def __contains__(self, key: object) -> bool:
        self._ensure_open()
        return key in self._h5 if self._h5 else False

    def __iter__(self) -> Iterator[str]:
        self._ensure_open()
        return iter(self._h5.keys()) if self._h5 else iter([])

    def __len__(self) -> int:
        self._ensure_open()
        return len(self._h5) if self._h5 else 0

    @property
    def attrs(self) -> dict:
        self._ensure_open()
        if self._h5 is not None:
            return dict(self._h5.attrs)
        return {}

    def __repr__(self) -> str:
        self._ensure_open()
        cols = list(self._h5.keys()) if self._h5 else []
        return f"H5TableGroup(columns={cols})"

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Detection helper
# ---------------------------------------------------------------------------


def detect_h5_quantities(
    zarr_path: str | Path,
) -> dict[str, H5QuantityGroup | H5TableGroup]:
    """Scan a .zarr directory for per-quantity HDF5 files.

    Returns a dict mapping quantity names to their H5 adapter objects.
    Only directories that actually contain a ``<name>.h5`` file are included.

    Parameters
    ----------
    zarr_path : str | Path
        Path to the ``.zarr`` directory.

    Returns
    -------
    dict[str, H5QuantityGroup | H5TableGroup]
        Mapping from quantity name to adapter. Empty if no H5 files found.
    """
    zarr_dir = Path(zarr_path)
    result: dict[str, H5QuantityGroup | H5TableGroup] = {}

    if not zarr_dir.is_dir():
        return result

    for child in zarr_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        h5_file = child / f"{name}.h5"
        if h5_file.is_file():
            if name == "table":
                result[name] = H5TableGroup(h5_file)
            else:
                result[name] = H5QuantityGroup(h5_file, name)

    return result
