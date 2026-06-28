"""Custom Zarr Group subclass and utilities for structured simulation output management.

This module provides the Pyzfn class, which extends Zarr Group functionality with
convenient methods for hierarchical dataset handling, array creation, metadata access,
and visual tree formatting for simulation outputs.
"""

import tempfile
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeVar

import numpy as np
import zarr
from numpy.typing import NDArray
from rich.console import Console
from rich.tree import Tree

# Handle both Zarr v2 and v3 imports
# Check if this is Zarr v2 (has zarr.core.group) or v3
try:
    import zarr.core.group
    # Zarr v2
    ZARR_VERSION = 2
    from zarr.core.group import AsyncGroup, Group
    from zarr.core.sync import sync
    from zarr.storage import StoreLike
except (ImportError, AttributeError):
    # Zarr v3
    ZARR_VERSION = 3
    from zarr import Group
    AsyncGroup = None
    sync = None
    StoreLike = None

warnings.filterwarnings(
    "ignore",
    message="Object at .* is not recognized as a component of a Zarr hierarchy.",
    category=UserWarning,
)

T = TypeVar("T", bound=np.generic)
IndexLike = int | slice | Sequence[int] | NDArray[np.int_]
SliceTuple = tuple[IndexLike, ...]


def inner_calc_modes(self: "Pyzfn", *args, **kwargs):
    """Lazy wrapper for mode extraction helpers."""
    from .calc_modes import inner_calc_modes as _inner_calc_modes

    return _inner_calc_modes(self, *args, **kwargs)


def inner_ispec(self: "Pyzfn", *args, **kwargs):
    """Lazy wrapper for interactive spectrum plotting helpers."""
    from .ispec import inner_ispec as _inner_ispec

    return _inner_ispec(self, *args, **kwargs)


def inner_snapshot(self: "Pyzfn", *args, **kwargs):
    """Lazy wrapper for snapshot plotting helpers."""
    from .snapshot import inner_snapshot as _inner_snapshot

    return _inner_snapshot(self, *args, **kwargs)


class Pyzfn:
    """A custom Zarr Group wrapper for structured simulation output management.

    Provides utility methods and properties for handling hierarchical datasets,
    including array creation, metadata access, and visual tree formatting.
    
    In Zarr v2, this inherits from Group. In Zarr v3, it wraps a Group object.
    """

    def __init__(
        self,
        store: str | Path,
        zarr_format: Literal[2, 3] = 2,
    ) -> None:
        """Initialize a Pyzfn group from a given Zarr store.

        Args:
            store (str | Path): The Zarr store to back the group. Most commonly a string
                to a Zarr directory, i.e. "path/to/simulation.zarr".
            zarr_format (Literal[2, 3], optional): Zarr format version. Defaults to 2.

        Raises:
            FileNotFoundError: If the provided path does not exist.
            NotADirectoryError: If the provided path is not a directory.

        """
        if isinstance(store, (str, Path)):
            p = Path(store)
            if not p.exists():
                msg = f"Path '{store}' does not exist."
                raise FileNotFoundError(msg)
            if not p.is_dir():
                msg = f"Path '{store}' is not a directory."
                raise NotADirectoryError(msg)
        
        # Open the zarr group
        self._group = zarr.open_group(store, mode='r')
        
        # Store clean path
        path_str = str(store) if isinstance(store, Path) else store
        self.clean_path: str = path_str.replace("file://", "")
    
    # Delegate all zarr Group methods to self._group
    def __getitem__(self, key):
        return self._group[key]
    
    def __contains__(self, key):
        return key in self._group
    
    def __iter__(self):
        return iter(self._group)
    
    def __len__(self):
        return len(self._group)

    def __delitem__(self, key: str) -> None:
        """Delete a dataset or group from the Zarr store."""
        del self._group[key]
    
    def keys(self):
        return self._group.keys()
    
    def values(self):
        return self._group.values()
    
    def items(self):
        return self._group.items()

    def members(self, max_depth: int | None = None):
        """Return (name, node) pairs for all immediate members of the group.

        Compatible with both Zarr v2 and v3.
        """
        if hasattr(self._group, "members"):
            return self._group.members() if max_depth is None else self._group.members(max_depth=max_depth)
        # Zarr v3 uses .items() as the equivalent
        return list(self._group.items())

    def create_array(
        self,
        name: str,
        *,
        shape,
        chunks=None,
        dtype=None,
        overwrite: bool = True,
        **kwargs,
    ) -> zarr.Array:
        """Create a new Zarr array in the group.

        Delegates to the underlying group with compatibility for Zarr v2 and v3.
        """
        if hasattr(self._group, "create_array"):
            return self._group.create_array(
                name,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                overwrite=overwrite,
                **kwargs,
            )
        # Zarr v2 uses require_dataset / open_array
        return self._group.require_dataset(
            name,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            overwrite=overwrite,
        )
    
    @property
    def attrs(self):
        return self._group.attrs
    
    @property
    def attributes(self):
        """Alias for attrs - returns dictionary of all zarr attributes."""
        return dict(self._group.attrs)
    
    @property
    def name(self):
        return self._group.name
    
    @property
    def store(self):
        return self._group.store

    calc_modes = inner_calc_modes
    ispec = inner_ispec
    snapshot = inner_snapshot

    def __repr__(self) -> str:
        """Return a string representation of the Pyzfn group.

        Returns:
            str: String representation of the Pyzfn group.

        """
        return f"Pyzfn('{self.name}')"

    def __str__(self) -> str:
        """Return a string representation of the Pyzfn group.

        Returns:
            str: String representation of the Pyzfn group.

        """
        return f"Pyzfn('{self.name}')"

    @property
    def path(self) -> str:
        """Full filesystem path of the Zarr store.

        Returns:
            str: Absolute file path with 'file://' prefix if available.

        """
        # In Zarr v3, use metadata.store_path if available
        if hasattr(self._group, 'metadata') and hasattr(self._group.metadata, 'store_path'):
            return str(self._group.metadata.store_path)
        # Fallback to clean_path
        return self.clean_path

    @property
    def p(self) -> None:
        """Print a visual tree representation of the Zarr group and its datasets.

        This method uses the rich library to display the hierarchical structure
        of the group, including subgroups and arrays, with their shapes and dtypes.

        """

        def add_to_tree(node_group, tree_node: Tree, prefix: str = "") -> None:
            for key, node in sorted(self.members() if node_group is self else list(node_group.items())):
                full_key = f"{prefix}/{key}" if prefix else key
                if isinstance(node, Group):
                    label = f"[bold]{key}[/bold]"
                    new_tree = tree_node.add(label)
                    add_to_tree(node, new_tree, full_key)
                else:
                    shape = getattr(node, "shape", "?")
                    dtype = getattr(node, "dtype", "?")
                    tree_node.add(f"[bold]{key}[/bold] {shape} {dtype}")

        tree = Tree(f"[bold]{self.name}[/bold]")
        add_to_tree(self, tree)
        Console().print(tree)

    def rm(self, name: str) -> None:
        """Remove a dataset or group from the Zarr store.

        Args:
            name (str): Name of the dataset or group to remove.

        Raises:
            KeyError: If the specified name does not exist in the group.
            ValueError: If ``name`` would escape the group directory.

        """
        # Security: reject paths that escape the base directory
        import shutil
        base = Path(self.clean_path).resolve()
        target = (base / name).resolve()
        if not str(target).startswith(str(base) + "/") and target != base:
            msg = f"Refusing to delete path outside job directory: {name!r}"
            raise ValueError(msg)

        if name not in self:
            msg = f"Dataset '{name}' not found in group '{self.name}'."
            raise KeyError(msg)
        del self[name]
        # Also clean up the filesystem entry if it still exists
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)

    def add_ndarray(
        self,
        name: str,
        data: NDArray[T],
        chunks: tuple[int, ...] | Literal["auto"] = "auto",
        *,
        overwrite: bool = True,
    ) -> zarr.Array:
        """Add a NumPy array to the Zarr group as a new dataset.

        Args:
            name (str): Name of the dataset to create.
            data (NDArray[T]): NumPy array to store.
            chunks (tuple[int, ...] | Literal["auto"], optional): Chunk shape or
                "auto". Defaults to "auto".
            overwrite (bool, optional): Overwrite existing dataset if it exists.
                Defaults to True.

        Returns:
            zarr.Array: The created Zarr array.

        """
        dset = self.create_array(
            name=name,
            shape=data.shape,
            chunks=chunks,
            dtype=data.dtype,
            overwrite=overwrite,
            shards=None,
        )
        dset[:] = data
        return dset

    def get_mode(self, dset_str: str, f: float) -> NDArray[np.complex64]:
        """Retrieve a specific mode from the dataset.

        Args:
            dset_str (str): Dataset name to retrieve modes from.
            f (float): Frequency to select the mode.

        Returns:
            NDArray[np.complex64]: The selected mode as a complex64 NumPy array.

        """
        freqs: NDArray[np.float64] = self.g(f"modes/{dset_str}/freqs")
        fi = int((np.abs(freqs - f)).argmin())
        return np.array(self.get_array(f"modes/{dset_str}/arr")[fi], dtype=np.complex64)

    def get_array(self, name: str) -> zarr.Array:
        """Retrieve a Zarr array by name.

        Args:
            name (str): Name of the dataset to retrieve.

        Returns:
            zarr.Array: The requested Zarr array.

        Raises:
            KeyError: If the dataset does not exist.
            TypeError: If the dataset is not a Zarr array.

        """
        if name not in self:
            msg = f"Dataset '{name}' not found in group '{self.name}'."
            raise KeyError(msg)
        array = self[name]
        if not isinstance(array, zarr.Array):
            msg = "Array must be a zarr array not a Group"
            raise TypeError(msg)
        return array

    def g(
        self,
        dset_in_str: str,
        slices: SliceTuple | slice | None = None,
    ) -> NDArray[T]:
        """Retrieve a sliced view of a dataset as a NumPy array.

        Args:
            dset_in_str (str): Name of the dataset to retrieve from the Zarr group.
            slices (slice | tuple[slice, ...], optional): Slice or tuple of slices to
                apply. Defaults to a full slice of 5 dimensions (i.e.,
                `slice(None)` * 5). Tip: use np.s_ to create complex slices.

        Returns:
            NDArray[T]: A NumPy array containing the sliced data, preserving the
                original dtype.

        Raises:
            ValueError: If too many slices are provided for the array's dimensions.

        """
        arr = self.get_array(dset_in_str)
        if slices is None:
            return np.asarray(arr[:])

        if isinstance(slices, slice):
            slice_tuple: tuple[slice, ...] = (slices,)
        else:
            slice_tuple = slices

        if isinstance(slice_tuple, tuple) and len(slice_tuple) > arr.ndim:
            msg = (
                f"Too many slices: got {len(slice_tuple)} for "
                f"{arr.ndim}-dimensional array."
            )
            raise ValueError(
                msg,
            )
        return np.asarray(arr[slice_tuple])
