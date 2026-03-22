import os
import glob
import json
import shutil
import warnings
import zarr
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from ..pyzfn import Pyzfn
from ..pyzfn.h5_backend import detect_h5_quantities
from ..cli.logging_config import get_mmpp_logger
from .constants import SPECIAL_ATTRS, ArraySlice, npf32, npc64, np1d, np2d, np3d, np4d, np5d, np4dc, RICH_AVAILABLE, FFT_AVAILABLE
from .attributes import AttributesView
from .dataset import DatasetAwareWrapper
from .dataset_geometry import AxisGeometry, DatasetGeometry

if RICH_AVAILABLE:
    from rich.console import Console
    from rich.syntax import Syntax

log = get_mmpp_logger("mmpp")


class _TreeDisplay:
    """Wrapper for zarr tree string that renders nicely in Jupyter and terminal."""

    def __init__(self, tree_str: str) -> None:
        self._tree = tree_str

    def __repr__(self) -> str:  # plain text (print / terminal)
        return self._tree

    def __str__(self) -> str:
        return self._tree

    def _repr_html_(self) -> str:  # rich Jupyter rendering
        import html
        escaped = html.escape(self._tree)
        return (
            "<pre style='"
            "font-family:monospace;font-size:0.85em;line-height:1.4;"
            "background:#0f172a;color:#e2e8f0;"
            "border:1px solid #334155;border-radius:8px;"
            "padding:12px 16px;overflow:auto;max-height:500px;"
            "margin:6px 0;'"
            f">{escaped}</pre>"
        )


@dataclass
class ScanResult:
    """Data class for storing scan results from a single zarr folder."""

    path: str
    attributes: dict[str, Any]
    error: Optional[str] = None


class ZarrJobResult:
    """Enhanced zarr job result with integrated Pyzfn functionality."""

    def __init__(self, path: str, attributes: dict[str, Any]):
        """
        Initialize ZarrJobResult with path and attributes.

        Parameters:
        -----------
        path : str
            Path to the zarr folder
        attributes : Dict[str, Any]
            Metadata attributes
        """
        self.path = path
        self.attributes = attributes
        self._mmpp_ref = None
        self._z = None
        self._h5_groups: dict | None = None
        self._path_obj = None
        self._name = None
        self._analyze = None
        self._mock_data = None

    def _build_mock_data_wrapper(self) -> DatasetAwareWrapper:
        """Create a deterministic in-memory dataset for plotting/debug parity work.

        The geometry follows MuMax3-style positive coordinates: the domain
        starts at the lower-left-bottom cell corner instead of being centred
        around the origin.
        """
        a, b, c = 5e-9, 3e-9, 2e-9
        cell = np.array((0.5e-9, 0.5e-9, 0.5e-9), dtype=float)
        pmin = np.array((0.0, 0.0, 0.0), dtype=float)
        pmax = np.array((2.0 * a, 2.0 * b, 2.0 * c), dtype=float)
        counts = np.rint((pmax - pmin) / cell).astype(int)

        xs = pmin[0] + (np.arange(counts[0], dtype=float) + 0.5) * cell[0]
        ys = pmin[1] + (np.arange(counts[1], dtype=float) + 0.5) * cell[1]
        zs = pmin[2] + (np.arange(counts[2], dtype=float) + 0.5) * cell[2]
        zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")

        x_rel = xx - a
        y_rel = yy - b
        z_rel = zz - c

        mask = ((x_rel / a) ** 2 + (y_rel / b) ** 2 + (z_rel / c) ** 2) <= 1.0
        volume_zyxc = np.stack(
            (-1e9 * y_rel, 1e9 * x_rel, 1e9 * z_rel),
            axis=-1,
        ).astype(np.float32)
        volume_zyxc[~mask] = 0.0
        data = volume_zyxc[np.newaxis, ...]  # (t, z, y, x, c)

        geometry = DatasetGeometry(
            shape=tuple(int(v) for v in data.shape),
            spatial_axes={"x": 3, "y": 2, "z": 1},
            axes={
                "x": AxisGeometry(
                    axis="x",
                    name="x",
                    index=3,
                    size=int(counts[0]),
                    min_m=float(pmin[0]),
                    max_m=float(pmax[0]),
                    cell_m=float(cell[0]),
                ),
                "y": AxisGeometry(
                    axis="y",
                    name="y",
                    index=2,
                    size=int(counts[1]),
                    min_m=float(pmin[1]),
                    max_m=float(pmax[1]),
                    cell_m=float(cell[1]),
                ),
                "z": AxisGeometry(
                    axis="z",
                    name="z",
                    index=1,
                    size=int(counts[2]),
                    min_m=float(pmin[2]),
                    max_m=float(pmax[2]),
                    cell_m=float(cell[2]),
                ),
            },
        )

        wrapper = DatasetAwareWrapper(
            self,
            "mock_data",
            data,
            geometry_override=geometry,
        )
        wrapper.attrs = {
            "dx": float(cell[0]),
            "dy": float(cell[1]),
            "dz": float(cell[2]),
            "pmin": tuple(float(v) for v in pmin),
            "pmax": tuple(float(v) for v in pmax),
            "x_name": "x",
            "y_name": "y",
            "z_name": "z",
        }
        return wrapper

    def _ensure_zarr_loaded(self) -> None:
        """Lazy load zarr group when needed."""
        if self._z is None:
            if not os.path.exists(self.path):
                raise FileNotFoundError(f"Path Not Found : '{self.path}'")

            z = zarr.open(self.path)
            if not isinstance(z, zarr.Group):
                raise TypeError(f"Path is not a zarr group : '{self.path}'")
            self._z = z
            self._path_obj = Path(self.path).absolute()
            self._name = self._path_obj.name.replace(self._path_obj.suffix, "")
            # Detect H5-backed quantities (amumax H5 storage mode)
            self._h5_groups = detect_h5_quantities(self.path)

    def _get_zarr_member(self, key: str) -> Union[zarr.Array, zarr.Group]:
        """Safely retrieve a dataset or subgroup from the underlying zarr store.

        H5-backed quantities (from amumax H5 storage mode) are checked first.
        """
        self._ensure_zarr_loaded()
        # Check H5-backed quantities first
        if self._h5_groups and key in self._h5_groups:
            return self._h5_groups[key]
        try:
            return self._z[key]
        except KeyError as exc:
            raise NameError(
                f"{self.path}: The dataset `{key}` does not exist."
            ) from exc
        except json.JSONDecodeError as exc:  # type: ignore[attr-defined]
            log.error(
                "Failed to decode metadata for '%s' in '%s': %s", key, self.path, exc
            )
            raise ValueError(
                f"{self.path}: Failed to decode zarr metadata for `{key}`. "
                "The store may contain corrupted or non-Zarr objects."
            ) from exc

    @property
    def z(self) -> zarr.Group:
        """Get the zarr group (lazy loaded)."""
        self._ensure_zarr_loaded()
        return self._z

    @property
    def name(self) -> str:
        """Get the name of the zarr folder."""
        if self._name is None:
            self._path_obj = Path(self.path).absolute()
            self._name = self._path_obj.name.replace(self._path_obj.suffix, "")
        return self._name

    @property
    def script(self) -> Optional["Syntax"]:
        """
        Check if there's a .mx3* file in the parent directory with the same name as the zarr simulation.
        If found, return syntax-highlighted content using rich.

        Returns:
            Optional[Syntax]: Syntax-highlighted script or None if no file found
        """
        try:
            # Get the zarr path and name
            zarr_path = self.path

            # Get zarr filename without extension
            zarr_filename = os.path.basename(zarr_path)
            base_name = zarr_filename.replace(".zarr", "")

            # Go to parent directory
            parent_dir = os.path.dirname(zarr_path)

            # Search for .mx3* file with the same name
            mx3_pattern = os.path.join(parent_dir, f"{base_name}.mx3*")
            mx3_files = glob.glob(mx3_pattern)

            if not mx3_files:
                log.info(f"No .mx3 file found for simulation {base_name}")
                return None

            # Take the first matching file
            mx3_file = mx3_files[0]

            # Read file content
            with open(mx3_file, encoding="utf-8") as f:
                mx3_content = f.read()

            # Create syntax-highlighted script
            if RICH_AVAILABLE and Syntax:
                syntax = Syntax(mx3_content, "go", theme="monokai", line_numbers=True)
                return syntax
            return None

        except (FileNotFoundError, PermissionError, OSError) as e:
            log.debug(f"Script file not found or not accessible: {str(e)}")
            return None
        except UnicodeDecodeError as e:
            log.error(f"Error decoding script file: {str(e)}")
            return None
        except ImportError:
            log.warning("Rich library not available for syntax highlighting")
            return None

    def display_script(self) -> None:
        """
        Display syntax-highlighted .mx3 script in console.
        """
        script = self.script
        if script:
            if RICH_AVAILABLE and Console:
                console = Console()
                console.print(script)
        else:
            log.warning("No script found to display.")




    def __getitem__(self, item: str) -> Union[zarr.Array, zarr.Group]:
        """Get zarr dataset or group by key.
        
        Prioritizes datasets over attributes. If a dataset with the given name
        exists, it will be returned. Use job[i].attrs[key] for attribute access.
        """
        self._ensure_zarr_loaded()
        try:
            member = self._get_zarr_member(item)
            if isinstance(member, zarr.Array):
                return DatasetAwareWrapper(self, item, member)
            return member
        except NameError:
            if item in self._z.attrs:
                warnings.warn(
                    f"Accessing zarr attribute '{item}' via job[i]['{item}'] is deprecated; "
                    f"use job[i].attrs['{item}'] instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return self._z.attrs[item]
            raise

    def __setitem__(self, key: str, value: str) -> None:
        """Set zarr dataset or attribute."""
        self._ensure_zarr_loaded()
        self._z[key] = value

    def __getattr__(
        self, name: str
    ) -> Union[zarr.Array, zarr.Group, int, float, str, "DatasetAwareWrapper"]:
        """Get zarr attribute or dataset by name."""
        if name.startswith("_") or name in ["path", "attributes"]:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        self._ensure_zarr_loaded()

        # Expose selected important attributes directly
        if name in SPECIAL_ATTRS:
            if name in self._z.attrs:
                return self._z.attrs[name]
            if name in self.attributes:
                return self.attributes[name]
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        # Try dataset or subgroup first
        try:
            zarr_item = self._get_zarr_member(name)
        except NameError:
            zarr_item = None

        if zarr_item is not None:
            if isinstance(zarr_item, zarr.Array):
                return DatasetAwareWrapper(self, name, zarr_item)
            return zarr_item

        # Backward-compatible access to attributes (deprecated)
        if name in self._z.attrs:
            warnings.warn(
                f"Accessing zarr attribute '{name}' via attribute access is deprecated; "
                f"use job[i].attrs['{name}'] instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return self._z.attrs[name]

        # Not found anywhere
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    @property
    def attrs(self):
        """Direct access to zarr group attributes."""
        self._ensure_zarr_loaded()
        return AttributesView(self._z.attrs)

    def __contains__(self, key: str) -> bool:
        """Return True if key exists as dataset/group, attribute, or H5 quantity."""
        self._ensure_zarr_loaded()
        try:
            if self._h5_groups and key in self._h5_groups:
                return True
            return key in self._z or key in self._z.attrs
        except Exception:
            return False

    @property
    def datasets(self) -> list[str]:
        """List top-level array datasets in this zarr group (including H5-backed)."""
        self._ensure_zarr_loaded()
        names = set(self._z.array_keys())
        if self._h5_groups:
            names.update(self._h5_groups.keys())
        return sorted(names)

    @property
    def mock_data(self) -> DatasetAwareWrapper:
        """Hidden in-memory debug dataset for plotting parity checks.

        This helper is intentionally not stored in the zarr tree and therefore
        does not appear in ``job.p`` or ``job.datasets``. It is available only
        through explicit access: ``job.mock_data``.
        """
        if self._mock_data is None:
            self._mock_data = self._build_mock_data_wrapper()
        return self._mock_data

    def keys(self) -> list[str]:
        """List top-level members (datasets, groups, and H5 quantities)."""
        self._ensure_zarr_loaded()
        names = set(self._z.keys())
        if self._h5_groups:
            names.update(self._h5_groups.keys())
        return sorted(names)

    def has_dataset(self, name: str) -> bool:
        """Check if a dataset (zarr.Array) with given name exists.
        
        Parameters
        ----------
        name : str
            Name of the dataset to check
            
        Returns
        -------
        bool
            True if dataset exists, False otherwise
            
        Example
        -------
        >>> job[0].has_dataset('alpha')  # Check if 'alpha' is a dataset
        True
        >>> job[0].has_dataset('nonexistent')
        False
        """
        self._ensure_zarr_loaded()
        try:
            member = self._z[name]
            return isinstance(member, zarr.Array)
        except KeyError:
            return False

    def has_attr(self, name: str) -> bool:
        """Check if an attribute with given name exists in zarr attrs.
        
        Parameters
        ----------
        name : str
            Name of the attribute to check
            
        Returns
        -------
        bool
            True if attribute exists, False otherwise
            
        Example
        -------
        >>> job[0].has_attr('alpha')  # Check if 'alpha' is an attribute
        True
        >>> job[0].has_attr('dx')
        True
        """
        self._ensure_zarr_loaded()
        return name in self._z.attrs

    def __repr__(self) -> str:
        return f"ZarrJobResult('{self.name}')"

    def __str__(self) -> str:
        return f"ZarrJobResult('{self.name}')"

    @property
    def pp(self):
        """Pretty print the zarr tree interactively using rich console.
        
        This displays a nicely formatted tree structure of the zarr group.
        Best used in interactive Python/IPython/Jupyter sessions.
        
        Example
        -------
        >>> job[0].pp  # Prints tree to console
        /
        ├── m_layer13 (443, 1, 94, 7520, 3) float32
        ├── m_layer14 (443, 1, 94, 7520, 3) float32
        └── table
            ├── t (3160,) float64
            └── step (3160,) float64
        """
        self._ensure_zarr_loaded()
        tree = self._z.tree()
        # Print using rich console for better formatting
        if RICH_AVAILABLE and Console:
            console = Console()
            console.print(f"[bold cyan]{self.name}[/bold cyan]")
            console.print(tree)
        else:
            print(f"{self.name}")
            print(tree)
        # Return None to avoid double printing in REPL
        return None

    @property
    def p(self):
        """Return the zarr tree as a string representation.
        
        Use this when you need the tree as a string (e.g., for logging or saving).
        For interactive display, use .pp instead.
        
        Returns
        -------
        str
            String representation of the zarr tree structure
            
        Example
        -------
        >>> tree_str = job[0].p
        >>> print(tree_str)
        """
        self._ensure_zarr_loaded()
        return _TreeDisplay(str(self._z.tree()))

    def rm(self, dset: str) -> None:
        """
        Remove a group or dataset.

        Parameters:
        -----------
        dset : str
            Name of dataset or group to remove
        """
        shutil.rmtree(f"{self.path}/{dset}", ignore_errors=True)

    def is_finished(self) -> bool:
        """Check if simulation is finished."""
        self._ensure_zarr_loaded()
        end_time: str = self._z.attrs.get("end_time", "")
        return end_time != ""

    def is_running(self) -> bool:
        """Check if simulation is still running."""
        return not self.is_finished()

    def mkdir(self, name: str) -> None:
        """
        Create nested directories.

        Parameters:
        -----------
        name : str
            Directory path to create
        """
        os.makedirs(f"{self.path}/{name}", exist_ok=True)

    def get_raw(
        self, dset: str, slices: ArraySlice = slice(None)
    ) -> Union[zarr.Array, np.ndarray]:
        """
        Get raw zarr dataset or data using direct indexing.
        Handles datasets with special characters (like minus) in names.
        Also supports H5-backed quantities from amumax H5 storage mode.

        Parameters:
        -----------
        dset : str
            Dataset name (can contain special characters)
        slices : ArraySlice, optional
            Array slicing specification (default: all data)

        Returns:
        --------
        Union[zarr.Array, np.ndarray]
            Raw zarr dataset or numpy array if sliced

        Example:
        --------
        # For dataset names with special characters like "m_z5-8"
        data = result.get_raw("m_z5-8")[:]
        # or with slicing
        data = result.get_raw("m_z5-8", slice(0, 100))
        """
        self._ensure_zarr_loaded()
        # Check H5-backed quantities first
        if self._h5_groups and dset in self._h5_groups:
            dataset = self._h5_groups[dset]
            if slices == slice(None):
                return dataset
            else:
                return dataset[slices]
        try:
            # Direct access using zarr indexing
            dataset = self._z[dset]
            if slices == slice(None):
                return dataset
            else:
                return dataset[slices]
        except KeyError as e:
            raise NameError(f"{self.path}: The dataset `{dset}` does not exist.") from e

    def get_raw_data(self, dset: str, slices: ArraySlice = slice(None)) -> np.ndarray:
        """
        Get raw data as numpy array from dataset with special characters.

        Parameters:
        -----------
        dset : str
            Dataset name (can contain special characters)
        slices : ArraySlice, optional
            Array slicing specification (default: all data)

        Returns:
        --------
        np.ndarray
            Numpy array with original dtype
        """
        dataset = self.get_raw(dset)
        return np.asarray(dataset[slices])

    def get_raw_f32(self, dset: str, slices: ArraySlice = slice(None)) -> np.ndarray:
        """
        Get raw data as float32 array from dataset with special characters.

        Parameters:
        -----------
        dset : str
            Dataset name (can contain special characters)
        slices : ArraySlice, optional
            Array slicing specification (default: all data)

        Returns:
        --------
        npf32
            Float32 numpy array
        """
        return np.asarray(self.get_raw(dset, slices), dtype=np.float32)

    def get_raw_c64(self, dset: str, slices: ArraySlice = slice(None)) -> npc64:
        """
        Get raw data as complex64 array from dataset with special characters.

        Parameters:
        -----------
        dset : str
            Dataset name (can contain special characters)
        slices : ArraySlice, optional
            Array slicing specification (default: all data)

        Returns:
        --------
        npc64
            Complex64 numpy array
        """
        return np.asarray(self.get_raw(dset, slices), dtype=np.complex64)

    def list_datasets(self) -> list[str]:
        """
        List all available datasets in the zarr group.
        Useful for finding datasets with special characters.

        Returns:
        --------
        List[str]
            List of dataset names
        """
        self._ensure_zarr_loaded()
        datasets = []

        def collect_datasets(group, prefix=""):
            for key in group.keys():
                full_key = f"{prefix}{key}" if prefix else key
                item = group[key]
                if isinstance(item, zarr.Array):
                    datasets.append(full_key)
                elif isinstance(item, zarr.Group):
                    collect_datasets(item, f"{full_key}/")

        collect_datasets(self._z)
        return datasets

    def find_datasets(self, pattern: str) -> list[str]:
        """
        Find datasets matching a pattern (supports wildcards).

        Parameters:
        -----------
        pattern : str
            Pattern to match (supports * and ? wildcards)

        Returns:
        --------
        List[str]
            List of matching dataset names
        """
        import fnmatch

        datasets = self.list_datasets()
        return [dset for dset in datasets if fnmatch.fnmatch(dset, pattern)]

    def get_dset(self, dset: str) -> zarr.Array:
        """
        Get zarr dataset.

        Parameters:
        -----------
        dset : str
            Dataset name

        Returns:
        --------
        zarr.Array
            The zarr dataset
        """
        member = self._get_zarr_member(dset)
        if isinstance(member, zarr.Group):
            raise ValueError(f"`{dset}` is a group, not a dataset.")
        return member

    def get_f32(self, dset: str, slices: ArraySlice) -> npf32:
        """
        Get float32 array from dataset.

        Parameters:
        -----------
        dset : str
            Dataset name
        slices : ArraySlice
            Array slicing specification

        Returns:
        --------
        npf32
            Float32 numpy array
        """
        return np.asarray(self.get_dset(dset)[slices], dtype=np.float32)

    def get_c64(self, dset: str, slices: ArraySlice) -> npc64:
        """
        Get complex64 array from dataset.

        Parameters:
        -----------
        dset : str
            Dataset name
        slices : ArraySlice
            Array slicing specification

        Returns:
        --------
        npc64
            Complex64 numpy array
        """
        return np.asarray(self.get_dset(dset)[slices], dtype=np.complex64)

    def get_np1d(self, dset_str: str, slices: ArraySlice) -> np1d:
        """Get 1D numpy array from dataset."""
        arr = self.get_f32(dset_str, slices)
        if arr.ndim != 1:
            raise ValueError("The dataset must be 1D")
        return arr

    def get_np2d(self, dset_str: str, slices: ArraySlice) -> np2d:
        """Get 2D numpy array from dataset."""
        arr = self.get_f32(dset_str, slices)
        if arr.ndim != 2:
            raise ValueError("The dataset must be 2D")
        return arr

    def get_np3d(self, dset_str: str, slices: ArraySlice) -> np3d:
        """Get 3D numpy array from dataset."""
        arr = self.get_f32(dset_str, slices)
        if arr.ndim != 3:
            raise ValueError("The dataset must be 3D")
        return arr

    def get_np4d(self, dset_str: str, slices: ArraySlice) -> np4d:
        """Get 4D numpy array from dataset."""
        arr = self.get_f32(dset_str, slices)
        if arr.ndim != 4:
            raise ValueError("The dataset must be 4D")
        return arr

    def get_np5d(self, dset_str: str, slices: ArraySlice) -> np5d:
        """Get 5D numpy array from dataset."""
        arr = self.get_f32(dset_str, slices)
        if arr.ndim != 5:
            raise ValueError("The dataset must be 5D")
        return arr

    def get_np4dc(self, dset_str: str, slices: ArraySlice) -> np4dc:
        """Get 4D complex array from modes dataset."""
        dset = self.get_dset(f"modes/{dset_str}/arr")
        return np.asarray(dset[slices], dtype=np.complex64)

    def _set_mmpp_ref(self, mmpp_instance) -> None:
        """Set reference to MMPP instance for plotting."""
        self._mmpp_ref = mmpp_instance

    @property
    def mpl(self):
        """Get matplotlib plotter for this single result."""
        from ..plotting import MMPPlotter
        return MMPPlotter([self], self._mmpp_ref)

    @property
    def matplotlib(self):
        """Get matplotlib plotter for this single result (alias for mpl)."""
        return self.mpl

    @property
    def get(self):
        """Access datasets with direct numpy output.
        
        Returns a NumpyGetter that provides direct numpy array access
        when slicing datasets. Unlike regular dataset access which returns
        DatasetAwareWrapper, this returns numpy arrays directly.
        
        Returns
        -------
        NumpyGetter
            Helper object for numpy-direct dataset access
        
        Example
        -------
        >>> # Direct numpy access
        >>> arr = job[0].get.m[:]  # Returns numpy.ndarray
        >>> arr = job[0].get.m[0:100, :, :, :, 0]
        >>> 
        >>> # Compare to regular access
        >>> wrapper = job[0].m[:]  # Returns DatasetAwareWrapper
        >>> arr = job[0].m[:].numpy()  # Explicit conversion
        """
        from .dataset import NumpyGetter
        return NumpyGetter(self)

    @property
    def fft(self):
        """Get FFT analyzer for this single result."""
        if not FFT_AVAILABLE:
            raise ImportError(
                "FFT functionality not available. Check fft module import."
            )
        from ..fft import FFT
        return FFT(self, self._mmpp_ref)

    @property
    def solitons(self):
        """Get soliton analyzer for this single result."""
        from ..solitons import SolitonInterface

        return SolitonInterface(self, self._mmpp_ref)

    @property
    def vortex(self):
        """Shortcut alias for ``self.solitons.vortex``."""
        return self.solitons.vortex

    @property
    def analyze(self):
        """Get unified analysis namespace for this result."""
        if self._analyze is None:
            from ..analyze import AnalyzeInterface

            self._analyze = AnalyzeInterface(self, self._mmpp_ref)
        return self._analyze

    def calculate_fft_data(self, **kwargs):
        """Direct method for FFT calculation."""
        return self.fft._compute_fft(**kwargs)

    def get_largest_m_dataset(self) -> str:
        """
        Automatically find the m dataset with the largest time dimension.

        Returns:
        --------
        str
            Name of the largest m dataset (e.g., "m_z5-8", "m-12", or fallback "m")
        """
        try:
            self._ensure_zarr_loaded()
            available_keys = list(self._z.keys())

            m_datasets = []
            for key in available_keys:
                if key.startswith("m") and not key.startswith("m_"):
                    m_datasets.append(key)
                elif key.startswith("m_"):
                    m_datasets.append(key)

            if not m_datasets:
                log.warning(f"No m datasets found in {self.path}, using fallback 'm'")
                return "m"

            largest_dataset = "m"
            largest_time_size = 0

            for dataset_name in m_datasets:
                try:
                    dataset = self._z[dataset_name]
                    if hasattr(dataset, "shape") and len(dataset.shape) >= 1:
                        time_size = dataset.shape[0]
                        if time_size > largest_time_size:
                            largest_time_size = time_size
                            largest_dataset = dataset_name
                except Exception as e:
                    log.debug(f"Could not check dataset {dataset_name}: {e}")
                    continue

            log.info(
                f"Auto-selected dataset '{largest_dataset}' with {largest_time_size} time steps"
            )
            return largest_dataset

        except Exception as e:
            log.warning(
                f"Error finding largest m dataset in {self.path}: {e}, using fallback 'm'"
            )
            return "m"
