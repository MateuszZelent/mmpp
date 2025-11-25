import glob
import json
import os
import pickle
import re
import shutil
import threading
import warnings
from uuid import uuid4
from html import escape
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union
from collections.abc import MutableMapping

import numpy as np
import pandas as pd
import zarr
from .pyzfn import Pyzfn
from rich.console import Console
from rich.syntax import Syntax

from .cli.logging_config import get_mmpp_logger, setup_mmpp_logging

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .batch_operations import BatchOperations

# Initialize rich logging - will be configured in MMPP.__init__
log = get_mmpp_logger("mmpp")

SPECIAL_ATTRS = {
    "dx",
    "dy",
    "dz",
    "Tx",
    "Ty",
    "Tz",
    "dt",
    "t_sampl",
    "fcut",
    "f_cut",
    "Nx",
    "Ny",
    "Nz",
    "cellsize_x",
    "cellsize_y",
    "cellsize_z",
    "total_time",
    "n_steps",
}

# Type aliases for numpy arrays
if TYPE_CHECKING:
    npf32 = NDArray[np.float32]
    npc64 = NDArray[np.complex64]
    np1d = NDArray[Any]
    np2d = NDArray[Any]
    np3d = NDArray[Any]
    np4d = NDArray[Any]
    np5d = NDArray[Any]
    np4dc = NDArray[np.complex64]
else:
    npf32 = np.ndarray
    npc64 = np.ndarray
    np1d = np.ndarray
    np2d = np.ndarray
    np3d = np.ndarray
    np4d = np.ndarray
    np5d = np.ndarray
    np4dc = np.ndarray
ArraySlice = Union[slice, tuple, int]

# Import for interactive display
try:
    # Import only if needed for interactive display
    ITABLES_AVAILABLE = True
except ImportError:
    ITABLES_AVAILABLE = False

try:
    from rich import print as rprint
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from IPython.display import HTML, display

    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

# Import plotting functionality
PLOTTING_AVAILABLE = True
from .plotting import MMPPlotter, PlotterProxy

# Import FFT functionality
try:
    from .fft import FFT

    FFT_AVAILABLE = True
except ImportError:
    FFT_AVAILABLE = False


@dataclass
class ScanResult:
    """Data class for storing scan results from a single zarr folder."""

    path: str
    attributes: dict[str, Any]
    error: Optional[str] = None


class AttributesView(MutableMapping):
    """Wrapper for zarr attrs with rich/Jupyter-friendly display."""

    def __init__(self, attrs: Any):
        self._attrs = attrs

    # Mapping protocol
    def __getitem__(self, key):
        return self._attrs[key]

    def __setitem__(self, key, value):
        self._attrs[key] = value

    def __delitem__(self, key):
        del self._attrs[key]

    def __iter__(self):
        return iter(self._attrs)

    def __len__(self):
        return len(self._attrs)

    def keys(self):
        return self._attrs.keys()

    def items(self):
        return self._attrs.items()

    def values(self):
        return self._attrs.values()

    def get(self, key, default=None):
        return self._attrs.get(key, default)

    def as_dict(self) -> dict[str, Any]:
        """Return attributes as a plain dict."""
        return dict(self._attrs)

    # Displays -------------------------------------------------------------
    def _rich_table(self):
        if not RICH_AVAILABLE:
            return None
        try:
            table = Table(
                title="Simulation attributes",
                show_header=True,
                header_style="bold cyan",
                box=None,
            )
            table.add_column("Key", style="magenta", no_wrap=True)
            table.add_column("Value", style="green")
            for key in sorted(self._attrs.keys()):
                val = self._attrs[key]
                table.add_row(str(key), repr(val))
            return table
        except Exception:
            return None

    def __repr__(self) -> str:
        # Try rich text table for terminals
        if RICH_AVAILABLE:
            try:
                console = Console(width=120, force_terminal=True, color_system="auto")
                table = self._rich_table()
                if table is not None:
                    console.print(table)
                    return console.export_text()
            except Exception:
                pass

        # Fallback plain text
        lines = ["Simulation attributes:"]
        for key in sorted(self._attrs.keys()):
            lines.append(f"- {key}: {self._attrs[key]!r}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.__repr__()

    def _repr_html_(self):
        """Interactive-ish HTML table for Jupyter."""
        try:
            table_id = f"attrs-{uuid4().hex}"
            header = """
            <style>
              table.attrs-table { border-collapse: collapse; width: 100%; }
              table.attrs-table th, table.attrs-table td { border: 1px solid #ddd; padding: 6px; font-family: monospace; font-size: 12px; }
              table.attrs-table th { background: #f4f4f4; cursor: pointer; text-align: left; }
              table.attrs-table tr:nth-child(even) { background: #fafafa; }
            </style>
            """
            rows = []
            for key in sorted(self._attrs.keys()):
                val = self._attrs[key]
                rows.append(
                    f"<tr><td>{escape(str(key))}</td><td><pre style='margin:0'>{escape(repr(val))}</pre></td></tr>"
                )
            body = "\n".join(rows)
            html_table = f"""
            {header}
            <table id="{table_id}" class="attrs-table">
              <thead>
                <tr><th>Key</th><th>Value</th></tr>
              </thead>
              <tbody>
                {body}
              </tbody>
            </table>
            <script>
              (function() {{
                const table = document.getElementById("{table_id}");
                const getCell = (tr, idx) => tr.children[idx].innerText || "";
                const comparer = (idx, asc) => (a, b) => ((v1, v2) => {{
                  const n1 = parseFloat(v1); const n2 = parseFloat(v2);
                  if(!isNaN(n1) && !isNaN(n2)) return n1 - n2;
                  return v1.localeCompare(v2);
                }})(getCell(asc ? a : b, idx), getCell(asc ? b : a, idx));
                table.querySelectorAll("th").forEach((th, idx) => {{
                  let asc = true;
                  th.addEventListener("click", () => {{
                    const tbody = table.querySelector("tbody");
                    Array.from(tbody.querySelectorAll("tr"))
                      .sort(comparer(idx, asc = !asc))
                      .forEach(tr => tbody.appendChild(tr));
                  }});
                }});
              }})();
            </script>
            """
            return html_table
        except Exception:
            return None


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
        self._path_obj = None
        self._name = None

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

    def _get_zarr_member(self, key: str) -> Union[zarr.Array, zarr.Group]:
        """Safely retrieve a dataset or subgroup from the underlying zarr store."""
        self._ensure_zarr_loaded()
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
    def script(self) -> Optional[Syntax]:
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
            syntax = Syntax(mx3_content, "go", theme="monokai", line_numbers=True)

            return syntax

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
        """Return True if key exists as dataset/group or attribute."""
        self._ensure_zarr_loaded()
        try:
            return key in self._z or key in self._z.attrs
        except Exception:
            return False

    @property
    def datasets(self) -> list[str]:
        """List top-level array datasets in this zarr group."""
        self._ensure_zarr_loaded()
        return sorted(list(self._z.array_keys()))

    def keys(self) -> list[str]:
        """List top-level members (datasets and groups) in this zarr group."""
        self._ensure_zarr_loaded()
        return list(self._z.keys())

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
        """Pretty print the zarr tree."""
        self._ensure_zarr_loaded()
        return self._z.tree()

    @property
    def p(self) -> None:
        """Print the zarr tree."""
        log.info(f"Database: {self.name}, Structure: {self.z.tree()}")

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
        dset_tmp = self[dset]
        if isinstance(dset_tmp, zarr.Group):
            raise ValueError(f"`{dset}` is a group, not a dataset.")
        return dset_tmp

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

    def _set_mmpp_ref(self, mmpp_instance: "MMPP") -> None:
        """Set reference to MMPP instance for plotting."""
        self._mmpp_ref = mmpp_instance

    @property
    def mpl(self) -> "MMPPlotter":
        return MMPPlotter([self], self._mmpp_ref)

    @property
    def matplotlib(self) -> "MMPPlotter":
        """Get matplotlib plotter for this single result (alias for mpl)."""
        return self.mpl

    @property
    def fft(self) -> "FFT":
        """Get FFT analyzer for this single result."""
        if not FFT_AVAILABLE:
            raise ImportError(
                "FFT functionality not available. Check fft module import."
            )
        if self._mmpp_ref is None:
            raise ValueError("MMPP reference not set. Use results from MMPP instance.")
        return FFT(self, self._mmpp_ref)

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
        # Import here to avoid circular import
        from .plotting import _find_largest_m_dataset

        return _find_largest_m_dataset(self.path)


# Dataset utilities
def find_largest_m_dataset(zarr_path: str) -> str:
    """
    Automatically find the m dataset with the largest time dimension.

    Parameters:
    -----------
    zarr_path : str
        Path to zarr file

    Returns:
    --------
    str
        Name of the largest m dataset (e.g., "m_z5-8", "m-12", or fallback "m")
    """
    try:
        job = Pyzfn(zarr_path)

        # Get all available datasets that start with "m"
        m_datasets = []

        # Access zarr file directly to get dataset keys
        import zarr

        try:
            zarr_file = zarr.open(zarr_path, mode="r")
            available_keys = list(zarr_file.keys())
            log.debug(f"Available keys in zarr file: {available_keys}")
        except Exception as e:
            log.debug(f"Could not access zarr file directly: {e}")
            available_keys = []

        for key in available_keys:
            if key.startswith("m") and not key.startswith("m_"):
                # Include base "m" dataset
                m_datasets.append(key)
            elif key.startswith("m_"):
                # Include cropped datasets like "m_z5-8", "m-12"
                m_datasets.append(key)

        if not m_datasets:
            log.warning(f"No m datasets found in {zarr_path}, using fallback 'm'")
            return "m"

        # Find dataset with largest time dimension
        largest_dataset = "m"
        largest_time_size = 0

        for dataset_name in m_datasets:
            try:
                # Use zarr file direct access instead of job.z
                dataset = zarr_file[dataset_name]
                if hasattr(dataset, "shape") and len(dataset.shape) >= 1:
                    time_size = dataset.shape[0]  # First dimension is usually time
                    log.debug(f"Dataset {dataset_name}: time size = {time_size}")

                    if time_size > largest_time_size:
                        largest_time_size = time_size
                        largest_dataset = dataset_name
            except (KeyError, IndexError, ValueError) as e:
                log.debug(f"Could not check dataset {dataset_name}: {e}")
                continue
            except Exception as e:
                log.debug(f"Unexpected error checking dataset {dataset_name}: {e}")
                continue

        log.info(
            f"Auto-selected dataset '{largest_dataset}' with {largest_time_size} time steps"
        )
        return largest_dataset

    except OSError as e:
        log.warning(f"File system error accessing {zarr_path}: {e}, using fallback 'm'")
        return "m"
    except ImportError as e:
        log.warning(f"Missing zarr dependency: {e}, using fallback 'm'")
        return "m"
    except Exception as e:
        log.error(
            f"Unexpected error finding largest m dataset in {zarr_path}: {e}, using fallback 'm'"
        )
        return "m"


# FFT availability check - moved here for DatasetSpecificFFT
try:
    from mmpp.fft.core import FFT

    FFT_AVAILABLE = True
except ImportError:
    FFT_AVAILABLE = False


class DatasetSpecificTransmissionInterface:
    """Wrapper for transmission interface that injects dataset_name."""

    def __init__(self, transmission_interface, dataset_name):
        self._transmission = transmission_interface
        self._dataset_name = dataset_name

    def __call__(self, config=None, /, **kwargs):
        """Inject dataset_name into config if not provided."""
        if config is None:
            # Add dataset_name to kwargs if not already present
            if "dataset_name" not in kwargs:
                kwargs["dataset_name"] = self._dataset_name
            return self._transmission(config, **kwargs)
        else:
            # Config object provided - check if dataset_name needs to be set
            if config.dataset_name is None:
                # Create a new config with dataset_name set
                from mmpp.fft.transmission import TransmissionConfig
                from dataclasses import asdict, replace

                config = replace(config, dataset_name=self._dataset_name)
            return self._transmission(config, **kwargs)

    def compute(self, config=None, /, **kwargs):
        """Alias for __call__."""
        return self.__call__(config, **kwargs)

    def plot_transmission(self, config=None, plot_config=None, **kwargs):
        """Inject dataset_name and call plot_transmission."""
        if config is None and "dataset_name" not in kwargs:
            kwargs["dataset_name"] = self._dataset_name
        return self._transmission.plot_transmission(config, plot_config, **kwargs)

    def __repr__(self):
        """Delegate to original transmission interface."""
        return repr(self._transmission)

    def __getattr__(self, name):
        """Delegate all other attributes to original transmission."""
        return getattr(self._transmission, name)


class DatasetSpecificFFT:
    """FFT wrapper with pre-set dataset"""

    def __init__(self, job_result, dataset_name, mmpp_instance=None, slice_info=None):
        self.dataset_name = dataset_name
        self.slice_info = slice_info
        # Create regular FFT instance
        self._fft = FFT(job_result, mmpp_instance)

    def __getattr__(self, name):
        """Delegate to FFT, injecting dataset context when appropriate."""
        attr = getattr(self._fft, name)

        if name == "dispersion" and attr is not None:
            return attr.clone_for_dataset(self.dataset_name, slice_info=self.slice_info)

        if name == "transmission" and attr is not None:
            return attr.clone_for_dataset(self.dataset_name, slice_info=self.slice_info)

        if callable(attr) and hasattr(attr, "__code__"):
            import inspect

            sig = inspect.signature(attr)
            if "dataset_name" in sig.parameters:

                def wrapper(*args, **kwargs):
                    if "dataset_name" not in kwargs:
                        kwargs["dataset_name"] = self.dataset_name
                    if (
                        self.slice_info is not None
                        and "slice_info" in sig.parameters
                        and "slice_info" not in kwargs
                    ):
                        kwargs["slice_info"] = self.slice_info
                    return attr(*args, **kwargs)

                return wrapper

        return attr

    def __repr__(self):
        """Rich documentation display for dataset-specific FFT interface."""
        try:
            return self._rich_dataset_fft_display()
        except Exception:
            return self._basic_dataset_fft_display()

    def _rich_dataset_fft_display(self) -> str:
        """Create rich documentation display for dataset-specific FFT."""
        try:
            import io
            from rich.columns import Columns
            from rich.console import Console
            from rich.panel import Panel
            from rich.syntax import Syntax
            from rich.table import Table
            from rich.text import Text

            console = Console(file=io.StringIO(), width=120, force_terminal=True)

            # Get dataset info
            try:
                dataset_shape = getattr(
                    self._fft.job_result._z[self.dataset_name], "shape", "Unknown"
                )
            except:
                dataset_shape = "Unknown"

            slice_str = f"[{self.slice_info}]" if self.slice_info else ""
            cache_size = len(self._fft._cache)

            # Summary panel content
            summary_text = Text()
            summary_text.append(
                "🎯 Dataset-Specific FFT Interface\n", style="bold cyan"
            )
            summary_text.append(
                f"📊 Dataset: {self.dataset_name}\n", style="bold green"
            )
            summary_text.append(f"📐 Shape: {dataset_shape}{slice_str}\n", style="dim")
            if self.slice_info:
                summary_text.append(f"✂️  Slicing: Active\n", style="yellow")
            summary_text.append(f"💾 Cache: {cache_size} entries\n", style="dim")

            # Core methods panel content
            core_methods_text = Text()
            core_methods_text.append(
                "🔧 FFT Methods (dataset pre-set):\n", style="bold yellow"
            )
            methods = [
                ("spectrum()", f"(freqs, spectrum) for {self.dataset_name}"),
                ("frequencies()", "Frequency array"),
                ("power()", f"|FFT|² power spectrum"),
                ("magnitude()", f"|FFT| magnitude"),
                ("phase()", "Phase spectrum"),
                ("plot_spectrum()", "Plot power spectrum"),
                ("dispersion", "Access dispersion analysis"),
            ]

            for method, desc in methods:
                core_methods_text.append("  • ", style="dim")
                core_methods_text.append(method, style="code")
                core_methods_text.append(f" - {desc}\n", style="dim")

            # Dataset-specific info panel
            dataset_info_text = Text()
            dataset_info_text.append("📋 Dataset Context:\n", style="bold magenta")
            dataset_info_text.append(
                f"  • Pre-selected: {self.dataset_name}\n", style="green"
            )
            dataset_info_text.append(
                "  • No need to specify dataset_name\n", style="dim"
            )
            if self.slice_info:
                dataset_info_text.append(
                    f"  • Sliced data: {self.slice_info}\n", style="yellow"
                )
                dataset_info_text.append(
                    "  • FFT operates on sliced data\n", style="dim"
                )

            # Usage examples
            example_code = f"""# Dataset-specific FFT usage (no dataset_name needed)
freqs, spectrum = m_layer.fft.spectrum()  # Uses '{self.dataset_name}' automatically
power = m_layer.fft.power()
frequencies = m_layer.fft.frequencies()

# Dispersion analysis
disp = m_layer.fft.dispersion
modes = disp.compute_1d()

# Plotting
fig, ax = m_layer.fft.plot_spectrum(log_scale=True)

# Compare with regular FFT (requires dataset_name)
freqs_full, regular_fft = job[0].fft.spectrum(dataset_name='{self.dataset_name}')
freqs_ds, dataset_fft = job[0].{self.dataset_name}.fft.spectrum()  # Same result!"""

            if self.slice_info:
                example_code += f"""

# Sliced dataset FFT
sliced_data = job[0].{self.dataset_name}{slice_str}
sliced_freqs, sliced_fft = sliced_data.fft.spectrum()  # FFT of sliced data
sliced_dispersion = sliced_data.fft.dispersion.compute_1d()"""

            syntax = Syntax(
                example_code, "python", theme="monokai", background_color="default"
            )

            # Build panels
            with console.capture() as capture:
                # Main summary panel
                console.print(
                    Panel.fit(
                        summary_text,
                        title=f"[bold cyan]Dataset FFT: {self.dataset_name}[/bold cyan]",
                        border_style="cyan",
                    )
                )
                console.print("")

                # Method panels side by side
                console.print(
                    Columns(
                        [
                            Panel.fit(
                                core_methods_text,
                                title="[bold yellow]Available Methods[/bold yellow]",
                                border_style="yellow",
                            ),
                            Panel.fit(
                                dataset_info_text,
                                title="[bold magenta]Dataset Context[/bold magenta]",
                                border_style="magenta",
                            ),
                        ]
                    )
                )
                console.print("")

                # Examples panel
                console.print(
                    Panel(
                        syntax,
                        title="[bold green]Usage Examples[/bold green]",
                        border_style="green",
                    )
                )

            output = capture.get()
            return output

        except ImportError:
            return self._basic_dataset_fft_display()

    def _basic_dataset_fft_display(self) -> str:
        """Fallback display without rich formatting."""
        slice_str = f"[{self.slice_info}]" if self.slice_info else ""

        return f"""DatasetSpecificFFT(dataset={self.dataset_name})

Dataset-specific FFT Interface:
• Pre-configured for dataset: {self.dataset_name}{slice_str}
• Available methods: spectrum(), power(), magnitude(), phase(), dispersion
• Usage: freq, spec = m_layer.fft.spectrum()  # No dataset_name needed!

Examples:
freqs, spectrum = m_layer.fft.spectrum()  # Auto-uses '{self.dataset_name}'
power = m_layer.fft.power()
dispersion = m_layer.fft.dispersion.compute_1d()

📖 For full documentation: help(job[0].fft)"""


class DatasetAwareWrapper:
    """Wrapper that acts like zarr.Array but has .fft property"""

    def __init__(self, job_result, dataset_name, zarr_array, slice_info=None):
        self.job_result = job_result
        self.dataset_name = dataset_name
        self.zarr_array = zarr_array
        self.slice_info = slice_info  # Store slicing information
        self._fft = None

    def _resolve_source(self):
        """Return underlying data respecting the stored slice."""
        if self.slice_info is not None:
            return self.zarr_array[self.slice_info]
        return self.zarr_array

    def __getattr__(self, name):
        """Delegate to zarr_array for most attributes (but not our own properties)"""
        # Don't delegate properties that are defined on this class
        if name in ('dt', 'fft', 'shape', 'data'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
        if self.slice_info is not None:
            # If sliced, get attribute from sliced data
            sliced_data = self._resolve_source()
            return getattr(sliced_data, name)
        return getattr(self.zarr_array, name)

    def __getitem__(self, key):
        """Return new DatasetAwareWrapper with slicing info preserved"""
        # Instead of returning raw numpy array, return new wrapper with slice info
        if self.slice_info is not None:
            # Combine existing slice with new slice - simplified for now
            combined_slice = key
        else:
            combined_slice = key

        return DatasetAwareWrapper(
            self.job_result,
            self.dataset_name,
            self.zarr_array,  # Keep original zarr reference
            slice_info=combined_slice,
        )

    @property
    def fft(self):
        """Return FFT with this dataset pre-selected"""
        if self._fft is None and FFT_AVAILABLE:
            # Create DatasetSpecificFFT with slicing info
            self._fft = DatasetSpecificFFT(
                self.job_result,
                self.dataset_name,
                getattr(self.job_result, "_mmpp_ref", None),
                slice_info=self.slice_info,
            )
        return self._fft

    @property
    def shape(self):
        """Shape accounting for slicing"""
        if self.slice_info is not None:
            sliced_data = self._resolve_source()
            return sliced_data.shape
        return self.zarr_array.shape

    @property
    def dt(self):
        """
        Get time step for this dataset.
        
        Algorithm:
        1. Check if 't_sampl' exists in job_result attrs (global)
        2. Check if 't' exists in THIS dataset's attrs and calculate dt
        3. Look for 't' array in various locations (root, table, etc.)
        4. Calculate dt = t[1] - t[0]
        
        Returns:
            float: Time step in seconds
        """
        # Method 1: Check for t_sampl in main attributes
        if hasattr(self.job_result, '_z') and self.job_result._z is not None:
            if 't_sampl' in self.job_result._z.attrs:
                return self.job_result._z.attrs['t_sampl']
        
        # Method 2: Check THIS dataset's attrs for 't' array (MOST SPECIFIC)
        if hasattr(self.job_result, '_z') and self.job_result._z is not None:
            try:
                dataset = self.job_result._z[self.dataset_name]
                if hasattr(dataset, 'attrs') and 't' in dataset.attrs:
                    t_attr = dataset.attrs['t']
                    # t_attr is a list or array in attrs
                    if hasattr(t_attr, '__len__') and len(t_attr) >= 2:
                        dt = float(t_attr[1] - t_attr[0])
                        return dt
            except (KeyError, NameError, AttributeError, IndexError, TypeError):
                pass
        
        # Method 3: Look for time array in various locations
        # Try common naming patterns and locations
        time_locations = [
            ('t',),  # Root level 't'
            ('table', 't'),  # Often in 'table' group
            ('time',),  # Alternative name
            (f't_{self.dataset_name}',),  # Dataset-specific time
        ]
        
        for location in time_locations:
            try:
                if hasattr(self.job_result, '_z'):
                    # Navigate through the location path
                    t_array = self.job_result._z
                    for key in location:
                        t_array = t_array[key]
                    
                    # Calculate dt from first two time points
                    if t_array.shape[0] >= 2:
                        dt = float(t_array[1] - t_array[0])
                        return dt
            except (KeyError, NameError, AttributeError, IndexError):
                continue
        
        # Method 4: Fallback - raise informative error
        raise AttributeError(
            f"Cannot determine time step for dataset '{self.dataset_name}'. "
            f"Neither 't_sampl' attribute nor time array 't' found in zarr file."
        )

    @property
    def data(self):
        """Return data as numpy array (loads into memory)."""
        return self.numpy(copy=False)

    def numpy(self, *, copy: bool = True, dtype=None, squeeze: bool = False):
        """Materialize the wrapped data as numpy array."""
        data = self._resolve_source()
        if isinstance(data, zarr.Array):
            data = data[:]
        array = np.array(data, copy=copy)
        if dtype is not None:
            array = array.astype(dtype, copy=copy)
        if squeeze:
            array = np.squeeze(array)
        return array

    def to_numpy(self, **kwargs):
        """Alias for numpy() to match common API naming."""
        return self.numpy(**kwargs)

    def as_zarr(self):
        """Return the underlying zarr.Array when no slicing is active."""
        if self.slice_info is not None:
            raise TypeError(
                "Sliced view has no standalone zarr representation; use numpy() instead"
            )
        return self.zarr_array

    def __array__(self, dtype=None):
        """Support implicit numpy conversions (e.g. np.asarray)."""
        array = self.numpy(copy=False)
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    def __iter__(self):
        return iter(self.numpy(copy=False))

    def __len__(self):
        return len(self.numpy(copy=False))

    def __repr__(self):
        slice_str = f"[{self.slice_info}]" if self.slice_info else ""
        return (
            f"DatasetAwareWrapper({self.dataset_name}{slice_str}, shape={self.shape})"
        )


class MMPP:
    """
    Multi-threaded scanner for zarr folders with pandas database creation and search functionality.

    This class scans directories recursively for .zarr folders, extracts metadata using Pyzfn,
    and creates a searchable pandas database.
    """

    def __init__(
        self,
        base_path: str,
        max_workers: int = 8,
        database_name: str = "mmpy_database",
        debug: bool = False,
        log_level: Optional[Union[str, int]] = None,
    ) -> None:
        """
        Initialize the MMPP.

        Parameters:
        -----------
        base_path : str
            Base directory path to scan for zarr folders OR direct path to .zarr file
        max_workers : int, optional
            Maximum number of worker threads for scanning (default: 8)
        database_name : str, optional
            Name of the database file (without extension, default: "mmpy_database")
        debug : bool, optional
            Enable debug logging (default: False)
        log_level : str or int, optional
            Set specific log level. Can be string ("DEBUG", "INFO", "WARNING", "ERROR")
            or integer constant (logging.DEBUG, logging.INFO, etc.).
            If provided, overrides debug parameter.
            Default: None (uses debug flag - DEBUG if True, INFO if False)
        """
        self.base_path: str = os.path.abspath(base_path)
        self.max_workers: int = max_workers
        self.database_name: str = database_name
        self.debug: bool = debug
        self.log_level: Optional[Union[str, int]] = (
            log_level  # Store for child components
        )
        self._lock: threading.Lock = threading.Lock()
        self._interactive_mode: bool = True  # Enable interactive mode by default
        self._single_zarr_mode: bool = False
        self._zarr_results: list[ZarrJobResult] = []

        # Configure rich logging for this instance with flexible level control
        import logging

        # Convert string log level to integer if needed
        numeric_level = None
        if log_level is not None:
            if isinstance(log_level, str):
                numeric_level = getattr(logging, log_level.upper(), None)
                if numeric_level is None:
                    raise ValueError(
                        f"Invalid log level: {log_level}. Use DEBUG, INFO, WARNING, or ERROR"
                    )
            else:
                numeric_level = log_level

        global log
        log = setup_mmpp_logging(debug=debug, logger_name="mmpp", level=numeric_level)

        # Check if base_path is a direct .zarr file
        if self.base_path.endswith(".zarr") and os.path.isdir(self.base_path):
            self._single_zarr_mode = True
            self.database_path = None
            self.dataframe = None
            self._load_single_zarr()
        else:
            # Original directory scanning mode
            self.database_path: str = os.path.join(
                self.base_path, f"{database_name}.pkl"
            )
            self.dataframe: Optional[pd.DataFrame] = None
            # Try to load existing database
            self._load_database()

        # Initialize rich console if available
        if RICH_AVAILABLE:
            self.console = Console()

    def _load_single_zarr(self) -> None:
        """Load a single .zarr file directly."""
        try:
            # Scan the single zarr file
            scan_result = self._scan_single_zarr(self.base_path)

            if scan_result.error:
                log.error(
                    f"Error loading zarr file {self.base_path}: {scan_result.error}"
                )
                return

            # Create ZarrJobResult
            result = ZarrJobResult(
                path=scan_result.path, attributes=scan_result.attributes
            )
            result._set_mmpp_ref(self)
            self._zarr_results = [result]

            log.info(f"Loaded single zarr file: {self.base_path}")

        except Exception as e:
            log.error(f"Error loading single zarr file: {e}")
            self._zarr_results = []

    def __len__(self) -> int:
        """Return number of zarr results available."""
        if self._single_zarr_mode:
            return len(self._zarr_results)
        elif self.dataframe is not None:
            return len(self.dataframe)
        else:
            return 0

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[ZarrJobResult, "BatchOperations"]:
        """
        Get zarr result by index or batch operations by slice.

        Parameters:
        -----------
        index : Union[int, slice]
            Index of the result to get or slice for batch operations

        Returns:
        --------
        Union[ZarrJobResult, BatchOperations]
            Single zarr result for integer index or batch operations for slice
        """
        # Handle slice notation for batch operations
        if isinstance(index, slice):
            # Import here to avoid circular imports
            from .batch_operations import BatchOperations

            if self._single_zarr_mode:
                results = self._zarr_results[index]
            else:
                # Database mode
                if self.dataframe is None or self.dataframe.empty:
                    raise IndexError("No database available. Run scan() first.")

                # Get slice of dataframe
                df_slice = self.dataframe.iloc[index]
                results = []
                for _, row in df_slice.iterrows():
                    path = row["path"]
                    attributes = {
                        col: row[col]
                        for col in self.dataframe.columns
                        if col != "path" and pd.notna(row[col])
                    }
                    result = ZarrJobResult(path=path, attributes=attributes)
                    result._set_mmpp_ref(self)
                    results.append(result)

            return BatchOperations(results, self)

        # Handle integer index for single result
        if not isinstance(index, int):
            raise TypeError(f"Index must be int or slice, got {type(index)}")

        if self._single_zarr_mode:
            if 0 <= index < len(self._zarr_results):
                return self._zarr_results[index]
            else:
                raise IndexError(
                    f"Index {index} out of range for {len(self._zarr_results)} results"
                )
        else:
            # Database mode
            if self.dataframe is None or self.dataframe.empty:
                raise IndexError("No database available. Run scan() first.")

            if 0 <= index < len(self.dataframe):
                row = self.dataframe.iloc[index]
                path = row["path"]
                attributes = {
                    col: row[col]
                    for col in self.dataframe.columns
                    if col != "path" and pd.notna(row[col])
                }
                result = ZarrJobResult(path=path, attributes=attributes)
                result._set_mmpp_ref(self)
                return result
            else:
                raise IndexError(
                    f"Index {index} out of range for {len(self.dataframe)} results"
                )

    def __iter__(self):
        """Make MMPP iterable."""
        for i in range(len(self)):
            yield self[i]

    @property
    def mpl(self) -> "MMPPlotter":
        """Get matplotlib plotter for all results."""
        if not PLOTTING_AVAILABLE:
            raise ImportError(
                "Plotting functionality not available. Check plotting.py import."
            )

        if self._single_zarr_mode:
            return MMPPlotter(self._zarr_results, self)
        else:
            all_results = self.get_all_jobs()
            return MMPPlotter(all_results, self)

    @property
    def matplotlib(self) -> "MMPPlotter":
        """Get matplotlib plotter for all results (alias for mpl)."""
        return self.mpl

    @property
    def fft(self) -> "FFT":
        """Get FFT analyzer for all results."""
        if not FFT_AVAILABLE:
            raise ImportError(
                "FFT functionality not available. Check fft module import."
            )

        # For single zarr mode, return FFT for the first result
        if self._single_zarr_mode and self._zarr_results:
            return FFT(self._zarr_results[0], self)
        else:
            # For database mode, get first result or create empty FFT
            all_results = self.get_all_jobs()
            if all_results:
                return FFT(all_results[0], self)
            else:
                raise ValueError("No results available for FFT analysis")

    def _find_zarr_folders(self) -> list[str]:
        """
        Recursively find all .zarr folders in the base path.

        Returns:
        --------
        List[str]
            List of paths to zarr folders
        """
        zarr_folders: list[str] = []

        for root, dirs, _files in os.walk(self.base_path):
            # Check if current directory is a zarr folder
            if root.endswith(".zarr") and os.path.isdir(root):
                zarr_folders.append(root)
                # Don't descend into zarr folders
                dirs.clear()

        return zarr_folders

    def _parse_path_parameters(self, zarr_path: str) -> dict[str, Any]:
        """
        Parse parameters from the folder path structure, including zarr folder name.

        Parameters:
        -----------
        zarr_path : str
            Dictionary of parameters extracted from the path
        """
        path_params: dict[str, Any] = {}

        try:
            # Get relative path from base_path to zarr folder
            rel_path = os.path.relpath(zarr_path, self.base_path)

            # Split the path into components
            path_parts = rel_path.split(os.sep)

            # Process all path components including the .zarr folder name
            for part in path_parts:
                # Skip empty parts and version folders (like 'v1')
                if not part or (
                    part.startswith("v") and part[1:].isdigit() and len(part) <= 3
                ):
                    continue

                # If this is a .zarr folder, remove the .zarr extension for parsing
                if part.endswith(".zarr"):
                    part = part[:-5]  # Remove .zarr extension

                # Parse parameters from this path component
                component_params = self._parse_single_path_component(part)
                path_params.update(component_params)

        except Exception as e:
            log.warning(f"Error parsing path parameters from {zarr_path}: {e}")

        return path_params

    def _parse_single_path_component(self, component: str) -> dict[str, Any]:
        """
        Parse parameters from a single path component.

        Parameters:
        -----------
        component : str
            Single path component (folder name)

        Returns:
        --------
        Dict[str, Any]
            Dictionary of parameters extracted from this component
        """
        params: dict[str, Any] = {}

        try:
            # Handle comma-separated parameters in a single component
            # Example: "param1_value1,param2_value2"
            if "," in component:
                sub_parts = component.split(",")
                for sub_part in sub_parts:
                    sub_params = self._parse_single_path_component(sub_part.strip())
                    params.update(sub_params)
                return params

            # Pattern 1: parameter_values_number (e.g., f0_values_2.15e+09)
            match = re.match(
                r"^(.+)_values_([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)$", component
            )
            if match:
                param_name, param_value = match.groups()
                try:
                    params[param_name] = float(param_value)
                except ValueError:
                    params[param_name] = param_value
                return params

            # Pattern 2: parameter_number (e.g., solver_3, maxerr_1e-06)
            match = re.match(
                r"^(.+?)_([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)$", component
            )
            if match:
                param_name, param_value = match.groups()
                try:
                    # Try to convert to float first
                    float_value = float(param_value)
                    # If it's a whole number, convert to int
                    if float_value.is_integer():
                        params[param_name] = int(float_value)
                    else:
                        params[param_name] = float_value
                except ValueError:
                    params[param_name] = param_value
                return params

            # Pattern 3: Multiple underscore-separated parameters
            # Example: "param1_val1_param2_val2"
            parts = component.split("_")
            if len(parts) >= 4 and len(parts) % 2 == 0:
                # Try to parse as alternating param_value pairs
                for i in range(0, len(parts), 2):
                    if i + 1 < len(parts):
                        param_name = parts[i]
                        param_value_str = parts[i + 1]
                        try:
                            # Try to convert to number
                            if "." in param_value_str or "e" in param_value_str.lower():
                                param_value = float(param_value_str)
                            else:
                                param_value = int(param_value_str)
                            params[param_name] = param_value
                        except ValueError:
                            params[param_name] = param_value_str
                if params:  # If we successfully parsed something
                    return params

            # Pattern 4: Just parameter name without underscore (treat as boolean flag)
            if "_" not in component and component.isalpha():
                params[component] = True
                return params

            # Pattern 5: If nothing else matches but contains underscores,
            # treat as a complex parameter name
            if "_" in component:
                params[component] = True

        except Exception as e:
            log.warning(f"Error parsing component '{component}': {e}")

        return params

    def _scan_single_zarr(self, zarr_path: str) -> ScanResult:
        """
        Scan a single zarr folder and extract metadata using Pyzfn.

        Parameters:
        -----------
        zarr_path : str
            Path to the zarr folder

        Returns:
        --------
        ScanResult
            Result containing path, attributes, and potential error
        """
        try:
            # Initialize Pyzfn job
            job = Pyzfn(zarr_path)

            # Extract all attributes from Pyzfn
            attributes: dict[str, Any] = {}
            for attr_name, attr_value in job.attrs.items():
                # Convert numpy arrays to lists for pandas compatibility
                if hasattr(attr_value, "tolist"):
                    attributes[attr_name] = attr_value.tolist()
                else:
                    attributes[attr_name] = attr_value

            # Parse parameters from full path (including zarr folder name)
            path_params = self._parse_path_parameters(zarr_path)

            # Merge path parameters with Pyzfn attributes
            # Pyzfn attributes take precedence over path parameters
            for param_name, param_value in path_params.items():
                if param_name not in attributes:
                    attributes[param_name] = param_value
                else:
                    # If parameter exists in both, keep Pyzfn version but add path version with suffix
                    attributes[f"{param_name}_path"] = param_value

            return ScanResult(path=zarr_path, attributes=attributes)

        except Exception as e:
            return ScanResult(path=zarr_path, attributes={}, error=str(e))

    def _scan_all_zarr_folders(self, zarr_folders: list[str]) -> list[ScanResult]:
        """
        Scan all zarr folders using multiple threads.

        Parameters:
        -----------
        zarr_folders : List[str]
            List of zarr folder paths to scan

        Returns:
        --------
        List[ScanResult]
            List of scan results
        """
        results: list[ScanResult] = []

        log.info(
            f"Scanning {len(zarr_folders)} zarr folders using {self.max_workers} threads..."
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(self._scan_single_zarr, zarr_path): zarr_path
                for zarr_path in zarr_folders
            }

            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_path), 1):
                result = future.result()
                results.append(result)

                # Print progress
                if i % 10 == 0 or i == len(zarr_folders):
                    log.debug(f"Progress: {i}/{len(zarr_folders)} folders processed")

                # Report errors
                if result.error:
                    log.error(f"Error processing {result.path}: {result.error}")

        return results

    def _create_dataframe(self, scan_results: list[ScanResult]) -> pd.DataFrame:
        """
        Create pandas DataFrame from scan results.

        Parameters:
        -----------
        scan_results : List[ScanResult]
            List of scan results

        Returns:
        --------
        pd.DataFrame
            DataFrame with paths and attributes
        """
        # Collect all data for DataFrame
        data_rows: list[dict[str, Any]] = []

        for result in scan_results:
            if not result.error:  # Only include successful scans
                row = {"path": result.path}
                row.update(result.attributes)
                data_rows.append(row)

        if not data_rows:
            log.warning("No valid zarr folders found!")
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(data_rows)

        log.info(
            f"Created database with {len(df)} entries and {len(df.columns)} columns"
        )
        log.debug(f"Columns: {list(df.columns)}")

        return df

    def _save_database(self) -> None:
        """Save the current DataFrame to pickle file."""
        if self.dataframe is not None:
            with self._lock:
                try:
                    with open(self.database_path, "wb") as f:
                        pickle.dump(self.dataframe, f)
                    log.info(f"Database saved to: {self.database_path}")
                except Exception as e:
                    log.error(f"Error saving database: {e}")

    def _load_database(self) -> bool:
        """
        Load existing database from pickle file.

        Returns:
        --------
        bool
            True if database was loaded successfully, False otherwise
        """
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, "rb") as f:
                    self.dataframe = pickle.load(f)
                log.info(f"Loaded existing database from: {self.database_path}")
                log.debug(f"Database contains {len(self.dataframe)} entries")
                return True
            except Exception as e:
                log.error(f"Error loading database: {e}")
                return False
        return False

    def scan(self, force: bool = False) -> pd.DataFrame:
        """
        Scan the base directory for zarr folders and create/update the database.

        Parameters:
        -----------
        force : bool, optional
            If True, force rescan even if database exists (default: False)

        Returns:
        --------
        pd.DataFrame
            The resulting database DataFrame
        """
        if self._single_zarr_mode:
            log.debug("Single zarr mode - no scanning needed.")
            return pd.DataFrame()  # Return empty DataFrame for single zarr mode

        # Check if we need to scan
        if not force and self.dataframe is not None:
            log.info("Database already loaded. Use force=True to rescan.")
            return self.dataframe

        # Find all zarr folders
        log.info(f"Searching for zarr folders in: {self.base_path}")
        zarr_folders = self._find_zarr_folders()

        if not zarr_folders:
            log.warning("No zarr folders found!")
            return pd.DataFrame()

        log.info(f"Found {len(zarr_folders)} zarr folders")

        # Scan all folders
        scan_results = self._scan_all_zarr_folders(zarr_folders)

        # Create DataFrame
        self.dataframe = self._create_dataframe(scan_results)

        # Save database
        self._save_database()

        return self.dataframe

    def force_rescan(self) -> pd.DataFrame:
        """
        Force a complete rescan of the directory structure.

        Returns:
        --------
        pd.DataFrame
            The resulting database DataFrame
        """
        log.info("Forcing complete rescan...")
        return self.scan(force=True)

    def get_parsing_examples(self, zarr_path: str) -> dict[str, Any]:
        """
        Get examples of how a specific path would be parsed.
        Useful for debugging path parsing.

        Parameters:
        -----------
        zarr_path : str
            Path to analyze

        Returns:
        --------
        Dict[str, Any]
            Dictionary showing parsing results for each component
        """
        rel_path = os.path.relpath(zarr_path, self.base_path)
        path_parts = rel_path.split(os.sep)

        examples = {
            "full_path": zarr_path,
            "relative_path": rel_path,
            "components": {},
            "final_params": self._parse_path_parameters(zarr_path),
        }

        for i, part in enumerate(path_parts):
            if part.endswith(".zarr"):
                clean_part = part[:-5]
                examples["components"][f"component_{i}_{part}"] = {
                    "original": part,
                    "cleaned": clean_part,
                    "parsed": self._parse_single_path_component(clean_part),
                }
            else:
                examples["components"][f"component_{i}_{part}"] = {
                    "original": part,
                    "parsed": self._parse_single_path_component(part),
                }

        return examples

    def find(self, **kwargs: Any) -> Union["PlotterProxy", list["ZarrJobResult"]]:
        """
        Find zarr folders that match the given criteria.
        Now returns a PlotterProxy with plotting capabilities.

        Parameters:
        -----------
        **kwargs : Any
            Attribute criteria to match (e.g., PBCx=1, Nx=1296, solver=3)

        Returns:
        --------
        PlotterProxy
            Proxy object containing ZarrJobResult objects with plotting capabilities
        """
        if self._single_zarr_mode:
            # In single zarr mode, just return the single result if no criteria or if it matches
            if not kwargs:
                # No criteria, return all (which is just one)
                if PLOTTING_AVAILABLE:
                    return PlotterProxy(self._zarr_results, self)
                else:
                    return self._zarr_results
            else:
                # Check if single result matches criteria
                matching_results = []
                for result in self._zarr_results:
                    matches = True
                    for key, value in kwargs.items():
                        if (
                            key not in result.attributes
                            or result.attributes[key] != value
                        ):
                            matches = False
                            break
                    if matches:
                        matching_results.append(result)

                log.debug(
                    f"Found {len(matching_results)} results matching criteria: {kwargs}"
                )
                if PLOTTING_AVAILABLE:
                    return PlotterProxy(matching_results, self)
                else:
                    return matching_results

        # Original database mode logic
        if self.dataframe is None or self.dataframe.empty:
            log.warning("No database available. Run scan() first.")
            if PLOTTING_AVAILABLE:
                return PlotterProxy([], self)
            else:
                return []

        # Start with all rows
        mask = pd.Series([True] * len(self.dataframe), index=self.dataframe.index)

        # Apply each filter criterion
        for key, value in kwargs.items():
            if key not in self.dataframe.columns:
                log.warning(f"Warning: Column '{key}' not found in database")
                continue

            # Handle different types of comparisons
            if isinstance(value, (list, tuple)):
                # If value is a list, check if the column value is in the list
                mask &= self.dataframe[key].isin(value)
            else:
                # Direct equality check
                mask &= self.dataframe[key] == value

        # Get matching rows
        matching_rows = self.dataframe.loc[mask]

        # Convert to ZarrJobResult objects
        results = []
        for _, row in matching_rows.iterrows():
            path = row["path"]
            attributes = {
                col: row[col]
                for col in self.dataframe.columns
                if col != "path" and pd.notna(row[col])
            }
            result = ZarrJobResult(path=path, attributes=attributes)
            result._set_mmpp_ref(self)
            results.append(result)

        log.debug(f"Found {len(results)} folders matching criteria: {kwargs}")

        if PLOTTING_AVAILABLE:
            return PlotterProxy(results, self)
        else:
            return results

    def find_paths(self, **kwargs: Any) -> list[str]:
        """
        Find zarr folder paths that match the given criteria.

        Parameters:
        -----------
        **kwargs : Any
            Attribute criteria to match (e.g., PBCx=1, Nx=1296)

        Returns:
        --------
        List[str]
            List of paths to zarr folders matching the criteria
        """
        results = self.find(**kwargs)
        return [result.path for result in results]

    def find_by_path_param(self, **kwargs: Any) -> list[ZarrJobResult]:
        """
        Find zarr folders that match path-extracted parameters specifically.

        Parameters:
        -----------
        **kwargs : Any
            Path parameter criteria to match (e.g., solver=3, f0=2.15e+09)

        Returns:
        --------
        List[ZarrJobResult]
            List of ZarrJobResult objects matching the criteria
        """
        if self.dataframe is None or self.dataframe.empty:
            log.warning("No database available. Run scan() first.")
            return []

        # Start with all rows
        mask = pd.Series([True] * len(self.dataframe), index=self.dataframe.index)

        # Apply each filter criterion
        for key, value in kwargs.items():
            # Check both the original parameter name and the _path suffix version
            param_found = False

            if key in self.dataframe.columns:
                mask &= self.dataframe[key] == value
                param_found = True
            elif f"{key}_path" in self.dataframe.columns:
                mask &= self.dataframe[f"{key}_path"] == value
                param_found = True

            if not param_found:
                print(
                    f"Warning: Parameter '{key}' not found in database (neither as '{key}' nor '{key}_path')"
                )
                continue

        # Get matching rows
        matching_rows = self.dataframe.loc[mask]

        # Convert to ZarrJobResult objects
        results = []
        for _, row in matching_rows.iterrows():
            path = row["path"]
            attributes = {
                col: row[col]
                for col in self.dataframe.columns
                if col != "path" and pd.notna(row[col])
            }
            results.append(ZarrJobResult(path=path, attributes=attributes))

        print(f"Found {len(results)} folders matching path criteria: {kwargs}")

        return results

    def find_by_path_param_paths(self, **kwargs: Any) -> list[str]:
        """
        Find zarr folder paths that match path-extracted parameters specifically.

        Parameters:
        -----------
        **kwargs : Any
            Path parameter criteria to match (e.g., solver=3, f0=2.15e+09)

        Returns:
        --------
        List[str]
            List of paths to zarr folders matching the criteria
        """
        results = self.find_by_path_param(**kwargs)
        return [result.path for result in results]

    def get_job(self, path: str) -> Optional[ZarrJobResult]:
        """
        Get a specific job by its path.

        Parameters:
        -----------
        path : str
            Path to the zarr folder

        Returns:
        --------
        Optional[ZarrJobResult]
            ZarrJobResult object or None if not found
        """
        if self.dataframe is None:
            print("No database available. Run scan() first.")
            return None

        matching_rows = self.dataframe[self.dataframe["path"] == path]
        if matching_rows.empty:
            print(f"No job found with path: {path}")
            return None

        row = matching_rows.iloc[0]
        attributes = {
            col: row[col]
            for col in self.dataframe.columns
            if col != "path" and pd.notna(row[col])
        }
        return ZarrJobResult(path=path, attributes=attributes)

    def get_all_jobs(self) -> list[ZarrJobResult]:
        """
        Get all jobs as ZarrJobResult objects.

        Returns:
        --------
        List[ZarrJobResult]
            List of all ZarrJobResult objects in the database
        """
        if self.dataframe is None or self.dataframe.empty:
            print("No database available. Run scan() first.")
            return []

        results = []
        for _, row in self.dataframe.iterrows():
            path = row["path"]
            attributes = {
                col: row[col]
                for col in self.dataframe.columns
                if col != "path" and pd.notna(row[col])
            }
            results.append(ZarrJobResult(path=path, attributes=attributes))

        return results

    def get_database(self) -> Optional[pd.DataFrame]:
        """
        Get the current database DataFrame.

        Returns:
        --------
        Optional[pd.DataFrame]
            The database DataFrame or None if not loaded
        """
        return self.dataframe

    def get_unique_values(self, column: str) -> list[Any]:
        """
        Get unique values for a specific column.

        Parameters:
        -----------
        column : str
            Column name

        Returns:
        --------
        List[Any]
            List of unique values in the column
        """
        if self.dataframe is None or column not in self.dataframe.columns:
            print(f"Column '{column}' not found in database")
            return []

        return sorted(self.dataframe[column].dropna().unique().tolist())

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of the database.

        Returns:
        --------
        Dict[str, Any]
            Summary information about the database
        """
        if self.dataframe is None:
            return {"status": "No database loaded"}

        summary = {
            "total_entries": len(self.dataframe),
            "columns": list(self.dataframe.columns),
            "column_count": len(self.dataframe.columns),
            "database_path": self.database_path,
        }

        return summary

    def get_path_parameters(self, zarr_path: str) -> dict[str, Any]:
        """
        Get parameters extracted from a specific zarr path.

        Parameters:
        -----------
        zarr_path : str
            Path to the zarr folder

        Returns:
        --------
        Dict[str, Any]
            Dictionary of parameters extracted from the path
        """
        return self._parse_path_parameters(zarr_path)

    def get_path_parameter_summary(self) -> dict[str, list[Any]]:
        """
        Get a summary of all path-extracted parameters and their unique values.

        Returns:
        --------
        Dict[str, List[Any]]
            Dictionary mapping parameter names to lists of unique values
        """
        if self.dataframe is None:
            return {}

        path_params = {}

        # Find columns that end with '_path' or are likely path parameters
        for col in self.dataframe.columns:
            if col.endswith("_path"):
                base_name = col[:-5]  # Remove '_path' suffix
                unique_vals = sorted(self.dataframe[col].dropna().unique().tolist())
                path_params[base_name] = unique_vals
            elif col in ["solver", "f0", "ky", "maxerr"]:  # Common path parameters
                if col not in [
                    c[:-5] for c in self.dataframe.columns if c.endswith("_path")
                ]:
                    unique_vals = sorted(self.dataframe[col].dropna().unique().tolist())
                    path_params[col] = unique_vals

        return path_params

    def list_data(self, limit: int = 10) -> None:
        """
        Display a formatted list of all data in the database.

        Parameters:
        -----------
        limit : int, optional
            Maximum number of entries to display (default: 10, use -1 for all)
        """
        if self.dataframe is None or self.dataframe.empty:
            print("No database available. Run scan() first.")
            return

        print("\n=== MMPP Database Summary ===")
        print(f"Total entries: {len(self.dataframe)}")
        print(f"Total columns: {len(self.dataframe.columns)}")
        print(f"Database path: {self.database_path}")

        # Show column names
        print(f"\nColumns: {list(self.dataframe.columns)}")

        # Show sample data
        display_count = (
            len(self.dataframe) if limit == -1 else min(limit, len(self.dataframe))
        )
        print(f"\n=== First {display_count} entries ===")

        for i, (_, row) in enumerate(self.dataframe.head(display_count).iterrows()):
            print(f"\n--- Entry {i + 1} ---")
            print(f"Path: {row['path']}")

            # Group parameters by type
            pyzfn_params = {}
            path_params = {}

            for col in self.dataframe.columns:
                if col == "path" or pd.isna(row[col]):
                    continue

                if col.endswith("_path"):
                    path_params[col[:-5]] = row[col]
                else:
                    pyzfn_params[col] = row[col]

            if pyzfn_params:
                print("Pyzfn attributes:")
                for key, value in pyzfn_params.items():
                    print(f"  {key}: {value}")

            if path_params:
                print("Path parameters:")
                for key, value in path_params.items():
                    print(f"  {key}: {value}")

        if len(self.dataframe) > display_count:
            print(f"\n... and {len(self.dataframe) - display_count} more entries")
            print("Use list_data(limit=-1) to see all entries")

    def __repr__(self) -> str:
        """Rich representation of MMPP object when printed."""
        if self._single_zarr_mode:
            return f"MMPP(single_zarr='{self.base_path}', results={len(self._zarr_results)})"

        if not self._interactive_mode:
            return f"MMPP(base_path='{self.base_path}', entries={len(self.dataframe) if self.dataframe is not None else 0})"

        return self._generate_interactive_display()

    def _generate_interactive_display(self) -> str:
        """Generate interactive display for the MMPP object."""
        if RICH_AVAILABLE:
            return self._rich_display()
        else:
            return self._basic_display()

    def _rich_display(self) -> str:
        """Generate rich console display."""
        if self.dataframe is None or self.dataframe.empty:
            return "[red]No database loaded. Run scan() first.[/red]"

        # Create summary panel
        summary_text = Text()
        summary_text.append("📊 Total entries: ", style="bold cyan")
        summary_text.append(f"{len(self.dataframe)}\n", style="bright_white")
        summary_text.append("📁 Database path: ", style="bold cyan")
        summary_text.append(f"{self.database_path}\n", style="dim")
        summary_text.append("🔍 Columns: ", style="bold cyan")
        summary_text.append(f"{len(self.dataframe.columns)}", style="bright_white")

        # Create methods panel
        methods_text = Text()
        methods_text.append("🔧 Available methods:\n", style="bold yellow")
        methods_text.append("  • ", style="dim")
        methods_text.append("find(**kwargs)", style="code")
        methods_text.append(" - Search by criteria\n", style="dim")
        methods_text.append("  • ", style="dim")
        methods_text.append("show()", style="code")
        methods_text.append(" - Interactive table view\n", style="dim")
        methods_text.append("  • ", style="dim")
        methods_text.append("list_data(limit=10)", style="code")
        methods_text.append(" - Formatted list view\n", style="dim")
        methods_text.append("  • ", style="dim")
        methods_text.append("get_summary()", style="code")
        methods_text.append(" - Database summary\n", style="dim")
        methods_text.append("  • ", style="dim")
        methods_text.append("force_rescan()", style="code")
        methods_text.append(" - Rescan directory", style="dim")

        # Create parameters panel
        param_summary = self.get_path_parameter_summary()
        param_text = Text()
        param_text.append("📋 Key parameters:\n", style="bold green")
        for param, values in list(param_summary.items())[:5]:  # Show first 5 parameters
            param_text.append(f"  • {param}: ", style="cyan")
            param_text.append(f"{len(values)} values ", style="bright_white")
            param_text.append(
                (
                    f"({min(values)} - {max(values)})\n"
                    if values and isinstance(values[0], (int, float))
                    else f"({', '.join(map(str, values[:3]))}{'...' if len(values) > 3 else ''})\n"
                ),
                style="dim",
            )

        if RICH_AVAILABLE:
            with self.console.capture() as capture:
                self.console.print(
                    Panel.fit(
                        summary_text,
                        title="[bold blue]MMPP Database[/bold blue]",
                        border_style="blue",
                    )
                )
                self.console.print("")
                self.console.print(
                    Columns(
                        [
                            Panel.fit(
                                methods_text,
                                title="[bold yellow]Methods[/bold yellow]",
                                border_style="yellow",
                            ),
                            Panel.fit(
                                param_text,
                                title="[bold green]Parameters[/bold green]",
                                border_style="green",
                            ),
                        ]
                    )
                )
            return capture.get()

        return str(summary_text) + "\n" + str(methods_text) + "\n" + str(param_text)

    def _basic_display(self) -> str:
        """Generate basic text display."""
        if self.dataframe is None or self.dataframe.empty:
            return "MMPP Database: No data loaded. Run scan() first."

        summary = f"""
MMPP Database Summary:
=====================
📊 Total entries: {len(self.dataframe)}
📁 Database path: {self.database_path}
🔍 Columns: {len(self.dataframe.columns)}

🔧 Available methods:
  • find(**kwargs) - Search by criteria
  • show() - Interactive table view
  • list_data(limit=10) - Formatted list view
  • get_summary() - Database summary
  • force_rescan() - Rescan directory

📋 Use 'jobs.show()' for interactive table view
"""
        return summary

    def show(self, max_rows: int = 1000, height: int = 400) -> None:
        """
        Show interactive pandas DataFrame viewer.

        Parameters:
        -----------
        max_rows : int, optional
            Maximum number of rows to display (default: 1000)
        height : int, optional
            Height of the table in pixels (default: 400)
        """
        if self.dataframe is None or self.dataframe.empty:
            print("No database available. Run scan() first.")
            return

        df_display = self.dataframe.head(max_rows).copy()

        # Try itables first (best option)
        if ITABLES_AVAILABLE and IPYTHON_AVAILABLE:
            try:
                # Configure itables for VS Code Jupyter
                import itables.options as opt

                opt.css = """
                .itables table td { text-align: left; }
                .itables table th { text-align: left; }
                """

                # Simplify DataFrame formatting for itables compatibility
                df_formatted = df_display.copy()

                # Format float columns with scientific notation for large numbers
                for col in df_formatted.columns:
                    if col != "path" and df_formatted[col].dtype in [
                        "float64",
                        "float32",
                    ]:
                        # Check if column has large numbers
                        if df_formatted[col].abs().max() > 1000:
                            df_formatted[col] = df_formatted[col].apply(
                                lambda x: f"{x:.2e}" if pd.notna(x) else "N/A"
                            )

                # Shorten paths for better display
                if "path" in df_formatted.columns:
                    df_formatted["path"] = df_formatted["path"].apply(
                        lambda x: (
                            str(x).replace(self.base_path, "...")
                            if pd.notna(x)
                            else "N/A"
                        )
                    )

                # Use itables with simplified parameters for VS Code
                from itables import to_html_datatable

                html_table = to_html_datatable(
                    df_formatted,
                    maxHeight=height,
                    scrollX=True,
                    scrollY=True,
                    classes="display compact stripe hover",
                    table_id="mmpp_table",
                )

                display(HTML(html_table))
                return

            except Exception as e:
                print(f"itables failed: {e}, falling back to rich display")

        # Try rich table display
        if RICH_AVAILABLE:
            try:
                self._show_rich_table(df_display, max_rows=50)  # Limit for rich display
                return
            except Exception as e:
                print(f"Rich display failed: {e}, falling back to pandas")

        # Fallback to pandas display with VS Code optimization
        try:
            # Create a nicely formatted HTML table for VS Code
            html = self._create_styled_html_table(df_display)
            if IPYTHON_AVAILABLE:
                display(HTML(html))
            else:
                print("HTML table created but cannot display in this environment")
        except Exception as e:
            print(f"HTML display failed: {e}, using basic pandas display")
            # Basic pandas display
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", None)
            pd.set_option("display.max_colwidth", 50)
            print(df_display.to_string(max_rows=20))

    def _show_rich_table(self, df: pd.DataFrame, max_rows: int = 50) -> None:
        """Show DataFrame using rich table."""
        table = Table(
            title="MMPP Database", show_header=True, header_style="bold magenta"
        )

        # Add columns
        for col in df.columns:
            if col == "path":
                table.add_column(col, style="cyan", max_width=40)
            elif col.endswith("_path"):
                table.add_column(col, style="yellow")
            elif col in ["solver", "f0", "ky", "maxerr"]:
                table.add_column(col, style="green")
            else:
                table.add_column(col, style="white")

        # Add rows
        for _i, (_, row) in enumerate(df.head(max_rows).iterrows()):
            row_data = []
            for col in df.columns:
                value = row[col]
                if pd.isna(value):
                    row_data.append("[dim]N/A[/dim]")
                elif col == "path":
                    # Shorten path for display
                    short_path = str(value).replace(self.base_path, "...")
                    row_data.append(f"[dim]{short_path}[/dim]")
                elif isinstance(value, float) and abs(value) > 1000:
                    row_data.append(f"{value:.2e}")
                else:
                    row_data.append(str(value))

            table.add_row(*row_data)

        if len(df) > max_rows:
            table.add_row(
                *[f"[dim]... and {len(df) - max_rows} more rows[/dim]"]
                + [""] * (len(df.columns) - 1)
            )

        self.console.print(table)

    def _create_styled_html_table(self, df: pd.DataFrame) -> str:
        """Create styled HTML table."""
        html = """
        <style>
        .mmpp-table {
            border-collapse: collapse;
            width: 100%;
            font-family: Arial, sans-serif;
            font-size: 12px;
        }
        .mmpp-table th, .mmpp-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .mmpp-table th {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        .mmpp-table .path-col {
            background-color: #f0f8ff;
            font-family: monospace;
            font-size: 10px;
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .mmpp-table .path-param {
            background-color: #fff3cd;
        }
        .mmpp-table .key-param {
            background-color: #d4edda;
        }
        .mmpp-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .mmpp-table tr:hover {
            background-color: #f5f5f5;
        }
        </style>
        <table class="mmpp-table">
        """

        # Header
        html += "<tr>"
        for col in df.columns:
            html += f"<th>{col}</th>"
        html += "</tr>"

        # Rows
        for _, row in df.head(100).iterrows():  # Limit for HTML
            html += "<tr>"
            for col in df.columns:
                value = row[col]
                css_class = ""

                if col == "path":
                    css_class = "path-col"
                    value = str(value).replace(self.base_path, "...")
                elif col.endswith("_path"):
                    css_class = "path-param"
                elif col in ["solver", "f0", "ky", "maxerr"]:
                    css_class = "key-param"

                if pd.isna(value):
                    value = "N/A"
                elif isinstance(value, float) and abs(value) > 1000:
                    value = f"{value:.2e}"

                html += f'<td class="{css_class}">{value}</td>'
            html += "</tr>"

        html += "</table>"

        if len(df) > 100:
            html += f"<p><i>... and {len(df) - 100} more rows. Use show() with itables for full view.</i></p>"

        return html

    def set_interactive_mode(self, enabled: bool = True) -> None:
        """
        Enable or disable interactive display mode.

        Parameters:
        -----------
        enabled : bool, optional
            Whether to enable interactive mode (default: True)
        """
        self._interactive_mode = enabled
        if enabled and RICH_AVAILABLE:
            rprint("[green]✓[/green] Interactive mode enabled")
        elif enabled:
            print("✓ Interactive mode enabled (basic)")
        else:
            print("Interactive mode disabled")

    def install_display_deps(self) -> None:
        """Print installation instructions for interactive display dependencies."""
        missing = []

        if not ITABLES_AVAILABLE:
            missing.append("itables")
        if not RICH_AVAILABLE:
            missing.append("rich")
        if not IPYTHON_AVAILABLE:
            missing.append("ipython")

        if missing:
            print("To enable full interactive features, install:")
            print(f"pip install {' '.join(missing)}")
        else:
            print("✓ All interactive display dependencies available!")

    def chunks(self, batch_size: int):
        """
        Generate chunks of results for batch processing.

        Parameters:
        -----------
        batch_size : int
            Number of results per chunk

        Yields:
        -------
        BatchOperations
            Batch operations object for each chunk
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        total_results = len(self)

        for i in range(0, total_results, batch_size):
            end_idx = min(i + batch_size, total_results)
            chunk_slice = slice(i, end_idx)
            yield self[chunk_slice]


def mmpp(base_path: str, force: bool = False, **kwargs: Any) -> MMPP:
    """
    Convenience function to create and initialize a MMPP.

    Parameters:
    -----------
    base_path : str
        Base directory path to scan
    force : bool, optional
        If True, force rescan even if database exists (default: False)
    **kwargs : Any
        Additional arguments passed to MMPP constructor

    Returns:
    --------
    MMPP
        Initialized processor instance
    """
    processor = MMPP(base_path, **kwargs)
    processor.scan(force=force)
    return processor
