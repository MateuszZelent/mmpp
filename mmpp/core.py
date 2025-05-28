from typing import Optional, Dict, List, Union, Any
import os
import pandas as pd
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
from dataclasses import dataclass
import re
from pyzfn import Pyzfn

# Import for interactive display
try:
    from itables import show, init_notebook_mode

    ITABLES_AVAILABLE = True
except ImportError:
    ITABLES_AVAILABLE = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    from rich import print as rprint

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from IPython.display import display, HTML
    import json

    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False

# Import plotting functionality
try:
    from .plotting import MMPPlotter, PlotterProxy

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


@dataclass
class ScanResult:
    """Data class for storing scan results from a single zarr folder."""

    path: str
    attributes: Dict[str, Any]
    error: Optional[str] = None


@dataclass
class ZarrJobResult:
    """Data class for storing information about a single zarr job/folder."""

    path: str
    attributes: Dict[str, Any]

    def __post_init__(self) -> None:
        """Post-initialization to add plotting capabilities."""
        self._mmpp_ref = None

    def _set_mmpp_ref(self, mmpp_instance: "MMPP") -> None:
        """Set reference to MMPP instance for plotting."""
        self._mmpp_ref = mmpp_instance

    @property
    def mpl(self) -> "MMPPlotter":
        """Get matplotlib plotter for this single result."""
        if not PLOTTING_AVAILABLE:
            raise ImportError(
                "Plotting functionality not available. Check plotting.py import."
            )
        if self._mmpp_ref is None:
            raise ValueError(
                "MMPP reference not set. Use results from MMPP.find() method."
            )
        return MMPPlotter([self], self._mmpp_ref)

    @property
    def matplotlib(self) -> "MMPPlotter":
        """Get matplotlib plotter for this single result (alias for mpl)."""
        return self.mpl

    def __getattr__(self, name: str) -> Any:
        """Allow accessing attributes as object properties."""
        if name in self.attributes:
            return self.attributes[name]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def __getitem__(self, key: str) -> Any:
        """Allow accessing attributes using dictionary-style notation."""
        return self.attributes[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute value with optional default."""
        return self.attributes.get(key, default)

    def keys(self) -> List[str]:
        """Get list of available attribute names."""
        return list(self.attributes.keys())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including path."""
        result = {"path": self.path}
        result.update(self.attributes)
        return result


class MMPP:
    """
    Multi-threaded scanner for zarr folders with pandas database creation and search functionality.

    This class scans directories recursively for .zarr folders, extracts metadata using Pyzfn,
    and creates a searchable pandas database.
    """

    def __init__(
        self, base_path: str, max_workers: int = 8, database_name: str = "mmpy_database"
    ) -> None:
        """
        Initialize the MMPP.

        Parameters:
        -----------
        base_path : str
            Base directory path to scan for zarr folders
        max_workers : int, optional
            Maximum number of worker threads for scanning (default: 8)
        database_name : str, optional
            Name of the database file (without extension, default: "mmpy_database")
        """
        self.base_path: str = os.path.abspath(base_path)
        self.max_workers: int = max_workers
        self.database_name: str = database_name
        self.database_path: str = os.path.join(self.base_path, f"{database_name}.pkl")
        self.dataframe: Optional[pd.DataFrame] = None
        self._lock: threading.Lock = threading.Lock()
        self._interactive_mode: bool = True  # Enable interactive mode by default

        # Initialize rich console if available
        if RICH_AVAILABLE:
            self.console = Console()

        # Try to load existing database
        self._load_database()

    def _find_zarr_folders(self) -> List[str]:
        """
        Recursively find all .zarr folders in the base path.

        Returns:
        --------
        List[str]
            List of paths to zarr folders
        """
        zarr_folders: List[str] = []

        for root, dirs, files in os.walk(self.base_path):
            # Check if current directory is a zarr folder
            if root.endswith(".zarr") and os.path.isdir(root):
                zarr_folders.append(root)
                # Don't descend into zarr folders
                dirs.clear()

        return zarr_folders

    def _parse_path_parameters(self, zarr_path: str) -> Dict[str, Any]:
        """
        Parse parameters from the folder path structure, including zarr folder name.

        Parameters:
        -----------
        zarr_path : str
            Full path to the zarr folder

        Returns:
        --------
        Dict[str, Any]
            Dictionary of parameters extracted from the path
        """
        path_params: Dict[str, Any] = {}

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
            print(f"Warning: Error parsing path parameters from {zarr_path}: {e}")

        return path_params

    def _parse_single_path_component(self, component: str) -> Dict[str, Any]:
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
        params: Dict[str, Any] = {}

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
            print(f"Warning: Error parsing component '{component}': {e}")

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
            attributes: Dict[str, Any] = {}
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

    def _scan_all_zarr_folders(self, zarr_folders: List[str]) -> List[ScanResult]:
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
        results: List[ScanResult] = []

        print(
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
                    print(f"Progress: {i}/{len(zarr_folders)} folders processed")

                # Report errors
                if result.error:
                    print(f"Error processing {result.path}: {result.error}")

        return results

    def _create_dataframe(self, scan_results: List[ScanResult]) -> pd.DataFrame:
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
        data_rows: List[Dict[str, Any]] = []

        for result in scan_results:
            if not result.error:  # Only include successful scans
                row = {"path": result.path}
                row.update(result.attributes)
                data_rows.append(row)

        if not data_rows:
            print("Warning: No valid zarr folders found!")
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(data_rows)

        print(f"Created database with {len(df)} entries and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")

        return df

    def _save_database(self) -> None:
        """Save the current DataFrame to pickle file."""
        if self.dataframe is not None:
            with self._lock:
                try:
                    with open(self.database_path, "wb") as f:
                        pickle.dump(self.dataframe, f)
                    print(f"Database saved to: {self.database_path}")
                except Exception as e:
                    print(f"Error saving database: {e}")

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
                print(f"Loaded existing database from: {self.database_path}")
                print(f"Database contains {len(self.dataframe)} entries")
                return True
            except Exception as e:
                print(f"Error loading database: {e}")
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
        # Check if we need to scan
        if not force and self.dataframe is not None:
            print("Database already loaded. Use force=True to rescan.")
            return self.dataframe

        # Find all zarr folders
        print(f"Searching for zarr folders in: {self.base_path}")
        zarr_folders = self._find_zarr_folders()

        if not zarr_folders:
            print("No zarr folders found!")
            return pd.DataFrame()

        print(f"Found {len(zarr_folders)} zarr folders")

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
        print("Forcing complete rescan...")
        return self.scan(force=True)

    def get_parsing_examples(self, zarr_path: str) -> Dict[str, Any]:
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

    def find(self, **kwargs: Any) -> Union["PlotterProxy", List["ZarrJobResult"]]:
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
        if self.dataframe is None or self.dataframe.empty:
            print("No database available. Run scan() first.")
            if PLOTTING_AVAILABLE:
                return PlotterProxy([], self)
            else:
                return []

        # Start with all rows
        mask = pd.Series([True] * len(self.dataframe), index=self.dataframe.index)

        # Apply each filter criterion
        for key, value in kwargs.items():
            if key not in self.dataframe.columns:
                print(f"Warning: Column '{key}' not found in database")
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

        print(f"Found {len(results)} folders matching criteria: {kwargs}")

        if PLOTTING_AVAILABLE:
            return PlotterProxy(results, self)
        else:
            return results

    def find_paths(self, **kwargs: Any) -> List[str]:
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

    def find_by_path_param(self, **kwargs: Any) -> List[ZarrJobResult]:
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
            print("No database available. Run scan() first.")
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

    def find_by_path_param_paths(self, **kwargs: Any) -> List[str]:
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

    def get_all_jobs(self) -> List[ZarrJobResult]:
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

    def get_unique_values(self, column: str) -> List[Any]:
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

    def get_summary(self) -> Dict[str, Any]:
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

    def get_path_parameters(self, zarr_path: str) -> Dict[str, Any]:
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

    def get_path_parameter_summary(self) -> Dict[str, List[Any]]:
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
            print(f"\n--- Entry {i+1} ---")
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
        methods_text.append("show_interactive()", style="code")
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
  • show_interactive() - Interactive table view
  • list_data(limit=10) - Formatted list view
  • get_summary() - Database summary
  • force_rescan() - Rescan directory

📋 Use 'jobs.show_interactive()' for interactive table view
"""
        return summary

    def show_interactive(self, max_rows: int = 1000, height: int = 400) -> None:
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
        for i, (_, row) in enumerate(df.head(max_rows).iterrows()):
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
            html += f"<p><i>... and {len(df) - 100} more rows. Use show_interactive() with itables for full view.</i></p>"

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
