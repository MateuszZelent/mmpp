import logging
import os
import glob
import json
import pickle
import re
import threading
import warnings
import uuid
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Optional, Union

from ..cli.logging_config import get_mmpp_logger
from .job import ScanResult, ZarrJobResult
from .constants import PLOTTING_AVAILABLE, FFT_AVAILABLE

if TYPE_CHECKING:
    from ..batch_operations import BatchOperations

if PLOTTING_AVAILABLE:
    from ..plotting import MMPPlotter, PlotterProxy

if FFT_AVAILABLE:
    from ..fft import FFT

log = get_mmpp_logger("mmpp")

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
        force_rescan: bool = False,
    ):
        """
        Initialize the MMPP.

        Parameters:
        -----------
        base_path : str
            Base directory path to scan for zarr folders OR direct path to .zarr file
            Can be virtual container path - will be translated to host path if needed.
        max_workers : int, optional
            Number of threads for scanning (default: 8)
        database_name : str, optional
            Name of the pickle file to store database (default: "mmpy_database")
        debug : bool, optional
            Enable debug logging (default: False)
        log_level : str or int, optional
            Set specific logging level (overrides debug flag)
        force_rescan : bool, optional
            Force rescan of directory even if cache exists (default: False)
        """
        # Configure logging - reconfigure if debug/level specified
        from ..cli.logging_config import setup_mmpp_logging
        
        if log_level is not None:
            # Explicit level always reconfigures
            level_int: int
            if isinstance(log_level, str):
                level_int = getattr(logging, log_level.upper(), logging.INFO)
            else:
                level_int = log_level
            setup_mmpp_logging(debug=False, level=level_int)
        elif debug:
            # Debug flag reconfigures to DEBUG level
            setup_mmpp_logging(debug=True, level=logging.DEBUG)
        
        self.debug = debug  # Store for child components (FFT, etc.)

        # Translate virtual path to host path if needed
        abs_path = os.path.abspath(base_path)
        if not os.path.exists(abs_path):
            translated_path = self._translate_path(base_path)
            if translated_path != base_path and os.path.exists(translated_path):
                log.info(f"Translated base_path: {base_path} -> {translated_path}")
                abs_path = translated_path
            else:
                # Try translating the absolute path
                translated_abs = self._translate_path(abs_path)
                if translated_abs != abs_path and os.path.exists(translated_abs):
                    log.info(f"Translated base_path: {abs_path} -> {translated_abs}")
                    abs_path = translated_abs

        self.base_path = abs_path
        self.max_workers = max_workers
        self.database_name = database_name
        self.df = pd.DataFrame()
        self.zarr_results: list[ZarrJobResult] = []

        # Check if base_path is a single .zarr file
        if self.base_path.endswith(".zarr") and os.path.isdir(self.base_path):
            self._load_single_zarr()
        else:
            # Try to load existing database (skip if force_rescan)
            if force_rescan:
                log.info("force_rescan=True: Skipping cache and rescanning directory")
                self.scan(force=True)
            elif not self._load_database():
                self.scan()

    def _load_single_zarr(self):
        """Load a single .zarr file directly."""
        try:
            # Create a single ZarrJobResult
            # We need to extract attributes manually since we're skipping the scan process
            from ..pyzfn import Pyzfn

            job = Pyzfn(self.base_path)
            attrs = job.attributes
            # Create a minimal DataFrame
            self.df = pd.DataFrame(
                [{"path": self.base_path, **attrs}]
            )
            # Create ZarrJobResult, not Pyzfn
            result = ZarrJobResult(self.base_path, attrs)
            result._set_mmpp_ref(self)
            self.zarr_results = [result]
            log.info(f"Loaded single zarr file: {self.base_path}")
        except Exception as e:
            log.error(f"Failed to load single zarr file: {e}")
            self.df = pd.DataFrame()
            self.zarr_results = []

    def __len__(self):
        """Return number of zarr results available."""
        return len(self.zarr_results)

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
        if isinstance(index, slice):
            # Return BatchOperations object for the slice
            from ..batch_operations import BatchOperations

            sliced_results = self.zarr_results[index]
            return BatchOperations(sliced_results, self)

        if index < 0:
            index += len(self.zarr_results)
        if 0 <= index < len(self.zarr_results):
            result = self.zarr_results[index]
            # Set reference to self for plotting
            result._set_mmpp_ref(self)
            return result
        raise IndexError("Index out of range")

    def __iter__(self):
        """Make MMPP iterable."""
        return iter(self.zarr_results)

    def __repr__(self) -> str:
        """Return concise text representation for console."""
        n_results = len(self.zarr_results)
        path_display = self.base_path
        if len(path_display) > 60:
            path_display = "..." + path_display[-57:]
        
        if n_results == 0:
            return f"<MMPP: {path_display} (empty)>"
        
        return f"<MMPP: {path_display} | {n_results} result{'s' if n_results != 1 else ''}>"

    def _repr_html_(self) -> str:
        """Return rich HTML representation for Jupyter notebooks."""
        n_results = len(self.zarr_results)
        
        # Elegant dark navy-charcoal gradient theme
        html = '<div style="font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Arial, sans-serif; border: 2px solid #334155; border-radius: 12px; padding: 18px; margin: 10px 0; background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%); color: #e2e8f0; box-shadow: 0 10px 25px rgba(0,0,0,0.3), 0 0 0 1px rgba(148,163,184,0.1) inset;">'
        html += '<h3 style="margin: 0 0 12px 0; color: #f1f5f9; font-weight: 600; letter-spacing: 0.5px; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">📊 MMPP Job Manager</h3>'
        html += f'<div style="background: linear-gradient(135deg, rgba(51,65,85,0.4) 0%, rgba(30,41,59,0.4) 100%); padding: 12px; border-radius: 8px; margin-bottom: 12px; border: 1px solid rgba(148,163,184,0.15); backdrop-filter: blur(10px);">'
        html += f'<b style="color: #94a3b8;">Path:</b> <code style="background: rgba(15,23,42,0.6); padding: 4px 10px; border-radius: 5px; font-family: \'Courier New\', monospace; font-size: 0.9em; color: #cbd5e1; border: 1px solid rgba(71,85,105,0.3);">{self.base_path}</code><br>'
        html += f'<b style="color: #94a3b8;">Results:</b> <span style="color: #60a5fa; font-weight: 600;">{n_results}</span> <span style="color: #cbd5e1;">zarr file{"s" if n_results != 1 else ""}</span>'
        html += '</div>'
        
        if n_results == 0:
            html += '<div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px;">'
            html += '⚠️ No simulation results found. Check path or run scan.'
            html += '</div></div>'
            return html
        
        # Get parameter statistics
        param_stats = self._get_parameter_stats()
        
        if param_stats:
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            
            html += '<div style="background: linear-gradient(135deg, rgba(51,65,85,0.4) 0%, rgba(30,41,59,0.4) 100%); padding: 12px; border-radius: 8px; margin-bottom: 12px; border: 1px solid rgba(148,163,184,0.15); backdrop-filter: blur(10px);">'
            html += '<b style="color: #94a3b8;">📋 Parameters:</b> <small style="color: #64748b; margin-left: 8px;">(click to see values)</small><br>'
            html += '<table style="width: 100%; margin-top: 8px; border-collapse: collapse; font-size: 0.9em;">'
            html += '<tr style="background: linear-gradient(135deg, rgba(71,85,105,0.3) 0%, rgba(51,65,85,0.3) 100%); border-bottom: 2px solid rgba(148,163,184,0.2);"><th style="text-align:left; padding: 8px; font-weight: 600; color: #cbd5e1;">Parameter</th><th style="text-align:left; padding: 8px; font-weight: 600; color: #cbd5e1;">Unique Values</th><th style="text-align:left; padding: 8px; font-weight: 600; color: #cbd5e1;">Range</th></tr>'
            
            # Show first 8 parameters
            for idx, (param, info) in enumerate(list(param_stats.items())[:8]):
                unique_count = info['unique']
                if unique_count > 1:
                    range_str = f"{info['min']:.4g} → {info['max']:.4g}"
                else:
                    range_str = f"{info['min']:.4g} (constant)"
                
                # Get all unique values for this parameter
                values_list = sorted(self.df[param].dropna().unique())
                values_str = ', '.join([f"{v:.6g}" if isinstance(v, (int, float)) else str(v) for v in values_list])
                param_detail_id = f"param-detail-{unique_id}-{idx}"
                
                html += f'<tr style="border-bottom: 1px solid rgba(71,85,105,0.3); cursor: pointer;" onclick="var elem = document.getElementById(\'{param_detail_id}\'); elem.style.display = elem.style.display === \'none\' ? \'table-row\' : \'none\';">'
                html += f'<td style="padding: 6px 8px;"><code style="background: rgba(15,23,42,0.6); padding: 3px 8px; border-radius: 4px; color: #60a5fa; border: 1px solid rgba(71,85,105,0.3); font-weight: 500;">{param}</code></td>'
                html += f'<td style="padding: 6px 8px; text-align: center; color: #a5b4fc; font-weight: 600;">{unique_count}</td>'
                html += f'<td style="padding: 6px 8px; font-family: monospace; color: #cbd5e1;">{range_str}</td>'
                html += '</tr>'
                
                # Hidden row with values
                html += f'<tr id="{param_detail_id}" style="display: none; background: rgba(15,23,42,0.4);">'
                html += f'<td colspan="3" style="padding: 8px 12px;">'
                html += f'<div style="color: #94a3b8; font-size: 0.85em; margin-bottom: 4px;">💡 Copy for find():</div>'
                html += f'<code style="display: block; background: rgba(15,23,42,0.8); padding: 8px; border-radius: 4px; color: #10b981; font-size: 0.85em; border: 1px solid rgba(71,85,105,0.4); overflow-x: auto; white-space: nowrap;">{param}=[{values_str}]</code>'
                html += '</td></tr>'
            
            # Add collapsible section for remaining parameters
            if len(param_stats) > 8:
                html += f'<tr id="more-params-{unique_id}" style="display: none;">'
                for idx, (param, info) in enumerate(list(param_stats.items())[8:], start=8):
                    unique_count = info['unique']
                    if unique_count > 1:
                        range_str = f"{info['min']:.4g} → {info['max']:.4g}"
                    else:
                        range_str = f"{info['min']:.4g} (constant)"
                    
                    # Get all unique values for this parameter
                    values_list = sorted(self.df[param].dropna().unique())
                    values_str = ', '.join([f"{v:.6g}" if isinstance(v, (int, float)) else str(v) for v in values_list])
                    param_detail_id = f"param-detail-{unique_id}-{idx}"
                    
                    html += f'</tr><tr id="more-params-{unique_id}" style="display: none; border-bottom: 1px solid rgba(71,85,105,0.3); cursor: pointer;" onclick="var elem = document.getElementById(\'{param_detail_id}\'); elem.style.display = elem.style.display === \'none\' ? \'table-row\' : \'none\';">'
                    html += f'<td style="padding: 6px 8px;"><code style="background: rgba(15,23,42,0.6); padding: 3px 8px; border-radius: 4px; color: #60a5fa; border: 1px solid rgba(71,85,105,0.3); font-weight: 500;">{param}</code></td>'
                    html += f'<td style="padding: 6px 8px; text-align: center; color: #a5b4fc; font-weight: 600;">{unique_count}</td>'
                    html += f'<td style="padding: 6px 8px; font-family: monospace; color: #cbd5e1;">{range_str}</td></tr>'
                    
                    # Hidden row with values
                    html += f'<tr id="{param_detail_id}" style="display: none; background: rgba(15,23,42,0.4);">'
                    html += f'<td colspan="3" style="padding: 8px 12px;">'
                    html += f'<div style="color: #94a3b8; font-size: 0.85em; margin-bottom: 4px;">💡 Copy for find():</div>'
                    html += f'<code style="display: block; background: rgba(15,23,42,0.8); padding: 8px; border-radius: 4px; color: #10b981; font-size: 0.85em; border: 1px solid rgba(71,85,105,0.4); overflow-x: auto; white-space: nowrap;">{param}=[{values_str}]</code>'
                    html += '</td></tr>'
                
                html += '</table>'
                html += f'<button onclick="var elems = document.querySelectorAll(\'#more-params-{unique_id}\'); elems.forEach(e => e.style.display = e.style.display === \'none\' ? \'table-row\' : \'none\'); this.textContent = this.textContent.includes(\'Show\') ? \'▲ Hide {len(param_stats) - 8} more parameters\' : \'▼ Show {len(param_stats) - 8} more parameters\';" style="margin-top: 10px; padding: 8px 16px; background: linear-gradient(135deg, rgba(96,165,250,0.2) 0%, rgba(79,70,229,0.2) 100%); border: 1px solid rgba(96,165,250,0.3); border-radius: 6px; color: #93c5fd; cursor: pointer; font-size: 0.85em; font-weight: 600; transition: all 0.2s; text-shadow: 0 1px 2px rgba(0,0,0,0.3);">▼ Show {len(param_stats) - 8} more parameters</button>'
            else:
                html += '</table>'
            
            html += '</div>'
        
        # Available methods
        html += '<div style="background: linear-gradient(135deg, rgba(51,65,85,0.4) 0%, rgba(30,41,59,0.4) 100%); padding: 12px; border-radius: 8px; border: 1px solid rgba(148,163,184,0.15); backdrop-filter: blur(10px);">'
        html += '<b style="color: #94a3b8;">🔧 Quick Start:</b><br>'
        html += '<code style="background: rgba(15,23,42,0.8); padding: 5px 10px; border-radius: 5px; display: inline-block; margin: 4px; font-family: \'Courier New\', monospace; font-size: 0.85em; color: #60a5fa; border: 1px solid rgba(71,85,105,0.4); font-weight: 500;">job.find(param=value)</code> '
        html += '<code style="background: rgba(15,23,42,0.8); padding: 5px 10px; border-radius: 5px; display: inline-block; margin: 4px; font-family: \'Courier New\', monospace; font-size: 0.85em; color: #60a5fa; border: 1px solid rgba(71,85,105,0.4); font-weight: 500;">job.columns</code> '
        html += '<code style="background: rgba(15,23,42,0.8); padding: 5px 10px; border-radius: 5px; display: inline-block; margin: 4px; font-family: \'Courier New\', monospace; font-size: 0.85em; color: #60a5fa; border: 1px solid rgba(71,85,105,0.4); font-weight: 500;">job[0].m</code> '
        html += '<code style="background: rgba(15,23,42,0.8); padding: 5px 10px; border-radius: 5px; display: inline-block; margin: 4px; font-family: \'Courier New\', monospace; font-size: 0.85em; color: #60a5fa; border: 1px solid rgba(71,85,105,0.4); font-weight: 500;">job[:].m.mpl</code><br>'
        html += '<small style="color: #94a3b8; margin-top: 6px; display: inline-block;">💡 Tip: Click on any parameter above to see all values and copy for <code style="background: rgba(15,23,42,0.6); padding: 2px 6px; border-radius: 3px; color: #93c5fd; border: 1px solid rgba(71,85,105,0.3);">find()</code></small>'
        html += '</div></div>'
        
        return html

    def _get_parameter_stats(self) -> dict:
        """Get statistics about parameter values across all results."""
        if self.df.empty:
            return {}
        
        stats = {}
        # Focus on numeric columns that vary
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            if col == 'path':
                continue
            try:
                values = self.df[col].dropna()
                if len(values) > 0:
                    stats[col] = {
                        'unique': values.nunique(),
                        'min': values.min(),
                        'max': values.max()
                    }
            except:
                continue
        
        # Sort by number of unique values (descending) - varying parameters first
        return dict(sorted(stats.items(), key=lambda x: x[1]['unique'], reverse=True))

    @property
    def mpl(self) -> "MMPPlotter":
        """Get matplotlib plotter for all results."""
        if not PLOTTING_AVAILABLE:
            raise ImportError(
                "Plotting functionality not available. Install matplotlib."
            )
        return MMPPlotter(self.zarr_results, self)

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
        # For MMPP level, we pass the first job as primary but provide full list context if needed
        # Actually FFT expects a single job usually, but let's see how it handles it.
        # If FFT is designed for single job, this property might need adjustment or return a BatchFFT.
        # For now, let's assume it takes the first job or we need a different approach.
        # Looking at FFT init: def __init__(self, job_result, mmpp_instance=None):
        if not self.zarr_results:
             raise ValueError("No zarr results available for FFT analysis.")
        
        return FFT(self.zarr_results[0], self)

    @property
    def columns(self) -> list[str]:
        """
        List of available column names for filtering with `find()`.
        
        Returns
        -------
        list[str]
            Column names from the database DataFrame.
            
        Examples
        --------
        >>> job.columns
        ['path', 'Nx', 'Ny', 'Nz', 'dx', 'dy', 'dz', 'PBCx', 'PBCy', 'solver', ...]
        
        >>> 'Nx' in job.columns
        True
        """
        return self.df.columns.tolist()

    def _find_zarr_folders(self) -> list[str]:
        """
        Recursively find all .zarr folders in the base path.

        Returns:
        --------
        List[str]
            List of paths to zarr folders
        """
        zarr_folders = []
        # Use glob for initial search (might be faster than os.walk for specific pattern)
        # But os.walk is more robust for deep recursion
        for root, dirs, _ in os.walk(self.base_path):
            for d in dirs:
                if d.endswith(".zarr"):
                    zarr_folders.append(os.path.join(root, d))
        return zarr_folders

    def _parse_path_parameters(self, zarr_path: str) -> dict[str, Any]:
        """
        Parse parameters from the folder path structure, including zarr folder name.

        Parameters:
        -----------
        zarr_path : str
            Dictionary of parameters extracted from the path
        """
        params = {}
        rel_path = os.path.relpath(zarr_path, self.base_path)
        path_components = rel_path.split(os.sep)

        for component in path_components:
            # Skip the .zarr folder itself for parameter parsing if it's just the container
            # But actually sometimes the zarr folder name contains info too.
            # Let's parse everything.
            component_params = self._parse_single_path_component(component)
            params.update(component_params)

        return params

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
        params = {}
        
        # Remove .zarr extension if present
        name = component.replace(".zarr", "")
        
        # Split by underscore or other delimiters
        # Common pattern: param1_val1_param2_val2
        # Or: param1=val1, param2=val2 (less common in folder names but possible)
        
        # Regex for key-value pairs like "key=value" or "key_value" where value is number
        # This is heuristic and might need adjustment based on specific naming conventions
        
        # Strategy 1: Look for explicit assignments (e.g. Nx=128)
        assignments = re.findall(r"([a-zA-Z0-9]+)=([a-zA-Z0-9\.\-+eE]+)", name)
        for key, val in assignments:
            try:
                # Try converting to number
                if "." in val or "e" in val.lower():
                    params[key] = float(val)
                else:
                    params[key] = int(val)
            except ValueError:
                params[key] = val
        
        # Strategy 2: Look for underscore-separated key_value patterns (e.g., kc2_60000.0, phi_0.5)
        # Match pattern: letters followed by underscore and numeric value (including scientific notation)
        underscore_patterns = re.findall(r"([a-zA-Z][a-zA-Z0-9]*)_([\-+]?[0-9]*\.?[0-9]+(?:[eE][\-+]?[0-9]+)?)", name)
        for key, val in underscore_patterns:
            if key not in params:  # Don't override = assignments
                try:
                    # Try converting to number
                    if "." in val or "e" in val.lower():
                        params[key] = float(val)
                    else:
                        params[key] = int(val)
                except ValueError:
                    params[key] = val
        
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
            # Use Pyzfn to extract attributes
            from ..pyzfn import Pyzfn

            job = Pyzfn(zarr_path)
            attributes = job.attributes

            # Add path parameters - these take precedence as they're explicit in folder structure
            path_params = self._parse_path_parameters(zarr_path)
            # Path parameters override zarr attributes (folder structure is explicit metadata)
            # This ensures parameters like kc2_60000.0 from folder names are preserved
            full_attributes = {**attributes, **path_params}

            return ScanResult(path=zarr_path, attributes=full_attributes)
        except Exception as e:
            log.warning(f"Error scanning {zarr_path}: {e}")
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
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(self._scan_single_zarr, path): path
                for path in zarr_folders
            }

            from rich.progress import track
            
            # Use rich progress bar if available, otherwise simple loop
            try:
                from rich.progress import Progress
                with Progress() as progress:
                    task = progress.add_task("[cyan]Scanning zarr folders...", total=len(zarr_folders))
                    
                    for future in as_completed(future_to_path):
                        path = future_to_path[future]
                        try:
                            result = future.result()
                            results.append(result)
                        except Exception as exc:
                            log.error(f"{path} generated an exception: {exc}")
                        finally:
                            progress.advance(task)
            except ImportError:
                # Fallback without rich
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as exc:
                        log.error(f"{path} generated an exception: {exc}")

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
        data = []
        for res in scan_results:
            if res.error is None:
                entry = {"path": res.path, **res.attributes}
                data.append(entry)
        
        if not data:
            return pd.DataFrame()
            
        return pd.DataFrame(data)

    def _save_database(self) -> None:
        """Save the current DataFrame to pickle file."""
        if self.df.empty:
            return
            
        db_path = os.path.join(self.base_path, f"{self.database_name}.pkl")
        try:
            with open(db_path, "wb") as f:
                pickle.dump(self.df, f)
            log.info(f"Database saved to {db_path}")
        except Exception as e:
            log.error(f"Failed to save database: {e}")

    @staticmethod
    def _translate_path(path: str) -> str:
        """
        Translate virtual container path to host filesystem path.
        
        Translation rules (NEW microlab structure):
        - /mnt/local/kkingstoun/{user}/pcss_storage/{project}/{rest} -> {STORAGE_ROOT}/projects/{project}/{rest}
        - /mnt/local/kkingstoun/{user}/projects/{project}/{rest} -> {STORAGE_ROOT}/projects/{project}/{rest}
        - /mnt/local/kkingstoun/{user}/{project}/{rest} -> {STORAGE_ROOT}/projects/{project}/{rest}
        
        Also supports legacy storage for backward compatibility.
        
        Parameters:
        -----------
        path : str
            Path to translate (may be virtual or real)
            
        Returns:
        --------
        str
            Translated path (or original if no translation needed)
        """
        if not path:
            return path
        
        # NEW unified microlab storage structure
        STORAGE_ROOT = "/mnt/storage_6/project_data/pl0095-01/mateuszz/microlab"
        # Legacy storage (for backward compatibility)
        LEGACY_STORAGE_ROOT = "/mnt/storage_2/scratch/pl0095-01/zelent"
        CONTAINER_PREFIX = "/mnt/local/kkingstoun"
        
        # Already in new host format
        if path.startswith(STORAGE_ROOT):
            return path
        
        # Already in legacy host format - check if should be migrated
        if path.startswith(LEGACY_STORAGE_ROOT):
            return path  # Keep using legacy path if it exists
        
        # Not a container path
        if not path.startswith(CONTAINER_PREFIX):
            return path
        
        # Pattern 1: pcss_storage paths (legacy)
        # /mnt/local/kkingstoun/{user}/pcss_storage/{project}/{rest}
        pcss_pattern = rf"^{re.escape(CONTAINER_PREFIX)}/[^/]+/pcss_storage/([^/]+)(/.*)?"
        pcss_match = re.match(pcss_pattern, path)
        if pcss_match:
            project = pcss_match.group(1)
            rest = pcss_match.group(2) or ""
            new_path = f"{STORAGE_ROOT}/projects/{project}{rest}".replace("//", "/")
            # Fallback to legacy if new path doesn't exist
            if not os.path.exists(new_path):
                legacy_path = f"{LEGACY_STORAGE_ROOT}/{project}{rest}".replace("//", "/")
                if os.path.exists(legacy_path):
                    return legacy_path
            return new_path
        
        # Pattern 2: /projects/ paths in container
        # /mnt/local/kkingstoun/{user}/projects/{project}/{rest}
        projects_pattern = rf"^{re.escape(CONTAINER_PREFIX)}/[^/]+/projects/([^/]+)(/.*)?"
        projects_match = re.match(projects_pattern, path)
        if projects_match:
            project = projects_match.group(1)
            rest = projects_match.group(2) or ""
            return f"{STORAGE_ROOT}/projects/{project}{rest}".replace("//", "/")
        
        # Pattern 3: Direct project paths (fallback)
        # /mnt/local/kkingstoun/{user}/{project}/{rest}
        standard_pattern = rf"^{re.escape(CONTAINER_PREFIX)}/[^/]+/([^/]+)(/.*)?"
        standard_match = re.match(standard_pattern, path)
        if standard_match:
            project_or_subdir = standard_match.group(1)
            rest = standard_match.group(2) or ""
            # Skip special directories that are not projects
            if project_or_subdir in ("pcss_storage", "projects", ".config", ".local", ".cache"):
                return path  # Don't translate special dirs
            new_path = f"{STORAGE_ROOT}/projects/{project_or_subdir}{rest}".replace("//", "/")
            # Fallback to legacy
            if not os.path.exists(new_path):
                legacy_path = f"{LEGACY_STORAGE_ROOT}/{project_or_subdir}{rest}".replace("//", "/")
                if os.path.exists(legacy_path):
                    return legacy_path
            return new_path
            
        return path

    def _load_database(self) -> bool:
        """
        Load existing database from pickle file.
        
        Validates and translates paths during loading. If a path doesn't exist,
        attempts to translate it from virtual to host path.

        Returns:
        --------
        bool
            True if database was loaded successfully, False otherwise
        """
        db_path = os.path.join(self.base_path, f"{self.database_name}.pkl")
        if not os.path.exists(db_path):
            return False
            
        try:
            with open(db_path, "rb") as f:
                self.df = pickle.load(f)
            
            # Reconstruct ZarrJobResult objects with path validation and translation
            self.zarr_results = []
            valid_paths = []
            
            for _, row in self.df.iterrows():
                path = row["path"]
                
                # Check if path exists, if not try to translate
                if not os.path.exists(path):
                    translated_path = self._translate_path(path)
                    if translated_path != path:
                        log.debug(f"Translated path: {path} -> {translated_path}")
                        if os.path.exists(translated_path):
                            path = translated_path
                        else:
                            log.warning(f"Path does not exist after translation: {translated_path}")
                            continue  # Skip this entry
                    else:
                        log.warning(f"Path does not exist: {path}")
                        continue  # Skip this entry
                
                # Filter out path from attributes
                attrs = {k: v for k, v in row.items() if k != "path"}
                self.zarr_results.append(ZarrJobResult(path, attrs))
                valid_paths.append(path)
            
            # Update DataFrame with valid paths only
            if len(valid_paths) < len(self.df):
                log.info(f"Filtered {len(self.df) - len(valid_paths)} invalid paths from database")
                self.df = self.df[self.df["path"].apply(lambda p: 
                    os.path.exists(p) or os.path.exists(self._translate_path(p))
                )]
                # Update paths in DataFrame to translated versions
                self.df["path"] = self.df["path"].apply(lambda p: 
                    self._translate_path(p) if not os.path.exists(p) else p
                )
                
            log.info(f"Loaded database from {db_path} ({len(self.zarr_results)} valid entries)")
            return True
        except Exception as e:
            log.warning(f"Failed to load database: {e}")
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
        if not force and not self.df.empty:
            return self.df

        log.info(f"Scanning {self.base_path} for .zarr folders...")
        zarr_folders = self._find_zarr_folders()
        log.info(f"Found {len(zarr_folders)} .zarr folders.")

        scan_results = self._scan_all_zarr_folders(zarr_folders)
        self.df = self._create_dataframe(scan_results)
        
        # Create ZarrJobResult objects
        self.zarr_results = []
        for res in scan_results:
            if res.error is None:
                self.zarr_results.append(ZarrJobResult(res.path, res.attributes))

        self._save_database()
        return self.df

    def force_rescan(self) -> pd.DataFrame:
        """
        Force a complete rescan of the directory structure.

        Returns:
        --------
        pd.DataFrame
            The resulting database DataFrame
        """
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
        return self._parse_path_parameters(zarr_path)

    def find(self, **kwargs: Any) -> "PlotterProxy":
        """
        Find zarr folders that match the given criteria.
        
        Returns a PlotterProxy with plotting capabilities containing 
        all matching ZarrJobResult objects.

        Parameters
        ----------
        **kwargs : Any
            Attribute criteria to filter by. Each keyword argument must match
            a column name in the database (see `job.columns` property or 
            `job.df.columns` for available columns).

        Common Simulation Parameters
        ----------------------------
        Grid dimensions:
            Nx, Ny, Nz : int
                Number of cells in x, y, z directions
            dx, dy, dz : float
                Cell size in meters (e.g., 5e-9 for 5 nm)
            cellsize_x, cellsize_y, cellsize_z : float
                Alternative cell size specification

        Time parameters:
            dt : float
                Simulation timestep in seconds
            t_sampl : float
                Sampling time interval
            total_time : float
                Total simulation time
            n_steps : int
                Number of simulation steps

        Boundary conditions:
            PBCx, PBCy, PBCz : int
                Periodic boundary conditions (0=off, 1=on)

        Frequency/FFT:
            fcut, f_cut : float
                Cutoff frequency for FFT analysis

        Solver and physics:
            solver : int
                Solver type (e.g., 3=RK3, 4=RK4, 5=RK45)
            alpha : float
                Gilbert damping constant
            Ms : float
                Saturation magnetization
            Bext : float
                External magnetic field

        Custom parameters:
            Any additional parameters saved in the zarr attributes
            will be available for filtering.

        Returns
        -------
        PlotterProxy
            Proxy object containing matching ZarrJobResult objects.
            Supports indexing like `result[0]`, iteration, and 
            plotting methods like `.mpl.plot()`.

        Examples
        --------
        Find simulations with specific grid size:
        
        >>> results = job.find(Nx=1296, Ny=1296)
        >>> len(results)
        5

        Find simulations with periodic boundary conditions:
        
        >>> results = job.find(PBCx=1, PBCy=1)

        Find simulations with specific solver:
        
        >>> results = job.find(solver=3)

        Combine multiple criteria:
        
        >>> results = job.find(Nx=1024, PBCx=1, alpha=0.01)

        Access matching jobs:
        
        >>> results = job.find(Nx=1296)
        >>> for res in results:
        ...     print(res.path)

        Get single matching job:
        
        >>> result = job.find(Nx=1296)[0]
        >>> result.m  # Access magnetization data

        See Also
        --------
        find_paths : Returns list of paths instead of PlotterProxy
        columns : Property showing all available column names
        df : Direct access to pandas DataFrame for complex queries

        Notes
        -----
        - All keyword arguments are combined with AND logic
        - Use `job.df.query()` directly for more complex filtering
        - Check available columns with `job.df.columns.tolist()`
        """
        if self.df.empty:
            log.warning("Database is empty. Run scan() first.")
            if PLOTTING_AVAILABLE:
                return PlotterProxy([], self)
            return [] # type: ignore

        # Filter DataFrame
        query_str = " & ".join([f"{k} == {repr(v)}" for k, v in kwargs.items()])
        try:
            filtered_df = self.df.query(query_str)
        except Exception as e:
            log.error(f"Query failed: {e}")
            if PLOTTING_AVAILABLE:
                return PlotterProxy([], self)
            return [] # type: ignore

        # Get matching ZarrJobResults
        matching_paths = set(filtered_df["path"])
        matching_results = [res for res in self.zarr_results if res.path in matching_paths]

        if PLOTTING_AVAILABLE:
            return PlotterProxy(matching_results, self)
        
        return matching_results # type: ignore

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
        proxy = self.find(**kwargs)
        if hasattr(proxy, "jobs"):
             return [job.path for job in proxy.jobs]
        return [job.path for job in proxy] # type: ignore
