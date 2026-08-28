import logging
import os
import pickle
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd

from ..cli.logging_config import get_mmpp_logger
from .constants import FFT_AVAILABLE
from .job import ScanResult, ZarrJobResult

if TYPE_CHECKING:
    from ..batch_operations import BatchOperations
    from ..fft import FFT
    from ..plotting import MMPPlotter

log = get_mmpp_logger("mmpp")


def _running_in_ipython_kernel() -> bool:
    """Return True inside Jupyter/VSCode notebook kernels."""
    try:
        import builtins

        get_ipython = getattr(builtins, "get_ipython", None)
        if get_ipython is None:
            return False
        shell = get_ipython().__class__.__name__
    except Exception:
        return False
    return shell == "ZMQInteractiveShell"


def _should_render_rich_progress() -> bool:
    """Use Rich live progress only where it can update a single terminal line."""
    return bool(sys.stderr.isatty() and not _running_in_ipython_kernel())


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
        log_level: str | int | None = None,
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
            self.df = pd.DataFrame([{"path": self.base_path, **attrs}])
            # Create ZarrJobResult, not Pyzfn
            result = ZarrJobResult(self.base_path, attrs)
            result._set_mmpp_ref(self)
            self.zarr_results = [result]
            log.info(f"Loaded single zarr file: {self.base_path}")
        except Exception as e:
            log.error(f"Failed to load single zarr file: {e}")
            self.df = pd.DataFrame()
            self.zarr_results = []

    @property
    def dataframe(self) -> "pd.DataFrame":
        """Alias for :attr:`df` — the main scan results DataFrame."""
        return self.df

    @dataframe.setter
    def dataframe(self, value: "pd.DataFrame") -> None:
        self.df = value

    def __len__(self):
        """Return number of zarr results available."""
        return len(self.zarr_results)

    def __getitem__(
        self, index: int | slice
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
        import html as _html
        import uuid as _uuid_mod

        from mmpp._repr_helpers import (
            NODE_COLOR_ANALYSIS,
            NODE_COLOR_COMPUTE,
            NODE_COLOR_PLOT,
            NODE_COLOR_UTIL,
            accessors_section_html,
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        n_results = len(self.zarr_results)
        unique_id = str(_uuid_mod.uuid4())[:8]
        api_card = api_help_html(
            self,
            title="MMPP API help",
            prefix="job",
            subtitle="Top-level navigation with live method signatures.",
            properties=[
                ("dataframe", "Scan results DataFrame alias"),
                ("columns", "Available parameter/filter columns"),
                (
                    "fft",
                    "FFT accessor for the first result; prefer job[0].fft or job[:].fft",
                ),
                (
                    "solitons",
                    "Soliton analysis entry point; use job[0].solitons or job[:].solitons",
                ),
                (
                    "analyze",
                    "Unified analysis namespace; use job[0].analyze or job[:].analyze",
                ),
                ("mpl", "Batch plotting accessor"),
                ("matplotlib", "Alias for mpl"),
            ],
            methods=[
                "scan",
                "force_rescan",
                "find",
            ],
            chrome=False,
        )

        sections = [
            metrics_section_html(
                [
                    ("path", self.base_path, None),
                    (
                        "results",
                        f"{n_results} zarr file{'s' if n_results != 1 else ''}",
                        NODE_COLOR_COMPUTE,
                    ),
                ]
            )
        ]

        if n_results == 0:
            sections.append(
                "<div style='background:rgba(255,255,255,0.1);padding:10px;"
                "border-radius:5px;margin-bottom:12px;'>"
                "⚠️ No simulation results found. Check path or run scan.</div>"
            )
        else:
            sections.append(self._repr_html_results_list(unique_id))

        param_stats = (
            self._get_parameter_stats() if n_results and hasattr(self, "df") else {}
        )
        if param_stats:
            param_html = (
                "<div style='background:linear-gradient(135deg,rgba(51,65,85,0.4) 0%,rgba(30,41,59,0.4) 100%);"
                "padding:12px;border-radius:8px;margin-bottom:12px;border:1px solid rgba(148,163,184,0.15);"
                "backdrop-filter:blur(10px);'>"
                "<b style='color:#bd93f9;'>📋 Parameters:</b> "
                "<small style='color:#6272a4;margin-left:8px;'>(first 8 varying numeric columns)</small><br>"
                "<table style='width:100%;margin-top:8px;border-collapse:collapse;font-size:0.9em;'>"
                "<tr style='border-bottom:2px solid rgba(98,114,164,0.25);'>"
                "<th style='text-align:left;padding:8px;color:#f8f8f2;'>Parameter</th>"
                "<th style='text-align:left;padding:8px;color:#f8f8f2;'>Unique Values</th>"
                "<th style='text-align:left;padding:8px;color:#f8f8f2;'>Range</th></tr>"
            )
            for param, info in list(param_stats.items())[:8]:
                unique_count = info["unique"]
                if unique_count > 1:
                    range_str = f"{info['min']:.4g} → {info['max']:.4g}"
                else:
                    range_str = f"{info['min']:.4g} (constant)"
                param_html += (
                    "<tr style='border-bottom:1px solid rgba(98,114,164,0.25);'>"
                    f"<td style='padding:6px 8px;'><code style='color:{NODE_COLOR_COMPUTE};'>"
                    f"{_html.escape(str(param))}</code></td>"
                    f"<td style='padding:6px 8px;color:{NODE_COLOR_PLOT};font-weight:600;'>"
                    f"{unique_count}</td>"
                    f"<td style='padding:6px 8px;color:#f8f8f2;font-family:monospace;'>"
                    f"{_html.escape(range_str)}</td></tr>"
                )
            param_html += "</table></div>"
            sections.append(param_html)

        sections.extend(
            [
                accessors_section_html(
                    [
                        (
                            "Navigation:",
                            [
                                ("job.find(...)", NODE_COLOR_COMPUTE),
                                ("job.columns", NODE_COLOR_COMPUTE),
                                ("job[0]", NODE_COLOR_COMPUTE),
                                ("job[:]", NODE_COLOR_COMPUTE),
                            ],
                        ),
                        (
                            "Analysis:",
                            [
                                ("job[0].fft", NODE_COLOR_ANALYSIS),
                                ("job[:].fft", NODE_COLOR_ANALYSIS),
                                ("job[0].analyze", NODE_COLOR_ANALYSIS),
                                ("job[0].solitons", NODE_COLOR_ANALYSIS),
                            ],
                        ),
                        (
                            "Plotting:",
                            [
                                ("job.mpl", NODE_COLOR_PLOT),
                                ("job[:].m.mpl", NODE_COLOR_PLOT),
                            ],
                        ),
                        (
                            "Utilities:",
                            [
                                ("job.force_rescan()", NODE_COLOR_UTIL),
                                ("job.dataframe", NODE_COLOR_UTIL),
                            ],
                        ),
                    ]
                ),
                examples_section_html(
                    "job.find(alpha=0.01)\njob[0].m\njob[0].fft.spectrum()\njob[:].fft.spectrum.compute_all()",
                    title="Quick Start",
                ),
            ]
        )

        return node_card_html(
            "MMPP Job Manager",
            icon="📊",
            subtitle="Top-level scan results, filtering and navigation.",
            badge=("ready", "#22c55e"),
            sections=sections,
            api=api_card,
            uid=f"mmpp-job-{unique_id}",
        )

    def _get_parameter_stats(self) -> dict:
        """Get statistics about parameter values across all results."""
        if self.df.empty:
            return {}

        stats = {}
        # Focus on numeric columns that vary
        numeric_cols = self.df.select_dtypes(include=["number"]).columns

        for col in numeric_cols:
            if col == "path":
                continue
            try:
                values = self.df[col].dropna()
                if len(values) > 0:
                    stats[col] = {
                        "unique": values.nunique(),
                        "min": values.min(),
                        "max": values.max(),
                    }
            except Exception:
                continue

        # Sort by number of unique values (descending) - varying parameters first
        return dict(sorted(stats.items(), key=lambda x: x[1]["unique"], reverse=True))

    def _repr_html_results_list(self, uid: str) -> str:
        """Build an expandable HTML block listing all scanned results with indices."""
        results = self.zarr_results
        if not results:
            return ""

        import html as _html

        # Determine which parameters vary so we can highlight them in the list
        param_stats = self._get_parameter_stats()
        varying = [p for p, s in param_stats.items() if s["unique"] > 1][
            :6
        ]  # at most 6 cols

        # Section container – collapsed by default when there are many results
        show_initially = len(results) <= 10
        list_id = f"results-list-{uid}"
        btn_id = f"results-btn-{uid}"
        n = len(results)

        # ---- header toggle button -------------------------------------------
        toggle_js = (
            f"var el=document.getElementById('{list_id}');"
            f"var btn=document.getElementById('{btn_id}');"
            f"if(el.style.display==='none'){{el.style.display='block';btn.textContent='▲ Hide {n} results';}}"
            f"else{{el.style.display='none';btn.textContent='▼ Show {n} results';}}"
        )
        open_label = f"▲ Hide {n} results" if show_initially else f"▼ Show {n} results"
        btn_html = (
            f'<button id="{btn_id}" onclick="{toggle_js}" '
            f'style="padding:4px 12px;background:linear-gradient(135deg,rgba(96,165,250,0.15),rgba(79,70,229,0.15));'
            f"border:1px solid rgba(96,165,250,0.3);border-radius:5px;color:#93c5fd;cursor:pointer;"
            f'font-size:0.8em;font-weight:600;float:right;">{open_label}</button>'
        )

        out = (
            f'<div style="background:linear-gradient(135deg,rgba(51,65,85,0.4),rgba(30,41,59,0.4));'
            f'padding:12px;border-radius:8px;margin-bottom:10px;border:1px solid rgba(148,163,184,0.15);">'
            f'<b style="color:#94a3b8;">📂 Scanned Results ({n}):</b>{btn_html}'
            f'<div style="clear:both;"></div>'
        )

        display_style = "block" if show_initially else "none"
        out += f'<div id="{list_id}" style="display:{display_style};margin-top:8px;overflow-x:auto;">'
        out += '<table style="width:100%;border-collapse:collapse;font-size:0.82em;">'

        # Table header
        th_style = "padding:5px 8px;font-weight:600;color:#cbd5e1;border-bottom:2px solid rgba(148,163,184,0.25);text-align:left;white-space:nowrap;"
        out += "<thead><tr>"
        out += f'<th style="{th_style}">#</th>'
        out += f'<th style="{th_style}">Path</th>'
        for p in varying:
            out += f'<th style="{th_style}">{_html.escape(p)}</th>'
        out += "</tr></thead><tbody>"

        # Rows – show first 30 directly, rest in a collapsible block
        VISIBLE = 30
        more_btn_id = f"results-more-btn-{uid}"

        def _fmt_val(v) -> str:
            if v is None:
                return '<span style="color:#475569;">—</span>'
            if isinstance(v, float):
                return f'<span style="color:#a5b4fc;">{v:.4g}</span>'
            return f'<span style="color:#a5b4fc;">{_html.escape(str(v))}</span>'

        def _short_path(full_path: str) -> str:
            """Return last 2 path components for display."""
            parts = str(full_path).replace("\\", "/").rstrip("/").split("/")
            short = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            return _html.escape(short)

        td_style = "padding:4px 8px;border-bottom:1px solid rgba(71,85,105,0.25);vertical-align:top;"
        code_style = (
            "background:rgba(15,23,42,0.7);padding:1px 6px;border-radius:4px;"
            "color:#60a5fa;font-family:monospace;cursor:pointer;"
            "border:1px solid rgba(71,85,105,0.4);"
        )
        path_style = (
            "font-family:monospace;font-size:0.88em;color:#cbd5e1;"
            "word-break:break-all;max-width:260px;"
        )

        for i, res in enumerate(results):
            attrs = res.attributes if isinstance(res.attributes, dict) else {}
            row_extra = (
                ""
                if i < VISIBLE
                else f'class="results-more-{uid}" style="display:none;"'
            )
            out += f"<tr {row_extra}>"
            # index cell with copyable job[i] snippet
            out += (
                f'<td style="{td_style}"><code style="{code_style}" '
                f'title="{_html.escape(str(res.path))}">'
                f"job[{i}]</code></td>"
            )
            # path cell
            out += f'<td style="{td_style}"><span style="{path_style}" title="{_html.escape(str(res.path))}">{_short_path(str(res.path))}</span></td>'
            # varying param values
            for p in varying:
                val = attrs.get(p)
                out += f'<td style="{td_style}">{_fmt_val(val)}</td>'
            out += "</tr>"

        out += "</tbody></table>"

        # "Show more" button if needed
        if n > VISIBLE:
            more_js = (
                f"var rows=document.querySelectorAll('.results-more-{uid}');"
                f"var btn=document.getElementById('{more_btn_id}');"
                f"var hidden=rows[0].style.display==='none';"
                f"rows.forEach(r=>r.style.display=hidden?'table-row':'none');"
                f"btn.textContent=hidden?'▲ Show fewer':'▼ Show {n - VISIBLE} more results';"
            )
            out += (
                f'<button id="{more_btn_id}" onclick="{more_js}" '
                f'style="margin-top:6px;padding:5px 14px;background:rgba(96,165,250,0.1);'
                f"border:1px solid rgba(96,165,250,0.25);border-radius:5px;color:#93c5fd;"
                f'cursor:pointer;font-size:0.8em;font-weight:600;">▼ Show {n - VISIBLE} more results</button>'
            )

        out += "</div></div>"
        return out

    @property
    def mpl(self) -> "MMPPlotter":
        """Get matplotlib plotter for all results."""
        try:
            from ..plotting import MMPPlotter
        except ImportError as exc:
            raise ImportError(
                "Plotting functionality not available. Install matplotlib."
            ) from exc
        return MMPPlotter(self.zarr_results, self)

    @property
    def matplotlib(self) -> "MMPPlotter":
        """Get matplotlib plotter for all results (alias for mpl)."""
        return self.mpl

    @property
    def fft(self) -> "FFT":
        """Get FFT analyzer.

        .. deprecated::
            Accessing ``job.fft`` returns the FFT interface for the **first**
            result only, which is misleading when multiple results are present.
            Use ``job[0].fft`` for a single result or ``job[:].fft`` for batch
            FFT across all results.
        """
        if not FFT_AVAILABLE:
            raise ImportError(
                "FFT functionality not available. Check fft module import."
            )
        if not self.zarr_results:
            raise ValueError("No zarr results available for FFT analysis.")

        if len(self.zarr_results) > 1:
            import warnings

            warnings.warn(
                f"job.fft operates on the FIRST result only ({len(self.zarr_results)} results "
                "available). Use job[0].fft for single-result FFT or job[:].fft for batch FFT.",
                stacklevel=2,
            )
        try:
            from ..fft import FFT
        except ImportError as exc:
            raise ImportError(
                "FFT functionality not available. Check fft module import."
            ) from exc
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
        underscore_patterns = re.findall(
            r"([a-zA-Z][a-zA-Z0-9]*)_([\-+]?[0-9]*\.?[0-9]+(?:[eE][\-+]?[0-9]+)?)", name
        )
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

            if _should_render_rich_progress():
                # Rich live progress is terminal-only. In VSCode/Jupyter it can
                # leave multiple stale render frames in the cell output.
                try:
                    from rich.progress import Progress

                    with Progress() as progress:
                        task = progress.add_task(
                            "[cyan]Scanning zarr folders...", total=len(zarr_folders)
                        )

                        for future in as_completed(future_to_path):
                            path = future_to_path[future]
                            try:
                                result = future.result()
                                results.append(result)
                            except Exception as exc:
                                log.error(f"{path} generated an exception: {exc}")
                            finally:
                                progress.advance(task)
                    return results
                except ImportError:
                    pass

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
        pcss_pattern = (
            rf"^{re.escape(CONTAINER_PREFIX)}/[^/]+/pcss_storage/([^/]+)(/.*)?"
        )
        pcss_match = re.match(pcss_pattern, path)
        if pcss_match:
            project = pcss_match.group(1)
            rest = pcss_match.group(2) or ""
            new_path = f"{STORAGE_ROOT}/projects/{project}{rest}".replace("//", "/")
            # Fallback to legacy if new path doesn't exist
            if not os.path.exists(new_path):
                legacy_path = f"{LEGACY_STORAGE_ROOT}/{project}{rest}".replace(
                    "//", "/"
                )
                if os.path.exists(legacy_path):
                    return legacy_path
            return new_path

        # Pattern 2: /projects/ paths in container
        # /mnt/local/kkingstoun/{user}/projects/{project}/{rest}
        projects_pattern = (
            rf"^{re.escape(CONTAINER_PREFIX)}/[^/]+/projects/([^/]+)(/.*)?"
        )
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
            if project_or_subdir in (
                "pcss_storage",
                "projects",
                ".config",
                ".local",
                ".cache",
            ):
                return path  # Don't translate special dirs
            new_path = f"{STORAGE_ROOT}/projects/{project_or_subdir}{rest}".replace(
                "//", "/"
            )
            # Fallback to legacy
            if not os.path.exists(new_path):
                legacy_path = (
                    f"{LEGACY_STORAGE_ROOT}/{project_or_subdir}{rest}".replace(
                        "//", "/"
                    )
                )
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
                            log.warning(
                                f"Path does not exist after translation: {translated_path}"
                            )
                            continue  # Skip this entry
                    else:
                        log.warning(f"Path does not exist: {path}")
                        continue  # Skip this entry

                # Filter out path from attributes
                attrs = {k: v for k, v in row.items() if k != "path"}
                result = ZarrJobResult(path, attrs)
                result._set_mmpp_ref(self)
                self.zarr_results.append(result)
                valid_paths.append(path)

            # Update DataFrame with valid paths only
            if len(valid_paths) < len(self.df):
                log.info(
                    f"Filtered {len(self.df) - len(valid_paths)} invalid paths from database"
                )
                self.df = self.df[
                    self.df["path"].apply(
                        lambda p: os.path.exists(p)
                        or os.path.exists(self._translate_path(p))
                    )
                ]
                # Update paths in DataFrame to translated versions
                self.df["path"] = self.df["path"].apply(
                    lambda p: self._translate_path(p) if not os.path.exists(p) else p
                )

            log.info(
                f"Loaded database from {db_path} ({len(self.zarr_results)} valid entries)"
            )
            self._sort_zarr_results()
            return True
        except Exception as e:
            log.warning(f"Failed to load database: {e}")
            return False

    def _sort_zarr_results(self) -> None:
        """Sort ``zarr_results`` by the first varying numeric parameter.

        When the folder structure encodes a swept parameter (e.g.
        ``b_0.010/``, ``b_0.020/``), this method detects that parameter
        from the parsed attributes and sorts in ascending order.

        Falls back to alphabetical path sorting when no single varying
        parameter is found.
        """
        if len(self.zarr_results) < 2:
            return

        # Collect numeric attributes for each job
        all_attrs: list[dict[str, Any]] = []
        for r in self.zarr_results:
            all_attrs.append(
                {
                    k: v
                    for k, v in r.attributes.items()
                    if isinstance(v, (int, float)) and k != "path"
                }
            )

        # Find the attribute(s) that actually vary across jobs
        if not all_attrs:
            self.zarr_results.sort(key=lambda r: r.path)
            return

        common_keys = set(all_attrs[0].keys())
        for a in all_attrs[1:]:
            common_keys &= set(a.keys())

        varying: list[tuple[str, int]] = []
        for key in common_keys:
            vals = {a[key] for a in all_attrs}
            if len(vals) > 1:
                varying.append((key, len(vals)))

        if len(varying) == 1:
            sort_key = varying[0][0]
            log.info(f"Auto-sorting {len(self.zarr_results)} jobs by '{sort_key}'")
            self.zarr_results.sort(key=lambda r: float(r.attributes.get(sort_key, 0)))
        elif len(varying) > 1:
            # Multiple varying params — pick the one with most unique values
            sort_key = max(varying, key=lambda x: x[1])[0]
            log.info(
                f"Multiple varying params detected ({[v[0] for v in varying]}); "
                f"sorting by '{sort_key}' (most unique values: {max(v[1] for v in varying)})"
            )
            self.zarr_results.sort(key=lambda r: float(r.attributes.get(sort_key, 0)))
        else:
            # No numeric variation — sort by path
            self.zarr_results.sort(key=lambda r: r.path)

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
                result = ZarrJobResult(res.path, res.attributes)
                result._set_mmpp_ref(self)
                self.zarr_results.append(result)

        # Sort by the first varying numeric parameter (auto-detected),
        # or fall back to path for deterministic ordering.
        self._sort_zarr_results()

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

    def find(self, **kwargs: Any) -> "BatchOperations":
        """
        Find zarr folders that match the given criteria.

        Returns a batch collection wrapper containing all matching
        :class:`~mmpp.core.job.ZarrJobResult` objects.

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
        BatchOperations
            Batch collection object containing matching ZarrJobResult objects.
            Supports indexing like `result[0]`, iteration, and
            plotting methods like `.mpl.plot()` and batch helpers.

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
        find_paths : Returns list of paths instead of batch wrapper
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
            from ..batch_operations import BatchOperations

            return BatchOperations([], self)

        # Filter DataFrame - use nearest match for numeric values
        filtered_df = self.df.copy()

        for key, target_value in kwargs.items():
            if key not in filtered_df.columns:
                log.error(
                    f"Column '{key}' not found in database. Available columns: {list(filtered_df.columns)}"
                )
                from ..batch_operations import BatchOperations

                return BatchOperations([], self)

            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(filtered_df[key]):
                # Find nearest value for numeric columns
                column_values = filtered_df[key].values

                # Handle NaN values
                valid_mask = ~pd.isna(column_values)
                if not valid_mask.any():
                    log.warning(f"All values in column '{key}' are NaN")
                    filtered_df = filtered_df.iloc[0:0]  # Empty DataFrame
                    break

                valid_values = column_values[valid_mask]
                differences = np.abs(valid_values - target_value)
                nearest_value = valid_values[np.argmin(differences)]

                log.info(
                    f"find({key}={target_value}): Using nearest value {nearest_value}"
                )

                # Filter to rows with nearest value
                filtered_df = filtered_df[filtered_df[key] == nearest_value]
            else:
                # Exact match for non-numeric columns
                filtered_df = filtered_df[filtered_df[key] == target_value]

        # Get matching ZarrJobResults
        matching_paths = set(filtered_df["path"])
        matching_results = [
            res for res in self.zarr_results if res.path in matching_paths
        ]

        from ..batch_operations import BatchOperations

        return BatchOperations(matching_results, self, _filter_kwargs=kwargs)

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
        if hasattr(proxy, "results"):
            return [job.path for job in proxy.results]
        return [job.path for job in proxy]  # type: ignore
