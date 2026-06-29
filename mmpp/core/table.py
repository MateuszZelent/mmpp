from __future__ import annotations

from html import escape as html_escape
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from .job import ZarrJobResult


def _as_list(value: Optional[Union[str, Sequence[str]]]) -> Optional[list[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


class TablePlotAccessor:
    """Callable plot namespace for ``job[0].table.plot``.

    Accessing the object in a notebook shows a helper card. Calling it keeps the
    short plotting syntax: ``job[0].table.plot(x="t", y="mx")``.
    """

    def __init__(self, table: "TableAwareWrapper"):
        self._table = table

    def __call__(
        self,
        x: Optional[str] = None,
        y: Optional[Union[str, Sequence[str]]] = None,
        *,
        kind: str = "line",
        max_rows: Optional[int] = None,
        start: int = 0,
        ax: Any = None,
        **kwargs: Any,
    ) -> Any:
        return self._table._plot(
            x=x,
            y=y,
            kind=kind,
            max_rows=max_rows,
            start=start,
            ax=ax,
            **kwargs,
        )

    def interactive(self, *, show: bool = True, max_rows: int = 2000) -> "TableInteractiveViewer":
        """Open the interactive table toolbar."""
        return self._table.interactive(show=show, max_rows=max_rows)

    def preview(
        self,
        n: int = 10,
        columns: Optional[Union[str, Sequence[str]]] = None,
        *,
        start: int = 0,
    ) -> Any:
        """Preview rows before plotting."""
        return self._table.preview(n=n, columns=columns, start=start)

    def _repr_html_(self) -> str:
        import uuid as _uuid
        from mmpp._repr_helpers import api_help_html, html_tabs

        uid = str(_uuid.uuid4())[:8]
        table = self._table
        default_x = table._default_x_column() or "row index"
        default_y = ", ".join(table._default_y_columns(None)[:4]) or "auto"

        _sec = (
            "background:linear-gradient(135deg,rgba(51,65,85,0.4) 0%,"
            "rgba(30,41,59,0.4) 100%);padding:12px;border-radius:8px;"
            "margin-bottom:12px;border:1px solid rgba(148,163,184,0.15);"
            "backdrop-filter:blur(10px);"
        )
        _code = (
            "background:rgba(15,23,42,0.8);padding:5px 10px;border-radius:5px;"
            "display:inline-block;margin:4px;font-family:'Courier New',monospace;"
            "font-size:0.85em;border:1px solid rgba(71,85,105,0.4);font-weight:500;"
        )

        def _metric(label: str, value: object, color: str = "#cbd5e1") -> str:
            return (
                f"<b style='color:#94a3b8'>{html_escape(label)}:</b> "
                f"<code style='background:rgba(15,23,42,0.6);padding:4px 10px;"
                f"border-radius:5px;font-size:0.9em;color:{color};"
                f"border:1px solid rgba(71,85,105,0.3);'>{html_escape(str(value))}</code><br>"
            )

        status_html = (
            f"<div style='{_sec}'>"
            + _metric("job", table._job_result.name)
            + _metric("table", table.name)
            + _metric("shape", f"{table.n_rows} rows x {len(table.columns)} columns")
            + _metric("default x", default_x)
            + _metric("default y", default_y)
            + "</div>"
        )

        groups = [
            ("Plot:", [("(...)", "#a78bfa"), ("(x='t', y='mx')", "#a78bfa")]),
            ("Interactive:", [(".interactive()", "#38bdf8")]),
            ("Preview:", [(".preview(n=20)", "#34d399")]),
        ]
        accessors_inner = ""
        for label, items in groups:
            chips = "".join(
                f"<code style='{_code}color:{color};'>{html_escape(name)}</code>"
                for name, color in items
            )
            accessors_inner += (
                f"<small style='color:#64748b;margin-right:6px;'>{html_escape(label)}</small>"
                f"{chips}<br>"
            )
        accessors_html = (
            f"<div style='{_sec}'>"
            "<b style='color:#94a3b8;'>ACCESSORS &amp; METHODS</b><br>"
            f"{accessors_inner}</div>"
        )

        example_code = (
            "job[0].table.plot\n"
            "job[0].table.plot(x='t', y='mx')\n"
            "job[0].table.plot(x='t', y=['mx', 'my'], kind='scatter')\n"
            "job[0].table.plot.interactive()"
        )
        examples_html = (
            f"<div style='{_sec}'>"
            "<b style='color:#94a3b8;'>Examples</b><br>"
            "<pre style='margin:6px 0 0 0;background:rgba(15,23,42,0.85);padding:10px;"
            "border-radius:6px;color:#e2e8f0;overflow-x:auto;font-size:0.85em;'>"
            f"<code>{html_escape(example_code)}</code></pre></div>"
        )

        api_card = api_help_html(
            self,
            title="Table plot API help",
            prefix="job[0].table.plot",
            subtitle="Callable plotting helper for table columns.",
            methods=["__call__", "interactive", "preview"],
            chrome=False,
        )

        overview_html = (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;padding:4px 0 0 0;\">"
            "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;margin-bottom:4px;'>"
            "Table plot helper"
            "<span style='background:#22c55e;color:#0f172a;padding:1px 6px;"
            "border-radius:10px;font-size:10px;margin-left:8px'>ready</span>"
            "</div>"
            "<div style='font-size:0.85em;color:#94a3b8;margin-bottom:12px;'>"
            "Plot scalar table columns or open the interactive table toolbar.</div>"
            + status_html
            + accessors_html
            + examples_html
            + "</div>"
        )

        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:18px;margin:10px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;box-shadow:0 10px 25px rgba(0,0,0,0.3),"
            "0 0 0 1px rgba(148,163,184,0.1) inset;\">"
            + html_tabs(
                [("Overview", overview_html), ("API", api_card)],
                uid=f"table-plot-{uid}",
            )
            + "</div>"
        )

    def __repr__(self) -> str:
        return "TablePlotAccessor(prefix='job[0].table.plot')"


class TableAwareWrapper:
    """Notebook-friendly wrapper around the top-level ``table`` zarr group.

    The wrapper intentionally keeps zarr-like access available through
    ``keys()``, ``array_keys()``, ``attrs`` and ``__getitem__`` while adding the
    high-level helpers users expect in notebooks.
    """

    def __init__(self, job_result: "ZarrJobResult", name: str, group: Any):
        self._job_result = job_result
        self.name = name
        self._group = group
        self._plot_accessor = TablePlotAccessor(self)

    @property
    def z(self) -> Any:
        """Return the underlying zarr group."""
        return self._group

    @property
    def attrs(self) -> Any:
        """Return table group attributes."""
        return getattr(self._group, "attrs", {})

    def __getitem__(self, key: str) -> Any:
        return self._group[key]

    def __contains__(self, key: str) -> bool:
        try:
            return key in self._group
        except Exception:
            return key in self.keys()

    def __iter__(self) -> Iterable[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.columns)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self.columns:
            return self._group[name]
        return getattr(self._group, name)

    def keys(self) -> list[str]:
        try:
            return list(self._group.keys())
        except Exception:
            return self.columns

    def array_keys(self) -> list[str]:
        try:
            return list(self._group.array_keys())
        except Exception:
            return [
                key
                for key in self.keys()
                if hasattr(self._group[key], "shape")
                and hasattr(self._group[key], "__getitem__")
            ]

    def group_keys(self) -> list[str]:
        try:
            return list(self._group.group_keys())
        except Exception:
            return []

    @property
    def columns(self) -> list[str]:
        """Return table columns stored as arrays."""
        return self.array_keys()

    @property
    def n_rows(self) -> int:
        """Best-effort number of rows inferred from the first table column."""
        lengths = []
        for column in self.columns:
            shape = getattr(self._group[column], "shape", None)
            if shape:
                lengths.append(int(shape[0]))
        return min(lengths) if lengths else 0

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(rows, columns)`` for the tabular view."""
        return (self.n_rows, len(self.columns))

    @property
    def plot(self) -> TablePlotAccessor:
        """Callable plotting helper shown as a notebook card."""
        return self._plot_accessor

    def _normalize_columns(self, columns: Optional[Union[str, Sequence[str]]]) -> list[str]:
        requested = _as_list(columns)
        available = self.columns
        if requested is None:
            return available
        missing = [column for column in requested if column not in available]
        if missing:
            raise KeyError(
                "Unknown table column(s): "
                + ", ".join(missing)
                + ". Available columns: "
                + ", ".join(available)
            )
        return requested

    def _read_column(self, column: str, start: int, stop: int) -> dict[str, Any]:
        array = self._group[column]
        data = np.asarray(array[start:stop])
        if data.ndim <= 1:
            return {column: data}
        flat = data.reshape(data.shape[0], -1)
        if flat.shape[1] == 1:
            return {column: flat[:, 0]}
        return {
            f"{column}_{component}": flat[:, component]
            for component in range(flat.shape[1])
        }

    def to_dataframe(
        self,
        columns: Optional[Union[str, Sequence[str]]] = None,
        *,
        max_rows: Optional[int] = None,
        start: int = 0,
    ) -> Any:
        """Load selected table columns into a pandas DataFrame.

        Parameters
        ----------
        columns:
            Column name or names. By default all array columns are loaded.
        max_rows:
            Optional row limit for quick notebook previews.
        start:
            First row index.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Table helpers require pandas. Install with: pip install pandas"
            ) from exc

        selected = self._normalize_columns(columns)
        start = max(int(start), 0)
        n_rows = self.n_rows
        if max_rows is None:
            stop = n_rows
        else:
            stop = min(n_rows, start + max(int(max_rows), 0))

        data: dict[str, Any] = {}
        for column in selected:
            data.update(self._read_column(column, start, stop))
        return pd.DataFrame(data)

    def preview(
        self,
        n: int = 10,
        columns: Optional[Union[str, Sequence[str]]] = None,
        *,
        start: int = 0,
    ) -> Any:
        """Return the first rows as a pandas DataFrame."""
        return self.to_dataframe(columns=columns, max_rows=n, start=start)

    def _default_x_column(self) -> str | None:
        candidates = ("t", "time", "Time", "step", "iteration")
        for candidate in candidates:
            if candidate in self.columns:
                return candidate
        return None

    def _default_y_columns(self, x: str | None = None) -> list[str]:
        numeric = []
        for column in self.columns:
            if column == x:
                continue
            dtype = getattr(self._group[column], "dtype", None)
            if dtype is None or np.issubdtype(np.dtype(dtype), np.number):
                numeric.append(column)
        return numeric[:6]

    def _plot(
        self,
        x: Optional[str] = None,
        y: Optional[Union[str, Sequence[str]]] = None,
        *,
        kind: str = "line",
        max_rows: Optional[int] = None,
        start: int = 0,
        ax: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Quick matplotlib plot for selected table columns."""
        import matplotlib.pyplot as plt

        x_column = x or self._default_x_column()
        y_columns = _as_list(y) or self._default_y_columns(x_column)
        load_columns = list(y_columns)
        if x_column is not None:
            load_columns = [x_column] + [column for column in load_columns if column != x_column]

        df = self.to_dataframe(load_columns, max_rows=max_rows, start=start)
        if ax is None:
            _, ax = plt.subplots()

        if not y_columns:
            ax.text(0.5, 0.5, "No numeric table columns to plot", ha="center", va="center")
            return ax

        for column in y_columns:
            expanded = [name for name in df.columns if name == column or name.startswith(f"{column}_")]
            for expanded_column in expanded:
                if kind == "scatter":
                    if x_column is None:
                        ax.scatter(df.index, df[expanded_column], label=expanded_column, **kwargs)
                    else:
                        ax.scatter(df[x_column], df[expanded_column], label=expanded_column, **kwargs)
                else:
                    if x_column is None:
                        ax.plot(df.index, df[expanded_column], label=expanded_column, **kwargs)
                    else:
                        ax.plot(df[x_column], df[expanded_column], label=expanded_column, **kwargs)

        ax.set_title(f"{self._job_result.name}.{self.name}")
        ax.set_xlabel(x_column or "row")
        ax.set_ylabel(", ".join(y_columns))
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        return ax

    def interactive(self, *, show: bool = True, max_rows: int = 2000) -> "TableInteractiveViewer":
        """Create an interactive table toolbar for notebook exploration."""
        viewer = TableInteractiveViewer(self, max_rows=max_rows)
        if show:
            viewer.show()
        return viewer

    def _repr_html_(self) -> str:
        import uuid as _uuid
        from mmpp._repr_helpers import api_help_html, html_tabs

        uid = str(_uuid.uuid4())[:8]
        columns = self.columns
        preview_columns = ", ".join(columns[:10])
        if len(columns) > 10:
            preview_columns += f", and {len(columns) - 10} more"

        _sec = (
            "background:linear-gradient(135deg,rgba(51,65,85,0.4) 0%,"
            "rgba(30,41,59,0.4) 100%);padding:12px;border-radius:8px;"
            "margin-bottom:12px;border:1px solid rgba(148,163,184,0.15);"
            "backdrop-filter:blur(10px);"
        )
        _code = (
            "background:rgba(15,23,42,0.8);padding:5px 10px;border-radius:5px;"
            "display:inline-block;margin:4px;font-family:'Courier New',monospace;"
            "font-size:0.85em;border:1px solid rgba(71,85,105,0.4);font-weight:500;"
        )

        def _metric(label: str, value: object, color: str = "#cbd5e1") -> str:
            return (
                f"<b style='color:#94a3b8'>{html_escape(label)}:</b> "
                f"<code style='background:rgba(15,23,42,0.6);padding:4px 10px;"
                f"border-radius:5px;font-size:0.9em;color:{color};"
                f"border:1px solid rgba(71,85,105,0.3);'>{html_escape(str(value))}</code><br>"
            )

        status_html = (
            f"<div style='{_sec}'>"
            + _metric("job", self._job_result.name)
            + _metric("table", self.name)
            + _metric("shape", f"{self.n_rows} rows x {len(columns)} columns")
            + _metric("columns", preview_columns or "none")
            + "</div>"
        )

        groups = [
            ("Preview:", [(".preview(n=10)", "#38bdf8"), (".to_dataframe()", "#38bdf8")]),
            ("Plotting:", [(".plot(x='t', y='mx')", "#a78bfa"), (".interactive()", "#a78bfa")]),
            ("Zarr:", [(".keys()", "#fb923c"), (".attrs", "#fb923c"), (".z", "#fb923c")]),
        ]
        accessors_inner = ""
        for label, items in groups:
            chips = "".join(
                f"<code style='{_code}color:{color};'>{html_escape(name)}</code>"
                for name, color in items
            )
            accessors_inner += (
                f"<small style='color:#64748b;margin-right:6px;'>{html_escape(label)}</small>"
                f"{chips}<br>"
            )
        accessors_html = (
            f"<div style='{_sec}'>"
            "<b style='color:#94a3b8;'>ACCESSORS &amp; METHODS</b><br>"
            f"{accessors_inner}</div>"
        )

        example_code = (
            "job[0].table\n"
            "job[0].table.preview(n=20)\n"
            "job[0].table.plot(x='t', y=['mx', 'my'])\n"
            "job[0].table.interactive()"
        )
        examples_html = (
            f"<div style='{_sec}'>"
            "<b style='color:#94a3b8;'>Examples</b><br>"
            "<pre style='margin:6px 0 0 0;background:rgba(15,23,42,0.85);padding:10px;"
            "border-radius:6px;color:#e2e8f0;overflow-x:auto;font-size:0.85em;'>"
            f"<code>{html_escape(example_code)}</code></pre></div>"
        )

        api_card = api_help_html(
            self,
            title="Table API help",
            prefix="job[0].table",
            subtitle="Interactive table preview and plotting helpers.",
            properties=[
                ("columns", "Available table columns"),
                ("shape", "Rows and columns in the tabular view"),
                ("attrs", "Underlying table group attributes"),
                ("z", "Raw zarr group"),
            ],
            methods=["preview", "to_dataframe", "plot", "interactive", "keys"],
            chrome=False,
        )

        overview_html = (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;padding:4px 0 0 0;\">"
            "<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;margin-bottom:4px;'>"
            "Table interface"
            "<span style='background:#22c55e;color:#0f172a;padding:1px 6px;"
            "border-radius:10px;font-size:10px;margin-left:8px'>ready</span>"
            "</div>"
            "<div style='font-size:0.85em;color:#94a3b8;margin-bottom:12px;'>"
            "Preview, DataFrame export and quick plotting for job table data.</div>"
            + status_html
            + accessors_html
            + examples_html
            + "</div>"
        )

        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:18px;margin:10px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;box-shadow:0 10px 25px rgba(0,0,0,0.3),"
            "0 0 0 1px rgba(148,163,184,0.1) inset;\">"
            + html_tabs(
                [("Overview", overview_html), ("API", api_card)],
                uid=f"table-{uid}",
            )
            + "</div>"
        )

    def __repr__(self) -> str:
        return f"TableAwareWrapper(columns={len(self.columns)}, rows={self.n_rows})"


class TableInteractiveViewer:
    """Small ipywidgets toolbar for table preview and plotting."""

    def __init__(self, table: TableAwareWrapper, *, max_rows: int = 2000):
        self.table = table
        self.max_rows = int(max_rows)
        self._widget = None
        self._output = None
        self._status = None

    def _make_widget(self) -> Any:
        try:
            import ipywidgets as widgets
        except ImportError as exc:
            raise ImportError(
                "Interactive table viewer requires ipywidgets. "
                "Install with: pip install ipywidgets"
            ) from exc

        columns = self.table.columns
        default_x = self.table._default_x_column()
        default_y = self.table._default_y_columns(default_x)

        x_options = [("(row index)", "__index__")] + [(column, column) for column in columns]
        y_options = [(column, column) for column in columns]

        x_dropdown = widgets.Dropdown(
            options=x_options,
            value=default_x or "__index__",
            description="X",
            layout=widgets.Layout(width="230px"),
        )
        y_select = widgets.SelectMultiple(
            options=y_options,
            value=tuple(default_y),
            description="Y",
            rows=min(max(len(columns), 4), 10),
            layout=widgets.Layout(width="260px"),
        )
        kind_dropdown = widgets.ToggleButtons(
            options=[("line", "line"), ("scatter", "scatter"), ("table", "table")],
            value="line",
            description="View",
            button_style="",
            layout=widgets.Layout(width="330px"),
            style={"button_width": "90px"},
        )
        start_box = widgets.BoundedIntText(
            value=0,
            min=0,
            max=max(self.table.n_rows - 1, 0),
            description="Start",
            layout=widgets.Layout(width="170px"),
        )
        rows_box = widgets.BoundedIntText(
            value=min(self.max_rows, max(self.table.n_rows, 1)),
            min=1,
            max=max(self.table.n_rows, 1),
            description="Rows",
            layout=widgets.Layout(width="170px"),
        )
        render_button = widgets.Button(
            description="Render",
            button_style="primary",
            icon="line-chart",
            layout=widgets.Layout(width="120px"),
        )
        preview_button = widgets.Button(
            description="Preview",
            button_style="info",
            icon="table",
            layout=widgets.Layout(width="120px"),
        )
        self._status = widgets.HTML(
            value=self._status_html(
                "ready",
                f"{self.table.n_rows} rows, {len(columns)} columns",
            )
        )
        self._output = widgets.Output(
            layout=widgets.Layout(
                border="1px solid #334155",
                padding="10px",
                min_height="260px",
            )
        )

        def render(as_table: bool = False) -> None:
            import matplotlib.pyplot as plt
            from IPython.display import display

            x_value = x_dropdown.value
            y_values = list(y_select.value)
            start = int(start_box.value)
            rows = int(rows_box.value)
            x_column = None if x_value == "__index__" else str(x_value)
            view_kind = "table" if as_table else kind_dropdown.value

            self._status.value = self._status_html("rendering", "updating output")
            with self._output:
                self._output.clear_output(wait=True)
                if view_kind == "table":
                    display(
                        self.table.preview(
                            n=rows,
                            columns=([x_column] if x_column else []) + y_values or None,
                            start=start,
                        )
                    )
                else:
                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    self.table.plot(
                        x=x_column,
                        y=y_values or None,
                        kind=str(view_kind),
                        max_rows=rows,
                        start=start,
                        ax=ax,
                    )
                    fig.tight_layout()
                    display(fig)
                    plt.close(fig)
            self._status.value = self._status_html("ready", "output updated")

        render_button.on_click(lambda _button: render(False))
        preview_button.on_click(lambda _button: render(True))

        controls = widgets.VBox(
            [
                widgets.HTML(
                    "<div style='font-weight:700;color:#e2e8f0;"
                    "background:#0f172a;border:1px solid #334155;border-radius:8px;"
                    "padding:8px 10px;'>Table toolbar</div>"
                ),
                widgets.HBox([x_dropdown, kind_dropdown]),
                widgets.HBox([y_select, widgets.VBox([start_box, rows_box])]),
                widgets.HBox([render_button, preview_button, self._status]),
            ],
            layout=widgets.Layout(
                border="1px solid #334155",
                padding="12px",
                margin="8px 0",
            ),
        )
        widget = widgets.VBox([controls, self._output])
        return widget

    @staticmethod
    def _status_html(state: str, message: str) -> str:
        color = "#22c55e" if state == "ready" else "#f59e0b"
        return (
            f"<span style='color:{color};font-weight:700'>{html_escape(state)}</span>"
            f"<span style='color:#64748b;margin-left:8px'>{html_escape(message)}</span>"
        )

    @property
    def widget(self) -> Any:
        if self._widget is None:
            self._widget = self._make_widget()
        return self._widget

    def show(self) -> "TableInteractiveViewer":
        from IPython.display import display

        display(self.widget)
        return self

    def _repr_mimebundle_(self, include: Any = None, exclude: Any = None) -> dict[str, Any]:
        widget = self.widget
        bundle = getattr(widget, "_repr_mimebundle_", None)
        if bundle is not None:
            return bundle(include=include, exclude=exclude)
        return {"text/plain": repr(self)}

    def __repr__(self) -> str:
        return f"TableInteractiveViewer({self.table!r})"
