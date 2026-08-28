from collections.abc import MutableMapping
from html import escape
from typing import Any

from .constants import RICH_AVAILABLE

if RICH_AVAILABLE:
    from rich.console import Console
    from rich.table import Table


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
        # Try rich text table for terminals (export only, no direct print to avoid
        # double-display in Jupyter where _repr_html_ provides the rich output)
        if RICH_AVAILABLE:
            try:
                from io import StringIO

                buf = StringIO()
                console = Console(
                    file=buf, width=120, force_terminal=False, color_system=None
                )
                table = self._rich_table()
                if table is not None:
                    console.print(table)
                    return buf.getvalue()
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
        """Rich HTML representation for Jupyter notebooks (app-consistent dark theme)."""
        try:
            from uuid import uuid4

            uid = uuid4().hex[:8]
            attr_count = len(self._attrs)
            rows = []
            for key in sorted(self._attrs.keys()):
                val = self._attrs[key]
                rows.append(
                    "<tr>"
                    f"<td style='padding:4px 10px;font-family:monospace;color:#93c5fd;"
                    f"border-bottom:1px solid #334155;white-space:nowrap'>{escape(str(key))}</td>"
                    f"<td style='padding:4px 10px;color:#cbd5e1;"
                    f"border-bottom:1px solid #334155;word-break:break-all'>{escape(str(val))}</td>"
                    "</tr>"
                )
            body = "\n".join(rows)
            return f"""
            <div style='background:#1e293b;border-radius:8px;padding:12px 16px;
                        font-size:13px;max-width:720px;border:1px solid #334155;
                        font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif'>
              <div style='color:#64748b;font-size:11px;text-transform:uppercase;
                          letter-spacing:.08em;margin-bottom:4px'>Simulation attributes</div>
              <div style='display:flex;align-items:center;justify-content:space-between;
                          margin-bottom:10px'>
                <span style='color:#475569;font-size:11px'>
                  {attr_count} {"entry" if attr_count == 1 else "entries"}
                </span>
                <input id='attr-search-{uid}'
                  type='text' placeholder='🔍  filter attributes…'
                  oninput='(function(inp){{
                    var q=inp.value.toLowerCase();
                    document.querySelectorAll("#attr-table-{uid} tbody tr").forEach(function(tr){{
                      var key=tr.children[0].textContent.toLowerCase();
                      var val=tr.children[1].textContent.toLowerCase();
                      tr.style.display=(key.includes(q)||val.includes(q))?"":"none";
                    }});
                  }})(this)'
                  style='background:#0f172a;border:1px solid #334155;border-radius:4px;
                         padding:4px 10px;color:#cbd5e1;font-size:12px;width:220px;
                         outline:none'/>
              </div>
              <table id='attr-table-{uid}' style='border-collapse:collapse;width:100%'>
                <thead>
                  <tr>
                    <th style='text-align:left;padding:4px 10px;color:#475569;
                               font-weight:500;font-size:11px;
                               border-bottom:1px solid #475569'>key</th>
                    <th style='text-align:left;padding:4px 10px;color:#475569;
                               font-weight:500;font-size:11px;
                               border-bottom:1px solid #475569'>value</th>
                  </tr>
                </thead>
                <tbody>
                  {body}
                </tbody>
              </table>
            </div>"""
        except Exception:
            return None
