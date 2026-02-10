from collections.abc import MutableMapping
from html import escape
from typing import Any
from uuid import uuid4

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
        """Rich HTML representation for Jupyter notebooks."""
        try:
            table_id = f"attrs-{uuid4().hex}"
            attr_count = len(self._attrs)
            rows = []
            for key in sorted(self._attrs.keys()):
                val = self._attrs[key]
                rows.append(
                    "<tr>"
                    f"<td style='padding:6px 8px; font-family:monospace; color:#93c5fd; border-bottom:1px solid rgba(71,85,105,0.35);'>{escape(str(key))}</td>"
                    f"<td style='padding:6px 8px; border-bottom:1px solid rgba(71,85,105,0.35);'><pre style='margin:0; white-space:pre-wrap; color:#cbd5e1; font-family:monospace;'>{escape(repr(val))}</pre></td>"
                    "</tr>"
                )
            body = "\n".join(rows)
            html_table = f"""
            <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif; border:2px solid #334155; border-radius:12px; padding:16px; margin:10px 0; background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%); color:#e2e8f0; box-shadow:0 10px 22px rgba(0,0,0,0.28);">
              <div style="margin-bottom:12px;">
                <div style="font-size:1.1em; font-weight:600; color:#f1f5f9;">Simulation Attributes</div>
                <div style="color:#94a3b8; margin-top:4px;">Entries: <span style="color:#cbd5e1;">{attr_count}</span></div>
              </div>

              <div style="background:rgba(15,23,42,0.6); padding:10px; border-radius:8px; border:1px solid rgba(148,163,184,0.2);">
                <table id="{table_id}" style="width:100%; border-collapse:collapse; font-size:0.9em;">
                  <thead>
                    <tr style="text-align:left; background:rgba(51,65,85,0.6);">
                      <th style="padding:6px 8px; color:#e2e8f0; cursor:pointer;">Key</th>
                      <th style="padding:6px 8px; color:#e2e8f0; cursor:pointer;">Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {body}
                  </tbody>
                </table>
              </div>
            </div>
            <script>
              (function() {{
                const table = document.getElementById("{table_id}");
                if (!table) return;
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

    def _repr_mimebundle_(self, include=None, exclude=None):
        """Return HTML + plain text bundle for notebook frontends."""
        html = self._repr_html_()
        text = self.__repr__()
        if html:
            return {"text/html": html, "text/plain": text}
        return {"text/plain": text}
