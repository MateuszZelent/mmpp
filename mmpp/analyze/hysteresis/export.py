"""Export utilities for hysteresis results."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pandas as pd


class HysteresisExporter:
    """Exporter for table/metrics/figure/report artifacts."""

    def __init__(self, result):
        self._result = result

    def to_dataframe(self) -> pd.DataFrame:
        """Return loop sample DataFrame."""
        return self._result.to_dataframe()

    def to_csv(self, path) -> Path:
        """Export loop samples to CSV."""
        target = Path(path)
        self.to_dataframe().to_csv(target, index=False)
        return target

    def to_json(self, path) -> Path:
        """Export full JSON payload (data + metrics + metadata)."""
        target = Path(path)
        payload = {
            "metadata": dict(self._result.metadata),
            "metrics": self._result.metrics.report().to_dict(orient="records"),
            "data": self.to_dataframe().to_dict(orient="records"),
        }
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return target

    def figure_to_svg(self, fig, path) -> Path:
        target = Path(path)
        fig.savefig(target, format="svg", bbox_inches="tight")
        return target

    def figure_to_pdf(self, fig, path) -> Path:
        target = Path(path)
        fig.savefig(target, format="pdf", bbox_inches="tight")
        return target

    def figure_to_png(self, fig, path) -> Path:
        target = Path(path)
        fig.savefig(target, format="png", bbox_inches="tight")
        return target

    def generate_report(self, path, fmt: str = "html") -> Path:
        """Generate a self-contained HTML report."""
        target = Path(path)
        fmt_norm = str(fmt).lower()
        if target.suffix:
            fmt_norm = target.suffix.lstrip(".").lower()
        if fmt_norm != "html":
            raise ValueError("Only fmt='html' is supported in this exporter")

        fig, _ax, _meta = self._result.plot.loop(show_hc=True, show_mr=True, show_ms=True)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self._result.config.dpi, bbox_inches="tight")
        buf.seek(0)
        image64 = base64.b64encode(buf.read()).decode("ascii")
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:
            pass

        metrics_html = self._result.metrics.report().to_html(index=False, border=0)
        metadata_items = "".join(
            f"<tr><td style='padding:4px 8px;color:#93c5fd;'>{key}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{value}</td></tr>"
            for key, value in sorted(self._result.metadata.items())
        )

        html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Hysteresis Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; margin: 20px; background:#0f172a; color:#e2e8f0; }}
    h1, h2 {{ color:#f1f5f9; }}
    .card {{ border:1px solid #334155; border-radius:10px; padding:12px; margin:12px 0; background:#1e293b; }}
    table {{ width:100%; border-collapse:collapse; }}
    td, th {{ padding:6px 8px; border-bottom:1px solid #334155; text-align:left; }}
    img {{ max-width:100%; border-radius:8px; border:1px solid #334155; background:#0b1220; }}
  </style>
</head>
<body>
  <h1>Hysteresis Report</h1>
  <div class="card">
    <h2>Loop</h2>
    <img src="data:image/png;base64,{image64}" alt="loop"/>
  </div>
  <div class="card">
    <h2>Metadata</h2>
    <table>{metadata_items}</table>
  </div>
  <div class="card">
    <h2>Metrics</h2>
    {metrics_html}
  </div>
</body>
</html>
"""
        target.write_text(html, encoding="utf-8")
        return target

    def export(self, path, fmt: str | None = None) -> Path:
        """Dispatch export based on explicit format or path suffix."""
        target = Path(path)
        fmt_norm = (fmt or target.suffix.lstrip(".") or self._result.config.default_export_format).lower()

        if fmt_norm == "csv":
            return self.to_csv(target)
        if fmt_norm == "json":
            return self.to_json(target)
        if fmt_norm == "html":
            return self.generate_report(target, fmt="html")
        raise ValueError(f"Unsupported export format: {fmt_norm}")


__all__ = ["HysteresisExporter"]
