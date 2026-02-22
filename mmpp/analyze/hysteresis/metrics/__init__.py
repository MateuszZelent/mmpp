"""Metrics accessor for hysteresis results."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from html import escape as _esc
from typing import Any

import numpy as np
import pandas as pd

from .core import (
    CoerciveFieldResult,
    RemanenceResult,
    SaturationResult,
    SusceptibilityResult,
    compute_coercive_field,
    compute_exchange_bias,
    compute_loop_area,
    compute_max_susceptibility,
    compute_remanence,
    compute_saturation_points,
    compute_squareness,
)
from .noise import (
    AnomalyReport,
    NoiseStats,
    auto_filter,
    detect_anomalies,
    estimate_noise_level,
)
from .registry import get_registered_metric, iter_registered_metrics
from .stability import CycleStabilityAnalysis, analyze_cycle_stability
from .uncertainty import ConfidenceIntervalResult, bootstrap_confidence_interval


class MetricsAccessor:
    """Lazy-access metrics interface for :class:`HysteresisResult`."""

    def __init__(self, result):
        self._result = result
        self._cache: dict[str, Any] = {}

    @property
    def _field(self) -> np.ndarray:
        return np.asarray(self._result.field, dtype=float)

    @property
    def _mag_raw(self) -> np.ndarray:
        return np.asarray(self._result.magnetization, dtype=float)

    @property
    def _branches(self):
        return self._result.branches

    @property
    def _field_unit(self) -> str:
        return str(self._result.metadata.get("field_unit", "input"))

    def _processed_magnetization(self) -> np.ndarray:
        key = "processed_magnetization"
        if key in self._cache:
            return self._cache[key]

        cfg = self._result.config
        mag = self._mag_raw.copy()

        noise_stats = estimate_noise_level(mag, self._field)
        self._cache["noise_stats"] = noise_stats

        should_filter = bool(cfg.auto_filter) or bool(cfg.filter_method)
        if should_filter:
            mag = auto_filter(mag, noise_stats, cfg)

        self._cache[key] = np.asarray(mag, dtype=float)
        return self._cache[key]

    @property
    def coercive_field(self) -> CoerciveFieldResult:
        if "coercive_field" not in self._cache:
            self._cache["coercive_field"] = compute_coercive_field(
                self._field,
                self._processed_magnetization(),
                self._branches,
                unit=self._field_unit,
            )
        return self._cache["coercive_field"]

    @property
    def remanence(self) -> RemanenceResult:
        if "remanence" not in self._cache:
            self._cache["remanence"] = compute_remanence(
                self._field,
                self._processed_magnetization(),
                self._branches,
            )
        return self._cache["remanence"]

    @property
    def saturation_points(self) -> SaturationResult:
        if "saturation_points" not in self._cache:
            self._cache["saturation_points"] = compute_saturation_points(
                self._field,
                self._processed_magnetization(),
                threshold=self._result.config.saturation_threshold,
                window=self._result.config.saturation_window,
            )
        return self._cache["saturation_points"]

    @property
    def loop_area(self) -> float:
        if "loop_area" not in self._cache:
            self._cache["loop_area"] = compute_loop_area(
                self._field,
                self._processed_magnetization(),
            )
        return float(self._cache["loop_area"])

    @property
    def squareness(self) -> float:
        if "squareness" not in self._cache:
            self._cache["squareness"] = compute_squareness(
                self.remanence,
                self.saturation_points,
            )
        return float(self._cache["squareness"])

    @property
    def max_susceptibility(self) -> SusceptibilityResult:
        if "max_susceptibility" not in self._cache:
            self._cache["max_susceptibility"] = compute_max_susceptibility(
                self._field,
                self._processed_magnetization(),
            )
        return self._cache["max_susceptibility"]

    @property
    def exchange_bias(self) -> float:
        if "exchange_bias" not in self._cache:
            self._cache["exchange_bias"] = compute_exchange_bias(self.coercive_field)
        return float(self._cache["exchange_bias"])

    @property
    def noise_stats(self) -> NoiseStats:
        if "noise_stats" not in self._cache:
            _ = self._processed_magnetization()
        return self._cache["noise_stats"]

    @property
    def anomalies(self) -> AnomalyReport:
        if "anomalies" not in self._cache:
            self._cache["anomalies"] = detect_anomalies(
                self._field,
                self._processed_magnetization(),
            )
        return self._cache["anomalies"]

    @property
    def cycle_stability(self) -> CycleStabilityAnalysis:
        if "cycle_stability" not in self._cache:
            self._cache["cycle_stability"] = analyze_cycle_stability(self._result)
        return self._cache["cycle_stability"]

    def confidence_interval(
        self,
        metric_name: str,
        *,
        n_samples: int | None = None,
        ci: float | None = None,
        seed: int = 123,
        block_size: int | None = None,
    ) -> ConfidenceIntervalResult:
        """Bootstrap confidence interval for selected metric."""
        key = (
            "ci",
            str(metric_name),
            int(n_samples) if n_samples is not None else None,
            float(ci) if ci is not None else None,
            int(seed),
            int(block_size) if block_size is not None else None,
        )
        if key not in self._cache:
            self._cache[key] = bootstrap_confidence_interval(
                self._result,
                metric_name=str(metric_name),
                n_samples=n_samples,
                ci=ci,
                seed=seed,
                block_size=block_size,
            )
        return self._cache[key]

    def _compute_plugin_metric(self, name: str):
        cache_key = f"plugin:{name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        func = get_registered_metric(name)
        if func is None:
            raise AttributeError(name)

        value = func(
            self._field,
            self._processed_magnetization(),
            self._branches,
        )
        self._cache[cache_key] = value
        return value

    def __getattr__(self, name: str):
        try:
            return self._compute_plugin_metric(name)
        except AttributeError as exc:
            raise AttributeError(f"Unknown metric '{name}'") from exc

    def _append_rows(
        self,
        rows: list[dict[str, Any]],
        metric: str,
        value: Any,
        source: str,
    ) -> None:
        if is_dataclass(value):
            mapping = asdict(value)
            unit_value = mapping.get("unit")
            for key, entry in mapping.items():
                if key == "unit":
                    continue
                rows.append(
                    {
                        "metric": f"{metric}.{key}",
                        "value": entry,
                        "unit": unit_value,
                        "source": source,
                    }
                )
            return

        if isinstance(value, dict):
            unit_value = value.get("unit")
            for key, entry in value.items():
                if key == "unit":
                    continue
                rows.append(
                    {
                        "metric": f"{metric}.{key}",
                        "value": entry,
                        "unit": unit_value,
                        "source": source,
                    }
                )
            return

        rows.append({"metric": metric, "value": value, "unit": None, "source": source})

    def report(self) -> pd.DataFrame:
        """Tabular report with core and plugin metrics."""
        rows: list[dict[str, Any]] = []

        core_values = {
            "coercive_field": self.coercive_field,
            "remanence": self.remanence,
            "saturation_points": self.saturation_points,
            "loop_area": self.loop_area,
            "squareness": self.squareness,
            "max_susceptibility": self.max_susceptibility,
            "exchange_bias": self.exchange_bias,
            "noise_stats": self.noise_stats,
            "anomalies": self.anomalies,
        }

        for name, value in core_values.items():
            self._append_rows(rows, name, value, source="core")

        for name, _func in iter_registered_metrics():
            try:
                plugin_value = self._compute_plugin_metric(name)
                self._append_rows(rows, name, plugin_value, source="plugin")
            except Exception as exc:  # pragma: no cover - plugin failure path
                rows.append(
                    {
                        "metric": name,
                        "value": f"ERROR: {exc}",
                        "unit": None,
                        "source": "plugin",
                    }
                )

        return pd.DataFrame(rows, columns=["metric", "value", "unit", "source"])

    def __repr__(self) -> str:
        return "<MetricsAccessor: core metrics + plugin metrics, use .report()>"

    def _repr_html_(self) -> str:
        plugin_count = len(iter_registered_metrics())
        methods = [
            ("coercive_field", "Hc- / Hc+ / mean / asymmetry"),
            ("remanence", "Mr- / Mr+ / mean"),
            ("saturation_points", "Ms+/Ms- and Hs+/Hs-"),
            ("loop_area", "Absolute loop area"),
            ("squareness", "Mr / Ms"),
            ("max_susceptibility", "max |dM/dB| and field"),
            ("exchange_bias", "(Hc+ + Hc-) / 2"),
            ("noise_stats", "SNR + RMS estimates"),
            ("anomalies", "Outliers / discontinuities / closure"),
            ("cycle_stability", "Drift/correlation between full cycles"),
            ("confidence_interval(...)", "Bootstrap CI for selected metric"),
            ("report()", "DataFrame summary (core + plugin)"),
        ]
        rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(name)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(desc)}</td></tr>"
            for name, desc in methods
        )

        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:14px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;\">"
            "<div style='font-size:1.03em;font-weight:600;color:#f1f5f9;'>Hysteresis Metrics</div>"
            "<div style='margin-top:6px;color:#94a3b8;font-size:0.9em;'>"
            f"Registered plugin metrics: {plugin_count}</div>"
            "<table style='width:100%;margin-top:8px;border-collapse:collapse;font-size:0.9em;'>"
            "<thead><tr style='text-align:left;background:rgba(51,65,85,0.6);'>"
            "<th style='padding:6px 8px;color:#e2e8f0;'>Metric / Method</th>"
            "<th style='padding:6px 8px;color:#e2e8f0;'>Description</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def _repr_mimebundle_(self, include=None, exclude=None):
        html = self._repr_html_()
        text = self.__repr__()
        if html:
            return {"text/html": html, "text/plain": text}
        return {"text/plain": text}


__all__ = ["MetricsAccessor"]
