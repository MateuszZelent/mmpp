"""Refactored ``SpectrumResult`` with fluent API additions."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ..filters import FilterPipeline


class SpectrumResult:
    """FFT spectrum result with fluent plotting and mode bridge helpers."""

    def __init__(
        self,
        frequencies: np.ndarray,
        spectrum: np.ndarray,
        peaks_info: dict | None = None,
        component_label: str | None = None,
        source_job: Any | None = None,
        source_fft: Any | None = None,
        mode_context: dict[str, Any] | None = None,
        filter_config: dict[str, Any] | None = None,
        raw_spectrum: np.ndarray | None = None,
        power_override: np.ndarray | None = None,
        scaling: str = "raw",
        spectrum_kind: str = "complex",
        power_quantity: str = "raw_power",
    ):
        self.frequencies = np.asarray(frequencies)
        self.spectrum = np.asarray(spectrum)
        self.peaks_info = peaks_info
        self.component_label = component_label
        self._source_job = source_job
        self._source_fft = source_fft
        self._mode_context = dict(mode_context or {})
        self._filter_config = filter_config
        self._raw_spectrum = np.asarray(raw_spectrum) if raw_spectrum is not None else np.asarray(spectrum)
        self._power_override = np.asarray(power_override) if power_override is not None else None
        self.scaling = str(scaling)
        self.spectrum_kind = str(spectrum_kind)
        self.power_quantity = str(power_quantity)
        self._single_component = False
        self._peaks_cache: list[dict[str, float]] | None = None

    @property
    def spectral_quantity(self) -> np.ndarray:
        """Return the non-negative spectral quantity consistent with ``scaling``.

        This accessor is the semantics-aware counterpart of the legacy ``power``
        property and respects ``power_quantity`` plus any filtered override.
        """
        if self._power_override is not None:
            return np.asarray(self._power_override, dtype=float)

        quantity = np.abs(self.spectrum) ** 2
        power_quantity = self.power_quantity.lower()
        if power_quantity in {
            "raw_power",
            "continuous_ft_power",
            "amplitude_squared",
            "power",
            "psd",
        }:
            return quantity

        return quantity

    @property
    def spectral_quantity_label(self) -> str:
        """Human-readable label for :attr:`spectral_quantity`."""
        return {
            "raw_power": "Raw power",
            "continuous_ft_power": "Continuous FT power",
            "amplitude_squared": "Amplitude²",
            "power": "Power",
            "psd": "PSD",
        }.get(self.power_quantity.lower(), "Spectral quantity")

    @property
    def peak_frequency_hz(self) -> float:
        """Frequency of the strongest spectral quantity sample."""
        if self.frequencies.size == 0 or self.spectral_quantity.size == 0:
            raise ValueError("Spectrum is empty; cannot determine peak frequency")

        quantity = np.asarray(self.spectral_quantity, dtype=float)
        if quantity.ndim > 1:
            reduction_axes = tuple(range(1, quantity.ndim))
            quantity = quantity.sum(axis=reduction_axes)

        peak_idx = int(np.argmax(quantity))
        if peak_idx >= self.frequencies.shape[0]:
            peak_idx = self.frequencies.shape[0] - 1
        return float(self.frequencies[peak_idx])

    @property
    def peak_frequency_ghz(self) -> float:
        """Peak frequency converted to GHz."""
        return self.peak_frequency_hz * 1e-9

    @property
    def is_complex_spectrum(self) -> bool:
        """Whether the spectrum preserves complex phase information."""
        return self.spectrum_kind == "complex"

    @property
    def frequencies_ghz(self) -> np.ndarray:
        """Frequency axis converted from Hz to GHz."""
        return np.asarray(self.frequencies, dtype=float) * 1e-9

    @property
    def freqs(self) -> np.ndarray:
        """Alias for :attr:`frequencies`."""
        return self.frequencies

    @property
    def power(self) -> np.ndarray:
        """Backward-compatible alias for :attr:`spectral_quantity`."""
        return self.spectral_quantity

    @property
    def amplitude(self) -> np.ndarray:
        """Amplitude spectrum ``|FFT|``."""
        return np.abs(self.spectrum)

    @property
    def magnitude(self) -> np.ndarray:
        """Backward-compatible alias for :attr:`amplitude`."""
        return self.amplitude

    @property
    def phase(self) -> np.ndarray:
        """Phase spectrum ``arg(FFT)``."""
        if not self.is_complex_spectrum:
            raise ValueError(
                "Phase is unavailable for magnitude-only spectra (e.g. method=2, power scaling, psd scaling, or filtered views)."
            )
        return np.angle(self.spectrum)

    @property
    def complex(self) -> np.ndarray:
        """Raw complex spectrum data."""
        if not self.is_complex_spectrum:
            raise ValueError(
                "Complex spectrum is unavailable for magnitude-only spectra (e.g. method=2, power scaling, psd scaling, or filtered views)."
            )
        return self.spectrum

    @property
    def data(self) -> np.ndarray:
        """Alias for :attr:`spectrum`."""
        return self.spectrum

    @property
    def filter_config(self) -> dict[str, Any] | None:
        """Filter configuration used for transformed representation."""
        return self._filter_config

    @property
    def peaks(self) -> list[dict[str, float]]:
        """Peak list derived from ``peaks_info``."""
        if self._peaks_cache is not None:
            return self._peaks_cache
        if not isinstance(self.peaks_info, dict):
            self._peaks_cache = []
            return self._peaks_cache

        indices = list(self.peaks_info.get("indices", []))
        freqs = np.asarray(self.peaks_info.get("frequencies", []), dtype=float)
        amps = np.asarray(self.peaks_info.get("amplitudes", []), dtype=float)
        peaks: list[dict[str, float]] = []
        for i, idx in enumerate(indices):
            freq = float(freqs[i]) if i < freqs.size else float("nan")
            amp = float(amps[i]) if i < amps.size else float("nan")
            peaks.append({"index": int(idx), "frequency_hz": freq, "frequency_ghz": freq * 1e-9, "amplitude": amp})
        self._peaks_cache = peaks
        return peaks

    @property
    def plot(self):
        """Fluent plotting namespace."""
        from ._plotting.accessor import SpectrumPlotAccessor

        return SpectrumPlotAccessor(self)

    @property
    def plt(self):
        """Deprecated alias for :attr:`plot`."""
        return self.plot

    @property
    def modes(self):
        """Bridge to mode analysis interface."""
        from .modes import SpectrumModes

        return SpectrumModes(self)

    def filtered(self, **kwargs) -> SpectrumResult:
        """Return filtered view using shared postprocessing pipeline.

        Supported keyword arguments include:
        ``normalize``, ``log_scale``, ``gamma``, ``smooth``, ``smooth_window``,
        ``smooth_sigma``, ``baseline``, ``percentile_clip``, ``soft_threshold``.
        """
        post: dict[str, Any] = {}

        if kwargs.get("normalize"):
            post["normalize"] = True
        if kwargs.get("log_scale") or kwargs.get("log_transform"):
            post["log_transform"] = True
        if "gamma" in kwargs and float(kwargs.get("gamma", 1.0)) != 1.0:
            post["gamma"] = {"gamma": float(kwargs["gamma"])}

        if "percentile_clip" in kwargs and kwargs["percentile_clip"] is not None:
            low, high = kwargs["percentile_clip"]
            post["percentile_clip"] = {"low": float(low), "high": float(high)}
        else:
            low = kwargs.get("clip_percentile_low")
            high = kwargs.get("clip_percentile_high")
            if low is not None or high is not None:
                post["percentile_clip"] = {
                    "low": float(0.0 if low is None else low),
                    "high": float(100.0 if high is None else high),
                }

        soft_threshold = kwargs.get("soft_threshold")
        if soft_threshold is None:
            soft_threshold = kwargs.get("soft_threshold_percentile")
        if soft_threshold is not None and float(soft_threshold) > 0:
            post["soft_threshold"] = {"percentile": float(soft_threshold)}

        baseline = kwargs.get("baseline")
        if baseline is None:
            baseline = kwargs.get("baseline_mode")
        if baseline is not None and str(baseline).lower() not in {"none", ""}:
            post["baseline_correction"] = {"mode": str(baseline).lower()}

        smooth = kwargs.get("smooth")
        if smooth is None:
            smooth = kwargs.get("smooth_filter")
        if smooth is not None and str(smooth).lower() not in {"none", ""}:
            smooth_mode = str(smooth).lower()
            if smooth_mode == "gaussian":
                filter_name = "gaussian_smooth"
            elif smooth_mode == "savgol":
                filter_name = "savgol_smooth"
            elif smooth_mode in {"moving_average", "moving"}:
                filter_name = "moving_average"
            else:
                filter_name = "gaussian_smooth"
            post[filter_name] = {
                "smooth_window": int(kwargs.get("smooth_window", 7)),
                "smooth_sigma": float(kwargs.get("smooth_sigma", 1.0)),
            }

        pipeline = FilterPipeline()
        filtered_power = pipeline.postprocess(
            np.asarray(self.spectral_quantity, dtype=float),
            np.asarray(self.frequencies, dtype=float),
            filters={"post": post} if post else None,
            stage="post",
        )
        filtered_spectrum = np.sqrt(np.clip(np.asarray(filtered_power, dtype=float), 0.0, None))
        return SpectrumResult(
            frequencies=self.frequencies,
            spectrum=filtered_spectrum,
            peaks_info=None,
            component_label=self.component_label,
            source_job=self._source_job,
            source_fft=self._source_fft,
            mode_context=self._mode_context,
            filter_config={"post": post},
            raw_spectrum=self._raw_spectrum,
            power_override=filtered_power,
            scaling=self.scaling,
            spectrum_kind="magnitude",
            power_quantity=self.power_quantity,
        )

    # Backward-compatible tuple behavior.
    def __iter__(self):
        yield self.frequencies
        yield self.spectrum
        if self.peaks_info is not None:
            yield self.peaks_info

    def __getitem__(self, index: int):
        items = [self.frequencies, self.spectrum]
        if self.peaks_info is not None:
            items.append(self.peaks_info)
        return items[index]

    def __len__(self):
        return 3 if self.peaks_info is not None else 2

    def plot_spectrum(self, **kwargs):
        """Deprecated alias for ``spec.plot.spectrum(...)``."""
        warnings.warn(
            "SpectrumResult.plot_spectrum() is deprecated. Use spec.plot.spectrum().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.plot.spectrum(**kwargs)

    def __repr__(self):
        label = f", label='{self.component_label}'" if self.component_label else ""
        peaks = len(self.peaks_info.get("indices", [])) if isinstance(self.peaks_info, dict) else "None"
        filtered = ", filtered=True" if self._filter_config else ""
        return (
            f"SpectrumResult(frequencies={len(self.frequencies)}, "
            f"spectrum_shape={self.spectrum.shape}, peaks={peaks}, scaling='{self.scaling}', "
            f"kind='{self.spectrum_kind}'{label}{filtered})"
        )

    def _repr_html_(self) -> str:
        from html import escape as _esc

        fmin = float(self.frequencies_ghz[0]) if self.frequencies.size else float("nan")
        fmax = float(self.frequencies_ghz[-1]) if self.frequencies.size else float("nan")
        n_points = int(self.frequencies.size)
        n_peaks = len(self.peaks)
        filtered_badge = (
            "<span style='background:#166534;color:#86efac;padding:2px 8px;"
            "border-radius:4px;font-size:0.8em;margin-left:8px;'>filtered</span>"
            if self._filter_config
            else ""
        )
        comp = _esc(self.component_label or "all")

        methods = [
            (".plot.spectrum(...)", "Static matplotlib spectrum plot"),
            (".plot.interactive(...)", "Interactive spectrum explorer"),
            (".modes.at(f=...)", "Mode profile at frequency [GHz]"),
            (".modes.at_peak(i)", "Mode profile at detected peak index"),
            (".modes.interactive()", "Full interactive modes explorer"),
            (".filtered(...)", "Non-destructive postprocessing (returns new result)"),
        ]
        method_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(m)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(d)}</td></tr>"
            for m, d in methods
        )
        filter_params = [
            ("normalize", "False", "Normalize spectrum amplitudes to [0, 1]"),
            ("log_scale", "False", "Apply logarithmic transform"),
            ("gamma", "None", "Gamma correction factor (float)"),
            ("percentile_clip", "None", "Clip range as (low, high) percentiles"),
            ("soft_threshold", "None", "Soft threshold by percentile"),
            ("baseline", "None", "Baseline removal ('none', 'median', ...)"),
            ("smooth", "None", "Smoothing: 'gaussian', 'savgol', 'moving_average'"),
            ("smooth_window", "7", "Smoothing window size"),
            ("smooth_sigma", "1.0", "Gaussian smooth sigma"),
        ]
        fp_rows = "".join(
            f"<tr><td style='padding:4px 8px;font-family:monospace;color:#93c5fd;'>{_esc(n)}</td>"
            f"<td style='padding:4px 8px;color:#a5b4fc;'>{_esc(d)}</td>"
            f"<td style='padding:4px 8px;color:#cbd5e1;'>{_esc(desc)}</td></tr>"
            for n, d, desc in filter_params
        )
        example = (
            "# Plot spectrum with peak markers\n"
            "spec.plot.spectrum(show_peaks=True)\n"
            "\n"
            "# Apply post-processing filters\n"
            "filtered = spec.filtered(normalize=True, gamma=0.5)\n"
            "filtered.plot.spectrum()\n"
            "\n"
            "# Mode analysis at peak\n"
            "mode = spec.modes.at_peak(0)\n"
            "mode.plot.imshow(component='z')"
        )
        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:2px solid #334155;border-radius:12px;padding:16px;margin:8px 0;"
            "background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#334155 100%);"
            "color:#e2e8f0;box-shadow:0 8px 20px rgba(0,0,0,0.25);\">"
            f"<div style='font-size:1.1em;font-weight:600;color:#f1f5f9;margin-bottom:4px;'>"
            f"SpectrumResult {filtered_badge}</div>"
            "<div style='font-size:0.85em;color:#94a3b8;margin-bottom:10px;'>"
            f"{fmin:.2f} – {fmax:.2f} GHz · {n_points} points · "
            f"component: {comp} · peaks: {n_peaks}</div>"
            # Methods
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Methods</div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            f"{method_rows}</table></div>"
            # Filter params
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "margin-bottom:10px;border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>"
            "Parameters <span style='color:#94a3b8;font-weight:400;'>(.filtered)</span></div>"
            "<table style='width:100%;border-collapse:collapse;font-size:0.9em;'>"
            "<thead><tr style='text-align:left;background:rgba(51,65,85,0.6);'>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Arg</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Default</th>"
            "<th style='padding:4px 8px;color:#e2e8f0;'>Description</th></tr></thead>"
            f"<tbody>{fp_rows}</tbody></table></div>"
            # Examples
            "<div style='background:rgba(15,23,42,0.6);padding:10px;border-radius:8px;"
            "border:1px solid rgba(148,163,184,0.2);'>"
            "<div style='font-weight:600;color:#e2e8f0;margin-bottom:6px;'>Examples</div>"
            "<pre style='margin:0;background:rgba(15,23,42,0.85);padding:10px;"
            "border-radius:6px;color:#e2e8f0;overflow-x:auto;font-size:0.85em;'>"
            f"<code>{example}</code></pre></div>"
            "</div>"
        )
