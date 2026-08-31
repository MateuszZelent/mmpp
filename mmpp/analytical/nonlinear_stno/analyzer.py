"""
mmpp.analytical.nonlinear_stno.analyzer
=========================================
Digital Signal Processing utilities for STNO time-series data.

The :class:`SpectrumAnalyzer` converts raw voltage-proxy trajectories
produced by the JIT engine into calibrated Power Spectral Density (PSD)
maps suitable for publication-quality spectrograms.

Example
-------
>>> from mmpp.analytical.nonlinear_stno import STNOParameters, run_all_sweeps_parallel
>>> from mmpp.analytical.nonlinear_stno import SpectrumAnalyzer
>>>
>>> device = STNOParameters()
>>> # … run simulation → all_V …
>>> analyzer = SpectrumAnalyzer(dt_out=0.5e-12, cut_time=100e-9)
>>> f_axis, psd_db = analyzer.compute_psd(all_V, f_max_ghz=10.0)
"""

import numpy as np


class SpectrumAnalyzer:
    """Relative spectrum analysis for experimental STNO signal proxies.

    Despite the historical ``compute_psd`` name, the output is a globally
    normalized relative FFT power in dB, not a unitful PSD in V²/Hz.
    """

    def __init__(self, dt_out: float, cut_time: float = 100e-9):
        self.dt_out = float(dt_out)
        cut = float(cut_time)
        if not np.isfinite(self.dt_out) or self.dt_out <= 0.0:
            raise ValueError("dt_out must be finite and positive [s]")
        if not np.isfinite(cut) or cut < 0.0:
            raise ValueError("cut_time must be finite and non-negative [s]")
        self.cut_idx = int(cut / self.dt_out)

    def _prepare_fft(self, all_V, f_min_ghz, f_max_ghz):
        values = np.asarray(all_V, dtype=float)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError(
                "all_V must have shape (n_sweeps, n_time) with n_sweeps > 0"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("all_V must contain only finite values")
        f_min = float(f_min_ghz)
        f_max = float(f_max_ghz)
        if not np.isfinite(f_min) or not np.isfinite(f_max) or not 0.0 <= f_min < f_max:
            raise ValueError("frequency bounds must satisfy 0 <= f_min_ghz < f_max_ghz")
        if self.cut_idx >= values.shape[1] - 3:
            raise ValueError("cut_time leaves fewer than four samples for the FFT")

        # Copy: centering must not mutate the caller's time-series array.
        V_steady = values[:, self.cut_idx :].copy()
        V_steady -= np.mean(V_steady, axis=1, keepdims=True)
        steps_steady = V_steady.shape[1]

        window = np.hanning(steps_steady)
        freqs = np.fft.rfftfreq(steps_steady, d=self.dt_out)

        f_mask = (freqs > f_min * 1e9) & (freqs <= f_max * 1e9)
        if not np.any(f_mask):
            raise ValueError("frequency interval contains no FFT bins")
        f_axis = freqs[f_mask] * 1e-9
        fft_power = np.abs(np.fft.rfft(V_steady * window, axis=1)) ** 2

        return f_axis, f_mask, fft_power

    def compute_psd(
        self,
        all_V: np.ndarray,
        f_min_ghz: float = 0.05,
        f_max_ghz: float = 10.0,
        noise_floor_db: float = -75.0,
    ):
        """Return frequency [GHz] and globally normalized relative power [dB]."""
        floor = float(noise_floor_db)
        if not np.isfinite(floor) or floor >= 0.0:
            raise ValueError("noise_floor_db must be a finite negative value")
        f_axis, f_mask, fft_power = self._prepare_fft(all_V, f_min_ghz, f_max_ghz)
        norm_factor = np.max(fft_power) + 1e-30
        psd_db = 10 * np.log10(fft_power[:, f_mask] / norm_factor + 10 ** (floor / 10))
        return f_axis, psd_db

    def peak_frequency(
        self, all_V: np.ndarray, f_min_ghz: float = 0.05, f_max_ghz: float = 10.0
    ):
        """
        Zwraca tablicę dominujących częstotliwości [GHz] dla każdej trajektorii.
        Wykorzystuje interpolację paraboliczną dla precyzji sub-bin.
        """
        f_axis, psd_db = self.compute_psd(
            all_V, f_min_ghz, f_max_ghz, noise_floor_db=-100.0
        )
        if f_axis.size == 0:
            raise ValueError("frequency interval contains no FFT bins")
        peak_indices = np.argmax(psd_db, axis=1)

        peaks = np.zeros(psd_db.shape[0])
        for i in range(psd_db.shape[0]):
            idx = peak_indices[i]
            if 0 < idx < len(f_axis) - 1 and len(f_axis) >= 2:
                alpha = psd_db[i, idx - 1]
                beta = psd_db[i, idx]
                gamma = psd_db[i, idx + 1]

                # Przesunięcie wierzchołka paraboli
                denom = alpha - 2 * beta + gamma
                if denom != 0:
                    p = 0.5 * (alpha - gamma) / denom
                    peaks[i] = f_axis[idx] + p * (f_axis[1] - f_axis[0])
                else:
                    peaks[i] = f_axis[idx]
            else:
                peaks[i] = f_axis[idx]

        return peaks
