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
    """Cyfrowa Analiza Sygnałów (DSP) i ekstrakcja widma z trajektorii czasowych."""
    
    def __init__(self, dt_out: float, cut_time: float = 100e-9):
        self.dt_out = dt_out
        self.cut_idx = int(cut_time / dt_out)

    def _prepare_fft(self, all_V, f_min_ghz, f_max_ghz):
        V_steady = all_V[:, self.cut_idx:]
        V_steady -= np.mean(V_steady, axis=1, keepdims=True)
        steps_steady = V_steady.shape[1]
        
        window = np.hanning(steps_steady)
        freqs = np.fft.rfftfreq(steps_steady, d=self.dt_out)
        
        f_mask = (freqs > f_min_ghz * 1e9) & (freqs <= f_max_ghz * 1e9)
        f_axis = freqs[f_mask] * 1e-9
        fft_power = np.abs(np.fft.rfft(V_steady * window, axis=1))**2
        
        return f_axis, f_mask, fft_power

    def compute_psd(self, all_V: np.ndarray, f_min_ghz: float = 0.05, f_max_ghz: float = 10.0, noise_floor_db: float = -75.0):
        """Zwraca oś F [GHz] oraz widmo w dB."""
        f_axis, f_mask, fft_power = self._prepare_fft(all_V, f_min_ghz, f_max_ghz)
        norm_factor = np.max(fft_power) + 1e-30
        psd_db = 10 * np.log10(fft_power[:, f_mask] / norm_factor + 10**(noise_floor_db/10))
        return f_axis, psd_db

    def peak_frequency(self, all_V: np.ndarray, f_min_ghz: float = 0.05, f_max_ghz: float = 10.0):
        """
        Zwraca tablicę dominujących częstotliwości [GHz] dla każdej trajektorii.
        Wykorzystuje interpolację paraboliczną dla precyzji sub-bin.
        """
        f_axis, psd_db = self.compute_psd(all_V, f_min_ghz, f_max_ghz, noise_floor_db=-100.0)
        peak_indices = np.argmax(psd_db, axis=1)
        
        peaks = np.zeros(psd_db.shape[0])
        for i in range(psd_db.shape[0]):
            idx = peak_indices[i]
            if 0 < idx < len(f_axis) - 1:
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
