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
    """Compute PSD maps from STNO voltage-proxy trajectories.

    Parameters
    ----------
    dt_out : float
        Output sampling interval used in the simulation [s].
    cut_time : float, optional
        Transient duration to discard before spectral analysis [s].
        Defaults to ``100e-9`` (100 ns).
    """

    def __init__(self, dt_out: float, cut_time: float = 100e-9) -> None:
        self.dt_out = dt_out
        self.cut_idx = int(cut_time / dt_out)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def compute_psd(
        self,
        all_V: np.ndarray,
        f_max_ghz: float = 10.0,
        noise_floor_db: float = -75.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute normalised Power Spectral Density for each trajectory.

        The method:

        1. Strips the initial *cut_time* transient.
        2. Removes the DC component (mean subtraction).
        3. Applies a Hann window to suppress spectral leakage.
        4. Computes the one-sided FFT power and converts to dBrelative.

        Parameters
        ----------
        all_V : np.ndarray, shape (n_sims, n_steps)
            Voltage-proxy time series returned by :func:`run_all_sweeps_parallel`.
        f_max_ghz : float, optional
            Upper frequency bound for the returned spectrum [GHz].
        noise_floor_db : float, optional
            Noise floor added in log-space to avoid ``log(0)`` [dB].

        Returns
        -------
        f_axis : np.ndarray, shape (n_freqs,)
            Frequency axis [GHz].
        psd_db : np.ndarray, shape (n_sims, n_freqs)
            Normalised PSD in dB.  Normalised by the global FFT maximum so
            that the peak is always 0 dB.
        """
        V_steady = all_V[:, self.cut_idx :]
        V_steady = V_steady - np.mean(V_steady, axis=1, keepdims=True)

        n_steps = V_steady.shape[1]
        window = np.hanning(n_steps)
        freqs = np.fft.rfftfreq(n_steps, d=self.dt_out)

        f_mask = (freqs > 0.05e9) & (freqs <= f_max_ghz * 1e9)
        f_axis = freqs[f_mask] * 1e-9

        fft_power = np.abs(np.fft.rfft(V_steady * window, axis=1)) ** 2
        peak = np.max(fft_power)
        if peak == 0.0:
            peak = 1.0
        psd_db = 10 * np.log10(
            fft_power[:, f_mask] / peak + 10 ** (noise_floor_db / 10)
        )

        return f_axis, psd_db

    def peak_frequency(self, all_V: np.ndarray, f_max_ghz: float = 10.0) -> np.ndarray:
        """Return the dominant frequency [GHz] for each trajectory.

        Convenience wrapper around :meth:`compute_psd` that extracts the
        frequency bin with maximum PSD for each simulation point.

        Parameters
        ----------
        all_V : np.ndarray, shape (n_sims, n_steps)
        f_max_ghz : float, optional

        Returns
        -------
        np.ndarray, shape (n_sims,)
            Peak frequency in GHz for each trajectory.
        """
        f_axis, psd_db = self.compute_psd(all_V, f_max_ghz=f_max_ghz)
        idx = np.argmax(psd_db, axis=1)
        return f_axis[idx]
