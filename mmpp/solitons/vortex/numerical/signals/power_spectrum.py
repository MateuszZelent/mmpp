"""Power-spectrum helpers for electrical signal traces."""

from __future__ import annotations

import numpy as np

from .models import SignalSpectrumResult

try:
    from scipy.signal import welch

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    welch = None  # type: ignore[assignment]
    SCIPY_AVAILABLE = False


def _infer_dt(time: np.ndarray) -> float:
    t = np.asarray(time, dtype=float).reshape(-1)
    if t.size < 2:
        return float("nan")
    dt = float(np.median(np.diff(t)))
    return dt if np.isfinite(dt) and dt > 0.0 else float("nan")


def _fft_periodogram(signal: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(signal.size)
    if n < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    x = np.asarray(signal, dtype=float) - float(np.mean(signal))
    window = np.hanning(n)
    xw = x * window

    spectrum = np.fft.rfft(xw)
    freq = np.fft.rfftfreq(n, d=float(dt))
    denom = max(float(np.sum(window**2)), 1e-30)
    power = (np.abs(spectrum) ** 2) / denom
    return np.asarray(freq, dtype=float), np.asarray(power, dtype=float)


def compute_signal_power_spectrum(
    time: np.ndarray,
    signal: np.ndarray,
    *,
    quantity: str,
    method: str = "welch",
    nperseg: int | None = None,
) -> SignalSpectrumResult:
    """Compute one-sided PSD using Welch (preferred) or FFT periodogram."""
    t = np.asarray(time, dtype=float).reshape(-1)
    x = np.asarray(signal, dtype=float).reshape(-1)
    if t.size != x.size:
        raise ValueError("time and signal must have the same length")

    dt = _infer_dt(t)
    if not np.isfinite(dt):
        return SignalSpectrumResult(
            frequencies_hz=np.array([], dtype=float),
            power=np.array([], dtype=float),
            quantity=str(quantity),
            metadata={"method": str(method).lower(), "status": "insufficient_time_resolution"},
        )

    method_norm = str(method).lower()
    if method_norm not in {"welch", "periodogram", "fft"}:
        raise ValueError("method must be 'welch', 'periodogram', or 'fft'")

    if method_norm == "welch" and SCIPY_AVAILABLE and welch is not None:
        nseg = int(nperseg) if nperseg is not None else min(max(64, x.size // 4), x.size)
        f, p = welch(
            x - float(np.mean(x)),
            fs=1.0 / dt,
            nperseg=max(8, nseg),
            detrend="constant",
            scaling="density",
        )
        return SignalSpectrumResult(
            frequencies_hz=np.asarray(f, dtype=float),
            power=np.asarray(p, dtype=float),
            quantity=str(quantity),
            metadata={"method": "welch", "dt": dt},
        )

    f, p = _fft_periodogram(x, dt)
    return SignalSpectrumResult(
        frequencies_hz=f,
        power=p,
        quantity=str(quantity),
        metadata={"method": "periodogram", "dt": dt},
    )


__all__ = ["compute_signal_power_spectrum"]
