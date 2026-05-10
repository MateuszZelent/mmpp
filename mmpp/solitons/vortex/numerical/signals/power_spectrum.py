"""Power-spectrum helpers for electrical signal traces."""

from __future__ import annotations

import numpy as np

from mmpp._shared.spectral import compute_psd, infer_dt

from .models import SignalSpectrumResult


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

    dt = infer_dt(t)
    if not np.isfinite(dt):
        return SignalSpectrumResult(
            frequencies_hz=np.array([], dtype=float),
            power=np.array([], dtype=float),
            quantity=str(quantity),
            metadata={
                "method": str(method).lower(),
                "status": "insufficient_time_resolution",
            },
        )

    method_norm = str(method).lower()
    if method_norm not in {"welch", "periodogram", "fft"}:
        raise ValueError("method must be 'welch', 'periodogram', or 'fft'")

    nseg = int(nperseg) if nperseg is not None else min(max(64, x.size // 4), x.size)
    f, p, used_method, metadata = compute_psd(
        x,
        dt=dt,
        method=method_norm,
        nperseg=max(8, nseg),
        scaling="density",
    )
    metadata["method"] = used_method
    return SignalSpectrumResult(
        frequencies_hz=np.asarray(f, dtype=float),
        power=np.asarray(p, dtype=float),
        quantity=str(quantity),
        metadata=metadata,
    )


__all__ = ["compute_signal_power_spectrum"]
