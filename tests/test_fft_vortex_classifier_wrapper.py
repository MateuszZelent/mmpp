from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np


def _make_mode_array(nx: int = 64, ny: int = 64) -> np.ndarray:
    cx = (nx - 1) * 0.5
    cy = (ny - 1) * 0.5
    x = np.arange(nx, dtype=float) - cx
    y = np.arange(ny, dtype=float) - cy
    xg, yg = np.meshgrid(x, y)
    radius = np.hypot(xg, yg)
    phi = np.arctan2(yg, xg)

    mz = np.exp(-((radius / 6.0) ** 2))
    m_perp = np.sqrt(np.clip(1.0 - mz**2, 0.0, 1.0))
    mx = -m_perp * np.sin(phi)
    my = m_perp * np.cos(phi)
    return np.stack([mx, my, mz], axis=-1).astype(np.complex128)


def test_legacy_vortex_classifier_wrapper_keeps_api():
    from mmpp.fft.vortex_classifier import (
        AdvancedVortexClassifier,
        VortexClassificationConfig,
        VortexModeResult,
    )

    cfg = VortexClassificationConfig()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        classifier = AdvancedVortexClassifier(cfg)
    assert any("deprecated" in str(w.message).lower() for w in caught)
    mode_data = SimpleNamespace(
        frequency=8.2,
        mode_array=_make_mode_array(),
        metadata={"spatial_resolution": (2.0e-9, 2.0e-9)},
    )
    result = classifier.classify_mode(mode_data)

    assert isinstance(result, VortexModeResult)
    assert result.mode_type in {"gyration", "breathing", "azimuthal"}
    assert np.isfinite(result.confidence)
    assert result.frequency == 8.2
