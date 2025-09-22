import numpy as np

from mmpp.fft.mode_characterization import ModeCharacterAnalyzer, ModeCharacteristicConfig
from mmpp.fft.modes import FMRModeData


def _make_grid(n=128):
    axis = np.linspace(-1.0, 1.0, n)
    return np.meshgrid(axis, axis)


def _make_mode(mx, my, mz, frequency=1.0):
    mode_array = np.stack([mx, my, mz], axis=-1)
    metadata = {
        "spatial_resolution": (1.0, 1.0),
        "core_position_px": ((mx.shape[1] - 1) / 2, (mx.shape[0] - 1) / 2),
    }
    return FMRModeData(frequency=frequency, mode_array=mode_array, metadata=metadata)


def test_gyration_mode_classification():
    x, y = _make_grid(96)
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    envelope = np.exp(-(r / 0.7) ** 2)

    mx = envelope * np.exp(1j * phi)
    my = envelope * np.exp(1j * (phi + np.pi / 2))
    mz = 0.1 * envelope * np.exp(1j * phi)

    mode = _make_mode(mx, my, mz, frequency=1.5)
    analyzer = ModeCharacterAnalyzer()
    result = analyzer.analyze(mode)

    assert result.primary_class == "gyration"
    assert result.m_index in {1, -1}
    assert result.rotation_sense in {"CCW", "CW"}
    assert result.radial_nodes == 0
    assert result.confidence > 0.4


def test_breathing_mode_classification():
    x, y = _make_grid(96)
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    mz_envelope = np.exp(-((r - 0.4) ** 2) / 0.02)
    mz = mz_envelope.astype(complex)
    mx = 0.02 * mz_envelope.astype(complex)
    my = 0.02 * mz_envelope.astype(complex)

    mode = _make_mode(mx, my, mz, frequency=2.2)
    config = ModeCharacteristicConfig(relative_amplitude_threshold=0.05)
    analyzer = ModeCharacterAnalyzer(config=config)
    result = analyzer.analyze(mode)

    assert result.primary_class == "breathing"
    assert result.m_index in (None, 0)
    assert result.phase_z_uniformity > 0.6
    assert result.confidence > 0.4


def test_azimuthal_mode_classification():
    x, y = _make_grid(96)
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    envelope = (r / 0.8) * np.exp(-(r / 0.9) ** 2)

    mx = envelope * np.exp(1j * 2 * phi)
    my = envelope * np.exp(1j * (2 * phi + np.pi / 3))
    mz = 0.05 * envelope

    mode = _make_mode(mx, my, mz, frequency=4.0)
    analyzer = ModeCharacterAnalyzer()
    result = analyzer.analyze(mode)

    assert result.primary_class == "azimuthal"
    assert result.m_index in {2, -2}
    assert result.radial_nodes >= 0
    assert result.m_quality > 0.3
