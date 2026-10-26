"""Functions for calculating spatially-resolved FFT modes."""

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..fft._compute_loading import (
    _resample_nonuniform_time_data,
    _time_axis_requires_resampling,
    _uniform_dt_from_time_axis,
)

if TYPE_CHECKING:  # pragma: no cover
    from .pyzfn import Pyzfn

NDIMS = 5


def inner_calc_modes(
    self: "Pyzfn",
    dset_in_str: str = "m",
    dset_out_str: str = "m",
    slices: tuple[slice, ...] | slice | None = None,
    *,
    window: bool = True,
    resample_nonuniform: bool = True,
) -> None:
    """Calculate spatially-resolved FFT modes and store the results in-place.

    This function computes the FFT of a 5-D dataset (time, z, y, x, c) and stores
    the results in a structured format under the `fft` and `modes` namespaces.

    Parameters
    ----------
    self : Pyzfn
        Instance of the Pyzfn class on which this method operates.
    dset_in_str : str
        Name of the input dataset to process.
    dset_out_str : str
        Name of the output dataset to create.
    slices : tuple[slice, ...] | slice
        Slices to apply to the input dataset. Defaults to all data.
        Tip: use np.s_ to create complex slices.
    window : bool
        Whether to apply a Hanning window to the time dimension before FFT.
        Defaults to True.
    resample_nonuniform : bool
        Whether to linearly resample a non-uniform time axis to an
        endpoint-preserving uniform grid before FFT. Defaults to True. Pass
        ``False`` to retain strict uniform-axis validation.

    Raises
    ------
    ValueError
        If the input dataset does not have the expected shape or
        lacks the required time attribute.

    Notes
    -----
    This function expects the input dataset to be a 5-D array with dimensions
    (t, z, y, x, c), where:
        - t: time dimension
        - z: spatial dimension (e.g., thickness)
        - y: spatial dimension (e.g., width)
        - x: spatial dimension (e.g., length)
        - c: vector dimension (e.g., magnetization components)
    The output datasets will be structured as follows:
    - `fft/{dset_out_str}/freqs`: Frequencies corresponding to the FFT.
    - `fft/{dset_out_str}/spec`: Maximum spectral amplitude across spatial dimensions.
    - `fft/{dset_out_str}/sum`: Sum of spectral amplitudes across spatial dimensions.
    - `modes/{dset_out_str}/freqs`: Frequencies corresponding to the FFT modes.
    - `modes/{dset_out_str}/arr`: Complex FFT modes array.

    """
    dset_in = self.get_array(dset_in_str)

    if not isinstance(resample_nonuniform, (bool, np.bool_)):
        raise TypeError("resample_nonuniform must be boolean")

    if slices is None:
        slices = (slice(None),) * NDIMS
    elif isinstance(slices, slice):
        slices = (slices,)

    if dset_in.ndim != NDIMS:
        msg = f"Expected a 5-D array (t,z,y,x,c); got {dset_in.ndim}-D."
        raise ValueError(msg)

    if "t" not in dset_in.attrs:
        msg = f"Dataset '{dset_in_str}' lacks required time attribute 't'."
        raise ValueError(msg)
    ts = np.asarray(dset_in.attrs["t"], dtype=np.float64)
    if ts.size != dset_in.shape[0]:
        msg = (
            f"len(attrs['t'])={ts.size} does not match time dimension "
            f"{dset_in.shape[0]}"
        )
        raise ValueError(msg)

    time_slice = (
        slices[0] if isinstance(slices, tuple) and len(slices) > 0 else slice(None)
    )
    ts = np.asarray(dset_in.attrs["t"], dtype=np.float64)[time_slice]
    arr = np.asarray(dset_in[slices], dtype=np.float32)
    if arr.shape[0] != ts.size:
        raise ValueError(
            "The selected time axis length does not match the selected data "
            f"shape: len(t)={ts.size}, data.shape[0]={arr.shape[0]}"
        )

    if _time_axis_requires_resampling(ts):
        if not resample_nonuniform:
            _uniform_dt_from_time_axis(ts, allow_nonuniform=False)
        arr, did_resample = _resample_nonuniform_time_data(arr, ts)
        if did_resample:
            mean_dt = float(np.mean(np.diff(ts)))
            max_deviation = float(np.max(np.abs(np.diff(ts) - mean_dt)))
            relative_deviation = max_deviation / abs(mean_dt)
            warnings.warn(
                "Pyzfn mode FFT detected a non-uniform time axis and linearly "
                "resampled it onto an endpoint-preserving uniform grid "
                f"(largest step deviation={relative_deviation:.3%} of mean dt). "
                "Interpolation may slightly attenuate or broaden high-frequency "
                "peaks and alter quantitative amplitudes or phases; pass "
                "resample_nonuniform=False to reject non-uniform input when strict "
                "sampling is required.",
                UserWarning,
                stacklevel=2,
            )
            ts = np.linspace(ts[0], ts[-1], ts.size)

    arr -= arr.mean(axis=0, keepdims=True)
    if window:
        arr *= np.hanning(arr.shape[0])[:, None, None, None, None]

    out = np.fft.rfft(arr, axis=0).astype(np.complex64)

    dt = _uniform_dt_from_time_axis(ts, allow_nonuniform=False)
    freqs = np.fft.rfftfreq(len(ts), dt) * 1e-9
    spec = np.abs(out)
    from ..cache.transactions import open_published_cache, publish_cache

    location = Path(dset_out_str)
    if not location.parts or location.is_absolute() or ".." in location.parts:
        raise ValueError(f"Invalid output dataset path {dset_out_str!r}")
    store = dset_in.store
    snapshot = store._field(dset_in.path) if hasattr(store, "_field") else None
    revision = None if snapshot is None else snapshot["revision"]
    if revision is None:
        warnings.warn(
            "The source has no committed revision; mode results are available "
            "in this Pyzfn instance but will not be reused by a new instance.",
            UserWarning,
            stacklevel=2,
        )
    parent = Path(self.clean_path) / "modes" / location.parent

    def populate(group):
        group.attrs["source_dataset"] = dset_in_str
        group.attrs["mode_algorithm"] = "pyzfn"
        group.attrs["window"] = bool(window)
        group.attrs["resample_nonuniform"] = bool(resample_nonuniform)
        group.create_dataset("freqs", data=freqs)
        group.create_dataset("arr", data=out, chunks=(1, *out.shape[1:]))
        fft = group.create_group("legacy_fft")
        fft.create_dataset("freqs", data=freqs)
        fft.create_dataset("spec", data=np.max(spec, axis=(1, 2, 3)))
        fft.create_dataset("sum", data=np.sum(spec, axis=(1, 2, 3)))

    publish_cache(
        parent, location.name, populate, source_revision=revision, overwrite=True
    )
    generation = open_published_cache(
        parent, location.name, source_revision=revision, allow_unversioned=True
    )
    if generation is None:
        raise RuntimeError("Unable to verify the published Pyzfn mode result")
    self._derived_modes[dset_out_str] = generation
