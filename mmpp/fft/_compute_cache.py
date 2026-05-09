"""Internal cache helpers for persisted FFT results."""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np


def load_existing_fft_result(
    *,
    zarr_path: str,
    dataset_name: str,
    result_cls: Any,
    config_cls: Any,
    psutil_module: Any | None,
    logger: Any,
) -> Any | None:
    """Load cached FFT result from zarr, or return ``None`` when unavailable."""
    try:
        start_time = time.time()
        process = None
        initial_memory = None
        if psutil_module is not None:
            process = psutil_module.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024

        logger.debug("Loading existing FFT data from: %s", zarr_path)
        logger.debug("FFT dataset: fft/%s", dataset_name)

        import zarr

        z = zarr.open(zarr_path, mode="r")
        fft_path = f"fft/{dataset_name}"
        if fft_path not in z:
            logger.debug("FFT dataset %s not found", fft_path)
            return None

        fft_group = z[fft_path]

        data_load_start = time.time()
        spectrum = np.array(fft_group["spectrum"])
        frequencies = np.array(fft_group["frequencies"])
        data_load_time = time.time() - data_load_start
        logger.debug("FFT data loading time: %.3fs", data_load_time)

        spectrum_size_mb = spectrum.nbytes / 1024 / 1024
        freq_size_mb = frequencies.nbytes / 1024 / 1024
        total_size_mb = spectrum_size_mb + freq_size_mb
        logger.debug("Spectrum size: %.1f MB", spectrum_size_mb)
        logger.debug("Frequencies size: %.1f MB", freq_size_mb)
        logger.debug("Total FFT data size: %.1f MB", total_size_mb)

        metadata = dict(fft_group.attrs)
        config = config_cls(
            window_function=metadata.pop("window_function", "hann"),
            filter_type=metadata.pop("filter_type", "remove_mean"),
            fft_engine=metadata.pop("fft_engine", "auto"),
            scaling=metadata.pop("scaling", "raw"),
            zero_padding=metadata.pop("zero_padding", True),
            nfft=metadata.pop("nfft", None),
        )

        total_time = time.time() - start_time
        if process is not None and initial_memory is not None:
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = final_memory - initial_memory
            logger.debug("Memory increase: %.1f MB", memory_increase)

        logger.info(
            "Loaded existing FFT data in %.3fs, spectrum shape: %s",
            total_time,
            spectrum.shape,
        )
        return result_cls(
            frequencies=frequencies,
            spectrum=spectrum,
            metadata=metadata,
            config=config,
        )
    except Exception as exc:
        logger.warning("Could not load existing FFT data: %s", exc)
        return None


def verify_fft_parameters(
    *,
    existing_result: Any,
    window: Any,
    filter_type: Any,
    engine: Any,
    scaling: Any,
    zero_padding: Any,
    nfft: Any,
    metadata_overrides: dict[str, Any],
    metadata_keys_to_check: Sequence[str] = ("z_layer", "source_dataset", "slice_identifier"),
) -> bool:
    """Verify whether request parameters match an existing cached FFT result."""
    engine_match = True
    if engine not in (None, "auto"):
        engine_match = existing_result.config.fft_engine == engine

    config_match = (
        existing_result.config.window_function == window
        and existing_result.config.filter_type == filter_type
        and engine_match
        and getattr(existing_result.config, "scaling", "raw") == scaling
        and existing_result.config.zero_padding == zero_padding
        and existing_result.config.nfft == nfft
    )

    metadata_match = True
    for key in metadata_keys_to_check:
        if key in metadata_overrides and key in existing_result.metadata:
            if metadata_overrides[key] != existing_result.metadata[key]:
                metadata_match = False
                break

    return config_match and metadata_match


__all__ = ["load_existing_fft_result", "verify_fft_parameters"]
