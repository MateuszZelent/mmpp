"""Internal cache helpers for persisted FFT results."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def load_existing_fft_result(
    *,
    zarr_path: str,
    dataset_name: str,
    result_cls: Any,
    config_cls: Any,
    psutil_module: Any | None,
    logger: Any,
    source_dataset: str | None = None,
    expected_source_revision: str | None = None,
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

        from ..cache.transactions import open_published_cache, peek_cache_revision
        from ..core.zarr_integrity import source_revision

        parent = Path(zarr_path) / "fft"
        revision = (
            expected_source_revision or source_revision(zarr_path, source_dataset)
            if source_dataset is not None
            else peek_cache_revision(parent, dataset_name)
        )
        fft_group = open_published_cache(parent, dataset_name, source_revision=revision)
        if fft_group is None:
            return None
        stored_dataset = fft_group.attrs.get("source_dataset")
        if (
            not isinstance(stored_dataset, str)
            or (source_dataset is not None and stored_dataset != source_dataset)
            or (
                expected_source_revision is None
                and source_revision(zarr_path, stored_dataset) != revision
            )
        ):
            logger.debug("Ignoring FFT cache with a stale or unknown source revision")
            return None
        fft_path = f"fft/{dataset_name}"
        required_arrays = {"spectrum", "frequencies"}
        missing_arrays = required_arrays.difference(fft_group.keys())
        if missing_arrays:
            logger.info(
                "Ignoring incomplete FFT cache %s (missing: %s)",
                fft_path,
                ", ".join(sorted(missing_arrays)),
            )
            return None

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
        required_config = {
            "window_function",
            "filter_type",
            "fft_engine",
            "scaling",
            "zero_padding",
        }
        if not required_config.issubset(metadata):
            logger.info(
                "Ignoring FFT cache %s without its complete configuration", fft_path
            )
            return None
        config = config_cls(
            window_function=metadata["window_function"],
            filter_type=metadata["filter_type"],
            fft_engine=metadata["fft_engine"],
            scaling=metadata["scaling"],
            zero_padding=metadata["zero_padding"],
            nfft=metadata.get("nfft"),
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
    metadata_keys_to_check: Sequence[str] = (
        "z_layer",
        "source_dataset",
        "slice_identifier",
        "resample_nonuniform",
    ),
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
        if key in metadata_overrides:
            if key not in existing_result.metadata:
                metadata_match = False
                break
            if metadata_overrides[key] != existing_result.metadata[key]:
                metadata_match = False
                break

    return config_match and metadata_match


__all__ = ["load_existing_fft_result", "verify_fft_parameters"]
