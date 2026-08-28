"""Soliton analysis namespace for MMPP."""

from ._coordinates import XYConvention, grid_xy
from ._field import select_snapshot, valid_magnetization_mask
from .batch import BatchSolitonsInterface
from .interface import DatasetSpecificSolitons, SolitonInterface
from .skyrmion import (
    BatchSkyrmionInterface,
    SizeFitConfig,
    SkyrmionAnalysisResult,
    SkyrmionConfig,
    SkyrmionInterface,
    SkyrmionSizeResult,
    SkyrmionTopologyConfig,
    SkyrmionTopologyResult,
    detect_skyrmion,
    fit_skyrmion_size,
)

__all__ = [
    "SolitonInterface",
    "DatasetSpecificSolitons",
    "BatchSolitonsInterface",
    "XYConvention",
    "grid_xy",
    "select_snapshot",
    "valid_magnetization_mask",
    "BatchSkyrmionInterface",
    "SizeFitConfig",
    "SkyrmionAnalysisResult",
    "SkyrmionConfig",
    "SkyrmionInterface",
    "SkyrmionSizeResult",
    "SkyrmionTopologyConfig",
    "SkyrmionTopologyResult",
    "detect_skyrmion",
    "fit_skyrmion_size",
]
