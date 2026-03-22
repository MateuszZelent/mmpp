"""Bridge namespace connecting numerical and analytical vortex flows."""

from .extract import AnalyticalParameterResolution, extract_model_defaults
from .interface import BridgeInterface

__all__ = ["BridgeInterface", "AnalyticalParameterResolution", "extract_model_defaults"]
