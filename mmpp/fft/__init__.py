"""
FFT Module

Provides comprehensive FFT analysis capabilities similar to numpy.fft.
Main entry point through the FFT class.
"""

from .core import FFT
from .compute_fft import FFTCompute, FFTComputeResult

__all__ = ['FFT', 'FFTCompute', 'FFTComputeResult']
