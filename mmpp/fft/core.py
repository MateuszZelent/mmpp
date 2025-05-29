"""
FFT Core Module

Main FFT class providing unified interface for FFT analysis.
"""

from typing import Optional, Dict, List, Union, Any, Tuple
import numpy as np

from .compute_fft import FFTCompute, FFTComputeResult
from .plot import FFTPlotter


class FFT:
    """
    Main FFT analysis class providing numpy.fft-like interface.
    
    This class aggregates FFT computation and plotting capabilities
    for MMPP job results.
    """
    
    def __init__(self, job_result, mmpp_instance: Optional[Any] = None):
        """
        Initialize FFT analyzer for a job result.
        
        Parameters:
        -----------
        job_result : ZarrJobResult
            Job result to analyze
        mmpp_instance : MMPP, optional
            Reference to parent MMPP instance
        """
        self.job_result = job_result
        self.mmpp = mmpp_instance
        
        # Initialize compute engine
        self._compute = FFTCompute()
        
        # Initialize plotter (lazy loaded)
        self._plotter = None
        
        # Cache for FFT results
        self._cache = {}
    
    @property
    def plotter(self) -> FFTPlotter:
        """Get plotter instance (lazy initialization)."""
        if self._plotter is None:
            self._plotter = FFTPlotter([self.job_result], self.mmpp)
        return self._plotter
    
    def _get_cache_key(self, dataset_name: str, z_layer: int, method: int, **kwargs) -> str:
        """Generate cache key for FFT results."""
        key_parts = [dataset_name, str(z_layer), str(method)]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        return "|".join(key_parts)
    
    def _compute_fft(self, dataset_name: str = "m_z11", 
                     z_layer: int = -1, 
                     method: int = 1,
                     use_cache: bool = True,
                     save: bool = False,
                     force: bool = False,
                     save_dataset_name: Optional[str] = None,
                     **kwargs) -> FFTComputeResult:
        """
        Compute FFT with caching and optional saving.
        
        Parameters:
        -----------
        dataset_name : str, optional
            Dataset name (default: "m_z11")
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        use_cache : bool, optional
            Use memory cache (default: True)
        save : bool, optional
            Save result to zarr file (default: False)
        force : bool, optional
            Force recalculation and overwrite existing (default: False)
        save_dataset_name : str, optional
            Custom name for saved dataset (default: auto-generated)
        **kwargs : Any
            Additional FFT configuration options
            
        Returns:
        --------
        FFTComputeResult
            FFT computation result
        """
        cache_key = self._get_cache_key(dataset_name, z_layer, method, **kwargs)
        
        # Check memory cache only if not forcing and not saving
        if use_cache and not force and not save and cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            result = self._compute.calculate_fft_data(
                self.job_result.path, dataset_name, z_layer, method, 
                save=save, force=force, save_dataset_name=save_dataset_name,
                save_dir='fft', chunks=None, **kwargs
            )
        except OSError as e:
            if "directory not empty" in str(e).lower():
                print("Warning: FFT directory already exists and is not empty. Use force=True to overwrite.")
            raise
            
        # Cache result only if not forcing
        if use_cache and not force:
            self._cache[cache_key] = result
            
        return result
    
    def spectrum(self, dset: str = "m_z11", 
                 z_layer: int = -1, 
                 method: int = 1,
                 save: bool = False,
                 force: bool = False,
                 save_dataset_name: Optional[str] = None,
                 **kwargs) -> np.ndarray:
        """
        Compute FFT spectrum.
        
        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m_z11")
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        save : bool, optional
            Save result to zarr file (default: False)
        force : bool, optional
            Force recalculation and overwrite existing (default: False)
        save_dataset_name : str, optional
            Custom name for saved dataset (default: auto-generated)
        **kwargs : Any
            Additional FFT configuration options
            
        Returns:
        --------
        np.ndarray
            Complex FFT spectrum
        """
        result = self._compute_fft(dset, z_layer, method, save=save, force=force, 
                                  save_dataset_name=save_dataset_name, **kwargs)
        return result.spectrum
    
    def frequencies(self, dset: str = "m_z11", 
                    z_layer: int = -1, 
                    method: int = 1,
                    save: bool = False,
                    force: bool = False,
                    save_dataset_name: Optional[str] = None,
                    **kwargs) -> np.ndarray:
        """
        Get frequency array for FFT.
        
        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m_z11")
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        save : bool, optional
            Save result to zarr file (default: False)
        force : bool, optional
            Force recalculation and overwrite existing (default: False)
        save_dataset_name : str, optional
            Custom name for saved dataset (default: auto-generated)
        **kwargs : Any
            Additional FFT configuration options
            
        Returns:
        --------
        np.ndarray
            Frequency array
        """
        result = self._compute_fft(dset, z_layer, method, save=save, force=force,
                                  save_dataset_name=save_dataset_name, **kwargs)
        return result.frequencies
    
    def power(self, dset: str = "m_z11", 
              z_layer: int = -1, 
              method: int = 1,
              save: bool = False,
              force: bool = False,
              save_dataset_name: Optional[str] = None,
              **kwargs) -> np.ndarray:
        """
        Compute power spectrum.
        
        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m_z11")
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        save : bool, optional
            Save result to zarr file (default: False)
        force : bool, optional
            Force recalculation and overwrite existing (default: False)
        save_dataset_name : str, optional
            Custom name for saved dataset (default: auto-generated)
        **kwargs : Any
            Additional FFT configuration options
            
        Returns:
        --------
        np.ndarray
            Power spectrum (|FFT|^2)
        """
        spectrum = self.spectrum(dset, z_layer, method, save=save, force=force,
                               save_dataset_name=save_dataset_name, **kwargs)
        return np.abs(spectrum)**2
    
    def phase(self, dset: str = "m_z11", 
              z_layer: int = -1, 
              method: int = 1,
              **kwargs) -> np.ndarray:
        """
        Compute phase spectrum.
        
        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m_z11")
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        **kwargs : Any
            Additional FFT configuration options
            
        Returns:
        --------
        np.ndarray
            Phase spectrum
        """
        spectrum = self.spectrum(dset, z_layer, method, **kwargs)
        return np.angle(spectrum)
    
    def magnitude(self, dset: str = "m_z11", 
                  z_layer: int = -1, 
                  method: int = 1,
                  **kwargs) -> np.ndarray:
        """
        Compute magnitude spectrum.
        
        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m_z11")
        z_layer : int, optional
            Z-layer (default: -1)
        method : int, optional
            FFT method (default: 1)
        **kwargs : Any
            Additional FFT configuration options
            
        Returns:
        --------
        np.ndarray
            Magnitude spectrum (|FFT|)
        """
        spectrum = self.spectrum(dset, z_layer, method, **kwargs)
        return np.abs(spectrum)
    
    def plot_spectrum(self, dset: str = "m_z11", 
                      method: int = 1,
                      z_layer: int = -1,
                      log_scale: bool = True,
                      normalize: bool = False,
                      save: bool = True,
                      force: bool = False,
                      save_dataset_name: Optional[str] = None,
                      **kwargs) -> Tuple[Any, Any]:
        """
        Plot power spectrum.
        
        Parameters:
        -----------
        dset : str, optional
            Dataset name (default: "m_z11")
        method : int, optional
            FFT method (default: 1)
        z_layer : int, optional
            Z-layer (default: -1)
        log_scale : bool, optional
            Use logarithmic scale (default: True)
        normalize : bool, optional
            Normalize spectrum (default: False)
        save : bool, optional
            Save FFT result to zarr file (default: True)
        force : bool, optional
            Force recalculation and overwrite existing (default: False)
        save_dataset_name : str, optional
            Custom name for saved dataset (default: auto-generated)
        **kwargs : Any
            Additional plotting options
            
        Returns:
        --------
        tuple
            (figure, axes) matplotlib objects
        """
        return self.plotter.power_spectrum(
            dataset_name=dset, method=method, z_layer=z_layer,
            log_scale=log_scale, normalize=normalize, 
            save=save, force=force, save_dataset_name=save_dataset_name,
            **kwargs
        )
    
    def clear_cache(self):
        """Clear FFT computation cache."""
        self._cache.clear()
    
    def __repr__(self) -> str:
        return f"FFT(path='{self.job_result.path}')"
