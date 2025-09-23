"""
FMR Mode Visualization Module

Professional implementation for visualizing FMR modes with interactive spectrum.
Provides both programmatic and interactive interfaces for mode analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Union

import math

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Import shared logging configuration
from ..cli.logging_config import get_mmpp_logger, setup_mmpp_logging

# Get logger for FMR modes
log = get_mmpp_logger("mmpp.fft.modes")

# Import electromagnetic analysis module
try:
    from .electromagnetic_analysis import (
        ElectromagneticAnalysisConfig,
        PoyntingVectorAnalysis,
        QFactorAnalysis,
        RadiationPatternAnalysis,
        analyze_electromagnetic_properties,
        create_comprehensive_em_report,
    )

    EM_ANALYSIS_AVAILABLE = True
except ImportError:
    EM_ANALYSIS_AVAILABLE = False
    log.warning("Electromagnetic analysis module not available")

# Import styling functions from plotting module
try:
    from ..plotting import apply_custom_colors, load_paper_style, setup_custom_fonts

    STYLING_AVAILABLE = True
except ImportError:
    STYLING_AVAILABLE = False
    log.warning("Styling functions not available - using default matplotlib styling")

# FFmpeg installation utilities
def _detect_platform():
    """Detect the current platform for FFmpeg installation."""
    import platform
    
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == 'linux':
        if machine in ['x86_64', 'amd64']:
            return 'linux-amd64'
        elif machine in ['aarch64', 'arm64']:
            return 'linux-arm64'
        elif machine.startswith('arm'):
            return 'linux-armhf'
    elif system == 'darwin':  # macOS
        if machine in ['x86_64', 'amd64']:
            return 'macos-amd64'
        elif machine in ['arm64', 'aarch64']:
            return 'macos-arm64'
    elif system == 'windows':
        if machine in ['x86_64', 'amd64']:
            return 'windows-amd64'
    
    return None


def _get_ffmpeg_download_info(platform_id):
    """Get download URL and extraction info for different platforms."""
    
    urls = {
        'linux-amd64': {
            'url': 'https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz',
            'format': 'tar.xz',
            'binary_pattern': '*/ffmpeg'
        },
        'linux-arm64': {
            'url': 'https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-arm64-static.tar.xz',
            'format': 'tar.xz', 
            'binary_pattern': '*/ffmpeg'
        },
        'linux-armhf': {
            'url': 'https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-armhf-static.tar.xz',
            'format': 'tar.xz',
            'binary_pattern': '*/ffmpeg'
        }
    }
    
    return urls.get(platform_id)


def _ensure_ffmpeg_available():
    """
    Ensure FFmpeg is available for animation saving.
    
    If FFmpeg is not found in system PATH, automatically downloads
    and installs a static build from appropriate source.
    
    Returns:
    --------
    str or None
        Path to FFmpeg executable if available, None if failed
    """
    import shutil
    import subprocess
    import platform
    import urllib.request
    import tarfile
    import tempfile
    from pathlib import Path
    
    # Check if FFmpeg is already in PATH
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        log.debug(f"FFmpeg found in system PATH: {ffmpeg_path}")
        return ffmpeg_path
    
    # Check if we already have a downloaded version
    mmpp_cache_dir = Path.home() / '.mmpp' / 'bin'
    cached_ffmpeg = mmpp_cache_dir / 'ffmpeg'
    
    if cached_ffmpeg.exists() and cached_ffmpeg.is_file():
        # Test if cached version works
        try:
            result = subprocess.run([str(cached_ffmpeg), '-version'], 
                                  capture_output=True, timeout=5)
            if result.returncode == 0:
                log.debug(f"Using cached FFmpeg: {cached_ffmpeg}")
                return str(cached_ffmpeg)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            log.warning("Cached FFmpeg not working, will re-download")
    
    # Attempt automatic installation
    log.info("FFmpeg not found in system, attempting automatic installation...")
    
    try:
        return _install_ffmpeg_automatic()
    except Exception as e:
        log.error(f"Automatic FFmpeg installation failed: {e}")
        log.warning("Please install FFmpeg manually or call mmpp.fft.modes.install_ffmpeg() for detailed installation help")
        return None


def _install_ffmpeg_automatic():
    """
    Automatically install FFmpeg with minimal user interaction.
    
    Returns:
    --------
    str
        Path to installed FFmpeg executable
        
    Raises:
    -------
    RuntimeError
        If installation fails
    """
    import shutil
    import subprocess
    import urllib.request
    import tarfile
    import tempfile
    from pathlib import Path
    
    # Detect platform
    platform_id = _detect_platform()
    if not platform_id:
        raise RuntimeError(f"Unsupported platform for automatic FFmpeg installation")
    
    # Get download info
    download_info = _get_ffmpeg_download_info(platform_id)
    if not download_info:
        raise RuntimeError(f"No download URL available for platform: {platform_id}")
    
    # Create cache directory
    mmpp_cache_dir = Path.home() / '.mmpp' / 'bin'
    mmpp_cache_dir.mkdir(parents=True, exist_ok=True)
    cached_ffmpeg = mmpp_cache_dir / 'ffmpeg'
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        archive_path = temp_path / f"ffmpeg-static.{download_info['format'].replace('.', '_')}"
        
        # Download with timeout and retry
        for attempt in range(3):
            try:
                log.info(f"Downloading FFmpeg from {download_info['url']} (attempt {attempt + 1}/3)...")
                
                # Simple download with timeout
                urllib.request.urlretrieve(download_info['url'], archive_path)
                log.info("Download completed, extracting...")
                break
                
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise RuntimeError(f"Failed to download FFmpeg after 3 attempts: {e}")
                log.warning(f"Download attempt {attempt + 1} failed: {e}, retrying...")
        
        # Extract archive
        if download_info['format'] == 'tar.xz':
            with tarfile.open(archive_path, 'r:xz') as tar:
                # Find ffmpeg binary in archive
                ffmpeg_member = None
                for member in tar.getmembers():
                    if member.name.endswith('/ffmpeg') and member.isfile():
                        ffmpeg_member = member
                        break
                
                if not ffmpeg_member:
                    raise RuntimeError("FFmpeg binary not found in downloaded archive")
                
                # Extract just the ffmpeg binary
                tar.extract(ffmpeg_member, temp_path)
                extracted_ffmpeg = temp_path / ffmpeg_member.name
                
                # Copy to cache directory
                shutil.copy2(extracted_ffmpeg, cached_ffmpeg)
                cached_ffmpeg.chmod(0o755)  # Make executable
        else:
            raise RuntimeError(f"Unsupported archive format: {download_info['format']}")
        
        log.info(f"FFmpeg installed successfully to {cached_ffmpeg}")
        
        # Test the installation
        result = subprocess.run([str(cached_ffmpeg), '-version'], 
                              capture_output=True, timeout=10)
        if result.returncode == 0:
            log.info("FFmpeg installation verified successfully")
            return str(cached_ffmpeg)
        else:
            raise RuntimeError(f"FFmpeg test failed: {result.stderr.decode()}")


def install_ffmpeg(force: bool = False, verbose: bool = True) -> Optional[str]:
    """
    Install FFmpeg for animation support with comprehensive error handling.
    
    This function provides multiple installation methods and detailed 
    troubleshooting information for FFmpeg installation.
    
    Parameters:
    -----------
    force : bool, default False
        Force reinstallation even if FFmpeg is already available
    verbose : bool, default True
        Show detailed installation progress and troubleshooting info
        
    Returns:
    --------
    str or None
        Path to ffmpeg executable if successful, None if failed
        
    Examples:
    ---------
    >>> import mmpp
    >>> # Install FFmpeg automatically
    >>> ffmpeg_path = mmpp.fft.modes.install_ffmpeg()
    >>> 
    >>> # Force reinstallation
    >>> ffmpeg_path = mmpp.fft.modes.install_ffmpeg(force=True)
    """
    import shutil
    import subprocess
    import platform
    from pathlib import Path
    
    if verbose:
        log.info("🔧 Installing FFmpeg for animation support...")
    
    # Check if already available (unless forcing)
    if not force:
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            if verbose:
                log.info(f"✅ FFmpeg already available in system PATH: {ffmpeg_path}")
            return ffmpeg_path
            
        # Check cached version
        cached_ffmpeg = Path.home() / '.mmpp' / 'bin' / 'ffmpeg'
        if cached_ffmpeg.exists():
            try:
                result = subprocess.run([str(cached_ffmpeg), '-version'], 
                                      capture_output=True, timeout=5)
                if result.returncode == 0:
                    if verbose:
                        log.info(f"✅ FFmpeg already available in cache: {cached_ffmpeg}")
                    return str(cached_ffmpeg)
            except:
                pass
    
    # Detect platform
    platform_id = _detect_platform()
    if verbose:
        log.info(f"🔍 Detected platform: {platform_id or 'Unsupported'}")
    
    if not platform_id:
        if verbose:
            log.error("❌ Unsupported platform for automatic installation")
            _show_manual_installation_help()
        return None
    
    # Attempt automatic installation
    try:
        ffmpeg_path = _install_ffmpeg_automatic()
        if verbose:
            log.info(f"✅ FFmpeg successfully installed: {ffmpeg_path}")
        return ffmpeg_path
        
    except Exception as e:
        if verbose:
            log.error(f"❌ Automatic installation failed: {e}")
            _show_manual_installation_help()
        return None


def _show_manual_installation_help():
    """Show detailed manual installation instructions."""
    import platform
    
    system = platform.system().lower()
    
    log.info("📖 Manual FFmpeg Installation Instructions:")
    log.info("=" * 50)
    
    if system == 'linux':
        log.info("🐧 Linux:")
        log.info("  Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg")
        log.info("  RHEL/CentOS:   sudo yum install ffmpeg  (or dnf)")
        log.info("  Arch Linux:    sudo pacman -S ffmpeg")
        log.info("  Conda:         conda install -c conda-forge ffmpeg")
    elif system == 'darwin':
        log.info("🍎 macOS:")
        log.info("  Homebrew:      brew install ffmpeg")
        log.info("  MacPorts:      sudo port install ffmpeg")
        log.info("  Conda:         conda install -c conda-forge ffmpeg")
    elif system == 'windows':
        log.info("🪟 Windows:")
        log.info("  Chocolatey:    choco install ffmpeg")
        log.info("  Scoop:         scoop install ffmpeg")
        log.info("  Manual:        Download from https://ffmpeg.org/download.html")
    
    log.info("")
    log.info("🔗 Official Downloads: https://ffmpeg.org/download.html")
    log.info("📚 Documentation:     https://ffmpeg.org/documentation.html")
    log.info("")
    log.info("💡 After installation, restart your Python session to use FFmpeg")
    log.info("=" * 50)


def check_ffmpeg_installation() -> dict:
    """
    Check current FFmpeg installation status and provide diagnostics.
    
    Returns:
    --------
    dict
        Dictionary containing installation status and diagnostics
        
    Examples:
    ---------
    >>> import mmpp
    >>> status = mmpp.fft.modes.check_ffmpeg_installation()
    >>> print(status['available'])  # True if FFmpeg is available
    >>> print(status['path'])       # Path to FFmpeg executable
    >>> print(status['version'])    # FFmpeg version string
    """
    import shutil
    import subprocess
    from pathlib import Path
    
    result = {
        'available': False,
        'path': None,
        'version': None,
        'cached_version': False,
        'platform': _detect_platform(),
        'diagnostics': []
    }
    
    # Check system PATH
    system_ffmpeg = shutil.which('ffmpeg')
    if system_ffmpeg:
        result['available'] = True
        result['path'] = system_ffmpeg
        result['diagnostics'].append(f"✅ Found in system PATH: {system_ffmpeg}")
        
        # Get version
        try:
            version_result = subprocess.run([system_ffmpeg, '-version'], 
                                          capture_output=True, timeout=10)
            if version_result.returncode == 0:
                version_line = version_result.stdout.decode().split('\n')[0]
                result['version'] = version_line
                result['diagnostics'].append(f"✅ Version: {version_line}")
            else:
                result['diagnostics'].append("⚠️  Could not determine version")
        except Exception as e:
            result['diagnostics'].append(f"⚠️  Version check failed: {e}")
    else:
        result['diagnostics'].append("❌ Not found in system PATH")
    
    # Check cached version
    cached_ffmpeg = Path.home() / '.mmpp' / 'bin' / 'ffmpeg'
    if cached_ffmpeg.exists():
        result['diagnostics'].append(f"🔍 Found cached version: {cached_ffmpeg}")
        
        if not result['available']:  # Only test if system version not found
            try:
                version_result = subprocess.run([str(cached_ffmpeg), '-version'], 
                                              capture_output=True, timeout=10)
                if version_result.returncode == 0:
                    result['available'] = True
                    result['path'] = str(cached_ffmpeg)
                    result['cached_version'] = True
                    version_line = version_result.stdout.decode().split('\n')[0]
                    result['version'] = version_line
                    result['diagnostics'].append(f"✅ Cached version working: {version_line}")
                else:
                    result['diagnostics'].append("❌ Cached version not working")
            except Exception as e:
                result['diagnostics'].append(f"❌ Cached version test failed: {e}")
    else:
        result['diagnostics'].append("❌ No cached version found")
    
    # Add platform info
    if result['platform']:
        result['diagnostics'].append(f"🔍 Platform: {result['platform']}")
    else:
        result['diagnostics'].append("⚠️  Unsupported platform for automatic installation")
    
    return result

# Import electromagnetic analysis module
try:
    # Electromagnetic analysis capabilities available
    EM_ANALYSIS_AVAILABLE = True
except ImportError:
    EM_ANALYSIS_AVAILABLE = False
    log.warning("Electromagnetic analysis module not available")

# Import dependencies with error handling
try:
    import zarr

    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    log.error("Zarr not available - mode analysis disabled")

try:
    from pyzfn import Pyzfn

    PYZFN_AVAILABLE = True
except ImportError:
    PYZFN_AVAILABLE = False

try:
    from matplotlib.axes import Axes
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    log.warning("Matplotlib not available - mode visualization disabled")

try:
    from scipy.signal import find_peaks

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    log.warning("SciPy not available - peak detection features limited")

from .metrics import (
    PeakWidth,
    compute_half_width_at_half_max,
    format_width_value,
    normalize_peak_width_option,
)

from .mode_characterization import (
    ModeCharacterAnalyzer,
    ModeCharacteristicConfig,
    ModeCharacterizationResult,
)

# Check for animation support
try:
    from matplotlib.animation import FuncAnimation, PillowWriter

    ANIMATION_AVAILABLE = True

    # Check for FFmpeg support
    try:
        from matplotlib.animation import FFMpegWriter

        FFMPEG_AVAILABLE = True
        log.debug("FFmpeg available for MP4 animations")
    except ImportError:
        FFMPEG_AVAILABLE = False
        log.debug("FFmpeg not available - MP4 animations will fallback to GIF")

except ImportError:
    ANIMATION_AVAILABLE = False
    FFMPEG_AVAILABLE = False
    log.warning("Animation support not available")

# Check for scientific colormaps
try:
    import cmcrameri.cm as cmc

    CMCRAMERI_AVAILABLE = True
    log.debug("cmcrameri colormaps available")
except ImportError:
    CMCRAMERI_AVAILABLE = False
    log.debug("cmcrameri not available - using standard matplotlib colormaps")

try:
    import cmocean

    CMOCEAN_AVAILABLE = True
    log.debug("cmocean colormaps available")
except ImportError:
    CMOCEAN_AVAILABLE = False
    log.debug("cmocean not available - using standard matplotlib colormaps")

try:
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    AXES_GRID_AVAILABLE = True
except ImportError:
    AXES_GRID_AVAILABLE = False
    log.warning(
        "mpl_toolkits.axes_grid1 not available - colorbar and scalebar enhancements disabled"
    )


class MidpointNormalize(mcolors.Normalize):
    """
    Matplotlib normalization class with symmetric colormap around midpoint.

    Useful for data that has meaningful zero point (like magnetic field components)
    where you want symmetric color scaling around zero.
    """

    def __init__(self, vmin=None, vmax=None, midpoint=0, clip=False):
        """
        Initialize normalization.

        Parameters:
        -----------
        vmin : float, optional
            Minimum value for normalization
        vmax : float, optional
            Maximum value for normalization
        midpoint : float, optional
            Value that should map to center of colormap (default: 0)
        clip : bool, optional
            Whether to clip values outside [vmin, vmax]
        """
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        """Apply normalization to values."""
        # Handle the case where vmin, vmax, or midpoint could be None
        if self.vmin is None or self.vmax is None:
            return mcolors.Normalize.__call__(self, value, clip)

        # Calculate normalized positions for min, mid, max
        normalized_min = max(
            0,
            1
            / 2
            * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))),
        )
        normalized_max = min(
            1,
            1
            / 2
            * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))),
        )
        normalized_mid = 0.5

        # Interpolate
        x = [self.vmin, self.midpoint, self.vmax]
        y = [normalized_min, normalized_mid, normalized_max]

        return np.ma.masked_array(np.interp(value, x, y))


def setup_animation_styling(
    use_paper_style: bool = True, use_custom_fonts: bool = True
) -> bool:
    """
    Setup styling for FMR mode animations using MMPP paper style.

    Parameters:
    -----------
    use_paper_style : bool, optional
        Whether to apply paper.mplstyle styling (default: True)
    use_custom_fonts : bool, optional
        Whether to setup custom fonts (default: True)

    Returns:
    --------
    bool
        True if styling was successfully applied
    """
    if not STYLING_AVAILABLE:
        log.warning(
            "Styling functions not available - using default matplotlib styling"
        )
        return False

    try:
        success = True

        # Setup custom fonts if requested
        if use_custom_fonts:
            font_success = setup_custom_fonts(verbose=False)
            if not font_success:
                log.warning("Custom font setup failed - using default fonts")
                success = False

        # Load paper style if requested
        if use_paper_style:
            style_success = load_paper_style(verbose=False)
            if style_success:
                log.debug("✓ Applied paper.mplstyle to mode animations")
            else:
                log.warning("Paper style loading failed - using default style")
                success = False

        # Apply custom colors for better visibility
        custom_colors = {
            "text": "#2E2E2E",  # Dark gray for text
            "axes": "#2E2E2E",  # Dark gray for axes
            "grid": "#CCCCCC",  # Light gray for grid
        }
        apply_custom_colors(custom_colors)

        return success

    except Exception as e:
        log.warning(f"Animation styling setup failed: {e}")
        return False


def check_ffmpeg_available() -> bool:
    """
    Check if ffmpeg is available for MP4 animations.

    Returns:
    --------
    bool
        True if ffmpeg is available
    """
    try:
        import subprocess

        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (
        subprocess.TimeoutExpired,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ):
        return False


# Simple wrapper for the main install function
def install_ffmpeg_simple() -> Optional[str]:
    """
    Install FFmpeg if not available on the system.
    
    This is a simple wrapper that calls the main install_ffmpeg function
    with user-friendly defaults.
    
    Returns:
    --------
    str or None
        Path to ffmpeg executable if successful, None if failed
        
    Examples:
    ---------
    >>> import mmpp
    >>> # Install FFmpeg with detailed output
    >>> ffmpeg_path = mmpp.fft.modes.install_ffmpeg_simple()
    >>> 
    >>> # Or use the main function with parameters
    >>> ffmpeg_path = mmpp.fft.modes.install_ffmpeg(force=True, verbose=True)
    """
    # Use the comprehensive function with user-friendly defaults
    return install_ffmpeg(force=False, verbose=True)


@dataclass
class ModeVisualizationConfig:
    """Configuration for mode visualization."""

    # Figure settings
    figsize: tuple[float, float] = (16, 10)
    dpi: int = 100

    # Spectrum settings
    spectrum_log_scale: bool = False
    spectrum_normalize: bool = True
    peak_threshold: float = 0.1
    peak_min_distance: int = 5

    # Mode visualization settings
    show_magnitude: bool = True
    show_phase: bool = True
    show_combined: bool = True
    colormap_magnitude: str = "cmc.berlin"  # cmcrameri berlin for amplitude data
    colormap_phase: str = "cmc.romaO"  # cmcrameri romaO for phase data
    colormap_animation: str = (
        "balance"  # cmocean.cm.balance for animations, RdBu_r fallback
    )
    interpolation: str = "nearest"
    use_midpoint_norm: bool = False  # Use MidpointNormalize for diverging data
    animation_time_steps: int = 60  # Number of time steps for one full phase cycle

    # Publication-style annotations
    show_scalebar: bool = True
    scalebar_length_nm: Optional[float] = None  # Auto-computed when None
    scalebar_location: str = "lower right"
    scalebar_pad: float = 0.3
    scalebar_color: str = "white"
    scalebar_fontsize: int = 9
    scalebar_frame: bool = False
    scalebar_height_fraction: float = 0.01
    scale_units: str = "nm"

    colorbar_fraction: float = 0.04   # Proper colorbar width
    colorbar_pad: float = 0.01       # Small padding for close positioning
    colorbar_ticklabel_size: int = 9  # Larger tick labels
    colorbar_label_size: int = 10     # Larger labels
    colorbar_labels: dict[str, str] = field(
        default_factory=lambda: {
            "magnitude": "Magnetization |m|",
            "phase": "Phase (rad)", 
            "combined": "Re(m) × cos(φ)",
        }
    )

    # Frequency range for analysis
    f_min: float = 0.0
    f_max: float = 40.0

    # Layout settings
    spectrum_width_ratio: float = 0.4
    modes_width_ratio: float = 0.6

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.f_min >= self.f_max:
            raise ValueError(
                f"f_min ({self.f_min}) must be less than f_max ({self.f_max})"
            )

        if self.peak_threshold < 0 or self.peak_threshold > 1:
            raise ValueError(
                f"peak_threshold must be between 0 and 1, got {self.peak_threshold}"
            )

        if self.peak_min_distance < 1:
            raise ValueError(
                f"peak_min_distance must be >= 1, got {self.peak_min_distance}"
            )

        if self.spectrum_width_ratio <= 0 or self.modes_width_ratio <= 0:
            raise ValueError("Width ratios must be positive")

        if self.dpi < 50 or self.dpi > 500:
            log.warning(f"Unusual DPI value: {self.dpi}")

        # Validate colormaps
        try:
            self._resolve_colormap(self.colormap_magnitude)
            self._resolve_colormap(self.colormap_phase)
        except Exception as e:
            log.warning(f"Colormap validation failed: {e}")

        if self.scalebar_length_nm is not None and self.scalebar_length_nm <= 0:
            raise ValueError("scalebar_length_nm must be positive when provided")

        if self.scalebar_height_fraction <= 0 or self.scalebar_height_fraction > 0.1:
            raise ValueError(
                "scalebar_height_fraction should be within (0, 0.1] for sensible display"
            )

        if self.colorbar_fraction <= 0 or self.colorbar_pad < 0:
            raise ValueError("colorbar_fraction must be > 0 and colorbar_pad >= 0")

    def _resolve_colormap(self, cmap_name: str):
        """
        Resolve colormap from various sources (cmcrameri, cmocean, matplotlib).

        Parameters:
        -----------
        cmap_name : str
            Name of the colormap

        Returns:
        --------
        matplotlib colormap object
        """
        # Try cmcrameri first (scientific colormaps)
        if CMCRAMERI_AVAILABLE:
            try:
                return getattr(cmc, cmap_name)
            except AttributeError:
                pass

        # Try cmocean (oceanographic colormaps)
        if CMOCEAN_AVAILABLE:
            try:
                import cmocean

                return getattr(cmocean.cm, cmap_name)
            except AttributeError:
                pass

        # Fallback to matplotlib
        import matplotlib.pyplot as plt

        return plt.get_cmap(cmap_name)


@dataclass
class Peak:
    """Peak data structure."""

    idx: int
    freq: float
    amplitude: float


class FMRModeData:
    """Container for FMR mode data at a specific frequency."""

    def __init__(
        self,
        frequency: float,
        mode_array: np.ndarray,
        extent: Optional[tuple[float, float, float, float]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize FMR mode data.

        Parameters:
        -----------
        frequency : float
            Frequency in GHz
        mode_array : np.ndarray
            Complex mode array with shape (ny, nx, 3) for spatial x-y and magnetization components
        extent : tuple, optional
            Spatial extent [x_min, x_max, y_min, y_max] in nm
        metadata : dict, optional
            Additional metadata
        """
        self.frequency = frequency
        self.mode_array = mode_array
        self.extent = extent or (0, mode_array.shape[1], 0, mode_array.shape[0])
        self.metadata = metadata or {}

        # Validate input
        if not isinstance(mode_array, np.ndarray):
            raise TypeError("mode_array must be numpy array")
        if mode_array.ndim != 3 or mode_array.shape[2] != 3:
            raise ValueError("mode_array must have shape (ny, nx, 3)")

    @property
    def magnitude(self) -> np.ndarray:
        """Get magnitude of mode for each component."""
        return np.abs(self.mode_array)

    @property
    def phase(self) -> np.ndarray:
        """Get phase of mode for each component."""
        return np.angle(self.mode_array)

    @property
    def total_magnitude(self) -> np.ndarray:
        """Get total magnitude across all components."""
        return np.sqrt(np.sum(self.magnitude**2, axis=2))

    def get_component(self, component: Union[int, str]) -> np.ndarray:
        """
        Get specific magnetization component.

        Parameters:
        -----------
        component : int or str
            Component index (0, 1, 2) or name ('x', 'y', 'z', 'mx', 'my', 'mz')

        Returns:
        --------
        np.ndarray
            Complex mode array for specified component
        """
        component_map = {"x": 0, "y": 1, "z": 2, "mx": 0, "my": 1, "mz": 2}

        if isinstance(component, str):
            if component.lower() not in component_map:
                raise ValueError(
                    f"Unknown component '{component}'. Use 'x', 'y', 'z' or 0, 1, 2"
                )
            component = component_map[component.lower()]

        if not 0 <= component <= 2:
            raise ValueError(f"Component index must be 0, 1, or 2, got {component}")

        return self.mode_array[:, :, component]


class FMRModeAnalyzer:
    """
    Professional FMR mode analyzer with interactive visualization.

    Provides both programmatic access to mode data and interactive
    spectrum visualization for frequency selection.
    """

    def __init__(
        self,
        zarr_path: str,
        dataset_name: Optional[str] = None,
        config: Optional[ModeVisualizationConfig] = None,
        mode_character_config: Optional[ModeCharacteristicConfig] = None,
        debug: bool = False,
    ):
        """
        Initialize FMR mode analyzer.

        Parameters:
        -----------
        zarr_path : str
            Path to zarr file containing mode data
        dataset_name : str, optional
            Base dataset name (default: auto-select largest m dataset)
        config : ModeVisualizationConfig, optional
            Visualization configuration
        debug : bool, optional
            Enable debug logging
        """
        if not ZARR_AVAILABLE:
            raise ImportError("Zarr is required for mode analysis")

        # Auto-select largest m dataset if none specified
        if dataset_name is None:
            from ..core import find_largest_m_dataset

            dataset_name = find_largest_m_dataset(zarr_path)

        self.zarr_path = zarr_path
        self.dataset_name = dataset_name
        self.config = config or ModeVisualizationConfig()
        self._character_analyzer = ModeCharacterAnalyzer(mode_character_config)

        # Set up logging
        setup_mmpp_logging(debug=debug, logger_name="mmpp.fft.modes")
        if debug:
            log.debug("FMR mode analyzer debug logging enabled")

        # Load data
        self._load_data()

        # Interactive state
        self._current_frequency = None
        self._interactive_fig = None
        self._frequency_line = None
        self._mode_axes = None
        self._row_colorbars: list[Any] = []
        self._fwhm_artists: list[Any] = []
        self._last_fwhm = None

        # Animation state tracking
        self._mode_animations = {}  # Dict to track active animations per axis
        self._animated_axes = set()  # Set of axes currently being animated

        # Mode data cache (LRU cache with max 10 entries)
        self._mode_cache = {}
        self._cache_order = []
        self._max_cache_size = 10

    @property
    def modes_available(self) -> bool:
        """Check if mode data is available."""
        return (
            self.modes_path is not None
            and self.freqs_path is not None
            and self.spectrum_path is not None
        )

    @property
    def last_fwhm(self) -> Optional[PeakWidth]:
        """Return the most recently computed half-width at half-maximum."""

        return getattr(self, "_last_fwhm", None)

    def _list_available_datasets(self) -> list[str]:
        """Enumerate top-level datasets available in the zarr archive."""

        try:
            root = zarr.open(self.zarr_path, mode="r")
            keys = set(root.group_keys()) | set(root.array_keys())
            return sorted({key.split("/")[0] for key in keys})
        except Exception as exc:
            log.debug("Unable to list datasets in %s: %s", self.zarr_path, exc)
            return []

    def _get_zarr_paths(self) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Unified path resolution for zarr datasets.

        Returns:
        --------
        Tuple[str, str, str]
            (modes_path, freqs_path, spectrum_path) or None if not found
        """
        # Possible base paths for modes/frequencies - consistent order
        base_paths = [f"modes/{self.dataset_name}", f"tmodes/{self.dataset_name}"]

        modes_path = None
        freqs_path = None

        # Find first existing base path
        for base_path in base_paths:
            if (
                f"{base_path}/arr" in self.zarr_file
                and f"{base_path}/freqs" in self.zarr_file
            ):
                modes_path = f"{base_path}/arr"
                freqs_path = f"{base_path}/freqs"
                break

        # If not found together, try separately (for backward compatibility)
        if modes_path is None:
            for base_path in base_paths:
                if f"{base_path}/arr" in self.zarr_file:
                    modes_path = f"{base_path}/arr"
                    break

        if freqs_path is None:
            for base_path in base_paths:
                if f"{base_path}/freqs" in self.zarr_file:
                    freqs_path = f"{base_path}/freqs"
                    break

        # Find spectrum - try multiple locations for consistency with plot_spectrum
        spectrum_path = None
        spectrum_candidates = [
            # Standard FFT locations (consistent with plot_spectrum)
            f"fft/{self.dataset_name}_z-1_m1/spectrum",  # Most common case
            f"fft/{self.dataset_name}_z0_m1/spectrum",
            f"fft/{self.dataset_name}/spectrum",
            # Legacy locations (from compute_modes)
            f"fft/{self.dataset_name}/spec",
            f"fft/{self.dataset_name}/sum",
            # Try other z_layers and methods
            *[f"fft/{self.dataset_name}_z{z}_m1/spectrum" for z in range(-5, 10)],
        ]
        
        for path in spectrum_candidates:
            if path in self.zarr_file:
                spectrum_path = path
                log.debug(f"Found spectrum at: {spectrum_path}")
                break

        return modes_path, freqs_path, spectrum_path

    def _load_data(self) -> None:
        """Load mode and spectrum data from zarr file."""
        try:
            self.zarr_file = zarr.open(self.zarr_path, mode="r")
            log.info(f"Opened zarr file: {self.zarr_path}")
        except Exception as e:
            log.error(f"Failed to open zarr file {self.zarr_path}: {e}")
            raise

        # Use unified path resolution
        self.modes_path, self.freqs_path, self.spectrum_path = self._get_zarr_paths()

        if not self.modes_path:
            log.debug(
                f"No mode data found for dataset '{self.dataset_name}'. "
                "Modes will need to be computed."
            )

        if not self.freqs_path:
            log.debug(
                f"No frequency data found for dataset '{self.dataset_name}'. "
                "Frequencies will be computed with modes."
            )

        if not self.spectrum_path:
            log.warning(
                f"No spectrum data found for dataset '{self.dataset_name}'. "
                f"Expected paths: fft/{self.dataset_name}/spec or fft/{self.dataset_name}/sum"
            )

        # Load frequency array if available
        if self.freqs_path:
            self.frequencies = np.array(self.zarr_file[self.freqs_path])
            log.info(
                f"Loaded frequencies: {len(self.frequencies)} points, "
                f"range {self.frequencies[0]:.3f} - {self.frequencies[-1]:.3f} GHz"
            )
        else:
            self.frequencies = None
            log.debug("No frequency data loaded - will be computed with modes")

        # Load spectrum if available
        if self.spectrum_path:
            self.spectrum = np.array(self.zarr_file[self.spectrum_path])
            if self.spectrum.ndim > 1:
                # Take first component if multi-component
                self.spectrum = (
                    self.spectrum[:, 0]
                    if self.spectrum.shape[1] == 3
                    else np.sum(self.spectrum, axis=tuple(range(1, self.spectrum.ndim)))
                )
            log.info(f"Loaded spectrum data: shape {self.spectrum.shape}")
        else:
            # Try to load power_sum from computed modes as fallback spectrum
            modes_power_path = f"modes/{self.dataset_name}/power_sum"
            if modes_power_path in self.zarr_file:
                self.spectrum = np.array(self.zarr_file[modes_power_path])
                log.info(f"Using computed modes power_sum as spectrum: shape {self.spectrum.shape}")
            else:
                self.spectrum = None

        # Get spatial information
        self._get_spatial_info()

    def _get_spatial_info(self) -> None:
        """Extract spatial information from zarr metadata."""
        # Try to get spatial resolution from attributes
        self.dx = 1.0  # Default spatial resolution in nm
        self.dy = 1.0

        # Look for spatial attributes in various locations
        attrs_to_check = [
            self.zarr_file.attrs,
            (
                self.zarr_file[self.dataset_name].attrs
                if self.dataset_name in self.zarr_file
                else {}
            ),
        ]

        for attrs in attrs_to_check:
            if "dx" in attrs:
                self.dx = float(attrs["dx"]) * 1e9  # Convert to nm
            if "dy" in attrs:
                self.dy = float(attrs["dy"]) * 1e9  # Convert to nm

        log.debug(f"Spatial resolution: dx={self.dx:.3f} nm, dy={self.dy:.3f} nm")

    def _detect_peaks(
        self, spectrum: np.ndarray, frequencies: np.ndarray
    ) -> list[Peak]:
        """
        Detect peaks in spectrum.

        Parameters:
        -----------
        spectrum : np.ndarray
            Power spectrum data
        frequencies : np.ndarray
            Frequency array in GHz

        Returns:
        --------
        List[Peak]
            List of detected peaks
        """
        if not SCIPY_AVAILABLE:
            log.warning("SciPy not available, using simple peak detection")
            # Simple peak detection without scipy
            peaks = []
            for i in range(1, len(spectrum) - 1):
                if (
                    spectrum[i] > spectrum[i - 1]
                    and spectrum[i] > spectrum[i + 1]
                    and spectrum[i] > self.config.peak_threshold * np.max(spectrum)
                ):
                    peaks.append(
                        Peak(idx=i, freq=frequencies[i], amplitude=spectrum[i])
                    )
            return peaks

        try:
            # Normalize spectrum for peak detection
            norm_spectrum = spectrum / np.max(spectrum)

            # Find peaks using scipy
            peak_indices, properties = find_peaks(
                norm_spectrum,
                height=self.config.peak_threshold,
                distance=self.config.peak_min_distance,
            )

            # Create peak objects
            peaks = []
            for idx in peak_indices:
                peaks.append(
                    Peak(idx=int(idx), freq=frequencies[idx], amplitude=spectrum[idx])
                )

            # Sort by amplitude (descending)
            peaks.sort(key=lambda p: p.amplitude, reverse=True)

            log.debug(f"Detected {len(peaks)} peaks")
            return peaks

        except Exception as e:
            log.error(f"Peak detection failed: {e}")
            return []

    def get_mode(self, frequency: float, z_layer: int = 0) -> FMRModeData:
        """
        Get mode data at specified frequency.

        Parameters:
        -----------
        frequency : float
            Frequency in GHz
        z_layer : int, optional
            Z-layer index (default: 0)

        Returns:
        --------
        FMRModeData
            Mode data at specified frequency

        Raises:
        -------
        ValueError
            If frequency or z_layer is out of range
        RuntimeError
            If mode data is not available
        """
        if self.frequencies is None:
            raise RuntimeError(
                "No frequency data available. Run compute_modes() first."
            )

        if self.modes_path is None:
            raise RuntimeError("No mode data available. Run compute_modes() first.")

        # Find closest frequency index
        freq_idx = np.argmin(np.abs(self.frequencies - frequency))
        actual_freq = self.frequencies[freq_idx]

        if abs(actual_freq - frequency) > 0.1:
            log.warning(
                f"Requested frequency {frequency:.3f} GHz not found, "
                f"using closest: {actual_freq:.3f} GHz"
            )

        # Validate and normalize z_layer bounds
        mode_shape = self.zarr_file[self.modes_path].shape
        n_layers = mode_shape[1]
        
        # Handle negative indexing (like Python lists)
        if z_layer < 0:
            z_layer = n_layers + z_layer
            log.debug(f"Converted negative z_layer to {z_layer}")
            
        if z_layer < 0 or z_layer >= n_layers:
            raise ValueError(
                f"z_layer {z_layer} out of range. Available layers: 0-{n_layers - 1} (or negative: -{n_layers} to -1)"
            )

        # Load mode data for this frequency with bounds checking
        try:
            mode_data = self.zarr_file[self.modes_path][freq_idx, z_layer, :, :, :]
        except IndexError as e:
            raise ValueError(
                f"Invalid indices: freq_idx={freq_idx}, z_layer={z_layer}. {e}"
            )

        # Create spatial extent
        ny, nx = mode_data.shape[:2]
        extent = (0, nx * self.dx, 0, ny * self.dy)

        # Metadata
        metadata = {
            "frequency_index": freq_idx,
            "requested_frequency": frequency,
            "actual_frequency": actual_freq,
            "z_layer": z_layer,
            "spatial_resolution": (self.dx, self.dy),
            "mode_shape": mode_shape,
        }

        # Update cache
        self._update_cache(
            frequency, z_layer, FMRModeData(actual_freq, mode_data, extent, metadata)
        )

        return FMRModeData(actual_freq, mode_data, extent, metadata)

    def characterize_mode(
        self,
        frequency: float,
        z_layer: int = 0,
        *,
        core_position: Optional[tuple[float, float]] = None,
        analysis_radius: Optional[float] = None,
        config: Optional[ModeCharacteristicConfig] = None,
    ) -> ModeCharacterizationResult:
        """Classify the mode at ``frequency`` into gyration/breathing/azimuthal families."""

        analyzer = self._character_analyzer if config is None else ModeCharacterAnalyzer(config)
        mode_data = self.get_mode(frequency, z_layer)
        return analyzer.analyze(
            mode_data,
            core_position=core_position,
            analysis_radius=analysis_radius,
        )

    def _update_cache(
        self, frequency: float, z_layer: int, mode_data: FMRModeData
    ) -> None:
        """Update mode data cache."""
        key = (frequency, z_layer)
        if key in self._mode_cache:
            # Move to end to indicate recent use
            self._cache_order.remove(key)
        elif len(self._mode_cache) >= self._max_cache_size:
            # Remove least recently used item
            oldest_key = self._cache_order.pop(0)
            del self._mode_cache[oldest_key]

        # Update cache with new mode data
        self._mode_cache[key] = mode_data
        self._cache_order.append(key)

    def find_peaks(
        self,
        threshold: Optional[float] = None,
        min_distance: Optional[int] = None,
        component: int = 0,
        spectrum: Optional[np.ndarray] = None,
        frequencies: Optional[np.ndarray] = None,
    ) -> list[Peak]:
        """
        Find peaks in the spectrum.

        Parameters:
        -----------
        threshold : float, optional
            Peak detection threshold (default: from config)
        min_distance : int, optional
            Minimum distance between peaks (default: from config)
        component : int, optional
            Spectrum component to analyze (default: 0)
        spectrum : np.ndarray, optional
            Spectrum data to use (default: self.spectrum)
        frequencies : np.ndarray, optional
            Frequency data to use (default: self.frequencies)

        Returns:
        --------
        List[Peak]
            List of detected peaks
        """
        # Use provided spectrum/frequencies or fallback to instance data
        spectrum_data = spectrum if spectrum is not None else self.spectrum
        freq_data = frequencies if frequencies is not None else self.frequencies
        
        if spectrum_data is None:
            log.warning("No spectrum data available for peak detection")
            return []

        threshold = threshold or self.config.peak_threshold
        min_distance = min_distance or self.config.peak_min_distance

        # Normalize spectrum for peak detection
        spectrum_work = spectrum_data.copy()
        if self.config.spectrum_normalize:
            spectrum_work = spectrum_work / np.max(spectrum_work)

        # Filter frequency range
        freq_mask = (freq_data >= self.config.f_min) & (
            freq_data <= self.config.f_max
        )
        freqs_filtered = freq_data[freq_mask]
        spectrum_filtered = spectrum_work[freq_mask]

        # Detect peaks
        peaks = self._detect_peaks(spectrum_filtered, freqs_filtered)

        # Convert to Peak objects with proper index mapping
        peaks_converted = []
        for peak in peaks:
            # Safely map back to original index
            try:
                orig_indices = np.where(freq_mask)[0]
                if peak.idx < len(orig_indices):
                    orig_idx = orig_indices[peak.idx]
                    peaks_converted.append(Peak(orig_idx, peak.freq, peak.amplitude))
                else:
                    log.warning(
                        f"Peak index {peak.idx} out of range for filtered array"
                    )
            except IndexError as e:
                log.warning(f"Index mapping error for peak {peak.idx}: {e}")
                continue

        log.info(
            f"Found {len(peaks_converted)} peaks in frequency range "
            f"{self.config.f_min}-{self.config.f_max} GHz"
        )

        return peaks_converted

    def plot_modes(
        self,
        frequency: float,
        z_layer: int = 0,
        components: Optional[list[Union[int, str]]] = None,
        save_path: Optional[str] = None,
    ) -> tuple[Figure, np.ndarray]:
        """
        Plot mode visualization for a specific frequency.

        Parameters:
        -----------
        frequency : float
            Frequency in GHz
        z_layer : int, optional
            Z-layer index (default: 0)
        components : list, optional
            List of components to plot (default: ['x', 'y', 'z'])
        save_path : str, optional
            Path to save the figure

        Returns:
        --------
        Tuple[Figure, np.ndarray]
            Matplotlib figure and axes array
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for plotting")

        # Setup professional styling for mode plots
        setup_animation_styling(use_paper_style=True, use_custom_fonts=True)

        components = components or ["x", "y", "z"]
        mode_data = self.get_mode(frequency, z_layer)

        # Create figure with subplots
        n_components = len(components)
        n_rows = (
            3
            if self.config.show_magnitude
            and self.config.show_phase
            and self.config.show_combined
            else 2
        )

        fig, axes = plt.subplots(
            n_rows,
            n_components,
            figsize=(4 * n_components, 3 * n_rows),
            dpi=self.config.dpi,
        )

        if n_components == 1:
            axes = axes.reshape(-1, 1)
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        # Plot each component
        for i, comp in enumerate(components):
            comp_data = mode_data.get_component(comp)
            magnitude = np.abs(comp_data)
            phase = np.angle(comp_data)

            row = 0

            # Magnitude plot
            if self.config.show_magnitude:
                im1 = axes[row, i].imshow(
                    magnitude,
                    cmap=self.config._resolve_colormap(self.config.colormap_magnitude),
                    extent=mode_data.extent,
                    aspect="equal",
                    interpolation=self.config.interpolation,
                    origin="lower",
                )
                axes[row, i].set_title(f"|m_{comp}| @ {frequency:.3f} GHz")
                axes[row, i].set_xlabel("x (nm)")
                if i == 0:
                    axes[row, i].set_ylabel("y (nm)")
                plt.colorbar(im1, ax=axes[row, i], shrink=0.8)
                row += 1

            # Phase plot
            if self.config.show_phase:
                im2 = axes[row, i].imshow(
                    phase,
                    cmap=self.config._resolve_colormap(self.config.colormap_phase),
                    extent=mode_data.extent,
                    aspect="equal",
                    interpolation=self.config.interpolation,
                    vmin=-np.pi,
                    vmax=np.pi,
                    origin="lower",
                )
                axes[row, i].set_title(f"arg(m_{comp}) @ {frequency:.3f} GHz")
                axes[row, i].set_xlabel("x (nm)")
                if i == 0:
                    axes[row, i].set_ylabel("y (nm)")
                plt.colorbar(im2, ax=axes[row, i], shrink=0.8)
                row += 1

            # Combined plot (phase with magnitude as alpha)
            if self.config.show_combined:
                # Create combined visualization: magnitude * cos(phase) for real part
                combined_data = magnitude * np.cos(phase)  # Real part
                # Alternative: combined_data = magnitude * np.sin(phase)  # Imaginary part

                im3 = axes[row, i].imshow(
                    combined_data,
                    cmap=self.config._resolve_colormap(self.config.colormap_phase),
                    extent=mode_data.extent,
                    aspect="equal",
                    interpolation=self.config.interpolation,
                    origin="lower",
                )
                axes[row, i].set_title(
                    f"m_{comp} combined (mag×cos(φ)) @ {frequency:.3f} GHz"
                )
                axes[row, i].set_xlabel("x (nm)")
                if i == 0:
                    axes[row, i].set_ylabel("y (nm)")
                plt.colorbar(im3, ax=axes[row, i], shrink=0.8)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")
            log.info(f"Saved mode plot to {save_path}")

        return fig, axes

    def interactive_spectrum(
        self,
        components: Optional[list[Union[int, str]]] = None,
        z_layer: int = 0,
        method: int = 1,
        show: bool = True,
        force: bool = False,
        use_fft_spectrum: bool = True,
        saveanim: Union[bool, str, None] = None,
        **kwargs,
    ) -> Figure:
        """
        Create interactive spectrum plot with mode visualization.

        Click on spectrum to select frequency and visualize corresponding mode.
        Right-click to snap to nearest peak.
        Double-click on mode plots to toggle animations.
        Press 'c' key to characterize the current mode.
        Press 's' key to save animated view (if saveanim enabled).
        Press 'h' key for help.

        **Interactive Requirements:**
        - Jupyter/IPython: Function automatically tries to enable `%matplotlib widget`
        - Standalone Python: Requires interactive backend (Qt5Agg, TkAgg, etc.)
        - If interactivity doesn't work, manually run: `%matplotlib widget` (Jupyter) 
          or install ipympl: `pip install ipympl`

        Each mode panel now includes a publication-ready scale bar (auto-sized in nm)
        and shared colorbars for magnitude/phase/combined maps when the required
        matplotlib toolkit extensions are available.

        Parameters:
        -----------
        components : list, optional
            List of components to plot (default: ['x', 'y', 'z'])
        z_layer : int, optional
            Z-layer index (default: 0)
        method : int, optional
            Visualization method (default: 1)
            1 = Standard interactive plot
            2 = Alternative layout (if implemented)
        show : bool, optional
            Whether to automatically display the figure (default: True)
        force : bool, optional
            Force reload of data from zarr file (default: False)
        use_fft_spectrum : bool, optional
            Use spectrum from standard FFT analysis instead of modes data (default: True)
            This ensures consistency with plot_spectrum results
        saveanim : bool, str, or None, optional
            Enable animation saving functionality (default: None)
            - None or False: No animation saving
            - True: Enable saving with default naming ('mode_animation_%Y%m%d_%H%M%S.mp4')
            - str: Custom path/filename for animation (e.g., 'my_animation.mp4')
            Supported formats: .mp4, .gif, .avi (depends on matplotlib writers)
            Press 's' key in interactive mode to save current animated view
        \\*\\*kwargs : dict
            Additional keyword arguments:
            - figsize : tuple, optional
                Figure size (width, height) in inches (default: from config)
            - dpi : int, optional
                Figure resolution in dots per inch (default: from config)
            - cmap : str, optional
                Colormap for all mode visualizations (overrides config colormaps)
                Examples: 'viridis', 'inferno', 'plasma', 'cividis', 'balance'
            - acmap : str, optional
                Colormap specifically for amplitude/magnitude plots (overrides cmap for magnitude)
                Examples: 'viridis', 'inferno', 'plasma', 'hot'
            - pcmap : str, optional
                Colormap specifically for phase plots (overrides cmap for phase)
                Examples: 'hsv', 'twilight', 'twilight_shifted', 'phase'
            - peak_width / fwhh / fwhm / hwfh : bool or str, optional
                Annotate the dominant peak's half-width at half-maximum on the
                spectrum panel. Strings control the label text.

        Returns:
        --------
        Figure
            Interactive matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for interactive plotting")

        # Force reload data if requested
        if force:
            log.info(f"Force reloading data for interactive spectrum (dataset: {self.dataset_name})")
            self._load_data()

        peak_width_option = kwargs.pop("peak_width", None)
        for alias in ("fwhh", "fwhm", "hwfh"):
            alias_value = kwargs.pop(alias, None)
            if alias_value is not None:
                peak_width_option = alias_value

        show_peak_width, peak_width_label = normalize_peak_width_option(
            peak_width_option
        )

        # Setup animation saving
        self._saveanim_enabled = saveanim is not None and saveanim is not False
        if self._saveanim_enabled:
            if isinstance(saveanim, str):
                self._saveanim_path = saveanim
            else:
                # Default filename with timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._saveanim_path = f"mode_animation_{timestamp}.mp4"
            log.info(f"Animation saving enabled. Press 's' to save to: {self._saveanim_path}")
        else:
            self._saveanim_path = None

        # Clear previous FWHM artists if figure is being re-used
        for artist in getattr(self, "_fwhm_artists", []):
            try:
                artist.remove()
            except ValueError:
                pass
        self._fwhm_artists: list[Any] = []
        self._last_fwhm = None

        # Use FFT spectrum for consistency with plot_spectrum
        spectrum_to_use = self.spectrum
        frequencies_to_use = self.frequencies
        
        if use_fft_spectrum:
            try:
                # Try to load spectrum from standard FFT analysis
                fft_spectrum_path = f"fft/{self.dataset_name}_z{z_layer}_m{method}/spectrum"
                fft_freqs_path = f"fft/{self.dataset_name}_z{z_layer}_m{method}/frequencies"
                
                if fft_spectrum_path in self.zarr_file and fft_freqs_path in self.zarr_file:
                    spectrum_to_use = np.abs(np.array(self.zarr_file[fft_spectrum_path])) ** 2
                    frequencies_to_use = np.array(self.zarr_file[fft_freqs_path]) / 1e9  # Convert to GHz
                    log.info(f"Using FFT spectrum from {fft_spectrum_path} for consistency with plot_spectrum")
                else:
                    log.warning(f"FFT spectrum not found at {fft_spectrum_path}, using modes spectrum")
            except Exception as e:
                log.warning(f"Failed to load FFT spectrum: {e}, using modes spectrum")

        if spectrum_to_use is None:
            raise ValueError("No spectrum data available for interactive mode")

        # Apply paper style for consistent visualization
        if STYLING_AVAILABLE:
            try:
                load_paper_style(verbose=False)
                log.debug("Applied paper style to interactive spectrum")
            except Exception as e:
                log.warning(f"Could not apply paper style: {e}")

        # Handle method parameter
        if method not in [1, 2]:
            log.warning(f"Unknown method {method}, using default method 1")
            method = 1

        components = components or ["x", "y", "z"]
        n_components = len(components)

        # Extract parameters from kwargs
        figsize = kwargs.get("figsize", self.config.figsize)
        dpi = kwargs.get("dpi", self.config.dpi)
        cmap = kwargs.get("cmap", None)
        acmap = kwargs.get("acmap", None)  # Amplitude/magnitude colormap
        pcmap = kwargs.get("pcmap", None)  # Phase colormap

        # Update colormaps if provided
        if cmap or acmap or pcmap:
            # Create a temporary copy of config with updated colormaps
            import copy

            temp_config = copy.deepcopy(self.config)

            # If cmap is provided, use it for all types unless specifically overridden
            if cmap:
                temp_config.colormap_magnitude = cmap
                temp_config.colormap_phase = cmap

            # Override with specific colormaps if provided
            if acmap:
                temp_config.colormap_magnitude = acmap
            if pcmap:
                temp_config.colormap_phase = pcmap

            self.config = temp_config

        # Update figure settings from kwargs
        self.config.figsize = figsize
        self.config.dpi = dpi

        # Validate number of components for layout
        if n_components > 5:
            raise ValueError(
                f"Too many components ({n_components}). Maximum supported: 5"
            )

        # Create figure with custom layout
        # Automatically configure interactive backend for Jupyter
        try:
            import matplotlib
            
            current_backend = matplotlib.get_backend()
            
            # Check if we're in Jupyter/IPython environment
            try:
                from IPython import get_ipython
                ipython = get_ipython()
                in_jupyter = ipython is not None and hasattr(ipython, 'kernel')
            except ImportError:
                in_jupyter = False
            
            # Auto-configure interactive backend for Jupyter
            if in_jupyter:
                if (
                    "ipympl" not in current_backend.lower() 
                    and "widget" not in current_backend.lower()
                    and "nbagg" not in current_backend.lower()
                ):
                    log.warning(
                        f"Current backend '{current_backend}' may not support full interactivity in Jupyter. "
                        "For best experience, run '%matplotlib widget' or install ipympl: pip install ipympl"
                    )
                    
                    # Try to automatically switch to widget backend
                    try:
                        ipython.run_line_magic('matplotlib', 'widget')
                        log.info("Automatically switched to matplotlib widget backend for interactivity")
                    except Exception as e:
                        log.debug(f"Could not auto-switch to widget backend: {e}")
                        # Try nbagg as fallback
                        try:
                            ipython.run_line_magic('matplotlib', 'nbagg')  
                            log.info("Switched to nbagg backend as fallback")
                        except Exception:
                            log.warning(
                                "Please run '%matplotlib widget' manually for full interactivity. "
                                "Install ipympl if needed: pip install ipympl"
                            )
                else:
                    log.info(f"Using Jupyter-compatible backend: {current_backend}")
            else:
                # Not in Jupyter - check for standalone interactive backends
                interactive_backends = ['qt5agg', 'tkagg', 'gtk3agg', 'wxagg', 'macosx']
                current_lower = current_backend.lower()
                if current_lower not in interactive_backends:
                    log.info(
                        f"Current backend: {current_backend}. Interactive features may be limited. "
                        f"Consider switching to: Qt5Agg, TkAgg, GTK3Agg, wxAgg"
                    )
                else:
                    log.info(f"Using interactive backend: {current_backend}")
                    
        except Exception as e:
            log.warning(f"Could not configure interactive backend: {e}")

        self._interactive_fig = plt.figure(figsize=figsize, dpi=dpi)

        # Create grid layout: spectrum on left, modes on right
        # Use dynamic number of rows (3 for all visualization types)
        n_vis_types = sum(
            [
                self.config.show_magnitude,
                self.config.show_phase,
                self.config.show_combined,
            ]
        )
        if n_vis_types == 0:
            raise ValueError("At least one visualization type must be enabled")

        gs = gridspec.GridSpec(
            n_vis_types,
            n_components + 1,
            width_ratios=[self.config.spectrum_width_ratio]
            + [self.config.modes_width_ratio / n_components] * n_components,
            height_ratios=[1] * n_vis_types,
        )

        # Spectrum plot spans all rows in first column
        ax_spectrum = self._interactive_fig.add_subplot(gs[:, 0])

        # Mode plots in remaining columns - dynamic based on enabled visualizations
        self._mode_axes = np.array(
            [
                [
                    self._interactive_fig.add_subplot(gs[row, col + 1])
                    for col in range(n_components)
                ]
                for row in range(n_vis_types)
            ]
        )

        # Plot spectrum
        freq_mask = (frequencies_to_use >= self.config.f_min) & (
            frequencies_to_use <= self.config.f_max
        )
        freqs_plot = frequencies_to_use[freq_mask]
        spectrum_segment = spectrum_to_use[freq_mask]
        spectrum_plot = spectrum_segment.copy()

        if self.config.spectrum_normalize and spectrum_plot.size:
            max_val = np.max(spectrum_plot)
            if max_val > 0:
                spectrum_plot = spectrum_plot / max_val

        if self.config.spectrum_log_scale:
            spectrum_plot = np.log10(spectrum_plot + 1e-10)
            ax_spectrum.set_ylabel("log₁₀(Power Spectrum)")
        else:
            ax_spectrum.set_ylabel("Power Spectrum")

        ax_spectrum.plot(freqs_plot, spectrum_plot, "b-", linewidth=1.5)
        ax_spectrum.set_xlabel("Frequency (GHz)")
        ax_spectrum.set_title("FMR Spectrum (Click to select frequency)")
        ax_spectrum.grid(True, alpha=0.3)

        # Find and mark peaks using the same spectrum data
        peaks = self.find_peaks(spectrum=spectrum_to_use, frequencies=frequencies_to_use)
        for peak in peaks:
            if self.config.f_min <= peak.freq <= self.config.f_max:
                y_val = spectrum_plot[np.argmin(np.abs(freqs_plot - peak.freq))]
                ax_spectrum.plot(peak.freq, y_val, "ro", markersize=4)
                ax_spectrum.text(
                    peak.freq,
                    y_val + 0.05 * np.max(spectrum_plot),
                    f"{peak.freq:.2f}",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        if show_peak_width and freqs_plot.size and spectrum_segment.size:
            width_info = compute_half_width_at_half_max(freqs_plot, spectrum_segment)
            if width_info is None:
                log.debug("FWHM annotation skipped: could not determine half-width")
            else:
                scale_factor = width_info.peak_value if self.config.spectrum_normalize else 1.0
                if scale_factor <= 0:
                    log.debug("FWHM annotation skipped: invalid scale factor")
                else:
                    half_level_plot = width_info.half_level / scale_factor
                    if self.config.spectrum_log_scale:
                        if half_level_plot > 0:
                            half_level_plot = np.log10(half_level_plot + 1e-10)
                            delta = 0.05
                            ymin = half_level_plot - delta
                            ymax = half_level_plot + delta
                        else:
                            half_level_plot = None
                    else:
                        delta = max(abs(half_level_plot) * 0.05, 0.02)
                        ymin = half_level_plot - delta
                        ymax = half_level_plot + delta

                    if half_level_plot is not None:
                        color = "tab:orange"
                        h_line = ax_spectrum.hlines(
                            half_level_plot,
                            width_info.left_frequency,
                            width_info.right_frequency,
                            colors=color,
                            linewidth=1.5,
                            linestyles="-",
                            alpha=0.9,
                            zorder=5,
                        )
                        self._fwhm_artists.append(h_line)

                        v_lines = ax_spectrum.vlines(
                            [width_info.left_frequency, width_info.right_frequency],
                            ymin=ymin,
                            ymax=ymax,
                            colors=color,
                            linewidth=1.2,
                            alpha=0.9,
                            zorder=5,
                        )
                        self._fwhm_artists.append(v_lines)

                        text = ax_spectrum.annotate(
                            f"{peak_width_label}: {format_width_value(width_info.width)}",
                            xy=(
                                (width_info.left_frequency + width_info.right_frequency) / 2.0,
                                half_level_plot,
                            ),
                            xytext=(0, 10),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            color=color,
                            bbox={
                                "boxstyle": "round,pad=0.2",
                                "facecolor": "white",
                                "edgecolor": color,
                                "linewidth": 0.8,
                                "alpha": 0.85,
                            },
                            zorder=6,
                        )
                        self._fwhm_artists.append(text)
                        self._last_fwhm = width_info
                    else:
                        log.debug("FWHM annotation skipped: non-positive half level for log axis")

        # Initial frequency line
        init_freq = peaks[0].freq if peaks else freqs_plot[len(freqs_plot) // 2]
        self._frequency_line = ax_spectrum.axvline(
            init_freq, color="red", linestyle="--", linewidth=2, alpha=0.8
        )

        # Plot initial mode
        self._current_frequency = init_freq
        self._update_mode_plots(components, z_layer)

        # Set up click handler with proper cleanup
        def on_click(event):
            # Handle spectrum clicks (single click only)
            if event.inaxes == ax_spectrum and event.xdata is not None:
                if event.button == 3:  # Right click - snap to peak
                    if peaks:
                        peak_freqs = [p.freq for p in peaks]
                        closest_peak_freq = peak_freqs[
                            np.argmin(np.abs(np.array(peak_freqs) - event.xdata))
                        ]
                        selected_freq = closest_peak_freq
                    else:
                        selected_freq = event.xdata
                else:  # Left click - exact frequency
                    selected_freq = event.xdata

                # Update frequency line and mode plots
                self._frequency_line.set_xdata([selected_freq, selected_freq])
                self._current_frequency = selected_freq
                self._update_mode_plots(components, z_layer)
                self._interactive_fig.canvas.draw()
            
            # Handle double-clicks on mode plots for animation
            elif event.dblclick and event.inaxes is not None:
                # Find which mode axis was double-clicked
                for row_idx, ax_row in enumerate(self._mode_axes):
                    for col_idx, ax in enumerate(ax_row):
                        if event.inaxes == ax:
                            component = components[col_idx]
                            self._toggle_mode_animation(ax, row_idx, col_idx, component, z_layer)
                            return

        # Store event connection for cleanup
        self._click_connection = self._interactive_fig.canvas.mpl_connect(
            "button_press_event", on_click
        )
        log.debug(f"Click handler connected with ID: {self._click_connection}")

        # Add keyboard handler for mode characterization
        def on_key_press(event):
            """Handle keyboard events for mode characterization"""
            if event.key == 'c' and self._current_frequency is not None:
                try:
                    log.info(f"Characterizing mode at {self._current_frequency:.3f} GHz...")
                    characterization = self.characterize_mode(self._current_frequency, z_layer)
                    
                    # Display characterization results
                    char_info = (
                        f"Mode Classification at {self._current_frequency:.3f} GHz:\n"
                        f"• Primary Class: {characterization.primary_class}\n"
                        f"• m-index: {characterization.m_index}\n"
                        f"• Rotation: {characterization.rotation_sense or 'N/A'}\n"
                        f"• Radial nodes: {characterization.radial_nodes}\n"
                        f"• Confidence: {characterization.confidence:.2f}\n"
                        f"• Labels: {', '.join(characterization.labels)}"
                    )
                    
                    print("\n" + "="*60)
                    print(char_info)
                    print("="*60)
                    
                    # Update figure title with classification
                    main_title = f"Interactive Mode Spectrum - {characterization.primary_class.upper()} mode"
                    if characterization.m_index is not None:
                        main_title += f" (m={characterization.m_index})"
                    if characterization.rotation_sense:
                        main_title += f" [{characterization.rotation_sense}]"
                    
                    self._interactive_fig.suptitle(main_title, fontsize=12, fontweight='bold')
                    self._interactive_fig.canvas.draw()
                    
                    log.info(f"Mode classified as: {characterization.primary_class} (confidence: {characterization.confidence:.2f})")
                    
                except Exception as e:
                    log.error(f"Failed to characterize mode: {e}")
                    print(f"Error characterizing mode: {e}")
                    
            elif event.key == 's' and self._saveanim_enabled:
                # Save animation of current animated modes
                try:
                    if not self._mode_animations:
                        print("❌ No active animations to save! Double-click mode plots first.")
                        return
                        
                    log.info(f"Saving animation to: {self._saveanim_path}")
                    print(f"💾 Saving animation with {len(self._mode_animations)} animated modes...")
                    
                    self._save_animated_view(self._saveanim_path)
                    print(f"✅ Animation saved to: {self._saveanim_path}")
                    
                except Exception as e:
                    log.error(f"Failed to save animation: {e}")
                    print(f"❌ Error saving animation: {e}")
            
            elif event.key == 'h':
                # Show help
                help_controls = [
                    "• Click spectrum: Select frequency",
                    "• Right-click spectrum: Snap to nearest peak", 
                    "• Double-click mode: Toggle animation",
                    "• 'c' key: Characterize current mode"
                ]
                
                if self._saveanim_enabled:
                    help_controls.append("• 's' key: Save animated view")
                    
                help_controls.append("• 'h' key: Show this help")
                
                help_text = f"""
Interactive Spectrum Controls:
============================
{chr(10).join(help_controls)}
                """
                print(help_text)

        # Connect keyboard handler
        self._key_connection = self._interactive_fig.canvas.mpl_connect(
            "key_press_event", on_key_press
        )
        log.debug(f"Keyboard handler connected with ID: {self._key_connection}")

        # Add cleanup method to figure
        def cleanup():
            # Stop all running animations first
            if hasattr(self, '_mode_animations') and self._mode_animations:
                log.debug(f"Stopping {len(self._mode_animations)} running animations...")
                for animation in self._mode_animations.values():
                    try:
                        animation.event_source.stop()
                    except AttributeError:
                        pass  # Animation might not have been started yet
                self._mode_animations.clear()
                if hasattr(self, '_animated_axes'):
                    self._animated_axes.clear()
            
            # Disconnect event handlers
            if hasattr(self, "_click_connection") and self._click_connection:
                self._interactive_fig.canvas.mpl_disconnect(self._click_connection)
                self._click_connection = None
                log.debug("Click handler disconnected")
                
            if hasattr(self, "_key_connection") and self._key_connection:
                self._interactive_fig.canvas.mpl_disconnect(self._key_connection)
                self._key_connection = None
                log.debug("Keyboard handler disconnected")
                
            log.debug("Interactive plot event handlers cleaned up")

        # Store cleanup function for later use
        self._interactive_fig._mmpp_cleanup = cleanup

        plt.tight_layout()
        
        # Update log message based on animation saving capability
        log_message = (
            "Interactive spectrum plot created. Click to select frequency, right-click to snap to peaks. "
            "Double-click mode plots for animations. Press 'c' to characterize current mode"
        )
        if self._saveanim_enabled:
            log_message += ", 's' to save animated view"
        log_message += ", 'h' for help."
        
        log.info(log_message)

        # Control figure display to avoid double showing
        if show:
            plt.show()
            return None  # Don't return figure to avoid Jupyter auto-display
        else:
            return self._interactive_fig

    def _update_mode_plots(
        self, components: list[Union[int, str]], z_layer: int
    ) -> None:
        """Update mode plots for current frequency."""
        if self._mode_axes is None or self._current_frequency is None:
            return

        # Clear previous shared colorbars
        for cbar in getattr(self, "_row_colorbars", []):
            try:
                cbar.remove()
            except ValueError:
                pass
        self._row_colorbars = []

        vis_types = []
        if self.config.show_magnitude:
            vis_types.append("magnitude")
        if self.config.show_phase:
            vis_types.append("phase")
        if self.config.show_combined:
            vis_types.append("combined")

        images_for_colorbar: list[Optional[Any]] = [None] * len(vis_types)

        # Clear all axes
        for ax_row in self._mode_axes:
            for ax in ax_row:
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])

        # Get mode data
        try:
            mode_data = self.get_mode(self._current_frequency, z_layer)
        except Exception as e:
            log.error(f"Failed to get mode data: {e}")
            # Show error message on plots instead of leaving them empty
            for ax_row in self._mode_axes:
                for ax in ax_row:
                    ax.text(
                        0.5,
                        0.5,
                        f"Error loading mode data:\n{str(e)}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=10,
                        color="red",
                        wrap=True,
                    )
            return

        # Plot each component
        for i, comp in enumerate(components):
            try:
                comp_data = mode_data.get_component(comp)
                magnitude = np.abs(comp_data)
                phase = np.angle(comp_data)

                row_idx = 0

                # Magnitude plot (if enabled)
                if self.config.show_magnitude:
                    img = self._mode_axes[row_idx, i].imshow(
                        magnitude,
                        cmap=self.config._resolve_colormap(
                            self.config.colormap_magnitude
                        ),
                        extent=mode_data.extent,
                        aspect="equal",
                        interpolation=self.config.interpolation,
                        origin="lower",
                    )
                    self._mode_axes[row_idx, i].set_title(f"|m_{comp}|")
                    if images_for_colorbar[row_idx] is None:
                        images_for_colorbar[row_idx] = img
                    if i == 0:
                        self._add_scale_bar(self._mode_axes[row_idx, i], mode_data.extent)
                    row_idx += 1

                # Phase plot (if enabled)
                if self.config.show_phase:
                    img = self._mode_axes[row_idx, i].imshow(
                        phase,
                        cmap=self.config._resolve_colormap(self.config.colormap_phase),
                        extent=mode_data.extent,
                        aspect="equal",
                        interpolation=self.config.interpolation,
                        vmin=-np.pi,
                        vmax=np.pi,
                        origin="lower",
                    )
                    self._mode_axes[row_idx, i].set_title(f"arg(m_{comp})")
                    if images_for_colorbar[row_idx] is None:
                        images_for_colorbar[row_idx] = img
                    if i == 0:
                        self._add_scale_bar(self._mode_axes[row_idx, i], mode_data.extent)
                    row_idx += 1

                # Combined plot (if enabled)
                if self.config.show_combined:
                    # Create combined visualization: magnitude * cos(phase) for real part
                    # or magnitude * sin(phase) for imaginary part
                    # This shows the actual complex amplitude with sign
                    combined_data = magnitude * np.cos(phase)  # Real part
                    # Alternative: combined_data = magnitude * np.sin(phase)  # Imaginary part

                    img = self._mode_axes[row_idx, i].imshow(
                        combined_data,
                        cmap=self.config._resolve_colormap(self.config.colormap_phase),
                        extent=mode_data.extent,
                        aspect="equal",
                        interpolation=self.config.interpolation,
                        origin="lower",
                    )
                    self._mode_axes[row_idx, i].set_title(
                        f"m_{comp} (mag×cos(φ))"
                    )
                    if images_for_colorbar[row_idx] is None:
                        images_for_colorbar[row_idx] = img
                    if i == 0:
                        self._add_scale_bar(self._mode_axes[row_idx, i], mode_data.extent)

            except Exception as e:
                log.error(f"Failed to plot component {comp}: {e}")
                continue

        # Add frequency info
        self._interactive_fig.suptitle(
            f"FMR Modes at {self._current_frequency:.3f} GHz",
            fontsize=14,
            fontweight="bold",
        )

        # Create shared colorbars per visualization type
        for row_idx, (vis_type, img) in enumerate(zip(vis_types, images_for_colorbar)):
            if img is None:
                continue
            try:
                # Get the rightmost axis in this row for colorbar placement
                rightmost_ax = self._mode_axes[row_idx, -1]
                
                if AXES_GRID_AVAILABLE:
                    from mpl_toolkits.axes_grid1 import make_axes_locatable
                    # Use make_axes_locatable for proper positioning
                    divider = make_axes_locatable(rightmost_ax)
                    cax = divider.append_axes("right", 
                                             size=f"{self.config.colorbar_fraction*100}%",
                                             pad=self.config.colorbar_pad)
                    cbar = self._interactive_fig.colorbar(img, cax=cax)
                else:
                    # Fallback to basic colorbar positioned at rightmost axis
                    cbar = self._interactive_fig.colorbar(
                        img,
                        ax=rightmost_ax,
                        fraction=self.config.colorbar_fraction,
                        pad=self.config.colorbar_pad,
                    )
            except Exception as exc:
                log.debug(f"Skipping colorbar for {vis_type}: {exc}")
                continue

            label = self.config.colorbar_labels.get(vis_type, vis_type.title())
            cbar.set_label(label, fontsize=self.config.colorbar_label_size)
            cbar.ax.tick_params(labelsize=self.config.colorbar_ticklabel_size)
            self._row_colorbars.append(cbar)

    def _toggle_mode_animation(
        self, ax: Any, row_idx: int, col_idx: int, component: Union[str, int], z_layer: int
    ) -> None:
        """Toggle between static mode plot and in-place animation."""
        if not ANIMATION_AVAILABLE:
            log.warning("Animation not available - matplotlib.animation required")
            return
            
        # Initialize animation tracking if needed
        if not hasattr(self, '_mode_animations'):
            self._mode_animations: dict[tuple[int, int], Any] = {}
            self._animated_axes: set[tuple[int, int]] = set()
        
        axis_key = (row_idx, col_idx)
        
        # Check if this axis is currently animated
        if axis_key in self._animated_axes:
            # Stop animation and revert to static
            self._stop_mode_animation(axis_key)
            # Redraw static mode
            self._update_single_mode_plot(ax, row_idx, col_idx, component, z_layer)
            self._interactive_fig.canvas.draw()
            log.info(f"Stopped animation for m_{component} (row {row_idx}, col {col_idx})")
        else:
            # Start animation
            self._start_mode_animation(ax, row_idx, col_idx, component, z_layer)
            log.info(f"Started animation for m_{component} (row {row_idx}, col {col_idx})")

    def _stop_mode_animation(self, axis_key: tuple[int, int]) -> None:
        """Stop animation for specific axis."""
        if axis_key in self._mode_animations:
            anim = self._mode_animations[axis_key]
            try:
                anim.event_source.stop()
                del self._mode_animations[axis_key]
            except Exception as e:
                log.debug(f"Error stopping animation: {e}")
        
        self._animated_axes.discard(axis_key)

def _create_ffmpeg_writer(ffmpeg_path: str, fps: int = 20, bitrate: int = 1800):
    """
    Create FFMpegWriter with compatibility across matplotlib versions.
    
    Parameters:
    -----------
    ffmpeg_path : str
        Path to ffmpeg executable
    fps : int, default 20
        Frames per second for animation
    bitrate : int, default 1800
        Bitrate for video encoding
        
    Returns:
    --------
    FFMpegWriter
        Configured writer instance
    """
    from matplotlib.animation import FFMpegWriter
    import inspect
    
    try:
        # Get the signature of FFMpegWriter.__init__ to check supported parameters
        init_signature = inspect.signature(FFMpegWriter.__init__)
        supported_params = set(init_signature.parameters.keys())
        
        # Build kwargs dict with only supported parameters
        kwargs = {}
        
        # Common parameters
        if 'fps' in supported_params:
            kwargs['fps'] = fps
        if 'bitrate' in supported_params:
            kwargs['bitrate'] = bitrate
            
        # Additional parameters that might be supported
        if 'codec' in supported_params:
            kwargs['codec'] = 'libx264'
        if 'extra_args' in supported_params:
            kwargs['extra_args'] = ['-pix_fmt', 'yuv420p']
            
        # Create writer with supported parameters
        writer = FFMpegWriter(**kwargs)
        
        # Set ffmpeg path - handle different matplotlib versions
        if hasattr(writer, '_args'):
            # In newer versions, modify the command args directly
            if hasattr(writer, '_args') and isinstance(writer._args, dict):
                writer._args['executable'] = ffmpeg_path
        
        # Also try setting via other methods for compatibility
        try:
            # Some versions use this approach
            writer.bin_path = lambda: ffmpeg_path
        except:
            pass
            
        return writer
        
    except Exception as e:
        log.warning(f"Failed to create FFMpegWriter with advanced options: {e}")
        
        # Fallback to minimal initialization
        try:
            writer = FFMpegWriter(fps=fps)
            
            # Try to set the path via different methods
            try:
                writer.bin_path = lambda: ffmpeg_path
            except:
                pass
                
            if hasattr(writer, '_args'):
                if hasattr(writer, '_args') and isinstance(writer._args, dict):
                    writer._args['executable'] = ffmpeg_path
                    
            return writer
        except Exception as e2:
            log.error(f"Failed to create basic FFMpegWriter: {e2}")
            raise


    def _save_animated_view(self, save_path: str) -> None:
        """Save current animated view to video file."""
        if not self._mode_animations:
            raise ValueError("No active animations to save")
            
        # Import required modules
        try:
            from matplotlib.animation import FuncAnimation, PillowWriter
        except ImportError:
            raise ImportError("Animation saving requires matplotlib.animation")
            
        log.info(f"Creating animation with {len(self._mode_animations)} animated modes")
        
        # Determine writer based on file extension
        file_ext = save_path.lower().split('.')[-1]
        
        if file_ext == 'mp4':
            # Ensure FFmpeg is available
            ffmpeg_path = _ensure_ffmpeg_available()
            if ffmpeg_path:
                try:
                    writer = _create_ffmpeg_writer(ffmpeg_path, fps=20, bitrate=1800)
                    writer_name = 'ffmpeg'
                except Exception as e:
                    log.warning(f"FFMpeg initialization failed: {e}, falling back to Pillow")
                    writer = PillowWriter(fps=10)
                    writer_name = 'pillow'
                    # Change extension to gif for Pillow
                    save_path = save_path.replace('.mp4', '.gif')
            else:
                log.warning("FFMpeg not available, falling back to Pillow")
                writer = PillowWriter(fps=10)
                writer_name = 'pillow'
                # Change extension to gif for Pillow
                save_path = save_path.replace('.mp4', '.gif')
                
        elif file_ext == 'gif':
            writer = PillowWriter(fps=10)
            writer_name = 'pillow'
        elif file_ext == 'avi':
            # Ensure FFmpeg is available
            ffmpeg_path = _ensure_ffmpeg_available()
            if ffmpeg_path:
                try:
                    writer = _create_ffmpeg_writer(ffmpeg_path, fps=20, bitrate=1800)
                    writer_name = 'ffmpeg'
                except Exception as e:
                    log.warning(f"FFMpeg initialization failed: {e}, falling back to GIF")
                    writer = PillowWriter(fps=10)
                    writer_name = 'pillow'
                    save_path = save_path.replace('.avi', '.gif')
            else:
                log.warning("FFMpeg not available, falling back to GIF")
                writer = PillowWriter(fps=10)
                writer_name = 'pillow'
                save_path = save_path.replace('.avi', '.gif')
        else:
            # Default to gif with Pillow
            writer = PillowWriter(fps=10)
            writer_name = 'pillow'
            save_path = save_path.replace(f'.{file_ext}', '.gif')
            
        # Create master animation that coordinates all mode animations
        def animate_all_modes(frame):
            for anim in self._mode_animations.values():
                anim(frame)
            return []  # Return empty list for blitting
            
        total_frames = max(len(anim_data['frames']) for anim_data in self._mode_animations.values())
        
        # Create animation object
        anim = FuncAnimation(
            self._interactive_fig, 
            animate_all_modes, 
            frames=total_frames,
            interval=50,  # 50ms between frames
            blit=False,   # Disable blitting for complex plots
            repeat=True
        )
        
        log.info(f"Saving {total_frames} frames using {writer_name} writer...")
        
        try:
            anim.save(save_path, writer=writer, dpi=150)
            log.info("✅ Animation saved successfully!")
        except Exception as e:
            log.error(f"Failed to save animation: {e}")
            # Try to save as static frames as fallback
            base_name = save_path.rsplit('.', 1)[0]
            for i in range(min(100, total_frames)):  # Limit to 100 frames
                animate_all_modes(i)
                self._interactive_fig.canvas.draw()
                static_path = f"{base_name}_frame_{i:03d}.png"
                self._interactive_fig.savefig(static_path, dpi=150, bbox_inches='tight')
                
            log.info(f"Saved static frames to {base_name}_frame_*.png")
            raise RuntimeError(f"Could not save as {file_ext}, saved static frames instead")

    def _start_mode_animation(
        self, ax: Any, row_idx: int, col_idx: int, component: Union[str, int], z_layer: int
    ) -> None:
        """Start in-place animation for specific mode axis."""
        from matplotlib.animation import FuncAnimation
        
        try:
            # Get mode data
            mode_data = self.get_mode(self._current_frequency, z_layer)
            comp_data = mode_data.get_component(component)
            
            # Determine visualization type based on row
            vis_types = []
            if self.config.show_magnitude:
                vis_types.append("magnitude")
            if self.config.show_phase:
                vis_types.append("phase") 
            if self.config.show_combined:
                vis_types.append("combined")
            
            if row_idx >= len(vis_types):
                log.error(f"Invalid row index {row_idx} for visualization types")
                return
                
            vis_type = vis_types[row_idx]
            
            # Clear the axis
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Setup animation data based on visualization type
            if vis_type == "magnitude":
                # Animate magnitude (static - just pulsing intensity)
                amplitude = np.abs(comp_data)
                time_steps = np.linspace(0, 2*np.pi, 30)  # 30 frames for smooth animation
                
                # Create initial plot
                im = ax.imshow(
                    amplitude,
                    cmap=self.config._resolve_colormap(self.config.colormap_magnitude),
                    extent=mode_data.extent,
                    aspect="equal",
                    interpolation=self.config.interpolation,
                    origin="lower",
                )
                ax.set_title(f"|m_{component}| (animated)")
                
                def animate_magnitude(frame):
                    # Gentle pulsing effect for magnitude
                    pulse = 0.8 + 0.2 * np.sin(time_steps[frame])
                    im.set_array(amplitude * pulse)
                    return [im]
                    
                # Create animation
                anim = FuncAnimation(
                    self._interactive_fig,
                    animate_magnitude,
                    frames=len(time_steps),
                    interval=100,  # 10 FPS
                    blit=True,
                    repeat=True
                )
                
            elif vis_type == "phase":
                # Animate phase rotation
                amplitude = np.abs(comp_data)
                phase = np.angle(comp_data)
                time_steps = np.linspace(0, 2*np.pi, 30)
                
                # Create initial plot
                current_phase = (phase + time_steps[0]) % (2 * np.pi)
                im = ax.imshow(
                    current_phase,
                    cmap=self.config._resolve_colormap(self.config.colormap_phase),
                    extent=mode_data.extent,
                    aspect="equal",
                    interpolation=self.config.interpolation,
                    vmin=-np.pi,
                    vmax=np.pi,
                    origin="lower",
                )
                ax.set_title(f"arg(m_{component}) (animated)")
                
                def animate_phase(frame):
                    # Rotating phase
                    current_phase = (phase + time_steps[frame]) % (2 * np.pi)
                    # Convert to -π to π range for better visualization
                    current_phase = np.where(current_phase > np.pi, current_phase - 2*np.pi, current_phase)
                    im.set_array(current_phase)
                    return [im]
                    
                # Create animation
                anim = FuncAnimation(
                    self._interactive_fig,
                    animate_phase,
                    frames=len(time_steps),
                    interval=100,  # 10 FPS
                    blit=True,
                    repeat=True
                )
                
            elif vis_type == "combined":
                # Animate real part oscillation (true temporal dynamics)
                amplitude = np.abs(comp_data)
                phase = np.angle(comp_data)
                time_steps = np.linspace(0, 2*np.pi, 30)
                
                # Setup symmetric normalization for oscillating data
                vmax = np.max(amplitude)
                
                # Create initial plot
                real_part = amplitude * np.cos(phase + time_steps[0])
                im = ax.imshow(
                    real_part,
                    cmap=self.config._resolve_colormap(self.config.colormap_phase),
                    extent=mode_data.extent,
                    aspect="equal", 
                    interpolation=self.config.interpolation,
                    vmin=-vmax,
                    vmax=vmax,
                    origin="lower",
                )
                ax.set_title(f"Re[m_{component}] (temporal)")
                
                def animate_combined(frame):
                    # True temporal oscillation: Re[A * e^(i*φ) * e^(i*ω*t)]
                    t = time_steps[frame]
                    real_part = amplitude * np.cos(phase + t)
                    im.set_array(real_part)
                    return [im]
                    
                # Create animation
                anim = FuncAnimation(
                    self._interactive_fig,
                    animate_combined,
                    frames=len(time_steps),
                    interval=100,  # 10 FPS
                    blit=True,
                    repeat=True
                )
            
            # Store animation and mark axis as animated
            axis_key = (row_idx, col_idx)
            self._mode_animations[axis_key] = anim
            self._animated_axes.add(axis_key)
            
        except Exception as e:
            log.error(f"Failed to start mode animation: {e}")

    def _update_single_mode_plot(
        self, ax: Any, row_idx: int, col_idx: int, component: Union[str, int], z_layer: int
    ) -> None:
        """Update single mode plot (used when stopping animation)."""
        try:
            # Get mode data
            mode_data = self.get_mode(self._current_frequency, z_layer)
            comp_data = mode_data.get_component(component)
            
            # Determine visualization type
            vis_types = []
            if self.config.show_magnitude:
                vis_types.append("magnitude")
            if self.config.show_phase:
                vis_types.append("phase")
            if self.config.show_combined:
                vis_types.append("combined")
                
            vis_type = vis_types[row_idx]
            
            # Clear and redraw
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            
            if vis_type == "magnitude":
                magnitude = np.abs(comp_data)
                ax.imshow(
                    magnitude,
                    cmap=self.config._resolve_colormap(self.config.colormap_magnitude),
                    extent=mode_data.extent,
                    aspect="equal",
                    interpolation=self.config.interpolation,
                    origin="lower",
                )
                ax.set_title(f"|m_{component}|")
                
            elif vis_type == "phase":
                phase = np.angle(comp_data)
                ax.imshow(
                    phase,
                    cmap=self.config._resolve_colormap(self.config.colormap_phase),
                    extent=mode_data.extent,
                    aspect="equal",
                    interpolation=self.config.interpolation,
                    vmin=-np.pi,
                    vmax=np.pi,
                    origin="lower",
                )
                ax.set_title(f"arg(m_{component})")
                
            elif vis_type == "combined":
                magnitude = np.abs(comp_data)
                phase = np.angle(comp_data)
                combined_data = magnitude * np.cos(phase)
                ax.imshow(
                    combined_data,
                    cmap=self.config._resolve_colormap(self.config.colormap_phase),
                    extent=mode_data.extent,
                    aspect="equal",
                    interpolation=self.config.interpolation,
                    origin="lower",
                )
                ax.set_title(f"m_{component} (mag×cos(φ))")
                
        except Exception as e:
            log.error(f"Failed to update single mode plot: {e}")

    def _add_scale_bar(
        self, ax: Any, extent: tuple[float, float, float, float]
    ) -> None:
        """Add a publication-style scale bar to the supplied axis."""
        if not (self.config.show_scalebar and AXES_GRID_AVAILABLE):
            return

        x_min, x_max, y_min, y_max = extent
        width_nm = float(x_max - x_min)
        height_nm = float(y_max - y_min)
        if width_nm <= 0 or height_nm <= 0:
            return

        bar_length = (
            self.config.scalebar_length_nm
            if self.config.scalebar_length_nm is not None
            else self._auto_scalebar_length(width_nm)
        )
        if bar_length is None or bar_length <= 0:
            return

        size_vertical = height_nm * self.config.scalebar_height_fraction
        label = self._format_scalebar_label(bar_length)

        try:
            scalebar = AnchoredSizeBar(
                ax.transData,
                bar_length,
                label,
                self.config.scalebar_location,
                pad=self.config.scalebar_pad,
                color=self.config.scalebar_color,
                frameon=self.config.scalebar_frame,
                size_vertical=size_vertical,
                fontproperties=fm.FontProperties(size=self.config.scalebar_fontsize),
            )
        except Exception as exc:
            log.debug(f"Could not create scale bar: {exc}")
            return

        ax.add_artist(scalebar)

    def _auto_scalebar_length(self, width_nm: float) -> Optional[float]:
        """Pick a nice scale bar length based on sample width."""
        if width_nm <= 0:
            return None

        target = width_nm / 4
        if target <= 0:
            return None

        exponent = math.floor(math.log10(target)) if target > 0 else 0
        best = None
        for multiplier in (1, 2, 5):
            candidate = multiplier * (10**exponent)
            if candidate <= width_nm * 0.9:
                best = candidate

        if best is None:
            best = width_nm / 2

        return round(best, 3)

    def _format_scalebar_label(self, length_nm: float) -> str:
        """Format scale bar label with sensible units."""
        units = self.config.scale_units.lower()
        if units == "nm" and length_nm >= 1000:
            value_um = length_nm / 1000.0
            return f"{value_um:g} um"
        return f"{length_nm:g} {self.config.scale_units}"

    def compute_modes(
        self,
        z_slice: slice = slice(None),
        window: bool = True,
        save: bool = True,
        force: bool = False,
    ) -> None:
        """
        Compute FMR modes from magnetization data.

        Parameters:
        -----------
        z_slice : slice
            Z-layer slice to process
        window : bool
            Apply Hanning window
        save : bool
            Save results to zarr
        force : bool
            Force recomputation even if data exists
        """
        if not force and f"modes/{self.dataset_name}/arr" in self.zarr_file:
            log.info("Mode data already exists, use force=True to recompute")
            return

        log.info(f"Computing FMR modes for dataset {self.dataset_name}")

        # Remove existing data if force=True
        if force:
            try:
                # Open in write mode for deletion
                zarr_write = zarr.open(self.zarr_path, mode="a")
                if f"modes/{self.dataset_name}" in zarr_write:
                    del zarr_write[f"modes/{self.dataset_name}"]
                    log.info(f"Removed existing modes data for {self.dataset_name}")
                if f"fft/{self.dataset_name}" in zarr_write:
                    del zarr_write[f"fft/{self.dataset_name}"]
                    log.info(f"Removed existing FFT data for {self.dataset_name}")
                zarr_write.close()
                # Important: Reopen in read mode and reload data paths
                self.zarr_file = zarr.open(self.zarr_path, mode="r")
                self._load_data()  # Reload paths after deletion
            except Exception as e:
                log.warning(f"Could not remove existing data: {e}")
                # Continue anyway - might be permission issue

        # Load magnetization data
        if self.dataset_name not in self.zarr_file:
            available = self._list_available_datasets()
            suggestion = (
                f" Available datasets: {', '.join(available)}"
                if available
                else ""
            )
            raise ValueError(
                f"Dataset '{self.dataset_name}' not found in zarr file '{self.zarr_path}'.{suggestion}"
            )

        dset = self.zarr_file[self.dataset_name]

        # Determine sampling interval dt
        dt: Optional[float] = None
        t_array: Optional[np.ndarray] = None
        try:
            raw_t = dset.attrs["t"][:]
            t_array = np.asarray(raw_t, dtype=float)
            if t_array.size > 1:
                diffs = np.diff(t_array)
                positive_diffs = diffs[diffs > 0]
                if positive_diffs.size:
                    dt = float(np.mean(positive_diffs))
                else:
                    dt = float(np.median(np.abs(diffs)))
                if not np.isfinite(dt) or dt <= 0:
                    raise ValueError("Invalid timestep derived from t attribute")
            else:
                raise ValueError("Insufficient time samples in attribute")
        except Exception as exc:
            log.debug(
                "Falling back to alternative dt sources for dataset %s: %s",
                self.dataset_name,
                exc,
            )
            t_array = None
            dt = None

        def _extract_dt(candidate: Any) -> Optional[float]:
            if candidate is None:
                return None
            try:
                value = float(np.asarray(candidate).item())
                if np.isfinite(value) and value > 0:
                    return value
            except Exception:
                return None
            return None

        if dt is None:
            for attrs in (getattr(dset, "attrs", {}), getattr(self.zarr_file, "attrs", {})):
                for key in ("t_sampl", "dt"):
                    dt_candidate = _extract_dt(attrs.get(key))
                    if dt_candidate is not None:
                        dt = dt_candidate
                        break
                if dt is not None:
                    break

        if dt is None and PYZFN_AVAILABLE:
            try:
                pyz_job = Pyzfn(self.zarr_path)
                dt_candidate = _extract_dt(pyz_job.attrs.get("t_sampl", None))
                if dt_candidate is not None:
                    dt = dt_candidate
            except Exception as exc:
                log.debug("Could not retrieve t_sampl via Pyzfn: %s", exc)

        if dt is None:
            dt = 1e-12
            log.warning(
                "Falling back to default timestep 1e-12 s for dataset %s."
                " Check zarr metadata for t or t_sampl attributes.",
                self.dataset_name,
            )

        if t_array is None:
            num_samples = dset.shape[0]
            t_array = np.arange(num_samples, dtype=float) * dt
        else:
            num_samples = t_array.size

        # Calculate frequencies using number of time samples
        if num_samples < 2:
            raise ValueError("Mode computation requires at least two time samples")

        freqs = np.fft.rfftfreq(num_samples, dt) * 1e-9  # Convert to GHz

        # Load and process data
        log.info(f"Loading magnetization data: {dset.shape}")
        arr = np.asarray(dset[:, z_slice])
        log.info("Loading magnetization data finished")

        # Remove DC component
        arr = arr - arr.mean(axis=0)[None, ...]

        # Apply window function
        if window:
            window_func = np.hanning(arr.shape[0])
            for _i in range(arr.ndim - 1):
                window_func = window_func[:, None]
            arr = arr * window_func
            log.info("Applied Hanning window")

        # Compute FFT
        log.info("Computing FFT...")
        fft_result = np.fft.rfft(arr, axis=0)
        log.info("Computing FFT finished.")

        # Save results
        if save:
            log.info("Saving mode data...")

            # Open in write mode
            zarr_write = zarr.open(self.zarr_path, mode="a")

            # Remove existing data if force=True to avoid conflicts
            if force:
                if f"modes/{self.dataset_name}" in zarr_write:
                    del zarr_write[f"modes/{self.dataset_name}"]
                if f"fft/{self.dataset_name}" in zarr_write:
                    del zarr_write[f"fft/{self.dataset_name}"]

            # Create groups
            modes_group = zarr_write.require_group(f"modes/{self.dataset_name}")
            # Don't create fft_group here - let plot_spectrum/calculate_fft_data handle it
            # This avoids conflicts with standard FFT data format

            # Save frequencies in modes group only
            modes_group.array(
                "freqs",
                data=freqs,
                shape=freqs.shape,
                dtype=freqs.dtype,
                chunks=freqs.shape,  # Use the data shape as chunk size
                overwrite=True,
            )

            # Save complex modes (chunked only on first dimension)
            chunks = (1,) + fft_result.shape[1:]
            modes_group.array(
                "arr",
                data=fft_result.astype(np.complex64, copy=False),
                shape=fft_result.shape,
                dtype=np.complex64,
                chunks=chunks,
                overwrite=True,
            )

            # Save power spectrum summary in modes group 
            power_spec = np.abs(fft_result)
            reduction_axes = tuple(range(1, power_spec.ndim)) if power_spec.ndim > 1 else None
            power_max = (
                np.max(power_spec, axis=reduction_axes)
                if reduction_axes
                else np.max(power_spec, keepdims=False)
            )
            power_sum = (
                np.sum(power_spec, axis=reduction_axes)
                if reduction_axes
                else np.sum(power_spec, keepdims=False)
            )
            modes_group.array(
                "power_max",
                data=power_max.astype(np.float32, copy=False),
                shape=power_max.shape,
                dtype=np.float32,
                chunks=power_max.shape,  # Use data shape as chunk size
                overwrite=True,
            )
            modes_group.array(
                "power_sum",
                data=power_sum.astype(np.float32, copy=False),
                shape=power_sum.shape,
                dtype=np.float32,
                chunks=power_sum.shape,  # Use data shape as chunk size
                overwrite=True,
            )

            # Save metadata
            modes_group.attrs["computed_at"] = str(datetime.now())
            modes_group.attrs["window_applied"] = window
            modes_group.attrs["z_slice"] = str(z_slice)
            modes_group.attrs["dt"] = dt

            # zarr groups don't have close() method, just let it go out of scope
            log.info("✅ Mode computation completed and saved")

        # Reload data
        self.zarr_file = zarr.open(self.zarr_path, mode="r")
        self._load_data()

    def save_modes_animation(
        self,
        frequency_range: tuple[float, float] = None,
        frequency: float = None,
        save_path: str = "mode_animation.gif",
        fps: int = 15,
        z_layer: int = 0,
        component: Union[str, int] = "z",
        animation_type: str = "temporal",
        colormap: str = None,
        use_midpoint_norm: bool = None,
        figsize: tuple[float, float] = None,
    ) -> None:
        """
        Save animation of FMR modes.

        Parameters:
        -----------
        frequency_range : tuple, optional
            (f_min, f_max) in GHz for frequency sweep animation
        frequency : float, optional
            Single frequency for temporal animation (in GHz)
        save_path : str
            Output file path (.gif or .mp4)
        fps : int
            Frames per second (default: 15)
        z_layer : int
            Z-layer to animate (default: 0)
        component : str or int
            Component to animate (default: 'z')
        animation_type : str
            Type of animation:
            - 'temporal': Real part of mode oscillating in time at fixed frequency
            - 'frequency': Mode amplitude across frequency range
            - 'phase': Phase evolution at fixed frequency
        colormap : str, optional
            Colormap name. If None, uses config defaults
        use_midpoint_norm : bool, optional
            Use symmetric normalization around zero (default: from config)
        figsize : tuple, optional
            Figure size (width, height)
        """
        if not MATPLOTLIB_AVAILABLE or not ANIMATION_AVAILABLE:
            raise ImportError(
                "Matplotlib and animation support are required for animations"
            )

        # Setup professional styling for animations
        setup_animation_styling(use_paper_style=True, use_custom_fonts=True)

        try:
            import matplotlib.cm as cm
            from matplotlib.animation import FuncAnimation

            # Parameter validation
            if frequency_range is None and frequency is None:
                raise ValueError(
                    "Either frequency_range or frequency must be specified"
                )

            if frequency_range is not None and frequency is not None:
                raise ValueError(
                    "Specify either frequency_range OR frequency, not both"
                )

            # Set defaults with intelligent choices for animation type
            figsize = figsize or (10, 8)
            colormap = colormap or self.config.colormap_animation

            # Auto-enable MidpointNormalize for temporal animations if not explicitly set
            if use_midpoint_norm is None:
                if animation_type == "temporal":
                    use_midpoint_norm = (
                        True  # Temporal animations benefit from symmetric normalization
                    )
                else:
                    use_midpoint_norm = self.config.use_midpoint_norm

            # Choose better colormap for temporal oscillations if default 'balance'
            if animation_type == "temporal" and colormap == "balance":
                log.info(
                    "Using diverging colormap 'balance' - perfect for oscillating modes"
                )

            # Try to get cmocean colormaps for better scientific visualization
            try:
                import cmocean

                if colormap == "balance":
                    cmap = cmocean.cm.balance  # Perfect for data with +/- symmetry
                elif colormap == "diff":
                    cmap = cmocean.cm.diff  # Another good diverging colormap
                elif colormap == "curl":
                    cmap = cmocean.cm.curl  # Good for circular/phase data
                elif colormap == "delta":
                    cmap = cmocean.cm.delta  # Good for deviations from mean
                elif colormap == "tarn":
                    cmap = cmocean.cm.tarn  # Good for complex data
                else:
                    # Try as regular matplotlib colormap
                    cmap = plt.get_cmap(colormap)
            except ImportError:
                log.warning("cmocean not available, using matplotlib colormaps")
                if colormap == "balance":
                    cmap = plt.get_cmap("RdBu_r")  # Best fallback for balance
                elif colormap == "diff":
                    cmap = plt.get_cmap("RdBu")  # Alternative diverging
                elif colormap == "curl" or colormap == "tarn":
                    cmap = plt.get_cmap("RdYlBu_r")  # Complex fallback
                elif colormap == "delta":
                    cmap = plt.get_cmap("PuOr_r")  # Another diverging option
                else:
                    cmap = plt.get_cmap(colormap)

            # Setup figure
            fig, ax = plt.subplots(figsize=figsize, dpi=self.config.dpi)

            if animation_type == "temporal" and frequency is not None:
                # Temporal animation: Real part oscillating in time (true physical dynamics)
                # Shows how the mode would actually oscillate at this frequency
                log.info(f"Creating temporal animation at {frequency:.3f} GHz")

                # Get mode data
                mode_data = self.get_mode(frequency, z_layer)
                comp_data = mode_data.get_component(component)

                # Get amplitude and phase - this is the complex mode from FFT
                amplitude = np.abs(comp_data)
                phase = np.angle(comp_data)

                # Setup normalization - MidpointNormalize is perfect for oscillating data
                if use_midpoint_norm:
                    vmax = np.max(amplitude)
                    norm = MidpointNormalize(vmin=-vmax, vmax=vmax, midpoint=0)
                else:
                    vmax = np.max(amplitude)
                    norm = plt.Normalize(vmin=-vmax, vmax=vmax)

                # Time steps for one full oscillation period
                time_steps = np.linspace(0, 2 * np.pi, self.config.animation_time_steps)

                # Create initial plot for colorbar setup
                t = time_steps[0]
                real_part = amplitude * np.cos(phase + t)
                im = ax.imshow(
                    real_part,
                    cmap=cmap,
                    norm=norm,
                    extent=mode_data.extent,
                    aspect="equal",
                    origin="lower",
                    interpolation=self.config.interpolation,
                )

                # Create colorbar once
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("Magnetization (arb. units)")

                # Set labels once (they won't change)
                ax.set_xlabel("x (nm)")
                ax.set_ylabel("y (nm)")

                def animate_temporal(frame):
                    # Calculate real part at this time step: Re[A * e^(i*φ) * e^(i*ω*t)]
                    # This is exactly: amplitude * cos(phase + ωt) where ωt = time_steps[frame]
                    t = time_steps[frame]
                    real_part = amplitude * np.cos(phase + t)

                    # Update image data instead of recreating
                    im.set_array(real_part)

                    # Show fraction of period completed
                    period_fraction = t / (2 * np.pi)
                    ax.set_title(
                        f"Re[m_{component}] @ {frequency:.3f} GHz (t = {period_fraction:.2f}T)"
                    )

                    return [im]

                # Create animation
                anim = FuncAnimation(
                    fig,
                    animate_temporal,
                    frames=len(time_steps),
                    interval=1000 / fps,
                    blit=True,
                    repeat=True,
                )

            elif animation_type == "frequency" and frequency_range is not None:
                # Frequency sweep animation
                f_min, f_max = frequency_range
                freq_mask = (self.frequencies >= f_min) & (self.frequencies <= f_max)
                freq_indices = np.where(freq_mask)[0]

                if len(freq_indices) == 0:
                    raise ValueError("No frequencies found in specified range")

                log.info(
                    f"Creating frequency sweep animation: {f_min:.3f} - {f_max:.3f} GHz"
                )

                # Pre-calculate all mode data for consistent normalization
                all_amplitudes = []
                for freq_idx in freq_indices:
                    freq = self.frequencies[freq_idx]
                    mode_data = self.get_mode(freq, z_layer)
                    comp_data = mode_data.get_component(component)
                    all_amplitudes.append(np.abs(comp_data))

                # Global normalization
                global_max = np.max([np.max(amp) for amp in all_amplitudes])
                norm = plt.Normalize(vmin=0, vmax=global_max)

                # Create initial plot for colorbar setup
                mode_data = self.get_mode(self.frequencies[freq_indices[0]], z_layer)
                im = ax.imshow(
                    all_amplitudes[0],
                    cmap=cmap,
                    norm=norm,
                    extent=mode_data.extent,
                    aspect="equal",
                    origin="lower",
                    interpolation=self.config.interpolation,
                )

                # Create colorbar once
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("|Magnetization| (arb. units)")

                # Set labels once (they won't change)
                ax.set_xlabel("x (nm)")
                ax.set_ylabel("y (nm)")

                def animate_frequency(frame):
                    freq_idx = freq_indices[frame]
                    frequency = self.frequencies[freq_idx]
                    amplitude = all_amplitudes[frame]

                    # Update image data instead of recreating
                    im.set_array(amplitude)

                    ax.set_title(f"|m_{component}| @ {frequency:.3f} GHz")

                    return [im]

                # Create animation
                anim = FuncAnimation(
                    fig,
                    animate_frequency,
                    frames=len(freq_indices),
                    interval=1000 / fps,
                    blit=True,
                    repeat=True,
                )

            elif animation_type == "phase" and frequency is not None:
                # Phase evolution animation
                log.info(f"Creating phase animation at {frequency:.3f} GHz")

                mode_data = self.get_mode(frequency, z_layer)
                comp_data = mode_data.get_component(component)

                amplitude = np.abs(comp_data)
                phase = np.angle(comp_data)

                # Phase steps
                phase_steps = np.linspace(
                    0, 2 * np.pi, self.config.animation_time_steps
                )

                # Create initial plot for colorbar setup
                current_phase = (phase + phase_steps[0]) % (2 * np.pi)
                im = ax.imshow(
                    current_phase,
                    cmap="hsv",  # HSV is perfect for phase
                    vmin=0,
                    vmax=2 * np.pi,
                    extent=mode_data.extent,
                    aspect="equal",
                    origin="lower",
                    interpolation=self.config.interpolation,
                )

                # Create colorbar once
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("Phase (rad)")
                cbar.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
                cbar.set_ticklabels(["0", "π/2", "π", "3π/2", "2π"])

                # Set labels once (they won't change)
                ax.set_xlabel("x (nm)")
                ax.set_ylabel("y (nm)")

                def animate_phase(frame):
                    # Add phase offset
                    current_phase = (phase + phase_steps[frame]) % (2 * np.pi)

                    # Update image data instead of recreating
                    im.set_array(current_phase)

                    ax.set_title(
                        f"Phase[m_{component}] @ {frequency:.3f} GHz (φ offset = {phase_steps[frame]:.2f})"
                    )

                    return [im]

                # Create animation
                anim = FuncAnimation(
                    fig,
                    animate_phase,
                    frames=len(phase_steps),
                    interval=1000 / fps,
                    blit=True,
                    repeat=True,
                )

            else:
                raise ValueError(
                    f"Invalid animation_type '{animation_type}' for given parameters"
                )

            # Save animation
            plt.tight_layout()

            # Choose writer based on file extension with fallback support
            if save_path.endswith(".mp4"):
                # Ensure FFmpeg is available with auto-installation
                ffmpeg_path = _ensure_ffmpeg_available()
                if ffmpeg_path:
                    writer = "ffmpeg"
                    log.info("Using FFmpeg writer for MP4 format")
                else:
                    log.warning(
                        "FFmpeg not available on system, converting to GIF format"
                    )
                    save_path = save_path.replace(".mp4", ".gif")
                    writer = "pillow"
            elif save_path.endswith(".gif"):
                writer = "pillow"
            else:
                writer = "pillow"  # Default to GIF
                if not save_path.endswith(".gif"):
                    save_path += ".gif"

            log.info(f"Saving animation to {save_path} (this may take a while...)")

            try:
                # Save animation with error handling
                if writer == "ffmpeg" and save_path.endswith(".mp4"):
                    # Try to save with ffmpeg writer
                    try:
                        anim.save(
                            save_path,
                            writer=writer,
                            fps=fps,
                            dpi=self.config.dpi // 2,
                        )
                    except Exception as ffmpeg_error:
                        # If FFmpeg fails, try with a more basic writer configuration
                        log.warning(f"FFmpeg save with advanced options failed: {ffmpeg_error}")
                        log.info("Trying basic FFmpeg configuration...")
                        anim.save(
                            save_path,
                            writer="ffmpeg",
                            fps=fps,
                            dpi=self.config.dpi // 2,
                        )
                else:
                    anim.save(
                        save_path, writer=writer, fps=fps, dpi=self.config.dpi // 2
                    )

                log.info("✅ Animation saved successfully!")

            except Exception as save_error:
                log.error(f"Failed to save animation with {writer}: {save_error}")

                # Fallback to GIF if MP4 fails
                if writer == "ffmpeg" and save_path.endswith(".mp4"):
                    log.info("Attempting fallback to GIF format...")
                    fallback_path = save_path.replace(".mp4", ".gif")
                    try:
                        anim.save(
                            fallback_path,
                            writer="pillow",
                            fps=fps,
                            dpi=self.config.dpi // 2,
                        )
                        log.info(f"✅ Animation saved as GIF: {fallback_path}")
                    except Exception as gif_error:
                        log.error(f"Fallback to GIF also failed: {gif_error}")
                        raise save_error
                else:
                    raise save_error

            plt.close(fig)

        except ImportError as e:
            log.error(f"Animation requires additional packages: {e}")
            raise
        except Exception as e:
            log.error(f"Failed to create animation: {e}")
            raise

    def install_ffmpeg(self) -> Optional[str]:
        """
        Install FFmpeg for MP4 animation support.
        
        This method ensures FFmpeg is available for high-quality video
        animation export. If FFmpeg is not found on the system, it will
        be automatically downloaded and installed.
        
        Returns:
        --------
        str or None
            Path to ffmpeg executable if successful, None if failed
            
        Examples:
        ---------
        >>> analyzer = FMRModeAnalyzer(zarr_file, dataset_name)
        >>> ffmpeg_path = analyzer.install_ffmpeg()
        """
        log.info("🔧 Installing FFmpeg for animation support...")
        return install_ffmpeg(force=False, verbose=True)


class FFTModeInterface:
    """
    Enhanced FFT interface with mode visualization capabilities.

    Provides elegant syntax like: job[0].fft[0][200].plot_modes()
    """

    def __init__(self, fft_result_index: int, parent_fft):
        """Initialize mode interface for specific FFT result."""
        self.fft_result_index = fft_result_index
        self.parent_fft = parent_fft
        self._mode_analyzer = None

    def __getitem__(self, frequency_index: int) -> "FrequencyModeInterface":
        """Get mode interface for specific frequency index."""
        return FrequencyModeInterface(frequency_index, self)

    def characterize_mode(
        self,
        frequency: float,
        **kwargs,
    ) -> ModeCharacterizationResult:
        """Characterize the mode at a given frequency (GHz)."""

        return self.mode_analyzer.characterize_mode(frequency, **kwargs)

    def __repr__(self) -> str:
        """Rich representation of the FFT mode interface."""
        try:
            from rich.console import Console
            from rich.text import Text

            RICH_AVAILABLE = True
        except ImportError:
            RICH_AVAILABLE = False

        if (
            RICH_AVAILABLE
            and self.parent_fft.mmpp
            and getattr(self.parent_fft.mmpp, "_interactive_mode", False)
        ):
            return self._rich_modes_display()
        else:
            return self._basic_modes_display()

    def _rich_modes_display(self) -> str:
        """Generate rich display for FFT modes interface."""
        try:
            from rich.columns import Columns
            from rich.console import Console
            from rich.panel import Panel
            from rich.text import Text

            console = Console()

            summary_text = Text()
            summary_text.append("🎯 MMPP FFT Mode Analyzer\n", style="bold cyan")
            summary_text.append(
                f"📁 Dataset: {getattr(self.mode_analyzer, 'dataset_name', 'Not loaded')}\n",
                style="dim",
            )
            summary_text.append(
                f"🌊 Modes available: {'Yes' if getattr(self.mode_analyzer, 'modes_available', False) else 'No'}\n",
                style="dim",
            )
            summary_text.append(
                f"📊 Z-layers: {getattr(self.mode_analyzer, 'n_z_layers', 'Unknown')}\n",
                style="dim",
            )

            methods_text = Text()
            methods_text.append("🔧 Available methods:\n", style="bold yellow")
            methods = [
                (
                    "interactive_spectrum(dset=None, **kwargs)",
                    "Interactive spectrum with modes",
                ),
                (
                    "plot_modes(frequency, dset=None, **kwargs)",
                    "Plot mode at specific frequency",
                ),
                (
                    "characterize_mode(frequency, **kwargs)",
                    "Return structured mode classification",
                ),
                ("save_modes_animation(**kwargs)", "Create mode animations"),
                ("install_ffmpeg()", "Install FFmpeg for MP4 animations"),
                ("compute_modes(dset=None, **kwargs)", "Compute/recompute modes"),
                ("[freq_index].plot_modes(**kwargs)", "Plot modes at frequency index"),
                ("[freq_index].characterize(**kwargs)", "Get mode labels at frequency index"),
            ]

            for method, description in methods:
                methods_text.append("  • ", style="dim")
                methods_text.append(method, style="code")
                methods_text.append(f" - {description}\n", style="dim")

            examples_text = Text()
            examples_text.append("💡 Usage examples:\n", style="bold green")
            examples = [
                "modes.interactive_spectrum(dset='m_z11')",
                "modes.plot_modes(frequency=1.5, dset='m_z11')",
                "modes.save_modes_animation(frequency=1.5, animation_type='temporal')",
                "modes[0][150].plot_modes()  # freq index 0, freq point 150",
                "modes.compute_modes(dset='m_z5-8')",
            ]

            for example in examples:
                examples_text.append(f"  {example}\n", style="code")

            try:
                with console.capture() as capture:
                    console.print(
                        Panel.fit(
                            summary_text,
                            title="[bold blue]MMPP FFT Modes[/bold blue]",
                            border_style="blue",
                        )
                    )
                    console.print("")
                    console.print(
                        Columns(
                            [
                                Panel.fit(
                                    methods_text,
                                    title="[bold yellow]Methods[/bold yellow]",
                                    border_style="yellow",
                                ),
                                Panel.fit(
                                    examples_text,
                                    title="[bold green]Examples[/bold green]",
                                    border_style="green",
                                ),
                            ]
                        )
                    )
                return capture.get()
            except Exception:
                pass

            return (
                str(summary_text) + "\n" + str(methods_text) + "\n" + str(examples_text)
            )

        except Exception:
            return self._basic_modes_display()

    def _basic_modes_display(self) -> str:
        """Generate basic display for FFT modes interface."""
        return f"""
MMPP FFT Mode Analyzer:
======================
🎯 Advanced FMR mode visualization and analysis
📁 Dataset: {getattr(self.mode_analyzer, "dataset_name", "Not loaded")}
🌊 Modes available: {"Yes" if getattr(self.mode_analyzer, "modes_available", False) else "No"}
📊 Z-layers: {getattr(self.mode_analyzer, "n_z_layers", "Unknown")}

🔧 Main methods:
  • interactive_spectrum(dset=None, **kwargs) - Interactive spectrum with modes
  • plot_modes(frequency, dset=None, **kwargs) - Plot mode at specific frequency
  • characterize_mode(frequency, **kwargs) - Structured mode classification
  • save_modes_animation(**kwargs) - Create mode animations
  • compute_modes(dset=None, **kwargs) - Compute/recompute modes
  • [freq_index].plot_modes(**kwargs) - Plot modes at frequency index
  • [freq_index].characterize(**kwargs) - Analyze mode at frequency index

💡 Animation examples:
  • modes.save_modes_animation(frequency=1.5, animation_type='temporal')
  • modes.save_modes_animation(frequency_range=(1.0, 3.0), animation_type='frequency')

🎬 Animation types: 'temporal', 'frequency', 'phase'
🎨 Supports MP4 (ffmpeg) and GIF (pillow) output formats
"""

    @property
    def mode_analyzer(self) -> FMRModeAnalyzer:
        """Get or create mode analyzer (lazy initialization)."""
        if self._mode_analyzer is None:
            # Get zarr path from parent FFT
            zarr_path = self.parent_fft.job_result.path
            debug_mode = (
                getattr(self.parent_fft.mmpp, "debug", False)
                if self.parent_fft.mmpp
                else False
            )
            self._mode_analyzer = FMRModeAnalyzer(zarr_path, debug=debug_mode)

        return self._mode_analyzer

    def interactive_spectrum(self, dset: str = None, force: bool = False, **kwargs) -> Figure:
        """Create interactive spectrum plot."""
        # If dset is specified, create a new analyzer for that dataset
        if dset is not None and dset != self.mode_analyzer.dataset_name:
            zarr_path = self.parent_fft.job_result.path
            debug_mode = (
                getattr(self.parent_fft.mmpp, "debug", False)
                if self.parent_fft.mmpp
                else False
            )
            temp_analyzer = FMRModeAnalyzer(
                zarr_path, dataset_name=dset, debug=debug_mode
            )

            # Check if modes exist or force recomputation
            if not temp_analyzer.modes_available or force:
                log.info(f"Computing modes for dataset '{dset}' (force={force})...")
                temp_analyzer.compute_modes(save=True, force=force)

            return temp_analyzer.interactive_spectrum(**kwargs)
        else:
            # Use default analyzer
            if not self.mode_analyzer.modes_available or force:
                log.info(
                    f"Computing modes for dataset '{self.mode_analyzer.dataset_name}' (force={force})..."
                )
                self.mode_analyzer.compute_modes(save=True, force=force)

            return self.mode_analyzer.interactive_spectrum(**kwargs)

    def compute_modes(self, dset: str = None, **kwargs) -> None:
        """Compute modes for specified dataset."""
        if dset is not None:
            zarr_path = self.parent_fft.job_result.path
            debug_mode = (
                getattr(self.parent_fft.mmpp, "debug", False)
                if self.parent_fft.mmpp
                else False
            )
            temp_analyzer = FMRModeAnalyzer(
                zarr_path, dataset_name=dset, debug=debug_mode
            )
            temp_analyzer.compute_modes(**kwargs)
        else:
            self.mode_analyzer.compute_modes(**kwargs)

    def plot_modes(
        self, frequency: float, dset: str = None, **kwargs
    ) -> tuple[Figure, np.ndarray]:
        """Plot modes at specified frequency."""
        # If dset is specified, create a new analyzer for that dataset
        if dset is not None and dset != self.mode_analyzer.dataset_name:
            zarr_path = self.parent_fft.job_result.path
            debug_mode = (
                getattr(self.parent_fft.mmpp, "debug", False)
                if self.parent_fft.mmpp
                else False
            )
            temp_analyzer = FMRModeAnalyzer(
                zarr_path, dataset_name=dset, debug=debug_mode
            )

            # Check if modes exist, if not compute them
            if not temp_analyzer.modes_available:
                log.info(f"Computing modes for dataset '{dset}'...")
                temp_analyzer.compute_modes(save=True)

            return temp_analyzer.plot_modes(frequency, **kwargs)
        else:
            # Use default analyzer
            if not self.mode_analyzer.modes_available:
                log.info(
                    f"Computing modes for dataset '{self.mode_analyzer.dataset_name}'..."
                )
                self.mode_analyzer.compute_modes(save=True)

            return self.mode_analyzer.plot_modes(frequency, **kwargs)

    def save_modes_animation(
        self,
        frequency_range: tuple[float, float] = None,
        frequency: float = None,
        save_path: str = "mode_animation.gif",
        dset: str = None,
        fps: int = 15,
        z_layer: int = 0,
        component: Union[str, int] = "z",
        animation_type: str = "temporal",
        **kwargs,
    ) -> None:
        """
        Save animation of FMR modes.

        Parameters:
        -----------
        frequency_range : tuple, optional
            (f_min, f_max) in GHz for frequency sweep animation
        frequency : float, optional
            Single frequency for temporal animation (in GHz)
        save_path : str
            Output file path (.gif or .mp4)
        dset : str, optional
            Dataset name. If None, uses default analyzer
        fps : int
            Frames per second (default: 15)
        z_layer : int
            Z-layer to animate (default: 0)
        component : str or int
            Component to animate (default: 'z')
        animation_type : str
            Type of animation ('temporal', 'frequency', 'phase')
        **kwargs
            Additional arguments passed to FMRModeAnalyzer.save_modes_animation
        """
        # If dset is specified, create a new analyzer for that dataset
        if dset is not None and dset != self.mode_analyzer.dataset_name:
            zarr_path = self.parent_fft.job_result.path
            debug_mode = (
                getattr(self.parent_fft.mmpp, "debug", False)
                if self.parent_fft.mmpp
                else False
            )
            temp_analyzer = FMRModeAnalyzer(
                zarr_path, dataset_name=dset, debug=debug_mode
            )

            # Check if modes exist, if not compute them
            if not temp_analyzer.modes_available:
                log.info(f"Computing modes for dataset '{dset}'...")
                temp_analyzer.compute_modes(save=True)

            return temp_analyzer.save_modes_animation(
                frequency_range=frequency_range,
                frequency=frequency,
                save_path=save_path,
                fps=fps,
                z_layer=z_layer,
                component=component,
                animation_type=animation_type,
                **kwargs,
            )
        else:
            # Use default analyzer
            if not self.mode_analyzer.modes_available:
                log.info(
                    f"Computing modes for dataset '{self.mode_analyzer.dataset_name}'..."
                )
                self.mode_analyzer.compute_modes(save=True)

            return self.mode_analyzer.save_modes_animation(
                frequency_range=frequency_range,
                frequency=frequency,
                save_path=save_path,
                fps=fps,
                z_layer=z_layer,
                component=component,
                animation_type=animation_type,
                **kwargs,
            )

    def install_ffmpeg(self) -> str:
        """
        Install FFmpeg for MP4 animation support.
        
        This method ensures FFmpeg is available for high-quality video
        animation export. If FFmpeg is not found on the system, it will
        be automatically downloaded and installed.
        
        Returns:
        --------
        str or None
            Path to ffmpeg executable if successful, None if failed
            
        Example:
        --------
        >>> job = load_zarr("data.zarr")  
        >>> ffmpeg_path = job[0].fft.modes.install_ffmpeg()
        >>> if ffmpeg_path:
        ...     job[0].fft.modes.save_modes_animation("animation.mp4")
        """
        return install_ffmpeg()


class FrequencyModeInterface:
    """Interface for mode operations at a specific frequency."""

    def __init__(self, frequency_index: int, parent_mode_interface):
        """Initialize frequency-specific mode interface."""
        self.frequency_index = frequency_index
        self.parent = parent_mode_interface

    @property
    def frequency(self) -> float:
        """Get frequency value for this index."""
        return self.parent.mode_analyzer.frequencies[self.frequency_index]

    def plot_modes(self, **kwargs) -> tuple[Figure, np.ndarray]:
        """Plot modes at this frequency."""
        return self.parent.mode_analyzer.plot_modes(self.frequency, **kwargs)

    def get_mode(self, **kwargs) -> FMRModeData:
        """Get mode data at this frequency."""
        return self.parent.mode_analyzer.get_mode(self.frequency, **kwargs)

    def characterize(self, **kwargs) -> ModeCharacterizationResult:
        """Return automatic mode classification for this frequency."""

        return self.parent.mode_analyzer.characterize_mode(self.frequency, **kwargs)

    def __repr__(self) -> str:
        """Rich string representation of FrequencyModeInterface."""
        try:
            # Try rich display first
            return self._rich_frequency_display()
        except ImportError:
            # Fallback to basic display
            return self._basic_frequency_display()

    def _rich_frequency_display(self) -> str:
        """Rich display with styling and detailed information."""
        import io

        from rich.console import Console
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.table import Table
        from rich.text import Text

        console = Console(file=io.StringIO(), width=100, force_terminal=True)

        # Main header
        header = Text("FrequencyModeInterface", style="bold cyan")

        # Frequency information table
        freq_table = Table(show_header=False, box=None, padding=(0, 1))
        freq_table.add_column("Property", style="bold yellow")
        freq_table.add_column("Value", style="white")

        freq_table.add_row("🎯 Frequency Index", f"{self.frequency_index}")
        freq_table.add_row("⚡ Frequency Value", f"{self.frequency:.2e} Hz")
        freq_table.add_row(
            "📊 Parent Modes",
            f"{len(self.parent.mode_analyzer.frequencies)} frequencies",
        )

        # Available methods
        methods_text = Text("Available Methods:", style="bold green")
        methods_list = [
            "• plot_modes(**kwargs) → Tuple[Figure, np.ndarray]",
            "• get_mode(**kwargs) → FMRModeData",
            "• characterize(**kwargs) → ModeCharacterizationResult",
            "• frequency → float (property)",
        ]
        methods_content = "\n".join(methods_list)

        # Usage examples
        example_code = f"""# Access frequency-specific operations
freq_interface = modes[{self.frequency_index}]

# Plot modes at this frequency
fig, axes = freq_interface.plot_modes()

# Get mode data
mode_data = freq_interface.get_mode()

# Automatic classification
characterization = freq_interface.characterize()
print(characterization.primary_class)

# Check frequency value
print(f"Frequency: {{freq_interface.frequency:.2e}} Hz")"""

        syntax = Syntax(
            example_code, "python", theme="monokai", background_color="default"
        )

        # Build the panel content
        content_parts = [
            freq_table,
            "",
            methods_text,
            Text(methods_content),
            "",
            Text("Usage Examples:", style="bold blue"),
            syntax,
        ]

        panel = Panel(
            "\n".join(str(part) for part in content_parts),
            title=str(header),
            border_style="cyan",
            width=98,
        )

        console.print(panel)
        return console.file.getvalue()

    def _basic_frequency_display(self) -> str:
        """Basic fallback display without rich formatting."""
        return (
            f"FrequencyModeInterface(frequency_index={self.frequency_index}, "
            f"frequency={self.frequency:.2e} Hz)\n"
            f"Methods: plot_modes(), get_mode(), characterize(), frequency (property)\n"
            f"Parent analyzer has {len(self.parent.mode_analyzer.frequencies)} frequencies"
        )
