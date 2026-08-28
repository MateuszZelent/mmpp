"""
FFmpeg utilities for animation support in MMPP.

This module provides automatic FFmpeg installation and management
for animation saving capabilities.
"""

import platform as platform_module
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from ...cli.logging_config import get_mmpp_logger

log = get_mmpp_logger(__name__)


def _detect_platform():
    """Detect the current platform for FFmpeg installation."""
    system = platform_module.system().lower()
    machine = platform_module.machine().lower()

    if system == "linux":
        if machine in ["x86_64", "amd64"]:
            return "linux-amd64"
        elif machine in ["aarch64", "arm64"]:
            return "linux-arm64"
        elif machine.startswith("arm"):
            return "linux-armhf"
    elif system == "darwin":  # macOS
        if machine in ["x86_64", "amd64"]:
            return "macos-amd64"
        elif machine in ["arm64", "aarch64"]:
            return "macos-arm64"
    elif system == "windows":
        if machine in ["x86_64", "amd64"]:
            return "windows-amd64"

    return None


def _get_ffmpeg_download_info(platform_id):
    """Get download URL and extraction info for different platforms."""
    urls = {
        "linux-amd64": {
            "url": "https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz",
            "format": "tar.xz",
            "binary_pattern": "*/ffmpeg",
        },
        "linux-arm64": {
            "url": "https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-arm64-static.tar.xz",
            "format": "tar.xz",
            "binary_pattern": "*/ffmpeg",
        },
        "linux-armhf": {
            "url": "https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-armhf-static.tar.xz",
            "format": "tar.xz",
            "binary_pattern": "*/ffmpeg",
        },
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
    # Check if FFmpeg is already in PATH
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        log.debug(f"FFmpeg found in system PATH: {ffmpeg_path}")
        return ffmpeg_path

    # Check if we already have a downloaded version
    mmpp_cache_dir = Path.home() / ".mmpp" / "bin"
    cached_ffmpeg = mmpp_cache_dir / "ffmpeg"

    if cached_ffmpeg.exists() and cached_ffmpeg.is_file():
        # Test if cached version works
        try:
            result = subprocess.run(
                [str(cached_ffmpeg), "-version"], capture_output=True, timeout=5
            )
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
        log.warning(
            "Please install FFmpeg manually or call mmpp.fft.modes.install_ffmpeg() for detailed installation help"
        )
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
    # Detect platform
    platform_id = _detect_platform()
    if not platform_id:
        raise RuntimeError("Unsupported platform for automatic FFmpeg installation")

    # Get download info
    download_info = _get_ffmpeg_download_info(platform_id)
    if not download_info:
        raise RuntimeError(f"No download URL available for platform: {platform_id}")

    # Create cache directory
    mmpp_cache_dir = Path.home() / ".mmpp" / "bin"
    mmpp_cache_dir.mkdir(parents=True, exist_ok=True)
    cached_ffmpeg = mmpp_cache_dir / "ffmpeg"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        archive_path = (
            temp_path / f"ffmpeg-static.{download_info['format'].replace('.', '_')}"
        )

        # Download with timeout and retry
        for attempt in range(3):
            try:
                log.info(
                    f"Downloading FFmpeg from {download_info['url']} (attempt {attempt + 1}/3)..."
                )

                # Simple download with timeout
                urllib.request.urlretrieve(download_info["url"], archive_path)
                log.info("Download completed, extracting...")
                break

            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise RuntimeError(
                        f"Failed to download FFmpeg after 3 attempts: {e}"
                    ) from e
                log.warning(f"Download attempt {attempt + 1} failed: {e}, retrying...")

        # Extract archive
        if download_info["format"] == "tar.xz":
            with tarfile.open(archive_path, "r:xz") as tar:
                # Find ffmpeg binary in archive
                ffmpeg_member = None
                for member in tar.getmembers():
                    if member.name.endswith("/ffmpeg") and member.isfile():
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
        result = subprocess.run(
            [str(cached_ffmpeg), "-version"], capture_output=True, timeout=10
        )
        if result.returncode == 0:
            log.info("FFmpeg installation verified successfully")
            return str(cached_ffmpeg)
        else:
            raise RuntimeError(f"FFmpeg test failed: {result.stderr.decode()}")


def install_ffmpeg(force: bool = False, verbose: bool = True) -> str | None:
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
    if verbose:
        log.info("🔧 Installing FFmpeg for animation support...")

    # Check if already available (unless forcing)
    if not force:
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            if verbose:
                log.info(f"✅ FFmpeg already available in system PATH: {ffmpeg_path}")
            return ffmpeg_path

        # Check cached version
        cached_ffmpeg = Path.home() / ".mmpp" / "bin" / "ffmpeg"
        if cached_ffmpeg.exists():
            try:
                result = subprocess.run(
                    [str(cached_ffmpeg), "-version"], capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    if verbose:
                        log.info(
                            f"✅ FFmpeg already available in cache: {cached_ffmpeg}"
                        )
                    return str(cached_ffmpeg)
            except Exception:
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
    system = platform_module.system().lower()

    log.info("📖 Manual FFmpeg Installation Instructions:")
    log.info("=" * 50)

    if system == "linux":
        log.info("🐧 Linux:")
        log.info("  Ubuntu/Debian: sudo apt update && sudo apt install ffmpeg")
        log.info("  RHEL/CentOS:   sudo yum install ffmpeg  (or dnf)")
        log.info("  Arch Linux:    sudo pacman -S ffmpeg")
        log.info("  Conda:         conda install -c conda-forge ffmpeg")
    elif system == "darwin":
        log.info("🍎 macOS:")
        log.info("  Homebrew:      brew install ffmpeg")
        log.info("  MacPorts:      sudo port install ffmpeg")
        log.info("  Conda:         conda install -c conda-forge ffmpeg")
    elif system == "windows":
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
    result = {
        "available": False,
        "path": None,
        "version": None,
        "cached_version": False,
        "platform": _detect_platform(),
        "diagnostics": [],
    }

    # Check system PATH
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        result["available"] = True
        result["path"] = system_ffmpeg
        result["diagnostics"].append(f"✅ Found in system PATH: {system_ffmpeg}")

        # Get version
        try:
            version_result = subprocess.run(
                [system_ffmpeg, "-version"], capture_output=True, timeout=10
            )
            if version_result.returncode == 0:
                version_line = version_result.stdout.decode().split("\n")[0]
                result["version"] = version_line
                result["diagnostics"].append(f"✅ Version: {version_line}")
            else:
                result["diagnostics"].append("⚠️  Could not determine version")
        except Exception as e:
            result["diagnostics"].append(f"⚠️  Version check failed: {e}")
    else:
        result["diagnostics"].append("❌ Not found in system PATH")

    # Check cached version
    cached_ffmpeg = Path.home() / ".mmpp" / "bin" / "ffmpeg"
    if cached_ffmpeg.exists():
        result["diagnostics"].append(f"🔍 Found cached version: {cached_ffmpeg}")

        if not result["available"]:  # Only test if system version not found
            try:
                version_result = subprocess.run(
                    [str(cached_ffmpeg), "-version"], capture_output=True, timeout=10
                )
                if version_result.returncode == 0:
                    result["available"] = True
                    result["path"] = str(cached_ffmpeg)
                    result["cached_version"] = True
                    version_line = version_result.stdout.decode().split("\n")[0]
                    result["version"] = version_line
                    result["diagnostics"].append(
                        f"✅ Cached version working: {version_line}"
                    )
                else:
                    result["diagnostics"].append("❌ Cached version not working")
            except Exception as e:
                result["diagnostics"].append(f"❌ Cached version test failed: {e}")
    else:
        result["diagnostics"].append("❌ No cached version found")

    # Add platform info
    if result["platform"]:
        result["diagnostics"].append(f"🔍 Platform: {result['platform']}")
    else:
        result["diagnostics"].append(
            "⚠️  Unsupported platform for automatic installation"
        )

    return result


def check_ffmpeg_available() -> bool:
    """
    Quick check if FFmpeg is available for animation saving.

    Returns:
    --------
    bool
        True if FFmpeg is available, False otherwise
    """
    return check_ffmpeg_installation()["available"]


def install_ffmpeg_simple() -> bool:
    """
    Simple FFmpeg installation without verbose output.

    Returns:
    --------
    bool
        True if installation successful, False otherwise
    """
    try:
        result = install_ffmpeg(verbose=False)
        return result is not None
    except Exception:
        return False


def _create_ffmpeg_writer(filename: str, fps: int = 30, bitrate: int = 1800):
    """
    Create FFmpeg writer for animation saving.

    Parameters:
    -----------
    filename : str
        Output filename
    fps : int, default 30
        Frames per second
    bitrate : int, default 1800
        Video bitrate in kbps

    Returns:
    --------
    matplotlib animation writer or None
    """
    try:
        from matplotlib.animation import FFMpegWriter

        # Get FFmpeg path
        ffmpeg_path = _ensure_ffmpeg_available()
        if not ffmpeg_path:
            log.error("FFmpeg not available for animation saving")
            return None

        # Create writer with proper bin_path assignment
        Writer = FFMpegWriter(fps=fps, bitrate=bitrate)
        # Set bin_path properly for FFMpegWriter - must be callable
        Writer.bin_path = lambda: ffmpeg_path  # type: ignore[method-assign]

        return Writer

    except ImportError:
        log.error("FFMpegWriter not available from matplotlib")
        return None
    except Exception as e:
        log.error(f"Error creating FFmpeg writer: {e}")
        return None
