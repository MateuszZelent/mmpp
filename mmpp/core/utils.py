from typing import Any

from ..cli.logging_config import get_mmpp_logger
from .mmpp import MMPP

log = get_mmpp_logger("mmpp")


def open(
    base_path: str = ".",
    max_workers: int = 8,
    database_name: str = "mmpy_database",
    debug: bool = False,
    log_level: str | int | None = None,
) -> MMPP:
    """
    Open a directory or .zarr file and return an MMPP instance.

    Parameters:
    -----------
    base_path : str, optional
        Base directory path to scan or .zarr file path (default: ".")
    max_workers : int, optional
        Number of threads for scanning (default: 8)
    database_name : str, optional
        Name of the database file (default: "mmpy_database")
    debug : bool, optional
        Enable debug logging (default: False)
    log_level : str or int, optional
        Set specific logging level (overrides debug flag)

    Returns:
    --------
    MMPP
        MMPP instance ready for use
    """
    return MMPP(
        base_path=base_path,
        max_workers=max_workers,
        database_name=database_name,
        debug=debug,
        log_level=log_level,
    )


def mmpp(base_path: str, force: bool = False, **kwargs: Any) -> MMPP:
    """
    Convenience function to create and initialize a MMPP.

    Parameters:
    -----------
    base_path : str
        Base directory path to scan
    force : bool, optional
        If True, force rescan even if database exists (default: False)
    **kwargs : Any
        Additional arguments passed to MMPP constructor

    Returns:
    --------
    MMPP
        Initialized processor instance
    """
    processor = MMPP(base_path, **kwargs)
    processor.scan(force=force)
    return processor


def install_ffmpeg() -> None:
    """
    Helper to install ffmpeg using imageio.
    Useful if system ffmpeg is not available.
    """
    try:
        import imageio

        imageio.plugins.ffmpeg.download()
        print("ffmpeg installed successfully via imageio")
    except ImportError:
        print("imageio not found. Please install it: pip install imageio")
    except Exception as e:
        print(f"Error installing ffmpeg: {e}")


def check_dependencies() -> dict[str, bool]:
    """
    Check availability of optional dependencies.

    Returns:
    --------
    Dict[str, bool]
        Dictionary of dependency names and their availability
    """
    dependencies = {
        "rich": False,
        "matplotlib": False,
        "imageio": False,
        "scipy": False,
        "pandas": False,
        "zarr": False,
        "numpy": False,
    }

    try:
        import rich  # noqa

        dependencies["rich"] = True
    except ImportError:
        pass

    try:
        import matplotlib  # noqa

        dependencies["matplotlib"] = True
    except ImportError:
        pass

    try:
        import imageio  # noqa

        dependencies["imageio"] = True
    except ImportError:
        pass

    try:
        import scipy  # noqa

        dependencies["scipy"] = True
    except ImportError:
        pass

    try:
        import pandas  # noqa

        dependencies["pandas"] = True
    except ImportError:
        pass

    try:
        import zarr  # noqa

        dependencies["zarr"] = True
    except ImportError:
        pass

    try:
        import numpy  # noqa

        dependencies["numpy"] = True
    except ImportError:
        pass

    return dependencies
