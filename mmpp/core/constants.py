import sys
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Type aliases for numpy arrays
if TYPE_CHECKING:
    npf32 = NDArray[np.float32]
    npc64 = NDArray[np.complex64]
    np1d = NDArray[Any]
    np2d = NDArray[Any]
    np3d = NDArray[Any]
    np4d = NDArray[Any]
    np5d = NDArray[Any]
    np4dc = NDArray[np.complex64]
else:
    npf32 = np.ndarray
    npc64 = np.ndarray
    np1d = np.ndarray
    np2d = np.ndarray
    np3d = np.ndarray
    np4d = np.ndarray
    np5d = np.ndarray
    np4dc = np.ndarray

ArraySlice = slice | tuple | int

SPECIAL_ATTRS = {
    "dx",
    "dy",
    "dz",
    "Tx",
    "Ty",
    "Tz",
    "dt",
    "t_sampl",
    "fcut",
    "f_cut",
    "Nx",
    "Ny",
    "Nz",
    "cellsize_x",
    "cellsize_y",
    "cellsize_z",
    "total_time",
    "n_steps",
}

# Feature flags and optional imports
try:
    import itables  # noqa: F401

    ITABLES_AVAILABLE = True
except ImportError:
    ITABLES_AVAILABLE = False


def _optional_module_available(module_name: str) -> bool:
    """Return whether an optional module can be imported safely.

    A module injected into ``sys.modules`` by an embedding application or test
    can legitimately have ``__spec__ = None``.  ``find_spec`` raises
    ``ValueError`` for that case, so check already-loaded modules first and
    keep optional-dependency probing non-fatal.
    """
    if module_name in sys.modules:
        return sys.modules[module_name] is not None
    try:
        return find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


RICH_AVAILABLE = _optional_module_available("rich")

# Do not probe notebook/plotting stacks during core import. Public dependency
# checks resolve these optional packages explicitly when requested.
IPYTHON_AVAILABLE = False
PLOTTING_AVAILABLE = False

FFT_AVAILABLE = True
