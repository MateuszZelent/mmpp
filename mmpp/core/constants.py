import numpy as np
from typing import TYPE_CHECKING, Any, Union

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

ArraySlice = Union[slice, tuple, int]

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

try:
    from rich import print as rprint
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Define dummy classes/functions if needed, or handle at usage site
    Console = None
    Syntax = None
    Table = None

# Do not probe notebook/plotting stacks during core import. Public dependency
# checks resolve these optional packages explicitly when requested.
IPYTHON_AVAILABLE = False
PLOTTING_AVAILABLE = False

FFT_AVAILABLE = True
