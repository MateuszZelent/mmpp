# ruff: noqa: N802, N803, N806
"""
Base classes for analytical model results.

Provides fluent plotting API through the `.plt` accessor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection
    from matplotlib.lines import Line2D


@dataclass
class AnalyticalResult:
    """
    Base class for analytical model results.

    Provides access to computed data and fluent plotting API.

    Attributes
    ----------
    model_name : str
        Name of the analytical model used
    params : dict
        Parameters used for the calculation
    metadata : dict
        Additional metadata (units, references, etc.)
    """

    model_name: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def plt(self) -> PlotAccessor:
        """Access plotting methods via fluent API."""
        return PlotAccessor(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name!r})"


@dataclass
class DispersionResult(AnalyticalResult):
    """
    Result of a dispersion relation calculation f(k).

    Stores wavevector and frequency arrays with plotting support.

    Attributes
    ----------
    k : np.ndarray
        Wavevector array in 1/m
    f : np.ndarray
        Frequency array in GHz
    omega : np.ndarray
        Angular frequency array in rad/s (computed from f)

    Examples
    --------
    >>> result = damon_eshbach(k=np.linspace(0, 1e7, 100), Ms=8e5, B=0.1, d=100e-9)
    >>> result.f[0]  # Frequency at k=0
    >>> result.plt.plot()
    """

    k: np.ndarray = field(default_factory=lambda: np.array([]))
    f: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def omega(self) -> np.ndarray:
        """Angular frequency ω = 2π·f in rad/s."""
        return 2.0 * np.pi * self.f * 1e9  # f is in GHz

    @property
    def f_hz(self) -> np.ndarray:
        """Frequency in Hz."""
        return self.f * 1e9

    @property
    def k_rad_per_um(self) -> np.ndarray:
        """Wavevector in rad/μm."""
        return self.k * 1e-6

    @property
    def k_rad_per_nm(self) -> np.ndarray:
        """Wavevector in rad/nm."""
        return self.k * 1e-9

    def __len__(self) -> int:
        return len(self.k)

    def __repr__(self) -> str:
        k_range = (
            f"[{self.k.min():.2e}, {self.k.max():.2e}]" if len(self.k) > 0 else "[]"
        )
        f_range = (
            f"[{self.f.min():.2f}, {self.f.max():.2f}]" if len(self.f) > 0 else "[]"
        )
        return (
            f"DispersionResult(model={self.model_name!r}, "
            f"k={k_range} 1/m, f={f_range} GHz, n={len(self)})"
        )


@dataclass
class FMRResult(AnalyticalResult):
    """
    Result of an FMR (ferromagnetic resonance) calculation f(B).

    Stores field and frequency arrays with plotting support.

    Attributes
    ----------
    B : np.ndarray
        Magnetic field array in Tesla
    f : np.ndarray
        Frequency array in GHz

    Examples
    --------
    >>> result = kittel(B=np.linspace(0, 0.5, 100), Ms=8e5, Ku=1e4)
    >>> result.f[50]  # Frequency at B=0.25 T
    >>> result.plt.plot()
    """

    B: np.ndarray = field(default_factory=lambda: np.array([]))
    f: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def H(self) -> np.ndarray:
        """Magnetic field in A/m (from B assuming vacuum)."""
        from .constants import MU0

        return self.B / MU0

    @property
    def omega(self) -> np.ndarray:
        """Angular frequency ω = 2π·f in rad/s."""
        return 2.0 * np.pi * self.f * 1e9

    @property
    def f_hz(self) -> np.ndarray:
        """Frequency in Hz."""
        return self.f * 1e9

    def __len__(self) -> int:
        return len(self.B)

    def __repr__(self) -> str:
        B_range = (
            f"[{self.B.min():.3f}, {self.B.max():.3f}]" if len(self.B) > 0 else "[]"
        )
        f_range = (
            f"[{self.f.min():.2f}, {self.f.max():.2f}]" if len(self.f) > 0 else "[]"
        )
        return (
            f"FMRResult(model={self.model_name!r}, "
            f"B={B_range} T, f={f_range} GHz, n={len(self)})"
        )


class PlotAccessor:
    """
    Fluent plotting API for analytical results.

    Provides `.plot()`, `.scatter()`, and other plotting methods.
    Accessed via `result.plt.plot(...)`.

    Examples
    --------
    >>> result = kittel(B=np.linspace(0, 0.5, 100), Ms=8e5)
    >>> result.plt.plot(color='blue', label='Kittel')
    >>> result.plt.scatter(s=10, alpha=0.5)
    """

    def __init__(self, result: AnalyticalResult):
        self._result = result

    def _get_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """Get x and y arrays for plotting."""
        if isinstance(self._result, DispersionResult):
            return self._result.k, self._result.f
        elif isinstance(self._result, FMRResult):
            return self._result.B, self._result.f
        else:
            raise TypeError(f"Unknown result type: {type(self._result)}")

    def _get_default_labels(self) -> tuple[str, str]:
        """Get default axis labels based on result type."""
        if isinstance(self._result, DispersionResult):
            return "k [1/m]", "f [GHz]"
        elif isinstance(self._result, FMRResult):
            return "B [T]", "f [GHz]"
        else:
            return "x", "y"

    def _ensure_axes(
        self,
        ax: Axes | None = None,
        figsize: tuple[float, float] = (8, 6),
    ) -> Axes:
        """Ensure we have a matplotlib Axes to plot on."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        return ax

    def plot(
        self,
        ax: Axes | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        figsize: tuple[float, float] = (8, 6),
        show: bool = False,
        **kwargs,
    ) -> Line2D:
        """
        Create a line plot of the analytical result.

        Parameters
        ----------
        ax : Axes, optional
            Matplotlib axes to plot on. If None, creates new figure.
        xlabel : str, optional
            X-axis label. Uses default if None.
        ylabel : str, optional
            Y-axis label. Uses default if None.
        title : str, optional
            Plot title
        figsize : tuple, optional
            Figure size if creating new figure
        show : bool, optional
            If True, call plt.show()
        **kwargs
            Additional arguments passed to ax.plot()

        Returns
        -------
        Line2D
            The plotted line object

        Examples
        --------
        >>> result.plt.plot(color='red', linewidth=2, label='DE mode')
        >>> result.plt.plot(ax=existing_ax, linestyle='--')
        """
        import matplotlib.pyplot as plt

        ax = self._ensure_axes(ax, figsize)
        x, y = self._get_xy()
        default_xlabel, default_ylabel = self._get_default_labels()

        # Set default styling
        kwargs.setdefault("linewidth", 1.5)
        if "label" not in kwargs:
            kwargs["label"] = self._result.model_name

        (line,) = ax.plot(x, y, **kwargs)

        ax.set_xlabel(xlabel or default_xlabel)
        ax.set_ylabel(ylabel or default_ylabel)
        if title:
            ax.set_title(title)

        ax.grid(True, alpha=0.3)

        if show:
            plt.show()

        return line

    def scatter(
        self,
        ax: Axes | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        title: str | None = None,
        figsize: tuple[float, float] = (8, 6),
        show: bool = False,
        **kwargs,
    ) -> PathCollection:
        """
        Create a scatter plot of the analytical result.

        Parameters
        ----------
        ax : Axes, optional
            Matplotlib axes to plot on
        xlabel : str, optional
            X-axis label
        ylabel : str, optional
            Y-axis label
        title : str, optional
            Plot title
        figsize : tuple, optional
            Figure size if creating new figure
        show : bool, optional
            If True, call plt.show()
        **kwargs
            Additional arguments passed to ax.scatter()

        Returns
        -------
        PathCollection
            The scatter plot collection
        """
        import matplotlib.pyplot as plt

        ax = self._ensure_axes(ax, figsize)
        x, y = self._get_xy()
        default_xlabel, default_ylabel = self._get_default_labels()

        # Set defaults
        kwargs.setdefault("s", 20)
        kwargs.setdefault("alpha", 0.7)
        if "label" not in kwargs:
            kwargs["label"] = self._result.model_name

        scatter = ax.scatter(x, y, **kwargs)

        ax.set_xlabel(xlabel or default_xlabel)
        ax.set_ylabel(ylabel or default_ylabel)
        if title:
            ax.set_title(title)

        ax.grid(True, alpha=0.3)

        if show:
            plt.show()

        return scatter

    def fill_between(
        self,
        other: AnalyticalResult | np.ndarray | float,
        ax: Axes | None = None,
        **kwargs,
    ) -> Any:
        """
        Fill between this result and another.

        Parameters
        ----------
        other : AnalyticalResult or array or float
            The other boundary for fill
        ax : Axes, optional
            Matplotlib axes
        **kwargs
            Arguments passed to ax.fill_between()

        Returns
        -------
        PolyCollection
            The fill object
        """
        ax = self._ensure_axes(ax)
        x, y1 = self._get_xy()

        if isinstance(other, AnalyticalResult):
            _, y2 = PlotAccessor(other)._get_xy()
        elif isinstance(other, np.ndarray):
            y2 = other
        else:
            y2 = float(other)  # type: ignore[assignment]

        kwargs.setdefault("alpha", 0.3)
        return ax.fill_between(x, y1, y2, **kwargs)
