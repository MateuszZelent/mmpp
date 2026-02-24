"""Widget constructors with optional ipywidgets dependency."""

from __future__ import annotations


def create_int_slider(*, min_value: int, max_value: int, value: int = 0, description: str = "Frame"):
    """Create an IntSlider when ipywidgets is available, else return None."""
    try:
        import ipywidgets as widgets
    except Exception:  # pragma: no cover - optional dependency
        return None

    lower = int(min_value)
    upper = int(max_value)
    current = int(max(lower, min(upper, value)))
    return widgets.IntSlider(
        value=current,
        min=lower,
        max=upper,
        step=1,
        description=str(description),
        continuous_update=False,
    )
