"""Styling utilities for FMR mode visualization."""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ...cli.logging_config import get_mmpp_logger

log = get_mmpp_logger("mmpp.fft.modes")

try:
    from ...plotting import apply_custom_colors, load_paper_style, setup_custom_fonts

    STYLING_AVAILABLE = True
except ImportError:
    apply_custom_colors = load_paper_style = setup_custom_fonts = None  # type: ignore
    STYLING_AVAILABLE = False
    log.warning("Styling functions not available - using default matplotlib styling")


class MidpointNormalize(mcolors.Normalize):
    """Matplotlib normalization with a configurable midpoint."""

    def __init__(self, vmin=None, vmax=None, midpoint=0, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        if self.vmin is None or self.vmax is None:
            return super().__call__(value, clip)

        normalized_min = max(0, 0.5 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 0.5 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5

        x = [self.vmin, self.midpoint, self.vmax]
        y = [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))


def setup_animation_styling(
    use_paper_style: bool = True,
    use_custom_fonts: bool = True,
) -> bool:
    """Apply MMPP styling helpers for animations."""
    if not STYLING_AVAILABLE:
        log.warning("Styling functions not available - using default matplotlib styling")
        return False

    try:
        success = True

        if use_custom_fonts and setup_custom_fonts is not None:
            font_success = setup_custom_fonts(verbose=False)
            if not font_success:
                log.warning("Custom font setup failed - using default fonts")
                success = False

        if use_paper_style and load_paper_style is not None:
            style_success = load_paper_style(verbose=False)
            if style_success:
                log.debug("✓ Applied paper.mplstyle to mode animations")
            else:
                log.warning("Paper style loading failed - using default style")
                success = False

        if apply_custom_colors is not None:
            custom_colors = {
                "text": "#2E2E2E",
                "axes": "#2E2E2E",
                "grid": "#CCCCCC",
            }
            apply_custom_colors(custom_colors)

        return success
    except Exception as exc:
        log.warning(f"Animation styling setup failed: {exc}")
        return False


__all__ = [
    "MidpointNormalize",
    "setup_animation_styling",
    "STYLING_AVAILABLE",
]
