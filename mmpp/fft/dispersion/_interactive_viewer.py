"""Lightweight interactive-dispersion controller.

This module intentionally has no Matplotlib/IPython/ipywidgets imports at module
import time. It is the stable, testable object returned by fluent interactive
APIs; notebook rendering can be layered on top by calling :meth:`show`.
"""

import json
from html import escape
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from ._json import json_safe

if TYPE_CHECKING:
    from .models import DispersionResult1D


def _normalize_interactive_options(
    *,
    components: Optional[list[str]] = None,
    mode_components: Optional[list[str]] = None,
    spectrum_components: Optional[list[str]] = None,
    animate: Optional[bool] = None,
    auto_animate: Optional[bool] = None,
    **kwargs: Any,
) -> tuple[Optional[list[str]], Optional[list[str]], dict[str, Any]]:
    """Normalize compatibility aliases used by spectrum/modes notebooks."""
    if components is not None:
        if mode_components is None:
            mode_components = list(components)
        if spectrum_components is None:
            spectrum_components = list(components)

    options = dict(kwargs)
    options.setdefault("positive_frequencies", True)
    if auto_animate is None and animate is not None:
        auto_animate = bool(animate)
    if auto_animate is not None:
        options["auto_animate"] = bool(auto_animate)

    return mode_components, spectrum_components, options


@dataclass
class DispersionInteractiveViewer:
    """Stable controller returned by dispersion interactive APIs."""

    result: "DispersionResult1D"
    show_requested: bool = True
    mode_components: Optional[list[str]] = None
    spectrum_components: Optional[list[str]] = None
    can_reconstruct_modes: bool = False
    mode_unavailable_reason: str = ""
    options: dict[str, Any] = field(default_factory=dict)
    _display_handle: Any = None

    @classmethod
    def from_result(
        cls,
        result: "DispersionResult1D",
        *,
        show: bool = True,
        can_reconstruct_modes: Optional[bool] = None,
        mode_unavailable_reason: str = "",
        components: Optional[list[str]] = None,
        mode_components: Optional[list[str]] = None,
        spectrum_components: Optional[list[str]] = None,
        animate: Optional[bool] = None,
        auto_animate: Optional[bool] = None,
        **kwargs: Any,
    ) -> "DispersionInteractiveViewer":
        mode_components, spectrum_components, options = _normalize_interactive_options(
            components=components,
            mode_components=mode_components,
            spectrum_components=spectrum_components,
            animate=animate,
            auto_animate=auto_animate,
            **kwargs,
        )
        if can_reconstruct_modes is None:
            can_reconstruct_modes = result.S_complex is not None
        if not can_reconstruct_modes and not mode_unavailable_reason:
            mode_unavailable_reason = (
                "Mode reconstruction requires S_complex and source FFT context."
            )

        viewer = cls(
            result=result,
            show_requested=bool(show),
            mode_components=mode_components,
            spectrum_components=spectrum_components,
            can_reconstruct_modes=bool(can_reconstruct_modes),
            mode_unavailable_reason=mode_unavailable_reason,
            options=options,
        )
        if show:
            viewer.show()
        return viewer

    def show(self) -> "DispersionInteractiveViewer":
        """Display a lightweight notebook representation when IPython exists."""
        self.show_requested = True
        try:
            from IPython.display import display
        except ImportError:
            return self

        self._display_handle = display(self, display_id=True)
        return self

    def close(self) -> None:
        """Best-effort close/update hook for notebook integrations."""
        if self._display_handle is not None and hasattr(self._display_handle, "update"):
            self._display_handle.update(None)
        self._display_handle = None
        self.show_requested = False

    @property
    def state(self) -> dict[str, Any]:
        """Serializable viewer state for tests, presets, and notebooks."""
        result_notes = list(getattr(self.result, "notes", None) or [])
        return {
            "show": self.show_requested,
            "mode_components": self.mode_components,
            "spectrum_components": self.spectrum_components,
            "can_reconstruct_modes": self.can_reconstruct_modes,
            "mode_unavailable_reason": self.mode_unavailable_reason,
            "result_notes": result_notes,
            "options": json_safe(self.options),
        }

    def export_selection(self, **selection: Any) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of viewer state and selection."""
        return {
            "viewer": self.state,
            "selection": json_safe(selection),
        }

    def save_preset(self, path: str | Path) -> Path:
        """Persist lightweight viewer state to a JSON preset file."""
        preset_path = Path(path)
        preset_path.parent.mkdir(parents=True, exist_ok=True)
        preset_path.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
        return preset_path

    def load_preset(self, path: str | Path) -> "DispersionInteractiveViewer":
        """Load lightweight viewer state from a JSON preset file."""
        preset_path = Path(path)
        payload = json.loads(preset_path.read_text())
        self.show_requested = bool(payload.get("show", self.show_requested))
        self.mode_components = payload.get("mode_components")
        self.spectrum_components = payload.get("spectrum_components")
        self.can_reconstruct_modes = bool(
            payload.get("can_reconstruct_modes", self.can_reconstruct_modes)
        )
        self.mode_unavailable_reason = str(
            payload.get("mode_unavailable_reason", self.mode_unavailable_reason)
        )
        options = payload.get("options", {})
        self.options = dict(options) if isinstance(options, dict) else {}
        self.options.setdefault("positive_frequencies", True)
        return self

    def _repr_html_(self) -> str:
        status = "mode-ready" if self.can_reconstruct_modes else "spectrum-only"
        notes = list(getattr(self.result, "notes", None) or [])
        notes_html = ""
        if notes:
            rows = "".join(f"<li>{escape(str(note))}</li>" for note in notes[:8])
            if len(notes) > 8:
                rows += f"<li>... {len(notes) - 8} more notes</li>"
            notes_html = f"<ul style='margin:6px 0 0 18px;padding:0;'>{rows}</ul>"
        return (
            "<div style='font-family:monospace;padding:8px;border:1px solid #334155;"
            "border-radius:6px;background:#0f172a;color:#e2e8f0;'>"
            f"DispersionInteractiveViewer: {status}"
            f"{notes_html}"
            "</div>"
        )
