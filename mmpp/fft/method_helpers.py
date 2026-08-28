"""Reusable callable helper wrappers for FFT public APIs."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from html import escape as _esc
from typing import Any


class CallableMethodHelper:
    """Callable helper object with concise text/HTML usage display."""

    def __init__(
        self,
        *,
        owner: str,
        name: str,
        target: Callable[..., Any],
        description: str = "",
        examples: Sequence[str] | None = None,
    ):
        self._owner = owner
        self._name = name
        self._target = target
        self._description = description or "Call helper to execute method."
        self._examples = list(examples or [])

    def __call__(self, *args, **kwargs):
        return self._target(*args, **kwargs)

    @property
    def target(self) -> Callable[..., Any]:
        """Underlying callable."""
        return self._target

    @property
    def signature(self) -> str:
        """Best-effort callable signature."""
        try:
            return str(inspect.signature(self._target))
        except Exception:
            return "(...)"

    def __repr__(self) -> str:
        return (
            f"<MethodHelper {self._owner}.{self._name}{self.signature}: "
            f"{self._description}>"
        )

    def _repr_html_(self) -> str:
        example_html = ""
        if self._examples:
            rows = "".join(
                f"<li style='margin:2px 0;'><code style='color:#93c5fd;'>{_esc(ex)}</code></li>"
                for ex in self._examples
            )
            example_html = (
                "<div style='margin-top:8px;'>"
                "<div style='font-weight:600;color:#e2e8f0;margin-bottom:4px;'>Examples</div>"
                f"<ul style='margin:0;padding-left:18px;color:#cbd5e1;'>{rows}</ul>"
                "</div>"
            )

        return (
            "<div style=\"font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "border:1px solid #334155;border-radius:10px;padding:12px;margin:6px 0;"
            'background:#0f172a;color:#e2e8f0;">'
            f"<div style='font-weight:600;color:#f1f5f9;'>{_esc(self._owner)}."
            f"{_esc(self._name)}{_esc(self.signature)}</div>"
            f"<div style='margin-top:4px;color:#cbd5e1;'>{_esc(self._description)}</div>"
            f"{example_html}</div>"
        )
