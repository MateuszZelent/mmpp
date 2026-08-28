"""Reusable callable notebook nodes for soliton public APIs."""

from __future__ import annotations

import inspect
import uuid
from collections.abc import Callable, Mapping, Set
from typing import Any


class CallableNodeHelper:
    """A callable method proxy that renders a canonical Overview/API card."""

    def __init__(
        self,
        *,
        owner: str,
        name: str,
        target: Callable[..., Any],
        description: str = "",
        examples: list[str] | None = None,
    ):
        self._owner = owner
        self._name = name
        self._target = target
        self.__wrapped__ = target
        target_doc = inspect.getdoc(target) or ""
        self.__doc__ = target_doc or self.__class__.__doc__
        self._description = description or (
            target_doc.splitlines()[0] if target_doc else "Run this analysis node."
        )
        self._examples = list(examples or [f"{owner}.{name}()"])

    def __call__(self, *args: Any, **kwargs: Any):
        return self._target(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any):
        """Explicit alias for calling the node."""
        return self(*args, **kwargs)

    @property
    def __signature__(self):
        return inspect.signature(self._target)

    @property
    def signature(self) -> str:
        """Live signature of the wrapped public method."""
        return str(self.__signature__)

    @property
    def target(self) -> Callable[..., Any]:
        """Underlying bound method."""
        return self._target

    def __repr__(self) -> str:
        return (
            f"<CallableNodeHelper {self._owner}.{self._name}{self.signature}: "
            f"{self._description}>"
        )

    def _repr_html_(self) -> str:
        from mmpp._repr_helpers import (
            NODE_COLOR_ANALYSIS,
            NODE_COLOR_COMPUTE,
            accessors_section_html,
            api_help_html,
            examples_section_html,
            metrics_section_html,
            node_card_html,
        )

        api = api_help_html(
            self,
            title=f"{self._name} callable API help",
            prefix=f"{self._owner}.{self._name}",
            methods=["run"],
            properties=[
                ("signature", "Live wrapped-method signature"),
                ("target", "Underlying bound method"),
            ],
            subtitle="Evaluate this node for help; call it with (...) to execute.",
            chrome=False,
        )
        return node_card_html(
            f"{self._name}{self.signature}",
            icon="▶",
            subtitle=self._description,
            sections=[
                metrics_section_html(
                    [
                        ("owner", self._owner, NODE_COLOR_ANALYSIS),
                        ("call", f"{self._name}{self.signature}", NODE_COLOR_COMPUTE),
                    ]
                ),
                accessors_section_html([("Execute:", [("(...)", NODE_COLOR_COMPUTE)])]),
                examples_section_html("\n".join(self._examples), title="Usage"),
            ],
            api=api,
            uid=f"soliton-method-{self._name}-{uuid.uuid4().hex[:8]}",
        )


class InteractiveNodeMixin:
    """Wrap selected public methods as callable notebook helper nodes."""

    _interactive_owner = "obj"
    _interactive_nodes: Set[str] = frozenset()
    _interactive_descriptions: Mapping[str, str] = {}
    _interactive_examples: Mapping[str, list[str]] = {}

    def __getattribute__(self, name: str):
        value = super().__getattribute__(name)
        if name.startswith("_") or isinstance(value, CallableNodeHelper):
            return value
        try:
            nodes = super().__getattribute__("_interactive_nodes")
        except AttributeError:
            return value
        if name not in nodes or not callable(value):
            return value
        owner = super().__getattribute__("_interactive_owner")
        descriptions = super().__getattribute__("_interactive_descriptions")
        examples = super().__getattribute__("_interactive_examples")
        return CallableNodeHelper(
            owner=owner,
            name=name,
            target=value,
            description=descriptions.get(name, ""),
            examples=examples.get(name),
        )


__all__ = ["CallableNodeHelper", "InteractiveNodeMixin"]
