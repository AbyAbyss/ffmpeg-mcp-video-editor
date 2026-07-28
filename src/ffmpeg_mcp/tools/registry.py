"""Tool registration.

Each tool is an async function taking one pydantic input model and returning one
pydantic output model. The decorator reads those two types off the signature and
derives the MCP JSON schemas from them, so the schema can never drift from the
implementation.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar, get_type_hints

from pydantic import BaseModel

TInput = TypeVar("TInput", bound=BaseModel)
TOutput = TypeVar("TOutput", bound=BaseModel)

ToolFn = Callable[[Any], Awaitable[BaseModel]]


@dataclass(frozen=True)
class ToolSpec:
    """One registered MCP tool."""

    name: str
    title: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    fn: ToolFn
    read_only: bool
    phase: int

    def input_schema(self) -> dict[str, Any]:
        """JSON schema for the tool's arguments."""
        return self.input_model.model_json_schema()

    def schema_fingerprint(self) -> str:
        """A short hash of this tool's argument schema.

        Jobs carry the fingerprint of the build that enqueued them so a worker
        running different code does not claim work it cannot faithfully run.
        A version number would not do: two builds can disagree about a schema
        while both calling themselves 0.1.0.
        """
        canonical = json.dumps(self.input_schema(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def output_schema(self) -> dict[str, Any]:
        """JSON schema for the tool's structured result."""
        return self.output_model.model_json_schema()

    async def call(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate arguments, run the tool, and return a JSON-ready result."""
        parsed = self.input_model.model_validate(arguments or {})
        result = await self.fn(parsed)
        return result.model_dump(mode="json")


_REGISTRY: dict[str, ToolSpec] = {}


def tool(
    name: str,
    *,
    title: str,
    phase: int,
    read_only: bool = False,
) -> Callable[[Callable[[TInput], Awaitable[TOutput]]], Callable[[TInput], Awaitable[TOutput]]]:
    """Register an async function as an MCP tool.

    The function must take exactly one parameter annotated with a pydantic model
    and return a pydantic model. Its docstring becomes the tool description that
    the calling model sees, so write it for that audience.
    """

    def decorate(
        fn: Callable[[TInput], Awaitable[TOutput]],
    ) -> Callable[[TInput], Awaitable[TOutput]]:
        if name in _REGISTRY:
            raise ValueError(f"Duplicate tool name {name!r}")
        signature = inspect.signature(fn)
        params = list(signature.parameters.values())
        if len(params) != 1:
            raise TypeError(f"Tool {name!r} must take exactly one argument model.")
        hints = get_type_hints(fn)
        input_model = hints.get(params[0].name)
        output_model = hints.get("return")
        if not (isinstance(input_model, type) and issubclass(input_model, BaseModel)):
            raise TypeError(f"Tool {name!r} argument must be annotated with a pydantic model.")
        if not (isinstance(output_model, type) and issubclass(output_model, BaseModel)):
            raise TypeError(f"Tool {name!r} return type must be a pydantic model.")
        description = inspect.getdoc(fn)
        if not description:
            raise TypeError(f"Tool {name!r} needs a docstring; it is the model-facing description.")
        _REGISTRY[name] = ToolSpec(
            name=name,
            title=title,
            description=description,
            input_model=input_model,
            output_model=output_model,
            fn=fn,  # type: ignore[arg-type]
            read_only=read_only,
            phase=phase,
        )
        return fn

    return decorate


def get_tool(name: str) -> ToolSpec | None:
    """Look up a registered tool by name."""
    return _REGISTRY.get(name)


def all_tools() -> list[ToolSpec]:
    """Every registered tool, ordered by phase then name."""
    return sorted(_REGISTRY.values(), key=lambda spec: (spec.phase, spec.name))


def load_all_tools() -> list[ToolSpec]:
    """Import every tool module so the decorators run, then return the registry."""
    from . import (  # noqa: F401
        captions,
        composition,
        core,
        grading,
        jobs,
        resize,
        transcription,
        vision,
    )
    from . import inspect as inspect_tools  # noqa: F401

    return all_tools()
