"""Small, provider-independent objects used by the agent runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool invocation requested by the model.

    Invalid JSON is represented explicitly instead of being silently converted.
    The registry can then return a useful observation for the model to correct.
    """

    call_id: str
    name: str
    arguments: Mapping[str, object] | None
    arguments_error: str | None = None


@dataclass(frozen=True, slots=True)
class ToolOutput:
    """The result sent back for one model-issued function call."""

    call_id: str
    output: str


@dataclass(frozen=True, slots=True)
class Usage:
    """Provider-neutral token accounting for one completed model response."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Normalized model output for one agent turn."""

    response_id: str
    text: str | None
    tool_calls: tuple[ToolCall, ...]
    usage: Usage | None = None
