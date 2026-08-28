"""OpenAI Responses API adapter.

The runtime depends on the small data types in ``agent.llm.models`` rather than
the SDK's response classes. This keeps provider-specific parsing at one boundary
and makes the agent loop straightforward to test with a fake client.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Protocol

from agent.config import Settings
from agent.llm.models import ModelResponse, ToolCall, ToolOutput


class LLMConfigurationError(ValueError):
    """Raised before a request when local LLM configuration is incomplete."""


class LLMRequestError(RuntimeError):
    """Raised when the provider rejects or cannot complete a request."""


class LLMClient(Protocol):
    """The provider boundary consumed by the runtime and its tests."""

    def respond(
        self,
        *,
        instructions: str,
        task: str | None,
        tools: Sequence[Mapping[str, object]],
        previous_response_id: str | None,
        tool_outputs: Sequence[ToolOutput],
    ) -> ModelResponse:
        """Request the next model turn."""


class OpenAIResponsesClient:
    """Call the OpenAI Responses API using custom function tools."""

    def __init__(self, settings: Settings) -> None:
        if not settings.api_key:
            raise LLMConfigurationError(
                "OPENAI_API_KEY must be set before an agent task can run"
            )

        # Keep the SDK import lazy: configuration checks and unit tests should
        # work even when the optional runtime dependency is not installed yet.
        try:
            from openai import OpenAI
        except ImportError as error:
            raise LLMConfigurationError(
                "OpenAI SDK is missing; install project dependencies before running a task"
            ) from error

        client_options: dict[str, str] = {"api_key": settings.api_key}
        if settings.base_url:
            client_options["base_url"] = settings.base_url
        self._client = OpenAI(**client_options)
        self._model = settings.model

    def respond(
        self,
        *,
        instructions: str,
        task: str | None,
        tools: Sequence[Mapping[str, object]],
        previous_response_id: str | None,
        tool_outputs: Sequence[ToolOutput],
    ) -> ModelResponse:
        """Create one response and normalize its function-call output."""

        if previous_response_id is None:
            if task is None:
                raise LLMRequestError("an initial model request requires a task")
            input_items: list[dict[str, object]] = [{"role": "user", "content": task}]
        else:
            # Function-call outputs are the only new inputs after the initial
            # task. ``previous_response_id`` carries the preceding context.
            input_items = [
                {
                    "type": "function_call_output",
                    "call_id": tool_output.call_id,
                    "output": tool_output.output,
                }
                for tool_output in tool_outputs
            ]

        request: dict[str, object] = {
            "model": self._model,
            "instructions": instructions,
            "input": input_items,
            "tools": list(tools),
            # The runtime executes tools serially so it can account for every
            # side effect and stop at its configured limits.
            "parallel_tool_calls": False,
        }
        if previous_response_id is not None:
            request["previous_response_id"] = previous_response_id

        try:
            response = self._client.responses.create(**request)
        except Exception as error:  # SDK exceptions vary by installed version.
            raise LLMRequestError(f"model request failed: {error}") from error
        return _parse_response(response)


def _parse_response(response: object) -> ModelResponse:
    """Translate the SDK response without leaking provider types upstream."""

    response_id = str(getattr(response, "id", ""))
    if not response_id:
        raise LLMRequestError("model response did not contain an id")

    tool_calls: list[ToolCall] = []
    for item in getattr(response, "output", ()):
        if getattr(item, "type", None) != "function_call":
            continue
        tool_calls.append(_parse_tool_call(item))

    text = str(getattr(response, "output_text", "") or "")
    return ModelResponse(
        response_id=response_id,
        text=text,
        tool_calls=tuple(tool_calls),
    )


def _parse_tool_call(item: object) -> ToolCall:
    """Parse function-call arguments while preserving malformed JSON as data."""

    call_id = str(getattr(item, "call_id", ""))
    name = str(getattr(item, "name", ""))
    raw_arguments = getattr(item, "arguments", "{}")

    try:
        parsed_arguments = json.loads(raw_arguments)
    except (TypeError, json.JSONDecodeError) as error:
        return ToolCall(
            call_id=call_id,
            name=name,
            arguments=None,
            arguments_error=f"tool arguments are not valid JSON: {error}",
        )

    if not isinstance(parsed_arguments, dict):
        return ToolCall(
            call_id=call_id,
            name=name,
            arguments=None,
            arguments_error="tool arguments must be a JSON object",
        )
    return ToolCall(call_id=call_id, name=name, arguments=parsed_arguments)
