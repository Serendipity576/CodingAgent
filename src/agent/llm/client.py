"""Provider adapters for OpenAI-compatible Responses APIs.

The runtime sees only ``LLMClient`` and provider-neutral model objects.  This
module owns the complete transcript for a task, so a request never relies on a
provider-side conversation or a previous response identifier.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Protocol

from agent.config import Settings
from agent.llm.models import ModelResponse, ToolCall, ToolOutput, Usage


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
        tool_outputs: Sequence[ToolOutput],
    ) -> ModelResponse:
        """Request the next model turn from locally supplied context."""


class ResponsesClient:
    """Call a Responses-compatible endpoint using only shared request fields."""

    def __init__(self, settings: Settings) -> None:
        if not settings.api_key or not settings.base_url or not settings.model:
            raise LLMConfigurationError(
                "CODING_AGENT_API_KEY, CODING_AGENT_BASE_URL, and CODING_AGENT_MODEL "
                "must be set before an agent task can run"
            )

        try:
            from openai import OpenAI
        except ImportError as error:
            raise LLMConfigurationError(
                "OpenAI SDK is missing; install project dependencies before running a task"
            ) from error

        self._client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
        self._model = settings.model
        self._max_output_tokens = settings.max_output_tokens
        # This is the sole conversation state.  It contains the user's task,
        # every provider output item, and each subsequent tool observation.
        self._history: list[dict[str, object]] = []

    def respond(
        self,
        *,
        instructions: str,
        task: str | None,
        tools: Sequence[Mapping[str, object]],
        tool_outputs: Sequence[ToolOutput],
    ) -> ModelResponse:
        """Create one stateless response from the client-owned transcript."""

        input_items = self._build_input(task, tool_outputs)
        request: dict[str, object] = {
            "model": self._model,
            "instructions": instructions,
            "input": input_items,
            "tools": list(tools),
            # Synchronous calls make each tool-calling turn explicit to the
            # bounded runtime and avoid exposing a streaming protocol upstream.
            "stream": False,
        }
        if self._max_output_tokens is not None:
            request["max_output_tokens"] = self._max_output_tokens
        request.update(self._provider_request_options())

        try:
            response = self._client.responses.create(**request)
        except Exception as error:  # SDK exceptions vary by installed version.
            raise LLMRequestError(f"model request failed: {error}") from error

        model_response = _parse_response(response)
        # Preserve every output item, including reasoning data required by a
        # provider to continue safely on the next independently sent request.
        self._history = [*input_items, *_response_input_items(response)]
        return model_response

    def _build_input(
        self,
        task: str | None,
        tool_outputs: Sequence[ToolOutput],
    ) -> list[dict[str, object]]:
        """Start a task or append tool observations to the local transcript."""

        if task is not None:
            # A non-empty history means a user is continuing the same local
            # conversation after an earlier Agent turn has finished.
            return [*self._history, {"role": "user", "content": task}]
        if not self._history:
            raise LLMRequestError("follow-up model request has no prior task context")
        return [
            *self._history,
            *[
                {
                    "type": "function_call_output",
                    "call_id": tool_output.call_id,
                    "output": tool_output.output,
                }
                for tool_output in tool_outputs
            ],
        ]

    def _provider_request_options(self) -> dict[str, object]:
        """Return optional parameters supported by this endpoint adapter."""

        return {}


class OpenAIResponsesClient(ResponsesClient):
    """Use OpenAI-only controls while keeping the transcript local."""

    def _provider_request_options(self) -> dict[str, object]:
        """Disable storage and retain encrypted reasoning for replay."""

        return {
            # The complete transcript remains in this client.  Do not create a
            # server-side conversation or rely on a previous response id.
            "store": False,
            "include": ["reasoning.encrypted_content"],
            # Serial execution is also enforced by the runtime; this avoids
            # asking OpenAI to return calls intended for parallel execution.
            "parallel_tool_calls": False,
        }


class DeepSeekResponsesClient(ResponsesClient):
    """DeepSeek adapter that deliberately omits OpenAI-only request options."""


def build_llm_client(settings: Settings) -> LLMClient:
    """Create the explicitly configured endpoint adapter."""

    return _client_class_for_provider(settings.provider)(settings)


def _client_class_for_provider(provider: str | None) -> type[ResponsesClient]:
    """Choose request behavior from the local provider configuration."""

    if provider == "openai":
        return OpenAIResponsesClient
    if provider == "deepseek":
        return DeepSeekResponsesClient
    if provider == "responses":
        return ResponsesClient
    raise LLMConfigurationError(
        "CODING_AGENT_PROVIDER must be set to openai, deepseek, or responses"
    )


def _parse_response(response: object) -> ModelResponse:
    """Translate one provider response without leaking SDK types upstream."""

    response_id = str(_field(response, "id", ""))
    if not response_id:
        raise LLMRequestError("model response did not contain an id")

    tool_calls: list[ToolCall] = []
    for item in _output_items(response):
        if _field(item, "type") != "function_call":
            continue
        tool_calls.append(_parse_tool_call(item))

    text = _response_text(response)
    return ModelResponse(
        response_id=response_id,
        text=text,
        tool_calls=tuple(tool_calls),
        usage=_parse_usage(_field(response, "usage")),
    )


def _response_text(response: object) -> str | None:
    """Read SDK convenience text or standard message content as a fallback."""

    text_value = _field(response, "output_text", "")
    if text_value:
        return str(text_value)

    fragments: list[str] = []
    for item in _output_items(response):
        if _field(item, "type") != "message":
            continue
        content = _field(item, "content", ())
        if not isinstance(content, Sequence) or isinstance(content, str):
            continue
        for part in content:
            if _field(part, "type") != "output_text":
                continue
            value = _field(part, "text")
            if value:
                fragments.append(str(value))
    return "".join(fragments) or None


def _parse_usage(raw_usage: object | None) -> Usage | None:
    """Normalize token accounting when the endpoint supplies it."""

    if raw_usage is None:
        return None
    input_tokens = _nonnegative_int(_field(raw_usage, "input_tokens"))
    output_tokens = _nonnegative_int(_field(raw_usage, "output_tokens"))
    total_tokens = _nonnegative_int(_field(raw_usage, "total_tokens"))
    if input_tokens is None or output_tokens is None:
        return None
    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens if total_tokens is not None else input_tokens + output_tokens,
    )


def _nonnegative_int(value: object | None) -> int | None:
    """Accept integer-like provider counters but reject invalid token values."""

    if isinstance(value, bool):
        return None
    try:
        parsed = int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return parsed if parsed is not None and parsed >= 0 else None


def _response_input_items(response: object) -> list[dict[str, object]]:
    """Serialize all returned items for the next locally managed request."""

    items: list[dict[str, object]] = []
    for item in _output_items(response):
        serialized = _serialize_output_item(item)
        if serialized is not None:
            items.append(serialized)
    return items


def _output_items(response: object) -> Sequence[object]:
    """Read output items from an SDK object or a mapping-based provider reply."""

    output = _field(response, "output", ())
    return output if isinstance(output, Sequence) and not isinstance(output, str) else ()


def _serialize_output_item(item: object) -> dict[str, object] | None:
    """Keep raw provider fields so future turns can replay their exact context."""

    dump = _field(item, "model_dump")
    if callable(dump):
        serialized = dump(mode="json", exclude_none=True)
        return serialized if isinstance(serialized, dict) else None
    if isinstance(item, Mapping):
        return dict(item)
    attributes = vars(item) if hasattr(item, "__dict__") else None
    return dict(attributes) if isinstance(attributes, dict) else None


def _parse_tool_call(item: object) -> ToolCall:
    """Parse function-call arguments while preserving malformed JSON as data."""

    call_id = str(_field(item, "call_id", ""))
    name = str(_field(item, "name", ""))
    raw_arguments = _field(item, "arguments", "{}")

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


def _field(source: object, field_name: str, default: object | None = None) -> object | None:
    """Read one field from either an SDK model or JSON-compatible mapping."""

    if isinstance(source, Mapping):
        return source.get(field_name, default)
    return getattr(source, field_name, default)
