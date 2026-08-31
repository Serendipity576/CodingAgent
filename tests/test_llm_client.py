from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

from agent.config import RuntimeLimits, Settings
from agent.llm.client import (
    DeepSeekResponsesClient,
    LLMConfigurationError,
    OpenAIResponsesClient,
    ResponsesClient,
    _client_class_for_provider,
)
from agent.llm.models import ToolOutput, Usage


class RecordingResponses:
    """Minimal SDK replacement that records a request and returns a fixed reply."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    def create(self, **request: object) -> object:
        """Record the SDK call made by the adapter."""

        self.requests.append(request)
        return self.response


class ResponsesClientTests(unittest.TestCase):
    def test_task_client_requires_all_generic_connection_fields(self) -> None:
        settings = Settings(
            workspace=Path(".").resolve(),
            model=None,
            api_key=None,
            base_url=None,
            limits=RuntimeLimits(),
        )

        with self.assertRaisesRegex(LLMConfigurationError, "CODING_AGENT_API_KEY"):
            ResponsesClient(settings)

    def test_openai_adapter_adds_openai_options_and_parses_usage(self) -> None:
        sdk_response = SimpleNamespace(
            id="response-1",
            output_text="",
            usage=SimpleNamespace(input_tokens=12, output_tokens=7, total_tokens=19),
            output=[
                SimpleNamespace(
                    type="function_call",
                    call_id="call-1",
                    name="read_file",
                    arguments='{"path": "README.md"}',
                )
            ],
        )
        recorder = RecordingResponses(sdk_response)
        client = _recording_client(OpenAIResponsesClient, recorder, max_output_tokens=128)

        response = client.respond(
            instructions="Use tools.",
            task="Inspect the project.",
            tools=({"type": "function", "name": "read_file"},),
            tool_outputs=(),
        )

        self.assertEqual(response.response_id, "response-1")
        self.assertEqual(response.text, None)
        self.assertEqual(response.tool_calls[0].name, "read_file")
        self.assertEqual(response.tool_calls[0].arguments, {"path": "README.md"})
        self.assertEqual(response.usage, Usage(12, 7, 19))
        request = recorder.requests[0]
        self.assertEqual(request["model"], "test-model")
        self.assertEqual(request["input"], [{"role": "user", "content": "Inspect the project."}])
        self.assertFalse(request["stream"])
        self.assertEqual(request["max_output_tokens"], 128)
        self.assertFalse(request["store"])
        self.assertEqual(request["include"], ["reasoning.encrypted_content"])
        self.assertFalse(request["parallel_tool_calls"])
        self.assertNotIn("previous_response_id", request)

    def test_generic_adapter_uses_only_shared_responses_parameters(self) -> None:
        recorder = RecordingResponses(
            {"id": "response-1", "output_text": "Done.", "output": []}
        )
        client = _recording_client(ResponsesClient, recorder)

        response = client.respond(
            instructions="Complete the task.",
            task="Inspect the project.",
            tools=(),
            tool_outputs=(),
        )

        self.assertEqual(response.text, "Done.")
        request = recorder.requests[0]
        self.assertFalse(request["stream"])
        self.assertNotIn("store", request)
        self.assertNotIn("include", request)
        self.assertNotIn("parallel_tool_calls", request)
        self.assertNotIn("previous_response_id", request)

    def test_message_output_is_used_when_a_compatible_response_has_no_output_text_property(self) -> None:
        recorder = RecordingResponses(
            {
                "id": "response-1",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "Message fallback works."}
                        ],
                    }
                ],
            }
        )
        client = _recording_client(ResponsesClient, recorder)

        response = client.respond(
            instructions="Complete the task.",
            task="Inspect the project.",
            tools=(),
            tool_outputs=(),
        )

        self.assertEqual(response.text, "Message fallback works.")

    def test_deepseek_adapter_replays_local_history_without_openai_options(self) -> None:
        first_response = SimpleNamespace(
            id="response-1",
            output_text="",
            output=[
                SimpleNamespace(
                    type="reasoning",
                    id="reasoning-1",
                    encrypted_content="provider-continuation-data",
                ),
                SimpleNamespace(
                    type="function_call",
                    call_id="call-1",
                    name="read_file",
                    arguments='{"path": "README.md"}',
                ),
            ],
        )
        recorder = RecordingResponses(first_response)
        client = _recording_client(DeepSeekResponsesClient, recorder)

        client.respond(
            instructions="Use tools.",
            task="Inspect the project.",
            tools=({"type": "function", "name": "read_file"},),
            tool_outputs=(),
        )
        recorder.response = SimpleNamespace(id="response-2", output_text="Done.", output=[])
        client.respond(
            instructions="Use tools.",
            task=None,
            tools=({"type": "function", "name": "read_file"},),
            tool_outputs=(ToolOutput(call_id="call-1", output='{"ok": true}'),),
        )

        initial_request, follow_up_request = recorder.requests
        self.assertNotIn("previous_response_id", initial_request)
        self.assertNotIn("previous_response_id", follow_up_request)
        self.assertNotIn("store", follow_up_request)
        self.assertNotIn("include", follow_up_request)
        self.assertNotIn("parallel_tool_calls", follow_up_request)
        self.assertEqual(
            follow_up_request["input"],
            [
                {"role": "user", "content": "Inspect the project."},
                {
                    "type": "reasoning",
                    "id": "reasoning-1",
                    "encrypted_content": "provider-continuation-data",
                },
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "read_file",
                    "arguments": '{"path": "README.md"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": '{"ok": true}',
                },
            ],
        )

    def test_follow_up_user_message_appends_to_the_same_local_history(self) -> None:
        recorder = RecordingResponses(
            SimpleNamespace(
                id="response-1",
                output_text="First answer.",
                output=[
                    SimpleNamespace(
                        type="message",
                        role="assistant",
                        content=[{"type": "output_text", "text": "First answer."}],
                    )
                ],
            )
        )
        client = _recording_client(ResponsesClient, recorder)

        client.respond(
            instructions="Answer clearly.",
            task="First question.",
            tools=(),
            tool_outputs=(),
        )
        recorder.response = SimpleNamespace(id="response-2", output_text="Second answer.", output=[])
        client.respond(
            instructions="Answer clearly.",
            task="Follow-up question.",
            tools=(),
            tool_outputs=(),
        )

        self.assertEqual(
            recorder.requests[1]["input"],
            [
                {"role": "user", "content": "First question."},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "First answer."}],
                },
                {"role": "user", "content": "Follow-up question."},
            ],
        )

    def test_explicit_provider_selects_the_matching_internal_adapter(self) -> None:
        self.assertIs(
            _client_class_for_provider("openai"),
            OpenAIResponsesClient,
        )
        self.assertIs(
            _client_class_for_provider("deepseek"),
            DeepSeekResponsesClient,
        )
        self.assertIs(
            _client_class_for_provider("responses"),
            ResponsesClient,
        )

    def test_missing_provider_is_rejected_before_creating_a_client(self) -> None:
        with self.assertRaisesRegex(LLMConfigurationError, "CODING_AGENT_PROVIDER"):
            _client_class_for_provider(None)


def _recording_client(
    client_type: type[ResponsesClient],
    recorder: RecordingResponses,
    *,
    max_output_tokens: int | None = None,
) -> ResponsesClient:
    """Construct one adapter without importing or contacting the real SDK."""

    client = object.__new__(client_type)
    client._client = SimpleNamespace(responses=recorder)
    client._model = "test-model"
    client._max_output_tokens = max_output_tokens
    client._history = []
    return client
