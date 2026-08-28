from __future__ import annotations

from types import SimpleNamespace
import unittest

from agent.llm.client import OpenAIResponsesClient
from agent.llm.models import ToolOutput


class RecordingResponses:
    """Minimal SDK replacement that exposes the request made by the adapter."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    def create(self, **request: object) -> object:
        self.requests.append(request)
        return self.response


class OpenAIResponsesClientTests(unittest.TestCase):
    def test_adapter_builds_initial_request_and_parses_function_call(self) -> None:
        sdk_response = SimpleNamespace(
            id="response-1",
            output_text="",
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
        client = object.__new__(OpenAIResponsesClient)
        client._client = SimpleNamespace(responses=recorder)
        client._model = "test-model"

        response = client.respond(
            instructions="Use tools.",
            task="Inspect the project.",
            tools=({"type": "function", "name": "read_file"},),
            previous_response_id=None,
            tool_outputs=(),
        )

        self.assertEqual(response.response_id, "response-1")
        self.assertEqual(response.tool_calls[0].name, "read_file")
        self.assertEqual(response.tool_calls[0].arguments, {"path": "README.md"})
        request = recorder.requests[0]
        self.assertEqual(request["model"], "test-model")
        self.assertEqual(request["input"], [{"role": "user", "content": "Inspect the project."}])
        self.assertFalse(request["parallel_tool_calls"])

    def test_adapter_sends_function_output_on_follow_up_turn(self) -> None:
        sdk_response = SimpleNamespace(id="response-2", output_text="Done.", output=[])
        recorder = RecordingResponses(sdk_response)
        client = object.__new__(OpenAIResponsesClient)
        client._client = SimpleNamespace(responses=recorder)
        client._model = "test-model"

        client.respond(
            instructions="Use tools.",
            task=None,
            tools=(),
            previous_response_id="response-1",
            tool_outputs=(ToolOutput(call_id="call-1", output='{"ok": true}'),),
        )

        request = recorder.requests[0]
        self.assertEqual(request["previous_response_id"], "response-1")
        self.assertEqual(
            request["input"],
            [
                {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": '{"ok": true}',
                }
            ],
        )
