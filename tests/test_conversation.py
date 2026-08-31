from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import tempfile
from time import monotonic
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.config import RuntimeLimits, Settings
from agent.conversation import ConversationManager, ConversationState
from agent.llm.models import ModelResponse, ToolOutput


class RecordingLLM:
    """Return final messages while retaining every user message passed by a session."""

    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def respond(
        self,
        *,
        instructions: str,
        task: str | None,
        tools: Sequence[Mapping[str, object]],
        tool_outputs: Sequence[ToolOutput],
    ) -> ModelResponse:
        self.requests.append({"instructions": instructions, "task": task})
        return ModelResponse(
            response_id=f"response-{len(self.requests)}",
            text=f"answer-{len(self.requests)}",
            tool_calls=(),
        )


class ConversationTests(unittest.TestCase):
    def test_session_serializes_two_user_turns_with_one_llm_client(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            llm = RecordingLLM()
            with patch("agent.conversation.build_llm_client", return_value=llm):
                manager = ConversationManager(_settings(workspace), workspace)
                session = manager.create()
                self.assertTrue(session.submit("First message."))
                _wait_for_finished_turn(session, 1)
                self.assertTrue(session.submit("Follow-up message."))
                _wait_for_finished_turn(session, 2)

        self.assertEqual([request["task"] for request in llm.requests], ["First message.", "Follow-up message."])
        self.assertEqual(session.state, ConversationState.IDLE)
        events = [item.event for item in session.events_after(0, timeout_seconds=0)]
        self.assertIn("conversation_turn_finished", events)

    def test_turn_limit_rejects_additional_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with patch("agent.conversation.build_llm_client", return_value=RecordingLLM()):
                manager = ConversationManager(_settings(workspace), workspace, max_turns=1)
                session = manager.create()
                self.assertTrue(session.submit("Only allowed message."))
                _wait_for_finished_turn(session, 1)
                self.assertFalse(session.submit("Rejected message."))

        self.assertEqual(session.state, ConversationState.LIMIT_REACHED)


def _wait_for_finished_turn(session: object, turn_id: int) -> None:
    """Wait for a worker event without relying on scheduler timing in assertions."""

    deadline = monotonic() + 3
    sequence = 0
    while monotonic() < deadline:
        for item in session.events_after(sequence, timeout_seconds=0.1):
            sequence = item.sequence
            if item.event == "conversation_turn_finished" and item.details["turn_id"] == turn_id:
                return
    raise AssertionError(f"turn {turn_id} did not finish")


def _settings(workspace: Path) -> Settings:
    """Return offline settings because the LLM constructor is patched in these tests."""

    return Settings(
        workspace=workspace.resolve(),
        model="test-model",
        api_key="test-key",
        base_url="https://llm.example.test/v1",
        provider="responses",
        limits=RuntimeLimits(max_steps=4, command_timeout_seconds=5, max_task_seconds=10),
    )
