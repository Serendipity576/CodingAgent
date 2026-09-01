from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
import sqlite3
import tempfile
from time import monotonic
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.config import RuntimeLimits, Settings
from agent.conversation import ConversationManager, ConversationState
from agent.conversation_store import ConversationStore
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


class DurableRecordingLLM(RecordingLLM):
    """Test client whose locally owned transcript can survive a manager restart."""

    def __init__(self) -> None:
        super().__init__()
        self._history: list[dict[str, object]] = []
        self.restored_history: list[dict[str, object]] | None = None

    def respond(
        self,
        *,
        instructions: str,
        task: str | None,
        tools: Sequence[Mapping[str, object]],
        tool_outputs: Sequence[ToolOutput],
    ) -> ModelResponse:
        """Append enough safe model state to verify continuation after recovery."""

        self.requests.append({"instructions": instructions, "task": task})
        if task is not None:
            self._history.append({"role": "user", "content": task})
        response_id = f"response-{len(self.requests)}"
        text = f"answer-{len(self.requests)}"
        self._history.append(
            {
                "type": "message",
                "id": response_id,
                "content": [{"type": "output_text", "text": text}],
            }
        )
        return ModelResponse(response_id=response_id, text=text, tool_calls=())

    def export_history(self) -> list[dict[str, object]]:
        """Return a copy so assertions cannot mutate the fake client state."""

        return deepcopy(self._history)

    def restore_history(self, history: Sequence[Mapping[str, object]]) -> None:
        """Record and restore the exact persisted transcript for assertions."""

        self._history = [dict(item) for item in history]
        self.restored_history = deepcopy(self._history)


class ContextDurableRecordingLLM(DurableRecordingLLM):
    """Durable fake that also persists context metadata beside its raw transcript."""

    def __init__(self) -> None:
        super().__init__()
        self.context_state: dict[str, object] = {
            "version": 1,
            "summary": {"current_goal": "继续会话"},
            "summary_version": 1,
            "covered_items": 2,
            "artifacts": [],
            "token_multiplier": 1.0,
        }
        self.restored_context_state: dict[str, object] | None = None

    def export_context_state(self) -> dict[str, object]:
        """Return a copy so persisted context metadata is never caller-owned."""

        return deepcopy(self.context_state)

    def restore_context_state(self, state: Mapping[str, object]) -> None:
        """Record the recovered state for the restart assertion."""

        self.context_state = dict(state)
        self.restored_context_state = deepcopy(self.context_state)

    def context_status(self) -> dict[str, object]:
        """Expose only compact context metadata to a session snapshot."""

        return {"summary_version": self.context_state.get("summary_version", 0)}


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

    def test_client_message_id_deduplicates_a_browser_retry(self) -> None:
        """A lost browser response must not enqueue an equivalent second turn."""

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            llm = RecordingLLM()
            with patch("agent.conversation.build_llm_client", return_value=llm):
                manager = ConversationManager(_settings(workspace), workspace)
                session = manager.create()
                self.assertTrue(session.submit("First message.", client_message_id="browser-1"))
                self.assertTrue(session.submit("Repeated message.", client_message_id="browser-1"))
                _wait_for_finished_turn(session, 1)

        user_events = [
            item for item in session.events_after(0, timeout_seconds=0)
            if item.event == "user_message"
        ]
        self.assertEqual([request["task"] for request in llm.requests], ["First message."])
        self.assertEqual(len(user_events), 1)
        self.assertEqual(user_events[0].details["client_message_id"], "browser-1")

    def test_session_restores_local_history_and_can_be_deleted(self) -> None:
        """Persist one completed turn, resume it in a new manager, then erase it."""

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            first_llm = DurableRecordingLLM()
            restored_llm = DurableRecordingLLM()
            with patch(
                "agent.conversation.build_llm_client",
                side_effect=[first_llm, restored_llm],
            ):
                manager = ConversationManager(_settings(workspace), workspace)
                session = manager.create()
                self.assertTrue(session.submit("First message."))
                _wait_for_finished_turn(session, 1)
                conversation_id = session.id
                saved_history = first_llm.export_history()
                database = workspace / ".agent" / "conversations" / "sessions.sqlite3"

                restored_manager = ConversationManager(_settings(workspace), workspace)
                restored = restored_manager.get(conversation_id)
                self.assertIsNotNone(restored)
                assert restored is not None
                self.assertEqual(restored.state, ConversationState.IDLE)
                self.assertEqual(restored_llm.restored_history, saved_history)
                self.assertTrue(restored.submit("Follow-up message."))
                _wait_for_finished_turn(restored, 2)
                self.assertTrue(restored_manager.delete(conversation_id))

            self.assertTrue(database.exists())
            self.assertNotIn(b"test-key", database.read_bytes())
            self.assertEqual(ConversationStore(workspace).load_conversations(), ())

        self.assertEqual([request["task"] for request in first_llm.requests], ["First message."])
        self.assertEqual([request["task"] for request in restored_llm.requests], ["Follow-up message."])

    def test_running_session_becomes_interrupted_without_replaying_work(self) -> None:
        """Never rerun a persisted active turn after a process restart."""

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            store = ConversationStore(workspace)
            store.save_conversation(
                conversation_id="interrupted-session",
                created_at=1.0,
                state=ConversationState.RUNNING.value,
                turn_count=1,
                max_turns=4,
                max_history_items=20,
                transcript=(),
                latest_result=None,
            )
            store.append_event(
                "interrupted-session",
                sequence=1,
                event="conversation_turn_started",
                timestamp=1.0,
                details={"turn_id": 1},
            )
            llm = DurableRecordingLLM()
            with patch("agent.conversation.build_llm_client", return_value=llm):
                manager = ConversationManager(_settings(workspace), workspace)
                session = manager.get("interrupted-session")
                self.assertIsNotNone(session)
                assert session is not None
                self.assertEqual(session.state, ConversationState.INTERRUPTED)
                self.assertEqual(llm.requests, [])
                events = session.events_after(0, timeout_seconds=0)
                self.assertEqual(events[-1].event, "conversation_interrupted")
                self.assertTrue(session.submit("Continue after review."))
                _wait_for_finished_turn(session, 2)

    def test_session_restores_context_metadata_separately_from_raw_history(self) -> None:
        """Summary metadata survives restart without replacing the complete transcript."""

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            first_llm = ContextDurableRecordingLLM()
            restored_llm = ContextDurableRecordingLLM()
            with patch(
                "agent.conversation.build_llm_client",
                side_effect=[first_llm, restored_llm],
            ):
                manager = ConversationManager(_settings(workspace), workspace)
                session = manager.create()
                self.assertTrue(session.submit("First message."))
                _wait_for_finished_turn(session, 1)
                restored_manager = ConversationManager(_settings(workspace), workspace)
                restored = restored_manager.get(session.id)

        self.assertIsNotNone(restored)
        self.assertEqual(restored_llm.restored_context_state, first_llm.context_state)

    def test_store_migrates_existing_sessions_without_context_metadata(self) -> None:
        """Older local databases gain the new column without losing their transcript."""

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            database_dir = workspace / ".agent" / "conversations"
            database_dir.mkdir(parents=True)
            database = database_dir / "sessions.sqlite3"
            with sqlite3.connect(database) as connection:
                connection.executescript(
                    """
                    CREATE TABLE conversations (
                        conversation_id TEXT PRIMARY KEY,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        state TEXT NOT NULL,
                        turn_count INTEGER NOT NULL,
                        max_turns INTEGER NOT NULL,
                        max_history_items INTEGER NOT NULL,
                        transcript_json TEXT NOT NULL,
                        latest_result_json TEXT
                    );
                    INSERT INTO conversations VALUES (
                        'legacy-session', 1, 1, 'idle', 1, 4, 20, '[]', NULL
                    );
                    """
                )

            loaded = ConversationStore(workspace).load_conversations()

        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].conversation_id, "legacy-session")
        self.assertEqual(loaded[0].context_state, {})


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
