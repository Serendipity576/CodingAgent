from __future__ import annotations

from pathlib import Path
import re
import tempfile
from threading import Thread
from time import monotonic, sleep
import unittest
from unittest.mock import patch
import warnings

from starlette.exceptions import StarletteDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message="Using `httpx` with `starlette.testclient` is deprecated",
    category=StarletteDeprecationWarning,
)
from fastapi.testclient import TestClient

from agent.config import RuntimeLimits, Settings
from agent.llm.models import ToolCall
from agent.security.policy_types import Decision, PolicyDecision
from agent.security.risk import RiskLevel
from agent.web.app import _event_cursor, create_app
from agent.web.approval import WebApproval


class WebApprovalTests(unittest.TestCase):
    def test_browser_decision_resolves_only_the_matching_pending_approval(self) -> None:
        events: list[tuple[str, dict[str, object]]] = []
        approval = WebApproval(lambda event, details: events.append((event, dict(details))))
        decision = PolicyDecision(
            decision=Decision.REQUIRE_APPROVAL,
            risk=RiskLevel.HIGH,
            reason="requires confirmation",
            policy="test_policy",
        )
        result: list[bool] = []
        worker = Thread(
            target=lambda: result.append(
                approval.request(
                    ToolCall("call-1", "run_command", {"command": ["python", "-V"]}),
                    decision,
                )
            )
        )
        worker.start()
        approval_id = _wait_for_approval(events)

        self.assertFalse(approval.resolve("unknown", True))
        self.assertTrue(approval.resolve(approval_id, True))
        worker.join(timeout=2)

        self.assertEqual(result, [True])
        requested = next(details for event, details in events if event == "approval_required")
        self.assertEqual(requested["arguments"], {"command": ["python", "-V"]})


class WebApplicationTests(unittest.TestCase):
    def test_event_cursor_prefers_a_valid_browser_reconnect_id(self) -> None:
        """An EventSource reconnect resumes after its last confirmed journal item."""

        self.assertEqual(_event_cursor(3, "8"), 8)
        self.assertEqual(_event_cursor(8, "3"), 8)
        self.assertEqual(_event_cursor(3, "not-a-number"), 3)

    def test_web_configuration_is_redacted_and_session_api_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            settings = _settings(workspace)
            with patch("agent.conversation.build_llm_client", return_value=_FinalLLM()):
                with TestClient(create_app(settings, workspace)) as client:
                    page = client.get("/")
                    config = client.get("/api/config")
                    created = client.post("/api/conversations")
                    session = created.json()
                    queued = client.post(
                        f"/api/conversations/{session['conversation_id']}/messages",
                        json={"text": "Explain the workspace."},
                    )
                    _wait_for_finished_turn(
                        client.app.state.manager.get(session["conversation_id"]),
                    )

        self.assertEqual(page.status_code, 200)
        self.assertIn("Coding Agent", page.text)
        self.assertIn('<div id="root">', page.text)
        script = re.search(r'src="/static/(assets/[^\"]+\.js)"', page.text)
        self.assertIsNotNone(script)
        bundle = client.get(f"/static/{script.group(1)}")
        self.assertEqual(bundle.status_code, 200)
        self.assertIn("javascript", bundle.headers["content-type"])
        self.assertEqual(config.status_code, 200)
        self.assertNotIn("test-key", config.text)
        self.assertEqual(created.status_code, 201)
        self.assertEqual(queued.status_code, 202)

    def test_web_restores_and_deletes_a_local_session(self) -> None:
        """Keep the browser-visible metadata across app instances until deletion."""

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            settings = _settings(workspace)
            with patch(
                "agent.conversation.build_llm_client",
                side_effect=[_FinalLLM(), _FinalLLM()],
            ):
                with TestClient(create_app(settings, workspace)) as first_client:
                    created = first_client.post("/api/conversations")
                    conversation_id = created.json()["conversation_id"]

                with TestClient(create_app(settings, workspace)) as restored_client:
                    listed = restored_client.get("/api/conversations")
                    deleted = restored_client.delete(f"/api/conversations/{conversation_id}")
                    deleted_again = restored_client.delete(f"/api/conversations/{conversation_id}")
                    missing = restored_client.get(f"/api/conversations/{conversation_id}")

        self.assertEqual(listed.status_code, 200)
        self.assertEqual([item["conversation_id"] for item in listed.json()], [conversation_id])
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(deleted_again.status_code, 204)
        self.assertEqual(missing.status_code, 404)


class _FinalLLM:
    """Offline LLM used only to let a Web session complete its first turn."""

    def respond(self, **_: object):
        from agent.llm.models import ModelResponse

        return ModelResponse("response-1", "done", ())


def _wait_for_approval(events: list[tuple[str, dict[str, object]]]) -> str:
    """Wait briefly for the worker to publish its approval request."""

    deadline = monotonic() + 2
    while monotonic() < deadline:
        for event, details in events:
            if event == "approval_required":
                return str(details["approval_id"])
        sleep(0.01)
    raise AssertionError("approval request was not published")


def _wait_for_finished_turn(session: object | None) -> None:
    """Wait for the daemon worker before the temporary workspace is removed."""

    if session is None:
        raise AssertionError("created Web session is unavailable")
    deadline = monotonic() + 2
    sequence = 0
    while monotonic() < deadline:
        for item in session.events_after(sequence, timeout_seconds=0.1):
            sequence = item.sequence
            if item.event == "conversation_turn_finished":
                return
    raise AssertionError("Web conversation turn did not finish")


def _settings(workspace: Path) -> Settings:
    """Create a redaction-focused local Web configuration."""

    return Settings(
        workspace=workspace.resolve(),
        provider="responses",
        model="test-model",
        api_key="test-key",
        base_url="https://llm.example.test/v1",
        limits=RuntimeLimits(),
    )
