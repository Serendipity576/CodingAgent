from __future__ import annotations

from pathlib import Path
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
from agent.web.app import create_app
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

        self.assertEqual(page.status_code, 200)
        self.assertIn("Coding Agent", page.text)
        self.assertEqual(config.status_code, 200)
        self.assertNotIn("test-key", config.text)
        self.assertEqual(created.status_code, 201)
        self.assertEqual(queued.status_code, 202)


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
