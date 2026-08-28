from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
import tempfile
import unittest

from agent.agent import CodingAgent, TaskStatus
from agent.config import RuntimeLimits, Settings
from agent.llm.models import ModelResponse, ToolCall, ToolOutput
from agent.tools import build_default_registry


class ScriptedLLM:
    """Return predetermined turns while retaining the runtime's requests."""

    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self._responses = iter(responses)
        self.requests: list[dict[str, object]] = []

    def respond(
        self,
        *,
        instructions: str,
        task: str | None,
        tools: Sequence[Mapping[str, object]],
        previous_response_id: str | None,
        tool_outputs: Sequence[ToolOutput],
    ) -> ModelResponse:
        self.requests.append(
            {
                "instructions": instructions,
                "task": task,
                "tools": tools,
                "previous_response_id": previous_response_id,
                "tool_outputs": tool_outputs,
            }
        )
        return next(self._responses)


def _settings(workspace: Path, *, max_consecutive_failures: int = 2) -> Settings:
    return Settings(
        workspace=workspace.resolve(),
        model="test-model",
        api_key=None,
        base_url=None,
        limits=RuntimeLimits(
            max_steps=10,
            command_timeout_seconds=10,
            max_output_chars=2_000,
            max_task_seconds=30,
            max_consecutive_tool_failures=max_consecutive_failures,
        ),
    )


class CodingAgentTests(unittest.TestCase):
    def test_agent_repairs_a_failing_test_in_a_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "app.py").write_text(
                "def add(left, right):\n    return left - right\n", encoding="utf-8"
            )
            (workspace / "test_app.py").write_text(
                "import unittest\n\n"
                "from app import add\n\n\n"
                "class AddTests(unittest.TestCase):\n"
                "    def test_add(self):\n"
                "        self.assertEqual(add(2, 3), 5)\n",
                encoding="utf-8",
            )
            llm = ScriptedLLM(
                [
                    _tool_response(
                        "response-1",
                        "call-1",
                        "run_command",
                        {"command": [sys.executable, "-m", "unittest", "-q"]},
                    ),
                    _tool_response(
                        "response-2", "call-2", "read_file", {"path": "app.py"}
                    ),
                    _tool_response(
                        "response-3",
                        "call-3",
                        "apply_patch",
                        {
                            "path": "app.py",
                            "expected_text": "return left - right",
                            "replacement_text": "return left + right + 0",
                        },
                    ),
                    _tool_response(
                        "response-4",
                        "call-4",
                        "run_command",
                        {"command": [sys.executable, "-m", "unittest", "-q"]},
                    ),
                    ModelResponse("response-5", "Fixed the add implementation and verified tests.", ()),
                ]
            )

            result = CodingAgent(
                settings=_settings(workspace),
                llm=llm,
                tools=build_default_registry(),
            ).run("Fix the failing test.")

            self.assertEqual(result.status, TaskStatus.COMPLETED)
            self.assertEqual(result.steps, 4)
            self.assertFalse(result.tool_calls[0].result.success)
            self.assertTrue(result.tool_calls[2].result.success)
            self.assertTrue(
                result.tool_calls[-1].result.success,
                (result.tool_calls[-1].result.error or "")
                + "\n"
                + result.tool_calls[-1].result.output,
            )
            self.assertIn("return left + right", (workspace / "app.py").read_text())
            self.assertIsNone(llm.requests[0]["previous_response_id"])
            self.assertEqual(llm.requests[1]["previous_response_id"], "response-1")

    def test_repeated_failed_call_stops_the_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            llm = ScriptedLLM(
                [
                    _tool_response(
                        "response-1", "call-1", "read_file", {"path": "missing.py"}
                    ),
                    _tool_response(
                        "response-2", "call-2", "read_file", {"path": "missing.py"}
                    ),
                ]
            )

            result = CodingAgent(
                settings=_settings(workspace),
                llm=llm,
                tools=build_default_registry(),
            ).run("Read the missing file.")

            self.assertEqual(result.status, TaskStatus.REPEATED_TOOL_FAILURE)
            self.assertEqual(result.steps, 2)
            self.assertEqual(len(result.tool_calls), 2)


def _tool_response(
    response_id: str,
    call_id: str,
    name: str,
    arguments: Mapping[str, object],
) -> ModelResponse:
    return ModelResponse(
        response_id=response_id,
        text="",
        tool_calls=(ToolCall(call_id=call_id, name=name, arguments=arguments),),
    )
