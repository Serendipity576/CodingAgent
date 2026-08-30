from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
import tempfile
import unittest

from agent.agent import CodingAgent, TaskStatus
from agent.change_tracker import GitStatusSnapshot
from agent.config import RuntimeLimits, Settings
from agent.llm.models import ModelResponse, ToolCall, ToolOutput
from agent.tools import build_default_registry
from agent.tools.base import ToolContext
from agent.tools.shell import RunCommandTool


class SingleResponseLLM:
    """Return one predetermined response for a runtime-limit test."""

    def __init__(self, response: ModelResponse) -> None:
        self._response = response

    def respond(
        self,
        *,
        instructions: str,
        task: str | None,
        tools: Sequence[Mapping[str, object]],
        previous_response_id: str | None,
        tool_outputs: Sequence[ToolOutput],
    ) -> ModelResponse:
        return self._response


class RuntimeBoundaryTests(unittest.TestCase):
    def test_command_timeout_is_reported_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            result = RunCommandTool().execute(
                {
                    "command": [
                        sys.executable,
                        "-c",
                        "import time; time.sleep(2)",
                    ]
                },
                _context(workspace, command_timeout_seconds=1),
            )

        self.assertFalse(result.success)
        self.assertEqual(result.error, "command timed out")
        self.assertTrue(result.metadata["timed_out"])

    def test_command_output_obeys_the_configured_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            result = RunCommandTool().execute(
                {"command": [sys.executable, "-c", "print('x' * 200)"]},
                _context(workspace, max_output_chars=60),
            )

        self.assertTrue(result.success)
        self.assertLessEqual(len(result.output), 60)
        self.assertIn("truncated", result.output)

    def test_invalid_tool_arguments_are_denied_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            result = build_default_registry(workspace).execute(
                ToolCall(
                    call_id="invalid-call",
                    name="read_file",
                    arguments=None,
                    arguments_error="malformed JSON",
                ),
                _context(workspace),
            )

        self.assertFalse(result.success)
        self.assertEqual(result.decision, "deny")
        self.assertEqual(result.policy, "invalid_arguments")

    def test_agent_stops_before_executing_a_call_past_max_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            llm = SingleResponseLLM(
                ModelResponse(
                    response_id="response-1",
                    text="",
                    tool_calls=(
                        ToolCall("call-1", "list_files", {"path": "."}),
                        ToolCall("call-2", "list_files", {"path": "."}),
                    ),
                )
            )
            result = CodingAgent(
                settings=_settings(workspace, max_steps=1),
                llm=llm,
                tools=build_default_registry(workspace),
                git_baseline=GitStatusSnapshot.unavailable("test baseline"),
            ).run("Inspect the workspace.")

        self.assertEqual(result.status, TaskStatus.MAX_STEPS_REACHED)
        self.assertEqual(result.steps, 1)
        self.assertEqual(len(result.tool_calls), 1)


def _context(
    workspace: Path,
    *,
    command_timeout_seconds: int = 10,
    max_output_chars: int = 2_000,
) -> ToolContext:
    return ToolContext(
        workspace=workspace.resolve(),
        limits=RuntimeLimits(
            max_steps=10,
            command_timeout_seconds=command_timeout_seconds,
            max_output_chars=max_output_chars,
            max_task_seconds=30,
        ),
    )


def _settings(workspace: Path, *, max_steps: int) -> Settings:
    return Settings(
        workspace=workspace.resolve(),
        model="test-model",
        api_key=None,
        base_url=None,
        limits=RuntimeLimits(
            max_steps=max_steps,
            command_timeout_seconds=10,
            max_output_chars=2_000,
            max_task_seconds=30,
        ),
    )
