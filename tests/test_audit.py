from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from agent.audit import AuditLogger
from agent.change_tracker import GitStatusSnapshot
from agent.llm.models import ToolCall
from agent.summary import build_task_summary
from agent.tools.base import ToolResult


class AuditLoggerTests(unittest.TestCase):
    def test_log_records_events_without_file_contents(self) -> None:
        """Audit records metadata and outcomes, but excludes patch and tool bodies."""

        patch_body = "API_KEY = 'do-not-log-this-source'\n"
        patch_call = ToolCall(
            call_id="call-1",
            name="apply_patch",
            arguments={
                "path": "src/example.py",
                "expected_text": "old implementation",
                "replacement_text": patch_body,
            },
        )
        patch_result = ToolResult.succeeded(
            "updated src/example.py",
            metadata={
                "path": "src/example.py",
                "change_type": "updated",
                "diff_summary": {"added_lines": 1, "removed_lines": 1},
            },
        )
        test_call = ToolCall(
            call_id="call-2",
            name="run_command",
            arguments={"command": ["python", "-m", "unittest", "-q"]},
        )
        test_result = ToolResult.succeeded("test output is intentionally omitted")
        blocked_call = ToolCall(
            call_id="call-3",
            name="run_command",
            arguments={"command": ["sudo", "whoami"]},
        )
        blocked_result = ToolResult.failed(
            "policy denied tool call",
            decision="deny",
            risk="critical",
            policy="command",
        )
        calls = (
            (patch_call, patch_result),
            (test_call, test_result),
            (blocked_call, blocked_result),
        )
        baseline = GitStatusSnapshot(True, (" M preexisting.py",))
        summary = build_task_summary(calls, baseline)

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            logger = AuditLogger.create(workspace, task_id="task-1")
            self.assertIsNotNone(logger)
            assert logger is not None
            logger.task_started("Update the example.", baseline)
            logger.tool_executed(
                step=1,
                call=patch_call,
                result=patch_result,
                duration_ms=12,
            )
            logger.tool_executed(
                step=2,
                call=test_call,
                result=test_result,
                duration_ms=34,
            )
            logger.task_finished(
                status="completed",
                message="Updated and tested.",
                steps=2,
                summary=summary,
            )
            content = logger.path.read_text(encoding="utf-8")

        events = [json.loads(line) for line in content.splitlines()]
        self.assertEqual([event["event"] for event in events], [
            "task_started",
            "tool_executed",
            "tool_executed",
            "task_finished",
        ])
        self.assertEqual(events[1]["arguments"]["replacement_text_chars"], len(patch_body))
        self.assertEqual(events[-1]["summary"]["modified_files"], ["src/example.py"])
        self.assertEqual(events[-1]["summary"]["modified_file_count"], 1)
        self.assertEqual(events[-1]["summary"]["tests"], "passed")
        self.assertEqual(events[-1]["summary"]["blocked_actions"], 1)
        self.assertEqual(events[-1]["summary"]["preexisting_git_changes"], [" M preexisting.py"])
        self.assertNotIn(patch_body, content)
        self.assertNotIn("test output is intentionally omitted", content)

    def test_summary_counts_approved_high_risk_actions(self) -> None:
        call = ToolCall(
            call_id="call-1",
            name="run_command",
            arguments={"command": ["rm", "temporary.txt"]},
        )
        result = ToolResult.succeeded(
            "exit code: 0",
            metadata={"approval": "approved", "exit_code": 0},
        )

        summary = build_task_summary(((call, result),), GitStatusSnapshot(True, ()))

        self.assertEqual(summary.approved_high_risk_actions, 1)
        self.assertEqual(summary.tests, "not_run")
