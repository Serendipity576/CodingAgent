"""Append-only JSONL audit logging without storing repository file contents."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from agent.change_tracker import GitStatusSnapshot
from agent.llm.models import ToolCall
from agent.summary import TaskSummary
from agent.tools.base import ToolResult


class AuditLogger:
    """Write minimal, structured events for one task to ``.agent/logs``."""

    def __init__(self, path: Path, task_id: str) -> None:
        self.path = path
        self.task_id = task_id

    @classmethod
    def create(cls, workspace: Path, *, task_id: str | None = None) -> "AuditLogger | None":
        """Create a task log when the workspace permits it, else stay nonfatal."""

        identifier = task_id or uuid4().hex
        path = workspace / ".agent" / "logs" / f"{identifier}.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=False)
        except OSError:
            return None
        return cls(path=path, task_id=identifier)

    def task_started(self, task: str, git_baseline: GitStatusSnapshot) -> None:
        """Record initial task and baseline state before any tool is executed."""

        self._write(
            "task_started",
            {
                "task": task,
                "git_baseline_available": git_baseline.available,
                "preexisting_git_changes": list(git_baseline.entries),
                "git_baseline_error": git_baseline.error,
            },
        )

    def tool_executed(
        self, *, step: int, call: ToolCall, result: ToolResult, duration_ms: int
    ) -> None:
        """Record authorization and execution facts without logging tool output."""

        self._write(
            "tool_executed",
            {
                "step": step,
                "tool": call.name,
                "arguments": _argument_summary(call),
                "success": result.success,
                "error": result.error,
                "risk": result.risk,
                "decision": result.decision,
                "policy": result.policy,
                "duration_ms": duration_ms,
                "output_chars": len(result.output),
                "metadata": dict(result.metadata),
            },
        )

    def task_finished(
        self, *, status: str, message: str, steps: int, summary: TaskSummary
    ) -> None:
        """Record the terminal outcome and concise change summary."""

        self._write(
            "task_finished",
            {
                "status": status,
                "message": message,
                "steps": steps,
                "summary": summary.as_dict(),
            },
        )

    def _write(self, event: str, details: Mapping[str, object]) -> None:
        payload = {
            "task_id": self.task_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **details,
        }
        try:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False, default=str))
                stream.write("\n")
        except OSError:
            # An audit write failure must not turn a recoverable coding task into
            # a crash. P3's task summary still exposes the log path when present.
            return


def _argument_summary(call: ToolCall) -> dict[str, object]:
    """Keep audit arguments useful while excluding file bodies and patch content."""

    arguments = call.arguments or {}
    if call.name in {"list_files", "read_file"}:
        return {"path": arguments.get("path")}
    if call.name == "apply_patch":
        return {
            "path": arguments.get("path"),
            "expected_text_chars": _text_length(arguments.get("expected_text")),
            "replacement_text_chars": _text_length(arguments.get("replacement_text")),
        }
    if call.name == "run_command":
        command = arguments.get("command")
        return {"command": command if isinstance(command, list) else None}
    return {"argument_keys": sorted(arguments.keys())}


def _text_length(value: object) -> int | None:
    return len(value) if isinstance(value, str) else None
