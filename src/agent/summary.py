"""Task-end summaries derived from Agent-owned tool results."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from agent.change_tracker import GitStatusSnapshot
from agent.llm.models import ToolCall
from agent.tools.base import ToolResult


@dataclass(frozen=True, slots=True)
class TaskSummary:
    """A concise, user-facing description of observable task effects."""

    modified_files: tuple[str, ...]
    diff_summaries: tuple[dict[str, object], ...]
    tests: str
    blocked_actions: int
    approved_high_risk_actions: int
    git_baseline_available: bool
    preexisting_git_changes: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return JSON-friendly task data for the CLI and audit finish event."""

        return {
            "modified_files": list(self.modified_files),
            "modified_file_count": len(self.modified_files),
            "diff_summaries": list(self.diff_summaries),
            "tests": self.tests,
            "blocked_actions": self.blocked_actions,
            "approved_high_risk_actions": self.approved_high_risk_actions,
            "git_baseline_available": self.git_baseline_available,
            "preexisting_git_changes": list(self.preexisting_git_changes),
        }


def build_task_summary(
    calls: Sequence[tuple[ToolCall, ToolResult]],
    git_baseline: GitStatusSnapshot,
) -> TaskSummary:
    """Summarize only the current run's successful patch and command results."""

    modified_files: list[str] = []
    diff_summaries: list[dict[str, object]] = []
    tests = "not_run"
    blocked_actions = 0
    approved_high_risk_actions = 0

    for call, result in calls:
        if result.metadata.get("approval") == "approved":
            approved_high_risk_actions += 1
        if not result.success and result.decision in {"deny", "require_approval"}:
            blocked_actions += 1

        if call.name == "apply_patch" and result.success:
            _record_file_change(result, modified_files, diff_summaries)
        elif call.name == "run_command" and _is_test_command(call):
            # The latest test command reflects the state after the latest patch.
            tests = "passed" if result.success else "failed"

    return TaskSummary(
        modified_files=tuple(modified_files),
        diff_summaries=tuple(diff_summaries),
        tests=tests,
        blocked_actions=blocked_actions,
        approved_high_risk_actions=approved_high_risk_actions,
        git_baseline_available=git_baseline.available,
        preexisting_git_changes=git_baseline.entries,
    )


def _record_file_change(
    result: ToolResult,
    modified_files: list[str],
    diff_summaries: list[dict[str, object]],
) -> None:
    path = result.metadata.get("path")
    if not isinstance(path, str):
        return
    if path not in modified_files:
        modified_files.append(path)

    change_type = result.metadata.get("change_type")
    diff = result.metadata.get("diff_summary")
    if isinstance(change_type, str) and isinstance(diff, dict):
        diff_summaries.append(
            {
                "path": path,
                "change_type": change_type,
                "added_lines": diff.get("added_lines", 0),
                "removed_lines": diff.get("removed_lines", 0),
            }
        )


def _is_test_command(call: ToolCall) -> bool:
    """Recognize common test commands without parsing arbitrary shell syntax."""

    command = call.arguments.get("command") if call.arguments else None
    if not isinstance(command, list) or not command:
        return False
    if not all(isinstance(part, str) for part in command):
        return False

    executable = Path(command[0]).name.casefold()
    if executable == "pytest":
        return True
    if executable.startswith("python") and len(command) >= 3:
        return command[1] == "-m" and command[2] in {"pytest", "unittest"}
    if executable == "cargo" and len(command) >= 2:
        return command[1] == "test"
    if executable == "go" and len(command) >= 2:
        return command[1] == "test"
    if executable == "node" and len(command) >= 2:
        return command[1] in {"--test", "test"}
    return False
