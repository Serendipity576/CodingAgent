from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from agent.config import RuntimeLimits
from agent.llm.models import ToolCall
from agent.security.policy import Decision, PolicyDecision
from agent.security.risk import RiskLevel
from agent.tools import build_default_registry
from agent.tools.base import ToolContext


class StaticApproval:
    """Test double that records the exact high-risk calls it receives."""

    def __init__(self, approved: bool) -> None:
        self.approved = approved
        self.calls: list[tuple[ToolCall, PolicyDecision]] = []

    def request(self, call: ToolCall, decision: PolicyDecision) -> bool:
        self.calls.append((call, decision))
        return self.approved


class SecurityPolicyTests(unittest.TestCase):
    def test_workspace_escape_is_denied_for_relative_and_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            registry = build_default_registry(workspace)
            context = _context(workspace)

            relative_result = registry.execute(
                _call("read_file", {"path": "../outside.txt"}), context
            )
            absolute_result = registry.execute(
                _call("read_file", {"path": str(outside)}), context
            )

            for result in (relative_result, absolute_result):
                self.assertFalse(result.success)
                self.assertEqual(result.decision, Decision.DENY.value)
                self.assertEqual(result.risk, RiskLevel.CRITICAL.value)
                self.assertEqual(result.policy, "workspace_boundary")

    def test_symlink_escape_is_denied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            link = workspace / "outside-link.txt"
            try:
                link.symlink_to(outside)
            except OSError as error:
                self.skipTest(f"symlinks unavailable: {error}")

            result = build_default_registry(workspace).execute(
                _call("read_file", {"path": "outside-link.txt"}), _context(workspace)
            )

            self.assertFalse(result.success)
            self.assertEqual(result.policy, "workspace_boundary")

    def test_sensitive_files_are_denied_for_read_and_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / ".env").write_text("TOKEN=secret", encoding="utf-8")
            registry = build_default_registry(workspace)
            context = _context(workspace)

            read_result = registry.execute(
                _call("read_file", {"path": ".env"}), context
            )
            command_result = registry.execute(
                _call("run_command", {"command": ["cat", ".env"]}), context
            )
            write_result = registry.execute(
                _call(
                    "apply_patch",
                    {
                        "path": "id_ed25519",
                        "expected_text": "",
                        "replacement_text": "private key",
                    },
                ),
                context,
            )

            for result in (read_result, command_result, write_result):
                self.assertFalse(result.success)
                self.assertEqual(result.decision, Decision.DENY.value)
                self.assertEqual(result.policy, "sensitive_data")
            self.assertFalse((workspace / "id_ed25519").exists())

    def test_local_conversation_database_is_denied_to_agent_tools(self) -> None:
        """Keep durable model context and event journals outside Agent tool access."""

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            database = workspace / ".agent" / "conversations" / "sessions.sqlite3"
            database.parent.mkdir(parents=True)
            database.write_text("local transcript", encoding="utf-8")

            result = build_default_registry(workspace).execute(
                _call("read_file", {"path": ".agent/conversations/sessions.sqlite3"}),
                _context(workspace),
            )
            listed = build_default_registry(workspace).execute(
                _call("list_files", {"path": ".", "max_depth": 2}),
                _context(workspace),
            )

        self.assertFalse(result.success)
        self.assertEqual(result.decision, Decision.DENY.value)
        self.assertEqual(result.policy, "sensitive_data")
        self.assertTrue(listed.success)
        self.assertNotIn(".agent", listed.output)

    def test_sensitive_parent_directory_does_not_block_the_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "secrets"
            workspace = root / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "normal.txt").write_text("safe", encoding="utf-8")

            result = build_default_registry(workspace).execute(
                _call("read_file", {"path": "normal.txt"}), _context(workspace)
            )

            self.assertTrue(result.success)
            self.assertEqual(result.policy, "workspace_read")

    def test_critical_commands_are_denied_without_prompting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            approval = StaticApproval(approved=True)
            registry = build_default_registry(workspace, approval=approval)

            result = registry.execute(
                _call("run_command", {"command": ["sudo", "whoami"]}), _context(workspace)
            )

            self.assertFalse(result.success)
            self.assertEqual(result.decision, Decision.DENY.value)
            self.assertEqual(result.policy, "critical_command")
            self.assertEqual(approval.calls, [])

    def test_high_risk_deletion_requires_and_honors_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            victim = workspace / "victim.txt"
            victim.write_text("temporary", encoding="utf-8")
            denied_approval = StaticApproval(approved=False)
            denied_registry = build_default_registry(workspace, approval=denied_approval)

            denied_result = denied_registry.execute(
                _call("run_command", {"command": ["rm", "victim.txt"]}),
                _context(workspace),
            )

            self.assertFalse(denied_result.success)
            self.assertEqual(denied_result.decision, Decision.REQUIRE_APPROVAL.value)
            self.assertEqual(denied_result.risk, RiskLevel.HIGH.value)
            self.assertEqual(denied_result.policy, "file_deletion")
            self.assertTrue(victim.exists())
            self.assertEqual(len(denied_approval.calls), 1)

            approved_approval = StaticApproval(approved=True)
            approved_registry = build_default_registry(workspace, approval=approved_approval)
            approved_result = approved_registry.execute(
                _call("run_command", {"command": ["rm", "victim.txt"]}),
                _context(workspace),
            )

            self.assertTrue(approved_result.success)
            self.assertEqual(approved_result.decision, Decision.REQUIRE_APPROVAL.value)
            self.assertFalse(victim.exists())
            self.assertEqual(len(approved_approval.calls), 1)

    def test_destructive_git_and_external_deletion_are_not_auto_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            approval = StaticApproval(approved=False)
            registry = build_default_registry(workspace, approval=approval)
            context = _context(workspace)

            git_result = registry.execute(
                _call("run_command", {"command": ["git", "reset", "--hard"]}), context
            )
            external_rm_result = registry.execute(
                _call("run_command", {"command": ["rm", "/tmp"]}), context
            )

            self.assertEqual(git_result.decision, Decision.REQUIRE_APPROVAL.value)
            self.assertEqual(git_result.policy, "git_destructive")
            self.assertEqual(external_rm_result.decision, Decision.DENY.value)
            self.assertEqual(external_rm_result.policy, "workspace_boundary")


def _call(name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(call_id=f"call-{name}", name=name, arguments=arguments)


def _context(workspace: Path) -> ToolContext:
    return ToolContext(
        workspace=workspace.resolve(),
        limits=RuntimeLimits(
            max_steps=10,
            command_timeout_seconds=10,
            max_output_chars=2_000,
            max_task_seconds=30,
        ),
    )
