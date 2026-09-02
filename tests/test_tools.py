from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from agent.config import RuntimeLimits
from agent.llm.models import ToolCall
from agent.sandbox import SandboxInvocation
from agent.security.policy import PolicyEngine
from agent.tools.base import ToolContext
from agent.tools.filesystem import ApplyPatchTool, ListFilesTool, ReadFileTool
from agent.tools.registry import ToolRegistry
from agent.tools.session import ReadSessionArtifactTool
from agent.tools.shell import RunCommandTool


class FilesystemToolTests(unittest.TestCase):
    def test_apply_patch_requires_a_unique_expected_fragment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            target = workspace / "example.txt"
            target.write_text("first\nfirst\n", encoding="utf-8")
            context = _context(workspace)

            result = ToolRegistry(
                [ApplyPatchTool()], policy=PolicyEngine(workspace)
            ).execute(
                ToolCall(
                    call_id="call-1",
                    name="apply_patch",
                    arguments={
                        "path": "example.txt",
                        "expected_text": "first",
                        "replacement_text": "second",
                    },
                ),
                context,
            )

            self.assertFalse(result.success)
            self.assertIn("exactly once", result.error or "")
            self.assertEqual(target.read_text(encoding="utf-8"), "first\nfirst\n")

    def test_read_file_truncates_oversized_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "long.txt").write_text("x" * 200, encoding="utf-8")
            context = _context(workspace, max_output_chars=100)

            result = ReadFileTool().execute({"path": "long.txt"}, context)

            self.assertTrue(result.success)
            self.assertLessEqual(len(result.output), 100)
            self.assertIn("file exceeds output limit", result.output)

    def test_list_files_stops_after_a_bounded_entry_prefix(self) -> None:
        """A dense directory cannot build an unbounded in-memory listing."""

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            for index in range(101):
                (workspace / f"entry-{index:03d}.txt").write_text("", encoding="utf-8")

            result = ListFilesTool().execute({"path": "."}, _context(workspace, max_output_chars=100))

        self.assertTrue(result.success)
        self.assertLessEqual(len(result.output), 100)
        self.assertIn("truncated", result.output)

    def test_command_output_is_drained_without_retaining_the_full_stream(self) -> None:
        """A noisy subprocess must leave only the configured output prefix in memory."""

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            result = RunCommandTool(sandbox=_PassthroughSandbox()).execute(
                {"command": [sys.executable, "-c", "print('x' * 250000)"]},
                _context(workspace, max_output_chars=200),
            )

        self.assertTrue(result.success, result.error)
        self.assertLessEqual(len(result.output), 200)
        self.assertIn("truncated", result.output)

    def test_run_command_normalizes_a_serialized_json_array(self) -> None:
        """A common model encoding error can recover without accepting shell text."""

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            command = json.dumps([sys.executable, "-m", "unittest", "-q"])
            result = ToolRegistry(
                [RunCommandTool(sandbox=_PassthroughSandbox())],
                policy=PolicyEngine(workspace),
            ).execute(
                ToolCall(call_id="call-1", name="run_command", arguments={"command": command}),
                _context(workspace),
            )

        self.assertTrue(result.success, result.error)
        self.assertEqual(
            result.metadata["normalization"]["command"], "parsed_json_string_array"
        )

    def test_list_files_empty_path_safely_defaults_to_workspace_root(self) -> None:
        """A common model shorthand lists the root without weakening other path tools."""

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "visible.txt").write_text("content", encoding="utf-8")
            result = ToolRegistry([ListFilesTool()], policy=PolicyEngine(workspace)).execute(
                ToolCall(call_id="call-1", name="list_files", arguments={"path": ""}),
                _context(workspace),
            )

        self.assertTrue(result.success)
        self.assertIn("visible.txt", result.output)
        self.assertEqual(result.metadata["normalization"]["path"], ".")

    def test_read_session_artifact_is_bounded_and_read_only(self) -> None:
        """Archived output is available only through the active session reader contract."""

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            reader = _ArtifactReader()
            context = ToolContext(
                workspace=workspace.resolve(),
                limits=RuntimeLimits(max_steps=10, command_timeout_seconds=10, max_task_seconds=30),
                artifact_reader=reader,
            )
            result = ToolRegistry(
                [ReadSessionArtifactTool()], policy=PolicyEngine(workspace)
            ).execute(
                ToolCall(
                    call_id="call-1",
                    name="read_session_artifact",
                    arguments={"artifact_id": "tool-1", "offset": 2, "max_chars": 4},
                ),
                context,
            )

        self.assertTrue(result.success)
        self.assertIn("cdef", result.output)
        self.assertEqual(reader.requests, [("tool-1", 2, 4)])


def _context(workspace: Path, *, max_output_chars: int = 2_000) -> ToolContext:
    return ToolContext(
        workspace=workspace.resolve(),
        limits=RuntimeLimits(
            max_steps=10,
            command_timeout_seconds=10,
            max_output_chars=max_output_chars,
            max_task_seconds=30,
        ),
    )


class _ArtifactReader:
    """Minimal active-session reader used to verify tool dispatch boundaries."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, int, int]] = []

    def read_session_artifact(
        self,
        artifact_id: str,
        offset: int,
        max_chars: int,
    ) -> tuple[str, dict[str, object]]:
        self.requests.append((artifact_id, offset, max_chars))
        return "abcdef"[offset : offset + max_chars], {"artifact_id": artifact_id}


class _PassthroughSandbox:
    """Test double that exposes command-output handling without host isolation setup."""

    def prepare(self, command: list[str], workspace: Path) -> SandboxInvocation:
        return SandboxInvocation(tuple(command), {"execution_scope": "test"})
