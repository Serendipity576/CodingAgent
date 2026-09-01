from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from agent.config import RuntimeLimits
from agent.llm.models import ToolCall
from agent.security.policy import PolicyEngine
from agent.tools.base import ToolContext
from agent.tools.filesystem import ApplyPatchTool, ListFilesTool, ReadFileTool
from agent.tools.registry import ToolRegistry


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
            (workspace / "long.txt").write_text("x" * 100, encoding="utf-8")
            context = _context(workspace, max_output_chars=30)

            result = ReadFileTool().execute({"path": "long.txt"}, context)

            self.assertTrue(result.success)
            self.assertLessEqual(len(result.output), 30)

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
