"""Workspace-scoped filesystem tools for the P1 runtime."""

from __future__ import annotations

from collections.abc import Mapping
from difflib import unified_diff
from pathlib import Path

from agent.security.path_guard import PathAccessError, WorkspacePathGuard
from agent.security.sensitive import SensitiveDataGuard
from agent.tools.base import ToolContext, ToolError, ToolResult, truncate_text


def resolve_workspace_path(context: ToolContext, requested_path: object) -> Path:
    """Resolve a requested path and require it to remain below the workspace."""

    if not isinstance(requested_path, str) or not requested_path.strip():
        raise ToolError("path must be a non-empty string")

    try:
        return WorkspacePathGuard(context.workspace).resolve(requested_path)
    except PathAccessError as error:
        # The registry runs the same check before this function. Rechecking at
        # execution time prevents a later path change from bypassing the guard.
        raise ToolError(str(error)) from error


def _required_string(arguments: Mapping[str, object], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise ToolError(f"{name} must be a string")
    return value


class ListFilesTool:
    """List a bounded directory tree without following symlinked directories."""

    name = "list_files"
    description = "List files and directories under a workspace-relative path. Use '.' for the workspace root."
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory to inspect, relative to the workspace. Use '.' for the workspace root.",
                "default": ".",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum recursion depth, from 0 to 5. Defaults to 2.",
                "minimum": 0,
                "maximum": 5,
            },
        },
        "additionalProperties": False,
    }

    def execute(
        self, arguments: Mapping[str, object], context: ToolContext
    ) -> ToolResult:
        requested_path = arguments.get("path", ".")
        if isinstance(requested_path, str) and not requested_path.strip():
            requested_path = "."
        target = resolve_workspace_path(context, requested_path)
        if not target.exists():
            raise ToolError(f"directory does not exist: {target}")
        if not target.is_dir():
            raise ToolError("path must identify a directory")

        max_depth = arguments.get("max_depth", 2)
        if type(max_depth) is not int or not 0 <= max_depth <= 5:
            raise ToolError("max_depth must be an integer from 0 to 5")

        entries: list[str] = []
        self._walk(
            target,
            context.workspace.resolve(),
            max_depth,
            0,
            entries,
            SensitiveDataGuard(),
        )
        output = "\n".join(entries) if entries else "(empty directory)"
        return ToolResult.succeeded(truncate_text(output, context.limits.max_output_chars))

    def _walk(
        self,
        directory: Path,
        workspace: Path,
        max_depth: int,
        current_depth: int,
        entries: list[str],
        sensitive_guard: SensitiveDataGuard,
    ) -> None:
        try:
            children = sorted(directory.iterdir(), key=lambda child: child.name.lower())
        except OSError as error:
            raise ToolError(f"could not list directory: {error}") from error

        for child in children:
            relative = child.relative_to(workspace).as_posix()
            if sensitive_guard.reason(Path(relative)):
                # A directory listing must not reveal protected local state
                # that the same Agent is prohibited from opening directly.
                continue
            if child.is_symlink():
                # Show links for diagnosis but never traverse them during P1.
                entries.append(f"{relative}@")
            elif child.is_dir():
                entries.append(f"{relative}/")
                if current_depth < max_depth:
                    self._walk(
                        child,
                        workspace,
                        max_depth,
                        current_depth + 1,
                        entries,
                        sensitive_guard,
                    )
            else:
                entries.append(relative)


class ReadFileTool:
    """Read one UTF-8 text file and return a bounded observation."""

    name = "read_file"
    description = "Read a text file inside the workspace."
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the workspace.",
            }
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def execute(
        self, arguments: Mapping[str, object], context: ToolContext
    ) -> ToolResult:
        target = resolve_workspace_path(context, _required_string(arguments, "path"))
        if not target.exists():
            raise ToolError(f"file does not exist: {target}")
        if not target.is_file():
            raise ToolError("path must identify a regular file")

        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            raise ToolError(f"could not read file: {error}") from error
        return ToolResult.succeeded(truncate_text(content, context.limits.max_output_chars))


class ApplyPatchTool:
    """Make one focused replacement or create one new text file."""

    name = "apply_patch"
    description = (
        "Replace one unique text fragment in a workspace file, or create a new file "
        "when expected_text is empty."
    )
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the workspace.",
            },
            "expected_text": {
                "type": "string",
                "description": "Exact existing text to replace; use an empty string only for a new file.",
            },
            "replacement_text": {
                "type": "string",
                "description": "Text that replaces expected_text.",
            },
        },
        "required": ["path", "expected_text", "replacement_text"],
        "additionalProperties": False,
    }

    def execute(
        self, arguments: Mapping[str, object], context: ToolContext
    ) -> ToolResult:
        target = resolve_workspace_path(context, _required_string(arguments, "path"))
        expected_text = _required_string(arguments, "expected_text")
        replacement_text = _required_string(arguments, "replacement_text")

        try:
            if target.exists():
                if not target.is_file():
                    raise ToolError("path must identify a regular file")
                if not expected_text:
                    raise ToolError("expected_text must not be empty when updating a file")
                original = target.read_text(encoding="utf-8", errors="replace")
                matches = original.count(expected_text)
                if matches == 0:
                    raise ToolError("expected_text was not found in the file")
                if matches > 1:
                    raise ToolError("expected_text must occur exactly once")
                updated = original.replace(expected_text, replacement_text, 1)
                target.write_text(updated, encoding="utf-8")
                action = "updated"
                diff_summary = _diff_summary(original, updated)
            else:
                if expected_text:
                    raise ToolError("expected_text must be empty when creating a file")
                original = None
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(replacement_text, encoding="utf-8")
                action = "created"
                diff_summary = _diff_summary("", replacement_text)
        except OSError as error:
            raise ToolError(f"could not write file: {error}") from error

        relative = target.relative_to(context.workspace.resolve()).as_posix()
        return ToolResult.succeeded(
            f"{action} {relative}",
            metadata={
                "path": relative,
                "change_type": action,
                "diff_summary": diff_summary,
            },
        )


def _diff_summary(before: str, after: str) -> dict[str, int]:
    """Count changed lines without placing full source code in the audit trail."""

    added_lines = 0
    removed_lines = 0
    for line in unified_diff(before.splitlines(), after.splitlines(), lineterm=""):
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added_lines += 1
        elif line.startswith("-"):
            removed_lines += 1
    return {"added_lines": added_lines, "removed_lines": removed_lines}
