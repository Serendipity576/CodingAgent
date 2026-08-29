"""Default tool set for the minimal coding-agent loop."""

from pathlib import Path

from agent.security.approval import ApprovalHandler
from agent.security.policy import PolicyEngine
from agent.tools.filesystem import ApplyPatchTool, ListFilesTool, ReadFileTool
from agent.tools.registry import ToolRegistry
from agent.tools.shell import RunCommandTool


def build_default_registry(
    workspace: Path, *, approval: ApprovalHandler | None = None
) -> ToolRegistry:
    """Create a gated registry with an explicit workspace policy dependency."""

    return ToolRegistry(
        [
            ListFilesTool(),
            ReadFileTool(),
            ApplyPatchTool(),
            RunCommandTool(),
        ],
        policy=PolicyEngine(workspace),
        approval=approval,
    )


__all__ = ["ToolRegistry", "build_default_registry"]
