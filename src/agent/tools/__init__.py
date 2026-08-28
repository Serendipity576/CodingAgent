"""Default tool set for the minimal coding-agent loop."""

from agent.tools.filesystem import ApplyPatchTool, ListFilesTool, ReadFileTool
from agent.tools.registry import ToolRegistry
from agent.tools.shell import RunCommandTool


def build_default_registry() -> ToolRegistry:
    """Create a fresh registry so each agent run has explicit dependencies."""

    return ToolRegistry(
        [
            ListFilesTool(),
            ReadFileTool(),
            ApplyPatchTool(),
            RunCommandTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_default_registry"]
