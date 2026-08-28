"""Tool registration, schema export, and recoverable execution failures."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from agent.llm.models import ToolCall
from agent.tools.base import Tool, ToolContext, ToolError, ToolResult


class ToolRegistry:
    """The only runtime entry point for invoking registered tools."""

    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    def schemas(self) -> tuple[Mapping[str, object], ...]:
        """Return Responses API function schemas for the available tools."""

        return tuple(
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                # P1 tools intentionally allow optional arguments such as
                # ``max_depth``; each tool still validates values at runtime.
                "strict": False,
            }
            for tool in self._tools.values()
        )

    def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Execute one call, converting ordinary input failures into observations."""

        if call.arguments_error:
            return ToolResult.failed(call.arguments_error)
        if call.arguments is None:
            return ToolResult.failed("tool arguments are required")

        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult.failed(f"unknown tool: {call.name}")

        try:
            return tool.execute(call.arguments, context)
        except ToolError as error:
            return ToolResult.failed(str(error))
        except Exception as error:  # Keep a tool bug from crashing the whole task.
            return ToolResult.failed(f"internal tool error: {error}")
