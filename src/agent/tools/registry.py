"""Tool registration, schema export, and recoverable execution failures."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
import json

from agent.llm.models import ToolCall
from agent.security.approval import ApprovalHandler, DenyAllApproval
from agent.security.policy import Decision, PolicyDecision, PolicyEngine
from agent.tools.base import Tool, ToolContext, ToolError, ToolResult


class ToolRegistry:
    """The only runtime entry point for invoking registered tools."""

    def __init__(
        self,
        tools: Iterable[Tool],
        *,
        policy: PolicyEngine,
        approval: ApprovalHandler | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._policy = policy
        self._approval = approval or DenyAllApproval()
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

        call, normalization_metadata = self._normalize_call(call)
        decision = self._policy.evaluate(call)
        if decision.decision is Decision.DENY:
            return self._blocked_result(
                decision,
                "policy denied tool call",
                metadata=normalization_metadata,
            )
        approval_metadata: dict[str, object] = {}
        if decision.decision is Decision.REQUIRE_APPROVAL:
            if not self._approval.request(call, decision):
                return self._blocked_result(
                    decision,
                    "user did not approve tool call",
                    metadata={"approval": "rejected"},
                )
            approval_metadata["approval"] = "approved"

        tool = self._tools.get(call.name)
        if tool is None:
            # This should only be reachable when a custom policy deliberately
            # permits a tool that was not actually registered.
            return ToolResult.failed(f"unknown tool: {call.name}")

        try:
            result = tool.execute(call.arguments or {}, context)
        except ToolError as error:
            result = ToolResult.failed(str(error))
        except Exception as error:  # Keep a tool bug from crashing the whole task.
            result = ToolResult.failed(f"internal tool error: {error}")

        # Keep the policy trace with normal results so P3 audit logging can use
        # the same object rather than reconstructing a security decision later.
        return replace(
            result,
            decision=decision.decision.value,
            risk=decision.risk.value,
            policy=decision.policy,
            metadata={**result.metadata, **normalization_metadata, **approval_metadata},
        )

    @staticmethod
    def _normalize_call(call: ToolCall) -> tuple[ToolCall, dict[str, object]]:
        """Normalize documented, unambiguous shorthand before policy evaluation."""

        if call.arguments is None:
            return call, {}
        if call.name == "list_files":
            return ToolRegistry._normalize_list_path(call)
        if call.name == "run_command":
            return ToolRegistry._normalize_command_array(call)
        return call, {}

    @staticmethod
    def _normalize_list_path(call: ToolCall) -> tuple[ToolCall, dict[str, object]]:
        """Treat an empty list-files path as the safe, documented workspace root."""

        path = call.arguments.get("path") if call.arguments is not None else None
        if path is not None and (not isinstance(path, str) or path.strip()):
            return call, {}
        normalized_arguments = {**call.arguments, "path": "."}
        return (
            replace(call, arguments=normalized_arguments),
            {"normalization": {"path": ".", "reason": "empty_path_means_workspace_root"}},
        )

    @staticmethod
    def _normalize_command_array(call: ToolCall) -> tuple[ToolCall, dict[str, object]]:
        """Accept only a JSON-encoded string array; never parse shell command text."""

        command = call.arguments.get("command") if call.arguments is not None else None
        if not isinstance(command, str):
            return call, {}
        try:
            parsed = json.loads(command)
        except json.JSONDecodeError:
            return call, {}
        if not isinstance(parsed, list) or not parsed or any(
            not isinstance(part, str) or not part for part in parsed
        ):
            return call, {}
        normalized_arguments = {**call.arguments, "command": parsed}
        return (
            replace(call, arguments=normalized_arguments),
            {
                "normalization": {
                    "command": "parsed_json_string_array",
                    "reason": "serialized_array_from_model",
                }
            },
        )

    @staticmethod
    def _blocked_result(
        decision: PolicyDecision,
        prefix: str,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> ToolResult:
        """Turn a denied or unapproved request into an LLM-visible observation."""

        return ToolResult.failed(
            f"{prefix}: {decision.reason}",
            decision=decision.decision.value,
            risk=decision.risk.value,
            policy=decision.policy,
            metadata=metadata,
        )
