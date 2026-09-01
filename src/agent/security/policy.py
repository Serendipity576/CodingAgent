"""Central deterministic authorization policy for every agent tool call."""

from __future__ import annotations

from pathlib import Path

from agent.llm.models import ToolCall
from agent.security.command_guard import CommandPolicy
from agent.security.path_guard import PathAccessError, WorkspacePathGuard
from agent.security.policy_types import Decision, PolicyDecision
from agent.security.risk import RiskLevel
from agent.security.sensitive import SensitiveDataGuard


class PolicyEngine:
    """Decide whether a model-requested action may execute, needs approval, or is denied."""

    _PATH_TOOLS = frozenset({"list_files", "read_file", "apply_patch"})

    def __init__(self, workspace: Path) -> None:
        self._path_guard = WorkspacePathGuard(workspace)
        self._sensitive_guard = SensitiveDataGuard()
        self._command_policy = CommandPolicy(self._path_guard, self._sensitive_guard)

    def evaluate(self, call: ToolCall) -> PolicyDecision:
        """Return an explainable authorization result without asking the LLM."""

        if call.arguments_error or call.arguments is None:
            return self._deny("invalid_arguments", "tool arguments are invalid")
        if call.name in self._PATH_TOOLS:
            return self._file_decision(call)
        if call.name == "read_session_artifact":
            return self._session_artifact_decision(call)
        if call.name == "run_command":
            return self._command_policy.evaluate(call.arguments.get("command"))
        return self._deny("unknown_tool", f"tool is not registered for policy: {call.name}")

    def _file_decision(self, call: ToolCall) -> PolicyDecision:
        try:
            target = self._path_guard.resolve(call.arguments["path"])
        except (KeyError, PathAccessError) as error:
            return self._deny("workspace_boundary", str(error))

        # Only the workspace-relative path is policy input. An unrelated parent
        # directory named ``secrets`` must not block every workspace file.
        relative_target = target.relative_to(self._path_guard.workspace)
        sensitive_reason = self._sensitive_guard.reason(relative_target)
        if sensitive_reason:
            return self._deny("sensitive_data", sensitive_reason)
        if call.name == "apply_patch":
            return PolicyDecision(
                Decision.ALLOW,
                RiskLevel.MEDIUM,
                "workspace file modification",
                "workspace_write",
            )
        return PolicyDecision(
            Decision.ALLOW,
            RiskLevel.LOW,
            "workspace read-only operation",
            "workspace_read",
        )

    def _session_artifact_decision(self, call: ToolCall) -> PolicyDecision:
        """Allow bounded reads from the already-owned active conversation only."""

        artifact_id = call.arguments.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            return self._deny("invalid_arguments", "artifact_id must be a non-empty string")
        for name in ("offset", "max_chars"):
            value = call.arguments.get(name)
            if value is not None and (type(value) is not int or value < 0):
                return self._deny("invalid_arguments", f"{name} must be a non-negative integer")
        return PolicyDecision(
            Decision.ALLOW,
            RiskLevel.LOW,
            "read-only local conversation artifact",
            "conversation_read",
        )

    @staticmethod
    def _deny(policy: str, reason: str) -> PolicyDecision:
        return PolicyDecision(Decision.DENY, RiskLevel.CRITICAL, reason, policy)


__all__ = ["Decision", "PolicyDecision", "PolicyEngine"]
