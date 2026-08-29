"""Conservative rule-based classification for argument-vector commands."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from agent.security.path_guard import PathAccessError, WorkspacePathGuard
from agent.security.policy_types import Decision, PolicyDecision
from agent.security.risk import RiskLevel
from agent.security.sensitive import SensitiveDataGuard


class CommandPolicy:
    """Classify common development commands without trying to parse every shell."""

    _CRITICAL_EXECUTABLES = frozenset(
        {"sudo", "doas", "su", "shutdown", "reboot", "halt", "poweroff", "mkfs"}
    )
    _NETWORK_EXECUTABLES = frozenset({"curl", "wget", "ssh", "scp", "sftp", "nc", "ncat"})
    _PACKAGE_MANAGERS = frozenset(
        {"pip", "pip3", "npm", "yarn", "pnpm", "poetry", "uv", "conda", "apt", "brew"}
    )
    _INTERPRETERS = frozenset({"python", "python3", "python3.10", "node", "bash", "sh", "zsh"})

    def __init__(
        self, path_guard: WorkspacePathGuard, sensitive_guard: SensitiveDataGuard
    ) -> None:
        self._path_guard = path_guard
        self._sensitive_guard = sensitive_guard

    def evaluate(self, command: object) -> PolicyDecision:
        """Return a decision using executable, arguments, and workspace targets."""

        if not isinstance(command, list) or not command:
            return self._deny("invalid_command", "command must be a non-empty string array")
        if any(not isinstance(part, str) or not part for part in command):
            return self._deny("invalid_command", "every command argument must be a non-empty string")

        sensitive_reason = self._sensitive_argument_reason(command[1:])
        if sensitive_reason:
            return self._deny("sensitive_data", sensitive_reason)

        executable = Path(command[0]).name.casefold()
        if executable in self._CRITICAL_EXECUTABLES:
            return self._deny("critical_command", f"{executable} is never allowed")
        if executable == "dd":
            return self._deny("critical_command", "dd can overwrite arbitrary block devices")
        if executable == "rm":
            return self._remove_decision(command)
        if executable == "git":
            return self._git_decision(command)
        if executable in self._PACKAGE_MANAGERS:
            return self._approval("package_management", "installing or changing dependencies")
        if executable in self._NETWORK_EXECUTABLES:
            return self._approval("network_command", "network-capable command")
        if executable in self._INTERPRETERS:
            return self._interpreter_decision(command, executable)
        if executable == "pytest":
            return self._allow("test_command", "Python test runner")
        if executable == "cargo":
            return self._build_tool_decision(command, {"test", "check", "build", "fmt"})
        if executable == "go":
            return self._build_tool_decision(command, {"test", "vet", "build", "fmt"})
        if executable == "make":
            # Makefiles are executable project code, so even test-like targets
            # deserve user confirmation until P2's policy can inspect them.
            return self._approval("build_script", "Makefile target can execute arbitrary commands")

        # Unknown executables may run arbitrary binaries. They are not denied,
        # but require a user to consciously expand the agent's capabilities.
        return self._approval("unknown_command", "unrecognized executable")

    def _remove_decision(self, command: Sequence[str]) -> PolicyDecision:
        targets = _non_option_arguments(command[1:])
        if not targets:
            return self._approval("file_deletion", "file deletion command")
        for target in targets:
            try:
                self._path_guard.resolve(target)
            except PathAccessError:
                return self._deny("workspace_boundary", "deletion target is outside workspace")
        return self._approval("file_deletion", "file deletion inside workspace")

    def _git_decision(self, command: Sequence[str]) -> PolicyDecision:
        arguments = set(command[1:])
        if {"reset", "--hard"}.issubset(arguments) or "clean" in arguments:
            return self._approval("git_destructive", "destructive Git operation")
        if "restore" in arguments or "checkout" in arguments:
            return self._approval("git_destructive", "Git operation may discard changes")
        if "commit" in arguments or "push" in arguments:
            return self._approval("git_write", "Git operation changes local or remote history")
        return self._allow("git_readonly", "read-only Git inspection command")

    def _interpreter_decision(
        self, command: Sequence[str], executable: str
    ) -> PolicyDecision:
        if executable.startswith("python") and len(command) >= 3:
            if command[1] == "-m" and command[2] in {"pytest", "unittest"}:
                return self._allow("test_command", "Python test runner")
        if executable == "node" and len(command) >= 2 and command[1] in {"--test", "test"}:
            return self._allow("test_command", "Node test runner")
        return self._approval("script_execution", "interpreter can execute arbitrary code")

    def _build_tool_decision(
        self, command: Sequence[str], allowed_subcommands: set[str]
    ) -> PolicyDecision:
        if len(command) >= 2 and command[1] in allowed_subcommands:
            return self._allow("development_command", "standard development command")
        return self._approval("build_tool", "build tool subcommand may change local state")

    def _sensitive_argument_reason(self, arguments: Sequence[str]) -> str | None:
        """Deny direct credential-path arguments even when a command is approved.

        This is intentionally narrow: a full interpretation of arbitrary scripts
        belongs outside this rule-based P2 policy. The file tools remain the
        normal path for repository reads and writes.
        """

        for argument in arguments:
            if argument.startswith("-"):
                continue
            try:
                target = self._path_guard.resolve(argument)
            except PathAccessError:
                continue
            relative_target = target.relative_to(self._path_guard.workspace)
            reason = self._sensitive_guard.reason(relative_target)
            if reason:
                return f"command argument targets sensitive data: {reason}"
        return None

    @staticmethod
    def _allow(policy: str, reason: str) -> PolicyDecision:
        return PolicyDecision(Decision.ALLOW, RiskLevel.LOW, reason, policy)

    @staticmethod
    def _approval(policy: str, reason: str) -> PolicyDecision:
        return PolicyDecision(Decision.REQUIRE_APPROVAL, RiskLevel.HIGH, reason, policy)

    @staticmethod
    def _deny(policy: str, reason: str) -> PolicyDecision:
        return PolicyDecision(Decision.DENY, RiskLevel.CRITICAL, reason, policy)


def _non_option_arguments(arguments: Sequence[str]) -> list[str]:
    """Return simple rm targets while respecting the conventional ``--`` marker."""

    targets: list[str] = []
    options_ended = False
    for argument in arguments:
        if argument == "--":
            options_ended = True
            continue
        if not options_ended and argument.startswith("-"):
            continue
        targets.append(argument)
    return targets
