"""Human approval interfaces for high-risk tool calls."""

from __future__ import annotations

from typing import Callable, Protocol, TextIO
import sys

from agent.llm.models import ToolCall
from agent.security.policy_types import PolicyDecision


class ApprovalHandler(Protocol):
    """Request an explicit user decision for an already-classified action."""

    def request(self, call: ToolCall, decision: PolicyDecision) -> bool:
        """Return ``True`` only when the user approves this exact call."""


class DenyAllApproval:
    """Safe default for non-interactive contexts and tests."""

    def request(self, call: ToolCall, decision: PolicyDecision) -> bool:
        return False


class ConsoleApproval:
    """Prompt only on an interactive terminal; deny when no user can answer."""

    def __init__(
        self,
        *,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
        read_input: Callable[[str], str] = input,
    ) -> None:
        self._input_stream = input_stream or sys.stdin
        self._output_stream = output_stream or sys.stderr
        self._read_input = read_input

    def request(self, call: ToolCall, decision: PolicyDecision) -> bool:
        if not self._input_stream.isatty():
            return False

        # Show exactly what is being approved; a model cannot reuse an earlier
        # approval for another command or file operation.
        print("HIGH RISK ACTION", file=self._output_stream)
        print(f"Tool: {call.name}", file=self._output_stream)
        print(f"Arguments: {dict(call.arguments or {})}", file=self._output_stream)
        print(f"Reason: {decision.reason}", file=self._output_stream)
        answer = self._read_input("Execute? [y/N] ").strip().casefold()
        return answer in {"y", "yes"}
