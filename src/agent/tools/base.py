"""Shared tool contracts and output handling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Protocol

from agent.config import RuntimeLimits


class ToolError(ValueError):
    """An expected tool failure that can be returned to the model."""


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Run-scoped values each tool needs without accessing global state."""

    workspace: Path
    limits: RuntimeLimits


@dataclass(frozen=True, slots=True)
class ToolResult:
    """A serializable success or failure observation for the model."""

    success: bool
    output: str
    error: str | None = None
    decision: str | None = None
    risk: str | None = None
    policy: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def succeeded(
        cls, output: str, *, metadata: Mapping[str, object] | None = None
    ) -> "ToolResult":
        return cls(success=True, output=output, metadata=metadata or {})

    @classmethod
    def failed(
        cls,
        error: str,
        output: str = "",
        *,
        decision: str | None = None,
        risk: str | None = None,
        policy: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> "ToolResult":
        return cls(
            success=False,
            output=output,
            error=error,
            decision=decision,
            risk=risk,
            policy=policy,
            metadata=metadata or {},
        )

    def as_observation(self) -> str:
        """Encode the result in the function-output format expected by the LLM."""

        payload: dict[str, object] = {"ok": self.success, "output": self.output}
        if self.error:
            payload["error"] = self.error
        if self.decision:
            payload["security"] = {
                "decision": self.decision,
                "risk": self.risk,
                "policy": self.policy,
            }
        return json.dumps(payload, ensure_ascii=False)


class Tool(Protocol):
    """The small interface used by the registry and function-call schema builder."""

    name: str
    description: str
    parameters: Mapping[str, object]

    def execute(
        self, arguments: Mapping[str, object], context: ToolContext
    ) -> ToolResult:
        """Execute a validated tool request."""


def truncate_text(text: str, maximum: int) -> str:
    """Keep observations within the configured context and terminal budget."""

    if len(text) <= maximum:
        return text
    marker = f"\n... [truncated {len(text) - maximum} characters]"
    if maximum <= len(marker):
        return text[:maximum]
    return f"{text[: maximum - len(marker)]}{marker}"
