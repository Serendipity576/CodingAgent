"""Browser-mediated approval for high-risk tool calls."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Event, Lock
from time import monotonic
from uuid import uuid4

from agent.llm.models import ToolCall
from agent.security.policy_types import PolicyDecision


@dataclass(slots=True)
class _PendingApproval:
    """One exact tool call awaiting one browser decision."""

    completed: Event
    approved: bool | None = None


class WebApproval:
    """Pause an Agent worker until the local page approves or denies a call."""

    def __init__(
        self,
        publish: Callable[[str, Mapping[str, object]], None],
        *,
        timeout_seconds: int = 300,
    ) -> None:
        self._publish = publish
        self._timeout_seconds = timeout_seconds
        self._pending: dict[str, _PendingApproval] = {}
        self._lock = Lock()

    def request(self, call: ToolCall, decision: PolicyDecision) -> bool:
        """Emit one approval request and wait for its matching browser action."""

        approval_id = uuid4().hex
        pending = _PendingApproval(completed=Event())
        with self._lock:
            self._pending[approval_id] = pending
        self._publish(
            "approval_required",
            {
                "approval_id": approval_id,
                "tool": call.name,
                "arguments": _safe_arguments(call),
                "risk": decision.risk.value,
                "reason": decision.reason,
                "timeout_seconds": self._timeout_seconds,
            },
        )
        pending.completed.wait(self._timeout_seconds)
        with self._lock:
            self._pending.pop(approval_id, None)
        approved = pending.approved is True
        self._publish(
            "approval_resolved",
            {"approval_id": approval_id, "approved": approved},
        )
        return approved

    def resolve(self, approval_id: str, approved: bool) -> bool:
        """Resolve one pending request; approvals cannot be reused."""

        with self._lock:
            pending = self._pending.get(approval_id)
            if pending is None or pending.completed.is_set():
                return False
            pending.approved = approved
            pending.completed.set()
        return True

    def deny_all(self) -> None:
        """Release pending workers safely when a conversation is cancelled."""

        with self._lock:
            pending = tuple(self._pending.values())
        for item in pending:
            item.approved = False
            item.completed.set()


def _safe_arguments(call: ToolCall) -> dict[str, object]:
    """Describe the approval target without copying patch or file content."""

    arguments = call.arguments or {}
    if call.name in {"list_files", "read_file"}:
        return {"path": arguments.get("path")}
    if call.name == "apply_patch":
        return {
            "path": arguments.get("path"),
            "expected_text_chars": _length(arguments.get("expected_text")),
            "replacement_text_chars": _length(arguments.get("replacement_text")),
        }
    if call.name == "run_command":
        command = arguments.get("command")
        return {"command": command if isinstance(command, list) else None}
    return {"argument_keys": sorted(arguments.keys())}


def _length(value: object) -> int | None:
    """Return text length without retaining user or repository data."""

    return len(value) if isinstance(value, str) else None
