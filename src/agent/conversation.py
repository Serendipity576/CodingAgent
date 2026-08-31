"""In-memory multi-turn conversations built from the bounded Agent runtime."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Condition, Event, Lock, RLock, Thread
from time import time
from typing import Protocol
from uuid import uuid4

from agent.agent import CodingAgent, TaskResult
from agent.audit import AuditLogger
from agent.change_tracker import GitStatusSnapshot
from agent.config import Settings
from agent.llm.client import LLMClient, build_llm_client
from agent.security.approval import ApprovalHandler
from agent.tools import build_default_registry
from agent.tools.registry import ToolRegistry


class ConversationState(str, Enum):
    """Visible lifecycle states for one conversation session."""

    IDLE = "idle"
    RUNNING = "running"
    CLOSED = "closed"
    LIMIT_REACHED = "limit_reached"


@dataclass(frozen=True, slots=True)
class ConversationEvent:
    """One safe, ordered fact rendered by terminal or Web observers."""

    sequence: int
    event: str
    timestamp: float
    details: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation without private model state."""

        return {
            "sequence": self.sequence,
            "event": self.event,
            "timestamp": self.timestamp,
            "details": dict(self.details),
        }


class ApprovalFactory(Protocol):
    """Build an approval handler connected to one conversation's events."""

    def __call__(
        self,
        publish: Callable[[str, Mapping[str, object]], None],
    ) -> ApprovalHandler | None:
        """Return the handler used by the session's ToolRegistry."""


class ConversationSession:
    """Own one LLM history, event stream, message queue, and turn worker."""

    def __init__(
        self,
        *,
        settings: Settings,
        workspace: Path,
        execution_lock: Lock,
        approval_factory: ApprovalFactory | None = None,
        max_turns: int | None = None,
        max_history_items: int | None = None,
    ) -> None:
        self.id = uuid4().hex
        self.workspace = workspace.resolve()
        self._settings = settings
        self._execution_lock = execution_lock
        self._max_turns = max_turns or settings.limits.max_conversation_turns
        self._max_history_items = max_history_items or settings.limits.max_history_items
        self._llm: LLMClient = build_llm_client(settings)
        self._events: list[ConversationEvent] = []
        self._events_ready = Condition(RLock())
        self._queue: deque[str] = deque()
        self._lock = RLock()
        self._cancel_event = Event()
        self._worker: Thread | None = None
        self._state = ConversationState.IDLE
        self._turn_count = 0
        self._latest_result: TaskResult | None = None
        # A session captures one baseline, so Agent edits in an earlier turn do
        # not become falsely labelled as user changes in a later turn.
        self._git_baseline = GitStatusSnapshot.capture(self.workspace)
        approval = approval_factory(self.publish) if approval_factory else None
        self._tools: ToolRegistry = build_default_registry(self.workspace, approval=approval)
        self.publish("conversation_created", {"workspace": str(self.workspace)})

    @property
    def state(self) -> ConversationState:
        """Return the current lifecycle state."""

        with self._lock:
            return self._state

    @property
    def latest_result(self) -> TaskResult | None:
        """Return the result from the last finished user turn."""

        with self._lock:
            return self._latest_result

    def submit(self, message: str) -> bool:
        """Queue a user message and start the serial worker when necessary."""

        text = message.strip()
        if not text:
            raise ValueError("message must not be empty")
        with self._lock:
            if self._state in {ConversationState.CLOSED, ConversationState.LIMIT_REACHED}:
                return False
            if self._history_item_count() >= self._max_history_items:
                self._state = ConversationState.LIMIT_REACHED
                self.publish(
                    "conversation_limit_reached",
                    {"max_history_items": self._max_history_items, "reason": "history_limit"},
                )
                return False
            if self._turn_count + len(self._queue) >= self._max_turns:
                self._state = ConversationState.LIMIT_REACHED
                self.publish(
                    "conversation_limit_reached",
                    {"max_turns": self._max_turns, "reason": "turn_limit"},
                )
                return False
            self._queue.append(text)
            self.publish("user_message", {"text": text})
            if self._worker is None or not self._worker.is_alive():
                self._worker = Thread(target=self._run_queue, daemon=True)
                self._worker.start()
        return True

    def cancel(self) -> bool:
        """Request cancellation and discard user messages not started yet."""

        with self._lock:
            if self._state is not ConversationState.RUNNING:
                return False
            self._cancel_event.set()
            self._queue.clear()
        self.publish("turn_cancel_requested", {})
        return True

    def close(self) -> None:
        """Reject future messages and request cancellation of active work."""

        with self._lock:
            self._state = ConversationState.CLOSED
            self._queue.clear()
            self._cancel_event.set()
        self.publish("conversation_closed", {})

    def publish(self, event: str, details: Mapping[str, object]) -> None:
        """Append a safe event and wake SSE or terminal observers."""

        with self._events_ready:
            item = ConversationEvent(
                sequence=len(self._events) + 1,
                event=event,
                timestamp=time(),
                details=dict(details),
            )
            self._events.append(item)
            self._events_ready.notify_all()

    def events_after(self, sequence: int, *, timeout_seconds: float = 15) -> list[ConversationEvent]:
        """Wait for events newer than ``sequence`` for event-stream clients."""

        with self._events_ready:
            matching = [item for item in self._events if item.sequence > sequence]
            if not matching:
                self._events_ready.wait(timeout_seconds)
                matching = [item for item in self._events if item.sequence > sequence]
            return matching

    def snapshot(self) -> dict[str, object]:
        """Return a compact session view without exposing local LLM history."""

        with self._lock:
            result = self._latest_result
            return {
                "conversation_id": self.id,
                "workspace": str(self.workspace),
                "state": self._state.value,
                "turn_count": self._turn_count,
                "queued_messages": len(self._queue),
                "history_items": self._history_item_count(),
                "max_history_items": self._max_history_items,
                "latest_status": result.status.value if result else None,
                "latest_message": result.message if result else None,
                "summary": result.summary.as_dict() if result and result.summary else None,
            }

    def _run_queue(self) -> None:
        """Process one workspace's queued turns serially in a daemon worker."""

        while True:
            with self._lock:
                if self._state is ConversationState.CLOSED or not self._queue:
                    if self._state is not ConversationState.CLOSED:
                        self._state = ConversationState.IDLE
                    return
                message = self._queue.popleft()
                self._state = ConversationState.RUNNING
                self._turn_count += 1
                turn_id = self._turn_count
                self._cancel_event.clear()

            self.publish("conversation_turn_started", {"turn_id": turn_id})
            with self._execution_lock:
                logger = AuditLogger.create(
                    self.workspace,
                    task_id=f"{self.id}-{turn_id}",
                    conversation_id=self.id,
                    turn_id=turn_id,
                )
                agent = CodingAgent(
                    settings=self._settings,
                    llm=self._llm,
                    tools=self._tools,
                    audit_logger=logger,
                    git_baseline=self._git_baseline,
                    event_callback=self.publish,
                    cancellation=self._cancel_event,
                )
                result = agent.run(message)

            with self._lock:
                self._latest_result = result
            self.publish(
                "conversation_turn_finished",
                {
                    "turn_id": turn_id,
                    "status": result.status.value,
                    "message": result.message,
                    "steps": result.steps,
                    "summary": result.summary.as_dict() if result.summary else None,
                    "audit_log": str(result.audit_log) if result.audit_log else None,
                },
            )

    def _history_item_count(self) -> int:
        """Read only a local adapter's item count; generic fakes report zero."""

        history = getattr(self._llm, "_history", ())
        return len(history) if isinstance(history, list) else 0


class ConversationManager:
    """Create and retain in-memory sessions for one workspace-bound process."""

    def __init__(
        self,
        settings: Settings,
        workspace: Path,
        *,
        max_turns: int | None = None,
        max_history_items: int | None = None,
    ) -> None:
        self._settings = settings
        self._workspace = workspace.resolve()
        self._max_turns = max_turns
        self._max_history_items = max_history_items
        self._execution_lock = Lock()
        self._sessions: dict[str, ConversationSession] = {}
        self._lock = RLock()

    def create(self, approval_factory: ApprovalFactory | None = None) -> ConversationSession:
        """Create a session with an independent local LLM transcript."""

        session = ConversationSession(
            settings=self._settings,
            workspace=self._workspace,
            execution_lock=self._execution_lock,
            approval_factory=approval_factory,
            max_turns=self._max_turns,
            max_history_items=self._max_history_items,
        )
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, conversation_id: str) -> ConversationSession | None:
        """Return one known session without creating implicit conversations."""

        with self._lock:
            return self._sessions.get(conversation_id)

    def snapshots(self) -> list[dict[str, object]]:
        """List conversation metadata in creation order for a UI sidebar."""

        with self._lock:
            return [session.snapshot() for session in self._sessions.values()]
