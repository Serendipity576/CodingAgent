"""Durable multi-turn conversations built from the bounded Agent runtime."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
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
from agent.conversation_store import ConversationStore, StoredConversation
from agent.llm.client import LLMClient, build_llm_client
from agent.security.approval import ApprovalHandler
from agent.tools import build_default_registry
from agent.tools.registry import ToolRegistry
from agent.trace import TurnTraceRecorder, public_turn_trace, trace_item_detail


class ConversationState(str, Enum):
    """Visible lifecycle states for one persisted conversation session."""

    IDLE = "idle"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
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
    """Own one local transcript, event journal, message queue, and turn worker."""

    def __init__(
        self,
        *,
        settings: Settings,
        workspace: Path,
        execution_lock: Lock,
        store: ConversationStore,
        approval_factory: ApprovalFactory | None = None,
        max_turns: int | None = None,
        max_history_items: int | None = None,
        restored: StoredConversation | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self._settings = settings
        self._execution_lock = execution_lock
        self._store = store
        self._max_turns = max_turns or settings.limits.max_conversation_turns
        self._max_history_items = max_history_items or settings.limits.max_history_items
        self.id = restored.conversation_id if restored else uuid4().hex
        self._created_at = restored.created_at if restored else time()
        self._events: list[ConversationEvent] = []
        # Browser retries reuse this bounded per-session id set so one accepted
        # message cannot become two turns when the HTTP response is lost.
        self._client_message_ids: set[str] = set()
        self._events_ready = Condition(RLock())
        self._queue: deque[str] = deque()
        self._lock = RLock()
        self._cancel_event = Event()
        self._worker: Thread | None = None
        self._state = ConversationState.IDLE
        self._turn_count = 0
        self._latest_result: TaskResult | None = None
        self._latest_result_data: dict[str, object] | None = None
        self._turn_traces: dict[int, dict[str, object]] = {}
        self._active_trace: TurnTraceRecorder | None = None
        self._deleted = False
        self._llm: LLMClient = build_llm_client(
            settings,
            trace_callback=self._record_llm_trace,
        )
        approval = approval_factory(self.publish) if approval_factory else None
        self._approval = approval
        self._tools: ToolRegistry = build_default_registry(self.workspace, approval=approval)

        if restored is None:
            self._persist()
            self.publish("conversation_created", {"workspace": str(self.workspace)})
        else:
            self._restore(restored)

    @property
    def state(self) -> ConversationState:
        """Return the current lifecycle state."""

        with self._lock:
            return self._state

    @property
    def latest_result(self) -> TaskResult | None:
        """Return the last in-process task result, if one exists."""

        with self._lock:
            return self._latest_result

    def submit(self, message: str, *, client_message_id: str | None = None) -> bool:
        """Queue one user message, accepting an exact browser retry only once."""

        text = message.strip()
        if not text:
            raise ValueError("message must not be empty")
        if client_message_id is not None and (
            not client_message_id.strip() or len(client_message_id) > 128
        ):
            raise ValueError("client_message_id must contain at most 128 characters")
        with self._lock:
            if client_message_id is not None and client_message_id in self._client_message_ids:
                return True
            if self._deleted or self._state in {
                ConversationState.CLOSED,
                ConversationState.LIMIT_REACHED,
            }:
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
            if client_message_id is not None:
                self._client_message_ids.add(client_message_id)
            details: dict[str, object] = {"text": text}
            if client_message_id is not None:
                details["client_message_id"] = client_message_id
            self.publish("user_message", details)
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
        self._deny_pending_approvals()
        self.publish("turn_cancel_requested", {})
        return True

    def close(self) -> None:
        """Reject future messages while retaining the transcript for later review."""

        with self._lock:
            if self._deleted:
                return
            self._state = ConversationState.CLOSED
            self._queue.clear()
            self._cancel_event.set()
        self._deny_pending_approvals()
        self.publish("conversation_closed", {})

    def delete(self) -> None:
        """Stop work and permanently erase this session's local stored state."""

        with self._lock:
            if self._deleted:
                return
            self._deleted = True
            self._state = ConversationState.CLOSED
            self._queue.clear()
            self._cancel_event.set()
            # Hold the same lock used by publish() until the database row is
            # gone, so a finishing worker cannot recreate a deleted session.
            self._store.delete_conversation(self.id)
        self._deny_pending_approvals()

    def resolve_approval(self, approval_id: str, approved: bool) -> bool:
        """Resolve one active Web approval without exposing its handler externally."""

        resolver = getattr(self._approval, "resolve", None)
        return bool(resolver(approval_id, approved)) if callable(resolver) else False

    def publish(self, event: str, details: Mapping[str, object]) -> None:
        """Append a safe event, durably journal it, and wake active observers."""

        with self._lock, self._events_ready:
            if self._deleted:
                return
            item = ConversationEvent(
                sequence=len(self._events) + 1,
                event=event,
                timestamp=time(),
                details=dict(details),
            )
            self._events.append(item)
            self._events_ready.notify_all()
            # Persist state after every visible transition. This includes model
            # transcript changes made immediately before runtime event emission.
            self._persist()
            self._store.append_event(
                self.id,
                sequence=item.sequence,
                event=item.event,
                timestamp=item.timestamp,
                details=item.details,
            )

    def events_after(self, sequence: int, *, timeout_seconds: float = 15) -> list[ConversationEvent]:
        """Wait for events newer than ``sequence`` for event-stream clients."""

        with self._events_ready:
            matching = [item for item in self._events if item.sequence > sequence]
            if not matching:
                self._events_ready.wait(timeout_seconds)
                matching = [item for item in self._events if item.sequence > sequence]
            return matching

    def latest_event_sequence(self) -> int:
        """Return the current journal position for terminal incremental rendering."""

        with self._events_ready:
            return len(self._events)

    def turn_traces(self) -> list[dict[str, object]]:
        """Return all turn traces without private request or response bodies."""

        with self._lock:
            return [
                public_turn_trace(self._turn_traces[turn_id])
                for turn_id in sorted(self._turn_traces)
            ]

    def turn_trace_item(self, turn_id: int, item_id: str) -> dict[str, object] | None:
        """Return one explicitly requested local trace detail record."""

        with self._lock:
            trace = self._turn_traces.get(turn_id)
            return trace_item_detail(trace, item_id) if trace is not None else None

    def snapshot(self) -> dict[str, object]:
        """Return a compact session view without exposing the local LLM transcript."""

        with self._lock:
            result = self._latest_result_data or _result_data(self._latest_result)
            return {
                "conversation_id": self.id,
                "workspace": str(self.workspace),
                "state": self._state.value,
                "turn_count": self._turn_count,
                "max_turns": self._max_turns,
                "queued_messages": len(self._queue),
                "history_items": self._history_item_count(),
                "max_history_items": self._max_history_items,
                "latest_status": result.get("status") if result else None,
                "latest_message": result.get("message") if result else None,
                "summary": result.get("summary") if result else None,
            }

    def _restore(self, restored: StoredConversation) -> None:
        """Rebuild a session from local storage without restarting active work."""

        with self._events_ready:
            self._events = [
                ConversationEvent(
                    sequence=item.sequence,
                    event=item.event,
                    timestamp=item.timestamp,
                    details=dict(item.details),
                )
                for item in restored.events
            ]
        self._client_message_ids = {
            client_message_id
            for item in self._events
            if item.event == "user_message"
            and isinstance(
                client_message_id := item.details.get("client_message_id"), str
            )
        }
        self._turn_count = restored.turn_count
        self._max_turns = restored.max_turns
        self._max_history_items = restored.max_history_items
        self._latest_result_data = dict(restored.latest_result) if restored.latest_result else None
        self._turn_traces = {
            item.turn_id: deepcopy(dict(item.data)) for item in restored.turn_traces
        }
        self._state = _stored_state(restored.state)
        _restore_history(self._llm, restored.transcript)

        if self._state is ConversationState.RUNNING:
            # A process restart cannot prove whether an external command or
            # request completed. Never automatically replay that active turn.
            self._state = ConversationState.INTERRUPTED
            self.publish(
                "conversation_interrupted",
                {"reason": "server_restart", "message": "上一轮在服务重启时中断，请确认后继续。"},
            )
        else:
            self._persist()

    def _run_queue(self) -> None:
        """Process one workspace's queued turns serially in a daemon worker."""

        while True:
            with self._lock:
                if self._deleted or self._state is ConversationState.CLOSED or not self._queue:
                    if (
                        not self._deleted
                        and self._state not in {ConversationState.CLOSED, ConversationState.IDLE}
                    ):
                        self._state = ConversationState.IDLE
                        self._persist()
                    return
                message = self._queue.popleft()
                self._state = ConversationState.RUNNING
                self._turn_count += 1
                turn_id = self._turn_count
                self._cancel_event.clear()

            self.publish("conversation_turn_started", {"turn_id": turn_id})
            with self._execution_lock:
                trace = TurnTraceRecorder(
                    conversation_id=self.id,
                    turn_id=turn_id,
                    on_change=self._save_turn_trace,
                )
                with self._lock:
                    self._active_trace = trace
                turn_git_baseline = GitStatusSnapshot.capture(self.workspace)
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
                    git_baseline=turn_git_baseline,
                    event_callback=self.publish,
                    cancellation=self._cancel_event,
                    trace=trace,
                )
                result = agent.run(message)
                with self._lock:
                    self._active_trace = None

            with self._lock:
                if self._deleted:
                    return
                self._latest_result = result
                self._latest_result_data = _result_data(result)
                # Persist the completed turn as idle before announcing it when
                # no message remains queued. A restart in this narrow window
                # must not misclassify finished work as active work.
                if not self._queue:
                    self._state = ConversationState.IDLE
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

    def _record_llm_trace(self, event: str, details: Mapping[str, object]) -> None:
        """Attach exact adapter payloads to the active turn without sending them to SSE."""

        with self._lock:
            trace = self._active_trace
        if trace is not None:
            trace.record_llm_payload(event, details)

    def _save_turn_trace(self, trace: dict[str, object]) -> None:
        """Persist trace updates locally and emit only a compact refresh signal."""

        turn_id = trace.get("turn_id")
        if not isinstance(turn_id, int) or turn_id <= 0:
            return
        with self._lock:
            if self._deleted:
                return
            self._turn_traces[turn_id] = deepcopy(trace)
            self._store.save_turn_trace(self.id, turn_id=turn_id, data=trace)
        self.publish(
            "trace_updated",
            {
                "turn_id": turn_id,
                "status": trace.get("status"),
                "item_count": len(trace.get("items", ())),
            },
        )

    def _persist(self) -> None:
        """Save a consistent metadata/transcript snapshot unless deletion won the race."""

        with self._lock:
            if self._deleted:
                return
            self._store.save_conversation(
                conversation_id=self.id,
                created_at=self._created_at,
                state=self._state.value,
                turn_count=self._turn_count,
                max_turns=self._max_turns,
                max_history_items=self._max_history_items,
                transcript=_export_history(self._llm),
                latest_result=self._latest_result_data or _result_data(self._latest_result),
            )

    def _history_item_count(self) -> int:
        """Read a local adapter's transcript size; generic test doubles report zero."""

        exported = _export_history(self._llm)
        if exported:
            return len(exported)
        history = getattr(self._llm, "_history", ())
        return len(history) if isinstance(history, list) else 0

    def _deny_pending_approvals(self) -> None:
        """Release a browser approval wait when cancellation, close, or delete occurs."""

        deny_all = getattr(self._approval, "deny_all", None)
        if callable(deny_all):
            deny_all()


class ConversationManager:
    """Load, create, resume, and delete workspace-scoped durable sessions."""

    def __init__(
        self,
        settings: Settings,
        workspace: Path,
        *,
        approval_factory: ApprovalFactory | None = None,
        max_turns: int | None = None,
        max_history_items: int | None = None,
    ) -> None:
        self._settings = settings
        self._workspace = workspace.resolve()
        self._approval_factory = approval_factory
        self._max_turns = max_turns
        self._max_history_items = max_history_items
        self._execution_lock = Lock()
        self._store = ConversationStore(self._workspace)
        self._sessions: dict[str, ConversationSession] = {}
        self._lock = RLock()
        for stored in self._store.load_conversations():
            session = self._build_session(restored=stored)
            self._sessions[session.id] = session

    def create(self, approval_factory: ApprovalFactory | None = None) -> ConversationSession:
        """Create a new transcript without sharing context with existing sessions."""

        session = self._build_session(approval_factory=approval_factory)
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, conversation_id: str) -> ConversationSession | None:
        """Return one known session without creating an implicit conversation."""

        with self._lock:
            return self._sessions.get(conversation_id)

    def resolve(self, identifier: str) -> ConversationSession | None:
        """Resolve a full id or one unambiguous prefix for terminal recovery commands."""

        with self._lock:
            if identifier in self._sessions:
                return self._sessions[identifier]
            matches = [item for item in self._sessions.values() if item.id.startswith(identifier)]
            return matches[0] if len(matches) == 1 else None

    def delete(self, conversation_id: str) -> bool:
        """Permanently delete one session's transcript, events, and metadata."""

        with self._lock:
            session = self._sessions.pop(conversation_id, None)
        if session is None:
            return False
        session.delete()
        return True

    def snapshots(self) -> list[dict[str, object]]:
        """List current and restored conversation metadata in recent-first order."""

        with self._lock:
            return [session.snapshot() for session in self._sessions.values()]

    def _build_session(
        self,
        *,
        approval_factory: ApprovalFactory | None = None,
        restored: StoredConversation | None = None,
    ) -> ConversationSession:
        """Construct a session with shared workspace serialization and durable storage."""

        return ConversationSession(
            settings=self._settings,
            workspace=self._workspace,
            execution_lock=self._execution_lock,
            store=self._store,
            approval_factory=approval_factory or self._approval_factory,
            max_turns=self._max_turns,
            max_history_items=self._max_history_items,
            restored=restored,
        )


def _stored_state(value: str) -> ConversationState:
    """Convert a database state string into the current explicit lifecycle enum."""

    try:
        return ConversationState(value)
    except ValueError as error:
        raise ValueError(f"stored conversation state is invalid: {value}") from error


def _export_history(llm: object) -> tuple[dict[str, object], ...]:
    """Copy optional transcript support without forcing lightweight test doubles."""

    exporter = getattr(llm, "export_history", None)
    if not callable(exporter):
        return ()
    exported = exporter()
    if not isinstance(exported, Sequence) or isinstance(exported, str):
        raise ValueError("LLM transcript export must return a sequence")
    if not all(isinstance(item, Mapping) for item in exported):
        raise ValueError("LLM transcript items must be mappings")
    return tuple(dict(item) for item in exported)


def _restore_history(llm: object, transcript: Sequence[Mapping[str, object]]) -> None:
    """Restore local transcript state when the configured client supports it."""

    if not transcript:
        return
    restorer = getattr(llm, "restore_history", None)
    if not callable(restorer):
        raise ValueError("configured LLM client cannot restore persisted conversation history")
    restorer(transcript)


def _result_data(result: TaskResult | None) -> dict[str, object] | None:
    """Persist only snapshot fields the UI can safely display after restart."""

    if result is None:
        return None
    return {
        "status": result.status.value,
        "message": result.message,
        "steps": result.steps,
        "summary": result.summary.as_dict() if result.summary else None,
        "audit_log": str(result.audit_log) if result.audit_log else None,
    }
