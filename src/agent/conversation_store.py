"""Durable, workspace-local storage for resumable Agent conversations."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from threading import Lock
from time import time


class ConversationStoreError(RuntimeError):
    """Raised when durable session data cannot be read or written safely."""


@dataclass(frozen=True, slots=True)
class StoredEvent:
    """One browser-safe event recovered from the local event journal."""

    sequence: int
    event: str
    timestamp: float
    details: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class StoredTurnTrace:
    """One private structured runtime trace associated with a completed or active turn."""

    turn_id: int
    data: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class StoredConversation:
    """Complete local state required to restore one durable conversation."""

    conversation_id: str
    created_at: float
    state: str
    turn_count: int
    max_turns: int
    max_history_items: int
    transcript: tuple[dict[str, object], ...]
    latest_result: Mapping[str, object] | None
    events: tuple[StoredEvent, ...]
    turn_traces: tuple[StoredTurnTrace, ...]


class ConversationStore:
    """Persist one workspace's sessions without exposing them to the Web client."""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.resolve()
        self._directory = self._workspace / ".agent" / "conversations"
        self._path = self._directory / "sessions.sqlite3"
        self._lock = Lock()
        self._directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        _protect_path(self._directory, 0o700)
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    state TEXT NOT NULL,
                    turn_count INTEGER NOT NULL,
                    max_turns INTEGER NOT NULL,
                    max_history_items INTEGER NOT NULL,
                    transcript_json TEXT NOT NULL,
                    latest_result_json TEXT
                );
                CREATE TABLE IF NOT EXISTS conversation_events (
                    conversation_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    details_json TEXT NOT NULL,
                    PRIMARY KEY (conversation_id, sequence),
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(conversation_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS conversation_updated_at
                    ON conversations(updated_at DESC);
                CREATE TABLE IF NOT EXISTS conversation_turn_traces (
                    conversation_id TEXT NOT NULL,
                    turn_id INTEGER NOT NULL,
                    data_json TEXT NOT NULL,
                    PRIMARY KEY (conversation_id, turn_id),
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(conversation_id)
                        ON DELETE CASCADE
                );
                """
            )

    @property
    def path(self) -> Path:
        """Return the local database path for diagnostics, never for the Web API."""

        return self._path

    def save_conversation(
        self,
        *,
        conversation_id: str,
        created_at: float,
        state: str,
        turn_count: int,
        max_turns: int,
        max_history_items: int,
        transcript: Sequence[Mapping[str, object]],
        latest_result: Mapping[str, object] | None,
    ) -> None:
        """Atomically upsert metadata and the client-owned model transcript."""

        transcript_json = _json_text([dict(item) for item in transcript])
        result_json = _json_text(dict(latest_result)) if latest_result is not None else None
        updated_at = time()
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO conversations (
                    conversation_id, created_at, updated_at, state, turn_count,
                    max_turns, max_history_items, transcript_json, latest_result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    state = excluded.state,
                    turn_count = excluded.turn_count,
                    max_turns = excluded.max_turns,
                    max_history_items = excluded.max_history_items,
                    transcript_json = excluded.transcript_json,
                    latest_result_json = excluded.latest_result_json
                """,
                (
                    conversation_id,
                    created_at,
                    updated_at,
                    state,
                    turn_count,
                    max_turns,
                    max_history_items,
                    transcript_json,
                    result_json,
                ),
            )

    def append_event(
        self,
        conversation_id: str,
        *,
        sequence: int,
        event: str,
        timestamp: float,
        details: Mapping[str, object],
    ) -> None:
        """Append one safe event after its session row has been persisted."""

        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO conversation_events (
                    conversation_id, sequence, event, timestamp, details_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, sequence, event, timestamp, _json_text(dict(details))),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (time(), conversation_id),
            )

    def save_turn_trace(
        self,
        conversation_id: str,
        *,
        turn_id: int,
        data: Mapping[str, object],
    ) -> None:
        """Upsert a bounded private trace without placing its bodies in the event journal."""

        if turn_id <= 0:
            raise ConversationStoreError("turn id must be positive")
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO conversation_turn_traces (conversation_id, turn_id, data_json)
                VALUES (?, ?, ?)
                ON CONFLICT(conversation_id, turn_id) DO UPDATE SET
                    data_json = excluded.data_json
                """,
                (conversation_id, turn_id, _json_text(dict(data))),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                (time(), conversation_id),
            )

    def load_conversations(self) -> tuple[StoredConversation, ...]:
        """Load all sessions and their safe event journals in recent-first order."""

        with self._lock, self._connection() as connection:
            rows = connection.execute(
                """
                SELECT conversation_id, created_at, state, turn_count, max_turns,
                       max_history_items, transcript_json, latest_result_json
                FROM conversations
                ORDER BY updated_at DESC, created_at DESC
                """
            ).fetchall()
            events = connection.execute(
                """
                SELECT conversation_id, sequence, event, timestamp, details_json
                FROM conversation_events
                ORDER BY conversation_id, sequence
                """
            ).fetchall()
            turn_traces = connection.execute(
                """
                SELECT conversation_id, turn_id, data_json
                FROM conversation_turn_traces
                ORDER BY conversation_id, turn_id
                """
            ).fetchall()

        events_by_conversation: dict[str, list[StoredEvent]] = {}
        for row in events:
            conversation_id = _required_text(row["conversation_id"], "conversation id")
            events_by_conversation.setdefault(conversation_id, []).append(
                StoredEvent(
                    sequence=_required_nonnegative_int(row["sequence"], "event sequence"),
                    event=_required_text(row["event"], "event name"),
                    timestamp=_required_float(row["timestamp"], "event timestamp"),
                    details=_mapping_json(row["details_json"], "event details"),
                )
            )

        traces_by_conversation: dict[str, list[StoredTurnTrace]] = {}
        for row in turn_traces:
            conversation_id = _required_text(row["conversation_id"], "conversation id")
            traces_by_conversation.setdefault(conversation_id, []).append(
                StoredTurnTrace(
                    turn_id=_required_positive_int(row["turn_id"], "turn id"),
                    data=_mapping_json(row["data_json"], "turn trace"),
                )
            )

        return tuple(
            StoredConversation(
                conversation_id=_required_text(row["conversation_id"], "conversation id"),
                created_at=_required_float(row["created_at"], "created timestamp"),
                state=_required_text(row["state"], "conversation state"),
                turn_count=_required_nonnegative_int(row["turn_count"], "turn count"),
                max_turns=_required_positive_int(row["max_turns"], "maximum turns"),
                max_history_items=_required_positive_int(
                    row["max_history_items"], "maximum history items"
                ),
                transcript=_transcript_json(row["transcript_json"]),
                latest_result=_optional_mapping_json(row["latest_result_json"], "latest result"),
                events=tuple(events_by_conversation.get(str(row["conversation_id"]), ())),
                turn_traces=tuple(traces_by_conversation.get(str(row["conversation_id"]), ())),
            )
            for row in rows
        )

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete one transcript and its safe event journal together."""

        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,)
            )
        return cursor.rowcount == 1

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Open a short-lived connection so worker threads never share SQLite state."""

        try:
            connection = sqlite3.connect(self._path, timeout=10)
        except sqlite3.Error as error:
            raise ConversationStoreError(f"cannot open conversation store: {error}") from error
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            _protect_path(self._path, 0o600)
            _protect_path(self._path.with_name(f"{self._path.name}-wal"), 0o600)
            _protect_path(self._path.with_name(f"{self._path.name}-shm"), 0o600)
            yield connection
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            connection.close()


def _protect_path(path: Path, mode: int) -> None:
    """Tighten permissions on POSIX while remaining usable on other platforms."""

    if not path.exists():
        return
    try:
        path.chmod(mode)
    except OSError:
        # Windows ACLs are not represented by chmod. The current user still
        # owns the application data directory created above.
        return


def _json_text(value: object) -> str:
    """Encode only JSON-compatible local state; never silently stringify secrets."""

    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ConversationStoreError("conversation state is not JSON serializable") from error


def _decode_json(raw: object, label: str) -> object:
    """Decode one persisted JSON column with an actionable corruption error."""

    if not isinstance(raw, str):
        raise ConversationStoreError(f"stored {label} is not text")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise ConversationStoreError(f"stored {label} is invalid JSON") from error


def _mapping_json(raw: object, label: str) -> Mapping[str, object]:
    """Decode a JSON object without trusting the database shape implicitly."""

    value = _decode_json(raw, label)
    if not isinstance(value, dict):
        raise ConversationStoreError(f"stored {label} is not an object")
    return value


def _optional_mapping_json(raw: object, label: str) -> Mapping[str, object] | None:
    """Decode a nullable JSON object column."""

    return None if raw is None else _mapping_json(raw, label)


def _transcript_json(raw: object) -> tuple[dict[str, object], ...]:
    """Decode the exact client-owned Response input transcript."""

    value = _decode_json(raw, "transcript")
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ConversationStoreError("stored transcript is not a list of objects")
    return tuple(dict(item) for item in value)


def _required_text(value: object, label: str) -> str:
    """Validate non-empty persisted identifiers and labels."""

    if not isinstance(value, str) or not value:
        raise ConversationStoreError(f"stored {label} is invalid")
    return value


def _required_nonnegative_int(value: object, label: str) -> int:
    """Validate counters read from SQLite's dynamically typed columns."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConversationStoreError(f"stored {label} is invalid")
    return value


def _required_positive_int(value: object, label: str) -> int:
    """Validate positive persisted limits."""

    parsed = _required_nonnegative_int(value, label)
    if parsed == 0:
        raise ConversationStoreError(f"stored {label} must be positive")
    return parsed


def _required_float(value: object, label: str) -> float:
    """Validate timestamps without accepting NaN or nonnumeric data."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConversationStoreError(f"stored {label} is invalid")
    parsed = float(value)
    if parsed != parsed:
        raise ConversationStoreError(f"stored {label} is invalid")
    return parsed
