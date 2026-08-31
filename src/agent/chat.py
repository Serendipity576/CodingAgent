"""Interactive terminal shell for one persistent Agent conversation."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from time import sleep

from agent.config import Settings
from agent.conversation import ConversationManager, ConversationSession
from agent.security.approval import ConsoleApproval


def run_chat(
    settings: Settings,
    workspace: Path,
    *,
    read_input: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> int:
    """Run a REPL that creates, lists, and resumes durable local sessions."""

    manager = ConversationManager(
        settings,
        workspace,
        approval_factory=lambda _publish: ConsoleApproval(),
    )
    session = _initial_session(manager)
    write(
        "Interactive Coding Agent. Commands: /help, /new, /sessions, /open ID, "
        "/status, /cancel, /quit"
    )

    while True:
        try:
            message = read_input("You> ").strip()
        except EOFError:
            return 0
        if not message:
            continue
        if message == "/quit":
            return 0
        if message == "/help":
            write("/new creates a session; /sessions lists saved sessions; /open ID resumes one.")
            continue
        if message == "/new":
            session = _new_session(manager)
            write("Started a new conversation.")
            continue
        if message == "/sessions":
            _print_sessions(manager, write)
            continue
        if message.startswith("/open "):
            identifier = message.removeprefix("/open ").strip()
            restored = manager.resolve(identifier)
            if restored is None:
                write("No saved conversation matches that id.")
            else:
                session = restored
                write(f"Opened conversation {session.id[:8]} ({session.state.value}).")
            continue
        if message == "/status":
            write(json.dumps(session.snapshot(), ensure_ascii=False, indent=2))
            continue
        if message == "/cancel":
            write("Cancellation requested." if session.cancel() else "No active turn to cancel.")
            continue

        sequence = session.latest_event_sequence()
        if not session.submit(message):
            write("Conversation is closed or reached its configured turn limit.")
            continue
        _print_turn_events(session, write, sequence=sequence)


def _new_session(manager: ConversationManager) -> ConversationSession:
    """Create a terminal session using the manager's ConsoleApproval factory."""

    return manager.create()


def _initial_session(manager: ConversationManager) -> ConversationSession:
    """Open the newest resumable session, creating one only when none exist."""

    for snapshot in manager.snapshots():
        if snapshot["state"] not in {"closed", "limit_reached"}:
            restored = manager.get(str(snapshot["conversation_id"]))
            if restored is not None:
                return restored
    return _new_session(manager)


def _print_sessions(manager: ConversationManager, write: Callable[[str], None]) -> None:
    """List persisted session ids without printing transcript or model internals."""

    snapshots = manager.snapshots()
    if not snapshots:
        write("No saved conversations.")
        return
    for snapshot in snapshots:
        write(
            f"{str(snapshot['conversation_id'])[:8]} "
            f"{snapshot['state']} {snapshot['turn_count']}/{snapshot['max_turns']} turns"
        )


def _print_turn_events(
    session: ConversationSession,
    write: Callable[[str], None],
    *,
    sequence: int,
) -> None:
    """Render concise progress without exposing provider reasoning data."""

    while True:
        events = session.events_after(sequence, timeout_seconds=1)
        for item in events:
            sequence = item.sequence
            if item.event == "assistant_message":
                write(f"Agent> {item.details.get('text', '')}")
            elif item.event == "tool_requested":
                write(f"[tool] {item.details.get('tool')}")
            elif item.event == "tool_finished":
                outcome = "ok" if item.details.get("success") else "failed"
                write(f"[tool] {item.details.get('tool')} {outcome}")
            elif item.event == "conversation_turn_finished":
                write(f"[status] {item.details.get('status')}")
                return
        sleep(0.05)
