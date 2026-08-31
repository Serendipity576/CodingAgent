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
    """Run a small REPL whose normal messages share one local LLM transcript."""

    manager = ConversationManager(settings, workspace)
    session = _new_session(manager)
    write("Interactive Coding Agent. Commands: /help, /new, /status, /cancel, /quit")

    while True:
        try:
            message = read_input("You> ").strip()
        except EOFError:
            session.close()
            return 0
        if not message:
            continue
        if message == "/quit":
            session.close()
            return 0
        if message == "/help":
            write("/new starts a clean conversation; /status shows the active session.")
            continue
        if message == "/new":
            session.close()
            session = _new_session(manager)
            write("Started a new conversation.")
            continue
        if message == "/status":
            write(json.dumps(session.snapshot(), ensure_ascii=False, indent=2))
            continue
        if message == "/cancel":
            write("Cancellation requested." if session.cancel() else "No active turn to cancel.")
            continue

        if not session.submit(message):
            write("Conversation is closed or reached its configured turn limit.")
            continue
        _print_turn_events(session, write)


def _new_session(manager: ConversationManager) -> ConversationSession:
    """Create a terminal session with the existing interactive approval prompt."""

    return manager.create(lambda _publish: ConsoleApproval())


def _print_turn_events(session: ConversationSession, write: Callable[[str], None]) -> None:
    """Render concise progress without exposing provider reasoning data."""

    sequence = 0
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
