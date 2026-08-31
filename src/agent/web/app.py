"""FastAPI application serving local Coding Agent conversations and events."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.config import Settings
from agent.conversation import ConversationManager, ConversationSession
from agent.web.approval import WebApproval


class MessageBody(BaseModel):
    """One user message submitted from the local browser page."""

    text: Annotated[str, Field(min_length=1, max_length=20_000)]


class ApprovalBody(BaseModel):
    """One explicit browser decision for a single high-risk tool call."""

    approved: bool


def create_app(settings: Settings, workspace: Path) -> FastAPI:
    """Build a loopback-only Web application backed by durable local sessions."""

    app = FastAPI(title="Coding Agent", docs_url=None, redoc_url=None)
    manager = ConversationManager(
        settings,
        workspace,
        approval_factory=lambda publish: WebApproval(publish),
    )
    static_dir = Path(__file__).with_name("static")
    app.state.manager = manager

    @app.get("/")
    async def index() -> FileResponse:
        """Serve the React application's built entry page."""

        return FileResponse(static_dir / "index.html")

    # StaticFiles confines requests to the pre-built Web bundle directory; it
    # never resolves browser-provided paths outside this application asset root.
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/api/config")
    async def config() -> dict[str, object]:
        """Expose a redacted configuration summary to the local page."""

        return settings.public_dict()

    @app.get("/api/conversations")
    async def conversations() -> list[dict[str, object]]:
        """List active in-memory conversations for the sidebar."""

        return manager.snapshots()

    @app.post("/api/conversations", status_code=201)
    async def create_conversation() -> dict[str, object]:
        """Create one durable local conversation with browser approvals."""

        try:
            session = manager.create()
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return session.snapshot()

    @app.get("/api/conversations/{conversation_id}")
    async def conversation(conversation_id: str) -> dict[str, object]:
        """Return current metadata for one known session."""

        return _session_or_404(manager, conversation_id).snapshot()

    @app.post("/api/conversations/{conversation_id}/messages", status_code=202)
    async def send_message(conversation_id: str, body: MessageBody) -> dict[str, object]:
        """Queue a user message without blocking the HTTP request on an LLM call."""

        session = _session_or_404(manager, conversation_id)
        try:
            accepted = session.submit(body.text)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        if not accepted:
            raise HTTPException(status_code=409, detail="conversation is closed or at its turn limit")
        return session.snapshot()

    @app.post("/api/conversations/{conversation_id}/cancel")
    async def cancel(conversation_id: str) -> dict[str, bool]:
        """Cancel current work and deny any approval waiting in that session."""

        session = _session_or_404(manager, conversation_id)
        return {"cancelled": session.cancel()}

    @app.post("/api/conversations/{conversation_id}/approvals/{approval_id}")
    async def resolve_approval(
        conversation_id: str,
        approval_id: str,
        body: ApprovalBody,
    ) -> dict[str, bool]:
        """Resolve exactly one active approval request from the local page."""

        session = _session_or_404(manager, conversation_id)
        if not session.resolve_approval(approval_id, body.approved):
            raise HTTPException(status_code=404, detail="approval is not pending")
        return {"accepted": True}

    @app.delete("/api/conversations/{conversation_id}", status_code=204)
    async def delete_conversation(conversation_id: str) -> Response:
        """Permanently erase one local transcript and its browser event journal."""

        if not manager.delete(conversation_id):
            raise HTTPException(status_code=404, detail="conversation not found")
        return Response(status_code=204)

    @app.get("/api/conversations/{conversation_id}/events")
    async def events(conversation_id: str, after: int = 0) -> StreamingResponse:
        """Stream safe session events, replaying any events after the given id."""

        session = _session_or_404(manager, conversation_id)
        return StreamingResponse(
            _event_stream(session, after),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


async def _event_stream(
    session: ConversationSession,
    after: int,
) -> AsyncIterator[str]:
    """Adapt the thread-backed event condition to a browser SSE stream."""

    sequence = after
    while True:
        events = await asyncio.to_thread(session.events_after, sequence)
        if not events:
            yield ": keep-alive\n\n"
            continue
        for item in events:
            sequence = item.sequence
            payload = json.dumps(item.as_dict(), ensure_ascii=False)
            yield f"id: {item.sequence}\nevent: {item.event}\ndata: {payload}\n\n"


def _session_or_404(manager: ConversationManager, conversation_id: str) -> ConversationSession:
    """Resolve an id without creating an accidental new conversation."""

    session = manager.get(conversation_id)
    if session is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return session


def run_server(settings: Settings, workspace: Path, *, host: str, port: int) -> None:
    """Run only on an explicit loopback address to keep local tools local."""

    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("the Web interface may only listen on a loopback address")
    import uvicorn

    uvicorn.run(create_app(settings, workspace), host=host, port=port)
