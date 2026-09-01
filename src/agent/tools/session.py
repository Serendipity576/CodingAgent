"""Read-only tools for data owned by the active local conversation."""

from __future__ import annotations

from collections.abc import Mapping
import json

from agent.conversation_memory import ARTIFACT_READ_CHARS, ContextStateError
from agent.tools.base import ToolContext, ToolError, ToolResult


class ReadSessionArtifactTool:
    """Read a bounded range from an oversized result preserved in local conversation history."""

    name = "read_session_artifact"
    description = (
        "Read a bounded range from a large tool output archived in this conversation. "
        "Use an artifact_id explicitly referenced in prior tool output."
    )
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "artifact_id": {
                "type": "string",
                "description": "Artifact id provided by an earlier truncated tool output.",
            },
            "offset": {
                "type": "integer",
                "description": "Zero-based character offset. Defaults to 0.",
                "minimum": 0,
            },
            "max_chars": {
                "type": "integer",
                "description": f"Maximum returned characters, from 1 to {ARTIFACT_READ_CHARS}. Defaults to {ARTIFACT_READ_CHARS}.",
                "minimum": 1,
                "maximum": ARTIFACT_READ_CHARS,
            },
        },
        "required": ["artifact_id"],
        "additionalProperties": False,
    }

    def execute(
        self, arguments: Mapping[str, object], context: ToolContext
    ) -> ToolResult:
        """Read only from the active conversation; this tool never accesses the workspace."""

        artifact_id = arguments.get("artifact_id")
        offset = arguments.get("offset", 0)
        max_chars = arguments.get("max_chars", ARTIFACT_READ_CHARS)
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ToolError("artifact_id must be a non-empty string")
        if type(offset) is not int or offset < 0:
            raise ToolError("offset must be a non-negative integer")
        if type(max_chars) is not int or not 0 < max_chars <= ARTIFACT_READ_CHARS:
            raise ToolError(f"max_chars must be an integer from 1 to {ARTIFACT_READ_CHARS}")
        if context.artifact_reader is None:
            raise ToolError("no archived tool output is available in this conversation")
        try:
            content, metadata = context.artifact_reader.read_session_artifact(
                artifact_id.strip(), offset, max_chars
            )
        except ContextStateError as error:
            raise ToolError(str(error)) from error
        return ToolResult.succeeded(
            json.dumps({"output": content, "artifact": dict(metadata)}, ensure_ascii=False),
            metadata=dict(metadata),
        )
