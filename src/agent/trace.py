"""Private, structured runtime traces for local conversation diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import re
from threading import RLock
from time import time
from typing import Any


TRACE_TEXT_LIMIT = 12_000
TRACE_COLLECTION_LIMIT = 100
_REDACTED = "[已脱敏]"
_SENSITIVE_KEY_NAMES = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "id_token",
    "client_secret",
}
_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:api[_-]?key|authorization|password|secret|token)\b\s*[:=]\s*[^\s,;]+"),
    re.compile(r"(?i)\bbearer\s+[a-z0-9._~+\-/=]{8,}"),
    re.compile(r"\bsk-[a-zA-Z0-9_-]{16,}\b"),
)


class TurnTraceRecorder:
    """Record one turn as a durable model/tool trace without using SSE for bodies."""

    def __init__(
        self,
        *,
        conversation_id: str,
        turn_id: int,
        on_change: Callable[[dict[str, object]], None],
    ) -> None:
        self._on_change = on_change
        self._lock = RLock()
        self._next_item = 1
        self._active_model_id: str | None = None
        self._trace: dict[str, object] = {
            "version": 1,
            "conversation_id": conversation_id,
            "turn_id": turn_id,
            "status": "running",
            "started_at": time(),
            "finished_at": None,
            "duration_ms": None,
            "summary": None,
            "items": [],
        }
        self._notify()

    def model_started(
        self,
        *,
        step: int,
        request: Mapping[str, object],
        title: str | None = None,
    ) -> str:
        """Start one model span with a generic fallback request representation."""

        item_id = self._append_item(
            kind="model",
            title=title or f"模型调用 #{step}",
            summary="正在请求模型",
            attributes={"step": step},
        )
        with self._lock:
            self._active_model_id = item_id
            self._item(item_id)["request"] = _sanitize(request)
        self._notify()
        return item_id

    def record_llm_payload(self, event: str, details: Mapping[str, object]) -> None:
        """Replace fallback model data with the adapter's exact normalized exchange."""

        with self._lock:
            item_id = self._active_model_id
            if item_id is None:
                return
            item = self._item(item_id)
            if event in {"request", "context_summary_request", "context_retry"}:
                request = details.get("request")
                if isinstance(request, Mapping):
                    item["request"] = _sanitize(request)
                    item["attributes"] = {
                        **_mapping(item.get("attributes")),
                        **_model_request_attributes(request),
                    }
                context = details.get("context")
                if isinstance(context, Mapping):
                    item["attributes"] = {
                        **_mapping(item.get("attributes")),
                        "context": _sanitize(context),
                    }
            elif event in {"response", "context_summary_response"}:
                response = details.get("response")
                if isinstance(response, Mapping):
                    item["response"] = _sanitize(response)
                context = details.get("context")
                if isinstance(context, Mapping):
                    item["attributes"] = {
                        **_mapping(item.get("attributes")),
                        "context": _sanitize(context),
                    }
            elif event in {"error", "context_summary_error"}:
                item["response"] = _sanitize({"error": details.get("error")})
        self._notify()

    def model_finished(
        self,
        item_id: str,
        *,
        response: Mapping[str, object],
        duration_ms: int,
    ) -> None:
        """Complete one model span after a normalized response reaches the runtime."""

        with self._lock:
            item = self._item(item_id)
            item["status"] = "completed"
            item["finished_at"] = time()
            item["duration_ms"] = duration_ms
            item["summary"] = _model_summary(response)
            if "response" not in item:
                item["response"] = _sanitize(response)
            item["attributes"] = {
                **_mapping(item.get("attributes")),
                **_model_response_attributes(response),
            }
            if self._active_model_id == item_id:
                self._active_model_id = None
        self._notify()

    def model_failed(self, item_id: str, *, error: str, duration_ms: int) -> None:
        """Close the active model span when the provider request fails."""

        with self._lock:
            item = self._item(item_id)
            item["status"] = "failed"
            item["finished_at"] = time()
            item["duration_ms"] = duration_ms
            item["summary"] = "模型请求失败"
            item["attributes"] = {**_mapping(item.get("attributes")), "error": _sanitize(error)}
            if self._active_model_id == item_id:
                self._active_model_id = None
        self._notify()

    def tool_started(self, *, step: int, parent_id: str | None, call: object) -> str:
        """Create a child tool span before policy evaluation and execution begin."""

        name = _string_field(call, "name") or "未知工具"
        arguments = _mapping_field(call, "arguments")
        item_id = self._append_item(
            kind="tool",
            parent_id=parent_id,
            title=f"工具：{name}",
            summary="正在执行",
            attributes={"step": step, "tool": name},
        )
        with self._lock:
            self._item(item_id)["request"] = _sanitize({"arguments": arguments})
        self._notify()
        return item_id

    def tool_finished(
        self,
        item_id: str,
        *,
        result: object,
        duration_ms: int,
    ) -> None:
        """Store a policy-aware tool result that remains hidden until explicitly opened."""

        success = bool(getattr(result, "success", False))
        decision = getattr(result, "decision", None)
        risk = getattr(result, "risk", None)
        error = getattr(result, "error", None)
        output = getattr(result, "output", "")
        metadata = getattr(result, "metadata", {})
        attributes: dict[str, object] = {
            "success": success,
            "decision": decision,
            "risk": risk,
            "policy": getattr(result, "policy", None),
        }
        if isinstance(metadata, Mapping) and "normalization" in metadata:
            attributes["normalization"] = metadata["normalization"]
        execution = _execution_attributes(metadata)
        if execution:
            attributes["execution"] = execution
        if isinstance(error, str) and error:
            attributes["error"] = error
        invalid_arguments = _is_invalid_argument_failure(
            decision=decision,
            policy=getattr(result, "policy", None),
            error=error,
        )
        with self._lock:
            item = self._item(item_id)
            item["status"] = "completed" if success else "skipped" if invalid_arguments else "failed"
            item["finished_at"] = time()
            item["duration_ms"] = duration_ms
            item["summary"] = (
                "执行成功"
                if success
                else "参数无效，未执行"
                if invalid_arguments
                else "执行失败"
            )
            item["attributes"] = _sanitize(attributes)
            item["response"] = _sanitize({"output": output, "error": error})
        self._notify()

    def tool_skipped(self, item_id: str, *, reason: str) -> None:
        """Close a model-requested tool span that a runtime limit prevented."""

        with self._lock:
            item = self._item(item_id)
            item["status"] = "skipped"
            item["finished_at"] = time()
            item["duration_ms"] = 0
            item["summary"] = "未执行"
            item["attributes"] = {"reason": _sanitize(reason)}
            item["response"] = {"error": _sanitize(reason)}
        self._notify()

    def finish(self, *, status: str, message: str, steps: int) -> None:
        """Seal a terminal turn while keeping all child spans available for inspection."""

        with self._lock:
            started_at = self._trace["started_at"]
            now = time()
            self._trace["status"] = status
            self._trace["finished_at"] = now
            self._trace["duration_ms"] = int((now - float(started_at)) * 1000)
            self._trace["summary"] = {"message": _sanitize(message), "steps": steps}
        self._notify()

    def snapshot(self) -> dict[str, object]:
        """Return a private JSON-safe copy for durable local storage."""

        with self._lock:
            return deepcopy(self._trace)

    def _append_item(
        self,
        *,
        kind: str,
        title: str,
        summary: str,
        attributes: Mapping[str, object],
        parent_id: str | None = None,
    ) -> str:
        with self._lock:
            item_id = f"{kind}-{self._next_item}"
            self._next_item += 1
            items = self._items()
            items.append(
                {
                    "item_id": item_id,
                    "parent_id": parent_id,
                    "kind": kind,
                    "status": "running",
                    "title": title,
                    "summary": summary,
                    "started_at": time(),
                    "finished_at": None,
                    "duration_ms": None,
                    "attributes": _sanitize(attributes),
                }
            )
        self._notify()
        return item_id

    def _items(self) -> list[dict[str, object]]:
        items = self._trace["items"]
        if not isinstance(items, list):
            raise RuntimeError("trace items must be a list")
        return items

    def _item(self, item_id: str) -> dict[str, object]:
        for item in self._items():
            if item.get("item_id") == item_id:
                return item
        raise RuntimeError(f"unknown trace item: {item_id}")

    def _notify(self) -> None:
        try:
            self._on_change(self.snapshot())
        except Exception:
            # Observability is strictly best-effort and cannot weaken execution.
            return


def public_turn_trace(trace: Mapping[str, object]) -> dict[str, object]:
    """Return a list-safe trace without model/tool bodies for the ordinary API."""

    safe = _public_trace(trace)
    safe["items"] = [
        {key: value for key, value in item.items() if key not in {"request", "response"}}
        for item in _mapping_list(safe.get("items"))
    ]
    return safe


def trace_item_detail(trace: Mapping[str, object], item_id: str) -> dict[str, object] | None:
    """Return one explicitly requested local detail record with sanitized bodies."""

    for item in _mapping_list(trace.get("items")):
        if item.get("item_id") == item_id:
            return _sanitize(item)
    return None


def _public_trace(trace: Mapping[str, object]) -> dict[str, object]:
    return _sanitize(trace)


def _model_request_attributes(request: Mapping[str, object]) -> dict[str, object]:
    input_items = request.get("input")
    tools = request.get("tools")
    return {
        "model": request.get("model"),
        "input_items": len(input_items) if isinstance(input_items, Sequence) and not isinstance(input_items, str) else 0,
        "tool_definitions": len(tools) if isinstance(tools, Sequence) and not isinstance(tools, str) else 0,
    }


def _model_response_attributes(response: Mapping[str, object]) -> dict[str, object]:
    calls = response.get("tool_calls")
    usage = response.get("usage")
    return {
        "response_id": response.get("response_id"),
        "tool_calls": len(calls) if isinstance(calls, Sequence) and not isinstance(calls, str) else 0,
        "usage": _sanitize(usage) if isinstance(usage, Mapping) else None,
    }


def _model_summary(response: Mapping[str, object]) -> str:
    calls = response.get("tool_calls")
    if isinstance(calls, Sequence) and not isinstance(calls, str) and calls:
        return f"返回 {len(calls)} 个工具调用"
    return "返回最终回复"


def _string_field(value: object, name: str) -> str | None:
    field = getattr(value, name, None)
    return field if isinstance(field, str) else None


def _mapping_field(value: object, name: str) -> Mapping[str, object]:
    field = getattr(value, name, None)
    return field if isinstance(field, Mapping) else {}


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_list(value: object) -> list[dict[str, object]]:
    return [dict(item) for item in value] if isinstance(value, list) and all(isinstance(item, Mapping) for item in value) else []


def _sanitize(value: object, *, key: str | None = None, depth: int = 0) -> Any:
    """Make trace content JSON-safe, bounded, and free of common credential values."""

    if key is not None and _is_sensitive_key(key):
        return _REDACTED
    if depth >= 12:
        return "[已省略：嵌套过深]"
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize(item_value, key=str(item_key), depth=depth + 1)
            for index, (item_key, item_value) in enumerate(value.items())
            if index < TRACE_COLLECTION_LIMIT
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [
            _sanitize(item, depth=depth + 1)
            for item in list(value)[:TRACE_COLLECTION_LIMIT]
        ]
    return _sanitize_text(str(value))


def _sanitize_text(value: str) -> str:
    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(_REDACTED, redacted)
    if len(redacted) <= TRACE_TEXT_LIMIT:
        return redacted
    return f"{redacted[:TRACE_TEXT_LIMIT]}\n… [已截断，原始长度 {len(redacted)} 字符]"


def _is_sensitive_key(key: str) -> bool:
    """Redact credential fields without mistaking token accounting for a secret."""

    normalized = key.casefold().replace("-", "_")
    return (
        normalized in _SENSITIVE_KEY_NAMES
        or normalized.endswith(("_api_key", "_password", "_secret"))
    )


def _execution_attributes(metadata: object) -> dict[str, object]:
    """Expose safe command-isolation facts without recording arbitrary metadata."""

    if not isinstance(metadata, Mapping):
        return {}
    allowed_names = {
        "execution_scope",
        "sandbox",
        "sandbox_available",
        "network",
        "workspace_access",
        "masked_path_count",
        "cancelled",
        "timed_out",
        "exit_code",
    }
    return {name: metadata[name] for name in allowed_names if name in metadata}


def _is_invalid_argument_failure(*, decision: object, policy: object, error: object) -> bool:
    """Distinguish malformed paths from an actual blocked workspace access attempt."""

    if decision != "deny" or policy not in {"invalid_arguments", "workspace_boundary"}:
        return False
    return isinstance(error, str) and "must be a non-empty string" in error
