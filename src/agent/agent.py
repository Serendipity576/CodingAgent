"""The bounded multi-turn runtime for a single coding task."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from time import monotonic

from agent.audit import AuditLogger
from agent.change_tracker import GitStatusSnapshot
from agent.config import Settings
from agent.context import TaskContext
from agent.llm.client import LLMClient, LLMRequestError
from agent.llm.models import ToolCall, ToolOutput
from agent.summary import TaskSummary, build_task_summary
from agent.trace import TurnTraceRecorder
from agent.tools.base import ToolContext, ToolResult
from agent.tools.registry import ToolRegistry


class TaskStatus(str, Enum):
    """Terminal states that tell the CLI why a run stopped."""

    COMPLETED = "completed"
    MAX_STEPS_REACHED = "max_steps_reached"
    TASK_TIMEOUT = "task_timeout"
    REPEATED_TOOL_FAILURE = "repeated_tool_failure"
    LLM_ERROR = "llm_error"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ExecutedToolCall:
    """One tool result retained for the final task summary and tests."""

    call: ToolCall
    result: ToolResult


@dataclass(frozen=True, slots=True)
class TaskResult:
    """The observable outcome of one full agent task."""

    status: TaskStatus
    message: str
    steps: int
    tool_calls: tuple[ExecutedToolCall, ...]
    task_id: str | None = None
    audit_log: Path | None = None
    summary: TaskSummary | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is TaskStatus.COMPLETED


class CodingAgent:
    """Coordinate model turns and tool results while enforcing runtime limits."""

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        tools: ToolRegistry,
        *,
        audit_logger: AuditLogger | None = None,
        git_baseline: GitStatusSnapshot | None = None,
        event_callback: Callable[[str, Mapping[str, object]], None] | None = None,
        cancellation: object | None = None,
        trace: TurnTraceRecorder | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._tools = tools
        self._audit_logger = audit_logger
        self._git_baseline = git_baseline
        self._event_callback = event_callback
        self._cancellation = cancellation
        self._trace = trace

    def run(self, task: str) -> TaskResult:
        """Run one task until the model finishes or a deterministic limit stops it."""

        task_context = TaskContext(task=task)
        tool_context = ToolContext(
            workspace=self._settings.workspace,
            limits=self._settings.limits,
            cancelled=self._cancellation if _is_cancellation_check(self._cancellation) else None,
        )
        audit_logger = self._audit_logger or AuditLogger.create(self._settings.workspace)
        git_baseline = self._git_baseline or GitStatusSnapshot.capture(
            self._settings.workspace
        )
        if audit_logger is not None:
            audit_logger.task_started(task_context.task, git_baseline)
        self._emit("turn_started", {"task": task_context.task})

        started_at = monotonic()
        next_task: str | None = task_context.task
        pending_outputs: tuple[ToolOutput, ...] = ()
        executed: list[ExecutedToolCall] = []
        steps = 0
        last_failure_signature: str | None = None
        consecutive_failures = 0

        def finish(status: TaskStatus, message: str) -> TaskResult:
            """Attach the run's audit facts whenever a terminal state is reached."""

            summary = build_task_summary(
                [(item.call, item.result) for item in executed], git_baseline
            )
            result = TaskResult(
                status=status,
                message=message,
                steps=steps,
                tool_calls=tuple(executed),
                task_id=audit_logger.task_id if audit_logger else None,
                audit_log=audit_logger.path if audit_logger else None,
                summary=summary,
            )
            if audit_logger is not None:
                audit_logger.task_finished(
                    status=status.value,
                    message=message,
                    steps=steps,
                    summary=summary,
                )
            if self._trace is not None:
                self._trace.finish(status=status.value, message=message, steps=steps)
            self._emit(
                "agent_finished",
                {"status": status.value, "message": message, "steps": steps},
            )
            return result

        while True:
            if self._task_timed_out(started_at):
                return finish(
                    TaskStatus.TASK_TIMEOUT,
                    "task time limit reached before the next model turn",
                )
            if self._is_cancelled():
                return finish(TaskStatus.CANCELLED, "task cancelled before the next model turn")

            model_started_at = monotonic()
            model_trace_id = (
                self._trace.model_started(
                    step=steps + 1,
                    request=_model_request_trace(
                        instructions=task_context.instructions,
                        task=next_task,
                        tools=self._tools.schemas(),
                        tool_outputs=pending_outputs,
                    ),
                )
                if self._trace is not None
                else None
            )
            try:
                self._emit("llm_request_started", {"step": steps + 1})
                response = self._llm.respond(
                    instructions=task_context.instructions,
                    task=next_task,
                    tools=self._tools.schemas(),
                    tool_outputs=pending_outputs,
                )
            except LLMRequestError as error:
                if self._trace is not None and model_trace_id is not None:
                    self._trace.model_failed(
                        model_trace_id,
                        error=str(error),
                        duration_ms=int((monotonic() - model_started_at) * 1000),
                    )
                return finish(TaskStatus.LLM_ERROR, str(error))

            if self._trace is not None and model_trace_id is not None:
                self._trace.model_finished(
                    model_trace_id,
                    response=_model_response_trace(response),
                    duration_ms=int((monotonic() - model_started_at) * 1000),
                )

            next_task = None
            if not response.tool_calls:
                self._emit("assistant_message", {"text": response.text or ""})
                return finish(
                    TaskStatus.COMPLETED,
                    response.text or "agent completed without a final message",
                )

            outputs: list[ToolOutput] = []
            for call_index, call in enumerate(response.tool_calls):
                self._emit(
                    "tool_requested",
                    {
                        "call_id": call.call_id,
                        "tool": call.name,
                        "arguments": _event_argument_summary(call),
                    },
                )
                if steps >= self._settings.limits.max_steps:
                    self._record_unexecuted_tool_calls(
                        response.tool_calls[call_index:],
                        "not executed because the maximum tool-call steps were reached",
                        parent_id=model_trace_id,
                    )
                    return finish(
                        TaskStatus.MAX_STEPS_REACHED,
                        "maximum tool-call steps reached",
                    )
                if self._task_timed_out(started_at):
                    self._record_unexecuted_tool_calls(
                        response.tool_calls[call_index:],
                        "not executed because the task time limit was reached",
                        parent_id=model_trace_id,
                    )
                    return finish(
                        TaskStatus.TASK_TIMEOUT,
                        "task time limit reached before the next tool call",
                    )
                if self._is_cancelled():
                    self._record_unexecuted_tool_calls(
                        response.tool_calls[call_index:],
                        "not executed because the task was cancelled",
                        parent_id=model_trace_id,
                    )
                    return finish(TaskStatus.CANCELLED, "task cancelled before tool execution")

                steps += 1
                tool_trace_id = (
                    self._trace.tool_started(
                        step=steps,
                        parent_id=model_trace_id,
                        call=call,
                    )
                    if self._trace is not None
                    else None
                )
                tool_started_at = monotonic()
                result = self._tools.execute(call, tool_context)
                duration_ms = int((monotonic() - tool_started_at) * 1000)
                if self._trace is not None and tool_trace_id is not None:
                    self._trace.tool_finished(tool_trace_id, result=result, duration_ms=duration_ms)
                executed.append(ExecutedToolCall(call=call, result=result))
                if audit_logger is not None:
                    audit_logger.tool_executed(
                        step=steps,
                        call=call,
                        result=result,
                        duration_ms=duration_ms,
                    )
                output = ToolOutput(call_id=call.call_id, output=result.as_observation())
                outputs.append(output)
                # ResponsesClient persists each observation immediately. A
                # process restart after a tool finishes must not leave the
                # restored transcript with an unanswered function call.
                _record_tool_outputs(self._llm, (output,))
                self._emit(
                    "tool_finished",
                    {
                        "call_id": call.call_id,
                        "tool": call.name,
                        "success": result.success,
                        "error": result.error,
                        "decision": result.decision,
                        "risk": result.risk,
                        "policy": result.policy,
                        "duration_ms": duration_ms,
                        "output_chars": len(result.output),
                    },
                )

                if result.success:
                    last_failure_signature = None
                    consecutive_failures = 0
                    continue

                failure_signature = _call_signature(call)
                if failure_signature == last_failure_signature:
                    consecutive_failures += 1
                else:
                    last_failure_signature = failure_signature
                    consecutive_failures = 1
                if (
                    consecutive_failures
                    >= self._settings.limits.max_consecutive_tool_failures
                ):
                    return finish(
                        TaskStatus.REPEATED_TOOL_FAILURE,
                        "the same tool call failed repeatedly",
                    )

            pending_outputs = tuple(outputs)

    def _record_unexecuted_tool_calls(
        self,
        calls: Sequence[ToolCall],
        reason: str,
        *,
        parent_id: str | None,
    ) -> None:
        """Close skipped model calls so durable history remains valid for a later turn."""

        outputs = tuple(
            ToolOutput(
                call_id=call.call_id,
                output=json.dumps({"success": False, "error": reason}, ensure_ascii=False),
            )
            for call in calls
        )
        _record_tool_outputs(self._llm, outputs)
        for call, output in zip(calls, outputs):
            if self._trace is not None:
                item_id = self._trace.tool_started(
                    step=0,
                    parent_id=parent_id,
                    call=call,
                )
                self._trace.tool_skipped(item_id, reason=reason)
            self._emit(
                "tool_finished",
                {
                    "call_id": call.call_id,
                    "tool": call.name,
                    "success": False,
                    "error": reason,
                    "decision": "not_executed",
                    "risk": "none",
                    "policy": "runtime_limit",
                    "duration_ms": 0,
                    "output_chars": len(output.output),
                },
            )

    def _task_timed_out(self, started_at: float) -> bool:
        return monotonic() - started_at >= self._settings.limits.max_task_seconds

    def _is_cancelled(self) -> bool:
        """Read a cooperative cancellation event without coupling to threading."""

        return _is_cancellation_check(self._cancellation) and self._cancellation.is_set()

    def _emit(self, event: str, details: Mapping[str, object]) -> None:
        """Publish safe runtime facts without making observers part of execution."""

        if self._event_callback is None:
            return
        try:
            self._event_callback(event, details)
        except Exception:
            # UI and terminal observers must never weaken the Agent loop.
            return


def _call_signature(call: ToolCall) -> str:
    """Create a stable identity for consecutive failed-call detection."""

    return json.dumps(
        {
            "name": call.name,
            "arguments": call.arguments,
            "arguments_error": call.arguments_error,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _is_cancellation_check(value: object | None) -> bool:
    """Accept event-like cancellation objects without importing threading types."""

    return callable(getattr(value, "is_set", None))


def _record_tool_outputs(llm: object, outputs: tuple[ToolOutput, ...]) -> None:
    """Use optional durable-history support without constraining test LLMs."""

    recorder = getattr(llm, "record_tool_outputs", None)
    if callable(recorder):
        recorder(outputs)


def _event_argument_summary(call: ToolCall) -> dict[str, object]:
    """Expose approval-relevant arguments without streaming file or patch bodies."""

    arguments = call.arguments or {}
    if call.name in {"list_files", "read_file"}:
        return {"path": arguments.get("path")}
    if call.name == "apply_patch":
        return {
            "path": arguments.get("path"),
            "expected_text_chars": _text_length(arguments.get("expected_text")),
            "replacement_text_chars": _text_length(arguments.get("replacement_text")),
        }
    if call.name == "run_command":
        command = arguments.get("command")
        return {"command": command if isinstance(command, list) else None}
    return {"argument_keys": sorted(arguments.keys())}


def _text_length(value: object) -> int | None:
    """Report a text length without copying its content into an event stream."""

    return len(value) if isinstance(value, str) else None


def _model_request_trace(
    *,
    instructions: str,
    task: str | None,
    tools: Sequence[Mapping[str, object]],
    tool_outputs: Sequence[ToolOutput],
) -> dict[str, object]:
    """Provide a generic fallback before an adapter records its exact request body."""

    return {
        "instructions": instructions,
        "task": task,
        "tools": list(tools),
        "tool_outputs": [
            {"call_id": output.call_id, "output": output.output}
            for output in tool_outputs
        ],
    }


def _model_response_trace(response: object) -> dict[str, object]:
    """Normalize a response for trace clients that do not expose provider payloads."""

    response_id = getattr(response, "response_id", None)
    text = getattr(response, "text", None)
    tool_calls = getattr(response, "tool_calls", ())
    usage = getattr(response, "usage", None)
    return {
        "response_id": response_id,
        "output_text": text,
        "tool_calls": [
            {
                "call_id": call.call_id,
                "name": call.name,
                "arguments": dict(call.arguments or {}),
                "arguments_error": call.arguments_error,
            }
            for call in tool_calls
        ],
        "usage": (
            {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            }
            if usage is not None
            else None
        ),
    }
