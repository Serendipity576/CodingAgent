"""The bounded multi-turn runtime for a single coding task."""

from __future__ import annotations

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
from agent.tools.base import ToolContext, ToolResult
from agent.tools.registry import ToolRegistry


class TaskStatus(str, Enum):
    """Terminal states that tell the CLI why a run stopped."""

    COMPLETED = "completed"
    MAX_STEPS_REACHED = "max_steps_reached"
    TASK_TIMEOUT = "task_timeout"
    REPEATED_TOOL_FAILURE = "repeated_tool_failure"
    LLM_ERROR = "llm_error"


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
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._tools = tools
        self._audit_logger = audit_logger
        self._git_baseline = git_baseline

    def run(self, task: str) -> TaskResult:
        """Run one task until the model finishes or a deterministic limit stops it."""

        task_context = TaskContext(task=task)
        tool_context = ToolContext(
            workspace=self._settings.workspace,
            limits=self._settings.limits,
        )
        audit_logger = self._audit_logger or AuditLogger.create(self._settings.workspace)
        git_baseline = self._git_baseline or GitStatusSnapshot.capture(
            self._settings.workspace
        )
        if audit_logger is not None:
            audit_logger.task_started(task_context.task, git_baseline)

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
            return result

        while True:
            if self._task_timed_out(started_at):
                return finish(
                    TaskStatus.TASK_TIMEOUT,
                    "task time limit reached before the next model turn",
                )

            try:
                response = self._llm.respond(
                    instructions=task_context.instructions,
                    task=next_task,
                    tools=self._tools.schemas(),
                    tool_outputs=pending_outputs,
                )
            except LLMRequestError as error:
                return finish(TaskStatus.LLM_ERROR, str(error))

            next_task = None
            if not response.tool_calls:
                return finish(
                    TaskStatus.COMPLETED,
                    response.text or "agent completed without a final message",
                )

            outputs: list[ToolOutput] = []
            for call in response.tool_calls:
                if steps >= self._settings.limits.max_steps:
                    return finish(
                        TaskStatus.MAX_STEPS_REACHED,
                        "maximum tool-call steps reached",
                    )
                if self._task_timed_out(started_at):
                    return finish(
                        TaskStatus.TASK_TIMEOUT,
                        "task time limit reached before the next tool call",
                    )

                steps += 1
                tool_started_at = monotonic()
                result = self._tools.execute(call, tool_context)
                duration_ms = int((monotonic() - tool_started_at) * 1000)
                executed.append(ExecutedToolCall(call=call, result=result))
                if audit_logger is not None:
                    audit_logger.tool_executed(
                        step=steps,
                        call=call,
                        result=result,
                        duration_ms=duration_ms,
                    )
                outputs.append(ToolOutput(call_id=call.call_id, output=result.as_observation()))

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

    def _task_timed_out(self, started_at: float) -> bool:
        return monotonic() - started_at >= self._settings.limits.max_task_seconds


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
