"""The bounded multi-turn runtime for a single coding task."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from time import monotonic

from agent.config import Settings
from agent.context import TaskContext
from agent.llm.client import LLMClient, LLMRequestError
from agent.llm.models import ToolCall, ToolOutput
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

    @property
    def succeeded(self) -> bool:
        return self.status is TaskStatus.COMPLETED


class CodingAgent:
    """Coordinate model turns and tool results while enforcing runtime limits."""

    def __init__(self, settings: Settings, llm: LLMClient, tools: ToolRegistry) -> None:
        self._settings = settings
        self._llm = llm
        self._tools = tools

    def run(self, task: str) -> TaskResult:
        """Run one task until the model finishes or a deterministic limit stops it."""

        task_context = TaskContext(task=task)
        tool_context = ToolContext(
            workspace=self._settings.workspace,
            limits=self._settings.limits,
        )
        started_at = monotonic()
        previous_response_id: str | None = None
        next_task: str | None = task_context.task
        pending_outputs: tuple[ToolOutput, ...] = ()
        executed: list[ExecutedToolCall] = []
        steps = 0
        last_failure_signature: str | None = None
        consecutive_failures = 0

        while True:
            if self._task_timed_out(started_at):
                return self._result(
                    TaskStatus.TASK_TIMEOUT,
                    "task time limit reached before the next model turn",
                    steps,
                    executed,
                )

            try:
                response = self._llm.respond(
                    instructions=task_context.instructions,
                    task=next_task,
                    tools=self._tools.schemas(),
                    previous_response_id=previous_response_id,
                    tool_outputs=pending_outputs,
                )
            except LLMRequestError as error:
                return self._result(TaskStatus.LLM_ERROR, str(error), steps, executed)

            previous_response_id = response.response_id
            next_task = None
            if not response.tool_calls:
                return self._result(
                    TaskStatus.COMPLETED,
                    response.text or "agent completed without a final message",
                    steps,
                    executed,
                )

            outputs: list[ToolOutput] = []
            for call in response.tool_calls:
                if steps >= self._settings.limits.max_steps:
                    return self._result(
                        TaskStatus.MAX_STEPS_REACHED,
                        "maximum tool-call steps reached",
                        steps,
                        executed,
                    )
                if self._task_timed_out(started_at):
                    return self._result(
                        TaskStatus.TASK_TIMEOUT,
                        "task time limit reached before the next tool call",
                        steps,
                        executed,
                    )

                steps += 1
                result = self._tools.execute(call, tool_context)
                executed.append(ExecutedToolCall(call=call, result=result))
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
                    return self._result(
                        TaskStatus.REPEATED_TOOL_FAILURE,
                        "the same tool call failed repeatedly",
                        steps,
                        executed,
                    )

            pending_outputs = tuple(outputs)

    def _task_timed_out(self, started_at: float) -> bool:
        return monotonic() - started_at >= self._settings.limits.max_task_seconds

    @staticmethod
    def _result(
        status: TaskStatus,
        message: str,
        steps: int,
        executed: list[ExecutedToolCall],
    ) -> TaskResult:
        return TaskResult(
            status=status,
            message=message,
            steps=steps,
            tool_calls=tuple(executed),
        )


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
