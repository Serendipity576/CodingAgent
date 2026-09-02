"""Local command tool for the P1 development loop."""

from __future__ import annotations

from collections.abc import Mapping
import os
import signal
import subprocess
from time import monotonic

from agent.sandbox import BubblewrapSandbox, CommandSandbox, SandboxUnavailableError
from agent.tools.base import ToolContext, ToolError, ToolResult, truncate_text


class RunCommandTool:
    """Run an argument-vector command in the configured workspace."""

    name = "run_command"
    description = "Run a local command in the workspace and return its exit status and output."
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "array",
                "description": "Executable and arguments as a JSON string array; shell syntax is not supported.",
                "items": {"type": "string"},
                "minItems": 1,
            }
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def __init__(self, sandbox: CommandSandbox | None = None) -> None:
        """Use Bubblewrap by default; tests may inject a sandbox test double."""

        self._sandbox = sandbox or BubblewrapSandbox()

    def execute(
        self, arguments: Mapping[str, object], context: ToolContext
    ) -> ToolResult:
        command = _command_argument(arguments)
        try:
            invocation = self._sandbox.prepare(command, context.workspace)
        except SandboxUnavailableError as error:
            return ToolResult.failed(
                f"default command sandbox unavailable: {error}",
                metadata={
                    "execution_scope": "sandbox",
                    "sandbox": "bubblewrap",
                    "sandbox_available": False,
                },
            )
        process_options: dict[str, object] = {
            "cwd": context.workspace,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "encoding": "utf-8",
            "errors": "replace",
            "text": True,
        }
        if os.name == "posix":
            # A separate process group lets cancellation stop child processes
            # started by an otherwise allowed test or build command.
            process_options["start_new_session"] = True
        try:
            process = subprocess.Popen(invocation.command, **process_options)
        except OSError as error:
            return ToolResult.failed(
                f"could not start sandboxed command: {error}", metadata=invocation.metadata
            )

        started_at = monotonic()
        while True:
            if context.cancelled is not None and context.cancelled.is_set():
                stdout, stderr = _stop_process(process)
                return ToolResult.failed(
                    "command cancelled",
                    _command_output(
                        stdout=stdout,
                        stderr=stderr,
                        status="cancelled by user",
                        limit=context.limits.max_output_chars,
                    ),
                    metadata={**invocation.metadata, "cancelled": True},
                )
            elapsed = monotonic() - started_at
            remaining = context.limits.command_timeout_seconds - elapsed
            if remaining <= 0:
                stdout, stderr = _stop_process(process)
                return _timeout_result(stdout, stderr, context, invocation.metadata)
            try:
                stdout, stderr = process.communicate(timeout=min(0.1, remaining))
                break
            except subprocess.TimeoutExpired:
                continue

        output = _command_output(
            stdout=stdout,
            stderr=stderr,
            status=f"exit code: {process.returncode}",
            limit=context.limits.max_output_chars,
        )
        if process.returncode != 0:
            return ToolResult.failed(
                f"command exited with code {process.returncode}",
                output,
                metadata={**invocation.metadata, "exit_code": process.returncode},
            )
        return ToolResult.succeeded(
            output, metadata={**invocation.metadata, "exit_code": process.returncode}
        )


def _timeout_result(
    stdout: str,
    stderr: str,
    context: ToolContext,
    metadata: Mapping[str, object],
) -> ToolResult:
    """Return a normal observation for a command stopped at its time limit."""

    output = _command_output(
        stdout=stdout,
        stderr=stderr,
        status=f"timed out after {context.limits.command_timeout_seconds} seconds",
        limit=context.limits.max_output_chars,
    )
    return ToolResult.failed(
        "command timed out", output, metadata={**metadata, "timed_out": True}
    )


def _stop_process(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Terminate a command and its children, then collect final captured output."""

    if process.poll() is None:
        _signal_process_group(process, signal.SIGTERM)
    try:
        return process.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        _signal_process_group(process, signal.SIGKILL)
        return process.communicate()


def _signal_process_group(process: subprocess.Popen[str], signal_number: int) -> None:
    """Stop an active command without failing when it exits between checks."""

    try:
        if os.name == "posix":
            os.killpg(process.pid, signal_number)
        elif signal_number == signal.SIGKILL:
            process.kill()
        else:
            process.terminate()
    except ProcessLookupError:
        return


def _command_argument(arguments: Mapping[str, object]) -> list[str]:
    command = arguments.get("command")
    if not isinstance(command, list) or not command:
        raise ToolError("command must be a non-empty JSON string array")
    if any(not isinstance(part, str) or not part for part in command):
        raise ToolError("every command element must be a non-empty string")
    return command


def _command_output(*, stdout: str, stderr: str, status: str, limit: int) -> str:
    """Label streams so the model can distinguish test failures from output."""

    sections = [status]
    if stdout:
        sections.extend(("stdout:", stdout))
    if stderr:
        sections.extend(("stderr:", stderr))
    return truncate_text("\n".join(sections), limit)
