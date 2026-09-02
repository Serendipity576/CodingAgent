"""Local command tool for the P1 development loop."""

from __future__ import annotations

from collections.abc import Mapping
import os
import signal
import subprocess
from threading import Thread
from time import monotonic
from typing import TextIO

from agent.sandbox import BubblewrapSandbox, CommandSandbox, SandboxUnavailableError
from agent.tools.base import ToolContext, ToolError, ToolResult


_STREAM_READ_CHARS = 4_096


class _BoundedStream:
    """Retain only a prefix while counting the complete stream length."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._parts: list[str] = []
        self._stored_chars = 0
        self.total_chars = 0

    def append(self, text: str) -> None:
        """Count all text but keep no more than the configured visible prefix."""

        self.total_chars += len(text)
        remaining = self._limit - self._stored_chars
        if remaining > 0:
            retained = text[:remaining]
            self._parts.append(retained)
            self._stored_chars += len(retained)

    @property
    def prefix(self) -> str:
        """Return the bounded prefix collected by the reader thread."""

        return "".join(self._parts)


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

        stdout, stderr, readers = _start_output_readers(process, context.limits.max_output_chars)
        started_at = monotonic()
        while True:
            if context.cancelled is not None and context.cancelled.is_set():
                _stop_process(process)
                _finish_output_readers(process, readers)
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
                _stop_process(process)
                _finish_output_readers(process, readers)
                return _timeout_result(stdout, stderr, context, invocation.metadata)
            try:
                process.wait(timeout=min(0.1, remaining))
                break
            except subprocess.TimeoutExpired:
                continue

        _finish_output_readers(process, readers)
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
    stdout: _BoundedStream,
    stderr: _BoundedStream,
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


def _start_output_readers(
    process: subprocess.Popen[str], limit: int
) -> tuple[_BoundedStream, _BoundedStream, tuple[Thread, Thread]]:
    """Drain both pipes concurrently so retained output never grows unbounded."""

    stdout = _BoundedStream(limit)
    stderr = _BoundedStream(limit)
    if process.stdout is None or process.stderr is None:  # Defensive Popen contract check.
        raise RuntimeError("command output pipes were not configured")
    readers = (
        Thread(target=_drain_stream, args=(process.stdout, stdout), daemon=True),
        Thread(target=_drain_stream, args=(process.stderr, stderr), daemon=True),
    )
    for reader in readers:
        reader.start()
    return stdout, stderr, readers


def _drain_stream(stream: TextIO, captured: _BoundedStream) -> None:
    """Consume a process pipe without retaining text beyond its output budget."""

    while chunk := stream.read(_STREAM_READ_CHARS):
        captured.append(chunk)


def _finish_output_readers(
    process: subprocess.Popen[str], readers: tuple[Thread, Thread]
) -> None:
    """Join readers and close parent pipe handles once the child has exited."""

    for reader in readers:
        reader.join()
    for stream in (process.stdout, process.stderr):
        if stream is not None:
            stream.close()


def _stop_process(process: subprocess.Popen[str]) -> None:
    """Terminate a command and its children before the reader threads are joined."""

    if process.poll() is None:
        _signal_process_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        _signal_process_group(process, signal.SIGKILL)
        process.wait()


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


def _command_output(
    *, stdout: _BoundedStream, stderr: _BoundedStream, status: str, limit: int
) -> str:
    """Render bounded stream prefixes with an accurate combined-size truncation marker."""

    sections = [status]
    total_chars = len(status)
    if stdout.total_chars:
        sections.extend(("stdout:", stdout.prefix))
        total_chars += len("\nstdout:\n") + stdout.total_chars
    if stderr.total_chars:
        sections.extend(("stderr:", stderr.prefix))
        total_chars += len("\nstderr:\n") + stderr.total_chars
    visible = "\n".join(sections)
    return _truncate_captured_text(visible, total_chars, limit)


def _truncate_captured_text(visible: str, total_chars: int, limit: int) -> str:
    """Keep the response within ``limit`` even when only stream prefixes are retained."""

    if total_chars <= limit:
        return visible
    marker = f"\n... [truncated {total_chars - limit} characters]"
    if limit <= len(marker):
        return visible[:limit]
    return f"{visible[: limit - len(marker)]}{marker}"
