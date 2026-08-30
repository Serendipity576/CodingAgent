"""Local command tool for the P1 development loop."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import subprocess

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

    def execute(
        self, arguments: Mapping[str, object], context: ToolContext
    ) -> ToolResult:
        command = _command_argument(arguments)
        try:
            completed = subprocess.run(
                command,
                cwd=context.workspace,
                capture_output=True,
                check=False,
                encoding="utf-8",
                errors="replace",
                text=True,
                timeout=context.limits.command_timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            output = _command_output(
                stdout=_as_text(error.stdout),
                stderr=_as_text(error.stderr),
                status=f"timed out after {context.limits.command_timeout_seconds} seconds",
                limit=context.limits.max_output_chars,
            )
            return ToolResult.failed(
                "command timed out",
                output,
                metadata={"timed_out": True},
            )
        except OSError as error:
            return ToolResult.failed(f"could not start command: {error}")

        output = _command_output(
            stdout=completed.stdout,
            stderr=completed.stderr,
            status=f"exit code: {completed.returncode}",
            limit=context.limits.max_output_chars,
        )
        if completed.returncode != 0:
            return ToolResult.failed(
                f"command exited with code {completed.returncode}",
                output,
                metadata={"exit_code": completed.returncode},
            )
        return ToolResult.succeeded(output, metadata={"exit_code": completed.returncode})


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


def _as_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""
