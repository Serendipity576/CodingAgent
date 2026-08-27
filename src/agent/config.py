"""Configuration loading for the command-line application.

This module deliberately contains no LLM client or tool execution logic. It only
normalizes startup configuration and keeps credentials out of user-facing output.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path


DEFAULT_MODEL = "gpt-5"
DEFAULT_MAX_STEPS = 20
DEFAULT_COMMAND_TIMEOUT_SECONDS = 60
DEFAULT_MAX_OUTPUT_CHARS = 20_000
DEFAULT_MAX_TASK_SECONDS = 900


class ConfigurationError(ValueError):
    """Raised when startup configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class RuntimeLimits:
    """Resource limits that later runtime and tool modules must enforce."""

    max_steps: int = DEFAULT_MAX_STEPS
    command_timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS
    max_task_seconds: int = DEFAULT_MAX_TASK_SECONDS

    def __post_init__(self) -> None:
        for field_name, value in (
            ("max_steps", self.max_steps),
            ("command_timeout_seconds", self.command_timeout_seconds),
            ("max_output_chars", self.max_output_chars),
            ("max_task_seconds", self.max_task_seconds),
        ):
            if value <= 0:
                raise ConfigurationError(f"{field_name} must be greater than zero")


@dataclass(frozen=True, slots=True)
class Settings:
    """Normalized application settings.

    ``api_key`` is retained for the future LLM client but is never included in
    ``public_dict`` or printed by the CLI.
    """

    workspace: Path
    model: str
    api_key: str | None
    base_url: str | None
    limits: RuntimeLimits

    def public_dict(self) -> dict[str, object]:
        """Return settings safe to display in terminal output and audit events."""

        # Report only whether a key exists; command output must never expose it.
        return {
            "workspace": str(self.workspace),
            "model": self.model,
            "base_url": self.base_url,
            "api_key_configured": self.api_key is not None,
            "limits": {
                "max_steps": self.limits.max_steps,
                "command_timeout_seconds": self.limits.command_timeout_seconds,
                "max_output_chars": self.limits.max_output_chars,
                "max_task_seconds": self.limits.max_task_seconds,
            },
        }


def load_settings(
    *,
    workspace: str | Path | None = None,
    model: str | None = None,
    max_steps: int | str | None = None,
    command_timeout_seconds: int | str | None = None,
    max_output_chars: int | str | None = None,
    max_task_seconds: int | str | None = None,
    environment: Mapping[str, str] | None = None,
) -> Settings:
    """Load settings from explicit arguments, environment variables, and defaults.

    Explicit arguments take precedence over environment variables. This function
    does not contact any external service and does not require an API key yet.
    """

    # Explicit CLI values come first so a single run is reproducible and does
    # not depend on unrelated values left in the caller's environment.
    env = os.environ if environment is None else environment
    resolved_workspace = _load_workspace(
        _first_value(workspace, env.get("CODING_AGENT_WORKSPACE"), Path.cwd())
    )
    selected_model = _load_text(
        "model", _first_value(model, env.get("MODEL_NAME"), DEFAULT_MODEL)
    )
    base_url = _optional_text(env.get("OPENAI_BASE_URL"))
    api_key = _optional_text(env.get("OPENAI_API_KEY"))

    limits = RuntimeLimits(
        max_steps=_positive_int(
            "max_steps",
            _first_value(max_steps, env.get("CODING_AGENT_MAX_STEPS")),
            DEFAULT_MAX_STEPS,
        ),
        command_timeout_seconds=_positive_int(
            "command_timeout_seconds",
            _first_value(
                command_timeout_seconds,
                env.get("CODING_AGENT_COMMAND_TIMEOUT_SECONDS"),
            ),
            DEFAULT_COMMAND_TIMEOUT_SECONDS,
        ),
        max_output_chars=_positive_int(
            "max_output_chars",
            _first_value(max_output_chars, env.get("CODING_AGENT_MAX_OUTPUT_CHARS")),
            DEFAULT_MAX_OUTPUT_CHARS,
        ),
        max_task_seconds=_positive_int(
            "max_task_seconds",
            _first_value(max_task_seconds, env.get("CODING_AGENT_MAX_TASK_SECONDS")),
            DEFAULT_MAX_TASK_SECONDS,
        ),
    )
    return Settings(
        workspace=resolved_workspace,
        model=selected_model,
        api_key=api_key,
        base_url=base_url,
        limits=limits,
    )


def _first_value(*values: object) -> object | None:
    """Return the first supplied value, preserving caller-defined precedence."""

    for value in values:
        if value is not None:
            return value
    return None


def _load_workspace(value: object) -> Path:
    text = _load_text("workspace", value)
    # Store one absolute root early. P2 will still validate every tool path
    # against this root before any file operation is authorized.
    workspace = Path(text).expanduser().resolve()
    if not workspace.exists():
        raise ConfigurationError(f"workspace does not exist: {workspace}")
    if not workspace.is_dir():
        raise ConfigurationError(f"workspace is not a directory: {workspace}")
    return workspace


def _load_text(field_name: str, value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ConfigurationError(f"{field_name} must not be empty")
    return text


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _positive_int(field_name: str, value: object | None, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(str(value))
    except ValueError as error:
        raise ConfigurationError(f"{field_name} must be an integer") from error
    if parsed <= 0:
        raise ConfigurationError(f"{field_name} must be greater than zero")
    return parsed
