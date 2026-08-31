"""Configuration loading for the command-line application.

This module deliberately contains no LLM client or tool execution logic. It
normalizes startup configuration and keeps credentials out of user-facing output.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path


DEFAULT_MAX_STEPS = 20
DEFAULT_COMMAND_TIMEOUT_SECONDS = 60
DEFAULT_MAX_OUTPUT_CHARS = 20_000
DEFAULT_MAX_TASK_SECONDS = 900
DEFAULT_MAX_CONSECUTIVE_TOOL_FAILURES = 2
DEFAULT_MAX_CONVERSATION_TURNS = 50
DEFAULT_MAX_HISTORY_ITEMS = 300
SUPPORTED_LLM_PROVIDERS = frozenset({"openai", "deepseek", "responses"})
LOCAL_CONNECTION_FIELDS = frozenset(
    {
        "CODING_AGENT_PROVIDER",
        "CODING_AGENT_API_KEY",
        "CODING_AGENT_BASE_URL",
        "CODING_AGENT_MODEL",
        "CODING_AGENT_MAX_OUTPUT_TOKENS",
    }
)


class ConfigurationError(ValueError):
    """Raised when startup configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class RuntimeLimits:
    """Resource limits that later runtime and tool modules must enforce."""

    max_steps: int = DEFAULT_MAX_STEPS
    command_timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS
    max_task_seconds: int = DEFAULT_MAX_TASK_SECONDS
    max_consecutive_tool_failures: int = DEFAULT_MAX_CONSECUTIVE_TOOL_FAILURES
    max_conversation_turns: int = DEFAULT_MAX_CONVERSATION_TURNS
    max_history_items: int = DEFAULT_MAX_HISTORY_ITEMS

    def __post_init__(self) -> None:
        for field_name, value in (
            ("max_steps", self.max_steps),
            ("command_timeout_seconds", self.command_timeout_seconds),
            ("max_output_chars", self.max_output_chars),
            ("max_task_seconds", self.max_task_seconds),
            ("max_consecutive_tool_failures", self.max_consecutive_tool_failures),
            ("max_conversation_turns", self.max_conversation_turns),
            ("max_history_items", self.max_history_items),
        ):
            if value <= 0:
                raise ConfigurationError(f"{field_name} must be greater than zero")


@dataclass(frozen=True, slots=True)
class Settings:
    """Normalized application settings.

    Connection fields are retained for the LLM client but are never included in
    ``public_dict`` or printed by the CLI.
    """

    workspace: Path
    model: str | None
    api_key: str | None
    base_url: str | None
    limits: RuntimeLimits
    provider: str | None = None
    max_output_tokens: int | None = None

    def public_dict(self) -> dict[str, object]:
        """Return settings safe to display in terminal output and audit events."""

        # Report only whether a key exists; command output must never expose it.
        return {
            "workspace": str(self.workspace),
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_configured": self.api_key is not None,
            "max_output_tokens": self.max_output_tokens,
            "limits": {
                "max_steps": self.limits.max_steps,
                "command_timeout_seconds": self.limits.command_timeout_seconds,
                "max_output_chars": self.limits.max_output_chars,
                "max_task_seconds": self.limits.max_task_seconds,
                "max_consecutive_tool_failures": self.limits.max_consecutive_tool_failures,
                "max_conversation_turns": self.limits.max_conversation_turns,
                "max_history_items": self.limits.max_history_items,
            },
        }


def load_settings(
    *,
    workspace: str | Path | None = None,
    model: str | None = None,
    base_url: str | None = None,
    config_file: str | Path | None = None,
    max_output_tokens: int | str | None = None,
    max_steps: int | str | None = None,
    command_timeout_seconds: int | str | None = None,
    max_output_chars: int | str | None = None,
    max_task_seconds: int | str | None = None,
    max_consecutive_tool_failures: int | str | None = None,
    max_conversation_turns: int | str | None = None,
    max_history_items: int | str | None = None,
    environment: Mapping[str, str] | None = None,
) -> Settings:
    """Load settings from explicit arguments, environment variables, and defaults.

    LLM connection settings are read from the local ``.env`` file. Explicit
    arguments are retained for local library callers and tests; environment
    variables continue to configure only workspace and runtime limits. This
    function does not contact an external service or validate an API key.
    """

    # Workspace and runtime limits retain their established explicit-argument
    # and environment-variable behavior. LLM connection settings below are
    # deliberately isolated from the process environment.
    env = os.environ if environment is None else environment
    file_values = _load_connection_file(config_file, environment)
    resolved_workspace = _load_workspace(
        _first_value(workspace, env.get("CODING_AGENT_WORKSPACE"), Path.cwd())
    )
    selected_model = _optional_text(
        _first_value(model, file_values.get("CODING_AGENT_MODEL"))
    )
    selected_provider = _optional_provider(
        file_values.get("CODING_AGENT_PROVIDER")
    )
    selected_base_url = _optional_text(
        _first_value(
            base_url,
            file_values.get("CODING_AGENT_BASE_URL"),
        )
    )
    api_key = _optional_text(
        file_values.get("CODING_AGENT_API_KEY")
    )
    selected_max_output_tokens = _optional_positive_int(
        "max_output_tokens",
        _first_value(
            max_output_tokens,
            file_values.get("CODING_AGENT_MAX_OUTPUT_TOKENS"),
        ),
    )

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
        max_consecutive_tool_failures=_positive_int(
            "max_consecutive_tool_failures",
            _first_value(
                max_consecutive_tool_failures,
                env.get("CODING_AGENT_MAX_CONSECUTIVE_TOOL_FAILURES"),
            ),
            DEFAULT_MAX_CONSECUTIVE_TOOL_FAILURES,
        ),
        max_conversation_turns=_positive_int(
            "max_conversation_turns",
            _first_value(
                max_conversation_turns,
                env.get("CODING_AGENT_MAX_CONVERSATION_TURNS"),
            ),
            DEFAULT_MAX_CONVERSATION_TURNS,
        ),
        max_history_items=_positive_int(
            "max_history_items",
            _first_value(max_history_items, env.get("CODING_AGENT_MAX_HISTORY_ITEMS")),
            DEFAULT_MAX_HISTORY_ITEMS,
        ),
    )
    return Settings(
        workspace=resolved_workspace,
        model=selected_model,
        api_key=api_key,
        base_url=selected_base_url,
        limits=limits,
        provider=selected_provider,
        max_output_tokens=selected_max_output_tokens,
    )


def _first_value(*values: object) -> object | None:
    """Return the first supplied value, preserving caller-defined precedence."""

    for value in values:
        if value is not None:
            return value
    return None


def _load_connection_file(
    config_file: str | Path | None,
    environment: Mapping[str, str] | None,
) -> dict[str, str]:
    """Load only connection fields from an explicit or default local ``.env`` file."""

    # Injected environments make tests deterministic and deliberately do not
    # read a developer's real local credentials. Runtime callers use ``.env``
    # in their current directory unless they select another file explicitly.
    if config_file is None and environment is not None:
        return {}
    path = Path(config_file) if config_file is not None else Path.cwd() / ".env"
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        if config_file is not None:
            raise ConfigurationError(f"configuration file does not exist: {path}")
        return {}
    if not path.is_file():
        raise ConfigurationError(f"configuration path is not a file: {path}")

    try:
        contents = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigurationError(f"cannot read configuration file: {path}") from error
    return _parse_connection_file(contents)


def _parse_connection_file(contents: str) -> dict[str, str]:
    """Parse simple ``NAME=value`` entries without expanding or executing text."""

    values: dict[str, str] = {}
    for line in contents.splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("export "):
            text = text.removeprefix("export ").strip()
        name, separator, value = text.partition("=")
        if not separator or name.strip() not in LOCAL_CONNECTION_FIELDS:
            continue
        values[name.strip()] = _unquote_env_value(value.strip())
    return values


def _unquote_env_value(value: str) -> str:
    """Remove matching outer quotes while keeping configuration data literal."""

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _optional_provider(value: object | None) -> str | None:
    """Normalize a provider identifier without deriving it from the endpoint URL."""

    if value is None:
        return None
    provider = str(value).strip().casefold()
    if not provider:
        return None
    if provider not in SUPPORTED_LLM_PROVIDERS:
        expected = ", ".join(sorted(SUPPORTED_LLM_PROVIDERS))
        raise ConfigurationError(f"provider must be one of: {expected}")
    return provider


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
    if value is None:
        raise ConfigurationError(f"{field_name} must not be empty")
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


def _optional_positive_int(field_name: str, value: object | None) -> int | None:
    """Parse an optional positive integer without creating an implicit model limit."""

    if value is None:
        return None
    return _positive_int(field_name, value, default=1)
