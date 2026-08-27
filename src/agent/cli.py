"""Command-line entry point for the Coding Agent."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from agent import __version__
from agent.config import ConfigurationError, load_settings


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without loading configuration or invoking an LLM."""

    parser = argparse.ArgumentParser(
        prog="coding-agent",
        description="A workspace-bounded coding agent with policy-controlled tools.",
    )
    parser.add_argument(
        "--workspace",
        metavar="PATH",
        help="Project directory the agent will operate in (default: current directory).",
    )
    parser.add_argument(
        "--model",
        help="LLM model name (default: MODEL_NAME or gpt-5).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum tool-calling steps for one task.",
    )
    parser.add_argument(
        "--command-timeout-seconds",
        type=int,
        help="Maximum duration for one local command.",
    )
    parser.add_argument(
        "--max-output-chars",
        type=int,
        help="Maximum captured output size for one tool call.",
    )
    parser.add_argument(
        "--max-task-seconds",
        type=int,
        help="Maximum duration for an entire agent task.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print sanitized startup configuration and exit.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the P0 command-line interface.

    Agent execution is intentionally not available until P1. ``--print-config``
    provides a safe way to verify configuration without exposing credentials.
    """

    parser = build_parser()
    arguments = parser.parse_args(argv)

    # P0 has no agent runtime yet, so invoking the command without an explicit
    # inspection request must not create a task or touch external services.
    if not arguments.print_config:
        parser.print_help()
        return 0

    try:
        settings = load_settings(
            workspace=arguments.workspace,
            model=arguments.model,
            max_steps=arguments.max_steps,
            command_timeout_seconds=arguments.command_timeout_seconds,
            max_output_chars=arguments.max_output_chars,
            max_task_seconds=arguments.max_task_seconds,
        )
    except ConfigurationError as error:
        parser.error(str(error))

    # Use the redacted representation instead of serializing Settings directly.
    print(json.dumps(settings.public_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0
