"""Command-line entry point for the Coding Agent."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from agent import __version__
from agent.agent import CodingAgent
from agent.config import ConfigurationError, load_settings
from agent.llm.client import LLMConfigurationError, OpenAIResponsesClient
from agent.security.approval import ConsoleApproval
from agent.tools import build_default_registry


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
        "--max-consecutive-tool-failures",
        type=int,
        help="Stop after this many identical consecutive failed tool calls.",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--print-config",
        action="store_true",
        help="Print sanitized startup configuration and exit.",
    )
    action.add_argument(
        "--task",
        metavar="TEXT",
        help="Coding task for the agent to complete.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run configuration inspection or one bounded coding task."""

    parser = build_parser()
    arguments = parser.parse_args(argv)

    # Require an explicit action so opening the CLI never starts a model request
    # or a local command by surprise.
    if not arguments.print_config and not arguments.task:
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
            max_consecutive_tool_failures=arguments.max_consecutive_tool_failures,
        )
    except ConfigurationError as error:
        parser.error(str(error))

    if arguments.print_config:
        # Use the redacted representation instead of serializing Settings directly.
        print(json.dumps(settings.public_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    try:
        agent = CodingAgent(
            settings=settings,
            llm=OpenAIResponsesClient(settings),
            # High-risk requests are approved at the terminal by the person
            # running this exact task; non-interactive use denies them by default.
            tools=build_default_registry(
                settings.workspace,
                approval=ConsoleApproval(),
            ),
        )
    except LLMConfigurationError as error:
        parser.error(str(error))

    result = agent.run(arguments.task)
    print(
        json.dumps(
            {
                "status": result.status.value,
                "steps": result.steps,
                "message": result.message,
                "tool_calls": len(result.tool_calls),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.succeeded else 1
