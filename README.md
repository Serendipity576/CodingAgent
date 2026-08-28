# Coding Agent

A coding agent with workspace-bounded local tools, deterministic policy checks, and audit logs. The system design follows one rule: the LLM proposes actions; application code decides whether they may execute.

## Current status

P1 (the minimal Agent loop) is complete. It supports model function calls, file tools, focused text patches, local commands, runtime limits, and a multi-turn task loop. P2 will add the complete safety policy and approval flow.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m pip install .
coding-agent --help
coding-agent --workspace . --print-config
python -m unittest discover -s tests -t .
```

`--print-config` never prints the API key. It is only a startup check; it does not call an LLM or execute any local tool.

To run a task, install dependencies and provide an API key through the environment:

```bash
python -m pip install .
export OPENAI_API_KEY="your_api_key"
coding-agent --workspace /path/to/project --task "Fix the failing tests."
```

The model may use `list_files`, `read_file`, `apply_patch`, and `run_command`. `apply_patch` replaces one exact, unique text fragment; `run_command` accepts an executable and arguments rather than shell syntax.

> P1 is not the final security boundary. Its workspace check and runtime limits are basic safeguards only. Do not use it on repositories containing credentials or important uncommitted work until P2 adds sensitive-file protection, command policy, and approval handling.

## Configuration

CLI arguments override environment variables.

| Setting | Environment variable | Default |
| --- | --- | --- |
| Workspace | `CODING_AGENT_WORKSPACE` | Current directory |
| Model | `MODEL_NAME` | `gpt-5` |
| API key | `OPENAI_API_KEY` | Not set |
| Base URL | `OPENAI_BASE_URL` | Not set |
| Maximum steps | `CODING_AGENT_MAX_STEPS` | `20` |
| Command timeout | `CODING_AGENT_COMMAND_TIMEOUT_SECONDS` | `60` seconds |
| Tool output limit | `CODING_AGENT_MAX_OUTPUT_CHARS` | `20000` characters |
| Task timeout | `CODING_AGENT_MAX_TASK_SECONDS` | `900` seconds |
| Repeated tool failure limit | `CODING_AGENT_MAX_CONSECUTIVE_TOOL_FAILURES` | `2` |

Store `OPENAI_API_KEY` in the process environment or an untracked local configuration file; never commit it. The official OpenAI documentation recommends loading API keys from environment variables rather than exposing them in application code.

## Development plan

Scope, phase boundaries, acceptance criteria, and progress updates live in [实施计划与进度.md](实施计划与进度.md). The detailed architecture is in [项目启动文档.md](项目启动文档.md).
