# Coding Agent

A coding agent with workspace-bounded local tools, deterministic policy checks, and audit logs. The system design follows one rule: the LLM proposes actions; application code decides whether they may execute.

## Current status

P0 (project skeleton and configuration) is complete. Agent execution, local tools, and security policies will be added in later phases.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m pip install .
coding-agent --help
coding-agent --workspace . --print-config
python -m unittest discover -s tests -t .
```

`--print-config` never prints the API key. It is only a startup check; it does not call an LLM or execute any local tool.

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

Store `OPENAI_API_KEY` in the process environment or an untracked local configuration file; never commit it. The official OpenAI documentation recommends loading API keys from environment variables rather than exposing them in application code.

## Development plan

Scope, phase boundaries, acceptance criteria, and progress updates live in [实施计划与进度.md](实施计划与进度.md). The detailed architecture is in [项目启动文档.md](项目启动文档.md).
