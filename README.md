# Coding Agent

A coding agent with workspace-bounded local tools, deterministic policy checks, and audit logs. The system design follows one rule: the LLM proposes actions; application code decides whether they may execute.

## Current status

P3 (observability and change control) is complete. The agent now gates every tool call through a deterministic policy, constrains filesystem access to the workspace, protects common credential paths, records structured audit events, and requests approval for high-risk commands.

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

## Safety boundary

Every tool request receives one deterministic decision:

- `ALLOW`: standard reads, tests, and workspace edits execute automatically;
- `REQUIRE_APPROVAL`: deletion, dependency management, network-capable commands, destructive Git operations, and unknown executables require an interactive `y`/`yes` confirmation;
- `DENY`: workspace escape, sensitive files, `sudo`, shutdown/reboot, formatting, and similar critical actions never execute.

The path guard resolves real paths before checking containment, so `../`, absolute escape, and symlink escape are rejected. Common credential paths such as `.env`, private keys, `credentials`, and `secrets` are also denied. The policy is rule-based rather than a complete static analyzer, and it is not container isolation.

## Audit and change summary

Each task writes append-only JSONL events to `.agent/logs/<task-id>.jsonl`. Events record the task, tool name, policy decision, risk, duration, exit metadata, and the final outcome. File bodies, patch text, and tool output are intentionally excluded from the audit log.

The final CLI JSON includes a task summary: changed-file count and paths from successful `apply_patch` calls, per-file added/removed-line counts, the latest recognized test result, blocked actions, and approved high-risk actions. When a task begins, the CLI also captures the read-only `git status --porcelain` baseline. Existing Git changes are reported separately and are never counted as Agent changes.

Automatic rollback is not implemented. The Agent never claims it can restore user changes that existed before a task began.

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
