# Coding Agent

A coding agent with workspace-bounded local tools, deterministic policy checks, and audit logs. The system design follows one rule: the LLM proposes actions; application code decides whether they may execute.

## Current status

P4 (security tests and demonstration) and P6 (persistent conversations and the local Web interface) are complete. The Agent gates every tool call through a deterministic policy, constrains filesystem access to the workspace, protects common credential paths, records structured audit events, and includes a reproducible prompt-injection demonstration.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m pip install .
coding-agent --help
coding-agent --workspace . --print-config
python -m unittest discover -s tests -t .
```

`--print-config` never prints the API key. It is only a startup check; it does not call an LLM or execute any local tool.

To run a task, put compatible Responses connection fields in an untracked `.env`
file in the directory where you run the command:

```bash
cat > .env <<'EOF'
CODING_AGENT_API_KEY="your_api_key"
CODING_AGENT_BASE_URL="https://llm.example.com/v1"
CODING_AGENT_MODEL="compatible-model"
CODING_AGENT_PROVIDER="responses"
CODING_AGENT_MAX_OUTPUT_TOKENS="2048"
EOF'

coding-agent \
  --workspace /path/to/project \
  --task "Fix the failing tests."
```

Every endpoint uses the same client-owned, stateless transcript: the initial task, original response items, and tool outputs are resubmitted on each turn. The client never sends `previous_response_id` or creates a server-side conversation. This follows DeepSeek's [Responses API compatibility guide](https://api-docs.deepseek.com/guides/responses_api/) and OpenAI's [stateless Responses guidance](https://developers.openai.com/api/reference/cli/resources/responses/methods/create).

Compatibility requires the service to support the OpenAI Responses API with `input` items and custom function tools. A service that implements only the legacy Chat Completions API is outside this interface. The CLI never asks for a provider name: `.env` selects the internal adapter with `CODING_AGENT_PROVIDER`, while `responses` uses the shared request format for other compatible endpoints.

The model may use `list_files`, `read_file`, `apply_patch`, and `run_command`. `apply_patch` replaces one exact, unique text fragment; `run_command` accepts an executable and arguments rather than shell syntax.

## Persistent conversations

`--task` runs one user turn. It can contain multiple model/tool exchanges, but it ends once the model returns a final message.

Start a terminal conversation to send several user messages through one locally held LLM transcript:

```bash
coding-agent chat --workspace /path/to/project
```

Terminal commands are `/help`, `/new`, `/sessions`, `/open <id>`, `/status`, `/cancel`, and `/quit`. `/new` creates a fresh local transcript; `/sessions` lists the workspace's saved sessions; `/open` resumes one using an unambiguous id prefix.

Each workspace stores its sessions in `.agent/conversations/sessions.sqlite3`. The database contains the client-owned model transcript, session metadata, and browser-safe event journal, but never the API key. On POSIX systems it is created with owner-only permissions and Agent tools are denied access to `.agent`. It is local runtime data, is ignored by Git, and should be backed up or deleted only deliberately. A turn that was active during a process restart is marked **interrupted**; the Agent never reruns its model or tool calls automatically.

Start the local Web interface:

```bash
coding-agent serve --workspace /path/to/project
```

Open `http://127.0.0.1:8765`. This is a React workbench with a session sidebar, typed event timeline, collapsible tool cards, execution summary, cancellation controls, high-risk approval dialog, and explicit local-session deletion. It communicates with the local server through REST and server-sent events; it never receives the API key or provider reasoning data. The server rejects non-loopback hosts.

### Web frontend development

The production bundle is served from `src/agent/web/static/` and is included in
the Python package. To rebuild it after changing the React source, use Node.js
20.19 or later:

```bash
cd frontend
npm ci
npm run build
```

`node_modules/` is local-only. Commit the React source, `package-lock.json`, and
the regenerated `src/agent/web/static/` bundle so `coding-agent serve` works
after a normal Python installation without a separate Node.js step.

For hot-reload development, start `coding-agent serve` in one terminal and run
`npm run dev` in another; then open `http://127.0.0.1:5173/static/`. Vite proxies
only `/api` to the loopback FastAPI service.

One workspace executes one Agent turn at a time, even across several browser conversations. Sessions have a turn limit, while their complete raw transcript remains local until explicit deletion. Each model request uses a separately selected, budgeted context view; completed older history can be summarized without losing the original record. Completed sessions can be resumed after restart.

## Safety boundary

Every tool request receives one deterministic decision:

- `ALLOW`: standard reads, tests, and workspace edits execute automatically;
- `REQUIRE_APPROVAL`: deletion, dependency management, network-capable commands, destructive Git operations, and unknown executables require an interactive `y`/`yes` confirmation;
- `DENY`: workspace escape, sensitive files, `sudo`, shutdown/reboot, formatting, and similar critical actions never execute.

The path guard resolves real paths before checking containment, so `../`, absolute escape, and symlink escape are rejected. Common credential paths such as `.env`, private keys, `credentials`, `secrets`, and the Agent's own `.agent` state are also denied. The policy is rule-based rather than a complete static analyzer, and it is not container isolation.

In Web mode, `REQUIRE_APPROVAL` pauses the worker and shows one browser prompt containing the exact tool, safe argument summary, risk, and reason. Approval applies only to that one call; rejection, timeout, cancellation, and an unknown approval id deny it. Cancelling a turn also stops an active local command and its POSIX child process group where supported.

## Audit and change summary

Each task writes append-only JSONL events to `.agent/logs/<task-id>.jsonl`. Events record the task, tool name, policy decision, risk, duration, exit metadata, and the final outcome. File bodies, patch text, and tool output are intentionally excluded from the audit log.

The final CLI JSON includes a task summary: changed-file count and paths from successful `apply_patch` calls, per-file added/removed-line counts, the latest recognized test result, blocked actions, and approved high-risk actions.

Web conversations additionally keep a private, per-turn runtime trace in the local session database. It groups model exchanges and their tool calls, including duration, policy decision, completion state, terminal result, and context-selection facts. The normal trace list and SSE updates never contain model request/response bodies. Opening one trace item in the local Web page requests its sanitized request or response on demand; API keys, authentication headers, common credential fields, and common secret patterns are redacted, oversized text is marked as truncated, and encrypted model reasoning is not recorded.

The context manager uses a fixed DeepSeek-V4-Flash baseline: a 1,048,576-token window, a 393,216-token output reserve, and a local safety margin. These are application defaults rather than environment variables. It preserves raw history in SQLite, stores structured summaries separately, retains oversized tool output locally as conversation artifacts, and lets the Agent read a referenced artifact in bounded chunks when needed.

## Security verification and demo

Run the complete automated suite:

```bash
python -m unittest discover -s tests -t .
```

The suite covers workspace escape (`../`, absolute paths, and symlinks), sensitive paths, dangerous commands, approval handling, command timeout, output truncation, invalid tool arguments, repeated failures, and maximum-step termination.

Run the offline prompt-injection demonstration (no API key or network request):

```bash
python examples/prompt_injection_demo/run_demo.py
```

It copies a deliberately malicious example repository into a temporary workspace. Its scripted model reads the malicious README, then attempts to read `.env`; the policy denies that request. The same run then diagnoses the failing test, patches `app.py`, and reruns the test successfully. Expected final fields are `blocked_actions: 1`, `tests: "passed"`, and `modified_files: ["app.py"]`. The script normally completes in a few seconds and removes its temporary workspace afterwards.

This is a deterministic policy demonstration, not evidence that every model will resist prompt injection. Repository text and tool output remain untrusted; the application policy is the enforcement boundary.

## Configuration

LLM connection settings are read from `.env` only. The main CLI does not accept
provider, API, endpoint, model, or generated-token settings as arguments.

| Setting | Configuration source | Default |
| --- | --- | --- |
| Workspace | `--workspace` or `CODING_AGENT_WORKSPACE` | Current directory |
| Provider | `CODING_AGENT_PROVIDER` in `.env` | Not set; required for a task |
| Model | `CODING_AGENT_MODEL` in `.env` | Not set; required for a task |
| API key | `CODING_AGENT_API_KEY` in `.env` | Not set; required for a task |
| Base URL | `CODING_AGENT_BASE_URL` in `.env` | Not set; required for a task |
| Maximum generated tokens | `CODING_AGENT_MAX_OUTPUT_TOKENS` in `.env` | Not set |
| Maximum steps | `CODING_AGENT_MAX_STEPS` | `20` |
| Command timeout | `CODING_AGENT_COMMAND_TIMEOUT_SECONDS` | `60` seconds |
| Tool output limit | `CODING_AGENT_MAX_OUTPUT_CHARS` | `20000` characters |
| Task timeout | `CODING_AGENT_MAX_TASK_SECONDS` | `900` seconds |
| Repeated tool failure limit | `CODING_AGENT_MAX_CONSECUTIVE_TOOL_FAILURES` | `2` |
| Conversation turn limit | `CODING_AGENT_MAX_CONVERSATION_TURNS` | `50` |
| Local working-context baseline | Fixed in application code | DeepSeek-V4-Flash: 1,048,576-token window / 393,216-token output reserve |

Store `CODING_AGENT_API_KEY` in the untracked local `.env` file; never commit it. The loader reads `CODING_AGENT_PROVIDER`, `CODING_AGENT_API_KEY`, `CODING_AGENT_BASE_URL`, `CODING_AGENT_MODEL`, and the optional `CODING_AGENT_MAX_OUTPUT_TOKENS` as literal values. It does not expand or execute its contents. Set `CODING_AGENT_PROVIDER` to `openai`, `deepseek`, or `responses`; the final value selects the corresponding internal adapter without inferring it from the URL.

## Development plan

Scope, phase boundaries, acceptance criteria, and progress updates live in [实施计划与进度.md](实施计划与进度.md). The detailed architecture is in [项目启动文档.md](项目启动文档.md).
