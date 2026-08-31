/** Safe API contracts consumed by the independently maintained React frontend. */

export interface TaskSummary {
  modified_files: string[];
  modified_file_count: number;
  diff_summaries: DiffSummary[];
  tests: string;
  blocked_actions: number;
  approved_high_risk_actions: number;
  git_baseline_available: boolean;
  preexisting_git_changes: string[];
}

export interface DiffSummary {
  path: string;
  change_type: string;
  added_lines: number;
  removed_lines: number;
}

export interface ConversationSnapshot {
  conversation_id: string;
  workspace: string;
  state: ConversationState;
  turn_count: number;
  max_turns: number;
  queued_messages: number;
  history_items: number;
  max_history_items: number;
  latest_status: string | null;
  latest_message: string | null;
  summary: TaskSummary | null;
}

export type ConversationState = "idle" | "running" | "closed" | "limit_reached";

export interface ConversationEvent {
  sequence: number;
  event: string;
  timestamp: number;
  details: Record<string, unknown>;
}

export interface PublicConfig {
  workspace: string;
  provider: string | null;
  model: string | null;
  base_url: string | null;
  api_key_configured: boolean;
  max_output_tokens: number | null;
  limits: {
    max_steps: number;
    command_timeout_seconds: number;
    max_output_chars: number;
    max_task_seconds: number;
    max_consecutive_tool_failures: number;
    max_conversation_turns: number;
    max_history_items: number;
  };
}

export interface ApprovalRequest {
  approvalId: string;
  tool: string;
  arguments: Record<string, unknown>;
  risk: string;
  reason: string;
  timeoutSeconds: number;
}
