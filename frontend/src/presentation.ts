// Pure formatting helpers keep event rendering components compact.
import type { ApprovalRequest, ConversationEvent, ConversationState } from "./types";

export interface TurnOutcome {
  description: string;
  tone: "error" | "warning" | "neutral";
  title: string;
}

/** Return a short human-facing label for a runtime event. */
export function eventLabel(event: string): string {
  const labels: Record<string, string> = {
    conversation_created: "会话已创建",
    conversation_turn_started: "开始处理任务",
    turn_started: "Agent 已启动",
    llm_request_started: "正在请求模型",
    tool_requested: "请求调用工具",
    tool_finished: "工具执行完成",
    assistant_message: "Agent 回复",
    agent_finished: "Agent 已结束本轮",
    conversation_turn_finished: "本轮处理已结束",
    context_compacted: "已更新本地上下文摘要",
    context_compaction_failed: "本地上下文摘要失败",
    conversation_interrupted: "服务重启后会话已中断",
    approval_required: "等待高风险操作确认",
    approval_resolved: "高风险操作已处理",
    conversation_limit_reached: "会话已达到上限",
    turn_cancel_requested: "已请求取消",
    conversation_closed: "会话已关闭",
  };
  return labels[event] ?? "运行事件";
}

/** Format an event timestamp without exposing browser locale implementation details. */
export function formatTime(timestamp: number): string {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(timestamp * 1000));
}

/** Describe a session state consistently in the sidebar and inspector. */
export function stateLabel(state: ConversationState): string {
  const labels: Record<ConversationState, string> = {
    idle: "等待输入",
    running: "正在执行",
    interrupted: "已中断，可继续",
    closed: "已关闭",
    limit_reached: "已达上限",
  };
  return labels[state];
}

/** Describe terminal results that need a visible user decision or follow-up. */
export function turnOutcome(status: string | null, maxSteps: number): TurnOutcome | null {
  switch (status) {
    case "max_steps_reached":
      return {
        title: "本轮未完成",
        description: `已达到 ${maxSteps} 次工具调用上限。发送下一条消息可继续，或提高 max steps。`,
        tone: "warning",
      };
    case "task_timeout":
      return {
        title: "本轮未完成",
        description: "任务达到执行时间上限。请查看记录后发送下一条消息继续，或提高任务超时限制。",
        tone: "warning",
      };
    case "repeated_tool_failure":
      return {
        title: "本轮未完成",
        description: "同一工具连续失败，运行已停止。请查看执行记录并明确下一步操作。",
        tone: "error",
      };
    case "llm_error":
      return {
        title: "模型请求失败",
        description: "本轮未能取得模型响应。请检查本地服务和模型配置后重试。",
        tone: "error",
      };
    case "cancelled":
      return {
        title: "本轮已取消",
        description: "已停止后续模型和工具调用。发送下一条消息即可继续处理。",
        tone: "neutral",
      };
    case "interrupted":
      return {
        title: "本轮已中断",
        description: "服务重启或中断后不会自动重放任务。请发送下一条消息决定是否继续。",
        tone: "warning",
      };
    default:
      return null;
  }
}

/** Convert a safe server event into the approval data needed by the dialog. */
export function approvalFromEvent(event: ConversationEvent): ApprovalRequest | null {
  if (event.event !== "approval_required") {
    return null;
  }
  const details = event.details;
  if (
    typeof details.approval_id !== "string" ||
    typeof details.tool !== "string" ||
    typeof details.risk !== "string" ||
    typeof details.reason !== "string" ||
    typeof details.timeout_seconds !== "number"
  ) {
    return null;
  }
  return {
    approvalId: details.approval_id,
    tool: details.tool,
    arguments: isRecord(details.arguments) ? details.arguments : {},
    risk: details.risk,
    reason: details.reason,
    timeoutSeconds: details.timeout_seconds,
  };
}

/** Render safe arguments in a stable readable format for cards and dialogs. */
export function formatDetails(value: unknown): string {
  return JSON.stringify(value, null, 2) ?? "{}";
}

/** Read one event field only when it is a string. */
export function stringDetail(event: ConversationEvent, name: string): string | null {
  const value = event.details[name];
  return typeof value === "string" ? value : null;
}

/** Read one event field only when it is a finite number. */
export function numberDetail(event: ConversationEvent, name: string): number | null {
  const value = event.details[name];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** Guard unknown JSON payloads before treating them as object maps. */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
