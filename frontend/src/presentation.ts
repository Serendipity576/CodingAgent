// Pure formatting helpers keep event rendering components compact.
import type { ApprovalRequest, ConversationEvent, ConversationState } from "./types";

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
    agent_finished: "Agent 已完成",
    conversation_turn_finished: "本轮任务已结束",
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
    closed: "已关闭",
    limit_reached: "已达上限",
  };
  return labels[state];
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
