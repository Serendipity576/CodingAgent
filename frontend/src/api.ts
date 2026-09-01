// The frontend calls only same-origin, safe local conversation endpoints.
import type { ConversationSnapshot, PublicConfig } from "./types";

/** Raised when a local API request returns a non-successful HTTP response. */
export class ApiError extends Error {
  constructor(message: string, public readonly status: number) {
    super(message);
    this.name = "ApiError";
  }
}

/** Send one same-origin JSON request and turn errors into displayable messages. */
async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, options);
  const body = (await response.json().catch(() => ({}))) as { detail?: unknown };

  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : response.statusText;
    throw new ApiError(detail || "本地服务请求失败", response.status);
  }
  return body as T;
}

/** Small typed facade around the existing local conversation endpoints. */
export const api = {
  config: () => request<PublicConfig>("/api/config"),
  conversations: () => request<ConversationSnapshot[]>("/api/conversations"),
  conversation: (conversationId: string) =>
    request<ConversationSnapshot>(`/api/conversations/${conversationId}`),
  createConversation: () =>
    request<ConversationSnapshot>("/api/conversations", { method: "POST" }),
  deleteConversation: (conversationId: string) =>
    request<void>(`/api/conversations/${conversationId}`, { method: "DELETE" }),
  sendMessage: (conversationId: string, text: string, clientMessageId: string) =>
    request<ConversationSnapshot>(`/api/conversations/${conversationId}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, client_message_id: clientMessageId }),
    }),
  cancel: (conversationId: string) =>
    request<{ cancelled: boolean }>(`/api/conversations/${conversationId}/cancel`, {
      method: "POST",
    }),
  resolveApproval: (conversationId: string, approvalId: string, approved: boolean) =>
    request<{ accepted: boolean }>(
      `/api/conversations/${conversationId}/approvals/${approvalId}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ approved }),
      },
    ),
};
