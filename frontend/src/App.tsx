// Root state composition for the independently maintained Web application.
import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "./api";
import {
  ApprovalDialog,
  Composer,
  Icon,
  Inspector,
  Sidebar,
  Timeline,
} from "./components";
import { approvalFromEvent, stateLabel } from "./presentation";
import type {
  ApprovalRequest,
  ConversationEvent,
  ConversationSnapshot,
  PublicConfig,
} from "./types";

const SSE_EVENT_NAMES = [
  "conversation_created",
  "conversation_turn_started",
  "turn_started",
  "llm_request_started",
  "tool_requested",
  "tool_finished",
  "assistant_message",
  "agent_finished",
  "conversation_turn_finished",
  "approval_required",
  "approval_resolved",
  "conversation_limit_reached",
  "turn_cancel_requested",
  "conversation_closed",
] as const;

type ConnectionState = "connecting" | "connected" | "reconnecting" | "error";

/** Compose the persistent-conversation workbench from safe local API state. */
export default function App() {
  const [config, setConfig] = useState<PublicConfig | null>(null);
  const [sessions, setSessions] = useState<ConversationSnapshot[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [eventsBySession, setEventsBySession] = useState<Record<string, ConversationEvent[]>>({});
  const [approvals, setApprovals] = useState<Record<string, ApprovalRequest>>({});
  const [message, setMessage] = useState("");
  const [notice, setNotice] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [resolvingApproval, setResolvingApproval] = useState(false);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const sequences = useRef(new Map<string, number>());

  const activeSession = sessions.find((session) => session.conversation_id === activeId) ?? null;
  const activeEvents = activeId ? eventsBySession[activeId] ?? [] : [];
  const activeApproval = activeId ? approvals[activeId] ?? null : null;

  /** Merge one fresh server snapshot without losing the sidebar's stable order. */
  const updateSession = useCallback((snapshot: ConversationSnapshot) => {
    setSessions((previous) => {
      const index = previous.findIndex((item) => item.conversation_id === snapshot.conversation_id);
      if (index === -1) {
        return [...previous, snapshot];
      }
      return previous.map((item) => item.conversation_id === snapshot.conversation_id ? snapshot : item);
    });
  }, []);

  /** Refresh metadata only; history and provider internals never reach the browser. */
  const refreshSessions = useCallback(async () => {
    const snapshots = await api.conversations();
    setSessions(snapshots);
    return snapshots;
  }, []);

  /** Create a separate local transcript and focus it immediately. */
  const createConversation = useCallback(async () => {
    setCreating(true);
    setNotice(null);
    try {
      const session = await api.createConversation();
      updateSession(session);
      setActiveId(session.conversation_id);
    } catch (error) {
      setNotice(messageFromError(error, "无法创建会话"));
    } finally {
      setCreating(false);
    }
  }, [updateSession]);

  /** Select an existing session and retrieve its current safe snapshot. */
  const selectConversation = useCallback(async (conversationId: string) => {
    setActiveId(conversationId);
    setNotice(null);
    try {
      updateSession(await api.conversation(conversationId));
    } catch (error) {
      setNotice(messageFromError(error, "无法读取会话"));
    }
  }, [updateSession]);

  /** Store one event once even when an SSE reconnect replays earlier events. */
  const appendEvent = useCallback((conversationId: string, event: ConversationEvent) => {
    const previousSequence = sequences.current.get(conversationId) ?? 0;
    if (event.sequence <= previousSequence) {
      return;
    }
    sequences.current.set(conversationId, event.sequence);
    setEventsBySession((previous) => ({
      ...previous,
      [conversationId]: [...(previous[conversationId] ?? []), event],
    }));
  }, []);

  /** Load configuration and any already-created local sessions on first render. */
  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const [nextConfig, snapshots] = await Promise.all([api.config(), refreshSessions()]);
        if (!active) {
          return;
        }
        setConfig(nextConfig);
        if (snapshots.length > 0) {
          setActiveId(snapshots[0].conversation_id);
        } else {
          setConnection("connected");
        }
      } catch (error) {
        if (active) {
          setConnection("error");
          setNotice(messageFromError(error, "无法连接本地服务"));
        }
      }
    })();
    return () => {
      active = false;
    };
  }, [refreshSessions]);

  /** Attach one named-event SSE stream to the currently visible session. */
  useEffect(() => {
    if (!activeId) {
      return;
    }
    const after = sequences.current.get(activeId) ?? 0;
    const source = new EventSource(`/api/conversations/${activeId}/events?after=${after}`);
    setConnection("connecting");

    const receive = (message: MessageEvent<string>) => {
      const event = parseConversationEvent(message.data);
      if (!event) {
        return;
      }
      appendEvent(activeId, event);
      const pending = approvalFromEvent(event);
      if (pending) {
        setApprovals((previous) => ({ ...previous, [activeId]: pending }));
      }
      if (event.event === "approval_resolved") {
        setApprovals((previous) => removeApproval(previous, activeId));
      }
      if (shouldRefreshSnapshot(event.event)) {
        void api.conversation(activeId).then(updateSession).catch(() => undefined);
      }
    };
    for (const name of SSE_EVENT_NAMES) {
      source.addEventListener(name, receive);
    }
    source.onopen = () => setConnection("connected");
    source.onerror = () => {
      setConnection(source.readyState === EventSource.CLOSED ? "error" : "reconnecting");
    };

    return () => {
      source.close();
    };
  }, [activeId, appendEvent, updateSession]);

  /** Queue a non-empty user message; the backend serializes execution safely. */
  const sendMessage = async () => {
    const text = message.trim();
    if (!activeSession || !text || submitting) {
      return;
    }
    setSubmitting(true);
    setNotice(null);
    try {
      const snapshot = await api.sendMessage(activeSession.conversation_id, text);
      updateSession(snapshot);
      setMessage("");
    } catch (error) {
      setNotice(messageFromError(error, "消息未能发送"));
    } finally {
      setSubmitting(false);
    }
  };

  /** Ask the backend to cancel the active turn and release any approval wait. */
  const cancel = async () => {
    if (!activeSession || cancelling) {
      return;
    }
    setCancelling(true);
    setNotice(null);
    try {
      const result = await api.cancel(activeSession.conversation_id);
      if (!result.cancelled) {
        setNotice("当前会话没有正在执行的任务。");
      }
      updateSession(await api.conversation(activeSession.conversation_id));
    } catch (error) {
      setNotice(messageFromError(error, "取消请求失败"));
    } finally {
      setCancelling(false);
    }
  };

  /** Resolve only the exact approval id emitted for the active session. */
  const resolveApproval = async (approved: boolean) => {
    if (!activeId || !activeApproval || resolvingApproval) {
      return;
    }
    setResolvingApproval(true);
    setNotice(null);
    try {
      await api.resolveApproval(activeId, activeApproval.approvalId, approved);
      setApprovals((previous) => removeApproval(previous, activeId));
    } catch (error) {
      setNotice(messageFromError(error, "审批决定未能提交"));
    } finally {
      setResolvingApproval(false);
    }
  };

  const composerDisabled = !activeSession || ["closed", "limit_reached"].includes(activeSession.state);
  return (
    <main className="app-shell">
      <Sidebar
        activeId={activeId}
        creating={creating}
        onCreate={() => void createConversation()}
        onSelect={(conversationId) => void selectConversation(conversationId)}
        sessions={sessions}
        workspace={config?.workspace ?? null}
      />
      <section className="workspace">
        <header className="workspace-header">
          <div>
            <p className="eyebrow">LOCAL WORKSPACE</p>
            <h1>{activeSession ? `会话 ${activeSession.conversation_id.slice(0, 8)}` : "Coding Agent 工作台"}</h1>
            {activeSession && <p className="session-subtitle">{stateLabel(activeSession.state)} · 本地 history 仅保留在当前进程</p>}
          </div>
          {activeSession && <span className={`header-state state-${activeSession.state}`}>{stateLabel(activeSession.state)}</span>}
        </header>
        {notice && (
          <div className="notice" role="alert">
            <span>{notice}</span>
            <button aria-label="关闭提示" type="button" onClick={() => setNotice(null)}><Icon name="close" size={16} /></button>
          </div>
        )}
        <section className="conversation-stage">
          <Timeline events={activeEvents} onCreate={() => void createConversation()} session={activeSession} />
        </section>
        <Composer
          disabled={composerDisabled}
          onChange={setMessage}
          onSubmit={() => void sendMessage()}
          submitting={submitting}
          value={message}
        />
      </section>
      <Inspector
        cancelling={cancelling}
        config={config}
        connection={connection}
        onCancel={() => void cancel()}
        session={activeSession}
      />
      <ApprovalDialog approval={activeApproval} onResolve={(approved) => void resolveApproval(approved)} resolving={resolvingApproval} />
    </main>
  );
}

/** Refresh state only when the event can change fields visible in the inspector. */
function shouldRefreshSnapshot(eventName: string): boolean {
  return [
    "conversation_turn_started",
    "conversation_turn_finished",
    "conversation_limit_reached",
    "conversation_closed",
  ].includes(eventName);
}

/** Ignore malformed SSE payloads rather than allowing one message to break the UI. */
function parseConversationEvent(raw: string): ConversationEvent | null {
  try {
    const value: unknown = JSON.parse(raw);
    if (!isRecord(value) || typeof value.sequence !== "number" || typeof value.event !== "string" || typeof value.timestamp !== "number" || !isRecord(value.details)) {
      return null;
    }
    return {
      sequence: value.sequence,
      event: value.event,
      timestamp: value.timestamp,
      details: value.details,
    };
  } catch {
    return null;
  }
}

/** Remove a completed approval immutably so the dialog cannot be reused. */
function removeApproval(
  approvals: Record<string, ApprovalRequest>,
  conversationId: string,
): Record<string, ApprovalRequest> {
  const { [conversationId]: _, ...remaining } = approvals;
  return remaining;
}

/** Keep operational errors readable while avoiding assumptions about thrown values. */
function messageFromError(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

/** Guard parsed JSON before reading object properties in the SSE handler. */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
