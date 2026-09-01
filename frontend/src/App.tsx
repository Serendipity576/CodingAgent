// Root state composition for the independently maintained Web application.
import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, api } from "./api";
import {
  ActivityDialog,
  ActivityStatus,
  ApprovalDialog,
  Composer,
  Icon,
  Inspector,
  type PendingUserMessage,
  Sidebar,
  Timeline,
  TurnOutcomeNotice,
} from "./components";
import { approvalFromEvent, stateLabel } from "./presentation";
import type {
  ApprovalRequest,
  ConversationEvent,
  ConversationSnapshot,
  PublicConfig,
  TraceItemDetail,
  TurnTrace,
} from "./types";

const SSE_EVENT_NAMES = [
  "conversation_created",
  "user_message",
  "conversation_turn_started",
  "turn_started",
  "llm_request_started",
  "tool_requested",
  "tool_finished",
  "assistant_message",
  "agent_finished",
  "conversation_turn_finished",
  "trace_updated",
  "conversation_interrupted",
  "approval_required",
  "approval_resolved",
  "conversation_limit_reached",
  "turn_cancel_requested",
  "conversation_closed",
] as const;

const LAST_ACTIVE_CONVERSATION_KEY = "coding-agent:last-active-conversation";
const BACKGROUND_REFRESH_INTERVAL_MS = 5_000;

type ConnectionState = "connecting" | "connected" | "reconnecting" | "error";

interface PendingMessage extends PendingUserMessage {
  conversationId: string;
}

interface LastActiveConversation {
  conversationId: string;
  workspace: string;
}

/** Restore a valid browser-local selection without exposing any conversation content. */
function restoreLastActiveConversation(workspace: string, sessions: ConversationSnapshot[]): string | null {
  try {
    const saved: unknown = JSON.parse(window.localStorage.getItem(LAST_ACTIVE_CONVERSATION_KEY) ?? "null");
    if (!isLastActiveConversation(saved) || saved.workspace !== workspace) {
      return null;
    }
    return sessions.some((session) => session.conversation_id === saved.conversationId)
      ? saved.conversationId
      : null;
  } catch {
    return null;
  }
}

/** Persist only the selected local session identifier for the current workspace. */
function saveLastActiveConversation(workspace: string, conversationId: string) {
  try {
    window.localStorage.setItem(
      LAST_ACTIVE_CONVERSATION_KEY,
      JSON.stringify({ conversationId, workspace } satisfies LastActiveConversation),
    );
  } catch {
    // Private browsing or storage restrictions should not affect conversation use.
  }
}

/** Narrow untrusted browser storage before using its values for session selection. */
function isLastActiveConversation(value: unknown): value is LastActiveConversation {
  return typeof value === "object"
    && value !== null
    && "conversationId" in value
    && "workspace" in value
    && typeof value.conversationId === "string"
    && typeof value.workspace === "string";
}

/** Compose the persistent-conversation workbench from safe local API state. */
export default function App() {
  const [config, setConfig] = useState<PublicConfig | null>(null);
  const [sessions, setSessions] = useState<ConversationSnapshot[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [eventsBySession, setEventsBySession] = useState<Record<string, ConversationEvent[]>>({});
  const [tracesBySession, setTracesBySession] = useState<Record<string, TurnTrace[]>>({});
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [pendingMessages, setPendingMessages] = useState<PendingMessage[]>([]);
  const [unreadSessions, setUnreadSessions] = useState<Record<string, true>>({});
  const [approvals, setApprovals] = useState<Record<string, ApprovalRequest>>({});
  const [notice, setNotice] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [resolvingApproval, setResolvingApproval] = useState(false);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [inspectorOpen, setInspectorOpen] = useState(() => window.innerWidth > 1160);
  const [sidebarOpen, setSidebarOpen] = useState(() => window.innerWidth > 760);
  const [followingLatest, setFollowingLatest] = useState(true);
  const [activityDialogOpen, setActivityDialogOpen] = useState(false);
  const sequences = useRef(new Map<string, number>());
  const pendingOrder = useRef(1_000_000);
  const acknowledgedMessages = useRef(new Set<string>());
  const activeIdRef = useRef<string | null>(null);
  const connectedStreams = useRef(new Set<string>());
  const deletedConversationIds = useRef(new Set<string>());
  const followingLatestRef = useRef(true);
  const conversationStage = useRef<HTMLElement>(null);

  const activeSession = sessions.find((session) => session.conversation_id === activeId) ?? null;
  const activeEvents = activeId ? eventsBySession[activeId] ?? [] : [];
  const activeTraces = activeId ? tracesBySession[activeId] ?? [] : [];
  const activePendingMessages = activeId ? pendingMessages.filter((message) => message.conversationId === activeId) : [];
  const activeApproval = activeId ? approvals[activeId] ?? null : null;
  const message = activeId ? drafts[activeId] ?? "" : "";

  /** Merge one fresh server snapshot without losing the sidebar's stable order. */
  const updateSession = useCallback((snapshot: ConversationSnapshot) => {
    if (deletedConversationIds.current.has(snapshot.conversation_id)) {
      return;
    }
    setSessions((previous) => {
      const index = previous.findIndex((item) => item.conversation_id === snapshot.conversation_id);
      if (index === -1) {
        return [...previous, snapshot];
      }
      return previous.map((item) => item.conversation_id === snapshot.conversation_id ? snapshot : item);
    });
  }, []);

  /** Retrieve list-safe runtime trace structure for one local conversation. */
  const refreshTraces = useCallback(async (conversationId: string): Promise<TurnTrace[]> => {
    const traces = await api.traces(conversationId);
    if (!deletedConversationIds.current.has(conversationId)) {
      setTracesBySession((previous) => ({ ...previous, [conversationId]: traces }));
    }
    return traces;
  }, []);

  /** Refresh metadata only; history and provider internals never reach the browser. */
  const refreshSessions = useCallback(async () => {
    const snapshots = await api.conversations();
    const visibleSnapshots = snapshots.filter(
      (snapshot) => !deletedConversationIds.current.has(snapshot.conversation_id),
    );
    setSessions(visibleSnapshots);
    return visibleSnapshots;
  }, []);

  /** Keep unsent text isolated so changing conversations cannot misroute it. */
  const updateDraft = useCallback((nextDraft: string) => {
    if (!activeId) {
      return;
    }
    setDrafts((previous) => ({ ...previous, [activeId]: nextDraft }));
  }, [activeId]);

  /** Track whether the reader expects new messages to remain in view. */
  const updateFollowingLatest = useCallback((nextValue: boolean) => {
    followingLatestRef.current = nextValue;
    setFollowingLatest((previous) => previous === nextValue ? previous : nextValue);
  }, []);

  /** Return the main chat to its newest message without changing its transcript. */
  const scrollToLatest = useCallback((behavior: ScrollBehavior = "smooth") => {
    const stage = conversationStage.current;
    if (!stage) {
      return;
    }
    updateFollowingLatest(true);
    stage.scrollTo({ top: stage.scrollHeight, behavior });
  }, [updateFollowingLatest]);

  /** Stop automatic scrolling only after the reader deliberately moves upward. */
  const observeConversationScroll = useCallback(() => {
    const stage = conversationStage.current;
    if (!stage) {
      return;
    }
    updateFollowingLatest(stage.scrollHeight - stage.scrollTop - stage.clientHeight < 80);
  }, [updateFollowingLatest]);

  /** Create a separate local transcript and focus it immediately. */
  const createConversation = useCallback(async () => {
    setCreating(true);
    setNotice(null);
    try {
      const session = await api.createConversation();
      updateSession(session);
      updateFollowingLatest(true);
      setActiveId(session.conversation_id);
    } catch (error) {
      setNotice(messageFromError(error, "无法创建会话"));
    } finally {
      setCreating(false);
    }
  }, [updateFollowingLatest, updateSession]);

  /** Select an existing session and retrieve its current safe snapshot. */
  const selectConversation = useCallback(async (conversationId: string) => {
    updateFollowingLatest(true);
    setActiveId(conversationId);
    setActivityDialogOpen(false);
    setUnreadSessions((previous) => removeConversationEntry(previous, conversationId));
    setNotice(null);
    try {
      updateSession(await api.conversation(conversationId));
    } catch (error) {
      setNotice(messageFromError(error, "无法读取会话"));
    }
  }, [updateFollowingLatest, updateSession]);

  /** Remove one local transcript only after the user confirms its permanent deletion. */
  const deleteConversation = useCallback(async (conversationId: string) => {
    if (deletingId || !window.confirm("删除后无法恢复该会话及其本地记录，确定继续吗？")) {
      return;
    }
    // Keep deletion authoritative while pre-existing SSE and fetch requests settle.
    deletedConversationIds.current.add(conversationId);
    setDeletingId(conversationId);
    setNotice(null);
    try {
      try {
        await api.deleteConversation(conversationId);
      } catch (error) {
        if (!(error instanceof ApiError && error.status === 404)) {
          deletedConversationIds.current.delete(conversationId);
          void refreshSessions().catch(() => undefined);
          setNotice(messageFromError(error, "会话未能删除"));
          return;
        }
      }
      sequences.current.delete(conversationId);
      setEventsBySession((previous) => removeConversationEntry(previous, conversationId));
      setTracesBySession((previous) => removeConversationEntry(previous, conversationId));
      setApprovals((previous) => removeConversationEntry(previous, conversationId));
      setDrafts((previous) => removeConversationEntry(previous, conversationId));
      setUnreadSessions((previous) => removeConversationEntry(previous, conversationId));
      setPendingMessages((previous) => previous.filter((message) => message.conversationId !== conversationId));
      const remaining = sessions.filter((item) => item.conversation_id !== conversationId);
      setSessions(remaining);
      if (activeId === conversationId) {
        setActiveId(remaining[0]?.conversation_id ?? null);
        setActivityDialogOpen(false);
      }
    } finally {
      setDeletingId(null);
    }
  }, [activeId, deletingId, refreshSessions, sessions]);

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
          setActiveId(
            restoreLastActiveConversation(nextConfig.workspace, snapshots)
            ?? snapshots[0].conversation_id,
          );
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

  /** Remember the selected transcript so a refresh returns to the same local session. */
  useEffect(() => {
    if (config && activeId) {
      saveLastActiveConversation(config.workspace, activeId);
    }
  }, [activeId, config]);

  /**
   * Keep one live stream for the visible transcript. Browsers cap concurrent
   * HTTP/1.1 connections per origin, so one stream per stored session can
   * otherwise block deletion, approval, and message requests.
   */
  useEffect(() => {
    if (!activeId || deletedConversationIds.current.has(activeId)) {
      setConnection("connected");
      return;
    }
    const conversationId = activeId;
    const after = sequences.current.get(conversationId) ?? 0;
    const source = new EventSource(`/api/conversations/${conversationId}/events?after=${after}`);
    const receive = (message: MessageEvent<string>) => {
      if (deletedConversationIds.current.has(conversationId)) {
        return;
      }
      const event = parseConversationEvent(message.data);
      if (!event) {
        return;
      }
      const clientMessageId = messageClientId(event);
      if (clientMessageId) {
        const key = pendingMessageKey(conversationId, clientMessageId);
        acknowledgedMessages.current.add(key);
        setPendingMessages((previous) => previous.filter((item) => pendingMessageKey(item.conversationId, item.clientMessageId) !== key));
      }
      appendEvent(conversationId, event);
      const pending = approvalFromEvent(event);
      if (pending) {
        setApprovals((previous) => ({ ...previous, [conversationId]: pending }));
      }
      if (event.event === "approval_resolved" || event.event === "conversation_interrupted") {
        setApprovals((previous) => removeApproval(previous, conversationId));
      }
      if (shouldRefreshSnapshot(event.event)) {
        void api.conversation(conversationId).then(updateSession).catch(() => undefined);
      }
      if (event.event === "trace_updated") {
        void refreshTraces(conversationId).catch(() => undefined);
      }
    };
    for (const name of SSE_EVENT_NAMES) {
      source.addEventListener(name, receive);
    }
    source.onopen = () => {
      connectedStreams.current.add(conversationId);
      if (conversationId === activeIdRef.current) {
        setConnection("connected");
      }
    };
    source.onerror = () => {
      connectedStreams.current.delete(conversationId);
      if (conversationId === activeIdRef.current) {
        setConnection(source.readyState === EventSource.CLOSED ? "error" : "reconnecting");
      }
    };
    return () => {
      source.close();
      connectedStreams.current.delete(conversationId);
    };
  }, [activeId, appendEvent, refreshTraces, updateSession]);

  /** Load the active trace on selection and after a browser refresh. */
  useEffect(() => {
    if (activeId && !deletedConversationIds.current.has(activeId)) {
      void refreshTraces(activeId).catch(() => undefined);
    }
  }, [activeId, refreshTraces]);

  /** Refresh inactive session metadata without holding a long-lived connection for each one. */
  useEffect(() => {
    const refreshWhenVisible = () => {
      if (document.visibilityState === "visible") {
        void refreshSessions().catch(() => undefined);
      }
    };
    const interval = window.setInterval(refreshWhenVisible, BACKGROUND_REFRESH_INTERVAL_MS);
    window.addEventListener("focus", refreshWhenVisible);
    return () => {
      window.clearInterval(interval);
      window.removeEventListener("focus", refreshWhenVisible);
    };
  }, [refreshSessions]);

  /** Update active-stream state and clear its unread marker after session selection. */
  useEffect(() => {
    activeIdRef.current = activeId;
    if (!activeId) {
      setConnection("connected");
      return;
    }
    setUnreadSessions((previous) => removeConversationEntry(previous, activeId));
    setConnection(connectedStreams.current.has(activeId) ? "connected" : "connecting");
  }, [activeId]);

  /** Keep new messages visible unless the reader intentionally inspected history. */
  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      if (followingLatestRef.current) {
        const stage = conversationStage.current;
        if (stage) {
          stage.scrollTop = stage.scrollHeight;
        }
      }
    });
    return () => window.cancelAnimationFrame(frame);
  }, [activeEvents.length, activeId, activePendingMessages.length]);

  /** Close side panels by default when the viewport enters their overlay layouts. */
  useEffect(() => {
    const compact = window.matchMedia("(max-width: 1160px)");
    const mobile = window.matchMedia("(max-width: 760px)");
    const syncPanels = () => {
      if (compact.matches) {
        setInspectorOpen(false);
      }
      if (mobile.matches) {
        setSidebarOpen(false);
      }
    };
    syncPanels();
    compact.addEventListener("change", syncPanels);
    mobile.addEventListener("change", syncPanels);
    return () => {
      compact.removeEventListener("change", syncPanels);
      mobile.removeEventListener("change", syncPanels);
    };
  }, []);

  /** Submit an optimistic message and keep it retryable until SSE confirms it. */
  const submitPendingMessage = async (pending: PendingMessage) => {
    const key = pendingMessageKey(pending.conversationId, pending.clientMessageId);
    setSubmitting(true);
    setNotice(null);
    try {
      const snapshot = await api.sendMessage(
        pending.conversationId,
        pending.text,
        pending.clientMessageId,
      );
      updateSession(snapshot);
    } catch (error) {
      if (!acknowledgedMessages.current.has(key)) {
        setPendingMessages((previous) => previous.map((item) => (
          pendingMessageKey(item.conversationId, item.clientMessageId) === key
            ? { ...item, state: "failed" }
            : item
        )));
        setNotice(messageFromError(error, "消息未能发送，可重试"));
      }
    } finally {
      setSubmitting(false);
    }
  };

  /** Add the user's message immediately instead of waiting for an SSE round trip. */
  const sendMessage = async () => {
    const text = message.trim();
    if (!activeSession || !text || submitting) {
      return;
    }
    const pending: PendingMessage = {
      clientMessageId: newClientMessageId(),
      conversationId: activeSession.conversation_id,
      order: pendingOrder.current++,
      state: "sending",
      text,
      timestamp: Date.now() / 1000,
    };
    const key = pendingMessageKey(pending.conversationId, pending.clientMessageId);
    setPendingMessages((previous) => acknowledgedMessages.current.has(key) ? previous : [...previous, pending]);
    updateDraft("");
    await submitPendingMessage(pending);
  };

  /** Retry the exact client id so the backend cannot create a duplicate turn. */
  const retryPendingMessage = async (clientMessageId: string) => {
    if (!activeId || submitting) {
      return;
    }
    const pending = pendingMessages.find((item) => (
      item.conversationId === activeId
      && item.clientMessageId === clientMessageId
      && item.state === "failed"
    ));
    if (!pending) {
      return;
    }
    setPendingMessages((previous) => previous.map((item) => (
      item.conversationId === pending.conversationId && item.clientMessageId === pending.clientMessageId
        ? { ...item, state: "sending" }
        : item
    )));
    await submitPendingMessage({ ...pending, state: "sending" });
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

  /** Refresh before opening so the dialog always starts with the newest trace structure. */
  const openActivityDialog = useCallback(async () => {
    if (activeId) {
      try {
        await refreshTraces(activeId);
      } catch (error) {
        setNotice(messageFromError(error, "无法读取运行追踪"));
      }
    }
    setActivityDialogOpen(true);
  }, [activeId, refreshTraces]);

  /** Fetch one explicit local payload only after the reader selects its trace node. */
  const loadTraceItem = useCallback((turnId: number, itemId: string): Promise<TraceItemDetail> => {
    if (!activeId) {
      return Promise.reject(new Error("未选择会话"));
    }
    return api.traceItem(activeId, turnId, itemId);
  }, [activeId]);

  const composerDisabled = !activeSession || ["closed", "limit_reached"].includes(activeSession.state);
  return (
    <main className={`app-shell ${inspectorOpen ? "inspector-open" : "inspector-closed"}`}>
      <Sidebar
        activeId={activeId}
        creating={creating}
        deletingId={deletingId}
        onClose={() => setSidebarOpen(false)}
        onCreate={() => {
          setSidebarOpen(false);
          void createConversation();
        }}
        onDelete={(conversationId) => void deleteConversation(conversationId)}
        onSelect={(conversationId) => {
          setSidebarOpen(false);
          void selectConversation(conversationId);
        }}
        open={sidebarOpen}
        sessions={sessions}
        unreadSessions={unreadSessions}
        workspace={config?.workspace ?? null}
      />
      <button aria-label="关闭会话列表" className="sidebar-backdrop" type="button" onClick={() => setSidebarOpen(false)} />
      <section className="workspace">
        <header className="workspace-header">
          <div>
            <p className="eyebrow">LOCAL WORKSPACE</p>
            <h1>{activeSession ? `会话 ${activeSession.conversation_id.slice(0, 8)}` : "Coding Agent 工作台"}</h1>
            {activeSession && <p className="session-subtitle">{stateLabel(activeSession.state)} · history 保存在当前 workspace</p>}
          </div>
          <div className="workspace-actions">
            <button
              aria-expanded={sidebarOpen}
              aria-label="打开会话列表"
              className="sidebar-toggle"
              type="button"
              onClick={() => setSidebarOpen(true)}
            >
              <Icon name="menu" size={18} />
            </button>
            {activeSession && activeSession.state !== "idle" && (
              <span className={`header-state state-${activeSession.state}`}>{stateLabel(activeSession.state)}</span>
            )}
            {activeSession && !followingLatest && (
              <button className="latest-button" type="button" onClick={() => scrollToLatest()}>
                回到最新
              </button>
            )}
            <button
              aria-label={inspectorOpen ? "收起运行详情" : "展开运行详情"}
              className={`inspector-toggle ${inspectorOpen ? "active" : ""}`}
              type="button"
              onClick={() => setInspectorOpen((open) => !open)}
            >
              <Icon name="panel" size={17} />
              <span>{inspectorOpen ? "收起详情" : "运行详情"}</span>
            </button>
          </div>
        </header>
        {notice && (
          <div className="notice" role="alert">
            <span>{notice}</span>
            <button aria-label="关闭提示" type="button" onClick={() => setNotice(null)}><Icon name="close" size={16} /></button>
          </div>
        )}
        <section className="conversation-stage" ref={conversationStage} onScroll={observeConversationScroll}>
          <Timeline
            events={activeEvents}
            onCreate={() => void createConversation()}
            onRetry={(clientMessageId) => void retryPendingMessage(clientMessageId)}
            pendingMessages={activePendingMessages}
            session={activeSession}
          />
        </section>
        <ActivityStatus events={activeEvents} session={activeSession} />
        <TurnOutcomeNotice
          maxSteps={config?.limits.max_steps ?? 20}
          onOpenActivity={() => void openActivityDialog()}
          session={activeSession}
        />
        <Composer
          disabled={composerDisabled}
          focusKey={activeId}
          onChange={updateDraft}
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
        onOpenActivity={() => void openActivityDialog()}
        open={inspectorOpen}
        session={activeSession}
        traces={activeTraces}
      />
      <ApprovalDialog approval={activeApproval} onResolve={(approved) => void resolveApproval(approved)} resolving={resolvingApproval} />
      <ActivityDialog
        onClose={() => setActivityDialogOpen(false)}
        onLoadItem={loadTraceItem}
        open={activityDialogOpen}
        traces={activeTraces}
      />
    </main>
  );
}

/** Refresh state only when the event can change fields visible in the inspector. */
function shouldRefreshSnapshot(eventName: string): boolean {
  return [
    "conversation_turn_started",
    "conversation_turn_finished",
    "conversation_interrupted",
    "conversation_limit_reached",
    "conversation_closed",
  ].includes(eventName);
}

/** Read the optional browser correlation id without trusting arbitrary event data. */
function messageClientId(event: ConversationEvent): string | null {
  if (event.event !== "user_message") {
    return null;
  }
  const value = event.details.client_message_id;
  return typeof value === "string" ? value : null;
}

/** Scope a local delivery id to exactly one durable conversation. */
function pendingMessageKey(conversationId: string, clientMessageId: string): string {
  return `${conversationId}:${clientMessageId}`;
}

/** Generate a local correlation id without exposing it outside the loopback API. */
function newClientMessageId(): string {
  if (typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
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

/** Remove stale state for a permanently deleted conversation without mutation. */
function removeConversationEntry<T>(
  entries: Record<string, T>,
  conversationId: string,
): Record<string, T> {
  const { [conversationId]: _, ...remaining } = entries;
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
