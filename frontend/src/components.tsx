// Presentational components deliberately receive safe API data rather than LLM history.
import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  eventLabel,
  formatDetails,
  formatTime,
  stateLabel,
  stringDetail,
  turnOutcome,
} from "./presentation";
import type {
  ApprovalRequest,
  ConversationEvent,
  ConversationSnapshot,
  PublicConfig,
  TaskSummary,
  TraceItem,
  TraceItemDetail,
  TurnTrace,
} from "./types";

type IconName = "add" | "arrow" | "close" | "code" | "delete" | "history" | "menu" | "panel" | "pause" | "send";

const MESSAGE_EVENTS = new Set(["user_message", "assistant_message"]);
const DISMISSED_OUTCOME_STORAGE_KEY = "coding-agent:dismissed-turn-outcomes";
const MAX_DISMISSED_OUTCOMES = 100;

export interface PendingUserMessage {
  clientMessageId: string;
  order: number;
  state: "sending" | "failed";
  text: string;
  timestamp: number;
}

interface IconProps {
  name: IconName;
  size?: number;
}

/** Render a small local SVG icon without adding an icon-library dependency. */
export function Icon({ name, size = 18 }: IconProps) {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.8,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };

  const paths: Record<IconName, ReactNode> = {
    add: <path d="M12 5v14M5 12h14" />,
    arrow: <path d="m9 18 6-6-6-6" />,
    close: <path d="m6 6 12 12M18 6 6 18" />,
    code: <path d="m8 9-3 3 3 3m8-6 3 3-3 3M14 5l-4 14" />,
    delete: <path d="M4 7h16m-10 4v5m4-5v5M9 7l1-3h4l1 3m-9 0 1 13h10l1-13" />,
    history: <path d="M3 12a9 9 0 1 0 3-6.7M3 4v5h5m4-4v7l5 3" />,
    menu: <path d="M4 7h16M4 12h16M4 17h16" />,
    panel: <path d="M4 5h16v14H4zm10 0v14M7 9h4m-4 3h4" />,
    pause: <path d="M8 5v14M16 5v14" />,
    send: <path d="m21 3-7.5 18-3.8-7.7L3 9.5 21 3Zm-11.4 10.3L15 9" />,
  };

  return <svg {...common}>{paths[name]}</svg>;
}

interface SidebarProps {
  sessions: ConversationSnapshot[];
  unreadSessions: Record<string, true>;
  activeId: string | null;
  workspace: string | null;
  creating: boolean;
  deletingId: string | null;
  open: boolean;
  onCreate: () => void;
  onClose: () => void;
  onDelete: (conversationId: string) => void;
  onSelect: (conversationId: string) => void;
}

/** Display local sessions and make the active session unambiguous. */
export function Sidebar({
  sessions,
  unreadSessions,
  activeId,
  workspace,
  creating,
  deletingId,
  open,
  onCreate,
  onClose,
  onDelete,
  onSelect,
}: SidebarProps) {
  return (
    <aside className={`sidebar ${open ? "open" : ""}`}>
      <div className="brand">
        <span className="brand-mark"><Icon name="code" size={20} /></span>
        <span className="brand-copy"><strong>Coding Agent</strong><small>LOCAL WORKBENCH</small></span>
        <button aria-label="关闭会话列表" className="sidebar-close" type="button" onClick={onClose}>
          <Icon name="close" size={17} />
        </button>
      </div>
      <div className="workspace-context">
        <span>WORKSPACE</span>
        <p className="workspace-path" title={workspace ?? undefined}>
          {workspace ?? "正在连接本地服务…"}
        </p>
      </div>
      <button className="new-conversation" type="button" onClick={onCreate} disabled={creating}>
        <Icon name="add" />
        {creating ? "正在创建" : "新建会话"}
      </button>
      <div className="sidebar-heading">
        <span>会话历史</span>
        <span className="count-badge">{sessions.length}</span>
      </div>
      <nav className="conversation-list" aria-label="会话列表">
        {sessions.length === 0 ? (
          <p className="empty-list">尚无会话</p>
        ) : (
          sessions.map((session) => {
            const isDeleting = deletingId === session.conversation_id;
            return (
              <div className="conversation-row" key={session.conversation_id}>
                <button
                  className={`conversation-item ${session.conversation_id === activeId ? "active" : ""}`}
                  type="button"
                  onClick={() => onSelect(session.conversation_id)}
                >
                  <span className="conversation-item-top">
                    <span className="conversation-name">
                      会话 {session.conversation_id.slice(0, 6)}
                      {unreadSessions[session.conversation_id] && <span aria-label="有新动态" className="unread-dot" />}
                    </span>
                    <StateDot state={session.state} />
                  </span>
                  <span className="conversation-item-bottom">
                    {stateLabel(session.state)}
                  </span>
                </button>
                <button
                  aria-label={`删除会话 ${session.conversation_id.slice(0, 8)}`}
                  className="delete-conversation"
                  disabled={isDeleting}
                  type="button"
                  onClick={() => onDelete(session.conversation_id)}
                >
                  <Icon name="delete" size={15} />
                </button>
              </div>
            );
          })
        )}
      </nav>
    </aside>
  );
}

/** Show a compact color-coded lifecycle state indicator. */
function StateDot({ state }: { state: ConversationSnapshot["state"] }) {
  return <span className={`state-dot state-${state}`} title={stateLabel(state)} />;
}

interface TimelineProps {
  events: ConversationEvent[];
  pendingMessages: PendingUserMessage[];
  session: ConversationSnapshot | null;
  onCreate: () => void;
  onRetry: (clientMessageId: string) => void;
}

/** Keep the chat surface focused on the messages authored by either participant. */
export function Timeline({ events, pendingMessages, session, onCreate, onRetry }: TimelineProps) {
  if (!session) {
    return (
      <div className="empty-stage">
        <span className="empty-stage-icon"><Icon name="history" size={28} /></span>
        <h1>开始一个本地会话</h1>
        <p>创建会话后，Agent 会在受限的 workspace 内持续处理你的后续消息。</p>
        <button className="primary-button" type="button" onClick={onCreate}>
          <Icon name="add" /> 新建会话
        </button>
      </div>
    );
  }

  // Runtime details remain durable and available in the inspector; the main
  // conversation intentionally contains just the two participants' messages.
  const messageItems = [
    ...events.filter(isMessageEvent).map((event) => ({ event, key: `event-${event.sequence}`, order: event.sequence, timestamp: event.timestamp, type: "event" as const })),
    ...pendingMessages.map((message) => ({ message, key: `pending-${message.clientMessageId}`, order: message.order, timestamp: message.timestamp, type: "pending" as const })),
  ].sort((left, right) => left.timestamp - right.timestamp || left.order - right.order);

  if (messageItems.length === 0) {
    return (
      <div className="empty-stage compact">
        <span className="empty-stage-icon"><Icon name="code" size={25} /></span>
        <h1>可以开始了</h1>
        <p>描述要处理的编码任务，或继续上一轮的工作。</p>
      </div>
    );
  }

  return (
    <div aria-relevant="additions" className="timeline" role="log">
      {messageItems.map((item) => item.type === "event" ? (
        <MessageEventCard event={item.event} key={item.key} />
      ) : (
        <PendingMessageCard key={item.key} message={item.message} onRetry={onRetry} />
      ))}
    </div>
  );
}

/** Render one user or assistant event after Timeline has filtered the journal. */
function MessageEventCard({ event }: { event: ConversationEvent }) {
  const text = stringDetail(event, "text");
  if (event.event === "user_message") {
    return <MessageCard role="user" label="你" time={formatTime(event.timestamp)}>{text ?? ""}</MessageCard>;
  }
  return <MessageCard role="assistant" label="Coding Agent" time={formatTime(event.timestamp)}>{text || "Agent 未返回文本。"}</MessageCard>;
}

/** Render a locally queued message before the server confirms its safe event. */
function PendingMessageCard({
  message,
  onRetry,
}: {
  message: PendingUserMessage;
  onRetry: (clientMessageId: string) => void;
}) {
  return (
    <MessageCard
      deliveryState={message.state}
      label="你"
      onRetry={() => onRetry(message.clientMessageId)}
      role="user"
      time={formatTime(message.timestamp)}
    >
      {message.text}
    </MessageCard>
  );
}

/** Render one message body while preserving its line breaks without injecting HTML. */
function MessageCard({
  role,
  label,
  time,
  children,
  deliveryState,
  onRetry,
}: {
  role: "user" | "assistant";
  label: string;
  time: string;
  children: string;
  deliveryState?: PendingUserMessage["state"];
  onRetry?: () => void;
}) {
  return (
    <article className={`message-card ${role} ${deliveryState ? `delivery-${deliveryState}` : ""}`}>
      <header className="message-meta">
        <span>{label}</span>
        <span className="message-meta-right">
          {deliveryState && <span className={`delivery-state ${deliveryState}`}>{deliveryState === "sending" ? "发送中" : "发送失败"}</span>}
          <time>{time}</time>
        </span>
      </header>
      {role === "assistant" ? <AssistantMarkdown content={children} /> : <p>{children}</p>}
      {deliveryState === "failed" && (
        <button className="message-retry" type="button" onClick={onRetry}>重试</button>
      )}
    </article>
  );
}

/** Render model Markdown without enabling untrusted raw HTML from the response. */
function AssistantMarkdown({ content }: { content: string }) {
  return (
    <div className="markdown-body">
      <ReactMarkdown
        components={{
          a: ({ children, href }) => <a href={href} rel="noreferrer" target="_blank">{children}</a>,
        }}
        remarkPlugins={[remarkGfm]}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

interface ActivityStatusProps {
  events: ConversationEvent[];
  session: ConversationSnapshot | null;
}

/** Show one changing status line while a turn is active, similar to a chat typing state. */
export function ActivityStatus({ events, session }: ActivityStatusProps) {
  if (session?.state !== "running") {
    return null;
  }
  const latest = [...events].reverse().find(isActivityEvent);
  const status = latest ? activityStatus(latest) : "正在准备任务";

  return (
    <div aria-live="polite" className="live-activity" role="status">
      <span className="activity-pulse" aria-hidden="true" />
      <span className="live-activity-text" key={latest?.sequence ?? "preparing"}>{status}</span>
    </div>
  );
}

interface TurnOutcomeNoticeProps {
  maxSteps: number;
  onOpenActivity: () => void;
  session: ConversationSnapshot | null;
}

/** Keep incomplete terminal turns visible until the user dismisses that exact turn result. */
export function TurnOutcomeNotice({ maxSteps, onOpenActivity, session }: TurnOutcomeNoticeProps) {
  const [dismissedOutcomes, setDismissedOutcomes] = useState(readDismissedOutcomes);
  if (!session || session.state === "running") {
    return null;
  }
  const outcome = turnOutcome(session.latest_status, maxSteps);
  const outcomeKey = outcome ? dismissedOutcomeKey(session) : null;
  if (!outcome || !outcomeKey || dismissedOutcomes.includes(outcomeKey)) {
    return null;
  }

  const dismiss = () => {
    const next = [...dismissedOutcomes.filter((key) => key !== outcomeKey), outcomeKey]
      .slice(-MAX_DISMISSED_OUTCOMES);
    setDismissedOutcomes(next);
    saveDismissedOutcomes(next);
  };

  return (
    <section aria-live="assertive" className={`turn-outcome turn-outcome-${outcome.tone}`} role="alert">
      <div>
        <strong>{outcome.title}</strong>
        <p>{outcome.description}</p>
      </div>
      <div className="turn-outcome-actions">
        <button type="button" onClick={onOpenActivity}>查看记录</button>
        <button aria-label="关闭本轮提醒" className="turn-outcome-close" type="button" onClick={dismiss}>
          <Icon name="close" size={15} />
        </button>
      </div>
    </section>
  );
}

/** Identify one terminal result without storing task text or conversation history. */
function dismissedOutcomeKey(session: ConversationSnapshot): string | null {
  return session.latest_status
    ? `${session.conversation_id}:${session.latest_event_sequence}:${session.latest_status}`
    : null;
}

/** Read a bounded list of locally dismissed results; unavailable storage is harmless. */
function readDismissedOutcomes(): string[] {
  try {
    const stored: unknown = JSON.parse(window.localStorage.getItem(DISMISSED_OUTCOME_STORAGE_KEY) ?? "[]");
    return Array.isArray(stored)
      ? stored.filter((value): value is string => typeof value === "string").slice(-MAX_DISMISSED_OUTCOMES)
      : [];
  } catch {
    return [];
  }
}

/** Persist only bounded result identifiers so a closed reminder stays closed after refresh. */
function saveDismissedOutcomes(outcomes: string[]) {
  try {
    window.localStorage.setItem(DISMISSED_OUTCOME_STORAGE_KEY, JSON.stringify(outcomes));
  } catch {
    // Storage restrictions should not prevent the user from working with a session.
  }
}

/** Keep a compact, grouped trace summary in the secondary inspector. */
function ActivityLog({ traces, onOpen }: { traces: TurnTrace[]; onOpen: () => void }) {
  const latestTrace = traces.at(-1);
  const items = latestTrace?.items.slice(-4).reverse() ?? [];
  return (
    <section className="activity-log">
      <div className="activity-log-heading">
        <h3>运行追踪</h3>
        <button aria-haspopup="dialog" className="activity-log-open" type="button" onClick={onOpen}>展开记录</button>
      </div>
      {items.length === 0 ? (
        <p className="activity-log-empty">模型调用、工具执行和安全决策会按轮次显示在这里。</p>
      ) : (
        <div className="activity-list">
          {items.map((item) => <TraceEntry item={item} key={item.item_id} />)}
        </div>
      )}
    </section>
  );
}

interface ActivityDialogProps {
  traces: TurnTrace[];
  open: boolean;
  onClose: () => void;
  onLoadItem: (turnId: number, itemId: string) => Promise<TraceItemDetail>;
}

type TraceFilter = "all" | "model" | "tool" | "security" | "error";

/** Show grouped model and tool traces without widening the chat layout. */
export function ActivityDialog({ traces, open, onClose, onLoadItem }: ActivityDialogProps) {
  const dialog = useRef<HTMLElement>(null);
  const closeButton = useRef<HTMLButtonElement>(null);
  const onCloseRef = useRef(onClose);
  const [filter, setFilter] = useState<TraceFilter>("all");
  const [selected, setSelected] = useState<{ turnId: number; itemId: string } | null>(null);
  const [detail, setDetail] = useState<TraceItemDetail | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  onCloseRef.current = onClose;
  const traceItems = traces.flatMap((trace) => trace.items.map((item) => ({ turnId: trace.turn_id, item }))).reverse();
  const visibleItems = traceItems.filter(({ item }) => matchesTraceFilter(item, filter));
  const selectedItem = visibleItems.find(({ turnId, item }) => (
    turnId === selected?.turnId && item.item_id === selected.itemId
  )) ?? visibleItems[0] ?? null;

  useEffect(() => {
    if (!open) {
      setSelected(null);
      setDetail(null);
      setDetailError(null);
      return;
    }
    setSelected((previous) => (
      previous !== null && visibleItems.some(({ turnId, item }) => (
        turnId === previous.turnId && item.item_id === previous.itemId
      ))
        ? previous
        : selectedItem ? { turnId: selectedItem.turnId, itemId: selectedItem.item.item_id } : null
    ));
  }, [filter, open, selectedItem, visibleItems]);

  useEffect(() => {
    if (!open || !selectedItem) {
      return;
    }
    let active = true;
    setLoadingDetail(true);
    setDetail(null);
    setDetailError(null);
    void onLoadItem(selectedItem.turnId, selectedItem.item.item_id)
      .then((nextDetail) => {
        if (active) {
          setDetail(nextDetail);
        }
      })
      .catch((error: unknown) => {
        if (active) {
          setDetail(null);
          setDetailError(error instanceof Error ? error.message : "无法读取追踪详情");
        }
      })
      .finally(() => {
        if (active) {
          setLoadingDetail(false);
        }
      });
    return () => {
      active = false;
    };
  }, [
    onLoadItem,
    open,
    selectedItem?.item.duration_ms,
    selectedItem?.item.item_id,
    selectedItem?.item.status,
    selectedItem?.turnId,
  ]);

  useEffect(() => {
    if (!open) {
      return;
    }
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const frame = window.requestAnimationFrame(() => closeButton.current?.focus());
    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.defaultPrevented) {
        return;
      }
      if (event.key === "Escape") {
        event.preventDefault();
        onCloseRef.current();
        return;
      }
      if (event.key !== "Tab") {
        return;
      }
      const focusable = dialog.current?.querySelectorAll<HTMLElement>(
        "button:not([disabled]), summary",
      );
      if (!focusable || focusable.length === 0) {
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      window.cancelAnimationFrame(frame);
      document.removeEventListener("keydown", handleKeyDown);
      previousFocus?.focus();
    };
  }, [open]);

  if (!open) {
    return null;
  }
  return (
    <div className="modal-backdrop activity-modal-backdrop" role="presentation">
      <section aria-labelledby="activity-dialog-title" aria-modal="true" className="activity-dialog" ref={dialog} role="dialog">
        <header className="activity-dialog-heading">
          <div>
            <p>当前会话</p>
            <h2 id="activity-dialog-title">执行记录</h2>
          </div>
          <button aria-label="关闭执行记录" className="activity-dialog-close" ref={closeButton} type="button" onClick={onClose}>
            <Icon name="close" size={18} />
          </button>
        </header>
        <p className="activity-dialog-description">请求与响应仅在此本地详情中按需读取；认证信息和常见密钥会被脱敏，模型原始推理不会记录。</p>
        {traceItems.length === 0 ? (
          <p className="activity-dialog-empty">当前会话尚未产生可查看的运行追踪。</p>
        ) : (
          <div className="activity-dialog-body">
            <nav aria-label="运行追踪列表" className="activity-event-menu">
              <div className="trace-filters" role="group" aria-label="追踪筛选">
                {(["all", "model", "tool", "security", "error"] as const).map((name) => (
                  <button
                    aria-pressed={filter === name}
                    className={filter === name ? "active" : ""}
                    key={name}
                    type="button"
                    onClick={() => setFilter(name)}
                  >
                    {traceFilterLabel(name)}
                  </button>
                ))}
              </div>
              {visibleItems.map(({ turnId, item }) => (
                <button
                  aria-pressed={turnId === selectedItem?.turnId && item.item_id === selectedItem.item.item_id}
                  className={`activity-event-option trace-option trace-${item.kind} ${item.parent_id ? "trace-child" : ""} ${turnId === selectedItem?.turnId && item.item_id === selectedItem.item.item_id ? "selected" : ""}`}
                  key={`${turnId}-${item.item_id}`}
                  type="button"
                  onClick={() => setSelected({ turnId, itemId: item.item_id })}
                >
                  <span className={`trace-kind trace-kind-${item.kind} trace-status-${item.status}`} aria-hidden="true" />
                  <span className="activity-entry-copy">
                    <strong>第 {turnId} 轮 · {item.title}</strong>
                    <small>{item.summary}{item.duration_ms !== null ? ` · ${item.duration_ms} ms` : ""}</small>
                  </span>
                  <time>{formatTime(item.started_at)}</time>
                </button>
              ))}
            </nav>
            {selectedItem && (
              <TraceDetail detail={detail} error={detailError} item={selectedItem.item} loading={loadingDetail} />
            )}
          </div>
        )}
      </section>
    </div>
  );
}

/** Render request and response bodies only after the user opened a trace item. */
function TraceDetail({
  detail,
  error,
  item,
  loading,
}: {
  detail: TraceItemDetail | null;
  error: string | null;
  item: TraceItem;
  loading: boolean;
}) {
  const visible = detail ?? item;
  const context = traceContext(detail?.attributes);
  const attributes = withoutTraceContext(detail?.attributes ?? {});
  return (
    <section aria-label="选中事件详情" className="activity-event-detail">
      <header>
        <div>
          <p>{visible.kind === "model" ? "模型交换" : "工具执行"}</p>
          <h3>{visible.title}</h3>
          <span>{visible.summary}</span>
        </div>
        <time>{formatTime(visible.started_at)}</time>
      </header>
      {loading && <p className="activity-detail-empty">正在读取本地详情…</p>}
      {error && <p className="activity-detail-empty">{error}</p>}
      {!loading && !error && detail && <>
        {context && <TracePayload title="上下文构建" value={context} />}
        <TracePayload title="摘要" value={attributes} />
        {"request" in detail && <TracePayload title={detail.kind === "model" ? "模型请求" : "工具参数"} value={detail.request} />}
        {"response" in detail && <TracePayload title={detail.kind === "model" ? "模型响应" : "工具结果"} value={detail.response} />}
      </>}
    </section>
  );
}

/** Read the optional selection facts without trusting arbitrary trace payload shapes. */
function traceContext(attributes: Record<string, unknown> | undefined): Record<string, unknown> | null {
  const value = attributes?.context;
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

/** Keep context selection in its own panel instead of duplicating it in the generic summary. */
function withoutTraceContext(attributes: Record<string, unknown>): Record<string, unknown> {
  const { context: _, ...remaining } = attributes;
  return remaining;
}

/** Keep JSON readable without allowing a detail payload to affect the DOM as HTML. */
function TracePayload({ title, value }: { title: string; value: unknown }) {
  return (
    <section className="trace-payload">
      <h4>{title}</h4>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </section>
  );
}

/** Render one compact trace row in the inspector. */
function TraceEntry({ item }: { item: TraceItem }) {
  return (
    <div className="activity-entry trace-entry">
      <span className={`trace-kind trace-kind-${item.kind} trace-status-${item.status}`} aria-hidden="true" />
      <span className="activity-entry-copy">
        <strong>{item.title}</strong>
        <small>{item.summary}{item.duration_ms !== null ? ` · ${item.duration_ms} ms` : ""}</small>
      </span>
    </div>
  );
}

/** Match the tree filter without duplicating security entries in the trace. */
function matchesTraceFilter(item: TraceItem, filter: TraceFilter): boolean {
  if (filter === "all") {
    return true;
  }
  if (filter === "error") {
    return item.status === "failed";
  }
  if (filter === "security") {
    const decision = item.attributes.decision;
    return typeof decision === "string" && decision !== "allow";
  }
  return item.kind === filter;
}

/** Keep filter names independent from persisted trace values. */
function traceFilterLabel(filter: TraceFilter): string {
  return { all: "全部", model: "模型", tool: "工具", security: "安全", error: "错误" }[filter];
}

/** Avoid treating future event types as chat content by default. */
function isMessageEvent(event: ConversationEvent): boolean {
  return MESSAGE_EVENTS.has(event.event);
}

/** Ignore private trace refresh signals when deriving the one-line live status. */
function isActivityEvent(event: ConversationEvent): boolean {
  return event.event !== "conversation_created" && event.event !== "trace_updated" && !isMessageEvent(event);
}

/** Derive one short in-place processing message from the latest safe event. */
function activityStatus(event: ConversationEvent): string {
  const tool = stringDetail(event, "tool");
  if (event.event === "tool_requested") {
    return tool ? `正在执行 ${tool}` : "正在准备调用工具";
  }
  if (event.event === "tool_finished") {
    return tool ? `已完成 ${tool}，正在继续处理` : "工具已完成，正在继续处理";
  }
  if (event.event === "approval_required") {
    return "正在等待你确认高风险操作";
  }
  if (event.event === "approval_resolved") {
    return "正在处理确认结果";
  }
  if (event.event === "turn_cancel_requested") {
    return "正在取消当前任务";
  }
  if (event.event === "agent_finished") {
    return "正在整理执行结果";
  }
  return eventLabel(event.event);
}

interface ComposerProps {
  value: string;
  disabled: boolean;
  focusKey: string | null;
  running: boolean;
  cancelling: boolean;
  submitting: boolean;
  onChange: (value: string) => void;
  onCancel: () => void;
  onSubmit: () => void;
}

/** Provide keyboard-friendly task entry without allowing a blank submission. */
export function Composer({
  value,
  disabled,
  focusKey,
  running,
  cancelling,
  submitting,
  onChange,
  onCancel,
  onSubmit,
}: ComposerProps) {
  const textarea = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const element = textarea.current;
    if (!element) {
      return;
    }
    // Let the field follow its content until it reaches a compact chat-sized
    // limit, then keep additional text inside the textarea's own scrollbar.
    element.style.height = "0px";
    const height = Math.min(element.scrollHeight, 216);
    element.style.height = `${height}px`;
    element.style.overflowY = element.scrollHeight > height ? "auto" : "hidden";
  }, [value]);

  useEffect(() => {
    if (!disabled) {
      textarea.current?.focus();
    }
  }, [disabled, focusKey]);

  const submit = (event: FormEvent) => {
    event.preventDefault();
    onSubmit();
  };
  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (!event.nativeEvent.isComposing && event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSubmit();
    }
  };

  return (
    <form className="composer" onSubmit={submit}>
      <div className="composer-field">
        <textarea
          aria-label="任务消息"
          disabled={disabled}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder={disabled ? "请先新建或选择一个可用会话" : "给 Coding Agent 发送消息"}
          ref={textarea}
          rows={1}
          value={value}
        />
        {running && (
          <button
            aria-label="停止生成"
            className="composer-cancel"
            disabled={cancelling}
            type="button"
            onClick={onCancel}
          >
            <Icon name="pause" size={15} />
            <span>{cancelling ? "正在停止" : "停止生成"}</span>
          </button>
        )}
        <button
          aria-label="发送消息"
          className="composer-send"
          disabled={disabled || submitting || !value.trim()}
          type="submit"
        >
          <Icon name="send" size={17} />
        </button>
      </div>
      <div className="composer-footer">
        <span>Enter 发送 · Shift + Enter 换行</span>
        {submitting && <span>正在发送…</span>}
      </div>
    </form>
  );
}

interface InspectorProps {
  config: PublicConfig | null;
  traces: TurnTrace[];
  session: ConversationSnapshot | null;
  connection: "connecting" | "connected" | "reconnecting" | "error";
  onOpenActivity: () => void;
  open: boolean;
}

/** Present session limits, safe execution details, and result summaries without LLM history. */
export function Inspector({ config, traces, session, connection, onOpenActivity, open }: InspectorProps) {
  return (
    <aside aria-hidden={!open} className={`inspector ${open ? "open" : "collapsed"}`} inert={!open}>
      <div className="inspector-title-row">
        <h2>运行信息</h2>
        <span className={`connection connection-${connection}`}>
          <span />{connection === "connected" ? "已连接" : connection === "reconnecting" ? "重连中" : connection === "error" ? "连接异常" : "连接中"}
        </span>
      </div>
      {session ? (
        <>
          <section className="status-overview">
            <div>
              <span>当前状态</span>
              <strong>{stateLabel(session.state)}</strong>
            </div>
            <StateDot state={session.state} />
          </section>
          <dl className="metrics">
            <div><dt>消息队列</dt><dd>{session.queued_messages}</dd></div>
            <div><dt>原始历史</dt><dd>{session.history_items} 项</dd></div>
            <div><dt>请求上下文</dt><dd>{formatTokenCount(session.context.estimated_input_tokens)} / {formatTokenCount(session.context.input_budget_tokens)}</dd></div>
            <div><dt>本地摘要</dt><dd>v{session.context.summary_version ?? 0} · {session.context.artifact_count ?? 0} 工件</dd></div>
            <div><dt>最近结果</dt><dd>{session.latest_status ?? "尚未执行"}</dd></div>
          </dl>
          <ActivityLog traces={traces} onOpen={onOpenActivity} />
          {session.summary && <Summary summary={session.summary} />}
        </>
      ) : (
        <p className="inspector-empty">选择或创建会话后，这里会显示运行状态与结果摘要。</p>
      )}
      <section className="configuration">
        <h3>本地连接</h3>
        <dl>
          <div><dt>服务商</dt><dd>{config?.provider ?? "未配置"}</dd></div>
          <div><dt>模型</dt><dd title={config?.model ?? undefined}>{config?.model ?? "未配置"}</dd></div>
        </dl>
      </section>
    </aside>
  );
}

/** Keep large token values readable in the compact inspector. */
function formatTokenCount(value: number | undefined): string {
  if (typeof value !== "number" || value <= 0) {
    return "尚未估算";
  }
  return value >= 1_000 ? `${(value / 1_000).toFixed(value >= 100_000 ? 0 : 1)}k` : String(value);
}

/** Render the bounded end-of-turn summary produced by the backend. */
function Summary({ summary }: { summary: TaskSummary }) {
  return (
    <section className="summary-panel">
      <h3>本轮摘要</h3>
      <div className="summary-grid">
        <div><strong>{summary.modified_file_count}</strong><span>修改文件</span></div>
        <div><strong>{summary.tests === "passed" ? "通过" : summary.tests === "failed" ? "失败" : "未运行"}</strong><span>测试</span></div>
        <div><strong>{summary.blocked_actions}</strong><span>已拦截</span></div>
      </div>
      {summary.modified_files.length > 0 && (
        <ul className="file-list">
          {summary.modified_files.map((path) => <li key={path}>{path}</li>)}
        </ul>
      )}
    </section>
  );
}

interface ApprovalDialogProps {
  approval: ApprovalRequest | null;
  resolving: boolean;
  onResolve: (approved: boolean) => void;
}

/** Force an explicit decision before a high-risk tool call can proceed. */
export function ApprovalDialog({ approval, resolving, onResolve }: ApprovalDialogProps) {
  const dialog = useRef<HTMLElement>(null);
  const rejectButton = useRef<HTMLButtonElement>(null);
  const resolveRef = useRef(onResolve);
  const resolvingRef = useRef(resolving);
  resolveRef.current = onResolve;
  resolvingRef.current = resolving;

  useEffect(() => {
    if (!approval) {
      return;
    }
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const focusDialog = () => rejectButton.current?.focus();
    const frame = window.requestAnimationFrame(focusDialog);
    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape" && !resolvingRef.current) {
        event.preventDefault();
        resolveRef.current(false);
        return;
      }
      if (event.key !== "Tab") {
        return;
      }
      const focusable = dialog.current?.querySelectorAll<HTMLElement>(
        "button:not([disabled]), summary",
      );
      if (!focusable || focusable.length === 0) {
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      window.cancelAnimationFrame(frame);
      document.removeEventListener("keydown", handleKeyDown);
      previousFocus?.focus();
    };
  }, [approval?.approvalId]);

  if (!approval) {
    return null;
  }
  return (
    <div className="modal-backdrop" role="presentation">
      <section aria-describedby="approval-description" aria-modal="true" className="approval-dialog" ref={dialog} role="dialog">
        <div className="approval-heading">
          <span className="approval-warning">!</span>
          <div><p>需要你的确认</p><h2>高风险工具调用</h2></div>
        </div>
        <p id="approval-description">该操作不会自动执行。请核对工具、参数摘要和策略原因。</p>
        <p className="dialog-hint">按 Esc 将拒绝本次操作。</p>
        <dl className="approval-facts">
          <div><dt>工具</dt><dd>{approval.tool}</dd></div>
          <div><dt>风险等级</dt><dd className="risk-high">{approval.risk}</dd></div>
          <div><dt>策略原因</dt><dd>{approval.reason}</dd></div>
          <div><dt>等待时限</dt><dd>{approval.timeoutSeconds} 秒</dd></div>
        </dl>
        <details className="approval-arguments" open>
          <summary>安全参数摘要</summary>
          <pre>{formatDetails(approval.arguments)}</pre>
        </details>
        <div className="approval-actions">
          <button className="secondary-button" disabled={resolving} ref={rejectButton} type="button" onClick={() => onResolve(false)}>拒绝</button>
          <button className="danger-button" disabled={resolving} type="button" onClick={() => onResolve(true)}>
            {resolving ? "正在提交" : "批准执行"}
          </button>
        </div>
      </section>
    </div>
  );
}
