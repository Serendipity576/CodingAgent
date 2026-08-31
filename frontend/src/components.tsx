// Presentational components deliberately receive safe API data rather than LLM history.
import { useEffect, useRef, type FormEvent, type KeyboardEvent, type ReactNode } from "react";

import {
  eventLabel,
  formatDetails,
  formatTime,
  numberDetail,
  stateLabel,
  stringDetail,
} from "./presentation";
import type {
  ApprovalRequest,
  ConversationEvent,
  ConversationSnapshot,
  PublicConfig,
  TaskSummary,
} from "./types";

type IconName = "add" | "arrow" | "close" | "code" | "delete" | "history" | "panel" | "pause" | "send";

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
    panel: <path d="M4 5h16v14H4zm10 0v14M7 9h4m-4 3h4" />,
    pause: <path d="M8 5v14M16 5v14" />,
    send: <path d="m21 3-7.5 18-3.8-7.7L3 9.5 21 3Zm-11.4 10.3L15 9" />,
  };

  return <svg {...common}>{paths[name]}</svg>;
}

interface SidebarProps {
  sessions: ConversationSnapshot[];
  activeId: string | null;
  workspace: string | null;
  creating: boolean;
  deletingId: string | null;
  onCreate: () => void;
  onDelete: (conversationId: string) => void;
  onSelect: (conversationId: string) => void;
}

/** Display local sessions and make the active session unambiguous. */
export function Sidebar({
  sessions,
  activeId,
  workspace,
  creating,
  deletingId,
  onCreate,
  onDelete,
  onSelect,
}: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark"><Icon name="code" size={20} /></span>
        <span className="brand-copy"><strong>Coding Agent</strong><small>LOCAL WORKBENCH</small></span>
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
                    <span className="conversation-name">会话 {session.conversation_id.slice(0, 6)}</span>
                    <StateDot state={session.state} />
                  </span>
                  <span className="conversation-item-bottom">
                    {stateLabel(session.state)} · {session.turn_count}/{session.max_turns} 轮
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
      <p className="local-only">仅本机访问<br />会话数据保留在 workspace</p>
    </aside>
  );
}

/** Show a compact color-coded lifecycle state indicator. */
function StateDot({ state }: { state: ConversationSnapshot["state"] }) {
  return <span className={`state-dot state-${state}`} title={stateLabel(state)} />;
}

interface TimelineProps {
  events: ConversationEvent[];
  session: ConversationSnapshot | null;
  onCreate: () => void;
}

/** Render safe runtime events as messages, tool cards, and concise notices. */
export function Timeline({ events, session, onCreate }: TimelineProps) {
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

  if (events.length === 0) {
    return (
      <div className="empty-stage compact">
        <span className="empty-stage-icon"><Icon name="code" size={25} /></span>
        <h1>可以开始了</h1>
        <p>描述要处理的编码任务，或继续上一轮的工作。</p>
      </div>
    );
  }

  return (
    <div className="timeline" aria-live="polite">
      {events.map((event) => <EventCard event={event} key={event.sequence} />)}
    </div>
  );
}

/** Select the most readable card treatment for each safe event category. */
function EventCard({ event }: { event: ConversationEvent }) {
  const text = stringDetail(event, "text");
  if (event.event === "user_message") {
    return <MessageCard role="user" label="你" time={formatTime(event.timestamp)}>{text ?? ""}</MessageCard>;
  }
  if (event.event === "assistant_message") {
    return <MessageCard role="assistant" label="Coding Agent" time={formatTime(event.timestamp)}>{text || "Agent 未返回文本。"}</MessageCard>;
  }
  if (event.event === "tool_requested" || event.event === "tool_finished") {
    return <ToolCard event={event} />;
  }
  return <SystemCard event={event} />;
}

/** Render one message body while preserving its line breaks without injecting HTML. */
function MessageCard({
  role,
  label,
  time,
  children,
}: {
  role: "user" | "assistant";
  label: string;
  time: string;
  children: string;
}) {
  return (
    <article className={`message-card ${role}`}>
      <header className="message-meta">
        <span>{label}</span>
        <time>{time}</time>
      </header>
      <p>{children}</p>
    </article>
  );
}

/** Render a tool call with detail kept collapsed until the user asks for it. */
function ToolCard({ event }: { event: ConversationEvent }) {
  const requested = event.event === "tool_requested";
  const tool = stringDetail(event, "tool") ?? "未知工具";
  const success = event.details.success === true;
  const duration = numberDetail(event, "duration_ms");
  const argumentsValue = event.details.arguments;
  const error = stringDetail(event, "error");
  const status = requested ? "执行中" : success ? "已完成" : "需要关注";

  return (
    <article className={`tool-card ${requested ? "pending" : success ? "success" : "failed"}`}>
      <div className="tool-card-heading">
        <span className="tool-card-icon"><Icon name="code" size={16} /></span>
        <div>
          <strong>{tool}</strong>
          <span>{requested ? "正在等待工具执行" : success ? "工具执行成功" : "工具未执行或执行失败"}</span>
        </div>
        <span className="tool-status">{status}</span>
        <time>{formatTime(event.timestamp)}</time>
      </div>
      {!requested && duration !== null && <p className="tool-result-meta">执行耗时 {duration} ms</p>}
      {error && <p className="tool-error">{error}</p>}
      {requested && argumentsValue !== undefined && (
        <details className="tool-details">
          <summary>查看安全参数摘要</summary>
          <pre>{formatDetails(argumentsValue)}</pre>
        </details>
      )}
    </article>
  );
}

/** Render non-message lifecycle events without flooding the conversation. */
function SystemCard({ event }: { event: ConversationEvent }) {
  const isAttention = [
    "approval_required",
    "conversation_interrupted",
    "conversation_limit_reached",
    "turn_cancel_requested",
  ].includes(event.event);
  const message = stringDetail(event, "message");
  const reason = stringDetail(event, "reason");
  return (
    <article className={`system-card ${isAttention ? "attention" : ""}`}>
      <span className="system-marker" />
      <div>
        <strong>{eventLabel(event.event)}</strong>
        {(message || reason) && <p>{message ?? reason}</p>}
      </div>
      <time>{formatTime(event.timestamp)}</time>
    </article>
  );
}

interface ComposerProps {
  value: string;
  disabled: boolean;
  submitting: boolean;
  onChange: (value: string) => void;
  onSubmit: () => void;
}

/** Provide keyboard-friendly task entry without allowing a blank submission. */
export function Composer({ value, disabled, submitting, onChange, onSubmit }: ComposerProps) {
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

  const submit = (event: FormEvent) => {
    event.preventDefault();
    onSubmit();
  };
  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
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
  session: ConversationSnapshot | null;
  connection: "connecting" | "connected" | "reconnecting" | "error";
  onCancel: () => void;
  cancelling: boolean;
  open: boolean;
}

/** Present session limits and result summaries without exposing LLM history. */
export function Inspector({ config, session, connection, onCancel, cancelling, open }: InspectorProps) {
  const isRunning = session?.state === "running";
  return (
    <aside aria-hidden={!open} className={`inspector ${open ? "open" : "collapsed"}`}>
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
            <div><dt>已处理轮次</dt><dd>{session.turn_count} / {session.max_turns}</dd></div>
            <div><dt>消息队列</dt><dd>{session.queued_messages}</dd></div>
            <div><dt>本地 history</dt><dd>{session.history_items} / {session.max_history_items}</dd></div>
            <div><dt>最近结果</dt><dd>{session.latest_status ?? "尚未执行"}</dd></div>
          </dl>
          {isRunning && (
            <button className="cancel-button" type="button" onClick={onCancel} disabled={cancelling}>
              <Icon name="pause" /> {cancelling ? "正在取消" : "取消当前任务"}
            </button>
          )}
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
          <div><dt>API Key</dt><dd>{config?.api_key_configured ? "已配置（不显示）" : "未配置"}</dd></div>
        </dl>
      </section>
    </aside>
  );
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
  if (!approval) {
    return null;
  }
  return (
    <div className="modal-backdrop" role="presentation">
      <section aria-describedby="approval-description" aria-modal="true" className="approval-dialog" role="dialog">
        <div className="approval-heading">
          <span className="approval-warning">!</span>
          <div><p>需要你的确认</p><h2>高风险工具调用</h2></div>
        </div>
        <p id="approval-description">该操作不会自动执行。请核对工具、参数摘要和策略原因。</p>
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
          <button className="secondary-button" disabled={resolving} type="button" onClick={() => onResolve(false)}>拒绝</button>
          <button className="danger-button" disabled={resolving} type="button" onClick={() => onResolve(true)}>
            {resolving ? "正在提交" : "批准执行"}
          </button>
        </div>
      </section>
    </div>
  );
}
