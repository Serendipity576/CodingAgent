const state = { sessionId: null, events: 0, source: null, approvalId: null };

const timeline = document.querySelector("#timeline");
const statusBox = document.querySelector("#status");
const form = document.querySelector("#message-form");
const messageInput = document.querySelector("#message");
const sendButton = document.querySelector("#send");
const cancelButton = document.querySelector("#cancel");
const approvalDialog = document.querySelector("#approval-dialog");

async function request(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
  return response.json();
}

function append(kind, text) {
  const item = document.createElement("article");
  item.className = `event ${kind}`;
  item.textContent = text;
  timeline.append(item);
  timeline.scrollTop = timeline.scrollHeight;
}

function showStatus(snapshot) {
  statusBox.textContent = JSON.stringify(snapshot, null, 2);
  document.querySelector("#session-title").textContent = snapshot.conversation_id || "尚未创建会话";
  cancelButton.disabled = snapshot.state !== "running";
}

async function refreshSessions() {
  const sessions = await request("/api/conversations");
  const list = document.querySelector("#conversations");
  list.replaceChildren();
  sessions.forEach((session) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `${session.conversation_id.slice(0, 8)} · ${session.state}`;
    button.onclick = () => selectSession(session.conversation_id);
    list.append(button);
  });
}

function connectEvents() {
  if (state.source) state.source.close();
  state.events = 0;
  state.source = new EventSource(`/api/conversations/${state.sessionId}/events`);
  state.source.onmessage = handleEvent;
  ["user_message", "assistant_message", "tool_requested", "tool_finished", "approval_required", "conversation_turn_finished", "agent_finished", "turn_cancel_requested"].forEach((name) => {
    state.source.addEventListener(name, handleEvent);
  });
}

function handleEvent(event) {
  const item = JSON.parse(event.data);
  state.events = item.sequence;
  const details = item.details || {};
  if (item.event === "user_message") append("user", details.text);
  else if (item.event === "assistant_message") append("assistant", details.text || "Agent 未返回文本。");
  else if (item.event === "tool_requested") append("tool", `请求工具：${details.tool}\n${JSON.stringify(details.arguments)}`);
  else if (item.event === "tool_finished") append("tool", `工具完成：${details.tool} · ${details.success ? "成功" : "失败"}`);
  else if (item.event === "approval_required") showApproval(details);
  else if (item.event === "conversation_turn_finished") showStatus({ conversation_id: state.sessionId, ...details });
  else if (item.event === "turn_cancel_requested") append("system", "已请求取消。");
}

function showApproval(details) {
  state.approvalId = details.approval_id;
  document.querySelector("#approval-details").textContent = JSON.stringify(details, null, 2);
  approvalDialog.showModal();
}

async function selectSession(id) {
  state.sessionId = id;
  timeline.replaceChildren();
  const snapshot = await request(`/api/conversations/${id}`);
  showStatus(snapshot);
  messageInput.disabled = false;
  sendButton.disabled = false;
  connectEvents();
}

async function createSession() {
  const session = await request("/api/conversations", { method: "POST" });
  await refreshSessions();
  await selectSession(session.conversation_id);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = messageInput.value.trim();
  if (!text || !state.sessionId) return;
  messageInput.value = "";
  await request(`/api/conversations/${state.sessionId}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  showStatus(await request(`/api/conversations/${state.sessionId}`));
});

cancelButton.onclick = async () => {
  if (!state.sessionId) return;
  await request(`/api/conversations/${state.sessionId}/cancel`, { method: "POST" });
};

document.querySelector("#approve").onclick = () => resolveApproval(true);
document.querySelector("#deny").onclick = () => resolveApproval(false);
async function resolveApproval(approved) {
  if (!state.sessionId || !state.approvalId) return;
  await request(`/api/conversations/${state.sessionId}/approvals/${state.approvalId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ approved }),
  });
  state.approvalId = null;
  approvalDialog.close();
}

document.querySelector("#new-conversation").onclick = createSession;

(async () => {
  const config = await request("/api/config");
  document.querySelector("#workspace").textContent = config.workspace;
  await refreshSessions();
  await createSession();
})().catch((error) => append("system", `页面初始化失败：${error.message}`));
