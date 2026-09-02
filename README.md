# Coding Agent

一个本地运行的编程智能体：LLM 只负责分析任务和提出工具调用，文件访问、命令执行与高风险操作均由确定性策略控制。

## 核心能力

- 读取、修改 workspace 内的文件，运行本地命令；
- 支持单任务、多轮终端对话和本地 Web 会话；
- 对话历史、上下文摘要、审计记录和 Trace 均保存在本地；
- 路径越界、敏感文件和关键危险命令直接拒绝；
- 高风险调用需逐次人工批准，命令默认运行在无网络的 Bubblewrap 沙箱中；
- Web 页面可查看会话、执行进度、脱敏后的模型请求/响应及工具详情。

## 安装

要求 Python 3.10+；需要执行本地命令时还需要 Linux、`bwrap` 和可用的用户命名空间。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

在项目根目录创建未提交的 `.env`：

```dotenv
CODING_AGENT_PROVIDER="responses"
CODING_AGENT_API_KEY="your_api_key"
CODING_AGENT_BASE_URL="https://llm.example.com/v1"
CODING_AGENT_MODEL="compatible-model"
CODING_AGENT_MAX_OUTPUT_TOKENS="128000"
```

支持 `openai`、`deepseek` 和通用 `responses` 适配器；服务必须兼容 Responses API 和自定义函数工具调用。API Key 只从本地 `.env` 读取。

## 使用

执行单个任务：

```bash
coding-agent --workspace /path/to/project --task "your task"
```

启动终端多轮对话：

```bash
coding-agent chat --workspace /path/to/project
```

启动本地 Web 页面：

```bash
coding-agent serve --workspace /path/to/project
```

打开 `http://127.0.0.1:8765`。服务仅监听 loopback 地址；同一 workspace 的 Agent 轮次串行执行。会话数据保存在 workspace 的 `.agent/conversations/`。

## 安全边界

每次工具调用都会得到 `ALLOW`、`REQUIRE_APPROVAL` 或 `DENY` 决策：

- `ALLOW`：常规 workspace 读取、修改和测试；
- `REQUIRE_APPROVAL`：删除、依赖管理、网络能力、破坏性 Git、解释器脚本及未知命令；
- `DENY`：workspace 越界、符号链接逃逸、`.env`、私钥、凭据目录、`sudo`、关机和格式化等操作。

命令在 Bubblewrap 沙箱中执行：网络关闭、环境变量最小化、HOME/TMP 隔离，并遮蔽 workspace 内的敏感路径。沙箱不可用时命令失败关闭，不会回退到宿主机执行。

## 可观测性与上下文

任务审计日志写入 `.agent/logs/`，记录工具、策略、风险、耗时和结果摘要，不保存文件正文、补丁正文或工具输出。Web 会话额外保存按轮次组织的 Trace；模型请求和响应仅在用户主动打开详情时以脱敏、截断后的形式提供。

完整对话转录由客户端本地维护，不使用服务端会话。长会话保留原始历史，同时使用本地摘要和工具输出工件构造受预算约束的模型上下文。

## 验证与开发

运行完整测试：

```bash
python -m unittest discover -s tests -t .
```

前端源码位于 `frontend/`。修改前端后，使用 Node.js 20.19+ 重新构建并提交生成的静态资源：

```bash
cd frontend
npm ci
npm run build
```
