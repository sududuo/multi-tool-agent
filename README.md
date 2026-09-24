# Multi-Tool AI Agent

一个基于 **Python、LangChain 和 LangGraph** 开发的多工具 AI Agent 项目。

项目目前支持天气查询、公司知识库 RAG 问答、通用知识问答、多轮对话状态管理，并使用 LangGraph 编排 Agent 工作流。

本项目主要用于学习和实践：

- LLM Tool Calling
- LangGraph 工作流
- RAG 检索
- Retriever + Reranker 两阶段检索
- 对话状态与持久化
- Python 项目模块化
- 日志与异常处理

---

## 项目功能

### 1. 天气查询

通过天气 API 查询指定城市的实时天气信息。

支持：

- 温度查询
- 湿度查询
- 天气状况查询
- 网络异常处理
- 返回数据异常处理

---

### 2. 公司知识库 RAG

使用 Chroma 构建本地向量知识库。

当前检索流程：

```text
用户问题
   ↓
Embedding
   ↓
Chroma Retriever
   ↓
Top 10 候选文档
   ↓
Qwen Reranker
   ↓
Top 3 文档
   ↓
LLM 生成最终回答
```

目前支持：

- Chroma 向量数据库
- 文档 Embedding
- Retriever Top-K 检索
- 专用 Reranker 重排序
- Reranker API 状态检查
- Reranker 失败自动降级
- Retriever 延迟初始化
- Retriever 缓存复用
- 知识库未初始化时友好提示

知识库构建与查询流程已经分离：

```text
scripts/build_index.py
→ 负责构建知识库

tools/rag.py
→ 负责知识库查询
```

---

### 3. 多工具 Agent

当前 Agent 可以根据用户问题自动选择工具。

已注册工具：

```text
get_weather
search_company_knowledge
```

例如：

```text
北京天气怎么样？
```

Agent 会调用天气工具。

```text
公司的总部在哪里？
```

Agent 会调用公司知识库工具。

普通知识问题则可以由 LLM 直接回答。

---

### 4. LangGraph 工作流

项目使用 LangGraph 编排 Agent。

当前主要工作流：

```text
START
  ↓
summary_router
  ↓
summarize / agent
  ↓
agent
  ↓
tools / END
  ↓
agent
```

工具执行完成后重新返回 Agent，由 LLM 根据工具结果生成最终回答。

---

### 5. 多轮对话与状态管理

使用 SQLite Checkpointer 保存 LangGraph 对话状态。

用户可以通过：

```text
thread_id
```

区分不同会话。

例如：

```text
你：我叫小明
AI：你好，小明！

你：我刚才说我叫什么？
AI：你刚才说你叫小明。
```

---

### 6. 对话上下文管理

项目包含基础的上下文管理机制：

- Token 数量统计
- 最近完整对话轮次裁剪
- 历史消息摘要
- 长对话上下文压缩
- Token 阈值触发摘要

用于避免长时间对话导致上下文无限增长。

---

### 7. 日志与异常处理

目前已经对部分关键流程加入日志：

- 天气查询
- RAG 检索
- Reranker 调用
- Retriever / Reranker 降级
- API 请求异常

Reranker 可以记录候选数量、返回数量以及相关性评分，方便后续进行 RAG Eval 和调试。

---

# 技术栈

- Python
- LangChain
- LangGraph
- Chroma
- SQLite
- DashScope
- Requests
- python-dotenv
- OpenAI Compatible API

---

# 项目结构

```text
multi-tool-agent/
│
├── agent/
│   ├── context.py
│   ├── graph.py
│   ├── llm.py
│   ├── memory.py
│   ├── model.py
│   ├── nodes.py
│   ├── state.py
│   └── summary.py
│
├── tools/
│   ├── rag.py
│   └── weather.py
│
├── services/
│   └── embeddings.py
│
├── scripts/
│   └── build_index.py
│
├── data/
│   └── company.txt
│
├── config.py
├── main.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

各模块主要职责：

```text
agent/llm.py
→ 创建基础 LLM

agent/model.py
→ 注册工具并创建 llm_with_tools

agent/state.py
→ 定义 LangGraph State

agent/context.py
→ Token 统计与上下文裁剪

agent/summary.py
→ 对话摘要与摘要路由

agent/nodes.py
→ Agent 节点

agent/memory.py
→ SQLite Checkpointer

agent/graph.py
→ LangGraph 工作流组装

services/embeddings.py
→ 创建共享 Embedding 模型

scripts/build_index.py
→ 构建 Chroma 向量知识库

tools/rag.py
→ RAG Retriever + Reranker

tools/weather.py
→ 天气查询工具
```

---

# 环境要求

推荐：

```text
Python 3.13+
```

同时需要：

- 可访问的模型 API
- DashScope API Key
- 网络连接

---

# 安装与配置

## 1. 克隆项目

```bash
git clone https://github.com/sududuo/multi-tool-agent.git
cd multi-tool-agent
```

---

## 2. 创建虚拟环境

```bash
python -m venv .venv
```

Windows：

```powershell
.venv\Scripts\activate
```

---

## 3. 安装依赖

```bash
pip install -r requirements.txt
```

---

## 4. 配置环境变量

复制：

```text
.env.example
```

并在项目根目录创建：

```text
.env
```

配置示例：

```env
DATA_BASE_URL=https://你的OpenAI兼容接口地址

DASHSCOPE_BASE_URL=https://你的DashScope原生接口地址

DASHSCOPE_API_KEY=你的API密钥

DATA_MODEL=你的聊天模型名称
```

请勿将真实 API Key 上传到 GitHub。

---

# 构建知识库

首次运行 RAG 前，需要先构建公司知识库。

在项目根目录执行：

```bash
python -m scripts.build_index
```

程序会：

```text
读取 data/company.txt
↓
文本切分
↓
Embedding
↓
创建 Chroma 向量数据库
↓
保存到 chroma_db/
```

如果已经存在旧知识库，当前构建脚本会删除旧库后重新构建。

修改：

```text
data/company.txt
```

后，也可以重新执行该命令更新知识库。

---

# 运行 Agent

知识库构建完成后：

```bash
python main.py
```

程序会要求输入：

```text
请输入thread_id：
```

例如：

```text
请输入thread_id：1
```

然后即可开始对话。

输入：

```text
q
```

或：

```text
quit
```

退出程序。

---

# 使用示例

## 普通知识问答

```text
你：法国的首都是哪里？

AI：法国的首都是巴黎。
```

---

## 天气查询

```text
你：北京天气怎么样？
```

Agent 会自动调用：

```text
get_weather
```

---

## 公司知识库查询

```text
你：公司总部在哪里？
```

Agent 会自动调用：

```text
search_company_knowledge
```

并经过：

```text
Retriever
→ Reranker
→ LLM
```

生成回答。

---

## 多轮对话

```text
你：我叫小明

你：我刚才说我叫什么？
```

通过 LangGraph Checkpointer 保留当前 `thread_id` 对应的对话状态。

---

# 本地生成文件

以下文件或目录不会提交到 GitHub：

```text
.env
.venv/
chroma_db/
checkpoints.sqlite
__pycache__/
.idea/
```

其中：

```text
chroma_db/
```

可以通过：

```bash
python -m scripts.build_index
```

重新生成。

---

# 当前项目状态

当前版本已经完成：

- [x] 基础 Agent 对话
- [x] Tool Calling
- [x] 天气工具
- [x] 公司知识库 RAG
- [x] Chroma Retriever
- [x] 专用 Reranker
- [x] Reranker 失败降级
- [x] LangGraph 工作流
- [x] SQLite Checkpointer
- [x] 多轮对话
- [x] Token 上下文裁剪
- [x] 历史对话摘要
- [x] 基础日志
- [x] 基础异常处理
- [x] 项目模块化重构
- [x] 知识库构建与查询分离
- [x] Retriever 延迟初始化与缓存

---

# 后续计划

- [ ] RAG Eval
- [ ] 优化 Chunk Size / Chunk Overlap
- [ ] 对比不同 Retriever Top-K 参数
- [ ] 对比 Rerank Top-N 参数
- [ ] 增加 pytest 单元测试
- [ ] 增加 Agent Eval
- [ ] 增加 HITL
- [ ] 使用 FastAPI 提供 API 服务
- [ ] PostgreSQL 数据库
- [ ] Async / 异步处理
- [ ] Docker
- [ ] Linux 部署
- [ ] CI/CD
- [ ] Observability
- [ ] MCP

---

# 学习目标

通过持续迭代本项目，学习：

- Python 工程化开发
- LLM Tool Calling
- Structured Agent Workflow
- RAG
- Retriever / Reranker
- LangGraph
- 对话状态管理
- Agent Memory
- 日志与异常处理
- Agent Evaluation
- FastAPI
- 数据库
- Docker 与部署

项目会随着学习过程持续更新。

---

# 免责声明

本项目主要用于个人学习和 AI Agent 工程实践。