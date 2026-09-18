
# Multi-Tool AI Agent

一个基于 Python、LangChain 和 LangGraph 开发的多工具 AI Agent 项目。

项目支持天气查询、公司知识库问答和通用知识问答，用于学习和实践 AI Agent 的工具调用、RAG 检索和工作流编排。

## 项目功能

### 1. 天气查询

- 通过天气 API 查询指定城市的天气。
- 获取温度、湿度和天气状况。
- 对网络请求失败和异常天气数据进行基础处理。
- 对无效城市提供友好的错误提示。

### 2. 公司知识库问答

- 使用 Chroma 向量数据库存储公司知识。
- 使用 RAG 技术检索相关信息。
- 根据用户问题查询公司知识库。
- 知识库没有相关信息时，明确提示用户，减少虚构内容。

### 3. 多工具协作

- 使用 LangGraph 编排 Agent 工作流。
- 根据用户问题选择合适的工具。
- 支持一次对话中处理多个不同任务。
- 将多个工具的结果整合为最终回答。

### 4. 对话状态管理

- 使用 SQLite 保存对话检查点。
- 支持基于会话的对话状态管理。
- 包含基础的历史消息摘要处理。

## 技术栈

- Python
- LangChain
- LangGraph
- Chroma
- SQLite
- Requests
- python-dotenv
- OpenAI 兼容 API

## 项目结构

```text
multi-tool-agent/
├── data/
│   └── company.txt
├── .env.example
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```

以下文件或目录在程序运行过程中可能生成，但不会提交到 GitHub：

```text
.venv/
chroma_db/
checkpoints.sqlite
.env
```

## 环境要求

- Python 3.13+
- 可用的 OpenAI 兼容 API
- 网络连接

## 安装与配置

### 1. 克隆项目

```bash
git clone https://github.com/sududuo/multi-tool-agent.git
cd multi-tool-agent
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
```

Windows 激活虚拟环境：

```powershell
.venv\Scripts\activate
```

### 3. 安装项目依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

复制 `.env.example`，在项目根目录创建 `.env` 文件。

根据自己的 API 服务商填写配置：

```env
DATA_BASE_URL=你的API地址
DATAOPAI_API_KEY=你的API密钥
DATA_MODEL=你的模型名称
```

请勿将真实 API Key 上传到 GitHub。

## 运行项目

在项目根目录执行：

```bash
python main.py
```

## 使用示例

### 天气查询

```text
查询北京天气
```

### 公司知识库查询

```text
公司的总部在哪里？
```

```text
公司主要从事什么业务？
```

### 多工具协作

```text
请告诉我北京今天的天气，以及公司的总部在哪里？
```

### 通用知识问答

```text
请介绍一下法国的历史。
```

## 当前项目状态

当前版本为 **V1**，主要用于学习和实践 AI Agent 的基础开发流程。

目前已经实现：

- 基础 Agent 对话
- 天气工具调用
- 公司知识库 RAG 检索
- 多工具协作
- 基础异常处理
- LangGraph 工作流
- SQLite 对话检查点

## 后续开发计划

- [ ] 优化项目代码结构
- [ ] 完善工具异常处理
- [ ] 增加单元测试和功能测试
- [ ] 增加日志和可观测性
- [ ] 优化 RAG 检索效果
- [ ] 使用 FastAPI 提供服务接口
- [ ] 增加数据库功能
- [ ] 使用 Docker 部署项目
- [ ] 完善 Agent 评估机制

## 学习目标

通过持续迭代本项目，学习以下技术：

- Python 工程化开发
- LLM 工具调用
- RAG 知识库
- LangGraph Agent 工作流
- API 服务开发
- 数据库与项目部署
- Agent 测试与评估

## 免责声明

本项目主要用于个人学习和技术实践。
