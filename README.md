# ChefAI · 垂直领域 RAG 知识库问答系统

ChefAI 是一个面向食谱/烹饪知识场景的 RAG 知识库问答系统，基于 **FastAPI + Streamlit + ChromaDB** 实现多格式资料入库、混合检索、Rerank、多轮 Query Rewrite、低置信度拒答与引用溯源。

> 项目定位：AI 应用开发 / RAG 工程实践项目。重点不在“调 API 生成回答”，而在于从文档入库、检索、生成、拒答到评估复测的完整闭环。

---

## 技术栈

- **后端**：Python, FastAPI, Pydantic
- **前端**：Streamlit
- **RAG**：LangChain, ChromaDB, BM25, Rerank, Parent-Child Chunking
- **模型服务**：GLM / Embedding / GLM-4V
- **工程化**：Docker, docker-compose, Pytest

---

## 核心功能

- 多格式资料入库：PDF、Word、Excel、CSV、TXT/Markdown、图片、网页 URL
- 文档切分：Parent-Child Chunking，small chunk 检索，parent chunk 回溯生成上下文
- 检索链路：向量检索 + BM25 混合检索 + Rerank 重排序
- 多轮问答：Query Rewrite 处理“它 / 这个 / 这道菜”等指代类追问
- 安全与可靠性：低置信度拒答、URL 安全校验、文件类型校验
- 可追溯回答：展示引用来源，降低资料外扩展

---

## RAG 评估与优化

项目构建了 **50 条中文 QA 评估集**，覆盖食材查询、步骤问答、细节判断、相似菜品对比、多轮追问和知识库外拒答。

| 指标 | 优化前 | 优化后 |
|---|---:|---:|
| Answer Correctness | 0.84 | 0.89 |
| Faithfulness | 0.47 | 0.91 |
| Refusal / Negation Correctness | 0.60 | 1.00 |

优化内容：

- 收紧生成 Prompt，限制资料外扩展
- 增加“根据资料 / 有没有提到 / 是否必须”类问题的严格判断规则
- 调整 Query Rewrite，仅对指代类问题进行改写
- 修复 Rerank 不可用时导致过度拒答的问题

评估文件位于：

```text
eval/
  eval_dataset.jsonl
  round1_formal_eval_results.csv
  round2_formal_eval_results.csv
  round1_scoring_summary.md
  round2_scoring_summary.md
```

---

## 项目截图

首页 / 上传 / 问答截图位于 `docs/` 目录：

```text
docs/homepage.png
docs/upload.png
docs/chat_demo.png
```

---

## 本地启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，并填写自己的 API Key：

```bash
cp .env.example .env
```

Windows PowerShell 可使用：

```powershell
copy .env.example .env
```

### 3. 启动后端

```bash
uvicorn backend.main:app --reload
```

### 4. 启动前端

```bash
streamlit run frontend/streamlit_app.py
```

---

## Docker 启动

```bash
docker compose up --build
```

前端默认访问：

```text
http://127.0.0.1:8501
```

后端默认访问：

```text
http://127.0.0.1:8000
```

---

## 测试

```bash
pytest
```

当前包含文件类型校验、URL 安全校验、检索置信度等单元测试。

---

## 注意事项

- `.env`、`chroma_db/`、`temp_upload/` 等本地敏感或临时文件不会上传 GitHub。
- 本项目的评估集规模较小，适合作为 RAG 应用开发实习项目展示，不应包装为生产级企业系统。
- “怎么做”类短 query 在部分场景下可能优先召回结构化食材表，后续可通过 intent routing 或 source-type weighting 继续优化。
