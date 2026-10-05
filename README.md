# 数码电商客服 RAG 项目（E-commerce Customer Service RAG）

![CI](https://github.com/vanvanovo/ecommerce-rag/actions/workflows/ci.yml/badge.svg)

> 面向数码电商客服场景的 RAG 知识问答系统：结构化建库 → 分层兜底问答 → 评测驱动迭代。
> 目标：让客服在文档中快速定位产品手册/售后政策/操作指南中的信息，统一答复口径。

## 技术选型（设计决策对照）

| 环节 | 选型 | 说明 |
|------|------|------|
| 文档转 Markdown | MinerU | 统一转 Markdown，保表格结构 |
| 存量 PDF 文本 | pdfplumber / PyMuPDF + OCR 兜底 | 商用 API 成本高 |
| 扫描件 OCR | PaddleOCR + YOLO 表格定位 + 跨页拼接 | 中文+版面+表格 |
| 图片 | 文字多→OCR；示意图→BLIP 生成描述 | 图片文案语义不丢 |
| 嵌入 | BGE-M3（稠密+稀疏双路） | 一模型两路，中文强 |
| 向量库 | Milvus（hybrid_search + WeightedRanker） | 原生混合检索 |
| 精排 | BGE-Reranker（仅精排 Top 少量） | cross-encoder 准但贵，末尾用 |
| 前置链路 | 正则 → Redis → MySQL BM25 → RAG 兜底 | 高频问题不配走全链路 |
| 意图分流 | BERT 三分类（闲聊/产品咨询/售后政策） | 小模型快、便宜；未训练时规则兜底 |
| 生成 | Qwen 流式 | 质量优先 + 流式体验 |

## 目录结构

```
ecommerce_rag/
├─ app/                    # FastAPI Web 层（REST + WebSocket 流式）
├─ base/                   # config.py 配置 / logger.py 日志
├─ core/                   # vector_store(BGE-M3+Milvus+rerank)、rag_system(生成)
├─ retrieval/              # 快路径：正则 → Redis → MySQL BM25
├─ routing/                # 意图分流：intent_classifier + train/(训练/评测闭环)
├─ ingest/                 # 离线数据管线：文档→Markdown/OCR→切分→入库
├─ data/ecommerce/         # 文档（手机/电脑/售后/指南 分目录，父目录名=source）
├─ static/                 # 前端聊天页（内置 index.html）
├─ classify_data/          # 意图训练种子 + 评测集
├─ models/                 # bge-m3 / bge-reranker / bert-base-chinese（本机已就位）
├─ config.ini / .env.example / .env.app.example
├─ docker-compose.yml      # 中间件：MySQL/Redis/Milvus+etcd+MinIO
├─ docker-compose.app.yml  # 应用容器编排
├─ Dockerfile / .dockerignore
└─ 接口文档.md / 六步跑通详解.md / 项目全流程讲解.md / requirements.txt
```

## 快速开始

```powershell
# 1. 中间件（MySQL8/Redis7/Milvus+etcd+MinIO；本机 MySQL 映射 3308）
docker compose up -d

# 2. Python 环境（本机可直接复用已有 conda 环境 EduRAG，依赖齐全）
conda activate EduRAG        # 或自建：conda create -n ecom python=3.10.18 -y
                             #        pip install -r requirements.txt

# 3. 建 Milvus 库（首次一次）
python -c "from pymilvus import MilvusClient; MilvusClient('http://127.0.0.1:19530').create_database('ecom')"

# 4. 改配置：config.ini 填 DASHSCOPE_API_KEY

# 5. 离线入库：文档放 data/ecommerce/ 后执行
python ingest/pipeline.py

# 6. 启动服务
python -m app.main
# 浏览器打开 http://localhost:8003
```

## 里程碑对照（M0~M9）

- M0 中间件容器 → `docker-compose.yml` ✅
- M1 数据接入 → `ingest/loaders.py`（MinerU 可选 + PyMuPDF 表格 + OCR + docx/pptx + BLIP 可选）✅
- M2 父子切分 → `ingest/chunkers.py`、`ingest/table_cross_page.py` ✅
- M3 向量入库 → `core/vector_store.py`（BGE-M3 双路 + Milvus）✅
- M4 检索+精排 → `core/vector_store.py` 的 `hybrid_search_with_rerank` ✅
- M5 前置链路 → `retrieval/`（正则/Redis/MySQL BM25）✅
- M6 意图分流 → `routing/`（三分类 + 训练/评测闭环）✅
- M7 生成+接口 → `core/rag_system.py`、`app/main.py`（双 Prompt + 指代消解 + 策略路由）✅
- M8 评估 → `routing/train/evaluate_intent.py`（意图回归门禁）；RAGAS 可接入（`rag_qa/rag_assessment` 参考）✅/可扩展
- M9 部署 → `Dockerfile` + `docker-compose.app.yml` + `.env.app.example` + `接口文档.md` ✅

---

> **仓库收录说明**：本仓库未收录模型权重（`models/`，约 10GB）、虚拟环境（`venv/`）、训练检查点（`routing/train/bert_results/`）与运行产物（`logs/` `results/` `volumes/`）。
> 本地运行请按上文准备中间件与模型文件；`config.ini` 中的密钥请使用自己的值（仓库内为占位符）。