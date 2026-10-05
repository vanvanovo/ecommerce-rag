# RAGAS 四维评估（电商知识库）

对项目真实 RAG 链路做自动化质量评估：**忠实度 / 答案相关性 / 上下文精确率 / 上下文召回率**。

## 被测链路

BERT 意图分流（闲聊 / 产品咨询 / 售后政策）→ 指代消解 → 策略路由（直检 / HyDE / 子查询）
→ BGE-M3 稠密+稀疏混合检索 → BGE-Reranker 精排 → 父子回上下文 → Qwen 生成。

> 评估运行时生成通道走**本地 Ollama**（OpenAI 兼容接口，qwen2.5:7b），无需云端 Key，全流程可复现。

## 评测集

`eval_set.json`：16 条电商域问题（产品参数/价格 8 条 + 售后政策 8 条），
每题带标准答案（ground_truth），内容取自知识库文档原文。

## 运行

```powershell
# 前置：Ollama（qwen2.5:7b + mxbai-embed-large）；Milvus / Redis / MySQL（docker compose up -d）
pip install -r evaluation/requirements-eval.txt   # RAGAS 评估附加依赖
python evaluation/run_ragas_eval.py               # 全量 16 条
python evaluation/run_ragas_eval.py --limit 3     # 冒烟 3 条
python evaluation/run_ragas_eval.py --judge deepseek   # 评委换云端（需 DEEPSEEK_API_KEY）
```

产物在 `evaluation/outputs/`：`eval_samples.json`（逐样本：问题/答案/上下文/意图/耗时）、
`ragas_results.csv`（逐样本四维分数）、`ragas_summary.json`（汇总）。

## 实测结果（2026-10-05，本机 CPU，16/16 有效）

| 指标 | 分数（16 条均值） |
|---|---|
| 忠实度 faithfulness | **0.79** |
| 答案相关性 answer_relevancy | **0.74** |
| 上下文精确率 context_precision | **0.94** |
| 上下文召回率 context_recall | **0.94** |

评委：本地 Qwen2.5-7B（Ollama）；嵌入：mxbai-embed-large。

## 首轮 badcase 归因（评测驱动迭代）

- **意图误判**：「视频播放续航」被意图模型判为"售后政策"→ 源过滤只取政策文档，
  错失产品手册 → 回答降级为"人工核实"（该条精确率/召回率为 0）。
  归因：意图模型训练样本未覆盖"续航"类表达，待补样重训后复测。
- **生成端冗余**：部分回答带营销/客套话（如"随时咨询我哦"），被忠实度评委判为
  上下文不支持，拉低忠实度——生成 Prompt 可增加"禁止客套话/营销话术"约束后复测。

> 本评估为**首轮基线**，评测集与脚本全部入库，可一键复现。
