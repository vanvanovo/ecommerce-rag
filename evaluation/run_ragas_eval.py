# evaluation/run_ragas_eval.py
"""
RAGAS 四维评估（电商知识库真实链路）

指标：faithfulness（忠实度）/ answer_relevancy（答案相关性）
      / context_precision（上下文精确率）/ context_recall（上下文召回率）

- 被测链路：项目真实 RAG 链路（BERT 意图 → 策略路由 → BGE-M3 混合检索 + Reranker → Qwen 生成）
- 评测模型：默认本地 Ollama qwen2.5:7b（评委）+ mxbai-embed-large（相似度嵌入），零云端依赖
- 用法：
    python evaluation/run_ragas_eval.py              # 全量 16 条
    python evaluation/run_ragas_eval.py --limit 3    # 冒烟 3 条
    python evaluation/run_ragas_eval.py --judge deepseek   # 需要用 DEEPSEEK_API_KEY 环境变量

产物（evaluation/outputs/）：
    eval_samples.json   # 每条样本：问题/生成答案/检索上下文/意图/耗时
    ragas_results.csv   # 每条样本的四维分数
    ragas_summary.json  # 四维均值汇总
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 生成通道：默认走本地 Ollama（OpenAI 兼容接口），无需云端 Key
os.environ.setdefault("DASHSCOPE_API_KEY", "ollama")
os.environ.setdefault("DASHSCOPE_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("LLM_MODEL", "qwen2.5:7b")

OLLAMA_V1 = "http://localhost:11434/v1"
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def build_llm(judge: str):
    from langchain_openai import ChatOpenAI

    if judge == "deepseek":
        return ChatOpenAI(
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key=os.environ["DEEPSEEK_API_KEY"],
            temperature=0,
        )
    return ChatOpenAI(model="qwen2.5:7b", base_url=OLLAMA_V1, api_key="ollama", temperature=0)


def build_embeddings():
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model="mxbai-embed-large:latest",
        base_url=OLLAMA_V1,
        api_key="ollama",
        check_embedding_ctx_length=False,   # 非 OpenAI 模型不做 tiktoken 预处理
    )


def generate_samples(questions):
    """跑项目真实 RAG 链路，捕获检索上下文与生成答案。"""
    from core.vector_store import VectorStore
    from core.rag_system import RAGSystem

    t0 = time.time()
    vs = VectorStore()
    print(f"[load] VectorStore 就绪 {time.time() - t0:.1f}s", flush=True)
    rag = RAGSystem(vs)

    captured = []
    orig = vs.hybrid_search_with_rerank

    def spy(query, k=None, source_filter=None):
        docs = orig(query, k=k, source_filter=source_filter)
        captured.append(docs)
        return docs

    vs.hybrid_search_with_rerank = spy

    samples = []
    for i, item in enumerate(questions, 1):
        captured.clear()
        t = time.time()
        label, _prob = rag.intent.predict(item["question"])
        answer = "".join(rag.generate_answer(item["question"], None, None))

        contexts, seen = [], set()
        for docs in captured:
            for d in docs:
                if d.page_content not in seen:
                    seen.add(d.page_content)
                    contexts.append(d.page_content)

        samples.append({
            "question": item["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
            "intent": label,
            "retrieval_calls": len(captured),
            "elapsed_s": round(time.time() - t, 1),
        })
        print(f"[{i}/{len(questions)}] {item['question'][:32]} | 意图={label} | "
              f"检索{len(captured)}次/上下文{len(contexts)}块 | {samples[-1]['elapsed_s']}s", flush=True)
    return samples


def run_ragas(samples, judge: str):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    ds = Dataset.from_dict({
        "question": [s["question"] for s in samples],
        "answer": [s["answer"] for s in samples],
        "contexts": [s["contexts"] for s in samples],
        "ground_truth": [s["ground_truth"] for s in samples],
    })
    return evaluate(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=build_llm(judge),
        embeddings=build_embeddings(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全量）")
    ap.add_argument("--judge", choices=["ollama", "deepseek"], default="ollama")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    out = root / "outputs"
    out.mkdir(exist_ok=True)

    questions = json.loads((root / "eval_set.json").read_text(encoding="utf-8"))
    if args.limit > 0:
        questions = questions[:args.limit]

    samples = generate_samples(questions)
    (out / "eval_samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[gen] 样本生成完成，开始 RAGAS 评测…", flush=True)
    t = time.time()
    result = run_ragas(samples, args.judge)
    df = result.to_pandas()
    df = df.rename(columns={
        "user_input": "question",
        "retrieved_contexts": "contexts",
        "response": "answer",
        "reference": "ground_truth",
    })
    df.to_csv(out / "ragas_results.csv", index=False, encoding="utf-8-sig")

    metric_cols = [m for m in METRIC_NAMES if m in df.columns]
    summary = {m: round(float(df[m].mean()), 4) for m in metric_cols}
    summary.update(
        samples=len(samples),
        valid_count={m: int(df[m].notna().sum()) for m in metric_cols},
        judge="ollama/qwen2.5:7b" if args.judge == "ollama" else "deepseek/deepseek-chat",
        embedding="ollama/mxbai-embed-large",
        elapsed_s=round(time.time() - t, 1),
    )
    (out / "ragas_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[ragas] 完成:", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
