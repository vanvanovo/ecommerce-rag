# evaluation/smoke_rag.py
# RAGAS 评估前冒烟：真实链路（意图 → 检索 → 本地 Qwen 生成）跑 2 条
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 本地生成通道：Ollama 的 OpenAI 兼容接口（无需云端 Key）
os.environ.setdefault("DASHSCOPE_API_KEY", "ollama")
os.environ.setdefault("DASHSCOPE_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("LLM_MODEL", "qwen2.5:7b")

from core.vector_store import VectorStore  # noqa: E402
from core.rag_system import RAGSystem      # noqa: E402


def main():
    t0 = time.time()
    vs = VectorStore()
    print(f"[load] VectorStore 就绪 {time.time() - t0:.1f}s")

    rag = RAGSystem(vs)

    captured = []
    orig = vs.hybrid_search_with_rerank

    def spy(query, k=None, source_filter=None):
        docs = orig(query, k=k, source_filter=source_filter)
        captured.append((query, source_filter, docs))
        return docs

    vs.hybrid_search_with_rerank = spy

    for q in ["iPhone 16 Pro 128G 官网售价多少？", "7 天无理由退货有什么要求？"]:
        captured.clear()
        t = time.time()
        label, prob = rag.intent.predict(q)
        ans = "".join(rag.generate_answer(q, None, None))
        print("=" * 70)
        print("Q:", q)
        print("intent:", label, round(prob, 2))
        for query, sf, docs in captured:
            print(f"  retrieve: {query!r} | filter={sf} | docs={len(docs)}")
            for d in docs[:2]:
                print("    -", d.page_content[:60].replace("\n", " "))
        print("A:", ans)
        print(f"time: {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
