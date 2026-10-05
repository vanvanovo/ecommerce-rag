# -*- coding: utf-8 -*-
"""
检索链路冒烟测试（不依赖 LLM，也不依赖大模型问答）。
用法：python smoke_test.py
验证：BGE-M3 双路向量 → Milvus 混合检索 → 父子聚合 → BGE-Reranker 精排。
"""
from core.vector_store import VectorStore

QUERIES = [
    ("iPhone 16 Pro 128G 多少钱", None),
    ("7天无理由怎么退", "售后政策"),
    ("保修几年", "售后政策"),
    ("拍照好的手机推荐", "手机"),
]

if __name__ == "__main__":
    vs = VectorStore()
    for q, sf in QUERIES:
        print("=" * 70)
        print(f"Q: {q}   | source_filter: {sf}")
        docs = vs.hybrid_search_with_rerank(q, source_filter=sf)
        if not docs:
            print("  （无结果）")
            continue
        for i, d in enumerate(docs, 1):
            snippet = d.page_content[:110].replace("\n", " ")
            print(f"  [{i}] {snippet}")
    print("=" * 70)
    print("冒烟测试完成")
