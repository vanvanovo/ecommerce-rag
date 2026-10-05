"""
向量存储：BGE-M3(稠密+稀疏双输出) → Milvus hybrid_search + WeightedRanker → 父子检索 → BGE-Reranker 精排

对应 EduRAG 的 vector_store.py，领域改成电商。
模型下载（放 models/ 下）：
  - bge-m3                    : https://huggingface.co/BAAI/bge-m3
  - bge-reranker-large        : https://huggingface.co/BAAI/bge-reranker-large
"""
import hashlib
import os
import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import MilvusClient, DataType, AnnSearchRequest, WeightedRanker

from base.config import config
from base.logger import logger


class VectorStore:
    def __init__(self,
                 collection_name=config.MILVUS_COLLECTION_NAME,
                 host=config.MILVUS_HOST,
                 port=config.MILVUS_PORT,
                 database=config.MILVUS_DATABASE_NAME):
        self.collection_name = collection_name
        self.database = database

        # 设备：auto → 有 CUDA 用 GPU，否则 CPU（写死 cpu/cuda/mps 亦可）
        self.device = self._resolve_device()

        # 本地模型路径（第一次要下载到 models/）
        bge_m3_path = os.path.join(config.MODELS_DIR, 'bge-m3')
        rerank_path = os.path.join(config.MODELS_DIR, 'bge-reranker-large')

        self.reranker = CrossEncoder(rerank_path, device=self.device)
        self.embedding_function = BGEM3EmbeddingFunction(
            model_name_or_path=bge_m3_path,
            use_f16=(self.device == 'cuda'),   # GPU 上 FP16 提速明显；CPU 上需关闭
            device=self.device
        )
        self.dense_dim = self.embedding_function.dim["dense"]

        self.client = MilvusClient(uri=f'http://{host}:{port}', db_name=database)
        self._create_or_load_collection()

    @staticmethod
    def _resolve_device():
        """解析运行设备：优先配置，auto 时自动探测 CUDA / MPS。"""
        want = getattr(config, 'MODEL_DEVICE', 'auto')
        if want in ('cpu', 'cuda', 'mps'):
            return want
        if torch.cuda.is_available():
            return 'cuda'
        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    # ---------- 建/加载 collection ----------
    def _create_or_load_collection(self):
        if not self.client.has_collection(self.collection_name):
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
            # 字段集对齐实战项目：子块id/父块id/子块内容/父块内容/两路向量/时间戳/tag/source/page_no/SM_数码产品类型
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.dense_dim)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=255)
            schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=50)        # 来源：手机/电脑/售后政策…
            schema.add_field(field_name="tag", datatype=DataType.VARCHAR, max_length=100)          # 标签：价格/保修/退换货/参数…
            schema.add_field(field_name="page_no", datatype=DataType.INT64)                        # 页码（用于溯源展示）
            schema.add_field(field_name="sm_ty", datatype=DataType.VARCHAR, max_length=50)         # 数码产品类型（SM-type）：phone/laptop/tablet
            schema.add_field(field_name="timestamp", datatype=DataType.VARCHAR, max_length=50)

            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="dense_vector", index_name="dense_index",
                                   index_type="HNSW", metric_type="IP", params={"M": 16, "efConstruction": 200})
            index_params.add_index(field_name="sparse_vector", index_name="sparse_index",
                                   index_type="SPARSE_INVERTED_INDEX", metric_type="IP",
                                   params={"drop_ratio_build": 0.2})
            self.client.create_collection(self.collection_name, schema=schema, index_params=index_params)
            logger.info(f"已创建集合 {self.collection_name}")
        self.client.load_collection(self.collection_name)

    # ---------- 入库 ----------
    def add_documents(self, documents, sm_ty_map=None):
        texts = [d.page_content for d in documents]
        embeddings = self.embedding_function(texts)     # BGE-M3 一模型两路输出
        rows = []
        for i, doc in enumerate(documents):
            md = doc.metadata
            rows.append({
                "id": hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest(),
                "text": doc.page_content,
                "dense_vector": embeddings["dense"][i],
                "sparse_vector": self._convert_sparse(embeddings["sparse"], i),
                "parent_id": md.get("parent_id", ""),
                "parent_content": md.get("parent_content", doc.page_content),
                "source": md.get("source", ""),
                "tag": md.get("tag", ""),                 # 价格/保修/退换货/参数…
                "page_no": md.get("page_no", 0),
                "sm_ty": md.get("sm_ty", ""),             # phone/laptop/tablet 或空
                "timestamp": md.get("timestamp", ""),
            })
        if rows:
            self.client.upsert(self.collection_name, data=rows)
            logger.info(f"已插入 {len(rows)} 条向量")

    def _convert_sparse(self, sparse_embeddings, index):
        """BGE-M3 稀疏向量 → Milvus {列索引: 权重} 字典。兼容 coo_array / csr_matrix。"""
        row = sparse_embeddings[index]
        indices = row.col if hasattr(row, 'col') else row.indices
        values = row.data
        return {int(i): float(v) for i, v in zip(indices, values)}

    # ---------- 混合检索 + 重排 ----------
    def hybrid_search_with_rerank(self, query, k=None, source_filter=None):
        """返回精排后的 top 文档（父块全文）。source_filter 示例："手机"。"""
        k = k or config.RETRIEVAL_K
        embeddings = self.embedding_function([query])
        dense_q = embeddings["dense"][0]
        sparse_q = self._convert_sparse(embeddings["sparse"], 0)
        expr = f'source == "{source_filter}"' if source_filter else ""

        dense_req = AnnSearchRequest(data=[dense_q], anns_field="dense_vector",
                                     param={"metric_type": "IP", "params": {"nprobe": 10}},
                                     limit=k, expr=expr)
        sparse_req = AnnSearchRequest(data=[sparse_q], anns_field="sparse_vector",
                                      param={"metric_type": "IP", "params": {}},
                                      limit=k, expr=expr)
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=WeightedRanker(1.0, 0.7),   # 稠密(语义)权重1.0 + 稀疏(关键词)权重0.7
            limit=k,
            output_fields=["id", "text", "parent_content", "source", "parent_id",
                           "tag", "page_no", "sm_ty", "timestamp"],
        )[0]

        # 子块 → 父块聚合去重（父子检索）
        parent_docs = self._unique_parents(results)
        if len(parent_docs) <= 1:
            return parent_docs

        # BGE-Reranker cross-encoder 精排（只排少量候选，成本可控）
        pairs = [[query, d.page_content] for d in parent_docs]
        scores = self.reranker.predict(pairs)
        ranked = [d for _, d in sorted(zip(scores, parent_docs), key=lambda x: -x[0])]
        return ranked[:config.CANDIDATE_M]

    def _unique_parents(self, results):
        """同一父块的多个子块命中 → 只留父块全文一次。"""
        seen = {}
        for hit in results:
            entity = hit["entity"]
            content = entity.get("parent_content") or entity.get("text", "")
            if content not in seen:
                seen[content] = Document(
                    page_content=content,
                    metadata={"id": entity.get("id"), "parent_id": entity.get("parent_id"),
                              "source": entity.get("source"), "tag": entity.get("tag"),
                              "page_no": entity.get("page_no"), "sm_ty": entity.get("sm_ty"),
                              "timestamp": entity.get("timestamp")},
                )
        return list(seen.values())


if __name__ == "__main__":
    # 自测：先入库 2 个文档，再检索
    vs = VectorStore()
    from ingest.chunkers import parent_child_split
    from langchain_core.documents import Document
    sample = Document(page_content="iPhone 16 Pro 128G 官网售价 7999 元，支持 12 期免息分期。",
                      metadata={"source": "手机", "file_path": "手机/价格.md",
                                "timestamp": "2026-01-01"})
    vs.add_documents(parent_child_split(sample))
    print(vs.hybrid_search_with_rerank("iPhone 16 Pro 128G 多少钱", source_filter="手机"))