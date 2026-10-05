"""
离线数据接入管线：文档 → Markdown/文本 → 父子切分 → BGE-M3 入库 Milvus

数据组织建议（对应 config.ini 的 valid_sources）：
    data/ecommerce/
        ├─ 手机/       产品手册、参数表、价格表
        ├─ 电脑/
        ├─ 平板/
        ├─ 售后政策/   退货、保修、价保、发票
        └─ 操作指南/   App使用、系统升级指引

用法： python ingest/pipeline.py     # 处理 data/ecommerce/ 下所有文档
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base.config import config
from base.logger import logger
from ingest.chunkers import parent_child_split
from ingest.loaders import load_document_to_text
from core.vector_store import VectorStore


def build_index(source_dir=None):
    """对 source_dir（默认 config.DATA_DIR）下所有文档做 加载→切分→入库。"""
    source_dir = source_dir or config.DATA_DIR
    vs = VectorStore()          # 连接 Milvus，建/加载 collection，载入 BGE-M3 + reranker
    docs = load_document_to_text(source_dir)   # 每种文件 → Document(page_content, metadata{source,...})
    logger.info(f"共加载 {len(docs)} 个文档")

    for doc in docs:
        # 商品/售后信息逐条切成父子块，子块带 parent_id/parent_content
        child_chunks = parent_child_split(doc)
        if child_chunks:
            vs.add_documents(child_chunks)
        logger.info(f"已入库: {doc.metadata.get('file_path')}")
    logger.info("向量入库完成")


if __name__ == '__main__':
    build_index()