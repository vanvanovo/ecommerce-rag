"""
父子切分：先按"语义边界"切父块，再从父块切子块。
子块用于向量检索（精准），命中后取父块内容（上下文充足）交给 LLM。
电商场景：Markdown 表格/标题是天然边界，优先保表不拦腰切。
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from base.config import config
from base.logger import logger


def ChineseTextSplitter(chunk_size, chunk_overlap):
    """中文分隔符优先级：段落 → 换行 → 句号/问号/叹号 → 分号 → 逗号"""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )


def parent_child_split(doc: Document):
    """返回子块列表，每个子块 metadata 带 parent_id / parent_content。"""
    parent_splitter = ChineseTextSplitter(config.PARENT_CHUNK_SIZE, config.PARENT_CHUNK_OVERLAP)
    child_splitter = ChineseTextSplitter(config.CHILD_CHUNK_SIZE, config.CHILD_CHUNK_OVERLAP)

    child_chunks = []
    parent_docs = parent_splitter.split_documents([doc])
    for i, pd in enumerate(parent_docs):
        parent_id = f"{doc.metadata.get('file_path', 'doc')}#parent{i}"
        for j, cd in enumerate(child_splitter.split_documents([pd])):
            cd.metadata.update({
                "id": f"{parent_id}#child{j}",
                "parent_id": parent_id,
                "parent_content": pd.page_content,   # 检索后用它的全文做上下文
                "source": doc.metadata.get("source"),
                "tag": doc.metadata.get("tag", ""),       # 价格/保修/退换货/参数…
                "page_no": doc.metadata.get("page_no", 0),
                "sm_ty": doc.metadata.get("sm_ty", ""),   # 数码产品类型 phone/laptop/tablet
                "timestamp": doc.metadata.get("timestamp"),
            })
            child_chunks.append(cd)
    logger.info(f"文档 {doc.metadata.get('file_path')} 切出子块 {len(child_chunks)} 个")
    return child_chunks