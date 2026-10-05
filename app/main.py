"""
FastAPI 服务层：REST + WebSocket 流式。
对齐一期 RAG：POST /api/query 探测是否流式，WS /api/stream 逐 token 推送。
"""
import asyncio
import json
import os
import sys
import uuid
from typing import Optional

# 允许直接用 `python app/main.py` 运行（否则用 `python -m app.main`）
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

from base.config import config
from base.logger import logger
from retrieval.mysql_client import MySQLClient
from retrieval.redis_cache import RedisClient
from retrieval.fast_path import FastPath
from core.vector_store import VectorStore
from core.rag_system import RAGSystem


class QueryRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    is_streaming: bool
    session_id: str


class QASystem:
    """编排层：快路径 → RAG 兜底 + 会话历史。"""

    def __init__(self):
        self.mysql = MySQLClient()
        self.redis = RedisClient()
        self.fast_path = FastPath(self.mysql, self.redis)
        self.rag = RAGSystem(VectorStore())

    def query(self, query, source_filter=None, session_id=None):
        history = self.mysql.fetch_recent_history(session_id) if session_id else []

        # 快路径：规则/缓存/BM25
        answer, need_rag = self.fast_path.search(query)
        if answer:
            if session_id:
                self.mysql.insert_history(session_id, query, answer)
                self.mysql.commit()
            yield answer, True
            return

        if need_rag:
            # 慢路径：RAG 流式
            collected = ""
            for token in self.rag.generate_answer(query, source_filter, history):
                collected += token
                yield token, False
            if session_id and collected:
                self.mysql.insert_history(session_id, query, collected)
                self.mysql.commit()
            yield "", True
            return

        yield "未找到答案", True


app = FastAPI(title="电商客服 RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
qa = QASystem()


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/create_session")
async def create_session():
    return {"session_id": str(uuid.uuid4())}


@app.get("/api/history/{session_id}")
async def history(session_id: str):
    return {"session_id": session_id, "history": qa.mysql.fetch_recent_history(session_id)}


@app.get("/api/sources")
async def sources():
    # 供前端"学科/品类"下拉，映射为 source_filter
    return {"sources": config.VALID_SOURCES}


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    session_id = req.session_id or str(uuid.uuid4())
    first = next(qa.query(req.query, req.source_filter, session_id), None)
    if not first:
        raise HTTPException(status_code=500, detail="查询失败")
    answer, is_complete = first
    if is_complete:
        return QueryResponse(answer=answer, is_streaming=False, session_id=session_id)
    return QueryResponse(answer="请通过 WebSocket 获取流式响应", is_streaming=True, session_id=session_id)


@app.websocket("/api/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = json.loads(await ws.receive_text())
            query = data.get("query")
            source_filter = data.get("source_filter")
            session_id = data.get("session_id") or str(uuid.uuid4())
            await ws.send_json({"type": "start", "session_id": session_id})

            for token, is_complete in qa.query(query, source_filter, session_id):
                if token:
                    await ws.send_json({"type": "token", "token": token, "session_id": session_id})
                if is_complete:
                    await ws.send_json({"type": "end", "session_id": session_id, "is_complete": True})
                    break
                await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        logger.info("WebSocket 断开")
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
        await ws.send_json({"type": "error", "error": str(e)})
    finally:
        await ws.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8003, reload=False)