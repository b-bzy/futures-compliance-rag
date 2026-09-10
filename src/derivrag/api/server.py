"""FastAPI 服务。

    GET  /health          模型与索引状态
    POST /search          只检索，返回带引用的候选（不调用 LLM，快）
    POST /chat            完整问答，支持多轮 session
    GET  /stats           语料与索引统计

启动:
    uvicorn derivrag.api.server:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import load_config
from ..pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(
    title="期货合规条款 RAG 检索系统",
    description="基于混合检索 + 多阶段重排的期权条款问答 API",
    version="0.1.0",
)

_cfg = load_config()
# service.lazy_load=true 时进程启动不加载模型。16GB 机器上
# bge-m3 + reranker + qwen3 无法同时常驻，按需加载可避免启动即 OOM。
# 设为 false 可换取首次请求的低延迟（适合内存充足的部署环境）。
_pipeline = RAGPipeline(_cfg, lazy=_cfg.get_path("service.lazy_load", True))

# 内存版会话存储。生产环境应换成 Redis，这里够用且零依赖。
_sessions: dict[str, dict[str, Any]] = {}


# =====================================================================
class SearchRequest(BaseModel):
    query: str = Field(..., description="检索问题")
    top_k: int = Field(5, ge=1, le=50)
    use_rerank: bool = True
    use_hyde: bool = False
    channels: list[str] = Field(default_factory=lambda: ["bm25", "dense", "keyword"])
    venue: str | None = Field(None, description="限定交易所，如 SSE / CFFEX")


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    top_k: int = Field(5, ge=1, le=20)
    use_hyde: bool | None = None
    use_rewrite: bool | None = None
    use_rerank: bool = True


class Citation(BaseModel):
    index: int
    doc_title: str
    doc_no: str
    clause_id: str
    venue: str
    effective_status: str
    source_url: str
    text: str
    score: float
    component_scores: dict[str, float] = {}
    component_ranks: dict[str, int] = {}


# =====================================================================
@app.get("/health")
def health() -> dict[str, Any]:
    """检查索引与各组件状态。不触发模型加载。"""
    chroma_dir = _cfg.resolve("paths.chroma")
    bm25_path = _cfg.resolve("paths.bm25")
    stats_path = _cfg.resolve("paths.processed") / "stats.json"

    vector_count = None
    try:
        from ..index.vectorstore import VectorStore

        vector_count = VectorStore(chroma_dir).count()
    except Exception as e:  # noqa: BLE001
        logger.debug("向量库探测失败: %s", e)

    provider_ok = False
    provider_name = None
    try:
        provider_name = _pipeline.provider.name
        provider_ok = True
    except Exception as e:  # noqa: BLE001
        logger.debug("LLM provider 不可用: %s", e)

    return {
        "status": "ok" if vector_count and bm25_path.exists() else "index_missing",
        "vector_count": vector_count,
        "bm25_index": bm25_path.exists(),
        "llm_provider": provider_name,
        "llm_available": provider_ok,
        "corpus_stats": json.loads(stats_path.read_text(encoding="utf-8"))
        if stats_path.exists()
        else None,
    }


@app.get("/stats")
def stats() -> dict[str, Any]:
    p = _cfg.resolve("paths.processed") / "stats.json"
    if not p.exists():
        raise HTTPException(404, "尚未生成统计，请先执行 scripts/02_parse.py")
    return json.loads(p.read_text(encoding="utf-8"))


@app.post("/search")
def search(req: SearchRequest) -> dict[str, Any]:
    """只做检索，不生成答案。延迟远低于 /chat，适合做条款查找。"""
    t = time.perf_counter()
    try:
        result = _pipeline.answer(
            req.query,
            use_hyde=req.use_hyde,
            use_rewrite=False,
            use_rerank=req.use_rerank,
            channels=tuple(req.channels),
            top_k=req.top_k,
            generate=False,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e

    citations = result.citations()
    if req.venue:
        citations = [c for c in citations if c["venue"] == req.venue]

    return {
        "query": req.query,
        "hits": citations,
        "channel_counts": result.channel_counts,
        "timings": result.timings,
        "elapsed": round(time.perf_counter() - t, 3),
    }


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    """完整问答。带 session_id 时启用多轮查询重写。"""
    session_id = req.session_id or str(uuid.uuid4())
    session = _sessions.setdefault(session_id, {"history": [], "created": time.time()})
    history = session["history"]

    try:
        result = _pipeline.answer(
            req.message,
            history=history,
            use_hyde=req.use_hyde,
            use_rewrite=req.use_rewrite,
            use_rerank=req.use_rerank,
            top_k=req.top_k,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e

    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": result.answer})
    # 只保留最近 10 轮，防止内存无限增长
    session["history"] = history[-20:]
    _evict_stale_sessions()

    return {
        "session_id": session_id,
        "question": req.message,
        "rewritten_query": result.rewritten_query,
        "answer": result.answer,
        "citations": result.citations(),
        "timings": result.timings,
    }


@app.delete("/chat/{session_id}")
def clear_session(session_id: str) -> dict[str, str]:
    _sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


def _evict_stale_sessions() -> None:
    ttl = _cfg.get_path("service.session_ttl_seconds", 3600)
    now = time.time()
    for sid in [s for s, v in _sessions.items() if now - v["created"] > ttl]:
        _sessions.pop(sid, None)


def main() -> None:
    import uvicorn

    uvicorn.run(
        app,
        host=_cfg.get_path("service.host", "127.0.0.1"),
        port=_cfg.get_path("service.port", 8000),
    )


if __name__ == "__main__":
    main()
