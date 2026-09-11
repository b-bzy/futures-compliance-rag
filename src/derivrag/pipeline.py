"""RAG 主链路。

    查询重写(多轮) -> HyDE -> 三路召回 -> score-aware RRF -> 父块回溯
    -> cross-encoder 重排 -> 带引用约束的生成

每一步都可以单独关闭，这既是消融实验的基础，也让"这一步到底带来了多少
提升"变成可测量的数字而不是口号。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from .config import Config, load_config
from .llm.prompts import ANSWER_NO_CONTEXT, build_answer_messages
from .schema import RetrievalHit

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """一次问答的完整结果，包含全过程的中间产物供 UI 可视化。"""

    question: str
    answer: str
    hits: list[RetrievalHit] = field(default_factory=list)
    rewritten_query: str | None = None
    hypothetical: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    channel_counts: dict[str, int] = field(default_factory=dict)

    def citations(self) -> list[dict[str, Any]]:
        """整理成 UI 用的引用卡片。"""
        out = []
        for i, h in enumerate(self.hits, 1):
            m = h.metadata
            out.append(
                {
                    "index": i,
                    "doc_title": m.get("doc_title", ""),
                    "doc_no": m.get("doc_no", ""),
                    "clause_id": m.get("clause_id", ""),
                    "venue": m.get("venue", ""),
                    "effective_status": m.get("effective_status", ""),
                    "source_url": m.get("source_url", ""),
                    "text": h.parent_text or h.text,
                    "snippet": h.text,
                    "score": round(h.score, 4),
                    "component_scores": {
                        k: round(v, 4) for k, v in h.component_scores.items()
                    },
                    "component_ranks": h.component_ranks,
                }
            )
        return out


class RAGPipeline:
    """端到端问答流水线。"""

    def __init__(self, cfg: Config | None = None, *, lazy: bool = True) -> None:
        self.cfg = cfg or load_config()
        self._retriever = None
        self._provider = None
        if not lazy:
            self.retriever  # noqa: B018 - 触发加载
            self.provider  # noqa: B018

    # ---------------------------------------------------------------
    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = build_retriever(self.cfg)
        return self._retriever

    @property
    def provider(self):
        if self._provider is None:
            from .llm.provider import get_provider

            self._provider = get_provider(self.cfg)
        return self._provider

    # ---------------------------------------------------------------
    def answer(
        self,
        question: str,
        *,
        history: Sequence[dict[str, str]] | None = None,
        use_hyde: bool | None = None,
        use_rewrite: bool | None = None,
        use_rerank: bool = True,
        channels: tuple[str, ...] = ("bm25", "dense", "keyword"),
        top_k: int | None = None,
        generate: bool = True,
    ) -> RAGResult:
        """完整问答。各开关默认读配置，显式传参可覆盖（消融实验用）。"""
        q_cfg = self.cfg.get_path("query", {})
        use_hyde = q_cfg.get("hyde", {}).get("enabled", True) if use_hyde is None else use_hyde
        use_rewrite = (
            q_cfg.get("rewrite", {}).get("enabled", True)
            if use_rewrite is None
            else use_rewrite
        )

        result = RAGResult(question=question, answer="")
        search_query = question
        t0 = time.perf_counter()

        # ---- 1. 多轮查询重写 ----
        if use_rewrite and history:
            from .retrieve.hyde import rewrite_query

            t = time.perf_counter()
            search_query = rewrite_query(
                self.provider,
                question,
                history,
                max_turns=q_cfg.get("rewrite", {}).get("max_history_turns", 4),
            )
            result.rewritten_query = search_query if search_query != question else None
            result.timings["rewrite"] = time.perf_counter() - t

        # ---- 2. HyDE ----
        expand: list[str] = []
        if use_hyde:
            from .retrieve.hyde import generate_hypothetical

            t = time.perf_counter()
            expand = generate_hypothetical(
                self.provider,
                search_query,
                n=q_cfg.get("hyde", {}).get("num_hypotheses", 1),
                # 默认值必须与 configs/config.yaml 一致。这里曾是 256，
                # 与 hyde.py 的默认值、config 的值构成三处独立的同名常量，
                # 改了其中一处另外两处还会复活这个 bug。
                max_tokens=q_cfg.get("hyde", {}).get("max_tokens", 2048),
            )
            result.hypothetical = expand
            result.timings["hyde"] = time.perf_counter() - t
            if not expand:
                # 「请求了 HyDE 但一份假设文档都没拿到」必须与「没请求 HyDE」
                # 可区分，否则消融实验会把失效的那轮当作 HyDE 档记录，
                # 测出来的「HyDE 无收益」其实是「HyDE 根本没跑」。
                result.timings["hyde_ineffective"] = 1.0
                logger.error(
                    "HyDE 已启用但未产出任何假设文档，本轮实际等价于未开 HyDE。"
                    "若正在跑消融实验，plus_hyde 档的数字不可用。"
                )

        # ---- 3. 检索（三路召回 + 融合 + 回溯 + 重排）----
        t = time.perf_counter()
        hits = self.retriever.retrieve(
            search_query,
            channels=channels,
            use_rerank=use_rerank,
            top_k=top_k,
            expand_queries=expand,
        )
        result.timings["retrieve"] = time.perf_counter() - t
        result.hits = hits
        for h in hits:
            for ch in h.component_ranks:
                result.channel_counts[ch] = result.channel_counts.get(ch, 0) + 1

        # ---- 4. 生成 ----
        if not generate:
            result.timings["total"] = time.perf_counter() - t0
            return result

        if not hits:
            result.answer = ANSWER_NO_CONTEXT
            result.timings["total"] = time.perf_counter() - t0
            return result

        context = self.retriever.context_for(
            hits, max_chars=self.cfg.get_path("generate.max_context_chars", 12000)
        )
        t = time.perf_counter()
        try:
            result.answer = self.provider.chat(build_answer_messages(question, context))
        except Exception as e:  # noqa: BLE001
            logger.error("生成失败: %s", e)
            result.answer = f"生成阶段出错：{e}\n\n检索到的条款仍可在下方引用中查看。"
        result.timings["generate"] = time.perf_counter() - t
        result.timings["total"] = time.perf_counter() - t0
        return result

    # ---------------------------------------------------------------
    def stream_answer(
        self,
        question: str,
        *,
        history: Sequence[dict[str, str]] | None = None,
        **kw,
    ) -> tuple[RAGResult, Iterator[str]]:
        """先返回检索结果，再流式产出答案 —— UI 可以立刻显示引用。"""
        result = self.answer(question, history=history, generate=False, **kw)
        if not result.hits:
            return result, iter([ANSWER_NO_CONTEXT])

        context = self.retriever.context_for(
            result.hits, max_chars=self.cfg.get_path("generate.max_context_chars", 12000)
        )
        return result, self.provider.stream(build_answer_messages(question, context))


# =====================================================================
def build_retriever(cfg: Config | None = None):
    """按配置装配 HybridRetriever（向量库 + BM25 + 重排 + 父块）。"""
    from .index.bm25 import BM25Index
    from .index.embed import get_embedder
    from .index.vectorstore import VectorStore
    from .retrieve.hybrid import HybridRetriever, load_parent_store
    from .retrieve.rerank import get_reranker

    cfg = cfg or load_config()

    vs = VectorStore(cfg.resolve("paths.chroma"))
    if vs.count() == 0:
        raise RuntimeError(
            "向量库为空。请先执行:\n"
            "  python scripts/01_crawl.py\n"
            "  python scripts/02_parse.py\n"
            "  python scripts/03_build_index.py"
        )

    bm25_path = cfg.resolve("paths.bm25")
    if not Path(bm25_path).exists():
        raise RuntimeError(f"BM25 索引不存在: {bm25_path}，请先跑 scripts/03_build_index.py")
    bm25 = BM25Index.load(bm25_path)

    parents = load_parent_store(cfg.resolve("paths.processed") / "parents.jsonl")

    return HybridRetriever(
        vs,
        bm25,
        get_embedder(cfg),
        reranker=get_reranker(cfg),
        parent_store=parents,
        config=cfg.get_path("retrieve", {}),
    )
