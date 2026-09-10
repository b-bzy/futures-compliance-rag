"""三路混合检索。

    BM25      —— 中文分词 + 术语词典。负责精确术语命中（"备兑开仓""行权价格间距"）
    Dense     —— bge-m3 稠密向量。负责语义泛化（"到期后怎么结算" -> "交割方式"条款）
    Keyword   —— 元数据精确匹配。负责结构化定位（合约代码、文号、"第X条"）

关键词通路常被忽略，但在条款检索里非常有用：用户问"《期权交易管理办法》
第十二条说了什么"时，前两路都只能靠文本相似度碰运气，而这一路能直接
按 clause_id + doc_title 精确定位。

三路结果经自研 score-aware RRF 融合（见 fusion.py），再交给
cross-encoder 精排。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ..schema import EffectiveStatus, RetrievalHit
from .fusion import dedupe_hits, merge_channels

logger = logging.getLogger(__name__)

# 用户问题里可能出现的结构化线索
_CLAUSE_REF = re.compile(r"第[一二三四五六七八九十百千零〇两\d]{1,10}条")
_DOC_NO_REF = re.compile(r"[一-鿿]{2,8}[〔\[（(]\s*(?:19|20)\d{2}\s*[〕\]）)]\s*第?\s*\d+\s*号")
_CONTRACT_CODE = re.compile(r"\b\d{6}\b|\b[A-Z]{2,3}\d{3,4}\b")


class HybridRetriever:
    """三路召回 + 融合 + 重排的完整检索器。"""

    def __init__(
        self,
        vectorstore,
        bm25_index,
        embedder,
        *,
        reranker=None,
        parent_store: dict[str, dict] | None = None,
        config: dict | None = None,
    ) -> None:
        self.vs = vectorstore
        self.bm25 = bm25_index
        self.embedder = embedder
        self.reranker = reranker
        self.parents = parent_store or {}
        cfg = config or {}
        self.bm25_top_k = cfg.get("bm25_top_k", 50)
        self.dense_top_k = cfg.get("dense_top_k", 50)
        self.keyword_top_k = cfg.get("keyword_top_k", 20)
        self.fusion_top_k = cfg.get("fusion_top_k", 50)
        self.rerank_top_k = cfg.get("rerank_top_k", 5)
        self.filter_repealed = cfg.get("filter_repealed", True)
        self.parent_expand = cfg.get("parent_expand", True)
        self.fusion_cfg = cfg.get("fusion", {})

    # ---------------------------------------------------------------
    def _where(self, extra: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """构造 Chroma 元数据过滤条件。

        默认排除已废止条款 —— 否则会拿失效规则回答问题。
        注意只排除**明确标记为已废止**的，"未知"状态一律保留：
        大多数交易所文档不写时效性字段，把未知当废止会丢掉绝大部分语料。
        """
        conditions: list[dict[str, Any]] = []
        if self.filter_repealed:
            conditions.append(
                {"effective_status": {"$ne": EffectiveStatus.REPEALED.value}}
            )
        for k, v in (extra or {}).items():
            conditions.append({k: v})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    # ---------------------------------------------------------------
    def _dense(self, query: str, top_k: int, where) -> list[dict[str, Any]]:
        vec = self.embedder.encode_dense([query])[0]
        return self.vs.search(vec, top_k=top_k, where=where)

    def _bm25(self, query: str, top_k: int) -> list[dict[str, Any]]:
        results = self.bm25.search(query, top_k=top_k * 2)
        if self.filter_repealed:
            results = [
                r
                for r in results
                if r["metadata"].get("effective_status") != EffectiveStatus.REPEALED.value
            ]
        return results[:top_k]

    def _keyword(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """元数据精确匹配通路。

        从问题里抽出条号 / 文号 / 合约代码这类结构化线索，
        直接按元数据检索，绕过文本相似度。
        """
        out: list[dict[str, Any]] = []
        seen: set[str] = set()

        def _collect(where: dict[str, Any]) -> None:
            try:
                for r in self.vs.query_metadata(where, limit=top_k):
                    if r["child_id"] not in seen:
                        seen.add(r["child_id"])
                        out.append(r)
            except Exception as e:  # noqa: BLE001 - 该通路失败不应拖垮整体检索
                logger.debug("关键词通路查询失败: %s", e)

        for m in _CLAUSE_REF.findall(query):
            _collect({"clause_id": m})
        for m in _DOC_NO_REF.findall(query):
            _collect({"doc_no": re.sub(r"\s+", "", m)})
        for m in _CONTRACT_CODE.findall(query):
            # 合约代码在正文里，用 BM25 补一刀更实际
            for r in self.bm25.search(m, top_k=top_k):
                if r["child_id"] not in seen:
                    seen.add(r["child_id"])
                    out.append(r)

        if self.filter_repealed:
            out = [
                r
                for r in out
                if r["metadata"].get("effective_status") != EffectiveStatus.REPEALED.value
            ]
        return out[:top_k]

    # ---------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        *,
        channels: tuple[str, ...] = ("bm25", "dense", "keyword"),
        use_rerank: bool = True,
        top_k: int | None = None,
        extra_filter: dict[str, Any] | None = None,
        expand_queries: list[str] | None = None,
    ) -> list[RetrievalHit]:
        """执行检索。

        参数:
            channels        启用哪些召回通路（消融实验用）
            use_rerank      是否走 cross-encoder 精排
            expand_queries  HyDE 假设文档或改写后的查询，与原查询一起召回
        """
        where = self._where(extra_filter)
        results: dict[str, list[dict[str, Any]]] = {}

        # 主查询 + 扩展查询都参与召回，再一起融合
        queries = [query] + list(expand_queries or [])

        if "dense" in channels:
            merged: dict[str, dict] = {}
            for q in queries:
                for r in self._dense(q, self.dense_top_k, where):
                    prev = merged.get(r["child_id"])
                    if prev is None or r["score"] > prev["score"]:
                        merged[r["child_id"]] = r
            results["dense"] = sorted(
                merged.values(), key=lambda r: r["score"], reverse=True
            )[: self.dense_top_k]

        if "bm25" in channels:
            merged = {}
            for q in queries:
                for r in self._bm25(q, self.bm25_top_k):
                    prev = merged.get(r["child_id"])
                    if prev is None or r["score"] > prev["score"]:
                        merged[r["child_id"]] = r
            results["bm25"] = sorted(
                merged.values(), key=lambda r: r["score"], reverse=True
            )[: self.bm25_top_k]

        if "keyword" in channels:
            # 关键词通路只用原始查询 —— HyDE 生成的假文档里没有真实条号
            results["keyword"] = self._keyword(query, self.keyword_top_k)

        fused = merge_channels(
            results,
            method=self.fusion_cfg.get("method", "score_rrf"),
            weights=self.fusion_cfg.get("weights"),
            k=self.fusion_cfg.get("k", 60),
            score_weight=self.fusion_cfg.get("score_weight", 0.3),
            top_k=self.fusion_top_k,
        )

        # 同一条款的多个子块只留最好的那个
        fused = dedupe_hits(fused, by="parent_id")

        if self.parent_expand:
            self._attach_parents(fused)

        if use_rerank and self.reranker is not None:
            fused = self.reranker.rerank(
                query, fused, top_k=top_k or self.rerank_top_k
            )
        else:
            fused = fused[: (top_k or self.rerank_top_k)]

        return fused

    # ---------------------------------------------------------------
    def _attach_parents(self, hits: list[RetrievalHit]) -> None:
        """父子索引的"回溯"一步：命中子块 -> 取回完整条款作为生成上下文。"""
        for h in hits:
            pid = h.metadata.get("parent_id")
            if pid and pid in self.parents:
                h.parent_text = self.parents[pid].get("text", "")

    def context_for(self, hits: list[RetrievalHit], max_chars: int = 12000) -> str:
        """把命中结果拼成带引用编号的上下文文本，供生成使用。"""
        blocks: list[str] = []
        used = 0
        for i, h in enumerate(hits, 1):
            body = h.parent_text or h.text
            meta = h.metadata
            header_bits = [f"[{i}]"]
            if meta.get("doc_title"):
                header_bits.append(meta["doc_title"])
            if meta.get("doc_no"):
                header_bits.append(meta["doc_no"])
            if meta.get("clause_id"):
                header_bits.append(meta["clause_id"])
            if meta.get("effective_status") and meta["effective_status"] != "未知":
                header_bits.append(f"[{meta['effective_status']}]")
            block = f"{' · '.join(header_bits)}\n{body}"
            if used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n---\n\n".join(blocks)


def load_parent_store(path: str | Path) -> dict[str, dict]:
    """把 parents.jsonl 读成 {parent_id: record} 供回溯使用。"""
    import json

    store: dict[str, dict] = {}
    p = Path(path)
    if not p.exists():
        logger.warning("父块文件不存在: %s", p)
        return store
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            store[rec["parent_id"]] = rec
    logger.info("已载入 %d 个父块", len(store))
    return store
