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
# 书名号里的文件名 ——「《上海证券交易所股票期权试点交易规则》第十二条」里的规则名。
# 少于 4 个字的不要：《》里塞短词多半不是文件名（如《说明》），会误扩大范围。
_DOC_TITLE_REF = re.compile(r"《([^》]{4,60})》")


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
        # 《规则名》-> doc_id 的反查索引，首次用到时才构建
        self._title_idx_cache: list[tuple[str, str]] | None = None
        cfg = config or {}
        self.bm25_top_k = cfg.get("bm25_top_k", 50)
        self.dense_top_k = cfg.get("dense_top_k", 50)
        self.keyword_top_k = cfg.get("keyword_top_k", 20)
        self.fusion_top_k = cfg.get("fusion_top_k", 50)
        self.rerank_top_k = cfg.get("rerank_top_k", 5)
        self.filter_repealed = cfg.get("filter_repealed", True)
        # 精确结构化匹配的置顶上限，见 _pin_exact
        self.max_pinned_exact = cfg.get("max_pinned_exact", 2)
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

    @property
    def _title_index(self) -> list[tuple[str, str]]:
        """(doc_title, doc_id) 列表，懒加载。用于把查询里的《规则名》反查成 doc_id。

        优先用已在内存里的父块存储（190 个文档、零额外 IO）；父块不可用时
        退回扫一遍向量库元数据。
        """
        if getattr(self, "_title_idx_cache", None) is not None:
            return self._title_idx_cache

        pairs: set[tuple[str, str]] = set()
        for p in (self.parents or {}).values():
            t, d = p.get("doc_title"), p.get("doc_id")
            if t and d:
                pairs.add((t, d))
        if not pairs:
            try:
                col = self.vs.client.get_collection(self.vs.collection_name)
                got = col.get(limit=self.vs.count(), include=["metadatas"])
                for m in got.get("metadatas", []) or []:
                    t, d = (m or {}).get("doc_title"), (m or {}).get("doc_id")
                    if t and d:
                        pairs.add((t, d))
            except Exception as e:  # noqa: BLE001 - 索引不可用时退化为不做文档约束
                logger.debug("构建标题索引失败: %s", e)

        self._title_idx_cache = sorted(pairs)
        logger.debug("标题索引就绪: %d 个文档", len(self._title_idx_cache))
        return self._title_idx_cache

    def _resolve_doc_ids(self, query: str) -> list[str]:
        """把查询里的《规则名》解析成 doc_id 列表。解析不出来返回空列表。

        用子串匹配而非相等：库里的 doc_title 往往是「关于发布《X》的通知（附件）」，
        而用户只写《X》，两者永远不相等。双向包含都算命中 —— 用户可能写全称
        也可能写得比标题更长（带了「的通知」）。
        """
        refs = _DOC_TITLE_REF.findall(query)
        if not refs:
            return []
        ids: list[str] = []
        for ref in refs:
            r = ref.strip()
            for title, doc_id in self._title_index:
                if (r in title or title in r) and doc_id not in ids:
                    ids.append(doc_id)
        if refs and not ids:
            logger.debug("查询里的书名号文件名未匹配到任何文档: %s", refs)
        return ids

    def _keyword(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """元数据精确匹配通路。

        从问题里抽出条号 / 文号 / 合约代码这类结构化线索，
        直接按元数据检索，绕过文本相似度。

        条号必须与文件名联合约束。库里「第十二条」有 59 个块，分布在几十份
        规则里 —— 只按 clause_id 查、再按 limit 截断，等于从 59 个同名条款里
        随机拿几个，目标文档那一条大概率不在其中（实测就不在）。用户写了
        《规则名》就是在告诉我们是哪一份，必须用上。
        """
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        doc_ids = self._resolve_doc_ids(query)

        def _scoped(where: dict[str, Any]) -> dict[str, Any]:
            """有文档线索时叠加 doc_id 约束。"""
            if not doc_ids:
                return where
            return {"$and": [where, {"doc_id": {"$in": doc_ids}}]}

        def _collect(where: dict[str, Any], *, exact: bool = False) -> None:
            try:
                for r in self.vs.query_metadata(where, limit=top_k):
                    if r["child_id"] not in seen:
                        seen.add(r["child_id"])
                        # exact=True 表示「用户点名了文档 + 条号，这一条就是他要的」，
                        # 不是一个相似度猜测。retrieve() 会保证它出现在结果里。
                        r["exact_match"] = exact
                        out.append(r)
            except Exception as e:  # noqa: BLE001 - 该通路失败不应拖垮整体检索
                logger.debug("关键词通路查询失败: %s", e)

        for m in _CLAUSE_REF.findall(query):
            before = len(out)
            _collect(_scoped({"clause_id": m}), exact=bool(doc_ids))
            if len(out) == before and doc_ids:
                # 该文档确实没有这一条 —— 退回全库查同名条号，好过什么都不返回。
                # 但这不算精确匹配：不知道用户要的是哪一份里的同名条号。
                logger.debug("文档 %s 内无 %s，退回全库匹配", doc_ids, m)
                _collect({"clause_id": m})
        # 文号本身就是文档的唯一标识，不需要书名号也算「点名了文档」。
        # 但只给文号不给条号时，是「这份文件说了什么」而非「这一条说了什么」——
        # 一个文号最多覆盖 117 个块（实测），全部置顶会挤爆 top_k，
        # 所以置顶数量在 _pin_exact 里有上限，这里只负责标记。
        for m in _DOC_NO_REF.findall(query):
            _collect({"doc_no": re.sub(r"\s+", "", m)}, exact=True)
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

        exact_ids: list[str] = []
        if "keyword" in channels:
            # 关键词通路只用原始查询 —— HyDE 生成的假文档里没有真实条号
            results["keyword"] = self._keyword(query, self.keyword_top_k)
            exact_ids = [
                r["child_id"] for r in results["keyword"] if r.get("exact_match")
            ]

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

        want = top_k or self.rerank_top_k
        pool = fused  # 截断前的候选池，置顶精确匹配时要从这里找回
        if use_rerank and self.reranker is not None:
            fused = self.reranker.rerank(query, fused, top_k=want)
        else:
            fused = fused[:want]

        return self._pin_exact(fused, pool, exact_ids, want)

    def _pin_exact(
        self,
        final: list[RetrievalHit],
        pool: list[RetrievalHit],
        exact_ids: list[str],
        want: int,
    ) -> list[RetrievalHit]:
        """把精确结构化匹配置顶，保证它不被融合或重排挤掉。

        为什么需要这一步：融合是加权投票，而 keyword 通路权重最低（0.20）。
        实测在 score_weight=0.3、k=60 下，只被 keyword 命中的块的**得分上限**是

            0.7·(0.20/61) + 0.3·0.20 = 0.0623

        而只被 dense 命中的块上限是 0.1557、bm25 是 0.0934 —— 也就是说
        keyword 单路命中**永远排不进 top-5**，无论它多准。这不是调参能解决的，
        提高 keyword 权重会连带抬高那些「同名条号但不知道是哪份文件」的弱匹配。

        真正的语义是：用户写「《X规则》第十二条」时，他不是在描述一个检索意图
        让各路去投票，而是在指定一条确定的记录。这种匹配不该参与投票。

        只在查询点名了文档（书名号文件名或文号）时触发，范围极窄。

        置顶数量有上限：一个文号最多覆盖 117 个块（实测），若用户只给文号
        不给条号，全部置顶会把 top_k 挤满同一份文件、挤掉其他相关条款。
        上限保证「该文档一定被代表」，剩下的名次交给融合决定。
        """
        if not exact_ids:
            return final
        present = {h.child_id for h in final}
        missing = [cid for cid in exact_ids if cid not in present]
        if not missing:
            return final

        by_id = {h.child_id: h for h in pool}
        pinned = [by_id[cid] for cid in missing if cid in by_id][: self.max_pinned_exact]
        if not pinned:
            return final
        logger.info(
            "精确匹配置顶 %d 条（融合/重排未保留它们）: %s",
            len(pinned),
            [h.metadata.get("clause_id") for h in pinned],
        )
        for h in pinned:
            h.component_ranks.setdefault("exact", 1)
        return (pinned + final)[:want]

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
