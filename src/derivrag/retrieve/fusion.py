"""多路召回融合。

提供两种融合方式，评测时作为消融对照：

  plain_rrf   标准 RRF：score = Σ w_i / (k + rank_i)
              这也是 langchain `EnsembleRetriever.weighted_reciprocal_rank`
              的实现（k=60）。它**只看名次、完全丢弃原始分数**。

  score_rrf   本项目自研：在 RRF 的基础上混入归一化后的原始分数
              score = (1-α)·Σ w_i/(k+rank_i) + α·Σ w_i·norm(s_i)

为什么要自研这一步：

纯 RRF 把"BM25 得分 38.5 的第 1 名"和"BM25 得分 2.1 的第 1 名"同等对待，
但在条款检索里这两者的置信度天差地别 —— 前者往往是术语精确命中
（"合约乘数"这种词在语料里只出现在少数几个块中），后者基本是噪声。
反过来，稠密向量的余弦分数分布很平（同一批候选常常挤在 0.6~0.7），
如果只按分数融合又会被 BM25 的长尾大分值压制。

所以两者都要：rank 部分保证鲁棒性（不受不同通路量纲影响），
score 部分保留"这一路到底有多确信"的信息。α 由配置控制，
α=0 时精确退化为标准 RRF，方便做 A/B 对比。
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

from ..schema import RetrievalHit

logger = logging.getLogger(__name__)


def _min_max_normalize(scores: Sequence[float]) -> list[float]:
    """把一路召回的分数归一到 [0,1]。

    全部相同时返回全 1（说明这一路没有区分度，交给 rank 部分决定）。
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def reciprocal_rank_fusion(
    channels: dict[str, list[dict[str, Any]]],
    *,
    weights: dict[str, float] | None = None,
    k: int = 60,
    score_weight: float = 0.0,
    top_k: int = 50,
) -> list[RetrievalHit]:
    """融合多路召回结果。

    参数:
        channels     {通路名: [{child_id, text, metadata, score}, ...]}，各路已按分数降序
        weights      各通路权重，缺省为等权
        k            RRF 平滑常数（标准取 60）
        score_weight α，归一化分数的占比。0 = 标准 RRF
        top_k        返回条数

    返回:
        按融合分降序的 RetrievalHit 列表，每条都带 component_scores /
        component_ranks，便于在 UI 上展示"这一条是被哪几路召回的"。
    """
    weights = weights or {name: 1.0 for name in channels}
    total_w = sum(weights.get(name, 0.0) for name in channels) or 1.0

    fused: dict[str, RetrievalHit] = {}

    for channel, results in channels.items():
        if not results:
            continue
        w = weights.get(channel, 0.0) / total_w
        if w <= 0:
            continue

        normed = _min_max_normalize([r.get("score", 0.0) for r in results])

        for rank, (r, norm_score) in enumerate(zip(results, normed), start=1):
            cid = r["child_id"]
            hit = fused.get(cid)
            if hit is None:
                hit = RetrievalHit(
                    child_id=cid,
                    text=r.get("text", ""),
                    score=0.0,
                    source="fused",
                    metadata=r.get("metadata", {}) or {},
                )
                fused[cid] = hit

            rank_part = w / (k + rank)
            score_part = w * norm_score
            hit.score += (1.0 - score_weight) * rank_part + score_weight * score_part
            hit.component_scores[channel] = float(r.get("score", 0.0))
            hit.component_ranks[channel] = rank

    ranked = sorted(fused.values(), key=lambda h: h.score, reverse=True)
    return ranked[:top_k]


def merge_channels(
    channels: dict[str, list[dict[str, Any]]],
    *,
    method: str = "score_rrf",
    weights: dict[str, float] | None = None,
    k: int = 60,
    score_weight: float = 0.3,
    top_k: int = 50,
) -> list[RetrievalHit]:
    """按配置选择融合方式。method='plain_rrf' 时强制 α=0。"""
    alpha = 0.0 if method == "plain_rrf" else score_weight
    return reciprocal_rank_fusion(
        channels, weights=weights, k=k, score_weight=alpha, top_k=top_k
    )


def dedupe_hits(hits: Iterable[RetrievalHit], *, by: str = "parent_id") -> list[RetrievalHit]:
    """按父块去重 —— 同一条款的多个子块只保留得分最高的那个。

    不去重的话，一条长条款会用它的 5 个子块占满 top-5，
    重排和生成阶段拿到的其实只有一条信息。
    """
    seen: set[str] = set()
    out: list[RetrievalHit] = []
    for h in hits:
        key = h.metadata.get(by) or h.child_id
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out
