"""检索评测指标。

金标 QA 对里记录了答案所在的 parent_id，所以"命中"的定义很干净：
返回的候选中是否包含那个 parent_id。这比用文本重叠判定可靠得多。

指标含义：
    Recall@k   前 k 条里是否包含正确条款（二值，对单答案场景等价于 Hit@k）
    MRR        正确条款排名的倒数，衡量"排得多靠前"
    nDCG@k     考虑位置折损的增益，与 MRR 在单答案场景高度相关但更平滑
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class RetrievalMetrics:
    n: int = 0
    recall: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: dict[int, float] = field(default_factory=dict)
    mean_latency: float = 0.0
    p95_latency: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "recall": {f"@{k}": round(v, 4) for k, v in sorted(self.recall.items())},
            "mrr": round(self.mrr, 4),
            "ndcg": {f"@{k}": round(v, 4) for k, v in sorted(self.ndcg.items())},
            "mean_latency_s": round(self.mean_latency, 3),
            "p95_latency_s": round(self.p95_latency, 3),
        }


def rank_of(retrieved_ids: Sequence[str], gold_id: str) -> int | None:
    """gold 在候选里的名次（1-based），未命中返回 None。"""
    for i, rid in enumerate(retrieved_ids, 1):
        if rid == gold_id:
            return i
    return None


def compute_metrics(
    ranks: Sequence[int | None],
    latencies: Sequence[float],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> RetrievalMetrics:
    """由每个查询的命中名次计算汇总指标。"""
    n = len(ranks)
    m = RetrievalMetrics(n=n)
    if n == 0:
        return m

    for k in ks:
        m.recall[k] = sum(1 for r in ranks if r is not None and r <= k) / n
        # 单条正确答案时 IDCG = 1，nDCG 即 1/log2(rank+1)
        m.ndcg[k] = (
            sum(1.0 / math.log2(r + 1) for r in ranks if r is not None and r <= k) / n
        )

    m.mrr = sum(1.0 / r for r in ranks if r is not None) / n

    if latencies:
        ordered = sorted(latencies)
        m.mean_latency = sum(ordered) / len(ordered)
        m.p95_latency = ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)]
    return m


_NUM_RE = None


def numeric_fidelity(answer: str, gold_answer: str) -> float | None:
    """金标答案里的数值有多少被生成答案原样保留。

    这是条款问答里**最该看的指标**。用户问"合约单位是多少"，答案错一个
    数量级就是灾难；而语义相似度、关键词覆盖率对数值错误都不敏感。

    与 keyword_match_score 的区别：后者会惩罚"答得比金标更全"的回答
    （实测有一条答案给出了 4 条带引用的要点，比金标那一条还完整，
    关键词覆盖率却只有 0.14）。数值保真度不受这个影响 ——
    它只问"该出现的数字出现了没有"。

    金标答案里没有任何数值时返回 None（该样本不参与统计）。
    """
    import re

    # 百分比、小数、带千分位的整数、时间点
    pattern = r"\d+(?:[,，]\d{3})*(?:\.\d+)?%?|\d{1,2}:\d{2}"
    gold_nums = re.findall(pattern, gold_answer)
    if not gold_nums:
        return None
    norm = lambda s: s.replace(",", "").replace("，", "")  # noqa: E731
    answer_norm = norm(answer)
    hit = sum(1 for n in set(gold_nums) if norm(n) in answer_norm)
    return hit / len(set(gold_nums))


def keyword_match_score(answer: str, gold_answer: str, *, min_len: int = 2) -> float:
    """关键词覆盖率 —— 生成答案里覆盖了金标答案多少关键片段。

    这是 RAGAS 之外的一个轻量、确定性的补充指标：不需要调用 LLM，
    对数值型答案（保证金公式、合约单位）尤其可靠。
    做法是把金标答案切成数字/术语片段，看生成答案里出现了多少。
    """
    import re

    if not gold_answer:
        return 0.0
    # 数字（含小数、百分比）与连续中文/英文词
    pieces = re.findall(r"\d+(?:\.\d+)?%?|[一-鿿]{2,}|[A-Za-z]{3,}", gold_answer)
    pieces = [p for p in pieces if len(p) >= min_len]
    if not pieces:
        return 0.0
    hit = sum(1 for p in set(pieces) if p in answer)
    return hit / len(set(pieces))
