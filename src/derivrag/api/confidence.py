"""检索置信度评估 —— 让"我不确定"这件事能被表达出来。

合规场景里答错的代价远大于答不出：一条已废止的规则被自信地引用，
比一句"没有检索到足够依据"糟糕得多。但 RAG 的默认行为恰恰相反 ——
只要检索返回了东西，生成端就会用同样笃定的语气把答案组织出来，
用户从措辞上完全看不出这次的证据到底扎不扎实。

这里把 cross-encoder 的原始 logit 折算成两个可解释的信号：

    绝对置信度   sigmoid(top1 logit)，回答"最佳候选到底像不像答案"
    区分度       top1 与 top2 的 **logit 差**，回答"是不是有好几条长得一样好"

区分度必须在 logit 空间算，不能用概率差。sigmoid 在高分区严重饱和：
实测 logit 6.685 与 6.157（差 0.53）过 sigmoid 后是 0.99876 与 0.99790，
概率差只剩 0.0009。若按概率差设阈值，只要 top1 分数高就几乎必然低于阈值，
ambiguous 会恒为真。logit 差是对数几率比，不随绝对分数漂移。

第二个信号在本项目里尤其必要。困难集的设计前提就是"15 个品种的条款表
都有'合约单位'这一行"，此时 top1 的绝对分数可以很高，但它与 top2 的
差距极小 —— 真正的风险不是没检索到，而是检索到了一批近似条款、
并从中挑错了一条。只看绝对分数会漏掉这类错误。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

# BGE-Reranker-v2-M3 输出未经归一化的 logit（见 retrieve/rerank.py:134，
# 直接取 logits 而非 sigmoid），正值表示相关。过 sigmoid 后才是可比的概率。
HIGH_PROB = 0.80
MEDIUM_PROB = 0.50

# top1 与 top2 的 logit 差小于此值时，认为存在难以区分的近似条款。
# 1.0 对应约 2.7 倍的几率比 —— 低于这个量级，两条候选的相关性没有实质区别。
AMBIGUOUS_LOGIT_MARGIN = 1.0

STALE_STATUSES = ("已废止", "被修订")

LEVEL_HINTS = {
    "high": "检索证据充分。",
    "medium": "检索证据一般，建议核对下方引用原文后再采用。",
    "low": "检索置信度偏低，答案可能不准确，请务必人工核对原文条款。",
    "unknown": "未启用交叉编码器重排，无法给出校准的置信度。",
}

AMBIGUOUS_HINT = "存在多条高度相似的候选条款，答案可能张冠李戴，请确认品种与合约是否对应。"


def _sigmoid(x: float) -> float:
    """数值稳定的 sigmoid —— x 为较大负数时 exp(-x) 会溢出，故分支处理。"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def assess(
    citations: Sequence[dict[str, Any]], *, reranked: bool = True
) -> dict[str, Any]:
    """由重排分数评估这次检索的可信程度。

    入参是 `RAGResult.citations()` 的输出。返回值直接进 JSON 响应，
    前端据此决定显示哪一档警告条。
    """
    if not citations:
        return {
            "level": "low",
            "probability": 0.0,
            "margin": None,
            "ambiguous": False,
            "stale_citations": 0,
            "hint": "没有检索到相关条款，请换一种问法或确认语料是否覆盖该主题。",
        }

    logits = [
        c["component_scores"]["rerank"]
        for c in citations
        if "rerank" in c.get("component_scores", {})
    ]

    # 关掉重排时只剩 RRF 融合分，那是名次倒数的加权和，不具备跨查询可比性，
    # 硬套阈值只会给出误导性的"高置信"。这种情况如实说不知道。
    if not reranked or not logits:
        return {
            "level": "unknown",
            "probability": None,
            "margin": None,
            "ambiguous": False,
            "stale_citations": _count_stale(citations),
            "hint": LEVEL_HINTS["unknown"],
        }

    top = _sigmoid(logits[0])
    margin = round(logits[0] - logits[1], 4) if len(logits) > 1 else None
    ambiguous = margin is not None and margin < AMBIGUOUS_LOGIT_MARGIN

    if top >= HIGH_PROB:
        level = "high"
    elif top >= MEDIUM_PROB:
        level = "medium"
    else:
        level = "low"

    # 低置信本身已经是最强提示，不必再被"疑似近似条款"覆盖
    hint = AMBIGUOUS_HINT if ambiguous and level != "low" else LEVEL_HINTS[level]

    stale = _count_stale(citations)
    if stale:
        hint += f" 另有 {stale} 条引用处于已废止或被修订状态，注意时效性。"

    return {
        "level": level,
        "probability": round(top, 4),
        "margin": margin,
        "ambiguous": ambiguous,
        "stale_citations": stale,
        "hint": hint,
    }


def _count_stale(citations: Sequence[dict[str, Any]]) -> int:
    return sum(1 for c in citations if c.get("effective_status") in STALE_STATUSES)
