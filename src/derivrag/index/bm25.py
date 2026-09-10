"""BM25 稀疏检索（中文分词）。

中文没有天然空格，直接用 rank_bm25 的默认按空格切会退化成整句一个 token，
BM25 完全失效。所以必须先分词。

金融领域词典是这里的关键 —— jieba 默认会把"行权价格间距"切成
"行权/价格/间距"、把"备兑开仓"切成"备兑/开仓"，导致这些术语在 BM25 里
无法作为整体匹配。加载自定义词典后这些术语保持完整，这也是混合检索里
BM25 通路真正能补上稠密向量短板的地方（精确术语命中）。
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Any, Sequence

import jieba
from rank_bm25 import BM25Okapi

from ..schema import ChildChunk

logger = logging.getLogger(__name__)

# 衍生品/期权领域术语。这些词 jieba 默认都会切碎。
FINANCE_TERMS = [
    "行权价格", "行权价格间距", "行权方式", "行权日", "行权指令", "到期日", "交收日",
    "合约标的", "合约单位", "合约乘数", "合约类型", "合约面值", "合约到期月份",
    "认购期权", "认沽期权", "看涨期权", "看跌期权", "欧式期权", "美式期权",
    "备兑开仓", "备兑平仓", "买入开仓", "买入平仓", "卖出开仓", "卖出平仓",
    "义务仓", "权利仓", "持仓限额", "限仓制度", "强行平仓", "强制平仓",
    "开仓保证金", "维持保证金", "保证金比例", "结算价格", "前结算价格",
    "涨跌停价格", "最大涨幅", "最大跌幅", "熔断机制", "集合竞价", "连续竞价",
    "实物交割", "现金交割", "平值合约", "虚值合约", "实值合约", "虚值", "实值",
    "做市商", "主做市商", "一般做市商", "流动性服务商",
    "投资者适当性", "适当性管理", "风险揭示书", "经纪合同",
    "股指期权", "个股期权", "股票期权", "商品期权", "指数期权", "期货期权",
    "沪深300", "上证50", "中证500", "中证1000", "科创50", "创业板", "深证100",
    "最小报价单位", "最小变动价位", "申报单位", "报价单位",
    "熔断", "限价指令", "市价指令", "组合策略", "证券保证金",
    "delta值", "希腊字母", "隐含波动率", "标的证券", "标的指数",
]


def _load_dict() -> None:
    """把领域术语注入 jieba。幂等。"""
    if getattr(_load_dict, "_done", False):
        return
    for term in FINANCE_TERMS:
        jieba.add_word(term, freq=10000)
    _load_dict._done = True  # type: ignore[attr-defined]
    logger.debug("已加载 %d 个金融术语到 jieba 词典", len(FINANCE_TERMS))


# 去掉纯符号 token，它们对 BM25 没有区分度还会拖慢检索
_JUNK = re.compile(r"^[\s\W_]+$")


def tokenize(text: str) -> list[str]:
    """中英混合分词。中文走 jieba，英文按词切并小写化。"""
    _load_dict()
    tokens: list[str] = []
    for tok in jieba.lcut(text):
        tok = tok.strip()
        if not tok or _JUNK.match(tok):
            continue
        # 英文统一小写，中文原样
        tokens.append(tok.lower() if tok.isascii() else tok)
    return tokens


class BM25Index:
    """BM25 索引 —— 可持久化到磁盘，避免每次启动重新分词。"""

    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self.child_ids: list[str] = []
        self.texts: list[str] = []
        self.metadatas: list[dict[str, Any]] = []

    # ---------------------------------------------------------------
    def build(self, chunks: Sequence[ChildChunk]) -> None:
        """从子块构建索引。"""
        self.child_ids = [c.child_id for c in chunks]
        self.texts = [c.text for c in chunks]
        self.metadatas = [c.to_metadata() for c in chunks]

        logger.info("BM25 分词中（%d 个块）…", len(chunks))
        corpus = [tokenize(t) for t in self.texts]
        # 全空的块会让 BM25Okapi 计算 avgdl 时除零
        corpus = [c if c else ["　"] for c in corpus]
        self.bm25 = BM25Okapi(corpus)
        logger.info("BM25 索引构建完成")

    # ---------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        top_k: int = 50,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """检索。where 为简单的元数据等值过滤。"""
        if self.bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        # 先按分数排序，再做元数据过滤。多取一些以免过滤后不够。
        n = top_k * 4 if where else top_k
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]

        out: list[dict[str, Any]] = []
        for i in idx:
            if scores[i] <= 0:
                continue
            meta = self.metadatas[i]
            if where and not all(meta.get(k) == v for k, v in where.items()):
                continue
            out.append(
                {
                    "child_id": self.child_ids[i],
                    "text": self.texts[i],
                    "metadata": meta,
                    "score": float(scores[i]),
                }
            )
            if len(out) >= top_k:
                break
        return out

    # ---------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "child_ids": self.child_ids,
                    "texts": self.texts,
                    "metadatas": self.metadatas,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("BM25 索引已保存 -> %s", path)

    @classmethod
    def load(cls, path: str | Path) -> BM25Index:
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls()
        idx.bm25 = data["bm25"]
        idx.child_ids = data["child_ids"]
        idx.texts = data["texts"]
        idx.metadatas = data["metadatas"]
        logger.info("BM25 索引已载入（%d 个块）", len(idx.child_ids))
        return idx
