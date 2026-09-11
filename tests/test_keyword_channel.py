"""关键词通路的文档维度约束与精确匹配置顶。

对应一个真实缺陷：用户问「《上海证券交易所股票期权试点交易规则》第十二条
规定了什么？」时系统答「无法确定」，而那一条明明在索引里。

两个独立成因，都要修：

1. `_keyword` 只按 clause_id 查，不用查询里的《规则名》。库里「第十二条」
   有 59 个块分布在几十份规则里，按 limit 截断等于随机取几个 —— 实测目标
   文档那一条不在返回集里。
2. 即使通路返回了正确那一条，它也进不了最终结果。keyword 权重最低(0.20)，
   在 score_weight=0.3 / k=60 下单路命中的得分上限是
   0.7·(0.20/61) + 0.3·0.20 = 0.0623，而 dense 单路上限 0.1557 ——
   无论多准都排不进 top-5。这是结构性的，调 keyword 权重会连带抬高那些
   「同名条号但不知道是哪份文件」的弱匹配。

全部用 stub，不加载模型、不访问索引。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from derivrag.retrieve.hybrid import HybridRetriever  # noqa: E402


# =====================================================================
# 三份文档都有「第十二条」，模拟真实语料里的同名条号
_CORPUS = [
    ("c-sse-12", "sse_att0", "关于发布《上海证券交易所股票期权试点交易规则》的通知（附件）",
     "第十二条", "第十二条合约标的发生除权、除息的…", "上证发〔2015〕23号"),
    ("c-cffex-12", "cffex_rule", "中国金融期货交易所交易规则",
     "第十二条", "第十二条会员应当…", ""),
    ("c-szse-12", "szse_guide", "深圳证券交易所期权业务指南",
     "第十二条", "第十二条投资者应当…", ""),
    ("c-sse-13", "sse_att0", "关于发布《上海证券交易所股票期权试点交易规则》的通知（附件）",
     "第十三条", "第十三条本所可以…", "上证发〔2015〕23号"),
]


class _FakeVS:
    """只实现 query_metadata，支持 clause_id / doc_no / $and / $in。"""

    def __init__(self) -> None:
        self.wheres: list[dict] = []

    def query_metadata(self, where, *, limit=50):  # noqa: ANN001, ANN201
        self.wheres.append(where)
        rows = []
        for cid, doc_id, title, clause, text, doc_no in _CORPUS:
            meta = {
                "doc_id": doc_id, "doc_title": title, "clause_id": clause,
                "doc_no": doc_no, "parent_id": f"p-{cid}", "effective_status": "未知",
            }
            if self._match(where, meta):
                rows.append({"child_id": cid, "text": text, "metadata": meta, "score": 1.0})
        return rows[:limit]

    def _match(self, where, meta) -> bool:
        if "$and" in where:
            return all(self._match(c, meta) for c in where["$and"])
        for k, v in where.items():
            if isinstance(v, dict) and "$in" in v:
                if meta.get(k) not in v["$in"]:
                    return False
            elif meta.get(k) != v:
                return False
        return True


class _FakeBM25:
    def search(self, q, top_k=10):  # noqa: ANN001, ANN201
        return []


def _retriever(**cfg):
    parents = {
        f"p-{cid}": {"doc_id": doc_id, "doc_title": title, "text": text}
        for cid, doc_id, title, _c, text, _n in _CORPUS
    }
    return HybridRetriever(
        _FakeVS(), _FakeBM25(), embedder=None, parent_store=parents, config=cfg
    )


# =====================================================================
class TestDocScoping:
    def test_rule_name_resolved_to_doc_ids(self):
        r = _retriever()
        ids = r._resolve_doc_ids("《上海证券交易所股票期权试点交易规则》第十二条规定了什么？")
        assert ids == ["sse_att0"], "应解析出目标文档"

    def test_substring_match_both_directions(self):
        """库里标题是「关于发布《X》的通知（附件）」，用户只写《X》，永不相等。"""
        r = _retriever()
        assert r._resolve_doc_ids("《中国金融期货交易所交易规则》第十条") == ["cffex_rule"]

    def test_short_bracket_content_ignored(self):
        """《说明》这类短词多半不是文件名，扩大范围反而有害。"""
        r = _retriever()
        assert r._resolve_doc_ids("《说明》第十二条") == []

    def test_unmatched_title_falls_back_to_no_scoping(self):
        r = _retriever()
        assert r._resolve_doc_ids("《不存在的某某管理办法》第十二条") == []

    def test_scoped_query_returns_only_target_document(self):
        """核心：带规则名时只返回那份文件的第十二条，不是三份里随机取。"""
        r = _retriever()
        out = r._keyword("《上海证券交易所股票期权试点交易规则》第十二条规定了什么？", top_k=20)
        assert len(out) == 1
        assert out[0]["child_id"] == "c-sse-12"
        assert out[0]["exact_match"] is True

    def test_unscoped_query_returns_all_homonyms(self):
        """不带规则名时无法区分，返回全部同名条号（原有行为），且不标精确。"""
        r = _retriever()
        out = r._keyword("第十二条规定了什么？", top_k=20)
        assert {h["child_id"] for h in out} == {"c-sse-12", "c-cffex-12", "c-szse-12"}
        assert all(not h["exact_match"] for h in out)

    def test_clause_absent_in_named_doc_falls_back(self):
        """点名的文件里没有这一条时退回全库，好过什么都不返回。"""
        r = _retriever()
        out = r._keyword("《深圳证券交易所期权业务指南》第十三条规定了什么？", top_k=20)
        assert out, "不应返回空"
        assert out[0]["child_id"] == "c-sse-13"
        assert not out[0]["exact_match"], "退回结果不是精确匹配"

    def test_doc_no_alone_counts_as_naming_a_document(self):
        """文号本身就是文档唯一标识，不需要书名号。"""
        r = _retriever()
        out = r._keyword("上证发〔2015〕23号 说了什么？", top_k=20)
        assert {h["child_id"] for h in out} == {"c-sse-12", "c-sse-13"}
        assert all(h["exact_match"] for h in out)


class TestPinExact:
    """精确匹配必须进最终结果 —— 否则修好通路也是白修，用户看不到变化。"""

    def _hit(self, cid):
        from derivrag.schema import RetrievalHit

        return RetrievalHit(child_id=cid, text="", score=0.1, source="fused", metadata={})

    def test_pinned_when_fusion_drops_it(self):
        r = _retriever()
        final = [self._hit(f"other-{i}") for i in range(5)]
        pool = final + [self._hit("c-sse-12")]
        out = r._pin_exact(final, pool, ["c-sse-12"], want=5)
        assert out[0].child_id == "c-sse-12", "精确匹配应置顶"
        assert len(out) == 5, "总数不应超过 want"

    def test_no_pin_when_already_present(self):
        r = _retriever()
        final = [self._hit("c-sse-12"), self._hit("other")]
        out = r._pin_exact(final, final, ["c-sse-12"], want=5)
        assert out == final, "已在结果里就不该改动顺序"

    def test_pin_count_capped(self):
        """一个文号最多覆盖 117 个块，全置顶会挤爆 top_k。"""
        r = _retriever(max_pinned_exact=2)
        final = [self._hit(f"other-{i}") for i in range(5)]
        extra = [self._hit(f"ex-{i}") for i in range(6)]
        out = r._pin_exact(final, final + extra, [h.child_id for h in extra], want=5)
        assert sum(1 for h in out if h.child_id.startswith("ex-")) == 2

    def test_no_exact_ids_is_noop(self):
        r = _retriever()
        final = [self._hit("a"), self._hit("b")]
        assert r._pin_exact(final, final, [], want=5) is final

    def test_pinned_hit_marked_in_component_ranks(self):
        """界面要能显示「这一条是精确匹配置顶的」，而非某路召回的。"""
        r = _retriever()
        final = [self._hit("other")]
        pool = final + [self._hit("c-sse-12")]
        out = r._pin_exact(final, pool, ["c-sse-12"], want=5)
        assert out[0].component_ranks.get("exact") == 1
