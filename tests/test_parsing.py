"""解析与分块的单元测试。

重点覆盖那些踩过坑的地方 —— 每个测试都对应一个真实出现过的 bug。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from derivrag.chunk.clause import (  # noqa: E402
    _cn_to_int,
    _title_is_question,
    split_clauses,
    split_table_rows,
)
from derivrag.chunk.semantic import semantic_split, split_sentences  # noqa: E402
from derivrag.parse.clean import (  # noqa: E402
    clean_cjk_spacing,
    clean_text,
    extract_doc_no,
    extract_effective_date,
    infer_effective_status,
    looks_chinese,
)
from derivrag.parse.docfile import _format_number, _int_to_chinese  # noqa: E402
from derivrag.parse.html import extract_qa_pairs, trim_boilerplate  # noqa: E402
from derivrag.schema import EffectiveStatus, RawDocument  # noqa: E402


# =====================================================================
class TestCleaning:
    def test_cjk_digit_spacing(self):
        """PDF 抽取会在数字与中文之间插入空格，必须修掉。

        真实样本来自 CFFEX 股指期权合约交易细则。
        """
        assert clean_cjk_spacing("沪深300 股指期权") == "沪深300股指期权"
        assert clean_cjk_spacing("2019 年12 月14 日") == "2019年12月14日"
        assert clean_cjk_spacing("10000 份") == "10000份"

    def test_english_spacing_untouched(self):
        """纯英文的正常词距不能被误删 —— SGX/HKEX 规则全是英文。"""
        text = "The Exercise Price shall be determined by the Clearing House"
        assert clean_text(text, is_cjk=False) == text

    def test_paragraph_structure_preserved(self):
        assert "\n\n" in clean_text("第一段\n\n\n\n第二段")

    def test_looks_chinese(self):
        assert looks_chinese("合约标的为上证50ETF")
        assert not looks_chinese("Contract Specifications for Index Futures")


class TestMetadataExtraction:
    def test_doc_no(self):
        assert extract_doc_no("……上证发〔2023〕48号……") == "上证发〔2023〕48号"
        assert extract_doc_no("没有文号的正文") is None

    def test_doc_no_without_institution_prefix(self):
        """郑商所《期权交易管理办法》正文只写「〔2026〕68号」，不带机构简称。

        机构前缀曾是必需的，导致这份文档抽不到文号。全量扫描 190 份文档，
        这是唯一一份这种形态 —— 修它的收益就是 49→50，如实记录。
        """
        assert extract_doc_no("……〔2026〕68号……") == "〔2026〕68号"

    @pytest.mark.parametrize(
        "text",
        [
            "上证发〔2023〕48号",
            "深证上〔2022〕1146号",
            "中金所发〔2026〕18号",
            "上证发〔2023〕第48号",  # 带「第」
        ],
    )
    def test_doc_no_accepted_forms(self, text):
        """这四种形态在 190 份语料里真实出现过（共 102 处，全是前缀紧贴括号）。"""
        assert extract_doc_no(f"前置正文……{text}……后续") == text

    def test_doc_no_keeps_institution_when_spaced(self):
        """带空格时必须仍抓到机构简称，不能退化成「〔2023〕48号」。

        当前语料里不存在这种形态，这是防御性测试：把机构前缀改为可选之后，
        带空格的输入会让可选组不参与匹配，静默丢掉机构简称 —— 而机构简称是
        引用真值的一部分，丢了比返回 None 更糟（用户无法判断这个文号出自哪家）。
        PDF 文本抽取常在原文无空格处插入空格，所以这条不是假想问题。
        """
        for raw in ["上证发 〔2023〕48号", "上证发　〔 2023 〕 48 号"]:
            assert extract_doc_no(f"……{raw}……") == "上证发〔2023〕48号"

    @pytest.mark.parametrize(
        "text",
        [
            "合约标的：华夏上证科创板50ETF　合约类型：认购期权、认沽期权",
            "本所交易时间为每周一至周五上午9:30-11:30，下午13:00-15:00。",
            "Chapter 10 Contract Specifications",
            "第十二条 期权交易采用集合竞价和连续竞价两种方式。",
            "合约乘数：每点人民币100元",
        ],
    )
    def test_doc_no_absent_is_not_a_miss(self, text):
        """返回 None 是正确行为，不是漏检 —— 这条容易被误当成 bug。

        190 份文档里 140 份抽不到文号，实测这 140 份在**原始文件**里
        也没有文号（含中金所 25 份「通知」类，原始 HTML 里 0/25 有文号）。
        它们是合约条款表、常设规则手册、投资者问答与英文 rulebook，
        本身不带发文字号。所以 26% 是语料属性，不是抽取缺陷。
        「第十二条」是条款号、「Chapter 10」是章节号，都不得误判成文号。
        """
        assert extract_doc_no(text) is None

    def test_effective_date(self):
        assert extract_effective_date("本办法自2026年5月1日起施行。") == "2026-05-01"

    def test_repealed_detected(self):
        """已废止条款必须能识别，否则检索会拿失效规则回答问题。"""
        status = infer_effective_status("关于XX的通知（已废止）", "")
        assert status == EffectiveStatus.REPEALED.value

    def test_unknown_is_not_repealed(self):
        """大多数交易所文档不写时效性，未知状态必须保留（不能当废止过滤掉）。"""
        assert infer_effective_status("某某业务指南", "正文") != EffectiveStatus.REPEALED.value


# =====================================================================
class TestWordNumbering:
    """SSE 的 .doc 用 Word 自动编号，"第X条"不是正文文本，必须还原。"""

    @pytest.mark.parametrize(
        "n,expected",
        [(1, "一"), (10, "十"), (11, "十一"), (20, "二十"), (103, "一百零三"),
         (110, "一百一十"), (170, "一百七十")],
    )
    def test_int_to_chinese(self, n, expected):
        assert _int_to_chinese(n) == expected

    def test_format_number_by_numfmt(self):
        assert _format_number(3, "chineseCountingThousand") == "三"
        assert _format_number(3, "decimal") == "3"
        assert _format_number(3, "upperRoman") == "III"

    def test_cn_to_int_roundtrip(self):
        for n in (1, 7, 15, 42, 99, 156):
            assert _cn_to_int(_int_to_chinese(n)) == n


# =====================================================================
class TestClauseSplitting:
    def _doc(self, text: str, **kw) -> RawDocument:
        defaults = dict(
            doc_id="d1", title="测试规则", venue="TEST", lang="zh",
            source_url="http://example.com", retrieval_date="2026-08-14",
            fmt="pdf", text=text,
        )
        defaults.update(kw)
        return RawDocument(**defaults)

    def test_split_by_clause(self):
        text = (
            "第一条 为规范期权交易，制定本规则。\n"
            "第二条 本规则适用于所有会员单位与投资者。\n"
            "第三条 本规则未规定的，适用其他规定。\n"
            "第四条 本规则自发布之日起施行。"
        )
        chunks = split_clauses(self._doc(text))
        assert len(chunks) == 4
        assert chunks[0].clause_id == "第一条"
        assert chunks[3].clause_id == "第四条"
        assert chunks[0].chunk_type == "clause"

    def test_too_few_clauses_returns_empty(self):
        """少于 3 条时不认为是条文式文档，交给兜底策略。"""
        assert split_clauses(self._doc("第一条 只有一条内容而已。")) == []

    def test_clause_reference_not_split(self):
        """"第X款/项/目"不是条，不能当切分点。"""
        text = (
            "第一条 本规则依据第三款制定，参见第二项与第五目。\n"
            "第二条 内容内容内容内容内容内容内容。\n"
            "第三条 内容内容内容内容内容内容内容。"
        )
        chunks = split_clauses(self._doc(text))
        assert [c.clause_id for c in chunks] == ["第一条", "第二条", "第三条"]

    def test_table_rows_become_atomic_facts(self):
        """条款表每行切成一个 (品种,字段,值) 原子事实。"""
        doc = self._doc(
            "",
            title="上证50ETF期权合约基本条款",
            tables=[{
                "index": 0, "n_rows": 2, "n_cols": 2,
                "rows": [["合约单位", "10000份"], ["行权方式", "到期日行权（欧式）"]],
            }],
        )
        chunks = split_table_rows(doc)
        assert len(chunks) == 2
        assert chunks[0].chunk_type == "table_row"
        assert chunks[0].clause_id == "合约单位"
        assert "10000份" in chunks[0].text
        assert "上证50ETF期权" in chunks[0].text


# =====================================================================
class TestQAExtraction:
    def test_numbered_with_answer_prefix(self):
        """SSE 50ETF FAQ 版式：N.问题？ + 答：……"""
        text = (
            "1.什么是股票期权？\n\n答：股票期权是一种合约。\n\n"
            "2.谁可以参与交易？\n\n答：符合适当性要求的投资者。\n"
        )
        pairs = extract_qa_pairs(text)
        assert len(pairs) == 2
        assert pairs[0]["question"] == "什么是股票期权？"
        assert pairs[0]["answer"] == "股票期权是一种合约。"

    def test_numbered_without_answer_prefix(self):
        """SSE 熔断机制问答版式：没有"答："前缀，答案直接跟在问题后。

        这是实测中的第二种版式，只支持一种会漏掉 8 对金标问答。
        """
        text = "1.熔断如何影响期权？\n\n期权市场同步暂停交易。\n\n2.何时恢复？\n\n按结束时间分情况处理。\n"
        pairs = extract_qa_pairs(text)
        assert len(pairs) == 2
        assert pairs[0]["answer"] == "期权市场同步暂停交易。"

    def test_title_is_question(self):
        """CFFEX 版式：一页一问，标题即问题。"""
        assert _title_is_question("我在商品期货交易所开过户，还要考试吗？")
        assert _title_is_question("如何办理期权账户开户手续")
        assert not _title_is_question("交易规则")

    def test_trim_boilerplate(self):
        """CFFEX 正文里混着面包屑和上下篇导航，必须裁掉。"""
        text = "首页\n>\n服务\n>\n常见问答\n\n分享：\n\n微信二维码\n\n这里是真正的答案内容。\n\n上一篇：\n某某\n下一篇：\n某某"
        out = trim_boilerplate(text)
        assert out == "这里是真正的答案内容。"


# =====================================================================
class TestSemanticSplit:
    def test_chinese_sentence_split(self):
        """中文断句要按 。；！？ 切，不能用英文句号规则。"""
        s = split_sentences("第一句话。第二句话；第三句话！")
        assert len(s) == 3

    def test_short_text_not_split(self):
        text = "很短的一句话。"
        assert semantic_split(text, None, max_chars=512) == [text]

    def test_length_fallback_without_model(self):
        """embed_fn 为 None 时退化为长度切分，整条流水线仍可跑通。"""
        text = "。".join([f"这是第{i}个句子，内容足够长以便触发切分" for i in range(40)])
        pieces = semantic_split(text, None, target_chars=200, max_chars=300, min_chars=50)
        assert len(pieces) > 1
        assert all(len(p) <= 300 for p in pieces)

    def test_no_content_loss(self):
        """切分不能丢字 —— 条款文本的完整性是硬要求。"""
        text = "。".join([f"条款内容第{i}段落说明事项" for i in range(30)])
        pieces = semantic_split(text, None, target_chars=150, max_chars=250, min_chars=40)
        joined = "".join(pieces)
        # 有重叠句，所以用"原文每个片段都出现在结果里"来验证
        for seg in text.split("。"):
            if seg.strip():
                assert seg in joined


# =====================================================================
class TestFusion:
    def test_plain_rrf_matches_formula(self):
        """α=0 时必须精确退化为标准 RRF，这是消融对照的前提。"""
        from derivrag.retrieve.fusion import reciprocal_rank_fusion

        channels = {
            "a": [{"child_id": "x", "text": "", "score": 10.0, "metadata": {}}],
            "b": [{"child_id": "x", "text": "", "score": 0.5, "metadata": {}}],
        }
        hits = reciprocal_rank_fusion(channels, k=60, score_weight=0.0)
        expected = 0.5 / 61 + 0.5 / 61
        assert abs(hits[0].score - expected) < 1e-9

    def test_score_weight_changes_ranking(self):
        """score-aware 融合要能让"分数高但名次相同"的候选胜出。"""
        from derivrag.retrieve.fusion import reciprocal_rank_fusion

        channels = {
            "bm25": [
                {"child_id": "high", "text": "", "score": 40.0, "metadata": {}},
                {"child_id": "low", "text": "", "score": 1.0, "metadata": {}},
            ],
        }
        plain = reciprocal_rank_fusion(channels, k=60, score_weight=0.0)
        aware = reciprocal_rank_fusion(channels, k=60, score_weight=0.9)
        gap_plain = plain[0].score - plain[1].score
        gap_aware = aware[0].score - aware[1].score
        assert gap_aware > gap_plain

    def test_dedupe_by_parent(self):
        """同一条款的多个子块只保留最好的一个，否则 top-5 会被一条条款占满。"""
        from derivrag.retrieve.fusion import dedupe_hits
        from derivrag.schema import RetrievalHit

        hits = [
            RetrievalHit("c1", "a", 0.9, "fused", {"parent_id": "p1"}),
            RetrievalHit("c2", "b", 0.8, "fused", {"parent_id": "p1"}),
            RetrievalHit("c3", "c", 0.7, "fused", {"parent_id": "p2"}),
        ]
        out = dedupe_hits(hits)
        assert [h.child_id for h in out] == ["c1", "c3"]
