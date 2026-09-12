"""QA 数据集构建的单元测试。

覆盖三条"判错了会静默教坏模型"的规则：

  1. `describe_gold`  报告里的评测集来源说明必须由数据推导。
     写死一句"人工撰写"，在 eval.gold_file 指向困难集时就是假陈述。
  2. `same_fact`      什么不能当负例。旧判据是字符重合度 > 0.85，
     会把"只差品种名"的难负例误删 —— 而那正是本任务要学的东西。
  3. `balance_metadata` 上下文里不能有"哪条带标题就选哪条"的捷径特征。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from derivrag.eval.dataset import describe_gold  # noqa: E402

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str, path: Path):
    """脚本文件名以数字开头，不能直接 import，按路径加载。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hardneg = _load("hardneg", _SCRIPTS / "05_mine_hard_negatives.py")
sftdata = _load("sftdata", _SCRIPTS / "08_build_sft_data.py")


# =====================================================================
# 1. 评测集来源说明
# =====================================================================
def test_describe_gold_human():
    gold = [{"mining_method": "human", "tier": 1}] * 3
    desc = describe_gold(gold)
    assert "交易所官方问答 3 条" in desc
    assert "tier-2" not in desc


def test_describe_gold_exposes_tier2():
    """困难集是模板生成 + LLM 改写的 tier-2，报告里必须写明，不能冒充人工金标。"""
    gold = [{"mining_method": "table_template+paraphrase", "tier": 2}] * 150
    desc = describe_gold(gold)
    assert "合约条款表模板 150 条" in desc
    assert "改写去泄漏" in desc
    assert "tier-2" in desc


def test_describe_gold_empty():
    assert describe_gold([]) == "来源未标注"


# =====================================================================
# 2. 难负例：什么是"同一个事实"
# =====================================================================
def _row(subject: str, field: str, value: str) -> dict:
    return {
        "chunk_type": "table_row",
        "section_path": [subject],
        "clause_id": field,
        "doc_title": f"{subject}合约基本条款",
        "text": f"{subject}的{field}：{value}",
    }


def test_same_field_different_product_is_a_valid_negative():
    """这是本任务最有价值的难负例，绝不能被剔除。

    旧的字符重合度判据在这对样本上是 0.893 > 0.85，会误删。
    """
    a = _row("沪深300ETF期权", "行权价格", "9个（1个平值合约、4个虚值合约、4个实值合约）")
    b = _row("深证100ETF期权", "行权价格", "9个（1个平值合约、4个虚值合约、4个实值合约）")
    assert hardneg.same_fact(a, b) is False


def test_same_product_same_field_is_not_a_negative():
    """《合约基本条款》与《上市交易通知》印的是同一张表，互为正确答案。"""
    a = _row("上证50ETF期权", "合约单位", "10000份")
    b = dict(_row("上证50ETF期权", "合约单位", "10000份"), doc_title="关于上证50ETF期权上市交易的通知")
    assert hardneg.same_fact(a, b) is True


def test_reprinted_clause_is_not_a_negative():
    """同一条款被另一份文档整段重印（修订版、制度汇编本）。"""
    body = "第十二条 会员应当按照本所规定报送交易结算数据，不得瞒报、漏报、迟报。"
    a = {"chunk_type": "clause", "text": body, "doc_title": "交易规则"}
    b = {"chunk_type": "clause", "text": body + "\n", "doc_title": "制度汇编"}
    assert hardneg.same_fact(a, b) is True


def test_short_generic_clause_does_not_swallow_negatives():
    """短通用句不能靠"互为子串"把正常负例判成同一事实。"""
    a = {"chunk_type": "clause", "text": "第三条 本细则由本所负责解释。", "doc_title": "结算细则"}
    b = {
        "chunk_type": "clause",
        "text": "第九条 会员应当在每日收市后核对结算结果，第三条 本细则由本所负责解释。",
        "doc_title": "交易细则",
    }
    assert hardneg.same_fact(a, b) is False


# =====================================================================
# 3. SFT 上下文：不能留捷径特征
# =====================================================================
def test_balance_metadata_strips_when_uneven():
    """一条带标题、一条不带，模型就能靠这个猜引用编号 —— 必须一起抹掉。"""
    records = [
        {"doc_title": "", "clause_id": "", "text": "干扰条款正文"},
        {"doc_title": "交易规则", "doc_no": "沪发〔2023〕1号", "clause_id": "第十二条", "text": "正例"},
    ]
    out = sftdata.balance_metadata(records)
    assert all(not r["doc_title"] and not r["clause_id"] for r in out)
    assert [r["text"] for r in out] == ["干扰条款正文", "正例"]


def test_balance_metadata_keeps_when_all_present():
    records = [
        {"doc_title": "交易规则", "clause_id": "第十二条", "text": "甲"},
        {"doc_title": "结算细则", "clause_id": "第三条", "text": "乙"},
    ]
    assert sftdata.balance_metadata(records) == records


def test_hardneg_records_requires_metadata():
    """旧版 05 产出的文件没有 neg_meta，必须整条放弃而不是造出裸文本干扰项。"""
    assert sftdata.hardneg_records({"neg": ["a", "b"]}, 2) == []
    assert sftdata.hardneg_records(None, 2) == []


def test_hardneg_records_pairs_text_with_metadata():
    hn = {
        "neg": ["条款甲正文", "条款乙正文"],
        "neg_meta": [
            {"doc_title": "交易规则", "clause_id": "第一条"},
            {"doc_title": "结算细则", "clause_id": "第二条"},
        ],
    }
    out = sftdata.hardneg_records(hn, 2)
    assert [r["text"] for r in out] == ["条款甲正文", "条款乙正文"]
    assert [r["doc_title"] for r in out] == ["交易规则", "结算细则"]
