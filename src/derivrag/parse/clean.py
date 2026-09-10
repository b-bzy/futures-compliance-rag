"""文本清洗。

目标是"保持文本完整性"——只去掉抽取过程引入的噪声，不改动条款原文的
任何实质内容。金融条款里一个数字、一个标点的差异都可能改变含义，
所以这里的每条规则都必须是可逆的格式修复，而不是语义改写。

实测到的抽取噪声（样本来自 CFFEX 股指期权合约交易细则、SZSE 合约条款）：

    "沪深300 股指期权"      -> 数字与中文之间被插入空格
    "2019 年12 月14 日"     -> 同上
    "10000 份"              -> 同上
    "第一条为规范..."        -> 条号与正文之间无分隔（这是原文如此，保留）
"""

from __future__ import annotations

import re

# 数字/字母 与 CJK 之间被 PDF 抽取插入的空格
_DIGIT_CJK = re.compile(r"(?<=[0-9A-Za-z%])[ \t]+(?=[一-鿿])")
_CJK_DIGIT = re.compile(r"(?<=[一-鿿])[ \t]+(?=[0-9A-Za-z])")
# CJK 之间的空格一律是噪声
_CJK_CJK = re.compile(r"(?<=[一-鿿])[ \t]+(?=[一-鿿])")

# 页眉页脚里的孤立页码行
_PAGE_NUM_LINE = re.compile(r"^[\s\-—–]*\d{1,4}[\s\-—–]*$")

# 连续 3 个以上换行压成 2 个
_MULTI_NEWLINE = re.compile(r"\n{3,}")
# 行尾空白
_TRAILING_WS = re.compile(r"[ \t]+$", re.M)

# 全角/半角混用的括号在条款里都有出现，统一不做转换（会改变原文外观），
# 但软连字符、零宽字符这类不可见噪声必须去掉
_INVISIBLE = re.compile(r"[​-‏‪-‮﻿­]")


def clean_cjk_spacing(text: str) -> str:
    """修复 PDF 抽取在中文与数字/字母之间插入的空格。

    只处理 CJK 相邻的情况，纯英文段落（如 SGX/HKEX 规则）的正常
    单词间距完全不受影响。
    """
    text = _DIGIT_CJK.sub("", text)
    text = _CJK_DIGIT.sub("", text)
    text = _CJK_CJK.sub("", text)
    return text


def strip_page_furniture(text: str, *, header_footer_lines: int = 0) -> str:
    """去掉孤立页码行。

    header_footer_lines > 0 时，额外裁掉每页首尾各 N 行——只在确认
    该文档有稳定页眉页脚时才启用，默认关闭以免误删正文。
    """
    lines = text.split("\n")
    kept = [ln for ln in lines if not _PAGE_NUM_LINE.match(ln)]
    if header_footer_lines > 0:
        kept = kept[header_footer_lines:-header_footer_lines] or kept
    return "\n".join(kept)


def normalize_whitespace(text: str) -> str:
    """规整空白，但保留段落结构（双换行）。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_WS.sub("", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def clean_text(text: str, *, is_cjk: bool = True) -> str:
    """标准清洗管线。"""
    if not text:
        return ""
    text = _INVISIBLE.sub("", text)
    text = normalize_whitespace(text)
    text = strip_page_furniture(text)
    if is_cjk:
        text = clean_cjk_spacing(text)
    return normalize_whitespace(text)


def looks_chinese(text: str, threshold: float = 0.15) -> bool:
    """判断文本是否以中文为主，用于决定走哪套清洗与切分规则。"""
    if not text:
        return False
    sample = text[:4000]
    cjk = sum(1 for c in sample if "一" <= c <= "鿿")
    return cjk / max(len(sample), 1) >= threshold


# ---------------------------------------------------------------------
# 文号与时效性
# ---------------------------------------------------------------------

# 上证发〔2023〕48号 / 深证上〔2022〕1146号 / 中金所发〔2019〕xx号
#
# 机构简称前缀是可选的：郑商所有一份文档正文只写「〔2026〕68号」而不带
# 「郑商所发」。年份括号 + 数字 + 号 这个结构本身已经足够特异，不会误命中。
#
# 只支持这一种形态是实测结论，不是偷懒：把全部 190 份文档的全文扫一遍，
# 「第N号」「N年第N号」「Circular No.」这些形态命中数都是 0。140 份抽不到
# 文号的文档里，0 份在原始文件里有文号可捞 —— 它们是合约条款表、常设规则
# 手册、投资者问答与英文 rulebook，本身就不带发文字号，返回 None 是正确
# 行为而非漏检。详见 docs/audit.md §2.2。
# 前缀后允许空白：前缀改可选之后，「上证发 〔2023〕48号」这种带空格的输入
# 会退化成部分匹配、静默返回「〔2023〕48号」，把机构简称吞掉 —— 而机构简称
# 是引用真值的一部分，丢了比返回 None 更糟。当前语料 102 处文号全是紧贴形态，
# 但 PDF 文本抽取常在原文无空格处插入空格，这里按防御处理。
_DOC_NO = re.compile(
    r"(?:[一-鿿]{2,8}\s*)?[〔\[（(]\s*(?:19|20)\d{2}\s*[〕\]）)]\s*第?\s*\d+\s*号"
)


def extract_doc_no(text: str) -> str | None:
    """从正文前部抽取文号，作为引用真值。"""
    m = _DOC_NO.search(text[:3000])
    return re.sub(r"\s+", "", m.group(0)) if m else None


_REPEALED_MARKERS = ("已废止", "已作废", "废止", "失效")
_SUPERSEDED_MARKERS = ("已修订", "被修订", "修订版")


def infer_effective_status(title: str, text: str = "") -> str:
    """从标题/正文推断时效性。

    交易所通常把废止标记写在标题后缀，如"……（已废止）"。
    至少 5 份 SSE 期权文档处于废止状态，检索时必须能过滤掉，
    否则会用失效规则回答问题。
    """
    from ..schema import EffectiveStatus

    haystack = f"{title}\n{text[:1500]}"
    if any(m in haystack for m in _REPEALED_MARKERS):
        return EffectiveStatus.REPEALED.value
    if any(m in haystack for m in _SUPERSEDED_MARKERS):
        return EffectiveStatus.SUPERSEDED.value
    # 有文号且无废止标记的规则文档，默认认为现行有效
    if extract_doc_no(haystack):
        return EffectiveStatus.ACTIVE.value
    return EffectiveStatus.UNKNOWN.value


_EFFECTIVE_DATE = re.compile(
    r"自\s*((?:19|20)\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日\s*起(?:施行|实施|执行|生效)"
)


def extract_effective_date(text: str) -> str | None:
    """抽取"自 YYYY 年 M 月 D 日起施行"中的生效日期。"""
    m = _EFFECTIVE_DATE.search(text)
    if not m:
        return None
    y, mo, d = m.groups()
    return f"{y}-{int(mo):02d}-{int(d):02d}"
