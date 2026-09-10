"""HTML 解析。

关键点是**必须裁到正文容器**。实测各站点的导航/页脚占比：

    SSE 合约条款页       正文 <div class="allZoom">，页面 85% 是站点导航
    CFFEX 规则/问答页    正文 <div class="content news_content allcontent">
    SGX Rule 10.3 页面   实质正文 562 字符 / 可见文本 19588 字符 = 2.9%

不裁剪的话，每个切出来的块都会包含同一份导航菜单，结果是：
BM25 的 IDF 被导航词彻底稀释、向量库里塞满近似重复的块、
重排看到的候选全长一个样。这是 HTML 语料最典型的翻车方式。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from selectolax.parser import HTMLParser

from .clean import clean_text, looks_chinese

logger = logging.getLogger(__name__)

# 站点正文容器选择器。按顺序尝试，第一个命中且有实质内容的胜出。
SITE_SELECTORS: dict[str, list[str]] = {
    "sse.com.cn": ["div.allZoom", "div.article_opt", "div.sse_content"],
    "cffex.com.cn": ["div.content.news_content.allcontent", "div.allcontent", "div.show_main"],
    "szse.cn": ["div.des-content", "div.article-content", "div#desContent"],
    "rulebook.sgx.com": ["div.rulebook-content", "main", "article", "div#content"],
    "hkex.com.hk": ["div.content", "main"],
    "investor.gov": ["div.field--name-body", "article", "main"],
}

# 一定要整块删掉的元素
_DROP_TAGS = (
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "iframe",
    "svg",
    "form",
    "button",
)

# 导航类容器的 class/id 关键词
_DROP_HINTS = (
    "nav",
    "menu",
    "breadcrumb",
    "sidebar",
    "footer",
    "header",
    "banner",
    "share",
    "related",
    "copyright",
    "search",
    "toolbar",
    "pagination",
)


def _looks_like_question(title: str) -> bool:
    """标题是否是一个问句（一问一答型页面的判据）。"""
    if not title:
        return False
    head = re.split(r"[|｜]", title)[0].strip()
    if head.endswith(("？", "?")):
        return True
    return any(
        w in head for w in ("如何", "怎么", "怎样", "是否", "能否", "可否", "哪些", "为何")
    )


def _host_of(url: str) -> str:
    from urllib.parse import urlparse

    return urlparse(url).netloc.lower().removeprefix("www.")


def _drop_noise(tree: HTMLParser) -> None:
    """删掉脚本、样式与明显的导航容器。"""
    for tag in _DROP_TAGS:
        for node in tree.css(tag):
            node.decompose()
    for node in tree.css("[class],[id]"):
        attr = f"{node.attributes.get('class', '')} {node.attributes.get('id', '')}".lower()
        if any(h in attr for h in _DROP_HINTS):
            node.decompose()


def _table_to_rows(node: Any) -> list[list[str]]:
    """把一个 <table> 抽成二维数组。"""
    rows: list[list[str]] = []
    for tr in node.css("tr"):
        cells = [
            clean_text(td.text(separator=" ", strip=True), is_cjk=True)
            for td in tr.css("td, th")
        ]
        if any(c for c in cells):
            rows.append(cells)
    return rows


def _render_rows(rows: list[list[str]]) -> str:
    """表格转文本：两列渲染成 "字段：值"，多列用 | 分隔。"""
    lines = []
    for row in rows:
        cells = [c for c in row if c]
        if len(cells) == 2:
            lines.append(f"{cells[0]}：{cells[1]}")
        elif cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _pick_content_node(tree: HTMLParser, url: str) -> Any:
    """按站点选择器裁到正文；都没命中就退回"文本最多的 div"。"""
    host = _host_of(url)
    selectors = next(
        (sels for domain, sels in SITE_SELECTORS.items() if domain in host), []
    )
    for sel in selectors:
        node = tree.css_first(sel)
        if node and len(node.text(strip=True)) > 100:
            return node

    # 兜底：选正文密度最高的块级元素。对未收录的站点也能给出可用结果，
    # 但会在解析警告里标注，提示应该为该站点补一条选择器。
    best, best_len = None, 0
    for node in tree.css("div, article, section, main"):
        # 只看叶子密度高的节点，避免选中 <body> 这种包住一切的容器
        text_len = len(node.text(strip=True))
        child_divs = len(node.css("div"))
        if text_len > best_len and child_divs <= 8:
            best, best_len = node, text_len
    return best or tree.body or tree.root


def parse_html(
    path: str | Path | None = None,
    *,
    html: str | None = None,
    url: str = "",
    extract_table: bool = True,
) -> dict[str, Any]:
    """解析 HTML，返回正文、表格与元信息。

    path 与 html 二选一。
    """
    if html is None:
        if path is None:
            raise ValueError("parse_html 需要 path 或 html 之一")
        raw = Path(path).read_bytes()
        html = _decode(raw)

    tree = HTMLParser(html)
    title = ""
    if tree.css_first("title"):
        title = clean_text(tree.css_first("title").text(strip=True), is_cjk=True)

    _drop_noise(tree)
    node = _pick_content_node(tree, url)

    warnings: list[str] = []
    host = _host_of(url)
    if url and not any(d in host for d in SITE_SELECTORS):
        warnings.append(f"站点 {host} 未配置正文选择器，使用了密度兜底策略")

    tables: list[dict[str, Any]] = []
    table_texts: list[str] = []
    if extract_table and node is not None:
        for i, tnode in enumerate(node.css("table")):
            rows = _table_to_rows(tnode)
            if not rows:
                continue
            tables.append(
                {
                    "index": i,
                    "n_rows": len(rows),
                    "n_cols": max(len(r) for r in rows),
                    "rows": rows,
                }
            )
            table_texts.append(_render_rows(rows))
            # 从 DOM 里摘掉，避免正文里再出现一份无结构的表格文本
            tnode.decompose()

    body = node.text(separator="\n", strip=True) if node is not None else ""
    is_cjk = looks_chinese(body or title)
    body = clean_text(body, is_cjk=is_cjk)
    body = trim_boilerplate(body)

    if table_texts:
        body = "\n\n".join([body, *table_texts]).strip()

    # 正文过短通常意味着选择器没命中或页面是 JS 渲染，但**一问一答型页面
    # 例外** —— CFFEX 常见问答里不少答案本来就只有一句话
    # （"是的，按客户号合并计算。"= 12 字符），对它们报警是误报。
    if len(body) < (15 if _looks_like_question(title) else 50):
        warnings.append(f"正文仅 {len(body)} 字符，疑似选择器未命中或页面为 JS 渲染")

    return {
        "text": body,
        "title": title,
        "tables": tables,
        "page_count": 1,
        "ocr_pages": [],
        "warnings": warnings,
        "lang": "zh" if is_cjk else "en",
    }


# 站点在正文容器**内部**残留的装饰块。CSS 选择器裁不掉这些，
# 因为它们和正文在同一个 div 里。
#
# CFFEX 的文章页结构实测为：
#     首页 > 服务 > … > 常见问答 / 服务 Service / 日期 / 分享：/ 微信二维码
#     <正文>
#     上一篇：… 下一篇：…
# 不裁掉的话，每篇 200 字的问答里有 80 字是导航面包屑，
# BM25 的 IDF 会被"首页""服务""分享"这些词严重稀释。
_LEAD_MARKERS = ("微信二维码", "分享：", "分享:")
# 只在正文开头这么多字符内寻找页头装饰标记
_LEAD_SCAN_CHARS = 200
_TAIL_MARKERS = ("上一篇：", "上一篇:", "下一篇：", "下一篇:", "打印本页", "关闭窗口")
_BREADCRUMB_LINE = re.compile(r"^(首页|Home)$|^>$|^[\s>]+$")


def trim_boilerplate(text: str) -> str:
    """去掉正文容器内残留的面包屑与上下篇导航。"""
    if not text:
        return text

    # 尾部：从第一个"上一篇/下一篇"开始整段丢弃
    cut = len(text)
    for marker in _TAIL_MARKERS:
        i = text.find(marker)
        if i != -1:
            cut = min(cut, i)
    text = text[:cut].rstrip()

    # 头部：正文从**最靠后**的那个分享控件标记之后开始。
    # 必须取最靠后的：CFFEX 的页头是"分享：… 微信二维码 …"，
    # 若在"分享："处就截断，"微信二维码"这行会残留在正文里。
    # 用绝对字符数而不是长度比例做保护：页头装饰块总是出现在最前面的
    # 一两百字内，而按比例判断在短文档上会失效（一篇 200 字的问答里，
    # 页头就占了一半以上，比例判断反而不敢裁）。
    cut_at = -1
    for marker in _LEAD_MARKERS:
        i = text.rfind(marker)
        if i != -1 and i < _LEAD_SCAN_CHARS:
            cut_at = max(cut_at, i + len(marker))
    if cut_at > 0:
        text = text[cut_at:].lstrip()

    # 残余的面包屑单行
    lines = [ln for ln in text.split("\n") if not _BREADCRUMB_LINE.match(ln.strip())]
    return "\n".join(lines).strip()


def _decode(body: bytes) -> str:
    for enc in ("utf-8", "gb18030", "gbk"):
        try:
            return body.decode(enc)
        except UnicodeDecodeError:
            continue
    return body.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------
# 问答页专用解析
# ---------------------------------------------------------------------

# SSE 50ETF 经纪业务 FAQ: "1. 什么是股票期权？" / "答：……"
_QA_NUMBERED = re.compile(
    r"^\s*(\d{1,3})\s*[.、．]\s*(.+?[？?])\s*$",
    re.M,
)
_ANSWER_PREFIX = re.compile(r"^\s*答[：:]\s*", re.M)


def extract_qa_pairs(text: str) -> list[dict[str, str]]:
    """从问答型文档里抽出 (问, 答) 对。

    实测两种版式，必须都支持：
      A) SSE 50ETF 经纪业务 FAQ —— "N. 问题？" 后跟 "答：……"（32 对）
      B) SSE 熔断机制问答      —— "N. 问题？" 后**直接**跟答案，没有"答："前缀（8 对）
    所以答案边界一律取"到下一个编号问题为止"，"答："前缀有则剥掉、无则忽略。
    """
    matches = list(_QA_NUMBERED.finditer(text))
    pairs: list[dict[str, str]] = []
    for i, m in enumerate(matches):
        question = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        answer = text[start:end].strip()
        answer = _ANSWER_PREFIX.sub("", answer, count=1).strip()
        if question and answer:
            pairs.append({"question": question, "answer": answer, "index": str(i + 1)})
    return pairs
