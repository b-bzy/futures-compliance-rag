"""站点适配器 —— 每个只负责一件事：把一个索引页展开成待下载的 URL 列表。

实际下载与内容校验统一走 Fetcher，保证校验逻辑只有一份实现。

各站点的真实结构（已实测）：

- SSE 合约条款: index 直接给出 5 个 c_YYYYMMDD_NNNNNNN.shtml
- SSE 业务规则: index -> article -> `<contentId>/files/<hash>.doc` （两跳）
- CFFEX 规则:   index -> article(/cn/<板块>/YYYYMMDD/NNNNN.html)
                      -> /u/cms/www/YYYYMM/<hash>.pdf （两跳）
- CFFEX 问答:   index 直接给出 /cn/cjwd1/YYYYMMDD/NNNNN.html，共 4 页
- SZSE:         列表页对 curl 返回 0 链接（客户端渲染），走固化 manifest
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import urljoin

from .fetcher import Fetcher

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredDoc:
    """适配器发现的一个待下载文档。"""

    id: str
    url: str
    title: str
    expect: str
    venue: str
    lang: str = "zh"
    notes: str | None = None


def _clean_text(html_fragment: str) -> str:
    """把 <a> 标签内的 HTML 片段压成一行纯文本标题。"""
    txt = re.sub(r"<[^>]+>", "", html_fragment)
    return re.sub(r"\s+", " ", txt).strip()


def _slug(text: str, maxlen: int = 40) -> str:
    """标题转成可用作文件名的片段。中文保留，其余符号替换成下划线。"""
    s = re.sub(r"[^\w一-鿿]+", "_", text).strip("_")
    return s[:maxlen] or "untitled"


def _decode(body: bytes) -> str:
    """交易所站点编码不统一，依次尝试。"""
    for enc in ("utf-8", "gb18030", "gbk"):
        try:
            return body.decode(enc)
        except UnicodeDecodeError:
            continue
    return body.decode("utf-8", errors="replace")


def _get_html(fetcher: Fetcher, url: str) -> str | None:
    """取一个 HTML 页面的文本内容（不落盘，只为发现链接）。"""
    try:
        fetcher._throttle(url)
        resp = fetcher.client.get(url)
        if resp.status_code >= 400:
            logger.warning("索引页 %s 返回 HTTP %d", url, resp.status_code)
            return None
        if len(resp.content) in fetcher.waf_fingerprints:
            logger.warning("索引页 %s 命中 WAF 拦截页指纹", url)
            return None
        return _decode(resp.content)
    except Exception as e:  # noqa: BLE001 - 发现阶段失败不应中断整个抓取
        logger.warning("索引页 %s 获取失败: %s", url, e)
        return None


# =====================================================================
# 上海证券交易所
# =====================================================================

_SSE_ARTICLE_RE = re.compile(
    r'<a[^>]+href=["\']([^"\']*c_\d+_\d+\.s?html)["\'][^>]*>(.*?)</a>', re.S
)
# SSE 附件形如 href="5717359/files/<32位hash>.doc"，也可能是绝对路径
_SSE_ATTACH_RE = re.compile(r'href=["\']([^"\']*/files/[0-9a-f]{16,}\.(?:doc|docx|pdf))["\']', re.I)


def _sse_articles(fetcher: Fetcher, index_url: str) -> list[tuple[str, str]]:
    """从 SSE 索引页抽 (绝对URL, 标题)，按 URL 去重且保持顺序。"""
    html = _get_html(fetcher, index_url)
    if not html:
        return []
    seen: dict[str, str] = {}
    for href, label in _SSE_ARTICLE_RE.findall(html):
        url = urljoin(index_url, href)
        title = _clean_text(label)
        if url not in seen and title:
            seen[url] = title
    return list(seen.items())


def sse_contract(fetcher: Fetcher, index_url: str, venue: str = "SSE") -> list[DiscoveredDoc]:
    """SSE 合约基本条款 —— 5 个品种，正文表格 inline 在文章页里。"""
    docs = []
    for url, title in _sse_articles(fetcher, index_url):
        docs.append(
            DiscoveredDoc(
                id=f"sse_contract_{_slug(title)}",
                url=url,
                title=title,
                expect="html",
                venue=venue,
                notes="合约条款表 inline，解析时只保留 <table>",
            )
        )
    logger.info("SSE 合约条款: 发现 %d 篇", len(docs))
    return docs


def _sse_with_attachments(
    fetcher: Fetcher, index_url: str, venue: str, id_prefix: str, notes: str
) -> list[DiscoveredDoc]:
    """SSE 规则/指南类：文章页本身要抓，附件（.doc/.docx）也要抓。

    实测 16 篇业务规则里 10 篇正文 inline、6 篇正文只在 .doc 附件中；
    2025 规则库更极端 —— HTML 壳只有约 1550 字符，正文完全在 .docx 里。
    所以两者都下，解析阶段再择优。
    """
    docs: list[DiscoveredDoc] = []
    for url, title in _sse_articles(fetcher, index_url):
        slug = _slug(title)
        docs.append(
            DiscoveredDoc(
                id=f"{id_prefix}_{slug}",
                url=url,
                title=title,
                expect="html",
                venue=venue,
                notes=notes,
            )
        )
        # 进文章页找附件
        article_html = _get_html(fetcher, url)
        if not article_html:
            continue
        for i, href in enumerate(dict.fromkeys(_SSE_ATTACH_RE.findall(article_html))):
            att_url = urljoin(url, href)
            ext = att_url.rsplit(".", 1)[-1].lower()
            docs.append(
                DiscoveredDoc(
                    id=f"{id_prefix}_{slug}_att{i}",
                    url=att_url,
                    title=f"{title}（附件）",
                    expect=ext,
                    venue=venue,
                    notes="正文附件 —— 部分文档正文只存在于此",
                )
            )
    logger.info("%s: 发现 %d 项（含附件）", id_prefix, len(docs))
    return docs


def sse_rule(fetcher: Fetcher, index_url: str, venue: str = "SSE") -> list[DiscoveredDoc]:
    """SSE 股票期权试点业务规则，16 篇，4 篇已废止（标题带标记）。"""
    return _sse_with_attachments(
        fetcher, index_url, venue, "sse_rule", "试点业务规则；标题含（已废止）者需标记 effective_status"
    )


def sse_guide(fetcher: Fetcher, index_url: str, venue: str = "SSE") -> list[DiscoveredDoc]:
    """SSE 配套业务指南与投资者问答，21 篇 —— 最高价值的中文 QA 来源。"""
    return _sse_with_attachments(
        fetcher, index_url, venue, "sse_guide", "含 50ETF 经纪业务 FAQ(32问) 与熔断机制问答(8问)"
    )


# =====================================================================
# 中国金融期货交易所
# =====================================================================

# 规则文章链接形如 /cn/gzjygz/20260410/43076.html 或 /cn/ssxz/20200301/43077.html
_CFFEX_ARTICLE_RE = re.compile(
    r'<a[^>]+href=["\'](/cn/[a-z0-9]+/\d{8}/\d+\.html)["\'][^>]*>(.*?)</a>', re.S | re.I
)
_CFFEX_PDF_RE = re.compile(r'href=["\']([^"\']*/u/cms/www/[^"\']*\.pdf)["\']', re.I)


def cffex_rules(fetcher: Fetcher, index_url: str, venue: str = "CFFEX") -> list[DiscoveredDoc]:
    """CFFEX 交易所规则总索引 —— 73 个规则，两跳到 PDF。

    实测：index 页面的 <a> 直接给出文章 URL，文章页里有唯一一个
    /u/cms/www/YYYYMM/<hash>.pdf 链接即正文。
    注意整站必须走 http://，https 会 TLS 握手失败。
    """
    html = _get_html(fetcher, index_url)
    if not html:
        return []

    articles: dict[str, str] = {}
    for href, label in _CFFEX_ARTICLE_RE.findall(html):
        title = _clean_text(label)
        # 过滤导航项（"交易所规则"、"规则"等短标签）
        if title and len(title) >= 6:
            articles.setdefault(urljoin(index_url, href), title)

    docs: list[DiscoveredDoc] = []
    for art_url, title in articles.items():
        art_html = _get_html(fetcher, art_url)
        if not art_html:
            continue
        pdfs = list(dict.fromkeys(_CFFEX_PDF_RE.findall(art_html)))
        if not pdfs:
            # 少数规则正文直接写在 HTML 里，没有 PDF 附件
            docs.append(
                DiscoveredDoc(
                    id=f"cffex_rule_{_slug(title)}",
                    url=art_url,
                    title=title,
                    expect="html",
                    venue=venue,
                    notes="无 PDF 附件，正文 inline",
                )
            )
            continue
        for i, pdf in enumerate(pdfs):
            suffix = "" if i == 0 else f"_v{i}"
            docs.append(
                DiscoveredDoc(
                    id=f"cffex_rule_{_slug(title)}{suffix}",
                    url=urljoin(art_url, pdf),
                    title=title,
                    expect="pdf",
                    venue=venue,
                    notes="第X条结构，切分友好" + ("（历史版本）" if i else ""),
                )
            )
    logger.info("CFFEX 规则: 发现 %d 项（来自 %d 篇文章）", len(docs), len(articles))
    return docs


_CFFEX_FAQ_RE = re.compile(
    r'<a[^>]+href=["\'](/cn/cjwd\d*/\d{8}/\d+\.html)["\'][^>]*>(.*?)</a>', re.S | re.I
)


def cffex_faq(fetcher: Fetcher, index_url: str, venue: str = "CFFEX") -> list[DiscoveredDoc]:
    """CFFEX 常见问答，共 4 页，约 40 条原子一问一答 —— Tier-1 金标来源。"""
    docs: list[DiscoveredDoc] = []
    seen: set[str] = set()

    # 分页形如 cjwd1.html / cjwd1_2.html ... 逐页尝试，连续两页无新增即停
    page_urls = [index_url]
    base = index_url.rsplit(".html", 1)[0]
    page_urls += [f"{base}_{i}.html" for i in range(2, 8)]

    misses = 0
    for page_url in page_urls:
        html = _get_html(fetcher, page_url)
        if not html:
            misses += 1
            if misses >= 2:
                break
            continue
        found_new = False
        for href, label in _CFFEX_FAQ_RE.findall(html):
            url = urljoin(page_url, href)
            title = _clean_text(label)
            if url in seen or not title:
                continue
            seen.add(url)
            found_new = True
            docs.append(
                DiscoveredDoc(
                    id=f"cffex_faq_{_slug(title, 30)}_{len(seen)}",
                    url=url,
                    title=title,
                    expect="html",
                    venue=venue,
                    notes="原子一问一答，作为 Tier-1 人工撰写金标",
                )
            )
        misses = 0 if found_new else misses + 1
        if misses >= 2:
            break

    logger.info("CFFEX 常见问答: 发现 %d 条", len(docs))
    return docs


# =====================================================================
# 深圳证券交易所
# =====================================================================


def szse_rule(fetcher: Fetcher, index_url: str, venue: str = "SZSE") -> list[DiscoveredDoc]:
    """SZSE 业务规则列表。

    ⚠️ www.szse.cn 的列表页对 curl 返回 0 个文章链接（内容由 JS 渲染），
    且无 sitemap、JSON list API 遍寻不获（/api/search/content=500,
    /api/search/=404）。因此这里优先读由 Playwright 脚本一次性产出的
    固化 manifest；没有 manifest 时降级为已知种子 URL 并 warn。

    生成 manifest: python scripts/00_harvest_szse_urls.py
    """
    from ..config import repo_root

    manifest = repo_root() / "data" / "interim" / "szse_url_manifest.json"
    if manifest.exists():
        import json

        entries = json.loads(manifest.read_text(encoding="utf-8"))
        docs = [
            DiscoveredDoc(
                id=f"szse_rule_{_slug(e['title'])}",
                url=e["url"],
                title=e["title"],
                expect=e.get("expect", "html"),
                venue=venue,
                notes="来自 Playwright 固化 manifest",
            )
            for e in entries
        ]
        logger.info("SZSE: 从 manifest 载入 %d 项", len(docs))
        return docs

    logger.warning(
        "未找到 %s —— SZSE 列表页无法用 curl 枚举。"
        "只抓 sources.yaml 中已验证的静态 PDF 直链。"
        "如需完整列表请先跑 scripts/00_harvest_szse_urls.py",
        manifest,
    )
    return []


# =====================================================================
# 派发表
# =====================================================================

ADAPTERS = {
    "sse_contract": sse_contract,
    "sse_rule": sse_rule,
    "sse_guide": sse_guide,
    "cffex_rules": cffex_rules,
    "cffex_faq": cffex_faq,
    "szse_rule": szse_rule,
}


def get_adapter(name: str):
    """按名字取适配器，未注册时抛出可读错误。"""
    if name not in ADAPTERS:
        raise KeyError(f"未注册的适配器 {name!r}，可用: {sorted(ADAPTERS)}")
    return ADAPTERS[name]
