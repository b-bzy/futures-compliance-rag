"""PDF 解析 —— PyMuPDF 主力。

为什么不用 PaddleOCR 做版面识别：实测抽样的 8 份交易所 PDF 里有 7 份
是原生数字排版（有完整文本层），根本不需要 OCR；而 paddlex 的依赖
会把 langchain 降到 <1.0，代价远大于收益。详见 constraints.txt。

OCR 只在真正必要时触发，且判定必须是**逐页双条件**：
文本极少 **且** 页面被大图覆盖。只用"文本少"这一个条件会在
DCE 豆粕期权制度汇编上误判 —— 那份文档第 0 页是扫描封面（0 字符），
但正文页都是原生文本，单条件会导致整篇被送去 OCR。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pymupdf

from .clean import clean_text, looks_chinese

logger = logging.getLogger(__name__)


def _page_image_ratio(page: pymupdf.Page) -> float:
    """页面被图片覆盖的面积占比。"""
    page_area = abs(page.rect.get_area())
    if page_area <= 0:
        return 0.0
    covered = 0.0
    for img in page.get_images(full=True):
        try:
            for rect in page.get_image_rects(img[0]):
                covered += abs(rect.get_area())
        except Exception:  # noqa: BLE001 - 个别损坏的图对象不应中断解析
            continue
    return min(covered / page_area, 1.0)


def _needs_ocr(page: pymupdf.Page, min_chars: int, min_image_ratio: float) -> bool:
    """逐页判断是否需要 OCR —— 双条件。"""
    text_len = len(page.get_text().strip())
    if text_len >= min_chars:
        return False
    return _page_image_ratio(page) >= min_image_ratio


def extract_tables(page: pymupdf.Page) -> list[dict[str, Any]]:
    """抽取页面上的表格。

    合约条款表是本项目最有价值的结构化资产：每一行就是一个
    (字段, 值) 原子事实，可以直接模板化成零幻觉的 QA 对。
    """
    tables: list[dict[str, Any]] = []
    try:
        found = page.find_tables()
    except Exception as e:  # noqa: BLE001
        logger.debug("第 %d 页表格检测失败: %s", page.number, e)
        return tables

    for t in found.tables:
        try:
            rows = t.extract()
        except Exception:  # noqa: BLE001
            continue
        # 清洗单元格：去掉 None，修复中文与数字间的抽取空格
        cleaned = [
            [clean_text(c or "", is_cjk=True) for c in row]
            for row in rows
            if any((c or "").strip() for c in row)
        ]
        if not cleaned:
            continue
        tables.append(
            {
                "page": page.number,
                "n_rows": len(cleaned),
                "n_cols": max(len(r) for r in cleaned),
                "rows": cleaned,
                "bbox": list(t.bbox),
            }
        )
    return tables


def _table_to_text(table: dict[str, Any]) -> str:
    """把表格转成正文里可读的文本形式，保证表格内容也能被全文检索到。

    两列表格（合约条款表的典型形态）渲染成 "字段：值"，
    多列表格渲染成用 | 分隔的行。
    """
    lines = []
    for row in table["rows"]:
        cells = [c for c in row if c]
        if len(cells) == 2:
            lines.append(f"{cells[0]}：{cells[1]}")
        elif cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _decode_pdf_string(value: str) -> str:
    """解码 PDF 元数据里的标题。

    交易所 PDF 的 Title 字段经常是**未解码的 PDF 十六进制串**，形如
    `<30333232B6B9C6C9...>`。大商所两份文档就是这样，直接用会得到一串乱码。
    这些字节实际是 GBK 编码的中文（B6B9=豆 C6C9=粕 …），解码后是
    "0322豆粕期权交易制度汇编-V2.0"。
    """
    if not value:
        return ""
    v = value.strip()
    if v.startswith("<") and v.endswith(">"):
        v = v[1:-1]
    if re.fullmatch(r"[0-9A-Fa-f]+", v) and len(v) >= 8:
        raw = bytes.fromhex(v[: len(v) - len(v) % 2])
        for enc in ("gbk", "gb18030", "utf-16-be", "utf-8"):
            try:
                decoded = raw.decode(enc).strip("\x00 ")
                # 解码结果必须像人话：至少含一个中文或字母
                if decoded and re.search(r"[一-鿿A-Za-z]", decoded):
                    return decoded
            except (UnicodeDecodeError, ValueError):
                continue
        return ""
    return v


# 页眉页脚里的页码行，抽标题时要跳过
_PAGE_MARK = re.compile(r"^[\s\-—–—]*[（(]?\s*\d{1,4}\s*[)）]?[\s\-—–—]*$")
# "附件"、"附件：" 这类前缀本身不是标题
_ATTACH_MARK = re.compile(r"^附件\s*[:：]?\s*$")


def extract_pdf_title(path: str | Path, *, max_lines: int = 12) -> str:
    """从 PDF 里抽标题。元数据优先，否则取首页字号最大的那一行。

    交易所 PDF 大多没有元数据标题，但首页排版很规整：标题一定是
    最大字号且居中。用字号而不是位置来判断，比"取第 N 行"稳健得多
    （首行常常是"— 1 —"页码或"附件"二字）。
    """
    path = Path(path)
    try:
        doc = pymupdf.open(path)
    except Exception:  # noqa: BLE001
        return ""

    try:
        meta_title = _decode_pdf_string((doc.metadata or {}).get("title", "") or "")
        # 元数据标题常带文件名残留（"SIF_CS"）或流水号前缀（"0322豆粕…"）。
        # 剔除的是**纯 ASCII 标识符**那种，注意必须写成 [A-Za-z0-9_.-] 而不是
        # \w —— Python 的 \w 在 Unicode 模式下把中文也算进去，
        # 用 \w 会把"0322豆粕期权交易制度汇编-V2.0"整个判成标识符而丢弃。
        if (
            meta_title
            and len(meta_title) >= 6
            and not re.fullmatch(r"[A-Za-z0-9_.\-]+", meta_title)
        ):
            # 元数据里常是排版源文件名，带流水号前缀和扩展名残留
            # （"0322豆粕期权交易制度汇编-V2.0.ai"）
            cleaned = re.sub(r"^\d{3,6}", "", meta_title)
            cleaned = re.sub(
                r"[-_ ]*(?:v?\d+(?:\.\d+)*|gai|final|new)?\.(?:ai|pdf|docx?|indd|cdr)$",
                "",
                cleaned,
                flags=re.I,
            )
            return cleaned.strip(" -_")

        if doc.page_count == 0:
            return ""
        blocks = doc[0].get_text("dict").get("blocks", [])
        candidates: list[tuple[float, str]] = []
        for b in blocks:
            for line in b.get("lines", []):
                text = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
                if not text or _PAGE_MARK.match(text) or _ATTACH_MARK.match(text):
                    continue
                if len(text) > 60:
                    continue
                size = max((s.get("size", 0) for s in line.get("spans", [])), default=0)
                candidates.append((size, text))
                if len(candidates) >= max_lines:
                    break
            if len(candidates) >= max_lines:
                break

        if not candidates:
            return ""
        top = max(c[0] for c in candidates)
        # 同为最大字号的相邻行拼起来（标题常被排成两行）
        parts = [t for s, t in candidates if s >= top - 0.1]
        return clean_text(" ".join(parts[:2]), is_cjk=True)
    finally:
        doc.close()


def _strip_table_lines(page_text: str, tables: list[dict[str, Any]]) -> str:
    """去掉页面文本里已被表格覆盖的行，保留标题与附注。

    表格单元格的内容在 page.get_text() 中是逐行出现的（多行单元格会被
    拆成多行），所以按行比对：某一行去空白后能在任一单元格的去空白形式
    中找到，就认为它已被表格表达，丢弃。
    """
    cell_blobs = {
        "".join(cell.split())
        for t in tables
        for row in t["rows"]
        for cell in row
        if cell and cell.strip()
    }
    if not cell_blobs:
        return page_text

    kept: list[str] = []
    for line in page_text.split("\n"):
        squeezed = "".join(line.split())
        if not squeezed:
            continue
        if any(squeezed in blob for blob in cell_blobs):
            continue
        kept.append(line.strip())
    return "\n".join(kept)


def parse_pdf(
    path: str | Path,
    *,
    ocr_enabled: bool = True,
    min_chars_per_page: int = 50,
    min_image_ratio: float = 0.30,
    extract_table: bool = True,
) -> dict[str, Any]:
    """解析一个 PDF，返回正文、表格与解析元信息。

    返回:
        {text, tables, page_count, ocr_pages, warnings}
    """
    path = Path(path)
    doc = pymupdf.open(path)

    page_texts: list[str] = []
    all_tables: list[dict[str, Any]] = []
    ocr_pages: list[int] = []
    warnings: list[str] = []

    for page in doc:
        text = page.get_text()

        if ocr_enabled and _needs_ocr(page, min_chars_per_page, min_image_ratio):
            ocr_text = _ocr_page(page)
            if ocr_text:
                text = ocr_text
                ocr_pages.append(page.number)
            else:
                warnings.append(f"第 {page.number} 页疑似扫描件但 OCR 未产出文本")

        if extract_table:
            tables = extract_tables(page)
            if tables:
                all_tables.extend(tables)
                table_text = "\n\n".join(_table_to_text(t) for t in tables)
                if table_text:
                    # 表格单元格在 page.get_text() 里也会出现一遍，只是丢失了
                    # 行列关系（字段与值被换行拆开）。直接拼接会得到两份内容
                    # 相同的块 —— 既污染 BM25 词频，又让重排看到重复候选。
                    # 所以只保留页面文本里**不属于表格**的部分（标题、附注），
                    # 再接上结构化的表格渲染。
                    residual = _strip_table_lines(text, tables)
                    text = f"{residual}\n\n{table_text}" if residual else table_text

        page_texts.append(text)

    doc.close()

    raw = "\n\n".join(page_texts)
    is_cjk = looks_chinese(raw)
    cleaned = clean_text(raw, is_cjk=is_cjk)

    if not cleaned.strip():
        warnings.append("解析结果为空 —— 该 PDF 可能是纯图像且 OCR 不可用")

    return {
        "text": cleaned,
        "tables": all_tables,
        "page_count": len(page_texts),
        "ocr_pages": ocr_pages,
        "warnings": warnings,
        "lang": "zh" if is_cjk else "en",
    }


# ---------------------------------------------------------------------
# OCR 兜底
# ---------------------------------------------------------------------

_ocr_engine = None


def _get_ocr():
    """惰性加载 RapidOCR。不可用时返回 None 而不是抛异常。"""
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine
    try:
        from rapidocr_onnxruntime import RapidOCR

        _ocr_engine = RapidOCR()
        logger.info("RapidOCR 已加载（ONNXRuntime 后端）")
    except Exception as e:  # noqa: BLE001
        logger.warning("RapidOCR 不可用，扫描页将被跳过: %s", e)
        _ocr_engine = False
    return _ocr_engine or None


def _ocr_page(page: pymupdf.Page, dpi: int = 200) -> str:
    """把一页渲染成位图后走 OCR。"""
    engine = _get_ocr()
    if engine is None:
        return ""
    try:
        import numpy as np

        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:  # RGBA -> RGB
            img = img[:, :, :3]
        result, _ = engine(img)
        if not result:
            return ""
        return "\n".join(line[1] for line in result)
    except Exception as e:  # noqa: BLE001
        logger.warning("第 %d 页 OCR 失败: %s", page.number, e)
        return ""
