#!/usr/bin/env python
"""解析 data/raw 下的原始文档，切成父块与子块。

用法:
    python scripts/02_parse.py                 # 全量解析
    python scripts/02_parse.py --no-semantic   # 跳过语义切分（不加载模型，快速验证）
    python scripts/02_parse.py --limit 20      # 只处理前 N 份

产出:
    data/interim/documents.jsonl    整篇文档（RawDocument）
    data/processed/parents.jsonl    条款级父块
    data/processed/children.jsonl   语义子块（向量库实际存储单元）
    data/processed/stats.json       真实块数统计 —— README 里的数字直接引用这个
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from derivrag.chunk.clause import split_document  # noqa: E402
from derivrag.chunk.semantic import build_children  # noqa: E402
from derivrag.config import load_config, load_sources, repo_root  # noqa: E402
from derivrag.parse.clean import (  # noqa: E402
    extract_doc_no,
    extract_effective_date,
    infer_effective_status,
)
from derivrag.schema import RawDocument  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("parse")
logging.getLogger("derivrag.parse.html").setLevel(logging.ERROR)


def load_manifest(path: Path) -> dict[str, dict]:
    """读 manifest，按 id 去重（后写的覆盖先写的，重跑时取最新一次结果）。"""
    records: dict[str, dict] = {}
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") in ("ok", "skipped") and rec.get("local_path"):
                records[rec["id"]] = rec
    return records


def title_from_notes(notes: str | None, fallback: str) -> str:
    """adapter 把真实标题写进了 notes 的 title= 字段。"""
    if notes and notes.startswith("title="):
        return notes.split("|", 1)[0][len("title=") :].strip()
    return fallback


# HTML 的 <title> 几乎都带站点名后缀，形式各站不同：
#   "……吗？|中国金融期货交易所"     (CFFEX，竖线前后无空格)
#   "上证50ETF期权合约基本条款 | 上海证券交易所"
#   "……_深圳证券交易所"
# 这些后缀如果留着，会污染标题（CFFEX 的问答标题本身就是问题文本）。
_SITE_SUFFIX = re.compile(
    r"\s*[|｜_\-–—]\s*(中国金融期货交易所|上海证券交易所|深圳证券交易所|上海期货交易所"
    r"|大连商品交易所|郑州商品交易所|SGX|HKEX|Rulebooks?)\s*$"
)


def clean_title(title: str) -> str:
    """去掉站点名后缀，保留真实标题。"""
    title = re.sub(r"\s+", " ", (title or "")).strip()
    for _ in range(3):  # 少数标题带两层后缀
        new = _SITE_SUFFIX.sub("", title).strip()
        if new == title:
            break
        title = new
    # 仍带竖线的，取第一段
    if "|" in title or "｜" in title:
        title = re.split(r"[|｜]", title)[0].strip()
    return title


def build_title_map(sources: dict) -> dict[str, str]:
    """从 sources.yaml 取权威标题 {doc_id: title}。

    PDF 直链条目没有 HTML <title>，早期版本会把标题回退成 doc_id，
    导致用户看到的引用是 "szse_sz100etf_contract" 而不是
    「深证100ETF期权合约基本条款」，条款表父块的主语也跟着变成 doc_id。
    所以 sources.yaml 里为每个 kind=doc 条目显式写了 title，作为最高优先级。
    """
    out: dict[str, str] = {}
    for group in ("zh", "en"):
        for entry in sources.get(group, []) or []:
            if entry.get("title") and entry.get("id"):
                out[entry["id"]] = entry["title"].strip()
    return out


def resolve_title(rec: dict, parsed: dict, path: Path, title_map: dict[str, str]) -> str:
    """标题优先级：sources.yaml > 解析出的 <title> > adapter 记录 > PDF 排版 > doc_id。"""
    doc_id = rec["id"]
    if doc_id in title_map:
        return title_map[doc_id]

    for candidate in (parsed.get("title"), title_from_notes(rec.get("notes"), "")):
        cleaned = clean_title(candidate or "")
        if cleaned and cleaned != doc_id:
            return cleaned

    if rec.get("expect") == "pdf":
        from derivrag.parse.pdf import extract_pdf_title

        cleaned = clean_title(extract_pdf_title(path))
        if len(cleaned) >= 6:
            return cleaned

    logger.warning("标题回退为 doc_id: %s", doc_id)
    return doc_id


def parse_one(rec: dict, cfg, title_map: dict[str, str] | None = None) -> RawDocument | None:
    """解析单份文档。"""
    # manifest 里的 local_path 可能是绝对路径（Fetcher 写入时 cfg.resolve 会
    # 无条件转绝对），也可能是相对路径。绝对路径在 data/ 被搬动后会失效，
    # 相对路径若按 CWD 解析则要求必须在仓库根目录下执行。两种都统一按
    # 仓库根目录还原，从而既不依赖 data/ 的历史位置、也不依赖 CWD。
    raw_path = Path(rec["local_path"])
    path = raw_path if raw_path.is_absolute() else repo_root() / raw_path
    if not path.exists():
        # 绝对路径失效的常见原因是 data/ 整体搬过位置，按文件名回退到当前 raw 目录
        fallback = cfg.resolve("paths.raw") / raw_path.name
        if fallback.exists():
            path = fallback
    if not path.exists():
        logger.warning("文件不存在: %s", path)
        return None

    fmt = rec.get("expect", path.suffix.lstrip("."))
    ocr_cfg = cfg.get_path("parse.ocr", {})

    try:
        if fmt == "pdf":
            from derivrag.parse.pdf import parse_pdf

            out = parse_pdf(
                path,
                ocr_enabled=ocr_cfg.get("enabled", True),
                min_chars_per_page=ocr_cfg.get("min_chars_per_page", 50),
                min_image_ratio=ocr_cfg.get("min_image_area_ratio", 0.30),
            )
        elif fmt == "html":
            from derivrag.parse.html import parse_html

            out = parse_html(path, url=rec["url"])
        elif fmt == "docx":
            from derivrag.parse.docfile import parse_docx

            out = parse_docx(path)
        elif fmt == "doc":
            from derivrag.parse.docfile import parse_doc

            out = parse_doc(path, soffice=cfg.get_path("parse.soffice_bin"))
        else:
            logger.warning("未知格式 %s: %s", fmt, path.name)
            return None
    except Exception as e:  # noqa: BLE001 - 单份失败不应中断全量解析
        logger.error("解析 %s 失败: %s", path.name, e)
        return None

    text = out.get("text", "")
    title = resolve_title(rec, out, path, title_map or {})

    return RawDocument(
        doc_id=rec["id"],
        title=title,
        venue=rec.get("venue", ""),
        lang=out.get("lang", rec.get("lang", "zh")),
        source_url=rec["url"],
        retrieval_date=rec.get("retrieval_date", ""),
        fmt=fmt,
        text=text,
        tables=out.get("tables", []),
        doc_no=extract_doc_no(text),
        effective_status=infer_effective_status(title, text),
        effective_date=extract_effective_date(text),
        page_count=out.get("page_count"),
        ocr_pages=out.get("ocr_pages", []),
        parse_warnings=out.get("warnings", []),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="解析与分块")
    ap.add_argument("--no-semantic", action="store_true", help="跳过语义切分，只按长度切子块")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    manifest = load_manifest(cfg.resolve("paths.manifest"))
    title_map = build_title_map(load_sources())
    logger.info(
        "manifest 中有 %d 份可解析文档，sources.yaml 提供 %d 个权威标题",
        len(manifest),
        len(title_map),
    )

    records = list(manifest.values())
    if args.limit:
        records = records[: args.limit]

    interim = cfg.resolve("paths.interim")
    processed = cfg.resolve("paths.processed")
    interim.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    # ---------- 解析 ----------
    docs: list[RawDocument] = []
    for i, rec in enumerate(records, 1):
        d = parse_one(rec, cfg, title_map)
        if d and d.text.strip():
            docs.append(d)
        if i % 25 == 0:
            logger.info("已解析 %d/%d", i, len(records))

    logger.info("解析完成: %d/%d 份产出非空正文", len(docs), len(records))

    with open(interim / "documents.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(d.to_json() + "\n")

    # ---------- 父块 ----------
    chunk_cfg = cfg.get_path("chunk", {})
    parents = []
    for d in docs:
        parents.extend(
            split_document(
                d,
                max_chars=chunk_cfg.get("parent", {}).get("max_chars", 4000),
                min_chars=chunk_cfg.get("parent", {}).get("min_chars", 30),
            )
        )
    logger.info("父块 %d 个", len(parents))

    with open(processed / "parents.jsonl", "w", encoding="utf-8") as f:
        for p in parents:
            f.write(p.to_json() + "\n")

    # ---------- 子块 ----------
    child_cfg = chunk_cfg.get("child", {})
    # 配置里的 strategy 与命令行 --no-semantic 都能关掉语义切分，
    # 命令行优先（方便快速验证时不加载 2.27GB 权重）
    use_semantic = (not args.no_semantic) and child_cfg.get("strategy", "semantic") == "semantic"

    embed_fn = None
    if use_semantic:
        from derivrag.index.embed import get_embedder

        embedder = get_embedder(cfg)
        embed_fn = embedder.encode_dense
        logger.info("语义切分已启用（%s）", embedder.model_name)
    else:
        logger.info("语义切分已跳过，按长度切分")

    children = []
    for i, p in enumerate(parents, 1):
        children.extend(
            build_children(
                p,
                embed_fn,
                target_chars=child_cfg.get("target_chars", 300),
                max_chars=child_cfg.get("max_chars", 512),
                min_chars=child_cfg.get("min_chars", 80),
                breakpoint_z=child_cfg.get("breakpoint_z", 1.0),
                overlap_sentences=child_cfg.get("overlap_sentences", 1),
            )
        )
        if i % 500 == 0:
            logger.info("子块进度 %d/%d 父块 -> %d 子块", i, len(parents), len(children))

    logger.info("子块 %d 个", len(children))
    with open(processed / "children.jsonl", "w", encoding="utf-8") as f:
        for c in children:
            f.write(c.to_json() + "\n")

    # ---------- 统计 ----------
    stats = {
        "documents_fetched": len(manifest),
        "documents_parsed": len(docs),
        "parents": len(parents),
        "children": len(children),
        "by_venue": dict(Counter(d.venue for d in docs)),
        "by_format": dict(Counter(d.fmt for d in docs)),
        "by_lang": dict(Counter(d.lang for d in docs)),
        "parent_types": dict(Counter(p.chunk_type for p in parents)),
        "effective_status": dict(Counter(p.effective_status for p in parents)),
        "total_chars": sum(len(d.text) for d in docs),
        "docs_with_warnings": sum(1 for d in docs if d.parse_warnings),
        "ocr_pages_total": sum(len(d.ocr_pages) for d in docs),
        "semantic_chunking": use_semantic,
    }
    (processed / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("统计: %s", json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
