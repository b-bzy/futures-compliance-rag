#!/usr/bin/env python
"""抓取语料。

用法:
    python scripts/01_crawl.py --dry-run          # 只探测，不落盘
    python scripts/01_crawl.py                    # 抓中文语料
    python scripts/01_crawl.py --include-en       # 同时抓英文层
    python scripts/01_crawl.py --only sse,cffex   # 只抓指定交易所
    python scripts/01_crawl.py --force            # 忽略本地缓存重抓

产出:
    data/raw/*.{pdf,html,doc,docx}
    data/raw/manifest.jsonl      每行一条 FetchRecord
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from derivrag.config import load_config, load_sources  # noqa: E402
from derivrag.crawl.adapters import DiscoveredDoc, get_adapter  # noqa: E402
from derivrag.crawl.fetcher import Fetcher  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crawl")


def expand_sources(fetcher: Fetcher, entries: list[dict], only: set[str] | None) -> list[DiscoveredDoc]:
    """把 sources.yaml 里的条目展开成扁平的待下载列表。

    kind=doc   直接就是一个待下载文档
    kind=index 交给对应 adapter 展开
    """
    discovered: list[DiscoveredDoc] = []
    for entry in entries:
        venue = entry.get("venue", "")
        if only and venue.lower() not in only:
            continue

        lang = entry.get("lang", "zh")
        if entry.get("kind") == "index":
            adapter_name = entry.get("adapter")
            if not adapter_name:
                logger.error("%s 是 index 但未指定 adapter，跳过", entry.get("id"))
                continue
            try:
                found = get_adapter(adapter_name)(fetcher, entry["url"], venue)
            except Exception as e:  # noqa: BLE001
                logger.error("适配器 %s 展开 %s 失败: %s", adapter_name, entry["url"], e)
                continue
            discovered.extend(found)
        else:
            discovered.append(
                DiscoveredDoc(
                    id=entry["id"],
                    url=entry["url"],
                    title=entry.get("id", ""),
                    expect=entry.get("expect", "html"),
                    venue=venue,
                    lang=lang,
                    notes=(entry.get("notes") or "").strip() or None,
                )
            )
    return discovered


def main() -> int:
    ap = argparse.ArgumentParser(description="抓取衍生品交易条款语料")
    ap.add_argument("--dry-run", action="store_true", help="只探测 URL 可用性，不下载")
    ap.add_argument("--include-en", action="store_true", help="同时抓取英文层")
    ap.add_argument("--only", default="", help="逗号分隔的交易所代码，如 sse,cffex")
    ap.add_argument("--force", action="store_true", help="忽略本地缓存重新下载")
    ap.add_argument("--limit", type=int, default=0, help="最多处理前 N 项（调试用）")
    args = ap.parse_args()

    cfg = load_config()
    sources = load_sources(cfg.get_path("crawl.sources_file"))
    only = {s.strip().lower() for s in args.only.split(",") if s.strip()} or None

    raw_dir = cfg.resolve("paths.raw")
    manifest_path = cfg.resolve("paths.manifest")
    crawl_cfg = cfg.get_path("crawl", {})

    entries = list(sources.get("zh", []))
    if args.include_en or cfg.get_path("crawl.include_en"):
        entries += list(sources.get("en", []))

    with Fetcher(
        raw_dir,
        rate_limit_seconds=crawl_cfg.get("rate_limit_seconds", 1.0),
        timeout_seconds=crawl_cfg.get("timeout_seconds", 60),
        max_retries=crawl_cfg.get("max_retries", 3),
        user_agent=sources.get("defaults", {}).get("user_agent", ""),
        waf_length_fingerprints=tuple(crawl_cfg.get("waf_length_fingerprints", [10648])),
    ) as fetcher:

        logger.info("展开语料源清单（%d 个入口）…", len(entries))
        docs = expand_sources(fetcher, entries, only)
        if args.limit:
            docs = docs[: args.limit]
        logger.info("共发现 %d 个待处理文档", len(docs))

        if args.dry_run:
            ok = bad = 0
            for d in docs:
                good, msg = fetcher.probe(d.url, expect=d.expect)
                status = "✓" if good else "✗"
                (logger.info if good else logger.error)(
                    "%s %-46s %s", status, d.id[:46], msg
                )
                ok, bad = (ok + 1, bad) if good else (ok, bad + 1)
            logger.info("探测完成: %d 可用 / %d 失败", ok, bad)
            return 0 if bad == 0 else 1

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        counts = {"ok": 0, "skipped": 0, "failed": 0}
        with open(manifest_path, "a", encoding="utf-8") as mf:
            for i, d in enumerate(docs, 1):
                rec = fetcher.fetch(
                    d.id,
                    d.url,
                    expect=d.expect,
                    venue=d.venue,
                    lang=d.lang,
                    force=args.force,
                    notes=d.notes,
                )
                # 把 adapter 发现的真实标题带进 manifest，解析阶段要用
                if d.title and d.title != d.id:
                    rec.notes = f"title={d.title}" + (f" | {rec.notes}" if rec.notes else "")
                mf.write(rec.to_json() + "\n")
                mf.flush()
                counts[rec.status] = counts.get(rec.status, 0) + 1
                if i % 20 == 0:
                    logger.info("进度 %d/%d  %s", i, len(docs), counts)

        logger.info("抓取完成: %s", counts)
        logger.info("清单写入 %s", manifest_path)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
