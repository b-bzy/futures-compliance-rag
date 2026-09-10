#!/usr/bin/env python
"""构建向量索引与 BM25 索引。

用法:
    python scripts/03_build_index.py            # 增量（已有则跳过）
    python scripts/03_build_index.py --rebuild  # 清空重建

产出:
    data/chroma/                  Chroma 持久化向量库
    data/interim/bm25_index.pkl   BM25 索引
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from derivrag.config import load_config  # noqa: E402
from derivrag.schema import ChildChunk  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("index")


def load_children(path: Path) -> list[ChildChunk]:
    chunks: list[ChildChunk] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunks.append(ChildChunk(**rec))
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser(description="构建索引")
    ap.add_argument("--rebuild", action="store_true", help="清空后重建")
    ap.add_argument("--batch-size", type=int, default=16, help="向量化批大小")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    children_path = cfg.resolve("paths.processed") / "children.jsonl"
    if not children_path.exists():
        logger.error("找不到 %s，请先执行 scripts/02_parse.py", children_path)
        return 1

    chunks = load_children(children_path)
    if args.limit:
        chunks = chunks[: args.limit]
    logger.info("载入 %d 个子块", len(chunks))

    # ---------- BM25 ----------
    from derivrag.index.bm25 import BM25Index

    t = time.perf_counter()
    bm25 = BM25Index()
    bm25.build(chunks)
    bm25.save(cfg.resolve("paths.bm25"))
    logger.info("BM25 构建耗时 %.1fs", time.perf_counter() - t)

    # ---------- 向量库 ----------
    from derivrag.index.embed import get_embedder
    from derivrag.index.vectorstore import VectorStore

    vs = VectorStore(cfg.resolve("paths.chroma"))
    if args.rebuild:
        vs.reset()

    existing = vs.count()
    if existing and not args.rebuild:
        logger.info("向量库已有 %d 条，只写入新增块（如需全量重建请加 --rebuild）", existing)
        have = set()
        # Chroma 没有"列出全部 id"的轻量接口，分页取回
        offset = 0
        while offset < existing:
            page = vs.collection.get(limit=2000, offset=offset, include=[])
            ids = page.get("ids", [])
            if not ids:
                break
            have.update(ids)
            offset += len(ids)
        chunks = [c for c in chunks if c.child_id not in have]
        logger.info("待写入 %d 个新块", len(chunks))

    if not chunks:
        logger.info("没有需要写入的新块")
        return 0

    embedder = get_embedder(cfg)
    embedder.batch_size = args.batch_size

    t = time.perf_counter()
    texts = [c.text for c in chunks]
    total = len(texts)
    written = 0
    # 分批编码 + 写入，避免一次性把 5 万条向量堆在内存里
    step = max(args.batch_size * 32, 256)
    for i in range(0, total, step):
        batch = chunks[i : i + step]
        vectors = embedder.encode_dense(texts[i : i + step])
        written += vs.add(batch, vectors)
        elapsed = time.perf_counter() - t
        rate = written / max(elapsed, 1e-6)
        eta = (total - written) / max(rate, 1e-6)
        logger.info(
            "向量化 %d/%d (%.1f 块/秒, 预计剩余 %.1f 分钟)", written, total, rate, eta / 60
        )

    logger.info("完成: 向量库共 %d 条，耗时 %.1f 分钟", vs.count(), (time.perf_counter() - t) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
