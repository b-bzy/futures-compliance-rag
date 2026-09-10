#!/usr/bin/env python
"""挖掘难负例，构造 reranker 训练集。

难负例是 reranker 训练成败的关键。随机负例太容易区分，模型学不到东西；
真正有用的负例是**检索器自己排在前面但其实是错的**那些 —— 它们与正例
共享大量术语（都在讲"保证金"、都在讲"行权"），只有细读才能分辨，
这正是 cross-encoder 该学会的能力。

做法：用当前检索器对每个 QA 的问题做检索，把 top-N 里 parent_id
不等于金标的候选取作负例。同时剔除与正例文本高度重合的候选 ——
它们往往是同一条款的不同子块，标成负例是错的标注。

产出 data/qa/reranker_train.jsonl，格式与 FlagEmbedding 训练脚本一致：
    {"query": ..., "pos": [...], "neg": [...]}

用法:
    python scripts/05_mine_hard_negatives.py --n-neg 7 --limit 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from derivrag.config import load_config  # noqa: E402
from derivrag.pipeline import RAGPipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hardneg")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def char_overlap(a: str, b: str) -> float:
    """字符集合的 Jaccard 相似度 —— 用来剔除"其实是正例"的伪负例。"""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def main() -> int:
    ap = argparse.ArgumentParser(description="挖掘难负例")
    ap.add_argument("--n-neg", type=int, default=7, help="每条样本的负例数")
    ap.add_argument("--pool", type=int, default=30, help="从 top-N 候选里挑负例")
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少条 QA")
    ap.add_argument(
        "--max-overlap",
        type=float,
        default=0.85,
        help="与正例字符重合度超过此值的候选不作为负例（大概率是同条款的别的子块）",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    cfg = load_config()
    qa_dir = cfg.resolve("paths.qa")

    # 训练集用金标 + 合成集；金标数量少，合成集提供规模。
    # 这里刻意用**原始**金标（未改写）—— 改写是为了评测去泄漏，
    # 训练数据反而应该覆盖用户真实会问的原话。
    pairs = load_jsonl(qa_dir / "gold.jsonl") + load_jsonl(
        cfg.resolve("eval.synthetic_file")
    )
    pairs = [p for p in pairs if p.get("parent_id") and p.get("question")]
    if not pairs:
        logger.error("没有可用的 QA 对，请先执行 scripts/04_mine_qa.py")
        return 1
    random.shuffle(pairs)
    if args.limit:
        pairs = pairs[: args.limit]
    logger.info("待处理 QA %d 条", len(pairs))

    pipeline = RAGPipeline(cfg, lazy=True)
    parents = pipeline.retriever.parents

    out_path = qa_dir / "reranker_train.jsonl"
    written = 0
    skipped_no_pos = 0
    neg_counts: list[int] = []

    with open(out_path, "w", encoding="utf-8") as f:
        for i, qa in enumerate(pairs, 1):
            gold_pid = qa["parent_id"]
            pos_rec = parents.get(gold_pid)
            if not pos_rec:
                skipped_no_pos += 1
                continue
            pos_text = pos_rec.get("text", "")
            if len(pos_text) < 20:
                skipped_no_pos += 1
                continue

            # 关掉重排，我们要的正是**粗排**里的难负例
            try:
                result = pipeline.answer(
                    qa["question"],
                    use_hyde=False,
                    use_rewrite=False,
                    use_rerank=False,
                    top_k=args.pool,
                    generate=False,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("检索失败: %s", e)
                continue

            negs: list[str] = []
            for hit in result.hits:
                if hit.metadata.get("parent_id") == gold_pid:
                    continue
                text = hit.parent_text or hit.text
                if len(text) < 20:
                    continue
                if char_overlap(text, pos_text) > args.max_overlap:
                    continue  # 疑似同条款，不能当负例
                negs.append(text[:1500])
                if len(negs) >= args.n_neg:
                    break

            if not negs:
                continue

            f.write(
                json.dumps(
                    {
                        "query": qa["question"],
                        "pos": [pos_text[:1500]],
                        "neg": negs,
                        "qa_id": qa.get("qa_id"),
                        "tier": qa.get("tier"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
            neg_counts.append(len(negs))

            if i % 50 == 0:
                logger.info("进度 %d/%d，已写 %d 条", i, len(pairs), written)

    avg_neg = sum(neg_counts) / max(len(neg_counts), 1)
    logger.info(
        "完成: %d 条训练样本，平均负例 %.1f 个，跳过（找不到正例）%d 条 -> %s",
        written, avg_neg, skipped_no_pos, out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
