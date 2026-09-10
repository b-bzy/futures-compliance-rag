#!/usr/bin/env python
"""构造 Qwen3 SFT 训练集。

每条样本 = (问题, 检索到的条款上下文, 带引用编号的答案)。

**拒答样本是这个脚本的重点**。只用"有答案"的样本训练，模型学到的行为是
"永远要给出一个答案"，遇到检索不到的问题时幻觉反而更严重。所以按比例
掺入拒答样本：给一组与问题无关的条款，标准答案是明确的"无法确定"。

三类样本:
    answer   正常问答，上下文含正确条款，答案带 [n] 引用
    refusal  上下文里**故意剔除**正确条款，标准答案是拒答
    distract 上下文含正确条款 + 若干强干扰条款，考验模型选对来源

用法:
    python scripts/08_build_sft_data.py --refusal-ratio 0.15 --limit 5000
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sft-data")

REFUSAL_ANSWER = "根据现有条款无法确定。提供的参考条款中没有涉及该问题的规定，建议补充相关交易所的业务规则或合约条款文件后再行查询。"


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


def format_context(records: list[dict]) -> str:
    """把条款列表渲染成带编号的上下文，格式与推理时 pipeline 用的完全一致。"""
    blocks = []
    for i, r in enumerate(records, 1):
        bits = [f"[{i}]"]
        for key in ("doc_title", "doc_no", "clause_id"):
            if r.get(key):
                bits.append(r[key])
        blocks.append(f"{' · '.join(bits)}\n{r.get('text', '')}")
    return "\n\n---\n\n".join(blocks)


def main() -> int:
    ap = argparse.ArgumentParser(description="构造 SFT 训练集")
    ap.add_argument("--refusal-ratio", type=float, default=0.15, help="拒答样本占比")
    ap.add_argument("--distract-ratio", type=float, default=0.35, help="强干扰样本占比")
    ap.add_argument("--n-context", type=int, default=4, help="每条样本的上下文条款数")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cfg = load_config()
    qa_dir = cfg.resolve("paths.qa")
    processed = cfg.resolve("paths.processed")

    parents = {p["parent_id"]: p for p in load_jsonl(processed / "parents.jsonl")}
    if not parents:
        logger.error("找不到父块，请先执行 scripts/02_parse.py")
        return 1

    pairs = load_jsonl(qa_dir / "gold.jsonl") + load_jsonl(qa_dir / "synthetic.jsonl")
    pairs = [p for p in pairs if p.get("parent_id") in parents and p.get("answer")]
    if not pairs:
        logger.error("找不到 QA 对，请先执行 scripts/04_mine_qa.py")
        return 1
    rng.shuffle(pairs)
    if args.limit:
        pairs = pairs[: args.limit]
    logger.info("可用 QA %d 条，父块 %d 个", len(pairs), len(parents))

    # 难负例可以直接复用，作为"强干扰条款"
    hardneg = {r["qa_id"]: r for r in load_jsonl(qa_dir / "reranker_train.jsonl") if r.get("qa_id")}
    logger.info("可复用的难负例记录 %d 条", len(hardneg))

    all_pids = list(parents)
    out_path = qa_dir / "sft_train.jsonl"
    counts = {"answer": 0, "refusal": 0, "distract": 0}

    with open(out_path, "w", encoding="utf-8") as f:
        for qa in pairs:
            gold = parents[qa["parent_id"]]
            roll = rng.random()

            if roll < args.refusal_ratio:
                # ---- 拒答样本：上下文里**没有**正确条款 ----
                pool = [
                    parents[pid]
                    for pid in rng.sample(all_pids, min(args.n_context * 3, len(all_pids)))
                    if pid != qa["parent_id"]
                ][: args.n_context]
                if not pool:
                    continue
                sample = {
                    "question": qa["question"],
                    "context": format_context(pool),
                    "answer": REFUSAL_ANSWER,
                    "is_refusal": True,
                    "sample_type": "refusal",
                }
                counts["refusal"] += 1

            else:
                # ---- 正常/强干扰样本：上下文含正确条款 ----
                others: list[dict] = []
                use_distract = roll < args.refusal_ratio + args.distract_ratio
                hn = hardneg.get(qa.get("qa_id"))
                if use_distract and hn and hn.get("neg"):
                    # 难负例作干扰项：与正确条款用词高度相似，最考验选源能力
                    others = [
                        {"doc_title": "", "clause_id": "", "text": t}
                        for t in hn["neg"][: args.n_context - 1]
                    ]
                if not others:
                    others = [
                        parents[pid]
                        for pid in rng.sample(all_pids, min(args.n_context * 2, len(all_pids)))
                        if pid != qa["parent_id"]
                    ][: args.n_context - 1]

                # 正确条款随机插入，避免模型学到"答案总在第 1 条"
                pos_idx = rng.randrange(len(others) + 1)
                ctx_records = others[:pos_idx] + [gold] + others[pos_idx:]
                citation = pos_idx + 1

                answer = f"{qa['answer']}[{citation}]"
                sample = {
                    "question": qa["question"],
                    "context": format_context(ctx_records),
                    "answer": answer,
                    "is_refusal": False,
                    "sample_type": "distract" if use_distract else "answer",
                    "gold_citation": citation,
                }
                counts["distract" if use_distract else "answer"] += 1

            sample["qa_id"] = qa.get("qa_id")
            sample["tier"] = qa.get("tier")
            sample["source_url"] = qa.get("source_url", "")
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    total = sum(counts.values())
    logger.info(
        "完成 %d 条 -> %s\n  正常 %d (%.0f%%) / 强干扰 %d (%.0f%%) / 拒答 %d (%.0f%%)",
        total, out_path,
        counts["answer"], 100 * counts["answer"] / max(total, 1),
        counts["distract"], 100 * counts["distract"] / max(total, 1),
        counts["refusal"], 100 * counts["refusal"] / max(total, 1),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
