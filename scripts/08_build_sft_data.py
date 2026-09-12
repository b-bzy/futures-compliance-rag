#!/usr/bin/env python
"""构造 Qwen3 SFT 训练集。

每条样本 = (问题, 检索到的条款上下文, 带引用编号的答案)。

**拒答样本是这个脚本的重点**。只用"有答案"的样本训练，模型学到的行为是
"永远要给出一个答案"，遇到检索不到的问题时幻觉反而更严重。所以按比例
掺入拒答样本：上下文里**故意剔除**正确条款，标准答案是明确的"无法确定"。

拒答的上下文优先用难负例而不是随机条款。生产里真正会出错的拒答场景是
"top-5 全是只差品种名的近似条款"，模型很容易顺手拿隔壁品种的数字作答；
"检索到一堆毫不相干的东西"那种太容易，学不到东西，只留作兜底。

⚠️ 上下文里不能有捷径特征。难负例若只存了裸文本、而正例带着标题和条号，
渲染出来就是「[1] 正文…」对「[2] 《XX办法》·第十二条 正文…」——
模型学会"选带元数据的那条"就能刷满引用准确率，根本不用读正文。
所以负例必须带上它自己的元数据（`05` 写的 `neg_meta`），
`balance_metadata()` 再兜一道底。

三类样本:
    answer   正常问答，上下文含正确条款，答案带 [n] 引用
    refusal  上下文里没有正确条款，标准答案是拒答
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

# 两种拒答场景的措辞不同。近似条款那种必须点明"讲的是别的品种/别的事项"，
# 否则模型学到的只是一句万能套话，遇到真正的近义干扰仍然会照抄隔壁的数字。
REFUSAL_ANSWERS = {
    "random": (
        "根据现有条款无法确定。提供的参考条款中没有涉及该问题的规定，"
        "建议补充相关交易所的业务规则或合约条款文件后再行查询。"
    ),
    "nearmiss": (
        "根据现有条款无法确定。提供的参考条款规定的是其他品种或其他事项，"
        "并未涵盖所问的内容，据此作答会张冠李戴；"
        "建议补充对应品种的合约条款或业务规则后再行查询。"
    ),
}


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


def balance_metadata(records: list[dict]) -> list[dict]:
    """保证同一条上下文里所有块的元数据完整度一致。

    只要有一个块缺标题，就把所有块的元数据一起抹掉 —— 宁可少一点信息，
    也不能留下"哪条带标题就选哪条"这种与内容无关的捷径。
    """
    if all(r.get("doc_title") for r in records):
        return records
    return [{**r, "doc_title": "", "doc_no": "", "clause_id": ""} for r in records]


def random_others(parents: dict, all_pids: list[str], rng, k: int, exclude: str) -> list[dict]:
    """随机采样 k 个父块作无关上下文。多采一些再排除金标，避免采空。"""
    picked = rng.sample(all_pids, min(k * 3, len(all_pids)))
    return [parents[pid] for pid in picked if pid != exclude][:k]


def hardneg_records(hn: dict | None, k: int) -> list[dict]:
    """把难负例还原成带元数据的上下文记录。没有 neg_meta 时返回空。"""
    if not hn:
        return []
    negs = hn.get("neg") or []
    metas = hn.get("neg_meta") or []
    if not negs or len(metas) < len(negs):
        return []  # 旧版 05 产出的文件没有元数据，交由调用方回退
    return [{**metas[i], "text": t} for i, t in enumerate(negs)][:k]


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

    # 难负例可以直接复用，作为"强干扰条款"与拒答样本的上下文
    hardneg = {r["qa_id"]: r for r in load_jsonl(qa_dir / "reranker_train.jsonl") if r.get("qa_id")}
    with_meta = sum(1 for r in hardneg.values() if r.get("neg_meta"))
    logger.info("可复用的难负例记录 %d 条（其中带元数据 %d 条）", len(hardneg), with_meta)
    if hardneg and with_meta < len(hardneg):
        logger.warning(
            "%d 条难负例没有 neg_meta，这些样本会回退到随机干扰块。"
            "请用当前版本的 scripts/05_mine_hard_negatives.py 重新生成。",
            len(hardneg) - with_meta,
        )

    all_pids = list(parents)
    out_path = qa_dir / "sft_train.jsonl"
    counts = {"answer": 0, "refusal_nearmiss": 0, "refusal_random": 0, "distract": 0}

    with open(out_path, "w", encoding="utf-8") as f:
        for qa in pairs:
            gold = parents[qa["parent_id"]]
            roll = rng.random()
            hn = hardneg.get(qa.get("qa_id"))
            gold_pid = qa["parent_id"]

            if roll < args.refusal_ratio:
                # ---- 拒答样本：上下文里**没有**正确条款 ----
                # 优先用难负例（检索器排在前面但其实答错的那些），
                # 这才是生产里真正会诱发幻觉的场景
                pool = hardneg_records(hn, args.n_context)
                kind = "nearmiss" if pool else "random"
                if not pool:
                    pool = random_others(parents, all_pids, rng, args.n_context, gold_pid)
                if not pool:
                    continue
                sample = {
                    "question": qa["question"],
                    "context": format_context(balance_metadata(pool)),
                    "answer": REFUSAL_ANSWERS[kind],
                    "is_refusal": True,
                    "sample_type": f"refusal_{kind}",
                }
                counts[f"refusal_{kind}"] += 1

            else:
                # ---- 正常/强干扰样本：上下文含正确条款 ----
                use_distract = roll < args.refusal_ratio + args.distract_ratio
                others: list[dict] = []
                if use_distract:
                    # 难负例作干扰项：与正确条款用词高度相似，最考验选源能力
                    others = hardneg_records(hn, args.n_context - 1)
                if not others:
                    others = random_others(parents, all_pids, rng, args.n_context - 1, gold_pid)

                # 正确条款随机插入，避免模型学到"答案总在第 1 条"
                pos_idx = rng.randrange(len(others) + 1)
                ctx_records = balance_metadata(others[:pos_idx] + [gold] + others[pos_idx:])
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
    pct = lambda n: 100 * n / max(total, 1)  # noqa: E731
    refusal = counts["refusal_nearmiss"] + counts["refusal_random"]
    logger.info(
        "完成 %d 条 -> %s\n"
        "  正常 %d (%.0f%%) / 强干扰 %d (%.0f%%) / 拒答 %d (%.0f%%)\n"
        "  拒答里近似条款 %d 条、随机条款 %d 条"
        "（近似条款那种才是生产里真正诱发幻觉的场景，占比越高越好）",
        total, out_path,
        counts["answer"], pct(counts["answer"]),
        counts["distract"], pct(counts["distract"]),
        refusal, pct(refusal),
        counts["refusal_nearmiss"], counts["refusal_random"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
