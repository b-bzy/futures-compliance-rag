#!/usr/bin/env python
"""挖掘难负例，构造 reranker 训练集。

难负例是 reranker 训练成败的关键。随机负例太容易区分，模型学不到东西；
真正有用的负例是**检索器自己排在前面但其实是错的**那些 —— 它们与正例
共享大量术语（都在讲"保证金"、都在讲"行权"），只有细读才能分辨，
这正是 cross-encoder 该学会的能力。

做法：用当前检索器对每个 QA 的问题做检索，把 top-N 里 parent_id
不等于金标的候选取作负例。

⚠️ 什么**不能**当负例 —— 这条判错会直接教坏 reranker:

    早先的判据是"与正例字符重合度 > 0.85 就剔除"，理由是它们多半是
    同一条款的不同子块。这条在中文条款表上是错的:

        沪深300ETF期权的行权价格：9个（1个平值合约、4个虚值…
        深证100ETF期权的行权价格：9个（1个平值合约、4个虚值…
        字符集 Jaccard = 0.893 -> 被当成"同一条款"剔掉

    可这正是本任务最该学的难负例（"在十几个只差品种名的近似块里选对一个"）。
    实测该规则会误删 25% 的同字段跨品种样本 —— 把最有价值的训练信号删掉了。

    现在改按**事实身份**判定，与字面相似度无关:
        条款表行  (品种, 字段) 相同 -> 同一个事实（《合约基本条款》与
                  《上市交易通知》印的是同一张表），互为正确答案，不能当负例
        条文      正文归一化后相同或互为子串 -> 同一条款在另一份文档里重印
    除此之外，只要 parent_id 不同就是合法负例，哪怕它长得几乎一样。

产出 data/qa/reranker_train.jsonl，格式与 FlagEmbedding 训练脚本一致：
    {"query": ..., "pos": [...], "neg": [...]}
额外写入 `pos_meta` / `neg_meta`（标题、条号）供 08_build_sft_data.py 使用 ——
FlagEmbedding 只读 query/pos/neg，多余字段会被忽略。缺了它，SFT 的干扰项
就只有裸文本、而正例带标题，模型能靠"哪条有元数据"猜引用编号。

用法:
    python scripts/05_mine_hard_negatives.py --n-neg 7 --limit 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
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


def _norm(s: str | None) -> str:
    """比对用的归一化：去掉全部空白。"""
    return re.sub(r"\s+", "", s or "")


def _subject_of(rec: dict) -> str:
    """条款表行的主语（品种名）。切分时写在 section_path[0]，兜底用文档标题。"""
    return _norm((rec.get("section_path") or [""])[0] or rec.get("doc_title", ""))


def same_fact(cand: dict, pos: dict) -> bool:
    """候选块与正例是否在陈述**同一个事实**（互为正确答案，不能当负例）。

    注意判的是事实身份，不是文本相似度 —— "只差品种名"的两行讲的是
    两个不同事实，必须保留为负例（见模块 docstring）。
    """
    if cand.get("chunk_type") == "table_row" and pos.get("chunk_type") == "table_row":
        return _subject_of(cand) == _subject_of(pos) and _norm(cand.get("clause_id")) == _norm(
            pos.get("clause_id")
        )

    a, b = _norm(cand.get("text")), _norm(pos.get("text"))
    if not a or not b:
        return False
    if a == b:
        return True
    # 同一条款被另一份文档整段重印（修订版、制度汇编本）。
    # 只在较短一方足够长时才认互为子串，否则"第三条 本细则未规定的……"
    # 这类通用短句会把正常负例误伤。
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return len(shorter) >= 30 and shorter in longer


def neg_meta_of(rec: dict) -> dict:
    """负例随身携带的元数据 —— 08 渲染 SFT 上下文时要用。"""
    return {
        "doc_title": rec.get("doc_title", ""),
        "doc_no": rec.get("doc_no") or "",
        "clause_id": rec.get("clause_id") or "",
        "parent_id": rec.get("parent_id", ""),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="挖掘难负例")
    ap.add_argument("--n-neg", type=int, default=7, help="每条样本的负例数")
    ap.add_argument("--pool", type=int, default=30, help="从 top-N 候选里挑负例")
    ap.add_argument("--limit", type=int, default=0, help="最多处理多少条 QA")
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
    dropped_same_fact = 0
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
            negs_meta: list[dict] = []
            seen_pids: set[str] = set()
            for hit in result.hits:
                cand_pid = hit.metadata.get("parent_id", "")
                if cand_pid == gold_pid or cand_pid in seen_pids:
                    continue
                # 同一父块的多个子块都可能召回，按 parent 去重后只留一条
                cand_rec = parents.get(cand_pid) or dict(hit.metadata)
                text = hit.parent_text or cand_rec.get("text") or hit.text
                if len(text) < 20:
                    continue
                if same_fact(cand_rec, pos_rec):
                    dropped_same_fact += 1
                    continue
                seen_pids.add(cand_pid)
                negs.append(text[:1500])
                negs_meta.append(neg_meta_of(cand_rec))
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
                        # 以下字段 FlagEmbedding 不读，供 08_build_sft_data.py 用
                        "pos_meta": neg_meta_of(pos_rec),
                        "neg_meta": negs_meta,
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
        "完成: %d 条训练样本，平均负例 %.1f 个\n"
        "  跳过（找不到正例）%d 条；剔除（与正例是同一个事实）%d 个候选 -> %s",
        written, avg_neg, skipped_no_pos, dropped_same_fact, out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
