#!/usr/bin/env python
"""消融实验 —— 逐级验证每个组件到底带来多少提升。

这张表是整个项目最有说服力的产出：它把"混合检索提升 12%""HyDE 提升 2%"
这类口号变成可复现的数字，也让面试时的每一句话都有据可查。

各档位:
    dense_only          仅稠密向量（近似于 ChatPDF 一类工具的做法，作为基线）
    bm25_only           仅 BM25
    bm25_dense_rrf      两路 + 标准 RRF（langchain EnsembleRetriever 的做法）
    three_score_rrf     三路 + 自研 score-aware RRF
    plus_rerank         + cross-encoder 重排
    plus_hyde           + HyDE
    (plus_rewrite 需要多轮上下文，在 06_eval.py 的多轮用例里单独测)

用法:
    python scripts/07_ablation.py                 # 全部档位
    python scripts/07_ablation.py --limit 40      # 抽样快跑
    python scripts/07_ablation.py --stages dense_only,plus_rerank
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
from derivrag.eval.dataset import describe_gold, load_gold  # noqa: E402
from derivrag.eval.metrics import compute_metrics, rank_of  # noqa: E402
from derivrag.pipeline import RAGPipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ablation")
logging.getLogger("derivrag.index.embed").setLevel(logging.WARNING)

# (档位名, 通路, 是否重排, 是否 HyDE, 融合方式)
STAGES = {
    "dense_only": (("dense",), False, False, "plain_rrf"),
    "bm25_only": (("bm25",), False, False, "plain_rrf"),
    "bm25_dense_rrf": (("bm25", "dense"), False, False, "plain_rrf"),
    "three_score_rrf": (("bm25", "dense", "keyword"), False, False, "score_rrf"),
    "plus_rerank": (("bm25", "dense", "keyword"), True, False, "score_rrf"),
    "plus_hyde": (("bm25", "dense", "keyword"), True, True, "score_rrf"),
}

STAGE_LABELS = {
    "dense_only": "仅稠密向量（基线）",
    "bm25_only": "仅 BM25",
    "bm25_dense_rrf": "BM25+稠密，标准 RRF",
    "three_score_rrf": "三路召回，自研 score-RRF",
    "plus_rerank": "+ 交叉编码器重排",
    "plus_hyde": "+ HyDE",
}


def run_stage(pipeline: RAGPipeline, gold: list[dict], stage: str, ks: list[int]) -> dict:
    channels, use_rerank, use_hyde, fusion_method = STAGES[stage]

    # 直接改检索器的融合配置，避免为每个档位重建索引
    pipeline.retriever.fusion_cfg = dict(pipeline.retriever.fusion_cfg)
    pipeline.retriever.fusion_cfg["method"] = fusion_method

    ranks: list[int | None] = []
    latencies: list[float] = []
    max_k = max(ks)

    for i, item in enumerate(gold, 1):
        t = time.perf_counter()
        try:
            result = pipeline.answer(
                item["question"],
                use_hyde=use_hyde,
                use_rewrite=False,
                use_rerank=use_rerank,
                channels=channels,
                top_k=max_k,
                generate=False,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("查询失败 %r: %s", item["question"][:40], e)
            ranks.append(None)
            continue
        latencies.append(time.perf_counter() - t)

        retrieved = [h.metadata.get("parent_id", "") for h in result.hits]
        ranks.append(rank_of(retrieved, item["parent_id"]))

        if i % 20 == 0:
            hit = sum(1 for r in ranks if r is not None and r <= 5)
            logger.info("  %s %d/%d (当前 Recall@5=%.1f%%)", stage, i, len(gold), 100 * hit / i)

    m = compute_metrics(ranks, latencies, ks)
    return m.to_dict()


def render_markdown(results: dict, baseline: str = "dense_only") -> str:
    """生成 README 里直接可用的消融表。"""
    lines = [
        "| 档位 | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@5 | 平均延迟 | P95 延迟 | 相对基线 Recall@5 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    base_r5 = results.get(baseline, {}).get("recall", {}).get("@5")
    for stage in STAGES:
        r = results.get(stage)
        if not r:
            continue
        rec = r["recall"]
        delta = ""
        if base_r5 and stage != baseline:
            diff = (rec.get("@5", 0) - base_r5) * 100
            delta = f"{diff:+.1f} pp"
        elif stage == baseline:
            delta = "—（基线）"
        lines.append(
            f"| {STAGE_LABELS.get(stage, stage)} "
            f"| {rec.get('@1', 0):.3f} | {rec.get('@5', 0):.3f} | {rec.get('@10', 0):.3f} "
            f"| {r['mrr']:.3f} | {r['ndcg'].get('@5', 0):.3f} "
            f"| {r['mean_latency_s']:.2f}s | {r['p95_latency_s']:.2f}s | {delta} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="消融实验")
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条金标（快跑）")
    ap.add_argument("--stages", default="", help="逗号分隔的档位名")
    ap.add_argument("--gold", default="", help="金标文件路径，默认读配置")
    args = ap.parse_args()

    cfg = load_config()
    gold_path = Path(args.gold) if args.gold else cfg.resolve("eval.gold_file")
    if not gold_path.exists():
        logger.error("金标文件不存在: %s，请先执行 scripts/04_mine_qa.py --tier1", gold_path)
        return 1

    gold = load_gold(gold_path, args.limit)
    if not gold:
        logger.error("金标集为空或缺少 parent_id 字段")
        return 1
    logger.info("评测集 %s：%d 条 —— %s", gold_path.name, len(gold), describe_gold(gold))

    # 优先级：命令行 --stages > 配置 eval.ablation_stages > 全部档位
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if not stages:
        stages = list(cfg.get_path("eval.ablation_stages", []) or STAGES)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        logger.error("未知档位 %s，可用: %s", unknown, list(STAGES))
        return 1

    ks = cfg.get_path("eval.retrieval_k", [1, 3, 5, 10])
    pipeline = RAGPipeline(cfg, lazy=True)

    results: dict[str, dict] = {}
    for stage in stages:
        logger.info("=== 档位 %s (%s) ===", stage, STAGE_LABELS.get(stage, ""))
        results[stage] = run_stage(pipeline, gold, stage, ks)
        logger.info("%s -> %s", stage, json.dumps(results[stage], ensure_ascii=False))

    report_dir = cfg.resolve("eval.report_dir")
    report_dir.mkdir(parents=True, exist_ok=True)

    (report_dir / "ablation.json").write_text(
        json.dumps({"n_gold": len(gold), "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    table = render_markdown(results)
    # 评测集描述必须如实反映用的是哪个集合 —— 写死"人工撰写金标"会在
    # 跑困难集（模板生成 + LLM 改写）时变成假陈述
    md = (
        f"# 检索消融实验\n\n"
        f"- 评测集：`{gold_path.name}`，{len(gold)} 条 —— {describe_gold(gold)}\n"
        f"- 命中判定：返回候选的 parent_id 是否等于金标答案所在条款\n"
        f"- 硬件：Apple M4 / 16GB / MPS\n\n"
        f"{table}\n"
    )
    (report_dir / "ablation.md").write_text(md, encoding="utf-8")

    print("\n" + table + "\n")
    logger.info("报告写入 %s", report_dir / "ablation.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
