#!/usr/bin/env python
"""端到端评测：检索指标 + RAGAS 生成质量 + 多轮改写效果。

⚠️ 关于"准确率"这个数字的诚实性:

    铁律是**答案不能由 LLM 撰写** —— 用同一个模型出题再自己判分是
    循环论证，任何有经验的面试官都会问"你的评测集哪来的"。
    所以 Tier-2b（LLM 从条款里挖的 QA）只用于训练，绝不用于报告指标。

    可以评测的有两类，都满足"答案非 LLM 撰写"：
      gold.jsonl / gold_eval.jsonl  交易所工作人员撰写的问答（问题经改写去泄漏）
      gold_hard.jsonl               合约条款表模板题，答案逐字取自表格单元格

    具体用哪个由 `eval.gold_file` 决定，报告里的来源说明由
    `derivrag.eval.dataset.describe_gold()` 按数据实际构成生成 ——
    **不要在这里写死一句"人工撰写"**，换了评测集它就是假陈述。

    RAGAS 的 judge 默认指向本地 ollama（OpenAI 兼容端点），可离线运行。
    judge 与被评测的生成模型是同一个时，faithfulness 会略偏乐观，
    报告里会标注这一点。若配了 DEEPSEEK_API_KEY，建议用它当 judge
    以获得独立评判。

用法:
    python scripts/06_eval.py --retrieval          # 只跑检索指标（快，不调 LLM）
    python scripts/06_eval.py --generation --limit 30
    python scripts/06_eval.py --multiturn          # 多轮改写效果
    python scripts/06_eval.py --all
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
from derivrag.eval.metrics import (  # noqa: E402
    compute_metrics,
    keyword_match_score,
    numeric_fidelity,
    rank_of,
)
from derivrag.pipeline import RAGPipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval")


# 多轮指代消解测试用例。第二问都省略了主体或谓语，
# 不做查询重写必然检索失败 —— 这正是要测的。
MULTITURN_CASES = [
    {
        "turns": ["上证50ETF期权的合约单位是多少？", "那深证100ETF期权呢？"],
        "expect_in_rewrite": ["深证100", "合约单位"],
    },
    {
        "turns": ["50ETF期权的到期日是如何规定的？", "行权日呢？"],
        "expect_in_rewrite": ["行权日"],
    },
    {
        "turns": ["沪深300股指期权的合约乘数是多少？", "它的最小变动价位是多少？"],
        "expect_in_rewrite": ["沪深300"],
    },
]


# =====================================================================
def eval_retrieval(pipeline: RAGPipeline, gold: list[dict], ks: list[int]) -> dict:
    """检索指标：Recall@k / MRR / nDCG。不调用生成模型。"""
    ranks: list[int | None] = []
    latencies: list[float] = []
    max_k = max(ks)

    for i, item in enumerate(gold, 1):
        if not item.get("parent_id"):
            continue
        t = time.perf_counter()
        result = pipeline.answer(item["question"], top_k=max_k, generate=False)
        latencies.append(time.perf_counter() - t)
        retrieved = [h.metadata.get("parent_id", "") for h in result.hits]
        ranks.append(rank_of(retrieved, item["parent_id"]))
        if i % 20 == 0:
            logger.info("检索评测 %d/%d", i, len(gold))

    return compute_metrics(ranks, latencies, ks).to_dict()


# =====================================================================
def eval_generation(pipeline: RAGPipeline, gold: list[dict], cfg) -> dict:
    """生成质量：关键词覆盖率（确定性）+ RAGAS（LLM 评判）。"""
    records = []
    for i, item in enumerate(gold, 1):
        t = time.perf_counter()
        result = pipeline.answer(item["question"], top_k=5)
        records.append(
            {
                "question": item["question"],
                "answer": result.answer,
                "ground_truth": item["answer"],
                "contexts": [h.parent_text or h.text for h in result.hits],
                "keyword_score": keyword_match_score(result.answer, item["answer"]),
                "numeric_fidelity": numeric_fidelity(result.answer, item["answer"]),
                "latency": time.perf_counter() - t,
            }
        )
        if i % 5 == 0:
            logger.info("生成评测 %d/%d", i, len(gold))

    kw_scores = [r["keyword_score"] for r in records]
    num_scores = [r["numeric_fidelity"] for r in records if r["numeric_fidelity"] is not None]
    out: dict = {
        "n": len(records),
        # 数值保真度是条款问答里最该看的指标：合约单位、保证金比例这类
        # 数字错一位就是灾难，而语义相似度对数值错误并不敏感。
        "numeric_fidelity_n": len(num_scores),
        "numeric_fidelity_mean": round(sum(num_scores) / max(len(num_scores), 1), 4)
        if num_scores
        else None,
        "numeric_fidelity_all_correct": round(
            sum(1 for s in num_scores if s == 1.0) / max(len(num_scores), 1), 4
        )
        if num_scores
        else None,
        # 关键词覆盖率作为参考。⚠️ 它会惩罚"答得比金标更全"的回答，
        # 实测有样本给出 4 条带引用的要点、比金标更完整，覆盖率却只有 0.14。
        "keyword_match_mean": round(sum(kw_scores) / max(len(kw_scores), 1), 4),
        "keyword_match_at_0.6": round(
            sum(1 for s in kw_scores if s >= 0.6) / max(len(kw_scores), 1), 4
        ),
        "mean_latency_s": round(sum(r["latency"] for r in records) / max(len(records), 1), 2),
    }

    ragas_scores = run_ragas(records, cfg)
    if ragas_scores:
        out["ragas"] = ragas_scores

    out["_records"] = records
    return out


def run_ragas(records: list[dict], cfg) -> dict | None:
    """跑 RAGAS。judge 走 OpenAI 兼容端点（本地 ollama 或云端 DeepSeek）。

    指标、并发数、超时全部来自 configs/config.yaml 的 eval 段：
        eval.ragas_metrics / eval.ragas_max_workers / eval.ragas_timeout_seconds
    """
    try:
        # ragas 0.4.3 在 langchain-community 0.4.x 上导入即失败，先打垫片
        from derivrag.compat import patch_ragas_vertexai

        patch_ragas_vertexai()

        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas import metrics as ragas_metrics
        from ragas.run_config import RunConfig
    except Exception as e:  # noqa: BLE001
        logger.warning("RAGAS 不可用，跳过（数值保真度指标仍然有效）: %s", e)
        return None

    wanted = cfg.get_path(
        "eval.ragas_metrics", ["faithfulness", "answer_correctness", "context_precision"]
    )
    metric_objs = []
    for name in wanted:
        obj = getattr(ragas_metrics, name, None)
        if obj is None:
            logger.warning("ragas 里没有指标 %r，跳过", name)
            continue
        metric_objs.append(obj)
    if not metric_objs:
        logger.warning("eval.ragas_metrics 里没有一个有效指标")
        return None

    timeout = cfg.get_path("eval.ragas_timeout_seconds", 900)
    max_workers = cfg.get_path("eval.ragas_max_workers", 1)

    judge_name = cfg.get_path("eval.judge_provider", "ollama")
    spec = cfg.get_path(f"llm.providers.{judge_name}", {})
    if not spec:
        logger.warning("找不到 judge provider %s 的配置", judge_name)
        return None

    try:
        llm = ChatOpenAI(
            model=spec.get("model"),
            base_url=spec.get("base_url"),
            api_key=spec.get("api_key") or "ollama",
            temperature=0.0,
            timeout=300,
        )
        # ⚠️ 必须显式传 embeddings。answer_correctness 需要向量相似度，
        # 而 ragas 在 embeddings=None 时会**默认回退到 OpenAI 的 embedding
        # 服务**，于是即便 judge 指向本地 ollama，整个评测仍然会因为
        # "Missing credentials / OPENAI_API_KEY" 直接失败。
        # 这里把本地 bge-m3 包装成 langchain 的 Embeddings 接口喂进去，
        # 整条评测链路即可完全离线。
        from derivrag.index.embed import LangChainEmbeddings, get_embedder

        embeddings = LangChainEmbeddings(get_embedder(cfg))
        dataset = Dataset.from_list(
            [
                {
                    "question": r["question"],
                    "answer": r["answer"],
                    "contexts": r["contexts"],
                    "ground_truth": r["ground_truth"],
                }
                for r in records
            ]
        )
        # ⚠️ 本地 judge 必须把并发降到 1 并放宽超时。
        # ragas 默认并行发起十几个 job、单 job 超时 180s，而本地 ollama
        # 同一时刻只能服务一个请求，结果是所有 job 一起排队、一起超时，
        # 三个指标全部返回 nan（实测 12 个 job 有 11 个 TimeoutError）。
        run_config = RunConfig(timeout=timeout, max_workers=max_workers, max_retries=2)
        logger.info(
            "RAGAS: judge=%s 指标=%s 并发=%d",
            judge_name, [m.name for m in metric_objs], max_workers,
        )
        result = evaluate(
            dataset,
            metrics=metric_objs,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
            run_config=run_config,
        )
        scores = {k: round(float(v), 4) for k, v in result._repr_dict.items()}
        scores["_judge"] = f"{judge_name}:{spec.get('model')}"
        scores["_note"] = (
            "judge 与生成模型相同，faithfulness 可能偏乐观；"
            "如需独立评判请配置 DEEPSEEK_API_KEY 并把 eval.judge_provider 改为 deepseek"
            if judge_name == cfg.get_path("llm.provider")
            else "judge 与生成模型不同，评判相对独立"
        )
        return scores
    except Exception as e:  # noqa: BLE001
        logger.warning("RAGAS 执行失败: %s", e)
        return None


# =====================================================================
def eval_multiturn(pipeline: RAGPipeline) -> dict:
    """多轮查询重写效果：开/关重写各跑一遍，比较改写命中率与检索结果差异。"""
    results = []
    for case in MULTITURN_CASES:
        turns = case["turns"]
        history: list[dict] = []
        # 先走完前置轮次，建立上下文
        for t in turns[:-1]:
            r = pipeline.answer(t, history=history, use_hyde=False)
            history.append({"role": "user", "content": t})
            history.append({"role": "assistant", "content": r.answer})

        follow_up = turns[-1]
        with_rw = pipeline.answer(
            follow_up, history=history, use_rewrite=True, use_hyde=False, generate=False
        )
        without_rw = pipeline.answer(
            follow_up, history=history, use_rewrite=False, use_hyde=False, generate=False
        )

        rewritten = with_rw.rewritten_query or follow_up
        expect = case["expect_in_rewrite"]
        results.append(
            {
                "follow_up": follow_up,
                "rewritten": rewritten,
                "rewrite_contains_expected": all(e in rewritten for e in expect),
                "top1_with_rewrite": (
                    with_rw.hits[0].metadata.get("doc_title", "") if with_rw.hits else None
                ),
                "top1_without_rewrite": (
                    without_rw.hits[0].metadata.get("doc_title", "")
                    if without_rw.hits
                    else None
                ),
                "results_differ": [h.child_id for h in with_rw.hits]
                != [h.child_id for h in without_rw.hits],
            }
        )
        logger.info("多轮: %r -> %r", follow_up, rewritten)

    ok = sum(1 for r in results if r["rewrite_contains_expected"])
    return {
        "n": len(results),
        "rewrite_success_rate": round(ok / max(len(results), 1), 3),
        "cases": results,
    }


# =====================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="端到端评测")
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--generation", action="store_true")
    ap.add_argument("--multiturn", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not (args.retrieval or args.generation or args.multiturn or args.all):
        args.all = True

    cfg = load_config()
    gold_path = cfg.resolve("eval.gold_file")
    if not gold_path.exists():
        logger.error(
            "评测集不存在: %s。gold.jsonl 由 scripts/04_mine_qa.py --tier1 产出，"
            "gold_hard.jsonl 由 --hard-eval N 产出",
            gold_path,
        )
        return 1

    gold = load_gold(gold_path, args.limit)
    if not gold:
        logger.error("评测集为空或缺少 parent_id 字段: %s", gold_path)
        return 1
    logger.info("评测集 %s：%d 条 —— %s", gold_path.name, len(gold), describe_gold(gold))

    pipeline = RAGPipeline(cfg, lazy=True)
    report: dict = {
        "gold_file": gold_path.name,
        "gold_size": len(gold),
        # 来源说明由数据推导，不写死 —— 见模块 docstring
        "gold_source": describe_gold(gold),
    }

    if args.retrieval or args.all:
        logger.info("=== 检索指标 ===")
        report["retrieval"] = eval_retrieval(
            pipeline, gold, cfg.get_path("eval.retrieval_k", [1, 3, 5, 10])
        )
        logger.info("%s", json.dumps(report["retrieval"], ensure_ascii=False))

    if args.generation or args.all:
        logger.info("=== 生成质量 ===")
        gen = eval_generation(pipeline, gold, cfg)
        records = gen.pop("_records", [])
        report["generation"] = gen
        logger.info("%s", json.dumps(gen, ensure_ascii=False))
        report_dir = cfg.resolve("eval.report_dir")
        report_dir.mkdir(parents=True, exist_ok=True)
        with open(report_dir / "generation_records.jsonl", "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.multiturn or args.all:
        logger.info("=== 多轮查询重写 ===")
        report["multiturn"] = eval_multiturn(pipeline)
        logger.info("重写成功率 %.0f%%", 100 * report["multiturn"]["rewrite_success_rate"])

    report_dir = cfg.resolve("eval.report_dir")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("报告写入 %s", report_dir / "eval.json")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
