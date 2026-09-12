"""评测集的读取与**如实描述**。

`eval.gold_file` 可以指向人工金标 `gold.jsonl`，也可以指向模板生成 +
LLM 改写的困难集 `gold_hard.jsonl`。所以报告里那句"评测集是什么"
必须**从数据本身推导**，不能在脚本里写死 —— 写死的那句话在换了
评测集之后就是假陈述，而且恰恰出现在最需要可信的位置。

这个模块被 06_eval.py 与 07_ablation.py 共用，保证两份报告口径一致。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def load_gold(path: Path, limit: int = 0, *, require_parent_id: bool = True) -> list[dict]:
    """读评测集。默认丢掉没有 parent_id 的条目 —— 检索指标算不了。"""
    items: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if require_parent_id and not rec.get("parent_id"):
                continue
            items.append(rec)
    return items[:limit] if limit else items


def describe_gold(gold: list[dict]) -> str:
    """按数据实际构成生成一句来源说明，用于报告抬头与 eval.json。

    刻意不接受任何"来源"入参：能被调用方覆盖的描述，迟早会和数据对不上。
    """
    methods = Counter(g.get("mining_method", "unknown") for g in gold)
    human = sum(v for k, v in methods.items() if k.startswith("human"))
    template = sum(v for k, v in methods.items() if k.startswith("table_template"))
    paraphrased = sum(1 for g in gold if "paraphrase" in (g.get("mining_method") or ""))

    parts = []
    if human:
        parts.append(f"交易所官方问答 {human} 条")
    if template:
        parts.append(f"合约条款表模板 {template} 条")
    other = len(gold) - human - template
    if other > 0:
        parts.append(f"其他 {other} 条")
    desc = "、".join(parts) if parts else "来源未标注"

    if paraphrased:
        desc += f"；其中 {paraphrased} 条经 LLM 改写去泄漏（问题不再是语料的连续子串）"

    # tier-2 是自动构造的。报任何指标时都必须让读者看见这一点，
    # 否则"93% 准确率"会被默认理解成人工金标上的成绩。
    n_t2 = sum(1 for g in gold if g.get("tier") == 2)
    if n_t2:
        desc += (
            f"；含 {n_t2} 条 tier-2 自动构造样本"
            "（答案逐字取自表格单元格，问题由模板生成，非 LLM 撰写答案）"
        )
    return desc
