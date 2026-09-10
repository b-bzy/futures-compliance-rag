#!/usr/bin/env python
"""构建 QA 数据集。

分两层，两层的用途完全不同，绝不能混用：

  Tier 1 —— 金标（人工撰写，零生成）
      来源: SSE 50ETF经纪业务FAQ / SSE熔断机制问答 / CFFEX 常见问答
      用途: **评测**。93% 这类数字只能报在这个集合上。
      理由: 用同一个 LLM 出题、再用它自己判分，是循环论证，
            面试时一问就穿。评测集必须是人写的。

  Tier 2 —— 自动挖掘（对应简历"自动挖掘的结构化 QA 数据集"）
      2a 条款表模板  (品种,字段,值) -> QA，纯模板零 LLM，答案是字面单元格值
      2b 条款 LLM 挖掘  要求答案必须是源条款的连续子串，生成后硬校验
      2c 跨交易所对比  同一字段在不同交易所的取值差异，多跳测试
      用途: **训练**（reranker / SFT），不用于报告准确率。

用法:
    python scripts/04_mine_qa.py --tier1              # 只抽金标，秒级完成
    python scripts/04_mine_qa.py --tier2 --limit 500  # LLM 挖掘（慢）
    python scripts/04_mine_qa.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from derivrag.config import load_config  # noqa: E402
from derivrag.schema import QAPair, stable_id  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("qa")


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


# =====================================================================
# Tier 1 —— 人工撰写金标
# =====================================================================
def mine_tier1(parents: list[dict]) -> list[QAPair]:
    """从问答型父块直接抽取。这些问答是交易所工作人员写的，不经任何生成。"""
    pairs: list[QAPair] = []
    for p in parents:
        if p.get("chunk_type") != "qa_pair":
            continue
        text = p.get("text", "")
        m = re.match(r"问：(.+?)\n答：(.+)", text, re.S)
        if not m:
            continue
        question, answer = m.group(1).strip(), m.group(2).strip()
        if len(question) < 5 or len(answer) < 5:
            continue
        pairs.append(
            QAPair(
                qa_id=stable_id(p["parent_id"], "t1", prefix="qa1_"),
                question=question,
                answer=answer,
                tier=1,
                source_url=p.get("source_url", ""),
                doc_id=p.get("doc_id"),
                parent_id=p["parent_id"],
                clause_id=p.get("clause_id"),
                venue=p.get("venue", ""),
                lang=p.get("lang", "zh"),
                mining_method="human",
                span_verified=True,  # 答案就是原文本身
            )
        )
    logger.info("Tier-1 金标: %d 对", len(pairs))
    return pairs


# =====================================================================
# Tier 2a —— 条款表模板（零 LLM，零幻觉）
# =====================================================================

# 每个字段配 3 种自然提问句式。答案是表格单元格的字面值。
FIELD_TEMPLATES = [
    "{subject}的{field}是什么？",
    "{subject}的{field}是多少？",
    "请问{subject}的{field}如何规定？",
]

# 这些字段问"是多少"不自然，只用"是什么/如何规定"
_NON_NUMERIC = {"合约标的", "合约类型", "行权方式", "交割方式", "买卖类型", "合约到期月份"}


def mine_tier2a(parents: list[dict]) -> list[QAPair]:
    """条款表的每个 (品种, 字段, 值) 三元组模板化成 QA。"""
    pairs: list[QAPair] = []
    for p in parents:
        if p.get("chunk_type") != "table_row":
            continue
        field = (p.get("clause_id") or "").strip()
        text = p.get("text", "")
        if not field or "：" not in text:
            continue
        subject = (p.get("section_path") or [""])[0] or p.get("doc_title", "")
        subject = subject.strip()
        value = text.split("：", 1)[1].strip()
        if not subject or not value or len(value) > 400:
            continue

        templates = (
            [FIELD_TEMPLATES[0], FIELD_TEMPLATES[2]]
            if field in _NON_NUMERIC
            else FIELD_TEMPLATES
        )
        for t in templates:
            question = t.format(subject=subject, field=field)
            pairs.append(
                QAPair(
                    qa_id=stable_id(p["parent_id"], t, prefix="qa2a_"),
                    question=question,
                    answer=value,
                    tier=2,
                    source_url=p.get("source_url", ""),
                    doc_id=p.get("doc_id"),
                    parent_id=p["parent_id"],
                    clause_id=field,
                    venue=p.get("venue", ""),
                    lang=p.get("lang", "zh"),
                    mining_method="table_template",
                    span_verified=True,  # 答案逐字来自单元格
                )
            )
    logger.info("Tier-2a 条款表模板: %d 对", len(pairs))
    return pairs


# =====================================================================
# Tier 2c —— 跨交易所对比（多跳）
# =====================================================================
def mine_tier2c(parents: list[dict], *, max_pairs: int = 300) -> list[QAPair]:
    """同一字段在不同交易所/品种上的取值差异 —— 多跳检索测试用例。"""
    by_field: dict[str, list[dict]] = defaultdict(list)
    for p in parents:
        if p.get("chunk_type") != "table_row":
            continue
        field = (p.get("clause_id") or "").strip()
        if field:
            by_field[field].append(p)

    pairs: list[QAPair] = []
    for field, rows in by_field.items():
        # 只挑取值确实不同的字段，取值相同的对比题没有意义
        seen_subjects: dict[str, dict] = {}
        for r in rows:
            subject = (r.get("section_path") or [""])[0] or r.get("doc_title", "")
            if subject and subject not in seen_subjects:
                seen_subjects[subject] = r
        if len(seen_subjects) < 2:
            continue

        items = list(seen_subjects.items())[:2]
        (s1, r1), (s2, r2) = items
        v1 = r1["text"].split("：", 1)[-1].strip()
        v2 = r2["text"].split("：", 1)[-1].strip()
        if v1 == v2 or len(v1) > 200 or len(v2) > 200:
            continue

        pairs.append(
            QAPair(
                qa_id=stable_id(r1["parent_id"], r2["parent_id"], "cmp", prefix="qa2c_"),
                question=f"{s1}与{s2}的{field}有什么区别？",
                answer=f"{s1}的{field}为{v1}；{s2}的{field}为{v2}。",
                tier=2,
                source_url=r1.get("source_url", ""),
                doc_id=r1.get("doc_id"),
                parent_id=r1["parent_id"],
                clause_id=field,
                venue=f"{r1.get('venue','')}/{r2.get('venue','')}",
                lang="zh",
                mining_method="cross_venue",
                span_verified=False,  # 答案是两段原文的拼接，非单一连续子串
            )
        )
        if len(pairs) >= max_pairs:
            break
    logger.info("Tier-2c 跨交易所对比: %d 对", len(pairs))
    return pairs


# =====================================================================
# Tier 2b —— 条款 LLM 挖掘 + span-grounding 硬校验
# =====================================================================
def _normalize(s: str) -> str:
    """比对用的归一化：去空白，统一常见的全半角差异。"""
    s = re.sub(r"\s+", "", s)
    table = str.maketrans("（）［］｛｝：；，。％－＋　", "()[]{}:;,.%-+ ")
    return s.translate(table)


def mine_tier2b(
    parents: list[dict],
    provider,
    *,
    limit: int = 0,
    min_len: int = 60,
    max_len: int = 1500,
) -> tuple[list[QAPair], dict[str, int]]:
    """让 LLM 从条款里出题，再用 span-grounding 硬校验答案。

    校验规则：答案归一化后必须是条款原文归一化后的**连续子串**。
    不满足直接丢弃 —— 这一条就把幻觉挡在数据集之外。
    """
    from derivrag.llm.prompts import build_qa_mine_messages

    candidates = [
        p
        for p in parents
        if p.get("chunk_type") == "clause" and min_len <= len(p.get("text", "")) <= max_len
    ]
    if limit:
        candidates = candidates[:limit]
    logger.info("Tier-2b 候选条款 %d 条", len(candidates))

    pairs: list[QAPair] = []
    stats = Counter()
    t0 = time.time()

    for i, p in enumerate(candidates, 1):
        clause = p["text"]
        try:
            raw = provider.chat(
                build_qa_mine_messages(
                    clause, p.get("doc_title", ""), p.get("clause_id", "") or ""
                ),
                max_tokens=512,
                temperature=0.2,
            )
        except Exception as e:  # noqa: BLE001
            stats["llm_error"] += 1
            logger.debug("条款 %s 生成失败: %s", p["parent_id"], e)
            continue

        items = _parse_json_array(raw)
        if items is None:
            stats["bad_json"] += 1
            continue

        for item in items[:2]:
            q = (item.get("question") or "").strip()
            a = (item.get("answer") or "").strip()
            if not q or not a:
                stats["empty"] += 1
                continue
            # ---- span-grounding 硬校验 ----
            if _normalize(a) not in _normalize(clause):
                stats["span_rejected"] += 1
                continue
            if len(a) > 300:
                stats["too_long"] += 1
                continue
            stats["accepted"] += 1
            pairs.append(
                QAPair(
                    qa_id=stable_id(p["parent_id"], q[:40], prefix="qa2b_"),
                    question=q,
                    answer=a,
                    tier=2,
                    source_url=p.get("source_url", ""),
                    doc_id=p.get("doc_id"),
                    parent_id=p["parent_id"],
                    clause_id=p.get("clause_id"),
                    venue=p.get("venue", ""),
                    lang=p.get("lang", "zh"),
                    mining_method="clause_llm",
                    span_verified=True,
                )
            )

        if i % 25 == 0:
            rate = i / max(time.time() - t0, 1e-6)
            logger.info(
                "进度 %d/%d (%.2f 条/秒, 已接受 %d, 预计剩余 %.0f 分钟) %s",
                i, len(candidates), rate, stats["accepted"],
                (len(candidates) - i) / max(rate, 1e-6) / 60, dict(stats),
            )

    total_generated = stats["accepted"] + stats["span_rejected"] + stats["empty"] + stats["too_long"]
    if total_generated:
        logger.info(
            "Tier-2b: 接受 %d / 生成 %d，span 存活率 %.1f%%",
            stats["accepted"], total_generated, 100 * stats["accepted"] / total_generated,
        )
    return pairs, dict(stats)


def _parse_json_array(raw: str) -> list[dict] | None:
    """从模型输出里抠出 JSON 数组，容忍 ```json 围栏和前后废话。"""
    if not raw:
        return None
    raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) else None


# =====================================================================
# 评测集去泄漏 —— 问题改写
# =====================================================================
#
# ⚠️ 这一步不是可选项，是评测有效性的前提。
#
# Tier-1 金标来自问答型文档，其父块文本就是"问：X\n答：Y"，
# 也就是说**金标问题原文 100% 出现在被索引的子块里**。
# 直接拿这些问题去评测，检索退化成精确字符串匹配 ——
# 实测六个消融档位全部 Recall@1 = 1.000，任何组件的效果都测不出来。
#
# 解决办法：把问题改写成语义相同但用词不同的形式，再校验改写后的问题
# 不是语料的连续子串。这样评测的才是真实的语义检索能力。

PARAPHRASE_SYSTEM = """你负责把金融业务问题改写成语义完全相同、但用词和句式不同的问法。

要求：
1. 保持问题的**意图和指向完全不变**，答案必须还是原来那个答案。
2. 换一种问法：调整语序、改用同义表达、改变提问角度（如"是多少"→"如何规定"）。
3. **保留关键实体**（品种名、交易所名、具体条款名），否则问题会变得无法回答。
4. 只输出改写后的问题本身，不要解释、不要引号、不要编号。
5. 不要照抄原句的任何一个长片段。"""


def paraphrase_gold(pairs: list[QAPair], provider, corpus_text: str) -> list[QAPair]:
    """改写金标问题以消除评测泄漏。改写失败或仍然泄漏的条目会被丢弃。"""
    out: list[QAPair] = []
    stats = Counter()

    for i, p in enumerate(pairs, 1):
        try:
            rewritten = provider.chat(
                [
                    {"role": "system", "content": PARAPHRASE_SYSTEM},
                    {"role": "user", "content": f"原问题：{p.question}\n\n改写后："},
                ],
                max_tokens=200,
                temperature=0.7,
            )
        except Exception as e:  # noqa: BLE001
            stats["llm_error"] += 1
            logger.debug("改写失败: %s", e)
            continue

        rewritten = (rewritten or "").strip().strip('"“”\'')
        if "\n" in rewritten:
            parts = [x.strip() for x in rewritten.split("\n") if x.strip()]
            rewritten = parts[-1] if parts else ""

        if not rewritten or len(rewritten) < 6:
            stats["too_short"] += 1
            continue
        # 仍然是语料的连续子串 -> 泄漏没消除，丢弃
        if _normalize(rewritten) in corpus_text:
            stats["still_leaked"] += 1
            continue
        # 与原问题几乎没变 -> 等同于没改写
        if _normalize(rewritten) == _normalize(p.question):
            stats["unchanged"] += 1
            continue

        out.append(
            QAPair(
                qa_id=f"{p.qa_id}_pp",
                question=rewritten,
                answer=p.answer,
                tier=p.tier,
                source_url=p.source_url,
                doc_id=p.doc_id,
                parent_id=p.parent_id,
                clause_id=p.clause_id,
                venue=p.venue,
                lang=p.lang,
                mining_method=f"{p.mining_method}+paraphrase",
                span_verified=p.span_verified,
            )
        )
        stats["accepted"] += 1
        if i % 10 == 0:
            logger.info("改写进度 %d/%d，已接受 %d", i, len(pairs), stats["accepted"])

    logger.info("金标改写: %s", dict(stats))
    return out


# =====================================================================
# 去重
# =====================================================================
def dedupe(pairs: list[QAPair], embedder=None, threshold: float = 0.93) -> list[QAPair]:
    """先按问题文本精确去重，再（可选）按 embedding 余弦去近似重复。"""
    seen: set[str] = set()
    unique: list[QAPair] = []
    for p in pairs:
        key = _normalize(p.question)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    logger.info("精确去重: %d -> %d", len(pairs), len(unique))

    if embedder is None or len(unique) < 2:
        return unique

    import numpy as np

    vecs = embedder.encode_dense([p.question for p in unique])
    keep: list[int] = []
    kept_vecs: list[np.ndarray] = []
    for i, v in enumerate(vecs):
        if kept_vecs:
            sims = np.dot(np.stack(kept_vecs), v)
            if float(sims.max()) > threshold:
                continue
        keep.append(i)
        kept_vecs.append(v)
    logger.info("语义去重(cos>%.2f): %d -> %d", threshold, len(unique), len(keep))
    return [unique[i] for i in keep]


# =====================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="构建 QA 数据集")
    ap.add_argument("--tier1", action="store_true", help="只抽人工金标")
    ap.add_argument("--tier2", action="store_true", help="自动挖掘")
    ap.add_argument(
        "--paraphrase",
        action="store_true",
        help="改写金标问题以消除评测泄漏，产出 gold_eval.jsonl（评测必须用这个）",
    )
    ap.add_argument("--all", action="store_true")
    ap.add_argument(
        "--skip-llm-mining",
        action="store_true",
        help="Tier-2 只跑模板与跨交易所对比，跳过慢速的 LLM 条款挖掘",
    )
    ap.add_argument(
        "--hard-eval",
        type=int,
        default=0,
        help="从条款表 QA 中抽 N 条并改写，产出困难评测集 gold_hard.jsonl。"
        "这些问题要在十几个只差品种名的近似块里选对一个，比 FAQ 金标难得多。",
    )
    ap.add_argument("--limit", type=int, default=0, help="Tier-2b 处理的条款数上限")
    ap.add_argument("--no-dedupe-embed", action="store_true", help="跳过语义去重（快）")
    args = ap.parse_args()

    # 只有在完全没有指定任何子任务时才默认跑全部。
    # --hard-eval / --paraphrase 也算显式子任务，否则单独传它们会连带
    # 触发 Tier-2b 的 LLM 挖掘（4000+ 条条款，数小时）。
    if not (args.tier1 or args.tier2 or args.all or args.paraphrase or args.hard_eval):
        args.all = True

    cfg = load_config()
    processed = cfg.resolve("paths.processed")
    qa_dir = cfg.resolve("paths.qa")
    qa_dir.mkdir(parents=True, exist_ok=True)

    parents = load_jsonl(processed / "parents.jsonl")
    if not parents:
        logger.error("找不到父块，请先执行 scripts/02_parse.py")
        return 1
    logger.info("载入 %d 个父块", len(parents))

    report: dict = {}

    # ---------- Tier 1 ----------
    if args.tier1 or args.all:
        gold = mine_tier1(parents)
        gold = dedupe(gold)
        with open(qa_dir / "gold.jsonl", "w", encoding="utf-8") as f:
            for p in gold:
                f.write(p.to_json() + "\n")
        report["tier1_gold"] = len(gold)
        logger.info("金标写入 %s", qa_dir / "gold.jsonl")

        # ---------- 去泄漏改写 ----------
        if args.paraphrase or args.all:
            from derivrag.llm.provider import get_provider

            try:
                provider = get_provider(cfg)
            except Exception as e:  # noqa: BLE001
                logger.warning("LLM 不可用，跳过金标改写: %s", e)
            else:
                # 语料全文用于校验改写后的问题不再是连续子串
                corpus = _normalize(
                    "\n".join(
                        json.loads(line).get("text", "")
                        for line in open(
                            processed / "children.jsonl", encoding="utf-8"
                        )
                    )
                )
                logger.info("语料长度 %d 字符，开始改写 %d 条金标问题…", len(corpus), len(gold))
                pp = paraphrase_gold(gold, provider, corpus)
                with open(qa_dir / "gold_eval.jsonl", "w", encoding="utf-8") as f:
                    for p in pp:
                        f.write(p.to_json() + "\n")
                report["tier1_gold_eval"] = len(pp)
                logger.info("去泄漏评测集写入 %s（%d 条）", qa_dir / "gold_eval.jsonl", len(pp))

    # ---------- Tier 2 ----------
    if args.tier2 or args.all:
        synthetic: list[QAPair] = []
        synthetic.extend(mine_tier2a(parents))
        synthetic.extend(mine_tier2c(parents))

        if args.skip_llm_mining:
            logger.info("已跳过 Tier-2b LLM 条款挖掘（--skip-llm-mining）")
        else:
            from derivrag.llm.provider import get_provider

            try:
                provider = get_provider(cfg)
                llm_pairs, mine_stats = mine_tier2b(parents, provider, limit=args.limit)
                synthetic.extend(llm_pairs)
                report["tier2b_stats"] = mine_stats
            except Exception as e:  # noqa: BLE001
                logger.warning("Tier-2b 跳过（LLM 不可用）: %s", e)

        embedder = None
        if not args.no_dedupe_embed:
            from derivrag.index.embed import get_embedder

            embedder = get_embedder(cfg)
        synthetic = dedupe(synthetic, embedder)

        with open(qa_dir / "synthetic.jsonl", "w", encoding="utf-8") as f:
            for p in synthetic:
                f.write(p.to_json() + "\n")
        report["tier2_synthetic"] = len(synthetic)
        report["tier2_by_method"] = dict(Counter(p.mining_method for p in synthetic))
        logger.info("合成集写入 %s", qa_dir / "synthetic.jsonl")

    # ---------- 困难评测集 ----------
    if args.hard_eval:
        import random as _random

        synth_path = qa_dir / "synthetic.jsonl"
        if not synth_path.exists():
            logger.error("需要先跑 --tier2 产出 synthetic.jsonl")
        else:
            # 只取中文合约条款表。SGX/HKEX 的表格大多是英文术语定义表
            # （"'Relevant Period' shall have the meaning ascribed…"），
            # 那类题考的是术语查找，不是本项目要突出的"同字段跨品种辨析"。
            CN_VENUES = {"SSE", "SZSE", "CFFEX", "SHFE", "DCE", "CZCE"}
            rows = [
                QAPair(**r)
                for r in (json.loads(l) for l in open(synth_path, encoding="utf-8"))
                if r.get("mining_method") == "table_template"
                and r.get("parent_id")
                and r.get("lang") == "zh"
                and r.get("venue") in CN_VENUES
            ]
            # 同一个父块只取一条，避免同一事实的三种句式重复进评测集
            seen_p: set[str] = set()
            uniq = []
            for r in rows:
                if r.parent_id in seen_p:
                    continue
                seen_p.add(r.parent_id)
                uniq.append(r)
            _random.Random(42).shuffle(uniq)
            sample = uniq[: args.hard_eval]
            logger.info("困难评测集抽样 %d 条（来自 %d 个不同条款）", len(sample), len(uniq))

            from derivrag.llm.provider import get_provider

            try:
                provider = get_provider(cfg)
            except Exception as e:  # noqa: BLE001
                logger.warning("LLM 不可用，跳过困难评测集: %s", e)
            else:
                corpus = _normalize(
                    "\n".join(
                        json.loads(line).get("text", "")
                        for line in open(processed / "children.jsonl", encoding="utf-8")
                    )
                )
                hard = paraphrase_gold(sample, provider, corpus)
                with open(qa_dir / "gold_hard.jsonl", "w", encoding="utf-8") as f:
                    for p in hard:
                        f.write(p.to_json() + "\n")
                report["hard_eval"] = len(hard)
                logger.info("困难评测集写入 %s（%d 条）", qa_dir / "gold_hard.jsonl", len(hard))

    (qa_dir / "mining_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("挖掘报告: %s", json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
