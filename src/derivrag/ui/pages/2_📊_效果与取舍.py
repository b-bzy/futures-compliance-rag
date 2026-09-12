"""效果与取舍 —— 这个系统到底行不行，以及我为什么这么做。

绝大多数 RAG 项目只有一个聊天框，看完不知道它准不准、也不知道作者
在哪些地方做过取舍。这一页把 eval/reports 下的消融数据和 docs/decisions.md
里的决策记录直接搬到界面上，让"效果"和"判断力"跟"能跑"一样可见。

数据全部读自仓库内已产出的报告文件，页面本身不跑任何评测 —— 数字改不了，
这也是它可信的前提。报告由 scripts/06_eval.py 与 scripts/07_ablation.py 产出。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from derivrag.config import repo_root  # noqa: E402

st.set_page_config(page_title="效果与取舍", page_icon="📊", layout="wide")

ROOT = repo_root()
REPORTS = ROOT / "eval" / "reports"

# 档位在报告 JSON 里的键 -> 展示名。顺序即消融的递进顺序。
STAGES = {
    "dense_only": "仅稠密向量（代理基线）",
    "bm25_only": "仅 BM25",
    "bm25_dense_rrf": "BM25+稠密，标准 RRF",
    "three_score_rrf": "三路召回 + 自研 score-RRF",
    "plus_rerank": "+ 交叉编码器重排",
    "plus_hyde": "+ HyDE",
}


@st.cache_data
def load_report(name: str) -> dict | None:
    p = REPORTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data
def load_decisions() -> list[dict[str, str]]:
    """把 docs/decisions.md 按 `## D<n>. <标题>` 切成卡片。"""
    p = ROOT / "docs" / "decisions.md"
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8")
    parts = re.split(r"^## (D\d+\.\s*.+)$", text, flags=re.MULTILINE)
    # split 结果形如 [前言, 标题1, 正文1, 标题2, 正文2, ...]
    return [
        {"title": parts[i].strip(), "body": parts[i + 1].strip().strip("-").strip()}
        for i in range(1, len(parts) - 1, 2)
    ]


def ablation_frame(report: dict) -> pd.DataFrame:
    rows = []
    for key, label in STAGES.items():
        r = report.get("results", {}).get(key)
        if not r:
            continue
        rows.append(
            {
                "档位": label,
                "Recall@1": r["recall"]["@1"],
                "Recall@5": r["recall"]["@5"],
                "Recall@10": r["recall"]["@10"],
                "MRR": r["mrr"],
                "nDCG@5": r["ndcg"]["@5"],
                "平均延迟(s)": r["mean_latency_s"],
                "P95(s)": r["p95_latency_s"],
            }
        )
    return pd.DataFrame(rows)


# =====================================================================
st.title("📊 效果与取舍")
st.caption(
    "全部数字读自 eval/reports 下已产出的报告，可用 scripts/06_eval.py 与 "
    "scripts/07_ablation.py 重跑复现。硬件：Apple M4 / 16GB / MPS。"
)

tab_ablation, tab_method, tab_decisions = st.tabs(
    ["检索消融", "评测集怎么来的", "技术决策记录"]
)

# ---------------------------------------------------------------------
with tab_ablation:
    hard = load_report("ablation_hard.json")
    if not hard:
        st.warning("未找到 eval/reports/ablation_hard.json，请先运行 scripts/07_ablation.py。")
    else:
        df = ablation_frame(hard)
        st.subheader(f"困难集消融（{hard.get('n_gold', '—')} 条，主表）")
        st.caption(
            "命中判定：返回候选的 parent_id 是否等于金标答案所在条款。"
            "困难集的问题要在十几个只差品种名的近似条款块里选对一个。"
        )

        base = df.loc[df["档位"] == STAGES["dense_only"], "Recall@1"]
        best = df.loc[df["档位"] == STAGES["plus_rerank"], "Recall@1"]
        std_rrf = df.loc[df["档位"] == STAGES["bm25_dense_rrf"], "Recall@1"]
        score_rrf = df.loc[df["档位"] == STAGES["three_score_rrf"], "Recall@1"]

        c1, c2, c3 = st.columns(3)
        if not base.empty and not best.empty:
            c1.metric(
                "端到端 Recall@1",
                f"{best.iloc[0]:.3f}",
                f"{(best.iloc[0] - base.iloc[0]) * 100:+.1f} pp vs 纯稠密基线",
            )
        if not std_rrf.empty and not score_rrf.empty:
            c2.metric(
                "自研 score-RRF",
                f"{score_rrf.iloc[0]:.3f}",
                f"{(score_rrf.iloc[0] - std_rrf.iloc[0]) * 100:+.1f} pp vs 标准 RRF",
            )
            c3.metric(
                "标准 RRF",
                f"{std_rrf.iloc[0]:.3f}",
                f"{(std_rrf.iloc[0] - base.iloc[0]) * 100:+.1f} pp vs 纯稠密基线",
                delta_color="inverse",
            )

        st.bar_chart(df.set_index("档位")[["Recall@1", "Recall@5", "MRR"]], height=340)
        st.dataframe(df, hide_index=True, use_container_width=True)

        st.markdown(
            """
**这张表里最值得说的一件事：混合检索并不天然更好。**

标准 RRF 只看名次不看分数。BM25 在这个任务上很弱（Recall@1 仅 0.407），
rank-only 融合会让它的弱结果把稠密向量的强结果挤下去 ——
**混合后（0.620）反而比纯稠密（0.727）差 10.7 pp**。
保留归一化原始分数后恢复到 0.740，比标准 RRF 高 **12.0 pp**。

这是本项目相对直接调用 langchain `EnsembleRetriever` 的实质改进，
也说明"混合检索一定比单路好"是个需要验证的假设，不是定理。
"""
        )

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            faq = load_report("ablation_faq.json")
            if faq:
                st.subheader(f"FAQ 档（{faq.get('n_gold', '—')} 条）")
                st.dataframe(ablation_frame(faq), hide_index=True, use_container_width=True)
                st.caption(
                    "改写去泄漏后，dense-only 基线仍有 Recall@1 = 0.986，"
                    "天花板效应明显 —— 这正是必须再建困难集的原因。"
                )
        with col_b:
            hyde = load_report("ablation_hyde50.json")
            if hyde:
                st.subheader(f"HyDE 增益（{hyde.get('n_gold', '—')} 条）")
                st.dataframe(ablation_frame(hyde), hide_index=True, use_container_width=True)
                st.caption(
                    "HyDE 在本语料上收益很小（Recall@1 持平），但要多调一次 LLM。"
                    "保留它是因为口语化提问场景下仍有用，默认可关。"
                )

# ---------------------------------------------------------------------
with tab_method:
    st.subheader("评测集的来源，决定了上面所有数字的可信度")

    ev = load_report("eval.json")
    st.markdown(
        """
| 数据集 | 规模 | 来源 | 用途 |
|---|---|---|---|
| `gold.jsonl` | 72 | 上交所 50ETF FAQ(31) + 熔断机制问答(8) + 中金所常见问答(33)，**交易所工作人员撰写** | 人工核对 |
| `gold_eval.jsonl` | 72 | 上面的问题经 LLM **改写去泄漏** | FAQ 档评测 |
| `gold_hard.jsonl` | 150 | 中文合约条款表 QA，同样改写去泄漏 | **主评测集** |
| `synthetic.jsonl` | 1747 | 条款表模板(1739) + 跨交易所对比(8) | **仅训练，不评测** |
"""
    )

    st.error(
        "**第一版消融表六个档位全是 Recall@1 = 1.000。**\n\n"
        "这不是效果好，是评测集泄漏：金标来自问答型文档，父块文本就是 "
        "`问：X\\n答：Y`，金标问题原文 100% 出现在被索引的子块里（实测 72/72），"
        "检索退化成了精确字符串匹配。改写去泄漏后数字才有意义。"
    )
    st.caption(
        "把这件事写进报告而不是把 1.000 拿去邀功，是这个项目里我最想被问到的一个决策。"
        "详见 docs/decisions.md 的 D14、D15。"
    )

    if ev:
        st.divider()
        st.subheader("生成质量与多轮改写")
        gen = ev.get("generation", {})
        mt = ev.get("multiturn", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("关键词覆盖率均值", f"{gen.get('keyword_match_mean', 0):.3f}", help=f"n={gen.get('n')}")
        c2.metric("覆盖率 ≥0.6 占比", f"{gen.get('keyword_match_at_0.6', 0):.0%}")
        c3.metric("多轮改写成功率", f"{mt.get('rewrite_success_rate', 0):.0%}", help=f"n={mt.get('n')}")
        st.caption(
            # 标签写"评测集"而不是"金标" —— eval.gold_file 可能指向模板生成的
            # 困难集，叫它金标是抬举。来源说明由 describe_gold() 按数据推导。
            f"评测集：{ev.get('gold_file', '—')}，{ev.get('gold_size', '—')} 条"
            f"（{ev.get('gold_source', '—')}）。"
            f"生成平均延迟 {gen.get('mean_latency_s', 0):.1f}s —— 该数字记录于本地 "
            "qwen3 时期，现已切换到 DeepSeek API，延迟大幅下降。"
        )

        if cases := mt.get("cases"):
            st.markdown("**多轮改写实例**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "追问": c["follow_up"],
                            "改写为": c["rewritten"],
                            "命中预期": "✅" if c.get("rewrite_contains_expected") else "❌",
                            "结果有变化": "是" if c.get("results_differ") else "否",
                        }
                        for c in cases
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )

# ---------------------------------------------------------------------
with tab_decisions:
    decisions = load_decisions()
    if not decisions:
        st.warning("未找到 docs/decisions.md。")
    else:
        st.subheader(f"技术决策记录（{len(decisions)} 条）")
        st.caption(
            "每条记录：当初的方案 → 实测发现了什么 → 最终怎么做。"
            "面试时被问「为什么不用 X」，答案不是「没想到」，"
            "而是「试过，因为具体的 Y 原因不可行，换成了 Z」。"
        )
        kw = st.text_input("搜索决策", placeholder="OCR / RRF / DeepSpeed / 泄漏 …").strip()
        shown = [
            d
            for d in decisions
            if not kw or kw.lower() in (d["title"] + d["body"]).lower()
        ]
        if kw and not shown:
            st.info(f"没有匹配「{kw}」的决策记录。")
        for d in shown:
            with st.expander(d["title"], expanded=bool(kw) and len(shown) <= 3):
                st.markdown(d["body"])
