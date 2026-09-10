"""检索过程剖析 —— 把"混合检索 + 多阶段重排"变成看得见的名次流转。

这个页面只调 /search，不调 LLM，所以是秒回的。之所以要跟问答页分开，
是因为演示检索链路时不该被十几秒的生成延迟打断 —— 提问、看名次怎么变、
改配置再看一次，这个循环要足够快才讲得动。

页面回答三个问题：
    1. 三路召回各自捞到了什么？谁独占、谁重合？
    2. RRF 融合后名次怎么排？
    3. 重排把谁提上来了、把谁压下去了？
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from derivrag.ui.api_client import APIError  # noqa: E402
from derivrag.ui.components import (  # noqa: E402
    CHANNEL_NAMES,
    STATUS_COLORS,
    get_client,
    render_backend_status,
    render_clause_text,
    render_confidence,
    render_timings,
)

st.set_page_config(page_title="检索过程剖析", page_icon="🔍", layout="wide")


def _label(hit: dict) -> str:
    """条款的短标识：优先用条号，没有就截标题。"""
    bits = [b for b in (hit.get("doc_title"), hit.get("clause_id")) if b]
    text = " · ".join(bits) if bits else hit.get("child_id", "")
    return text if len(text) <= 48 else text[:47] + "…"


def _delta(before: int | None, after: int) -> str:
    """融合后名次 → 重排后名次的变化。正数表示被提上来了。"""
    if before is None:
        return "新进"
    diff = before - after
    if diff > 0:
        return f"▲{diff}"
    if diff < 0:
        return f"▼{-diff}"
    return "－"


st.title("🔍 检索过程剖析")
st.caption("一个查询在三路召回、RRF 融合、交叉编码器重排之间的名次流转。不调用 LLM，秒回。")

health = render_backend_status()

with st.sidebar:
    st.divider()
    st.header("召回通路")
    ch_bm25 = st.checkbox("BM25（分词精确匹配）", value=True)
    ch_dense = st.checkbox("稠密向量（语义泛化）", value=True)
    ch_keyword = st.checkbox("关键词（条号/文号定位）", value=True)

    st.divider()
    use_rerank = st.toggle("交叉编码器重排", value=True)
    use_hyde = st.toggle("HyDE 假设文档", value=False, help="会调用一次 LLM，慢一些")
    top_k = st.slider("返回条款数", 1, 20, 8)
    venue = st.text_input("限定交易所（可空）", placeholder="SSE / CFFEX / DCE …").strip()

channels = tuple(
    c for c, on in (("bm25", ch_bm25), ("dense", ch_dense), ("keyword", ch_keyword)) if on
)

EXAMPLES = [
    "50ETF期权的合约单位是多少？",
    "沪深300股指期权的最小变动价位",
    "熔断机制触发后集合竞价持续多长时间",
    "豆粕期权的行权方式是什么",
]

st.write("**示例问题**")
cols = st.columns(len(EXAMPLES))
for col, ex in zip(cols, EXAMPLES):
    if col.button(ex, use_container_width=True):
        st.session_state["probe_query"] = ex

query = st.text_input(
    "检索问题",
    value=st.session_state.get("probe_query", ""),
    placeholder="输入一个问题，看它在检索链路里的名次怎么变化",
)

if not query:
    st.info("输入问题或点上面的示例开始。建议先用默认配置跑一次，再关掉某一路召回对比差异。")
    st.stop()

if not channels:
    st.error("至少要启用一条召回通路。")
    st.stop()

if health is None:
    st.stop()

with st.spinner("检索中…"):
    try:
        res = get_client().search(
            query,
            top_k=top_k,
            use_rerank=use_rerank,
            use_hyde=use_hyde,
            channels=channels,
            venue=venue or None,
        )
    except APIError as e:
        st.error(str(e))
        st.stop()

hits = res.get("hits", [])
if not hits:
    st.warning("没有检索到条款。试试放宽交易所限制，或换一种问法。")
    st.stop()

render_confidence(res.get("confidence"))

# ---------- 1. 各路召回规模 ----------
st.subheader("① 各路召回命中数")
counts = res.get("channel_counts", {})
if counts:
    cols = st.columns(len(counts))
    for col, (ch, n) in zip(cols, counts.items()):
        col.metric(CHANNEL_NAMES.get(ch, ch), f"{n} 条")
    st.caption("统计的是最终返回条款中，各路召回曾经命中过的条数（同一条可被多路命中）。")

# ---------- 2. 名次流转表 ----------
st.subheader("② 名次流转：谁被提上来了，谁被压下去了")

rows = []
for h in hits:
    ranks = h.get("component_ranks") or {}
    scores = h.get("component_scores") or {}
    final_rank = h["index"]
    # 不走重排时融合名次就是最终名次；走重排时由 rerank 记下候选的原始融合名次
    fused_rank = ranks.get("fused", final_rank if not use_rerank else None)
    rows.append(
        {
            "最终": final_rank,
            "条款": _label(h),
            "状态": STATUS_COLORS.get(h.get("effective_status", "未知"), "⚪"),
            "BM25": ranks.get("bm25"),
            "稠密": ranks.get("dense"),
            "关键词": ranks.get("keyword"),
            "融合后": fused_rank,
            "重排后": final_rank if use_rerank else None,
            "名次变化": _delta(fused_rank, final_rank) if use_rerank else "—",
            "重排分": scores.get("rerank"),
            "融合分": scores.get("fused"),
        }
    )

df = pd.DataFrame(rows)
st.dataframe(
    df,
    hide_index=True,
    use_container_width=True,
    column_config={
        "最终": st.column_config.NumberColumn(width="small"),
        "条款": st.column_config.TextColumn(width="large"),
        "状态": st.column_config.TextColumn(width="small"),
    },
)
st.caption(
    "空值表示该路召回没有命中这一条。「名次变化」是融合后名次与重排后名次之差，"
    "正数表示被重排提上来了。"
)

# ---------- 3. 重排效果 ----------
if use_rerank:
    moved_up = [r for r in rows if r["名次变化"].startswith("▲")]
    moved_down = [r for r in rows if r["名次变化"].startswith("▼")]
    if moved_up or moved_down:
        st.subheader("③ 重排做了什么")
        c1, c2 = st.columns(2)
        for col, title, group in (
            (c1, "**被提上来的**", moved_up),
            (c2, "**被压下去的**", moved_down),
        ):
            with col:
                st.markdown(title)
                if not group:
                    st.write("（无）")
                for r in group[:5]:
                    st.write(f"- {r['条款']} {r['名次变化']}")
        st.caption(
            "粗排负责「别漏」（Recall@10 已达 0.980），精排负责「别错」——"
            "把对的那一条顶到第 1 位。代价是延迟从 0.10s 涨到 4.53s。"
        )

# ---------- 4. 条款原文 ----------
st.subheader("④ 条款原文")
for h in hits:
    flag = STATUS_COLORS.get(h.get("effective_status", "未知"), "⚪")
    with st.expander(f"[{h['index']}] {flag} {_label(h)}", expanded=h["index"] == 1):
        render_clause_text(h.get("text", ""))
        bits = [h.get("venue", ""), f"重排分 {h.get('score', 0)}"]
        for ch, rank in (h.get("component_ranks") or {}).items():
            bits.append(f"{CHANNEL_NAMES.get(ch, ch)}#{rank}")
        line = " · ".join(b for b in bits if b)
        if h.get("source_url"):
            line += f" · [原文链接]({h['source_url']})"
        st.caption(line)

st.divider()
render_timings(res.get("timings"))
st.caption(f"端到端 {res.get('elapsed', 0):.2f}s（不含生成）")
