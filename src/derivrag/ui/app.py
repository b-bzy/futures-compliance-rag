"""Streamlit 演示界面。

除了问答本身，这个界面重点展示**检索过程**：三路召回各自命中了什么、
RRF 融合后名次如何变化、重排把哪些候选提上来又把哪些压下去。
对一个作品集项目来说，能把"混合检索 + 多阶段重排"这句话变成看得见的
名次变化表，比任何文字描述都有说服力。

启动:
    streamlit run src/derivrag/ui/app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from derivrag.config import load_config  # noqa: E402
from derivrag.pipeline import RAGPipeline  # noqa: E402

st.set_page_config(page_title="期货合规条款 RAG 检索系统", page_icon="📑", layout="wide")

STATUS_COLORS = {"现行有效": "🟢", "已废止": "🔴", "被修订": "🟡", "未知": "⚪"}


@st.cache_resource(show_spinner="正在加载模型与索引（首次约需 1 分钟）…")
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(load_config(), lazy=True)


@st.cache_data(ttl=60)
def get_stats() -> dict:
    import json

    p = load_config().resolve("paths.processed") / "stats.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# =====================================================================
st.title("📑 期货合规条款 RAG 检索系统")
st.caption(
    "混合检索（BM25 + BGE-M3 稠密 + 关键词） · score-aware RRF 融合 · "
    "BGE-Reranker-v2-M3 重排 · HyDE · 多轮查询重写"
)

with st.sidebar:
    st.header("检索配置")
    use_hyde = st.toggle("HyDE 假设文档", value=True, help="解决口语提问与条款书面语的语义不对称")
    use_rewrite = st.toggle("多轮查询重写", value=True, help="把追问补全成独立可检索的问题")
    use_rerank = st.toggle("交叉编码器重排", value=True, help="从 top-50 精排出 top-5")
    top_k = st.slider("返回条款数", 1, 15, 5)

    st.divider()
    st.subheader("召回通路（消融）")
    ch_bm25 = st.checkbox("BM25（分词精确匹配）", value=True)
    ch_dense = st.checkbox("稠密向量（语义泛化）", value=True)
    ch_keyword = st.checkbox("关键词（条号/文号定位）", value=True)

    st.divider()
    stats = get_stats()
    if stats:
        st.subheader("语料规模")
        c1, c2 = st.columns(2)
        c1.metric("原始文档", stats.get("documents_parsed", "—"))
        c2.metric("条款父块", f"{stats.get('parents', 0):,}")
        c1.metric("检索子块", f"{stats.get('children', 0):,}")
        c2.metric("总字数", f"{stats.get('total_chars', 0):,}")
        with st.expander("按交易所"):
            for v, n in sorted(
                stats.get("by_venue", {}).items(), key=lambda x: -x[1]
            ):
                st.write(f"- {v}: {n} 份")

    st.divider()
    if st.button("🗑 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

channels = tuple(
    c
    for c, on in (("bm25", ch_bm25), ("dense", ch_dense), ("keyword", ch_keyword))
    if on
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- 历史消息 ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            _render_citations = st.session_state.get("_render_fn")
            if _render_citations:
                _render_citations(msg["citations"], msg.get("meta", {}))


def render_citations(citations: list[dict], meta: dict) -> None:
    """引用卡片 + 检索过程可视化。"""
    if meta.get("rewritten_query"):
        st.info(f"🔄 查询已改写为：**{meta['rewritten_query']}**")

    with st.expander(f"📎 引用条款（{len(citations)} 条）", expanded=True):
        for c in citations:
            flag = STATUS_COLORS.get(c["effective_status"], "⚪")
            header = f"**[{c['index']}]** {flag} {c['doc_title']}"
            if c["clause_id"]:
                header += f" · {c['clause_id']}"
            if c["doc_no"]:
                header += f" · {c['doc_no']}"
            st.markdown(header)

            st.markdown(
                f"<div style='background:#f6f8fa;border-left:3px solid #0969da;"
                f"padding:8px 12px;margin:4px 0;font-size:0.9em;white-space:pre-wrap'>"
                f"{c['text'][:1200]}</div>",
                unsafe_allow_html=True,
            )

            bits = [f"{c['venue']}", f"重排分 {c['score']}"]
            for ch, rank in (c.get("component_ranks") or {}).items():
                bits.append(f"{ch}#{rank}")
            st.caption(" · ".join(bits) + f" · [原文链接]({c['source_url']})")
            st.divider()

    if meta.get("timings"):
        cols = st.columns(len(meta["timings"]))
        for col, (k, v) in zip(cols, meta["timings"].items()):
            col.metric(k, f"{v:.2f}s")


st.session_state["_render_fn"] = render_citations

# ---------- 输入 ----------
if prompt := st.chat_input("请输入关于期权条款的问题，例如：50ETF期权的合约单位是多少？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not channels:
            st.error("至少要启用一条召回通路")
            st.stop()

        pipeline = get_pipeline()
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        with st.spinner("检索中…"):
            t = time.perf_counter()
            try:
                result = pipeline.answer(
                    prompt,
                    history=history,
                    use_hyde=use_hyde,
                    use_rewrite=use_rewrite,
                    use_rerank=use_rerank,
                    channels=channels,
                    top_k=top_k,
                )
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

        st.markdown(result.answer)
        citations = result.citations()
        meta = {
            "rewritten_query": result.rewritten_query,
            "timings": result.timings,
        }
        render_citations(citations, meta)

        if result.hypothetical:
            with st.expander("🧪 HyDE 生成的假设条款（仅用于检索，不作为答案依据）"):
                for h in result.hypothetical:
                    st.text(h)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "citations": citations,
                "meta": meta,
            }
        )
