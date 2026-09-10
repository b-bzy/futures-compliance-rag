"""Streamlit 演示界面 —— 问答主页。

界面通过 HTTP 访问 FastAPI，自身不加载任何模型（原因见 ui/api_client.py
的模块说明）。答案走 SSE 流式返回：引用先到、答案逐字补，避免演示时
空白干等 —— RAG 的首 token 延迟天然很高，一次性返回会毁掉现场节奏。

另外两个页面在 ui/pages/ 下：
    1_检索过程剖析   一个查询在三路召回与重排之间的名次流转
    2_效果与取舍     消融实验数据与技术决策记录

启动（需先起后端）:
    uvicorn derivrag.api.server:app --port 8000
    streamlit run src/derivrag/ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from derivrag.ui.api_client import APIError  # noqa: E402
from derivrag.ui.components import (  # noqa: E402
    get_client,
    render_backend_status,
    render_citations,
    render_confidence,
    render_timings,
)

st.set_page_config(page_title="期货合规条款 RAG 检索系统", page_icon="📑", layout="wide")

st.title("📑 期货合规条款 RAG 检索系统")
st.caption(
    "混合检索（BM25 + BGE-M3 稠密 + 关键词） · score-aware RRF 融合 · "
    "BGE-Reranker-v2-M3 重排 · HyDE · 多轮查询重写"
)

# ---------- 侧边栏 ----------
health = render_backend_status()

with st.sidebar:
    st.divider()
    st.header("检索配置")
    use_hyde = st.toggle("HyDE 假设文档", value=True, help="解决口语提问与条款书面语的语义不对称")
    use_rewrite = st.toggle("多轮查询重写", value=True, help="把追问补全成独立可检索的问题")
    use_rerank = st.toggle("交叉编码器重排", value=True, help="从 top-50 精排出 top-5")
    top_k = st.slider("返回条款数", 1, 15, 5)

    if not use_rerank:
        st.caption("⚠️ 关掉重排后没有校准的分数，置信度会显示为“未知”。")

    st.divider()
    stats = (health or {}).get("corpus_stats") or {}
    if stats:
        st.subheader("语料规模")
        c1, c2 = st.columns(2)
        c1.metric("原始文档", stats.get("documents_parsed", "—"))
        c2.metric("条款父块", f"{stats.get('parents', 0):,}")
        c1.metric("检索子块", f"{stats.get('children', 0):,}")
        c2.metric("总字数", f"{stats.get('total_chars', 0):,}")
        with st.expander("按交易所"):
            for v, n in sorted(stats.get("by_venue", {}).items(), key=lambda x: -x[1]):
                st.write(f"- {v}: {n} 份")

    st.divider()
    if st.button("🗑 清空对话", use_container_width=True):
        if sid := st.session_state.get("session_id"):
            get_client().clear_session(sid)
        st.session_state.messages = []
        st.session_state.pop("session_id", None)
        st.rerun()

# ---------- 会话状态 ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- 历史消息 ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("rewritten_query"):
                st.info(f"🔄 查询已改写为：**{msg['rewritten_query']}**")
            render_confidence(msg.get("confidence"))
            render_citations(msg.get("citations", []), expanded=False)

# ---------- 输入 ----------
PLACEHOLDER = "请输入关于期权条款的问题，例如：50ETF期权的合约单位是多少？"

if prompt := st.chat_input(PLACEHOLDER, disabled=health is None):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        meta_slot = st.empty()      # 查询改写提示
        conf_slot = st.container()  # 置信度提示条
        cite_slot = st.container()  # 引用卡片
        answer_slot = st.empty()    # 流式答案

        citations: list[dict] = []
        confidence: dict = {}
        rewritten: str | None = None
        timings: dict = {}
        parts: list[str] = []
        failed: str | None = None

        with st.spinner("检索中…"):
            try:
                for event, data in get_client().chat_stream(
                    prompt,
                    session_id=st.session_state.get("session_id"),
                    top_k=top_k,
                    use_hyde=use_hyde,
                    use_rewrite=use_rewrite,
                    use_rerank=use_rerank,
                ):
                    if event == "meta":
                        st.session_state["session_id"] = data.get("session_id")
                        timings = data.get("timings", {})
                        rewritten = data.get("rewritten_query")
                        if rewritten:
                            meta_slot.info(f"🔄 查询已改写为：**{rewritten}**")
                    elif event == "citations":
                        citations = data.get("citations", [])
                        confidence = data.get("confidence", {})
                        # 引用先于答案落地 —— 用户可以立刻开始读条款原文
                        with conf_slot:
                            render_confidence(confidence)
                        with cite_slot:
                            render_citations(citations)
                    elif event == "token":
                        parts.append(data.get("text", ""))
                        answer_slot.markdown("".join(parts) + "▌")
                    elif event == "done":
                        timings = data.get("timings", timings)
                    elif event == "error":
                        failed = data.get("detail", "未知错误")
                        break
            except APIError as e:
                failed = str(e)

        answer = "".join(parts)
        if answer:
            answer_slot.markdown(answer)  # 去掉光标
        else:
            answer_slot.empty()

        if failed:
            st.error(failed)
            if citations:
                # 检索结果仍然有价值，别让这一轮白跑
                st.caption("生成未完成，但已检索到的条款仍显示在上方引用中。")
        render_timings(timings)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer or f"_（本轮未生成答案：{failed or '无输出'}）_",
                "citations": citations,
                "confidence": confidence,
                "rewritten_query": rewritten,
            }
        )
