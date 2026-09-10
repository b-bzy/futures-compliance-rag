"""三个页面共用的渲染组件。

抽出来的主要原因是引用卡片和置信度提示要在"问答"和"检索剖析"两处
保持完全一致的样式 —— 演示时如果同一条条款在两个页面长得不一样，
观众会先怀疑数据对不对，而不是听你讲检索链路。
"""

from __future__ import annotations

import os
from typing import Any

import streamlit as st

from .api_client import APIClient

STATUS_COLORS = {"现行有效": "🟢", "已废止": "🔴", "被修订": "🟡", "未知": "⚪"}

# 置信度四档对应的展示方式：(streamlit 组件, 图标, 档位中文名)
CONFIDENCE_STYLES = {
    "high": (st.success, "✅", "高"),
    "medium": (st.warning, "⚠️", "中"),
    "low": (st.error, "🛑", "低"),
    "unknown": (st.info, "❔", "未知"),
}

CHANNEL_NAMES = {
    "bm25": "BM25 分词匹配",
    "dense": "稠密向量语义",
    "keyword": "关键词条号定位",
    "fused": "RRF 融合",
    "rerank": "交叉编码器重排",
}


@st.cache_resource
def get_client() -> APIClient:
    """APIClient 内含连接池，跨 rerun 复用同一个实例。"""
    return APIClient()


def render_confidence(conf: dict[str, Any] | None) -> None:
    """置信度提示条。

    这是整个界面里唯一一个"劝用户别全信"的控件。合规问答场景下，
    把不确定性显式说出来比多两个点的召回率更有价值。
    """
    if not conf:
        return
    render, icon, label = CONFIDENCE_STYLES.get(
        conf.get("level", "unknown"), CONFIDENCE_STYLES["unknown"]
    )

    bits = [f"{icon} **检索置信度：{label}**"]
    if conf.get("probability") is not None:
        bits.append(f"最佳候选相关概率 {conf['probability']:.1%}")
    if conf.get("margin") is not None:
        # margin 是 logit 差而非概率差（原因见 api/confidence.py 模块说明），
        # 所以按分差展示，不能写成百分比
        bits.append(f"与次优分差 {conf['margin']:.2f}")
    render(" · ".join(bits) + f"\n\n{conf.get('hint', '')}")


def render_citations(citations: list[dict[str, Any]], *, expanded: bool = True) -> None:
    """引用卡片：条款原文 + 时效状态 + 各路召回名次 + 原文链接。"""
    if not citations:
        return
    with st.expander(f"📎 引用条款（{len(citations)} 条）", expanded=expanded):
        for c in citations:
            flag = STATUS_COLORS.get(c.get("effective_status", "未知"), "⚪")
            header = f"**[{c['index']}]** {flag} {c.get('doc_title', '')}"
            if c.get("clause_id"):
                header += f" · {c['clause_id']}"
            if c.get("doc_no"):
                header += f" · {c['doc_no']}"
            st.markdown(header)
            render_clause_text(c.get("text", ""))

            bits = [c.get("venue", ""), f"重排分 {c.get('score', 0)}"]
            for ch, rank in (c.get("component_ranks") or {}).items():
                bits.append(f"{CHANNEL_NAMES.get(ch, ch)}#{rank}")
            line = " · ".join(b for b in bits if b)
            if c.get("source_url"):
                line += f" · [原文链接]({c['source_url']})"
            st.caption(line)
            st.divider()


def render_clause_text(text: str, *, limit: int = 1200) -> None:
    """条款正文块。等宽缩进 + 左侧色条，和答案正文区分开。"""
    body = text[:limit] + ("…" if len(text) > limit else "")
    st.markdown(
        "<div style='background:#f6f8fa;border-left:3px solid #0969da;"
        "padding:8px 12px;margin:4px 0;font-size:0.9em;white-space:pre-wrap'>"
        f"{_escape(body)}</div>",
        unsafe_allow_html=True,
    )


def render_timings(timings: dict[str, float] | None) -> None:
    """各阶段耗时。演示时用来说明"慢在哪一步"。"""
    if not timings:
        return
    items = [(k, v) for k, v in timings.items() if isinstance(v, (int, float))]
    if not items:
        return
    cols = st.columns(len(items))
    for col, (k, v) in zip(cols, items):
        col.metric(k, f"{v:.2f}s")


def render_backend_status() -> dict[str, Any] | None:
    """侧边栏的后端连接状态。连不上时给出可直接照做的启动命令。"""
    client = get_client()
    try:
        h = client.health()
    except Exception as e:  # noqa: BLE001
        st.sidebar.error(f"后端未连接\n\n{e}")
        st.sidebar.code(
            "uvicorn derivrag.api.server:app --port 8000\n# 或\ndocker compose up",
            language="bash",
        )
        return None

    ok = h.get("status") == "ok"
    st.sidebar.success("后端已连接") if ok else st.sidebar.warning(
        f"后端已连接，但索引未就绪（{h.get('status')}）"
    )
    st.sidebar.caption(
        f"{os.environ.get('DERIVRAG_API_URL', 'http://127.0.0.1:8000')} · "
        f"向量 {h.get('vector_count') or 0:,} 条 · "
        f"生成模型 {h.get('llm_provider') or '不可用'}"
    )
    return h


def _escape(text: str) -> str:
    """条款正文按 unsafe_allow_html 渲染，必须先转义，否则含 < 的内容会被吞。"""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
