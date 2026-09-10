"""语义切分 —— 子块生成。

父块（条款）是给 LLM 看的上下文，子块才是向量库里实际存的检索单元。
小块检索、大块生成：命中子块后通过 parent_id 回溯完整条款。

为什么自己写而不用现成的：
  langchain 1.x 的 `langchain_text_splitters` 里**没有** SemanticChunker，
  它在 `langchain_experimental` 包里（本项目未安装，且它会引入额外依赖）。
  自己实现只有几十行，而且能针对中文法规做两处现成实现做不到的处理：

  1. 中文断句要按 。；！？ 切，而不是英文的 . ! ?
     ——法规里"第3.5条"这种编号会被英文句号切碎
  2. 断点阈值用 (mean + z*std) 自适应，而不是固定余弦阈值
     ——条款长度差异极大，固定阈值在短条款上会一刀不切、长条款上切碎

切分策略：句子级 embedding -> 相邻句余弦距离 -> 距离显著高于均值处断开
-> 合并过短片段、硬切过长片段。
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Sequence

import numpy as np

from ..schema import ChildChunk, ParentChunk, stable_id

logger = logging.getLogger(__name__)

# 中文句末标点 + 英文句末，保留标点在句尾
_SENT_SPLIT = re.compile(r"(?<=[。；！？!?;])\s*|(?<=[.!?])\s+(?=[A-Z])")
# 换行也是天然边界（条款表渲染出来的"字段：值"每行一条）
_LINE_SPLIT = re.compile(r"\n+")


def split_sentences(text: str) -> list[str]:
    """切句。先按换行切，再按句末标点切，保证不丢内容。"""
    out: list[str] = []
    for line in _LINE_SPLIT.split(text):
        line = line.strip()
        if not line:
            continue
        for s in _SENT_SPLIT.split(line):
            s = (s or "").strip()
            if s:
                out.append(s)
    return out


def _cosine_distances(vectors: np.ndarray) -> np.ndarray:
    """相邻句向量的余弦距离。返回长度 n-1 的数组。"""
    if len(vectors) < 2:
        return np.array([])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normed = vectors / np.clip(norms, 1e-12, None)
    sims = np.sum(normed[:-1] * normed[1:], axis=1)
    return 1.0 - sims


def find_breakpoints(distances: np.ndarray, z: float = 1.0) -> list[int]:
    """距离显著高于均值的位置作为断点。

    阈值 = mean + z * std。z 越大切得越少、块越大。
    """
    if distances.size == 0:
        return []
    threshold = float(distances.mean() + z * distances.std())
    return [i for i, d in enumerate(distances) if d > threshold]


def _merge_and_cap(
    sentences: Sequence[str],
    breakpoints: Sequence[int],
    *,
    target_chars: int,
    max_chars: int,
    min_chars: int,
    overlap_sentences: int,
) -> list[str]:
    """按断点组装片段，再做长度约束。

    - 短于 min_chars 的片段并入下一块（避免产生"第X条"这种无信息量的碎块）
    - 长于 max_chars 的片段按句子硬切
    - overlap_sentences 让相邻块共享尾句，避免答案正好落在边界上被切断
    """
    cut = set(breakpoints)
    groups: list[list[str]] = []
    cur: list[str] = []
    for i, s in enumerate(sentences):
        cur.append(s)
        cur_len = sum(len(x) for x in cur)
        # 到达断点、或已经超过目标长度，就收一块
        if (i in cut and cur_len >= target_chars * 0.5) or cur_len >= target_chars:
            groups.append(cur)
            cur = cur[-overlap_sentences:] if overlap_sentences else []
    if cur and (not groups or cur != groups[-1][-len(cur) :]):
        groups.append(cur)

    # 长度约束
    out: list[str] = []
    for g in groups:
        text = "".join(g) if _is_cjk_join(g) else " ".join(g)
        if len(text) <= max_chars:
            out.append(text)
            continue
        # 硬切
        buf = ""
        for s in g:
            if len(buf) + len(s) > max_chars and buf:
                out.append(buf)
                buf = s
            else:
                buf = f"{buf}{s}" if _is_cjk_join([buf, s]) else f"{buf} {s}".strip()
        if buf:
            out.append(buf)

    # 合并过短块
    merged: list[str] = []
    for t in out:
        if merged and len(t) < min_chars:
            merged[-1] = f"{merged[-1]}\n{t}"
        else:
            merged.append(t)
    return [m for m in merged if m.strip()]


def _is_cjk_join(parts: Sequence[str]) -> bool:
    """中文句子拼接不加空格，英文要加。"""
    sample = "".join(parts)[:200]
    if not sample:
        return True
    cjk = sum(1 for c in sample if "一" <= c <= "鿿")
    return cjk / len(sample) > 0.15


def semantic_split(
    text: str,
    embed_fn: Callable[[list[str]], np.ndarray] | None,
    *,
    target_chars: int = 300,
    max_chars: int = 512,
    min_chars: int = 80,
    breakpoint_z: float = 1.0,
    overlap_sentences: int = 1,
) -> list[str]:
    """对一段文本做语义切分，返回子块文本列表。

    embed_fn 为 None 时退化为纯长度切分（用于消融对比，也让不加载
    模型就能跑通整条流水线）。
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    if embed_fn is None:
        breakpoints: list[int] = []
    else:
        try:
            vectors = np.asarray(embed_fn(sentences))
            breakpoints = find_breakpoints(_cosine_distances(vectors), breakpoint_z)
        except Exception as e:  # noqa: BLE001 - 切分不应因模型问题整体失败
            logger.warning("语义切分回退到长度切分: %s", e)
            breakpoints = []

    return _merge_and_cap(
        sentences,
        breakpoints,
        target_chars=target_chars,
        max_chars=max_chars,
        min_chars=min_chars,
        overlap_sentences=overlap_sentences,
    )


def build_children(
    parent: ParentChunk,
    embed_fn: Callable[[list[str]], np.ndarray] | None,
    **kwargs,
) -> list[ChildChunk]:
    """把一个父块切成若干子块，并冗余检索期要用的元数据。"""
    # 条款表行本身就是原子事实，不再切
    if parent.chunk_type == "table_row":
        pieces = [parent.text]
    else:
        pieces = semantic_split(parent.text, embed_fn, **kwargs)

    children: list[ChildChunk] = []
    for i, piece in enumerate(pieces):
        children.append(
            ChildChunk(
                child_id=stable_id(parent.parent_id, str(i), prefix="c_"),
                parent_id=parent.parent_id,
                doc_id=parent.doc_id,
                text=piece,
                venue=parent.venue,
                lang=parent.lang,
                source_url=parent.source_url,
                doc_title=parent.doc_title,
                doc_no=parent.doc_no,
                clause_id=parent.clause_id,
                effective_status=parent.effective_status,
                chunk_type=parent.chunk_type,
                position=i,
            )
        )
    return children
