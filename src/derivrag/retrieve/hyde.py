"""HyDE（Hypothetical Document Embeddings）与多轮查询重写。

两者都是"在检索之前先改造查询"，但解决的问题不同：

  HyDE     解决**语义不对称**。用户的口语化提问和条款的书面语在向量
           空间里距离偏远。先让 LLM 生成一段"像条款"的假设文本，用它
           去检索，命中率显著高于原始问题。
           ⚠️ 假设文档只参与检索，绝不进入生成上下文 —— 否则模型会把
           自己编的内容当依据，这是 HyDE 最常见的误用。

  Rewrite  解决**多轮指代**。"那深证100ETF呢？"这种追问单独拿去检索
           必然失败，要先补全成"深证100ETF期权的到期日是什么时候？"。
"""

from __future__ import annotations

import logging
from typing import Sequence

from ..llm.prompts import build_hyde_messages, build_rewrite_messages

logger = logging.getLogger(__name__)


def generate_hypothetical(
    provider,
    question: str,
    *,
    n: int = 1,
    max_tokens: int = 256,
) -> list[str]:
    """生成假设性条款文本。失败时返回空列表（调用方降级为不用 HyDE）。"""
    out: list[str] = []
    for i in range(max(1, n)):
        try:
            # 多份假设文档之间用温度制造差异
            text = provider.chat(
                build_hyde_messages(question),
                max_tokens=max_tokens,
                temperature=0.1 if i == 0 else 0.7,
            )
        except Exception as e:  # noqa: BLE001 - HyDE 失败不应让检索整体失败
            logger.warning("HyDE 生成失败，降级为原始查询检索: %s", e)
            break
        text = (text or "").strip()
        if text:
            out.append(text)
    return out


def format_history(history: Sequence[dict[str, str]], max_turns: int = 4) -> str:
    """把对话历史格式化成重写提示里的 history 段。"""
    recent = list(history)[-max_turns * 2 :]
    lines = []
    for m in recent:
        role = "用户" if m.get("role") == "user" else "助手"
        content = (m.get("content") or "").strip()
        if not content:
            continue
        # 助手的长回答截断，重写只需要知道谈过什么主体
        if role == "助手" and len(content) > 200:
            content = content[:200] + "…"
        lines.append(f"{role}：{content}")
    return "\n".join(lines) if lines else "（无）"


def rewrite_query(
    provider,
    question: str,
    history: Sequence[dict[str, str]],
    *,
    max_turns: int = 4,
    max_tokens: int = 200,
) -> str:
    """把追问改写成独立可检索的问题。无历史或失败时原样返回。"""
    if not history:
        return question

    try:
        rewritten = provider.chat(
            build_rewrite_messages(question, format_history(history, max_turns)),
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("查询重写失败，使用原始问题: %s", e)
        return question

    rewritten = (rewritten or "").strip().strip('"“”\'')
    # 模型偶尔会输出解释性前缀，取最后一行非空内容
    if "\n" in rewritten:
        parts = [p.strip() for p in rewritten.split("\n") if p.strip()]
        rewritten = parts[-1] if parts else ""

    if not rewritten or len(rewritten) > len(question) * 8:
        return question

    if rewritten != question:
        logger.info("查询重写: %r -> %r", question, rewritten)
    return rewritten
