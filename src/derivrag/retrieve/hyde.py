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
    # 推理模型的思考与正文共用预算。256 时 deepseek-flash 的 reasoning 就能
    # 用满，正文长度 0，HyDE 静默失效（付了钱、等了时间、拿到空文档）。
    max_tokens: int = 2048,
) -> list[str]:
    """生成假设性条款文本。失败时返回空列表（调用方降级为不用 HyDE）。

    降级本身是对的 —— HyDE 失败不该让整条检索挂掉。但**降级必须是响亮的**：
    HyDE 每次调用都要花钱和时间，静默返回空列表会让消融实验把「HyDE 档」
    测成「没开 HyDE 档」，结论直接失真。所以这里按失败原因分级：

    - 预算不足导致正文为空（LLMTruncatedError）→ ERROR，这是配置错误，
      一定要有人看见；
    - 其他异常（网络、限流、provider 不可用）→ WARNING，属于运行时波动。
    """
    from ..llm.provider import LLMTruncatedError

    out: list[str] = []
    for i in range(max(1, n)):
        try:
            # 多份假设文档之间用温度制造差异
            text = provider.chat(
                build_hyde_messages(question),
                max_tokens=max_tokens,
                temperature=0.1 if i == 0 else 0.7,
            )
        except LLMTruncatedError as e:
            logger.error(
                "HyDE 未产出假设文档（已付费但无收益），降级为原始查询检索。"
                "这是预算配置问题，请调大 configs/config.yaml 的 "
                "query.hyde.max_tokens 或 llm.min_output_tokens: %s",
                e,
            )
            break
        except Exception as e:  # noqa: BLE001 - HyDE 失败不应让检索整体失败
            logger.warning("HyDE 生成失败，降级为原始查询检索: %s", e)
            break
        text = (text or "").strip()
        if text:
            out.append(text)
        else:
            logger.error(
                "HyDE 返回空文本（已付费但无收益），降级为原始查询检索。"
                "第 %d/%d 份假设文档为空。", i + 1, n,
            )
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
    # 同 generate_hypothetical：200 在推理模型上会被思考吃光，改写静默失效、
    # 原样返回未补全的追问，多轮效果归零。改写输出本身很短，1024 足够有余。
    max_tokens: int = 1024,
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
