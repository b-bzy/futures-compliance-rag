"""HyDE 失效时的降级行为测试 —— 「白花钱」那类缺陷的回归。

背景：HyDE 每次调用都要付费和等待。它失败时降级为原始查询检索是对的
（不该让整条检索挂掉），但**降级必须是响亮的**：

1. 静默返回空列表，会让消融实验把「plus_hyde 档」测成「没开 HyDE 档」，
   于是得出「HyDE 无收益」的结论 —— 而真相是 HyDE 根本没跑。
   RESULTS.md 里那张 HyDE 配对表就有这个风险。
2. 用户侧则是付了钱、等了 16 倍延迟，什么也没换来。

实测证据：query.hyde.max_tokens=256 时 deepseek-flash 返回
finish_reason=length、reasoning_tokens=256、content 长度 0。

全部用 stub，不联网、不花钱。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from derivrag.llm.provider import LLMTruncatedError  # noqa: E402
from derivrag.retrieve.hyde import generate_hypothetical, rewrite_query  # noqa: E402


class _Provider:
    """按脚本返回或抛异常，并记录收到的 max_tokens。"""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.budgets: list[int] = []
        self.calls = 0

    def chat(self, messages, *, max_tokens=None, temperature=None, stop=None):  # noqa: ANN001, ANN003
        self.budgets.append(max_tokens)
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


# =====================================================================
class TestHydeDegradation:
    def test_truncation_logged_as_error_not_warning(self, caplog):
        """预算不足是配置错误，必须 ERROR 级别 —— WARNING 会淹没在日志里。"""
        p = _Provider([LLMTruncatedError("finish_reason=length")])
        with caplog.at_level(logging.DEBUG):
            out = generate_hypothetical(p, "备兑开仓的保证金怎么算？")
        assert out == [], "仍应降级，不能让检索挂掉"
        errs = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errs, "截断必须记 ERROR"
        assert "已付费但无收益" in errs[0].message

    def test_empty_text_also_logged_as_error(self, caplog):
        """provider 返回空串（而非抛错）时，同样是白花钱，不能静默。"""
        p = _Provider(["", "  "])
        with caplog.at_level(logging.DEBUG):
            out = generate_hypothetical(p, "问题", n=2)
        assert out == []
        assert [r for r in caplog.records if r.levelno >= logging.ERROR]

    def test_runtime_failure_is_warning_not_error(self, caplog):
        """网络/限流属于运行时波动，不该和配置错误同级 —— 否则告警失去区分度。"""
        p = _Provider([ConnectionError("连接超时")])
        with caplog.at_level(logging.DEBUG):
            out = generate_hypothetical(p, "问题")
        assert out == []
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_partial_success_kept(self):
        """n=3 时第二份为空，不该丢掉已经付费拿到的另外两份。"""
        p = _Provider(["文档一", "", "文档三"])
        assert generate_hypothetical(p, "问题", n=3) == ["文档一", "文档三"]

    def test_default_budget_not_the_old_256(self):
        """默认值回归 —— 256 曾同时存在于三处，改一处另两处会复活 bug。"""
        p = _Provider(["文档"])
        generate_hypothetical(p, "问题")
        assert p.budgets[0] >= 1024, f"HyDE 默认预算 {p.budgets[0]} 过小，推理模型会被截断"

    def test_rewrite_default_budget_not_the_old_200(self):
        p = _Provider(["改写后的问题"])
        rewrite_query(p, "它的最小变动价位是多少？", [{"role": "user", "content": "沪深300股指期权"}])
        assert p.budgets[0] >= 1024, f"查询改写默认预算 {p.budgets[0]} 过小"

    def test_rewrite_falls_back_to_original_on_failure(self):
        """改写失败必须原样返回问题，不能返回空串把查询清空。"""
        p = _Provider([LLMTruncatedError("length")])
        q = "它的最小变动价位是多少？"
        assert rewrite_query(p, q, [{"role": "user", "content": "沪深300"}]) == q


class TestPipelineHydeObservability:
    """「开了 HyDE 但没生效」必须与「没开 HyDE」可区分。

    否则消融实验的 plus_hyde 档会静默退化成 plus_rerank 档，
    报告里那句「HyDE 没有观察到可辨别收益」就变成了自证预言。
    """

    def test_config_default_matches_yaml(self):
        """pipeline 的兜底默认值必须与 configs/config.yaml 一致。"""
        import re

        src = (Path(__file__).resolve().parents[1] / "src/derivrag/pipeline.py").read_text(
            encoding="utf-8"
        )
        m = re.search(r'"hyde", \{\}\)\.get\("max_tokens", (\d+)\)', src)
        assert m, "没找到 pipeline 里的 hyde max_tokens 兜底默认值"
        assert int(m.group(1)) >= 1024, f"兜底默认值 {m.group(1)} 过小"

    def test_yaml_budget_is_safe(self):
        """配置文件里的值本身也要够用。"""
        import yaml

        root = Path(__file__).resolve().parents[1]
        cfg = yaml.safe_load((root / "configs/config.yaml").read_text(encoding="utf-8"))
        assert cfg["query"]["hyde"]["max_tokens"] >= 1024
        assert cfg["llm"]["min_output_tokens"] >= 1024
        assert cfg["llm"]["max_tokens"] >= 4096
