"""推理模型正文被截断时的行为测试。

对应一个真实缺陷：deepseek-flash 的思考 token 与正文共用 max_tokens，
预算不足时思考阶段就撞上 length 上限，正文长度 0。原先 chat() 直接
`content or ""`，调用方拿到空串，界面显示一个空气泡 —— 无从判断是模型
没话说还是被截断。实测证据（max_tokens=2048，问「备兑开仓的保证金怎么算？」）：

    finish_reason: length
    completion_tokens: 2048  其中 reasoning_tokens: 2048
    content 长度: 0

全部用 stub 替换 client，不联网、不消耗 API 额度。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from derivrag.llm.provider import (  # noqa: E402
    LLMTruncatedError,
    OpenAICompatProvider,
)


# =====================================================================
class _Delta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str | None, finish_reason: str | None) -> None:
        self.message = _Delta(content)
        self.delta = _Delta(content)
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self, completion_tokens: int) -> None:
        self.completion_tokens = completion_tokens


class _Resp:
    def __init__(self, content: str | None, finish_reason: str, used: int) -> None:
        self.choices = [_Choice(content, finish_reason)]
        self.usage = _Usage(used)


class _FakeCompletions:
    """按预设脚本逐次返回，并记录每次收到的 max_tokens。"""

    def __init__(self, script: list[_Resp] | list[list[_Resp]]) -> None:
        self.script = list(script)
        self.budgets: list[int] = []
        self.calls = 0

    def create(self, **kw):  # noqa: ANN003, ANN201
        self.budgets.append(kw.get("max_tokens"))
        self.calls += 1
        item = self.script[min(self.calls - 1, len(self.script) - 1)]
        if kw.get("stream"):
            return iter(item if isinstance(item, list) else [item])
        return item


class _FakeClient:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.chat = type("C", (), {"completions": completions})()


def _provider(script, *, max_tokens: int = 2048) -> tuple[OpenAICompatProvider, _FakeCompletions]:
    comp = _FakeCompletions(script)
    p = OpenAICompatProvider(
        name="deepseek", base_url="http://stub", api_key="k",
        model="deepseek-flash", max_tokens=max_tokens,
    )
    p._client = _FakeClient(comp)
    return p, comp


# =====================================================================
class TestChatTruncation:
    def test_normal_answer_no_retry(self):
        """正常返回时不应重试，也不应改动预算。"""
        p, comp = _provider([_Resp("合约单位是 10000 份。", "stop", 42)])
        assert p.chat([{"role": "user", "content": "q"}]) == "合约单位是 10000 份。"
        assert comp.calls == 1
        assert comp.budgets == [2048]

    def test_empty_on_length_retries_with_bigger_budget(self):
        """思考吃满预算 → 提高预算重试一次并拿到答案。"""
        p, comp = _provider([
            _Resp("", "length", 2048),          # 第一次：思考烧光预算
            _Resp("备兑开仓不收维持保证金。", "stop", 900),  # 重试成功
        ])
        assert p.chat([{"role": "user", "content": "q"}]) == "备兑开仓不收维持保证金。"
        assert comp.calls == 2
        assert comp.budgets == [2048, 6144], "重试预算应为 3 倍"

    def test_retry_budget_capped(self):
        """重试预算有上限，避免成本失控。"""
        p, comp = _provider([_Resp("", "length", 9000), _Resp("答案", "stop", 10)], max_tokens=8192)
        p.chat([{"role": "user", "content": "q"}])
        assert comp.budgets[1] == 16384, "3 倍会超过上限，应被压到 16384"

    def test_still_empty_after_retry_raises(self):
        """重试后仍无正文必须显式失败，不能返回空串。"""
        p, comp = _provider([_Resp("", "length", 2048)])  # 脚本耗尽后一直返回同一条
        with pytest.raises(LLMTruncatedError) as e:
            p.chat([{"role": "user", "content": "q"}])
        assert comp.calls == 2, "只重试一次"
        assert "length" in str(e.value)

    def test_empty_on_stop_raises_without_retry(self):
        """finish_reason 不是 length 时说明不是预算问题，不该浪费一次调用。"""
        p, comp = _provider([_Resp("", "stop", 5)])
        with pytest.raises(LLMTruncatedError):
            p.chat([{"role": "user", "content": "q"}])
        assert comp.calls == 1

    def test_thinking_only_content_treated_as_empty(self):
        """部分后端把未闭合的 <think> 直接写进 content，等同没有答案。"""
        p, _ = _provider([_Resp("<think>让我想想", "length", 2048), _Resp("答案", "stop", 9)])
        assert p.chat([{"role": "user", "content": "q"}]) == "答案"


class TestOutputBudgetFloor:
    """输出预算下限 —— 在 provider 边界上堵死整类「小预算 → 正文为空」。

    这是对「逐个调用点改数字」的替代：仓库里曾同时存在 HyDE 256、查询改写
    200、QA 挖掘 512、去泄漏改写 200 四个独立的小预算，且 HyDE 的 256 还
    分散在 config、hyde.py 默认值、pipeline.py 兜底默认三处。改其中一处，
    其余几处照样复活同一个 bug。
    """

    def test_small_budget_raised_to_floor(self):
        p, comp = _provider([_Resp("假设条款", "stop", 60)], max_tokens=8192)
        p.min_output_tokens = 1024
        p.chat([{"role": "user", "content": "q"}], max_tokens=256)
        assert comp.budgets == [1024], "256 应被抬到下限 1024"

    def test_large_budget_untouched(self):
        p, comp = _provider([_Resp("答案", "stop", 60)], max_tokens=8192)
        p.min_output_tokens = 1024
        p.chat([{"role": "user", "content": "q"}], max_tokens=4096)
        assert comp.budgets == [4096], "高于下限的预算不应被干预"

    def test_floor_can_be_disabled(self):
        """非推理模型不该被强行抬高 —— 下限必须可关。"""
        p, comp = _provider([_Resp("答案", "stop", 60)])
        p.min_output_tokens = 0
        p.chat([{"role": "user", "content": "q"}], max_tokens=200)
        assert comp.budgets == [200]

    def test_floor_warns_only_once(self, caplog):
        """下限告警不能刷屏 —— 去泄漏改写会连着调用 222 次。"""
        import logging as _logging

        p, _ = _provider([_Resp("答案", "stop", 60)], max_tokens=8192)
        p.min_output_tokens = 1024
        with caplog.at_level(_logging.WARNING):
            for _ in range(5):
                p.chat([{"role": "user", "content": "q"}], max_tokens=256)
        hits = [r for r in caplog.records if "低于下限" in r.message]
        assert len(hits) == 1, f"应只告警一次，实际 {len(hits)} 次"

    def test_floor_applies_to_stream(self):
        p, comp = _provider([[_Resp("片段", "stop", 5)]], max_tokens=8192)
        p.min_output_tokens = 1024
        list(p.stream([{"role": "user", "content": "q"}], max_tokens=200))
        assert comp.budgets == [1024], "流式路径同样要受下限保护"

    def test_floor_applies_to_default_budget(self):
        """连 provider 自己的 max_tokens 过小时也要兜住。"""
        p, comp = _provider([_Resp("答案", "stop", 60)], max_tokens=128)
        p.min_output_tokens = 1024
        p.chat([{"role": "user", "content": "q"}])
        assert comp.budgets == [1024]


class TestStreamTruncation:
    def test_stream_yields_content(self):
        p, _ = _provider([[_Resp("上证", "", 0), _Resp("50ETF", "stop", 8)]])
        assert "".join(p.stream([{"role": "user", "content": "q"}])) == "上证50ETF"

    def test_stream_without_content_raises(self):
        """流式下不能重试（已经在推事件了），必须抛错让 SSE 层发 error 事件。

        否则界面会留一个永远补不上的空气泡 —— 这正是修复前的表现。
        """
        p, _ = _provider([[_Resp(None, "length", 2048)]])
        with pytest.raises(LLMTruncatedError) as e:
            list(p.stream([{"role": "user", "content": "q"}]))
        assert "length" in str(e.value)

    def test_truncated_error_is_runtime_error(self):
        """api/server.py 捕获 RuntimeError 转 503，继承关系不能断。"""
        assert issubclass(LLMTruncatedError, RuntimeError)
