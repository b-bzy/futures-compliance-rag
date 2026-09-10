"""SSE 流式端点与置信度评估的测试。

用 stub 替换 pipeline，所以整套测试不加载任何模型、不访问网络，
在 CI 或没下载模型的机器上都能跑。测的是端点本身的契约：
事件顺序、会话累积、以及各种半失败状态下的行为。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import derivrag.api.server as server  # noqa: E402
from derivrag.api.confidence import assess  # noqa: E402
from derivrag.ui.api_client import _parse_sse  # noqa: E402


# =====================================================================
# 测试替身
# =====================================================================
def make_citation(index: int, rerank: float, status: str = "现行有效") -> dict:
    return {
        "index": index,
        "doc_title": f"条款{index}",
        "doc_no": "",
        "clause_id": "合约单位",
        "venue": "SSE",
        "effective_status": status,
        "source_url": "http://example.com",
        "text": "合约单位为10000份",
        "score": rerank,
        "component_scores": {"rerank": rerank},
        "component_ranks": {"dense": index},
    }


@dataclass
class FakeResult:
    rewritten_query: str | None = None
    hypothetical: list = field(default_factory=list)
    channel_counts: dict = field(default_factory=lambda: {"bm25": 2, "dense": 3})
    timings: dict = field(default_factory=lambda: {"retrieve": 0.12})
    _citations: list = field(default_factory=list)

    def citations(self) -> list:
        return self._citations


class FakePipeline:
    """记录每次调用的入参，供断言多轮历史是否正确传入。"""

    DEFAULT_TOKENS = ["合约单位", "为 10000 份", "。含换行\n的 token"]

    def __init__(self, tokens: list[str] | None = None) -> None:
        self.calls: list[dict] = []
        # 不能写 `tokens or DEFAULT` —— 空列表是合法入参（用于测"没生成出内容"）
        self.tokens = self.DEFAULT_TOKENS if tokens is None else tokens

    def stream_answer(self, question, *, history=None, **kw):
        self.calls.append({"question": question, "history": list(history or []), **kw})
        result = FakeResult(
            rewritten_query="50ETF期权的合约单位是多少？",
            _citations=[make_citation(1, 3.1), make_citation(2, -1.0, "已废止")],
        )
        return result, iter(self.tokens)


def read_events(response) -> list[tuple[str, dict]]:
    """复用界面侧的 SSE 解析器 —— 顺带验证前后端对同一份数据的理解一致。"""
    return list(_parse_sse(iter(response.text.splitlines())))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "_pipeline", FakePipeline())
    server._sessions.clear()
    return TestClient(server.app)


# =====================================================================
# 置信度
# =====================================================================
class TestConfidence:
    def test_high_when_top_hit_is_strong_and_distinct(self):
        c = assess([make_citation(1, 3.0), make_citation(2, -2.0)])
        assert c["level"] == "high"
        assert c["probability"] == pytest.approx(0.9526, abs=1e-3)
        assert not c["ambiguous"]

    def test_flags_near_duplicate_clauses(self):
        """困难集的核心风险：十几个品种的条款表都有"合约单位"这一行。

        此时 top1 绝对分数很高，但与 top2 几乎无差别 —— 只看绝对分数
        会把这种"挑错了一条"的情况误报成高置信。
        """
        c = assess([make_citation(1, 3.0), make_citation(2, 2.95)])
        assert c["ambiguous"] is True
        assert "张冠李戴" in c["hint"]

    def test_margin_is_measured_in_logit_space(self):
        """真实观测：查"50ETF期权的合约单位"时 top1/top2 的 logit 是 6.685/6.157。

        过 sigmoid 后是 0.99876 / 0.99790，概率差只有 0.0009。如果 margin 在
        概率空间算，任何高分结果的 margin 都会趋近 0，ambiguous 将恒为真。
        """
        c = assess([make_citation(1, 6.685), make_citation(2, 6.157)])
        assert c["margin"] == pytest.approx(0.528, abs=1e-3), "margin 应是 logit 差"
        assert c["ambiguous"] is True, "两条都是《上证50ETF期权合约基本条款》，确实难区分"

    def test_well_separated_high_scores_are_not_flagged_ambiguous(self):
        """概率空间的回归测试：这一对在 sigmoid 下概率差仅 0.017，会被误报。

        但 logit 差 2.685（几率比约 15 倍）是干脆的区分，不该报歧义。
        """
        c = assess([make_citation(1, 6.685), make_citation(2, 4.0)])
        assert c["level"] == "high"
        assert c["ambiguous"] is False
        assert "张冠李戴" not in c["hint"]

    def test_low_when_negative_logit(self):
        c = assess([make_citation(1, -1.5), make_citation(2, -3.0)])
        assert c["level"] == "low"
        assert "人工核对" in c["hint"]

    def test_unknown_without_rerank(self):
        """关掉重排只剩 RRF 融合分，那不具备跨查询可比性，不能假装有校准。"""
        c = assess([make_citation(1, 1.0)], reranked=False)
        assert c["level"] == "unknown"
        assert c["probability"] is None

    def test_empty_hits(self):
        c = assess([])
        assert c["level"] == "low"
        assert c["margin"] is None

    def test_single_hit_has_no_margin(self):
        c = assess([make_citation(1, 3.0)])
        assert c["margin"] is None
        assert c["ambiguous"] is False

    def test_counts_stale_citations(self):
        c = assess([make_citation(1, 3.0), make_citation(2, -2.0, "已废止")])
        assert c["stale_citations"] == 1
        assert "废止" in c["hint"]

    @pytest.mark.parametrize("logit,expected", [(-800.0, 0.0), (800.0, 1.0)])
    def test_sigmoid_does_not_overflow(self, logit, expected):
        from derivrag.api.confidence import _sigmoid

        assert _sigmoid(logit) == expected


# =====================================================================
# SSE 端点
# =====================================================================
class TestChatStream:
    def test_event_sequence_and_payloads(self, client):
        r = client.post("/chat/stream", json={"message": "50ETF的合约单位?"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")

        events = read_events(r)
        assert [name for name, _ in events] == [
            "meta", "citations", "token", "token", "token", "done",
        ]

        meta = events[0][1]
        assert meta["session_id"]
        assert meta["rewritten_query"] == "50ETF期权的合约单位是多少？"

        cites = events[1][1]
        assert len(cites["citations"]) == 2
        assert cites["confidence"]["level"] == "high"
        assert cites["confidence"]["stale_citations"] == 1

        done = events[-1][1]
        assert done["answer"] == "合约单位为 10000 份。含换行\n的 token"
        assert "generate" in done["timings"]

    def test_citations_arrive_before_any_token(self, client):
        """引用必须先于答案落地 —— 这是流式改造的全部意义所在。"""
        names = [n for n, _ in read_events(client.post("/chat/stream", json={"message": "x"}))]
        assert names.index("citations") < names.index("token")

    def test_multiturn_history_accumulates(self, client):
        first = client.post("/chat/stream", json={"message": "50ETF的合约单位?"})
        sid = read_events(first)[0][1]["session_id"]

        second = client.post(
            "/chat/stream", json={"message": "那深证100ETF呢?", "session_id": sid}
        )
        assert read_events(second)[0][1]["session_id"] == sid

        history = server._pipeline.calls[1]["history"]
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "50ETF的合约单位?"}
        assert history[1]["role"] == "assistant"

    def test_options_forwarded_to_pipeline(self, client):
        client.post(
            "/chat/stream",
            json={"message": "x", "top_k": 9, "use_hyde": False, "use_rerank": False},
        )
        call = server._pipeline.calls[0]
        assert call["top_k"] == 9
        assert call["use_hyde"] is False
        assert call["use_rerank"] is False

    def test_confidence_unknown_when_rerank_disabled(self, client):
        r = client.post("/chat/stream", json={"message": "x", "use_rerank": False})
        conf = [d for n, d in read_events(r) if n == "citations"][0]["confidence"]
        assert conf["level"] == "unknown"

    def test_missing_index_yields_error_event(self, client, monkeypatch):
        """索引没建好时要发 error 事件，而不是 500 —— 流已经开始了，
        此时抛 HTTPException 客户端只会看到连接中断，得不到可操作的提示。"""

        class Broken:
            def stream_answer(self, *a, **k):
                raise RuntimeError("向量库为空。请先执行 scripts/03_build_index.py")

        monkeypatch.setattr(server, "_pipeline", Broken())
        r = client.post("/chat/stream", json={"message": "x"})
        events = read_events(r)
        assert [n for n, _ in events] == ["error"]
        assert "03_build_index" in events[0][1]["detail"]

    def test_partial_answer_survives_mid_stream_failure(self, client, monkeypatch):
        """生成中途断掉时，已经推出去的 token 不该丢。"""

        class HalfBroken(FakePipeline):
            def stream_answer(self, question, *, history=None, **kw):
                result, _ = FakePipeline.stream_answer(self, question, history=history, **kw)

                def gen():
                    yield "前半段"
                    raise RuntimeError("上游连接中断")

                return result, gen()

        monkeypatch.setattr(server, "_pipeline", HalfBroken())
        r = client.post("/chat/stream", json={"message": "x"})
        events = read_events(r)
        assert [n for n, _ in events] == ["meta", "citations", "token", "error", "done"]
        assert events[-1][1]["answer"] == "前半段"

    def test_empty_answer_not_written_to_history(self, client, monkeypatch):
        """没生成出内容就不该写回历史，否则下一轮会带着空回答做查询重写。"""
        pipeline = FakePipeline(tokens=[])
        monkeypatch.setattr(server, "_pipeline", pipeline)
        r = client.post("/chat/stream", json={"message": "x"})
        sid = read_events(r)[0][1]["session_id"]
        assert server._sessions[sid]["history"] == []

    def test_clear_session(self, client):
        r = client.post("/chat/stream", json={"message": "x"})
        sid = read_events(r)[0][1]["session_id"]
        assert sid in server._sessions

        assert client.delete(f"/chat/{sid}").status_code == 200
        assert sid not in server._sessions


# =====================================================================
# 重排记录融合名次
# =====================================================================
class TestRerankRecordsFusedRank:
    """「检索过程剖析」页面要画出"重排把谁提上来了"，这依赖重排前的融合名次。

    只保留融合分是不够的：返回的是重排后的 top_k，它是送进重排的候选集的
    子集，界面无法从子集反推候选在全量候选里的原始名次。
    """

    @staticmethod
    def _fake_reranker(scores):
        from derivrag.retrieve.rerank import Reranker

        r = Reranker.__new__(Reranker)  # 绕过 __init__，避免加载模型
        r.score = lambda query, documents: scores
        return r

    def _hits(self, n):
        from derivrag.schema import RetrievalHit

        # 模拟融合输出：已按融合分降序
        return [
            RetrievalHit(
                child_id=f"c{i}",
                text=f"条款{i}",
                score=1.0 - i * 0.1,
                source="fused",
                component_ranks={"dense": i + 1},
            )
            for i in range(n)
        ]

    def test_fused_rank_recorded_before_reordering(self):
        hits = self._hits(4)
        # 让原本第 4 名（索引 3）得分最高，被提到第 1
        reranker = self._fake_reranker([0.1, 0.2, 0.3, 9.9])
        out = reranker.rerank("q", hits, top_k=4)

        assert out[0].component_ranks["fused"] == 4, "被提上来的那条，融合名次应是 4"
        assert out[0].component_scores["rerank"] == 9.9
        # 融合分要保留原值，不能被重排分覆盖
        assert out[0].component_scores["fused"] == pytest.approx(0.7)

        assert [h.component_ranks["fused"] for h in out] == [4, 3, 2, 1]

    def test_fused_rank_survives_truncation_to_top_k(self):
        """截断到 top_k 后，仍能看出候选原本排在第几 —— 这正是子集反推不出来的信息。"""
        hits = self._hits(10)
        scores = [0.0] * 10
        scores[7] = 5.0  # 融合第 8 名被提到第 1
        out = self._fake_reranker(scores).rerank("q", hits, top_k=3)

        assert len(out) == 3
        assert out[0].component_ranks["fused"] == 8

    def test_does_not_clobber_channel_ranks(self):
        out = self._fake_reranker([1.0, 2.0]).rerank("q", self._hits(2), top_k=2)
        assert all("dense" in h.component_ranks for h in out)


# =====================================================================
# SSE 解析
# =====================================================================
class TestSSEParsing:
    def test_parses_events_and_restores_newlines(self):
        raw = [
            "event: meta", 'data: {"session_id": "abc"}', "",
            "event: token", 'data: {"text": "含换行\\n的中文"}', "",
        ]
        got = list(_parse_sse(iter(raw)))
        assert got[0] == ("meta", {"session_id": "abc"})
        assert got[1][1]["text"] == "含换行\n的中文"

    def test_ignores_blank_and_comment_lines(self):
        raw = ["", ": keep-alive", "event: done", 'data: {"answer": "x"}', ""]
        assert list(_parse_sse(iter(raw))) == [("done", {"answer": "x"})]
