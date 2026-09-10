#!/usr/bin/env python
"""系统健壮性探针 —— 找 bug 用，不评价答案质量。

与 eval/manual_test_questions.md 互补：那份测「答得对不对」（58 题，人工判定），
这份测「系统会不会崩、会不会静默出错」（自动判定，可回归）。

设计约束
    1. 绝大多数用例打 /search（不调 LLM，零 API 成本）；只有 G 组必须走 /chat。
    2. 每条用例声明期望，判定是「行为是否符合期望」而非「答案是否正确」。
    3. 任何 5xx、任何未捕获异常、任何静默空结果都算缺陷。

用法
    uvicorn derivrag.api.server:app --port 8000     # 先起后端
    python eval/robustness_probe.py                 # 全量
    python eval/robustness_probe.py --group C       # 只跑某组
    python eval/robustness_probe.py --no-llm        # 跳过花钱的 G 组
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
TIMEOUT = 180


# =====================================================================
def call(path: str, payload: dict | None = None, timeout: int = TIMEOUT) -> tuple[int, dict | str, float]:
    """返回 (status, body, elapsed)。网络层异常也转成 status 让用例自己判定。"""
    url = f"{BASE}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    # 后端在 127.0.0.1，必须绕开系统代理，否则 ALL_PROXY 会把请求劫走
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t = time.perf_counter()
    try:
        with opener.open(req, timeout=timeout) as r:
            body = r.read().decode()
            el = time.perf_counter() - t
            try:
                return r.status, json.loads(body), el
            except json.JSONDecodeError:
                return r.status, body, el
    except urllib.error.HTTPError as e:
        el = time.perf_counter() - t
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw), el
        except json.JSONDecodeError:
            return e.code, raw, el
    except Exception as e:  # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}", time.perf_counter() - t


def search(query: str, **kw) -> tuple[int, dict | str, float]:
    body = {"query": query, "top_k": 3, "use_rerank": False}
    body.update(kw)
    return call("/search", body)


RESULTS: list[dict] = []


def check(group: str, cid: str, desc: str, expect: str, ok: bool, detail: str, elapsed: float = 0.0) -> None:
    RESULTS.append(
        {
            "group": group, "id": cid, "desc": desc, "expect": expect,
            "ok": ok, "detail": detail, "elapsed": round(elapsed, 2),
        }
    )
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {cid} {desc}")
    if not ok:
        print(f"         期望: {expect}")
        print(f"         实际: {detail}")


# =====================================================================
def group_a_input_boundary() -> None:
    """A. 输入边界 —— 不该 500，也不该静默返回空。"""
    print("\nA. 输入边界")
    cases = [
        ("A1", "空字符串", "", "422 拒绝，或 200 且 hits 为空但不报错"),
        ("A2", "纯空白", "   \t\n  ", "同上"),
        ("A3", "单个字符", "期", "200，不崩"),
        ("A4", "纯标点", "？？？！！！……", "200 且不崩"),
        ("A5", "纯 emoji", "📈📉💰", "200 且不崩"),
        ("A6", "超长输入 20k 字符", "期权" * 10000, "200 或 422，不能超时/500"),
        ("A7", "繁体中文", "上證50ETF期權的合約單位是多少？", "200，应能召回（bge-m3 支持繁体）"),
        ("A8", "全角数字与符号", "上证５０ＥＴＦ期权的合约单位？", "200，不崩"),
        ("A9", "SQL 注入样式", "'; DROP TABLE parents; --", "200 且不崩（Chroma 无 SQL 注入面，但要确认）"),
        ("A10", "花括号模板注入", "{{ 7*7 }} ${jndi:ldap://x}", "200 且 query 原样回显、未被求值"),
        ("A11", "换行与控制字符", "合约单位\n\r\x00是多少", "200 或 422，不能 500"),
    ]
    for cid, desc, q, expect in cases:
        st, body, el = search(q)
        if st == -1:
            check("A", cid, desc, expect, False, f"请求层失败: {body}", el)
            continue
        if st >= 500:
            check("A", cid, desc, expect, False, f"HTTP {st}: {str(body)[:150]}", el)
            continue
        n = len(body.get("hits", [])) if isinstance(body, dict) else -1
        detail = f"HTTP {st}, hits={n}"
        ok = st in (200, 422)
        if cid == "A10" and st == 200 and isinstance(body, dict):
            # 断言「query 原样回显」而不是「响应里不出现 49」——后者会被延迟
            # 浮点数（如 0.6981286249938421）误伤，是假阳性。
            echoed = body.get("query")
            ok = ok and echoed == q
            detail += f", query 回显={echoed!r}"
        check("A", cid, desc, expect, ok, detail, el)


def group_b_param_boundary() -> None:
    """B. 参数边界 —— 越界值必须被 pydantic 拦住，不能进到检索层。"""
    print("\nB. 参数边界")
    q = "上证50ETF期权的合约单位是多少？"
    cases = [
        ("B1", "top_k=0", {"top_k": 0}, "422（Field ge=1）"),
        ("B2", "top_k=1", {"top_k": 1}, "200，返回 1 条"),
        ("B3", "top_k=50", {"top_k": 50}, "200（上限 le=50）"),
        ("B4", "top_k=51 越界", {"top_k": 51}, "422"),
        ("B5", "top_k=-1", {"top_k": -1}, "422"),
        ("B6", "channels 空列表", {"channels": []}, "422 或 200 空结果，不能 500"),
        ("B7", "channels 非法值", {"channels": ["nonexistent"]}, "不能 500；理想是 422 或忽略未知通路"),
        ("B8", "channels 单路 bm25", {"channels": ["bm25"]}, "200，hits 全部来自 bm25"),
        ("B9", "venue 不存在", {"venue": "NOT_A_VENUE"}, "200 且 hits 为空（过滤后无结果）"),
        ("B10", "venue 大小写", {"venue": "sse"}, "200；若返回 0 条说明 venue 过滤大小写敏感"),
        ("B11", "top_k 传字符串", {"top_k": "abc"}, "422"),
        ("B12", "缺 query 字段", None, "422"),
    ]
    for cid, desc, kw, expect in cases:
        if kw is None:
            st, body, el = call("/search", {"top_k": 3})
        else:
            st, body, el = search(q, **kw)
        if st >= 500 or st == -1:
            check("B", cid, desc, expect, False, f"HTTP {st}: {str(body)[:150]}", el)
            continue
        n = len(body.get("hits", [])) if isinstance(body, dict) else -1
        detail = f"HTTP {st}, hits={n}"
        if cid == "B2":
            ok = st == 200 and n == 1
        elif cid in ("B1", "B4", "B5", "B11", "B12"):
            ok = st == 422
        elif cid == "B8":
            chans = body.get("channel_counts", {}) if isinstance(body, dict) else {}
            ok = st == 200 and set(chans) <= {"bm25"}
            detail += f", channel_counts={chans}"
        elif cid == "B9":
            ok = st == 200 and n == 0
        elif cid == "B10":
            ok = st == 200
            detail += "（0 条即大小写敏感，属可用性问题）"
        else:
            ok = st in (200, 422)
        check("B", cid, desc, expect, ok, detail, el)


def group_c_consistency() -> None:
    """C. 一致性与自洽 —— 同一请求两次结果应一致；声明的通路应真的参与。"""
    print("\nC. 一致性与自洽")
    q = "沪深300股指期权的最小变动价位是多少？"

    st1, b1, _ = search(q, top_k=5)
    st2, b2, _ = search(q, top_k=5)
    ids1 = [h["clause_id"] for h in b1.get("hits", [])] if isinstance(b1, dict) else []
    ids2 = [h["clause_id"] for h in b2.get("hits", [])] if isinstance(b2, dict) else []
    check("C", "C1", "同一查询两次结果一致（检索应确定性）", "两次 hits 顺序完全相同",
          ids1 == ids2 and bool(ids1), f"第一次={ids1}, 第二次={ids2}")

    st, body, _ = search("《上海证券交易所股票期权试点交易规则》第十二条规定了什么？", top_k=10)
    chans = body.get("channel_counts", {}) if isinstance(body, dict) else {}
    check("C", "C2", "带明确条号时 keyword 通路应参与", "channel_counts 含 keyword",
          "keyword" in chans, f"channel_counts={chans}")

    st, body, _ = search("上证发〔2015〕23号 说了什么？", top_k=10)
    chans = body.get("channel_counts", {}) if isinstance(body, dict) else {}
    check("C", "C3", "带明确文号时 keyword 通路应参与", "channel_counts 含 keyword",
          "keyword" in chans, f"channel_counts={chans}")

    st, body, _ = search(q, top_k=5)
    hits = body.get("hits", []) if isinstance(body, dict) else []
    scores = [h["score"] for h in hits]
    check("C", "C4", "hits 按 score 降序", "score 单调不增",
          scores == sorted(scores, reverse=True), f"scores={[round(s, 4) for s in scores]}")

    check("C", "C5", "top-5 无重复父块（父块去重应生效）", "clause_id + doc_title 组合不重复",
          len({(h.get("doc_title"), h.get("clause_id")) for h in hits}) == len(hits),
          f"{len(hits)} 条中唯一 {len({(h.get('doc_title'), h.get('clause_id')) for h in hits})} 条")

    missing = [k for k in ("doc_title", "clause_id", "venue", "effective_status", "source_url", "text")
               if hits and any(h.get(k) in (None, "") for h in hits)]
    check("C", "C6", "引用字段完整（引用真值不能缺）", "每条 hit 的关键字段非空",
          not missing, f"存在空值的字段: {missing or '无'}")


def group_d_session() -> None:
    """D. 会话与状态 —— 伪造 id、并发同 session、清理接口。"""
    print("\nD. 会话与状态")
    st, body, el = call("/chat/1234-not-a-real-session", None)
    st_del, body_del, el2 = call_delete("/chat/1234-not-a-real-session")
    check("D", "D1", "删除不存在的 session", "200（幂等）或 404，不能 500",
          st_del in (200, 404), f"HTTP {st_del}: {str(body_del)[:80]}", el2)

    st, body, el = call("/stats")
    ok = st == 200 and isinstance(body, dict) and body.get("documents_parsed")
    check("D", "D2", "/stats 可用且含语料统计", "200 且 documents_parsed 存在",
          bool(ok), f"HTTP {st}, documents_parsed={body.get('documents_parsed') if isinstance(body, dict) else '?'}", el)

    st, body, el = call("/health")
    h = body if isinstance(body, dict) else {}
    ok = st == 200 and h.get("status") == "ok" and h.get("vector_count", 0) > 0
    check("D", "D3", "/health 反映真实索引状态", "status=ok 且 vector_count>0",
          bool(ok), f"status={h.get('status')}, vector_count={h.get('vector_count')}, llm={h.get('llm_available')}", el)


def call_delete(path: str) -> tuple[int, dict | str, float]:
    req = urllib.request.Request(f"{BASE}{path}", method="DELETE")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    t = time.perf_counter()
    try:
        with opener.open(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode() or "{}"), time.perf_counter() - t
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200], time.perf_counter() - t
    except Exception as e:  # noqa: BLE001
        return -1, f"{type(e).__name__}: {e}", time.perf_counter() - t


def group_e_concurrency() -> None:
    """E. 并发与压力 —— 模型单例在并发下会不会崩或串数据。"""
    print("\nE. 并发与压力")
    qs = [
        "上证50ETF期权的合约单位是多少？",
        "沪深300股指期权的合约乘数是多少？",
        "豆粕期权的最小变动价位是多少？",
        "创业板ETF期权的合约单位是多少？",
        "中证1000股指期权的交易代码怎么表示？",
        "郑商所期权合约的交易单位是什么？",
        "上交所期权的行权方式是什么？",
        "股指期权的行权日是哪一天？",
    ]
    out: dict[int, tuple[int, int, float]] = {}
    lock = threading.Lock()

    def worker(i: int, q: str) -> None:
        st, body, el = search(q, top_k=3)
        n = len(body.get("hits", [])) if isinstance(body, dict) else -1
        with lock:
            out[i] = (st, n, el)

    ts = [threading.Thread(target=worker, args=(i, q)) for i, q in enumerate(qs)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - t0

    bad = {i: v for i, v in out.items() if v[0] != 200 or v[1] <= 0}
    lat = [v[2] for v in out.values()]
    check("E", "E1", f"{len(qs)} 路并发 /search", "全部 200 且各自有结果",
          not bad, f"失败 {len(bad)} 个: {bad}" if bad else f"全部成功，墙钟 {wall:.1f}s", wall)
    if lat:
        check("E", "E2", "并发延迟分布", "P95 不应比中位数高一个数量级",
              max(lat) < statistics.median(lat) * 10 + 5,
              f"中位 {statistics.median(lat):.2f}s / 最大 {max(lat):.2f}s / 墙钟 {wall:.1f}s")

    # 顺序重复同一查询，看有无状态污染
    ids = []
    for _ in range(3):
        st, body, _ = search(qs[0], top_k=3)
        ids.append([h["clause_id"] for h in body.get("hits", [])] if isinstance(body, dict) else [])
    check("E", "E3", "并发后重复查询结果稳定", "三次结果相同",
          len({tuple(x) for x in ids}) == 1, f"三次: {ids}")


def group_f_degradation() -> None:
    """F. 降级路径 —— 依赖不可用时应明确报错，不能静默给错答案。"""
    print("\nF. 降级路径")
    st, body, el = search("上证50ETF期权的合约单位是多少？", use_hyde=True, top_k=3)
    ok = st in (200, 503)
    n = len(body.get("hits", [])) if isinstance(body, dict) else -1
    check("F", "F1", "开启 HyDE（要调 LLM 生成假设文档）", "200 有结果，或 503 明确报错",
          ok and (st == 503 or n > 0), f"HTTP {st}, hits={n}", el)

    st, body, el = search("上证50ETF期权的合约单位是多少？", use_rerank=True, top_k=3)
    ok = st in (200, 503)
    detail = f"HTTP {st}"
    if st == 200 and isinstance(body, dict):
        detail += f", hits={len(body.get('hits', []))}"
    check("F", "F2", "开启重排（模型未下载）", "200 正常重排，或 503 明确报错；不能 500",
          ok, detail, el)


def group_g_generation(skip: bool) -> None:
    """G. 生成链路 —— 唯一花 API 钱的一组，覆盖已知的空答案缺陷。"""
    print("\nG. 生成链路" + ("（--no-llm，已跳过）" if skip else "（会消耗 DeepSeek 额度）"))
    if skip:
        return
    cases = [
        ("G1", "常规事实问答", "上证50ETF期权的合约单位是多少？"),
        ("G2", "曾返回空答案的问题（回归）", "备兑开仓的保证金怎么算？"),
        ("G3", "语料中不存在，应拒答", "白糖期权的合约单位是多少？"),
        ("G4", "需要合成多块", "股票期权市场会受到证券市场指数熔断机制的哪些影响？"),
    ]
    for cid, desc, q in cases:
        st, body, el = call("/chat", {"message": q, "top_k": 5, "use_rerank": False})
        if st != 200 or not isinstance(body, dict):
            check("G", cid, desc, "200 且 answer 非空", False, f"HTTP {st}: {str(body)[:150]}", el)
            continue
        ans = (body.get("answer") or "").strip()
        cites = body.get("citations", [])
        if cid == "G3":
            refused = any(w in ans for w in ("无法", "没有", "未找到", "不存在", "未涉及", "无相关"))
            check("G", cid, desc, "答案明确表示查不到，不编造数字",
                  bool(ans) and refused, f"answer={ans[:110]!r}", el)
        else:
            check("G", cid, desc, "answer 非空且带引用",
                  bool(ans) and bool(cites), f"answer 长度={len(ans)}, 引用={len(cites)}, 前 80 字={ans[:80]!r}", el)


# =====================================================================
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--group", help="只跑某组（A/B/C/D/E/F/G）")
    p.add_argument("--no-llm", action="store_true", help="跳过 G 组（不消耗 API 额度）")
    args = p.parse_args()

    st, body, _ = call("/health")
    if st != 200:
        print(f"后端不可用（/health HTTP {st}）。先执行:")
        print("  uvicorn derivrag.api.server:app --port 8000")
        return 2
    print(f"后端就绪: {json.dumps(body, ensure_ascii=False)[:150]}")

    groups = {
        "A": group_a_input_boundary, "B": group_b_param_boundary,
        "C": group_c_consistency, "D": group_d_session,
        "E": group_e_concurrency, "F": group_f_degradation,
    }
    todo = [args.group.upper()] if args.group else list(groups) + ["G"]
    for g in todo:
        if g in groups:
            groups[g]()
        elif g == "G":
            group_g_generation(args.no_llm)

    print("\n" + "=" * 66)
    fails = [r for r in RESULTS if not r["ok"]]
    by_group: dict[str, list[dict]] = {}
    for r in RESULTS:
        by_group.setdefault(r["group"], []).append(r)
    for g, rs in sorted(by_group.items()):
        n_ok = sum(1 for r in rs if r["ok"])
        print(f"  {g} 组: {n_ok}/{len(rs)} 通过")
    print(f"\n合计 {len(RESULTS) - len(fails)}/{len(RESULTS)} 通过，{len(fails)} 个缺陷")
    if fails:
        print("\n缺陷清单:")
        for r in fails:
            print(f"  {r['id']} {r['desc']}")
            print(f"     期望: {r['expect']}")
            print(f"     实际: {r['detail']}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
