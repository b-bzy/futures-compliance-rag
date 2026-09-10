"""界面层访问后端的唯一入口。

改造前 Streamlit 直接 `from derivrag.pipeline import RAGPipeline`，在自己
进程里加载模型。这有两个实际问题：

1. 模型会被加载两份。bge-m3 + reranker 在 16GB 机器上跑两份直接 OOM，
   而 FastAPI 那套端点等于白写 —— 系统里存在两条互不相干的调用路径。
2. 前后端无法分离部署。界面和推理绑死在同一个进程、同一台机器上。

现在界面只通过 HTTP 说话，模型只在 API 进程里存在一份。

后端地址由环境变量 DERIVRAG_API_URL 指定，docker-compose 里指向服务名。
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterator

import httpx

DEFAULT_BASE_URL = os.environ.get("DERIVRAG_API_URL", "http://127.0.0.1:8000")

# 连接要快速失败（后端没起来时立刻报错），读取要给足时间：
# 首次请求会触发模型懒加载，叠加重排与生成，几分钟是正常的。
TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0)


class APIError(RuntimeError):
    """后端不可用或返回了错误 —— 界面捕获它并显示可操作的提示。"""


class APIClient:
    """后端 REST/SSE 接口的薄封装。"""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=TIMEOUT)

    # ---------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def stats(self) -> dict[str, Any]:
        return self._request("GET", "/stats")

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        use_rerank: bool = True,
        use_hyde: bool = False,
        channels: tuple[str, ...] = ("bm25", "dense", "keyword"),
        venue: str | None = None,
    ) -> dict[str, Any]:
        """只检索不生成。用于"检索过程剖析"页面 —— 不调 LLM，秒回。"""
        return self._request(
            "POST",
            "/search",
            json={
                "query": query,
                "top_k": top_k,
                "use_rerank": use_rerank,
                "use_hyde": use_hyde,
                "channels": list(channels),
                "venue": venue,
            },
        )

    def clear_session(self, session_id: str) -> None:
        try:
            self._request("DELETE", f"/chat/{session_id}")
        except APIError:
            # 会话清理失败不该阻塞用户重开对话，后端 TTL 兜底
            pass

    # ---------------------------------------------------------------
    def chat_stream(
        self,
        message: str,
        *,
        session_id: str | None = None,
        top_k: int = 5,
        use_hyde: bool | None = None,
        use_rewrite: bool | None = None,
        use_rerank: bool = True,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        """逐个产出 (事件名, 数据)，调用方据此驱动界面更新。

        事件序列见 api/server.py:chat_stream —— meta / citations / token / done。
        """
        body = {
            "message": message,
            "session_id": session_id,
            "top_k": top_k,
            "use_hyde": use_hyde,
            "use_rewrite": use_rewrite,
            "use_rerank": use_rerank,
        }
        try:
            with self._client.stream("POST", "/chat/stream", json=body) as r:
                if r.status_code >= 400:
                    r.read()
                    raise APIError(_detail(r))
                yield from _parse_sse(r.iter_lines())
        except httpx.RequestError as e:
            raise APIError(f"无法连接后端 {self.base_url}：{e}") from e

    # ---------------------------------------------------------------
    def _request(self, method: str, path: str, **kw: Any) -> dict[str, Any]:
        try:
            r = self._client.request(method, path, **kw)
        except httpx.RequestError as e:
            raise APIError(f"无法连接后端 {self.base_url}{path}：{e}") from e
        if r.status_code >= 400:
            raise APIError(_detail(r))
        return r.json()


# =====================================================================
def _parse_sse(lines: Iterator[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """解析 SSE 帧。httpx 的 iter_lines 已按 UTF-8 解码并去掉了行尾。"""
    event = "message"
    for line in lines:
        if not line:  # 空行表示一帧结束，事件名回到默认值
            event = "message"
        elif line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            payload = line[len("data:") :].strip()
            if payload:
                yield event, json.loads(payload)


def _detail(r: httpx.Response) -> str:
    """把 FastAPI 的错误体转成人话。非 JSON 响应（如网关报错）也要能读。"""
    try:
        body = r.json()
    except ValueError:
        return f"后端返回 {r.status_code}：{r.text[:200]}"
    return str(body.get("detail", body)) if isinstance(body, dict) else str(body)
