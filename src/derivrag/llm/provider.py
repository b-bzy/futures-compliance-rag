"""LLM 接入层 —— 双通道，配置切换。

    ollama    本地 qwen3:8b（Q4_K_M, 40960 ctx），零成本、可离线
    deepseek  api.deepseek.com，需要 DEEPSEEK_API_KEY
    openai    任意 OpenAI 兼容端点

⚠️ 为什么 ollama 不走 OpenAI 兼容端点而单独实现：

qwen3 默认开启 thinking。在 ollama 0.31 的 OpenAI 兼容端点上：
  - 思考内容被放进一个**独立字段**，`message.content` 是空字符串
  - 思考本身照常消耗 max_tokens，实测 300 tokens 全部耗尽在思考上，
    `finish_reason` 返回 `length` 而 content 长度为 0
  - 官方的 `/no_think` 软开关时灵时不灵（同一模型不同问题表现不一致）
  - 通过 extra_body 传 `think=False` 被静默忽略

而 ollama 的原生 `/api/chat` 接口支持 `"think": false`，实测稳定关闭思考、
content 正常返回。所以这里对 ollama 用原生接口，对其他 provider 用
OpenAI 协议 —— 两者都实现同一个 chat()/stream() 接口，上层无感知。

设计要点：**provider 不可用时自动降级而不是崩溃**。没配 DEEPSEEK_API_KEY
就退回本地 ollama 并 warning，让这个仓库在任何机器上都能 clone 下来直接跑。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str  # system | user | assistant
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMUnavailableError(RuntimeError):
    """所有 provider 都不可用。"""


class LLMTruncatedError(RuntimeError):
    """模型没有产出任何正文 —— 通常是思考过程吃满了 token 预算。

    推理模型（deepseek-flash、qwen3 的 think 模式等）的思考 token 与正文
    共用 max_tokens。预算不足时会在思考阶段就撞上 length 上限，正文一个
    token 都没输出。实测：max_tokens=2048 时「备兑开仓的保证金怎么算？」
    返回 finish_reason=length、reasoning_tokens=2048、content 长度 0。

    这个错误存在的意义是**别让它静默**：原先 chat() 直接 `content or ""`，
    调用方拿到空串，界面显示一个空气泡，无从判断是模型没话说还是被截断。
    继承 RuntimeError，api/server.py 会把它转成 503 并带上原因。
    """


def _strip_thinking(text: str) -> str:
    """剥掉内联的思考段（部分后端会把 <think> 直接写进 content）。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif text.lstrip().startswith("<think>"):
        return ""  # 思考未结束就被截断，没有有效答案
    return text.strip()


def _to_payload(messages: Sequence[Message | dict[str, str]]) -> list[dict[str, str]]:
    return [m.to_dict() if isinstance(m, Message) else dict(m) for m in messages]


# =====================================================================
class BaseProvider:
    """统一接口。"""

    name: str
    model: str

    def health(self) -> bool:
        raise NotImplementedError

    def chat(self, messages, **kw) -> str:
        raise NotImplementedError

    def stream(self, messages, **kw) -> Iterator[str]:
        raise NotImplementedError


# =====================================================================
class OllamaProvider(BaseProvider):
    """ollama 原生 /api/chat —— 可靠地关闭 qwen3 思考模式。"""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str = "qwen3:latest",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: float = 300.0,
        enable_thinking: bool = False,
    ) -> None:
        self.name = "ollama"
        # 配置里给的是 OpenAI 兼容地址（.../v1），这里去掉后缀取根地址
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        self._client: Any = None

    @property
    def client(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def health(self) -> bool:
        try:
            r = self.client.get(f"{self.base_url}/api/tags", timeout=10)
            return r.status_code == 200
        except Exception as e:  # noqa: BLE001
            logger.debug("ollama 健康检查失败: %s", e)
            return False

    def _body(self, messages, temperature, max_tokens, stream: bool) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": _to_payload(messages),
            "think": self.enable_thinking,
            "stream": stream,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": self.max_tokens if max_tokens is None else max_tokens,
            },
        }

    def chat(
        self,
        messages: Sequence[Message | dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        body = self._body(messages, temperature, max_tokens, stream=False)
        if stop:
            body["options"]["stop"] = stop
        r = self.client.post(f"{self.base_url}/api/chat", json=body)
        r.raise_for_status()
        data = r.json()
        return _strip_thinking(data.get("message", {}).get("content", "") or "")

    def stream(
        self,
        messages: Sequence[Message | dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        body = self._body(messages, temperature, max_tokens, stream=True)
        with self.client.stream("POST", f"{self.base_url}/api/chat", json=body) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    break


# =====================================================================
class OpenAICompatProvider(BaseProvider):
    """OpenAI 兼容协议 —— deepseek / openai / 任意兼容端点。"""

    def __init__(
        self,
        name: str,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: float = 180.0,
    ) -> None:
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client: Any = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "not-needed",
                timeout=self.timeout,
            )
        return self._client

    def health(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception as e:  # noqa: BLE001
            logger.debug("provider %s 健康检查失败: %s", self.name, e)
            return False

    def chat(
        self,
        messages: Sequence[Message | dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> str:
        budget = self.max_tokens if max_tokens is None else max_tokens
        text, reason, used = self._once(messages, temperature, budget, stop)
        if text:
            return text

        # 正文为空。推理模型把预算烧在思考上时 finish_reason 会是 length ——
        # 提一次预算重试，比直接失败更可能拿到答案；只重试一次，避免成本失控。
        if reason == "length":
            retry = min(budget * 3, 16384)
            logger.warning(
                "%s 正文为空且 finish_reason=length（思考用了 %s tokens），"
                "把 max_tokens 从 %d 提到 %d 重试一次",
                self.name, used, budget, retry,
            )
            text, reason, used = self._once(messages, temperature, retry, stop)
            if text:
                return text

        raise LLMTruncatedError(
            f"{self.name} 未产出正文（finish_reason={reason}，"
            f"completion_tokens={used}）。若 finish_reason 为 length，"
            f"请调大 configs/config.yaml 的 llm.max_tokens；"
            f"当前值 {self.max_tokens}。"
        )

    def _once(
        self,
        messages: Sequence[Message | dict[str, str]],
        temperature: float | None,
        budget: int,
        stop: list[str] | None,
    ) -> tuple[str, str | None, int]:
        """发一次请求，返回 (正文, finish_reason, completion_tokens)。"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=_to_payload(messages),
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=budget,
            stop=stop,
        )
        choice = resp.choices[0]
        used = getattr(getattr(resp, "usage", None), "completion_tokens", 0) or 0
        return (
            _strip_thinking(choice.message.content or ""),
            getattr(choice, "finish_reason", None),
            used,
        )

    def stream(
        self,
        messages: Sequence[Message | dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        budget = self.max_tokens if max_tokens is None else max_tokens
        s = self.client.chat.completions.create(
            model=self.model,
            messages=_to_payload(messages),
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=budget,
            stream=True,
        )
        emitted = False
        reason: str | None = None
        for chunk in s:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            reason = getattr(choice, "finish_reason", None) or reason
            delta = choice.delta
            if delta and delta.content:
                emitted = True
                yield delta.content

        # 一个正文 token 都没产出时必须显式失败。流式路径下不能重试
        # （已经开始给前端推事件了），所以直接抛错让 SSE 层发 error 事件，
        # 而不是让界面留一个空气泡。
        if not emitted:
            raise LLMTruncatedError(
                f"{self.name} 流式响应未产出正文（finish_reason={reason}）。"
                f"若为 length，请调大 configs/config.yaml 的 llm.max_tokens；"
                f"当前值 {self.max_tokens}。"
            )


# =====================================================================
def build_provider(cfg: dict | None = None, *, override: str | None = None) -> BaseProvider:
    """按配置构造 provider，不可用时按 fallback_order 降级。"""
    if cfg is None:
        from ..config import load_config

        cfg = load_config()

    llm_cfg = cfg.get_path("llm", {}) if hasattr(cfg, "get_path") else cfg
    providers = llm_cfg.get("providers", {})
    wanted = override or llm_cfg.get("provider", "ollama")
    order = [wanted] + [p for p in llm_cfg.get("fallback_order", []) if p != wanted]

    last_error: str | None = None
    for name in order:
        spec = providers.get(name)
        if not spec:
            last_error = f"配置里没有 provider {name!r}"
            continue

        api_key = spec.get("api_key") or ""
        if not api_key and name != "ollama":
            last_error = (
                f"{name} 缺少 API key，请设置对应环境变量"
                f"（如 export DEEPSEEK_API_KEY=...）"
            )
            logger.warning("%s —— 跳过。", last_error)
            continue

        if name == "ollama":
            provider: BaseProvider = OllamaProvider(
                base_url=spec.get("base_url", "http://localhost:11434/v1"),
                model=spec.get("model", "qwen3:latest"),
                temperature=llm_cfg.get("temperature", 0.1),
                max_tokens=llm_cfg.get("max_tokens", 2048),
                timeout=llm_cfg.get("timeout_seconds", 300),
                enable_thinking=llm_cfg.get("enable_thinking", False),
            )
        else:
            provider = OpenAICompatProvider(
                name,
                base_url=spec.get("base_url", ""),
                api_key=api_key,
                model=spec.get("model", ""),
                temperature=llm_cfg.get("temperature", 0.1),
                max_tokens=llm_cfg.get("max_tokens", 2048),
                timeout=llm_cfg.get("timeout_seconds", 180),
            )

        if provider.health():
            if name != wanted:
                logger.warning("provider %s 不可用，已降级到 %s", wanted, name)
            logger.info("LLM provider: %s (%s)", name, provider.model)
            return provider

        last_error = f"{name} 健康检查失败"
        logger.warning("%s", last_error)

    raise LLMUnavailableError(
        f"没有可用的 LLM provider。最后一次错误: {last_error}\n"
        f"本地方案: 安装 ollama 后执行 `ollama pull qwen3`；"
        f"云端方案: export DEEPSEEK_API_KEY=..."
    )


_default_provider: BaseProvider | None = None


def get_provider(cfg: dict | None = None) -> BaseProvider:
    global _default_provider
    if _default_provider is None:
        _default_provider = build_provider(cfg)
    return _default_provider


def reset_provider() -> None:
    global _default_provider
    _default_provider = None
