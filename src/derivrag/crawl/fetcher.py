"""带内容校验的下载器。

这是全项目最容易翻车的地方。实测结论：**HTTP 200 在这批站点上是无效的
健康检查**，必须校验内容本身：

- SHFE/INE 对所有 HTML 请求返回 HTTP 200 + 10,648 字节的
  `<title>WEB 应用防火墙</title>` 验证码页
- www.sgx.com/sites/default/files/*.pdf 返回 200 但 Content-Type 是
  text/html（SPA 壳），真实文件只在 api2.sgx.com
- CFFEX 的 https 直接 TLS 握手失败（curl exit 35），必须走 http
- DCE / CZCE 所有 HTML 返回 412，只有静态 PDF 路径可取

所以每次下载都断言三件事：Content-Type 匹配、magic bytes 正确、
长度不在 WAF 指纹表里。任何一条不过就记为 failed 并保留错误原因，
而不是把一个验证码页当成规则文档存进语料库。
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from ..schema import FetchRecord

logger = logging.getLogger(__name__)

# 各格式的文件头。PDF 的 %PDF 可能前面有少量空白字节，所以在前 1KB 内搜索。
MAGIC = {
    "pdf": [b"%PDF"],
    "doc": [b"\xd0\xcf\x11\xe0"],  # OLE2 复合文档 (Word 97-2003)
    "docx": [b"PK\x03\x04"],  # ZIP 容器 (OOXML)
    "xlsx": [b"PK\x03\x04"],
}

# Content-Type 前缀白名单
CONTENT_TYPE_OK = {
    "pdf": ("application/pdf", "application/octet-stream"),
    "html": ("text/html", "application/xhtml"),
    "doc": ("application/msword", "application/octet-stream"),
    "docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml",
        "application/octet-stream",
    ),
}

DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


class ValidationError(Exception):
    """内容校验失败 —— 拿到了 2xx，但内容不是我们要的东西。"""


class Fetcher:
    """限速 + 缓存 + 内容校验的下载器。

    adapter 只负责产出 URL 列表，所有实际下载都经过这里，
    保证校验逻辑只有一份实现。
    """

    def __init__(
        self,
        raw_dir: Path,
        *,
        rate_limit_seconds: float = 1.0,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        user_agent: str = DEFAULT_UA,
        waf_length_fingerprints: tuple[int, ...] = (10648,),
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit_seconds
        self.max_retries = max_retries
        self.user_agent = user_agent
        self.waf_fingerprints = set(waf_length_fingerprints)
        self._last_request_at: dict[str, float] = {}

        self.client = httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "application/pdf;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
            # 部分交易所站点证书链不全，且我们只读公开文档，这里放宽校验。
            # 语料的真实性由后续的 magic bytes + 内容断言保证，不依赖 TLS。
            verify=False,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> Fetcher:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------
    # 限速
    # ---------------------------------------------------------------
    def _throttle(self, url: str) -> None:
        """按 host 限速，不同站点之间互不阻塞。"""
        host = urlparse(url).netloc
        last = self._last_request_at.get(host)
        if last is not None:
            wait = self.rate_limit - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
        self._last_request_at[host] = time.monotonic()

    # ---------------------------------------------------------------
    # 校验
    # ---------------------------------------------------------------
    def _validate(self, body: bytes, content_type: str, expect: str, url: str) -> None:
        """三重校验。任一不过就抛 ValidationError。"""
        if not body:
            raise ValidationError("响应体为空")

        # 1) WAF 拦截页长度指纹。SHFE/INE 返回 200 + 10648 字节验证码页。
        if len(body) in self.waf_fingerprints:
            raise ValidationError(
                f"命中 WAF 拦截页长度指纹 ({len(body)} 字节) —— "
                f"该站点的 HTML 走 JS-CAPTCHA 防火墙，请改用静态文件路径"
            )

        # 2) Content-Type。注意 sgx.com 会用 text/html 冒充 PDF。
        ct = (content_type or "").split(";")[0].strip().lower()
        allowed = CONTENT_TYPE_OK.get(expect)
        if allowed and ct and not any(ct.startswith(a) for a in allowed):
            raise ValidationError(
                f"Content-Type 不符: 期望 {expect}，实际 {ct!r}"
                + ("（SPA 壳冒充文件下载）" if expect != "html" and ct.startswith("text/html") else "")
            )

        # 3) magic bytes
        magics = MAGIC.get(expect)
        if magics:
            head = body[:1024]
            if not any(m in head for m in magics):
                raise ValidationError(
                    f"文件头不符: 期望 {expect}，实际前 8 字节 {body[:8]!r}"
                )

        # HTML 额外做一个下限检查 —— 太短几乎一定是错误页而非正文
        if expect == "html" and len(body) < 512:
            raise ValidationError(f"HTML 响应仅 {len(body)} 字节，疑似错误页")

    # ---------------------------------------------------------------
    # 下载
    # ---------------------------------------------------------------
    def _local_path(self, doc_id: str, url: str, expect: str) -> Path:
        """落盘路径。用 doc_id 而非 URL 派生的名字，便于人工核对。"""
        suffix = {"pdf": ".pdf", "html": ".html", "doc": ".doc", "docx": ".docx"}.get(
            expect, ""
        )
        if not suffix:
            name = unquote(urlparse(url).path).rsplit("/", 1)[-1]
            suffix = Path(name).suffix or ".bin"
        return self.raw_dir / f"{doc_id}{suffix}"

    def fetch(
        self,
        doc_id: str,
        url: str,
        *,
        expect: str = "html",
        venue: str = "",
        lang: str = "zh",
        force: bool = False,
        notes: str | None = None,
    ) -> FetchRecord:
        """下载单个 URL 并校验。已存在且非 force 时跳过（重跑不重下）。"""
        path = self._local_path(doc_id, url, expect)

        if path.exists() and not force:
            body = path.read_bytes()
            return FetchRecord(
                id=doc_id,
                url=url,
                venue=venue,
                lang=lang,
                expect=expect,
                status="skipped",
                bytes=len(body),
                sha256=hashlib.sha256(body).hexdigest(),
                local_path=str(path),
                notes="本地已存在，跳过下载",
            )

        last_error: str | None = None
        http_status: int | None = None
        content_type: str | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                self._throttle(url)
                resp = self.client.get(url)
                http_status = resp.status_code
                content_type = resp.headers.get("content-type")

                if resp.status_code >= 400:
                    last_error = f"HTTP {resp.status_code}"
                    # 4xx 基本不会因重试而改变，直接放弃（429 除外）
                    if resp.status_code != 429 and resp.status_code < 500:
                        break
                    time.sleep(2**attempt)
                    continue

                body = resp.content
                self._validate(body, content_type or "", expect, url)

                path.write_bytes(body)
                logger.info("已下载 %s (%s, %d 字节)", doc_id, expect, len(body))
                return FetchRecord(
                    id=doc_id,
                    url=url,
                    venue=venue,
                    lang=lang,
                    expect=expect,
                    status="ok",
                    http_status=resp.status_code,
                    content_type=content_type,
                    bytes=len(body),
                    sha256=hashlib.sha256(body).hexdigest(),
                    local_path=str(path),
                    notes=notes,
                )

            except ValidationError as e:
                # 内容校验失败重试没有意义，站点就是这么返回的
                last_error = f"内容校验失败: {e}"
                break
            except httpx.HTTPError as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt < self.max_retries:
                    time.sleep(2**attempt)

        logger.warning("下载失败 %s <%s>: %s", doc_id, url, last_error)
        return FetchRecord(
            id=doc_id,
            url=url,
            venue=venue,
            lang=lang,
            expect=expect,
            status="failed",
            http_status=http_status,
            content_type=content_type,
            error=last_error,
            notes=notes,
        )

    def probe(self, url: str, *, expect: str = "html") -> tuple[bool, str]:
        """只探测不落盘，供 --dry-run 使用。

        用 GET 而非 HEAD：多个交易所站点对 HEAD 的响应与 GET 不一致
        （有的直接 405，有的返回不带 Content-Length 的空响应）。
        """
        try:
            self._throttle(url)
            with self.client.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    return False, f"HTTP {resp.status_code}"
                ct = resp.headers.get("content-type", "")
                # 只读前 64KB 足够做 magic bytes 与 WAF 指纹判断
                head = b""
                for chunk in resp.iter_bytes(8192):
                    head += chunk
                    if len(head) >= 65536:
                        break
                declared_len = resp.headers.get("content-length")
                total = int(declared_len) if declared_len else len(head)

            if total in self.waf_fingerprints:
                return False, f"WAF 拦截页 ({total} 字节)"
            ct_main = ct.split(";")[0].strip().lower()
            allowed = CONTENT_TYPE_OK.get(expect)
            if allowed and ct_main and not any(ct_main.startswith(a) for a in allowed):
                return False, f"Content-Type {ct_main!r} != 期望 {expect}"
            magics = MAGIC.get(expect)
            if magics and not any(m in head[:1024] for m in magics):
                return False, f"文件头不符 {head[:8]!r}"
            return True, f"OK {ct_main} {total} 字节"
        except httpx.HTTPError as e:
            return False, f"{type(e).__name__}: {e}"
