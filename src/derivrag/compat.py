"""第三方库兼容垫片。

目前只有一处：ragas 0.4.3 无法在 langchain-community 0.4.x 上导入。

原因（实测）：
    ragas/llms/base.py 第 12-13 行无条件执行
        from langchain_community.chat_models.vertexai import ChatVertexAI
        from langchain_community.llms import VertexAI
    但 langchain-community 0.4.x 已经把 Vertex AI 集成拆分到独立包
    langchain-google-vertexai，`chat_models.vertexai` 子模块不复存在。
    ragas 的 metadata 里 langchain-community 是完全不带版本约束的，
    属于 ragas 自身的打包缺陷。

为什么不降级 langchain-community：
    langchain-community 0.3.x pin langchain-core<0.4，而本项目用的是
    langchain-core 1.x。降级会连锁摧毁整个 langchain 1.x 检索层。

垫片做什么：
    在 import ragas 之前，往 sys.modules 里注册两个只提供
    ChatVertexAI / VertexAI 名字的占位模块。ragas 只把这两个类用于
    isinstance 判断和 llm_factory 的分支派发；本项目的 judge 走的是
    OpenAI 兼容协议（指向本地 ollama），永远不会命中 Vertex 分支，
    所以占位类不被实例化，行为完全等价。

    如果哪天真的要用 Vertex AI 做 judge，装 langchain-google-vertexai，
    本垫片会自动改为从那里转发真实实现。
"""

from __future__ import annotations

import logging
import sys
import types

logger = logging.getLogger(__name__)

_PATCHED = False


def _resolve_vertex_classes() -> tuple[type, type]:
    """优先返回真实实现，拿不到就返回占位类。"""
    try:
        from langchain_google_vertexai import ChatVertexAI, VertexAI  # type: ignore

        logger.debug("垫片: 使用 langchain-google-vertexai 的真实实现")
        return ChatVertexAI, VertexAI
    except Exception:  # noqa: BLE001
        pass

    class _UnavailableVertex:
        """占位类。只用于 isinstance 判断；被实例化说明配置走错了分支。"""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError(
                "本项目未安装 Vertex AI 支持。若确实要用 Vertex 作为 ragas judge，"
                "请 pip install langchain-google-vertexai；"
                "默认配置的 judge 是本地 ollama（OpenAI 兼容协议）。"
            )

    class ChatVertexAI(_UnavailableVertex):  # type: ignore[no-redef]
        pass

    class VertexAI(_UnavailableVertex):  # type: ignore[no-redef]
        pass

    return ChatVertexAI, VertexAI


def patch_ragas_vertexai() -> None:
    """注册占位模块。必须在 `import ragas` 之前调用。幂等。"""
    global _PATCHED
    if _PATCHED:
        return

    try:
        import langchain_community.chat_models as _cm  # noqa: F401
    except ImportError:
        # 连 langchain_community 都没有，说明没打算用 ragas，无需垫片
        _PATCHED = True
        return

    chat_vertex, llm_vertex = _resolve_vertex_classes()

    mod_name = "langchain_community.chat_models.vertexai"
    if mod_name not in sys.modules:
        try:
            __import__(mod_name)
        except ImportError:
            stub = types.ModuleType(mod_name)
            stub.ChatVertexAI = chat_vertex  # type: ignore[attr-defined]
            sys.modules[mod_name] = stub
            # 同时挂到父包上，`from ...chat_models import vertexai` 也能work
            import langchain_community.chat_models as cm

            cm.vertexai = stub  # type: ignore[attr-defined]
            logger.debug("垫片: 已注册 %s", mod_name)

    # ragas 还要 `from langchain_community.llms import VertexAI`
    try:
        import langchain_community.llms as llms

        if not hasattr(llms, "VertexAI"):
            llms.VertexAI = llm_vertex  # type: ignore[attr-defined]
            logger.debug("垫片: 已补 langchain_community.llms.VertexAI")
    except ImportError:
        pass

    _PATCHED = True


def import_ragas():
    """打好垫片后导入 ragas，返回模块本身。"""
    patch_ragas_vertexai()
    import ragas

    return ragas
