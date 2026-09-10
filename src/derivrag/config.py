"""配置加载。

支持 ${VAR} 与 ${VAR:default} 形式的环境变量插值，这样 API key 之类的
敏感值不必写进 yaml。
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def repo_root() -> Path:
    """仓库根目录 —— 本文件位于 <root>/src/derivrag/config.py。"""
    return Path(__file__).resolve().parents[2]


def _interpolate(value: Any) -> Any:
    """递归地把 ${VAR} / ${VAR:default} 替换成环境变量值。

    未设置且无默认值时返回空字符串，由调用方决定如何处理（例如
    LLMProvider 会在 api_key 为空时降级到 ollama 并 warn）。
    """
    if isinstance(value, str):

        def _sub(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            return os.environ.get(name, default if default is not None else "")

        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


class Config(dict):
    """字典的薄封装，额外支持点号路径取值。"""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """cfg.get_path("models.embedding.name") 等价于逐级 get。"""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def resolve(self, dotted: str, default: Any = None) -> Path:
        """取一个配置项并解析成绝对路径（相对仓库根目录）。"""
        raw = self.get_path(dotted, default)
        if raw is None:
            raise KeyError(f"配置项不存在且无默认值: {dotted}")
        p = Path(raw)
        return p if p.is_absolute() else repo_root() / p


@lru_cache(maxsize=8)
def load_config(path: str | Path | None = None) -> Config:
    """加载主配置。默认读 <root>/configs/config.yaml。"""
    cfg_path = Path(path) if path else repo_root() / "configs" / "config.yaml"
    if not cfg_path.is_absolute():
        cfg_path = repo_root() / cfg_path
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(_interpolate(raw))


@lru_cache(maxsize=8)
def load_sources(path: str | Path | None = None) -> dict:
    """加载语料源清单 configs/sources.yaml。"""
    src_path = Path(path) if path else repo_root() / "configs" / "sources.yaml"
    if not src_path.is_absolute():
        src_path = repo_root() / src_path
    with open(src_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
