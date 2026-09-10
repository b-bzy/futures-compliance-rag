"""向量化。

⚠️ 关于 bge-m3 的一个容易被忽略的坑：
用 sentence-transformers 加载 BAAI/bge-m3 拿到的**只有 dense 向量**。
仓库里确实带了 `colbert_linear.pt` 和 `sparse_linear.pt`，但
`modules.json` 只声明了 Transformer -> Pooling -> Normalize 三层，
sentence-transformers 永远不会去加载那两个头。也就是说，你以为自己在跑
dense+sparse+colbert 三模混合检索，实际上跑的是纯 dense。

要拿到真正的三模，必须走 `FlagEmbedding.BGEM3FlagModel`。本模块默认用
FlagEmbedding 后端；sentence-transformers 后端保留下来是为了消融对比，
且在配置里如实标注它是 dense-only。

实测（本机 M4 / 16GB / MPS，transformers 5.15）：
    模型加载约 15s（权重已缓存），2 条短文本编码 <0.1s，
    dense 维度 1024，sparse 返回 token_id -> 权重 的稀疏字典。
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """统一的向量化接口。

    encode_documents / encode_queries 分开是因为部分模型对查询和文档
    使用不同的前缀指令；bge-m3 不需要前缀，但保持接口一致便于替换模型。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        backend: Literal["flagembedding", "sentence-transformers"] = "flagembedding",
        device: str = "mps",
        batch_size: int = 8,
        max_length: int = 1024,
        use_fp16: bool = False,
        return_dense: bool = True,
        return_sparse: bool = True,
        return_colbert: bool = False,
    ) -> None:
        self.model_name = model_name
        self.backend = backend
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.return_dense = return_dense
        self.return_sparse = return_sparse and backend == "flagembedding"
        self.return_colbert = return_colbert and backend == "flagembedding"
        self._model: Any = None

        if backend == "sentence-transformers" and (return_sparse or return_colbert):
            logger.warning(
                "sentence-transformers 后端只能产出 dense 向量 —— "
                "bge-m3 的 sparse/colbert 头不会被加载。已自动关闭稀疏与 colbert 输出。"
            )

    # ---------------------------------------------------------------
    def _load(self) -> Any:
        """惰性加载。16GB 机器上模型常驻代价高，用到才加载。"""
        if self._model is not None:
            return self._model

        if self.backend == "flagembedding":
            from FlagEmbedding import BGEM3FlagModel

            logger.info("加载 %s (FlagEmbedding, device=%s)…", self.model_name, self.device)
            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.use_fp16,
                devices=self.device,
            )
        else:
            from sentence_transformers import SentenceTransformer

            logger.info(
                "加载 %s (sentence-transformers, device=%s) —— 仅 dense",
                self.model_name,
                self.device,
            )
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def unload(self) -> None:
        """释放模型。qwen3 + bge-m3 + reranker 在 16GB 上无法同时常驻。"""
        self._model = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # ---------------------------------------------------------------
    def encode(
        self,
        texts: Sequence[str],
        *,
        show_progress: bool = False,
    ) -> dict[str, Any]:
        """编码文本，返回 {dense, sparse}。sparse 为 token_id->weight 字典列表。"""
        texts = [t if t.strip() else " " for t in texts]
        model = self._load()

        if self.backend == "flagembedding":
            out = model.encode(
                list(texts),
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=self.return_dense,
                return_sparse=self.return_sparse,
                return_colbert_vecs=self.return_colbert,
            )
            result: dict[str, Any] = {}
            if self.return_dense:
                result["dense"] = np.asarray(out["dense_vecs"], dtype=np.float32)
            if self.return_sparse:
                result["sparse"] = out["lexical_weights"]
            if self.return_colbert:
                result["colbert"] = out["colbert_vecs"]
            return result

        vecs = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return {"dense": np.asarray(vecs, dtype=np.float32)}

    def encode_dense(self, texts: Sequence[str], **kw) -> np.ndarray:
        """只要 dense 向量 —— 语义切分和向量库写入都走这个。"""
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        return self.encode(texts, **kw)["dense"]

    def encode_queries(self, queries: Sequence[str]) -> dict[str, Any]:
        """编码查询。bge-m3 查询与文档同构，无需额外前缀。"""
        return self.encode(queries)

    # ---------------------------------------------------------------
    @staticmethod
    def sparse_dot(a: dict[str, float], b: dict[str, float]) -> float:
        """两个稀疏权重字典的点积 —— bge-m3 的 lexical matching 分数。"""
        if not a or not b:
            return 0.0
        # 遍历较短的一侧
        if len(a) > len(b):
            a, b = b, a
        return float(sum(w * b.get(k, 0.0) for k, w in a.items()))


class LangChainEmbeddings:
    """把 Embedder 适配成 langchain 的 Embeddings 接口，供 Chroma 使用。"""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.encode_dense(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.encode_dense([text])[0].tolist()


_default_embedder: Embedder | None = None


def get_embedder(cfg: dict | None = None) -> Embedder:
    """按配置构造全局 Embedder（进程内单例，避免重复加载 2.27GB 权重）。"""
    global _default_embedder
    if _default_embedder is not None:
        return _default_embedder

    if cfg is None:
        from ..config import load_config

        cfg = load_config()

    e = cfg.get_path("models.embedding", {}) if hasattr(cfg, "get_path") else cfg
    _default_embedder = Embedder(
        e.get("name", "BAAI/bge-m3"),
        backend=e.get("backend", "flagembedding"),
        device=e.get("device", "mps"),
        batch_size=e.get("batch_size", 8),
        max_length=e.get("max_length", 1024),
        use_fp16=e.get("use_fp16", False),
        return_dense=e.get("return_dense", True),
        return_sparse=e.get("return_sparse", True),
        return_colbert=e.get("return_colbert", False),
    )
    return _default_embedder
