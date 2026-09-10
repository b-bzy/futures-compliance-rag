"""Chroma 向量库封装。

chromadb 1.x 与网上大多数教程不同的两点（实测）：
  1. **没有 .persist() 了** —— PersistentClient 的写入是自动落盘的，
     调用 .persist() 会 AttributeError。
  2. collection 名字必须 3-512 字符且只含 [a-zA-Z0-9._-]，
     1-2 个字符会抛 InvalidArgumentError。

这里直接用 chromadb 原生客户端而不是 langchain-chroma，原因是我们要
自己控制 dense 向量的产生（bge-m3 走 FlagEmbedding 而非 ST），
并且需要拿到原始距离分数做 score-aware RRF —— langchain 的
`similarity_search` 会把分数丢掉或做不透明的转换。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..schema import ChildChunk

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "derivrag_clauses"


class VectorStore:
    """Chroma 持久化向量库。"""

    def __init__(
        self,
        persist_dir: str | Path,
        *,
        collection_name: str = DEFAULT_COLLECTION,
        distance: str = "cosine",
    ) -> None:
        import chromadb

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        # hnsw:space 只能在创建时指定，之后不可改
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance},
        )
        logger.info(
            "Chroma 就绪: %s (collection=%s, 现有 %d 条)",
            self.persist_dir,
            collection_name,
            self.collection.count(),
        )

    # ---------------------------------------------------------------
    def add(
        self,
        chunks: Sequence[ChildChunk],
        vectors: np.ndarray,
        *,
        batch_size: int = 512,
    ) -> int:
        """写入子块及其向量。"""
        if len(chunks) != len(vectors):
            raise ValueError(f"块数 {len(chunks)} 与向量数 {len(vectors)} 不一致")

        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self.collection.upsert(
                ids=[c.child_id for c in batch],
                embeddings=[v.tolist() for v in vectors[i : i + batch_size]],
                documents=[c.text for c in batch],
                metadatas=[c.to_metadata() for c in batch],
            )
            total += len(batch)
            if total % 2048 == 0:
                logger.info("已写入 %d/%d", total, len(chunks))
        return total

    # ---------------------------------------------------------------
    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 50,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """向量检索。返回带**相似度**（非距离）的结果。

        Chroma 的 cosine space 返回的是距离 (1 - cos)，这里统一转成
        相似度，让所有召回通路的分数都是"越大越好"，融合时才好归一化。
        """
        res = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        out: list[dict[str, Any]] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            out.append(
                {
                    "child_id": cid,
                    "text": doc,
                    "metadata": dict(meta or {}),
                    "score": 1.0 - float(dist),
                }
            )
        return out

    def get_by_ids(self, ids: Sequence[str]) -> list[dict[str, Any]]:
        """按 id 批量取回，用于关键词通路命中后补全内容。"""
        if not ids:
            return []
        res = self.collection.get(ids=list(ids), include=["documents", "metadatas"])
        return [
            {"child_id": i, "text": d, "metadata": dict(m or {})}
            for i, d, m in zip(
                res.get("ids", []), res.get("documents", []), res.get("metadatas", [])
            )
        ]

    def query_metadata(
        self, where: dict[str, Any], *, limit: int = 50
    ) -> list[dict[str, Any]]:
        """纯元数据过滤检索 —— 关键词通路走这条（如按 clause_id 精确匹配）。"""
        res = self.collection.get(
            where=where, limit=limit, include=["documents", "metadatas"]
        )
        return [
            {"child_id": i, "text": d, "metadata": dict(m or {}), "score": 1.0}
            for i, d, m in zip(
                res.get("ids", []), res.get("documents", []), res.get("metadatas", [])
            )
        ]

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """删除并重建 collection —— 重建索引时用。"""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001 - 不存在时忽略
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        logger.info("collection %s 已重置", self.collection_name)
