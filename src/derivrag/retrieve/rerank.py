"""交叉编码器重排。

⚠️ 简历/网上常见的 "BGE-Reranker-M3" 这个模型 **不存在**。
HuggingFace 上真实存在的是 `BAAI/bge-reranker-v2-m3`（apache-2.0，568M 参数，
XLM-RoBERTa-large 底座，其 config 里 _name_or_path 正是 BAAI/bge-m3）。
本项目已在 configs/config.yaml 里用正确 ID。

两阶段召回的分工：
    粗排（向量+BM25+关键词 → RRF）负责"别漏"，取 top-50；
    精排（cross-encoder）负责"别错"，从 50 条里挑 top-5。
双塔模型把 query 和 doc 分别编码，交互信息全靠一次点积；cross-encoder
让两者在每一层做完整 attention，代价是必须逐对前向，所以只能用在
小候选集上 —— 这就是为什么要分两阶段，而不是直接用 cross-encoder 检索。
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from ..schema import RetrievalHit

logger = logging.getLogger(__name__)


class Reranker:
    """bge-reranker 交叉编码器。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        device: str = "mps",
        batch_size: int = 4,
        max_length: int = 512,
        use_onnx: bool = False,
        onnx_model: str = "BAAI/bge-reranker-base",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_onnx = use_onnx
        self.onnx_model = onnx_model
        self._model: Any = None
        self._tokenizer: Any = None

    # ---------------------------------------------------------------
    def _load(self) -> None:
        if self._model is not None:
            return

        if self.use_onnx:
            # bge-reranker-base 是唯一 ONNX 自包含（不需要外部 .onnx_data）的
            # BGE reranker，配合 CoreMLExecutionProvider 在 Mac 上延迟最低。
            self._load_onnx()
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("加载重排模型 %s (device=%s)…", self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        model.eval()
        if self.device == "mps" and torch.backends.mps.is_available():
            model = model.to("mps")
        elif self.device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
        else:
            self.device = "cpu"
        self._model = model

    def _load_onnx(self) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        logger.info("加载 ONNX 重排模型 %s…", self.onnx_model)
        self._tokenizer = AutoTokenizer.from_pretrained(self.onnx_model)
        path = hf_hub_download(self.onnx_model, "onnx/model.onnx")
        providers = [
            p
            for p in ("CoreMLExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()
        ]
        self._model = ort.InferenceSession(path, providers=providers)
        logger.info("ONNX providers: %s", self._model.get_providers())

    def unload(self) -> None:
        """释放显存/内存。16GB 上 qwen3 + bge-m3 + reranker 不能同时常驻。"""
        self._model = None
        self._tokenizer = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # ---------------------------------------------------------------
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """给 (query, doc) 打分。分数越大越相关（未归一化的 logit）。"""
        if not documents:
            return []
        self._load()

        if self.use_onnx:
            return self._score_onnx(query, documents)

        import torch

        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i : i + self.batch_size]
                inputs = self._tokenizer(
                    [query] * len(batch),
                    list(batch),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self._model(**inputs).logits
                # bge-reranker 是单 logit 回归头
                scores.extend(logits.view(-1).float().cpu().tolist())
        return scores

    def _score_onnx(self, query: str, documents: Sequence[str]) -> list[float]:
        scores: list[float] = []
        input_names = {i.name for i in self._model.get_inputs()}
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            enc = self._tokenizer(
                [query] * len(batch),
                list(batch),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            feed = {k: v for k, v in enc.items() if k in input_names}
            out = self._model.run(None, feed)[0]
            scores.extend(np.asarray(out).reshape(-1).tolist())
        return scores

    # ---------------------------------------------------------------
    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        top_k: int = 5,
        use_parent_text: bool = False,
    ) -> list[RetrievalHit]:
        """对候选重排，返回 top_k。

        use_parent_text=True 时用父条款全文参与打分 —— 更准（上下文完整），
        但更慢且容易超 max_length，默认关闭。
        """
        if not hits:
            return []
        docs = [
            (h.parent_text if use_parent_text and h.parent_text else h.text) for h in hits
        ]
        scores = self.score(query, docs)

        for h, s in zip(hits, scores):
            h.component_scores["rerank"] = float(s)
            # 保留融合分，便于消融时对比重排前后的名次变化
            h.component_scores.setdefault("fused", h.score)
            h.score = float(s)
            h.source = "rerank"

        return sorted(hits, key=lambda h: h.score, reverse=True)[:top_k]


_default_reranker: Reranker | None = None


def get_reranker(cfg: dict | None = None) -> Reranker:
    """进程内单例，避免重复加载 2.27GB 权重。"""
    global _default_reranker
    if _default_reranker is not None:
        return _default_reranker

    if cfg is None:
        from ..config import load_config

        cfg = load_config()
    r = cfg.get_path("models.reranker", {}) if hasattr(cfg, "get_path") else cfg
    _default_reranker = Reranker(
        r.get("name", "BAAI/bge-reranker-v2-m3"),
        device=r.get("device", "mps"),
        batch_size=r.get("batch_size", 4),
        max_length=r.get("max_length", 512),
        use_onnx=r.get("use_onnx", False),
        onnx_model=r.get("onnx_fallback", "BAAI/bge-reranker-base"),
    )
    return _default_reranker
