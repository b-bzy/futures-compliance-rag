"""全流程共用的数据结构。

抓取 -> 解析 -> 分块 -> 检索 各阶段的产物都在这里定义，
所有中间文件都是 JSONL，字段名与这些 dataclass 一一对应。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum
from typing import Any


class EffectiveStatus(str, Enum):
    """条款时效性。检索默认只返回 ACTIVE。

    交易所文档里"已废止"通常体现在标题后缀（如"（已废止）"）或
    正文的时效性字段。至少 5 份 SSE 期权文档处于 REPEALED 状态，
    不过滤会用失效规则回答问题。
    """

    ACTIVE = "现行有效"
    REPEALED = "已废止"
    SUPERSEDED = "被修订"
    UNKNOWN = "未知"


class DocFormat(str, Enum):
    PDF = "pdf"
    HTML = "html"
    DOC = "doc"
    DOCX = "docx"


@dataclass
class FetchRecord:
    """一次下载的结果，逐行写入 data/raw/manifest.jsonl。"""

    id: str
    url: str
    venue: str
    lang: str
    expect: str
    status: str  # ok | skipped | failed
    http_status: int | None = None
    content_type: str | None = None
    bytes: int | None = None
    sha256: str | None = None
    local_path: str | None = None
    retrieval_date: str = field(default_factory=lambda: date.today().isoformat())
    error: str | None = None
    notes: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class RawDocument:
    """解析后的整篇文档（尚未分块）。写入 data/interim/documents.jsonl。"""

    doc_id: str
    title: str
    venue: str
    lang: str
    source_url: str
    retrieval_date: str
    fmt: str
    text: str
    # 表格单独存，条款表要按行拆成 (品种, 字段, 值) 原子事实
    tables: list[dict[str, Any]] = field(default_factory=list)
    doc_no: str | None = None  # 文号，如 上证发〔2023〕48号
    effective_status: str = EffectiveStatus.UNKNOWN.value
    effective_date: str | None = None
    page_count: int | None = None
    ocr_pages: list[int] = field(default_factory=list)
    parse_warnings: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class ParentChunk:
    """条款级父块 —— 中文按"第X条"切，合约条款表按行切。

    父块是给 LLM 看的上下文单位：语义完整、可直接引用。
    """

    parent_id: str
    doc_id: str
    text: str
    venue: str
    lang: str
    source_url: str
    retrieval_date: str
    doc_title: str
    doc_no: str | None = None
    clause_id: str | None = None  # "第十二条" / "2.5(d)" / "合约乘数"
    section_path: list[str] = field(default_factory=list)  # ["第三章 交易", "第一节 ..."]
    effective_status: str = EffectiveStatus.UNKNOWN.value
    effective_date: str | None = None
    chunk_type: str = "clause"  # clause | table_row | qa_pair | prose

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class ChildChunk:
    """语义切分出的子块 —— 这才是向量库里实际存的东西。

    子块用于精确召回，命中后通过 parent_id 回溯到完整条款作为上下文
    （即父子索引：小块检索、大块生成）。
    """

    child_id: str
    parent_id: str
    doc_id: str
    text: str
    # 冗余一份检索期要用的元数据，避免每次都回查 docstore
    venue: str = ""
    lang: str = ""
    source_url: str = ""
    doc_title: str = ""
    doc_no: str | None = None
    clause_id: str | None = None
    effective_status: str = EffectiveStatus.UNKNOWN.value
    chunk_type: str = "clause"
    position: int = 0  # 在父块内的序号

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    def to_metadata(self) -> dict[str, Any]:
        """Chroma metadata 只接受标量，这里做一次扁平化。"""
        return {
            "parent_id": self.parent_id,
            "doc_id": self.doc_id,
            "venue": self.venue,
            "lang": self.lang,
            "source_url": self.source_url,
            "doc_title": self.doc_title,
            "doc_no": self.doc_no or "",
            "clause_id": self.clause_id or "",
            "effective_status": self.effective_status,
            "chunk_type": self.chunk_type,
            "position": self.position,
        }


@dataclass
class RetrievalHit:
    """单路召回或融合后的一条命中。"""

    child_id: str
    text: str
    score: float
    source: str  # bm25 | dense | keyword | fused | rerank
    metadata: dict[str, Any] = field(default_factory=dict)
    # 各路原始分数，用于 score-aware RRF 与调试可视化
    component_scores: dict[str, float] = field(default_factory=dict)
    component_ranks: dict[str, int] = field(default_factory=dict)
    parent_text: str | None = None

    @property
    def citation(self) -> str:
        """人可读的引用串，UI 与生成的答案里都用它。"""
        m = self.metadata
        parts = [p for p in (m.get("doc_title"), m.get("doc_no"), m.get("clause_id")) if p]
        return " · ".join(parts) if parts else m.get("source_url", self.child_id)


@dataclass
class QAPair:
    """QA 数据集条目。tier=1 为人工撰写金标，tier=2 为自动挖掘。"""

    qa_id: str
    question: str
    answer: str
    tier: int
    source_url: str
    doc_id: str | None = None
    parent_id: str | None = None
    clause_id: str | None = None
    venue: str = ""
    lang: str = "zh"
    mining_method: str = ""  # human | table_template | clause_llm | cross_venue
    # span-grounding: 自动挖掘的答案必须是源条款的连续子串
    span_verified: bool = False
    human_verified: bool | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def stable_id(*parts: str, prefix: str = "", length: int = 16) -> str:
    """由内容派生的确定性 ID —— 重跑流水线不会产生新 ID。"""
    h = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:length]
    return f"{prefix}{h}" if prefix else h
