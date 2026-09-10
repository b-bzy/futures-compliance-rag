"""条款级父块切分。

父块是"给 LLM 看的上下文单位"，必须语义完整且可直接引用。中文交易所
规则天然按"第X条"组织，这是比任何固定长度切分都好的边界 —— 一条就是
一个完整的规范性陈述，引用时能精确到"《XX办法》第十二条"。

三种父块形态：

  clause     法规条文，按"第X条"切（实测每份规则 40-80 条）
  table_row  合约条款表的一行，即一个 (品种, 字段, 值) 原子事实
  qa_pair    问答型文档的一对问答
  prose      切不出结构时的兜底，按标题层级 + 长度切
"""

from __future__ import annotations

import logging
import re

from ..schema import ParentChunk, RawDocument, stable_id

logger = logging.getLogger(__name__)

# "第一条" "第十二条" "第一百零三条" "第3条"
CLAUSE_RE = re.compile(r"第[一二三四五六七八九十百千零〇两\d]{1,10}条(?![款项目])")
# "第一章 总则" / "第二节 交易"
SECTION_RE = re.compile(
    r"^\s*(第[一二三四五六七八九十百千零〇\d]{1,10}[章节编部分])\s*(.{0,40})$", re.M
)
# 英文规则编号: "7.04.5 Exercise" / "Rule 2.2.8"
EN_RULE_RE = re.compile(r"^\s*(?:Rule\s+)?(\d+(?:\.\d+){1,4}[A-Z]?)\s+(?=[A-Z(])", re.M)

_CN_DIGITS = {c: i for i, c in enumerate("零一二三四五六七八九")}


def _cn_to_int(s: str) -> int | None:
    """把"一百零三"这类中文数字转成 int，用于条款排序与去重。"""
    s = s.strip()
    if s.isdigit():
        return int(s)
    if not s:
        return None
    total, section, number = 0, 0, 0
    unit_map = {"十": 10, "百": 100, "千": 1000}
    for ch in s:
        if ch in _CN_DIGITS:
            number = _CN_DIGITS[ch]
        elif ch in unit_map:
            unit = unit_map[ch]
            section += (number or 1) * unit
            number = 0
        elif ch == "两":
            number = 2
        else:
            return None
    return total + section + number


def _section_path_at(text: str, pos: int) -> list[str]:
    """给定字符位置，回溯出它所处的章/节层级。"""
    path: list[str] = []
    for m in SECTION_RE.finditer(text, 0, pos):
        label = f"{m.group(1)} {m.group(2)}".strip()
        marker = m.group(1)
        level = next(
            (i for i, k in enumerate(["编", "部分", "章", "节"]) if k in marker), 2
        )
        path = path[:level] + [label]
    return path


def split_clauses(doc: RawDocument) -> list[ParentChunk]:
    """按"第X条"切分中文法规。切不出来时返回空列表，由调用方走兜底。"""
    text = doc.text
    matches = list(CLAUSE_RE.finditer(text))
    if len(matches) < 3:  # 少于 3 条基本说明这不是条文式文档
        return []

    chunks: list[ParentChunk] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if len(body) < 10:
            continue

        clause_label = m.group(0)
        # 目录页里也会出现一堆"第X条"，但后面只跟标题和页码，不是正文。
        # 判据不能只用长度 —— 交易所规则里大量存在 20 字上下的短条款
        # （"第三条 本细则未规定的，按照相关业务规则执行。"），单纯按长度过滤
        # 会把它们连同目录一起丢掉。真正的区分点是**有没有句末标点**：
        # 正文条款一定以句号/分号收尾，目录条目不会。
        if len(body) < 25 and not body.rstrip().endswith(("。", "；", "！", "？", ".", ";")):
            continue

        chunks.append(
            ParentChunk(
                parent_id=stable_id(doc.doc_id, clause_label, str(i), prefix="p_"),
                doc_id=doc.doc_id,
                text=body,
                venue=doc.venue,
                lang=doc.lang,
                source_url=doc.source_url,
                retrieval_date=doc.retrieval_date,
                doc_title=doc.title,
                doc_no=doc.doc_no,
                clause_id=clause_label,
                section_path=_section_path_at(text, start),
                effective_status=doc.effective_status,
                effective_date=doc.effective_date,
                chunk_type="clause",
            )
        )
    return chunks


def split_en_rules(doc: RawDocument) -> list[ParentChunk]:
    """按编号规则切分英文规则手册（SGX/HKEX 的 7.04.5 这类）。"""
    text = doc.text
    matches = list(EN_RULE_RE.finditer(text))
    if len(matches) < 3:
        return []

    chunks: list[ParentChunk] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if len(body) < 25:
            continue
        chunks.append(
            ParentChunk(
                parent_id=stable_id(doc.doc_id, m.group(1), str(i), prefix="p_"),
                doc_id=doc.doc_id,
                text=body,
                venue=doc.venue,
                lang=doc.lang,
                source_url=doc.source_url,
                retrieval_date=doc.retrieval_date,
                doc_title=doc.title,
                doc_no=doc.doc_no,
                clause_id=m.group(1),
                section_path=[],
                effective_status=doc.effective_status,
                effective_date=doc.effective_date,
                chunk_type="clause",
            )
        )
    return chunks


def split_table_rows(doc: RawDocument) -> list[ParentChunk]:
    """把合约条款表的每一行切成一个原子事实父块。

    这是本项目最高价值的结构化资产：每行就是一个 (字段, 值) 对，
    既是最精确的检索单元（问"合约乘数是多少"应该只命中这一行，
    而不是整张表），也是零幻觉 QA 模板的直接来源。
    """
    chunks: list[ParentChunk] = []
    # 用文档标题里的品种名做主语，例如"上证50ETF期权合约基本条款" -> "上证50ETF期权"
    subject = re.sub(r"合约基本条款|合约条款|基本条款|附件", "", doc.title).strip(" -—·|")

    for t_idx, table in enumerate(doc.tables):
        rows = table.get("rows", [])
        # 只处理两列的"字段/值"表；多列表格（如费率表）留给正文切分
        if table.get("n_cols") != 2:
            continue
        for r_idx, row in enumerate(rows):
            cells = [c for c in row if c and c.strip()]
            if len(cells) != 2:
                continue
            field, value = cells[0].strip(), cells[1].strip()
            if not field or not value or len(field) > 30:
                continue
            body = f"{subject}的{field}：{value}" if subject else f"{field}：{value}"
            chunks.append(
                ParentChunk(
                    parent_id=stable_id(doc.doc_id, str(t_idx), str(r_idx), prefix="p_"),
                    doc_id=doc.doc_id,
                    text=body,
                    venue=doc.venue,
                    lang=doc.lang,
                    source_url=doc.source_url,
                    retrieval_date=doc.retrieval_date,
                    doc_title=doc.title,
                    doc_no=doc.doc_no,
                    clause_id=field,
                    section_path=[subject] if subject else [],
                    effective_status=doc.effective_status,
                    effective_date=doc.effective_date,
                    chunk_type="table_row",
                )
            )
    return chunks


def split_qa(doc: RawDocument) -> list[ParentChunk]:
    """把问答型文档切成一对一问答父块。

    支持两种版式：
      A) 一篇文档含多组编号问答（SSE 的 FAQ / 熔断机制问答）
      B) 一篇文档就是一组问答，**标题即问题、正文即答案**
         （CFFEX 常见问答，每条问答独占一个页面，共 30+ 条）
    """
    from ..parse.html import extract_qa_pairs

    pairs = extract_qa_pairs(doc.text)

    # 版式 B：标题本身是一个问句
    if not pairs and _title_is_question(doc.title):
        answer = doc.text.strip()
        if 10 <= len(answer) <= 3000:
            pairs = [{"question": doc.title.strip(), "answer": answer, "index": "1"}]

    if not pairs:
        return []

    chunks: list[ParentChunk] = []
    for p in pairs:
        body = f"问：{p['question']}\n答：{p['answer']}"
        chunks.append(
            ParentChunk(
                parent_id=stable_id(doc.doc_id, p["index"], p["question"][:40], prefix="p_"),
                doc_id=doc.doc_id,
                text=body,
                venue=doc.venue,
                lang=doc.lang,
                source_url=doc.source_url,
                retrieval_date=doc.retrieval_date,
                doc_title=doc.title,
                doc_no=doc.doc_no,
                clause_id=f"问答{p['index']}",
                section_path=[],
                effective_status=doc.effective_status,
                effective_date=doc.effective_date,
                chunk_type="qa_pair",
            )
        )
    return chunks


def _title_is_question(title: str) -> bool:
    """标题是否是一个问句。

    HTML 的 <title> 常带站点后缀（"…吗？| 中国金融期货交易所"），
    所以先切掉后缀再判断结尾标点；同时用疑问词兜底，覆盖
    "如何办理开户手续" 这类不带问号的标题。
    """
    if not title:
        return False
    head = re.split(r"[|｜]", title)[0].strip()
    if not head or len(head) < 6:
        return False
    if head.endswith(("？", "?")):
        return True
    return any(
        w in head
        for w in ("如何", "怎么", "怎样", "是否", "能否", "可否", "什么", "哪些", "为何", "为什么")
    )


def split_prose(doc: RawDocument, *, max_chars: int = 4000, min_chars: int = 30) -> list[ParentChunk]:
    """兜底切分：先按章节标题切，再按长度硬切。"""
    text = doc.text
    if not text.strip():
        return []

    # 先按章节切
    sec_matches = list(SECTION_RE.finditer(text))
    if sec_matches:
        segments = []
        for i, m in enumerate(sec_matches):
            start = m.start()
            end = sec_matches[i + 1].start() if i + 1 < len(sec_matches) else len(text)
            segments.append((f"{m.group(1)} {m.group(2)}".strip(), text[start:end]))
        if sec_matches[0].start() > min_chars:
            segments.insert(0, ("", text[: sec_matches[0].start()]))
    else:
        segments = [("", text)]

    chunks: list[ParentChunk] = []
    for label, seg in segments:
        # 段落过长再按空行硬切
        pieces = [seg] if len(seg) <= max_chars else _split_by_length(seg, max_chars)
        for j, piece in enumerate(pieces):
            piece = piece.strip()
            if len(piece) < min_chars:
                continue
            chunks.append(
                ParentChunk(
                    parent_id=stable_id(doc.doc_id, label, str(j), piece[:60], prefix="p_"),
                    doc_id=doc.doc_id,
                    text=piece,
                    venue=doc.venue,
                    lang=doc.lang,
                    source_url=doc.source_url,
                    retrieval_date=doc.retrieval_date,
                    doc_title=doc.title,
                    doc_no=doc.doc_no,
                    clause_id=label or None,
                    section_path=[label] if label else [],
                    effective_status=doc.effective_status,
                    effective_date=doc.effective_date,
                    chunk_type="prose",
                )
            )
    return chunks


def _split_by_length(text: str, max_chars: int) -> list[str]:
    """按段落边界累加到长度上限，不在句子中间切断。"""
    paras = re.split(r"\n\s*\n", text)
    out, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                out.append(buf)
            # 单段就超长时按句号再切
            if len(p) > max_chars:
                sentences = re.split(r"(?<=[。；！？!?;])", p)
                cur = ""
                for s in sentences:
                    if len(cur) + len(s) <= max_chars:
                        cur += s
                    else:
                        if cur:
                            out.append(cur)
                        cur = s
                buf = cur
            else:
                buf = p
    if buf:
        out.append(buf)
    return out


def split_document(doc: RawDocument, *, max_chars: int = 4000, min_chars: int = 30) -> list[ParentChunk]:
    """按文档形态自动选择切分策略。

    顺序有讲究：条款表要单独抽出来（它们是最精确的检索单元），
    然后才轮到条文/问答/兜底处理正文部分。
    """
    chunks: list[ParentChunk] = []

    # 1) 条款表 -> 原子事实
    chunks.extend(split_table_rows(doc))

    # 2) 正文：条文 -> 问答 -> 英文规则 -> 兜底
    body_chunks = (
        split_clauses(doc)
        or split_qa(doc)
        or split_en_rules(doc)
        or split_prose(doc, max_chars=max_chars, min_chars=min_chars)
    )
    chunks.extend(body_chunks)

    if not chunks:
        logger.warning("文档 %s (%s) 未能切出任何父块", doc.doc_id, doc.title[:40])
    return chunks
