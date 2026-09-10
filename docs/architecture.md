# 系统架构

## 全链路

```
                        ┌──────────────── 离线构建 ────────────────┐

  交易所官网 ──▶ Fetcher ──▶ Parser ──▶ Chunker ──▶ Embedder ──▶ Chroma
   10 家        内容校验     版面/表格   父子索引     bge-m3       向量库
                            OCR 兜底   语义切分                    │
                                          │                        │
                                          ├──▶ parents.jsonl ──────┤
                                          └──▶ BM25 索引 ──────────┤
                                                                   │
                        └──────────────────────────────────────────┘
                                                                   │
  ┌──────────────── 在线检索 ────────────────────────────────────┐ │
  │                                                              │ │
  │  用户问题                                                     │ │
  │     │                                                        │ │
  │     ├─▶ 查询重写（多轮指代消解）                               │ │
  │     │      "那深证100ETF呢？" → "深证100ETF期权的合约单位是多少？" │ │
  │     │                                                        │ │
  │     ├─▶ HyDE（生成假设条款，仅用于检索）                        │ │
  │     │                                                        │ │
  │     ├───────┬──────────────┬──────────────┐                  │ │
  │     ▼       ▼              ▼              ▼                  │ │
  │   BM25    稠密向量       关键词          （原始查询 +           │ │
  │  jieba分词  bge-m3      条号/文号/代码      HyDE 文档）  ◀───────┘ │
  │  top-50    top-50        top-20                              │
  │     └───────┴──────────────┘                                 │
  │                  │                                           │
  │                  ▼                                           │
  │        自研 score-aware RRF 融合                              │
  │        (1-α)·Σw/(k+rank) + α·Σw·minmax(score)                │
  │                  │                                           │
  │                  ▼                                           │
  │        按 parent_id 去重  →  父块回溯（子块命中→完整条款）      │
  │                  │                                           │
  │                  ▼                                           │
  │        bge-reranker-v2-m3 交叉编码器精排  top-50 → top-5      │
  │                  │                                           │
  │                  ▼                                           │
  │        强引用约束生成（Qwen3 / DeepSeek）                      │
  │        只依据条款 · 标注 [n] · 无据则拒答                       │
  │                  │                                           │
  └──────────────────┼───────────────────────────────────────────┘
                     ▼
              答案 + 引用卡片（文号/条号/生效状态/原文/源链接）
```

## 为什么是父子索引

检索单元和生成单元的最优粒度不一样：

- **检索**要小块。问"合约乘数是多少"，理想命中是"合约乘数：每点人民币100元"
  这一行，而不是整张 25 行的条款表 —— 大块会稀释向量、拉低精度。
- **生成**要大块。只给一行"每点人民币100元"，模型无法判断这是哪个品种、
  有无例外情形，容易答错。

所以：**子块进向量库负责召回，命中后用 `parent_id` 回溯完整条款作为生成上下文**。

```
父块 (ParentChunk)  条款级 —— "第十二条 …"、条款表的一行、一组问答
   │
   └── 子块 (ChildChunk) 语义切分，200-300 字，存进 Chroma
        ├── 命中 → 回溯父块
        └── 同一父块的多个子块在融合后去重（否则一条条款会占满 top-5）
```

## 三路召回的分工

| 通路 | 擅长 | 典型场景 |
|---|---|---|
| BM25 + jieba | 精确术语命中 | "备兑开仓的保证金怎么算" —— "备兑开仓"是低频术语，IDF 高 |
| 稠密向量 bge-m3 | 语义泛化 | "到期没行权会怎样" → 命中"交割方式""到期日行权（欧式）"条款 |
| 关键词元数据 | 结构化定位 | "《期权交易管理办法》第十二条说了什么" —— 直接按 clause_id 匹配 |

BM25 的金融词典是关键。jieba 默认会把"行权价格间距"切成"行权/价格/间距"、
"备兑开仓"切成"备兑/开仓"，这些术语就无法作为整体匹配。
[`index/bm25.py`](../src/derivrag/index/bm25.py) 里注入了约 80 个衍生品术语。

## 元数据与时效性

每个块携带统一 schema：

```
venue           交易所代码
doc_title       文档标题
doc_no          文号（上证发〔2023〕48号）—— 引用真值
clause_id       第X条 / 字段名 / 规则编号
effective_status 现行有效 | 已废止 | 被修订 | 未知
source_url      原文链接
retrieval_date  抓取日期
```

**检索默认过滤"已废止"**，但保留"未知"—— 大多数交易所文档不写时效性字段，
把未知当废止会丢掉绝大部分语料。这个取舍写在
[`retrieve/hybrid.py::_where`](../src/derivrag/retrieve/hybrid.py) 里。

## 模块地图

| 路径 | 职责 |
|---|---|
| [`crawl/fetcher.py`](../src/derivrag/crawl/fetcher.py) | 带内容校验的下载器（Content-Type + magic bytes + WAF 指纹） |
| [`crawl/adapters.py`](../src/derivrag/crawl/adapters.py) | 各站点索引页 → URL 列表 |
| [`parse/pdf.py`](../src/derivrag/parse/pdf.py) | PyMuPDF 版面/表格，逐页双条件 OCR 兜底 |
| [`parse/html.py`](../src/derivrag/parse/html.py) | 站点正文选择器 + 页头页脚裁剪 + 问答抽取 |
| [`parse/docfile.py`](../src/derivrag/parse/docfile.py) | .doc/.docx，含 Word 自动编号还原 |
| [`parse/clean.py`](../src/derivrag/parse/clean.py) | 中文抽取噪声清洗、文号/时效性抽取 |
| [`chunk/clause.py`](../src/derivrag/chunk/clause.py) | 条款级父块切分（条文/表格行/问答/兜底） |
| [`chunk/semantic.py`](../src/derivrag/chunk/semantic.py) | 自研语义切分（句向量余弦断点） |
| [`index/embed.py`](../src/derivrag/index/embed.py) | bge-m3 三模编码（FlagEmbedding） |
| [`index/vectorstore.py`](../src/derivrag/index/vectorstore.py) | Chroma 封装，保留原始距离分数 |
| [`index/bm25.py`](../src/derivrag/index/bm25.py) | jieba + 金融词典 + BM25Okapi |
| [`retrieve/fusion.py`](../src/derivrag/retrieve/fusion.py) | **自研 score-aware RRF** + 父块去重 |
| [`retrieve/rerank.py`](../src/derivrag/retrieve/rerank.py) | bge-reranker-v2-m3 交叉编码器 |
| [`retrieve/hyde.py`](../src/derivrag/retrieve/hyde.py) | HyDE + 多轮查询重写 |
| [`retrieve/hybrid.py`](../src/derivrag/retrieve/hybrid.py) | 三路召回编排 |
| [`llm/provider.py`](../src/derivrag/llm/provider.py) | ollama 原生 / OpenAI 兼容 双通道 |
| [`llm/prompts.py`](../src/derivrag/llm/prompts.py) | 强引用约束提示词 |
| [`pipeline.py`](../src/derivrag/pipeline.py) | 端到端编排，各组件可单独开关（消融基础） |
| [`compat.py`](../src/derivrag/compat.py) | ragas × langchain-community 兼容垫片 |

## 性能特征（Apple M4 / 16GB / MPS 实测）

| 环节 | 耗时 |
|---|---|
| 抓取 191 份文档 | 约 6 分钟（1 req/s 限速） |
| 解析 + 语义切分 | 约 15 分钟 |
| 向量化 14,567 子块 | 约 10 分钟（23.8 块/秒） |
| 单次检索（不重排） | < 1 秒 |
| 单次检索（含重排 top-50→5） | 数秒 |
| 生成（qwen3:8b 本地） | 3–25 秒（视答案长度） |

⚠️ qwen3:8b + bge-m3 + reranker **无法在 16GB 上同时常驻**，
服务层用 `lazy_load` 按需加载。
