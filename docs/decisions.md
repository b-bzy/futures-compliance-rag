# 技术决策记录

每条都记录：**当初的方案 → 实测发现了什么 → 最终怎么做**。
这份文档的价值在于，面试时被问到"为什么不用 X"，答案不是"没想到"，
而是"试过，因为具体的 Y 原因不可行，换成了 Z"。

---

## D1. PaddleOCR → PyMuPDF + RapidOCR

**原方案**：PaddleOCR / PP-Structure 做版面识别。

**实测**：
- `paddlepaddle` 确实有 cp312/cp313 macOS arm64 轮子，**轮子不是障碍**
- 真正的障碍是 `paddlex[base]` 的依赖 pin：
  `langchain<1.0`、`langchain-community<1.0`、`langchain-openai<1.0`、`numpy<2.4`
  装上去 pip 会静默把 langchain 1.x 降级，整个检索层全部 ImportError
- 更关键的是**根本不需要 OCR**：抽样 8 份交易所 PDF，7 份有完整原生文本层。
  最终全量 190 份文档里只有 6 页触发了 OCR

**决策**：PyMuPDF 做主力解析（`find_tables()` 抽条款表实测 100% 正确），
RapidOCR 做兜底。若一定要 PP-Structure，放独立 py3.12 环境用文件系统交换数据。

**OCR 触发判据的坑**：不能只用"页面文本少"这一个条件。DCE 豆粕期权制度汇编
第 0 页是扫描封面（0 字符），但正文页都是原生文本 —— 单条件判据会把整篇
送去 OCR。改成**逐页双条件**：文本 < 50 字符 **且** 图片覆盖面积 > 30%。

---

## D2. `BGE-Reranker-M3` 不存在

**原方案**：简历/网上教程里写的 `BAAI/bge-reranker-m3`。

**实测**：HF API 返回 401。用一个确定不存在的假 ID 做对照实验，同样返回 401
（HF 对未认证请求把 404 伪装成 401），再通过 `author=BAAI&search=bge-reranker`
枚举确认：该模型**从未存在**。

**决策**：改用 `BAAI/bge-reranker-v2-m3`（apache-2.0，568M，XLM-RoBERTa-large，
其 config 的 `_name_or_path` 正是 `BAAI/bge-m3`）。

---

## D3. bge-m3 走 sentence-transformers 只有 dense

**原方案**：`SentenceTransformer("BAAI/bge-m3")` 做稠密向量，以为拿到了三模。

**实测**：仓库里确实带 `colbert_linear.pt` 和 `sparse_linear.pt`，但
`modules.json` 只声明 `Transformer → Pooling → Normalize` 三层 ——
sentence-transformers **永远不会加载那两个头**。也就是说会在
"我在跑 dense+sparse+colbert 混合检索"的认知下，实际只跑了 dense。

**决策**：用 `FlagEmbedding.BGEM3FlagModel(return_dense=True, return_sparse=True)`。
配置里保留 sentence-transformers 后端做消融对照，并在代码注释里写明它是 dense-only。

---

## D4. DeepSpeed 在 macOS 上是结构性不可用

**原方案**：本机用 DeepSpeed 做增量微调。

**实测**（不是"慢"，是装不上也跑不了）：
- PyPI 只有 sdist，**任何平台都没有预编译 wheel**
- `op_builder/builder.py` 里 `darwin` / `macos` 出现 **0 次**，只对 nvcc/hipcc 构建
- 自带的 MPS accelerator 自报 `supported_dtypes=[float32]`、
  `_communication_backend_name=None` → ZeRO 无法初始化
- Qwen3-8B 的 bf16 权重 16.38 GB > 本机 16 GiB 总内存，权重本身就放不下

**决策**：训练脚本写完整（ZeRO-2 给 reranker、ZeRO-3+offload 给 Qwen3-8B），
标注为租用 GPU 执行；本机提供 `--smoke`，在 MPS 上真实跑前向/反向/优化器更新
3 步，验证数据管线、标签掩码、LoRA 注入。见 [training/README_GPU.md](../training/README_GPU.md)。

**附带的坑**：`peft` 的 target-module 自动映射表里**没有任何 qwen 条目**，
不显式传 `target_modules` 会直接 `ValueError`。脚本里已写死七个投影层。

---

## D5. langchain 1.x 的 import 路径全变了

**实测**：`langchain.retrievers` 整个命名空间和 `langchain.text_splitter`
在 1.x 里**都不存在**了，网上所有教程的 import 都会 `ModuleNotFoundError`。

**正确路径**：
| 组件 | 1.x 位置 |
|---|---|
| EnsembleRetriever / ParentDocumentRetriever / ContextualCompressionRetriever | `langchain_classic.retrievers` |
| MultiQueryRetriever | `langchain_classic.retrievers.multi_query` |
| CrossEncoderReranker | `langchain_classic.retrievers.document_compressors` |
| BM25Retriever | `langchain_community.retrievers` |
| 各类 splitter | `langchain_text_splitters` |
| **SemanticChunker** | **不在 langchain_text_splitters**，在 `langchain_experimental` |

**决策**：语义切分自己实现（见 D6）；检索器自己组装而不是用
`EnsembleRetriever`，因为需要拿到原始分数（见 D7）。

---

## D6. 语义切分自己写

**原因**：
1. `SemanticChunker` 在 `langchain_experimental`，会引入额外依赖
2. 现成实现按英文 `.!?` 断句，中文法规里"第3.5条"会被切碎
3. 现成实现用固定余弦阈值；条款长度差异极大，固定阈值在短条款上不切、
   长条款上切碎。自己实现用 `mean + z*std` 自适应

只有几十行代码，可控性和可讲性都更好。`embed_fn=None` 时退化为纯长度切分，
让不加载模型也能跑通整条流水线（也是消融对照的一档）。

---

## D7. 自研 score-aware RRF

**现成方案**：`EnsembleRetriever.weighted_reciprocal_rank`，标准 RRF，k=60。

**问题**：它**只看名次，完全丢弃原始分数**。但在条款检索里：
- BM25 得分 38.5 的第 1 名（术语精确命中，如"合约乘数"只出现在少数块中）
  和 BM25 得分 2.1 的第 1 名，置信度天差地别
- 稠密向量的余弦分数分布很平（候选常挤在 0.6~0.7），只按分数融合又会被
  BM25 的长尾大分值压制

**决策**：`score = (1-α)·Σ w/(k+rank) + α·Σ w·minmax(s)`。
rank 部分保证鲁棒（不受量纲影响），score 部分保留"这一路有多确信"。
**α=0 时精确退化为标准 RRF**，单元测试里断言了这一点，消融实验直接切换 α 做 A/B。

---

## D8. ragas 0.4.3 在 langchain-community 0.4.x 上导入即崩

**实测**：`ragas/llms/base.py` 第 12 行无条件执行
`from langchain_community.chat_models.vertexai import ChatVertexAI`，
但 langchain-community 0.4.x 已把 Vertex AI 拆分到独立包，该子模块不存在。
ragas 的 metadata 里 `langchain-community` 完全不带版本约束 —— 是 ragas 自身的打包缺陷。

**为什么不降级 langchain-community**：0.3.x pin `langchain-core<0.4`，
而本项目用 langchain-core 1.x，降级会连锁摧毁整个检索层。

**决策**：写一个受控垫片 [`src/derivrag/compat.py`](../src/derivrag/compat.py)，
在 import ragas 前往 `sys.modules` 注册占位模块。ragas 只把这两个类用于
isinstance 判断和 llm_factory 分支派发，而本项目 judge 走 OpenAI 兼容协议
（指向本地 ollama），永远不会命中 Vertex 分支，行为完全等价。
装了 `langchain-google-vertexai` 时垫片会自动转发真实实现。

---

## D9. ollama 不走 OpenAI 兼容端点

**原方案**：ollama 提供 `/v1` OpenAI 兼容端点，用同一个 OpenAI 客户端即可。

**实测**（qwen3 默认开启 thinking）：
- 思考内容被放进独立字段，`message.content` 是**空字符串**
- 思考照常消耗 max_tokens：300 tokens 全部耗尽在思考上，
  `finish_reason='length'` 而 content 长度为 0
- `/no_think` 软开关**时灵时不灵**（同一模型不同问题表现不一致）
- `extra_body={"think": False}` 被静默忽略
- 实测一个 44 字的回答耗时 42.6 秒

**决策**：ollama 走原生 `/api/chat` 并传 `"think": false`，实测稳定关闭思考、
content 正常返回、同样的问题 **3.0 秒**返回。deepseek / openai 仍走 OpenAI 协议。
两者实现同一个 `chat()/stream()` 接口，上层无感知。

---

## D10. Word 自动编号还原

**实测**：SSE 的规则 `.doc` 里，"第一条""第二条"这些条号**不是正文文本**，
而是 Word 的自动列表编号。python-docx 的 `paragraph.text` 完全读不到，
LibreOffice 转 HTML 也不渲染。

《上海证券交易所股票期权试点交易规则》里正则能匹配到的"第X条"只有 **10 处**
（全是"依照本规则第X条"这类交叉引用），而真实条数是 **170 条**。
不还原编号，这份文档只能退化成按段落硬切，彻底失去"精确到条"的引用能力
—— 而这正是本项目最核心的价值。

**决策**：直接读 `word/numbering.xml`，它明确写着
```xml
<w:numFmt w:val="chineseCountingThousand"/>
<w:lvlText w:val="第%1条"/>
```
按文档顺序对 `numPr` 段落计数，再套用该模板，完全复现 Word 的显示结果。

**踩到的坑**：计数器必须以 **abstractNumId** 为键而不是 numId。
本例中 `numId=3` 和 `numId=4` 指向同一个 `abstractNum`，Word 是共用一条
编号序列继续往下编的；按 numId 分别计数会让正文里冒出两个"第一条"。

还原后：第一条 → 第一百七十一条，连续无重复。

---

## D11. 条款切分的目录页过滤

**原方案**：正文短于 25 字符的"第X条"视为目录条目，丢弃。

**实测**：这条规则会把大量**合法的短条款**一并丢掉，例如
"第三条 本细则未规定的，按照交易所相关业务规则的规定执行。"
单元测试里 4 条测试条款全被误杀，返回 0 个块。

**决策**：判据改成"短 **且** 没有句末标点"。正文条款一定以 。；！？ 收尾，
目录条目不会。

---

## D12. 表格内容重复

**实测**：`page.get_text()` 会把表格单元格逐行吐出来（丢失行列关系），
如果再把 `find_tables()` 的结构化渲染拼上去，同一份内容会出现两遍 ——
既污染 BM25 词频，又让重排看到大量重复候选。

**决策**：逐行比对，丢掉页面文本里已被表格覆盖的行，保留标题与附注，
再接上结构化的 `字段：值` 渲染。SZSE 深证100ETF 条款表实测从 2000 字符
降到 1021 字符，标题保留，零重复。

---

## D13. HTTP 200 不是有效的健康检查

**实测**：
| 站点 | 现象 |
|---|---|
| SHFE / INE | 所有 HTML 返回 **200 + 10,648 字节**的 WAF 验证码页 |
| www.sgx.com/sites/default/files/*.pdf | 返回 **200 但 Content-Type 是 text/html**（SPA 壳） |
| DCE / CZCE | 所有 HTML 返回 **412**，只有静态 PDF 路径可取 |
| CFFEX | **https 直接 TLS 握手失败**，必须走 http |
| HuggingFace | 不存在的 repo 返回 **401 而不是 404** |
| ModelScope | 不存在的 repo 也返回 **200**，要看 body 里的 Code 字段 |

**决策**：`Fetcher` 每次下载断言三件事 —— Content-Type 匹配、magic bytes 正确
（PDF `%PDF`、doc `d0cf11e0`、docx `PK`）、长度不在 WAF 指纹表里。
实测 190 份文档全部通过，0 个验证码页混入语料。

---

## D14. 评测集必须是人写的

**问题**：如果用 LLM 挖掘的 QA 去评测 LLM 生成的答案，再用 LLM 当 judge，
这是三重循环论证。

**决策**：数据集严格分层。
- **Tier-1 金标**：交易所官方问答（SSE 50ETF FAQ 31 对、熔断机制问答 8 对、
  CFFEX 常见问答 30+ 条），**纯人工撰写**。所有报告出去的指标只在这上面算。
- **Tier-2 合成**：条款表模板（零 LLM）+ LLM 挖掘（span-grounding 硬校验）
  + 跨交易所对比。**只用于训练**，不用于报告准确率。

LLM 挖掘的 span-grounding 校验：答案归一化后必须是源条款归一化后的**连续子串**，
不满足直接丢弃。这样"自动挖掘的结构化 QA"这句话才站得住脚。

---

## D15. 评测集泄漏 —— 第一版消融表全是 1.000

**发生了什么**：第一次跑完消融实验，六个档位**全部** Recall@1 = 1.000。

看起来像大获全胜，实际是评测彻底失效。

**原因**：Tier-1 金标是从问答型文档抽的，而这些文档的父块文本就是
`问：X\n答：Y` —— 也就是说**金标问题的原文 100% 出现在被索引的子块里**
（实测 72/72）。检索退化成了精确字符串匹配，任何组件的效果都测不出来。

```
被索引的子块: "问：证券市场实施指数熔断机制对股票期权市场有何影响？\n答：证券市场将于…"
金标问题:     "证券市场实施指数熔断机制对股票期权市场有何影响？"   ← 一字不差
```

**决策**：加一步**去泄漏改写**。用 LLM 把金标问题改写成语义相同、用词不同的
形式，再校验改写后的问题不再是语料的连续子串，产出 `gold_eval.jsonl`。
`configs/config.yaml` 里 `eval.gold_file` 指向的是改写后的版本，
原始金标保留在 `gold_raw_file` 供人工核对。

```
原: 指数熔断期间，期权投资者能否申报或撤销申报？
改: 在指数触发熔断机制时，期权交易者是否可以提交或撤回申报？
```

改写后重跑，指标立刻分化，各档位差异清晰可见。

**第二个问题：FAQ 评测集太容易，重排看不出价值。**
改写后 FAQ 档的 dense-only 基线就有 Recall@1 = 0.986，天花板效应明显 ——
因为 FAQ 答案块在语料里非常独特，召回几乎不会错。

**决策**：再造一个**困难评测集** `gold_hard.jsonl`：从中文合约条款表 QA 里
抽 150 条并同样改写。这些题要在十几个只差品种名的近似块里选对一个
（"易方达科创50ETF期权的每张合约对应多少份ETF？"—— 15 个品种的条款表
都有"合约单位"这一行），是真正考验辨析能力的任务。

在困难集上，各组件的价值才显现出来：自研 score-RRF 比标准 RRF 高 12.0 pp，
重排再提 6.7 pp。两张表都保留在 `eval/reports/`，README 里以困难集为主表。

**教训**：指标好看的第一反应应该是怀疑评测集，而不是庆祝。

**后续补充（重要）**：困难集因修复文档标题而重新生成过一次，
同一套构造流程的两次抽样，各档位有 **±5–7 pp** 的摆动
（基线 0.660↔0.727，重排后 0.860↔0.807）。
所以 n=150 时**小于约 6 pp 的差距不能当作真实差异**。
"score-RRF 优于标准 RRF"两次都成立（+6.7 / +12.0 pp），可以讲；
"重排提升幅度"只能说方向确定。这是当前评测最大的方法论短板，
解法是把评测集扩到 500+ 条，见 [audit.md](audit.md) 第二批计划。
