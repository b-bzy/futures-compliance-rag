# 期货合规条款 RAG 检索系统

**覆盖 10 家境内外交易所的 190 份期权业务规则与合约条款，回答精确引用到「某规则第 X 条」。**

> A clause-level RAG system for exchange-traded options rules — hybrid retrieval
> (BM25 + BGE-M3 + metadata) → custom score-aware RRF → cross-encoder reranking,
> with measured, reproducible ablations. Docs are in Chinese.

![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Corpus](<https://img.shields.io/badge/corpus-190%20docs%20%C2%B7%205.4k%20clauses-orange>)

> 仓库含两部分：根目录代码是**个人实现**（公开语料，可复现，本 README 的主体）；
> [docs/product_case.md](docs/product_case.md) 是**公司项目的产品复盘**（已脱敏、不含源码）。
> 两者不是同一套系统 —— 公司项目受涉密约束无法开源，所以我在公开语料上把同一套设计取舍
> 自己实现并实测了一遍。

---

## 效果

<!-- TODO: 补 Streamlit 界面截图或 GIF，放在 docs/assets/demo.png -->

<!-- ![演示](docs/assets/demo.png) -->

实测输出（节选自 `scripts/06_eval.py --generation` 的生成记录）：

```
问：股票期权市场会受到证券市场指数熔断机制的哪些影响？

答：① 指数熔断期间，期权市场同步暂停交易及恢复交易 [1]
    ② 熔断期间不揭示虚拟参考价格与虚拟匹配量 [2]
    ③ 指数熔断优先于个股/期权的交易熔断执行 [3]
    ④ 行权日熔断至收盘的，最后交易日顺延、当日行权申报无效 [4]

引用 [1] 《上海证券交易所股票期权试点交易规则》第 X 条
        文号 上证发〔20XX〕XX号 · 现行有效 · 原文链接
```

每条引用都带**文号 / 条号 / 生效状态 / 原文 / 源链接**，无据可依时拒答而不是编。
完整记录与逐条人工复核见 [eval/reports/RESULTS.md](eval/reports/RESULTS.md)。

---

## 快速开始

```bash
# 1. 独立环境（务必独立，原因见下方「已知边界」）
conda create -n derivrag python=3.12 -y
conda activate derivrag
pip install -r requirements.txt

# 2. 本地生成模型（零 API 成本方案）
ollama pull qwen3
#    或用 DeepSeek API：export DEEPSEEK_API_KEY=sk-...
#    再把 configs/config.yaml 的 llm.provider 改成 deepseek

# 3. 解析 .doc 附件需要 LibreOffice（上交所 16 份规则正文只存在于 .doc 里）
brew install --cask libreoffice

# 4. 跑通全流程
python scripts/01_crawl.py --include-en     # 抓取语料，约 6 分钟
python scripts/02_parse.py                  # 解析 + 分块，约 15 分钟
python scripts/03_build_index.py --rebuild  # 建索引，约 10 分钟
python scripts/04_mine_qa.py --tier1        # 抽取人工金标 QA

# 5. 用起来
streamlit run src/derivrag/ui/app.py        # 演示界面
uvicorn derivrag.api.server:app --port 8000 # API 服务
```

⚠️ 语料版权归各交易所所有，本仓库**只提交抓取脚本、不提交语料**。

---

## 核心结果

困难评测集 150 条，命中判定为返回候选的 `parent_id` 是否等于金标条款所在块。
硬件 Apple M4 / 16GB / MPS。

| 档位                               | Recall@1        | Recall@5        | MRR             | 平均延迟 |
| ---------------------------------- | --------------- | --------------- | --------------- | -------- |
| 仅稠密向量（代理基线）             | 0.727           | 0.967           | 0.831           | 0.23s    |
| 仅 BM25                            | 0.407           | 0.607           | 0.491           | 0.01s    |
| BM25 + 稠密，标准 RRF              | 0.620           | 0.887           | 0.738           | 0.10s    |
| 三路召回 +**自研 score-RRF** | 0.740           | 0.967           | 0.836           | 0.10s    |
| **+ 交叉编码器重排**         | **0.807** | **0.993** | **0.888** | 4.53s    |

**端到端 Recall@1 +8.0 pp（相对提升 11.0%）。最值得说的一行是第 3 → 第 4 行：
标准 RRF 只看名次不看分数，弱的 BM25 结果会把强的稠密结果挤下去 —— 混合后（0.620）
反而比纯稠密（0.727）差 10.7 pp；保留归一化原始分数后恢复到 0.740。
「混合检索一定比单路好」是需要验证的假设，不是定理。**

> ⚠️ **两个我下错又自己纠正的判断**，比上面任何一个数字都更能说明这个项目怎么做的：
> ① 第一版消融表六档**全是 Recall@1 = 1.000** —— 实为评测集泄漏，金标问题原文
> 100% 出现在被索引的块里；② 我曾断言「HyDE 是负收益 -4.0 pp」，换一版评测集重测
> **完全打平**，结论被自己的数据推翻。两件事指向同一个短板：**n=150 统计功效不足，
> 小于约 6 pp 的差距不能当作真实差异**。全过程记录在
> [RESULTS.md](eval/reports/RESULTS.md)，已列为下一步最高优先级。

完整消融、HyDE 配对对比、多轮改写、生成质量与全部原始数据：
**[eval/reports/RESULTS.md](eval/reports/RESULTS.md)**

```bash
python scripts/04_mine_qa.py --hard-eval 150                # 重建困难评测集
python scripts/07_ablation.py --gold data/qa/gold_hard.jsonl # 重跑消融
```

---

## 技术要点

- **三路混合召回** —— BM25（注入约 80 个衍生品术语的 jieba 词典）+ bge-m3 稠密向量
  + 关键词元数据。不加词典，"备兑开仓"会被切成"备兑/开仓"，术语无法整体匹配。
    → [architecture.md](docs/architecture.md#三路召回的分工)
- **自研 score-aware RRF** —— `(1-α)·Σw/(k+rank) + α·Σw·minmax(score)`，
  在标准 RRF 的名次鲁棒性之上保留原始分数的置信度；`α=0` 时精确退化为标准 RRF
  （单元测试断言了这一点），消融实验直接切 α 做 A/B。
  → [fusion.py](src/derivrag/retrieve/fusion.py) · [decisions.md D7](docs/decisions.md#d7-自研-score-aware-rrf)
- **父子索引** —— 子块进向量库负责召回，命中后用 `parent_id` 回溯完整条款供生成；
  融合后按 parent_id 去重，否则一条长条款的 5 个子块会占满 top-5。
  → [architecture.md](docs/architecture.md#为什么是父子索引)
- **评测集去泄漏 + 困难集构造** —— 两个评测集的答案都不是 LLM 生成的（分别是交易所
  原文和条款表单元格字面值），只有问题措辞经过改写，不存在「LLM 出题、LLM 作答、
  LLM 判分」的循环论证。 → [RESULTS.md §0](eval/reports/RESULTS.md)
- **微调** —— reranker + Qwen3-8B LoRA 训练代码完整，本机只跑 `--smoke`（MPS 上真实
  前反向 3 步验证数据管线与 LoRA 注入）。DeepSpeed 在 macOS 上结构性不可用、
  8B 的 bf16 权重 16.38 GB 超出本机 16 GiB 内存。
  → [training/README_GPU.md](training/README_GPU.md) · [decisions.md D4](docs/decisions.md#d4-deepspeed-在-macos-上是结构性不可用)

---

## 文档导航

| 文档                                                        | 内容                                                                                  |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **[eval/reports/RESULTS.md](eval/reports/RESULTS.md)** | 全部评测结果 —— 消融、HyDE 配对、多轮改写、生成质量，**含推翻自己结论的记录** |
| **[docs/audit.md](docs/audit.md)**                     | 自我审计与执行计划 —— 逐条列出当前缺陷（含已修/未修），每条都有诊断脚本证据         |
| **[docs/decisions.md](docs/decisions.md)**             | 15 条技术决策记录 —— 「想怎么做 → 实测发现什么 → 最终怎么做」                     |
| [docs/architecture.md](docs/architecture.md)                 | 全链路架构图、模块地图、性能特征                                                      |
| [docs/corpus.md](docs/corpus.md)                             | 语料规模、来源、覆盖边界与「10 万条条款」口径澄清                                     |
| [docs/interview_qa.md](docs/interview_qa.md)                 | 针对每条简历表述的追问准备                                                            |
| [docs/product_case.md](docs/product_case.md)                 | 公司项目的产品复盘（已脱敏，无源码）                                                  |

---

## 项目结构

```
configs/          config.yaml（主配置）· sources.yaml（已验证的语料源清单）
src/derivrag/
  crawl/          带内容校验的下载器 + 各站点适配器
  parse/          PDF / HTML / Word 解析，含 Word 自动编号还原
  chunk/          条款级父块 + 自研语义子块切分
  index/          bge-m3 编码 · Chroma · BM25
  retrieve/       三路召回 · 自研 RRF · 重排 · HyDE · 查询重写
  llm/            双 provider（ollama 原生 / OpenAI 兼容）+ 提示词
  api/ ui/        FastAPI 服务 · Streamlit 演示界面
scripts/          01 抓取 → 08 构造 SFT 数据，编号即执行顺序
training/         reranker + qwen3 训练脚本与 DeepSpeed 配置
eval/reports/     评测与消融报告
docs/             架构 · 决策 · 审计 · 语料 · 面试准备 · 产品复盘
tests/            32 个单元测试，每个都对应一个真实踩过的坑
constraints.txt   禁止安装清单 —— 装之前务必先读
```

---

## 已知边界

诚实列出，不掩饰：

1. **评测集统计功效不足** —— n=150，小于约 6 pp 的差距不应当作真实差异。
   解法是扩到 500+ 条。→ [RESULTS.md](eval/reports/RESULTS.md)
2. **深交所列表页无法枚举** —— 客户端渲染 + 无 sitemap，只抓到已知直链的 4 份 PDF。
   代码里预留了 Playwright manifest 入口。→ [corpus.md §5.1](docs/corpus.md#51-深交所只抓到-4-份)
3. **时效性字段覆盖率低** —— 多数文档不写生效/废止信息，检索只过滤**明确标记已废止**的，
   保留"未知"。→ [corpus.md §5.2](docs/corpus.md#52-时效性字段覆盖率低)
4. **bge-m3 稀疏向量还没进融合** —— 编码时已算出 `lexical_weights`，当前融合只用三路。
5. **微调未在全量数据上执行** —— 见上方「技术要点」。
6. **RAGAS 数字本报告不提供** —— 代码路径已修好，但本地 8B judge 实测 3 条样本跑 35
   分钟未完成，需要云端 judge。不拿没跑出来的指标充数。→ [RESULTS.md §5](eval/reports/RESULTS.md)
7. **必须用独立环境** —— `paddlex` / `surya-ocr` / `mlx-lm>=0.31` 等包会静默降级
   langchain 或 transformers，摧毁整个依赖树。装任何新包前先读
   **[constraints.txt](constraints.txt)** 并跑
   `pip install --dry-run <pkg> 2>&1 | grep -E "Would (install|uninstall)"`。

---

## 许可与数据

代码、配置与文档以 [MIT License](LICENSE) 发布。

⚠️ **许可不覆盖语料。** 抓取脚本获取的交易所文档版权归各自发布机构所有，
已通过 `.gitignore` 排除在仓库之外 —— 这里没有任何一份语料。
详细范围说明与合规要点见 **[NOTICE.md](NOTICE.md)**。
