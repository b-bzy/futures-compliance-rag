# 许可范围说明 / Scope of License

## 中文

本仓库的 [MIT License](LICENSE) **仅覆盖源代码、配置与文档**，
不覆盖、也无权授予 `scripts/01_crawl.py` 所抓取的交易所文档的任何权利。

那些文档 —— 上海证券交易所、深圳证券交易所、中国金融期货交易所、
上海期货交易所、大连商品交易所、郑州商品交易所、SGX、HKEX、OCC、SEC
发布的业务规则、合约条款与投资者问答 —— **版权归各自发布机构所有**，
已通过 `.gitignore` 明确排除在本仓库之外。仓库里没有任何一份语料。

若你运行抓取脚本，需自行遵守各来源站点的使用条款与 robots.txt。
本项目的抓取器已做了两件事：

- 限速 **1 请求/秒**（`configs/config.yaml` 的 `crawl.rate_limit_seconds`）
- 排除 robots.txt 禁止抓取的主机 —— `api2.sgx.com` 的 robots.txt 是
  `Disallow: /`，因此其文件在 `configs/sources.yaml` 里标记为 `manual_only`，
  不参与批量抓取

评测报告（`eval/reports/*.md`、`*.json`）只含汇总指标。
含条款原文的评测明细 `generation_records.jsonl` 同样被 `.gitignore` 排除，
需要时用 `python scripts/06_eval.py` 在本地重新生成。

## English

The [MIT License](LICENSE) in this repository covers **only the source code,
configuration and documentation**. It does not cover, and cannot grant any
rights to, the exchange documents retrieved by `scripts/01_crawl.py`.

Those documents — business rules, contract specifications and investor Q&A
published by SSE, SZSE, CFFEX, SHFE, DCE, CZCE, SGX, HKEX, OCC and the SEC —
remain the copyright of their respective publishers. They are deliberately
excluded from this repository via `.gitignore`. **No corpus is committed here.**

If you run the crawler, you are responsible for complying with each source
site's terms of use and robots.txt. This crawler rate-limits to 1 request per
second and excludes hosts whose robots.txt disallows crawling (`api2.sgx.com`
returns `Disallow: /`, so its files are marked `manual_only` in
`configs/sources.yaml` and are never fetched in bulk).

Evaluation reports under `eval/reports/` contain aggregate metrics only.
The per-record evaluation dump, which embeds verbatim clause text, is likewise
gitignored and can be regenerated locally with `python scripts/06_eval.py`.
