# A 股场内 ETF 动量轮动

当前默认策略：**16 只 ETF 池、每周信号、20/40 日快速重入、7%年化风险仓波动预算、前4名排名缓冲并必选合格池第1名、风险 ETF 总目标仓位上限70%、单只风险 ETF 目标仓位上限30%**。2026-10-06 经用户选择启用，首次按新规则执行为10月8日（9月30日周信号）。银行 ETF `512800` 和其他风险 ETF 使用同一套筛选及仓位规则。风险 ETF 最多3只，另可持有十年国债 ETF；国债不受单只风险 ETF 上限约束，未分配的资金为现金。策略只使用 Tushare 的 `fund_daily`、`fund_adj` 和交易日历。科创50 ETF `588000` 当前未纳入默认标的池。

## 在 PyCharm 中运行

1. 将本文件夹作为项目打开，设置 Python 解释器并在项目终端执行 `python -m pip install -r requirements.txt`。
2. 在 `data/tushare_token.txt` 放一行自己的 Tushare Token，或设置环境变量 `TUSHARE_TOKEN`。Token 文件已被 `.gitignore` 排除。
3. 右键运行 `run_backtest.py` 查看历史结果；右键运行 `live_signal.py` 获取下一次调仓计划。两个入口的默认参数相同。

新克隆的项目尚无行情缓存，第一次请联网运行 `python run_backtest.py --end 2026-09-30`，下载后再使用 `--offline`。每位成员使用自己的Tushare Token及数据权限。可将 `data/tushare_token.example.txt` 复制为 `data/tushare_token.txt` 后填写；也可在启动Python前设置环境变量 `TUSHARE_TOKEN`。程序不会自动读取 `.env`。

## 团队协作

仓库共享源代码、因子库定义、测试、策略初始化状态及研究结论。`data/` 默认排除，仅提交 `.gitkeep`、`rotation_state.json` 和空白Token示例；真实Token、个人账户数据库与快照、行情缓存、因子缓存、生成报告、虚拟环境均保留在成员本地。

`rotation_state.json` 只含公开策略参数和周目标代码，用于一致重放排名缓冲；它不是账户持仓明细。更改选仓规则或这个锚点时，请在分支上修改并通过Pull Request讨论。协作步骤见 [CONTRIBUTING.md](CONTRIBUTING.md)。文档中的 `output/` 路径是本地生成结果，克隆后需自行运行对应脚本；仓库未分发Tushare行情。

终端示例：

```powershell
python run_backtest.py --offline --end 2026-09-30
python plot_benchmarks.py --offline
python live_signal.py --capital 100000
python -m unittest discover -s tests
```

`--offline` 只读本地 Tushare 缓存。平时不加该参数即可增量更新：当结束日期向后延长时，程序复查最后 21 个自然日并追加新数据；`--refresh` 才会重新下载完整区间。若当天收盘价已到而基金复权因子尚未更新，程序会停止并提示，等数据完整后重试。缓存保存在 `data/cache/`，因子缓存在 `data/factors.sqlite`。

真实账户的周度持仓、入出金和收益另用 `portfolio_tracker.py` 记在 `data/live_portfolio.sqlite`。录入模板是 `portfolio_entry.example.json`；操作步骤和收益口径见 [LIVE_GUIDE.md](LIVE_GUIDE.md)。这部分只读你提供的券商记录，不会把回测净值当作真实账户收益。

因子库在 `etf_rotation/factors.py` 定义十三种价格因子：动量、均线趋势、年化波动率、下行波动率、前高突破、单窗口风险调整动量、趋势效率、短长波动率比、均线偏离标准分、10/30 日与 5/10 日均线同侧持续性，以及滚动窗口的高点回撤和低点涨幅。`FactorSpec` 指定因子、交易日窗口及跳过最近交易日的天数；`FactorStore` 按需计算并写入 SQLite，价格矩阵变化时自动失效重算。默认实盘策略实际调用 20/40 日动量、40 日趋势、60 日波动率及国债 60 日动量；其他因子目前只供研究。默认排名使用两段动量均值除以波动率，是策略中的组合评分，不等同于库内单窗口的 `risk_adjusted_momentum`。

## 规则

- 风险资产使用不跳过近期交易日的 **20/40 日动量**和 **40 日均线趋势**筛选，按两段动量均值除以60日历史波动率排名。先选入合格池第1名；剩余最多2个名额，按当前排名优先保留仍合格且位于前4名的旧周目标，再按评分补足3只。绝对筛选失败的旧仓不能靠缓冲保留，第1名不保证权重最大。风险仓内部按逆波动率分配，以过去60日协方差估计总波动率。沪深300低于40日均线时，风险仓目标再减半。
- 风险仓目标受 **7%年化波动预算**、**70%总目标仓位**和**单只风险ETF 30%目标仓位**共同限制。单只超限部分先按现有权重分给其他已入选且未触顶的风险ETF，剩余仓位在十年国债ETF `511260` 的60日动量为正时配置国债，否则留现金。国债不受30%上限限制。
- 每个交易周最后一个开市日收盘后生成信号，**下一交易日收盘**按目标权重模拟成交。默认单边手续费 3 bps、滑点 2 bps；已有仓位在两次调仓之间自然漂移。回测与实盘信号均不会连接券商下单。

标的池在 [etf_rotation/config.py](etf_rotation/config.py)：`510300` 沪深300、`159915` 创业板、`159941` 纳指、`518880` 黄金、`511260` 十年国债、`159995` 半导体、`159875` 新能源、`159928` 消费、`512660` 军工、`515070` 人工智能、`512880` 券商、`512800` 银行、`513520` 日经、`561360` 石油产业、`159985` 豆粕期货、`159980` 有色。`510300` 还用作大盘趋势过滤；新上市 ETF 在积累足够历史数据前不会入选。`512800` 是风险 ETF，不充当国债防守仓。

## 文件与输出

| 文件 | 用途 |
| --- | --- |
| `etf_rotation/data.py` | Tushare 下载、同日基金复权和增量缓存 |
| `etf_rotation/factors.py`、`factor_store.py` | 因子计算与 SQLite 因子库 |
| `etf_rotation/predictive_factors.py` | 研究用途：只用既有周收益滚动拟合跨行业 PCR 预测 |
| `etf_rotation/market_factors.py` | 研究用途：四种只用历史收盘价的大盘状态候选因子 |
| `etf_rotation/risk_strategy.py` | 当前唯一选仓与防守规则 |
| `etf_rotation/strategy_state.py`、`data/rotation_state.json` | 保存切换锚点，从旧周目标重放排名缓冲 |
| `etf_rotation/backtest.py` | 下一交易日成交、费用和缺价处理 |
| `run_backtest.py` | PyCharm 回测入口 |
| `plot_benchmarks.py` | 将当前回测与六类宽基 ETF 买入持有收益画在同一张图 |
| `live_signal.py` | 盘后调仓计划入口 |
| `portfolio_tracker.py` | 录入实际持仓、入出金，查看真实账户收益 |

回测输出位于 `output/backtest/`：`equity.csv`、`daily_weights.csv`、`signal_weights.csv`、`trades.csv`、`summary.json`，有 Matplotlib 时还会生成 `equity.png`。实盘计划位于 `output/live_signal.json`。历史表现与限制见 [RESEARCH.md](RESEARCH.md)；实际操作步骤见 [LIVE_GUIDE.md](LIVE_GUIDE.md)。

当前策略的稳健性审计保留在 `research_strategy_audit.py`，离线结果在 `output/strategy_audit/`。已结束的均线回归、事件仓和 588000 扩池探索保留结论于 [RESEARCH.md](RESEARCH.md)，不再保留中间研究脚本和输出。

2026-10-06 的扩池及规则优化研究保留一个复现入口 `research_strategy_optimization.py`，固定比较 18 个方案；中文报告、对比图及稳健性结果在 `output/optimization_20261006/`。可运行 `python research_strategy_optimization.py --offline --robustness --end 2026-09-30` 复现，或用 `--summarize` 仅重生成报告和图。这些候选没有接入默认实盘信号。

7%预算与原排名缓冲的历史比较使用同一入口：`python research_strategy_optimization.py --combined --offline --robustness --end 2026-09-30`。报告和收益/回撤图位于 `output/optimization_20261006/combined_7pct/`；`--combined --summarize` 可仅重生成报告。原缓冲历史年化12.54%、夏普1.78、最大回撤4.85%，高成本压力下回撤5.47%；它与现在的必选第1名版本不同。研究基线冻结为切换前6%无缓冲版本，避免升级实盘后改写旧比较。

最近回撤归因可运行 `python research_strategy_optimization.py --recent-audit --offline --end 2026-09-30`，结果位于上述目录的 `recent_drawdown/`。它核对原净值、分解ETF收益贡献，并记录保留第4名后的收益及周中筛选失败至退出的时间；仅作解释，不更改默认规则。

强制最强候选入选的研究使用 `python research_strategy_optimization.py --priority-buffer --offline --robustness --end 2026-09-30`，报告在 `output/optimization_20261006/priority_buffer/`。**7%＋缓冲必选第1名现已接入默认实盘与回测**。2022-01-04至2026-09-30、单边5bps的历史年化12.49%、零无风险利率夏普1.77、最大回撤4.22%；近期回撤未改善，这些年份均已参加研究。必选前2名仅作历史对照，未启用。

排名缓冲以旧周目标为状态。`data/rotation_state.json` 保存9月24日旧目标（纳指、豆粕、有色），从下一周9月30日开始应用新规则；每天运行都从该锚点逐周重放，不会把未执行的计划当成券商成交。请备份该文件，缺失时程序会停止发布。回测始终从固定历史起点重建选仓，再截取指定收益区间；切换前的回测是新规则的假设历史，不代表真实账户曾使用新规则。

市场状态与 PCR 研究的可复用因子仍在 `etf_rotation/`，研究结论与限制见 [RESEARCH.md](RESEARCH.md)。未入选的方案运行脚本和中间表格已清理；这些候选因子未接入当前实盘信号。

宽基收益图与最大回撤图位于 `output/benchmark_comparison/`，同时保存逐日净值、汇总表及数据检查记录。使用 `plot_benchmarks.py --offline` 可从缓存重画；不加 `--offline` 可为六只宽基 ETF 增量更新数据。图中的上证50、沪深300、中证500、中证1000、创业板和科创50均由代表 ETF 的后复权价格买入持有表示，**不是指数官方全收益序列**。512100 在 2022-09-02 缺少报价，图表仅在估值时沿用前一日价格，并在 `data_checks.json` 标记。

复权收盘价用于研究连续收益，**不是可保证成交的价格**。模型未模拟 ETF 实盘价差、溢折价、整手数量、停牌时的真实成交和资金容量。历史低回撤不构成未来 5% 或 10% 的硬上限。
