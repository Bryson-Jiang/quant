# 每周实盘信号使用说明

当前默认：**17只核心轮动＋512890独立红利反转，共18只**。2026-10-31按用户指令删除尚未启用的513500标普500，在原16只基础上仅新增513180恒生科技，首次纳入信号2026-11-06、首次计划执行11月9日接近收盘；此前周目标仍按原16只。核心继续用20/40日动量、40日趋势、8%波动目标预算、20日均线反转0.5及前4名缓冲必选第1名。红利规则仍为**MA200／入0%／出+4%／不限日／先现金再国债／账户上限30%**。整个账户可能最高100%风险仓、最多4只风险ETF加国债共5只；8%/70%只描述核心目标，不能当作全账户限制。

红利模块原首次信号为**2026-10-23收盘后**、计划**10月26日接近收盘执行**。本次扩池不重置红利持有状态或原核心缓冲，不继承研究里的虚拟持仓。

日常仍运行`live_signal.py --capital 总资产 --show-scores`。JSON中`core_universe`为配置的17只，`active_core_universe`为该信号日已启用的成员，`applied_strategy`反映实际应用版本。10月30日信号不因周末改池而改写；不要手动按研究回测中的虚拟仓位提前交易。

普通回测默认按实际启用日期。研究新17只完整历史时同时加`--hypothetical-universe --hypothetical-satellite`；只有后一个参数仅放开红利历史，仍遵守核心扩池日期。旧16只研究及相关性审计中的18只研究池均已冻结，不受新默认影响。

## 准备本地环境

1. 在PyCharm为项目创建虚拟环境，执行 `python -m pip install -r requirements.txt`。
2. 将 `data/tushare_token.example.txt` 复制为 `data/tushare_token.txt`，填入自己的Token；也可使用环境变量 `TUSHARE_TOKEN`。程序不会自动加载 `.env`。
3. 每位成员需具备Tushare ETF日线、基金复权因子和交易日历对应权限。首次联网运行会下载历史，之后在区间延长时增量更新并复查最近21个自然日。

Token、行情缓存、因子数据库、个人账户明细及 `output/` 均保留在本机。账户数据不用于共享策略初始化。

## 本地文件与输出速查

先在PowerShell或PyCharm的Terminal切换到项目根目录，再使用本项目解释器。下列项目路径是示意，请替换为实际路径：

```powershell
cd C:\path\to\Quant
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

若已有 `.venv` 无需重建；若在新电脑尚未建立环境，先执行 `py -m venv .venv`。PyCharm的项目解释器选择本项目 `.venv\Scripts\python.exe`，运行配置的工作目录选项目根目录。直接指定解释器不需要激活PowerShell虚拟环境。

### 经常使用的入口

| 文件 | 何时运行 | 主要输出 |
| --- | --- | --- |
| `live_signal.py` | 每周盘后检查计划，日常也可查询状态 | 终端显示状态、信号日、执行日；有效时显示目标比例/金额及评分。完整计划保存在 `output/live_signal.json` |
| `run_backtest.py` | 检查历史表现 | `output/backtest/` 下的 `equity.csv`、`daily_weights.csv`、`signal_weights.csv`、`trades.csv`、`summary.json`、`equity.png` |
| `run_backtrader.py` | 用独立订单和现金引擎复核当前17只核心池 | `output/backtrader_comparison_current/` 下的 `metrics.csv`、`reconciliation.csv`、逐日净值/实际权重/订单等；报告在 `docs/backtrader_comparison_current.md`，旧16只报告单独保留 |
| `plot_benchmarks.py` | 已有回测结果后，与常见宽基比较 | 默认 `output/benchmark_comparison/` 下的 `broad_market_comparison.png`、`drawdown_comparison.png`、`comparison_summary.csv`、`comparison_nav.csv` |
| `portfolio_tracker.py` | 录入券商真实持仓、入出金并统计真实收益 | 数据库 `data/live_portfolio.sqlite`；`report --export` 导出 `output/live_portfolio_history.csv`、`output/live_portfolio_positions.csv` |
| `research_overfitting_audit.py` | 重新做2018年以来的参数和过拟合研究 | `output/overfitting_audit_20261021/` 数表与检查记录；报告在 `docs/overfitting_audit_20261021.md` |
| `research_rsrs.py` | 探索RSRS高低价评分与不同窗口，首次加`--download` | `output/rsrs_20261031/`全部场景、IC、预测R²、预热与成本对照；报告在`docs/rsrs_research_20261031.md`，不改默认 |
| `analyze_performance.py` | 离线计算默认策略七项指标，核对ETF完整回合及账户周期 | `output/performance_metrics_current/`指标JSON、闭合/开放回合、账户周期和年度汇总；报告`docs/performance_metrics_current.md`；需匹配截止日的Backtrader订单 |

`research_reversal.py`、`research_vol8.py`、`research_dividend_reversion.py`、`research_strategy_optimization.py` 等是历史研究复现入口，有各自冻结的基线；平时不用逐个运行，也不会自动切换实盘默认。查看某入口支持的参数可在命令末尾加 `--help`。

### 策略内部模块

`etf_rotation/` 内主要文件由上述入口自动调用，通常不直接右键运行：

- `config.py`：ETF代码、名称及策略版本；`risk_strategy.py`：核心评分、资格过滤、排名缓冲和风险仓配置。
- `portfolio_strategy.py`、`mean_reversion.py`：叠加红利低波卫星及其入出场状态，组合成完整目标。
- `factors.py`：十九种因子的定义；`factor_store.py`：计算结果缓存到 `data/factors.sqlite`，不是实盘账户数据库。
- `data.py`：Tushare行情、同日复权因子、校验与增量缓存。
- `backtest.py`：自写成交/费用/净值引擎；`backtrader_engine.py`：Backtrader对照；`report.py`：保存回测表格和图片。
- `strategy_state.py`与`data/rotation_state.json`：重放缓冲锚点和红利启用状态。请保留并备份，真实持仓另行记录。

### 推荐命令

**日常获取计划**，示例总资产100000元，应改成券商当前持仓市值加现金：

```powershell
.\.venv\Scripts\python.exe live_signal.py --capital 100000 --show-scores
```

`--capital`只影响金额换算，不改变策略目标比例。不会自动读取券商持仓，也不会给出已扣除现有份额的买卖差额；需按本指南的份额公式另算。

实盘入口按当前有效周信号日取行情，非调仓日不一定下载当天行情。`output/live_signal.json`会被本次计划覆盖，账户历史则另记在SQLite数据库中。

**用现有缓存重现固定历史区间**，显式采用全历史红利假设：

```powershell
.\.venv\Scripts\python.exe run_backtest.py --offline --end 2026-10-20 --hypothetical-satellite
.\.venv\Scripts\python.exe plot_benchmarks.py --offline
```

普通回测无参数时从2022年开始，默认终点为运行日，默认按红利及核心扩池实际启用日期回放。上例只放开红利历史，截至10月20日仍重现原16只池；研究新17只完整历史再加`--hypothetical-universe`。`plot_benchmarks.py`默认读取刚生成的 `output/backtest/`，首次缺少宽基行情时去掉绘图命令的 `--offline`；指定其他回测输出目录后，绘图也要传对应的 `--backtest-dir`。

**按主要研究范围从2018年回测**，使用已准备的独立历史缓存：

```powershell
.\.venv\Scripts\python.exe run_backtest.py --start 2018-01-01 --end 2026-10-20 --cache-dir data/audit_history --offline --hypothetical-satellite --output-dir output/backtest_2018
```

2018年前的数据只用于指标/状态预热，不计入2018起始账户收益。该结果的净值和沪深300对比看 `output/backtest_2018/equity.png`。默认六宽基绘图包含不能覆盖2018全区间的科创50ETF，因此不能直接套用到2018起始回测；完整六宽基比较使用前面的2022起始示例。

相同输出目录下重复回测会覆盖文件；保留多次结果时，用不同的 `--output-dir`。

**Backtrader复核**：

```powershell
.\.venv\Scripts\python.exe run_backtrader.py --end 2026-10-20
```

它默认从2018年开始，使用 `data/audit_history/` 的共同缓存终点，**不会联网刷新行情**，也不会自动复用 `live_signal.py` 更新的 `data/cache/`。默认整段采用当前红利规则作历史假设；加 `--from-adoption` 才按实际启用日期回放。对照框架时应统一区间、缓存、费用与红利模式。

**2018主区间参数复验**：

```powershell
.\.venv\Scripts\python.exe research_overfitting_audit.py --focus-2018 --end 2026-10-20
```

包含7个配置对照与125个参数点，耗时比日常信号更长；不需要每周重跑。现有 `data/audit_history/` 缺失时，可先用该脚本的 `--download-history --end 2026-10-20` 准备独立历史缓存。

### PyCharm参数填写

右键入口文件运行一次，再在其运行配置中填写参数。例如 `live_signal.py` 的脚本参数填写：

```text
--capital 100000 --show-scores
```

参数栏不再填写 `python`、解释器路径或脚本文件名。回测参数也用同样方式填写。终端看运行日志，项目文件树中打开 `output/` 看JSON、CSV和PNG；CSV可用Excel查看，Markdown研究报告在 `docs/`。

以上带 `2026-10-20` 的命令用于复现固定历史结果，不是今天的实盘建议。实时计划使用 `live_signal.py`；`--offline`始终只读取已有缓存，不会自动补新数据。

## 每周运行

每个交易周最后一个开市日收盘后生成信号，下一交易日接近收盘执行。节假日按交易日历调整，不能机械理解为固定每隔五个交易日交易。

在上海时间18:00以后等待ETF日线及复权因子完整发布，再运行：

```powershell
.\.venv\Scripts\python.exe live_signal.py --capital 100000
```

这条命令已默认使用核心＋红利方案，无需额外传参数。加`--show-scores`查看核心动量评分和排名；红利独立输出MA200偏离、入出场状态及目标仓位。JSON中`satellite_details.exit_deviation=0.04`、`max_weight=0.3`、`funding=cash_then_bond`可核对版本；`target_vol=0.08`仅是核心预算。修改`--target-vol`不会缩减或取消红利模块。

将示例 `100000` 换成实时账户总资产（全部持仓市值加现金，外部入金实际到账后计入），而非仅填现金或最初投入本金。PyCharm直接运行时可在运行配置中填入同样的 `--capital` 参数。

程序输出信号日、计划调仓日、各ETF目标比例、目标金额、现金及每100份的参考金额，保存到 `output/live_signal.json`。只有 `status=ready` 时才是可用于对应计划日的信号；`historical_only`、`no_new_rebalance`、`execution_window_closed` 均不能当作新计划。若行情、复权因子或日历不完整，程序拒绝发布。

调仓日以券商实时可成交价格换算份额：

```text
目标总份额 = 向下取整(目标金额 / 当前价格 / 100) × 100
买卖份额 = 目标总份额 − 现有份额
```

正数表示买入，负数表示卖出。确认现金与手续费后先处理超额仓位，再买入不足部分。国债一手金额可能较大，取整后的剩余资金暂留现金，并记录实际权重偏差。信号日收盘价只是参考，程序不会连接券商下单。

## 排名缓冲与初始化状态

核心先选入合格候选中的第1名，再保留仍合格且位于前4名的旧周目标，最后补足最多3只。失去40日动量或40日均线趋势资格时，不能靠缓冲保留。红利单独判断：周信号偏离MA200≤0%入场，持有至偏离≥+4%退出，每周按账户30%恢复目标，资金先现金后国债。周中触及退出线不会触发每日或盘中交易；均线在移动，超过均线4%不等于赚4%。

`data/rotation_state.json`保存核心公共缓冲锚点及红利启用日期，不含个人金额、份额或Token。原9月24日、10月9日核心锚点均保留；新增红利从10月23日信号开始以未持有状态重放。缺少状态文件会停止发布。`rotation_state_20261006.json`及`rotation_state_20261018.json`为旧研究冻结状态，平时无需更改。真实账户偏差另记，不能把计划等同于成交。

## 离线回测

新克隆后先联网下载，再离线复用缓存：

```powershell
.\.venv\Scripts\python.exe run_backtest.py --end 2026-09-30
.\.venv\Scripts\python.exe run_backtest.py --offline --end 2026-09-30
```

默认仅从红利实际启用日期开始叠加。要复现新方案的全历史假设表现，请显式运行：

```powershell
.\.venv\Scripts\python.exe run_backtest.py --offline --end 2026-10-20 --hypothetical-satellite --output-dir output\backtest_selected_dividend_model
```

全历史假设模式在2022-01-04至2026-10-20、单边5bps下，年化17.57%、夏普2.006、收盘最大回撤4.43%。这不是策略实际启用记录，更不是账户真实收益；默认重放模式在启用前仍沿用原核心。

`plot_benchmarks.py --offline` 从本地行情与回测结果生成宽基对比，尚无所需行情时先联网运行相应脚本。研究命令和已完成比较见 [RESEARCH.md](RESEARCH.md)。仓库不包含个人生成报告或行情；文档所指的 `output/` 文件需在本地生成。

回测打印的“期末持仓”是截至最后一个回测收盘已执行的持仓；该日刚生成、下一交易日才执行的目标请看`live_signal.py`，两者可能不同。

## 记录个人账户收益

每位成员独立使用本地 `data/live_portfolio.sqlite`。新建个人数据库：

```powershell
.\.venv\Scripts\python.exe portfolio_tracker.py init
```

将 `portfolio_entry.example.json` 复制到 `data/portfolio_entry_YYYY-MM-DD.json`，用券商真实账户快照填写日期、总资产、现金、ETF份额、现价和市值，可附成本价与浮动盈亏。模板数值为虚构示例。第一份收盘快照可作为收益起点，首笔快照的 `cash_flows` 留空；以后填写自上一份快照以来每笔外部入金（正）或出金（负）的日期和金额。买卖、佣金与留在账户内的分红不是外部入出金。

```powershell
.\.venv\Scripts\python.exe portfolio_tracker.py record --file data/portfolio_entry_YYYY-MM-DD.json
.\.venv\Scripts\python.exe portfolio_tracker.py report --export
```

已有数据库可直接继续录入，不必重新初始化。若确认首次买入前真实资产，并希望采用投资前基准，可使用 `portfolio_tracker.py opening --date YYYY-MM-DD --equity 实际金额`；若已选择首笔收盘快照为起点，无需另设投资前基准。

同日重复录入会拒绝；核实后可用 `record --replace` 修正，旧JSON保存在 `snapshot_revisions` 表。报表导出到 `output/live_portfolio_history.csv` 和 `output/live_portfolio_positions.csv`。无外部入出金的相邻快照区间收益为精确账户收益；有入出金时使用Modified Dietz日期加权近似，不是逐笔现金流时点的精确TWR。最大回撤仅覆盖已记录快照。

关闭账户数据库读写程序后自行备份数据库。上传代码前查看 `git status`，个人数据库、快照、Token及本地文档备份应处于忽略状态；共享文档只保留通用操作与研究结论。
