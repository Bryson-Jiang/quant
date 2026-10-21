# 每周实盘信号使用说明

当前默认：**16只核心轮动＋512890独立红利反转**。核心继续用20/40日动量、40日趋势、8%波动目标预算、20日均线反转0.5及前4名缓冲必选第1名。2026-10-21选定红利规则：**MA200／入0%／出+4%／不限日／先现金再国债／账户上限30%**。原核心风险目标保持不变，因此整个账户可能最高100%风险仓、最多4只风险ETF加国债共5只；8%/70%只描述核心目标，不能当作全账户限制。

首次新规则信号为**2026-10-23收盘后**、计划**10月26日接近收盘执行**。卫星从未持有状态启动，不继承研究里的虚拟旧仓；必须在启用后的周信号触发入场才买入。

## 准备本地环境

1. 在PyCharm为项目创建虚拟环境，执行 `python -m pip install -r requirements.txt`。
2. 将 `data/tushare_token.example.txt` 复制为 `data/tushare_token.txt`，填入自己的Token；也可使用环境变量 `TUSHARE_TOKEN`。程序不会自动加载 `.env`。
3. 每位成员需具备Tushare ETF日线、基金复权因子和交易日历对应权限。首次联网运行会下载历史，之后在区间延长时增量更新并复查最近21个自然日。

Token、行情缓存、因子数据库、个人账户明细及 `output/` 均保留在本机。账户数据不用于共享策略初始化。

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
