# 每周实盘信号使用说明

当前默认策略：16只ETF池（含银行ETF `512800`）、20/40日动量、40日趋势、**7%年化风险仓波动预算＋前4名排名缓冲＋必选合格池第1名**。风险ETF最多3只，总目标仓位上限70%、单只目标仓位上限30%；防守资产为十年国债ETF，未分配资金留现金。7%是历史风险估计预算，历史回撤不构成未来回撤上限。

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

将示例 `100000` 换成实时账户总资产（全部持仓市值加现金，外部入金实际到账后计入），而非仅填现金或最初投入本金。PyCharm直接运行时可在运行配置中填入同样的 `--capital` 参数。

程序输出信号日、计划调仓日、各ETF目标比例、目标金额、现金及每100份的参考金额，保存到 `output/live_signal.json`。只有 `status=ready` 时才是可用于对应计划日的信号；`historical_only`、`no_new_rebalance`、`execution_window_closed` 均不能当作新计划。若行情、复权因子或日历不完整，程序拒绝发布。

调仓日以券商实时可成交价格换算份额：

```text
目标总份额 = 向下取整(目标金额 / 当前价格 / 100) × 100
买卖份额 = 目标总份额 − 现有份额
```

正数表示买入，负数表示卖出。确认现金与手续费后先处理超额仓位，再买入不足部分。国债一手金额可能较大，取整后的剩余资金暂留现金，并记录实际权重偏差。信号日收盘价只是参考，程序不会连接券商下单。

## 排名缓冲与初始化状态

先选入合格候选中的第1名，再按当前排名保留仍合格且位于前4名的旧周目标，最后补足最多3只。失去40日动量或40日均线趋势资格时，不能靠缓冲保留；周信号确认失败后在下一交易日收盘退出。当前没有每日或盘中止损。

`data/rotation_state.json` 保存从旧规则切换到新规则的公共初始化锚点及ETF代码，不含个人金额、份额或Token。程序每次从同一锚点逐周重放，缺少该文件会停止发布。JSON中的 `rank_buffer_size=4`、`mandatory_top_rank=1` 与 `rotation_state` 可用于核对版本。真实账户偏差另记，不能把计划等同于成交。

## 离线回测

新克隆后先联网下载，再离线复用缓存：

```powershell
.\.venv\Scripts\python.exe run_backtest.py --end 2026-09-30
.\.venv\Scripts\python.exe run_backtest.py --offline --end 2026-09-30
```

`plot_benchmarks.py --offline` 从本地行情与回测结果生成宽基对比，尚无所需行情时先联网运行相应脚本。研究命令和已完成比较见 [RESEARCH.md](RESEARCH.md)。仓库不包含个人生成报告或行情；文档所指的 `output/` 文件需在本地生成。

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
