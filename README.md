# A 股场内 ETF 动量轮动

当前默认代号：**澄流120（CL120）**。2026-11-09按用户指令，在原组合上增加**120日市场IVOL高四分位过滤**：沪深300单因子回归，只比较A股股票ETF；先过滤、再执行原评分与排名缓冲。新规则于11月9日收盘执行前启用，首次使用11月6日完整收盘数据修订尚未执行的目标，未改写此前已执行历史。[启用说明](docs/cl120_default_20261109.md)记录参数、边界和验证；原17只默认状态冻结在`data/rotation_state_20261109.json`。

当前默认策略：**17只ETF核心轮动＋512890红利低波独立卫星，共18只**。2026-10-31相关性研究后按用户指令移除尚未启用的513500标普500，保留新增513180恒生科技；首次新池信号2026-11-06、计划11月9日收盘执行，此前完整周计划仍按原16只重放。核心保留20/40日动量、8%波动目标预算、20日均线反转0.5、前4名缓冲必选第1名、最多3只风险ETF及70%总目标上限。红利继续**MA200、入0%、出+4%、不限日、先现金再国债、账户上限30%**，其原启用日期不变。叠加后最多4只风险ETF加国债共5只，风险总目标最高100%，全账户不受8%/70%硬约束。行情使用Tushare日线、同日复权因子和交易日历；588000未加入核心池。

新加入/删减对照见[相关性与共线性研究](docs/correlation_audit_20261031.md)。当前默认对应报告中的`drop_513500`方案；原18只研究池与差方案保留，便于复现。扩池和红利历史假设分别用`--hypothetical-universe`与`--hypothetical-satellite`显式开启，普通回测和实盘默认遵守实际启用日期。

## 在 PyCharm 中运行

1. 将本文件夹作为项目打开，设置 Python 解释器并在项目终端执行 `python -m pip install -r requirements.txt`。
2. 在 `data/tushare_token.txt` 放一行自己的 Tushare Token，或设置环境变量 `TUSHARE_TOKEN`。Token 文件已被 `.gitignore` 排除。
3. 右键运行 `run_backtest.py` 查看历史结果；右键运行 `live_signal.py` 获取下一次调仓计划。两个入口共用生产策略函数，但日期和输出用途不同；本地文件作用、运行参数和输出速查见 [LIVE_GUIDE.md](LIVE_GUIDE.md#本地文件与输出速查)。

新克隆的项目尚无行情缓存，第一次请联网运行 `python run_backtest.py --end 2026-09-30`，下载后再使用 `--offline`。每位成员使用自己的Tushare Token及数据权限。可将 `data/tushare_token.example.txt` 复制为 `data/tushare_token.txt` 后填写；也可在启动Python前设置环境变量 `TUSHARE_TOKEN`。程序不会自动读取 `.env`。

## 团队协作

仓库共享源代码、因子库定义、测试、策略初始化状态及研究结论。`data/`默认排除，仅共享`.gitkeep`、当前`rotation_state.json`、历史研究锚点`rotation_state_20261006.json`/`rotation_state_20261018.json`/`rotation_state_20261031.json`和空白Token示例；真实Token、个人账户数据库与快照、行情缓存、因子缓存、生成报告、虚拟环境均保留在成员本地。

`rotation_state.json` 只含公开策略参数和周目标代码，用于一致重放排名缓冲；它不是账户持仓明细。更改选仓规则或这个锚点时，请在分支上修改并通过Pull Request讨论。协作步骤见 [CONTRIBUTING.md](CONTRIBUTING.md)。文档中的 `output/` 路径是本地生成结果，克隆后需自行运行对应脚本；仓库未分发Tushare行情。

终端示例：

```powershell
python run_backtest.py --offline --end 2026-09-30
python plot_benchmarks.py --offline
python live_signal.py --capital 100000
python live_signal.py --capital 100000 --show-scores
python -m unittest discover -s tests
```

`--offline` 只读本地 Tushare 缓存。平时不加该参数即可增量更新：当结束日期向后延长时，程序复查最后 21 个自然日并追加新数据；`--refresh` 才会重新下载完整区间。若当天收盘价已到而基金复权因子尚未更新，程序会停止并提示，等数据完整后重试。缓存保存在 `data/cache/`，因子缓存在 `data/factors.sqlite`。

真实账户的周度持仓、入出金和收益另用 `portfolio_tracker.py` 记在 `data/live_portfolio.sqlite`。录入模板是 `portfolio_entry.example.json`；操作步骤和收益口径见 [LIVE_GUIDE.md](LIVE_GUIDE.md)。这部分只读你提供的券商记录，不会把回测净值当作真实账户收益。

因子库在`etf_rotation/factors.py`定义十九种价格因子：动量、均线趋势、年化波动率、下行波动率、前高突破、单窗口风险调整动量、趋势效率、短长波动率比、均线偏离标准分、10/30日与5/10日均线同侧持续性、滚动高点回撤与低点涨幅、短期反转、简单移动平均RSI、市场残差反转、回撤后反弹确认、相对大盘强弱、涨跌波动平衡。`FactorSpec`指定因子、交易日窗口及跳过天数；`FactorStore`按需计算并写入SQLite，价格矩阵变化时失效重算。默认实际调用20/40日动量、40日趋势、60日波动率、20日均线偏离标准分及国债60日动量。相对强弱等其他因子供研究。实盘JSON包含原始评分、均线修正、最终评分和资格排名，终端加`--show-scores`可查看合格标的评分。

另保留RSRS高低价因子：`compute_rsrs(high, low, RSRSSpec(N, M))`返回斜率、拟合R²、Z标准分、Z×R²修正版及Z×R²×斜率右偏版；`FactorStore.get_rsrs`按高低价、窗口及版本缓存。高低价须真实取得并使用同日复权因子，不能用收盘价代替。默认策略未调用RSRS。[RSRS研究报告](docs/rsrs_research_20261031.md)保留91场景及全部窗口/成本对照；入口`python research_rsrs.py --download`，复现用`python research_rsrs.py`。

量能、高波动反转、IVOL残差代理及MACD因子在`etf_rotation/extended_factors.py`定义，使用`ExtendedFactorSpec`、`compute_extended`与`FactorStore.get_extended`调用。IVOL主要在A股ETF同类内比较，市场/成长代理不是FF3或期权隐含波动率。[新因子研究](docs/factor_extensions_20261109.md)包含117个固定场景及后设解释对照；首次运行`python research_factor_extensions.py --download-only`，再运行`python research_factor_extensions.py`。当前默认采用市场IVOL120资格过滤，其余新增增强继续供研究；旧研究固定调用切换前17只状态，避免改变原比较基线。

2026-11-09先登记`ivol_market_120_exclude_top_quartile`为候选，同日按用户后续指令启用为澄流120；[参数与状态记录](docs/strategy_candidates.md)保留。在原核心选仓前排除A股同类中市场回归残差波动最高的四分位尾部，不叠加新评分。普通回测按实际启用日期重放；全历史假设须显式加`--hypothetical-ivol`，完整当前方案假设还应加`--hypothetical-universe --hypothetical-satellite`。

默认策略的年化、夏普、卡玛、回撤、完整ETF回合胜率/盈亏比及换手见[七项指标报告](docs/performance_metrics_current.md)，入口`python analyze_performance.py`。该入口离线统计，不切换策略；将账户周期与ETF持仓回合、双边换手与单边等效、金额盈亏比与收益率盈亏比分开，闭合和开放回合均核对终值。

## 规则

- 风险资产先通过40日动量为正、价格高于40日均线的资格筛选。原评分为20/40日动量均值除以60日历史年化波动率（下限5%），再加`0.5 × D × clip(-Z20/3,-1,1)`；`Z20`为20日均线价格标准偏离，`D`为当日已启用风险ETF原评分截面标准差（扩池前15只、扩池后16只，国债及红利卫星不参与）。短期偏热扣分、趋势内回调加分。原评分、修正及选仓只使用信号日及以前数据。
- 按修正后的评分先选入合格池第1名；剩余最多2个名额优先保留仍合格且位于前4名的旧周目标，再补足3只。资格失败不能靠缓冲保留，第1名不保证权重最大。风险仓内部按逆波动率分配，以过去60日协方差估计波动。沪深300低于40日均线时，风险仓目标再减半。
- 核心使用**8%年化波动目标预算**、**70%总目标仓位上限**和**单只风险ETF 30%目标上限**。单只超限部分先分配给其他已入选且未触顶的核心ETF；剩余资金在国债 `511260` 的60日动量为正时配置国债，否则留现金。最终限仓可能改变预测波动，8%是目标预算，不能视为波动或回撤硬上限。
- 红利512890独立计算真实报价复权收盘价相对MA200偏离，周信号偏离≤0%入场，持有至偏离≥+4%退出，不限日；每周目标为账户30%，先用现金再减少国债，可用防守资金不足时少买。原核心风险ETF名单与目标权重全部保留，红利不参加核心排名、40日趋势资格或截面评分。叠加后风险总目标最高100%，最多5只ETF；目标权重在非调仓日可自然漂移。
- 每个交易周最后一个开市日收盘后生成信号，**下一交易日收盘**按目标权重模拟成交。默认单边手续费 3 bps、滑点 2 bps；已有仓位在两次调仓之间自然漂移。回测与实盘信号均不会连接券商下单。

标的池在 [etf_rotation/config.py](etf_rotation/config.py)：`510300` 沪深300、`159915` 创业板、`159941` 纳指、`518880` 黄金、`511260` 十年国债、`159995` 半导体、`159875` 新能源、`159928` 消费、`512660` 军工、`515070` 人工智能、`512880` 券商、`512800` 银行、`513520` 日经、`561360` 石油产业、`159985` 豆粕期货、`159980` 有色。`510300` 还用作大盘趋势过滤；新上市 ETF 在积累足够历史数据前不会入选。`512800` 是风险 ETF，不充当国债防守仓。

## 文件与输出

| 文件 | 用途 |
| --- | --- |
| `etf_rotation/data.py` | Tushare 下载、同日基金复权和增量缓存 |
| `etf_rotation/factors.py`、`factor_store.py` | 因子计算与 SQLite 因子库 |
| `etf_rotation/predictive_factors.py` | 研究用途：只用既有周收益滚动拟合跨行业 PCR 预测 |
| `etf_rotation/market_factors.py` | 研究用途：四种只用历史收盘价的大盘状态候选因子 |
| `etf_rotation/risk_strategy.py` | 当前17只核心ETF的动量、缓冲、启用资格与防守规则 |
| `etf_rotation/portfolio_strategy.py`、`mean_reversion.py` | 实盘/回测共用的红利状态机与国债现金拨款 |
| `etf_rotation/strategy_state.py`、`data/rotation_state.json` | 保存切换锚点，从旧周目标重放排名缓冲 |
| `etf_rotation/backtest.py` | 下一交易日成交、费用和缺价处理 |
| `run_backtest.py` | PyCharm 回测入口 |
| `run_backtrader.py`、`etf_rotation/backtrader_engine.py` | 原生Backtrader订单与现金记账的独立对照 |
| `plot_benchmarks.py` | 将当前回测与六类宽基 ETF 买入持有收益画在同一张图 |
| `live_signal.py` | 盘后调仓计划入口 |
| `portfolio_tracker.py` | 录入实际持仓、入出金，查看真实账户收益 |

回测输出位于 `output/backtest/`：`equity.csv`、`daily_weights.csv`、`signal_weights.csv`、`trades.csv`、`summary.json`，有 Matplotlib 时还会生成 `equity.png`。实盘计划位于 `output/live_signal.json`。历史表现与限制见 [RESEARCH.md](RESEARCH.md)；实际操作步骤见 [LIVE_GUIDE.md](LIVE_GUIDE.md)。

默认回测仅从2026-10-23信号开始叠加红利，避免把研究中的虚拟旧仓当作实盘持仓。若要复现新方案2022年以来的全历史假设表现，运行`python run_backtest.py --offline --end 2026-10-20 --hypothetical-satellite --output-dir output/backtest_selected_dividend_model`；该模式重现年化17.57%、夏普2.006、收盘回撤4.43%，不是历史实盘记录。实盘JSON的`satellite_details`显示4%退出、30%上限、启用日期、MA偏离与状态。

当前默认与六类宽基、固定80%科技重仓示例及红利边际归因见[收益回撤与归因报告](docs/default_benchmark_attribution_20261021.md)。运行`python plot_benchmarks.py --dividend-attribution --offline --end 2026-10-20`可复现；首次缺少宽基行情时去掉`--offline`增量更新。全历史假设、原核心、每个持仓周期的有利与不利效果同时保留，实盘策略不变。

当前策略的稳健性审计保留在 `research_strategy_audit.py`，离线结果在 `output/strategy_audit/`。已结束的均线回归、事件仓和 588000 扩池探索保留结论于 [RESEARCH.md](RESEARCH.md)，不再保留中间研究脚本和输出。

2026-10-17反转与反弹研究见[最终报告](docs/reversal_research_20261017.md)，包含银行事件检验、50项固定比较、10项后设分类补充、时间分段、交易成本、IC/R²及单标的依赖检查。首轮历史最佳的20日均线反转年化13.05%、夏普1.82、回撤3.96%，但参数、后段及统计证据不足，默认实盘保持原规则。统一复现入口为 `python research_reversal.py --end 2026-10-16`；`--summarize` 仅重建报告，`--classified-only` 仅重现已有首轮结果后的分类补充。最终汇总表位于 `output/reversal_20261017/`，弱项只保留统计记录，未增加独立默认策略分支。

8%预算研究使用`python research_vol8.py --end 2026-10-16`，报告见[8%预算研究](docs/vol8_research_20261017.md)，数表位于`output/vol8_20261017/`；`--report-only`重建报告和图。比较35组固定参数及逐年选择路径，所有年份都已参与历史研究。2026-10-18用户已选择8%＋20日均线反转0.5作为默认。原7%研究基线冻结为`RiskConfig.buffered_momentum()`及`data/rotation_state_20261006.json`，防止重复叠加均线因子或改写旧结果。

2026-10-21红利低波长期均线探索见[研究报告](docs/dividend_reversion_20261021.md)。主研究512890，用515100及515450复核；比较150/200日均线、偏离入出场、持有期和卫星配置方式，共72个周度受约束组合。复用因子库的`trend`和独立状态机`etf_rotation/mean_reversion.py`，统一入口为`python research_dividend_reversion.py --end 2026-10-20`；首次先加`--download-only`更新数据，`--report-only`从汇总重建报告和图。结果在`output/dividend_reversion_20261021/`。候选未接入默认实盘；完整失败项、开放周期及后设对照均保留供审计。

用户指定的MA200入0%、出+4%/+5%、替换风险仓上限40%补充见[40%上限报告](docs/dividend_cap40_20261021.md)，运行`python research_dividend_reversion.py --cap40 --end 2026-10-20`。仅红利上限放宽，其他风险约束保留，正式默认未变。

从国债/现金拨款、保持原核心目标的20%/30%卫星补充见[资金来源对照报告](docs/dividend_defensive_funding_20261021.md)，运行`python research_dividend_reversion.py --defensive-funding --end 2026-10-20`。主研究允许全账户超过原8%/70%额度并最多持有5只ETF，另列只使用剩余风险额度的对照。2026-10-21用户选择其中出+4%、上限30%成为新默认，其余为历史对照。

2026-10-06 的扩池及规则优化研究保留一个复现入口 `research_strategy_optimization.py`，固定比较 18 个方案；中文报告、对比图及稳健性结果在 `output/optimization_20261006/`。可运行 `python research_strategy_optimization.py --offline --robustness --end 2026-09-30` 复现，或用 `--summarize` 仅重生成报告和图。这些候选没有接入默认实盘信号。

7%预算与原排名缓冲的历史比较使用同一入口：`python research_strategy_optimization.py --combined --offline --robustness --end 2026-09-30`。报告和收益/回撤图位于 `output/optimization_20261006/combined_7pct/`；`--combined --summarize` 可仅重生成报告。原缓冲历史年化12.54%、夏普1.78、最大回撤4.85%，高成本压力下回撤5.47%；它与现在的必选第1名版本不同。研究基线冻结为切换前6%无缓冲版本，避免升级实盘后改写旧比较。

最近回撤归因可运行 `python research_strategy_optimization.py --recent-audit --offline --end 2026-09-30`，结果位于上述目录的 `recent_drawdown/`。它核对原净值、分解ETF收益贡献，并记录保留第4名后的收益及周中筛选失败至退出的时间；仅作解释，不更改默认规则。

强制最强候选入选的历史研究使用`python research_strategy_optimization.py --priority-buffer --offline --robustness --end 2026-09-30`，报告在`output/optimization_20261006/priority_buffer/`。7%＋缓冲必选第1名曾于2026-10-06接入默认，现作为旧版对照；其2022-01-04至2026-09-30、单边5bps历史年化12.49%、零利率夏普1.77、最大回撤4.22%。必选前2名仅作历史对照，未启用。

排名缓冲以旧周目标为状态。`data/rotation_state.json`保留9月24日历史初始化记录，并追加本次10月9日旧目标（纳指、银行、豆粕）作为切换锚点，从10月16日信号开始生效，10月19日执行。每天按时间顺序重放各次锚点，后一次切换不会改写此前信号；周目标不等同于券商实际成交。请备份该文件，缺失时程序停止发布。回测从固定历史起点重建，再截取收益区间；新策略的历史回测不是账户真实使用记录。

市场状态与 PCR 研究的可复用因子仍在 `etf_rotation/`，研究结论与限制见 [RESEARCH.md](RESEARCH.md)。未入选的方案运行脚本和中间表格已清理；这些候选因子未接入当前实盘信号。

宽基收益图与最大回撤图位于 `output/benchmark_comparison/`，同时保存逐日净值、汇总表及数据检查记录。使用 `plot_benchmarks.py --offline` 可从缓存重画；不加 `--offline` 可为六只宽基 ETF 增量更新数据。图中的上证50、沪深300、中证500、中证1000、创业板和科创50均由代表 ETF 的后复权价格买入持有表示，**不是指数官方全收益序列**。512100 在 2022-09-02 缺少报价，图表仅在估值时沿用前一日价格，并在 `data_checks.json` 标记。

复权收盘价用于研究连续收益，**不是可保证成交的价格**。模型未模拟 ETF 实盘价差、溢折价、整手数量、停牌时的真实成交和资金容量。历史低回撤不构成未来 5% 或 10% 的硬上限。

## 过拟合与参数审计

2026-10-21的[完整审计报告](docs/overfitting_audit_20261021.md)按用户指定以2018年至今为主要研究范围，包含7个配置对照、125个参数点复验及2018—2021/2022后稳定性检查；此前554个案例按原区间保留，其中2022后的选池、IC/R²、多重比较及执行诊断属于开发期证据。2018年前结果仅留作附录，历史行情可用于预热因子和状态。只复验主区间可运行 `python research_overfitting_audit.py --focus-2018 --end 2026-10-20`，完整入口为 `python research_overfitting_audit.py --end 2026-10-20`；独立历史缓存下载加 `--download-history`，已有结果重建报告加 `--report-only`。该审计没有修改默认策略。2018年以来夏普约1.32、最大回撤约10.11%，不能把近年夏普2和约5%回撤视为稳定实盘水平。

## Backtrader独立复核

2026-10-23的[旧16只Backtrader报告](docs/backtrader_comparison_20261023.md)保留。当前17只独立复核见[当前报告](docs/backtrader_comparison_current.md)。运行 `python run_backtrader.py`，或在PyCharm直接运行该文件；先按 `requirements.txt` 安装依赖。默认只读覆盖17只核心及红利的`data/audit_history`缓存，不自动调用Tushare。当前输出在`output/backtrader_comparison_current/`，不覆盖旧报告。

默认是全历史采用当前红利模块的假设研究，`--from-adoption`按实际启用日期回放，与 `run_backtest.py` 的默认启用模式对应。收盘目标比例场景保留原模型按成交参考价确定数量的理想假设，使用原生经纪商订单与现金记账；另列前日锁定份额的原生 `Close` 订单场景及资金不足拒单记录。没有用同日新信号同日成交。无成本逐日净值一致，单边5bps下年化差不到0.0002个百分点；这一复核检验实现，不构成过拟合已被排除或未来盈利的证明。

## 行业扩池研究

2026-10-31的[行业覆盖与扩池报告](docs/pool_expansion_20261031.md)测试医药、地产、煤炭、电力、化工五个规模较大且2021年末前成立的ETF，包含32种池组合、历史准入和费用对照，行情截止2026-10-20。本轮没有改善默认的历史夏普，保留原池；候选保留在研究入口而非实盘配置。复现使用`python research_pool_expansion.py --download`准备独立数据，再运行`python research_pool_expansion.py`；后设缩放诊断加`--diagnose-spread`，最后用`--report-only`重建报告。完整107项测试通过，不代表选池后视或过拟合已消除。

同日的[境外指数扩池报告](docs/overseas_expansion_20261031.md)覆盖标普500、恒生、德国、法国及恒科/中概风格与日本指数替换，共20个固定场景。恒科有小幅历史改善，但不确定区间包含零；标普＋恒生较依赖2022后区间。按用户要求不更改默认。入口`research_overseas_expansion.py`先加`--download`准备独立行情，再无参数运行主试验，`--diagnostics`保存后设解释与区间，`--report-only`生成报告/图。43项研究检查、完整110项测试通过；没有把同日晚间海外收盘或尚未公布NAV提前用于信号。
