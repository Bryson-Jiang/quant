# 团队协作

## 首次使用

1. 克隆私有仓库 `Bryson-Jiang/quant`，用PyCharm打开项目。
2. 创建本地虚拟环境，安装 `requirements.txt`。
3. 使用自己的Tushare账号、Token和数据权限，参见 [LIVE_GUIDE.md](LIVE_GUIDE.md)。Token放入 `data/tushare_token.txt` 或环境变量 `TUSHARE_TOKEN`。
4. 首次联网下载行情后再使用 `--offline`。运行 `python -m unittest discover -s tests` 检查本地环境，测试使用构造行情与临时目录，无需真实账户或Tushare Token。

## 提交修改

从 `main` 创建描述清楚的分支，围绕一个研究问题或修复提交，通过Pull Request合并。策略或因子变化说明公式、数据截止日、成交时点、费用、对照组及样本选择限制；避免只汇报最优参数。

通用因子放入 `etf_rotation/factors.py`，研究结果写入 `RESEARCH.md`。涉及默认策略时核对 `risk_strategy.py`、`config.py`、实盘和回测入口；排名缓冲依赖历史状态，修改 `rotation_state.json` 会改变未来选仓，需在PR中解释并核对截断及未来扰动检查。

## 共享范围

- 提交源码、测试、依赖清单、策略初始化状态、模板与研究结论。
- 真实Token、个人账户数据库与快照、行情/因子缓存、生成输出、虚拟环境与本机IDE配置由 `.gitignore` 排除。
- `data/` 使用允许清单：只有 `.gitkeep`、公共策略初始化状态及Token空白示例可提交。新增公共数据文件前先讨论用途与数据权限。
- 查看差异时留意手工写入的密钥、个人资产与本机路径；不要用 `git add -f` 强行加入本地敏感文件。

回测使用连续权重和日收盘净值，历史结果已经历多次研究选择。整手、最低佣金、跨境溢折价、数据修订和盘中回撤等限制须随研究结论一并说明。
