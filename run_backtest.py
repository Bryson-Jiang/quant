"""PyCharm 直接右键运行本文件，或从终端传入参数。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
from pathlib import Path

from etf_rotation.backtest import run_backtest
from etf_rotation.config import ETF_NAMES, PORTFOLIO_ETF_NAMES, STRATEGY_NAME
from etf_rotation.factor_store import FactorStore
from etf_rotation.report import save_report
from etf_rotation.risk_strategy import RiskConfig
from etf_rotation.portfolio_strategy import load_portfolio_data, portfolio_signals
from etf_rotation.strategy_state import load_rotation_state


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="中国场内 ETF 多窗口动量轮动回测")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2022, 1, 1), help="回测起始日 YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, default=date.today(), help="回测结束日 YYYY-MM-DD")
    parser.add_argument("--target-vol", type=float, default=RiskConfig.current_default().target_vol,
                        help="核心年化风险仓波动预算8%%；红利卫星不受该全账户限制")
    parser.add_argument("--max-risky-weight", type=float, default=0.70,
                        help="核心风险ETF总仓位上限70%%；另有最多30%%红利卫星")
    parser.add_argument("--hypothetical-satellite",action="store_true",
                        help="显式模拟2022年以来均已采用红利卫星的假设历史；默认从实际启用日开始")
    parser.add_argument("--hypothetical-universe",action="store_true",
                        help="假设当前核心池全历史可用；默认仅从扩池实际生效日加入恒生科技")
    parser.add_argument("--fee-bps", type=float, default=3.0, help="单边手续费，基点")
    parser.add_argument("--slippage-bps", type=float, default=2.0, help="单边滑点，基点")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_DIR / "data" / "cache")
    parser.add_argument("--state-file", type=Path, default=PROJECT_DIR / "data" / "rotation_state.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "output" / "backtest")
    parser.add_argument("--factor-db", type=Path, default=PROJECT_DIR / "data" / "factors.sqlite",
                        help="因子库 SQLite 文件")
    parser.add_argument("--offline", action="store_true", help="仅用本地 Tushare 缓存")
    parser.add_argument("--refresh", action="store_true", help="全部标的重新下载")
    args = parser.parse_args()
    if args.offline and args.refresh:
        parser.error("--offline 和 --refresh 不能同时使用")
    return args


def main() -> None:
    args = parse_args()
    # 从固定起点重放排名缓冲，再截取收益区间；不能随 --start 改变旧选仓。
    history_start = min(args.start, date(2022, 1, 1))
    core_prices,satellite_history=load_portfolio_data(history_start,args.end,args.cache_dir,
                                                     offline=args.offline,refresh=args.refresh)
    store = FactorStore(args.factor_db)
    risk_config = RiskConfig.current_default(target_vol=args.target_vol,
                                             max_risky_weight=args.max_risky_weight)
    state = load_rotation_state(args.state_file)
    bundle=portfolio_signals(core_prices,satellite_history,state,risk_config,store,
                             hypothetical_satellite_history=args.hypothetical_satellite,
                             hypothetical_universe_history=args.hypothetical_universe)
    signals=bundle.signals;prices=bundle.prices
    mode="hypothetical_full_history" if args.hypothetical_satellite else "from_actual_adoption_date"
    print(f"核心参数：{risk_config}")
    print(f"红利卫星：{state.dividend_policy.metadata()}；回测模式：{mode}")
    print(f"标的池模式：{'hypothetical_full_history' if args.hypothetical_universe else 'from_actual_adoption_date'}")
    print("提示：默认仅从卫星实际启用日期叠加；--hypothetical-satellite为全历史假设测试，均不是账户实盘收益。")
    result = run_backtest(prices, signals, args.start, args.end,
                          fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
    reported_signals = signals.loc[signals.index <= result.equity.index[-1]]
    outputs = save_report(result, reported_signals, args.output_dir,
                          metadata={"strategy": {
                              "name": STRATEGY_NAME,
                              "config": asdict(risk_config),
                              "config_scope":"core_only",
                              "simulation_mode":mode,
                              "universe_simulation_mode":"hypothetical_full_history" if args.hypothetical_universe else "from_actual_adoption_date",
                              "dividend_policy":state.dividend_policy.metadata(),
                              "rotation_state": state.metadata(),
                              "universe":list(PORTFOLIO_ETF_NAMES),
                              "core_universe":list(ETF_NAMES),
                          }})

    print(f"因子库：{args.factor_db}")
    print(f"回测区间：{result.equity.index[0].date()} 至 {result.equity.index[-1].date()}")
    for label, key in (
        ("总收益", "total_return"), ("年化收益", "annual_return"),
        ("年化波动", "annual_volatility"), ("最大回撤", "max_drawdown"),
        ("基准收益", "benchmark_total_return"),
    ):
        print(f"{label}：{result.metrics[key]:.2%}")
    print(f"夏普比率（无风险利率为零）：{result.metrics['sharpe_zero_rf']:.2f}")
    print(f"换仓次数：{result.metrics['trade_count']}")
    if not result.weights.empty:
        latest = result.weights.iloc[-1]
        holdings = [f"{PORTFOLIO_ETF_NAMES.get(code, code)}({code}) {weight:.0%}"
                    for code, weight in latest.items() if code != "CASH" and weight > 0]
        print("期末持仓：" + ("、".join(holdings) if holdings else "现金"))
    print("输出文件：")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
