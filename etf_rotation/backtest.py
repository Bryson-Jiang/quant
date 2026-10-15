"""日频收盘成交回测：信号收盘后生成，下一交易日收盘执行。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .config import BENCHMARK
from .factors import validate_prices


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float | int]


def _metrics(equity: pd.DataFrame, trades: pd.DataFrame) -> dict[str, float | int]:
    returns = equity["daily_return"]
    nav = equity["strategy_nav"]
    trading_days = len(equity)
    drawdown = nav / nav.cummax().clip(lower=1.0) - 1
    annual_vol = float(returns.std(ddof=1) * np.sqrt(252)) if trading_days > 1 else 0.0
    annual_return = float(nav.iloc[-1] ** (252 / trading_days) - 1)
    daily_std = float(returns.std(ddof=1)) if trading_days > 1 else 0.0
    return {
        "trading_days": trading_days,
        "total_return": float(nav.iloc[-1] - 1),
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_zero_rf": float(returns.mean() / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
        "benchmark_total_return": float(equity["benchmark_nav"].iloc[-1] - 1),
        "traded_notional_sum": float(trades["turnover"].sum()) if not trades.empty else 0.0,
        "trade_count": int((trades["turnover"] > 0).sum()) if not trades.empty else 0,
    }


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    start: date,
    end: date,
    *,
    fee_bps: float = 3.0,
    slippage_bps: float = 2.0,
    max_stale_days: int = 3,
) -> BacktestResult:
    """执行与估值均使用后复权收盘价；成本按 ETF 买卖权重绝对变化之和计算。"""
    if fee_bps < 0 or slippage_bps < 0 or max_stale_days < 0:
        raise ValueError("成本和允许的缺价天数不能为负")
    validate_prices(prices)
    if BENCHMARK not in prices:
        raise ValueError("prices 必须含基准 ETF")
    if not isinstance(signals.index, pd.DatetimeIndex) or signals.index.has_duplicates:
        raise ValueError("signals 日期必须是无重复的 DatetimeIndex")
    if not signals.index.isin(prices.index).all():
        raise ValueError("信号日期必须是价格矩阵中的交易日")
    unknown = signals.columns.difference(prices.columns)
    if len(unknown):
        raise ValueError(f"信号包含未知标的：{list(unknown)}")
    dates = prices.index[(prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end))]
    if len(dates) < 2:
        raise ValueError("回测区间至少需要两个交易日")
    columns = list(prices.columns)
    signals = signals.reindex(columns=columns, fill_value=0.0)

    # 下一交易日收盘才成交，避免信号日同价成交带来的未来函数。
    executions: dict[pd.Timestamp, tuple[pd.Timestamp, pd.Series]] = {}
    for signal_date, target in signals.iterrows():
        loc = prices.index.searchsorted(signal_date, side="right")
        if loc < len(prices.index):
            execution_date = prices.index[loc]
            if execution_date in dates:
                executions[execution_date] = (signal_date, target)

    valuation_prices = prices.ffill(limit=max_stale_days) if max_stale_days else prices
    held = pd.Series(0.0, index=columns)
    nav = 1.0
    records: list[dict] = []
    weights: list[pd.Series] = []
    trades: list[dict] = []
    benchmark_base = float(prices.loc[dates[0], BENCHMARK])
    if not np.isfinite(benchmark_base) or benchmark_base <= 0:
        raise ValueError("回测首日基准收盘价无效")

    for i, day in enumerate(dates):
        gross_return = 0.0
        if i > 0 and (held > 0).any():
            yesterday = dates[i - 1]
            owned = held[held > 0].index
            before = valuation_prices.loc[yesterday, owned]
            today = valuation_prices.loc[day, owned]
            if before.isna().any() or today.isna().any():
                missing = list(before.index[before.isna() | today.isna()])
                raise ValueError(f"持仓 {missing} 在 {day.date()} 缺价超过 {max_stale_days} 天，停止回测")
            asset_growth = today / before
            gross_return = float((held.loc[owned] * (asset_growth - 1)).sum())
            if 1 + gross_return <= 0:
                raise ValueError(f"{day.date()} 的组合价值非正")
            # 持有期间价格变化会令权重自然漂移；非调仓日不做免费再平衡。
            held.loc[owned] = held.loc[owned] * asset_growth / (1 + gross_return)
        nav *= 1 + gross_return
        turnover = 0.0
        cost = 0.0
        if day in executions:
            signal_date, desired = executions[day]
            desired = desired.astype(float).copy()
            if not np.isfinite(desired).all() or (desired < 0).any() or desired.sum() > 1 + 1e-8:
                raise ValueError(f"{signal_date.date()} 的目标权重无效")
            # 任何待买/待卖的 ETF 无真实当日收盘价，整笔调仓取消；不可用估值填充值成交。
            changed = (desired - held).abs() > 1e-10
            unavailable = [symbol for symbol in columns if changed[symbol] and pd.isna(prices.at[day, symbol])]
            if not unavailable:
                turnover = float((desired - held).abs().sum())
                cost = turnover * (fee_bps + slippage_bps) / 10_000
                nav *= 1 - cost
                held = desired
            trades.append({
                "signal_date": signal_date,
                "execution_date": day,
                "status": "skipped_no_quote" if unavailable else "executed",
                "holdings": ",".join(held.index[held > 0]),
                "cash_weight": float(1 - held.sum()),
                "turnover": turnover,
                "cost_fraction": cost,
                "unavailable": ",".join(unavailable),
            })
        weights.append(held.rename(day))
        records.append({
            "date": day,
            "strategy_nav": nav,
            "benchmark_nav": float(prices.at[day, BENCHMARK]) / benchmark_base,
            "gross_return": gross_return,
            "turnover": turnover,
            "cost_fraction": cost,
        })

    equity = pd.DataFrame(records).set_index("date")
    equity["daily_return"] = equity["strategy_nav"].pct_change()
    equity.iloc[0, equity.columns.get_loc("daily_return")] = equity.iloc[0]["strategy_nav"] - 1
    equity["drawdown"] = equity["strategy_nav"] / equity["strategy_nav"].cummax().clip(lower=1.0) - 1
    weight_frame = pd.DataFrame(weights)
    weight_frame.index.name = "date"
    weight_frame["CASH"] = 1 - weight_frame.sum(axis=1)
    trade_frame = pd.DataFrame(trades, columns=[
        "signal_date", "execution_date", "status", "holdings", "cash_weight",
        "turnover", "cost_fraction", "unavailable",
    ])
    return BacktestResult(equity, weight_frame, trade_frame, _metrics(equity, trade_frame))
