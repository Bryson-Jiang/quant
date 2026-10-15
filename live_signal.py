"""生成下一次周度调仓计划；只输出目标权重，不连接券商下单。"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from etf_rotation.config import ETF_NAMES, STRATEGY_NAME
from etf_rotation.data import _client, load_close_prices
from etf_rotation.factor_store import FactorStore
from etf_rotation.risk_strategy import RiskConfig, risk_controlled_signals
from etf_rotation.strategy_state import load_rotation_state


ROOT = Path(__file__).resolve().parent
SHANGHAI = ZoneInfo("Asia/Shanghai")


def trade_calendar(as_of: date, path: Path, *, offline: bool) -> pd.DataFrame:
    left, right = as_of - timedelta(days=21), as_of + timedelta(days=21)
    existing = pd.read_csv(path, dtype={"cal_date": str}) if path.exists() else pd.DataFrame()
    if offline:
        if existing.empty:
            raise FileNotFoundError(f"离线交易日历不存在：{path}")
        frame = existing
    else:
        fetched = _client().trade_cal(exchange="SSE", start_date=left.strftime("%Y%m%d"),
                                      end_date=right.strftime("%Y%m%d"), fields="cal_date,is_open")
        if not isinstance(fetched, pd.DataFrame) or fetched.empty:
            raise RuntimeError("Tushare trade_cal 未返回交易日历")
        fetched["cal_date"] = fetched["cal_date"].astype(str)
        frame = pd.concat([existing, fetched], ignore_index=True).drop_duplicates("cal_date", keep="last")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.sort_values("cal_date").to_csv(path, index=False, encoding="utf-8-sig")
    if not {"cal_date", "is_open"}.issubset(frame.columns):
        raise ValueError("交易日历缺少 cal_date 或 is_open")
    dates = pd.to_datetime(frame["cal_date"].astype(str), format="%Y%m%d", errors="coerce")
    flags = pd.to_numeric(frame["is_open"], errors="coerce")
    if dates.isna().any() or (~flags.isin([0, 1])).any() or dates.duplicated().any():
        raise ValueError("交易日历日期或开市标记无效")
    out = pd.DataFrame({"is_open": flags.to_numpy(dtype=int)}, index=pd.DatetimeIndex(dates))
    required = pd.date_range(left, right)
    if not required.isin(out.index).all():
        raise ValueError("交易日历没有完整覆盖前后三周，请联网更新")
    return out.sort_index().loc[str(left):str(right)]


def latest_actionable_week(calendar: pd.DataFrame, as_of: date,
                           now: datetime) -> tuple[date, date, str]:
    """依正式开市日历识别最后一个完整交易周及其下一交易日。"""
    opens = calendar.index[calendar["is_open"].eq(1)]
    cutoff = pd.Timestamp(as_of)
    if as_of == now.date() and now.time() < time(18):
        cutoff -= pd.Timedelta(days=1)
    completed = opens[opens <= cutoff]
    if len(completed) == 0:
        raise ValueError("交易日历中没有已经完成的交易日")
    week_ends = [day for i, day in enumerate(opens[:-1])
                 if day <= completed[-1] and day.to_period("W-FRI") != opens[i + 1].to_period("W-FRI")]
    if not week_ends:
        raise ValueError("交易日历中没有已完成的交易周")
    signal = week_ends[-1]
    execution = opens[opens > signal][0]
    status = "ready" if as_of <= execution.date() else "no_new_rebalance"
    if as_of < now.date():
        status = "historical_only"
    if as_of == execution.date() and as_of == now.date() and now.time() >= time(15):
        status = "execution_window_closed"
    return signal.date(), execution.date(), status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=date.fromisoformat, default=None,
                        help="查询日期，默认上海时区今天")
    parser.add_argument("--offline", action="store_true", help="仅使用已缓存行情与交易日历")
    parser.add_argument("--capital", type=float, default=None,
                        help="可选账户总资产（人民币），输出目标金额；不是下单数量")
    parser.add_argument("--target-vol", type=float, default=RiskConfig.current_default().target_vol,
                        help="当前策略的年化风险仓波动预算，默认 7%%")
    parser.add_argument("--max-risky-weight", type=float, default=0.70,
                        help="风险 ETF 总目标权重上限，默认 70%%")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--state-file", type=Path, default=ROOT / "data" / "rotation_state.json",
                        help="排名缓冲切换记录；请保留并备份")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "live_signal.json")
    args = parser.parse_args()
    if args.capital is not None and args.capital <= 0:
        parser.error("--capital 必须大于零")
    now = datetime.now(SHANGHAI)
    as_of = args.as_of or now.date()
    if as_of > now.date():
        parser.error("--as-of 不能晚于上海当前日期")
    calendar = trade_calendar(as_of, args.cache_dir / "tushare_trade_cal_SSE.csv",
                              offline=args.offline)
    signal_day, execution_day, status = latest_actionable_week(calendar, as_of, now)
    prices = load_close_prices(date(2022, 1, 1), signal_day, args.cache_dir, offline=args.offline)
    if prices.index[-1].date() != signal_day or prices.iloc[-1].isna().any():
        missing = prices.columns[prices.iloc[-1].isna()].tolist()
        raise RuntimeError(f"信号日 {signal_day} 行情不完整；缺少 {missing}，不能发布实盘计划")
    store = FactorStore(ROOT / "data" / "factors.sqlite")
    config = RiskConfig.current_default(target_vol=args.target_vol,
                                        max_risky_weight=args.max_risky_weight)
    state = load_rotation_state(args.state_file)
    signals = risk_controlled_signals(prices, config, store, **state.signal_kwargs(prices))
    target = signals.loc[pd.Timestamp(signal_day)]
    weights = {symbol: round(float(value), 6) for symbol, value in target.items() if value > 1e-8}
    cash = round(max(0.0, 1.0 - float(target.sum())), 6)
    lot_notional = {}
    for symbol in weights:
        cached = pd.read_csv(args.cache_dir / f"tushare_{symbol}_hfq.csv",
                             usecols=["date", "raw_close"])
        quote = cached.loc[cached["date"].eq(signal_day.isoformat()), "raw_close"]
        if len(quote) != 1 or float(quote.iloc[0]) <= 0:
            raise RuntimeError(f"{symbol} 的信号日原始收盘价缺失，不能估计交易手数")
        lot_notional[symbol] = round(float(quote.iloc[0]) * 100, 2)
    plan = {
        "strategy": STRATEGY_NAME,
        "status": status,
        "as_of": as_of.isoformat(),
        "signal_date": signal_day.isoformat(),
        "planned_execution_date": execution_day.isoformat(),
        "execution_assumption": "下一交易日收盘；实际价格和能否成交未知",
        "target_vol": args.target_vol,
        "max_risky_weight": args.max_risky_weight,
        "max_risky_positions": config.max_risky_positions,
        "bear_scale": config.bear_scale,
        "short_momentum_days": config.short_momentum_days,
        "long_momentum_days": config.long_momentum_days,
        "momentum_skip_days": config.momentum_skip_days,
        "trend_days": config.trend_days,
        "max_single_risky_weight": config.max_single_risky_weight,
        "rank_buffer_size": config.rank_buffer_size,
        "mandatory_top_rank": config.mandatory_top_rank,
        "rotation_state": state.metadata(),
        "universe": list(ETF_NAMES),
        "target_weights": weights,
        "cash_weight": cash,
        "estimated_100_share_lot_cny_at_signal_close": lot_notional,
        "target_notional_cny": ({code: round(weight * args.capital, 2) for code, weight in weights.items()}
                                if args.capital is not None else None),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"状态：{status}；信号日：{signal_day}；计划调仓日：{execution_day}；"
          f"目标波动 {args.target_vol:.0%}；风险仓上限 {args.max_risky_weight:.0%}")
    print("选仓：合格池第1名必选；剩余名额优先保留前4名旧目标，再按评分补足")
    if status == "ready":
        for code, weight in weights.items():
            amount = f"，目标约 {weight * args.capital:,.0f} 元" if args.capital else ""
            print(f"  {code} {ETF_NAMES[code]}：{weight:.2%}{amount}")
        print(f"  现金：{cash:.2%}")
        if args.capital:
            large_lots = {code: value for code, value in lot_notional.items()
                          if value / args.capital >= 0.02}
            for code, value in large_lots.items():
                print(f"  手数提示：{code} 按信号日收盘价每 100 份约 {value:,.0f} 元，"
                      f"占账户 {value / args.capital:.1%}；目标权重可能无法精确实现。")
    else:
        print("当前没有可执行的新调仓计划；请等下一次完整交易周的盘后信号。")
    print(f"已保存：{args.output}")


if __name__ == "__main__":
    main()
