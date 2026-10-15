"""现行默认 ETF 池的可复现稳健性审计；仅使用本地数据。"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from etf_rotation.backtest import BacktestResult, run_backtest
from etf_rotation.config import ETF_NAMES
from etf_rotation.data import load_close_prices
from etf_rotation.factor_store import FactorStore
from etf_rotation.factors import FactorSpec
from etf_rotation.risk_strategy import RiskConfig, risk_controlled_signals


ROOT = Path(__file__).resolve().parent
START = date(2022, 1, 1)
EARLY_START = date(2021, 7, 1)
BOND = "511260"


def period_metrics(equity: pd.DataFrame, left: str, right: str) -> tuple[float, float]:
    daily = equity.loc[left:right, "daily_return"]
    if daily.empty:
        return float("nan"), float("nan")
    nav = (1 + daily).cumprod()
    return (float(nav.iloc[-1] ** (252 / len(nav)) - 1),
            float((nav / nav.cummax().clip(lower=1) - 1).min()))


def worst_windows(prices: pd.DataFrame, symbol: str, *, horizon: int = 20,
                  count: int = 5) -> pd.DataFrame:
    """事后挑选的市场下跌窗口只用于诊断，不参与策略参数选择。"""
    series = prices.loc[str(START):, symbol].dropna()
    forward = series / series.shift(horizon) - 1
    positions = pd.Series(np.arange(len(series)), index=series.index)
    selected: list[tuple[int, int]] = []
    rows = []
    for end, loss in forward.sort_values().items():
        if pd.isna(loss):
            continue
        last = int(positions.at[end])
        first = last - horizon
        if first < 0 or any(first <= hi and last >= lo for lo, hi in selected):
            continue
        selected.append((first, last))
        rows.append({"benchmark": symbol, "start": series.index[first],
                     "end": end, "benchmark_return": float(loss)})
        if len(rows) == count:
            break
    return pd.DataFrame(rows)


def window_metrics(result: BacktestResult, left: pd.Timestamp,
                   right: pd.Timestamp) -> tuple[float, float]:
    nav = result.equity.loc[left:right, "strategy_nav"]
    normalized = nav / nav.iloc[0]
    return float(normalized.iloc[-1] - 1), float((normalized / normalized.cummax() - 1).min())


def paired_bootstrap(challenger: pd.Series, baseline: pd.Series, *, block: int,
                     repetitions: int = 2000) -> tuple[float, float]:
    paired = pd.concat([challenger, baseline], axis=1).dropna().to_numpy()
    rng = np.random.default_rng(20260926)
    n = len(paired)
    estimates = np.empty(repetitions)
    for i in range(repetitions):
        starts = rng.integers(0, n, size=(n + block - 1) // block)
        locations = (starts[:, None] + np.arange(block)) % n
        sample = paired[locations.ravel()[:n]]
        annual = np.prod(1 + sample, axis=0) ** (252 / n) - 1
        estimates[i] = annual[0] - annual[1]
    return tuple(float(x) for x in np.quantile(estimates, [0.025, 0.975]))


def needed_frames(prices: pd.DataFrame, config: RiskConfig,
                  store: FactorStore) -> dict[FactorSpec, pd.DataFrame]:
    specs = {FactorSpec("momentum", config.short_momentum_days, config.momentum_skip_days),
             FactorSpec("momentum", config.long_momentum_days, config.momentum_skip_days),
             FactorSpec("trend", config.trend_days), FactorSpec("volatility", 60),
             FactorSpec("momentum", 60)}
    return {spec: store.get(prices, spec) for spec in specs}


def excluded_frames(frames: dict[FactorSpec, pd.DataFrame], config: RiskConfig,
                    members: tuple[str, ...]) -> dict[FactorSpec, pd.DataFrame]:
    altered = frames.copy()
    if BOND in members:
        spec = FactorSpec("momentum", 60)
        bond = altered[spec].copy()
        bond[BOND] = -1.0
        altered[spec] = bond
    else:
        spec = FactorSpec("momentum", config.long_momentum_days,
                          config.momentum_skip_days)
        gate = altered[spec].copy()
        gate.loc[:, list(members)] = -1.0
        altered[spec] = gate
    return altered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "strategy_audit")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prices = load_close_prices(EARLY_START, args.end, ROOT / "data" / "cache",
                               offline=True, warmup_days=200)
    store = FactorStore(ROOT / "data" / "factors.sqlite")
    legacy = RiskConfig()
    fast = RiskConfig(short_momentum_days=20, long_momentum_days=40,
                      momentum_skip_days=0, trend_days=40)
    current = RiskConfig.current_default()
    old_frames = needed_frames(prices, legacy, store)
    new_frames = needed_frames(prices, current, store)
    variants = {
        "legacy_60_120": (legacy, old_frames),
        "fast_40_20": (fast, new_frames),
        "fast_40_20_single30": (current, new_frames),
    }
    bear = store.get(prices, FactorSpec("trend", 120))["510300"].shift(1).le(0)
    results: dict[str, BacktestResult] = {}
    rows = []
    for name, (config, frames) in variants.items():
        target = risk_controlled_signals(prices, config, factor_frames=frames)
        normal = run_backtest(prices, target, START, args.end, fee_bps=5, slippage_bps=0)
        stress = run_backtest(prices, target, START, args.end, fee_bps=20, slippage_bps=0)
        early = run_backtest(prices, target, EARLY_START, date(2021, 12, 31),
                             fee_bps=5, slippage_bps=0)
        results[name] = normal
        mask = bear.reindex(normal.equity.index, fill_value=False)
        sample = normal.equity.loc[mask, "daily_return"]
        risk_columns = [s for s in ETF_NAMES if s != BOND]
        risky_target = target[risk_columns]
        gross = risky_target.sum(axis=1).replace(0, np.nan)
        hhi = risky_target.div(gross, axis=0).pow(2).sum(axis=1)
        owned = normal.weights.loc[normal.equity.index].shift(1)
        row = {"variant": name, "annual": normal.metrics["annual_return"],
               "sharpe": normal.metrics["sharpe_zero_rf"],
               "max_drawdown": normal.metrics["max_drawdown"],
               "stress20_annual": stress.metrics["annual_return"],
               "stress20_max_drawdown": stress.metrics["max_drawdown"],
               "early_2021_h2_total_return": early.metrics["total_return"],
               "early_2021_h2_max_drawdown": early.metrics["max_drawdown"],
               "max_single_risky_target": float(risky_target.max().max()),
               "mean_risky_hhi": float(hhi[gross.notna()].mean()),
               "bear_daily_mean": float(sample.mean()),
               "bear_daily_5pct": float(sample.quantile(0.05)),
               "bear_mean_risk_weight": float(owned.loc[mask, risk_columns].sum(axis=1).mean())}
        for label, left, right in (("train", "2022-01-01", "2024-12-31"),
                                   ("validation", "2025-01-01", "2025-12-31"),
                                   ("observed_2026", "2026-01-01", str(args.end))):
            row[f"{label}_annual"], row[f"{label}_drawdown"] = period_metrics(
                normal.equity, left, right)
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.output_dir / "final_candidate_audit.csv",
                              index=False, encoding="utf-8-sig")

    stress_rows = []
    for _, item in pd.concat([worst_windows(prices, code) for code in
                              ("510300", "159915")], ignore_index=True).iterrows():
        row = {"benchmark": item["benchmark"], "start": item["start"].date(),
               "end": item["end"].date(), "benchmark_return": item["benchmark_return"]}
        for name, result in results.items():
            row[f"{name}_return"], row[f"{name}_drawdown"] = window_metrics(
                result, item["start"], item["end"])
        stress_rows.append(row)
    pd.DataFrame(stress_rows).to_csv(args.output_dir / "final_candidate_bear_windows.csv",
                                     index=False, encoding="utf-8-sig")

    exclusions = [(code, (code,)) for code in ETF_NAMES if code != BOND]
    exclusions += [("gold_and_commodity", ("518880", "159985")),
                   ("overseas", ("159941", "513520")),
                   ("bond_to_cash", (BOND,))]
    ablations = []
    for name, members in exclusions:
        old_target = risk_controlled_signals(
            prices, legacy, factor_frames=excluded_frames(old_frames, legacy, members))
        new_target = risk_controlled_signals(
            prices, current, factor_frames=excluded_frames(new_frames, current, members))
        old_result = run_backtest(prices, old_target, START, args.end,
                                  fee_bps=5, slippage_bps=0)
        new_result = run_backtest(prices, new_target, START, args.end,
                                  fee_bps=5, slippage_bps=0)
        ablations.append({"excluded": name, "members": ",".join(members),
                          "old_annual": old_result.metrics["annual_return"],
                          "new_annual": new_result.metrics["annual_return"],
                          "annual_advantage": new_result.metrics["annual_return"] -
                                              old_result.metrics["annual_return"],
                          "old_sharpe": old_result.metrics["sharpe_zero_rf"],
                          "new_sharpe": new_result.metrics["sharpe_zero_rf"],
                          "sharpe_advantage": new_result.metrics["sharpe_zero_rf"] -
                                              old_result.metrics["sharpe_zero_rf"],
                          "new_max_drawdown": new_result.metrics["max_drawdown"]})
    pd.DataFrame(ablations).sort_values("annual_advantage").to_csv(
        args.output_dir / "final_candidate_matched_ablation.csv",
        index=False, encoding="utf-8-sig")

    bootstrap = {str(block): paired_bootstrap(
        results["fast_40_20_single30"].equity["daily_return"],
        results["legacy_60_120"].equity["daily_return"], block=block)
        for block in (10, 20)}
    (args.output_dir / "final_candidate_bootstrap.json").write_text(json.dumps({
        "paired_block_bootstrap_95pct_annual_difference": bootstrap,
        "not_corrected_for_strategy_search": True,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("paired bootstrap:", bootstrap)


if __name__ == "__main__":
    main()
