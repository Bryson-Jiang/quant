"""将默认策略回测与主要 A 股宽基 ETF 的买入持有收益画在同一张图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from etf_rotation.config import STRATEGY_NAME
from etf_rotation.data import load_symbol


ROOT = Path(__file__).resolve().parent
BENCHMARKS: dict[str, tuple[str, str]] = {
    "510050": ("上证50", "#9467bd"),
    "510300": ("沪深300", "#1f77b4"),
    "510500": ("中证500", "#ff7f0e"),
    "512100": ("中证1000", "#d62728"),
    "159915": ("创业板", "#2ca02c"),
    "588000": ("科创50", "#8c564b"),
}


def make_comparison(equity: pd.DataFrame, benchmarks: dict[str, pd.Series]) -> pd.DataFrame:
    """起点都归一为 1；行情缺失时停止，不用未来价格或长时填补。"""
    if "strategy_nav" not in equity or equity.empty:
        raise ValueError("回测净值缺少 strategy_nav")
    if not isinstance(equity.index, pd.DatetimeIndex) or equity.index.has_duplicates:
        raise ValueError("回测日期索引无效")
    out = pd.DataFrame(index=equity.index)
    strategy = pd.to_numeric(equity["strategy_nav"], errors="coerce")
    if strategy.isna().any() or (strategy <= 0).any():
        raise ValueError("策略净值包含无效数值")
    # 回测首日已经扣除了建仓费用，保留该费用而不重新归一化。
    out["策略"] = strategy
    for code, (label, _) in BENCHMARKS.items():
        price = benchmarks[code].reindex(equity.index)
        # 停牌或份额折算造成的短暂缺报价，只能沿用已知旧价估值。
        price = price.ffill(limit=3)
        if price.isna().any() or (price <= 0).any():
            missing = price.index[price.isna()][:3].strftime("%Y-%m-%d").tolist()
            raise ValueError(f"{code} 在回测日缺少有效复权价：{missing}")
        out[label] = price / price.iloc[0]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest-dir", type=Path, default=ROOT / "output" / "backtest")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "benchmark_comparison")
    parser.add_argument("--offline", action="store_true", help="只读本地 Tushare 缓存")
    args = parser.parse_args()

    summary = json.loads((args.backtest_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("strategy", {}).get("name") != STRATEGY_NAME:
        raise ValueError("回测摘要不是当前默认策略；请先运行 run_backtest.py")
    equity = pd.read_csv(args.backtest_dir / "equity.csv", index_col=0, parse_dates=True,
                         encoding="utf-8-sig")
    equity.index = pd.DatetimeIndex(equity.index, name="date")
    first, last = equity.index[0].date(), equity.index[-1].date()
    prices = {code: load_symbol(code, first, last, args.cache_dir, offline=args.offline)
              for code in BENCHMARKS}
    missing_quotes = {code: prices[code].reindex(equity.index).index[
        prices[code].reindex(equity.index).isna()].strftime("%Y-%m-%d").tolist()
        for code in BENCHMARKS}
    nav = make_comparison(equity, prices)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nav.to_csv(args.output_dir / "comparison_nav.csv", encoding="utf-8-sig", index_label="date")
    (args.output_dir / "data_checks.json").write_text(json.dumps({
        "strategy": STRATEGY_NAME,
        "source": "Tushare fund_daily close × same-day fund_adj factor",
        "proxy": "buy-and-hold broad-index ETFs, not official index total-return series",
        "benchmark_missing_quote_days": missing_quotes,
        "valuation_of_missing_quote": "prior available close; at most 3 trading days",
        "period": {"start": str(first), "end": str(last)},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    days = len(nav)
    rows = []
    for label in nav:
        series = nav[label]
        rows.append({"name": label, "total_return": float(series.iloc[-1] - 1),
                     "annual_return": float(series.iloc[-1] ** (252 / days) - 1),
                     "max_drawdown": float((series / series.cummax().clip(lower=1) - 1).min())})
    metrics = pd.DataFrame(rows).sort_values("total_return", ascending=True)
    metrics.to_csv(args.output_dir / "comparison_summary.csv", index=False, encoding="utf-8-sig")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax, bars) = plt.subplots(2, 1, figsize=(13, 8.5),
                                   gridspec_kw={"height_ratios": [3, 1.35]})
    colors = {"策略": "#111827", **{label: color for label, color in BENCHMARKS.values()}}
    for label in nav:
        ax.plot(nav.index, (nav[label] - 1) * 100, label=label,
                color=colors[label], linewidth=2.6 if label == "策略" else 1.5,
                alpha=1.0 if label == "策略" else 0.85)
    ax.axhline(0, color="#94a3b8", linewidth=0.8)
    ax.set_title(f"默认策略与主要 A 股宽基 ETF 收益对比  |  {first} 至 {last}",
                 fontsize=15, pad=12)
    ax.set_ylabel("累计收益率（%）")
    ax.grid(alpha=0.2)
    ax.legend(ncol=4, loc="upper left", frameon=False)

    values = metrics["total_return"].to_numpy() * 100
    labels = metrics["name"].tolist()
    bars.barh(labels, values, color=[colors[label] for label in labels], alpha=0.88)
    bars.axvline(0, color="#64748b", linewidth=0.8)
    bars.set_xlabel("全区间累计收益率（%）")
    bars.grid(axis="x", alpha=0.2)
    for y, value in enumerate(values):
        bars.annotate(f"{value:+.1f}%", (value, y),
                      xytext=(4 if value >= 0 else -4, 0), textcoords="offset points",
                      va="center", ha="left" if value >= 0 else "right", fontsize=9)
    bars.set_xlim(min(-15, float(values.min()) - 10), max(65, float(values.max()) + 10))
    fig.text(0.5, 0.015,
             "宽基曲线为代表 ETF 的后复权收盘价买入持有；策略含单边 3bp 手续费及 2bp 滑点，ETF 买入持有未另计交易费。",
             ha="center", color="#475569", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    chart = args.output_dir / "broad_market_comparison.png"
    fig.savefig(chart, dpi=170)
    plt.close(fig)

    drawdowns = metrics.sort_values("max_drawdown")
    fig_dd, ax_dd = plt.subplots(figsize=(10, 5.5))
    dd_labels = drawdowns["name"].tolist()
    dd_values = drawdowns["max_drawdown"].to_numpy() * 100
    ax_dd.barh(dd_labels, dd_values, color=[colors[label] for label in dd_labels], alpha=0.9)
    ax_dd.axvline(0, color="#64748b", linewidth=0.8)
    ax_dd.set_xlim(min(-60, float(dd_values.min()) - 6), 4)
    ax_dd.set_xlabel("最大回撤（%）")
    ax_dd.set_title(f"默认策略与主要 A 股宽基 ETF 最大回撤  |  {first} 至 {last}",
                    fontsize=14, pad=12)
    ax_dd.grid(axis="x", alpha=0.18)
    for y, value in enumerate(dd_values):
        ax_dd.text(value - 0.8, y, f"{value:.1f}%", va="center", ha="right", fontsize=10)
    fig_dd.text(0.5, 0.018,
                "按统一起点后的净值计算历史最高点至其后最低点；宽基为代表 ETF 后复权买入持有。",
                ha="center", color="#475569", fontsize=9)
    fig_dd.tight_layout(rect=(0, 0.04, 1, 1))
    dd_chart = args.output_dir / "drawdown_comparison.png"
    fig_dd.savefig(dd_chart, dpi=170)
    plt.close(fig_dd)
    print(metrics.sort_values("total_return", ascending=False).to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Chart: {chart}")
    print(f"Drawdown chart: {dd_chart}")


if __name__ == "__main__":
    main()
