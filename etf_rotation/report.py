"""保存可用 Excel 打开的回测表格、摘要与净值图。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .backtest import BacktestResult


def save_report(
    result: BacktestResult,
    signals: pd.DataFrame,
    output_dir: Path,
    *,
    metadata: dict | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for filename, frame in (
        ("equity.csv", result.equity),
        ("daily_weights.csv", result.weights),
        ("trades.csv", result.trades),
        ("signal_weights.csv", signals),
    ):
        path = output_dir / filename
        frame.to_csv(path, encoding="utf-8-sig", index_label=frame.index.name if filename != "trades.csv" else None,
                     index=filename != "trades.csv")
        outputs.append(path)
    summary_path = output_dir / "summary.json"
    summary = {**result.metrics, **(metadata or {})}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs.append(summary_path)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return outputs
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    result.equity[["strategy_nav", "benchmark_nav"]].plot(ax=axes[0])
    axes[0].set_ylabel("Net asset value")
    axes[0].set_title("ETF momentum rotation vs CSI 300 ETF")
    axes[0].grid(alpha=0.25)
    result.equity["drawdown"].plot(ax=axes[1], color="firebrick", legend=False)
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    chart_path = output_dir / "equity.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    outputs.append(chart_path)
    return outputs
