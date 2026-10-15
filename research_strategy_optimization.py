"""固定候选的扩池/规则研究；不修改默认策略或实盘信号文件。"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from etf_rotation.backtest import BacktestResult, run_backtest
from etf_rotation.config import ETF_NAMES, PREVIOUS_STRATEGY_NAME as STRATEGY_NAME
from etf_rotation.data import load_symbol
from etf_rotation.factors import FactorSpec, compute_factor
from etf_rotation.risk_strategy import (BOND, RiskConfig, priority_buffer_selection,
                                        risk_controlled_signals, weekly_signal_dates)

ROOT = Path(__file__).resolve().parent
START = date(2022, 1, 1)
FETCH_START = date(2020, 11, 27)
CANDIDATES = {
    "510500": "中证500", "512100": "中证1000", "512890": "红利低波",
    "159992": "创新药", "513500": "标普500", "513180": "恒生科技",
    "511010": "5年期国债",
}
EXPANSIONS = tuple(s for s in CANDIDATES if s != "511010")
RULES = ("ensemble", "correlation", "rank_buffer", "fast_vol", "full_vol")
LABELS = {"default16": "默认16只", "add_mid_small": "加入中证500及1000",
          "add_dividend_health": "加入红利低波及创新药", "add_us_hk": "加入标普500及恒生科技",
          "expanded22": "全部扩池至22只", "ensemble": "20/40/80日动量平均",
          "correlation": "相关性筛选（0.8）", "rank_buffer": "排名缓冲（前4名）",
          "fast_vol": "短期波动放大时降仓", "full_vol": "完整组合波动上限",
          "risk_budget7": "波动预算升至7%", "shorter_treasury": "5年国债替换10年国债"}
LABELS["risk7_rank_buffer"] = "7%预算＋前4名排名缓冲"
LABELS["risk7_priority1"] = "7%缓冲＋必选第1名"
LABELS["risk7_priority2"] = "7%缓冲＋必选前2名（补充对照）"
LABELS.update({f"add_{s}": f"加入{name}（{s}）" for s, name in CANDIDATES.items() if s != "511010"})


def factor_frames(prices: pd.DataFrame) -> dict[FactorSpec, pd.DataFrame]:
    specs = [FactorSpec("momentum", n) for n in (20, 40, 60, 80)]
    specs += [FactorSpec("trend", n) for n in (40, 120)]
    specs += [FactorSpec("volatility", n) for n in (20, 60)]
    return {s: compute_factor(prices, s) for s in specs}


def score_frame(frames: dict[FactorSpec, pd.DataFrame], *, ensemble: bool = False) -> pd.DataFrame:
    windows = (20, 40, 80) if ensemble else (20, 40)
    momentum = sum(frames[FactorSpec("momentum", n)] for n in windows) / len(windows)
    return momentum / frames[FactorSpec("volatility", 60)].clip(lower=0.05)


def portfolio_vol(history: pd.DataFrame, weights: pd.Series, *, minimum: int) -> float | None:
    history = history.loc[:, weights.index].dropna()
    if len(history) < minimum:
        return None
    cov = history.cov().to_numpy(dtype=float) * 252
    vector = weights.to_numpy(dtype=float)
    return float(np.sqrt(max(0.0, vector @ cov @ vector)))


def research_signals(prices: pd.DataFrame, mode: str = "base", *, parameter: float | None = None,
                     config: RiskConfig | None = None,
                     frames: dict[FactorSpec, pd.DataFrame] | None = None) -> pd.DataFrame:
    # 冻结本轮研究的6%无缓冲基线，避免默认实盘升级后历史比较随之改变。
    config = config or RiskConfig.previous_default()
    frames = frames or factor_frames(prices)
    if mode == "ensemble":
        return risk_controlled_signals(prices, config, factor_frames=frames,
                                       score_overlay=score_frame(frames, ensemble=True) - score_frame(frames))
    if mode in {"rank_buffer", "rank_buffer_top1", "rank_buffer_top2"}:
        mandatory = {"rank_buffer": 0, "rank_buffer_top1": 1, "rank_buffer_top2": 2}[mode]
        buffered = replace(config, rank_buffer_size=int(parameter or 4), mandatory_top_rank=mandatory)
        return risk_controlled_signals(prices, buffered, factor_frames=frames)
    if mode == "correlation":
        risky = [s for s in prices if s != BOND]
        score = score_frame(frames)
        momentum = frames[FactorSpec("momentum", 40)].copy()
        trend = frames[FactorSpec("trend", 40)]
        volatility = frames[FactorSpec("volatility", 60)]
        returns = prices.pct_change(fill_method=None)
        previous: list[str] = []
        for day in weekly_signal_dates(prices.index):
            eligible = momentum.loc[day, risky].gt(0) & trend.loc[day, risky].gt(0)
            ranked = score.loc[day, risky].where(eligible & volatility.loc[day, risky].gt(0))
            ranked = ranked.dropna().sort_index().sort_values(ascending=False, kind="stable")
            threshold = parameter if parameter is not None else 0.8
            history = returns.loc[:day, ranked.index].tail(60)
            corr = history.corr(min_periods=40)
            selected = []
            for symbol in ranked.index:
                if all(pd.notna(corr.at[symbol, other]) and corr.at[symbol, other] <= threshold
                       for other in selected):
                    selected.append(symbol)
                if len(selected) == config.max_risky_positions:
                    break
            momentum.loc[day, [s for s in risky if s not in selected]] = np.nan
            previous = selected
        altered = dict(frames)
        altered[FactorSpec("momentum", 40)] = momentum
        return risk_controlled_signals(prices, config, factor_frames=altered)
    target = risk_controlled_signals(prices, config, factor_frames=frames)
    if mode == "base":
        return target
    if mode not in {"fast_vol", "full_vol"}:
        raise ValueError(mode)
    returns = prices.pct_change(fill_method=None)
    for day, row in target.iterrows():
        symbols = row.index[row.gt(1e-12) & (row.index != BOND)] if mode == "fast_vol" else row.index[row.gt(1e-12)]
        weights = row.loc[symbols]
        if weights.empty:
            continue
        slow = portfolio_vol(returns.loc[:day, symbols].tail(60), weights, minimum=40)
        fast_window = int(parameter or 20)
        fast = portfolio_vol(returns.loc[:day, symbols].tail(fast_window), weights,
                             minimum=int(np.ceil(fast_window * 0.9)))
        if slow is None or fast is None:
            continue
        if mode == "fast_vol":
            scale = min(1.0, slow / max(slow, fast, 1e-12))
            released = float(weights.sum() * (1 - scale))
            target.loc[day, symbols] *= scale
            if row[BOND] > 0:
                target.at[day, BOND] += released
        else:
            # 国债也有波动；完整组合预算超限时统一降仓，释放为现金。
            scale = min(1.0, config.target_vol / max(slow, fast, 1e-12))
            target.loc[day, symbols] *= scale
    return target


def subset_frames(frames: dict[FactorSpec, pd.DataFrame], columns: list[str]) -> dict[FactorSpec, pd.DataFrame]:
    return {s: f.loc[:, columns] for s, f in frames.items()}


def evaluate(prices: pd.DataFrame, target: pd.DataFrame, end: date, cost: int = 5) -> BacktestResult:
    return run_backtest(prices, target, START, end, fee_bps=cost, slippage_bps=0)


def period_attribution(prices: pd.DataFrame, result: BacktestResult,
                       left: pd.Timestamp, right: pd.Timestamp) -> pd.Series:
    """按区间起点资产计量逐项盈亏，含权重漂移、收益复利和收益后扣费。"""
    equity = result.equity
    days = equity.index[(equity.index > left) & (equity.index <= right)]
    asset_returns = prices.ffill(limit=3).pct_change(fill_method=None).reindex(equity.index)
    daily_pnl = result.weights[prices.columns].shift(1).fillna(0) * asset_returns.fillna(0)
    np.testing.assert_allclose(daily_pnl.loc[days].sum(axis=1), equity.loc[days, "gross_return"], atol=1e-12)
    base_nav = float(equity.at[left, "strategy_nav"])
    capital = equity.strategy_nav.shift(1).loc[days] / base_nav
    components = daily_pnl.loc[days].mul(capital, axis=0).sum()
    components["TRADING_COST"] = float((-capital * (1 + equity.loc[days, "gross_return"])
                                         * equity.loc[days, "cost_fraction"]).sum())
    observed = float(equity.at[right, "strategy_nav"] / base_nav - 1)
    np.testing.assert_allclose(components.sum(), observed, atol=1e-12)
    return components


def period_stats(daily: pd.Series) -> dict[str, float]:
    nav = (1 + daily).cumprod()
    sigma = daily.std(ddof=1)
    return {"annual": float(nav.iloc[-1] ** (252 / len(daily)) - 1),
            "sharpe": float(daily.mean() / sigma * np.sqrt(252)) if sigma > 0 else 0.0,
            "drawdown": float((nav / nav.cummax().clip(lower=1) - 1).min())}


def bootstrap_difference(daily: pd.Series, baseline: pd.Series, block: int = 20) -> tuple[float, float]:
    paired = pd.concat([daily, baseline], axis=1).dropna().to_numpy()
    rng = np.random.default_rng(20261006)
    n = len(paired)
    differences = []
    for _ in range(2000):
        starts = rng.integers(0, n, size=(n + block - 1) // block)
        positions = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        sample = paired[positions]
        differences.append(float(252 * np.mean(sample[:, 0] - sample[:, 1])))
    return tuple(float(v) for v in np.quantile(differences, [0.025, 0.975]))


def transition_target(prices: pd.DataFrame, config: RiskConfig, previous: list[str],
                      *, buffer_size: int = 4,
                      mandatory_top: int = 0,
                      frames: dict[FactorSpec, pd.DataFrame] | None = None) -> pd.Series:
    """切换规则时由既有周目标初始化，不把假想历史持仓当成实际前仓。"""
    frames = frames or factor_frames(prices)
    day = prices.index[-1]
    risky = [s for s in prices if s != BOND]
    eligible = frames[FactorSpec("momentum", 40)].loc[day, risky].gt(0)
    eligible &= frames[FactorSpec("trend", 40)].loc[day, risky].gt(0)
    eligible &= frames[FactorSpec("volatility", 60)].loc[day, risky].gt(0)
    ranked = score_frame(frames).loc[day, risky].where(eligible).dropna()
    ranked = ranked.sort_index().sort_values(ascending=False, kind="stable")
    if mandatory_top:
        selected = priority_buffer_selection(ranked.index, previous, mandatory_top=mandatory_top,
                                              buffer_size=buffer_size, slots=config.max_risky_positions)
    else:
        selected = [s for s in dict.fromkeys(previous) if s in ranked.head(buffer_size).index][:3]
        selected += [s for s in ranked.index if s not in selected][:3 - len(selected)]
    altered = dict(frames)
    altered[FactorSpec("momentum", 40)] = frames[FactorSpec("momentum", 40)].copy()
    altered[FactorSpec("momentum", 40)].loc[day, [s for s in risky if s not in selected]] = np.nan
    unbuffered = replace(config, rank_buffer_size=None, mandatory_top_rank=0)
    return risk_controlled_signals(prices, unbuffered, factor_frames=altered).loc[day]


def combined_report(directory: Path) -> None:
    table = pd.read_csv(directory / "comparison.csv").set_index("variant")
    checks = pd.read_csv(directory / "neighbors.csv")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    intervals = json.loads((directory / "bootstrap.json").read_text(encoding="utf-8"))
    pct = lambda value: f"{value:.2%}"
    lines = ["# 7%波动预算与排名缓冲组合研究", "",
             f"区间2022-01-04至{manifest['end']}；16只默认池；周信号、下一交易日收盘执行。"
             "最多3只风险ETF和1只国债，风险总目标上限70%、单只风险目标上限30%；夏普为零无风险利率口径。", "",
             "这是在上一轮研究后提出的组合检验；四个主方案、三个邻近参数方案事先列明。"
             "所有年份已参加历史研究，不是独立盲测。排名缓冲只改变选择顺序：前次周目标仍合格且位于前4名时优先保留。", "",
             "| 方案 | 年化 | 夏普 | 最大回撤 | 实现年化波动 | 单边20bps年化 | 20bps回撤 | 年买卖权重总和 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in table.iterrows():
        lines.append(f"| {LABELS[name]} | {pct(r.annual_return)} | {r.sharpe_zero_rf:.2f} | {pct(r.max_drawdown)} | {pct(r.annual_volatility)} | {pct(r.stress20_annual)} | {pct(r.stress20_drawdown)} | {r.annual_turnover:.2f} |")
    base, combo, risk7 = (table.loc[n] for n in ("default16", "risk7_rank_buffer", "risk_budget7"))
    lines += ["", "## 判断", "",
              f"组合相对默认年化提高{(combo.annual_return - base.annual_return) * 100:.2f}个百分点，"
              f"夏普变化{combo.sharpe_zero_rf - base.sharpe_zero_rf:+.3f}，"
              f"最大回撤扩大{(base.max_drawdown - combo.max_drawdown) * 100:.2f}个百分点。"
              "全样本点估计符合‘提高收益、夏普相近、历史回撤约5%’的偏好，但分段和敏感性检查并未证明稳定优于默认。", "",
              f"与7%无缓冲相比，缓冲使年买卖权重总和减少{1 - combo.annual_turnover / risk7.annual_turnover:.1%}；"
              f"与6%默认相比减少{1 - combo.annual_turnover / base.annual_turnover:.1%}。"
              f"单边成本升至20bps后，组合年化{pct(combo.stress20_annual)}、回撤{pct(combo.stress20_drawdown)}，"
              f"高于5%的跌幅；默认相应为{pct(base.stress20_annual)}、{pct(base.stress20_drawdown)}。", "",
              "7%是用历史协方差估计的风险仓年化波动预算，不是风险仓固定7%、组合回撤上限或完整组合波动率保证。"
              "国债另有波动；调仓之间权重自然漂移，目标仓位上限不等于每天实际仓位硬上限。", "",
              "按照本次用户选择，仅展示对比，保留6%默认及既有实盘计划。"]
    lines += ["", "## 分段与早期复查", "", "| 方案 | 2022–2024年化 | 夏普 | 回撤 | 2025–2026已观察年化 | 夏普 | 回撤 | 2021下半年收益 | 回撤 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in table.iterrows():
        lines.append(f"| {LABELS[name]} | {pct(r['early_annual'])} | {r['early_sharpe']:.2f} | {pct(r['early_drawdown'])} | {pct(r['later_annual'])} | {r['later_sharpe']:.2f} | {pct(r['later_drawdown'])} | {pct(r['2021h2_return'])} | {pct(r['2021h2_drawdown'])} |")
    yearly = pd.read_csv(directory / "yearly.csv")
    annual_returns = yearly.pivot(index="year", columns="variant", values="total_return")
    lines += ["", "以下为各年的实际累计收益；2026只到9月30日，不是全年收益。", "",
              "| 年份 | 6%默认 | 6%＋缓冲 | 7%无缓冲 | 7%＋缓冲 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for year, r in annual_returns.iterrows():
        lines.append(f"| {year} | {pct(r['default16'])} | {pct(r['rank_buffer'])} | {pct(r['risk_budget7'])} | {pct(r['risk7_rank_buffer'])} |")
    lines += ["", "组合相对默认在2024和2026年截至9月底的收益较低；2022–2024夏普也较低。"
              "整体收益增益在2025较集中，应以冻结规则后的新行情检验能否持续。"]
    lines += ["", "## 邻近参数", "", "| 预算 | 缓冲排名范围 | 年化 | 夏普 | 回撤 | 20bps年化 | 20bps回撤 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in checks.itertuples():
        lines.append(f"| {pct(r.target_vol)} | 前{r.buffer_size}名 | {pct(r.annual)} | {r.sharpe:.2f} | {pct(r.drawdown)} | {pct(r.stress20_annual)} | {pct(r.stress20_drawdown)} |")
    lines += ["", "## 持仓剔除与抽样", ""]
    path = directory / "ablation.csv"
    if manifest.get("robustness_enabled", False) and path.exists():
        ablations = pd.read_csv(path, dtype={"removed": str})
        lines += [f"逐一剔除15只风险ETF后，组合相对6%默认的年化增益在"
                  f"{int(ablations.annual_vs_default.gt(0).sum())}/15个池中为正，夏普增益在"
                  f"{int(ablations.sharpe_vs_default.gt(0).sum())}/15个池中为正。"
                  "剔除创业板、半导体、新能源或人工智能时，组合相对默认的收益优势会消失；"
                  "不能把组合优势视为与标的池无关的规律。", ""]
        lines += ["同时从三个方案可交易池剔除相同标的；沪深300仍保留为市场过滤与业绩基准。", "",
                  "| 剔除 | 组合年化 | 夏普 | 回撤 | 对6%默认年化差 | 夏普差 | 对7%无缓冲年化差 | 夏普差 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in ablations.itertuples():
            lines.append(f"| {r.removed} | {pct(r.annual)} | {r.sharpe:.2f} | {pct(r.drawdown)} | {r.annual_vs_default * 100:+.2f}个百分点 | {r.sharpe_vs_default:+.2f} | {r.annual_vs_risk7 * 100:+.2f}个百分点 | {r.sharpe_vs_risk7:+.2f} |")
    lines += ["", "20日成对循环块抽样的年化算术日均收益差探索性95%区间：", ""]
    for name, bounds in intervals.items():
        lines.append(f"- 组合相对{LABELS[name]}：{bounds[0] * 100:+.2f}至{bounds[1] * 100:+.2f}个百分点。")
    lines += ["", "抽样区间均包含零；抽样未修正此前多次参数/标的选择，不能作为未来增益或回撤上限的证明。"
              "重复在同一历史样本上选择策略会增加回测过拟合风险，参见"
              "[Bailey等：The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)。", "",
              "## 时序检查", "",
              "四个主方案在2023-06-30、2025-06-30截断后重新计算，8项过去目标权重最大差均为0。"
              "未来价格扰动、前4名保留边界及切换初始化测试在`tests/test_research_optimization.py`。"
              "这些检查未发现新增规则使用未来行情，但不能还原Tushare历史发布版本或排除所有现实执行偏差。", "",
              "## 切换时的状态", "",
              "缓冲有历史状态。历史组合回放一直沿用自己的前次周目标；若从旧规则切换，应由旧规则最近的实际目标或经核对的持仓初始化。"
              "本次另存`transition_preview.json`，由旧默认前次周目标初始化9月30日候选，未改写`output/live_signal.json`。"
              "该文件是研究预览，不是已经发布的实盘计划。", "",
              "## 限制与复现", "",
              "连续权重、日收盘成交和净值，未模拟整手、最低佣金、盘中价差、跨境溢价、资金容量和现金利息。"
              "5%左右历史回撤不能保证今后或盘中最大回撤也在5%左右。", "",
              "```powershell", ".\\.venv\\Scripts\\python.exe research_strategy_optimization.py --combined --offline --robustness --end 2026-09-30", "```", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    navs = pd.read_csv(directory / "nav.csv", index_col=0, parse_dates=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    colors = ["#244a73", "#24998c", "#999999", "#df8b20"]
    for (name, nav), color in zip(navs.items(), colors):
        axes[0].plot(nav.index, (nav - 1) * 100, label=LABELS[name], color=color, linewidth=1.7)
        axes[1].plot(nav.index, (nav / nav.cummax().clip(lower=1) - 1) * 100, color=color, linewidth=1.3)
    axes[0].set(title=f"7%波动预算＋排名缓冲｜2022-01-04 至 {manifest['end']}\n单边5 bps；连续权重；历史研究结果", ylabel="累计收益（%）")
    axes[1].set(ylabel="相对历史高点的回撤（%）")
    axes[1].axhline(-5, color="#a64b47", linestyle="--", linewidth=0.8, alpha=0.6)
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(directory / "comparison.png", dpi=170)
    plt.close(fig)


def run_combined(end: date, directory: Path, *, offline: bool, robustness: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    current = RiskConfig.previous_default()
    increased = replace(current, target_vol=0.07)
    cases = (("default16", "base", current), ("rank_buffer", "rank_buffer", current),
             ("risk_budget7", "base", increased), ("risk7_rank_buffer", "rank_buffer", increased))
    neighbor_specs = ((0.065, 4), (0.075, 4), (0.07, 5))
    manifest = {"end": str(end), "default_strategy": STRATEGY_NAME,
                "research_only": True, "default_unchanged": True,
                "primary_cases": [{"name": n, "mode": m, "target_vol": c.target_vol} for n, m, c in cases],
                "neighbors": list(neighbor_specs), "universe": list(ETF_NAMES),
                "sample_previously_used_for_selection": True, "robustness_enabled": robustness,
                "start": str(START), "warmup_start": str(FETCH_START),
                "cost_bps": [5, 20], "rank_buffer": 4,
                "execution": "weekly last trading close -> next trading close",
                "max_risky_positions": current.max_risky_positions,
                "max_risky_weight": current.max_risky_weight,
                "max_single_risky_weight": current.max_single_risky_weight}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    series = {s: load_symbol(s, FETCH_START, end, ROOT / "data" / "cache", offline=offline) for s in ETF_NAMES}
    prices = pd.DataFrame({s: p.reindex(series["510300"].index) for s, p in series.items()})
    manifest["input_sha256"] = {s: hashlib.sha256((ROOT / "data" / "cache" / f"tushare_{s}_hfq.csv").read_bytes()).hexdigest() for s in series}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    frames = factor_frames(prices)
    bear = frames[FactorSpec("trend", 120)]["510300"].shift(1).le(0)
    targets, results, rows, years, prefixes = {}, {}, [], [], []
    for name, mode, config in cases:
        print(f"combined evaluate {name}", flush=True)
        target = research_signals(prices, mode, config=config, frames=frames)
        normal, stress = evaluate(prices, target, end), evaluate(prices, target, end, cost=20)
        early = run_backtest(prices, target, date(2021, 7, 1), date(2021, 12, 31), fee_bps=5, slippage_bps=0)
        targets[name], results[name] = target, normal
        daily = normal.equity["daily_return"]
        row = {"variant": name, **normal.metrics, "stress20_annual": stress.metrics["annual_return"],
               "stress20_sharpe": stress.metrics["sharpe_zero_rf"], "stress20_drawdown": stress.metrics["max_drawdown"],
               "annual_turnover": float(normal.metrics["traded_notional_sum"] * 252 / len(daily)),
               "weak_market_q05": float(daily.loc[bear.reindex(daily.index)].quantile(0.05)),
               "2021h2_return": early.metrics["total_return"], "2021h2_drawdown": early.metrics["max_drawdown"]}
        for label, left, right in (("early", "2022-01-01", "2024-12-31"), ("later", "2025-01-01", str(end))):
            row.update({f"{label}_{k}": v for k, v in period_stats(daily.loc[left:right]).items()})
        rows.append(row)
        for year, returns in daily.groupby(daily.index.year):
            years.append({"variant": name, "year": int(year), **period_stats(returns),
                          "total_return": float((1 + returns).prod() - 1)})
        for cutoff in ("2023-06-30", "2025-06-30"):
            truncated = prices.loc[:cutoff]
            prefix = research_signals(truncated, mode, config=config)
            common = target.index.intersection(prefix.index)
            common = common[common < truncated.index[-1]]
            error = float((target.loc[common] - prefix.loc[common]).abs().to_numpy().max())
            if error > 1e-12:
                raise AssertionError((name, cutoff, error))
            prefixes.append({"variant": name, "cutoff": cutoff, "max_difference": error})
    pd.DataFrame(rows).to_csv(directory / "comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(years).to_csv(directory / "yearly.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prefixes).to_csv(directory / "prefix_audit.csv", index=False, encoding="utf-8-sig")
    pd.concat({n: r.equity["strategy_nav"] for n, r in results.items()}, axis=1).to_csv(directory / "nav.csv", encoding="utf-8-sig")
    print(pd.DataFrame(rows)[["variant", "annual_return", "sharpe_zero_rf", "max_drawdown", "stress20_annual"]].round(4).to_string(index=False), flush=True)
    neighbors = []
    for vol, size in neighbor_specs:
        print(f"neighbor {vol} buffer {size}", flush=True)
        target = research_signals(prices, "rank_buffer", parameter=size, config=replace(current, target_vol=vol), frames=frames)
        normal, stress = evaluate(prices, target, end), evaluate(prices, target, end, cost=20)
        neighbors.append({"target_vol": vol, "buffer_size": size, "annual": normal.metrics["annual_return"],
                          "sharpe": normal.metrics["sharpe_zero_rf"], "drawdown": normal.metrics["max_drawdown"],
                          "stress20_annual": stress.metrics["annual_return"], "stress20_drawdown": stress.metrics["max_drawdown"]})
    pd.DataFrame(neighbors).to_csv(directory / "neighbors.csv", index=False, encoding="utf-8-sig")
    if robustness:
        ablations = []
        for removed in (s for s in prices if s != BOND):
            print(f"ablate {removed}", flush=True)
            altered = dict(frames)
            altered[FactorSpec("momentum", 40)] = frames[FactorSpec("momentum", 40)].copy()
            altered[FactorSpec("momentum", 40)][removed] = np.nan
            matched = {}
            for name, mode, config in (cases[0], cases[2], cases[3]):
                target = research_signals(prices, mode, config=config, frames=altered)
                matched[name] = evaluate(prices, target, end)
            combo, base, risk7 = (matched[n] for n in ("risk7_rank_buffer", "default16", "risk_budget7"))
            ablations.append({"removed": removed, "annual": combo.metrics["annual_return"],
                              "sharpe": combo.metrics["sharpe_zero_rf"], "drawdown": combo.metrics["max_drawdown"],
                              "annual_vs_default": combo.metrics["annual_return"] - base.metrics["annual_return"],
                              "sharpe_vs_default": combo.metrics["sharpe_zero_rf"] - base.metrics["sharpe_zero_rf"],
                              "annual_vs_risk7": combo.metrics["annual_return"] - risk7.metrics["annual_return"],
                              "sharpe_vs_risk7": combo.metrics["sharpe_zero_rf"] - risk7.metrics["sharpe_zero_rf"]})
        pd.DataFrame(ablations).to_csv(directory / "ablation.csv", index=False, encoding="utf-8-sig")
    intervals = {n: bootstrap_difference(results["risk7_rank_buffer"].equity["daily_return"], results[n].equity["daily_return"])
                 for n in ("default16", "risk_budget7")}
    (directory / "bootstrap.json").write_text(json.dumps(intervals, indent=2), encoding="utf-8")
    previous_day = targets["default16"].index[-2]
    prior = targets["default16"].loc[previous_day]
    old_risky = prior.index[prior.gt(0) & (prior.index != BOND)].tolist()
    transition = transition_target(prices, increased, old_risky, frames=frames)
    replay = targets["risk7_rank_buffer"].iloc[-1]
    (directory / "transition_preview.json").write_text(json.dumps({
        "research_only": True, "signal_date": str(prices.index[-1].date()),
        "prior_default_signal_date": str(previous_day.date()), "initial_risk_selection": old_risky,
        "target_vol": 0.07, "rank_buffer_size": 4,
        "transition_weights": {s: float(w) for s, w in transition.items() if w > 1e-8},
        "hypothetical_replay_weights": {s: float(w) for s, w in replay.items() if w > 1e-8},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    combined_report(directory)
    print(f"combined report saved: {directory / 'report.md'}", flush=True)


def recent_drawdown_audit(directory: Path, end: date) -> None:
    """只解释已保存回放的最近回撤，不据这段下跌重选参数或改变默认。"""
    output = directory / "recent_drawdown"
    output.mkdir(parents=True, exist_ok=True)
    saved = pd.read_csv(directory / "nav.csv", index_col=0, parse_dates=True)
    if saved.index[-1].date() != end:
        raise ValueError("审计结束日必须与已保存对比结果一致")
    series = {s: load_symbol(s, FETCH_START, end, ROOT / "data" / "cache", offline=True) for s in ETF_NAMES}
    prices = pd.DataFrame({s: p.reindex(series["510300"].index) for s, p in series.items()})
    frames = factor_frames(prices)
    config = RiskConfig.previous_default()
    increased = replace(config, target_vol=0.07)
    cases = (("default16", "base", config), ("rank_buffer", "rank_buffer", config),
             ("risk_budget7", "base", increased), ("risk7_rank_buffer", "rank_buffer", increased))
    results, targets = {}, {}
    for name, mode, rule in cases:
        print(f"drawdown audit {name}", flush=True)
        target = research_signals(prices, mode, config=rule, frames=frames)
        result = evaluate(prices, target, end)
        np.testing.assert_allclose(result.equity.strategy_nav, saved[name], atol=1e-12, rtol=0)
        results[name], targets[name] = result, target
    nav = saved.risk7_rank_buffer
    peak = nav.idxmax()
    trough = nav.loc[peak:].idxmin()
    latest = nav.index[-1]
    attributions, summaries = {}, []
    risky = [s for s in prices if s != BOND]
    for name, result in results.items():
        own = result.equity.strategy_nav
        own_peak = own.idxmax()
        own_trough = own.loc[own_peak:].idxmin()
        row = {"variant": name, "peak": str(own_peak.date()), "trough": str(own_trough.date()),
               "recent_max_drawdown": float(own.loc[own_trough] / own.loc[own_peak] - 1),
               "latest_drawdown": float(own.iloc[-1] / own.loc[own_peak] - 1)}
        for label, right in (("trough", trough), ("latest", latest)):
            components = period_attribution(prices, result, peak, right)
            attributions[(name, label)] = components
            row[f"common_{label}_return"] = float(components.sum())
        row["mean_risk_weight"] = float(result.weights[risky].shift(1).loc[peak:latest].iloc[1:].sum(axis=1).mean())
        summaries.append(row)
    summary = pd.DataFrame(summaries).set_index("variant")
    summary.to_csv(output / "summary.csv", encoding="utf-8-sig")
    contribution = pd.DataFrame(attributions)
    contribution.columns.names = ["variant", "period"]
    contribution.to_csv(output / "attribution.csv", encoding="utf-8-sig")
    score = score_frame(frames)
    eligible = frames[FactorSpec("momentum", 40)].gt(0) & frames[FactorSpec("trend", 40)].gt(0)
    eligible &= frames[FactorSpec("volatility", 60)].gt(0)
    decisions = []
    for day in targets["risk7_rank_buffer"].index:
        if day < peak - pd.Timedelta(days=14):
            continue
        ranked = score.loc[day, risky].where(eligible.loc[day, risky]).dropna()
        ranked = ranked.sort_index().sort_values(ascending=False, kind="stable")
        ranks = pd.Series(range(1, len(ranked) + 1), index=ranked.index)
        selected = targets["risk7_rank_buffer"].loc[day].drop(BOND)
        base = targets["risk_budget7"].loc[day].drop(BOND)
        for symbol in ranked.index:
            if selected[symbol] <= 0 and base[symbol] <= 0:
                continue
            decisions.append({"day": str(day.date()), "symbol": symbol, "rank": int(ranks[symbol]),
                              "score": float(ranked[symbol]), "buffer_weight": float(selected[symbol]),
                              "plain_weight": float(base[symbol]),
                              "retained_rank4": bool(ranks[symbol] == 4 and selected[symbol] > 0),
                              "blocked_rank1": bool(ranks[symbol] == 1 and selected[symbol] <= 0)})
    decisions = pd.DataFrame(decisions)
    decisions.to_csv(output / "weekly_decisions.csv", index=False, encoding="utf-8-sig")
    episodes = []
    for day, selected in targets["risk7_rank_buffer"].iterrows():
        if day < peak:
            continue
        plain = targets["risk_budget7"].loc[day]
        kept = [s for s in risky if selected[s] > 0 and plain[s] <= 0]
        missed = [s for s in risky if plain[s] > 0 and selected[s] <= 0]
        if not kept:
            continue
        entry_pos = prices.index.searchsorted(day, side="right")
        row = {"signal_date": str(day.date()), "kept": ",".join(kept), "missed": ",".join(missed)}
        if entry_pos >= len(prices):
            row["status"] = "审计截止时尚未执行，不影响此前回撤"
            episodes.append(row)
            continue
        entry = prices.index[entry_pos]
        future_signals = targets["risk7_rank_buffer"].index[targets["risk7_rank_buffer"].index > day]
        next_pos = prices.index.searchsorted(future_signals[0], side="right") if len(future_signals) else len(prices)
        exit_day = prices.index[next_pos] if next_pos < len(prices) else latest
        kept_returns = prices.loc[exit_day, kept] / prices.loc[entry, kept] - 1
        missed_returns = prices.loc[exit_day, missed] / prices.loc[entry, missed] - 1
        a = period_attribution(prices, results["risk7_rank_buffer"], entry, exit_day)
        b = period_attribution(prices, results["risk_budget7"], entry, exit_day)
        row.update({"status": "已执行", "entry_close": str(entry.date()), "next_execution_close": str(exit_day.date()),
                    "kept_price_return": ";".join(f"{s}:{kept_returns[s]:.6%}" for s in kept),
                    "missed_price_return": ";".join(f"{s}:{missed_returns[s]:.6%}" for s in missed),
                    "gross_portfolio_difference": float(a.drop("TRADING_COST").sum() - b.drop("TRADING_COST").sum())})
        episodes.append(row)
    pd.DataFrame(episodes).to_csv(output / "buffer_episodes.csv", index=False, encoding="utf-8-sig")
    delays = []
    for name, result in results.items():
        # 记录收盘已失去资格但仍有仓位的连续交易日；可能等周信号、次日成交或恢复资格。
        for symbol in risky:
            failed = result.weights[symbol].gt(1e-8) & ~eligible[symbol].reindex(result.weights.index)
            group = failed.ne(failed.shift(fill_value=False)).cumsum()
            for _, spell in failed.loc[peak:].groupby(group.loc[peak:]):
                if not bool(spell.iloc[0]):
                    continue
                first, last = spell.index[0], spell.index[-1]
                next_pos = result.weights.index.searchsorted(last, side="right")
                next_day = result.weights.index[next_pos] if next_pos < len(result.weights) else None
                resolution = "截至审计日仍不合格"
                if next_day is not None:
                    resolution = "退出" if result.weights.at[next_day, symbol] <= 1e-8 else "恢复资格"
                delays.append({"variant": name, "symbol": symbol, "first_invalid_close": str(first.date()),
                               "last_invalid_held_close": str(last.date()), "trading_days": len(spell),
                               "resolution_date": str(next_day.date()) if next_day is not None else "",
                               "resolution": resolution})
        signal_eligibility = eligible.loc[targets[name].index, risky]
        if (targets[name][risky].gt(1e-8) & ~signal_eligibility).any().any():
            raise AssertionError(f"{name} 选中筛选已失败的ETF")
    pd.DataFrame(delays).to_csv(output / "filter_delays.csv", index=False, encoding="utf-8-sig")
    lines = ["# 最近回撤归因与排名缓冲退出审计", "",
             f"固定原方案与原数据回放。最近共同高点{peak.date()}，组合低点{trough.date()}，截至{latest.date()}。"
             "低点由事后数据定位，仅用于解释历史，不参与交易信号。", "",
             "| 方案 | 最近最大回撤 | 截至末日回撤 | 平均风险仓位 |", "| --- | ---: | ---: | ---: |"]
    for name, r in summary.iterrows():
        lines.append(f"| {LABELS[name]} | {r.recent_max_drawdown:.2%} | {r.latest_drawdown:.2%} | {r.mean_risk_weight:.2%} |")
    base_return = summary.at["default16", "common_trough_return"]
    plain_return = summary.at["risk_budget7", "common_trough_return"]
    buffer_return = summary.at["risk7_rank_buffer", "common_trough_return"]
    lines += ["", "## 主要原因", "",
              f"按共同日期{peak.date()}至{trough.date()}比较，6%默认收益{base_return:.2%}，"
              f"7%无缓冲{plain_return:.2%}，7%缓冲{buffer_return:.2%}。"
              f"从6%默认先升预算到7%，这段收益减少{(base_return - plain_return) * 100:.3f}个百分点；"
              f"在7%基础上加入缓冲，再减少{(plain_return - buffer_return) * 100:.3f}个百分点。"
              "这是依照该变更顺序的历史方案差，不能解释为两种效果始终相互独立。", "",
              f"7%缓冲与7%无缓冲平均风险仓分别为{summary.at['risk7_rank_buffer', 'mean_risk_weight']:.2%}、"
              f"{summary.at['risk_budget7', 'mean_risk_weight']:.2%}。额外回撤主要来自选仓时点和组合结构差异。"
              "保留旧仓会改变其他仓位、历史协方差及风险缩放，也会错过新候选的上涨。"
              "不能把ETF贡献差全部解读为多持有某ETF导致的止损延误。", "",
              "近期例子包括：8月14日保留有色、未换入黄金；8月28日继续保留有色、未换入豆粕；"
              "9月4日保留黄金、未换入银行。完整后续收益见下文。9月30日豆粕替代日经的信号尚未成交，"
              "不可能是此前回撤的原因。"]
    for label, right in (("trough", trough), ("latest", latest)):
        lines += ["", f"## {peak.date()}收盘至{right.date()}收盘收益贡献", "",
                  "贡献按共同高点的账户资产计量，含期间复利和自然权重漂移；成本按每日收益后实际扣除金额计量。"
                  "各项之和与区间收益严格核对。", "",
                  "| ETF | 6%默认 | 6%缓冲 | 7%无缓冲 | 7%缓冲 | 缓冲对7%无缓冲差 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for symbol in contribution.index:
            values = [attributions[(n, label)][symbol] for n, _, _ in cases]
            if max(abs(v) for v in values) < 1e-8:
                continue
            title = "交易成本" if symbol == "TRADING_COST" else f"{symbol} {ETF_NAMES[symbol]}"
            lines.append(f"| {title} | " + " | ".join(f"{v * 100:+.3f}pp" for v in values)
                         + f" | {(values[-1] - values[-2]) * 100:+.3f}pp |")
        totals = [float(attributions[(n, label)].sum()) for n, _, _ in cases]
        lines.append("| 合计 | " + " | ".join(f"{v:.2%}" for v in totals)
                     + f" | {(totals[-1] - totals[-2]) * 100:+.3f}pp |")
    lines += ["", "## 周选择与退出条件", "",
              f"审计窗口中有{int(decisions.retained_rank4.sum())}次周信号保留第4名；"
              f"有{int(decisions.blocked_rank1.sum())}次周信号未选第1名。逐项见`weekly_decisions.csv`。", "",
              "全部四方案在周信号日均未选中40日动量非正或40日均线趋势非正的风险ETF。"
              "缓冲会延后排名淘汰，但不覆盖这些绝对筛选。周中失去资格仍可能持有，见`filter_delays.csv`；"
              "当前四方案没有每日、盘中或固定亏损比例止损。", "",
              "## 每次保留旧仓的后续价格", "",
              "从信号的下一交易日收盘到下一次调仓收盘，使用复权价格衡量。差值为该周期按各自期初账户资产计量的"
              "组合毛收益差，不含手续费；周期不含开始成交前的涨跌，也不能直接加总成整段回撤差。", "",
              "| 信号 | 持仓区间 | 保留旧仓/价格变化 | 未选新仓/价格变化 | 缓冲毛收益差 |",
              "| --- | --- | --- | --- | ---: |"]
    for r in episodes:
        if r["status"] != "已执行":
            lines.append(f"| {r['signal_date']} | 尚未执行 | {r['kept']} | {r['missed']} | 不影响此前回撤 |")
        else:
            lines.append(f"| {r['signal_date']} | {r['entry_close']}至{r['next_execution_close']} | {r['kept_price_return']} | {r['missed_price_return']} | {r['gross_portfolio_difference'] * 100:+.3f}pp |")
    lines += ["",
              "窗口归因仅说明这段历史的组合差异，不证明未来缓冲一定有益或有害；未据近期低点调参，默认6%及实盘计划未改变。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.round(6).to_string(), flush=True)
    print("attribution through trough (percentage points)", flush=True)
    print((contribution.xs("trough", level="period", axis=1) * 100).round(4).to_string(), flush=True)
    print("rank4 decisions", flush=True)
    print(decisions.loc[decisions.retained_rank4 | decisions.blocked_rank1].to_string(index=False), flush=True)
    print(f"audit saved: {output}", flush=True)


def run_priority_buffer(end: date, directory: Path, *, robustness: bool) -> None:
    """主检验为必选第1名，必选前2名只作预先固定的补充对照。"""
    directory.mkdir(parents=True, exist_ok=True)
    config = RiskConfig.previous_default()
    increased = replace(config, target_vol=0.07)
    cases = (("default16", "base", config), ("risk_budget7", "base", increased),
             ("risk7_rank_buffer", "rank_buffer", increased),
             ("risk7_priority1", "rank_buffer_top1", increased),
             ("risk7_priority2", "rank_buffer_top2", increased))
    manifest = {"start": str(START), "end": str(end), "research_only": True,
                "default_unchanged": True, "primary_candidate": "risk7_priority1",
                "secondary_predeclared": "risk7_priority2", "buffer_size": 4,
                "universe": list(ETF_NAMES), "cost_bps": [5, 20],
                "sample_previously_used_for_selection": True,
                "execution": "weekly last trading close -> next trading close",
                "selection_rule": "guarantee eligible rank1/rank1-2, retain prior eligible top4 in current-rank order, fill to3",
                "risk_controls": "same 70% total / 30% single caps and weekly absolute filters; no daily stop",
                "robustness_enabled": robustness}
    series = {s: load_symbol(s, FETCH_START, end, ROOT / "data" / "cache", offline=True) for s in ETF_NAMES}
    prices = pd.DataFrame({s: p.reindex(series["510300"].index) for s, p in series.items()})
    frames = factor_frames(prices)
    manifest["input_sha256"] = {s: hashlib.sha256((ROOT / "data" / "cache" / f"tushare_{s}_hfq.csv").read_bytes()).hexdigest() for s in series}
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    score = score_frame(frames)
    risky = [s for s in prices if s != BOND]
    eligible = frames[FactorSpec("momentum", 40)].gt(0) & frames[FactorSpec("trend", 40)].gt(0)
    eligible &= frames[FactorSpec("volatility", 60)].gt(0)
    targets, results, rows, years, selection_rows, prefixes = {}, {}, [], [], [], []
    fixed_peak, fixed_low = pd.Timestamp("2026-06-30"), pd.Timestamp("2026-09-28")
    for name, mode, rule in cases:
        print(f"priority evaluate {name}", flush=True)
        target = research_signals(prices, mode, config=rule, frames=frames)
        normal, stress = evaluate(prices, target, end), evaluate(prices, target, end, cost=20)
        targets[name], results[name] = target, normal
        daily = normal.equity.daily_return
        row = {"variant": name, **normal.metrics, "stress20_annual": stress.metrics["annual_return"],
               "stress20_sharpe": stress.metrics["sharpe_zero_rf"], "stress20_drawdown": stress.metrics["max_drawdown"],
               "annual_turnover": float(normal.metrics["traded_notional_sum"] * 252 / len(daily))}
        for label, left, right in (("early", "2022-01-01", "2024-12-31"), ("later", "2025-01-01", str(end))):
            row.update({f"{label}_{k}": v for k, v in period_stats(daily.loc[left:right]).items()})
        nav = normal.equity.strategy_nav
        if fixed_peak in nav.index and fixed_low in nav.index:
            row["recent_common_return"] = float(nav.loc[fixed_low] / nav.loc[fixed_peak] - 1)
            row["recent_latest_return"] = float(nav.iloc[-1] / nav.loc[fixed_peak] - 1)
        for year, group in daily.groupby(daily.index.year):
            years.append({"variant": name, "year": int(year), **period_stats(group),
                          "total_return": float((1 + group).prod() - 1)})
        for day in target.index[target.index >= pd.Timestamp(START)]:
            ranked = score.loc[day, risky].where(eligible.loc[day, risky]).dropna()
            ranked = ranked.sort_index().sort_values(ascending=False, kind="stable")
            owned = target.loc[day, risky].loc[lambda r: r.gt(1e-10)].index.tolist()
            first = ranked.index[0] if len(ranked) else ""
            second = ranked.index[1] if len(ranked) > 1 else ""
            if mode == "rank_buffer_top1" and first and first not in owned:
                raise AssertionError((name, day, "first not selected"))
            if mode == "rank_buffer_top2" and any(s and s not in owned for s in (first, second)):
                raise AssertionError((name, day, "top2 not selected"))
            selection_rows.append({"variant": name, "day": str(day.date()), "first": first, "second": second,
                                   "selected": ",".join(owned), "blocked_first": bool(first and first not in owned),
                                   "blocked_second": bool(second and second not in owned)})
        if name in {"risk7_priority1", "risk7_priority2"}:
            for cutoff in ("2023-06-30", "2025-06-30"):
                truncated = prices.loc[:cutoff]
                prefix = research_signals(truncated, mode, config=rule)
                common = target.index.intersection(prefix.index)
                common = common[common < truncated.index[-1]]
                error = float((target.loc[common] - prefix.loc[common]).abs().to_numpy().max())
                if error > 1e-12:
                    raise AssertionError((name, cutoff, error))
                prefixes.append({"variant": name, "cutoff": cutoff, "max_difference": error})
        if "risk7_rank_buffer" in targets:
            old = targets["risk7_rank_buffer"].loc[str(START):, risky].gt(1e-10)
            row["different_selection_weeks_vs_old"] = int(target.loc[old.index, risky].gt(1e-10).ne(old).any(axis=1).sum())
        rows.append(row)
    table = pd.DataFrame(rows).set_index("variant")
    table.to_csv(directory / "comparison.csv", encoding="utf-8-sig")
    pd.DataFrame(years).to_csv(directory / "yearly.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selection_rows).to_csv(directory / "weekly_selection.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prefixes).to_csv(directory / "prefix_audit.csv", index=False, encoding="utf-8-sig")
    pd.concat({n: r.equity.strategy_nav for n, r in results.items()}, axis=1).to_csv(directory / "nav.csv", encoding="utf-8-sig")
    saved_path = directory.parent / "combined_7pct" / "nav.csv"
    if saved_path.exists():
        saved = pd.read_csv(saved_path, index_col=0, parse_dates=True)
        if saved.index.equals(results["default16"].equity.index):
            for n in ("default16", "risk_budget7", "risk7_rank_buffer"):
                np.testing.assert_allclose(saved[n], results[n].equity.strategy_nav, atol=1e-12, rtol=0)
    intervals = {reference: bootstrap_difference(results["risk7_priority1"].equity.daily_return,
                                                 results[reference].equity.daily_return)
                 for reference in ("risk7_rank_buffer", "risk_budget7")}
    (directory / "bootstrap.json").write_text(json.dumps({"primary_vs_reference_annual_arithmetic_95pct": intervals,
                                                          "not_corrected_for_multiple_trials": True}, indent=2), encoding="utf-8")
    print(table[["annual_return", "sharpe_zero_rf", "max_drawdown", "recent_common_return",
                 "different_selection_weeks_vs_old"]].round(6).to_string(), flush=True)
    if robustness:
        ablations = []
        for removed in risky:
            print(f"priority ablate {removed}", flush=True)
            altered = dict(frames)
            altered[FactorSpec("momentum", 40)] = frames[FactorSpec("momentum", 40)].copy()
            altered[FactorSpec("momentum", 40)][removed] = np.nan
            matched = {}
            for name, mode, rule in cases[1:]:
                target = research_signals(prices, mode, config=rule, frames=altered)
                matched[name] = evaluate(prices, target, end)
            for name in ("risk7_priority1", "risk7_priority2"):
                r, old, plain = (matched[n].metrics for n in (name, "risk7_rank_buffer", "risk_budget7"))
                ablations.append({"variant": name, "removed": removed, "annual": r["annual_return"],
                                  "sharpe": r["sharpe_zero_rf"], "drawdown": r["max_drawdown"],
                                  "annual_vs_old": r["annual_return"] - old["annual_return"],
                                  "sharpe_vs_old": r["sharpe_zero_rf"] - old["sharpe_zero_rf"],
                                  "annual_vs_plain": r["annual_return"] - plain["annual_return"],
                                  "sharpe_vs_plain": r["sharpe_zero_rf"] - plain["sharpe_zero_rf"]})
        pd.DataFrame(ablations).to_csv(directory / "ablation.csv", index=False, encoding="utf-8-sig")
    prior = targets["default16"].iloc[-2]
    previous = prior.index[prior.gt(0) & (prior.index != BOND)].tolist()
    preview = {"research_only": True, "signal_date": str(prices.index[-1].date()), "initial_selection": previous,
               "weights": {}}
    for name, count in (("risk7_priority1", 1), ("risk7_priority2", 2)):
        transition = transition_target(prices, increased, previous, mandatory_top=count, frames=frames)
        preview["weights"][name] = {s: float(v) for s, v in transition.items() if v > 1e-8}
    (directory / "transition_preview.json").write_text(json.dumps(preview, indent=2), encoding="utf-8")
    selections = pd.DataFrame(selection_rows)
    lines = ["# 强制最强候选入选的缓冲规则研究", "",
             f"区间2022-01-04至{end}，默认16只池，单边成本5bps及20bps压力测试，夏普为零无风险利率。"
             "周末交易日收盘信号，下一交易日收盘执行，3只风险ETF加国债/现金，风险总目标70%、单只风险目标30%。", "",
             "主检验为用户提出的必选第1名；必选前2名是开始检验前固定的补充对照。"
             "先过滤不合格资产，再保证前1/2名，按当前排名保留仍在前4名的旧仓，最后按排名补齐3只。"
             "前1名不一定权重最高，权重仍由逆波动及协方差决定。没有加入每日止损。", "",
             "此前全部年份已参与研究，本轮是针对观察到的问题做的探索性比较，非独立盲测。", "",
             "| 方案 | 年化 | 夏普 | 最大回撤 | 年买卖权重总和 | 20bps年化 | 20bps回撤 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in table.iterrows():
        lines.append(f"| {LABELS[name]} | {r.annual_return:.2%} | {r.sharpe_zero_rf:.2f} | {r.max_drawdown:.2%} | {r.annual_turnover:.2f} | {r.stress20_annual:.2%} | {r.stress20_drawdown:.2%} |")
    first, old = table.loc["risk7_priority1"], table.loc["risk7_rank_buffer"]
    lines += ["", "## 判断", "",
              f"主方案相对原缓冲，年化变化{(first.annual_return - old.annual_return) * 100:+.3f}个百分点，"
              f"夏普变化{first.sharpe_zero_rf - old.sharpe_zero_rf:+.3f}，"
              f"最大回撤幅度变化{(abs(first.max_drawdown) - abs(old.max_drawdown)) * 100:+.3f}个百分点（负数表示缩小）。"
              f"6月30日至9月28日收益差为{(first.recent_common_return - old.recent_common_return) * 100:+.3f}个百分点。", "",
              "强制第1名入选可以修复最强候选被旧仓挤掉的情况，但不保证第2、第3名入选，也没有加快绝对风控退出。"
              "近期错过的黄金、豆粕、银行主要位于第2或第3名，不能指望仅此一条规则消除这些机会成本。"
              "历史回撤改善应与逐年、同池剔除及成本压力一起判断，不能视为更高收益或未来回撤上限的证明。"]
    lines += ["", "## 分段与选仓变化", "",
              "| 方案 | 2022–2024年化 | 夏普 | 2025–2026观察年化 | 夏普 | 6/30至9/28收益 | 排除第1名的周数 | 排除第2名的周数 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, r in table.iterrows():
        subset = selections.loc[selections.variant.eq(name)]
        lines.append(f"| {LABELS[name]} | {r.early_annual:.2%} | {r.early_sharpe:.2f} | {r.later_annual:.2%} | {r.later_sharpe:.2f} | {r.recent_common_return:.2%} | {int(subset.blocked_first.sum())} | {int(subset.blocked_second.sum())} |")
    for name in ("risk7_priority1", "risk7_priority2"):
        r = table.loc[name]
        lines += ["", f"{LABELS[name]}与原缓冲的选仓有{int(r.different_selection_weeks_vs_old)}个周信号不同。"]
    lines += ["", "## 逐年实际收益", "", "| 年份 | 6%默认 | 7%无缓冲 | 7%原缓冲 | 必选第1名 | 必选前2名 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for year, r in pd.DataFrame(years).pivot(index="year", columns="variant", values="total_return").iterrows():
        lines.append(f"| {year} | " + " | ".join(f"{r[n]:.2%}" for n, _, _ in cases) + " |")
    lines += ["", "2026只到9月30日，不是全年收益。近期窗口和年份分段均已被观察，不能作为盲测。", "",
              "## 稳健性与时序", ""]
    if robustness:
        for name, group in pd.DataFrame(ablations).groupby("variant"):
            lines += [f"- {LABELS[name]}：逐一剔除15只风险ETF后，相对同池原缓冲年化改善"
                      f"{int(group.annual_vs_old.gt(1e-10).sum())}/15、夏普改善{int(group.sharpe_vs_old.gt(1e-10).sum())}/15；"
                      f"相对同池7%无缓冲年化改善{int(group.annual_vs_plain.gt(1e-10).sum())}/15、"
                      f"夏普改善{int(group.sharpe_vs_plain.gt(1e-10).sum())}/15。"]
    for reference, bounds in intervals.items():
        lines += [f"- 主方案对{LABELS[reference]}的20日成对循环块抽样，年化算术收益差探索性95%区间"
                  f"为{bounds[0] * 100:+.2f}至{bounds[1] * 100:+.2f}个百分点；未调整重复选择。"]
    lines += ["", "新增两个规则在两个截断日期的4项历史权重差均为0；每个周信号都检查最强候选实际入选。"
              "原三个基线净值与已保存结果一致。新增选仓边界和未来行情扰动测试在`tests/test_research_optimization.py`。", "",
              "## 实盘状态与限制", "",
              "默认及实盘计划未切换。研究预览保存在`transition_preview.json`，由上次默认目标初始化。"
              "只保证排名第1名并不意味着必须退出第4名；原策略每周绝对筛选和退出时序仍相同。"
              "历史最大回撤不保证未来或盘中回撤，连续权重未模拟整手、最低佣金、价差与溢折价。", "",
              "```powershell", ".\\.venv\\Scripts\\python.exe research_strategy_optimization.py --priority-buffer --offline --robustness --end 2026-09-30", "```", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"priority report saved: {directory / 'report.md'}", flush=True)


def factor_diagnostics(prices: pd.DataFrame, frames: dict[FactorSpec, pd.DataFrame],
                       *, ensemble: bool) -> dict:
    """未来收益仅作评价标签。回归拟合只用在当周信号时已结束的标签。"""
    score = score_frame(frames, ensemble=ensemble)
    weeks = weekly_signal_dates(prices.index)
    risky = [s for s in prices if s != BOND]
    samples = []
    for left, right in zip(weeks[:-1], weeks[1:]):
        a, b = prices.index.searchsorted(left, side="right"), prices.index.searchsorted(right, side="right")
        if b >= len(prices):
            continue
        eligible = frames[FactorSpec("momentum", 40)].loc[left, risky].gt(0)
        eligible &= frames[FactorSpec("trend", 40)].loc[left, risky].gt(0)
        label = prices.iloc[b][risky] / prices.iloc[a][risky] - 1
        frame = pd.DataFrame({"score": score.loc[left, risky].where(eligible), "y": label}).dropna()
        if len(frame) < 5 or frame["score"].std(ddof=1) <= 0:
            continue
        frame["z"] = (frame["score"] - frame["score"].mean()) / frame["score"].std(ddof=1)
        samples.append((left, prices.index[b], frame))
    sse_model, sse_mean, prediction_count = 0.0, 0.0, 0
    ic_rows = []
    for day, exit_day, frame in samples:
        if day < pd.Timestamp(START):
            continue
        # Spearman等于平均秩的Pearson相关；不额外依赖SciPy。
        ic_rows.append((day, float(frame["score"].rank().corr(frame["y"].rank()))))
        # 最近104个完成的周截面；当前可交易信号尚不能观察下一周收益。
        known = [f for d, end_day, f in samples if end_day <= day][-104:]
        if len(known) < 52:
            continue
        training = pd.concat(known, ignore_index=True)
        x = np.column_stack([np.ones(len(training)), training["z"]])
        beta = np.linalg.lstsq(x, training["y"].to_numpy(), rcond=None)[0]
        prediction = beta[0] + beta[1] * frame["z"].to_numpy()
        y = frame["y"].to_numpy()
        sse_model += float(np.sum((prediction - y) ** 2))
        sse_mean += float(np.sum((training["y"].mean() - y) ** 2))
        prediction_count += len(frame)
    ic = pd.Series({d: value for d, value in ic_rows}, dtype=float)
    return {"mean_weekly_spearman_ic": float(ic.mean()), "ic_weeks": len(ic),
            "early_ic": float(ic.loc[:"2024-12-31"].mean()),
            "later_ic": float(ic.loc["2025-01-01":].mean()),
            "rolling_linear_oos_r2_vs_known_mean": 1 - sse_model / sse_mean if sse_mean else None,
            "prediction_count": prediction_count,
            "note": "评分线性回归的滚动预测诊断；非策略净值R²，也不是未参与选择的盲测"}


def write_report(directory: Path) -> None:
    table = pd.read_csv(directory / "comparison.csv").set_index("variant")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    coverage = json.loads((directory / "coverage.json").read_text(encoding="utf-8"))
    factors = json.loads((directory / "factor_diagnostics.json").read_text(encoding="utf-8"))
    intervals = json.loads((directory / "bootstrap.json").read_text(encoding="utf-8"))["annual_arithmetic_return_difference_95pct"]
    corr = pd.read_csv(directory / "correlation.csv", index_col=0)
    corr.index = corr.index.astype(str)
    holdings = pd.read_csv(directory / "holdings.csv")
    pct = lambda value: f"{value:.2%}"
    lines = ["# ETF 标的池与规则优化研究（2026-10-06）", "",
             f"区间：2022-01-04 至 {manifest['end']}；{int(table.loc['default16', 'trading_days'])} 个交易日。"
             "所有方案每周最后交易日盘后生成目标，下一交易日收盘成交；单边总成本5 bps，另以20 bps压力测试。"
             "最多3只风险ETF加1只国债，风险总目标上限70%、单只风险目标上限30%。",
             "", "本轮固定18个方案（包含默认），没有按结果修改候选参数。仅对全期夏普前两名进一步做主要持仓剔除和相邻参数检查。"
             "领先方案的选择本身仍有选择偏差。历史2022–2026已用于此前研究，以下全部为探索性回查。默认实盘与10月8日计划未切换。", "",
             "## 标的池覆盖", "",
             "现有16只包含10只国内股票ETF、2只海外股票ETF、3只黄金/商品ETF与1只国债ETF。"
             "主要缺口是中小盘、红利低波、医药与港股；创业板、半导体、新能源、人工智能存在重叠。"
             "石油产业ETF是行业股票暴露；有色和豆粕ETF是商品期货类暴露，不应把所有行业名称看成同一资产类别。", "",
             f"主区间日收益中，创业板与新能源相关约{corr.at['159915', '159875']:.2f}，"
             f"半导体与人工智能约{corr.at['159995', '515070']:.2f}，纳指与标普500约{corr.at['159941', '513500']:.2f}。"
             f"16只共同有数据的{coverage['common_days']}个交易日，相关矩阵特征值参与率约为{coverage['effective_correlation_dimension']:.2f}，"
             "说明代码数量并不等于独立收益来源数量；该维度仅是描述统计，不用于交易。", "",
             "## 完整比较", "", "| 方案 | 年化收益 | 夏普（零无风险利率） | 最大回撤 | 单边20bps年化 | 年买卖权重总和 |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, row in table.iterrows():
        lines.append(f"| {LABELS[name]} | {pct(row.annual_return)} | {row.sharpe_zero_rf:.2f} | {pct(row.max_drawdown)} | {pct(row.stress20_annual)} | {row.annual_one_way_turnover:.2f} |")
    lines += ["", "年买卖权重总和包含买和卖，单位为账户资产倍数，不是交易次数。2026年未结束，相关年化值为已观察区间的年化。", "",
              "## 研究判断", "",
              "16只池已具备较宽覆盖。本样本中，补齐全部六只风险ETF使年化从11.21%降至9.40%，扩池不能直接作为提高收益的理由。"
              "中证500降低了本样本回撤、略提高夏普，但年化几乎不变，早期分段夏普还略下降；标普500的小幅改善亦需前瞻观察。", "",
              "排名缓冲值得作为成本优化候选：年买卖权重总和约下降15%，20 bps成本下年化9.09%高于默认8.42%。"
              "不过前4名放宽到前5名后年化降至10.34%、夏普1.71，参数敏感性仍明显；收益差的抽样区间包含0。"
              "建议固定前4名版本做前瞻对照，暂不以本轮最高历史表现切换实盘。", "",
              "短期波动降仓在15/20/30日和五个主要标的剔除中均提高夏普，但收益都比相应默认低。"
              "它适合优先降低波动/回撤的需求，是防守力度的取舍。7%预算能提高本样本收益，但全期夏普下降，不能据此保证今后回撤低于5%。", "",
              "## 改进规则的实际含义", "",
              "- 排名缓冲：前次周目标中的ETF只要仍满足趋势/动量资格且在前4名，就优先保留；空位再按评分补足。仍每周重算权重，不等于取消周度调仓。",
              "- 短期波动降仓：用已知20日和60日协方差估计风险仓波动，前者更高时按两者波动之比降低风险仓；释放资金在国债动量为正时进入国债，否则现金。",
              "- 完整组合预算：把国债也计入20/60日组合波动，估计值超过6%时统一降仓并留现金。",
              "- 相关性筛选：按动量评分依次选择，但与已选资产60日相关系数大于0.8的候选跳过。",
              "- 多周期：评分改为20、40、80日动量等权平均除以60日波动；资格筛选仍为40日动量和40日趋势。",
              "- 7%预算：只提高风险预算，属于提高风险承担的对照，不能把额外收益全部解释成预测能力改善。", "",
              "## 分段表现", "", "| 方案 | 2022–2024年化 | 夏普 | 2025–2026已观察年化 | 夏普 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    names = ["default16", "add_510500", "expanded22", "correlation", "rank_buffer", "fast_vol", "risk_budget7"]
    for name in names:
        r = table.loc[name]
        lines.append(f"| {LABELS[name]} | {pct(r['2022_2024_annual'])} | {r['2022_2024_sharpe']:.2f} | {pct(r['2025_2026_annual'])} | {r['2025_2026_sharpe']:.2f} |")
    lines += ["", "## 稳健性复查", ""]
    robust_path = directory / "robustness.csv"
    if robust_path.exists():
        robust = pd.read_csv(robust_path)
        lines += ["| 方案 | 检查 | 年化 | 夏普 | 最大回撤 | 相对同样剔除后的默认年化差 | 夏普差 |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for r in robust.itertuples():
            lines.append(f"| {LABELS[r.variant]} | {r.check} | {pct(r.annual)} | {r.sharpe:.2f} | {pct(r.drawdown)} | {r.annual_difference * 100:+.2f}个百分点 | {r.sharpe_difference:+.2f} |")
    lines += ["", "20交易日成对循环块抽样的年化算术日均收益差探索性95%区间：", ""]
    for name in ("add_510500", "correlation", "rank_buffer", "fast_vol"):
        low, high = intervals[name]
        lines.append(f"- {LABELS[name]}：{low * 100:+.2f} 至 {high * 100:+.2f}个百分点。")
    lines += ["", "这些区间没有修正本轮及此前全部策略搜索；不能直接用来宣称统计显著或未来有效。", "",
              "## 因子预测诊断", "",
              "未来收益标签从信号后下一交易日收盘，算到下周信号后下一交易日收盘；仅用于评价。"
              "每周截面至少5只合格ETF才计IC。回归只使用当时已结束的标签，最近104个周截面、至少52周才拟合。", "",
              "| 评分 | 平均周度Spearman IC | 周数 | 2022–2024 IC | 2025–2026 IC | 滚动线性预测R² |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, f in factors.items():
        label = "默认20/40日" if name == "default_score" else "20/40/80日平均"
        lines.append(f"| {label} | {f['mean_weekly_spearman_ic']:.3f} | {f['ic_weeks']} | {f['early_ic']:.3f} | {f['later_ic']:.3f} | {f['rolling_linear_oos_r2_vs_known_mean']:.4f} |")
    lines += ["", "R²相对当时已知训练标签均值预测计算，只评价评分的简单线性预测，不等于策略净值拟合优度，也不是独立盲测。", "",
              "## 默认池实际持有分布", "", "| ETF | 持有交易日 | 全期平均仓位 |", "| --- | ---: | ---: |"]
    names_all = ETF_NAMES | CANDIDATES
    for r in holdings.query("variant == 'default16'").sort_values("mean_weight", ascending=False).itertuples():
        lines.append(f"| {r.symbol} {names_all[str(r.symbol)] if str(r.symbol) in names_all else ''} | {r.holding_days} | {pct(r.mean_weight)} |")
    lines += ["", "## 限制及复现", "",
              "36项真实数据截断检查过去目标权重差为0；另有未来行情扰动、排名缓冲边界及含国债组合预算的单元检查。"
              "数据是现在取得的版本，无法还原历史发布后修订。回测为连续权重与日收盘净值；未模拟整手、最低佣金、跨境溢价、盘中价差、现金利息及容量。"
              "小规模账户中，国债100份一手的金额可能较大，目标比例无法精确实现，未来优先补做整手/现金与真实费用的实盘验证。", "",
              "```powershell", ".\\.venv\\Scripts\\python.exe research_strategy_optimization.py --offline --robustness --end 2026-09-30", "```", "",
              "## 参考资料", "",
              "- [Time Series Momentum，Moskowitz等](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum)：多资产趋势思路，本地方案并非论文原策略复刻。",
              "- [Volatility Managed Portfolios，Moreira与Muir](https://www.nber.org/papers/w22208)：高波动时减少风险的启发；本地20/60日降仓规则为自行构造。",
              "- [Portfolio Implementation / Craftsmanship Alpha，AQR](https://www.aqr.com/-/media/AQR/Documents/Insights/Working-Papers/AQR--Craftsmanship-Alpha.pdf)：交易成本与调仓收益的取舍。",
              "- [The Probability of Backtest Overfitting，Bailey等](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)：重复选择带来的过拟合风险。",
              "- [红利低波512890产品资料](https://www.sse.com.cn/disclosure/fund/announcement/c/new/2026-03-18/512890_20260318_FNWM.pdf)、[创新药159992上市通知](https://www.szse.cn/disclosure/notice/t20200407_575761.html)。",
              "- [中证500产品资料](https://www.nffund.com/main/files/2023/02/27/810779453725.pdf)、[中证1000基金资料](https://www.nffund.com/new/personal-financing/detail.html?fundCode=512100)。",
              "- [标普500基金资料](https://www.bosera.com/column/infoDetail.do?classid=0002000200020005&infoid=2975929)、[恒生科技基金资料](https://fund.chinaamc.com/fund/513180/jijinfeilv.shtml)。",
              "- [5年期国债511010产品资料](https://www.sse.com.cn/disclosure/fund/announcement/c/new/2025-03-21/511010_20250321_5LLX.pdf)。", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    navs = pd.read_csv(directory / "nav.csv", index_col=0, parse_dates=True)
    display = ["default16", "rank_buffer", "fast_vol", "expanded22"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    colors = ["#244a73", "#df8b20", "#24998c", "#999999"]
    for name, color in zip(display, colors):
        nav = navs[name]
        axes[0].plot(nav.index, (nav - 1) * 100, label=LABELS[name], linewidth=1.7, color=color)
        axes[1].plot(nav.index, (nav / nav.cummax().clip(lower=1) - 1) * 100, linewidth=1.3, color=color)
    axes[0].set(title=f"ETF轮动策略研究对比｜2022-01-04 至 {manifest['end']}\n单边5 bps；连续权重；历史研究结果", ylabel="累计收益（%）")
    axes[1].set(ylabel="相对历史高点的回撤（%）")
    axes[1].axhline(-5, color="#a64b47", linestyle="--", linewidth=0.8, alpha=0.6)
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(directory / "comparison.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 9, 30))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--robustness", action="store_true")
    parser.add_argument("--combined", action="store_true", help="只检验7%%预算与排名缓冲的组合及邻近参数")
    parser.add_argument("--recent-audit", action="store_true", help="离线归因已保存组合回放的最近回撤，不重选策略")
    parser.add_argument("--priority-buffer", action="store_true", help="离线检验必选第1名及固定前2名对照，不切换默认")
    parser.add_argument("--summarize", action="store_true", help="仅从已保存结果生成中文报告和图，不重跑回测")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "optimization_20261006")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.priority_buffer:
        run_priority_buffer(args.end, args.output_dir / "priority_buffer", robustness=args.robustness)
        return
    if args.recent_audit:
        recent_drawdown_audit(args.output_dir / "combined_7pct", args.end)
        return
    if args.combined:
        directory = args.output_dir / "combined_7pct"
        if args.summarize:
            combined_report(directory)
        else:
            run_combined(args.end, directory, offline=args.offline, robustness=args.robustness)
        return
    if args.summarize:
        write_report(args.output_dir)
        print(f"report saved: {args.output_dir / 'report.md'}")
        return
    current = RiskConfig.previous_default()
    base_symbols = list(ETF_NAMES)
    cases = [("default16", base_symbols, "base", current)]
    cases += [(f"add_{s}", base_symbols + [s], "base", current) for s in EXPANSIONS]
    cases += [("add_mid_small", base_symbols + ["510500", "512100"], "base", current),
              ("add_dividend_health", base_symbols + ["512890", "159992"], "base", current),
              ("add_us_hk", base_symbols + ["513500", "513180"], "base", current),
              ("expanded22", base_symbols + list(EXPANSIONS), "base", current)]
    cases += [(mode, base_symbols, mode, current) for mode in RULES]
    cases += [("risk_budget7", base_symbols, "base", replace(current, target_vol=0.07)),
              ("shorter_treasury", base_symbols, "base", current)]
    manifest = {"start": str(START), "end": str(args.end), "default_strategy": STRATEGY_NAME,
                "research_only": True, "default_unchanged": True,
                "fixed_cases": [{"name": n, "symbols": s, "mode": m, "target_vol": c.target_vol}
                                for n, s, m, c in cases],
                "shorter_treasury_alias": {BOND: "511010"},
                "correlation_threshold": 0.8, "rank_buffer": 4,
                "execution": "weekly last trading close -> next trading close",
                "cost_bps": [5, 20], "max_risk_positions": 3,
                "not_independent_holdout": "2022-2026已用于此前选择，本研究全部为探索性复查",
                "assumptions": "连续权重；不含整手、最低佣金、溢折价限制、盘中成交与现金利息"}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    series = {}
    for symbol in base_symbols + list(CANDIDATES):
        print(f"load {symbol}", flush=True)
        series[symbol] = load_symbol(symbol, FETCH_START, args.end, ROOT / "data" / "cache", offline=args.offline)
    prices_all = pd.DataFrame({s: f.reindex(series["510300"].index) for s, f in series.items()})
    if prices_all.iloc[-1].isna().any():
        raise ValueError(f"末日行情缺失：{prices_all.columns[prices_all.iloc[-1].isna()].tolist()}")
    manifest["input_sha256"] = {s: hashlib.sha256((ROOT / "data" / "cache" / f"tushare_{s}_hfq.csv").read_bytes()).hexdigest() for s in series}
    manifest["download_first_last_dates"] = {s: [str(f.index[0].date()), str(f.index[-1].date())] for s, f in series.items()}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.download_only:
        print("download complete", flush=True)
        return
    all_frames = factor_frames(prices_all)
    prices_base = prices_all.loc[:, base_symbols]
    frames_base = subset_frames(all_frames, base_symbols)
    baseline_target = research_signals(prices_base, frames=frames_base)
    baseline = evaluate(prices_base, baseline_target, args.end)
    np.testing.assert_allclose(baseline_target.to_numpy(), risk_controlled_signals(prices_base, current).to_numpy())
    results, targets, inputs = {}, {}, {}
    summary, yearly, bootstrap, holdings, prefix = [], [], {}, [], []
    bear = frames_base[FactorSpec("trend", 120)]["510300"].shift(1).le(0)
    for name, symbols, mode, config in cases:
        print(f"evaluate {name}", flush=True)
        prices = prices_all.loc[:, symbols].copy()
        frames = subset_frames(all_frames, symbols)
        if name == "shorter_treasury":
            prices[BOND] = prices_all["511010"]
            frames = factor_frames(prices)
        target = research_signals(prices, mode, config=config, frames=frames)
        result = evaluate(prices, target, args.end)
        stress = evaluate(prices, target, args.end, cost=20)
        results[name], targets[name], inputs[name] = result, target, prices
        daily = result.equity["daily_return"]
        risky = [s for s in symbols if s != BOND]
        row = {"variant": name, **result.metrics, "stress20_annual": stress.metrics["annual_return"],
               "stress20_drawdown": stress.metrics["max_drawdown"],
               "mean_risk_weight": float(result.weights[risky].sum(axis=1).mean()),
               "mean_bond_weight": float(result.weights[BOND].mean()),
               "annual_one_way_turnover": float(result.metrics["traded_notional_sum"] * 252 / len(daily)),
               "weak_market_daily_q05": float(daily.loc[bear.reindex(daily.index)].quantile(0.05)),
               "worst20day_return": float(((1 + daily).rolling(20).apply(np.prod, raw=True) - 1).min())}
        for label, left, right in (("2022_2024", "2022-01-01", "2024-12-31"),
                                   ("2025_2026", "2025-01-01", str(args.end))):
            row.update({f"{label}_{key}": value for key, value in period_stats(daily.loc[left:right]).items()})
        summary.append(row)
        for year, group in daily.groupby(daily.index.year):
            yearly.append({"variant": name, "year": int(year), **period_stats(group),
                           "total_return": float((1 + group).prod() - 1)})
        if name != "default16":
            bootstrap[name] = bootstrap_difference(daily, baseline.equity["daily_return"])
        for symbol in symbols:
            actual_symbol = "511010" if name == "shorter_treasury" and symbol == BOND else symbol
            owned = result.weights[symbol]
            contributions = owned.shift(1).fillna(0) * prices[symbol].pct_change(fill_method=None).reindex(owned.index).fillna(0)
            holdings.append({"variant": name, "symbol": actual_symbol,
                             "holding_days": int(owned.gt(1e-8).sum()), "mean_weight": float(owned.mean()),
                             "max_weight": float(owned.max()), "selected_weeks": int(target.loc[str(START):, symbol].gt(1e-8).sum()),
                             "annual_arithmetic_gross_contribution": float(contributions.sum() * 252 / len(daily))})
        # 截断后重新计算：只比较两边都已完整结束的周，排除截断造成的临时周末。
        for cutoff in ("2023-06-30", "2025-06-30"):
            shortened = prices.loc[:cutoff]
            recalculated = research_signals(shortened, mode, config=config)
            dates = target.index.intersection(recalculated.index)
            dates = dates[dates < shortened.index[-1]]
            error = float((target.loc[dates] - recalculated.loc[dates]).abs().to_numpy().max())
            if error > 1e-12:
                raise AssertionError(f"{name} {cutoff} prefix error {error}")
            prefix.append({"variant": name, "cutoff": cutoff, "max_weight_difference": error})
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(args.output_dir / "comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(args.output_dir / "yearly.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(holdings).to_csv(args.output_dir / "holdings.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prefix).to_csv(args.output_dir / "prefix_audit.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "bootstrap.json").write_text(json.dumps({"annual_arithmetic_return_difference_95pct": bootstrap,
                                                               "not_corrected_for_multiple_trials": True}, indent=2), encoding="utf-8")
    diagnostics = {"default_score": factor_diagnostics(prices_base, frames_base, ensemble=False),
                   "ensemble_score": factor_diagnostics(prices_base, frames_base, ensemble=True)}
    (args.output_dir / "factor_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    returns = prices_all.pct_change(fill_method=None).loc[str(START):]
    returns.corr(min_periods=252).to_csv(args.output_dir / "correlation.csv", encoding="utf-8-sig")
    common = returns[base_symbols].dropna()
    eigen = np.linalg.eigvalsh(common.corr())
    (args.output_dir / "coverage.json").write_text(json.dumps({"common_days": len(common),
                 "effective_correlation_dimension": float(eigen.sum() ** 2 / np.square(eigen).sum()),
                 "dimension_is_descriptive_not_trading_input": True}, indent=2), encoding="utf-8")
    pd.concat({n: r.equity["strategy_nav"] for n, r in results.items()}, axis=1).to_csv(
        args.output_dir / "nav.csv", encoding="utf-8-sig")
    leaders = summary_frame.query("variant != 'default16' and max_drawdown >= -0.10").sort_values(
        "sharpe_zero_rf", ascending=False).head(2)["variant"].tolist()
    if args.robustness:
        robust_rows = []
        case_map = {n: (s, m, c) for n, s, m, c in cases}
        for name in leaders:
            symbols, mode, config = case_map[name]
            for removed in ("518880", "159941", "159980", "159985", "512800"):
                altered = inputs[name].drop(columns=removed)
                base_altered = prices_base.drop(columns=removed)
                r = evaluate(altered, research_signals(altered, mode, config=config), args.end)
                b = evaluate(base_altered, research_signals(base_altered), args.end)
                robust_rows.append({"variant": name, "check": f"remove_{removed}",
                                    "annual": r.metrics["annual_return"], "sharpe": r.metrics["sharpe_zero_rf"],
                                    "drawdown": r.metrics["max_drawdown"],
                                    "annual_difference": r.metrics["annual_return"] - b.metrics["annual_return"],
                                    "sharpe_difference": r.metrics["sharpe_zero_rf"] - b.metrics["sharpe_zero_rf"]})
            if mode in {"correlation", "rank_buffer", "fast_vol", "full_vol"}:
                neighbors = {"correlation": (0.7, 0.9), "rank_buffer": (3, 5),
                             "fast_vol": (15, 30), "full_vol": (15, 30)}[mode]
                for parameter in neighbors:
                    r = evaluate(inputs[name], research_signals(inputs[name], mode, parameter=parameter, config=config), args.end)
                    robust_rows.append({"variant": name, "check": f"neighbor_{parameter}",
                                        "annual": r.metrics["annual_return"], "sharpe": r.metrics["sharpe_zero_rf"],
                                        "drawdown": r.metrics["max_drawdown"],
                                        "annual_difference": r.metrics["annual_return"] - baseline.metrics["annual_return"],
                                        "sharpe_difference": r.metrics["sharpe_zero_rf"] - baseline.metrics["sharpe_zero_rf"]})
        pd.DataFrame(robust_rows).to_csv(args.output_dir / "robustness.csv", index=False, encoding="utf-8-sig")
    write_report(args.output_dir)
    print(summary_frame[["variant", "annual_return", "sharpe_zero_rf", "max_drawdown", "stress20_annual"]].round(4).to_string(index=False), flush=True)
    print(json.dumps(diagnostics, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
