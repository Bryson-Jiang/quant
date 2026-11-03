"""固定反转/反弹研究网格；共用正式选仓与成交引擎，不改动实盘计划。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from etf_rotation.backtest import run_backtest
from etf_rotation.config import LEGACY_CORE_NAMES as ETF_NAMES, BUFFERED_MOMENTUM_STRATEGY_NAME as STRATEGY_NAME
from etf_rotation.data import load_close_prices
from etf_rotation.factors import FactorSpec, compute_factor
from etf_rotation.risk_strategy import BOND, RiskConfig, risk_controlled_signals, weekly_signal_dates
from etf_rotation.strategy_state import load_research_rotation_state

ROOT = Path(__file__).resolve().parent
START = date(2022, 1, 1)
DESIGN_END = "2024-12-31"
DOMESTIC = ("510300", "159915", "159995", "159875", "159928", "512660",
            "515070", "512880", "512800", "561360")
FAMILIES = {"base": "当前默认", "short_term_reversal": "短期反转", "ma_zscore": "均线反转",
            "rsi": "RSI反转", "residual_reversal": "残差反转", "drawdown_rebound": "反弹确认",
            "crowding": "弱市拥挤防守", "unwind": "反弹切换防守"}
SOURCES = [
    ("Nagel：Evaporating Liquidity", "https://www.nber.org/papers/w17653"),
    ("Da等：Decomposing Short-Term Return Reversal", "https://www.newyorkfed.org/research/staff_reports/sr513.html"),
    ("Dai等：Reversals and the Returns to Liquidity Provision", "https://www.nber.org/papers/w30917"),
    ("Bailey等：The Probability of Backtest Overfitting", "https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf"),
]


@dataclass(frozen=True)
class Trial:
    family: str = "base"
    window: int = 0
    strength: float = 0.0
    scope: str = "all"

    @property
    def key(self) -> str:
        return "default" if self.family == "base" else f"{self.family}_{self.window}_{self.strength:g}_{self.scope}"

    @property
    def label(self) -> str:
        if self.family == "base":
            return FAMILIES[self.family]
        action = ("减仓50%" if self.strength == 0.5 else "禁选重排") if self.family in {"crowding", "unwind"} else f"强度{self.strength:g}"
        scope = {"all": "全池", "bank": "银行", "domestic": "国内股票"}[self.scope]
        return f"{FAMILIES[self.family]}／{self.window}日／{action}／{scope}"


def fixed_trials() -> list[Trial]:
    cases = [Trial()]
    for family, windows, strengths in (
            ("short_term_reversal", (3, 5, 10), (0.25, 0.5, 1.0)),
            ("ma_zscore", (10, 20, 40), (0.25, 0.5)),
            ("rsi", (7, 14), (0.25, 0.5)),
            ("residual_reversal", (3, 5, 10), (0.25, 0.5)),
            ("drawdown_rebound", (10, 20, 40), (0.25, 0.5))):
        cases.extend(Trial(family, k, a) for k in windows for a in strengths)
    cases.extend(Trial("crowding", k, a, scope) for k in (5, 10, 20)
                 for a in (0.5, 1.0) for scope in ("bank", "domestic"))
    cases.extend(Trial("unwind", k, a, "bank") for k in (5, 10, 20) for a in (0.5, 1.0))
    return cases


def feature_bundle(prices: pd.DataFrame, cases: list[Trial]) -> tuple[dict, dict, pd.DataFrame]:
    specs = {FactorSpec("momentum", n) for n in (20, 40, 60)}
    specs |= {FactorSpec("trend", 40), FactorSpec("volatility", 60)}
    specs |= {FactorSpec(c.family, c.window) for c in cases if c.family not in {"base", "crowding", "unwind"}}
    frames = {s: compute_factor(prices, s) for s in specs}
    vol = frames[FactorSpec("volatility", 60)]
    score = (frames[FactorSpec("momentum", 20)] + frames[FactorSpec("momentum", 40)]) * 0.5 / vol.clip(lower=0.05)
    spread = score.drop(columns=BOND).std(axis=1, ddof=1).fillna(0)
    returns = prices.pct_change(fill_method=None)
    market = prices["510300"]
    feature = {}
    for c in cases:
        if c.family == "base":
            continue
        if c.family not in {"crowding", "unwind"}:
            raw = frames[FactorSpec(c.family, c.window)]
            if c.family == "ma_zscore":
                raw = -raw / 3
            elif c.family == "rsi":
                raw = (50 - raw) / 50
            else:
                raw = raw / 3
            feature[c.key] = raw.clip(-1, 1).fillna(0).mul(spread * c.strength, axis=0)
            if c.scope != "all":
                columns = ["512800"] if c.scope == "bank" else list(DOMESTIC)
                feature[c.key].loc[:, ~prices.columns.isin(columns)] = 0.0
        else:
            k = c.window
            own = prices / prices.shift(k) - 1
            market_move = market / market.shift(k) - 1
            if c.family == "crowding":
                down_share = returns["510300"].lt(0).where(returns["510300"].notna()).rolling(k, min_periods=k).mean()
                weak = market_move.lt(0) & down_share.ge(0.6) & frames[FactorSpec("trend", 40)]["510300"].le(0)
                extended = (prices / prices.rolling(k, min_periods=k).mean() - 1).gt(0)
                flag = own.gt(0) & extended
                flag = flag.mul(weak, axis=0).astype(bool)
            else:
                # 先观察旧弱市，再观察最近3日市场已反弹、银行已转弱；不用未来反弹标签。
                old_decline = (market.shift(3) / market.shift(k + 3) - 1).lt(0)
                market_bounce = (market / market.shift(3) - 1).gt(0)
                old_relative = (prices.shift(3) / prices.shift(k + 3)).div(market.shift(3) / market.shift(k + 3), axis=0).gt(1)
                flag = old_relative & (prices / prices.shift(3) - 1).le(0)
                flag = flag.mul(old_decline & market_bounce, axis=0).astype(bool)
            columns = ["512800"] if c.scope == "bank" else list(DOMESTIC)
            flag.loc[:, ~flag.columns.isin(columns)] = False
            feature[c.key] = flag.fillna(False).astype(bool)
    return frames, feature, score


def targets(prices: pd.DataFrame, cases: list[Trial], choices: pd.Series,
            frames: dict, features: dict, *, removed: str | None = None) -> pd.DataFrame:
    """choices仅使用当日已选定配置；半仓不改变非零旧目标，禁选在核心筛选前执行。"""
    overlay = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    block = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    halve = pd.DataFrame(False, index=prices.index, columns=prices.columns)
    mapping = {c.key: c for c in cases}
    for name in choices.unique():
        c = mapping[name]
        if c.family == "base":
            continue
        dates = choices.index[choices.eq(name)]
        if c.family in {"crowding", "unwind"}:
            destination = block if c.strength == 1 else halve
            destination.loc[dates] = features[name].loc[dates]
        else:
            overlay.loc[dates] = features[name].loc[dates]
    overlay[BOND] = 0.0
    if removed:
        block[removed] = True
    altered = dict(frames)
    altered[FactorSpec("momentum", 40)] = frames[FactorSpec("momentum", 40)].mask(block)
    # 屏蔽项失去资格，其余项保留原始动量评分。
    state = load_research_rotation_state(ROOT)
    result = risk_controlled_signals(prices, RiskConfig.buffered_momentum(), factor_frames=altered,
                                     score_overlay=overlay, **state.signal_kwargs(prices))
    reductions = result.where(halve.reindex(result.index), 0) * 0.5
    reductions[BOND] = 0.0
    result -= reductions
    bond_ok = frames[FactorSpec("momentum", 60)][BOND].reindex(result.index).gt(0)
    result.loc[bond_ok, BOND] += reductions.sum(axis=1).loc[bond_ok]
    return result


def stats(daily: pd.Series) -> dict:
    nav = (1 + daily).cumprod()
    sigma = float(daily.std(ddof=1))
    return {"annual": float(nav.iloc[-1] ** (252 / len(daily)) - 1),
            "sharpe": float(daily.mean() / sigma * np.sqrt(252)) if sigma else 0.0,
            "drawdown": float((nav / nav.cummax().clip(lower=1) - 1).min()),
            "total": float(nav.iloc[-1] - 1)}


def choose_case(daily: pd.DataFrame, cutoff: pd.Timestamp, years: list[int]) -> str:
    """参数选择器只读截止日已实现净收益；固定训练期规则，不访问后段表现。"""
    known = daily.loc[:cutoff]
    base = stats(known["default"])
    ranked = []
    for name in known:
        metric = stats(known[name])
        wins = sum(stats(known[name].loc[str(year)])["total"] >=
                   stats(known["default"].loc[str(year)])["total"] - 1e-12 for year in years)
        if metric["drawdown"] >= -0.10 and metric["annual"] >= base["annual"] - 0.005 and wins >= np.ceil(2 * len(years) / 3):
            ranked.append((metric["sharpe"], name))
    return max(ranked)[1] if ranked else "default"


def evaluate(prices: pd.DataFrame, signal: pd.DataFrame, end: date):
    return run_backtest(prices, signal, START, end, fee_bps=3, slippage_bps=2)


def record(name: str, result, target: pd.DataFrame, baseline_target: pd.DataFrame) -> dict:
    daily = result.equity.daily_return
    stress = (1 + result.equity.gross_return) * (1 - result.equity.turnover * 20 / 10000) - 1
    row = {"variant": name, **stats(daily), **{f"stress20_{k}": v for k, v in stats(stress).items()},
           "turnover_per_year": float(result.equity.turnover.sum() * 252 / len(daily)),
           "mean_risk_weight": float(result.weights.drop(columns=[BOND, "CASH"]).sum(axis=1).mean()),
           "mean_bank_weight": float(result.weights["512800"].mean()),
           "changed_target_weeks": int((target - baseline_target).abs().max(axis=1).loc[str(START):].gt(1e-10).sum())}
    for prefix, left, right in (("design", "2022", DESIGN_END), ("validation", "2025", str(daily.index[-1].date()))):
        row.update({f"{prefix}_{k}": v for k, v in stats(daily.loc[left:right]).items()})
    return row


def bank_events(prices: pd.DataFrame, directory: Path) -> pd.DataFrame:
    market, bank = prices["510300"], prices["512800"]
    down = market.diff().lt(0)
    streak = down.astype(int).groupby((~down).cumsum()).cumsum()
    bank5 = bank / bank.shift(5) - 1
    market5 = market / market.shift(5) - 1
    conditions = {"大盘连跌至少3日": streak.ge(3),
                  "大盘连跌3日且银行近5日上涨": streak.ge(3) & bank5.gt(0),
                  "大盘弱5日且银行短期偏热": market5.lt(0) & down.rolling(5).sum().ge(3) & bank5.gt(0)
                                                  & bank.gt(bank.rolling(5).mean())}
    fbank = bank.shift(-6) / bank.shift(-1) - 1
    fmarket = market.shift(-6) / market.shift(-1) - 1
    details, summaries = [], []
    for label, condition in conditions.items():
        available = condition & fbank.notna() & fmarket.notna() & (prices.index >= pd.Timestamp(START))
        last = -100
        for i in np.flatnonzero(available.to_numpy()):
            if i > last + 6:
                last = i
                details.append({"condition": label, "signal_date": prices.index[i],
                                "entry_date": prices.index[i + 1], "exit_date": prices.index[i + 6],
                                "bank_return": float(fbank.iloc[i]), "market_return": float(fmarket.iloc[i])})
        observations = pd.DataFrame([r for r in details if r["condition"] == label])
        if observations.empty:
            continue
        for group, mask in (("全部", np.ones(len(observations), dtype=bool)),
                            ("2022–2024", observations.signal_date.le(pd.Timestamp(DESIGN_END))),
                            ("2025以后", observations.signal_date.gt(pd.Timestamp(DESIGN_END))),
                            ("随后大盘上涨（仅事后归因）", observations.market_return.gt(0)),
                            ("随后大盘未上涨（仅事后归因）", observations.market_return.le(0))):
            sample = observations.loc[mask]
            if len(sample):
                values = sample.bank_return.to_numpy()
                rng = np.random.default_rng(20261017)
                ci = np.quantile(rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1), [0.025, 0.975])
                summaries.append({"condition": label, "group": group, "events": len(sample),
                                  "bank_mean": float(sample.bank_return.mean()),
                                  "bank_median": float(sample.bank_return.median()),
                                  "bank_loss_rate": float(sample.bank_return.lt(0).mean()),
                                  "mean_ci_low": float(ci[0]), "mean_ci_high": float(ci[1]),
                                  "market_mean": float(sample.market_return.mean())})
    pd.DataFrame(details).to_csv(directory / "bank_events.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(summaries)
    summary.to_csv(directory / "bank_event_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def diagnostics(prices: pd.DataFrame, frames: dict, features: dict, score: pd.DataFrame,
                cases: list[Trial]) -> pd.DataFrame:
    risky = prices.columns.drop(BOND)
    eligible = (frames[FactorSpec("momentum", 40)].gt(0) & frames[FactorSpec("trend", 40)].gt(0)
                & frames[FactorSpec("volatility", 60)].gt(0))
    selected = [Trial(), Trial("short_term_reversal", 5, 0.5), Trial("ma_zscore", 20, 0.5),
                Trial("rsi", 14, 0.5), Trial("residual_reversal", 5, 0.5), Trial("drawdown_rebound", 20, 0.5)]
    week = weekly_signal_dates(prices.index)
    results = []
    for c in selected:
        values = score if c.family == "base" else features[c.key]
        groups, ics = [], []
        for day, following in zip(week[:-1], week[1:]):
            entry = prices.index.searchsorted(day, side="right")
            exit_ = prices.index.searchsorted(following, side="right")
            if exit_ >= len(prices) or day < pd.Timestamp(START):
                continue
            label = prices.iloc[exit_][risky] / prices.iloc[entry][risky] - 1
            x = values.loc[day, risky].where(eligible.loc[day, risky])
            pair = pd.DataFrame({"x": x, "y": label}).dropna()
            if len(pair) < 5 or pair.x.std() < 1e-12:
                continue
            ic = float(pair.x.rank().corr(pair.y.rank()))
            ics.append((day, ic))
            groups.append((day, prices.index[exit_], pair.x.to_numpy(), pair.y.to_numpy()))
        sse = denominator = 0.0
        predictions = 0
        for day, _, x, y in groups:
            known = [g for g in groups if g[1] <= day and g[0] >= day - pd.DateOffset(years=2)]
            if len(known) < 52:
                continue
            tx = np.concatenate([g[2] for g in known])
            ty = np.concatenate([g[3] for g in known])
            xm, ym = tx.mean(), ty.mean()
            beta = float(((tx - xm) * (ty - ym)).sum() / (((tx - xm)**2).sum() + 1e-6))
            predicted = ym + beta * (x - xm)
            sse += float(((y - predicted)**2).sum())
            denominator += float(((y - ym)**2).sum())
            predictions += len(y)
        ic_series = pd.Series({d: v for d, v in ics}, dtype=float)
        results.append({"factor": c.key, "ic": float(ic_series.mean()), "ic_weeks": len(ics),
                        "design_ic": float(ic_series.loc[:DESIGN_END].mean()),
                        "validation_ic": float(ic_series.loc["2025":].mean()),
                        "rolling_known_label_r2": 1 - sse / denominator if denominator else None,
                        "predicted_observations": predictions})
    return pd.DataFrame(results)


def bootstrap(daily: pd.DataFrame, leader_names: list[str], count: int = 2000) -> dict:
    """同一组20日循环块：成对区间与当前整组尝试的最大均值差检验。"""
    matrix = daily.to_numpy()
    base_pos = daily.columns.get_loc("default")
    diff = matrix - matrix[:, [base_pos]]
    centered = diff - diff.mean(axis=0)
    observed = float(diff.mean(axis=0).max())
    rng = np.random.default_rng(20261017)
    maxima, intervals = [], {name: [] for name in leader_names}
    for _ in range(count):
        starts = rng.integers(0, len(matrix), (len(matrix) + 19) // 20)
        index = ((starts[:, None] + np.arange(20)) % len(matrix)).ravel()[:len(matrix)]
        maxima.append(float(centered[index].mean(axis=0).max()))
        base = matrix[index, base_pos]
        base_sharpe = float(base.mean() / base.std(ddof=1) * np.sqrt(252))
        for name in leader_names:
            sample = matrix[index, daily.columns.get_loc(name)]
            intervals[name].append((float((sample - base).mean() * 252),
                                    float(sample.mean() / sample.std(ddof=1) * np.sqrt(252) - base_sharpe)))
    return {"block_days": 20, "replicates": count,
            "family_max_mean_return_p": float((1 + np.sum(np.array(maxima) >= observed)) / (count + 1)),
            "corrected_grid_trials": len(daily.columns) - 1,
            "observed_max_annual_arithmetic_return_difference": observed * 252,
            "paired_unadjusted_95pct": {name: {"annual_arithmetic_difference": np.quantile(np.array(v)[:, 0], [0.025, 0.975]).tolist(),
                                              "sharpe_difference": np.quantile(np.array(v)[:, 1], [0.025, 0.975]).tolist()}
                                         for name, v in intervals.items()},
            "limitation": "仅校正本轮固定网格的均值收益搜索；不校正此前研究、夏普挑选或非平稳市场"}


def classified_check(prices: pd.DataFrame, directory: Path, end: date) -> None:
    """首轮结果后的分类补充，单独记账，不冒称预先冻结或独立验证。"""
    cases = [Trial()]
    cases += [Trial(family, window, strength, "bank") for family, window in
              (("short_term_reversal", 5), ("ma_zscore", 20), ("rsi", 14), ("residual_reversal", 5))
              for strength in (0.25, 0.5)]
    cases += [Trial("ma_zscore", 20, strength, "domestic") for strength in (0.25, 0.5)]
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["software"] = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}
    manifest["classified_followup_cases"] = [asdict(c) | {"key": c.key} for c in cases[1:]]
    manifest["classified_followup_is_posthoc"] = True
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    frames, feature, _ = feature_bundle(prices, cases)
    results, signals, rows = {}, {}, []
    for c in cases:
        signal = targets(prices, cases, pd.Series(c.key, index=prices.index), frames, feature)
        result = evaluate(prices, signal, end)
        results[c.key], signals[c.key] = result, signal
        if c.family == "base":
            baseline_target = signal
        rows.append(record(c.key, result, signal, baseline_target) | asdict(c))
        print(f"classified {c.key}: SR={rows[-1]['sharpe']:.3f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(directory / "classified_comparison.csv", index=False, encoding="utf-8-sig")
    winner = table.loc[table.variant.ne("default")].sort_values("sharpe", ascending=False, kind="stable").iloc[0].variant
    daily = pd.concat({name: result.equity.daily_return for name, result in results.items()}, axis=1)
    check = bootstrap(daily, [winner])
    check["posthoc_followup_not_independent"] = True
    check["leader"] = winner
    (directory / "classified_bootstrap.json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
    baseline = pd.read_csv(directory / "comparison.csv").set_index("variant").loc["default"]
    np.testing.assert_allclose(rows[0]["annual"], baseline.annual, atol=1e-12)
    cutoff = "2025-06-30"
    short = prices.loc[:cutoff]
    sf, sx, _ = feature_bundle(short, cases)
    fresh = targets(short, cases, pd.Series(winner, index=short.index), sf, sx)
    dates = fresh.index[fresh.index < short.index[-1]]
    delta = float((fresh.loc[dates] - signals[winner].loc[dates]).abs().to_numpy().max())
    if delta > 1e-12:
        raise AssertionError("Classified follow-up prefix failed")
    changed = prices.copy()
    changed.loc[changed.index > pd.Timestamp(cutoff), "512800"] *= 9
    cf, cx, _ = feature_bundle(changed, cases)
    perturbed = targets(changed, cases, pd.Series(winner, index=prices.index), cf, cx)
    future_delta = float((perturbed.loc[:cutoff] - signals[winner].loc[:cutoff]).abs().to_numpy().max())
    if future_delta > 1e-12:
        raise AssertionError("Classified follow-up future perturbation failed")
    check.update({"max_prefix_weight_difference": delta, "max_future_perturbation_difference": future_delta,
                  "latest_target_difference_vs_default": float((signals[winner].iloc[-1] - baseline_target.iloc[-1]).abs().max())})
    (directory / "classified_bootstrap.json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(directory: Path, end: date, manifest: dict) -> Path:
    summary = pd.read_csv(directory / "comparison.csv").set_index("variant")
    yearly = pd.read_csv(directory / "yearly.csv")
    events = pd.read_csv(directory / "bank_event_summary.csv")
    diagnostic = pd.read_csv(directory / "factor_diagnostics.csv")
    ablation = pd.read_csv(directory / "ablation.csv")
    audits = pd.read_csv(directory / "time_audit.csv")
    boot = json.loads((directory / "bootstrap.json").read_text(encoding="utf-8"))
    choices = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    labels = {c.key: c.label for c in fixed_trials()} | {"walk_forward": "逐年仅按过去选择"}
    champion, design = choices["full_sample_leader"], choices["design_period_selection"]
    primary = Trial("short_term_reversal", 5, 0.5).key
    bank_primary = Trial("crowding", 10, 0.5, "bank").key
    names = list(dict.fromkeys(["default", primary, bank_primary, design, champion, "walk_forward"]))
    pct = lambda x: f"{x:.2%}"
    winner, base = summary.loc[champion], summary.loc["default"]
    selected_result = summary.loc[design]
    primary_events = events.loc[events.condition.eq("大盘连跌3日且银行近5日上涨") & events.group.eq("全部")].iloc[0]
    rebound_events = events.loc[events.condition.eq("大盘连跌3日且银行近5日上涨") & events.group.eq("随后大盘上涨（仅事后归因）")].iloc[0]
    lines = ["# ETF反转与反弹因子研究报告", "", f"研究日期：2026-10-17；行情截止：{end}。", "",
             "> 历史研究快照：本报告中的默认为当时7%版本；2026-10-18已选择8%＋20日均线反转0.5，现行参数见README。", "",
             "## 结论与适用范围", "",
             "研究已完成；保留当前7%预算＋前4名缓冲＋必选第1名的默认实盘。下面的最优项属于历史候选，未改写live_signal或10月19日计划。",
             f"固定比较{len(fixed_trials())}个方案（含默认），时间2022-01-04至{end}，16只ETF；标准单边5bps，高成本20bps。"
             "夏普为零无风险利率，回撤为日收盘净值。所有这些年份已参与此前研究；2025以后仅为按时间分段的复核，不能称为独立盲测。", "",
             f"首轮50项全样本夏普最高：**{labels[champion]}**；仅按2022–2024选择：**{labels[design]}**。"
             "请同时查看两者的后段与逐年表现，不能用全样本冠军替换训练期选择后再称样本外收益。", "",
             f"- 银行普遍在此后下跌的假说没有得到稳定支持：最贴近观察的{int(primary_events.events)}次事件，随后5日银行均值{pct(primary_events.bank_mean)}、亏损比例{pct(primary_events.bank_loss_rate)}；"
             f"随后大盘上涨的{int(rebound_events.events)}次，银行均值仍为{pct(rebound_events.bank_mean)}。",
             f"- 历史冠军年化比默认增加{(winner.annual-base.annual)*100:.2f}个百分点、夏普增加{winner.sharpe-base.sharpe:.3f}；"
             f"2025以后年化仅增加{(winner.validation_annual-base.validation_annual)*100:.2f}个百分点，夏普变化{winner.validation_sharpe-base.validation_sharpe:+.3f}。",
             f"- 真正按设计段选择的方案，2025以后年化{pct(selected_result.validation_annual)}、夏普{selected_result.validation_sharpe:.3f}，"
             f"同期默认为{pct(base.validation_annual)}、{base.validation_sharpe:.3f}；滚动选择路径全期夏普{summary.loc['walk_forward','sharpe']:.3f}。",
             f"- 首轮49个非默认尝试校正后均值收益差探索性p值{boot['family_max_mean_return_p']:.3f}；冠军的收益差和夏普差成对区间均包含零。"
             "收益点估计值得继续跟踪，证据不足以替换默认策略。", "",
             "## 一、银行短期高点假说", "",
             "先把观察写成固定事件：大盘至少连跌3日；在此基础上要求银行近5日上涨；另比较大盘近5日走弱且银行位于短均线之上。"
             "信号日之后下一交易日收盘才计入，再持有5个交易日。每组事件间隔至少7个交易日，组内持有窗口不重叠。"
             "后续大盘上涨/未上涨的分组只用于事后归因，绝不作为交易输入；事件少时不据概率点估计下结论。", "",
             "| 条件 | 分组 | 事件数 | 银行均值 | 银行中位数 | 银行亏损比例 | 大盘均值 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in events.itertuples():
        lines.append(f"| {r.condition} | {r.group} | {r.events} | {pct(r.bank_mean)} | {pct(r.bank_median)} | {pct(r.bank_loss_rate)} | {pct(r.market_mean)} |")
    if "mean_ci_low" in events:
        lines += ["", f"‘大盘连跌3日且银行近5日上涨’组的事件均值重抽样探索性95%区间为[{pct(primary_events.mean_ci_low)}, {pct(primary_events.mean_ci_high)}]，包含零；"
                  "事件间隔虽去掉组内持有期重叠，仍可能存在市场状态相关性，区间未作多重比较修正。"]
    lines += ["", f"值得关注的是相对表现：随后大盘上涨的{int(rebound_events.events)}次，大盘均值{pct(rebound_events.market_mean)}，银行为{pct(rebound_events.bank_mean)}。"
              "这提示反弹时持有防御资产可能有机会成本；低β及有限样本也可能解释差异，尚不能证明可交易择时优势。"
              "如果其他ETF仍未通过40日资格，单纯调整评分也无法让它们入选。", "",
              "这些条件无法直接确认‘总处于高点’。收益条件、观察窗口与样本年份都会影响结论；不使用未来回调来反向定义入场日。", "",
              "## 二、因子与策略定义", "",
              "1. **短期反转**：过去k日对数收益取负，再除以60日历史日波动×√k。窗口3/5/10日，叠加强度0.25/0.5/1。",
              "2. **均线反转**：价格相对10/20/40日均线的标准偏离取负；强度0.25/0.5。",
              "3. **RSI反转**：(50−RSI)/50，RSI采用7/14日简单移动平均涨跌幅，平盘为50；非Wilder版本。",
              "4. **残差反转**：短期对数收益剔除β×同期沪深300对数收益后取负。β与残差波动只用冲击开始前60日；窗口3/5/10日。",
              "5. **反弹确认**：t−3之前10/20/40日窗口回撤深度，乘最近3日已发生的正对数收益，分别按60日波动缩放。",
              "6. **弱市拥挤防守**：沪深300k日下跌、窗口至少60%交易日下跌且低于40日均线；风险ETF同期上涨且高于k日均线。比较仅银行和全部10只国内股票ETF，窗口5/10/20日。",
              "7. **反弹切换防守**：前k日市场下跌且银行相对跑赢；最近3日市场已反弹而银行已转弱。比较窗口5/10/20日。", "",
              "前五类保留40日动量与趋势资格，把有界因子乘**当日**原评分截面标准差后小幅叠加，避免对不同量纲直接相加。"
              "短期/残差因子先除3再截断到[−1,1]，均线偏离同样处理；反弹因子只做正向奖励。国债不参与评分。"
              "因此超跌但尚未通过40日资格的ETF不会因本研究被买入。", "",
              "因子库另保留标准化相对大盘强弱 `relative_strength`：k日自身对数收益减沪深300同期对数收益，再除以60日自身波动×√k。"
              "它是可复用描述因子，不作为本次新增第51个调参方案。", "",
              "两类防守分别测试半仓防守和禁选重排：半仓将受影响资产的周目标减半，释放至正60日动量国债，否则现金；禁选在排名之前执行并按原规则重排。"
              "它们不是同一连续参数的两点。半仓目标仍为正，缓冲历史一致；禁选通过核心资格筛选更新历史，不保留已归零目标。", "",
              "## 三、组合表现", "",
              "| 方案 | 年化 | 夏普 | 最大回撤 | 20bps年化 | 20bps回撤 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in names:
        r = summary.loc[name]
        lines.append(f"| {labels[name]} | {pct(r.annual)} | {r.sharpe:.3f} | {pct(r.drawdown)} | {pct(r.stress20_annual)} | {pct(r.stress20_drawdown)} |")
    lines += ["", f"冠军的全期平均银行仓位{pct(winner.mean_bank_weight)}，默认为{pct(base.mean_bank_weight)}。"
              "因此其改善不能简单归因为少持银行。反转评分只在合格池内改变排名，合格标的不超过3只时，纯排名叠加无法改变入选集合。", "",
              "| 家族的全样本最优项（事后描述） | 年化 | 夏普 | 回撤 |", "| --- | ---: | ---: | ---: |"]
    raw_cases = summary.loc[summary.index != "walk_forward"]
    for _, group in raw_cases.groupby("family", sort=False):
        best = group.sort_values("sharpe", ascending=False, kind="stable").iloc[0]
        lines.append(f"| {labels[best.name]} | {pct(best.annual)} | {best.sharpe:.3f} | {pct(best.drawdown)} |")
    lines += ["", "![净值、回撤及银行敞口](assets/reversal_20261017.png)", "",
              "## 四、时间分段与逐年复核", "",
              "2022–2024作为事先指定设计段。设计段候选须满足年化不低于默认超过0.5个百分点、最大回撤不超过10%、至少2/3年份收益差非负，再按夏普选择；没有合格项就保留默认。", "",
              "| 方案 | 设计段年化 | 设计段夏普 | 2025以后年化 | 2025以后夏普 | 2025以后回撤 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in names:
        r = summary.loc[name]
        lines.append(f"| {labels[name]} | {pct(r.design_annual)} | {r.design_sharpe:.3f} | {pct(r.validation_annual)} | {r.validation_sharpe:.3f} | {pct(r.validation_drawdown)} |")
    lines += ["", "| 方案 | " + " | ".join(str(y) for y in sorted(yearly.year.unique())) + " |",
              "| --- | " + " | ".join("---:" for _ in sorted(yearly.year.unique())) + " |"]
    for name in names:
        rows = yearly.loc[yearly.variant.eq(name)].set_index("year")
        lines.append("| " + labels[name] + " | " + " | ".join(pct(rows.at[y, "total"]) for y in sorted(yearly.year.unique())) + " |")
    lines += ["", "2026年尚未结束，逐年表列实际累计收益；分段年化采用252交易日口径。滚动路径2022–2023沿用默认，2024/2025/2026每年仅看当时已经实现的收益选择；训练末端留5个交易日间隔。"
              "路径在同一核心引擎中重放并支付真实切换成本，不拼接不同历史仓位的假想收益。它仍是已观察历史上的复核，不是前瞻实盘。", "",
              "逐年选择：" + "；".join(f"{r['year']}→{labels[r['variant']]}" for r in choices["walk_forward_choices"]) + "。", "",
              "## 五、参数敏感性与完整尝试记录", "",
              "下表保留与全样本冠军同一家族、同一作用范围的全部固定参数。全部方案的汇总在本地 `comparison.csv`，包含失败项；未保留逐项独立策略脚本。", "",
              "| 方案 | 年化 | 夏普 | 回撤 | 后段年化 | 后段夏普 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    champion_case = next(c for c in fixed_trials() if c.key == champion)
    for c in fixed_trials():
        if c.family == champion_case.family and c.scope == champion_case.scope:
            r = summary.loc[c.key]
            lines.append(f"| {c.label} | {pct(r.annual)} | {r.sharpe:.3f} | {pct(r.drawdown)} | {pct(r.validation_annual)} | {r.validation_sharpe:.3f} |")
    classified_path = directory / "classified_comparison.csv"
    if classified_path.exists():
        classified = pd.read_csv(classified_path)
        classified_ci = json.loads((directory / "classified_bootstrap.json").read_text(encoding="utf-8"))
        lines += ["", "### 分类补充：仅调整银行或国内股票ETF", "",
                  "在看过首轮结果后增加10项分类检验：银行专用5日反转、20日均线反转、14日RSI、5日残差反转，强度各0.25/0.5；另测国内10只股票ETF的20日均线反转。"
                  "这些是后续探索，不能称为预先固定或独立盲测。全轮共60个静态参数配置，另有1条滚动选择路径。按年滚动选择仅针对首轮50项，不能拿分类补充的事后最优替换它。", "",
                  "| 分类方案 | 年化 | 夏普 | 回撤 | 后段年化 | 后段夏普 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for r in classified.loc[classified.variant.ne("default")].itertuples():
            c = Trial(r.family, int(r.window), r.strength, r.scope)
            lines.append(f"| {c.label} | {pct(r.annual)} | {r.sharpe:.3f} | {pct(r.drawdown)} | {pct(r.validation_annual)} | {r.validation_sharpe:.3f} |")
        lines += ["", f"分类10项内部的居中最大均值差探索性p值{classified_ci['family_max_mean_return_p']:.3f}；"
                  "不包含其在首轮结果之后才设定的自适应选择偏差，不能与前面的首轮49项校正混为一谈。"
                  f"分类冠军的历史截断及未来扰动最大权重差为{max(classified_ci['max_prefix_weight_difference'],classified_ci['max_future_perturbation_difference']):.3g}。"]
        followup = classified.loc[classified.variant.ne("default")].sort_values("sharpe", ascending=False, kind="stable").iloc[0]
        lines += ["", f"分类补充的最高夏普{followup.sharpe:.3f}，默认为{base.sharpe:.3f}；"
                  "本次没有发现只针对银行或只针对国内股票反转更优的稳定证据。"]
    lines += ["", "## 六、预测性与统计不确定性", "",
              "因子IC只在至少5只合格风险ETF、因子有截面差异的周计算，标签为下一次收盘执行到再下次收盘执行的真实可持有区间，非信号当日收益。"
              "R²是过去2年滚动线性预测，至少52个已结束标签周，所有训练标签在预测日已结束；对照为当时已知训练标签均值。标签未扣个股交易费用。", "",
              "| 因子 | 平均IC | IC周数 | 前段IC | 后段IC | 滚动R² |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in diagnostic.itertuples():
        lines.append(f"| {labels[r.factor]} | {r.ic:.4f} | {r.ic_weeks} | {r.design_ic:.4f} | {r.validation_ic:.4f} | {r.rolling_known_label_r2:.4f} |")
    lines += ["", "短期反转、均线反转和残差反转的IC在本样本为负，滚动线性R²也未提供稳定预测增益。"
              "个别组合净值改善可能来自少数周的换仓路径、风险结构与缓冲交互，不能据净值点估计声称反转因子已具有独立预测能力。"
              "周度信号及下一交易日收盘执行也可能错过更快的反转；本轮没有为追求更好回测而改成当天收盘成交。"]
    lines += ["", f"20日成对循环块抽样{boot['replicates']}次，对本轮{boot['corrected_grid_trials']}个非默认尝试的最大均值收益差进行居中最大值校正，"
              f"探索性p值 **{boot['family_max_mean_return_p']:.3f}**。这项检验针对日均收益差，不等同于冠军夏普的显著性，也没有涵盖此前所有调参。", "",
              "| 候选 | 年化算术收益差95%区间 | 夏普差95%区间 |", "| --- | ---: | ---: |"]
    for name, ci in boot["paired_unadjusted_95pct"].items():
        a, b = ci["annual_arithmetic_difference"], ci["sharpe_difference"]
        lines.append(f"| {labels[name]} | [{pct(a[0])}, {pct(a[1])}] | [{b[0]:.3f}, {b[1]:.3f}] |")
    lines += ["", "成对区间本身未作多重比较修正，且市场非平稳、ETF之间相关。均值差区间与复利年化差定义不同。", "",
              "## 七、单标的依赖与未来信息检查", "",
              "逐一禁用15只风险ETF的交易资格，对候选和默认使用相同禁用池；价格仍保留作为市场/截面信息源，尤其沪深300继续作为大盘过滤。", "",
              "| 候选 | 同池年化提高次数 | 同池夏普提高次数 |", "| --- | ---: | ---: |"]
    for name, group in ablation.groupby("variant"):
        lines.append(f"| {labels[name]} | {int(group.annual_delta.gt(1e-10).sum())}/{len(group)} | {int(group.sharpe_delta.gt(1e-10).sum())}/{len(group)} |")
    lines += ["", f"时点审计共{len(audits)}项，最大过去目标差 **{audits.max_weight_difference.max():.3g}**。"
              "包含真实历史截断和未来行情扰动；滚动选择只用当时已实现净收益，收益后扣费、周中漂移、缺价取消成交均复用正式引擎。"
              "新增因子另有方向、缺价、跳过窗口、残差β时点测试。", "",
              "项目62项单元测试通过，分类补充另做2项时点检查；主要审计与分类审计共22项。"
              "清理后的统一实现再次重放默认与历史冠军，净值最大差约2.3×10⁻¹⁶，属于CSV浮点舍入。", "",
              "## 八、最终处置与复现", "",
              "保留统一研究入口 `research_reversal.py`，五种通用新因子保留在因子库，含有目前未改善组合的因子，供后续研究。"
              "淘汰候选只保留汇总结果与固定网格记录，不保留逐策略脚本、逐策略数据副本或中间净值图。默认策略、实盘信号和账户数据库保持研究前状态。", "",
              "**建议**：以冻结规则后的新行情验证候选；先改善证据强度，再决定是否接入实盘。不能因为银行近期强就单独对其加一个未经跨年份验证的规则。", "",
              "复现：", "", "```powershell", f".\\.venv\\Scripts\\python.exe research_reversal.py --end {end}", "```", "",
              "结果目录 `output/reversal_20261017/`；首次复现需自行下载相同区间行情。数据为当前Tushare版本，无法还原历史发布版本。"
              "回测未模拟整手、最低佣金、跨境溢价、盘中波动及现金利息；收益差不保证能在真实账户实现。"]
    latest_path = directory / "latest_targets.csv"
    if latest_path.exists():
        latest = pd.read_csv(latest_path, index_col=0)
        difference = float((latest.loc["default"] - latest.loc[champion]).abs().max())
        lines += ["", "## 当前周目标核对", "",
                  f"首轮历史冠军与默认的最新目标最大差为{difference:.3g}。本周只有银行与纳指通过原绝对筛选，纯排名叠加没有改变入选集合；"
                  "研究不会改写10月19日原计划。若日后采用新规则，须明确切换时间并沿用经过核对的历史状态。"]
    lines += ["", "## 资料依据", "",
              "流动性供给研究支持把短期反转视为条件性假说；行业动量与反转机制可能并存，因此本研究没有假定个股研究直接适用于A股ETF。"
              "上述因子、阈值和周度实现是本地构造，不是论文原策略复刻。", ""]
    lines.extend(f"- [{title}]({url})" for title, url in SOURCES)
    lines += ["", f"数据指纹：`{manifest['price_sha256']}`。实盘文件保护指纹前后校验通过。", ""]
    path = ROOT / "docs/reversal_research_20261017.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 16))
    parser.add_argument("--summarize", action="store_true", help="只从已保存的最终汇总表重建Markdown报告")
    parser.add_argument("--classified-only", action="store_true", help="已有首轮汇总后，仅复现10项分类补充")
    args = parser.parse_args()
    directory = ROOT / "output/reversal_20261017"
    directory.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        saved = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        print(write_report(directory, date.fromisoformat(saved["end"]), saved))
        return
    if args.classified_only:
        prices = load_close_prices(START, args.end, ROOT / "data/cache", offline=True)
        classified_check(prices, directory, args.end)
        saved = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        print(write_report(directory, args.end, saved))
        return
    protected = [ROOT / p for p in ("live_signal.py", "run_backtest.py", "etf_rotation/risk_strategy.py",
                                    "etf_rotation/config.py", "data/rotation_state.json", "output/live_signal.json")]
    before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.exists()}
    cases = fixed_trials()
    prices = load_close_prices(START, args.end, ROOT / "data/cache", offline=True)
    manifest = {"start": str(START), "end": str(args.end), "strategy": STRATEGY_NAME,
                "cases": [{**asdict(c), "key": c.key} for c in cases], "design_end": DESIGN_END,
                "price_sha256": hashlib.sha256(pd.util.hash_pandas_object(prices, index=True).values.tobytes()).hexdigest(),
                "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
                "default_unchanged": True, "past_data_already_used_in_research": True,
                "training_selection_rule": "DD<=10%, annual >= baseline-0.5pp, annual return improvement in >=2/3 training years, highest Sharpe; otherwise default",
                "primary_predeclared": Trial("short_term_reversal", 5, 0.5).key,
                "bank_predeclared": Trial("crowding", 10, 0.5, "bank").key}
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    frames, features, score = feature_bundle(prices, cases)
    results, all_targets, rows, yearly = {}, {}, [], []
    for c in cases:
        signal = targets(prices, cases, pd.Series(c.key, index=prices.index), frames, features)
        result = evaluate(prices, signal, args.end)
        if c.family == "base":
            baseline_target = signal
            stress_engine = run_backtest(prices, signal, START, args.end, fee_bps=20, slippage_bps=0)
            analytic = (1 + result.equity.gross_return) * (1 - result.equity.turnover * 20 / 10000) - 1
            np.testing.assert_allclose(stress_engine.equity.daily_return, analytic, atol=1e-12)
        results[c.key], all_targets[c.key] = result, signal
        rows.append({**record(c.key, result, signal, baseline_target), **asdict(c)})
        for year, daily in result.equity.daily_return.groupby(result.equity.index.year):
            yearly.append({"variant": c.key, "year": int(year), **stats(daily)})
        print(f"evaluated {c.key}: {rows[-1]['annual']:.3%} SR={rows[-1]['sharpe']:.3f}", flush=True)
    table = pd.DataFrame(rows).set_index("variant")
    past_daily = pd.concat({c.key: results[c.key].equity.daily_return for c in cases}, axis=1)
    design = choose_case(past_daily, pd.Timestamp(DESIGN_END), [2022, 2023, 2024])
    champion = table.loc[table.drawdown.ge(-0.10)].sort_values("sharpe", ascending=False, kind="stable").index[0]
    choices = pd.Series("default", index=prices.index)
    fold_rows = []
    for year in (2024, 2025, 2026):
        prior_days = prices.index[prices.index < pd.Timestamp(f"{year}-01-01")]
        cutoff = prior_days[-6]
        selected = choose_case(past_daily, cutoff, list(range(2022, year)))
        choices.loc[choices.index.year == year] = selected
        fold_rows.append({"year": year, "training_end": str(cutoff.date()), "variant": selected})
    wf_target = targets(prices, cases, choices, frames, features)
    wf = evaluate(prices, wf_target, args.end)
    results["walk_forward"], all_targets["walk_forward"] = wf, wf_target
    rows.append(record("walk_forward", wf, wf_target, baseline_target))
    for year, daily in wf.equity.daily_return.groupby(wf.equity.index.year):
        yearly.append({"variant": "walk_forward", "year": int(year), **stats(daily)})
    pd.DataFrame(rows).to_csv(directory / "comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(directory / "yearly.csv", index=False, encoding="utf-8-sig")
    primary = Trial("short_term_reversal", 5, 0.5).key
    bank_primary = Trial("crowding", 10, 0.5, "bank").key
    selection = {"full_sample_leader": champion, "design_period_selection": design, "walk_forward_choices": fold_rows}
    (directory / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    leaders = list(dict.fromkeys([champion, design, primary, bank_primary]))
    main_daily = pd.concat({c.key: results[c.key].equity.daily_return for c in cases}, axis=1)
    boot = bootstrap(main_daily, leaders)
    (directory / "bootstrap.json").write_text(json.dumps(boot, ensure_ascii=False, indent=2), encoding="utf-8")
    bank_events(prices, directory)
    diagnostics(prices, frames, features, score, cases).to_csv(directory / "factor_diagnostics.csv", index=False, encoding="utf-8-sig")
    audited = list(dict.fromkeys(["default", primary, bank_primary, champion, design, "walk_forward"]))
    checks = []
    for cutoff in ("2023-06-30", "2025-06-30", "2026-09-30"):
        shorter = prices.loc[:cutoff]
        short_frames, short_features, _ = feature_bundle(shorter, cases)
        for name in audited:
            path = choices.loc[shorter.index] if name == "walk_forward" else pd.Series(name, index=shorter.index)
            fresh = targets(shorter, cases, path, short_frames, short_features)
            dates = fresh.index.intersection(all_targets[name].index)
            dates = dates[dates < shorter.index[-1]]
            error = float((fresh.loc[dates] - all_targets[name].loc[dates]).abs().to_numpy().max())
            if error > 1e-12:
                raise AssertionError(f"prefix failure {name} {cutoff}: {error}")
            checks.append({"variant": name, "check": f"prefix_{cutoff}", "max_weight_difference": error})
    changed = prices.copy()
    changed.loc[changed.index > pd.Timestamp("2025-06-30"), "512800"] *= 9
    changed_frames, changed_features, _ = feature_bundle(changed, cases)
    for name in audited:
        path = choices if name == "walk_forward" else pd.Series(name, index=prices.index)
        fresh = targets(changed, cases, path, changed_frames, changed_features)
        error = float((fresh.loc[:"2025-06-30"] - all_targets[name].loc[:"2025-06-30"]).abs().to_numpy().max())
        if error > 1e-12:
            raise AssertionError(f"future perturbation failure: {name}")
        checks.append({"variant": name, "check": "future_bank_x9", "max_weight_difference": error})
    pd.DataFrame(checks).to_csv(directory / "time_audit.csv", index=False, encoding="utf-8-sig")
    ablations = []
    ablation_leaders = list(dict.fromkeys([champion, design, bank_primary]))
    ablation_leaders = [name for name in ablation_leaders if name != "default"]
    for removed in prices.columns.drop(BOND):
        same_base = targets(prices, cases, pd.Series("default", index=prices.index), frames, features, removed=removed)
        base_result = evaluate(prices, same_base, args.end)
        for name in ablation_leaders:
            signal = targets(prices, cases, pd.Series(name, index=prices.index), frames, features, removed=removed)
            result = evaluate(prices, signal, args.end)
            ablations.append({"variant": name, "removed": removed, "annual": result.metrics["annual_return"],
                              "sharpe": result.metrics["sharpe_zero_rf"], "drawdown": result.metrics["max_drawdown"],
                              "annual_delta": result.metrics["annual_return"] - base_result.metrics["annual_return"],
                              "sharpe_delta": result.metrics["sharpe_zero_rf"] - base_result.metrics["sharpe_zero_rf"]})
        print(f"ablation {removed} complete", flush=True)
    pd.DataFrame(ablations).to_csv(directory / "ablation.csv", index=False, encoding="utf-8-sig")
    plotted = list(dict.fromkeys(["default", primary, bank_primary, design, champion, "walk_forward"]))
    pd.DataFrame({n: all_targets[n].iloc[-1] for n in ["default", champion]}).T.to_csv(
        directory / "latest_targets.csv", encoding="utf-8-sig")
    pd.concat({n: results[n].equity.strategy_nav for n in plotted}, axis=1).to_csv(directory / "nav.csv", encoding="utf-8-sig")
    holdings = []
    for name in plotted:
        for symbol in results[name].weights:
            owned = results[name].weights[symbol]
            holdings.append({"variant": name, "symbol": symbol, "holding_days": int(owned.gt(1e-10).sum()),
                             "mean_weight": float(owned.mean()), "max_weight": float(owned.max())})
    pd.DataFrame(holdings).to_csv(directory / "holdings.csv", index=False, encoding="utf-8-sig")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True, gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    labels = {c.key: c.label for c in cases} | {"walk_forward": "逐年仅按过去选择"}
    for name in plotted:
        nav = results[name].equity.strategy_nav
        width = 2.5 if name == "default" else 1.2
        axes[0].plot(nav.index, (nav - 1) * 100, label=labels[name], linewidth=width)
        axes[1].plot(nav.index, (nav / nav.cummax().clip(lower=1) - 1) * 100, linewidth=width)
        axes[2].plot(nav.index, results[name].weights["512800"].rolling(20).mean() * 100, linewidth=width)
    axes[0].set_title(f"ETF反转与反弹研究｜2022-01-04至{args.end}｜单边5bps\n历史研究比较，未切换默认实盘")
    axes[0].set_ylabel("累计收益（%）")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].set_ylabel("收盘回撤（%）")
    axes[2].set_ylabel("银行仓位20日均值（%）")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.axvline(pd.Timestamp("2025-01-01"), color="grey", linestyle="--", alpha=0.5)
    fig.tight_layout()
    assets = ROOT / "docs/assets"
    assets.mkdir(parents=True, exist_ok=True)
    fig.savefig(assets / "reversal_20261017.png", dpi=150)
    plt.close(fig)
    after = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.exists()}
    if before != after:
        raise AssertionError("Protected production file changed during research")
    manifest.update({"protected_files": before, "protected_files_unchanged": True,
                     "time_audits": len(checks), "eligible_count_le_three_share": float(
                         (frames[FactorSpec("momentum", 40)].drop(columns=BOND).gt(0)
                          & frames[FactorSpec("trend", 40)].drop(columns=BOND).gt(0)).sum(axis=1).loc[str(START):].le(3).mean())})
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    classified_check(prices, directory, args.end)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    report = write_report(directory, args.end, manifest)
    print(f"COMPLETE {report}", flush=True)
    print(json.dumps(selection, ensure_ascii=True), flush=True)
    print(pd.DataFrame(rows)[["variant", "annual", "sharpe", "drawdown"]].sort_values("sharpe", ascending=False).head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
