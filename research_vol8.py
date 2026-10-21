"""8%波动预算与有界因子叠加研究；只读离线行情，不修改实盘默认。"""

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
from etf_rotation.config import ETF_NAMES, BUFFERED_MOMENTUM_STRATEGY_NAME as STRATEGY_NAME
from etf_rotation.data import load_close_prices
from etf_rotation.factors import FactorSpec, compute_factor
from etf_rotation.risk_strategy import BOND, RiskConfig, risk_controlled_signals, weekly_signal_dates
from etf_rotation.strategy_state import load_research_rotation_state
from research_reversal import bootstrap, choose_case, stats

ROOT = Path(__file__).resolve().parent
START = date(2022, 1, 1)
OUT = ROOT / "output/vol8_20261017"
EXTRAS = {"trend_efficiency": "趋势效率", "downside_balance": "涨跌波动平衡",
          "relative_strength": "相对强弱", "breakout": "突破", "volatility_ratio": "波动稳定性"}
SOURCES = [
    ("Moreira与Muir：Volatility Managed Portfolios", "https://www.nber.org/papers/w22208"),
    ("Hurst等：A Century of Evidence on Trend-Following Investing",
     "https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing"),
    ("Da等：Frog in the Pan: Continuous Information and Momentum",
     "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2370931"),
    ("Bailey等：The Probability of Backtest Overfitting", "https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf"),
]


@dataclass(frozen=True)
class Case:
    budget: float = 0.07
    ma_window: int = 0
    ma_strength: float = 0.0
    extra: str = ""
    strength: float = 0.0

    @property
    def key(self) -> str:
        if self == Case():
            return "default"
        stem = f"v{self.budget * 100:g}"
        if self.ma_window:
            stem += f"_ma{self.ma_window}_{self.ma_strength:g}"
        if self.extra:
            stem += f"_{self.extra}_{self.strength:g}"
        return stem + ("_plain" if not self.ma_window and not self.extra else "")

    @property
    def label(self) -> str:
        stem = f"{self.budget:.0%}预算" if self.budget in (0.07, 0.08, 0.09) else f"{self.budget:.1%}预算"
        if self.ma_window:
            stem += f"＋均线{self.ma_window}/{self.ma_strength:g}"
        if self.extra:
            stem += f"＋{EXTRAS[self.extra]}/{self.strength:g}"
        return stem


def fixed_cases() -> list[Case]:
    # 在读取本轮回测结果之前冻结：10预算对照、5均线邻参、20额外因子单独/组合。
    cases = [Case()]
    cases += [Case(v, w, a) for v in (0.07, 0.075, 0.08, 0.085, 0.09)
              for w, a in ((0, 0.0), (20, 0.5)) if (v, w) != (0.07, 0)]
    cases += [Case(0.08, w, a) for w, a in ((10, 0.25), (10, 0.5), (20, 0.25), (40, 0.25), (40, 0.5))]
    cases += [Case(0.08, w, a, name, strength) for name in EXTRAS
              for w, a in ((0, 0.0), (20, 0.5)) for strength in (0.25, 0.5)]
    return cases


def feature_bundle(prices: pd.DataFrame) -> tuple[dict, dict[str, pd.DataFrame], pd.DataFrame]:
    specs = {FactorSpec("momentum", n) for n in (20, 40, 60)}
    specs |= {FactorSpec("trend", 40), FactorSpec("volatility", 60)}
    specs |= {FactorSpec("ma_zscore", n) for n in (10, 20, 40)}
    specs |= {FactorSpec("trend_efficiency", 40), FactorSpec("downside_balance", 60),
              FactorSpec("relative_strength", 20), FactorSpec("breakout", 20), FactorSpec("volatility_ratio", 20)}
    frames = {s: compute_factor(prices, s) for s in specs}
    vol = frames[FactorSpec("volatility", 60)]
    score = 0.5 * (frames[FactorSpec("momentum", 20)] + frames[FactorSpec("momentum", 40)]) / vol.clip(lower=0.05)
    raw = {f"ma{n}": -frames[FactorSpec("ma_zscore", n)] / 3 for n in (10, 20, 40)}
    raw.update({"trend_efficiency": frames[FactorSpec("trend_efficiency", 40)],
                "downside_balance": frames[FactorSpec("downside_balance", 60)],
                "relative_strength": frames[FactorSpec("relative_strength", 20)] / 3,
                "breakout": frames[FactorSpec("breakout", 20)] / (vol * np.sqrt(20 / 252)).replace(0, np.nan) / 3,
                "volatility_ratio": -np.log(frames[FactorSpec("volatility_ratio", 20)].where(
                    frames[FactorSpec("volatility_ratio", 20)].gt(0))) / np.log(2)})
    raw = {name: x.clip(-1, 1).replace([np.inf, -np.inf], np.nan).fillna(0) for name, x in raw.items()}
    return frames, raw, score


def overlay(case: Case, raw: dict, score: pd.DataFrame) -> pd.DataFrame:
    out = score * 0.0
    out = out.fillna(0.0)
    if case.ma_window:
        out += case.ma_strength * raw[f"ma{case.ma_window}"]
    if case.extra:
        out += case.strength * raw[case.extra]
    spread = score.drop(columns=BOND).std(axis=1, ddof=1).fillna(0)
    out = out.mul(spread, axis=0)
    out[BOND] = 0.0
    return out


def targets(prices: pd.DataFrame, cases: list[Case], choices: pd.Series,
            frames: dict, raw: dict, score: pd.DataFrame, *, removed: tuple[str, ...] = ()) -> pd.DataFrame:
    """变预算路径共用同一评分历史；预算只影响权重，不影响排名缓冲状态。"""
    mapping = {c.key: c for c in cases}
    changes = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for name in choices.unique():
        dates = choices.index[choices.eq(name)]
        changes.loc[dates] = overlay(mapping[name], raw, score).loc[dates]
    altered = dict(frames)
    if removed:
        altered[FactorSpec("momentum", 40)] = frames[FactorSpec("momentum", 40)].copy()
        altered[FactorSpec("momentum", 40)].loc[:, list(removed)] = np.nan
    state = load_research_rotation_state(ROOT)
    result = None
    for budget in sorted({mapping[name].budget for name in choices.unique()}):
        signal = risk_controlled_signals(prices, RiskConfig.buffered_momentum(target_vol=budget),
                                         factor_frames=altered, score_overlay=changes, **state.signal_kwargs(prices))
        if result is None:
            result = signal * 0.0
        selected = choices.reindex(signal.index).map(lambda name: mapping[name].budget).eq(budget)
        result.loc[selected] = signal.loc[selected]
    return result


def evaluate(prices, target, end):
    return run_backtest(prices, target, START, end, fee_bps=3, slippage_bps=2)


def record(name, result, target, baseline):
    daily = result.equity.daily_return
    stress = (1 + result.equity.gross_return) * (1 - result.equity.turnover * 20 / 10000) - 1
    row = {"variant": name, **stats(daily), "realized_vol": float(daily.std(ddof=1) * np.sqrt(252)),
           "stress20_annual": stats(stress)["annual"], "stress20_drawdown": stats(stress)["drawdown"],
           "mean_risk_weight": float(result.weights.drop(columns=[BOND, "CASH"]).sum(axis=1).mean()),
           "mean_bond_weight": float(result.weights[BOND].mean()),
           "turnover_per_year": float(result.equity.turnover.sum() * 252 / len(daily))}
    for prefix, left, right in (("design", "2022", "2024-12-31"), ("later", "2025", str(daily.index[-1].date()))):
        row.update({f"{prefix}_{k}": v for k, v in stats(daily.loc[left:right]).items()})
    dd = result.equity.drawdown
    trough = dd.idxmin()
    row.update({"trough": str(trough.date()), "worst_day": float(daily.min()),
                "changed_target_weeks": int((baseline - target).abs().max(axis=1).loc[str(START):].gt(1e-10).sum())})
    return row


def diagnostics(prices, frames, raw, score):
    dates = weekly_signal_dates(prices.index)
    risky = prices.columns.drop(BOND)
    eligible = frames[FactorSpec("momentum", 40)].gt(0) & frames[FactorSpec("trend", 40)].gt(0)
    rows = []
    for name, value in {"original_score": score, "ma20": raw["ma20"], **{n: raw[n] for n in EXTRAS}}.items():
        groups, ic = [], {}
        for i, day in enumerate(dates[:-1]):
            enter, exit_ = prices.index.searchsorted(day, side="right"), prices.index.searchsorted(dates[i+1], side="right")
            if exit_ >= len(prices):
                continue
            label = prices.iloc[exit_][risky] / prices.iloc[enter][risky] - 1
            pair = pd.DataFrame({"x": value.loc[day, risky].where(eligible.loc[day, risky]), "y": label}).dropna()
            if len(pair) < 5 or pair.x.std() < 1e-12 or day < pd.Timestamp(START):
                continue
            ic[day] = float(pair.x.rank().corr(pair.y.rank()))
            groups.append((day, prices.index[exit_], pair.x.to_numpy(), pair.y.to_numpy()))
        sse = denominator = 0.0
        predictions = 0
        for day, _, x, y in groups:
            known = [g for g in groups if g[1] <= day and g[0] >= day - pd.DateOffset(years=2)]
            if len(known) < 52:
                continue
            tx, ty = np.concatenate([g[2] for g in known]), np.concatenate([g[3] for g in known])
            xm, ym = tx.mean(), ty.mean()
            beta = float(((tx-xm)*(ty-ym)).sum() / (((tx-xm)**2).sum()+1e-6))
            sse += float(((y-(ym+beta*(x-xm)))**2).sum())
            denominator += float(((y-ym)**2).sum())
            predictions += len(y)
        series = pd.Series(ic, dtype=float)
        rows.append({"factor": name, "ic": series.mean(), "weeks": len(series),
                     "design_ic": series.loc[:"2024"].mean(), "later_ic": series.loc["2025":].mean(),
                     "rolling_known_label_r2": 1-sse/denominator if denominator else None,
                     "predicted_observations": predictions})
    return pd.DataFrame(rows)


def protected_hashes():
    files = ("live_signal.py", "run_backtest.py", "etf_rotation/risk_strategy.py", "etf_rotation/config.py",
             "data/rotation_state.json", "output/live_signal.json")
    return {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files if (ROOT/f).exists()}


def checks(prices, cases, frames, raw, score, signals, daily, selection, end):
    primary, plain = Case(0.08, 20, 0.5).key, Case(0.08).key
    leaders = list(dict.fromkeys(["default", plain, primary, selection["full_sample_leader"], selection["design_selection"]]))
    audited = leaders + ["walk_forward"]
    paths = {n: pd.Series(n, index=prices.index) for n in leaders}
    paths["walk_forward"] = pd.Series("default", index=prices.index)
    for fold in selection["walk_forward_choices"]:
        paths["walk_forward"].loc[prices.index.year == fold["year"]] = fold["variant"]
    audits = []
    for cutoff in ("2023-06-30", "2025-06-30", "2026-09-30"):
        short = prices.loc[:cutoff]
        sf, sr, ss = feature_bundle(short)
        for name in audited:
            fresh = targets(short, cases, paths[name].loc[short.index], sf, sr, ss)
            dates = fresh.index[fresh.index < short.index[-1]]
            error = float((fresh.loc[dates] - signals[name].loc[dates]).abs().to_numpy().max())
            if error > 1e-12:
                raise AssertionError(f"prefix failed {name} {cutoff}")
            audits.append({"variant": name, "check": f"prefix_{cutoff}", "max_weight_difference": error})
    changed = prices.copy()
    cutoff = "2025-06-30"
    for i, symbol in enumerate(prices.columns):
        changed.loc[changed.index > cutoff, symbol] *= 1.5 + i * 0.17
    cf, cr, cs = feature_bundle(changed)
    for name in audited:
        # 滚动配置的历史选择只依赖已实现过去收益，未来配置本身不改动。
        fresh = targets(changed, cases, paths[name], cf, cr, cs)
        error = float((fresh.loc[:cutoff] - signals[name].loc[:cutoff]).abs().to_numpy().max())
        if error > 1e-12:
            raise AssertionError(f"future perturbation failed {name}")
        audits.append({"variant": name, "check": "all_future_prices_changed", "max_weight_difference": error})
    pd.DataFrame(audits).to_csv(OUT/"time_audit.csv", index=False, encoding="utf-8-sig")
    print(f"time audits complete: {len(audits)}", flush=True)
    ablation_names = list(dict.fromkeys(["default", plain, primary, selection["full_sample_leader"]]))
    ablations = []
    # 禁止交易同一ETF，其价格仍可作为共同历史信息。并增加黄金/纳指同时禁选。
    removals = [(symbol,) for symbol in prices.columns.drop(BOND)] + [("518880", "159941")]
    for removed in removals:
        for name in ablation_names:
            signal = targets(prices, cases, paths[name], frames, raw, score, removed=removed)
            result = evaluate(prices, signal, end)
            ablations.append({"variant": name, "removed": ",".join(removed), **stats(result.equity.daily_return)})
        print(f"ablation {','.join(removed)} complete", flush=True)
    # 同样风险策略下防守资金改持现金，识别国债收益依赖。
    for name in ablation_names:
        signal = signals[name].copy()
        signal[BOND] = 0.0
        result = evaluate(prices, signal, end)
        ablations.append({"variant": name, "removed": "BOND_TO_CASH", **stats(result.equity.daily_return)})
    pd.DataFrame(ablations).to_csv(OUT/"ablation.csv", index=False, encoding="utf-8-sig")
    bootstrap_daily = daily[[c.key for c in cases]]
    boot = {"versus_7pct": bootstrap(bootstrap_daily, leaders[1:])}
    same8 = [c.key for c in cases if c.budget == 0.08 and c.key != plain]
    boot["factor_increment_versus_8pct"] = bootstrap(daily[[plain]+same8].rename(columns={plain: "default"}),
                                                    list(dict.fromkeys([primary, selection["full_sample_leader"]
                                                                        if selection["full_sample_leader"] != plain else "default"])))
    (OUT/"bootstrap.json").write_text(json.dumps(boot, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(audits)


def selection_changes():
    saved=pd.read_csv(OUT/"target_weights.csv",index_col=[0,1],parse_dates=[1])
    baseline=saved.loc[Case(0.08).key].drop(columns=BOND).gt(1e-10)
    rows=[]
    for name in saved.index.get_level_values(0).unique():
        if name=="walk_forward":
            continue
        held=saved.loc[name].drop(columns=BOND).gt(1e-10)
        changed=held.ne(baseline).any(axis=1) & (held.index>=pd.Timestamp(START))
        for day in held.index[changed]:
            rows.append({"variant":name,"signal_date":str(day.date()),
                         "plain8_selection":",".join(held.columns[baseline.loc[day]]),
                         "candidate_selection":",".join(held.columns[held.loc[day]])})
    result=pd.DataFrame(rows,columns=["variant","signal_date","plain8_selection","candidate_selection"])
    result.to_csv(OUT/"selection_changes.csv",index=False,encoding="utf-8-sig")
    return result


def tail_analysis(prices, signals, daily):
    """重建主方案最差日的开盘前旧仓；当日收盘新目标不承受当日收益。"""
    primary=Case(0.08,20,0.5).key
    day=daily[primary].idxmin()
    prior=prices.index[prices.index.get_loc(day)-1]
    names=list(dict.fromkeys(["default",Case(0.08).key,primary,Case(0.08,extra="relative_strength",strength=0.25).key]))
    rows=[]
    for name in names:
        signal=signals[name]
        execution=pd.Series({d:prices.index[prices.index.searchsorted(d,side="right")]
                             for d in signal.index if prices.index.searchsorted(d,side="right")<len(prices)})
        old_signal=execution[execution.lt(day)].index[-1]
        entry=execution[old_signal]
        initial=signal.loc[old_signal]
        growth=prices.loc[prior]/prices.loc[entry]
        held=(initial*growth)/(1+(initial*(growth-1)).sum())
        asset_return=prices.loc[day]/prices.loc[prior]-1
        contribution=held*asset_return
        gross=float(contribution.sum())
        turnover=0.0
        if execution.eq(day).any():
            desired=signal.loc[execution[execution.eq(day)].index[-1]]
            after_return=held*(1+asset_return)/(1+gross)
            turnover=float((desired-after_return).abs().sum())
        predicted=(1+gross)*(1-turnover*5/10000)-1
        np.testing.assert_allclose(predicted,daily.loc[day,name],atol=1e-12)
        for symbol in held.index[held.gt(1e-10)]:
            rows.append({"variant":name,"date":str(day.date()),"prior_signal":str(old_signal.date()),"symbol":symbol,
                         "prior_close_weight":held[symbol],"asset_return":asset_return[symbol],
                         "gross_contribution":contribution[symbol],"verified_net_return":predicted})
    pd.DataFrame(rows).to_csv(OUT/"tail_event.csv",index=False,encoding="utf-8-sig")


def report(end):
    table = pd.read_csv(OUT/"comparison.csv").set_index("variant")
    yearly = pd.read_csv(OUT/"yearly.csv")
    selection = json.loads((OUT/"selection.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))
    boots = json.loads((OUT/"bootstrap.json").read_text(encoding="utf-8"))
    labels = {c.key:c.label for c in fixed_cases()} | {"walk_forward":"逐年仅按过去选参数"}
    plain, primary, ma7 = Case(0.08).key, Case(0.08,20,0.5).key, Case(0.07,20,0.5).key
    relative=Case(0.08,extra="relative_strength",strength=0.25).key
    changed=selection_changes()
    change_count=lambda key: int(changed.variant.eq(key).sum())
    names = list(dict.fromkeys(["default", ma7, plain, primary, selection["full_sample_leader"], selection["design_selection"], "walk_forward"]))
    pct = lambda x: f"{x:.2%}"
    lines = ["# 8%波动预算与因子叠加研究", "", f"研究日期2026-10-17；行情截止{end}。", "",
             "> 历史研究快照：本报告中的默认为当时7%版本；2026-10-18已选择8%＋20日均线反转0.5，现行参数见README。", "",
             "## 研究范围", "", "当前默认仍为7%预算＋前4名缓冲＋必选第1名。所有候选均为离线研究，未切换实盘。",
             "7%/8%表示风险篮子的年化波动预算，非最大波动或回撤上限，也非包含国债后的组合波动保证。"
             "固定风险ETF目标总仓位上限70%、单只30%、最多3只；弱市减半和国债60日动量规则不变。", "",
             "本轮在读取结果前固定35组参数：10组预算对照、5组均线邻参、20组其他因子单独/与均线组合。"
             "8%＋20日均线反转0.5为主检验；历史最优均线参数来自上一轮研究，不能当成本轮新发现。"
             "2022–2024为设计段，2025以后为时间分段复核。全部历史年份已参与既往研究，没有独立盲测样本。", "",
             "## 本轮判断", "",
             f"- 单独提高预算：年化由{pct(table.loc['default','annual'])}升至{pct(table.loc[plain,'annual'])}，"
             f"夏普由{table.loc['default','sharpe']:.3f}降至{table.loc[plain,'sharpe']:.3f}；风险增加有收益回报，但效率下降。",
             f"- 主方案8%＋均线反转：年化{pct(table.loc[primary,'annual'])}、夏普{table.loc[primary,'sharpe']:.3f}、"
             f"回撤{pct(table.loc[primary,'drawdown'])}；相对8%原评分改善，但7%均线候选的全期夏普仍较高。",
             f"- 8%中最高历史夏普为{labels[selection['full_sample_leader']]}。相对强弱0.25仅改变{change_count(relative)}次周选仓，"
             f"均线主方案改变{change_count(primary)}次；少量事件、邻参不稳定及多轮搜索削弱了证据强度。",
             "- 保留8%均线反转和8%相对强弱为前瞻跟踪候选；不继续增加组合复杂度，也不因8.5%/9%更高年化而升级预算。"
             "当前7%实盘保持既有规则，不能承诺历史4%～5%回撤能够延续。", "",
             "## 数据、成本和时序", "", f"16只ETF，收益区间2022-01-04至{end}，约400个自然日预热；"
             "沿用本地Tushare后复权收盘价与真实缺价，不下载或覆盖行情。每周最后一个开市日盘后信号于下一交易日收盘执行，"
             "执行前仍计旧仓收益，按实际漂移仓位计买卖双向换手。单边手续费3bps＋滑点2bps，并检验20bps总成本。"
             "复权连续价格不等于实际成交价；不模拟最低佣金、整手、跨境溢价和盘中回撤。"
             "周目标沿用同一已保存切换锚点，历史假设持仓不代表真实账户交易。", "",
             "## 因子构造", "", "原评分为20/40日动量各半，除60日日收益年化波动（下限5%）。"
             "所有新增因子先限制于[-1,1]，再乘当日15只风险ETF原评分截面标准差D和强度a，叠加至原评分。"
             "截面只使用当日已知数据；仍需40日动量与40日均线资格。国债不参与评分。", "",
             "| 因子 | 本地构造与假说 |", "| --- | --- |",
             "| 均线反转 | −(价格−N日均价)/价格标准差/3；N=10/20/40。偏离过大扣分、趋势内回调加分。 |",
             "| 趋势效率 | 40日价格净变化/40日绝对日变动之和；奖励更连续的上行。 |",
             "| 涨跌波动平衡 | 1−2×近60日负收益平方和/全部收益平方和；奖励上涨贡献更大的路径，不是基本面质量。 |",
             "| 相对强弱 | 自身20日对数收益减沪深300同期对数收益，除自身60日对数波动×√20，再除3。 |",
             "| 突破 | 价格/此前20日最高价−1，除60日年化波动×√(20/252)，再除3；奖励已确认突破。 |",
             "| 波动稳定性 | −log(20日波动/60日波动)/log(2)；惩罚短期波动抬升。 |", "",
             "额外因子强度固定0.25/0.5，分别单独加入或叠加于20日均线反转0.5。"
             "合格标的不超过3只时评分通常不改变入选集合；组合权重仍由逆波动、协方差与仓位约束决定。", "",
             "## 主要结果", "", "夏普按零无风险利率；回撤按日收盘净值。全样本最优仅为事后描述。", "",
             "| 方案 | 年化 | 夏普 | 回撤 | 实现波动 | 平均风险仓 | 20bps年化 | 20bps回撤 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n in names:
        r=table.loc[n]
        lines.append(f"| {labels[n]} | {pct(r.annual)} | {r.sharpe:.3f} | {pct(r.drawdown)} | {pct(r.realized_vol)} | {pct(r.mean_risk_weight)} | {pct(r.stress20_annual)} | {pct(r.stress20_drawdown)} |")
    lines += ["", "![收益与回撤](assets/vol8_20261017.png)", "", "## 尾部风险与熊市日", "",
              "最大回撤较低也可能伴随更大的单日损失。以下弱市按前一交易日沪深300低于40日均线界定，"
              "不使用当日收盘反向划分交易时已知的市场状态。", "",
              "| 方案 | 最差单日 | 弱市日收益5%分位 | 弱市平均风险仓 |", "| --- | ---: | ---: | ---: |"]
    regimes=pd.read_csv(OUT/"regimes.csv").query("regime=='weak'").set_index("variant")
    for n in ("default",plain,primary,relative):
        lines.append(f"| {labels[n]} | {pct(table.loc[n,'worst_day'])} | {pct(regimes.loc[n,'daily_q05'])} | {pct(regimes.loc[n,'mean_risk_weight'])} |")
    tail_path=OUT/"tail_event.csv"
    if tail_path.exists():
        tail=pd.read_csv(tail_path,dtype={"symbol":str})
        main=tail.loc[tail.variant.eq(primary)].set_index("symbol")
        worst=main.gross_contribution.nsmallest(2)
        attribution="、".join(f"{ETF_NAMES[symbol]}贡献{pct(value)}" for symbol,value in worst.items())
        lines += ["", f"主方案最差日为{tail.date.iloc[0]}，前一期已持有的{attribution}。"
                  "归因重建执行前旧仓并计入当日收盘换手费用，与正式净收益逐项核对通过；详见tail_event.csv。"]
    lines += ["", "## 时间分段与逐年", "",
              "设计段仅允许8%候选或7%默认回退：回撤≤10%、年化不比默认低0.5个百分点以上、至少2/3训练年份收益不低于默认，再选夏普最高。"
              "滚动路径2022–2023用默认，2024起每年仅依据此前已实现收益重选，训练末端空出5个交易日。"
              "各年在同一评分历史及缓冲状态中重放并支付切换成本，不拼接不同旧仓的收益。", "",
              "| 方案 | 设计段年化 | 设计段夏普 | 后段年化 | 后段夏普 | 后段回撤 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for n in names:
        r=table.loc[n]
        lines.append(f"| {labels[n]} | {pct(r.design_annual)} | {r.design_sharpe:.3f} | {pct(r.later_annual)} | {r.later_sharpe:.3f} | {pct(r.later_drawdown)} |")
    lines += ["", "| 方案 | 2022 | 2023 | 2024 | 2025 | 2026至今 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for n in names:
        r=yearly.loc[yearly.variant.eq(n)].set_index("year")
        lines.append("| "+labels[n]+" | "+" | ".join(pct(r.loc[y,"total"]) for y in range(2022,2027))+" |")
    lines += ["", "滚动选择记录：", ""]
    lines += [f"- {f['year']}：{labels[f['variant']]}；训练收益截至{f['training_end']}。" for f in selection["walk_forward_choices"]]
    lines += ["", "## 预算及均线参数稳定性", "", "| 方案 | 年化 | 夏普 | 回撤 | 后段夏普 |", "| --- | ---: | ---: | ---: | ---: |"]
    for c in fixed_cases():
        if c.extra:
            continue
        r=table.loc[c.key]
        lines.append(f"| {c.label} | {pct(r.annual)} | {r.sharpe:.3f} | {pct(r.drawdown)} | {r.later_sharpe:.3f} |")
    lines += ["", "## 新因子增量", "", "每家族的历史最高夏普项，含未改善项；不可把事后赢家解释为已知有效。", "",
              "| 家族 | 全样本最佳配置 | 年化 | 夏普 | 回撤 | 后段夏普 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for family in EXTRAS:
        r=table.loc[table.extra.eq(family)].sort_values("sharpe",ascending=False).iloc[0]
        lines.append(f"| {EXTRAS[family]} | {labels[r.name]} | {pct(r.annual)} | {r.sharpe:.3f} | {pct(r.drawdown)} | {r.later_sharpe:.3f} |")
    diag=pd.read_csv(OUT/"factor_diagnostics.csv")
    lines += ["", "单因子诊断：合格标的≥5只的周截面，预测下一执行收盘至再下一执行收盘收益。"
              "滚动R²仅用过去2年且标签已结束的≥52周拟合，基准为训练标签均值。", "",
              "| 因子 | 周数 | 平均秩IC | 后段IC | 滚动R² |", "| --- | ---: | ---: | ---: | ---: |"]
    for _,r in diag.iterrows():
        lines.append(f"| {r.factor} | {int(r.weeks)} | {r.ic:.3f} | {r.later_ic:.3f} | {r.rolling_known_label_r2:.4f} |")
    abl=pd.read_csv(OUT/"ablation.csv")
    lines += ["", "## 资产依赖与统计不确定性", "", "逐一禁止交易同一只风险ETF，所有方案保持共同信息集；比较15次，不重新选择参数。", "",
              "| 比较 | 年化更高次数 | 夏普更高次数 |", "| --- | ---: | ---: |"]
    for left,right in ((plain,"default"),(primary,plain),(relative,plain),(selection["full_sample_leader"],primary)):
        if left==right:
            continue
        l=abl.loc[abl.variant.eq(left)].set_index("removed").reindex(prices_symbols())
        r=abl.loc[abl.variant.eq(right)].set_index("removed").reindex(prices_symbols())
        lines.append(f"| {labels[left]} 对 {labels[right]} | {int(l.annual.gt(r.annual).sum())}/15 | {int(l.sharpe.gt(r.sharpe).sum())}/15 |")
    lines += ["", "联合剔除与防守依赖：", "", "| 方案 | 同时禁选黄金/纳指年化 | 同时禁选黄金/纳指夏普 | 国债改现金年化 | 国债改现金夏普 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for n in ("default",plain,primary,relative):
        a=abl.loc[abl.variant.eq(n)].set_index("removed")
        lines.append(f"| {labels[n]} | {pct(a.loc['518880,159941','annual'])} | {a.loc['518880,159941','sharpe']:.3f} | {pct(a.loc['BOND_TO_CASH','annual'])} | {a.loc['BOND_TO_CASH','sharpe']:.3f} |")
    lines += ["", "相对强弱逐只剔除的优势在联合剔除黄金与纳指时未保留；这说明15次单标的通过仍不足以证明资产依赖已消除。"
              "国债改现金会改变防守经济含义，比较目的为识别依赖，不能要求收益完全不受影响。", ""]
    for title,key in (("相对7%默认的本轮搜索","versus_7pct"),("同为8%时因子增量","factor_increment_versus_8pct")):
        b=boots[key]
        lines += [f"- {title}：20日循环块、{b['replicates']}次成对抽样；校正{b['corrected_grid_trials']}项的最大均值收益差探索性p={b['family_max_mean_return_p']:.3f}。"]
    ci=boots["factor_increment_versus_8pct"]["paired_unadjusted_95pct"][primary]
    lines += [f"- 8%均线反转相对8%原评分：年化算术均值收益差的未校正95%区间为[{pct(ci['annual_arithmetic_difference'][0])}, {pct(ci['annual_arithmetic_difference'][1])}]，"
              f"夏普差区间[{ci['sharpe_difference'][0]:.3f}, {ci['sharpe_difference'][1]:.3f}]。",
              "- 上述检验不是夏普选优校正，也不覆盖之前多轮研究与非平稳市场。提高预算带来的收益不能直接解释为新增alpha。", "",
              "## 当前目标及后续使用", "", "`latest_targets.csv`列出同一最新信号日的各方案目标，作为研究对照。"
              "实际实盘继续调用默认7%方案；本研究没有修改调仓建议文件、账户数据库或标的池。", "",
              "## 检查、完整记录及复现", "", f"通过{manifest.get('time_audits',0)}项历史截断/未来行情扰动检查，默认实盘文件指纹保持一致。"
              "因子资格、排名状态与权重只读信号日及以前的数据。因子通过检查仍不等于消除了历史调参偏差。", "",
              "保留一个统一研究入口、因子库新增`downside_balance`以及完整汇总；没有逐候选脚本或逐候选目录。"
              "输出包含comparison/yearly/daily_returns/target_weights/holdings/regimes/ablation/factor_diagnostics/time_audit/selection_changes/tail_event等汇总文件，"
              "以及固定参数、代码/数据指纹和选择记录；保留失败项统计供审计。", "", "```powershell",
              f".\\.venv\\Scripts\\python.exe research_vol8.py --end {end}",
              ".\\.venv\\Scripts\\python.exe research_vol8.py --report-only", "```", "", "## 资料依据", "",
              "波动管理、跨资产趋势及连续信息动量研究支持检验风险缩放与趋势路径；本文额外因子及窗口是本地假说，"
              "不是论文策略复刻，论文也不能证明本ETF池采用8%预算一定优于7%。", ""]
    lines += [f"- [{title}]({url})" for title,url in SOURCES]
    lines += ["", f"数据指纹：`{manifest['price_sha256']}`。", ""]
    path=ROOT/"docs/vol8_research_20261017.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines),encoding="utf-8")
    # 只展示主要对照，所有其他候选保留数表。
    plotted=list(dict.fromkeys(["default",ma7,plain,primary,selection["full_sample_leader"]]))
    daily=pd.read_csv(OUT/"daily_returns.csv",index_col=0,parse_dates=True)
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"]=["Microsoft YaHei","DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"]=False
    fig,axes=plt.subplots(2,1,figsize=(13,9),sharex=True,gridspec_kw={"height_ratios":[2.3,1]})
    for name in plotted:
        nav=(1+daily[name]).cumprod()
        label=labels[name]
        axes[0].plot(nav.index,(nav-1)*100,label=label,linewidth=2 if name in ("default",primary) else 1.3)
        axes[1].plot(nav.index,(nav/nav.cummax().clip(lower=1)-1)*100,linewidth=1.5)
    axes[0].set_title(f"8%波动预算与因子研究｜2022-01-04至{end}｜单边5bps\n全样本最优为事后描述；尚未切换默认实盘")
    axes[0].set_ylabel("累计收益（%）")
    axes[0].legend(fontsize=8,loc="upper left")
    axes[1].set_ylabel("收盘回撤（%）")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.axvline(pd.Timestamp("2025-01-01"),color="grey",linestyle="--",alpha=0.5)
    fig.tight_layout()
    assets=ROOT/"docs/assets"
    assets.mkdir(parents=True,exist_ok=True)
    fig.savefig(assets/"vol8_20261017.png",dpi=150)
    plt.close(fig)
    return path


def prices_symbols():
    return [s for s in ETF_NAMES if s!=BOND]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end",type=date.fromisoformat,default=date(2026,10,16))
    parser.add_argument("--report-only",action="store_true")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.report_only:
        manifest=json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))
        manifest["report_generation_code_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
        print(report(date.fromisoformat(manifest["end"])))
        return
    before=protected_hashes()
    cases=fixed_cases()
    prices=load_close_prices(START,args.end,ROOT/"data/cache",offline=True)
    manifest={"start":str(START),"end":str(args.end),"strategy":STRATEGY_NAME,"case_count":len(cases),
              "cases":[asdict(c)|{"key":c.key} for c in cases],"primary":Case(0.08,20,0.5).key,
              "price_sha256":hashlib.sha256(pd.util.hash_pandas_object(prices,index=True).values.tobytes()).hexdigest(),
              "past_data_already_used_in_research":True,"software":{"python":platform.python_version(),"pandas":pd.__version__,"numpy":np.__version__},
              "selection_rule":"8% candidates or default; DD<=10%, annual>=default-0.5pp, >=2/3 training years nonnegative return difference, highest Sharpe",
              "protected_before":before}
    (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    frames,raw,score=feature_bundle(prices)
    rows,yearly,holding_rows,regimes=[],[],[],[]
    results,signals={},{}
    for c in cases:
        signal=targets(prices,cases,pd.Series(c.key,index=prices.index),frames,raw,score)
        result=evaluate(prices,signal,args.end)
        if c.key=="default":
            base_signal=signal
            original=risk_controlled_signals(prices,RiskConfig.buffered_momentum(),
                                             **load_research_rotation_state(ROOT).signal_kwargs(prices))
            pd.testing.assert_frame_equal(signal,original)
            prior_path=ROOT/"output/reversal_20261017/comparison.csv"
            prior_manifest=ROOT/"output/reversal_20261017/manifest.json"
            if prior_path.exists() and prior_manifest.exists() and json.loads(prior_manifest.read_text(encoding="utf-8")).get("end")==str(args.end):
                prior=pd.read_csv(prior_path).set_index("variant").loc["default"]
                np.testing.assert_allclose(stats(result.equity.daily_return)["annual"],prior.annual,atol=1e-12)
        rows.append(record(c.key,result,signal,base_signal)|asdict(c))
        results[c.key],signals[c.key]=result,signal
        print(f"evaluated {c.key}: annual={rows[-1]['annual']:.3%} SR={rows[-1]['sharpe']:.3f} DD={rows[-1]['drawdown']:.3%}",flush=True)
    daily=pd.concat({n:r.equity.daily_return for n,r in results.items()},axis=1)
    table=pd.DataFrame(rows).set_index("variant")
    candidates=["default"]+[c.key for c in cases if c.budget==0.08]
    design=choose_case(daily[candidates],pd.Timestamp("2024-12-31"),[2022,2023,2024])
    winner=table.loc[table.budget.eq(0.08)&table.drawdown.ge(-0.10)].sort_values("sharpe",ascending=False,kind="stable").index[0]
    choices=pd.Series("default",index=prices.index)
    folds=[]
    for year in (2024,2025,2026):
        cutoff=prices.index[prices.index<pd.Timestamp(f"{year}-01-01")][-6]
        selected=choose_case(daily[candidates],cutoff,list(range(2022,year)))
        choices.loc[choices.index.year==year]=selected
        folds.append({"year":year,"training_end":str(cutoff.date()),"variant":selected})
    selection={"full_sample_leader":winner,"design_selection":design,"walk_forward_choices":folds}
    (OUT/"selection.json").write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding="utf-8")
    wf_signal=targets(prices,cases,choices,frames,raw,score)
    wf=evaluate(prices,wf_signal,args.end)
    rows.append(record("walk_forward",wf,wf_signal,base_signal))
    results["walk_forward"],signals["walk_forward"]=wf,wf_signal
    daily["walk_forward"]=wf.equity.daily_return
    pd.DataFrame(rows).to_csv(OUT/"comparison.csv",index=False,encoding="utf-8-sig")
    daily.to_csv(OUT/"daily_returns.csv",encoding="utf-8-sig")
    pd.concat(signals,names=["variant","signal_date"]).to_csv(OUT/"target_weights.csv",encoding="utf-8-sig")
    pd.DataFrame({n:s.iloc[-1] for n,s in signals.items()}).T.to_csv(OUT/"latest_targets.csv",encoding="utf-8-sig")
    weak=prices["510300"].shift(1).le(prices["510300"].rolling(40,min_periods=40).mean().shift(1))
    for name,r in results.items():
        for year,segment in r.equity.daily_return.groupby(r.equity.index.year):
            yearly.append({"variant":name,"year":int(year),**stats(segment)})
        for symbol in r.weights:
            w=r.weights[symbol]
            holding_rows.append({"variant":name,"symbol":symbol,"holding_days":int(w.gt(1e-10).sum()),"mean_weight":float(w.mean()),"max_weight":float(w.max())})
        for label,mask in (("weak",weak),("strong",~weak)):
            segment=r.equity.daily_return.loc[mask.reindex(r.equity.index)]
            regimes.append({"variant":name,"regime":label,"days":len(segment),"mean_daily":float(segment.mean()),
                            "daily_q05":float(segment.quantile(0.05)),"worst_day":float(segment.min()),
                            "mean_risk_weight":float(r.weights.drop(columns=[BOND,"CASH"]).sum(axis=1).loc[segment.index].mean())})
    pd.DataFrame(yearly).to_csv(OUT/"yearly.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(holding_rows).to_csv(OUT/"holdings.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(regimes).to_csv(OUT/"regimes.csv",index=False,encoding="utf-8-sig")
    diagnostics(prices,frames,raw,score).to_csv(OUT/"factor_diagnostics.csv",index=False,encoding="utf-8-sig")
    manifest["time_audits"]=checks(prices,cases,frames,raw,score,signals,daily,selection,args.end)
    tail_analysis(prices,signals,daily)
    if protected_hashes()!=before:
        raise AssertionError("Production file changed")
    manifest["protected_files_unchanged"]=True
    manifest["research_code_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["factor_library_sha256"]=hashlib.sha256((ROOT/"etf_rotation/factors.py").read_bytes()).hexdigest()
    (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"COMPLETE {report(args.end)}",flush=True)
    print(json.dumps(selection,ensure_ascii=True),flush=True)


if __name__=="__main__":
    main()
