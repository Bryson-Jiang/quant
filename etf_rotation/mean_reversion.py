"""研究用途：长期均线偏离状态机，不接入默认实盘。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .factors import FactorSpec, compute_factor


@dataclass(frozen=True)
class ReversionRule:
    window: int = 200
    entry: float = -0.03
    exit: float = 0.03
    max_days: int | None = 120
    confirmation: str = "none"

    def __post_init__(self):
        if self.window < 2 or not -0.5 < self.entry <= 0 or not 0 < self.exit < 0.5:
            raise ValueError("长期均线窗口或入出场偏离无效")
        if self.max_days is not None and self.max_days < 1:
            raise ValueError("最长持有交易日必须为正")
        if self.confirmation not in {"none", "bounce3", "slope20"}:
            raise ValueError("未知确认规则")

    @property
    def key(self):
        return f"ma{self.window}_in{abs(self.entry):.0%}_out{self.exit:.0%}_max{self.max_days or 0}_{self.confirmation}".replace("%", "")

    @property
    def label(self):
        confirm = {"none": "", "bounce3": "／3日反弹确认", "slope20": "／长均线不下降"}[self.confirmation]
        return f"MA{self.window}／入{self.entry:.0%}／出+{self.exit:.0%}／{self.max_days or '不限'}日{confirm}"


def reversion_states(prices: pd.DataFrame, symbol: str, rule: ReversionRule,
                     decision_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """信号日收盘决策，下一交易日收盘执行；强制退出后等重新越过入场线才再武装。

    max_days为信号袖套的年龄，自预定下一交易日开始计数。使用滚动均线卖出，
    并不保证退出价高于买入价；实际交易损益由统一回测引擎另算。
    """
    # 例如拆分暂停交易日：用过去N个真实报价计算窗口，不填造缺价，也不让一个
    # 暂停日使后续200个交易日的均线全部失效。缺价当日仍不产生新交易信号。
    quoted = prices[[symbol]].dropna()
    observed = compute_factor(quoted, FactorSpec("trend", rule.window))[symbol]
    p = quoted[symbol]
    average = p / (1 + observed)
    deviation = observed.reindex(prices.index)
    bounce = (p / p.shift(3) - 1).reindex(prices.index)
    slope = (average / average.shift(20) - 1).reindex(prices.index)
    decision = set(decision_dates)
    active, armed, entry_execution_pos = False, True, None
    rows = []
    for i, day in enumerate(prices.index):
        reason = "hold" if active else "flat"
        if day in decision and pd.notna(deviation.loc[day]):
            value = float(deviation.loc[day])
            if active:
                age = i - entry_execution_pos
                if value >= rule.exit:
                    active, armed, reason = False, False, "ma_exit"
                elif rule.max_days is not None and age >= rule.max_days:
                    active, armed, reason = False, False, "time_exit"
            else:
                if value > rule.entry:
                    armed = True
                confirmed = (rule.confirmation == "none"
                             or rule.confirmation == "bounce3" and bounce.loc[day] > 0
                             or rule.confirmation == "slope20" and slope.loc[day] >= 0)
                if armed and value <= rule.entry and confirmed:
                    active, reason, entry_execution_pos = True, "entry", i + 1
        rows.append({"date": day, "active": active, "reason": reason,
                     "deviation": deviation.loc[day], "decision": day in decision})
    return pd.DataFrame(rows).set_index("date")


def capped_risk_weights(desired: pd.Series, returns: pd.DataFrame, day: pd.Timestamp,
                        *, target_vol: float, bear: bool, risk_ceiling: float | None = None,
                        max_weights: pd.Series | None = None) -> pd.Series:
    """与默认相同60日风险估计、70%总上限、弱市减半、单只30%约束。"""
    desired = desired.loc[desired.gt(1e-12)]
    if desired.empty:
        return desired
    within = desired / desired.sum()
    history = returns.loc[:day, desired.index].tail(60).dropna()
    individual = returns.loc[:day, desired.index].tail(60).std(ddof=1) * np.sqrt(252)
    if len(history) >= 40:
        covariance = history.cov().to_numpy() * 252
        sigma = np.sqrt(max(float(within.to_numpy() @ covariance @ within.to_numpy()), 0))
    else:
        sigma = float(np.sqrt(((within * individual) ** 2).sum()))
        covariance = np.diag(individual.to_numpy() ** 2)
    gross = min(0.70, target_vol / max(float(sigma), 0.02)) * (0.5 if bear else 1)
    if risk_ceiling is not None:
        gross = min(gross, risk_ceiling)
    out = within * gross
    cap = pd.Series(0.30, index=out.index) if max_weights is None else max_weights.reindex(out.index)
    protected = set(out.index[out.gt(cap)])
    excess = float((out - cap).clip(lower=0).sum())
    out = out.clip(upper=cap)
    while excess > 1e-12:
        available = out.index[out.lt(cap - 1e-12) & ~out.index.isin(protected)]
        if len(available) == 0:
            break
        allocation = excess * out.loc[available] / out.loc[available].sum()
        allocation = allocation.clip(upper=cap.loc[available] - out.loc[available])
        out.loc[available] += allocation
        excess -= float(allocation.sum())
        protected.update(out.index[out.ge(cap - 1e-12)])
    # 单只限仓会改变组合结构，最终再检验风险预算，防止移除负相关资产后风险上升。
    projected = float(np.sqrt(max(out.to_numpy() @ covariance @ out.to_numpy(), 0)))
    limit = target_vol * (0.5 if bear else 1)
    if projected > limit:
        out *= limit / projected
    return out


def defensive_funding_targets(prices: pd.DataFrame, base: pd.DataFrame,
                              state: pd.DataFrame, symbol: str, cap: float,
                              *, bond: str = "511260", bear: pd.Series | None = None,
                              guard: bool = False) -> pd.DataFrame:
    """保持核心风险目标，仅从现金再从国债拨给反转卫星；研究用途。

    guard=False时不再对整个账户强加原8%/70%额度；guard=True只给卫星使用
    剩余风险额度，不缩减核心。核心本身若已超预测预算，无法用此函数修复。
    可能有4只风险ETF及1只国债；目标上限允许在非调仓日自然漂移。
    """
    if not 0 < cap <= 1 or symbol == bond:
        raise ValueError("卫星上限或标的无效")
    if guard and bear is None:
        raise ValueError("风险额度对照必须提供已知市场状态")
    changes=state.index[state.active.ne(state.active.shift(1,fill_value=False))]
    decisions=base.index.union(changes).sort_values()
    decisions=decisions[decisions>=base.index[0]]
    original=base.reindex(base.index.union(decisions)).sort_index().ffill().reindex(decisions)
    original=original.reindex(columns=prices.columns,fill_value=0.)
    returns=prices.pct_change(fill_method=None)
    rows=[]
    for day in decisions:
        row=original.loc[day].copy()
        if state.at[day,"active"]:
            cash=max(0.,1-float(row.sum()))
            amount=min(cap,cash+float(row[bond]))
            if guard and amount>0:
                core=row.drop([bond,symbol]);core=core[core.gt(1e-12)]
                weak=bool(bear.loc[day]);limit=.04 if weak else .08
                amount=min(amount,max(0.,(.35 if weak else .70)-float(core.sum())))
                names=list(core.index)+[symbol]
                history=returns.loc[:day,names].tail(60).dropna()
                if len(history)<40:
                    amount=0.
                elif amount>0:
                    cov=history.cov().to_numpy()*252
                    weights=core.to_numpy()
                    a=float(cov[-1,-1])
                    b=float(weights @ cov[:-1,-1])
                    c=float(weights @ cov[:-1,:-1] @ weights)-limit**2
                    # var(core + x*satellite) = a*x^2 + 2*b*x + c + limit^2.
                    if a>1e-16:
                        discriminant=b*b-a*c
                        if discriminant<0:
                            amount=0.
                        else:
                            root=np.sqrt(max(discriminant,0))
                            lower=max(0.,(-b-root)/a);upper=(-b+root)/a
                            amount=min(amount,max(0.,upper))
                            if amount<lower-1e-12:amount=0.
                    elif c>1e-12:
                        amount=0.
            row[symbol]=amount
            row[bond]=max(0.,float(row[bond])-max(0.,amount-cash))
        rows.append(row.rename(day))
    result=pd.DataFrame(rows);result.index.name="signal_date"
    return result
