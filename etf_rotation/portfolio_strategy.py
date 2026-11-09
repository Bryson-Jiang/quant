"""默认组合：17只核心轮动＋独立红利反转卫星，共用实盘和回测路径。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from .config import BENCHMARK, ETF_NAMES, PORTFOLIO_ETF_NAMES
from .data import load_close_prices, load_symbol
from .factor_store import FactorStore
from .ivol_filter import IVOLFilterResult, market_ivol_filter
from .mean_reversion import ReversionRule, defensive_funding_targets, reversion_states
from .risk_strategy import RiskConfig, risk_controlled_signals, weekly_signal_dates
from .strategy_state import RotationState


@dataclass(frozen=True)
class PortfolioSignals:
    prices: pd.DataFrame
    core_signals: pd.DataFrame
    signals: pd.DataFrame
    satellite_states: pd.DataFrame
    hypothetical_satellite_history: bool
    ivol_filter: IVOLFilterResult | None = None
    hypothetical_ivol_history: bool = False


def load_portfolio_data(start: date, end: date, cache_dir: Path, *,
                        offline: bool = False, refresh: bool = False) -> tuple[pd.DataFrame,pd.DataFrame]:
    """完整红利历史只用于长均线状态，不扩入核心排名；原16只加载口径不变。"""
    core=load_close_prices(min(start,date(2022,1,1)),end,cache_dir,offline=offline,refresh=refresh)
    market_start=min(date(2019,1,1),core.index[0].date())
    market=load_symbol(BENCHMARK,market_start,end,cache_dir,offline=offline,refresh=refresh)
    satellite=(load_symbol("512890",date(2019,1,1),end,cache_dir,offline=offline,refresh=refresh)
               if end>=date(2019,1,1) else pd.Series(dtype=float))
    history=satellite.reindex(market.index).to_frame("512890")
    return core,history


def portfolio_signals(core_prices: pd.DataFrame, satellite_history: pd.DataFrame,
                      state: RotationState, config: RiskConfig | None = None,
                      store: FactorStore | None = None, *,
                      hypothetical_satellite_history: bool = False,
                      hypothetical_universe_history: bool = False,
                      hypothetical_ivol_history: bool = False,
                      as_of: date | None = None) -> PortfolioSignals:
    """默认按启用日期从未持有状态重放；全历史卫星模拟须显式请求。"""
    policy=state.dividend_policy
    if policy is None:
        raise ValueError("默认组合缺少红利卫星参数与启用日期")
    symbols=list(state.core_symbols or ETF_NAMES)
    if not set(symbols).issubset(core_prices.columns):
        raise ValueError(f"核心行情缺少策略所需{len(symbols)}只ETF")
    config=config or RiskConfig.current_default()
    core_prices=core_prices.loc[:,symbols]
    activation=state.universe_activation_dates(hypothetical=hypothetical_universe_history)
    ivol=(market_ivol_filter(core_prices,state.ivol_policy,store,
                            hypothetical_history=hypothetical_ivol_history,
                            as_of=as_of,
                            universe_activation_dates=activation)
          if state.ivol_policy is not None else None)
    core=risk_controlled_signals(core_prices,config,store,**state.signal_kwargs(core_prices),
                                universe_activation_dates=activation,
                                eligibility_mask=ivol.eligibility_mask if ivol is not None else None)
    # 不读取查询日以后的行情，连卫星状态的最后一个日期也与核心截止日一致。
    history=satellite_history.loc[:core_prices.index[-1],[policy.symbol]]
    # 上市前没有报价也必须有明确的False状态；NaN的布尔值会误激活卫星。
    history=history.reindex(history.index.union(core_prices.index)).sort_index()
    decisions=weekly_signal_dates(history.index)
    if not hypothetical_satellite_history:
        decisions=decisions[decisions>=pd.Timestamp(policy.effective_signal_date)]
    rule=ReversionRule(policy.ma_days,policy.entry_deviation,policy.exit_deviation,policy.max_holding_days)
    if history[policy.symbol].notna().any():
        satellite_state=reversion_states(history,policy.symbol,rule,decisions)
    else:
        satellite_state=pd.DataFrame({"active":False,"reason":"flat","deviation":float("nan"),
                                      "decision":history.index.isin(decisions)},index=history.index)
        satellite_state.index.name="date"
    prices=core_prices.copy()
    prices[policy.symbol]=history[policy.symbol].reindex(prices.index)
    signals=defensive_funding_targets(prices,core,satellite_state.reindex(prices.index),
                                     policy.symbol,policy.max_weight)
    return PortfolioSignals(prices,core,signals,satellite_state,hypothetical_satellite_history,ivol,hypothetical_ivol_history)


def ivol_details(bundle: PortfolioSignals, state: RotationState, signal_day: pd.Timestamp) -> dict:
    policy=state.ivol_policy;result=bundle.ivol_filter
    if policy is None or result is None:return {"enabled":False,"effective_for_signal":False}
    u=result.utility.loc[signal_day]
    active=bool(result.active.loc[signal_day])
    return {**policy.metadata(),"enabled":True,"effective_for_signal":active,
            "simulation_mode":"hypothetical_full_history" if bundle.hypothetical_ivol_history else "from_actual_adoption_date",
            "model":"market_ols_with_intercept","annualization":252,"residual_std_ddof":1,
            "valid_comparison_assets":int(u.notna().sum()),
            "excluded_symbols":list(u.index[(u.notna() & ~result.eligibility_mask.loc[signal_day])]) if active else [],
            "missing_factor_behavior":"neutral; original trend and momentum requirements still apply",
            "adoption_note":"2026-11-09收盘执行前按用户指令切换；首次使用11月6日收盘数据，不表示11月6日已采用新规则"}


def satellite_details(bundle: PortfolioSignals, state: RotationState,
                      signal_day: pd.Timestamp) -> dict:
    policy=state.dividend_policy
    if policy is None:
        raise ValueError("缺少红利卫星启用记录")
    row=bundle.satellite_states.loc[signal_day]
    deviation=float(row.deviation) if pd.notna(row.deviation) else None
    close=bundle.prices.at[signal_day,policy.symbol]
    eligible_date=signal_day>=pd.Timestamp(policy.effective_signal_date)
    return {**policy.metadata(),"name":PORTFOLIO_ETF_NAMES[policy.symbol],
            "simulation_mode":"hypothetical_full_history" if bundle.hypothetical_satellite_history else "from_actual_adoption_date",
            "effective_for_signal":bool(eligible_date),"active":bool(row.active),
            "state_reason":row.reason,"deviation_from_ma":deviation,
            "adjusted_signal_close":float(close) if pd.notna(close) else None,
            "adjusted_ma":float(close/(1+deviation)) if deviation is not None and pd.notna(close) else None,
            "target_weight":float(bundle.signals.at[signal_day,policy.symbol]),
            "core_targets_preserved":True,"global_eight_percent_budget_applied":False,
            "funding_description":"先使用现金，不足部分减少国债；保留所有核心风险ETF目标",
            "state_basis":"从启用日空仓开始按周信号重放，不代表券商已成交"}
