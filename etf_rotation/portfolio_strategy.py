"""默认组合：原16只核心轮动＋独立红利反转卫星，共用实盘和回测路径。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from .config import BENCHMARK, ETF_NAMES, PORTFOLIO_ETF_NAMES
from .data import load_close_prices, load_symbol
from .factor_store import FactorStore
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
                      hypothetical_satellite_history: bool = False) -> PortfolioSignals:
    """默认按启用日期从未持有状态重放；全历史卫星模拟须显式请求。"""
    policy=state.dividend_policy
    if policy is None:
        raise ValueError("默认组合缺少红利卫星参数与启用日期")
    if not set(ETF_NAMES).issubset(core_prices.columns):
        raise ValueError("核心行情缺少默认16只ETF")
    config=config or RiskConfig.current_default()
    core_prices=core_prices.loc[:,list(ETF_NAMES)]
    core=risk_controlled_signals(core_prices,config,store,**state.signal_kwargs(core_prices))
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
    return PortfolioSignals(prices,core,signals,satellite_state,hypothetical_satellite_history)


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
