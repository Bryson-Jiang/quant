"""澄流120的市场残差波动过滤；因子与2026-11-09研究候选共用定义。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from .extended_factors import ExtendedFactorSpec, compute_extended, low_ivol_utility
from .factor_store import FactorStore
from .risk_strategy import universe_activation_mask

if TYPE_CHECKING:
    from .strategy_state import IVOLPolicy


@dataclass(frozen=True)
class IVOLFilterResult:
    ivol: pd.DataFrame
    utility: pd.DataFrame
    eligibility_mask: pd.DataFrame
    active: pd.Series
    comparison_symbols: tuple[str, ...]


def market_ivol_filter(prices: pd.DataFrame, policy: IVOLPolicy,
                       store: FactorStore | None = None, *,
                       hypothetical_history: bool = False,
                       as_of: date | None = None,
                       universe_activation_dates: dict | None = None) -> IVOLFilterResult:
    """缺因子中性；启用前不影响资格，启用后的过滤优先于排名缓冲。"""
    spec=ExtendedFactorSpec('ivol',window=policy.window,model='market')
    ivol=store.get_extended(prices,spec) if store is not None else compute_extended(prices,spec)
    if universe_activation_dates:
        ivol=ivol.where(universe_activation_mask(prices,universe_activation_dates))
    members=tuple(s for s in policy.comparison_symbols if s in prices.columns)
    utility=low_ivol_utility(ivol,list(members))
    enough=ivol.loc[:,list(members)].notna().sum(axis=1).ge(policy.min_assets)
    utility=utility.where(enough,axis=0)
    active=pd.Series(True,index=prices.index,name='ivol_filter_active')
    if not hypothetical_history:
        active.loc[prices.index<pd.Timestamp(policy.effective_signal_date)]=False
        if as_of is not None and as_of<policy.adopted_on:active.loc[:]=False
    allowed=utility.ge(1-2*policy.max_normalized_rank)|utility.isna()
    allowed.loc[~active]=True
    return IVOLFilterResult(ivol,utility,allowed,active,members)
