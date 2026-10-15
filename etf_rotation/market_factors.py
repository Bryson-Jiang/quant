"""可复用的大盘状态因子；仅使用信号日及此前的 ETF 收盘价。"""

from __future__ import annotations

import pandas as pd

from .factor_store import FactorStore
from .factors import FactorSpec, validate_prices


DOMESTIC_EQUITY = (
    "510300", "159915", "159995", "159875", "159928", "512660", "515070", "512880",
)


def regime_gates(prices: pd.DataFrame, store: FactorStore) -> pd.DataFrame:
    """保留未启用的四种市场状态候选因子，False 表示历史不足或条件未满足。

    这些列仅供后续前瞻研究；当前策略不会据此增加风险仓位。
    """
    validate_prices(prices)
    required = set(DOMESTIC_EQUITY)
    if not required.issubset(prices.columns):
        raise ValueError(f"市场状态因子缺少 ETF：{sorted(required - set(prices.columns))}")
    market_momentum = store.get(prices, FactorSpec("momentum", 40))["510300"]
    trend = store.get(prices, FactorSpec("trend", 40))
    market_up = market_momentum.gt(0) & trend["510300"].gt(0)
    ma_20 = store.get(prices, FactorSpec("ma_persistence_10_30", 20))["510300"]
    ma_40 = store.get(prices, FactorSpec("ma_persistence_10_30", 40))["510300"]
    fast_20 = store.get(prices, FactorSpec("ma_persistence_5_10", 20))["510300"]
    breadth = trend.loc[:, list(DOMESTIC_EQUITY)].gt(0).sum(axis=1)
    return pd.DataFrame({
        "direction_only": market_up,
        "ma_20_persistence": market_up & ma_20.eq(1),
        "report_strict_ma": market_up & ma_40.eq(1) & fast_20.eq(1),
        "domestic_breadth": market_up & breadth.ge(5),
    }, index=prices.index)
