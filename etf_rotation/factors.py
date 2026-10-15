"""只使用信号日及之前价格的 ETF 因子库。

所有窗口均以交易日计数；输入必须是未填补缺价的后复权收盘价。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorSpec:
    name: str
    window: int
    skip: int = 0

    def __post_init__(self) -> None:
        if self.name not in FACTOR_NAMES:
            raise ValueError(f"未知因子：{self.name}")
        if self.window < 1 or self.skip < 0:
            raise ValueError("window 必须为正，skip 不能为负")


FACTOR_NAMES = frozenset({
    "momentum", "trend", "volatility", "downside_volatility",
    "breakout", "risk_adjusted_momentum", "trend_efficiency", "volatility_ratio",
    "ma_zscore", "ma_persistence_10_30", "ma_persistence_5_10",
    "rolling_drawdown", "rolling_runup",
})


def validate_prices(prices: pd.DataFrame) -> None:
    if (prices.empty or not isinstance(prices.index, pd.DatetimeIndex)
            or not prices.index.is_monotonic_increasing or prices.index.has_duplicates
            or prices.columns.has_duplicates):
        raise ValueError("prices 必须是日期升序、无重复的非空价格矩阵")
    values = prices.to_numpy(dtype=float)
    if np.isinf(values).any() or (values[~np.isnan(values)] <= 0).any():
        raise ValueError("价格必须为正且有限")


def compute_factor(prices: pd.DataFrame, spec: FactorSpec) -> pd.DataFrame:
    """因子值仅依赖 t 日及以前的观测，信号日无报价时值为 NaN。"""
    validate_prices(prices)
    shifted = prices.shift(spec.skip)
    if spec.name == "momentum":
        out = shifted / prices.shift(spec.skip + spec.window) - 1
    elif spec.name == "trend":
        out = shifted / shifted.rolling(spec.window, min_periods=spec.window).mean() - 1
    elif spec.name in {"ma_persistence_10_30", "ma_persistence_5_10"}:
        # 报告中的均线同侧持续性。+1/-1 表示最近 window 日始终在同一侧，
        # 0 表示期间发生交叉；缺少完整历史时保持 NaN。
        fast, slow = ((10, 30) if spec.name == "ma_persistence_10_30" else (5, 10))
        spread = (shifted.rolling(fast, min_periods=fast).mean()
                  - shifted.rolling(slow, min_periods=slow).mean())
        positive = spread.gt(0).rolling(spec.window, min_periods=spec.window).sum().eq(spec.window)
        negative = spread.lt(0).rolling(spec.window, min_periods=spec.window).sum().eq(spec.window)
        valid = spread.notna().rolling(spec.window, min_periods=spec.window).sum().eq(spec.window)
        out = (positive.astype(float) - negative.astype(float)).where(valid)
    elif spec.name == "ma_zscore":
        average = shifted.rolling(spec.window, min_periods=spec.window).mean()
        dispersion = shifted.rolling(spec.window, min_periods=spec.window).std(ddof=1)
        out = (shifted - average) / dispersion.replace(0, np.nan)
    elif spec.name == "rolling_drawdown":
        high = shifted.rolling(spec.window, min_periods=spec.window).max()
        out = shifted / high - 1
    elif spec.name == "rolling_runup":
        low = shifted.rolling(spec.window, min_periods=spec.window).min()
        out = shifted / low - 1
    elif spec.name == "breakout":
        past_high = shifted.shift(1).rolling(spec.window, min_periods=spec.window).max()
        out = shifted / past_high - 1
    elif spec.name == "trend_efficiency":
        path_length = shifted.diff().abs().rolling(spec.window, min_periods=spec.window).sum()
        out = (shifted - shifted.shift(spec.window)) / path_length.replace(0, np.nan)
    else:
        returns = prices.pct_change(fill_method=None).shift(spec.skip)
        if spec.name == "volatility":
            out = returns.rolling(spec.window, min_periods=spec.window).std(ddof=1) * np.sqrt(252)
        elif spec.name == "downside_volatility":
            downside = returns.clip(upper=0)
            out = np.sqrt(downside.pow(2).rolling(spec.window, min_periods=spec.window).mean() * 252)
        elif spec.name == "volatility_ratio":
            recent = returns.rolling(spec.window, min_periods=spec.window).std(ddof=1)
            older = returns.rolling(spec.window * 3, min_periods=spec.window * 3).std(ddof=1)
            out = recent / older.replace(0, np.nan)
        else:
            momentum = shifted / prices.shift(spec.skip + spec.window) - 1
            vol = returns.rolling(spec.window, min_periods=spec.window).std(ddof=1) * np.sqrt(252)
            out = momentum / vol.clip(lower=0.05)
    return out.where(prices.notna()).replace([np.inf, -np.inf], np.nan)
