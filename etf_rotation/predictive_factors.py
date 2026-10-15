"""跨行业周收益预测因子；所有模型仅以信号日及此前观测拟合。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .factors import validate_prices
from .risk_strategy import weekly_signal_dates


@dataclass(frozen=True)
class PCRSpec:
    train_weeks: int
    components: int
    mad_limit: float = 3.0

    def __post_init__(self) -> None:
        if self.train_weeks < 30 or self.components < 1 or self.mad_limit <= 0:
            raise ValueError("PCR 参数无效")


def weekly_pcr_forecast(prices: pd.DataFrame, symbols: tuple[str, ...],
                        spec: PCRSpec) -> pd.DataFrame:
    """在本周末收盘后，预测下周各 symbol 的收盘到收盘对数收益。

    训练配对为 (r[w-1], r[w])，最后一个目标是已观察到的本周收益 r[t]；
    用 r[t] 预测 r[t+1]。去极值、标准化、PCA 与回归每周只在训练窗口内重算。
    """
    validate_prices(prices)
    if len(symbols) != len(set(symbols)) or not set(symbols).issubset(prices.columns):
        raise ValueError("symbols 必须是价格矩阵中的不重复标的")
    if spec.components > len(symbols):
        raise ValueError("主成分个数不能超过标的数")
    weekly = prices.loc[weekly_signal_dates(prices.index), list(symbols)]
    weekly_returns = np.log(weekly / weekly.shift(1))
    forecasts = pd.DataFrame(np.nan, index=weekly.index, columns=list(symbols), dtype=float)
    values = weekly_returns.to_numpy(dtype=float)
    for t in range(spec.train_weeks + 1, len(weekly_returns)):
        predictors = values[t - spec.train_weeks:t]
        targets = values[t - spec.train_weeks + 1:t + 1]
        current = values[t]
        if not (np.isfinite(predictors).all() and np.isfinite(targets).all()
                and np.isfinite(current).all()):
            continue
        median = np.median(predictors, axis=0)
        mad = np.median(np.abs(predictors - median), axis=0)
        lower = median - spec.mad_limit * mad
        upper = median + spec.mad_limit * mad
        clipped = np.clip(predictors, lower, upper)
        mean = clipped.mean(axis=0)
        scale = clipped.std(axis=0, ddof=1)
        if (scale <= 1e-12).any():
            continue
        standardized = (clipped - mean) / scale
        _, _, directions = np.linalg.svd(standardized, full_matrices=False)
        axes = directions[:spec.components].T
        latent = standardized @ axes
        design = np.column_stack([np.ones(len(latent)), latent])
        coefficients = np.linalg.lstsq(design, targets, rcond=None)[0]
        current_latent = ((np.clip(current, lower, upper) - mean) / scale) @ axes
        forecasts.iloc[t] = np.r_[1.0, current_latent] @ coefficients
    forecasts.index.name = "signal_date"
    return forecasts
