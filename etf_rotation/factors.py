"""只使用信号日及之前价格的 ETF 因子库。

所有窗口均以交易日计数；收盘因子输入为未填补缺价的后复权收盘价。
RSRS另用同日复权最高/最低价，不能从收盘价伪造。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BENCHMARK


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
    "short_term_reversal", "rsi", "residual_reversal",
    "drawdown_rebound", "relative_strength",
    "downside_balance",
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
    elif spec.name == "rsi":
        # 简单移动平均版RSI（非Wilder平滑）；平盘窗口取中性50。
        change = shifted.diff()
        gain = change.clip(lower=0).rolling(spec.window, min_periods=spec.window).mean()
        loss = -change.clip(upper=0).rolling(spec.window, min_periods=spec.window).mean()
        total = gain + loss
        out = (100 * gain / total.replace(0, np.nan)).where(total.ne(0), 50).where(total.notna()).clip(0, 100)
    elif spec.name in {"short_term_reversal", "residual_reversal", "relative_strength", "drawdown_rebound"}:
        # 固定60日风险尺度；window是短期冲击/回撤窗口，不是风险估计窗口。
        log_returns = np.log(shifted / shifted.shift(1))
        daily_vol = log_returns.rolling(60, min_periods=60).std(ddof=1)
        move = np.log(shifted / shifted.shift(spec.window))
        scale = daily_vol * np.sqrt(spec.window)
        if spec.name == "short_term_reversal":
            out = -move / scale.replace(0, np.nan)
        elif spec.name == "drawdown_rebound":
            # t-3以前的窗口回撤深度 × 最近3日已确认反弹，均按历史波动缩放。
            before = shifted.shift(3)
            depth = (1 - before / before.rolling(spec.window, min_periods=spec.window).max()).clip(lower=0)
            bounce = np.log(shifted / before).clip(lower=0)
            out = depth / scale.replace(0, np.nan) * bounce / (daily_vol * np.sqrt(3)).replace(0, np.nan)
        else:
            if BENCHMARK not in prices:
                raise ValueError(f"{spec.name} 需要基准列 {BENCHMARK}")
            market = log_returns[BENCHMARK]
            market_move = move[BENCHMARK]
            if spec.name == "relative_strength":
                out = move.sub(market_move, axis=0) / scale.replace(0, np.nan)
            else:
                # beta及残差波动只用短期冲击开始前的60日，避免用该冲击拟合自身。
                covariance = log_returns.rolling(60, min_periods=60).cov(market)
                market_var = market.rolling(60, min_periods=60).var(ddof=1)
                beta = covariance.div(market_var.replace(0, np.nan), axis=0).shift(spec.window)
                residual_var = (log_returns.rolling(60, min_periods=60).var(ddof=1)
                                - covariance.pow(2).div(market_var.replace(0, np.nan), axis=0))
                residual_scale = np.sqrt(residual_var.clip(lower=0)).shift(spec.window) * np.sqrt(spec.window)
                residual = move - beta.mul(market_move, axis=0)
                out = -residual / residual_scale.where(residual_scale > 1e-10)
                out[BENCHMARK] = np.nan  # 基准对自身没有可识别的残差风险。
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
        elif spec.name == "downside_balance":
            # 收益平方中上涨与下跌的占比差；正值表示上涨贡献更大，范围[-1,1]。
            # 使用非中心二阶矩，平盘窗口无可识别方向，保持NaN。
            total = returns.pow(2).rolling(spec.window, min_periods=spec.window).sum()
            negative = returns.clip(upper=0).pow(2).rolling(spec.window, min_periods=spec.window).sum()
            out = (1 - 2 * negative / total.replace(0, np.nan)).clip(-1, 1)
        elif spec.name == "volatility_ratio":
            recent = returns.rolling(spec.window, min_periods=spec.window).std(ddof=1)
            older = returns.rolling(spec.window * 3, min_periods=spec.window * 3).std(ddof=1)
            out = recent / older.replace(0, np.nan)
        else:
            momentum = shifted / prices.shift(spec.skip + spec.window) - 1
            vol = returns.rolling(spec.window, min_periods=spec.window).std(ddof=1) * np.sqrt(252)
            out = momentum / vol.clip(lower=0.05)
    return out.where(prices.notna()).replace([np.inf, -np.inf], np.nan)


@dataclass(frozen=True)
class RSRSSpec:
    """N日OLS斜率，最近M个斜率（含当日）的总体标准差Z分数。"""

    regression_days: int = 18
    standardization_days: int = 600

    def __post_init__(self) -> None:
        if (not isinstance(self.regression_days, int) or self.regression_days < 3
                or not isinstance(self.standardization_days, int) or self.standardization_days < 2):
            raise ValueError("RSRS回归窗口至少3日，标准化窗口至少2日")


def validate_high_low(high: pd.DataFrame, low: pd.DataFrame) -> None:
    validate_prices(high)
    validate_prices(low)
    if not high.index.equals(low.index) or not high.columns.equals(low.columns):
        raise ValueError("RSRS最高/最低价必须严格对齐")
    if high.notna().ne(low.notna()).any().any() or (high < low).any().any():
        raise ValueError("RSRS高低价必须成对存在，且最高价不能低于最低价")


def compute_rsrs(high: pd.DataFrame, low: pd.DataFrame,
                 spec: RSRSSpec = RSRSSpec()) -> dict[str, pd.DataFrame]:
    """H=alpha+beta*L+epsilon；全部滚动窗口只读当日及以前完整报价。

    zscore=(beta-mean_M(beta))/std_M(beta, ddof=0)
    adjusted=zscore*R²；right_skew=zscore*R²*beta。
    这是价格回归的拟合R²，不是预测未来收益的R²。常量、缺价或未预热
    窗口保留NaN；不前填、不用收盘价充当高低价、不用全样本标准化。
    """
    validate_high_low(high, low)
    n, m = spec.regression_days, spec.standardization_days
    x = low.rolling(n, min_periods=n)
    y = high.rolling(n, min_periods=n)
    vx, vy = x.var(ddof=0), y.var(ddof=0)
    covariance = x.cov(high, ddof=0)
    stable_x = vx > 1e-12 * x.mean().pow(2)
    stable_y = vy > 1e-12 * y.mean().pow(2)
    slope = covariance / vx.where(stable_x)
    fit = (covariance.pow(2) / (vx * vy).where(stable_x & stable_y)).clip(0, 1)
    history = slope.rolling(m, min_periods=m)
    sigma = history.std(ddof=0)
    z = (slope - history.mean()) / sigma.where(sigma > 1e-10)
    values = {"slope": slope, "r_squared": fit, "zscore": z,
              "adjusted": z * fit, "right_skew": z * fit * slope}
    return {name: frame.where(high.notna()).replace([np.inf, -np.inf], np.nan)
            for name, frame in values.items()}
