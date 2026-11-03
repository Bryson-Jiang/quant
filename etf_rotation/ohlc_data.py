"""RSRS研究独立高低价数据：与已有收盘快照逐日核对，同日复权。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import _normalize_history


def adjusted_high_low(daily: pd.DataFrame, history: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """history为已有raw_close/adj_factor/close缓存；缺价、修订或错误均停止。

    不获取最新因子来缩放整段历史，也不将close复制为high/low。
    返回的日期严格等于现有收盘快照，避免两份数据悄悄错位。
    """
    cached = _normalize_history(history, symbol)
    if not {"trade_date", "high", "low", "close"}.issubset(daily.columns):
        raise ValueError(f"{symbol}日线缺少RSRS所需字段")
    dates = pd.to_datetime(daily.trade_date.astype(str), format="%Y%m%d", errors="coerce")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError(f"{symbol}日线日期无效或重复")
    raw = daily[["high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    raw.index = pd.DatetimeIndex(dates, name="date")
    raw = raw.sort_index().reindex(cached.index)
    a = raw.to_numpy(float)
    if not np.isfinite(a).all() or (a <= 0).any():
        raise ValueError(f"{symbol}高低价缺失、非正或未覆盖既有收盘快照")
    if ((raw.high < raw.low) | (raw.high + 1e-9 < raw.close)
            | (raw.low - 1e-9 > raw.close)).any():
        raise ValueError(f"{symbol}高低价区间未包含收盘价")
    if not np.allclose(raw.close, cached.raw_close, rtol=0, atol=1e-9):
        raise ValueError(f"{symbol}新日线收盘与冻结快照不一致；不能混用修订前后数据")
    return pd.DataFrame({"raw_high": raw.high, "raw_low": raw.low,
                         "adj_factor": cached.adj_factor,
                         "high": raw.high * cached.adj_factor,
                         "low": raw.low * cached.adj_factor,
                         "close": cached.close}, index=cached.index)
