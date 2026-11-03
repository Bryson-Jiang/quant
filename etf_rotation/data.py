"""Tushare ETF 日线、基金复权因子与本地 CSV 缓存。"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ADJUST, BENCHMARK, ETF_NAMES


SOURCE = "tushare"
CACHE_SCHEMA_VERSION = 2
PROJECT_DIR = Path(__file__).resolve().parent.parent
TOKEN_FILE = PROJECT_DIR / "data" / "tushare_token.txt"
CHUNK_DAYS = 1460  # fund_adj 每次最多 2000 行；四个自然年通常低于上限。
UPDATE_OVERLAP_DAYS = 21  # 增量更新时复查最近三周，吸收近期行情/复权因子修订。


def tushare_code(symbol: str) -> str:
    """将沪深 ETF 六位代码转为 Tushare ts_code。"""
    if len(symbol) != 6 or not symbol.isdigit():
        raise ValueError(f"无效 ETF 代码：{symbol}")
    if symbol.startswith("5"):
        return f"{symbol}.SH"
    if symbol.startswith("1"):
        return f"{symbol}.SZ"
    raise ValueError(f"不支持的沪深 ETF 代码：{symbol}")


def _token() -> str:
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token and TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8-sig").strip()
    if not token:
        raise RuntimeError(
            "未找到 Tushare Token。请设置环境变量 TUSHARE_TOKEN，"
            "或将 Token 单独写入 data/tushare_token.txt（不要上传该文件）。"
        )
    return token


def _client():
    try:
        import tushare as ts
    except ImportError as exc:
        raise RuntimeError("请先安装依赖：python -m pip install -r requirements.txt") from exc
    return ts.pro_api(_token())


def _windows(start: date, end: date):
    cursor = start
    while cursor <= end:
        stop = min(cursor + timedelta(days=CHUNK_DAYS - 1), end)
        yield cursor, stop
        cursor = stop + timedelta(days=1)


def _download_table(pro, method: str, ts_code: str, start: date, end: date) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for left, right in _windows(start, end):
        frame = getattr(pro, method)(
            ts_code=ts_code,
            start_date=left.strftime("%Y%m%d"),
            end_date=right.strftime("%Y%m%d"),
        )
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"{method} 返回了非 DataFrame 数据")
        if not frame.empty:
            parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _normalize_table(frame: pd.DataFrame, value_column: str, symbol: str) -> pd.Series:
    required = {"trade_date", value_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"{symbol} 数据缺少列：{sorted(required - set(frame.columns))}")
    dates = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    values = pd.to_numeric(frame[value_column], errors="coerce")
    if dates.isna().any() or values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all() or (values <= 0).any():
        raise ValueError(f"{symbol} 的 {value_column} 含无效日期或非正数")
    series = pd.Series(values.to_numpy(dtype=float), index=pd.DatetimeIndex(dates), name=value_column)
    if series.index.has_duplicates:
        raise ValueError(f"{symbol} 的 {value_column} 含重复交易日")
    return series.sort_index()


def _normalize_history(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """校验缓存中的原始价、复权因子和后复权价。"""
    required = {"date", "raw_close", "adj_factor", "close"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{symbol} 缓存缺少列：{sorted(required - set(frame.columns))}")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    numeric = frame[["raw_close", "adj_factor", "close"]].apply(pd.to_numeric, errors="coerce")
    if (dates.isna().any() or numeric.isna().any().any()
            or not np.isfinite(numeric.to_numpy(dtype=float)).all() or (numeric <= 0).any().any()):
        raise ValueError(f"{symbol} 缓存含无效日期或价格")
    expected = numeric["raw_close"] * numeric["adj_factor"]
    if not np.allclose(numeric["close"], expected, rtol=1e-9, atol=1e-9):
        raise ValueError(f"{symbol} 缓存中的后复权价与原始价×复权因子不一致")
    out = pd.DataFrame(numeric.to_numpy(dtype=float), index=pd.DatetimeIndex(dates, name="date"),
                       columns=["raw_close", "adj_factor", "close"])
    if out.index.has_duplicates or out.empty:
        raise ValueError(f"{symbol} 缓存日期重复或为空")
    return out.sort_index()


def _fetch_history(symbol: str, start: date, end: date, *, allow_empty: bool = False) -> pd.DataFrame:
    pro = _client()
    ts_code = tushare_code(symbol)
    daily = _download_table(pro, "fund_daily", ts_code, start, end)
    if daily.empty:
        if allow_empty:
            return pd.DataFrame(columns=["raw_close", "adj_factor", "close"],
                                index=pd.DatetimeIndex([], name="date"))
        raise ValueError(f"fund_daily 未返回 {ts_code} 在 {start} 至 {end} 的行情")
    raw_close = _normalize_table(daily, "close", symbol)
    factors = _download_table(pro, "fund_adj", ts_code, start, end)
    if factors.empty:
        raise ValueError(f"fund_adj 未返回 {ts_code} 的基金复权因子；请检查接口权限")
    factor = _normalize_table(factors, "adj_factor", symbol).reindex(raw_close.index)
    if factor.isna().any():
        missing = raw_close.index[factor.isna()][:5].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"{ts_code} 在 {missing} 等日期缺少复权因子，已停止以免误算收益")
    # 用同一交易日的因子计算后复权价，不使用未来交易日的因子。
    return pd.DataFrame({
        "raw_close": raw_close.to_numpy(dtype=float),
        "adj_factor": factor.to_numpy(dtype=float),
        "close": (raw_close * factor).to_numpy(dtype=float),
    }, index=pd.DatetimeIndex(raw_close.index, name="date"))


def load_symbol(
    symbol: str,
    start: date,
    end: date,
    cache_dir: Path,
    *,
    offline: bool = False,
    refresh: bool = False,
) -> pd.Series:
    """读取 Tushare 日线；向后延长时仅补近期数据并复查重叠区。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{SOURCE}_{symbol}_{ADJUST}"
    csv_path = cache_dir / f"{prefix}.csv"
    meta_path = cache_dir / f"{prefix}.json"
    frame: pd.DataFrame | None = None
    meta: dict | None = None
    if csv_path.exists() and meta_path.exists() and not refresh:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (meta["source"] == SOURCE and meta["schema_version"] == CACHE_SCHEMA_VERSION
                    and meta["symbol"] == symbol and meta["adjust"] == ADJUST):
                frame = _normalize_history(pd.read_csv(csv_path), symbol)
                if meta["start"] > meta["end"]:
                    raise ValueError("缓存日期范围无效")
            else:
                meta = None
        except (KeyError, ValueError, OSError, json.JSONDecodeError):
            frame = None
            meta = None
    covered = (frame is not None and meta is not None
               and meta["start"] <= start.isoformat() and meta["end"] >= end.isoformat())
    if not covered:
        if offline:
            raise FileNotFoundError(f"Tushare 离线缓存未覆盖 {symbol} 的 {start} 至 {end}：{csv_path}")
        try:
            if frame is None or meta is None or refresh or meta["start"] > start.isoformat():
                # 首次请求、强制刷新或历史起点前移：重新获取完整区间。
                frame = _fetch_history(symbol, start, end)
                cache_start = start
                cache_end = end
            else:
                # 仅终点向后移动：重新核对最后三周并追加新交易日。
                cache_start = date.fromisoformat(meta["start"])
                cache_end = end
                overlap_start = max(cache_start, date.fromisoformat(meta["end"])
                                    - timedelta(days=UPDATE_OVERLAP_DAYS - 1))
                latest = _fetch_history(symbol, overlap_start, end, allow_empty=True)
                if not latest.empty:
                    frame = pd.concat([frame, latest])
                    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
                    frame = _normalize_history(frame.reset_index(), symbol)
        except Exception as exc:
            raise RuntimeError(f"下载 {symbol}（{ETF_NAMES.get(symbol, symbol)}）失败：{exc}") from exc
        tmp_csv = csv_path.with_name(csv_path.name + ".tmp")
        tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
        frame.to_csv(tmp_csv, encoding="utf-8-sig", index_label="date")
        tmp_meta.write_text(json.dumps({
            "source": SOURCE, "schema_version": CACHE_SCHEMA_VERSION,
            "symbol": symbol, "adjust": ADJUST,
            "start": cache_start.isoformat(), "end": cache_end.isoformat(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_csv.replace(csv_path)
        tmp_meta.replace(meta_path)
    return frame.loc[(frame.index.date >= start) & (frame.index.date <= end), "close"].rename(symbol)


def load_close_prices(
    start: date,
    end: date,
    cache_dir: Path,
    *,
    offline: bool = False,
    refresh: bool = False,
    warmup_days: int = 400,
    symbols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """以沪深300 ETF 的交易日为主日历，其他 ETF 上市前保持空值。"""
    if start >= end:
        raise ValueError("开始日期必须早于结束日期")
    fetch_start = start - timedelta(days=warmup_days)
    series = {
        symbol: load_symbol(symbol, fetch_start, end, cache_dir,
                            offline=offline, refresh=refresh)
        for symbol in (symbols or tuple(ETF_NAMES))
    }
    calendar = series[BENCHMARK].index
    if calendar.empty:
        raise ValueError("基准 ETF 没有交易日")
    prices = pd.DataFrame({symbol: s.reindex(calendar) for symbol, s in series.items()}, index=calendar)
    prices.index.name = "date"
    if prices.loc[prices.index >= pd.Timestamp(start)].empty:
        raise ValueError("回测日期范围内没有交易日")
    return prices
