"""SQLite 因子缓存：参数或输入价格变化时自动重算。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from .factors import FactorSpec, compute_factor, validate_prices


def _price_digest(prices: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps([str(c) for c in prices.columns], ensure_ascii=False).encode())
    digest.update(np.asarray(prices.index.asi8, dtype="<i8").tobytes())
    values = prices.to_numpy(dtype="<f8", copy=True)
    values[np.isnan(values)] = np.nan  # 统一 NaN 的位模式
    digest.update(values.tobytes())
    return digest.hexdigest()


class FactorStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS factor_meta (key TEXT PRIMARY KEY, price_digest TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS factor_values (key TEXT NOT NULL, date TEXT NOT NULL, symbol TEXT NOT NULL, value REAL NOT NULL, PRIMARY KEY (key, date, symbol))")

    def get(self, prices: pd.DataFrame, spec: FactorSpec) -> pd.DataFrame:
        validate_prices(prices)
        key = json.dumps({"name": spec.name, "window": spec.window, "skip": spec.skip}, sort_keys=True)
        digest = _price_digest(prices)
        with closing(sqlite3.connect(self.path)) as db, db:
            row = db.execute("SELECT price_digest FROM factor_meta WHERE key = ?", (key,)).fetchone()
            if row is not None and row[0] == digest:
                records = db.execute("SELECT date, symbol, value FROM factor_values WHERE key = ?", (key,)).fetchall()
                out = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns, dtype=float)
                for day, symbol, value in records:
                    out.at[pd.Timestamp(day), symbol] = value
                return out
            out = compute_factor(prices, spec)
            values = [
                (key, day.isoformat(), str(symbol), float(value))
                for day, series in out.iterrows()
                for symbol, value in series.items() if pd.notna(value)
            ]
            db.execute("DELETE FROM factor_values WHERE key = ?", (key,))
            db.executemany("INSERT INTO factor_values (key, date, symbol, value) VALUES (?, ?, ?, ?)", values)
            db.execute("INSERT OR REPLACE INTO factor_meta (key, price_digest) VALUES (?, ?)", (key, digest))
            return out
