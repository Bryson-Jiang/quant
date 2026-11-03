"""2026-10-31之前的16只研究基线；防止默认扩池改写旧报告。"""
from pathlib import Path
from datetime import date
import pandas as pd
from .config import LEGACY_PORTFOLIO_STRATEGY_NAME, LEGACY_CORE_NAMES, BENCHMARK
from .data import load_close_prices, load_symbol
from .strategy_state import load_rotation_state as _load


def load_rotation_state(path: Path, *, expected_strategy=LEGACY_PORTFOLIO_STRATEGY_NAME):
    path=Path(path)
    if expected_strategy==LEGACY_PORTFOLIO_STRATEGY_NAME and path.name=='rotation_state.json':
        path=path.with_name('rotation_state_20261031.json')
    return _load(path,expected_strategy=expected_strategy)


def load_portfolio_data(start,end,cache_dir,*,offline=False,refresh=False):
    core=load_close_prices(min(start,date(2022,1,1)),end,cache_dir,offline=offline,refresh=refresh,
                           symbols=tuple(LEGACY_CORE_NAMES))
    market=load_symbol(BENCHMARK,min(date(2019,1,1),core.index[0].date()),end,cache_dir,offline=offline,refresh=refresh)
    sat=(load_symbol('512890',date(2019,1,1),end,cache_dir,offline=offline,refresh=refresh)
         if end>=date(2019,1,1) else pd.Series(dtype=float))
    return core,sat.reindex(market.index).to_frame('512890')
