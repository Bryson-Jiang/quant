"""记录策略切换时的旧周目标，回测与实盘从同一历史锚点重放缓冲状态。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from .config import (BUFFERED_MOMENTUM_STRATEGY_NAME, CORE_STRATEGY_NAME, ETF_NAMES,
                     LEGACY_CORE_NAMES, LEGACY_CORE_STRATEGY_NAME, LEGACY_PORTFOLIO_STRATEGY_NAME,
                     SATELLITE_ETF_NAMES, STRATEGY_NAME)
from .risk_strategy import BOND


@dataclass(frozen=True)
class SelectionAnchor:
    selection_anchor_date: date
    effective_signal_date: date
    initial_risky_selection: tuple[str, ...]


@dataclass(frozen=True)
class DividendPolicy:
    symbol: str
    ma_days: int
    entry_deviation: float
    exit_deviation: float
    max_holding_days: int | None
    max_weight: float
    funding: str
    adopted_on: date
    effective_signal_date: date
    first_execution_date: date
    initially_active: bool = False

    def metadata(self) -> dict:
        from dataclasses import asdict
        return {k: v.isoformat() if isinstance(v,date) else v for k,v in asdict(self).items()}


@dataclass(frozen=True)
class UniversePolicy:
    added_symbols: tuple[str, ...]
    adopted_on: date
    effective_signal_date: date
    first_execution_date: date
    previous_strategy: str

    def metadata(self):
        return {"added_symbols":list(self.added_symbols),"adopted_on":self.adopted_on.isoformat(),
                "effective_signal_date":self.effective_signal_date.isoformat(),
                "first_execution_date":self.first_execution_date.isoformat(),"previous_strategy":self.previous_strategy}


@dataclass(frozen=True)
class RotationState(SelectionAnchor):
    historical_anchors: tuple[SelectionAnchor, ...] = ()
    dividend_policy: DividendPolicy | None = None
    universe_policy: UniversePolicy | None = None
    core_symbols: tuple[str, ...] | None = None

    def universe_activation_dates(self, *, hypothetical: bool = False) -> dict:
        if hypothetical or self.universe_policy is None:return {}
        return {s:self.universe_policy.effective_signal_date for s in self.universe_policy.added_symbols}

    def signal_kwargs(self, prices: pd.DataFrame) -> dict:
        """锚点之后必须保有锚点行情；历史切片不读取未来的切换状态。"""
        anchors = (*self.historical_anchors, SelectionAnchor(self.selection_anchor_date,
                    self.effective_signal_date, self.initial_risky_selection))
        active = [a for a in anchors if prices.index[-1] > pd.Timestamp(a.selection_anchor_date)]
        if not active:
            return {}
        for a in active:
            if pd.Timestamp(a.selection_anchor_date) not in prices.index:
                raise ValueError("行情未覆盖策略切换锚点，请从2022年起加载历史后再截取回测区间")
        first, *later = active
        result = {"initial_risky_selection": list(first.initial_risky_selection),
                  "initial_selection_date": pd.Timestamp(first.selection_anchor_date)}
        if later:
            result["selection_resets"] = [(pd.Timestamp(a.selection_anchor_date), list(a.initial_risky_selection)) for a in later]
        return result

    def metadata(self) -> dict:
        result = {"selection_anchor_date": self.selection_anchor_date.isoformat(),
                "effective_signal_date": self.effective_signal_date.isoformat(),
                "initial_risky_selection": list(self.initial_risky_selection),
                "state_basis": "前次周目标选仓；固定锚点后逐周重放，不是券商成交记录"}
        if self.historical_anchors:
            result["historical_selection_anchors"] = [
                {"selection_anchor_date": a.selection_anchor_date.isoformat(),
                 "effective_signal_date": a.effective_signal_date.isoformat(),
                 "initial_risky_selection": list(a.initial_risky_selection)} for a in self.historical_anchors]
        if self.dividend_policy is not None:
            result["dividend_policy"]=self.dividend_policy.metadata()
        if self.universe_policy is not None:
            result["universe_policy"]=self.universe_policy.metadata()
        return result


def load_rotation_state(path: Path, *, expected_strategy: str = STRATEGY_NAME) -> RotationState:
    """缺失或错误时停止发布，避免丢失缓冲历史后静默改变实盘计划。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in {1, 2, 3, 4} or payload.get("strategy") != expected_strategy:
        raise ValueError("策略切换记录的版本或策略名称不匹配")

    def parse_anchor(item):
        anchor = date.fromisoformat(item["selection_anchor_date"])
        effective = date.fromisoformat(item["effective_signal_date"])
        symbols = item["initial_risky_selection"]
        if (not isinstance(symbols, list) or any(not isinstance(s, str) for s in symbols)
                or len(symbols) > 3 or len(set(symbols)) != len(symbols)
                or not set(symbols).issubset(set(ETF_NAMES) - {BOND}) or anchor >= effective):
            raise ValueError("策略切换记录的日期或旧选仓无效")
        return SelectionAnchor(anchor, effective, tuple(symbols))

    latest = parse_anchor(payload)
    history = tuple(parse_anchor(item) for item in payload.get("historical_selection_anchors", []))
    sequence = (*history, latest)
    if any(a.effective_signal_date > b.selection_anchor_date for a, b in zip(sequence, sequence[1:])):
        raise ValueError("策略切换锚点必须按时间排序且不能重叠")
    policy=None
    if payload.get("schema_version") in {3,4}:
        pairs={STRATEGY_NAME:CORE_STRATEGY_NAME,LEGACY_PORTFOLIO_STRATEGY_NAME:LEGACY_CORE_STRATEGY_NAME}
        if expected_strategy not in pairs or payload.get("core_strategy")!=pairs[expected_strategy]:
            raise ValueError("红利卫星的核心策略版本不匹配")
        item=payload["dividend_policy"]
        policy=DividendPolicy(**{**item,**{k:date.fromisoformat(item[k]) for k in
                                      ("adopted_on","effective_signal_date","first_execution_date")}})
        if (policy.symbol not in SATELLITE_ETF_NAMES or policy.ma_days!=200
                or policy.entry_deviation!=0 or policy.exit_deviation!=.04
                or policy.max_holding_days is not None or policy.max_weight!=.30
                or policy.funding!="cash_then_bond" or policy.initially_active is not False
                or policy.adopted_on>policy.effective_signal_date
                or policy.effective_signal_date>=policy.first_execution_date):
            raise ValueError("红利卫星参数或启用日期无效")
    elif expected_strategy==STRATEGY_NAME:
        raise ValueError("当前默认必须保有红利卫星启用记录")
    universe=None
    if payload.get("schema_version")==4:
        item=payload['universe_policy']
        universe=UniversePolicy(tuple(item['added_symbols']),*(date.fromisoformat(item[k]) for k in
                                ('adopted_on','effective_signal_date','first_execution_date')),item['previous_strategy'])
        added_symbols=set(ETF_NAMES)-set(LEGACY_CORE_NAMES)
        if (expected_strategy!=STRATEGY_NAME or set(universe.added_symbols)!=added_symbols
                or len(universe.added_symbols)!=len(added_symbols) or universe.previous_strategy!=LEGACY_PORTFOLIO_STRATEGY_NAME
                or not universe.adopted_on<=universe.effective_signal_date<universe.first_execution_date):
            raise ValueError('扩池启用记录无效')
    elif expected_strategy==STRATEGY_NAME:
        raise ValueError('当前默认必须保存扩池启用记录')
    symbols=tuple(ETF_NAMES) if expected_strategy==STRATEGY_NAME else tuple(LEGACY_CORE_NAMES)
    return RotationState(latest.selection_anchor_date, latest.effective_signal_date,
                         latest.initial_risky_selection, history,policy,universe,symbols)


def load_research_rotation_state(root: Path) -> RotationState:
    """2026-10-17研究固定使用原7%版本锚点，不随实盘升级改写历史比较。"""
    return load_rotation_state(root / "data/rotation_state_20261006.json",
                               expected_strategy=BUFFERED_MOMENTUM_STRATEGY_NAME)
