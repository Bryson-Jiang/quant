"""记录策略切换时的旧周目标，回测与实盘从同一历史锚点重放缓冲状态。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from .config import ETF_NAMES, STRATEGY_NAME
from .risk_strategy import BOND


@dataclass(frozen=True)
class RotationState:
    selection_anchor_date: date
    effective_signal_date: date
    initial_risky_selection: tuple[str, ...]

    def signal_kwargs(self, prices: pd.DataFrame) -> dict:
        """锚点之后必须保有锚点行情；历史切片不读取未来的切换状态。"""
        anchor = pd.Timestamp(self.selection_anchor_date)
        if prices.index[-1] <= anchor:
            return {}
        if anchor not in prices.index:
            raise ValueError("行情未覆盖策略切换锚点，请从2022年起加载历史后再截取回测区间")
        return {"initial_risky_selection": list(self.initial_risky_selection),
                "initial_selection_date": anchor}

    def metadata(self) -> dict:
        return {"selection_anchor_date": self.selection_anchor_date.isoformat(),
                "effective_signal_date": self.effective_signal_date.isoformat(),
                "initial_risky_selection": list(self.initial_risky_selection),
                "state_basis": "前次周目标选仓；固定锚点后逐周重放，不是券商成交记录"}


def load_rotation_state(path: Path) -> RotationState:
    """缺失或错误时停止发布，避免丢失缓冲历史后静默改变实盘计划。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("strategy") != STRATEGY_NAME:
        raise ValueError("策略切换记录的版本或策略名称不匹配")
    anchor = date.fromisoformat(payload["selection_anchor_date"])
    effective = date.fromisoformat(payload["effective_signal_date"])
    symbols = payload["initial_risky_selection"]
    if (not isinstance(symbols, list) or any(not isinstance(s, str) for s in symbols)
            or len(symbols) > 3 or len(set(symbols)) != len(symbols)
            or not set(symbols).issubset(set(ETF_NAMES) - {BOND}) or anchor >= effective):
        raise ValueError("策略切换记录的日期或旧选仓无效")
    return RotationState(anchor, effective, tuple(symbols))
