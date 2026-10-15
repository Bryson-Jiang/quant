"""Local ledger for actual account snapshots and external cash flows.

Amounts in SQLite are integer fen. Dates are Shanghai calendar dates. Weekly
returns use daily-weighted external flows (Modified Dietz) when there is no
account valuation at the exact time of a deposit or withdrawal.
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any


DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "live_portfolio.sqlite"
CENT = Decimal("0.01")
RECONCILIATION_TOLERANCE_CENTS = 200


def _day(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field} 必须为 YYYY-MM-DD 日期")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field} 必须为 YYYY-MM-DD 日期") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{field} 必须为 YYYY-MM-DD 日期")
    return parsed


def _decimal(value: Any, field: str, *, positive: bool = False,
             nonnegative: bool = False) -> Decimal:
    if isinstance(value, bool) or value is None or isinstance(value, (dict, list)):
        raise ValueError(f"{field} 必须是金额或价格")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{field} 必须是金额或价格") from exc
    if not result.is_finite() or (positive and result <= 0) or (nonnegative and result < 0):
        raise ValueError(f"{field} 数值无效")
    return result


def _cents(value: Any, field: str, *, nonnegative: bool = False) -> int:
    amount = _decimal(value, field, nonnegative=nonnegative)
    if amount != amount.quantize(CENT):
        raise ValueError(f"{field} 最多只能有两位小数（人民币元）")
    return int(amount * 100)


def _yuan(cents: int) -> str:
    return str((Decimal(cents) / 100).quantize(CENT))


def connect(path: Path = DEFAULT_DB) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS account_opening (
            id INTEGER PRIMARY KEY CHECK(id = 1),
            opening_date TEXT NOT NULL,
            equity_cents INTEGER NOT NULL CHECK(equity_cents > 0),
            note TEXT NOT NULL,
            recorded_at_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS opening_revisions (
            id INTEGER PRIMARY KEY,
            old_opening_date TEXT NOT NULL,
            old_equity_cents INTEGER NOT NULL,
            old_note TEXT NOT NULL,
            replaced_at_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS snapshots (
            valuation_date TEXT PRIMARY KEY,
            total_equity_cents INTEGER NOT NULL CHECK(total_equity_cents >= 0),
            cash_cents INTEGER NOT NULL CHECK(cash_cents >= 0),
            other_assets_cents INTEGER NOT NULL CHECK(other_assets_cents >= 0),
            note TEXT NOT NULL,
            source_json TEXT NOT NULL,
            recorded_at_utc TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS positions (
            valuation_date TEXT NOT NULL REFERENCES snapshots(valuation_date) ON DELETE CASCADE,
            symbol TEXT NOT NULL,
            shares INTEGER NOT NULL CHECK(shares > 0),
            last_price TEXT NOT NULL,
            market_value_cents INTEGER NOT NULL CHECK(market_value_cents >= 0),
            avg_cost TEXT,
            unrealized_pnl_cents INTEGER,
            PRIMARY KEY (valuation_date, symbol)
        );
        CREATE TABLE IF NOT EXISTS cash_flows (
            id INTEGER PRIMARY KEY,
            valuation_date TEXT NOT NULL REFERENCES snapshots(valuation_date) ON DELETE CASCADE,
            flow_date TEXT NOT NULL,
            amount_cents INTEGER NOT NULL CHECK(amount_cents != 0),
            note TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS snapshot_revisions (
            id INTEGER PRIMARY KEY,
            valuation_date TEXT NOT NULL,
            old_source_json TEXT NOT NULL,
            replaced_at_utc TEXT NOT NULL
        );
    """)
    return conn


def set_opening(conn: sqlite3.Connection, opening_date: str, equity: Any,
                note: str = "", *, replace: bool = False) -> None:
    """Set an actual pre-trade account valuation before recording snapshots."""
    day = _day(opening_date, "opening date")
    amount = _cents(equity, "opening equity", nonnegative=True)
    if amount == 0:
        raise ValueError("期初账户总资产必须大于 0")
    first = conn.execute("SELECT MIN(valuation_date) FROM snapshots").fetchone()[0]
    if first and day > date.fromisoformat(first):
        raise ValueError("期初日期不能晚于第一笔账户快照")
    if first and not replace:
        raise ValueError("已有账户快照；补录或修正期初基准须明确使用 --replace")
    existing = conn.execute("SELECT * FROM account_opening WHERE id = 1").fetchone()
    if existing and not replace:
        raise ValueError("期初基准已存在；确认需要修正时使用 --replace")
    if first:
        first_flows = conn.execute("SELECT flow_date FROM cash_flows WHERE valuation_date = ?",
                                   (first,)).fetchall()
        if any(not (day < date.fromisoformat(flow["flow_date"]) <= date.fromisoformat(first))
               for flow in first_flows):
            raise ValueError("期初日期与首笔快照中的资金进出日期冲突")
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        if existing:
            conn.execute("INSERT INTO opening_revisions VALUES (NULL, ?, ?, ?, ?)",
                         (existing["opening_date"], existing["equity_cents"],
                          existing["note"], now))
        conn.execute("INSERT OR REPLACE INTO account_opening VALUES (1, ?, ?, ?, ?)",
                     (day.isoformat(), amount, note, now))


def record_snapshot(conn: sqlite3.Connection, payload: dict[str, Any], *,
                    replace: bool = False) -> str:
    """Validate a full post-rebalance account valuation, then save atomically."""
    if not isinstance(payload, dict):
        raise ValueError("记录必须是 JSON 对象")
    valuation_date = _day(payload.get("date"), "date")
    day_text = valuation_date.isoformat()
    total = _cents(payload.get("total_equity"), "total_equity", nonnegative=True)
    cash = _cents(payload.get("cash"), "cash", nonnegative=True)
    other = _cents(payload.get("other_assets", "0"), "other_assets", nonnegative=True)
    note = str(payload.get("note", ""))

    existing = conn.execute("SELECT source_json FROM snapshots WHERE valuation_date = ?",
                            (day_text,)).fetchone()
    latest = conn.execute("SELECT MAX(valuation_date) FROM snapshots").fetchone()[0]
    if existing and not replace:
        raise ValueError(f"{day_text} 已有记录；核实后才可用 --replace 修正（旧版会留档）")
    if not existing and latest and day_text <= latest:
        raise ValueError("新记录必须晚于现有最后一个快照；已有日期请用 --replace")
    previous = conn.execute("SELECT MAX(valuation_date) FROM snapshots WHERE valuation_date < ?",
                            (day_text,)).fetchone()[0]
    opening = conn.execute("SELECT opening_date FROM account_opening WHERE id = 1").fetchone()
    if not previous and opening and valuation_date < date.fromisoformat(opening["opening_date"]):
        raise ValueError("快照日期不能早于期初基准日期")
    flow_boundary = previous or (opening["opening_date"] if opening else None)

    raw_positions = payload.get("positions")
    if not isinstance(raw_positions, list):
        raise ValueError("positions 必须是数组")
    positions: list[tuple[Any, ...]] = []
    seen: set[str] = set()
    for number, item in enumerate(raw_positions, 1):
        if not isinstance(item, dict):
            raise ValueError(f"positions[{number}] 必须是对象")
        symbol = str(item.get("symbol", ""))
        if not re.fullmatch(r"\d{6}", symbol) or symbol in seen:
            raise ValueError(f"positions[{number}] 代码必须为不重复的六位数字")
        seen.add(symbol)
        shares = item.get("shares")
        if isinstance(shares, bool) or not isinstance(shares, int) or shares <= 0:
            raise ValueError(f"{symbol} shares 必须是正整数份额")
        last_price = _decimal(item.get("last_price"), f"{symbol} last_price", positive=True)
        computed = int((last_price * shares * 100).quantize(Decimal("1"),
                                                            rounding=ROUND_HALF_UP))
        market_value = (_cents(item["market_value"], f"{symbol} market_value",
                               nonnegative=True) if "market_value" in item else computed)
        if abs(computed - market_value) > RECONCILIATION_TOLERANCE_CENTS:
            raise ValueError(f"{symbol} 市值与份额×现价不符，请确认 last_price 是现价")
        avg_cost = (str(_decimal(item["avg_cost"], f"{symbol} avg_cost", nonnegative=True))
                    if item.get("avg_cost") is not None else None)
        unrealized = (_cents(item["unrealized_pnl"], f"{symbol} unrealized_pnl")
                      if item.get("unrealized_pnl") is not None else None)
        positions.append((day_text, symbol, shares, str(last_price), market_value,
                          avg_cost, unrealized))

    represented = cash + other + sum(row[4] for row in positions)
    if abs(total - represented) > RECONCILIATION_TOLERANCE_CENTS:
        raise ValueError("账户总资产与现金＋持仓市值＋其他资产相差超过 2 元，请核对漏填资产")

    raw_flows = payload.get("cash_flows")
    if not isinstance(raw_flows, list):
        raise ValueError("cash_flows 必须是数组")
    if not flow_boundary and raw_flows:
        raise ValueError("首笔快照是收益基准，不要在其中重复填写初始本金；后续入金填 cash_flows")
    flows: list[tuple[Any, ...]] = []
    for number, item in enumerate(raw_flows, 1):
        if not isinstance(item, dict):
            raise ValueError(f"cash_flows[{number}] 必须是对象")
        flow_date = _day(item.get("date"), f"cash_flows[{number}].date")
        if flow_boundary and not (date.fromisoformat(flow_boundary) < flow_date <= valuation_date):
            raise ValueError("每笔资金进出日期必须晚于前一快照、且不晚于本次快照")
        amount = _cents(item.get("amount"), f"cash_flows[{number}].amount")
        if amount == 0:
            raise ValueError("资金进出金额不能为 0；入金为正，出金为负")
        flows.append((day_text, flow_date.isoformat(), amount, str(item.get("note", ""))))

    source = json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True)
    now = datetime.now(timezone.utc).isoformat()
    with conn:
        if existing:
            conn.execute("INSERT INTO snapshot_revisions VALUES (NULL, ?, ?, ?)",
                         (day_text, existing["source_json"], now))
            conn.execute("DELETE FROM snapshots WHERE valuation_date = ?", (day_text,))
        conn.execute("INSERT INTO snapshots VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (day_text, total, cash, other, note, source, now))
        conn.executemany("INSERT INTO positions VALUES (?, ?, ?, ?, ?, ?, ?)", positions)
        conn.executemany("INSERT INTO cash_flows VALUES (NULL, ?, ?, ?, ?)", flows)
    return day_text


def history(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    snapshots = conn.execute("SELECT * FROM snapshots ORDER BY valuation_date").fetchall()
    if not snapshots:
        return []
    flows_by_snapshot: dict[str, list[sqlite3.Row]] = {}
    for flow in conn.execute("SELECT * FROM cash_flows ORDER BY flow_date, id"):
        flows_by_snapshot.setdefault(flow["valuation_date"], []).append(flow)
    opening = conn.execute("SELECT * FROM account_opening WHERE id = 1").fetchone()
    baseline = (int(opening["equity_cents"]) if opening else
                int(snapshots[0]["total_equity_cents"]))
    previous_date = (date.fromisoformat(opening["opening_date"]) if opening else
                     date.fromisoformat(snapshots[0]["valuation_date"]))
    previous_equity = baseline
    net_flows = 0
    nav = 1.0
    peak = 1.0
    rows: list[dict[str, Any]] = []
    for index, snapshot in enumerate(snapshots):
        day = date.fromisoformat(snapshot["valuation_date"])
        equity = int(snapshot["total_equity_cents"])
        flows = flows_by_snapshot.get(snapshot["valuation_date"], [])
        period_flow = sum(int(flow["amount_cents"]) for flow in flows)
        if index or opening:
            days = (day - previous_date).days
            if days == 0 and flows:
                raise ValueError("同日期初基准与快照之间有入金，无法确定资金进出时点")
            weighted = (sum(int(flow["amount_cents"]) *
                            (days - (date.fromisoformat(flow["flow_date"]) - previous_date).days)
                            / days for flow in flows) if days else 0)
            denominator = previous_equity + weighted
            if denominator <= 0:
                raise ValueError(f"{day} 的收益计算分母不为正，请检查出入金和账户市值")
            period_return = (equity - previous_equity - period_flow) / denominator
            if period_return < -1:
                raise ValueError(f"{day} 的区间收益小于 -100%，请核对数据")
            nav *= 1 + period_return
            net_flows += period_flow
        else:
            period_return = None
        peak = max(peak, nav)
        rows.append({
            "date": day.isoformat(),
            "total_equity_cny": _yuan(equity),
            "cash_cny": _yuan(int(snapshot["cash_cents"])),
            "cash_weight": int(snapshot["cash_cents"]) / equity if equity else 0.0,
            "other_assets_cny": _yuan(int(snapshot["other_assets_cents"])),
            "net_external_flows_cny": _yuan(net_flows),
            "net_invested_cny": _yuan(baseline + net_flows),
            "cumulative_pnl_cny": _yuan(equity - baseline - net_flows),
            "period_return": period_return,
            "time_weighted_nav": nav,
            "time_weighted_return": nav - 1,
            "observed_drawdown": nav / peak - 1,
        })
        previous_date, previous_equity = day, equity
    return rows


def latest_positions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    day = conn.execute("SELECT MAX(valuation_date) FROM snapshots").fetchone()[0]
    if day is None:
        return []
    rows = conn.execute("SELECT * FROM positions WHERE valuation_date = ? ORDER BY symbol",
                        (day,)).fetchall()
    equity = conn.execute("SELECT total_equity_cents FROM snapshots WHERE valuation_date = ?",
                          (day,)).fetchone()[0]
    return [{"date": day, "symbol": row["symbol"], "shares": row["shares"],
             "last_price": row["last_price"], "market_value_cny": _yuan(row["market_value_cents"]),
             "weight": row["market_value_cents"] / equity if equity else 0.0,
             "avg_cost": row["avg_cost"],
             "unrealized_pnl_cny": (_yuan(row["unrealized_pnl_cents"])
                                    if row["unrealized_pnl_cents"] is not None else None)}
            for row in rows]


def export_csv(conn: sqlite3.Connection, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "live_portfolio_history.csv"
    positions_path = output_dir / "live_portfolio_positions.csv"
    data = history(conn)
    with history_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "total_equity_cny", "cash_cny",
            "cash_weight", "other_assets_cny", "net_external_flows_cny",
            "net_invested_cny", "cumulative_pnl_cny",
            "period_return", "time_weighted_nav", "time_weighted_return", "observed_drawdown"])
        writer.writeheader()
        writer.writerows(data)
    with positions_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "symbol", "shares", "last_price", "market_value_cny", "weight",
                         "avg_cost", "unrealized_pnl_cny"])
        for row in conn.execute("SELECT p.*, s.total_equity_cents FROM positions p "
                                "JOIN snapshots s USING(valuation_date) "
                                "ORDER BY p.valuation_date, p.symbol"):
            writer.writerow([row["valuation_date"], row["symbol"], row["shares"],
                             row["last_price"], _yuan(row["market_value_cents"]),
                             row["market_value_cents"] / row["total_equity_cents"]
                             if row["total_equity_cents"] else 0.0,
                             row["avg_cost"], _yuan(row["unrealized_pnl_cents"])
                             if row["unrealized_pnl_cents"] is not None else ""])
    return history_path, positions_path
