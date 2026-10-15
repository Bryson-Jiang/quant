"""Record actual ETF account valuations; does not place brokerage orders."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from decimal import Decimal
from pathlib import Path

from etf_rotation.portfolio_tracker import DEFAULT_DB, connect, export_csv, history, \
    latest_positions, record_snapshot, set_opening


def main() -> None:
    parser = argparse.ArgumentParser(description="实盘账户快照、入金及真实收益记录")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite 数据库路径")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="建立空数据库，不会虚构持仓或收益")
    opening = sub.add_parser("opening", help="录入真实的投资前账户总资产，作为收益起点")
    opening.add_argument("--date", required=True)
    opening.add_argument("--equity", required=True, help="当日投资前账户总资产，人民币元")
    opening.add_argument("--note", default="")
    opening.add_argument("--replace", action="store_true", help="录入首笔快照前修正基准")
    record = sub.add_parser("record", help="录入一个完整账户快照")
    record.add_argument("--file", type=Path, required=True, help="UTF-8 JSON 录入文件")
    record.add_argument("--replace", action="store_true", help="修正同日记录，旧内容留档")
    report = sub.add_parser("report", help="查看资金调整后的真实收益")
    report.add_argument("--export", action="store_true", help="同时导出 CSV")
    report.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    with connect(args.db) as conn:
        if args.command == "init":
            print(f"数据库已建立：{args.db}")
        elif args.command == "opening":
            set_opening(conn, args.date, args.equity, args.note, replace=args.replace)
            print(f"已设置投资前基准：{args.date}，{args.equity} 元")
        elif args.command == "record":
            with args.file.open(encoding="utf-8-sig") as handle:
                payload = json.load(handle, parse_float=Decimal)
            day = record_snapshot(conn, payload, replace=args.replace)
            print(f"已记录 {day} 的完整账户快照：{args.db}")
        else:
            rows = history(conn)
            if not rows:
                print(f"数据库已就绪，尚无实盘快照：{args.db}")
                return
            latest = rows[-1]
            print(f"记录区间：{rows[0]['date']} 至 {latest['date']}（{len(rows)} 次快照）")
            print(f"最新账户总资产：{latest['total_equity_cny']} 元；现金：{latest['cash_cny']} 元")
            print(f"收益起点以来净入金：{latest['net_external_flows_cny']} 元")
            print(f"累计净投入：{latest['net_invested_cny']} 元")
            print(f"收益起点以来累计盈亏：{latest['cumulative_pnl_cny']} 元")
            print(f"资金调整后累计收益率：{latest['time_weighted_return']:.2%}（有入金时为近似值）")
            print(f"已记录快照中的最大回撤：{min(row['observed_drawdown'] for row in rows):.2%}")
            if conn.execute("SELECT 1 FROM account_opening WHERE id = 1").fetchone() is None:
                print("尚未录入首次买入前资产：首笔快照暂作收益基准，首日交易盈亏未计入累计收益。")
            print("当前持仓：")
            for item in latest_positions(conn):
                unrealized = (f"；浮盈亏 {item['unrealized_pnl_cny']} 元"
                              if item["unrealized_pnl_cny"] is not None else "")
                print(f"  {item['symbol']} {item['shares']} 份；市值 {item['market_value_cny']} 元；"
                      f"仓位 {item['weight']:.2%}"
                      f"{unrealized}")
            if args.export:
                files = export_csv(conn, args.output_dir)
                print("已导出：" + "、".join(str(path) for path in files))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, json.JSONDecodeError, sqlite3.Error) as exc:
        print(f"录入失败：{exc}", file=sys.stderr)
        raise SystemExit(2) from None
