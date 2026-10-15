"""Cash injections must never appear as strategy profit."""

import json
import tempfile
import unittest
from pathlib import Path

from etf_rotation.portfolio_tracker import (connect, export_csv, history,
    record_snapshot, set_opening)


class PortfolioTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.conn = connect(Path(self.temp.name) / "account.sqlite")

    def tearDown(self) -> None:
        self.conn.close()
        self.temp.cleanup()

    @staticmethod
    def snapshot(day: str, equity: str, flows=None):
        return {"date": day, "total_equity": equity, "cash": equity,
                "positions": [], "cash_flows": [] if flows is None else flows}

    def test_deposit_is_excluded_from_profit_and_daily_weighted_return(self) -> None:
        record_snapshot(self.conn, self.snapshot("2026-09-28", "60000"))
        record_snapshot(self.conn, self.snapshot("2026-10-05", "81000", [
            {"date": "2026-10-01", "amount": "20000", "note": "追加本金"}]))
        rows = history(self.conn)
        self.assertEqual(rows[-1]["net_external_flows_cny"], "20000.00")
        self.assertEqual(rows[-1]["cumulative_pnl_cny"], "1000.00")
        self.assertAlmostEqual(rows[-1]["period_return"], 1000 / (60000 + 20000 * 4 / 7))

    def test_opening_equity_includes_first_trade_costs(self) -> None:
        set_opening(self.conn, "2026-09-28", "60000")
        record_snapshot(self.conn, self.snapshot("2026-09-28", "59985"))
        row = history(self.conn)[0]
        self.assertEqual(row["cumulative_pnl_cny"], "-15.00")
        self.assertAlmostEqual(row["time_weighted_return"], -15 / 60000)

    def test_opening_correction_is_explicit_and_audited(self) -> None:
        set_opening(self.conn, "2026-09-28", "60000")
        record_snapshot(self.conn, self.snapshot("2026-09-28", "59985"))
        with self.assertRaisesRegex(ValueError, "--replace"):
            set_opening(self.conn, "2026-09-28", "60010")
        set_opening(self.conn, "2026-09-28", "60010", replace=True)
        self.assertEqual(history(self.conn)[0]["cumulative_pnl_cny"], "-25.00")
        previous = self.conn.execute("SELECT old_equity_cents FROM opening_revisions").fetchone()[0]
        self.assertEqual(previous, 6000000)

    def test_complete_position_reconciliation_and_missing_flow_rejection(self) -> None:
        record_snapshot(self.conn, {
            "date": "2026-09-28", "total_equity": "60000", "cash": "19540.50",
            "positions": [{"symbol": "511260", "shares": 300,
                           "last_price": "134.865", "market_value": "40459.50"}],
            "cash_flows": [],
        })
        with self.assertRaisesRegex(ValueError, "cash_flows"):
            record_snapshot(self.conn, {"date": "2026-10-05", "total_equity": "60000",
                                        "cash": "60000", "positions": []})
        with self.assertRaisesRegex(ValueError, "相差超过"):
            record_snapshot(self.conn, {"date": "2026-10-05", "total_equity": "61000",
                                        "cash": "60000", "positions": [], "cash_flows": []})

    def test_duplicate_requires_explicit_replace_and_retains_prior_version(self) -> None:
        original = self.snapshot("2026-09-28", "60000")
        record_snapshot(self.conn, original)
        with self.assertRaisesRegex(ValueError, "已有记录"):
            record_snapshot(self.conn, original)
        corrected = self.snapshot("2026-09-28", "59990")
        record_snapshot(self.conn, corrected, replace=True)
        self.assertEqual(history(self.conn)[0]["total_equity_cny"], "59990.00")
        old = self.conn.execute("SELECT old_source_json FROM snapshot_revisions").fetchone()[0]
        self.assertEqual(json.loads(old)["total_equity"], "60000")

    def test_flow_must_fall_between_snapshots(self) -> None:
        record_snapshot(self.conn, self.snapshot("2026-09-28", "60000"))
        with self.assertRaisesRegex(ValueError, "资金进出日期"):
            record_snapshot(self.conn, self.snapshot("2026-10-05", "80000", [
                {"date": "2026-09-28", "amount": "20000"}]))

    def test_export_has_history_and_holdings_headers(self) -> None:
        record_snapshot(self.conn, self.snapshot("2026-09-28", "60000"))
        output = Path(self.temp.name) / "output"
        files = export_csv(self.conn, output)
        self.assertTrue(all(path.exists() for path in files))
        self.assertIn("cumulative_pnl_cny", files[0].read_text(encoding="utf-8-sig"))

if __name__ == "__main__":
    unittest.main()
