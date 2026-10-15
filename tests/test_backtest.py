"""验证所选策略使用的下一交易日成交、费用与缺价处理。"""

import unittest
from datetime import date

import pandas as pd

from etf_rotation.backtest import run_backtest

class BacktestTests(unittest.TestCase):
    def test_signal_executes_next_close_and_costs_both_sides(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=5)
        prices = pd.DataFrame({"510300": [1, 1, 1, 1, 1],
                               "A": [100, 100, 100, 110, 110],
                               "B": [100, 100, 100, 100, 100]}, index=days)
        signals = pd.DataFrame({"510300": [0.0, 0.0], "A": [1.0, 0.0], "B": [0.0, 1.0]},
                               index=[days[0], days[2]])
        result = run_backtest(prices, signals, date(2024, 1, 1), date(2024, 1, 5),
                              fee_bps=3, slippage_bps=2)
        nav = result.equity["strategy_nav"]
        self.assertAlmostEqual(nav.iloc[0], 1)
        self.assertAlmostEqual(nav.iloc[1], 0.9995)  # 首次买入单边成本
        self.assertAlmostEqual(nav.iloc[2], 0.9995)
        self.assertAlmostEqual(nav.iloc[3], 0.9995 * 1.1 * 0.999)  # 先获利，再卖 A 买 B
        self.assertEqual(result.trades["turnover"].tolist(), [1.0, 2.0])
        self.assertEqual(result.trades["execution_date"].tolist(), [days[1], days[3]])

    def test_missing_execution_quote_stays_in_cash(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=3)
        prices = pd.DataFrame({"510300": [1, 1, 1], "A": [1, float("nan"), 2]}, index=days)
        signals = pd.DataFrame({"A": [1.0]}, index=[days[0]])
        result = run_backtest(prices, signals, days[0].date(), days[-1].date())
        self.assertEqual(result.weights.at[days[1], "CASH"], 1.0)
        self.assertEqual(result.trades.at[0, "unavailable"], "A")
        self.assertEqual(result.trades.at[0, "status"], "skipped_no_quote")

    def test_missing_quote_cannot_sell_existing_position_at_stale_price(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=4)
        prices = pd.DataFrame({"510300": [1.0] * 4,
                               "A": [1.0, 1.0, float("nan"), 1.2],
                               "B": [1.0] * 4}, index=days)
        signals = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]},
                               index=[days[0], days[1]])
        result = run_backtest(prices, signals, days[0].date(), days[-1].date(),
                              fee_bps=0, slippage_bps=0)
        self.assertEqual(result.trades["status"].tolist(), ["executed", "skipped_no_quote"])
        self.assertEqual(result.weights.at[days[2], "A"], 1.0)
        self.assertEqual(result.weights.at[days[2], "B"], 0.0)
        self.assertAlmostEqual(result.equity.at[days[3], "strategy_nav"], 1.2)

    def test_rejects_signal_from_non_market_day(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=3)
        prices = pd.DataFrame({"510300": [1.0] * 3}, index=days)
        signals = pd.DataFrame({"510300": [1.0]}, index=[pd.Timestamp("2024-01-06")])
        with self.assertRaisesRegex(ValueError, "信号日期"):
            run_backtest(prices, signals, days[0].date(), days[-1].date())

    def test_weights_drift_between_rebalances(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=4)
        prices = pd.DataFrame({"510300": [1, 1, 1, 1],
                               "A": [1, 1, 2, 2], "B": [1, 1, 1, 1]}, index=days)
        signals = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[days[0]])
        result = run_backtest(prices, signals, days[0].date(), days[-1].date(),
                              fee_bps=0, slippage_bps=0)
        self.assertAlmostEqual(result.equity.at[days[2], "strategy_nav"], 1.5)
        self.assertAlmostEqual(result.weights.at[days[2], "A"], 2 / 3)
        self.assertAlmostEqual(result.weights.at[days[2], "B"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
