"""宽基图使用同一起点，短缺价仅沿用旧价估值。"""

import unittest

import pandas as pd

from plot_benchmarks import BENCHMARKS, make_comparison


class BenchmarkPlotTests(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=6)
        self.equity = pd.DataFrame({"strategy_nav": [0.9995, 1.01, 1.02, 1.03, 1.04, 1.05]},
                                   index=dates)
        self.prices = {code: pd.Series([1.0, 1.01, 1.02, 1.03, 1.04, 1.05], index=dates)
                       for code in BENCHMARKS}

    def test_keeps_initial_strategy_cost_and_fills_one_missing_quote_from_past(self) -> None:
        missing_day = self.equity.index[2]
        self.prices["512100"] = self.prices["512100"].drop(missing_day)
        nav = make_comparison(self.equity, self.prices)
        self.assertAlmostEqual(nav.iloc[0]["策略"], 0.9995)
        self.assertAlmostEqual(nav.loc[missing_day, "中证1000"], 1.01)
        self.assertAlmostEqual(nav.iloc[3]["中证1000"], 1.03)

    def test_rejects_long_missing_quote_run(self) -> None:
        self.prices["512100"] = self.prices["512100"].iloc[:2]
        with self.assertRaises(ValueError):
            make_comparison(self.equity, self.prices)


if __name__ == "__main__":
    unittest.main()
