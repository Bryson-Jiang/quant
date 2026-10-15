"""因子只读历史数据，缓存按输入内容更新。"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from etf_rotation.factor_store import FactorStore
from etf_rotation.factors import FactorSpec, compute_factor


class FactorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.days = pd.bdate_range("2024-01-01", periods=15)
        self.prices = pd.DataFrame({"A": [1 + i / 20 for i in range(15)],
                                    "B": [2 + i / 30 for i in range(15)]}, index=self.days)

    def test_all_factors_are_prefix_invariant(self) -> None:
        specs = [FactorSpec("momentum", 3, 1), FactorSpec("trend", 3),
                 FactorSpec("volatility", 3), FactorSpec("downside_volatility", 3),
                 FactorSpec("breakout", 3), FactorSpec("risk_adjusted_momentum", 3, 1),
                 FactorSpec("trend_efficiency", 3), FactorSpec("volatility_ratio", 3),
                 FactorSpec("ma_zscore", 3), FactorSpec("rolling_drawdown", 3),
                 FactorSpec("rolling_runup", 3),
                 FactorSpec("ma_persistence_5_10", 3),
                 FactorSpec("ma_persistence_10_30", 3)]
        altered = self.prices.copy()
        altered.loc[self.days[10]:, "A"] *= 2
        for spec in specs:
            with self.subTest(spec=spec):
                pd.testing.assert_frame_equal(compute_factor(self.prices, spec).iloc[:10],
                                              compute_factor(altered, spec).iloc[:10])

    def test_store_roundtrip_and_invalidation(self) -> None:
        spec = FactorSpec("momentum", 3, 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            store = FactorStore(Path(temp_dir) / "factors.sqlite")
            first = store.get(self.prices, spec)
            pd.testing.assert_frame_equal(first, store.get(self.prices, spec), check_freq=False)
            changed = self.prices.copy()
            changed.at[self.days[8], "A"] = 4.0
            new = store.get(changed, spec)
            self.assertNotEqual(first.at[self.days[9], "A"], new.at[self.days[9], "A"])
            pd.testing.assert_frame_equal(new, store.get(changed, spec), check_freq=False)

    def test_ma_persistence_requires_complete_history_and_reports_direction(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=55)
        rising = pd.Series(range(1, 56), index=days, dtype=float)
        prices = pd.DataFrame({"A": rising, "B": rising.iloc[::-1].to_numpy()}, index=days)
        factor = compute_factor(prices, FactorSpec("ma_persistence_10_30", 20))
        self.assertTrue(factor.iloc[:48].isna().all().all())
        self.assertEqual(factor.iloc[48]["A"], 1.0)
        self.assertEqual(factor.iloc[48]["B"], -1.0)

    def test_rolling_drawdown_and_runup_use_only_prior_window(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=5)
        prices = pd.DataFrame({"A": [10.0, 12.0, 11.0, 13.0, 9.0]}, index=days)
        drawdown = compute_factor(prices, FactorSpec("rolling_drawdown", 3))
        runup = compute_factor(prices, FactorSpec("rolling_runup", 3))
        self.assertAlmostEqual(drawdown.iloc[-1, 0], 9 / 13 - 1)
        self.assertAlmostEqual(runup.iloc[-2, 0], 13 / 11 - 1)


if __name__ == "__main__":
    unittest.main()
