"""防守轮动的持仓上限、防守切换与历史不变性。"""

import unittest

import numpy as np
import pandas as pd

from etf_rotation.risk_strategy import RiskConfig, risk_controlled_signals


class RiskStrategyTests(unittest.TestCase):
    def test_four_position_limit_and_bond_remainder(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=180)
        x = np.arange(len(days))
        prices = pd.DataFrame({
            "510300": 1 + 0.001 * x + 0.01 * np.sin(x / 8),
            "511260": 1 + 0.0002 * x + 0.001 * np.sin(x / 9),
            "A": 1 + 0.002 * x + 0.02 * np.sin(x / 7),
            "B": 1 + 0.0015 * x + 0.01 * np.sin(x / 6),
            "C": 1 + 0.0012 * x + 0.02 * np.sin(x / 5),
            "D": 1 + 0.0011 * x + 0.01 * np.sin(x / 4),
        }, index=days)
        config = RiskConfig()
        signals = risk_controlled_signals(prices, config)
        self.assertTrue(((signals > 1e-8).sum(axis=1) <= 4).all())
        self.assertTrue((signals.sum(axis=1) <= 1 + 1e-9).all())
        self.assertEqual(config.max_risky_weight, 0.70)
        self.assertTrue((signals.drop(columns="511260").sum(axis=1) <= 0.70 + 1e-9).all())
        self.assertGreater(signals["511260"].max(), 0)

        altered = prices.copy()
        altered.loc[days[160]:, "A"] *= 3
        past = signals.loc[signals.index < days[160]]
        pd.testing.assert_frame_equal(
            past, risk_controlled_signals(altered, config).loc[past.index])

    def test_all_risky_assets_failing_filter_moves_to_bond(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=180)
        x = np.arange(len(days))
        prices = pd.DataFrame({
            "510300": 2 - 0.002 * x,
            "511260": 1 + 0.0003 * x,
            "A": 2 - 0.003 * x,
        }, index=days)
        signals = risk_controlled_signals(prices, RiskConfig())
        self.assertEqual(signals.iloc[-1]["511260"], 1.0)
        self.assertEqual(signals.iloc[-1]["A"], 0.0)


if __name__ == "__main__":
    unittest.main()
