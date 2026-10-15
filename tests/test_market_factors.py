"""Candidate market-state factors must not read future prices."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from etf_rotation.config import ETF_NAMES
from etf_rotation.factor_store import FactorStore
from etf_rotation.market_factors import regime_gates


class MarketFactorTests(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.bdate_range("2023-01-02", periods=330)
        x = np.arange(len(dates))
        prices = {symbol: 1.0 + (0.0004 + i * 0.00005) * x
                  + 0.01 * np.sin(x / (9 + i))
                  for i, symbol in enumerate(ETF_NAMES)}
        prices["511260"] = 1 + 0.0001 * x
        self.prices = pd.DataFrame(prices, index=dates)

    def test_future_change_cannot_rewrite_past_gates(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = FactorStore(Path(folder) / "factors.sqlite")
            original = regime_gates(self.prices, store)
            cut = self.prices.index[280]
            changed = self.prices.copy()
            changed.loc[cut:, "510300"] *= 1.25
            observed = regime_gates(changed, store)
            pd.testing.assert_frame_equal(original.loc[original.index < cut],
                                          observed.loc[observed.index < cut])

    def test_missing_component_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = FactorStore(Path(folder) / "factors.sqlite")
            with self.assertRaisesRegex(ValueError, "缺少 ETF"):
                regime_gates(self.prices.drop(columns="510300"), store)


if __name__ == "__main__":
    unittest.main()
