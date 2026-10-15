"""周频主成分回归只能使用信号日已经观察到的收益。"""

import unittest

import numpy as np
import pandas as pd

from etf_rotation.predictive_factors import PCRSpec, weekly_pcr_forecast


class PredictiveFactorTests(unittest.TestCase):
    def test_future_prices_do_not_change_earlier_forecasts(self) -> None:
        dates = pd.bdate_range("2020-01-06", periods=5 * 150)
        weekly = np.arange(150)
        levels = {
            name: np.exp(np.cumsum(0.002 + 0.01 * np.sin(weekly / (5 + i))
                                   + 0.003 * np.cos(weekly / (9 + i))))
            for i, name in enumerate(("A", "B", "C"))
        }
        prices = pd.DataFrame({name: np.repeat(values, 5)
                               for name, values in levels.items()}, index=dates)
        spec = PCRSpec(train_weeks=52, components=2)
        original = weekly_pcr_forecast(prices, ("A", "B", "C"), spec)
        self.assertTrue(original.notna().all(axis=1).sum() > 40)
        cut = dates[5 * 125]
        changed = prices.copy()
        changed.loc[cut:, "A"] *= 1.4
        observed = weekly_pcr_forecast(changed, ("A", "B", "C"), spec)
        pd.testing.assert_frame_equal(original.loc[original.index < cut],
                                      observed.loc[observed.index < cut])


if __name__ == "__main__":
    unittest.main()
