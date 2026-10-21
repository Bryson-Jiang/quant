"""反转因子的方向、数据时点及市场残差定义。"""

import unittest

import numpy as np
import pandas as pd

from etf_rotation.factors import FactorSpec, compute_factor


class ReversalFactorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        days = pd.bdate_range("2022-01-03", periods=190)
        returns = rng.normal(0.0003, 0.01, (len(days), 3))
        returns[:, 1] += 0.5 * returns[:, 0]
        self.prices = pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=days,
                                   columns=["510300", "512800", "159941"])
        self.names = ("short_term_reversal", "rsi", "residual_reversal",
                      "drawdown_rebound", "relative_strength")

    def test_prefix_and_future_perturbation(self):
        changed = self.prices.copy()
        cutoff = self.prices.index[140]
        changed.loc[changed.index > cutoff, "512800"] *= 5
        for name in self.names:
            with self.subTest(name=name):
                spec = FactorSpec(name, 10)
                full = compute_factor(self.prices, spec)
                pd.testing.assert_frame_equal(full.loc[:cutoff], compute_factor(self.prices.loc[:cutoff], spec))
                pd.testing.assert_frame_equal(full.loc[:cutoff], compute_factor(changed, spec).loc[:cutoff])

    def test_skip_is_a_shift_of_observed_information(self):
        for name in self.names:
            with self.subTest(name=name):
                expected = compute_factor(self.prices, FactorSpec(name, 10)).shift(3)
                observed = compute_factor(self.prices, FactorSpec(name, 10, 3))
                pd.testing.assert_frame_equal(expected, observed)

    def test_rsi_and_bounce_directions(self):
        rsi = compute_factor(self.prices, FactorSpec("rsi", 7)).stack().dropna()
        self.assertTrue(rsi.between(0, 100).all())
        flat = self.prices * 0 + 100
        self.assertTrue(compute_factor(flat, FactorSpec("rsi", 7)).iloc[7:].eq(50).all().all())
        bounce = compute_factor(self.prices, FactorSpec("drawdown_rebound", 20))
        falling = self.prices.lt(self.prices.shift(3))
        self.assertTrue(bounce.where(falling).stack().dropna().eq(0).all())

    def test_reversal_is_negative_standardized_log_return(self):
        observed = compute_factor(self.prices, FactorSpec("short_term_reversal", 5))
        daily = np.log(self.prices / self.prices.shift(1)).rolling(60).std(ddof=1)
        expected = -np.log(self.prices / self.prices.shift(5)) / (daily * np.sqrt(5))
        pd.testing.assert_frame_equal(observed, expected)

    def test_missing_signal_quote_is_never_filled(self):
        prices = self.prices.copy()
        missing = prices.index[100]
        prices.at[missing, "512800"] = np.nan
        for name in self.names:
            for skip in (0, 3):
                with self.subTest(name=name, skip=skip):
                    result = compute_factor(prices, FactorSpec(name, 10, skip))
                    self.assertTrue(pd.isna(result.at[missing, "512800"]))
        result = compute_factor(prices, FactorSpec("short_term_reversal", 10))
        self.assertTrue(pd.isna(result.at[prices.index[101], "512800"]))

    def test_market_adjusted_factors_require_market_and_use_prior_beta(self):
        for name in ("relative_strength", "residual_reversal"):
            with self.assertRaises(ValueError):
                compute_factor(self.prices.drop(columns="510300"), FactorSpec(name, 5))
        p = self.prices
        lr = np.log(p / p.shift(1))
        market = lr["510300"]
        cov = lr.rolling(60).cov(market)
        mv = market.rolling(60).var(ddof=1)
        beta = cov.div(mv, axis=0).shift(5)
        residual_std = np.sqrt((lr.rolling(60).var(ddof=1) - cov.pow(2).div(mv, axis=0)).clip(lower=0)).shift(5)
        move = np.log(p / p.shift(5))
        expected = -(move - beta.mul(move["510300"], axis=0)) / (residual_std * np.sqrt(5)).where(residual_std > 1e-10)
        expected["510300"] = np.nan
        observed = compute_factor(p, FactorSpec("residual_reversal", 5))
        pd.testing.assert_frame_equal(observed, expected)


if __name__ == "__main__":
    unittest.main()
