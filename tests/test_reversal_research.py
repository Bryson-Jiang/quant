"""研究叠加不得更改默认、引入未来数据或破坏周目标约束。"""

import unittest

import numpy as np
import pandas as pd

from etf_rotation.config import ETF_NAMES
from etf_rotation.risk_strategy import BOND, RiskConfig, risk_controlled_signals
from research_reversal import Trial, choose_case, feature_bundle, fixed_trials, targets


class ReversalResearchTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(73)
        days = pd.bdate_range("2024-01-01", periods=220)
        self.prices = pd.DataFrame(np.exp(np.cumsum(rng.normal(0.0006, 0.01, (220, 16)), axis=0)),
                                   index=days, columns=ETF_NAMES)
        self.cases = fixed_trials()

    def test_fixed_grid_and_no_overlay_reproduces_default(self):
        self.assertEqual(len(self.cases), 50)
        self.assertEqual(len({c.key for c in self.cases}), 50)
        frames, features, _ = feature_bundle(self.prices, self.cases)
        wrapped = targets(self.prices, self.cases, pd.Series("default", index=self.prices.index), frames, features)
        original = risk_controlled_signals(self.prices, RiskConfig.buffered_momentum())
        pd.testing.assert_frame_equal(wrapped, original)

    def test_all_mechanisms_are_causal_and_respect_caps(self):
        frames, features, _ = feature_bundle(self.prices, self.cases)
        changed = self.prices.copy()
        cutoff = self.prices.index[150]
        changed.loc[changed.index > cutoff, "512800"] *= 7
        cf, cx, _ = feature_bundle(changed, self.cases)
        chosen = [Trial("short_term_reversal", 5, 0.5), Trial("rsi", 14, 0.5),
                  Trial("residual_reversal", 5, 0.5), Trial("drawdown_rebound", 20, 0.5),
                  Trial("ma_zscore", 20, 0.5), Trial("crowding", 10, 0.5, "bank"),
                  Trial("crowding", 10, 1, "domestic"), Trial("unwind", 10, 0.5, "bank"),
                  Trial("unwind", 10, 1, "bank")]
        for case in chosen:
            with self.subTest(case=case.key):
                path = pd.Series(case.key, index=self.prices.index)
                original = targets(self.prices, self.cases, path, frames, features)
                perturbed = targets(changed, self.cases, path, cf, cx)
                pd.testing.assert_frame_equal(original.loc[:cutoff], perturbed.loc[:cutoff])
                risky = original.drop(columns=BOND)
                self.assertTrue(risky.le(0.30 + 1e-12).all().all())
                self.assertTrue(risky.sum(axis=1).le(0.70 + 1e-12).all())
                self.assertTrue(risky.gt(1e-10).sum(axis=1).le(3).all())
                self.assertTrue(original.ge(-1e-12).all().all())
                self.assertTrue(original.sum(axis=1).le(1 + 1e-12).all())

    def test_disabled_asset_cannot_enter(self):
        frames, features, _ = feature_bundle(self.prices, self.cases)
        result = targets(self.prices, self.cases, pd.Series("default", index=self.prices.index),
                         frames, features, removed="512800")
        self.assertTrue(result["512800"].eq(0).all())

    def test_parameter_selection_cannot_read_future_strategy_returns(self):
        days = pd.bdate_range("2022-01-03", "2025-12-31")
        base = pd.Series(0.0002 + 0.002 * np.sin(np.arange(len(days)) / 7), index=days)
        daily = pd.DataFrame({"default": base, "candidate": base + 0.0001})
        cutoff = pd.Timestamp("2024-12-31")
        selected = choose_case(daily, cutoff, [2022, 2023, 2024])
        self.assertEqual(selected, "candidate")
        changed = daily.copy()
        changed.loc[changed.index > cutoff, "candidate"] = -0.3
        self.assertEqual(selected, choose_case(changed, cutoff, [2022, 2023, 2024]))

    def test_classified_overlay_is_confined_to_declared_assets(self):
        bank = Trial("ma_zscore", 20, 0.5, "bank")
        domestic = Trial("ma_zscore", 20, 0.5, "domestic")
        cases = [Trial(), bank, domestic]
        _, feature, _ = feature_bundle(self.prices, cases)
        self.assertTrue(feature[bank.key].drop(columns="512800").eq(0).all().all())
        self.assertGreater(float(feature[bank.key]["512800"].abs().sum()), 0)
        for symbol in (BOND, "159941", "513520", "159985", "518880"):
            self.assertTrue(feature[domestic.key][symbol].eq(0).all())


if __name__ == "__main__":
    unittest.main()
