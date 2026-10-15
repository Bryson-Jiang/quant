"""研究规则的时点、缓冲和组合风险约束。"""

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from etf_rotation.factors import FactorSpec
from etf_rotation.risk_strategy import BOND, RiskConfig
from research_strategy_optimization import (factor_frames, portfolio_vol, priority_buffer_selection,
                                           research_signals, transition_target)


class ResearchOptimizationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        dates = pd.bdate_range("2021-01-01", periods=360)
        symbols = ["510300", BOND, "159941", "159915", "518880"]
        returns = rng.normal(0.0008, 0.008, (len(dates), len(symbols)))
        returns[:, 1] = rng.normal(0.0001, 0.001, len(dates))
        self.prices = pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=dates, columns=symbols)

    def test_future_prices_do_not_change_past_targets(self):
        cutoff = self.prices.index[240]
        changed = self.prices.copy()
        changed.loc[changed.index > cutoff] *= np.linspace(1, 6, len(changed.loc[changed.index > cutoff]))[:, None]
        for mode in ("base", "ensemble", "correlation", "rank_buffer", "rank_buffer_top1",
                     "rank_buffer_top2", "fast_vol", "full_vol"):
            with self.subTest(mode=mode):
                original = research_signals(self.prices, mode).loc[:cutoff]
                alternative = research_signals(changed, mode).loc[:cutoff]
                pd.testing.assert_frame_equal(original, alternative)

    def test_rank_buffer_keeps_previous_target_at_rank_four(self):
        frames = factor_frames(self.prices)
        scores = pd.DataFrame({"510300": 0.9, BOND: 0.0, "159941": 0.8,
                               "159915": 0.7, "518880": 0.6}, index=self.prices.index)
        scores.loc[self.prices.index[200]:, "518880"] = 0.75
        frames[FactorSpec("volatility", 60)] = scores * 0 + 0.1
        frames[FactorSpec("trend", 40)] = scores * 0 + 0.1
        frames[FactorSpec("momentum", 20)] = scores * 0.1
        frames[FactorSpec("momentum", 40)] = scores * 0.1
        frames[FactorSpec("momentum", 60)] = scores * 0 + 0.01
        baseline = research_signals(self.prices, frames=frames)
        buffered = research_signals(self.prices, "rank_buffer", frames=frames)
        day = buffered.index[buffered.index >= self.prices.index[205]][0]
        self.assertGreater(baseline.at[day, "518880"], 0)
        self.assertEqual(baseline.at[day, "159915"], 0)
        self.assertGreater(buffered.at[day, "159915"], 0)
        self.assertEqual(buffered.at[day, "518880"], 0)

    def test_priority_first_displaces_worst_old_position(self):
        ranked = pd.Index(["new_first", "old_second", "old_third", "old_fourth", "other"])
        previous = ["old_fourth", "old_third", "old_second"]
        selected = priority_buffer_selection(ranked, previous, mandatory_top=1)
        self.assertEqual(selected, ["new_first", "old_second", "old_third"])

    def test_priority_first_can_still_block_second_while_top_two_cannot(self):
        ranked = pd.Index(["old_first", "new_second", "old_third", "old_fourth"])
        previous = ["old_fourth", "old_third", "old_first"]
        first = priority_buffer_selection(ranked, previous, mandatory_top=1)
        two = priority_buffer_selection(ranked, previous, mandatory_top=2)
        self.assertEqual(first, ["old_first", "old_third", "old_fourth"])
        self.assertEqual(two, ["old_first", "new_second", "old_third"])

    def test_priority_buffer_handles_empty_and_small_eligible_pool(self):
        self.assertEqual(priority_buffer_selection(pd.Index([]), ["old"], mandatory_top=1), [])
        self.assertEqual(priority_buffer_selection(pd.Index(["only"]), [], mandatory_top=2), ["only"])

    def test_full_portfolio_budget_includes_bond_and_cash(self):
        prices = self.prices.copy()
        prices[BOND] = 100 * np.cumprod(1 + 0.015 * np.sin(np.arange(len(prices)) / 3) + 0.0001)
        target = research_signals(prices, "full_vol")
        returns = prices.pct_change(fill_method=None)
        reduced = False
        for day, row in target.iterrows():
            self.assertLessEqual(float(row.sum()), 1 + 1e-10)
            self.assertTrue(row.ge(0).all())
            weights = row.loc[row.gt(0)]
            if weights.empty:
                continue
            for window, minimum in ((20, 18), (60, 40)):
                sigma = portfolio_vol(returns.loc[:day].tail(window), weights, minimum=minimum)
                if sigma is not None:
                    self.assertLessEqual(sigma, RiskConfig.previous_default().target_vol + 1e-10)
            reduced |= float(row.sum()) < 0.999 and row[BOND] > 0
        self.assertTrue(reduced)

    def test_rank_buffer_cannot_override_momentum_or_trend_failure(self):
        scores = pd.DataFrame({"510300": 0.9, BOND: 0.0, "159941": 0.8,
                               "159915": 0.7, "518880": 0.6}, index=self.prices.index)
        scores.loc[self.prices.index[200]:, "518880"] = 0.75
        for failed_spec in (FactorSpec("momentum", 40), FactorSpec("trend", 40)):
            with self.subTest(failed_spec=failed_spec):
                frames = factor_frames(self.prices)
                frames[FactorSpec("volatility", 60)] = scores * 0 + 0.1
                frames[FactorSpec("trend", 40)] = scores * 0 + 0.1
                frames[FactorSpec("momentum", 20)] = scores * 0.1
                frames[FactorSpec("momentum", 40)] = scores * 0.1
                frames[FactorSpec("momentum", 60)] = scores * 0 + 0.01
                frames[failed_spec].loc[self.prices.index[200]:, "159915"] = -0.01
                target = research_signals(self.prices, "rank_buffer", frames=frames)
                before = target.loc[target.index < self.prices.index[200], "159915"]
                after = target.loc[target.index >= self.prices.index[200], "159915"]
                self.assertTrue(before.gt(0).all())
                self.assertTrue(after.eq(0).all())

    def test_seven_percent_buffer_is_causal_and_respects_caps(self):
        config = replace(RiskConfig.previous_default(), target_vol=0.07)
        target = research_signals(self.prices, "rank_buffer", config=config)
        risky = target.drop(columns=BOND)
        self.assertTrue(risky.le(0.30 + 1e-12).all().all())
        self.assertTrue(risky.sum(axis=1).le(0.70 + 1e-12).all())
        self.assertTrue(risky.gt(1e-12).sum(axis=1).le(3).all())
        cutoff = self.prices.index[240]
        modified = self.prices.copy()
        modified.loc[modified.index > cutoff] *= 5
        repeated = research_signals(modified, "rank_buffer", config=config)
        pd.testing.assert_frame_equal(target.loc[:cutoff], repeated.loc[:cutoff])

    def test_transition_preserves_prior_rank_four_only_when_eligible(self):
        frames = factor_frames(self.prices)
        score = pd.DataFrame({"510300": 0.9, BOND: 0.0, "159941": 0.8,
                              "159915": 0.7, "518880": 0.75}, index=self.prices.index)
        frames[FactorSpec("volatility", 60)] = score * 0 + 0.1
        frames[FactorSpec("trend", 40)] = score * 0 + 0.1
        frames[FactorSpec("momentum", 20)] = score * 0.1
        frames[FactorSpec("momentum", 40)] = score * 0.1
        frames[FactorSpec("momentum", 60)] = score * 0 + 0.01
        config = replace(RiskConfig.previous_default(), target_vol=0.07)
        retained = transition_target(self.prices, config, ["159915"], frames=frames)
        self.assertGreater(retained["159915"], 0)
        frames[FactorSpec("momentum", 40)].loc[self.prices.index[-1], "159915"] = -0.01
        invalid = transition_target(self.prices, config, ["159915"], frames=frames)
        self.assertEqual(invalid["159915"], 0)


if __name__ == "__main__":
    unittest.main()
