"""新默认40/20日策略须遵守仓位约束，且只使用已知历史。"""

import unittest

import numpy as np
import pandas as pd

from etf_rotation.config import ETF_NAMES
from etf_rotation.factors import FactorSpec
from etf_rotation.risk_strategy import BOND, RiskConfig, risk_controlled_signals, risk_score_components, weekly_signal_dates
from etf_rotation.strategy_state import RotationState, SelectionAnchor


class CurrentStrategyTests(unittest.TestCase):
    def test_bank_etf_is_in_default_universe(self) -> None:
        self.assertEqual(len(ETF_NAMES), 16)
        self.assertEqual(ETF_NAMES["512800"], "银行ETF（华宝）")

    def test_selected_default_is_eight_percent_ma20_buffer_mandatory_first(self) -> None:
        config = RiskConfig.current_default()
        self.assertEqual(config.target_vol, 0.08)
        self.assertEqual(config.ma_reversion_days, 20)
        self.assertEqual(config.ma_reversion_strength, 0.5)
        self.assertEqual(RiskConfig.buffered_momentum().target_vol, 0.07)
        self.assertEqual(RiskConfig.buffered_momentum().ma_reversion_strength, 0.0)
        self.assertEqual(config.rank_buffer_size, 4)
        self.assertEqual(config.mandatory_top_rank, 1)
        self.assertIsNone(RiskConfig.previous_default().rank_buffer_size)
        self.assertEqual(RiskConfig.previous_default().target_vol, 0.06)

    def setUp(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=180)
        x = np.arange(len(dates))
        series = {code: 1 + (0.0007 + 0.0001 * i) * x +
                  (0.005 + 0.001 * i) * np.sin(x / (5 + i))
                  for i, code in enumerate(ETF_NAMES)}
        series["511260"] = 1 + 0.0002 * x + 0.001 * np.sin(x / 11)
        self.prices = pd.DataFrame(series, index=dates)

    def test_default_respects_risk_limits_and_bond_exemption(self) -> None:
        current = risk_controlled_signals(self.prices, RiskConfig.current_default())
        self.assertTrue((current.drop(columns="511260").max(axis=1) <= 0.30 + 1e-9).all())
        self.assertTrue((current.drop(columns="511260").sum(axis=1) <= 0.70 + 1e-9).all())
        self.assertTrue((current.sum(axis=1) <= 1 + 1e-9).all())
        self.assertGreater(current["511260"].max(), 0.30)

    def test_future_price_change_does_not_modify_prior_signals(self) -> None:
        original = risk_controlled_signals(self.prices, RiskConfig.current_default())
        changed = self.prices.copy()
        changed.loc[changed.index[150]:, "159915"] *= 2
        observed = risk_controlled_signals(changed, RiskConfig.current_default())
        pd.testing.assert_frame_equal(original.loc[original.index < changed.index[150]],
                                      observed.loc[observed.index < changed.index[150]])

    def controlled_frames(self) -> dict:
        score = pd.DataFrame(0.1, index=self.prices.index, columns=self.prices.columns)
        score["510300"] = 0.9
        score["159941"] = 0.8
        score["159915"] = 0.7
        score["518880"] = 0.6
        return {FactorSpec("momentum", 20): score * 0.1,
                FactorSpec("momentum", 40): score * 0.1,
                FactorSpec("trend", 40): score * 0 + 0.1,
                FactorSpec("volatility", 60): score * 0 + 0.1,
                FactorSpec("momentum", 60): score * 0 + 0.01,
                FactorSpec("ma_zscore", 20): score * 0}

    def test_production_first_is_selected_and_eligible_fourth_old_is_kept(self) -> None:
        frames = self.controlled_frames()
        anchor = weekly_signal_dates(self.prices.index)[20]
        # 金价提高到第1名；旧目标中的第4名创业板可优先于新第3名纳指。
        for n in (20, 40):
            frames[FactorSpec("momentum", n)].loc[anchor + pd.Timedelta(days=1):, "518880"] = 0.1
        target = risk_controlled_signals(self.prices, RiskConfig.current_default(), factor_frames=frames,
                    initial_risky_selection=["510300", "159915", "159980"], initial_selection_date=anchor)
        row = target.loc[target.index > anchor].iloc[0]
        self.assertGreater(row["518880"], 0)
        self.assertGreater(row["159915"], 0)
        self.assertEqual(row["159941"], 0)

    def test_production_buffer_does_not_keep_ineligible_old_position(self) -> None:
        anchor = weekly_signal_dates(self.prices.index)[20]
        for spec in (FactorSpec("momentum", 40), FactorSpec("trend", 40)):
            with self.subTest(spec=spec):
                frames = self.controlled_frames()
                frames[spec].loc[anchor + pd.Timedelta(days=1):, "159915"] = -0.01
                target = risk_controlled_signals(self.prices, RiskConfig.current_default(), factor_frames=frames,
                    initial_risky_selection=["510300", "159941", "159915"], initial_selection_date=anchor)
                self.assertTrue(target.loc[target.index > anchor, "159915"].eq(0).all())

    def test_transition_state_is_causal_and_replay_is_stable(self) -> None:
        anchor = weekly_signal_dates(self.prices.index)[20]
        config = RiskConfig.current_default()
        plain = risk_controlled_signals(self.prices, config)
        kwargs = {"initial_risky_selection": ["159941", "159915", "518880"],
                  "initial_selection_date": anchor}
        seeded = risk_controlled_signals(self.prices, config, **kwargs)
        pd.testing.assert_frame_equal(plain.loc[:anchor], seeded.loc[:anchor])
        cutoff = weekly_signal_dates(self.prices.index)[25]
        prefix = risk_controlled_signals(self.prices.loc[:cutoff], config, **kwargs)
        pd.testing.assert_frame_equal(seeded.loc[:cutoff], prefix)
        pd.testing.assert_frame_equal(seeded, risk_controlled_signals(self.prices, config, **kwargs))
        changed = self.prices.copy()
        changed.loc[changed.index > cutoff, "518880"] *= 7
        perturbed = risk_controlled_signals(changed, config, **kwargs)
        pd.testing.assert_frame_equal(seeded.loc[:cutoff], perturbed.loc[:cutoff])

    def test_state_rejects_missing_anchor_history(self) -> None:
        anchor = weekly_signal_dates(self.prices.index)[20]
        state = RotationState(anchor.date(), weekly_signal_dates(self.prices.index)[21].date(), ("159941",))
        self.assertEqual(state.signal_kwargs(self.prices.loc[:anchor]), {})
        with self.assertRaises(ValueError):
            state.signal_kwargs(self.prices.loc[self.prices.index > anchor])
        with self.assertRaises(ValueError):
            risk_controlled_signals(self.prices.loc[self.prices.index > anchor], RiskConfig.current_default(),
                    initial_risky_selection=["159941"], initial_selection_date=anchor)

    def test_current_default_equals_frozen_research_candidate(self) -> None:
        from research_vol8 import Case, feature_bundle, targets
        cases = [Case(), Case(0.08, 20, 0.5)]
        frames, raw, score = feature_bundle(self.prices)
        expected = targets(self.prices, cases, pd.Series(cases[1].key, index=self.prices.index), frames, raw, score)
        actual = risk_controlled_signals(self.prices, RiskConfig.current_default())
        pd.testing.assert_frame_equal(actual, expected, atol=1e-12, rtol=1e-12)

    def test_ma_adjustment_changes_ranking_and_excludes_bond(self) -> None:
        frames = self.controlled_frames()
        z = frames[FactorSpec("ma_zscore", 20)].copy()
        z["510300"] = 3.0
        z["159941"] = -3.0
        z[BOND] = -3.0
        result = risk_score_components(frames[FactorSpec("momentum", 20)],
                 frames[FactorSpec("momentum", 40)], frames[FactorSpec("volatility", 60)], RiskConfig.current_default(), z)
        self.assertTrue(result["base_score"]["510300"].gt(result["base_score"]["159941"]).all())
        self.assertTrue(result["score"]["159941"].gt(result["score"]["510300"]).all())
        self.assertTrue(result["ma_adjustment"][BOND].eq(0).all())

    def test_multiple_adoption_anchors_preserve_prior_history(self) -> None:
        dates = weekly_signal_dates(self.prices.index)
        old = SelectionAnchor(dates[10].date(), dates[11].date(), ("159941", "159915"))
        state = RotationState(dates[20].date(), dates[21].date(), ("518880", "159980"), (old,))
        config = RiskConfig.current_default()
        first = risk_controlled_signals(self.prices, config,
                        initial_risky_selection=list(old.initial_risky_selection), initial_selection_date=dates[10])
        full = risk_controlled_signals(self.prices, config, **state.signal_kwargs(self.prices))
        pd.testing.assert_frame_equal(first.loc[:dates[20]], full.loc[:dates[20]])
        prefix_prices = self.prices.loc[:dates[18]]
        prefix = risk_controlled_signals(prefix_prices, config, **state.signal_kwargs(prefix_prices))
        pd.testing.assert_frame_equal(full.loc[:dates[18]], prefix)
        changed = self.prices.copy()
        changed.loc[changed.index > dates[25], "159941"] *= 5
        perturbed = risk_controlled_signals(changed, config, **state.signal_kwargs(changed))
        pd.testing.assert_frame_equal(full.loc[:dates[25]], perturbed.loc[:dates[25]])


if __name__ == "__main__":
    unittest.main()
