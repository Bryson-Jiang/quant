"""验证新增风险研究的方向、时序、仓位和预算路径。"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from etf_rotation.config import ETF_NAMES
from etf_rotation.factors import FactorSpec, compute_factor
from etf_rotation.risk_strategy import BOND, RiskConfig, risk_controlled_signals
from research_vol8 import Case, evaluate, feature_bundle, fixed_cases, overlay, tail_analysis, targets


class Vol8ResearchTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(315)
        days=pd.bdate_range("2023-01-02",periods=240)
        changes=rng.normal(0.0006,0.012,(len(days),len(ETF_NAMES)))
        self.prices=pd.DataFrame(np.exp(np.cumsum(changes,axis=0)),index=days,columns=ETF_NAMES)
        self.cases=fixed_cases()

    def test_fixed_grid_and_plain_default_equivalence(self):
        self.assertEqual(len(self.cases),35)
        self.assertEqual(len({c.key for c in self.cases}),35)
        f,r,s=feature_bundle(self.prices)
        actual=targets(self.prices,self.cases,pd.Series("default",index=self.prices.index),f,r,s)
        expected=risk_controlled_signals(self.prices,RiskConfig.buffered_momentum())
        pd.testing.assert_frame_equal(actual,expected)

    def test_downside_balance_has_correct_direction_bounds_and_flat_missing(self):
        days=pd.bdate_range("2024-01-01",periods=65)
        p=pd.DataFrame({"up":1.01**np.arange(65),"down":0.99**np.arange(65),"flat":1.0},index=days)
        actual=compute_factor(p,FactorSpec("downside_balance",20))
        self.assertAlmostEqual(actual.up.iloc[-1],1.0)
        self.assertAlmostEqual(actual.down.iloc[-1],-1.0)
        self.assertTrue(actual.flat.isna().all())
        delayed=compute_factor(p,FactorSpec("downside_balance",20,3))
        np.testing.assert_allclose(actual.shift(3),delayed,equal_nan=True)

    def test_all_new_factor_overlays_are_causal_and_no_bond_score(self):
        cutoff=self.prices.index[160]
        changed=self.prices.copy()
        changed.loc[changed.index>cutoff] *= np.linspace(2,9,len(ETF_NAMES))
        f,r,s=feature_bundle(self.prices)
        cf,cr,cs=feature_bundle(changed)
        for case in self.cases:
            with self.subTest(case=case.key):
                before=overlay(case,r,s)
                after=overlay(case,cr,cs)
                pd.testing.assert_frame_equal(before.loc[:cutoff],after.loc[:cutoff])
                self.assertTrue(before[BOND].eq(0).all())
        original=compute_factor(self.prices,FactorSpec("downside_balance",60))
        perturbed=compute_factor(changed,FactorSpec("downside_balance",60))
        pd.testing.assert_frame_equal(original.loc[:cutoff],perturbed.loc[:cutoff])
        missing=self.prices.copy()
        missing.iloc[-1,0]=np.nan
        self.assertTrue(pd.isna(compute_factor(missing,FactorSpec("downside_balance",60)).iloc[-1,0]))

    def test_varying_budget_preserves_selection_and_respects_caps(self):
        f,r,s=feature_bundle(self.prices)
        plain8=Case(0.08)
        choices=pd.Series("default",index=self.prices.index)
        choices.loc[choices.index>=choices.index[170]]=plain8.key
        actual=targets(self.prices,self.cases,choices,f,r,s)
        low=risk_controlled_signals(self.prices,RiskConfig.buffered_momentum())
        high=risk_controlled_signals(self.prices,RiskConfig.buffered_momentum(target_vol=0.08))
        for day in actual.index:
            expected=low.loc[day] if day<choices.index[170] else high.loc[day]
            pd.testing.assert_series_equal(actual.loc[day],expected)
        risky=actual.drop(columns=BOND)
        self.assertTrue(risky.le(0.30+1e-12).all().all())
        self.assertTrue(risky.sum(axis=1).le(0.70+1e-12).all())
        self.assertTrue(risky.gt(1e-10).sum(axis=1).le(3).all())
        self.assertTrue(actual.ge(0).all().all())
        self.assertTrue(actual.sum(axis=1).le(1+1e-12).all())

    def test_tail_attribution_matches_net_return_and_uses_old_holdings(self):
        f,r,s=feature_bundle(self.prices)
        names=["default",Case(0.08).key,Case(0.08,20,0.5).key,
               Case(0.08,extra="relative_strength",strength=0.25).key]
        signals={n:targets(self.prices,self.cases,pd.Series(n,index=self.prices.index),f,r,s) for n in names}
        daily=pd.DataFrame({n:evaluate(self.prices,t,self.prices.index[-1].date()).equity.daily_return
                            for n,t in signals.items()})
        with tempfile.TemporaryDirectory() as folder:
            with patch("research_vol8.OUT",Path(folder)):
                tail_analysis(self.prices,signals,daily)
            actual=pd.read_csv(Path(folder)/"tail_event.csv")
        self.assertTrue(pd.to_datetime(actual.prior_signal).lt(pd.to_datetime(actual.date)).all())
        for n,rows in actual.groupby("variant"):
            day=pd.Timestamp(rows.date.iloc[0])
            self.assertAlmostEqual(rows.verified_net_return.iloc[0],daily.loc[day,n],places=12)


if __name__=="__main__":
    unittest.main()
