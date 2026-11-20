"""Wilder种子/缺价、防未来信息、因子缓存与RSI研究选择口径。"""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from etf_rotation.factor_store import FactorStore
from etf_rotation.factors import FactorSpec, compute_factor
import research_rsi as research


class RSIResearchTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(20261120)
        self.prices=pd.DataFrame(np.exp(np.cumsum(rng.normal(.0005,.01,(400,4)),axis=0)),
             index=pd.bdate_range('2020-01-01',periods=400),columns=['510300','159915','159941','511260'])

    def test_wilder_sma_seed_and_following_update_manual(self):
        p=pd.DataFrame({'A':[10.,11.,10.,12.,11.]},index=pd.bdate_range('2020-01-01',periods=5))
        r=compute_factor(p,FactorSpec('rsi_wilder',3)).A
        self.assertTrue(r.iloc[:3].isna().all())
        self.assertAlmostEqual(r.iloc[3],75.)
        self.assertAlmostEqual(r.iloc[4],100*(2/3)/(2/3+5/9))

    def test_flat_up_down_and_warmup(self):
        d=pd.bdate_range('2020-01-01',periods=30)
        p=pd.DataFrame({'flat':1.,'up':np.arange(30)+1.,'down':40.-np.arange(30)},index=d)
        r=compute_factor(p,FactorSpec('rsi_wilder',14))
        self.assertTrue(r.iloc[:14].isna().all().all())
        np.testing.assert_allclose(r.iloc[14:].to_numpy(),np.tile([50.,100.,0.],(16,1)))

    def test_missing_quote_restarts_complete_warmup(self):
        p=self.prices.iloc[:60].copy();p.iloc[30,0]=np.nan
        r=compute_factor(p,FactorSpec('rsi_wilder',7))
        self.assertTrue(r.iloc[30:38,0].isna().all())
        self.assertTrue(np.isfinite(r.iloc[38,0]))
        truncated=compute_factor(p.iloc[31:],FactorSpec('rsi_wilder',7))
        pd.testing.assert_series_equal(r.iloc[38:,0],truncated.iloc[7:,0])

    def test_factors_and_score_utilities_are_prefix_and_future_invariant(self):
        cut=self.prices.index[280];changed=self.prices.copy();changed.loc[changed.index>cut]*=[3,.3,1.8,1.1]
        for method in ('rsi','rsi_wilder'):
            spec=FactorSpec(method,14)
            full=compute_factor(self.prices,spec);prefix=compute_factor(self.prices.loc[:cut],spec)
            future=compute_factor(changed,spec)
            pd.testing.assert_frame_equal(full.loc[:cut],prefix)
            pd.testing.assert_frame_equal(full.loc[:cut],future.loc[:cut])
            for mode in research.MODES:
                pd.testing.assert_frame_equal(research.rsi_utility(full,mode).loc[:cut],research.rsi_utility(prefix,mode))

    def test_price_units_and_skip(self):
        for method in ('rsi','rsi_wilder'):
            spec=FactorSpec(method,14)
            np.testing.assert_allclose(compute_factor(self.prices,spec),compute_factor(self.prices.mul([100,.1,7,2]),spec),rtol=1e-10,atol=1e-10,equal_nan=True)
            pd.testing.assert_frame_equal(compute_factor(self.prices,FactorSpec(method,14,2)),compute_factor(self.prices,spec).shift(2))

    def test_sma_and_wilder_cache_keys_and_price_invalidation(self):
        with tempfile.TemporaryDirectory() as folder:
            store=FactorStore(Path(folder)/'f.sqlite')
            simple=store.get(self.prices,FactorSpec('rsi',14))
            wilder=store.get(self.prices,FactorSpec('rsi_wilder',14))
            self.assertGreater(np.nanmax(np.abs(simple-wilder)),1.)
            pd.testing.assert_frame_equal(wilder,store.get(self.prices,FactorSpec('rsi_wilder',14)))
            changed=self.prices.copy();changed.iloc[-1,0]*=1.1
            new=store.get(changed,FactorSpec('rsi_wilder',14))
            self.assertNotEqual(new.iloc[-1,0],wilder.iloc[-1,0])
            pd.testing.assert_frame_equal(new,compute_factor(changed,FactorSpec('rsi_wilder',14)))

    def test_fixed_grid_and_neutral_default_score(self):
        entries=research.cases()
        self.assertEqual(len(entries),84)
        self.assertEqual(len({c.name for c in entries}),84)
        comp=research.components(self.prices);frames=research.feature_frames(self.prices)
        pd.testing.assert_frame_equal(research.case_score(comp,frames,entries[0]),comp['score'])
        c=next(c for c in entries if c.name=='wilder14_trend_a0.25')
        score=research.case_score(comp,frames,c)
        pd.testing.assert_series_equal(score['511260'],comp['score']['511260'])

    def test_utility_definitions_do_not_confuse_trend_with_cooling(self):
        raw=pd.DataFrame({'A':[20.,50.,70.,80.,100.,np.nan]})
        np.testing.assert_allclose(research.rsi_utility(raw,'trend').A,[-1,0,.8,1,1,np.nan],equal_nan=True)
        np.testing.assert_allclose(research.rsi_utility(raw,'cooling').A,[0,0,0,-1/3,-1,np.nan],equal_nan=True)
        pd.testing.assert_frame_equal(research.rsi_utility(raw,'reversal'),-research.rsi_utility(raw,'trend'))

    def test_predictive_regression_excludes_not_yet_matured_training_labels(self):
        rng=np.random.default_rng(5)
        def sample(n,day,end):
            b=rng.normal(size=n);x=rng.normal(size=n)
            return pd.DataFrame({'date':pd.Timestamp(day),'label_end':pd.Timestamp(end),'base':b,'x':x,'y':.01*b+.02*x+rng.normal(0,.001,n)})
        train=sample(250,'2018-06-01','2018-06-30');test=sample(30,'2022-06-01','2022-06-30')
        future=sample(30,'2018-07-01','2024-06-30');future['y']=1e9
        a=research.predictive_metrics(pd.concat([train,test],ignore_index=True))
        b=research.predictive_metrics(pd.concat([train,test,future],ignore_index=True))
        self.assertEqual(a,b)
        self.assertEqual(a['oos_observations'],30)
        self.assertGreater(a['incremental_predictive_r2'],.9)

    def test_yearly_choice_cannot_see_future_returns(self):
        days=pd.bdate_range('2018-01-01','2026-11-06')
        rng=np.random.default_rng(2)
        p=pd.DataFrame(np.exp(np.cumsum(rng.normal(.0003,.005,(len(days),4)),axis=0)),index=days,columns=self.prices.columns)
        comp=research.components(p);frames=research.feature_frames(p)
        entries=[research.Case('default'),research.Case('candidate','wilder',14,'trend',.25)]
        daily=pd.DataFrame({'default':rng.normal(.0003,.005,len(days)),'candidate':rng.normal(.0006,.005,len(days))},index=days)
        a,old=research.walk_score(p,comp,frames,entries,daily)
        future=daily.copy();future.loc['2025':,'candidate']=-.05
        b,new=research.walk_score(p,comp,frames,entries,future)
        self.assertEqual(old[:3],new[:3])
        pd.testing.assert_frame_equal(a.loc[:'2024'],b.loc[:'2024'])


if __name__=='__main__':unittest.main()
