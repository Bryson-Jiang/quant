"""独立OLS、量纲不变、历史截断及量能缓存验证。"""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from etf_rotation.extended_factors import ExtendedFactorSpec as Spec,compute_extended,low_ivol_utility
from etf_rotation.factor_store import FactorStore


class ExtendedFactorTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(20261109);n=360
        m=rng.normal(.0004,.01,n);g=rng.normal(.0002,.012,n);noise=rng.normal(0,.003,n)
        r=pd.DataFrame({'510300':m,'159915':m+g,'A':.0001+.8*m+.4*g+noise,
                         'B':.0002+.5*m-.2*g-noise},index=pd.bdate_range('2020-01-01',periods=n))
        self.p=10*(1+r).cumprod();self.v=pd.DataFrame(rng.uniform(100,300,r.shape),index=r.index,columns=r.columns)
        self.specs=[Spec('reversal'),Spec('high_vol_reversal'),Spec('abnormal_volume',window=5,slow=50),
                    Spec('volume_reversal',window=5,fast=5),Spec('ivol',window=60),
                    Spec('ivol',window=60,model='style'),Spec('macd_trend',fast=12,slow=26,signal=9),
                    Spec('macd_histogram',fast=12,slow=26,signal=9)]

    def test_all_inputs_prefix_and_future_invariant(self):
        cut=self.p.index[260];p=self.p.copy();v=self.v.copy()
        p.loc[p.index>cut]*=[.4,2,3,5];v.loc[v.index>cut]*=[2,3,4,8]
        for spec in self.specs:
            full=compute_extended(self.p,spec,self.v)
            prefix=compute_extended(self.p.loc[:cut],spec,self.v.loc[:cut])
            future=compute_extended(p,spec,v)
            pd.testing.assert_frame_equal(full.loc[:cut],prefix)
            pd.testing.assert_frame_equal(full.loc[:cut],future.loc[:cut])

    def test_ivol_agrees_with_direct_intercept_ols(self):
        r=self.p.pct_change(fill_method=None).iloc[-60:]
        for model in ('market','style'):
            x=np.column_stack([np.ones(60),r['510300']])
            if model=='style':x=np.column_stack([x,r['159915']-r['510300']])
            beta=np.linalg.lstsq(x,r.A,rcond=None)[0]
            expected=np.std(r.A.to_numpy()-x@beta,ddof=1)*np.sqrt(252)
            result=compute_extended(self.p,Spec('ivol',window=60,model=model))
            self.assertAlmostEqual(result.A.iloc[-1],expected,places=10)
            self.assertTrue(result['510300'].isna().all())
            if model=='style':self.assertTrue(result['159915'].isna().all())

    def test_activity_signal_is_separate_from_return_reversal(self):
        p=self.p.copy();v=self.v*0+100.
        p.iloc[-1,p.columns.get_loc('A')]=p.A.iloc[-6]*1.2
        p.iloc[-1,p.columns.get_loc('B')]=p.B.iloc[-6]*.8
        v.iloc[-5:]=400.
        premium=compute_extended(p,Spec('abnormal_volume',window=5,slow=50),v)
        reversal=compute_extended(p,Spec('volume_reversal',window=5,fast=5,slow=50),v)
        self.assertEqual(premium.A.iloc[-1],premium.B.iloc[-1])
        self.assertGreater(premium.A.iloc[-1],0)
        self.assertLess(reversal.A.iloc[-1],0)
        self.assertGreater(reversal.B.iloc[-1],0)

    def test_macd_and_reversal_are_invariant_to_price_units(self):
        p=self.p.mul([100,.01,10,.001])
        for spec in self.specs:
            np.testing.assert_allclose(compute_extended(self.p,spec,self.v),compute_extended(p,spec,self.v),
                                       atol=1e-7,rtol=1e-6,equal_nan=True)

    def test_missing_quotes_are_not_filled_into_macd_or_ivol(self):
        p=self.p.copy();p.iloc[-5,p.columns.get_loc('A')]=np.nan
        for name in ('ivol','macd_trend','high_vol_reversal'):
            spec=Spec(name,fast=12,slow=26) if name=='macd_trend' else Spec(name)
            result=compute_extended(p,spec,self.v)
            self.assertTrue(result.A.iloc[-5:].isna().all())

    def test_ivol_scope_requires_three_assets_and_leaves_others_neutral(self):
        raw=pd.DataFrame({'A':[.1,.1],'B':[.2,.2],'C':[.4,np.nan],'D':[.5,.5]})
        out=low_ivol_utility(raw,['A','B','C'])
        np.testing.assert_allclose(out.iloc[0,:3],[1,0,-1])
        self.assertTrue(out.D.isna().all())
        self.assertTrue(out.iloc[1].isna().all())

    def test_cache_detects_volume_change_when_prices_unchanged(self):
        spec=Spec('abnormal_volume',window=5,slow=50)
        with tempfile.TemporaryDirectory() as directory:
            store=FactorStore(Path(directory)/'f.sqlite')
            first=store.get_extended(self.p,spec,self.v)
            pd.testing.assert_frame_equal(first,store.get_extended(self.p,spec,self.v))
            volume=self.v.copy();volume.iloc[-1,0]*=10
            second=store.get_extended(self.p,spec,volume)
            self.assertNotEqual(first.iloc[-1,0],second.iloc[-1,0])
            pd.testing.assert_frame_equal(second,compute_extended(self.p,spec,volume))

    def test_fixed_experiment_grid(self):
        from research_factor_extensions import cases,features
        entries=cases()
        self.assertEqual(len(features()),40)
        self.assertEqual(len(entries),117)
        self.assertEqual(len({c.name for c in entries}),117)
        self.assertEqual(sum(c.family=='warmup_control' for c in entries),6)

    def test_midweek_cutoff_excludes_unexecuted_placeholder_signal(self):
        from research_factor_extensions import executable_signal_dates
        from etf_rotation.risk_strategy import weekly_signal_dates
        from research_overfitting_audit import simulate
        days=pd.bdate_range('2024-01-01',periods=20);cut=days[11]
        p=pd.DataFrame({'510300':np.linspace(1,1.2,len(days))},index=days)
        old=pd.DataFrame(.5,index=weekly_signal_dates(days[:12]),columns=p.columns)
        full=pd.DataFrame(.5,index=weekly_signal_dates(days),columns=p.columns)
        past=executable_signal_dates(old,days[:12])
        self.assertNotIn(cut,past)
        pd.testing.assert_frame_equal(old.loc[past],full.loc[past])
        a=simulate(p.loc[:cut],old,days[0]).series()
        b=simulate(p,full,days[0]).series().loc[:cut]
        pd.testing.assert_series_equal(a,b)


if __name__=='__main__':unittest.main()
