"""RSRS必须与直接OLS一致，不能读未来或把缺报价补成可交易因子。"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from etf_rotation.factors import RSRSSpec,compute_rsrs
from etf_rotation.factor_store import FactorStore
from etf_rotation.ohlc_data import adjusted_high_low


class RSRSTests(unittest.TestCase):
    def setUp(self):
        days=pd.bdate_range('2020-01-01',periods=150)
        x=np.arange(150)
        self.low=pd.DataFrame({'A':2+.004*x+.07*np.sin(x/7),
                               'B':4+.008*x+.12*np.cos(x/9)},index=days)
        self.high=self.low+pd.DataFrame({'A':.05+.03*np.sin(x/5),
                                        'B':.07+.04*np.cos(x/6)},index=days)
        self.spec=RSRSSpec(10,20)

    def test_regression_and_zscore_match_direct_ols(self):
        actual=compute_rsrs(self.high,self.low,self.spec)
        slopes=[];fits=[]
        for i in range(9,len(self.low)):
            x=self.low.A.iloc[i-9:i+1].to_numpy();y=self.high.A.iloc[i-9:i+1].to_numpy()
            xc=x-x.mean();yc=y-y.mean()
            slopes.append(xc@yc/(xc@xc));fits.append((xc@yc)**2/((xc@xc)*(yc@yc)))
        np.testing.assert_allclose(actual['slope'].A.iloc[9:],slopes,rtol=1e-8,atol=1e-8)
        np.testing.assert_allclose(actual['r_squared'].A.iloc[9:],fits,atol=1e-8)
        beta=np.asarray(slopes[-20:]);z=(beta[-1]-beta.mean())/beta.std(ddof=0)
        self.assertAlmostEqual(actual['zscore'].A.iloc[-1],z,places=7)
        self.assertAlmostEqual(actual['adjusted'].A.iloc[-1],z*fits[-1],places=7)
        self.assertAlmostEqual(actual['right_skew'].A.iloc[-1],z*fits[-1]*slopes[-1],places=7)

    def test_prefix_and_future_perturbation_invariant(self):
        full=compute_rsrs(self.high,self.low,self.spec);cut=self.low.index[100]
        high=self.high.copy();low=self.low.copy();high.loc[high.index>cut]*=3;low.loc[low.index>cut]*=3
        altered=compute_rsrs(high,low,self.spec)
        prefix=compute_rsrs(self.high.loc[:cut],self.low.loc[:cut],self.spec)
        for name in full:
            pd.testing.assert_frame_equal(full[name].loc[:cut],prefix[name])
            pd.testing.assert_frame_equal(full[name].loc[:cut],altered[name].loc[:cut])

    def test_quote_units_do_not_change_dimensionless_scores(self):
        original=compute_rsrs(self.high,self.low,self.spec)
        rescaled=compute_rsrs(self.high.mul([100,.001]),self.low.mul([100,.001]),self.spec)
        for name in original:
            np.testing.assert_allclose(original[name],rescaled[name],atol=1e-6,rtol=1e-6,equal_nan=True)

    def test_missing_quotes_need_full_prewarm_and_flat_prices_are_invalid(self):
        h=self.high.copy();l=self.low.copy();h.iloc[50]=np.nan;l.iloc[50]=np.nan
        out=compute_rsrs(h,l,self.spec)
        self.assertTrue(out['zscore'].iloc[:28].isna().all().all())
        self.assertTrue(out['zscore'].iloc[50:79].isna().all().all())
        self.assertTrue(out['zscore'].iloc[79].notna().all())
        flat=compute_rsrs(self.low*0+2.1,self.low*0+2,self.spec)
        self.assertTrue(flat['slope'].isna().all().all())

    def test_invalid_high_low_stops_computation(self):
        for high,low in ((self.high.iloc[1:],self.low),(self.low,self.high),
                          (self.high.mask(self.high.index.to_series().eq(self.high.index[40]),axis=0),self.low)):
            with self.assertRaises(ValueError):compute_rsrs(high,low,self.spec)

    def test_sqlite_cache_includes_high_low_windows_and_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            store=FactorStore(Path(directory)/'f.sqlite')
            first=store.get_rsrs(self.high,self.low,self.spec)
            pd.testing.assert_frame_equal(first,store.get_rsrs(self.high,self.low,self.spec))
            changed=self.high.copy();changed.iloc[-1,0]+=.03
            new=store.get_rsrs(changed,self.low,self.spec)
            self.assertNotAlmostEqual(first.iloc[-1,0],new.iloc[-1,0])
            expected=compute_rsrs(changed,self.low,self.spec)['right_skew']
            pd.testing.assert_frame_equal(expected,store.get_rsrs(changed,self.low,self.spec,'right_skew'))

    def test_data_uses_same_day_adjustment_and_rejects_snapshot_revision(self):
        days=pd.bdate_range('2020-01-01',periods=3)
        daily=pd.DataFrame({'trade_date':days.strftime('%Y%m%d'),'high':[2.1,1.1,1.2],
                             'low':[1.9,.9,1.0],'close':[2.,1.,1.1]})
        history=pd.DataFrame({'date':days,'raw_close':[2.,1.,1.1],'adj_factor':[1.,2.,2.],
                               'close':[2.,2.,2.2]})
        got=adjusted_high_low(daily,history,'A')
        np.testing.assert_allclose(got.high,[2.1,2.2,2.4])
        changed=daily.copy();changed.at[1,'close']=1.01
        with self.assertRaises(ValueError):adjusted_high_low(changed,history,'A')
        with self.assertRaises(ValueError):adjusted_high_low(daily.iloc[1:],history,'A')

    def test_research_design_is_fixed_and_has_matched_warmup_controls(self):
        from research_rsrs import cases
        entries=cases()
        self.assertEqual(len(entries),91)
        self.assertEqual(len({c.key for c in entries}),91)
        self.assertEqual(sum(c.mode=='maturity' for c in entries),12)


if __name__=='__main__':unittest.main()
