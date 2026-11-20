"""相关性诊断使用收益、成对样本和正确符号，防止缺失值冒充低相关。"""
import unittest
import numpy as np
import pandas as pd

from analyze_pool_correlations import daily_returns,correlation_matrix,priority_table,window_definitions


class PoolCorrelationTests(unittest.TestCase):
    def test_positive_negative_and_constant_series(self):
        dates=pd.bdate_range('2024-01-01',periods=20);x=np.linspace(-.02,.02,20)
        r=pd.DataFrame({'candidate':x,'same':x*2,'opposite':-x,'constant':0.},index=dates)
        rho,n=correlation_matrix(r,['candidate'],['same','opposite','constant'],10)
        self.assertAlmostEqual(rho.at['candidate','same'],1)
        self.assertAlmostEqual(rho.at['candidate','opposite'],-1)
        self.assertTrue(pd.isna(rho.at['candidate','constant']))
        self.assertTrue(n.eq(20).all().all())

    def test_missing_quotes_are_not_filled_and_counts_are_pairwise(self):
        dates=pd.bdate_range('2024-01-01',periods=7)
        p=pd.DataFrame({'candidate':[1,1.1,np.nan,1.2,1.3,1.4,1.6],
                        'existing':[2,2.1,2.2,2.3,2.4,np.nan,2.5]},index=dates)
        r=daily_returns(p)
        self.assertTrue(r.loc[dates[2:4],'candidate'].isna().all())
        rho,n=correlation_matrix(r,['candidate'],['existing'],3)
        self.assertEqual(n.at['candidate','existing'],2)
        self.assertTrue(pd.isna(rho.at['candidate','existing']))

    def test_independent_price_units_do_not_change_returns_or_correlation(self):
        rng=np.random.default_rng(4);days=pd.bdate_range('2024-01-01',periods=30)
        p=pd.DataFrame(np.exp(np.cumsum(rng.normal(0,.01,(30,2)),axis=0)),index=days,columns=['a','b'])
        q=p.mul(pd.Series({'a':1000.,'b':.001}))
        a,_=correlation_matrix(daily_returns(p),['a'],['b'],20)
        b,_=correlation_matrix(daily_returns(q),['a'],['b'],20)
        pd.testing.assert_frame_equal(a,b,atol=1e-12,rtol=1e-12)

    def test_ranking_does_not_treat_missing_as_zero_or_penalize_negative_hedge(self):
        candidates=['positive','negative','missing'];existing=['risk','bond']
        matrix=pd.DataFrame({'risk':[.4,-.8,np.nan],'bond':[.0,.5,np.nan]},index=candidates)
        matrices={k:matrix.copy() for k in ['long','recent','down','crash']}
        summary=priority_table(matrices,candidates,['risk'])
        self.assertEqual(summary.iloc[0].symbol,'negative')
        self.assertEqual(summary.iloc[-1].symbol,'missing')
        self.assertAlmostEqual(summary.iloc[0].worst_of_three,-.8)
        self.assertTrue(pd.isna(summary.iloc[-1].worst_of_three))

    def test_down_and_crash_windows_use_same_day_market_return_condition(self):
        days=pd.bdate_range('2024-01-01',periods=6)
        r=pd.DataFrame({'510300':[-.02,-.01,-.005,0,.01,.02],'candidate':[.01]*6},index=days)
        windows=window_definitions(r)
        self.assertEqual(windows['down'][0].index.tolist(),days[:3].tolist())
        self.assertEqual(windows['crash'][0].index.tolist(),days[:2].tolist())


if __name__=='__main__':unittest.main()
