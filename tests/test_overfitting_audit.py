"""审计加速内核必须遵守正式引擎的成交、估值及因果时序。"""
import unittest
from datetime import date

import numpy as np
import pandas as pd

from etf_rotation.backtest import run_backtest
from etf_rotation.risk_strategy import RiskConfig, risk_controlled_signals
from research_overfitting_audit import CoreAudit, global_risk_guard, simulate


class OverfittingAuditTests(unittest.TestCase):
    def test_missing_execution_and_weight_drift_match_production(self):
        dates=pd.bdate_range("2022-01-03",periods=12)
        p=pd.DataFrame({"510300":np.arange(100.,112.),"511260":np.linspace(100,101,12)},index=dates)
        p.loc[dates[5],"510300"]=np.nan
        signals=pd.DataFrame([[.6,.4],[0.,1.],[.5,.5]],index=dates[[0,4,7]],columns=p.columns)
        for cost in (0,5,20):
            slow=run_backtest(p,signals,dates[0].date(),dates[-1].date(),fee_bps=0,slippage_bps=cost)
            fast=simulate(p,signals,dates[0],cost_bps=cost)
            np.testing.assert_allclose(fast.returns,slow.equity.daily_return,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(fast.weights,slow.weights.drop(columns="CASH"),rtol=1e-12,atol=1e-12)
            self.assertEqual(fast.skipped,1)

    def test_audit_core_and_future_invariance(self):
        rng=np.random.default_rng(14)
        dates=pd.bdate_range("2020-01-01",periods=420)
        p=pd.DataFrame(np.cumprod(1+rng.normal(.0005,.009,(420,4)),axis=0)*100,
                       index=dates,columns=["510300","511260","159941","518880"])
        cfg=RiskConfig.current_default()
        slow=risk_controlled_signals(p,cfg)
        fast=CoreAudit(p).signals(cfg)
        pd.testing.assert_frame_equal(slow,fast,check_exact=False,rtol=1e-10,atol=1e-12)
        cutoff=fast.index[55]
        changed=p.copy();changed.loc[changed.index>cutoff]*=3
        newer=CoreAudit(changed).signals(cfg)
        prefix=CoreAudit(p.loc[:cutoff]).signals(cfg)
        pd.testing.assert_frame_equal(fast.loc[:cutoff],newer.loc[:cutoff])
        pd.testing.assert_frame_equal(fast.loc[:cutoff],prefix)

    def test_maturity_and_exclusion_do_not_create_prelisting_positions(self):
        dates=pd.bdate_range("2020-01-01",periods=250)
        p=pd.DataFrame({"510300":np.exp(np.arange(250)*.001),
                        "511260":np.exp(np.arange(250)*.0001),
                        "159941":np.r_[np.full(150,np.nan),np.exp(np.arange(100)*.003)]},index=dates)
        e=CoreAudit(p)
        t=e.signals(min_age=120)
        self.assertTrue(t["159941"].eq(0).all())
        excluded=e.signals(excluded=["159941"])
        self.assertTrue(excluded["159941"].eq(0).all())

    def test_global_guard_uses_past_risk_and_preserves_cash(self):
        dates=pd.bdate_range("2022-01-03",periods=150)
        rng=np.random.default_rng(7)
        p=pd.DataFrame(np.cumprod(1+rng.normal(.0005,.03,(150,3)),axis=0),index=dates,
                       columns=["510300","511260","512890"])
        t=pd.DataFrame([[.5,0,.5]]*20,index=dates[-20:],columns=p.columns)
        result=global_risk_guard(p,t)
        self.assertTrue(result.drop(columns="511260").sum(axis=1).le(.7+1e-12).all())
        self.assertTrue(result["511260"].eq(0).all())
        self.assertTrue(result.sum(axis=1).le(1+1e-12).all())


if __name__=="__main__":
    unittest.main()
