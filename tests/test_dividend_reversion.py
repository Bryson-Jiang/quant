"""长期均线卫星：时序、超时再武装、动态退出及组合仓位约束。"""

import unittest
import numpy as np
import pandas as pd

from etf_rotation.config import LEGACY_CORE_NAMES as ETF_NAMES
from etf_rotation.factors import FactorSpec, compute_factor
from etf_rotation.mean_reversion import ReversionRule, defensive_funding_targets, reversion_states
from etf_rotation.risk_strategy import RiskConfig, risk_controlled_signals, risk_score_components, weekly_signal_dates
from research_dividend_reversion import PRIMARY, combine, episodes, fixed_rules


class DividendReversionTests(unittest.TestCase):
    def test_grid_frozen_and_thresholds_stateful(self):
        rules=fixed_rules()
        self.assertEqual(len(rules),18)
        self.assertEqual(len({r.key for r in rules}),18)
        days=pd.bdate_range("2024-01-01",periods=20)
        p=pd.DataFrame({PRIMARY:[100]*10+[90,90,91,91,91,110,110,110,110,110]},index=days)
        state=reversion_states(p,PRIMARY,ReversionRule(5,-.03,.03,None),days)
        self.assertEqual(state.loc[days[10],"reason"],"entry")
        self.assertTrue(state.loc[days[11],"active"])
        self.assertEqual(state.loc[days[15],"reason"],"ma_exit")

    def test_timeout_does_not_immediately_reenter_same_decline(self):
        days=pd.bdate_range("2024-01-01",periods=45)
        p=pd.DataFrame({PRIMARY:[100]*20+list(np.linspace(85,45,25))},index=days)
        state=reversion_states(p,PRIMARY,ReversionRule(20,-.03,.03,3),days)
        self.assertEqual(int(state.reason.eq("entry").sum()),1)
        self.assertEqual(int(state.reason.eq("time_exit").sum()),1)
        self.assertFalse(state.active.iloc[-1])

    def test_rolling_ma_profit_line_can_exit_with_loss(self):
        days=pd.bdate_range("2024-01-01",periods=24)
        p=pd.DataFrame({PRIMARY:[100]*10+[90,90]+[80]*10+[84,84]},index=days)
        rule=ReversionRule(5,-.03,.03,None)
        state=reversion_states(p,PRIMARY,rule,days)
        trade=episodes(p,state,PRIMARY,rule).iloc[0]
        self.assertEqual(trade.exit_reason,"ma_exit")
        self.assertAlmostEqual(trade.asset_return,84/90-1)
        self.assertLess(trade.asset_return,0)

    def test_temporary_no_quote_does_not_erase_long_ma_history(self):
        days=pd.bdate_range("2024-01-01",periods=20)
        values=[100.]*20;values[10]=np.nan;values[11]=90.
        p=pd.DataFrame({PRIMARY:values},index=days)
        state=reversion_states(p,PRIMARY,ReversionRule(5,-.03,.03,None),days)
        self.assertTrue(pd.isna(state.loc[days[10],"deviation"]))
        self.assertNotEqual(state.loc[days[10],"reason"],"entry")
        self.assertEqual(state.loc[days[11],"reason"],"entry")

    def test_prefix_and_future_invariance_for_every_rule(self):
        rng=np.random.default_rng(28);days=pd.bdate_range("2022-01-03",periods=500)
        p=pd.DataFrame({PRIMARY:np.exp(np.cumsum(rng.normal(.0001,.014,len(days))))},index=days)
        cutoff=days[350];changed=p.copy();changed.loc[changed.index>cutoff]*=8
        for rule in fixed_rules():
            a=reversion_states(p,PRIMARY,rule,weekly_signal_dates(p.index))
            b=reversion_states(changed,PRIMARY,rule,weekly_signal_dates(changed.index))
            pd.testing.assert_frame_equal(a.loc[:cutoff],b.loc[:cutoff])
            prefix=p.loc[:cutoff]
            c=reversion_states(prefix,PRIMARY,rule,weekly_signal_dates(prefix.index))
            pd.testing.assert_frame_equal(a.loc[a.index<cutoff],c.loc[c.index<cutoff])

    def test_integrations_preserve_caps_and_inactive_baseline(self):
        rng=np.random.default_rng(42);days=pd.bdate_range("2024-01-01",periods=260)
        p=pd.DataFrame(np.exp(np.cumsum(rng.normal(.0007,.01,(260,17)),axis=0)),index=days,columns=list(ETF_NAMES)+[PRIMARY])
        core=p.drop(columns=PRIMARY);cfg=RiskConfig.current_default()
        base=risk_controlled_signals(core,cfg)
        factor=lambda n,k:compute_factor(core,FactorSpec(n,k))
        score=risk_score_components(factor("momentum",20),factor("momentum",40),factor("volatility",60),cfg,factor("ma_zscore",20))["score"]
        bear=factor("trend",40)['510300'].le(0);bond=factor("momentum",60)['511260'].gt(0)
        state=pd.DataFrame({"active":False},index=days)
        original=combine(p,base,score,bear,bond,state,PRIMARY,"budget_add",.3)
        pd.testing.assert_frame_equal(original,base.reindex(columns=p.columns,fill_value=0.0))
        state.loc[days[150]:,"active"]=True
        for mode in ("risk_replace","budget_add"):
            for cap in (.15,.30,.40):
                target=combine(p,base,score,bear,bond,state,PRIMARY,mode,cap)
                risky=target.drop(columns='511260')
                self.assertTrue(np.isfinite(target).all().all())
                self.assertTrue(target[PRIMARY].le(cap+1e-12).all())
                self.assertTrue(risky.drop(columns=PRIMARY).le(.30+1e-12).all().all())
                self.assertTrue(risky.sum(axis=1).le(.70+1e-12).all())
                self.assertTrue(risky.gt(1e-10).sum(axis=1).le(3).all())
                self.assertTrue(target.sum(axis=1).le(1+1e-12).all())

    def test_cap40_allows_dividend_above30_but_retains_bear_budget(self):
        days=pd.bdate_range("2024-01-01",periods=100)
        columns=list(ETF_NAMES)+[PRIMARY]
        growth=.0003+.001*np.sin(np.arange(100))
        p=pd.DataFrame({s:np.exp(np.cumsum(growth*(1+i/30))) for i,s in enumerate(columns)},index=days)
        dates=days[[80,90]]
        base=pd.DataFrame(0.,index=dates,columns=list(ETF_NAMES))
        base.loc[:,['510300','159915','159941','511260']]=[.25,.25,.20,.30]
        score=pd.DataFrame(1.,index=days,columns=list(ETF_NAMES))
        score.loc[:,['510300','159915','159941']]=[3.,2.,1.]
        state=pd.DataFrame({'active':True},index=days)
        bond=pd.Series(True,index=days)
        normal=combine(p,base,score,pd.Series(False,index=days),bond,state,PRIMARY,'risk_replace',.40)
        self.assertAlmostEqual(normal.loc[dates[0],PRIMARY],.40)
        self.assertAlmostEqual(normal.loc[dates[0]].drop('511260').sum(),.70)
        weak=combine(p,base,score,pd.Series(True,index=days),bond,state,PRIMARY,'risk_replace',.40)
        self.assertLessEqual(weak.loc[dates[0]].drop('511260').sum(),.35+1e-12)
        self.assertLessEqual(weak.loc[dates[0],PRIMARY],.40)

    def test_defensive_funding_preserves_core_and_spends_cash_before_bond(self):
        days=pd.bdate_range("2024-01-01",periods=3)
        p=pd.DataFrame(100.,index=days,columns=['510300','159915','159941','511260',PRIMARY])
        base=pd.DataFrame([[.2,.2,.2,.3],[.3,.3,.3,.1]],index=days[[0,2]],columns=p.columns[:-1])
        state=pd.DataFrame({'active':True},index=days)
        target=defensive_funding_targets(p,base,state,PRIMARY,.2)
        pd.testing.assert_frame_equal(target[base.columns[:-1]],base[base.columns[:-1]],check_names=False)
        self.assertAlmostEqual(target.iloc[0][PRIMARY],.2)
        self.assertAlmostEqual(target.iloc[0]['511260'],.2)
        self.assertAlmostEqual(target.iloc[0].sum(),1.)
        self.assertAlmostEqual(target.iloc[1][PRIMARY],.1)
        self.assertAlmostEqual(target.iloc[1]['511260'],0.)
        self.assertEqual(int(target.iloc[0].gt(0).sum()),5)
        inactive=state.assign(active=False)
        unchanged=defensive_funding_targets(p,base,inactive,PRIMARY,.3)
        pd.testing.assert_frame_equal(unchanged,base.reindex(columns=p.columns,fill_value=0.),check_names=False)

    def test_defensive_guard_uses_spare_gross_budget_without_cutting_core(self):
        days=pd.bdate_range("2024-01-01",periods=100)
        growth=.0002+.001*np.sin(np.arange(100))
        p=pd.DataFrame({s:np.exp(np.cumsum(growth)) for s in ['510300','511260',PRIMARY]},index=days)
        dates=days[[80,90]]
        base=pd.DataFrame([[.7,.3],[.2,.8]],index=dates,columns=['510300','511260'])
        state=pd.DataFrame({'active':True},index=days)
        weak=pd.Series(False,index=days);weak.loc[dates[1]]=True
        target=defensive_funding_targets(p,base,state,PRIMARY,.3,bear=weak,guard=True)
        np.testing.assert_allclose(target['510300'],base['510300'])
        self.assertAlmostEqual(target.loc[dates[0],PRIMARY],0.)
        self.assertAlmostEqual(target.loc[dates[1],PRIMARY],.15)
        self.assertAlmostEqual(target.loc[dates[1],'511260'],.65)

    def test_defensive_guard_reduces_incremental_weight_to_covariance_budget(self):
        days=pd.bdate_range("2024-01-01",periods=100)
        growth=.001+.02*np.sin(np.arange(100))
        p=pd.DataFrame({s:np.exp(np.cumsum(growth)) for s in ['510300','511260',PRIMARY]},index=days)
        base=pd.DataFrame([[.2,.8]],index=days[[-1]],columns=['510300','511260'])
        state=pd.DataFrame({'active':True},index=days)
        target=defensive_funding_targets(p,base,state,PRIMARY,.3,bear=pd.Series(False,index=days),guard=True)
        added=float(target.iloc[0][PRIMARY])
        self.assertGreater(added,0.)
        self.assertLess(added,.3)
        sigma=float(p['510300'].pct_change().tail(60).std(ddof=1)*np.sqrt(252))
        self.assertAlmostEqual((.2+added)*sigma,.08)
        self.assertAlmostEqual(target.iloc[0]['510300'],.2)


if __name__=="__main__":unittest.main()
