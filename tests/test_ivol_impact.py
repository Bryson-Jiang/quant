"""IVOL事件解释的相同起点、局部费用、缓冲状态及持仓统计口径。"""
from datetime import date
import unittest
import numpy as np
import pandas as pd

from etf_rotation.backtest import run_backtest,BacktestResult
from etf_rotation.risk_strategy import risk_controlled_signals,weekly_signal_dates
from etf_rotation.strategy_state import RotationState
import research_ivol_impact as research


class IVOLImpactTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(121)
        self.days=pd.bdate_range('2024-01-01',periods=160)
        self.prices=pd.DataFrame(np.exp(np.cumsum(rng.normal(.0008,.008,(160,3)),axis=0)),
                                 index=self.days,columns=['510300','511260','159915'])
        weeks=weekly_signal_dates(self.days)
        self.signals=pd.DataFrame(0.,index=weeks,columns=self.prices.columns)
        self.signals['510300']=np.where(np.arange(len(weeks))%2,.4,.1)
        self.signals['159915']=np.where(np.arange(len(weeks))%2,.1,.5)
        self.signals['511260']=1-self.signals[['510300','159915']].sum(axis=1)

    def test_conditional_replay_matches_production_from_existing_positions(self):
        full=run_backtest(self.prices,self.signals,self.days[0].date(),self.days[-1].date())
        start=self.days[73];end=self.days[135]
        result=research.conditional_replay(self.prices,self.signals,start,end,full.weights.loc[start])
        expected=full.equity.strategy_nav.loc[start:end];expected=expected/expected.iloc[0]
        pd.testing.assert_series_equal(result.nav,expected,check_names=False,atol=1e-12,rtol=1e-12)

    def test_missing_execution_quote_cancels_same_batch_as_production(self):
        p=self.prices.copy();signal=self.signals.index[20];execution=p.index[p.index.get_loc(signal)+1]
        p.loc[execution,'159915']=np.nan
        full=run_backtest(p,self.signals,p.index[0].date(),p.index[-1].date())
        start=self.days[70];result=research.conditional_replay(p,self.signals,start,self.days[-1],full.weights.loc[start])
        expected=full.equity.strategy_nav.loc[start:]/full.equity.strategy_nav.loc[start]
        pd.testing.assert_series_equal(result.nav,expected,check_names=False,atol=1e-12,rtol=1e-12)
        self.assertEqual(result.at[execution,'cost'],0)

    def test_single_cycle_includes_initial_switch_cost_and_cash(self):
        days=pd.bdate_range('2024-01-01',periods=3)
        p=pd.DataFrame({'510300':[1,1.05,1.1],'511260':[1,1,1]},index=days)
        target=pd.Series({'510300':.5,'511260':0.});pre=pd.Series({'510300':0.,'511260':1.})
        curve,cost=research.cycle_curve(p,target,pre,days[0],days[-1],5.)
        self.assertAlmostEqual(cost,1.5*.0005)
        self.assertAlmostEqual(curve.iloc[0],1-cost)
        self.assertAlmostEqual(curve.iloc[-1],(1-cost)*1.05)

    def test_common_selection_states_preserve_adoption_anchor(self):
        p=self.prices;weeks=weekly_signal_dates(p.index)
        rotation=RotationState(weeks[15].date(),weeks[16].date(),('159915',))
        mask=pd.DataFrame(True,index=p.index,columns=p.columns)
        mask.loc[p.index>=weeks[20],'159915']=False
        actual=risk_controlled_signals(p,research.BASE,eligibility_mask=mask,**rotation.signal_kwargs(p))
        prior,resets,_=research.selection_states(p,actual,rotation)
        self.assertEqual(prior[weeks[16]],('159915',))
        reconstructed=risk_controlled_signals(p,research.BASE,eligibility_mask=mask,selection_resets=resets)
        pd.testing.assert_frame_equal(actual,reconstructed)

    def test_average_weight_is_over_all_days_not_only_held_days(self):
        days=pd.bdate_range('2024-01-01',periods=6)
        weights=pd.DataFrame({'511260':[0.,.2,.2,0.,0.,0.],'CASH':[1.,.8,.8,1.,1.,1.]},index=days)
        result=BacktestResult(pd.DataFrame(),weights,pd.DataFrame(),{})
        row=research.holding_summary(result,'2024-01-01','test')[0]
        self.assertAlmostEqual(row['average_weight'],.4/6)
        self.assertAlmostEqual(row['mean_weight_when_held'],.2)
        self.assertEqual(row['held_days'],2)
        self.assertEqual(row['episodes_including_open'],1)

    def test_reentry_reports_actual_held_date_within_horizon(self):
        days=pd.bdate_range('2024-01-01',periods=5)
        w=pd.DataFrame({'510300':[0.,0.,0.,.2,.2]},index=days)
        self.assertIsNone(research.next_holding_execution(w,'510300',days[0],days[2]))
        self.assertEqual(research.next_holding_execution(w,'510300',days[0],days[-1]),days[3])


if __name__=='__main__':unittest.main()
