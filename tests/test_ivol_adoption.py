"""IVOL生产接入、缓冲优先级、启用边界与既有研究基线的回归检查。"""
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from etf_rotation.config import ETF_NAMES, IVOL_COMPARISON_SYMBOLS, PRE_IVOL_PORTFOLIO_STRATEGY_NAME
from etf_rotation.extended_factors import ExtendedFactorSpec, compute_extended, low_ivol_utility
from etf_rotation.ivol_filter import market_ivol_filter
from etf_rotation.portfolio_strategy import portfolio_signals, ivol_details
from etf_rotation.risk_strategy import RiskConfig, risk_controlled_signals, weekly_signal_dates
from etf_rotation.strategy_state import load_rotation_state, load_pre_ivol_rotation_state
from live_signal import signal_score_details

ROOT=Path(__file__).resolve().parents[1]


class IVOLAdoptionTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(120)
        self.days=pd.bdate_range('2024-01-01',periods=380)
        m=rng.normal(.0005,.008,len(self.days))
        returns=np.column_stack([.0006+(1+i*.04)*m+rng.normal(0,.001+i*.0007,len(m)) for i in range(len(ETF_NAMES))])
        self.prices=pd.DataFrame(np.exp(np.cumsum(returns,axis=0)),index=self.days,columns=ETF_NAMES)
        self.prices['510300']=np.exp(np.cumsum(m))
        self.history=pd.DataFrame({'512890':np.linspace(1,1.02,len(m))},index=self.days)
        self.effective=weekly_signal_dates(self.days)[50]
        state=load_rotation_state(ROOT/'data/rotation_state.json')
        policy=replace(state.ivol_policy,adopted_on=self.effective.date()+timedelta(days=3),
                       effective_signal_date=self.effective.date(),first_execution_date=self.effective.date()+timedelta(days=3))
        self.state=replace(state,ivol_policy=policy,universe_policy=None)

    def test_current_identity_and_frozen_seventeen_asset_baseline(self):
        current=load_rotation_state(ROOT/'data/rotation_state.json')
        old=load_pre_ivol_rotation_state(ROOT/'data/rotation_state.json')
        self.assertIsNotNone(current.ivol_policy)
        self.assertEqual(current.ivol_policy.window,120)
        self.assertEqual(current.ivol_policy.comparison_symbols,IVOL_COMPARISON_SYMBOLS)
        self.assertEqual(current.ivol_policy.previous_strategy,PRE_IVOL_PORTFOLIO_STRATEGY_NAME)
        self.assertIsNone(old.ivol_policy)
        self.assertEqual(len(old.core_symbols),17)

    def test_adoption_preserves_all_prior_targets_and_buffer_history(self):
        old=portfolio_signals(self.prices,self.history,replace(self.state,ivol_policy=None))
        new=portfolio_signals(self.prices,self.history,self.state)
        before=new.signals.index<self.effective
        pd.testing.assert_frame_equal(new.signals.loc[before],old.signals.loc[before])
        self.assertTrue(new.ivol_filter.eligibility_mask.loc[self.days<self.effective].all().all())
        self.assertFalse(new.ivol_filter.eligibility_mask.loc[self.effective].all())
        for day in new.core_signals.index:
            self.assertTrue(new.core_signals.loc[day,~new.ivol_filter.eligibility_mask.loc[day]].eq(0).all())

    def test_full_history_filter_requires_explicit_option(self):
        actual=portfolio_signals(self.prices,self.history,self.state)
        hypothetical=portfolio_signals(self.prices,self.history,self.state,hypothetical_ivol_history=True)
        self.assertFalse(actual.ivol_filter.active.loc[self.days<self.effective].any())
        self.assertTrue(hypothetical.ivol_filter.active.all())
        self.assertEqual(ivol_details(hypothetical,self.state,self.effective)['simulation_mode'],'hypothetical_full_history')

    def test_backdated_query_does_not_use_later_adoption_decision(self):
        before=portfolio_signals(self.prices.loc[:self.effective],self.history,self.state,
                                 as_of=self.effective.date())
        old=portfolio_signals(self.prices.loc[:self.effective],self.history,replace(self.state,ivol_policy=None))
        self.assertFalse(before.ivol_filter.active.any())
        pd.testing.assert_frame_equal(before.signals,old.signals)

    def test_filter_exactly_matches_existing_research_definition(self):
        result=market_ivol_filter(self.prices,self.state.ivol_policy,hypothetical_history=True)
        raw=compute_extended(self.prices,ExtendedFactorSpec('ivol',window=120,model='market'))
        u=low_ivol_utility(raw,list(IVOL_COMPARISON_SYMBOLS))
        pd.testing.assert_frame_equal(result.ivol,raw)
        pd.testing.assert_frame_equal(result.eligibility_mask,u.ge(-.5)|u.isna())
        self.assertTrue(result.eligibility_mask['510300'].all())
        self.assertEqual(int((~result.eligibility_mask.iloc[-1]).sum()),2)
        for symbol in set(ETF_NAMES)-set(IVOL_COMPARISON_SYMBOLS):
            self.assertTrue(result.eligibility_mask[symbol].all())

    def test_missing_factor_is_neutral_and_inactive_asset_is_not_ranked(self):
        missing=self.prices.copy();missing.loc[:,'159915']=np.nan
        result=market_ivol_filter(missing,self.state.ivol_policy,hypothetical_history=True,
                                 universe_activation_dates={'159995':self.days[-1]+timedelta(days=1)})
        self.assertTrue(result.eligibility_mask['159915'].all())
        self.assertTrue(result.utility['159995'].isna().all())
        self.assertTrue(result.eligibility_mask['159995'].all())
        early=result.ivol.iloc[:119].notna().sum(axis=1)
        self.assertTrue(early.eq(0).all())

    def test_future_perturbation_and_prefix_do_not_change_decisions(self):
        cutoff=weekly_signal_dates(self.days)[60]
        full=portfolio_signals(self.prices,self.history,self.state)
        changed=self.prices.copy();changed.loc[changed.index>cutoff]*=np.linspace(1,5,len(changed.loc[changed.index>cutoff]))[:,None]
        altered=portfolio_signals(changed,self.history,self.state)
        prefix=portfolio_signals(self.prices.loc[:cutoff],self.history.loc[:cutoff],self.state)
        pd.testing.assert_frame_equal(full.signals.loc[:cutoff],altered.signals.loc[:cutoff])
        pd.testing.assert_frame_equal(full.signals.loc[:cutoff],prefix.signals)
        pd.testing.assert_frame_equal(full.ivol_filter.utility.loc[:cutoff],prefix.ivol_filter.utility)

    def test_filtered_old_top_rank_cannot_survive_buffer(self):
        prices=self.prices
        class Store:
            def get(self,p,spec):
                frame=pd.DataFrame(.01,index=p.index,columns=p.columns)
                if spec.name=='volatility':return frame*20
                if spec.name=='ma_zscore':return frame*0
                if spec.name=='momentum' and spec.window in (20,40):
                    frame['512800']=.2;frame['510300']=.15;frame['159941']=.12;frame['159928']=.1
                return frame
            def get_extended(self,p,spec):
                frame=pd.DataFrame(np.nan,index=p.index,columns=p.columns)
                for i,s in enumerate(IVOL_COMPARISON_SYMBOLS[1:]):frame[s]=.05+i*.01
                frame['512800']=1.;frame['515070']=.9
                return frame
        cfg=RiskConfig.current_default();store=Store()
        state=replace(self.state,selection_anchor_date=weekly_signal_dates(self.days)[49].date(),
                      effective_signal_date=self.effective.date(),initial_risky_selection=('512800','159941'),historical_anchors=())
        result=portfolio_signals(prices,self.history,state,cfg,store)
        self.assertEqual(result.core_signals.at[self.effective,'512800'],0)
        self.assertGreater(result.core_signals.at[self.effective,'510300'],0)
        rows=signal_score_details(prices,cfg,result.core_signals.loc[self.effective],store,ivol_filter=result.ivol_filter)
        bank=next(r for r in rows if r['symbol']=='512800')
        self.assertTrue(bank['original_trend_eligible'])
        self.assertFalse(bank['eligible']);self.assertFalse(bank['ivol_filter_pass'])
        first=next(r for r in rows if r['rank']==1)
        self.assertEqual(first['symbol'],'510300')

    def test_state_rejects_parameter_drift_or_late_adoption(self):
        payload=json.loads((ROOT/'data/rotation_state.json').read_text(encoding='utf-8'))
        for key,value in [('window',60),('max_normalized_rank',.8),('adopted_on','2026-11-10')]:
            bad=json.loads(json.dumps(payload));bad['ivol_policy'][key]=value
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'state.json';path.write_text(json.dumps(bad),encoding='utf-8')
                with self.assertRaises(ValueError):load_rotation_state(path)


if __name__=='__main__':unittest.main()
