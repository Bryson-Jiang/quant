"""默认组合启用时点、卫星资金来源与共用回测时序。"""

from datetime import timedelta
from datetime import datetime
from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from etf_rotation.backtest import run_backtest
from etf_rotation.config import ETF_NAMES, PORTFOLIO_ETF_NAMES, STRATEGY_NAME
from etf_rotation.portfolio_strategy import portfolio_signals, satellite_details
from etf_rotation.risk_strategy import weekly_signal_dates
from etf_rotation.strategy_state import DividendPolicy, RotationState, load_rotation_state


class PortfolioStrategyTests(unittest.TestCase):
    def setUp(self):
        days=pd.bdate_range("2024-01-01",periods=430)
        rng=np.random.default_rng(39)
        self.core=pd.DataFrame(np.exp(np.cumsum(rng.normal(.0007,.007,(len(days),16)),axis=0)),index=days,columns=ETF_NAMES)
        self.history=pd.DataFrame({"512890":np.r_[np.full(210,100.),np.full(220,80.)]},index=days)
        weeks=weekly_signal_dates(days);self.effective=weeks[65]
        execution=days[days.get_loc(self.effective)+1]
        self.policy=DividendPolicy("512890",200,0.,.04,None,.3,"cash_then_bond",
                                   (self.effective-timedelta(days=2)).date(),self.effective.date(),execution.date())
        self.state=RotationState(weeks[10].date(),weeks[11].date(),("159941","159915"),(),self.policy)

    def test_adoption_starts_flat_and_preserves_core_targets(self):
        result=portfolio_signals(self.core,self.history,self.state)
        before=result.signals.index<self.effective
        self.assertTrue(result.signals.loc[before,"512890"].eq(0).all())
        pd.testing.assert_frame_equal(result.signals.loc[before,list(ETF_NAMES)],result.core_signals.loc[before])
        risky=list(set(ETF_NAMES)-{"511260"})
        pd.testing.assert_frame_equal(result.signals[risky],result.core_signals[risky])
        self.assertAlmostEqual(result.signals.at[self.effective,"512890"],.3)
        self.assertTrue(result.signals.sum(axis=1).le(1+1e-12).all())
        self.assertTrue(result.signals.gt(1e-10).sum(axis=1).le(5).all())

    def test_full_history_is_explicit_and_does_not_seed_actual_virtual_holdings(self):
        actual=portfolio_signals(self.core,self.history,self.state)
        hypothetical=portfolio_signals(self.core,self.history,self.state,hypothetical_satellite_history=True)
        before=hypothetical.signals.index<self.effective
        self.assertTrue(hypothetical.signals.loc[before,"512890"].gt(0).any())
        self.assertTrue(actual.signals.loc[before,"512890"].eq(0).all())
        details=satellite_details(actual,self.state,self.effective)
        self.assertEqual(details['exit_deviation'],.04)
        self.assertEqual(details['max_weight'],.3)
        self.assertTrue(details['core_targets_preserved'])
        self.assertFalse(details['global_eight_percent_budget_applied'])

    def test_next_close_execution(self):
        result=portfolio_signals(self.core,self.history,self.state)
        backtest=run_backtest(result.prices,result.signals,self.core.index[0].date(),self.core.index[-1].date())
        self.assertEqual(backtest.weights.at[self.effective,'512890'],0.)
        execution=self.core.index[self.core.index.get_loc(self.effective)+1]
        self.assertAlmostEqual(backtest.weights.at[execution,'512890'],.3)

    def test_later_satellite_history_never_creates_prelisting_targets(self):
        # 模拟ETF尚未上市，数据表只在核心历史的中间才开始。
        later=self.history.iloc[180:]
        for hypothetical in (False,True):
            result=portfolio_signals(self.core,later,self.state,hypothetical_satellite_history=hypothetical)
            before=result.satellite_states.index<later.index[0]
            self.assertTrue(result.satellite_states.loc[before,'active'].eq(False).all())
            self.assertTrue(result.signals.loc[result.signals.index<later.index[0],'512890'].eq(0).all())
            pd.testing.assert_index_equal(result.signals.index,result.core_signals.index)

    def test_empty_satellite_history_keeps_core_without_daily_phantom_signals(self):
        for hypothetical in (False,True):
            result=portfolio_signals(self.core,self.history.iloc[:0],self.state,hypothetical_satellite_history=hypothetical)
            self.assertTrue(result.signals['512890'].eq(0).all())
            self.assertTrue(result.satellite_states['active'].eq(False).all())
            pd.testing.assert_frame_equal(result.signals[list(ETF_NAMES)],result.core_signals)

    def test_loader_does_not_request_satellite_before_its_supported_history(self):
        from etf_rotation.portfolio_strategy import load_portfolio_data
        early=self.core.iloc[:5].copy()
        early.index=pd.bdate_range('2018-01-02',periods=5)
        with patch('etf_rotation.portfolio_strategy.load_close_prices',return_value=early), \
                patch('etf_rotation.portfolio_strategy.load_symbol',return_value=early['510300']) as load:
            _,h=load_portfolio_data(early.index[0].date(),early.index[-1].date(),Path('unused'),offline=True)
        self.assertEqual(load.call_count,1)
        self.assertEqual(load.call_args.args[0],'510300')
        pd.testing.assert_index_equal(h.index,early.index)
        self.assertTrue(h['512890'].isna().all())

    def test_future_prices_and_cutoff_do_not_change_old_portfolio(self):
        full=portfolio_signals(self.core,self.history,self.state)
        cutoff=weekly_signal_dates(self.core.index)[75]
        changed=self.core.copy();changed.loc[changed.index>cutoff]*=5
        history=self.history.copy();history.loc[history.index>cutoff]*=7
        future=portfolio_signals(changed,history,self.state)
        prefix=portfolio_signals(self.core.loc[:cutoff],self.history.loc[:cutoff],self.state)
        pd.testing.assert_frame_equal(full.signals.loc[:cutoff],future.signals.loc[:cutoff])
        pd.testing.assert_frame_equal(full.signals.loc[:cutoff],prefix.signals)

    def test_invalid_adoption_policy_stops_publication(self):
        root=Path(__file__).resolve().parents[1]
        payload=json.loads((root/'data/rotation_state.json').read_text(encoding='utf-8'))
        valid=load_rotation_state(root/'data/rotation_state.json')
        self.assertEqual(valid.dividend_policy.exit_deviation,.04)
        payload['dividend_policy']['exit_deviation']=.05
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.json';path.write_text(json.dumps(payload),encoding='utf-8')
            with self.assertRaises(ValueError):load_rotation_state(path)

    def test_live_entry_plan_includes_satellite_amount_and_independent_scores(self):
        import live_signal
        day=self.effective
        days=pd.date_range(day-timedelta(days=21),day+timedelta(days=21))
        calendar=pd.DataFrame({'is_open':(days.weekday<5).astype(int)},index=days)
        now=datetime(day.year,day.month,day.day,19,tzinfo=live_signal.SHANGHAI)
        with tempfile.TemporaryDirectory() as folder:
            cache=Path(folder);output=cache/'signal.json'
            for symbol in PORTFOLIO_ETF_NAMES:
                pd.DataFrame({'date':[day.date().isoformat()],'raw_close':[10.]}).to_csv(cache/f'tushare_{symbol}_hfq.csv',index=False)
            argv=['live_signal.py','--as-of',day.date().isoformat(),'--cache-dir',str(cache),'--output',str(output),'--capital','100000']
            with patch('sys.argv',argv),patch('live_signal.datetime') as clock, \
                    patch('live_signal.trade_calendar',return_value=calendar), \
                    patch('live_signal.load_portfolio_data',return_value=(self.core.loc[:day],self.history.loc[:day])), \
                    patch('live_signal.load_rotation_state',return_value=self.state), \
                    patch('live_signal.FactorStore',return_value=None),redirect_stdout(io.StringIO()):
                clock.now.return_value=now
                live_signal.main()
            plan=json.loads(output.read_text(encoding='utf-8'))
        self.assertEqual(plan['strategy'],STRATEGY_NAME)
        self.assertEqual(plan['status'],'ready')
        self.assertEqual(plan['target_weights']['512890'],.3)
        self.assertEqual(plan['target_notional_cny']['512890'],30000.)
        self.assertTrue(plan['satellite_details']['active'])
        self.assertEqual(len(plan['score_details']),15)
        self.assertNotIn('512890',[row['symbol'] for row in plan['score_details']])
        self.assertEqual(plan['max_total_etf_positions'],5)


if __name__=='__main__':unittest.main()
