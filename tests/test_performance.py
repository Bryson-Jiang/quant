"""核对回合现金流、未平仓排除及收益区块，避免把口径误当策略优势。"""
import unittest
import numpy as np
import pandas as pd

from etf_rotation.performance import return_metrics,outcome_metrics,holding_episodes,rebalance_periods,turnover_metrics


class PerformanceTests(unittest.TestCase):
    def test_payoff_is_not_profit_factor_and_flats_stay_in_denominator(self):
        stats=outcome_metrics([1.,2.,-2.,0.])
        self.assertEqual(stats['count'],4)
        self.assertEqual(stats['win_rate'],.5)
        self.assertEqual(stats['payoff_ratio'],.75)
        self.assertEqual(stats['profit_factor'],1.5)

    def test_partial_exits_additions_and_open_mark_reconcile(self):
        orders=pd.DataFrame([
            ['2024-01-01','A','Completed',10.,10.,1.],
            ['2024-01-02','A','Completed',-4.,12.,.48],
            ['2024-01-03','A','Completed',2.,11.,.22],
            ['2024-01-04','A','Completed',-8.,9.,.72],
            ['2024-01-05','B','Completed',5.,20.,1.],
            ['2024-01-05','C','Rejected',5.,0.,0.],
        ],columns=['execution_date','symbol','status','size','price','commission'])
        episodes=holding_episodes(orders,pd.Series({'B':22.}),initial_cash=1000.)
        self.assertEqual(len(episodes),2)
        closed=episodes.loc[episodes.closed]
        self.assertEqual(len(closed),1)
        self.assertEqual(int(closed.iloc[0].orders),4)
        self.assertAlmostEqual(float(closed.iloc[0].net_pnl),-4.42/1000)
        self.assertAlmostEqual(float(episodes.net_pnl.sum()),4.58/1000)
        self.assertAlmostEqual(float(episodes.fees.sum()),3.42/1000)
        self.assertEqual(outcome_metrics(closed.net_pnl)['wins'],0)

    def test_closed_position_is_not_invented_by_terminal_mark(self):
        orders=pd.DataFrame([['2024-01-01','A','Completed',1.,10.,.1]],
             columns=['execution_date','symbol','status','size','price','commission'])
        episodes=holding_episodes(orders,pd.Series({'A':100.}),initial_cash=100.)
        self.assertFalse(bool(episodes.iloc[0].closed))
        self.assertEqual(outcome_metrics(episodes.loc[episodes.closed,'net_pnl'])['count'],0)
        self.assertAlmostEqual(float(episodes.net_pnl.sum()),.899)

    def test_first_day_cost_counts_as_drawdown_and_sharpe_uses_mean(self):
        r=pd.Series([-.1,1/9,0.],index=pd.bdate_range('2024-01-01',periods=3))
        stats=return_metrics(r)
        self.assertAlmostEqual(stats['max_drawdown'],-.1)
        self.assertAlmostEqual(stats['annual_return'],0.)
        self.assertAlmostEqual(stats['calmar'],0.)
        self.assertGreater(stats['sharpe_zero_rf'],0.)

    def test_periods_are_disjoint_and_keep_last_incomplete(self):
        r=pd.Series([-.001,.01,-.01,.02,-.01,.005,.003,-.002,.01,.004],
                    index=pd.bdate_range('2024-01-01',periods=10))
        periods=rebalance_periods(r,r.index[[0,5,9]])
        self.assertEqual(periods.days.tolist(),[5,4,1])
        self.assertEqual(periods.complete.tolist(),[True,True,False])
        self.assertAlmostEqual(float(np.prod(1+periods.net_return)),float(np.prod(1+r)))

    def test_two_sided_and_half_equivalent_turnover(self):
        e=pd.DataFrame({'turnover':[1.,.4,0.,.6],'cost_fraction':[.0005,.0002,0.,.0003]})
        stats=turnover_metrics(e)
        self.assertEqual(stats['annual_two_sided_turnover'],126.)
        self.assertEqual(stats['annual_half_two_sided_equivalent'],63.)
        self.assertAlmostEqual(stats['annual_additive_cost_fraction'],.063)


if __name__=='__main__':unittest.main()
