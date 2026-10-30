"""核对真实Backtrader订单、现金、下一收盘时序和缺价边界。"""
import unittest

import numpy as np
import pandas as pd

from etf_rotation.backtest import run_backtest
from etf_rotation.backtrader_engine import run_backtrader


class BacktraderEngineTests(unittest.TestCase):
    def prices(self):
        d=pd.bdate_range("2022-01-03",periods=10)
        return pd.DataFrame({"510300":[10,11,12,13,12,11,12,11,10,12],
                             "511260":[100,100,101,101,102,102,102,103,103,104]},index=d,dtype=float)

    def test_zero_cost_native_broker_matches_drift_and_next_close(self):
        p=self.prices();d=p.index
        s=pd.DataFrame([[.5,.5],[0,1]],index=d[[0,8]],columns=p.columns)
        slow=run_backtest(p,s,d[0].date(),d[-1].date(),fee_bps=0,slippage_bps=0)
        result,orders,meta=run_backtrader(p,s,d[0].date(),d[-1].date(),fee_bps=0,slippage_bps=0)
        np.testing.assert_allclose(result.equity.strategy_nav,slow.equity.strategy_nav,atol=1e-9)
        self.assertEqual(result.weights.loc[d[0],"CASH"],1.)
        self.assertAlmostEqual(result.weights.loc[d[1],"510300"],.5,places=8)
        self.assertAlmostEqual(result.weights.loc[d[-1],"511260"],1.,places=8)
        filled=orders[orders.status.eq("Completed")]
        self.assertTrue((pd.to_datetime(filled.execution_date)>pd.to_datetime(filled.signal_date)).all())
        self.assertEqual(meta["orders_margin_or_rejected"],0)
        self.assertLess(meta["native_timereturn_max_error"],1e-12)
        self.assertAlmostEqual(meta["native_drawdown"],result.metrics["max_drawdown"],places=10)

    def test_commission_is_paid_from_actual_broker_cash(self):
        p=self.prices().iloc[:3].copy();p["510300"]=10.;p["511260"]=100.
        s=pd.DataFrame([[1.,0.]],columns=p.columns,index=p.index[[0]])
        result,orders,_=run_backtrader(p,s,p.index[0].date(),p.index[-1].date(),fee_bps=10,slippage_bps=0)
        # 满仓实际成交额+手续费=初始资金，而不是无成本更新持仓或注入现金。
        expected=1/(1+.001)
        self.assertAlmostEqual(result.equity.strategy_nav.iloc[1],expected,places=9)
        filled=orders[orders.status.eq("Completed")]
        cost=filled.commission.sum()
        self.assertAlmostEqual(cost,float(result.equity.commission.sum()),places=8)
        self.assertGreaterEqual(result.equity.cash.min(),0.)

    def test_cross_start_signal_executes_on_first_evaluation_day(self):
        p=self.prices();d=p.index
        s=pd.DataFrame([[.4,.4]],columns=p.columns,index=d[[1]])
        result,orders,_=run_backtrader(p,s,d[2].date(),d[-1].date(),fee_bps=0,slippage_bps=0)
        self.assertAlmostEqual(result.equity.strategy_nav.iloc[0],1.,places=12)
        self.assertAlmostEqual(result.weights["510300"].iloc[0],.4,places=8)
        self.assertTrue(orders.execution_date.eq(d[2]).all())

    def test_missing_execution_quote_cancels_whole_batch_in_both_modes(self):
        p=self.prices();d=p.index;p.loc[d[1],"510300"]=np.nan
        s=pd.DataFrame([[.3,.3]],columns=p.columns,index=d[[0]])
        for mode in ("target_close","fixed_close"):
            result,orders,_=run_backtrader(p,s,d[0].date(),d[-1].date(),mode=mode)
            self.assertTrue(result.weights.CASH.eq(1.).all())
            self.assertEqual(int(orders.status.eq("Completed").sum()),0)
            self.assertEqual(result.trades.status.iloc[0],"skipped_no_quote")

    def test_placeholder_prices_cannot_buy_before_first_real_quote(self):
        p=self.prices();d=p.index;p["159875"]=np.nan;p.loc[d[5]:,"159875"]=2.
        s=pd.DataFrame([[0.,0.,.5]],columns=p.columns,index=d[[0]])
        result,orders,_=run_backtrader(p,s,d[0].date(),d[-1].date())
        self.assertTrue(result.weights["159875"].eq(0).all())
        self.assertEqual(int(orders.status.eq("Completed").sum()),0)

    def test_long_missing_held_quote_stops_instead_of_free_valuation(self):
        p=self.prices();d=p.index;p.loc[d[3:8],"510300"]=np.nan
        s=pd.DataFrame([[.5,0]],columns=p.columns,index=d[[0]])
        with self.assertRaises(ValueError):
            run_backtrader(p,s,d[0].date(),d[-1].date())

    def test_fixed_close_order_locks_previous_day_units(self):
        p=self.prices();d=p.index
        s=pd.DataFrame([[.3,.2]],columns=p.columns,index=d[[0]])
        result,orders,meta=run_backtrader(p,s,d[0].date(),d[-1].date(),fee_bps=0,slippage_bps=0,mode="fixed_close")
        stock=orders[(orders.symbol=="510300")&orders.status.eq("Completed")].iloc[0]
        self.assertAlmostEqual(stock["size"],1_000_000*.3/10,delta=1e-5)
        self.assertEqual(stock.execution_date,d[1])
        self.assertEqual(stock.price,11.)
        self.assertEqual(meta["orders_margin_or_rejected"],0)

    def test_future_prices_do_not_rewrite_past_backtrader_equity(self):
        p=self.prices();d=p.index
        s=pd.DataFrame([[.4,.4],[.2,.6]],columns=p.columns,index=d[[0,6]])
        original,_,_=run_backtrader(p,s,d[0].date(),d[-1].date())
        changed=p.copy();changed.loc[d[5]:]*=2.5
        altered,_,_=run_backtrader(changed,s,d[0].date(),d[-1].date())
        pd.testing.assert_series_equal(original.equity.strategy_nav.loc[:d[4]],altered.equity.strategy_nav.loc[:d[4]])


if __name__=="__main__":unittest.main()
