"""宽基图使用同一起点，短缺价仅沿用旧价估值。"""

import unittest

import pandas as pd

from plot_benchmarks import BENCHMARKS, buy_hold_nav, incremental_attribution, make_comparison


class BenchmarkPlotTests(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.bdate_range("2024-01-01", periods=6)
        self.equity = pd.DataFrame({"strategy_nav": [0.9995, 1.01, 1.02, 1.03, 1.04, 1.05]},
                                   index=dates)
        self.prices = {code: pd.Series([1.0, 1.01, 1.02, 1.03, 1.04, 1.05], index=dates)
                       for code in BENCHMARKS}

    def test_keeps_initial_strategy_cost_and_fills_one_missing_quote_from_past(self) -> None:
        missing_day = self.equity.index[2]
        self.prices["512100"] = self.prices["512100"].drop(missing_day)
        nav = make_comparison(self.equity, self.prices)
        self.assertAlmostEqual(nav.iloc[0]["策略"], 0.9995)
        self.assertAlmostEqual(nav.loc[missing_day, "中证1000"], 1.01)
        self.assertAlmostEqual(nav.iloc[3]["中证1000"], 1.03)

    def test_rejects_long_missing_quote_run(self) -> None:
        self.prices["512100"] = self.prices["512100"].iloc[:2]
        with self.assertRaises(ValueError):
            make_comparison(self.equity, self.prices)

    def test_technology_comparator_keeps_shares_and_initial_cost(self) -> None:
        prices=pd.DataFrame({'a':[1.,2.,2.],'b':[1.,1.,2.]},index=self.equity.index[:3])
        nav=buy_hold_nav(prices,{'a':.5,'b':.5})
        self.assertAlmostEqual(nav.iloc[0],.9995)
        self.assertAlmostEqual(nav.iloc[1],1.5*.9995)
        self.assertAlmostEqual(nav.iloc[2],2.*.9995)

    def test_incremental_attribution_equals_realized_wealth_difference(self) -> None:
        import numpy as np
        from etf_rotation.backtest import run_backtest
        days=pd.bdate_range('2024-01-01',periods=8)
        prices=pd.DataFrame({'510300':[100,102,106,95,110,115,116,111],
                             '511260':[100,100.3,100.5,100.4,100.8,101,101.1,101.3],
                             '512890':[100,110,90,120,135,110,130,125]},index=days)
        base=pd.DataFrame([[.5,.5,0.],[.5,.5,0.]],index=days[[0,5]],columns=prices.columns)
        added=base.copy();added.iloc[0]=[.5,.2,.3]
        old=run_backtest(prices,base,days[0].date(),days[-1].date())
        new=run_backtest(prices,added,days[0].date(),days[-1].date())
        daily,accumulated=incremental_attribution(prices,new,old)
        np.testing.assert_allclose(daily.sum(axis=1),new.equity.daily_return-old.equity.daily_return,atol=1e-12)
        np.testing.assert_allclose(accumulated.net_nav_difference,new.equity.strategy_nav-old.equity.strategy_nav,atol=1e-12)
        np.testing.assert_allclose(accumulated.drop(columns='net_nav_difference').sum(axis=1),accumulated.net_nav_difference,atol=1e-12)


if __name__ == "__main__":
    unittest.main()
