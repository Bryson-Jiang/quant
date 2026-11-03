"""行业准入必须依赖过去资料，不因未来规模/成交额变化改写历史。"""
import unittest

import numpy as np
import pandas as pd

from research_pool_expansion import historical_eligibility


class PoolExpansionTests(unittest.TestCase):
    def setUp(self):
        self.days=pd.bdate_range('2020-01-01',periods=300)
        self.price=pd.Series(1.,index=self.days)
        self.amount=pd.Series(20_000_000.,index=self.days)
        self.size=pd.Series(300_000_000.,index=self.days)

    def test_maturity_counts_real_quotes_not_calendar_rows(self):
        price=self.price.copy();price.iloc[:10]=np.nan
        allowed=historical_eligibility(price,self.amount,self.size)
        self.assertFalse(allowed.iloc[:261].any())
        self.assertTrue(allowed.iloc[261])

    def test_size_is_lagged_and_missing_current_quote_is_blocked(self):
        size=self.size.copy();size.iloc[270]=1.
        result=historical_eligibility(self.price,self.amount,size)
        self.assertTrue(result.iloc[270])
        self.assertFalse(result.iloc[271])
        price=self.price.copy();price.iloc[275]=np.nan
        self.assertFalse(historical_eligibility(price,self.amount,self.size).iloc[275])

    def test_future_data_and_truncation_do_not_change_past_eligibility(self):
        cutoff=self.days[275]
        base=historical_eligibility(self.price,self.amount,self.size)
        amount=self.amount.copy();size=self.size.copy()
        amount.loc[amount.index>cutoff]=0.;size.loc[size.index>cutoff]=0.
        changed=historical_eligibility(self.price,amount,size)
        prefix=historical_eligibility(self.price.loc[:cutoff],self.amount.loc[:cutoff],self.size.loc[:cutoff])
        pd.testing.assert_series_equal(base.loc[:cutoff],changed.loc[:cutoff])
        pd.testing.assert_series_equal(base.loc[:cutoff],prefix)


if __name__=='__main__':
    unittest.main()
