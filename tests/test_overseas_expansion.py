"""跨境研究不得提前使用尚未公布的NAV，不得破坏原切换状态和仓位约束。"""
import unittest
from datetime import date

import numpy as np
import pandas as pd

from etf_rotation.strategy_state import RotationState
from research_overseas_expansion import cap_foreign,filtered_state_kwargs,known_nav


class OverseasExpansionTests(unittest.TestCase):
    def test_nav_available_by_announcement_date_not_valuation_date(self):
        nav=pd.DataFrame({'nav_date':['20260929','20260930','20260930'],
                          'ann_date':['20260930','20261002','20261005'],
                          'unit_nav':[1.,1.1,1.2]})
        self.assertEqual(known_nav(nav,pd.Timestamp('2026-09-30')).unit_nav,1.)
        self.assertEqual(known_nav(nav,pd.Timestamp('2026-10-03')).unit_nav,1.1)
        self.assertEqual(known_nav(nav,pd.Timestamp('2026-10-06')).unit_nav,1.2)
        with self.assertRaises(ValueError):known_nav(nav,pd.Timestamp('2026-09-28'))

    def test_replacement_filters_public_seed_without_mutation(self):
        p=pd.DataFrame(1.,index=pd.bdate_range('2026-09-01','2026-10-20'),columns=['510300','511260','513500','512800'])
        state=RotationState(date(2026,10,9),date(2026,10,16),('159941','512800'))
        result=filtered_state_kwargs(state,p)
        self.assertEqual(result['initial_risky_selection'],['512800'])
        self.assertEqual(state.initial_risky_selection,('159941','512800'))

    def test_foreign_cap_reduces_only_foreign_returns_to_existing_defense(self):
        t=pd.DataFrame([[.3,.3,.1,.3],[.3,.3,.1,0.]],
                       index=pd.bdate_range('2026-10-15',periods=2),columns=['159941','513500','510300','511260'])
        original=t.copy(deep=True);c=cap_foreign(t)
        np.testing.assert_allclose(c[['159941','513500']].sum(axis=1),.3)
        self.assertEqual(c.iloc[0]['511260'],.6)
        self.assertEqual(c.iloc[1]['511260'],0.)
        pd.testing.assert_series_equal(c['510300'],t['510300'])
        self.assertTrue(c.sum(axis=1).le(1+1e-12).all())
        pd.testing.assert_frame_equal(t,original)


if __name__=='__main__':unittest.main()
