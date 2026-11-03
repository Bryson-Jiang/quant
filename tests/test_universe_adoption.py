"""扩池先预热、后启用；不得改写既有周目标或提前改变截面缩放。"""
import unittest
import json
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from etf_rotation.config import ETF_NAMES,LEGACY_CORE_NAMES,LEGACY_PORTFOLIO_STRATEGY_NAME
from etf_rotation.risk_strategy import RiskConfig,risk_controlled_signals,weekly_signal_dates
from etf_rotation.strategy_state import UniversePolicy,load_rotation_state
from live_signal import signal_score_details


class UniverseAdoptionTests(unittest.TestCase):
    def setUp(self):
        self.days=pd.bdate_range('2024-01-01',periods=430)
        x=np.arange(len(self.days));self.prices=pd.DataFrame(index=self.days)
        for i,s in enumerate(ETF_NAMES):
            self.prices[s]=np.exp((.0002 if s=='511260' else .0007)*x+.01*np.sin(x/(4+i)))
        self.prices['513180']=np.exp(.004*x+.015*np.sin(x/4))
        self.effective=weekly_signal_dates(self.days)[65]
        self.activation={'513180':self.effective}
        self.cfg=RiskConfig.current_default()

    def test_original_targets_and_dispersion_unchanged_before_activation(self):
        old=risk_controlled_signals(self.prices[list(LEGACY_CORE_NAMES)],self.cfg)
        new=risk_controlled_signals(self.prices,self.cfg,universe_activation_dates=self.activation)
        before=new.index<self.effective
        pd.testing.assert_frame_equal(new.loc[before,list(LEGACY_CORE_NAMES)],old.loc[before])
        self.assertTrue(new.loc[before,['513180']].eq(0).all().all())
        self.assertGreater(new.at[self.effective,'513180'],0)

    def test_activation_preserves_factor_warmup_and_future_invariance(self):
        full=risk_controlled_signals(self.prices,self.cfg,universe_activation_dates=self.activation)
        cut=self.effective
        changed=self.prices.copy();changed.loc[changed.index>cut]*=4
        altered=risk_controlled_signals(changed,self.cfg,universe_activation_dates=self.activation)
        prefix=risk_controlled_signals(self.prices.loc[:cut],self.cfg,universe_activation_dates=self.activation)
        pd.testing.assert_frame_equal(full.loc[:cut],altered.loc[:cut])
        pd.testing.assert_frame_equal(full.loc[:cut],prefix)
        hypothetical=risk_controlled_signals(self.prices,self.cfg)
        self.assertTrue(hypothetical.loc[hypothetical.index<cut,['513180']].gt(0).any().any())

    def test_score_explanation_uses_same_activation_mask(self):
        targets=risk_controlled_signals(self.prices,self.cfg,universe_activation_dates=self.activation)
        for day in (targets.index[60],self.effective):
            rows=signal_score_details(self.prices,self.cfg,targets.loc[day],universe_activation_dates=self.activation)
            qualified=sorted([r for r in rows if r['eligible']],key=lambda r:r['rank'])
            self.assertGreater(targets.at[day,qualified[0]['symbol']],0)
            if day<self.effective:
                self.assertTrue(all(not r['eligible'] for r in rows if r['symbol'] in self.activation))

    def test_current_and_frozen_states_have_distinct_universes(self):
        root=Path(__file__).resolve().parents[1]
        current=load_rotation_state(root/'data/rotation_state.json')
        old=load_rotation_state(root/'data/rotation_state_20261031.json',expected_strategy=LEGACY_PORTFOLIO_STRATEGY_NAME)
        self.assertEqual(len(current.core_symbols),17);self.assertEqual(len(old.core_symbols),16)
        self.assertNotIn('513500',current.core_symbols)
        self.assertEqual(set(current.universe_activation_dates()),{'513180'})
        self.assertEqual(current.universe_activation_dates(hypothetical=True),{})
        self.assertEqual(old.universe_activation_dates(),{})

    def test_pending_policy_cannot_restore_removed_sp500(self):
        root=Path(__file__).resolve().parents[1]
        payload=json.loads((root/'data/rotation_state.json').read_text(encoding='utf-8'))
        payload['universe_policy']['added_symbols'].append('513500')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.json'
            path.write_text(json.dumps(payload),encoding='utf-8')
            with self.assertRaises(ValueError):load_rotation_state(path)


if __name__=='__main__':unittest.main()
