"""相关筛选只能删正高相关者；资格过滤不能改写弱市判断。"""
import unittest
import numpy as np
import pandas as pd
from research_correlation_audit import greedy_uncorrelated
from etf_rotation.risk_strategy import risk_controlled_signals,RiskConfig


class CorrelationAuditTests(unittest.TestCase):
    def test_original_research_pool_is_frozen_after_sp500_removal(self):
        from research_correlation_audit import ETF_NAMES as frozen_names,static_cases
        from etf_rotation.config import ETF_NAMES as current_names
        self.assertEqual(len(frozen_names),18)
        self.assertEqual(len(static_cases())+1,41)  # 静态40场景＋动态筛除。
        self.assertEqual(set(frozen_names)-{'513500'},set(current_names))

    def test_keeps_strongest_positive_cluster_and_negative_hedge(self):
        c=pd.DataFrame([[1.,.9,-.95],[.9,1.,-.9],[-.95,-.9,1.]],index=['A','B','H'],columns=['A','B','H'])
        self.assertEqual(greedy_uncorrelated(['A','B','H'],c),['A','H'])
        self.assertEqual(greedy_uncorrelated(['B','A','H'],c),['B','H'])

    def test_missing_history_does_not_create_fake_correlation(self):
        c=pd.DataFrame([[1.,np.nan],[np.nan,1.]],index=['A','B'],columns=['A','B'])
        self.assertEqual(greedy_uncorrelated(['A','B'],c),['A','B'])

    def test_eligibility_mask_keeps_benchmark_regime_and_bond_defense(self):
        days=pd.bdate_range('2024-01-01',periods=200);x=np.arange(200)
        p=pd.DataFrame({'510300':np.exp(-.001*x),'511260':np.exp(.0001*x),
                        'A':np.exp(.002*x+.001*np.sin(x/5)),'B':np.exp(.0015*x+.001*np.sin(x/7))},index=days)
        mask=pd.DataFrame(True,index=days,columns=p.columns);mask['510300']=False
        cfg=RiskConfig.current_default()
        original=risk_controlled_signals(p,cfg);filtered=risk_controlled_signals(p,cfg,eligibility_mask=mask)
        pd.testing.assert_frame_equal(original,filtered)
        self.assertLessEqual(filtered.drop(columns='511260').sum(axis=1).max(),.35+1e-10)
        self.assertGreater(filtered['511260'].iloc[-1],.6)


if __name__=='__main__':unittest.main()
