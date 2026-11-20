"""扩池研究须覆盖新增股票IVOL、动态准入、交易名额和因果边界。"""
import unittest
import numpy as np
import pandas as pd

import research_cl120_pool_expansion as research


class CL120PoolExpansionTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(410)
        days=pd.bdate_range('2020-01-01',periods=340)
        market=rng.normal(.0005,.006,len(days))
        cols=[*research.CORE_NAMES,'512400','512070','159938']
        returns=np.column_stack([.0004+(1+i*.015)*market+rng.normal(0,.002+i*.0003,len(days)) for i in range(len(cols))])
        self.prices=pd.DataFrame(np.exp(np.cumsum(returns,axis=0)),index=days,columns=cols)
        self.prices['510300']=np.exp(np.cumsum(market));self.prices['512890']=np.linspace(1,1.02,len(days))
        # 新增有色股票的特有波动明显高，必须能触发过滤，不能绕过新规则。
        self.prices['512400']=np.exp(np.cumsum(.0005+market+rng.normal(0,.04,len(days))))

    def test_new_equity_is_subject_to_ivol_and_never_bought_when_filtered(self):
        case=research.Case('add_512400',('512400',))
        _,_,core,mask=research.build_targets(self.prices,case)
        self.assertTrue((~mask['512400']).any())
        self.assertTrue(core.loc[~mask['512400'].reindex(core.index),'512400'].eq(0).all())
        self.assertTrue(mask['510300'].all())

    def test_research_and_production_targets_agree_for_add_and_replace(self):
        for case in (research.Case('default'),research.Case('add_512400',('512400',)),
                     research.Case('replace',('512070',),('512880',))):
            with self.subTest(case=case.name):
                _,fast,_,_=research.build_targets(self.prices,case)
                _,slow,_,_=research.build_targets(self.prices,case,production=True)
                pd.testing.assert_frame_equal(fast,slow,atol=1e-11,rtol=1e-11)
                if case.removed:self.assertNotIn('512880',fast)

    def test_new_asset_not_selectable_diagnostic_cannot_buy_it(self):
        case=research.Case('add_159938',('159938',))
        _,targets,_,mask=research.build_targets(self.prices,case,diagnostic='new_assets_not_selectable')
        self.assertTrue(mask['159938'].eq(False).all())
        self.assertTrue(targets['159938'].eq(0).all())

    def test_unlisted_asset_does_not_change_existing_targets(self):
        prices=self.prices.copy();start=prices.index[180]
        prices.loc[prices.index<start,'159938']=np.nan
        _,old,_,_=research.build_targets(prices,research.Case('default'))
        _,new,_,_=research.build_targets(prices,research.Case('add',('159938',)))
        pd.testing.assert_frame_equal(new.loc[new.index<start,old.columns],old.loc[old.index<start])
        self.assertTrue(new.loc[new.index<start,'159938'].eq(0).all())

    def test_prefix_and_future_perturbation_invariance(self):
        cut=research.weekly_signal_dates(self.prices.index)[50]
        case=research.Case('add',('512400','159938'))
        _,full,_,_=research.build_targets(self.prices,case)
        _,prefix,_,_=research.build_targets(self.prices.loc[:cut],case)
        changed=self.prices.copy();changed.loc[changed.index>cut]*=np.linspace(.2,4,len(changed.columns))
        _,future,_,_=research.build_targets(changed,case)
        pd.testing.assert_frame_equal(full.loc[:cut],prefix)
        pd.testing.assert_frame_equal(full.loc[:cut],future.loc[:cut])

    def test_dynamic_admission_excludes_unavailable_assets_from_peer_group(self):
        case=research.Case('add',('159938',))
        admission=pd.DataFrame(True,index=self.prices.index,columns=self.prices.columns)
        admission.loc[:,'159938']=False
        _,actual,core,mask=research.build_targets(self.prices,case,allowed=admission)
        _,old,_,_=research.build_targets(self.prices,research.Case('default'))
        self.assertTrue(core['159938'].eq(0).all())
        self.assertTrue(mask['159938'].eq(False).all())
        pd.testing.assert_frame_equal(actual[old.columns],old)


if __name__=='__main__':unittest.main()
