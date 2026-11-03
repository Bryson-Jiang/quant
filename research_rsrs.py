"""RSRS评分研究；保持默认策略、真实持仓和原实盘信号不变。

首次运行 python research_rsrs.py --download
离线复现 python research_rsrs.py
重建报告 python research_rsrs.py --report-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict,dataclass,replace
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from etf_rotation.backtest import run_backtest
from etf_rotation.config import LEGACY_CORE_NAMES
from etf_rotation.data import _token,_download_table,load_symbol,tushare_code
from etf_rotation.factors import FactorSpec,RSRSSpec,compute_factor,compute_rsrs
from etf_rotation.factor_store import FactorStore
from etf_rotation.legacy_research import load_rotation_state
from etf_rotation.mean_reversion import ReversionRule,reversion_states,defensive_funding_targets
from etf_rotation.ohlc_data import adjusted_high_low
from etf_rotation.portfolio_strategy import portfolio_signals
from etf_rotation.risk_strategy import RiskConfig,risk_controlled_signals,risk_score_components,weekly_signal_dates
from research_overfitting_audit import CoreAudit,simulate,statistics
from research_pool_expansion import protected_fingerprints,paired_bootstrap,markdown_table

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/rsrs_20261031';CACHE=ROOT/'data/rsrs_20261031'
CLOSE_CACHE=ROOT/'data/overseas_expansion_20261031';END=date(2026,10,20);SAT='512890'
ETF_NAMES={**LEGACY_CORE_NAMES,'513180':'恒生科技ETF（华夏）'}
BASE=RiskConfig(target_vol=.08,short_momentum_days=20,long_momentum_days=40,momentum_skip_days=0,
                trend_days=40,max_single_risky_weight=.30,rank_buffer_size=4,mandatory_top_rank=1,
                ma_reversion_days=20,ma_reversion_strength=.5)
N_VALUES=(10,18,30,60);M_VALUES=(120,250,600)
MOMENTUM_PAIRS=((10,20),(20,40),(40,80),(60,120))
VARIANTS=('zscore','adjusted','right_skew')


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8')


@dataclass(frozen=True)
class Case:
    mode:str='baseline'
    n:int=18
    m:int=600
    variant:str='adjusted'
    strength:float=.25
    short:int=20
    long:int=40

    @property
    def key(self):
        if self.mode=='baseline':return 'default' if (self.short,self.long)==(20,40) else f'momentum_{self.short}_{self.long}'
        return f'{self.mode}_{self.variant}_N{self.n}_M{self.m}_a{self.strength:g}_mom{self.short}_{self.long}'


def cases():
    entries=[Case(short=a,long=b) for a,b in MOMENTUM_PAIRS]
    for n in N_VALUES:
        for m in M_VALUES:
            entries.extend(Case('replace',n,m,v) for v in VARIANTS)
            entries.extend(Case('blend',n,m,strength=w) for w in (.25,.5))
            entries.append(Case('maturity',n,m))
    entries.extend(Case('filter',n,m) for n in (18,30,60) for m in (250,600))
    for a,b in MOMENTUM_PAIRS:
        if (a,b)!=(20,40):
            entries.append(Case('replace',short=a,long=b))
            entries.extend(Case('blend',strength=w,short=a,long=b) for w in (.25,.5))
    return list({c.key:c for c in entries}.values())


def plan():
    return {'research_date':'2026-10-31','data_end':END,'universe':ETF_NAMES,'base_config':asdict(BASE),
      'case_count':len(cases()),'cases':[{'key':c.key,**asdict(c)} for c in cases()],
      'design':'N=10/18/30/60、M=120/250/600；Z、Z*R²、Z*R²*beta替换评分；修正版强度.25/.5叠加；>0资格过滤；12个相同预热资格的原评分对照；10/20、20/40、40/80、60/120收益动量对照',
      'baseline':'原16只+恒科，不含标普；核心及红利均全历史假设，非真实账户或实际启用后业绩',
      'frozen_rules':'周末最后交易日信号，下一交易日收盘；40日趋势过滤、60日波动及协方差、8%核心目标、70%核心风险上限、单只30%、3只核心风险、前4缓冲必选1；红利MA200入0出4先现金再国债30%',
      'replace':'仅替换最终横截面排序；仍需原收益动量long>0及MA40资格，缺RSRS时不能入选',
      'blend':'score_old + strength * D_old * clip(RSRS_adjusted/3,-1,1)；缺RSRS时原评分回退',
      'filter':'原评分排序且RSRS_adjusted>0，缺RSRS时不入选',
      'warmup':'完整N日高低价，再完整M日斜率；Z总体标准差ddof0、窗口含信号日；不按上市前指数补历史',
      'costs':[5,20],'starts':['2018-01-01','2022-01-01'],
      'selection':'全部历史已被观察；按时间先选参数再看后段仍非独立盲测；不自动切换默认',
      'protected_before':protected_fingerprints()}


def download():
    CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    pro=None;status=[]
    for symbol in ETF_NAMES:
        path=CACHE/f'daily_{symbol}.csv'
        try:
            if not path.exists():
                old=CLOSE_CACHE/f'daily_{symbol}.csv'
                if old.exists():daily=pd.read_csv(old,dtype={'trade_date':str})
                else:
                    if pro is None:pro=ResearchClient()
                    daily=_download_table(pro,'fund_daily',tushare_code(symbol),date(2010,1,1),END)
                if daily.empty:raise ValueError('Empty daily quotes')
                daily.to_csv(path,index=False,encoding='utf-8-sig')
            daily=pd.read_csv(path,dtype={'trade_date':str})
            history=pd.read_csv(CLOSE_CACHE/f'tushare_{symbol}_hfq.csv')
            history=history.loc[pd.to_datetime(history.date).le(pd.Timestamp(END))]
            adjusted=adjusted_high_low(daily,history,symbol)
            adjusted.to_csv(CACHE/f'high_low_{symbol}.csv',encoding='utf-8-sig')
            status.append({'symbol':symbol,'rows':len(adjusted),'first':str(adjusted.index[0].date()),
                           'last':str(adjusted.index[-1].date()),'same_snapshot_close':True})
            write_json(OUT/'download_status.json',status)
            print(f'data {symbol}: {len(adjusted)} paired high/low quotes',flush=True)
        except Exception as exc:
            # API异常可能含请求信息，只披露类型，禁止把凭据写入日志或报告。
            status.append({'symbol':symbol,'failed':type(exc).__name__})
            write_json(OUT/'download_status.json',status)
            raise RuntimeError(f'{symbol}: high/low preparation failed ({type(exc).__name__}); details withheld') from None


class ResearchClient:
    """官方REST契约；研究专用HTTPS入口，避开当前SDK不可达的waditu地址。

    不修改SDK、全局_client、代理或TLS验证；不接受自定义凭据目的地址。
    """
    def __init__(self):
        self.session=requests.Session()
        self.token=_token()

    def fund_daily(self,**params):
        response=self.session.post('https://api.tushare.pro',json={
            'api_name':'fund_daily','token':self.token,'params':params,
            'fields':'ts_code,trade_date,high,low,close'},timeout=30,allow_redirects=False)
        response.raise_for_status()
        value=response.json()
        if value.get('code')!=0:raise RuntimeError('Tushare query failed; response details withheld')
        return pd.DataFrame(value['data']['items'],columns=value['data']['fields'])


def read_data():
    close={s:load_symbol(s,date(2010,1,1),END,CLOSE_CACHE,offline=True) for s in [*ETF_NAMES,SAT]}
    prices=pd.DataFrame(close,index=close['510300'].index).rename_axis('date')
    highs={};lows={}
    for symbol in ETF_NAMES:
        daily=pd.read_csv(CACHE/f'daily_{symbol}.csv',dtype={'trade_date':str})
        history=pd.read_csv(CLOSE_CACHE/f'tushare_{symbol}_hfq.csv')
        history=history.loc[pd.to_datetime(history.date).le(pd.Timestamp(END))]
        frame=adjusted_high_low(daily,history,symbol)
        highs[symbol]=frame.high;lows[symbol]=frame.low
    return prices,pd.DataFrame(highs,index=prices.index),pd.DataFrame(lows,index=prices.index)


def case_inputs(core,engine,case,rsrs):
    cfg=replace(BASE,short_momentum_days=case.short,long_momentum_days=case.long)
    f={FactorSpec(name,window):pd.DataFrame(engine.factor(name,window),index=core.index,columns=core.columns)
       for name,window in [('momentum',case.short),('momentum',case.long),('volatility',60),('ma_zscore',20)]}
    comp=risk_score_components(f[FactorSpec('momentum',case.short)],f[FactorSpec('momentum',case.long)],
                               f[FactorSpec('volatility',60)],cfg,f[FactorSpec('ma_zscore',20)])
    score=comp['score'];allowed=None
    if case.mode!='baseline':
        value=rsrs[(case.n,case.m)][case.variant]
        if case.mode=='replace':score=value;allowed=value.notna()
        elif case.mode=='blend':
            spread=comp['base_score'].drop(columns='511260').std(axis=1,ddof=1).fillna(0)
            score=score+value.div(3).clip(-1,1).fillna(0).mul(spread*case.strength,axis=0)
        elif case.mode=='filter':allowed=value.gt(0)&value.notna()
        elif case.mode=='maturity':allowed=value.notna()
        else:raise ValueError('Unknown research mode')
    return cfg,score,allowed,comp['score']


def research_targets(prices,high,low,state,case):
    """用于截断/未来扰动检查；从原始高低价重算因子与全部状态。"""
    core=prices[list(ETF_NAMES)];engine=CoreAudit(core,state)
    rsrs={(case.n,case.m):compute_rsrs(high,low,RSRSSpec(case.n,case.m))}
    cfg,score,allowed,_=case_inputs(core,engine,case,rsrs)
    target=engine.signals(cfg,score_override=score,eligibility_mask=allowed)
    states=reversion_states(prices[[SAT]],SAT,ReversionRule(200,0,.04,None),weekly_signal_dates(prices.index))
    return defensive_funding_targets(prices,target,states,SAT,.30)


def factor_diagnostics(prices,rsrs,baseline_score):
    risky=[s for s in ETF_NAMES if s!='511260'];days=weekly_signal_dates(prices.index)
    days=days[days>=pd.Timestamp('2018-01-01')]
    vol=compute_factor(prices[list(ETF_NAMES)],FactorSpec('volatility',60))
    summaries=[];ic_rows=[];by_symbol=[]
    # 每个标签都从下一交易日收盘入场，完整h日后收盘离场，末端不足不使用。
    for (n,m),bundle in rsrs.items():
        for variant in VARIANTS:
            value=bundle[variant][risky]
            for horizon in (5,20):
                future=prices[risky].shift(-(horizon+1))/prices[risky].shift(-1)-1
                sample=[];outcomes=[];ics=[];base_ics=[]
                for day in days:
                    valid=value.loc[day].notna()&future.loc[day].notna()&vol.loc[day,risky].gt(0)
                    names=valid.index[valid]
                    if len(names)<5:continue
                    x=value.loc[day,names];y=future.loc[day,names]
                    if x.std()==0 or y.std()==0:continue
                    rank_ic=float(x.rank().corr(y.rank()));base=baseline_score.loc[day,names]
                    base_ic=float(base.rank().corr(y.rank()))
                    ics.append(rank_ic);base_ics.append(base_ic)
                    ic_rows.append({'date':day,'n':n,'m':m,'variant':variant,'horizon':horizon,
                                    'rank_ic':rank_ic,'baseline_matched_ic':base_ic,'assets':len(names)})
                    pos=prices.index.get_loc(day);end_day=prices.index[pos+horizon+1]
                    for symbol in names:
                        sample.append({'date':day,'label_end':end_day,'symbol':symbol,
                                       'x':float(value.at[day,symbol]),'y':float(y[symbol])})
                samples=pd.DataFrame(sample)
                sse=sse_null=0.;tested=0
                if not samples.empty:
                    for year in range(2019,2027):
                        left=pd.Timestamp(year,1,1);right=pd.Timestamp(year+1,1,1)
                        train=samples.loc[samples.label_end<left];test=samples.loc[(samples.date>=left)&(samples.date<right)]
                        if len(train)<200 or test.empty:continue
                        coef=np.linalg.lstsq(np.column_stack([np.ones(len(train)),train.x]),train.y,rcond=None)[0]
                        pred=coef[0]+coef[1]*test.x
                        sse+=float(np.square(test.y-pred).sum())
                        sse_null+=float(np.square(test.y-train.y.mean()).sum());tested+=len(test)
                    for symbol,g in samples.groupby('symbol'):
                        by_symbol.append({'n':n,'m':m,'variant':variant,'horizon':horizon,'symbol':symbol,
                                          'observations':len(g),'time_series_rank_ic':float(g.x.rank().corr(g.y.rank()))})
                summaries.append({'n':n,'m':m,'variant':variant,'horizon':horizon,'weeks':len(ics),
                    'mean_rank_ic':float(np.mean(ics)) if ics else None,
                    'matched_baseline_rank_ic':float(np.nanmean(base_ics)) if base_ics else None,
                    'ic_std':float(np.std(ics,ddof=1)) if len(ics)>1 else None,
                    'positive_ic_fraction':float(np.mean(np.asarray(ics)>0)) if ics else None,
                    'predictive_oos_r2':1-sse/sse_null if sse_null else None,'oos_observations':tested,
                    'mean_price_fit_r2':float(bundle['r_squared'][risky].loc['2018':].stack().mean())})
    pd.DataFrame(summaries).to_csv(OUT/'factor_diagnostics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(ic_rows).to_csv(OUT/'weekly_factor_ic.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(by_symbol).to_csv(OUT/'factor_ic_by_etf.csv',index=False,encoding='utf-8-sig')


def execute():
    before=protected_fingerprints();prices,high,low=read_data();core=prices[list(ETF_NAMES)]
    state=replace(load_rotation_state(ROOT/'data/rotation_state.json'),core_symbols=tuple(ETF_NAMES))
    engine=CoreAudit(core,state);entries=cases();rsrs={}
    for n in N_VALUES:
        for m in M_VALUES:rsrs[n,m]=compute_rsrs(high,low,RSRSSpec(n,m))
    sat=reversion_states(prices[[SAT]],SAT,ReversionRule(200,0,.04,None),weekly_signal_dates(prices.index))
    records=[];daily={};signals={};checks=[];holding=[];coverage=[]
    cash_start_daily={start:{} for start in ('2018-01-01','2022-01-01')}
    for (n,m),frames in rsrs.items():
        for s in ETF_NAMES:
            valid=frames['adjusted'][s].dropna()
            coverage.append({'n':n,'m':m,'symbol':s,'first_valid':str(valid.index[0].date()) if len(valid) else '',
                             'valid_days':len(valid),'last_valid':str(valid.index[-1].date()) if len(valid) else ''})
    pd.DataFrame(coverage).to_csv(OUT/'factor_coverage.csv',index=False,encoding='utf-8-sig')
    baseline_score=None
    for i,case in enumerate(entries):
        cfg,score,allowed,original=case_inputs(core,engine,case,rsrs)
        t=engine.signals(cfg,score_override=score,eligibility_mask=allowed)
        target=defensive_funding_targets(prices,t,sat,SAT,.30);signals[case.key]=target
        if case.key=='default':baseline_score=score
        for start in ('2018-01-01','2022-01-01'):
            for cost in (5.,20.):
                led=simulate(prices,target,pd.Timestamp(start),cost_bps=cost)
                records.append({'case':case.key,**asdict(case),'start':start,'cost':cost,'skipped':led.skipped,
                   'annual_turnover':float(led.turnover.sum()*252/len(led.dates)),**statistics(led.returns)})
                if cost==5:cash_start_daily[start][case.key]=led.series()
                if start=='2018-01-01' and cost==5:
                    daily[case.key]=led.series()
                    if case.mode!='maturity':
                        weights=pd.DataFrame(led.weights,index=led.dates,columns=prices.columns)
                        for s in prices:
                            holding.append({'case':case.key,'symbol':s,'held_fraction':float(weights[s].gt(1e-10).mean()),
                               'average_weight':float(weights[s].mean()),'max_weight':float(weights[s].max())})
        risky=t.drop(columns='511260')
        ok=(risky.max().max()<=.3+1e-9 and risky.sum(axis=1).max()<=.7+1e-9
            and risky.gt(1e-10).sum(axis=1).max()<=3 and target.sum(axis=1).max()<=1+1e-9
            and target.gt(1e-10).sum(axis=1).max()<=5)
        checks.append({'check':case.key+'_constraints','passed':bool(ok)})
        print(f'case {i+1}/{len(entries)} {case.key}',flush=True)
    table=pd.DataFrame(records);table.to_csv(OUT/'all_results.csv',index=False,encoding='utf-8-sig')
    daily=pd.DataFrame(daily);daily.to_csv(OUT/'daily_returns.csv',encoding='utf-8-sig')
    for start,values in cash_start_daily.items():
        pd.DataFrame(values).to_csv(OUT/f'plot_daily_{start}.csv',encoding='utf-8-sig')
    pd.DataFrame(holding).to_csv(OUT/'holdings.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'case':name,'period':period,**statistics(daily.loc[left:right,name])}
       for name in daily for period,left,right in [('2018-2021','2018','2021'),('2022-2024','2022','2024'),
                                                  ('2025+','2025',str(END))]]).to_csv(OUT/'segments.csv',index=False,encoding='utf-8-sig')
    main=table.query("start=='2018-01-01' and cost==5 and mode not in ['baseline','maturity']")
    winner=main.sort_values(['sharpe','case'],ascending=[False,True]).iloc[0]['case']
    representative=[Case(),Case('replace'),Case('blend'),Case('filter'),next(c for c in entries if c.key==winner)]
    for case in {c.key:c for c in representative}.values():
        cfg,score,allowed,original=case_inputs(core,engine,case,rsrs)
        overlay=score-original if case.mode!='baseline' else None
        production=risk_controlled_signals(core,cfg,**state.signal_kwargs(core),score_overlay=overlay,
                                           eligibility_mask=allowed)
        composed=defensive_funding_targets(prices,production,sat,SAT,.30)
        error=float((composed-signals[case.key]).abs().to_numpy().max())
        checks.append({'check':case.key+'_production_match','error':error,'passed':error<1e-10})
        for cost in (5.,20.):
            slow=run_backtest(prices,signals[case.key],date(2018,1,1),END,fee_bps=0,slippage_bps=cost)
            fast=simulate(prices,signals[case.key],pd.Timestamp('2018-01-01'),cost_bps=cost)
            error=float(np.max(np.abs(fast.returns-slow.equity.daily_return)))
            checks.append({'check':case.key+f'_ledger_{cost:g}','error':error,'passed':error<1e-10})
    integrated=portfolio_signals(core,prices[[SAT]],state,hypothetical_satellite_history=True,hypothetical_universe_history=True)
    error=float((integrated.signals-signals['default']).abs().to_numpy().max())
    checks.append({'check':'portfolio_baseline_match','error':error,'passed':error<1e-10})
    # 三种用途、多截点的原始数据截断和未来扰动；不得依赖全样本标准化。
    for case in (Case('replace'),Case('blend'),Case('filter')):
        for cutoff in (pd.Timestamp('2020-12-25'),pd.Timestamp('2023-12-29'),pd.Timestamp('2025-12-26')):
            prefix=research_targets(prices.loc[:cutoff],high.loc[:cutoff],low.loc[:cutoff],state,case)
            cp=prices.copy();ch=high.copy();cl=low.copy();mask=cp.index>cutoff
            scale=np.linspace(.4,2.8,prices.shape[1]);cp.loc[mask]*=scale
            ch.loc[mask]*=scale[:-1];cl.loc[mask]*=scale[:-1]
            altered=research_targets(cp,ch,cl,state,case)
            error=max(float((prefix-signals[case.key].loc[:cutoff]).abs().to_numpy().max()),
                      float((altered.loc[:cutoff]-signals[case.key].loc[:cutoff]).abs().to_numpy().max()))
            checks.append({'check':case.key+'_future_'+str(cutoff.date()),'error':error,'passed':error<1e-10})
    # 与前轮已保存的17只方案逐点核对，不把数据口径变化当成因子收益。
    old_path=ROOT/'output/correlation_research_20261031/drop_513500_targets.csv'
    if old_path.exists():
        old=pd.read_csv(old_path,index_col=0,parse_dates=True).drop(columns='513500')
        error=float((old-signals['default']).abs().to_numpy().max())
        checks.append({'check':'frozen_17_etf_baseline_match','error':error,'passed':error<1e-10})
    factor_diagnostics(prices,rsrs,baseline_score)
    non_maturity=['default']+[c.key for c in entries if c.mode!='maturity' and c.key!='default']
    write_json(OUT/'family_bootstrap.json',paired_bootstrap(daily[non_maturity].to_numpy(),reps=1000))
    choices=[];walk_targets=signals['default'].copy()
    for train_end,left,right in [('2021','2022','2024'),('2024','2025',str(END))]:
        scores={name:statistics(daily.loc['2018':train_end,name])['sharpe'] for name in non_maturity}
        chosen=sorted(scores,key=lambda name:(-scores[name],name))[0]
        walk_targets.loc[left:right]=signals[chosen].loc[left:right]
        choices.append({'train_start':'2018','train_end':train_end,'test_start':left,'test_end':right,'chosen':chosen,
                        'chosen_test':statistics(daily.loc[left:right,chosen]),'baseline_test':statistics(daily.loc[left:right,'default'])})
    # 不能把不同策略的已实现日收益直接拼接：参数切换时必须重新成交并计费。
    walk_ledger=simulate(prices,walk_targets,pd.Timestamp('2018-01-01'),cost_bps=5.)
    walk_ledger.series().to_csv(OUT/'time_ordered_returns.csv',header=['daily_return'])
    walk_targets.to_csv(OUT/'time_ordered_targets.csv')
    for item in choices:
        item['executed_test']=statistics(walk_ledger.series().loc[item['test_start']:item['test_end']])
    slow=run_backtest(prices,walk_targets,date(2018,1,1),END,fee_bps=0,slippage_bps=5.)
    error=float(np.max(np.abs(walk_ledger.returns-slow.equity.daily_return)))
    checks.append({'check':'time_ordered_switch_cost_ledger_match','error':error,'passed':error<1e-10})
    write_json(OUT/'time_ordered_choices.json',choices)
    checks.append({'check':'default_and_live_files_unchanged','passed':before==protected_fingerprints()})
    write_json(OUT/'checks.json',checks)
    for name in dict.fromkeys(['default',winner,Case('replace').key,Case('blend').key,Case('filter').key]):
        signals[name].to_csv(OUT/f'{name}_targets.csv',encoding='utf-8-sig')
    write_json(OUT/'manifest.json',{'cases':len(entries),'result_rows':len(table),'winner_ex_post':winner,
      'data_end':END,'all_checks_passed':all(c['passed'] for c in checks),'protected_before':before,
      'input_hashes':{name:hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()
                      for name,frame in [('prices',prices),('high',high),('low',low)]},
      'protected_after':protected_fingerprints(),'source_hashes':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
       for p in ['research_rsrs.py','etf_rotation/factors.py','etf_rotation/ohlc_data.py','research_overfitting_audit.py']}})
    if not all(c['passed'] for c in checks):raise AssertionError('RSRS research verification failed')
    print(main.sort_values('sharpe',ascending=False)[['case','annual','sharpe','drawdown']].head(8).to_string(index=False),flush=True)


def report():
    # 报告生成函数在全部数据和研究检查通过后运行。
    build_report()


def case_label(key):
    c=next(c for c in cases() if c.key==key)
    if c.mode=='baseline':return '当前默认20/40' if key=='default' else f'收益动量{c.short}/{c.long}'
    mode={'replace':'替换评分','blend':'叠加','filter':'正值过滤','maturity':'相同预热原评分'}[c.mode]
    variant={'zscore':'Z','adjusted':'Z×R²','right_skew':'Z×R²×β'}[c.variant]
    suffix=f'，λ={c.strength:g}' if c.mode=='blend' else ''
    mom=f'，动量{c.short}/{c.long}' if (c.short,c.long)!=(20,40) else ''
    return f'{mode}{variant} N{c.n}/M{c.m}{suffix}{mom}'


def build_report():
    table=pd.read_csv(OUT/'all_results.csv')
    daily=pd.read_csv(OUT/'daily_returns.csv',index_col=0,parse_dates=True)
    checks=json.loads((OUT/'checks.json').read_text(encoding='utf-8'))
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    if not manifest['all_checks_passed'] or not all(c['passed'] for c in checks):
        raise ValueError('RSRS checks failed')
    if protected_fingerprints()!=manifest['protected_after']:
        raise ValueError('默认配置或实盘文件已改变，请重跑研究；不能把旧结果标成新默认')
    main=table.query("start=='2018-01-01' and cost==5").set_index('case')
    recent=table.query("start=='2022-01-01' and cost==5").set_index('case')
    stress=table.query("start=='2018-01-01' and cost==20").set_index('case')
    best={mode:main.loc[main['mode'].eq(mode)].sort_values('sharpe',ascending=False).index[0]
          for mode in ('replace','blend','filter','baseline')}
    chosen=list(dict.fromkeys(['default',Case('replace',variant='zscore').key,Case('replace').key,
                              best['replace'],best['blend'],best['filter'],best['baseline']]))
    def metric_rows(frame,names):
        return [[case_label(k),f'{frame.at[k,"annual"]:.2%}',f'{frame.at[k,"sharpe"]:.3f}',
                 f'{frame.at[k,"drawdown"]:.2%}',f'{frame.at[k,"annual_turnover"]:.2f}'] for k in names]
    headers=['方案','年化','零利率夏普','收盘最大回撤','年均双边名义换手/NAV']
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
    assets=ROOT/'docs/assets';assets.mkdir(parents=True,exist_ok=True)
    plotted=list(dict.fromkeys(['default',Case('replace').key,best['replace'],best['blend']]))
    fig,axes=plt.subplots(2,2,figsize=(16,10))
    for row,start in enumerate(('2018-01-01','2022-01-01')):
        # 第二行独立现金起点；不能直接归一化携带旧仓的收益充当现金回测。
        # 研究输出在execute中保存了两种起点的日收益。
        path=OUT/f'plot_daily_{start}.csv'
        r=pd.read_csv(path,index_col=0,parse_dates=True)
        for name in plotted:
            nav=(1+r[name]).cumprod();dd=nav/nav.cummax().clip(lower=1)-1
            axes[row,0].plot(nav.index,nav,label=case_label(name),lw=1.6)
            axes[row,1].plot(dd.index,dd*100,label=case_label(name),lw=1.3)
        axes[row,0].set(title=f'{start[:4]}起：扣费净值',ylabel='净值')
        axes[row,1].set(title=f'{start[:4]}起：收盘回撤',ylabel='%')
        axes[row,0].legend(fontsize=8,loc='upper left')
        for ax in axes[row]:ax.grid(alpha=.25)
    fig.suptitle('RSRS评分探索｜17只核心＋红利卫星｜单边5bps｜全历史假设，非实盘记录',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(assets/'rsrs_comparison_20261031.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,strength in zip(axes,(.25,.5)):
        matrix=np.array([[main.at[Case('blend',n,m,strength=strength).key,'sharpe'] for m in M_VALUES] for n in N_VALUES])
        im=ax.imshow(matrix-main.at['default','sharpe'],cmap='RdYlGn',vmin=-.25,vmax=.25,aspect='auto')
        for i in range(len(N_VALUES)):
            for j in range(len(M_VALUES)):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',fontsize=11)
        ax.set(xticks=range(3),xticklabels=M_VALUES,yticks=range(4),yticklabels=N_VALUES,
               xlabel='标准化窗口 M',ylabel='回归窗口 N',title=f'修正版叠加强度 λ={strength:g}')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='相对默认的夏普变化',shrink=.8)
    fig.suptitle(f'参数敏感性｜格内为2018以来夏普｜默认{main.at["default","sharpe"]:.3f}｜固定20/40动量')
    fig.savefig(assets/'rsrs_sensitivity_20261031.png',dpi=150,bbox_inches='tight');plt.close(fig)
    # 对照长预热本身的影响。
    warm=[]
    for key in dict.fromkeys([Case('replace').key,best['replace']]):
        c=next(c for c in cases() if c.key==key)
        control=Case('maturity',c.n,c.m).key
        warm.extend(metric_rows(main,[control,key]))
    diagnostics=pd.read_csv(OUT/'factor_diagnostics.csv')
    diagnostic_rows=[]
    for n,m,v in dict.fromkeys([(18,600,'zscore'),(18,600,'adjusted'),(60,250,'adjusted'),
                 (int(main.at[manifest['winner_ex_post'],'n']),int(main.at[manifest['winner_ex_post'],'m']),
                  main.at[manifest['winner_ex_post'],'variant'])]):
        for h in (5,20):
            q=diagnostics.query('n==@n and m==@m and variant==@v and horizon==@h').iloc[0]
            diagnostic_rows.append([f'{v} N{n}/M{m}',h,int(q.weeks),f'{q.mean_rank_ic:.4f}',
                f'{q.matched_baseline_rank_ic:.4f}',f'{q.predictive_oos_r2:.3%}',f'{q.mean_price_fit_r2:.3f}'])
    family=json.loads((OUT/'family_bootstrap.json').read_text(encoding='utf-8'))
    choices=json.loads((OUT/'time_ordered_choices.json').read_text(encoding='utf-8'))
    walk_rows=[[q['train_end'],q['test_start']+'—'+q['test_end'],case_label(q['chosen']),
                f"{q['executed_test']['annual']:.2%} / {q['executed_test']['sharpe']:.3f}",
                f"{q['baseline_test']['annual']:.2%} / {q['baseline_test']['sharpe']:.3f}"] for q in choices]
    holding=pd.read_csv(OUT/'holdings.csv',dtype={'symbol':str})
    holdrows=[]
    for key in ('default',manifest['winner_ex_post']):
        top=holding.loc[holding.case.eq(key)&holding.symbol.ne('511260')].nlargest(5,'average_weight')
        for _,r in top.iterrows():
            holdrows.append([case_label(key),r.symbol,ETF_NAMES.get(r.symbol,'红利低波卫星'),
                             f'{r.average_weight:.1%}',f'{r.held_fraction:.1%}'])
    fixed_blends=main.loc[main['mode'].eq('blend')&main['short'].eq(20)&main['long'].eq(40)]
    improved_blends=int(fixed_blends.sharpe.gt(main.at['default','sharpe']).sum())
    winner=manifest['winner_ex_post'];canonical=Case('replace').key
    momentum_names=[c.key for c in cases() if c.mode=='baseline']
    cycle_blend=best['blend']
    matched_cycle=Case(short=int(main.at[cycle_blend,'short']),long=int(main.at[cycle_blend,'long'])).key
    text=f'''# RSRS评分、窗口与ETF轮动研究

研究日期2026-10-31，行情统一截止2026-10-20。当前默认17只核心（原16只＋513180恒科，不含513500标普）及512890红利卫星，全历史假设；默认配置、周状态、实盘计划与真实账户均未改变。

## 结论与建议

本轮保持默认。现有默认2018以来年化{main.at['default','annual']:.2%}、夏普{main.at['default','sharpe']:.3f}、回撤{main.at['default','drawdown']:.2%}；事后最高夏普RSRS为{case_label(winner)}，年化{main.at[winner,'annual']:.2%}、夏普{main.at[winner,'sharpe']:.3f}、回撤{main.at[winner,'drawdown']:.2%}，并未同时提高收益。该方案2022现金起点年化{recent.at[winner,'annual']:.2%}、夏普{recent.at[winner,'sharpe']:.3f}，低于默认{recent.at['default','annual']:.2%}、{recent.at['default','sharpe']:.3f}；20bps成本下2018年化{stress.at[winner,'annual']:.2%}，也低于默认{stress.at['default','annual']:.2%}。

经典N18/M600修正版直接替换仅年化{main.at[canonical,'annual']:.2%}、夏普{main.at[canonical,'sharpe']:.3f}。保持20/40收益动量的24个叠加方案中，{improved_blends}个夏普超过默认；不同动量窗口联合试验的最好叠加为{case_label(cycle_blend)}，年化{main.at[cycle_blend,'annual']:.2%}、夏普{main.at[cycle_blend,'sharpe']:.3f}、回撤{main.at[cycle_blend,'drawdown']:.2%}，不是原20/40评分的普遍增益。正值过滤的事后最好方案降低长期回撤，但牺牲年化且近期较弱。因子可保留作观察与后续前向检验，当前证据不足以替换默认评分或增加实盘规则。

## 方法与来源

截图定义H=α+βL+ε，N日最高价对最低价做带截距OLS；β=Cov(L,H)/Var(L)。再取最近M个β（包含信号日、总体标准差ddof=0）计算Z=(β−mean_M)/std_M。本轮同时比较Z、修正版Z×R²、右偏版Z×R²×β。拟合R²是当期高低价的解释程度，不能视为未来收益预测R²，也不能把β较大直接等同未来一定上涨。

[光大证券研究团队对RSRS系列的原始介绍](https://www.sohu.com/a/246541595_195566)区分市场择时与横截面行业排序；历史报告不构成当前ETF池有效性的证明。[Tushare fund_daily文档](https://tushare.pro/document/2?doc_id=127)提供真实high/low/close，[fund_adj文档](https://tushare.pro/document/2?doc_id=199)提供同日基金复权因子。

原缓存仅收盘。本次为17只核心单独补高低价，每个交易日用既有同日因子计算H'=H×factor_t、L'=L×factor_t，并与冻结raw_close逐日一致核对。没有用今日因子统一复权历史，没有将close复制为高低价，没有拼接ETF上市前指数，没有向后填缺价。当前SDK的waditu下载地址连接失败，研究下载使用可达的官方api.tushare.pro HTTPS REST入口，不修改SDK或生产数据接口；TLS验证保留，Token仅在请求体中发送，不写入报告或日志。

## 事先固定的91场景

回归N=10/18/30/60，标准化M=120/250/600；36个三版本替换、24个修正版叠加强度0.25/0.5、6个正值过滤、12个相同预热原评分对照，另有4个收益动量周期对照及9个经典RSRS与其他动量周期组合。共91场景、两个现金起点×两档成本={len(table)}行结果，差方案全部保留。没有按结果继续细调阈值或切换默认。

- 替换：用RSRS代替最终排序评分，保留long期收益动量>0、MA40、vol60等绝对资格。它不是完全移除现有趋势防守的裸RSRS策略。缺RSRS时不入选。
- 叠加：score_new=score_old+λ×D_old×clip(RSRS_adjusted/3,−1,1)，D为原风险ETF基础评分当日离散度；缺RSRS回退原评分，国债不参与排名。
- 过滤：保持原评分，仅在RSRS_adjusted>0时允许入选；缺值不入选。阈值0事前固定，不搜索。
- 收益动量：10/20、20/40、40/80、60/120；只改变收益动量及对应long正值资格，MA40市场过滤、MA20反转、60日风险估计保持固定。
- 全部保留8%核心波动目标、70%核心风险上限、单只30%、最多3只核心风险、前4缓冲必选第1名。红利MA200入0出4、先现金再国债、最多30%不变；全账户并无8%/70%硬上限。

每交易周最后开市日收盘形成信号，下一交易日收盘成交。2018前只预热；每个起点从现金开始。单边5bps=3bps费用＋2bps滑点预算，另测20bps；252日年化、零无风险利率夏普、收盘回撤。分数复权单位及成交日参考价确定仓位仍是理想执行假设；未模拟整数手、盘口容量、QDII溢价或单独现金分红账簿。

## 2018年以来的结果

下列“最好”均是已观察全样本内的事后筛选，不能称为样本外最优。

{markdown_table(headers,metric_rows(main,chosen))}

## 2022年独立现金起点

{markdown_table(headers,metric_rows(recent,chosen))}

![收益与回撤](assets/rsrs_comparison_20261031.png)

## 交易成本压力

单边20bps，2018现金起点：

{markdown_table(headers,metric_rows(stress,chosen))}

## 参数敏感性与预热差异

收益动量本身的四组对照（不加RSRS）：

{markdown_table(headers,metric_rows(main,momentum_names))}

最好叠加与其对应的原收益动量周期单独比较，避免把周期变化误归因于RSRS：

{markdown_table(headers,metric_rows(main,[matched_cycle,cycle_blend]))}

![叠加参数敏感性](assets/rsrs_sensitivity_20261031.png)

M=600要求至少N+599个完整交易日报价才能出现标准分，会明显延后新ETF参与。叠加缺值回退默认；替换/过滤必须与相同预热资格的默认评分比较，避免把临时缩小ETF池误当成RSRS能力：

{markdown_table(headers,warm)}

首次可用日期及每个ETF样本长度见本地`factor_coverage.csv`。M窗口遇缺失β需重新积累完整窗口，没有缩短min_periods来提高样本量。

## 因子预测性：IC与R²

每个周信号只用当时已有因子，标签从下一交易日收盘到其后5/20个交易日收盘；末端不完整标签丢弃。每期至少5只有效风险ETF，计算横截面Spearman RankIC，并列同一标的/日期样本上的原评分IC。此表不依赖事后持有名单；原收益动量资格不用于预筛该IC样本。

预测R²使用按年扩展训练的一元OLS：只纳入标签结束日在测试年开始前的训练样本，至少200个ETF-周样本，预测测试年未来收益，以各年训练均值为基准计算合并OOS R²。它是时间顺序诊断，年份早已被看过，不是真正盲测。20日标签重叠且ETF相关，不能把全部样本视为独立显著性观察。

{markdown_table(['因子','未来交易日','有效周','RankIC','同样本默认IC','预测OOS R²','高低价拟合R²'],diagnostic_rows)}

完整72组因子×标签诊断、逐周IC及逐ETF时序IC均保留。若高低价拟合R²很高而预测R²接近零或为负，说明高低价可以相互解释，不代表能预测未来盈利。

## 时间顺序与多重比较

分别仅使用截至2021年和2024年的历史夏普，从非预热对照场景中选参数，再看下一段；下一段均已在项目中观察过。各候选缓冲状态作为虚拟子策略从历史持续计算，只在训练段结束后选择；将所选候选在后段的周目标接入同一账户，从账户当前权重真实模拟下一交易日换仓及费用，未直接拼接不同策略收益。区间统计来自该连续交易路径，区别于上面的2022现金起点；这是候选策略切换诊断，并非已部署规则。

{markdown_table(['训练截止','后段','所选方案','后段年化/夏普','默认年化/夏普'],walk_rows)}

非预热对照79方案的配对循环块抽样，每档1000次；零假设为本研究家族内没有平均日收益增益，块长20/60/120，家族校正p分别为{', '.join(f"{q['family_mean_gain_p']:.3f}" for q in family)}。检验的是日均增益、不是夏普差异；只覆盖本轮，不能消除此前多轮选池和调参偏差。所有已观察区间都不是独立盲测。

## 持仓集中度

默认与事后最高夏普RSRS方案，各列平均权重最大的5只风险资产。平均权重按所有交易日计算，不只在持有日计算；缺省包含卫星，国债另作防守不列此表。

{markdown_table(['方案','代码','ETF','平均权重','持有日占比'],holdrows)}

## 验证与本地复现

{len(checks)}项研究检查全部通过：默认与生产组合及前轮17只目标对账、代表性评分与生产选仓对账、费用/收益逐日对账、仓位限制，以及三类用途在三个截点的原始高低价截断/未来扰动检查。因子回归另与直接OLS核对，SQLite键包含两份高低价、窗口与版本。生产配置/状态/实盘计划指纹未变；真实账户数据库未读写。

```powershell
python research_rsrs.py --download
python research_rsrs.py
python research_rsrs.py --report-only
```

高低价缓存`data/rsrs_20261031/`，输入收盘快照`data/overseas_expansion_20261031/`，完整结果`output/rsrs_20261031/`；均本地保存且被Git排除，不分发授权行情。因子库调用`compute_rsrs(high,low,RSRSSpec(N,M))`返回斜率、拟合R²、Z、修正、右偏五个矩阵；`FactorStore.get_rsrs`可缓存单一版本。原close因子接口及默认实盘评分保持不变。
'''
    (ROOT/'docs/rsrs_research_20261031.md').write_text(text,encoding='utf-8')
    print('Report: docs/rsrs_research_20261031.md',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--download-only',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if not args.report_only:
        if not (OUT/'plan.json').exists():write_json(OUT/'plan.json',plan())
        if args.download or args.download_only:download()
        if args.download_only:return
        execute()
    report()


if __name__=='__main__':main()
