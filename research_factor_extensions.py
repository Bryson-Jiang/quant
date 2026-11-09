"""量能、波动率反转、IVOL代理及MACD增强研究，保持默认不变。

python research_factor_extensions.py --download-only
python research_factor_extensions.py
python research_factor_extensions.py --report-only
"""
from __future__ import annotations
import argparse
import hashlib
import json
from dataclasses import asdict,dataclass,replace
from datetime import date,timedelta
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from etf_rotation.config import LEGACY_CORE_NAMES
from etf_rotation.data import _token,_windows,_normalize_history,_normalize_table,load_symbol,tushare_code
from etf_rotation.extended_factors import ExtendedFactorSpec as Spec,compute_extended,low_ivol_utility
from etf_rotation.factor_store import FactorStore
from etf_rotation.factors import FactorSpec,compute_factor
from etf_rotation.risk_strategy import RiskConfig,risk_score_components,risk_controlled_signals,weekly_signal_dates
from etf_rotation.portfolio_strategy import portfolio_signals
from etf_rotation.mean_reversion import ReversionRule,reversion_states,defensive_funding_targets
from etf_rotation.strategy_state import load_pre_ivol_rotation_state as load_rotation_state
from etf_rotation.backtest import run_backtest
from research_overfitting_audit import CoreAudit,simulate,statistics
from research_pool_expansion import protected_fingerprints,paired_bootstrap,markdown_table

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/factor_extensions_20261109';CACHE=ROOT/'data/factor_extensions_20261109'
OLD=ROOT/'data/overseas_expansion_20261031'
END=date(2026,10,20);EXTEND=date(2026,11,6);SAT='512890'
ETF_NAMES={**LEGACY_CORE_NAMES,'513180':'恒生科技ETF（华夏）'}
A_STOCKS=['510300','159915','159995','159875','159928','512660','515070','512880','512800','561360']
RISKY=[s for s in ETF_NAMES if s!='511260']
BASE=RiskConfig(target_vol=.08,short_momentum_days=20,long_momentum_days=40,momentum_skip_days=0,
                trend_days=40,max_single_risky_weight=.30,rank_buffer_size=4,mandatory_top_rank=1,
                ma_reversion_days=20,ma_reversion_strength=.5)


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8')


def features():
    result={}
    for k in (3,5,10):
        for slow in (60,120):
            result[f'reversal_k{k}_v{slow}']=Spec('reversal',window=k,slow=slow)
            for fast in (10,20):result[f'hvrev_k{k}_v{fast}_{slow}']=Spec('high_vol_reversal',window=k,fast=fast,slow=slow)
    for w in (1,5):
        for b in (50,120):result[f'volume_w{w}_b{b}']=Spec('abnormal_volume',window=w,slow=b)
        for k in (3,5,10):result[f'volrev_k{k}_w{w}_b60']=Spec('volume_reversal',window=k,fast=w,slow=60)
    for model in ('market','style'):
        for w in (20,60,120):result[f'ivol_{model}_{w}']=Spec('ivol',window=w,model=model)
    for fast,slow,signal in ((6,13,5),(12,26,9),(24,52,18)):
        for kind in ('trend','histogram'):result[f'macd_{kind}_{fast}_{slow}_{signal}']=Spec('macd_'+kind,fast=fast,slow=slow,signal=signal)
    return result


CANONICAL=('hvrev_k5_v10_60','volume_w5_b50','volrev_k5_w5_b60','ivol_market_60','macd_histogram_12_26_9')


@dataclass(frozen=True)
class Case:
    name:str
    family:str
    components:tuple[tuple[str,float],...]=()
    mode:str='overlay'
    scope:str='default'


def cases():
    rows=[Case('default','baseline',mode='baseline')]
    for name,spec in features().items():
        family=spec.name
        if family.startswith('macd_'):family='macd'
        for strength in (.25,.5):rows.append(Case(f'{name}_a{strength:g}',family,((name,strength),)))
    for name in ('hvrev_k5_v10_60','volume_w5_b50','volrev_k5_w5_b60','macd_histogram_12_26_9'):
        for strength in (.25,.5):rows.append(Case(f'{name}_ashare_a{strength:g}',features()[name].name,((name,strength),),scope='ashare'))
    for w in (20,60,120):
        name=f'ivol_market_{w}'
        for strength in (.25,.5):rows.append(Case(f'{name}_global_a{strength:g}','ivol_global_diagnostic',((name,strength),),scope='global'))
    for name,spec in features().items():
        if spec.name.startswith('macd_'):
            rows.append(Case(name+'_positive_filter','macd_filter',((name,0.),),mode='filter'))
            rows.append(Case(name+'_matched_warmup','warmup_control',((name,0.),),mode='maturity'))
        elif spec.name=='ivol':rows.append(Case(name+'_exclude_top_quartile','ivol_filter',((name,0.),),mode='filter'))
    pairs=[('hv_macd',('hvrev_k5_v10_60','macd_histogram_12_26_9')),
           ('ivol_macd',('ivol_market_60','macd_trend_12_26_9')),
           ('volume_macd',('volume_w5_b50','macd_histogram_12_26_9')),
           ('hv_ivol_macd',('hvrev_k5_v10_60','ivol_market_60','macd_histogram_12_26_9'))]
    rows.extend(Case('combo_'+name,'combination',tuple((f,.25) for f in names)) for name,names in pairs)
    return rows


def plan():
    return {'date':'2026-11-09','main_end':END,'extension_end':EXTEND,'core_universe':ETF_NAMES,
       'base_config':asdict(BASE),'features':{n:asdict(s) for n,s in features().items()},
       'cases':[asdict(c) for c in cases()],'case_count':len(cases()),'overlay_strengths':[.25,.5],
       'ivol':'A股ETF市场/成长风格代理残差波动；不是隐含波动或FF3论文精确复现；全池应用另列为诊断',
       'literature':'Gervais 2001为高成交量溢价，不是高波动反转；反转交互属于本研究假设',
       'evaluation':'2018/2022现金起点×5/20bps；2018-21/2022-24/2025+分段；按时间选参、块抽样、多重比较；全部已经观察，不称盲测',
       'extension':'仅在主区间选好或事先指定方案上核对10/21-11/6新增数据，不以短期表现重新筛选参数；该区间部分曾用于实盘查询，也非严格盲测',
       'robust_screen':'两个起点5/20bps的年化和夏普均不低于默认，且5bps两起点最大回撤均不超过10%；不据此自动切换实盘',
       'sources':['https://doi.org/10.1111/0022-1082.00349','https://doi.org/10.1111/j.1540-6261.2006.00836.x',
                  'https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/macd'],
       'protected_before':protected_fingerprints()}


class ResearchClient:
    def __init__(self):self.s=requests.Session();self.token=_token()
    def query(self,method,symbol,start,end):
        response=self.s.post('https://api.tushare.pro',json={'api_name':method,'token':self.token,
          'params':{'ts_code':tushare_code(symbol),'start_date':start.strftime('%Y%m%d'),'end_date':end.strftime('%Y%m%d')},
          'fields':'ts_code,trade_date,close,vol,amount' if method=='fund_daily' else 'ts_code,trade_date,adj_factor'},
          timeout=30,allow_redirects=False)
        response.raise_for_status();payload=response.json()
        if payload.get('code')!=0:raise RuntimeError('Provider error; API details withheld')
        return pd.DataFrame(payload['data']['items'],columns=payload['data']['fields'])


def download():
    CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    client=None;records=[]
    for symbol in [*ETF_NAMES,SAT]:
        try:
            raw_path=CACHE/f'daily_{symbol}.csv'
            if not raw_path.exists():
                if client is None:client=ResearchClient()
                parts=[client.query('fund_daily',symbol,a,b) for a,b in _windows(date(2010,1,1),EXTEND)]
                raw=pd.concat([p for p in parts if not p.empty],ignore_index=True)
                raw.to_csv(raw_path,index=False,encoding='utf-8-sig')
            raw=pd.read_csv(raw_path,dtype={'trade_date':str})
            old=_normalize_history(pd.read_csv(OLD/f'tushare_{symbol}_hfq.csv'),symbol)
            raw_close=_normalize_table(raw,'close',symbol)
            old_part=raw_close.reindex(old.index)
            if not np.allclose(old_part,old.raw_close,rtol=0,atol=1e-9):raise ValueError('Historical close snapshot revised')
            new_days=raw_close.index[raw_close.index>old.index[-1]]
            factor_path=CACHE/f'new_factor_{symbol}.csv'
            if not factor_path.exists():
                if client is None:client=ResearchClient()
                client.query('fund_adj',symbol,END-timedelta(days=21),EXTEND).to_csv(factor_path,index=False,encoding='utf-8-sig')
            f=_normalize_table(pd.read_csv(factor_path,dtype={'trade_date':str}),'adj_factor',symbol)
            overlap=old.index.intersection(f.index)
            if not np.allclose(old.loc[overlap,'adj_factor'],f.reindex(overlap),rtol=1e-9):raise ValueError('Historical factor snapshot revised')
            new_factor=f.reindex(new_days)
            if new_factor.isna().any():raise ValueError('Missing same-day extension factors')
            extra=pd.DataFrame({'raw_close':raw_close.reindex(new_days),'adj_factor':new_factor,
                                'close':raw_close.reindex(new_days)*new_factor},index=new_days)
            joined=pd.concat([old,extra]);joined.index.name='date'
            joined=_normalize_history(joined.reset_index(),symbol)
            joined.to_csv(CACHE/f'tushare_{symbol}_hfq.csv',encoding='utf-8-sig')
            write_json(CACHE/f'tushare_{symbol}_hfq.json',{'source':'tushare','schema_version':2,'symbol':symbol,
                   'adjust':'hfq','start':'2010-01-01','end':str(EXTEND)})
            for field in ('vol','amount'):
                a=pd.to_numeric(raw[field],errors='coerce')
                if a.isna().any() or not np.isfinite(a).all() or a.lt(0).any():raise ValueError('Invalid volume/amount')
            records.append({'symbol':symbol,'rows':len(joined),'new_days':len(new_days),
                            'historical_close_match':True,'historical_factor_match':True,'last_quote':str(joined.index[-1].date())})
            write_json(OUT/'download_status.json',records)
            print(f'data {symbol}: {len(joined)} quotes, {len(new_days)} new days',flush=True)
        except Exception as exc:
            records.append({'symbol':symbol,'failed_type':type(exc).__name__})
            write_json(OUT/'download_status.json',records)
            raise RuntimeError(f'{symbol}: preparation failed ({type(exc).__name__}); private API details withheld') from None


def read_data(end=END):
    values={s:load_symbol(s,date(2010,1,1),end,CACHE,offline=True) for s in [*ETF_NAMES,SAT]}
    prices=pd.DataFrame(values,index=values['510300'].index).rename_axis('date')
    volumes={};amounts={}
    for s in ETF_NAMES:
        raw=pd.read_csv(CACHE/f'daily_{s}.csv',dtype={'trade_date':str})
        raw.index=pd.to_datetime(raw.trade_date,format='%Y%m%d')
        if raw.index.has_duplicates:raise ValueError('Duplicate activity dates')
        volumes[s]=pd.to_numeric(raw.vol).reindex(prices.index)
        amounts[s]=pd.to_numeric(raw.amount).reindex(prices.index)*1000
    return prices,pd.DataFrame(volumes,index=prices.index),pd.DataFrame(amounts,index=prices.index)


def feature_frames(prices,volume,*,use_cache=False):
    core=prices[list(ETF_NAMES)];result={}
    store=FactorStore(OUT/'factor_cache.sqlite') if use_cache else None
    for name,spec in features().items():
        activity=volume if spec.name in {'abnormal_volume','volume_reversal'} else None
        if use_cache and name in CANONICAL:
            result[name]=store.get_extended(core,spec,activity)
            pd.testing.assert_frame_equal(result[name],compute_extended(core,spec,activity))
        else:result[name]=compute_extended(core,spec,activity)
    return result


def utility(name,frames,scope='default'):
    raw=frames[name];spec=features()[name]
    if spec.name=='ivol':out=low_ivol_utility(raw,RISKY if scope=='global' else A_STOCKS)
    else:out=raw.clip(-1,1)
    out=out.copy()
    if scope=='ashare':out.loc[:,[s for s in out if s not in A_STOCKS]]=np.nan
    out['511260']=np.nan
    return out


def base_components(core):
    cfg=BASE
    f={n:compute_factor(core,FactorSpec(name,window)) for n,name,window in [
        ('short','momentum',20),('long','momentum',40),('vol','volatility',60),('ma','ma_zscore',20)]}
    return risk_score_components(f['short'],f['long'],f['vol'],cfg,f['ma'])


def case_inputs(core,frames,case,components):
    score=components['score'].copy();allowed=None
    spread=components['base_score'].drop(columns='511260').std(axis=1,ddof=1).fillna(0)
    if case.mode=='overlay':
        for name,strength in case.components:
            u=utility(name,frames,case.scope)
            score+=u.fillna(0).mul(spread*strength,axis=0)
    elif case.mode in {'filter','maturity'}:
        name=case.components[0][0];u=utility(name,frames,case.scope)
        if features()[name].name=='ivol':allowed=u.ge(-.5)|u.isna()
        else:allowed=u.notna() if case.mode=='maturity' else u.gt(0)&u.notna()
    elif case.mode!='baseline':raise ValueError('Unknown case mode')
    return score,allowed


def compose(prices,core_targets):
    sat=reversion_states(prices[[SAT]],SAT,ReversionRule(200,0,.04,None),weekly_signal_dates(prices.index))
    return defensive_funding_targets(prices,core_targets,sat,SAT,.30)


def executable_signal_dates(signals,calendar):
    """截断区间末尾可能有尚无下一交易日的占位目标，不能当成已执行历史。"""
    return signals.index[calendar.searchsorted(signals.index,side='right')<len(calendar)]


def targets(prices,volume,state,case,*,frames=None,engine=None,components=None):
    core=prices[list(ETF_NAMES)]
    frames=frames if frames is not None else feature_frames(prices,volume)
    engine=engine if engine is not None else CoreAudit(core,state)
    components=components if components is not None else base_components(core)
    score,allowed=case_inputs(core,frames,case,components)
    return compose(prices,engine.signals(BASE,score_override=score,eligibility_mask=allowed))


def diagnostics(prices,frames,components):
    core=prices[list(ETF_NAMES)];v=compute_factor(core,FactorSpec('volatility',60))
    days=weekly_signal_dates(prices.index);days=days[days>=pd.Timestamp('2018-01-01')]
    summaries=[];daily_rows=[];symbol_rows=[]
    for fi,name in enumerate(frames):
        x=utility(name,frames);relevance=[];risk_corr=[]
        for horizon in (5,20):
            future=prices[RISKY].shift(-(horizon+1))/prices[RISKY].shift(-1)-1
            samples=[];ics=[];baseics=[]
            for day in days:
                valid=x.loc[day,RISKY].notna()&future.loc[day].notna()&v.loc[day,RISKY].gt(0)
                names=valid.index[valid]
                if len(names)<5:continue
                xx=x.loc[day,names];yy=future.loc[day,names]
                if xx.std()<1e-12 or yy.std()<1e-12:continue
                ic=float(xx.rank().corr(yy.rank()))
                base=float(components['score'].loc[day,names].rank().corr(yy.rank()))
                ics.append(ic);baseics.append(base)
                daily_rows.append({'name':name,'horizon':horizon,'date':day,'rank_ic':ic,'matched_base_ic':base,'assets':len(names)})
                if horizon==5:
                    relevance.append(float(xx.rank().corr(components['score'].loc[day,names].rank())))
                    risk_corr.append(float(xx.rank().corr(v.loc[day,names].rank())))
                label_end=prices.index[prices.index.get_loc(day)+horizon+1]
                samples.extend({'date':day,'label_end':label_end,'symbol':s,'x':float(xx[s]),'y':float(yy[s])} for s in names)
            sample=pd.DataFrame(samples);sse=null=0.;count=0
            if not sample.empty:
                for year in range(2019,2027):
                    left=pd.Timestamp(year,1,1);right=pd.Timestamp(year+1,1,1)
                    train=sample.loc[sample.label_end<left];test=sample.loc[(sample.date>=left)&(sample.date<right)]
                    if len(train)<200 or test.empty:continue
                    coef=np.linalg.lstsq(np.column_stack([np.ones(len(train)),train.x]),train.y,rcond=None)[0]
                    prediction=coef[0]+coef[1]*test.x
                    sse+=float(np.square(test.y-prediction).sum());null+=float(np.square(test.y-train.y.mean()).sum());count+=len(test)
                for s,g in sample.groupby('symbol'):
                    corr=float(g.x.rank().corr(g.y.rank())) if g.x.std()>1e-12 else None
                    symbol_rows.append({'name':name,'horizon':horizon,'symbol':s,'observations':len(g),'rank_ic':corr})
            summaries.append({'name':name,'family':features()[name].name,'horizon':horizon,'weeks':len(ics),
                'mean_rank_ic':float(np.mean(ics)) if ics else None,'matched_base_rank_ic':float(np.nanmean(baseics)) if ics else None,
                'positive_ic_fraction':float(np.mean(np.array(ics)>0)) if ics else None,
                'predictive_oos_r2':1-sse/null if null else None,'oos_observations':count,
                'original_score_rank_correlation':float(np.nanmean(relevance)) if relevance else None,
                'volatility_rank_correlation':float(np.nanmean(risk_corr)) if risk_corr else None})
        print(f'factor diagnostic {fi+1}/{len(frames)} {name}',flush=True)
    pd.DataFrame(summaries).to_csv(OUT/'factor_diagnostics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(daily_rows).to_csv(OUT/'weekly_ic.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(symbol_rows).to_csv(OUT/'etf_factor_ic.csv',index=False,encoding='utf-8-sig')


def execute():
    before=protected_fingerprints();prices,volume,amount=read_data()
    core=prices[list(ETF_NAMES)];state=load_rotation_state(ROOT/'data/rotation_state.json')
    engine=CoreAudit(core,state);frames=feature_frames(prices,volume,use_cache=True);components=base_components(core)
    entries=cases();signals={};records=[];daily={};recent_daily={};holdings=[];checks=[]
    for i,case in enumerate(entries):
        t=targets(prices,volume,state,case,frames=frames,engine=engine,components=components);signals[case.name]=t
        for start in ('2018-01-01','2022-01-01'):
            for cost in (5.,20.):
                led=simulate(prices,t,pd.Timestamp(start),cost_bps=cost)
                metric=statistics(led.returns)
                records.append({'case':case.name,'family':case.family,'mode':case.mode,'scope':case.scope,
                                'start':start,'cost':cost,'skipped':led.skipped,
                                'annual_turnover':float(led.turnover.sum()*252/len(led.dates)),**metric})
                if cost==5:
                    (daily if start=='2018-01-01' else recent_daily)[case.name]=led.series()
                    if start=='2018-01-01':
                        w=pd.DataFrame(led.weights,index=led.dates,columns=prices.columns)
                        for s in prices:
                            holdings.append({'case':case.name,'symbol':s,'average_weight':float(w[s].mean()),
                                             'held_fraction':float(w[s].gt(1e-10).mean()),'max_weight':float(w[s].max())})
        risky=t.drop(columns=['511260',SAT])
        checks.append({'check':case.name+'_constraints','passed':bool(risky.max().max()<=.3+1e-9
            and risky.sum(axis=1).max()<=.7+1e-9 and risky.gt(1e-10).sum(axis=1).max()<=3
            and t.sum(axis=1).max()<=1+1e-9 and t.gt(1e-10).sum(axis=1).max()<=5)})
        print(f'case {i+1}/{len(entries)} {case.name}',flush=True)
    table=pd.DataFrame(records);table.to_csv(OUT/'all_results.csv',index=False,encoding='utf-8-sig')
    daily=pd.DataFrame(daily);recent=pd.DataFrame(recent_daily)
    daily.to_csv(OUT/'daily_2018.csv',encoding='utf-8-sig');recent.to_csv(OUT/'daily_2022.csv',encoding='utf-8-sig')
    pd.DataFrame(holdings).to_csv(OUT/'holdings.csv',index=False,encoding='utf-8-sig')
    segments=pd.DataFrame([{'case':n,'period':period,**statistics(daily.loc[a:b,n])} for n in daily
                          for period,a,b in [('2018-2021','2018','2021'),('2022-2024','2022','2024'),('2025+','2025',str(END))]])
    segments.to_csv(OUT/'segments.csv',index=False,encoding='utf-8-sig')
    main=table.query("start=='2018-01-01' and cost==5").set_index('case')
    eligible=[c.name for c in entries if c.family not in {'warmup_control','ivol_global_diagnostic'}]
    winner=main.loc[[n for n in eligible if n!='default']].sort_values('sharpe',ascending=False).index[0]
    family_best={family:main.loc[main.family.eq(family)].sharpe.idxmax() for family in
                 ('high_vol_reversal','abnormal_volume','volume_reversal','ivol','macd','combination','macd_filter','ivol_filter')}
    passing=[]
    for name in eligible:
        rows=table.loc[table.case.eq(name)].set_index(['start','cost'])
        base=table.loc[table.case.eq('default')].set_index(['start','cost'])
        ok=bool((rows.annual>=base.annual-1e-12).all() and (rows.sharpe>=base.sharpe-1e-12).all()
                and rows.loc[pd.IndexSlice[:,5.],'drawdown'].ge(-.10-1e-12).all())
        if name!='default' and ok:passing.append(name)
    nomination=main.loc[passing].sharpe.idxmax() if passing else 'default'
    representatives=list(dict.fromkeys(['default',winner,nomination,*family_best.values(),
                                      'hvrev_k5_v10_60_a0.25','ivol_market_60_a0.25','macd_histogram_12_26_9_a0.25']))
    for name in representatives:
        case=next(c for c in entries if c.name==name)
        score,admission=case_inputs(core,frames,case,components)
        prod_core=risk_controlled_signals(core,BASE,**state.signal_kwargs(core),
                  score_overlay=score-components['score'],eligibility_mask=admission)
        prod=compose(prices,prod_core);error=float((prod-signals[name]).abs().to_numpy().max())
        checks.append({'check':name+'_production_targets','error':error,'passed':error<1e-10})
        for cost in (5.,20.):
            slow=run_backtest(prices,signals[name],date(2018,1,1),END,fee_bps=0,slippage_bps=cost)
            led=simulate(prices,signals[name],pd.Timestamp('2018-01-01'),cost_bps=cost)
            error=float(np.max(np.abs(led.returns-slow.equity.daily_return)))
            checks.append({'check':name+f'_ledger_{cost:g}','error':error,'passed':error<1e-10})
    integrated=portfolio_signals(core,prices[[SAT]],state,hypothetical_satellite_history=True,hypothetical_universe_history=True)
    err=float((integrated.signals-signals['default']).abs().to_numpy().max())
    checks.append({'check':'unchanged_production_baseline','error':err,'passed':err<1e-10})
    prior=ROOT/'output/rsrs_20261031/default_targets.csv'
    if prior.exists():
        old=pd.read_csv(prior,index_col=0,parse_dates=True)
        err=float((old-signals['default']).abs().to_numpy().max())
        checks.append({'check':'prior_17_etf_baseline','error':err,'passed':err<1e-10})
    for name in [winner,'hvrev_k5_v10_60_a0.25','ivol_market_60_a0.25','macd_histogram_12_26_9_a0.25','volrev_k5_w5_b60_a0.25']:
        case=next(c for c in entries if c.name==name)
        for cut in (pd.Timestamp('2020-12-25'),pd.Timestamp('2023-12-29'),pd.Timestamp('2025-12-26')):
            p=prices.loc[:cut];v=volume.loc[:cut]
            prefix=targets(p,v,state,case)
            cp=prices.copy();cv=volume.copy();future=cp.index>cut
            cp.loc[future]*=np.linspace(.4,2.8,prices.shape[1]);cv.loc[future]*=np.linspace(2,9,volume.shape[1])
            changed=targets(cp,cv,state,case)
            error=max(float((prefix-signals[name].loc[:cut]).abs().to_numpy().max()),
                      float((changed.loc[:cut]-signals[name].loc[:cut]).abs().to_numpy().max()))
            checks.append({'check':name+'_future_'+str(cut.date()),'error':error,'passed':error<1e-10})
    diagnostics(prices,frames,components)
    trials=['default']+[c.name for c in entries if c.name!='default' and c.family!='warmup_control']
    write_json(OUT/'family_bootstrap.json',paired_bootstrap(daily[trials].to_numpy(),reps=1000))
    # 参数切换使用同一账户重新成交，不能直接拼接候选的已实现日收益。
    walk=signals['default'].copy();choices=[]
    for train_end,left,right in [('2021','2022','2024'),('2024','2025',str(END))]:
        scores={n:statistics(daily.loc['2018':train_end,n])['sharpe'] for n in eligible}
        selected=sorted(scores,key=lambda n:(-scores[n],n))[0]
        walk.loc[left:right]=signals[selected].loc[left:right]
        choices.append({'train_end':train_end,'test_start':left,'test_end':right,'chosen':selected,
                        'baseline':statistics(daily.loc[left:right,'default'])})
    walk_ledger=simulate(prices,walk,pd.Timestamp('2018-01-01'))
    for q in choices:q['test_result']=statistics(walk_ledger.series().loc[q['test_start']:q['test_end']])
    write_json(OUT/'time_ordered_choices.json',choices)
    walk.to_csv(OUT/'time_ordered_targets.csv');walk_ledger.series().to_csv(OUT/'time_ordered_returns.csv')
    # 预先限定：仅事先指定和主区间选出者检查新增13日，不以此调参。
    p,v,a=read_data(EXTEND);f=feature_frames(p,v);e=CoreAudit(p[list(ETF_NAMES)],state);bc=base_components(p[list(ETF_NAMES)])
    extension=[]
    for name in representatives:
        case=next(c for c in entries if c.name==name)
        t=targets(p,v,state,case,frames=f,engine=e,components=bc)
        past=executable_signal_dates(signals[name],prices.index)
        pd.testing.assert_index_equal(t.loc[past].index,signals[name].loc[past].index)
        err=float((t.loc[past]-signals[name].loc[past]).abs().to_numpy().max())
        checks.append({'check':name+'_new_data_preserves_past_targets','error':err,'passed':err<1e-10})
        led=simulate(p,t,pd.Timestamp('2018-01-01'))
        err=float((led.series().reindex(daily.index)-daily[name]).abs().max())
        checks.append({'check':name+'_new_data_preserves_past_returns','error':err,'passed':err<1e-10})
        new=led.series().loc[str(END+timedelta(days=1)):]
        extension.append({'case':name,'days':len(new),'total':float((1+new).prod()-1),
                          'max_drawdown':statistics(new)['drawdown']})
    pd.DataFrame(extension).to_csv(OUT/'short_extension.csv',index=False,encoding='utf-8-sig')
    # 成交额替代份额成交量：固定代表参数，检验拆分/价格量纲的解释。
    amount_rows=[]
    for name in ('volume_w5_b50_a0.25','volrev_k5_w5_b60_a0.25'):
        case=next(c for c in entries if c.name==name);name_f=case.components[0][0]
        alternative=frames.copy();alternative[name_f]=compute_extended(core,features()[name_f],amount)
        t=targets(prices,amount,state,case,frames=alternative,engine=engine,components=components)
        for start in ('2018-01-01','2022-01-01'):
            for cost in (5.,20.):
                led=simulate(prices,t,pd.Timestamp(start),cost_bps=cost)
                amount_rows.append({'case':name,'activity':'cash_amount','start':start,'cost':cost,**statistics(led.returns)})
    pd.DataFrame(amount_rows).to_csv(OUT/'amount_robustness.csv',index=False,encoding='utf-8-sig')
    ivol_attribution(prices,volume,state,frames,components,engine,signals,table,checks)
    for name in representatives:signals[name].to_csv(OUT/f'{name}_targets.csv',encoding='utf-8-sig')
    checks.append({'check':'default_state_live_unchanged','passed':before==protected_fingerprints()})
    write_json(OUT/'checks.json',checks)
    write_json(OUT/'manifest.json',{'main_end':END,'extension_end':EXTEND,'cases':len(entries),'result_rows':len(table),
      'features':len(frames),'winner_ex_post':winner,'family_best':family_best,'strict_screen_passed':passing,
      'nomination_for_observation':nomination,'all_checks_passed':all(c['passed'] for c in checks),
      'protected_before':before,'protected_after':protected_fingerprints(),
      'input_hashes':{name:hashlib.sha256(pd.util.hash_pandas_object(df,index=True).values.tobytes()).hexdigest()
                      for name,df in [('prices',prices),('volume',volume),('amount',amount)]},
      'source_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                      ['research_factor_extensions.py','etf_rotation/extended_factors.py','etf_rotation/factor_store.py']}})
    if not all(c['passed'] for c in checks):raise AssertionError('Factor research validation failed')
    print(main.loc[eligible].sort_values('sharpe',ascending=False)[['annual','sharpe','drawdown']].head(10).to_string(),flush=True)


def ivol_attribution(prices,volume,state,frames,components,engine,signals,table,checks):
    """事后归因诊断，不并入原先117场景寻优或扩大显著性结论。"""
    core=prices[list(ETF_NAMES)];columns=core.columns
    reference='ivol_style_120_exclude_top_quartile'
    ref_case=next(c for c in cases() if c.name==reference)
    _,reference_mask=case_inputs(core,frames,ref_case,components)
    weeks=reference_mask.loc[signals[reference].index]
    eligible=(compute_factor(core,FactorSpec('momentum',40))>0)&(compute_factor(core,FactorSpec('trend',40))>0)
    blocked=(~weeks)&eligible.reindex(weeks.index).fillna(False)
    # 仅用2018-2021中可知的屏蔽频率，2022以后固定排除2只；之前仍用默认。
    train=blocked.loc['2018':'2021',A_STOCKS].sum().sort_values(ascending=False,kind='stable')
    fixed_names=train.head(2).index.tolist()
    fixed=pd.DataFrame(True,index=core.index,columns=columns)
    fixed.loc[fixed.index>=pd.Timestamp('2022-01-01'),fixed_names]=False
    masks={'static2_chosen_2018_2021':fixed}
    protected=['510300','159915']
    for window in (60,120):
        # 对齐基准自身不可评价的范围，排除“只因保护创业板”造成的差异。
        raw=frames[f'ivol_market_{window}'].copy();raw.loc[:,protected]=np.nan
        low=low_ivol_utility(raw,A_STOCKS)
        masks[f'market_ivol_{window}_same_protected_broads']=low.ge(-.5)|low.isna()
        sigma=compute_factor(core,FactorSpec('volatility',window));sigma.loc[:,protected]=np.nan
        low=low_ivol_utility(sigma,A_STOCKS)
        masks[f'total_volatility_{window}_same_protected_broads']=low.ge(-.5)|low.isna()
    # 反向四分位是方向诊断，并非看完结果再推荐的另一个交易方案。
    ref_u=utility('ivol_style_120',frames)
    masks['exclude_lowest_ivol_quartile']=ref_u.le(.5)|ref_u.isna()
    rows=[]
    for name,mask in masks.items():
        t=compose(prices,engine.signals(BASE,score_override=components['score'],eligibility_mask=mask))
        for start in ('2018-01-01','2022-01-01'):
            for cost in (5.,20.):
                led=simulate(prices,t,pd.Timestamp(start),cost_bps=cost)
                rows.append({'case':name,'start':start,'cost':cost,**statistics(led.returns)})
        if name=='static2_chosen_2018_2021':
            prior=t.index<pd.Timestamp('2022-01-01')
            err=float((t.loc[prior]-signals['default'].loc[prior]).abs().to_numpy().max())
            checks.append({'check':'static_control_does_not_use_training_selection_in_past','error':err,'passed':err<1e-10})
    pd.DataFrame(rows).to_csv(OUT/'ivol_posthoc_controls.csv',index=False,encoding='utf-8-sig')
    counts=pd.DataFrame({'blocked_weeks':blocked.loc['2018':].sum(),
                         'eligible_weeks':eligible.reindex(weeks.index).loc['2018':].sum(),
                         'training_2018_2021_blocked_weeks':train.reindex(columns).fillna(0)})
    counts.to_csv(OUT/'ivol_exclusions.csv',index_label='symbol',encoding='utf-8-sig')
    write_json(OUT/'ivol_posthoc_design.json',{'phase':'after_observing_main_results','fixed_names':fixed_names,
       'purpose':'separate volatility preference, protected benchmark assets and persistent sector exclusion; not extra candidate search',
       'models':list(masks),'original_family_p_covers':'original prespecified trials only'})
    # 可加的对数收益归因：各资产贡献＋交易成本正好还原账户净增长。
    attribution=[];drawdowns=[]
    for name in ('default',reference):
        for start in ('2018-01-01','2022-01-01'):
            led=simulate(prices,signals[name],pd.Timestamp(start),cost_bps=5.)
            weights=pd.DataFrame(led.weights,index=led.dates,columns=prices.columns).shift(1).fillna(0)
            changes=prices.ffill(limit=3).pct_change(fill_method=None).reindex(led.dates).fillna(0)
            gross_contribution=weights*changes
            log_scale=np.ones(len(led.dates));nonzero=np.abs(led.gross)>1e-14
            log_scale[nonzero]=np.log1p(led.gross[nonzero])/led.gross[nonzero]
            contribution=gross_contribution.mul(log_scale,axis=0).sum()
            cost=float(np.log1p(-led.turnover*.0005).sum())
            err=abs(float(contribution.sum()+cost)-float(np.log1p(led.returns).sum()))
            checks.append({'check':name+'_'+start+'_log_attribution_reconcile','error':err,'passed':err<1e-10})
            attribution.extend({'case':name,'start':start,'symbol':s,'log_growth_contribution':float(contribution[s])} for s in prices)
            attribution.append({'case':name,'start':start,'symbol':'COST','log_growth_contribution':cost})
            nav=pd.Series(np.cumprod(1+led.returns),index=led.dates)
            dd=nav/nav.cummax().clip(lower=1)-1;trough=dd.idxmin();prior_nav=nav.loc[:trough]
            peak=prior_nav.idxmax() if prior_nav.max()>=1 else led.dates[0]
            drawdowns.append({'case':name,'start':start,'peak':peak,'trough':trough,
               'maximum_drawdown':float(dd.min()),'worst_day_date':led.dates[np.argmin(led.returns)],
               'worst_day':float(led.returns.min())})
    pd.DataFrame(attribution).to_csv(OUT/'log_return_attribution.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(drawdowns).to_csv(OUT/'drawdown_episodes.csv',index=False,encoding='utf-8-sig')


def label(name):
    c=next(c for c in cases() if c.name==name)
    if name=='default':return '当前默认'
    if c.family=='combination':
        return {'combo_hv_macd':'高波动反转＋MACD柱', 'combo_ivol_macd':'低IVOL＋MACD趋势',
                'combo_volume_macd':'量能溢价＋MACD柱','combo_hv_ivol_macd':'反转＋IVOL＋MACD柱'}[name]+'（各0.25）'
    feature,strength=c.components[0];s=features()[feature]
    if s.name=='ivol':base=f'低IVOL{"市场" if s.model=="market" else "风格"}{s.window}日'
    elif s.name=='high_vol_reversal':base=f'高波动反转k{s.window}/σ{s.fast},{s.slow}'
    elif s.name=='reversal':base=f'短反转k{s.window}/σ{s.slow}'
    elif s.name=='abnormal_volume':base=f'量能溢价W{s.window}/B{s.slow}'
    elif s.name=='volume_reversal':base=f'量能反转k{s.window}/W{s.fast}/B{s.slow}'
    else:base=f'MACD{"趋势" if s.name=="macd_trend" else "柱"}{s.fast}/{s.slow}/{s.signal}'
    if c.mode=='maturity':return base+'同预热对照'
    if c.mode=='filter':return base+('剔除最高1/4' if s.name=='ivol' else '>0过滤')
    scope='（A股）' if c.scope=='ashare' or s.name=='ivol' and c.scope!='global' else '（全池诊断）' if c.scope=='global' else ''
    return base+f'×{strength:g}'+scope


def build_report():
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    checks=json.loads((OUT/'checks.json').read_text(encoding='utf-8'))
    if not manifest['all_checks_passed'] or protected_fingerprints()!=manifest['protected_after']:
        raise ValueError('研究未通过，或默认策略文件已改变；须重新核对')
    table=pd.read_csv(OUT/'all_results.csv')
    main=table.query("start=='2018-01-01' and cost==5").set_index('case')
    recent=table.query("start=='2022-01-01' and cost==5").set_index('case')
    stress=table.query("start=='2018-01-01' and cost==20").set_index('case')
    best=manifest['family_best'];winner=manifest['winner_ex_post']
    selected=list(dict.fromkeys(['default',best['high_vol_reversal'],best['abnormal_volume'],best['volume_reversal'],
                               best['ivol'],best['macd'],best['combination'],best['macd_filter'],best['ivol_filter'],winner]))
    headers=['方案','年化','夏普Rf0','最大回撤','卡玛','年均双边换手/NAV']
    def metric_rows(frame,names):
        return [[label(n),f'{frame.at[n,"annual"]:.2%}',f'{frame.at[n,"sharpe"]:.3f}',
                 f'{frame.at[n,"drawdown"]:.2%}',f'{frame.at[n,"annual"]/abs(frame.at[n,"drawdown"]):.3f}',
                 f'{frame.at[n,"annual_turnover"]:.2f}'] for n in names]
    daily=pd.read_csv(OUT/'daily_2018.csv',index_col=0,parse_dates=True)
    recent_daily=pd.read_csv(OUT/'daily_2022.csv',index_col=0,parse_dates=True)
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
    assets=ROOT/'docs/assets';assets.mkdir(parents=True,exist_ok=True)
    plotted=list(dict.fromkeys(['default',best['high_vol_reversal'],best['ivol_filter'],best['macd'],best['abnormal_volume']]))
    fig,axes=plt.subplots(2,2,figsize=(16,10))
    for row,(start,returns) in enumerate([('2018',daily),('2022',recent_daily)]):
        for name in plotted:
            nav=(1+returns[name]).cumprod();dd=nav/nav.cummax().clip(lower=1)-1
            axes[row,0].plot(nav.index,nav,label=label(name),lw=1.4)
            axes[row,1].plot(dd.index,dd*100,lw=1.2)
        axes[row,0].set(title=f'{start}现金起点：扣费净值',ylabel='净值')
        axes[row,1].set(title=f'{start}现金起点：收盘回撤',ylabel='%')
        axes[row,0].legend(fontsize=7.5,loc='upper left')
        for ax in axes[row]:ax.grid(alpha=.25)
    fig.suptitle('因子增强对照｜数据至2026-10-20｜单边5bps｜各类事后最好，非独立验证',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(assets/'factor_extensions_comparison_20261109.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(13,16))
    for col,strength in enumerate((.25,.5)):
        matrix=np.array([[main.at[f'hvrev_k{k}_v{fast}_{slow}_a{strength:g}','sharpe'] for k in (3,5,10)]
                         for fast,slow in ((10,60),(20,60),(10,120),(20,120))])
        axes[0,col].imshow(matrix-main.at['default','sharpe'],cmap='RdYlGn',vmin=-.3,vmax=.3,aspect='auto')
        axes[0,col].set(xticks=range(3),xticklabels=[3,5,10],yticks=range(4),yticklabels=['10/60','20/60','10/120','20/120'],
                        xlabel='反转窗口k',ylabel='快/慢波动率',title=f'高波动反转 λ={strength:g}')
        m2=np.array([[main.at[f'ivol_{model}_{w}_a{strength:g}','sharpe'] for w in (20,60,120)] for model in ('market','style')])
        axes[1,col].imshow(m2-main.at['default','sharpe'],cmap='RdYlGn',vmin=-.3,vmax=.3,aspect='auto')
        axes[1,col].set(xticks=range(3),xticklabels=[20,60,120],yticks=range(2),yticklabels=['市场','市场＋成长'],
                        xlabel='回归窗口',title=f'A股低IVOL λ={strength:g}')
        m3=np.array([[main.at[f'macd_{kind}_{f}_{s}_{g}_a{strength:g}','sharpe'] for f,s,g in ((6,13,5),(12,26,9),(24,52,18))]
                     for kind in ('trend','histogram')])
        axes[2,col].imshow(m3-main.at['default','sharpe'],cmap='RdYlGn',vmin=-.3,vmax=.3,aspect='auto')
        axes[2,col].set(xticks=range(3),xticklabels=['6/13/5','12/26/9','24/52/18'],yticks=range(2),yticklabels=['趋势线','柱状'],
                        xlabel='快/慢/信号EMA窗口',title=f'MACD λ={strength:g}')
        for ax,array in zip(axes[:3,col],(matrix,m2,m3)):
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):ax.text(j,i,f'{array[i,j]:.3f}',ha='center',va='center',fontsize=10)
    ivol_sr=np.array([[main.at[f'ivol_{model}_{w}_exclude_top_quartile','sharpe'] for w in (20,60,120)] for model in ('market','style')])
    ivol_dd=np.array([[-100*recent.at[f'ivol_{model}_{w}_exclude_top_quartile','drawdown'] for w in (20,60,120)] for model in ('market','style')])
    for col,array,title in [(0,ivol_sr,'IVOL高四分位过滤：2018以来夏普'),(1,ivol_dd,'IVOL高四分位过滤：2022以来回撤幅度%')]:
        axes[3,col].imshow(array-main.at['default','sharpe'] if col==0 else array,
                          cmap='RdYlGn' if col==0 else 'RdYlGn_r',vmin=-.3 if col==0 else 4,vmax=.3 if col==0 else 8,aspect='auto')
        axes[3,col].set(xticks=range(3),xticklabels=[20,60,120],yticks=range(2),yticklabels=['市场','市场＋成长'],xlabel='回归窗口',title=title)
        for i in range(2):
            for j in range(3):axes[3,col].text(j,i,f'{array[i,j]:.3f}' if col==0 else f'{array[i,j]:.2f}%',ha='center',va='center')
    fig.suptitle(f'参数敏感性｜前3行为2018以来夏普，默认{main.at["default","sharpe"]:.3f}｜末行另标注过滤与近期回撤',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(assets/'factor_extensions_sensitivity_20261109.png',dpi=150);plt.close(fig)
    diagnostics=pd.read_csv(OUT/'factor_diagnostics.csv')
    diag_names=list(dict.fromkeys([*CANONICAL,*[next(c for c in cases() if c.name==n).components[0][0]
                                              for n in (best['high_vol_reversal'],best['ivol'],best['macd'])]]))
    diag_rows=[]
    for n in diag_names:
        for horizon in (5,20):
            q=diagnostics.loc[diagnostics.name.eq(n)&diagnostics.horizon.eq(horizon)].iloc[0]
            diag_rows.append([n,horizon,int(q.weeks),f'{q.mean_rank_ic:.4f}',f'{q.matched_base_rank_ic:.4f}',
                              f'{q.predictive_oos_r2:.3%}',f'{q.original_score_rank_correlation:.3f}'])
    choices=json.loads((OUT/'time_ordered_choices.json').read_text(encoding='utf-8'))
    walkrows=[[q['train_end'],q['test_start']+'—'+q['test_end'],label(q['chosen']),
               f"{q['test_result']['annual']:.2%}/{q['test_result']['sharpe']:.3f}",
               f"{q['baseline']['annual']:.2%}/{q['baseline']['sharpe']:.3f}"] for q in choices]
    family=json.loads((OUT/'family_bootstrap.json').read_text(encoding='utf-8'))
    ext=pd.read_csv(OUT/'short_extension.csv').set_index('case')
    extrows=[[label(n),int(ext.at[n,'days']),f'{ext.at[n,"total"]:.2%}',f'{ext.at[n,"max_drawdown"]:.2%}']
             for n in dict.fromkeys(['default',*best.values(),winner])]
    hold=pd.read_csv(OUT/'holdings.csv',dtype={'symbol':str});holding_rows=[]
    for n in ('default',winner):
        for _,r in hold.loc[hold.case.eq(n)&hold.symbol.ne('511260')].nlargest(5,'average_weight').iterrows():
            holding_rows.append([label(n),r.symbol,ETF_NAMES.get(r.symbol,'红利低波卫星'),f'{r.average_weight:.1%}',f'{r.held_fraction:.1%}'])
    # 逐窗口邻域全表留档；不只展示获胜的单一点。
    group_rows=[]
    for f in ('high_vol_reversal','abnormal_volume','volume_reversal','ivol','macd','combination'):
        g=main.loc[main.family.eq(f)]
        count=int(g.sharpe.gt(main.at['default','sharpe']).sum())
        both=[n for n in g.index if g.at[n,'sharpe']>main.at['default','sharpe']
              and recent.at[n,'sharpe']>recent.at['default','sharpe']]
        group_rows.append([f,len(g),count,len(both),f'{g.sharpe.median():.3f}',f'{g.sharpe.max():.3f}'])
    reverse_control=[]
    c=next(c for c in cases() if c.name==best['high_vol_reversal']);n,strength=c.components[0];s=features()[n]
    counterpart=f'reversal_k{s.window}_v{s.slow}_a{strength:g}'
    reverse_control=metric_rows(main,['default',counterpart,c.name])
    warm_names=['default',best['macd_filter']]
    warm_names.insert(1,best['macd_filter'].replace('_positive_filter','_matched_warmup'))
    passed=manifest['strict_screen_passed'];screen_text='、'.join(label(n) for n in passed) if passed else '没有方案通过'
    amount=pd.read_csv(OUT/'amount_robustness.csv');amountrows=[]
    for _,r in amount.loc[amount.start.eq('2018-01-01')&amount.cost.eq(5)].iterrows():
        amountrows.append([label(r['case']),f'{main.at[r["case"],"annual"]:.2%}/{main.at[r["case"],"sharpe"]:.3f}',
                           f'{r.annual:.2%}/{r.sharpe:.3f}'])
    controls=pd.read_csv(OUT/'ivol_posthoc_controls.csv')
    control_labels={'static2_chosen_2018_2021':'2018-21固定选出的2行业剔除',
       'market_ivol_60_same_protected_broads':'市场IVOL60，保护相同宽基','market_ivol_120_same_protected_broads':'市场IVOL120，保护相同宽基',
       'total_volatility_60_same_protected_broads':'普通波动率60，保护相同宽基','total_volatility_120_same_protected_broads':'普通波动率120，保护相同宽基',
       'exclude_lowest_ivol_quartile':'反方向：剔除最低IVOL四分位'}
    control_rows=[]
    for n in controls.case.unique():
        aa=controls.loc[controls.case.eq(n)&controls.start.eq('2018-01-01')&controls.cost.eq(5)].iloc[0]
        bb=controls.loc[controls.case.eq(n)&controls.start.eq('2022-01-01')&controls.cost.eq(5)].iloc[0]
        control_rows.append([control_labels[n],f'{aa.annual:.2%}/{aa.sharpe:.3f}',f'{aa.drawdown:.2%}',
                             f'{bb.annual:.2%}/{bb.sharpe:.3f}',f'{bb.drawdown:.2%}'])
    control_design=json.loads((OUT/'ivol_posthoc_design.json').read_text(encoding='utf-8'))
    attribution=pd.read_csv(OUT/'log_return_attribution.csv',dtype={'symbol':str})
    logrows=[]
    diff=attribution.loc[attribution.start.eq('2018-01-01')].pivot(index='symbol',columns='case',values='log_growth_contribution')
    diff['increment']=diff['ivol_style_120_exclude_top_quartile']-diff['default']
    for s,r in diff.sort_values('increment',ascending=False).iterrows():
        logrows.append([s,ETF_NAMES.get(s,'交易成本' if s=='COST' else '红利卫星'),f'{r["default"]:.4f}',
                        f'{r["ivol_style_120_exclude_top_quartile"]:.4f}',f'{r["increment"]:+.4f}'])
    drawdown=pd.read_csv(OUT/'drawdown_episodes.csv')
    ddrows=[[label(r['case']),r.start,r.peak[:10],r.trough[:10],f'{r.maximum_drawdown:.2%}',r.worst_day_date[:10],f'{r.worst_day:.2%}'] for _,r in drawdown.iterrows()]
    text=f'''# 高波动反转、量能、IVOL代理与MACD增强研究

研究日期2026-11-09；主比较2018/2022现金起点至2026-10-20，辅助新增数据2026-10-21至2026-11-06。当前17只核心＋512890卫星，包含恒科、不包含标普。默认配置、公共切换状态、原实盘信号及个人账户均未改变。本轮不自动采用研究冠军。

## 结论概览

当前默认主区间年化{main.at['default','annual']:.2%}、夏普{main.at['default','sharpe']:.3f}、回撤{main.at['default','drawdown']:.2%}。本轮非诊断候选的事后最高夏普为{label(winner)}：年化{main.at[winner,'annual']:.2%}、夏普{main.at[winner,'sharpe']:.3f}、回撤{main.at[winner,'drawdown']:.2%}；2022起点{recent.at[winner,'annual']:.2%}/{recent.at[winner,'sharpe']:.3f}，20bps主区间{stress.at[winner,'annual']:.2%}/{stress.at[winner,'sharpe']:.3f}。

事先固定的严格筛选要求：两个起点在5/20bps下年化与夏普均不低于默认，且5bps两起点回撤均≤10%。结果：{screen_text}。这只是一组必要条件，仍需结合邻近参数、分段/时序选择、持仓集中度、成本和统计不确定性；通过也不代表消除过拟合或自动切换默认。

## 文献核对及迁移边界

1. [Gervais、Kaniel、Mingelgrin（2001），The High-Volume Return Premium](https://doi.org/10.1111/0022-1082.00349)：讨论异常成交量与随后股票收益的关系，不能直接作为“高波动率反转”或“放量后必反转”的证据。本轮分别测试相对量能溢价与量能×反转；后者是新增交互假设。ETF申赎、套利和规模扩张会影响量能，机制不等同个股关注度。
2. [Ang、Hodrick、Xing、Zhang（2006），The Cross-Section of Volatility and Expected Returns](https://doi.org/10.1111/j.1540-6261.2006.00836.x)：高特质波动股票的收益偏低，IVOL是特质波动率。原文含FF3残差与波动风险分析；本轮没有FF3和VIX创新因子，不声称完整复现。仅检验ETF行情可支持的含截距市场/成长风格残差模型，并主要在A股股票ETF内比较；全池市场模型仅作错配风险诊断。
3. [Fidelity MACD说明](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/macd)用于公式核对。MACD是EMA差及信号线的趋势/动能工具，既有原评分已经包含趋势信息，是否增加预测能力需要实测。

## 固定因子与参数

- 普通短期反转对照：k=3/5/10日，σ估计60/120日。短期冲击z=log(P_t/P_t-k)/(σ_过去、在冲击前结束×sqrt(k))，基础反转u=clip(−z/3,−1,1)。
- 高波动率反转：u×clip(σ_fast/σ_slow_冲击前−1,0,1)，fast=10/20，slow=60/120。只有近期波动超过冲击前背景波动时增强反转；仍保留原40日动量为正和MA40资格，避免绕过趋势条件。
- 量能溢价：u=clip(log(最近W日平均成交量/事件窗口之前B日平均成交量)/log(3),−1,1)，W=1/5、B=50/120。相对自身历史做比较，不按成交量绝对大小给大基金加分，也不是原论文分组实验的逐字复制。
- 量能反转：短反转u×max(异常量能u,0)，k=3/5/10、W=1/5、B=60。代表性参数另以成交额替代成交量，检查价格/份额量纲的影响。
- IVOL：过去N=20/60/120日的OLS残差样本标准差×sqrt252，包含截距。market为r_i~r_510300；style增加r_159915−r_510300成长代理。原始IVOL按本类资产当日排名转换为低波动高分的[−1,1]效用，至少3只有效资产。CSI300不回归自身；style还排除创业板自身，缺评分时中性，不人为给零残差资产优势。高IVOL剔除为排除本类最高四分位。
- MACD：DIF=EMA_fast−EMA_slow，DEA=EMA_signal(DIF)，柱=2(DIF−DEA)；参数6/13/5、12/26/9、24/52/18，分别比较趋势线与柱。趋势分DIF/(P×σ60_daily×sqrt(slow))；柱分HIST/(P×σ60_daily×sqrt(signal))，截为[−1,1]。不直接比较价格量纲不同的ETF裸MACD值，EMA仅使用历史，缺价后要求连续预热。

主叠加公式score_new=score_old+λ×D_old×u，λ=0.25/0.5；多因子组合各项固定0.25。D与原动量/均线反转评分相同，国债不进入排名；缺新因子时叠加回退原评分。MACD正值过滤必须有有效信号，并列同预热的原评分对照；IVOL剔除只针对可评分的本类资产，其余保留原资格。

40个原始因子、117个预先固定方案，共{len(table)}个主区间/成本结果，另有固定量能金额代理诊断。包括6个普通反转对照、12个高波动反转、4个量能溢价、6个量能反转、6个IVOL、6个MACD原始信号；所有差方案和范围诊断保留。没有按看到的结果继续细调参数。

## 数据与执行

主区间与前次17只默认回测使用同一收盘快照；补成交量/额后逐日核对旧收盘及复权因子，未重写旧快照。新增13日仅在独立研究缓存按同日因子追加，不改生产数据。[Tushare字段](https://tushare.pro/document/2?doc_id=127)：vol为手、amount为千元，金额诊断换算为元；自身比值中固定单位抵消。

全历史假设当前池与红利规则，但上市前及预热不足保持缺值，不补未上市指数历史。2018前仅预热；两个起点各从现金开始，起点区间相互重叠，并非两组独立样本。周末最后交易日收盘信号，下一交易日收盘执行；8%核心预算、70%核心风险上限、单只30%、3只核心风险、前4缓冲必选第1名、红利MA200入0出4/先现金再国债30%全部保留。叠加后全账户可能超过8%波动目标及70%风险仓，这不是账户硬上限。

5bps=单边3bps费用＋2bps滑点预算，压力测试20bps；252日年化、无风险收益0、每日收盘最大回撤。分数复权单位、按成交参考价确定目标数量、未模拟整手/最低佣金/QDII溢价/盘口容量等原模型限制保留。与A股宽基比较的超额也可能来自国债、黄金、境外资产配置，而非新增因子的单独预测能力。

## 2018年以来：各类事后最好

{markdown_table(headers,metric_rows(main,selected))}

## 2022年现金起点

{markdown_table(headers,metric_rows(recent,selected))}

![收益与回撤](assets/factor_extensions_comparison_20261109.png)

## 成本压力：单边20bps、2018起点

{markdown_table(headers,metric_rows(stress,selected))}

## 参数邻域、消融和范围

{markdown_table(['因子类','方案数','长期夏普提升数','两个起点夏普均提升数','长期夏普中位数','长期夏普最好'],group_rows)}

![参数敏感性](assets/factor_extensions_sensitivity_20261109.png)

高波动反转与相同k、相同背景σ、同强度的普通反转对照，区分“反转本身”与“只在波动放大时反转”的增益：

{markdown_table(headers,reverse_control)}

MACD过滤与同样预热资格的原评分对照，避免把延迟新ETF参与误当因子能力：

{markdown_table(headers,metric_rows(main,warm_names))}

IVOL主方案仅针对A股股票ETF，含石油产业、不含黄金、境外股指和商品期货ETF；全部成员在脚本A_STOCKS显式冻结。相对A股市场解释不掉的纳指/黄金涨跌不能直接称为不应持有的个股特质风险。全池IVOL诊断列于完整结果，未纳入可部署参数时序选择；也没有真实SMB/HML/Rf或ETF成分股逐日数据，因此结果只能称代理增强。

固定代表性量能方案的成交额替代对照（2018起点5bps，列年化/夏普）：

{markdown_table(['方案','成交量版本','成交额版本'],amountrows)}

## IC、预测R²与信息重复

以有效因子所在风险ETF为样本，周信号至少5只有效资产；标签从t+1交易日收盘进入，到其后5/20日收盘，末端不完整不纳入。IVOL风格模型可能更晚达到足够横截面数量；表内原评分IC严格使用同一日期和ETF样本。IC不预筛选实际持仓名单或未来胜者。

预测R²为逐年扩展的一元OLS，训练仅使用标签结束日在测试年之前的样本、至少200条，以该年训练均值为基准计算合并OOS R²。用于检查预测增益，不等于价格回归拟合度，也不是新的独立盲测。20日标签重叠、ETF相关，不能按原始观察数计算独立显著性。

{markdown_table(['因子','未来日','有效周','RankIC','同样本默认IC','预测OOS R²','与原评分秩相关'],diag_rows)}

完整80组诊断、逐周IC、逐ETF时序IC和波动率相关性见本地CSV。IVOL已与逆波动权重及原评分的波动率分母存在信息重叠，MACD也与20/40动量及MA40过滤重叠；需从增量表现而非因子名字判断价值。

## IVOL改善来源：后设归因，不继续寻优

观察到IVOL高四分位过滤较强后，另外固定6项解释对照（24个区间/成本结果）：与风格模型一样保护CSI300/创业板的市场IVOL、同样保护宽基的普通波动率过滤、反向剔除低IVOL，以及仅依据2018-2021屏蔽频率在2022以后固定剔除两行业。它们没有并入原117场景挑冠军，不能拿事后对照提高原家族的显著性结论。

固定剔除代码为{'、'.join(control_design['fixed_names'])}，2022之前仍用默认目标，未把训练段末才知道的名单反用于训练段。普通波动率对照检验是否只是低波动倾向；同宽基保护对照检验是否只是“创业板恰好不被筛掉”的结构差异。

{markdown_table(['后设对照','2018年化/夏普','长期回撤','2022年化/夏普','近期回撤'],control_rows)}

进一步把账户总对数净增长精确分为各资产收益贡献与成本。每期资产简单收益贡献乘log(1+组合毛收益)/组合毛收益，再累加；成本使用log(1−当日成本率)。它们相加还原log(期末NAV)，不是把不同资产买入持有收益直接相加。以下比较风格IVOL120过滤与默认，所有增量均列出，避免只展示赢家：

{markdown_table(['代码','资产','默认对数贡献','IVOL过滤对数贡献','差额'],logrows)}

最大回撤和最差单日的日期也需对比，防止总样本回撤改善掩盖近期风险上升：

{markdown_table(['方案','起点','峰值日','谷值日','最大回撤','最差单日日期','单日收益'],ddrows)}

## 时间顺序选择与统计不确定性

训练分别截至2021和2024，只用此前收益选择方案，下一段才使用相应周目标。各候选的缓冲状态作为虚拟子策略持续计算；使用同一账户模拟参数切换时的实际买卖及费用，不拼接不同账户收益。下一段已在项目中被观察，只称时间顺序诊断：

{markdown_table(['训练截止','测试段','训练选中','后段年化/夏普','默认年化/夏普'],walkrows)}

除6个预热控制外的111个方案做配对循环块抽样，每个块长1000次；20/60/120日块长的本轮家族平均日收益增益校正p分别为{', '.join(f"{q['family_mean_gain_p']:.3f}" for q in family)}。这个检验不是夏普差异检验，也不覆盖此前反复选池、因子及仓位试验。全样本事后冠军存在选择偏差；区间、费用或邻近参数不一致时不能仅凭一项最好指标推广。

## 新增13交易日：10月21日至11月6日

仅对事先指定或主区间已选出的代表性方案核对，未用于继续筛参或重新命名冠军。资产价格截至11月6日，当日信号只会在11月9日执行，未把未来成交收益放进该窗口。该段较短，且部分日期曾用于实盘查询，不能称严格独立盲测，不年化也不据此判断有效性：

{markdown_table(['方案','交易日','区间净收益','区间最大回撤'],extrows)}

## 集中度观察

默认和事后最高夏普方案，列平均权重最高的5只风险资产（均包含红利卫星，不列国债）；均值按全部交易日计算。换成新评分后若收益主要依赖少数资产，需要结合此前选池偏差继续观察：

{markdown_table(['方案','代码','ETF','平均权重','持有日占比'],holding_rows)}

## 验证与复现

{len(checks)}项研究检查全部通过：仓位约束、默认与此前17只目标/当前生产组合对账、代表性目标与生产选仓对账、5/20bps逐日净值对账、多截点原始价格与量能截断/未来扰动，以及新增数据不改变过去可执行目标和已实现净值。主区间截止10月20日为周二，截断价格日历末尾可能保留一个尚无下一成交日的研究占位目标；延长日历时仅核对已有下一成交日的历史目标，并逐日核对整个旧区间的收益，未将该占位目标当作已交易信号。IVOL另与直接含截距OLS核对，MACD/量能做价格和份额单位不变性与缓存失效测试。所有实盘配置/状态/原信号指纹未变；个人账户数据库未读写。

```powershell
python research_factor_extensions.py --download-only
python research_factor_extensions.py
python research_factor_extensions.py --report-only
```

研究高低量/价格缓存`data/factor_extensions_20261109/`，完整结果`output/factor_extensions_20261109/`，均Git忽略。因子定义`etf_rotation/extended_factors.py`，`FactorStore.get_extended(prices,spec,activity)`可直接缓存复用，不必改实盘选仓。默认保持当前17只核心＋红利，RSRS仍观察，未自动增加上述任何增强。
'''
    (ROOT/'docs/factor_extensions_20261109.md').write_text(text,encoding='utf-8')
    print('Report: docs/factor_extensions_20261109.md',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download-only',action='store_true');p.add_argument('--report-only',action='store_true')
    args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'plan.json').exists():write_json(OUT/'plan.json',plan())
    if args.download_only:download();return
    if not args.report_only:execute()
    build_report()


if __name__=='__main__':main()
