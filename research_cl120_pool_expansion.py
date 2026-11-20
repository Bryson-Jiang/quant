"""CL120大行业扩池研究；不修改默认池、实盘信号或真实账户。

python research_cl120_pool_expansion.py --screen
python research_cl120_pool_expansion.py --download
python research_cl120_pool_expansion.py
python research_cl120_pool_expansion.py --report-only
"""
from __future__ import annotations
import argparse
from dataclasses import asdict,dataclass,replace
from datetime import date
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from etf_rotation.config import LEGACY_CORE_NAMES
from etf_rotation.data import _token,_windows,_normalize_history,_normalize_table,load_symbol,tushare_code
from etf_rotation.extended_factors import ExtendedFactorSpec,compute_extended,low_ivol_utility
from etf_rotation.factors import FactorSpec,compute_factor
from etf_rotation.risk_strategy import RiskConfig,risk_score_components,risk_controlled_signals,weekly_signal_dates
from etf_rotation.mean_reversion import ReversionRule,reversion_states,defensive_funding_targets
from etf_rotation.portfolio_strategy import portfolio_signals
from etf_rotation.strategy_state import IVOLPolicy,load_pre_ivol_rotation_state
from etf_rotation.backtest import run_backtest
from research_overfitting_audit import CoreAudit,simulate,statistics
from research_pool_expansion import historical_eligibility,paired_bootstrap,markdown_table

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'data/cl120_pool_expansion_20261109';OUT=ROOT/'output/cl120_pool_expansion_20261109'
CORE_CACHE=ROOT/'data/factor_extensions_20261109'
END=date(2026,11,6);SAT='512890'
CORE_NAMES={**LEGACY_CORE_NAMES,'513180':'恒生科技ETF（华夏）'}
A_BASE=('510300','159915','159995','159875','159928','512660','515070','512880','512800','561360')
SCREEN={
 '159938':('医药ETF广发','医药卫生：制药、器械、服务等，补大健康缺口'),
 '159611':('电力ETF广发','电力公用事业，区别于新能源设备制造'),
 '512200':('房地产ETF南方','房地产开发与运营'),
 '159870':('化工ETF鹏华','多类化工产业，补材料周期'),
 '515220':('煤炭ETF国泰','煤炭能源，和石油有部分重叠'),
 '512400':('有色金属ETF南方','有色股票，区别于现有有色期货'),
 '512070':('证券保险ETF易方达','非银金融含保险，与原证券ETF重叠；另测替换'),
 '159825':('农业ETF富国','种植、养殖及农业产业，区别于豆粕期货'),
 '516110':('汽车ETF国泰','整车与零部件，区别于只覆盖新能源车'),
 '159996':('家电ETF国泰','家电全产业，补可选消费'),
 '159936':('可选消费ETF广发','较宽的可选消费'),
 '159944':('材料ETF广发','原材料大行业'),
 '515210':('钢铁ETF国泰','钢铁全行业'),
 '516970':('基建ETF广发','基础设施与建筑工程'),
 '159666':('交通运输ETF华夏','航空、铁路、港口、航运、物流等运输'),
 '561320':('交运ETF国泰','内地运输主题'),
 '159683':('运输ETF嘉实','运输历史产品，用于记录清盘/终止状态，不复活交易'),
}
BASE=RiskConfig(target_vol=.08,short_momentum_days=20,long_momentum_days=40,momentum_skip_days=0,
                trend_days=40,max_single_risky_weight=.30,rank_buffer_size=4,mandatory_top_rank=1,
                ma_reversion_days=20,ma_reversion_strength=.5)

def json_file(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8')

def protected():
    names=['etf_rotation/config.py','etf_rotation/risk_strategy.py','etf_rotation/portfolio_strategy.py',
           'etf_rotation/ivol_filter.py','etf_rotation/strategy_state.py','etf_rotation/extended_factors.py',
           'etf_rotation/factor_store.py','live_signal.py','run_backtest.py','data/rotation_state.json','output/live_signal.json']
    return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names if (ROOT/n).exists()}

def state():
    old=load_pre_ivol_rotation_state(ROOT/'data/rotation_state_20261109.json')
    policy=IVOLPolicy('510300',120,3,.75,A_BASE,date(2026,11,9),date(2026,11,6),date(2026,11,9),
                      'weekly_core17_vol8_ma20a05_div512890_ma200_in0_out4_defense30',
                      'weekly_17_etf_fast_40_20_vol8_ma20a05_buffer4_top1_single30')
    return replace(old,ivol_policy=policy,core_symbols=tuple(CORE_NAMES))

class Client:
    def __init__(self):self.session=requests.Session();self.token=_token()
    def query(self,method,params,fields=''):
        try:
            response=self.session.post('https://api.tushare.pro',json={'api_name':method,'token':self.token,
                        'params':params,'fields':fields},timeout=30,allow_redirects=False)
            response.raise_for_status();result=response.json()
            if result.get('code')!=0:raise ValueError('Provider refused query')
            return pd.DataFrame(result['data']['items'],columns=result['data']['fields'])
        except Exception as exc:
            raise RuntimeError(f'{method}: {type(exc).__name__}; private request details withheld') from None

def fetch_cached(client,path,method,params,fields=''):
    if path.exists():return pd.read_csv(path,dtype=str)
    frame=client.query(method,params,fields)
    frame.to_csv(path,index=False,encoding='utf-8-sig')
    return frame

def screen_plan():
    return {'research_date':'2026-11-09','end':END,'screen':SCREEN,'main_asset_floor_yi':20.,
            'observation_asset_floor_yi':10.,'minimum_median_amount20_yuan':10_000_000,
            'founded_on_or_before':'20211231','freeze':'先按行业、年龄、规模和流动性筛选，之后才读取完整价格回测；不因回测更换同类代表',
            'grid':'默认、各单只、主层全部两两组合、事先指定医药电力/资源/内需篮子、全主层、主层加观察层、非银替换证券；不寻优因子或仓位参数',
            'cost_bps':[5.,20.],'cash_starts':['2018-01-01','2022-01-01'],
            'ivol_scope':'新增A股ETF也加入IVOL比较，移除的旧ETF不再参加；门槛75%、120日不变',
            'robustness':'新增252真实报价、前一日20日成交额中位数>=1000万元、前一日份额估计市值>=2亿元；另查2日份额滞后',
            'mechanism_diagnostics':'固定旧池评分离散度D；禁止新增ETF入选但保留其对D及IVOL横截面的影响；均为归因而非额外选优',
            'protected_before':protected(),'base_config':asdict(BASE)}

def screen():
    CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'screen_plan.json').exists():json_file(OUT/'screen_plan.json',screen_plan())
    client=Client()
    basic_path=CACHE/'fund_basic_snapshot.csv'
    if not basic_path.exists():
        frames=[client.query('fund_basic',{'market':'E','status':s},
                 'ts_code,name,found_date,list_date,delist_date,benchmark,status') for s in ('L','D')]
        pd.concat(frames,ignore_index=True).to_csv(basic_path,index=False,encoding='utf-8-sig')
    basic=pd.read_csv(basic_path,dtype=str).set_index('ts_code');rows=[]
    for symbol,(name,coverage) in SCREEN.items():
        meta=basic.loc[tushare_code(symbol)]
        row={'symbol':symbol,'name':name,'coverage':coverage,'found_date':meta.found_date,'list_date':meta.list_date,
             'status':meta.status,'benchmark':meta.benchmark,'nav_date':None,'share_date':None,
             'assets_yi':None,'median_amount20_yuan':None,'tier':'excluded','reason':''}
        if meta.status!='L':row['reason']='非存续产品，不复活历史清盘ETF'
        elif str(meta.found_date)>'20211231':row['reason']='2021年末以后成立，本轮历史过短'
        else:
            params={'ts_code':tushare_code(symbol),'start_date':'20260901','end_date':END.strftime('%Y%m%d')}
            daily=fetch_cached(client,CACHE/f'recent_daily_{symbol}.csv','fund_daily',params,'ts_code,trade_date,close,vol,amount')
            shares=fetch_cached(client,CACHE/f'recent_share_{symbol}.csv','fund_share',params,'ts_code,trade_date,fd_share')
            nav=fetch_cached(client,CACHE/f'recent_nav_{symbol}.csv','fund_nav',params)
            if daily.empty or shares.empty or nav.empty:raise ValueError(f'{symbol}: missing screening data')
            daily=daily.assign(trade_date=daily.trade_date.astype(str)).sort_values('trade_date')
            shares=shares.assign(trade_date=shares.trade_date.astype(str)).sort_values('trade_date')
            nav=nav.assign(nav_date=nav.nav_date.astype(str)).sort_values('nav_date')
            nav=nav.loc[nav.nav_date.le(END.strftime('%Y%m%d'))]
            if 'ann_date' in nav:nav=nav.loc[nav.ann_date.fillna('99999999').astype(str).le('20261109')]
            n=nav.iloc[-1];available=shares.loc[shares.trade_date.le(n.nav_date)]
            if available.empty:raise ValueError('No known shares before NAV date')
            share=available.iloc[-1]
            gap=int(daily.trade_date.between(str(share.trade_date),str(n.nav_date),inclusive='right').sum())
            if gap>5:raise ValueError('Screening shares too stale')
            value=float(share.fd_share)*10000*float(n.unit_nav)/1e8
            amount=float(pd.to_numeric(daily.amount).tail(20).median()*1000)
            row.update(nav_date=str(n.nav_date),share_date=str(share.trade_date),assets_yi=value,median_amount20_yuan=amount)
            if len(daily)<20:row['reason']='20日流动性历史不足'
            elif amount<10_000_000:row['reason']='近期20日成交额中位数低于1000万元'
            elif value>=20:row.update(tier='main',reason='规模和流动性合格')
            elif value>=10:row.update(tier='observation',reason='10—20亿元观察层，单列检验')
            else:row['reason']='规模低于10亿元观察门槛'
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT/'screening.csv',index=False,encoding='utf-8-sig')
        print(f'screen {symbol}: {row["tier"]}; assets_yi={row["assets_yi"]}',flush=True)
    spec={'main':[r['symbol'] for r in rows if r['tier']=='main'],
          'observation':[r['symbol'] for r in rows if r['tier']=='observation']}
    json_file(OUT/'selected_universe.json',spec)
    json_file(OUT/'experiment_plan.json',{**screen_plan(),**spec,'cases':[asdict(c) for c in experiment_cases(spec)]})

def download():
    if not (OUT/'selected_universe.json').exists():screen()
    client=Client();spec=json.loads((OUT/'selected_universe.json').read_text(encoding='utf-8'))
    included=[*spec['main'],*spec['observation']];status=[]
    for symbol in [*CORE_NAMES,SAT]:
        for suffix in ('.csv','.json'):
            src=CORE_CACHE/f'tushare_{symbol}_hfq{suffix}';dst=CACHE/src.name
            if not dst.exists():shutil.copy2(src,dst)
    for symbol in included:
        old_folder=ROOT/'data/pool_expansion_20261031'
        daily_path=CACHE/f'daily_{symbol}.csv';factor_path=CACHE/f'adj_{symbol}.csv'
        prefix=CACHE/f'tushare_{symbol}_hfq'
        old_price=old_folder/f'tushare_{symbol}_hfq.csv'
        if not Path(str(prefix)+'.csv').exists():
            if old_price.exists():
                old=_normalize_history(pd.read_csv(old_price),symbol)
                daily=pd.read_csv(CACHE/f'recent_daily_{symbol}.csv',dtype={'trade_date':str})
                adj=fetch_cached(client,factor_path,'fund_adj',{'ts_code':tushare_code(symbol),'start_date':'20260901','end_date':END.strftime('%Y%m%d')},'ts_code,trade_date,adj_factor')
                raw=_normalize_table(daily,'close',symbol);factor=_normalize_table(adj,'adj_factor',symbol)
                overlap=old.index.intersection(raw.index)
                np.testing.assert_allclose(old.loc[overlap,'raw_close'],raw.reindex(overlap),rtol=0,atol=1e-9)
                np.testing.assert_allclose(old.loc[overlap,'adj_factor'],factor.reindex(overlap),rtol=1e-9)
                extra=raw.loc[raw.index>old.index[-1]];f=factor.reindex(extra.index)
                if f.isna().any():raise ValueError('Missing extension adjustment')
                joined=pd.concat([old,pd.DataFrame({'raw_close':extra,'adj_factor':f,'close':extra*f})])
                old_daily=pd.read_csv(old_folder/f'daily_{symbol}.csv',dtype={'trade_date':str})
                pd.concat([old_daily,daily]).drop_duplicates('trade_date',keep='last').to_csv(daily_path,index=False,encoding='utf-8-sig')
            else:
                frames={}
                for method,path,fields in [('fund_daily',daily_path,'ts_code,trade_date,close,vol,amount'),('fund_adj',factor_path,'ts_code,trade_date,adj_factor')]:
                    if path.exists():frames[method]=pd.read_csv(path,dtype={'trade_date':str});continue
                    parts=[client.query(method,{'ts_code':tushare_code(symbol),'start_date':a.strftime('%Y%m%d'),'end_date':b.strftime('%Y%m%d')},fields)
                           for a,b in _windows(date(2010,1,1),END)]
                    frames[method]=pd.concat(parts,ignore_index=True)
                    frames[method].to_csv(path,index=False,encoding='utf-8-sig')
                raw=_normalize_table(frames['fund_daily'],'close',symbol)
                factor=_normalize_table(frames['fund_adj'],'adj_factor',symbol).reindex(raw.index)
                if factor.isna().any():raise ValueError('Missing same-day fund adjustment')
                joined=pd.DataFrame({'raw_close':raw,'adj_factor':factor,'close':raw*factor})
            joined.index.name='date';joined=_normalize_history(joined.reset_index(),symbol)
            joined.to_csv(Path(str(prefix)+'.csv'),encoding='utf-8-sig')
            json_file(Path(str(prefix)+'.json'),{'source':'tushare','schema_version':2,'symbol':symbol,'adjust':'hfq','start':'2010-01-01','end':str(END)})
        share_path=CACHE/f'share_{symbol}.csv'
        if not share_path.exists():
            old_share=old_folder/f'share_{symbol}.csv'
            if old_share.exists():
                frames=[pd.read_csv(old_share,dtype={'trade_date':str}),pd.read_csv(CACHE/f'recent_share_{symbol}.csv',dtype={'trade_date':str})]
            else:
                frames=[client.query('fund_share',{'ts_code':tushare_code(symbol),'start_date':a.strftime('%Y%m%d'),'end_date':b.strftime('%Y%m%d')},'ts_code,trade_date,fd_share')
                        for a,b in _windows(date(2010,1,1),END)]
            pd.concat(frames,ignore_index=True).drop_duplicates('trade_date',keep='last').to_csv(share_path,index=False,encoding='utf-8-sig')
        status.append({'symbol':symbol,'complete':True})
        json_file(OUT/'download_status.json',status);print(f'download {symbol}: complete',flush=True)

@dataclass(frozen=True)
class Case:
    name:str
    added:tuple[str,...]=()
    removed:tuple[str,...]=()
    family:str='main'

def experiment_cases(spec):
    main=spec['main'];observe=spec['observation'];result=[Case('default')]
    result += [Case('add_'+s,(s,)) for s in main]
    result += [Case('add_'+s,(s,),family='observation') for s in observe]
    result += [Case('add_'+'_'.join(c),tuple(c)) for c in itertools.combinations(main,2)]
    baskets=[('health_power',('159938','159611')),('resources',('515220','159870','512400')),
             ('domestic_demand',('512200','516110','159938'))]
    for name,members in baskets:
        members=tuple(s for s in members if s in main)
        if len(members)>=2:result.append(Case('basket_'+name,members))
    result.append(Case('all_main',tuple(main)))
    if observe:result.append(Case('all_with_observation',tuple(main+observe),family='observation'))
    if '512070' in main:result.append(Case('replace_securities_with_nonbank',('512070',),('512880',)))
    seen=set();out=[]
    for c in result:
        key=(tuple(sorted(c.added)),tuple(sorted(c.removed)))
        if key in seen:continue
        seen.add(key);out.append(c)
    return out

def read_data():
    spec=json.loads((OUT/'selected_universe.json').read_text(encoding='utf-8'))
    included=spec['main']+spec['observation']
    series={s:load_symbol(s,date(2010,1,1),END,CACHE,offline=True) for s in [*CORE_NAMES,*included,SAT]}
    return pd.DataFrame(series,index=series['510300'].index).rename_axis('date'),spec

def admissions(prices,included,lag=1):
    allowed=pd.DataFrame(True,index=prices.index,columns=prices.columns)
    for s in included:
        raw=_normalize_history(pd.read_csv(CACHE/f'tushare_{s}_hfq.csv'),s)
        daily=pd.read_csv(CACHE/f'daily_{s}.csv',dtype={'trade_date':str})
        amount=pd.Series(pd.to_numeric(daily.amount).to_numpy()*1000,index=pd.to_datetime(daily.trade_date,format='%Y%m%d')).sort_index().reindex(prices.index)
        share=pd.read_csv(CACHE/f'share_{s}.csv',dtype={'trade_date':str})
        units=pd.Series(pd.to_numeric(share.fd_share).to_numpy()*10000,index=pd.to_datetime(share.trade_date,format='%Y%m%d')).sort_index().reindex(prices.index).ffill(limit=5)
        value=units*raw.raw_close.reindex(prices.index)
        allowed[s]=(prices[s].notna().cumsum().ge(252)&prices[s].notna()
                    &amount.rolling(20,min_periods=20).median().shift(1).ge(10_000_000)
                    &value.shift(lag).ge(200_000_000))
    return allowed

def build_targets(prices,case,*,allowed=None,diagnostic=None,production=False):
    symbols=[s for s in CORE_NAMES if s not in case.removed]+list(case.added)
    core=prices[symbols];policy=state().ivol_policy
    frames={spec:compute_factor(core,spec) for spec in [FactorSpec('momentum',20),FactorSpec('momentum',40),
             FactorSpec('volatility',60),FactorSpec('trend',40),FactorSpec('momentum',60),FactorSpec('ma_zscore',20)]}
    admission=pd.DataFrame(True,index=core.index,columns=core.columns) if allowed is None else allowed.loc[:,symbols]
    for spec in (FactorSpec('momentum',20),FactorSpec('momentum',40)):frames[spec]=frames[spec].where(admission)
    components=risk_score_components(frames[FactorSpec('momentum',20)],frames[FactorSpec('momentum',40)],
                                      frames[FactorSpec('volatility',60)],BASE,frames[FactorSpec('ma_zscore',20)])
    score=components['score']
    if diagnostic=='fixed_old_spread':
        old=prices[list(CORE_NAMES)]
        c=risk_score_components(compute_factor(old,FactorSpec('momentum',20)),compute_factor(old,FactorSpec('momentum',40)),
                                 compute_factor(old,FactorSpec('volatility',60)),BASE,compute_factor(old,FactorSpec('ma_zscore',20)))
        spread=c['base_score'].drop(columns='511260').std(axis=1,ddof=1).fillna(0)
        ma=(-frames[FactorSpec('ma_zscore',20)]/3).clip(-1,1).fillna(0).mul(BASE.ma_reversion_strength*spread,axis=0);ma['511260']=0.
        score=components['base_score']+ma
    peers=[s for s in A_BASE if s not in case.removed]+list(case.added)
    ivol=compute_extended(core,ExtendedFactorSpec('ivol',window=120,model='market')).where(admission)
    utility=low_ivol_utility(ivol,peers)
    mask=(utility.ge(-.5)|utility.isna())&admission
    if diagnostic=='new_assets_not_selectable':mask.loc[:,list(case.added)]=False
    if production:
        core_targets=risk_controlled_signals(core,BASE,factor_frames=frames,eligibility_mask=mask,
                         score_overlay=score-components['score'],**state().signal_kwargs(core))
    else:core_targets=CoreAudit(core,state()).signals(BASE,score_override=score,eligibility_mask=mask)
    p=core.copy();p[SAT]=prices[SAT]
    sat=reversion_states(p[[SAT]],SAT,ReversionRule(200,0,.04,None),weekly_signal_dates(p.index))
    return p,defensive_funding_targets(p,core_targets,sat,SAT,.30),core_targets,mask

def holding_rows(p,ledger,name):
    weights=pd.DataFrame(ledger.weights,index=ledger.dates,columns=p.columns);rows=[]
    for s in p:
        held=weights[s].gt(1e-10);groups=held.ne(held.shift(fill_value=False)).cumsum()
        lengths=held.loc[held].groupby(groups.loc[held]).size()
        rows.append({'case':name,'symbol':s,'average_weight':float(weights[s].mean()),'max_drifted_weight':float(weights[s].max()),
                     'held_days':int(held.sum()),'held_fraction':float(held.mean()),'episodes_including_open':len(lengths),
                     'median_episode_days':float(lengths.median()) if len(lengths) else 0.,'open_at_end':bool(held.iloc[-1])})
    return rows

def evaluate(p,t,name,**extra):
    rows=[]
    for start in ('2018-01-01','2022-01-01'):
        for cost in (5.,20.):
            led=simulate(p,t,pd.Timestamp(start),cost_bps=cost)
            rows.append({'case':name,'start':start,'cost':cost,'skipped':led.skipped,
                         'annual_turnover':float(led.turnover.sum()*252/len(led.dates)),**extra,**statistics(led.returns)})
    return rows

def execute():
    before=protected();prices,spec=read_data();entries=experiment_cases(spec)
    frozen=json.loads((OUT/'experiment_plan.json').read_text(encoding='utf-8'))
    assert frozen['cases']==json.loads(json.dumps([asdict(c) for c in entries])),'Experiment plan changed'
    if frozen['protected_before']!=before:raise ValueError('Production changed after freezing this experiment')
    included=spec['main']+spec['observation'];allowed=admissions(prices,included)
    results=[];holdings=[];daily={};recent={};data={};checks=[]
    targets_dir=OUT/'targets';targets_dir.mkdir(exist_ok=True)
    for i,case in enumerate(entries):
        p,t,core,mask=build_targets(prices,case);data[case.name]=(p,t)
        results.extend(evaluate(p,t,case.name,family=case.family,added=','.join(case.added),removed=','.join(case.removed)))
        led=simulate(p,t,pd.Timestamp('2018-01-01'))
        daily[case.name]=led.series();holdings.extend(holding_rows(p,led,case.name))
        recent[case.name]=simulate(p,t,pd.Timestamp('2022-01-01')).series()
        t.to_csv(targets_dir/f'{case.name}.csv',encoding='utf-8-sig')
        risky=core.drop(columns='511260')
        valid=bool(risky.max().max()<=.3+1e-10 and risky.sum(axis=1).max()<=.7+1e-10
                   and risky.gt(1e-10).sum(axis=1).max()<=3 and t.sum(axis=1).max()<=1+1e-10
                   and t.gt(1e-10).sum(axis=1).max()<=5)
        for s in case.added:
            valid &= not bool(t.loc[t.index<p[s].first_valid_index(),s].gt(0).any())
        filtered=core.where(~mask.reindex(core.index),0.)
        valid &= bool(filtered.abs().max().max()<1e-10)
        checks.append({'check':case.name+'_listing_limits_and_ivol','passed':valid})
        print(f'case {i+1}/{len(entries)} {case.name}',flush=True)
    table=pd.DataFrame(results);table.to_csv(OUT/'all_results.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(holdings).to_csv(OUT/'holdings.csv',index=False,encoding='utf-8-sig')
    daily=pd.DataFrame(daily);recent=pd.DataFrame(recent)
    daily.to_csv(OUT/'daily_2018.csv',encoding='utf-8-sig');recent.to_csv(OUT/'daily_2022.csv',encoding='utf-8-sig')
    main=table.query("start=='2018-01-01' and cost==5 and family=='main'").set_index('case')
    champion=main.sort_values(['sharpe','annual'],ascending=False).index[0]
    nondefault=main.drop(index='default').sort_values(['sharpe','annual'],ascending=False).index[0]
    by_name={c.name:c for c in entries};representatives=list(dict.fromkeys(['default','all_main',champion,nondefault,
                      *['add_'+s for s in included],*(['replace_securities_with_nonbank'] if '512070' in spec['main'] else [])]))
    default=portfolio_signals(prices[list(CORE_NAMES)],prices[[SAT]],state(),BASE,
                 hypothetical_satellite_history=True,hypothetical_universe_history=True,hypothetical_ivol_history=True)
    err=float((default.signals-data['default'][1]).abs().to_numpy().max())
    checks.append({'check':'default_CL120_production_match','error':err,'passed':err<1e-10})
    for name in representatives:
        p,t=data[name];_,production,_,_=build_targets(prices,by_name[name],production=True)
        err=float((production-t).abs().to_numpy().max())
        checks.append({'check':name+'_production_targets','error':err,'passed':err<1e-10})
        if name in {'default','all_main',champion,nondefault}:
            for cost in (5.,20.):
                a=run_backtest(p,t,date(2018,1,1),END,fee_bps=0,slippage_bps=cost)
                b=simulate(p,t,pd.Timestamp('2018-01-01'),cost_bps=cost)
                err=float(np.max(np.abs(a.equity.daily_return.to_numpy()-b.returns)))
                checks.append({'check':name+f'_ledger_cost{cost:g}','error':err,'passed':err<1e-10})
    robust=[];robust_targets={}
    for i,case in enumerate(entries):
        p,t,core,mask=build_targets(prices,case,allowed=allowed)
        robust.extend(evaluate(p,t,case.name));robust_targets[case.name]=(p,t)
        checks.append({'check':case.name+'_historical_admission','passed':bool(core.where(~allowed.reindex(index=core.index,columns=core.columns),0.).abs().max().max()<1e-10)})
        print(f'admission {i+1}/{len(entries)} {case.name}',flush=True)
    pd.DataFrame(robust).to_csv(OUT/'historical_admission.csv',index=False,encoding='utf-8-sig')
    lag2=admissions(prices,included,lag=2);delay_rows=[]
    for name in representatives:
        p,t,_,_=build_targets(prices,by_name[name],allowed=lag2)
        delay_rows.extend(evaluate(p,t,name))
    pd.DataFrame(delay_rows).to_csv(OUT/'two_day_share_lag.csv',index=False,encoding='utf-8-sig')
    segments=[]
    for name in daily:
        for period,a,b in [('2018-2021','2018','2021'),('2022-2024','2022','2024'),('2025+','2025',str(END))]:
            segments.append({'case':name,'period':period,**statistics(daily.loc[a:b,name])})
    pd.DataFrame(segments).to_csv(OUT/'segments.csv',index=False,encoding='utf-8-sig')
    eligible=[c.name for c in entries if c.family=='main'];walk=data['default'][1].reindex(columns=prices.columns,fill_value=0.)
    choices=[]
    for train_end,test_start,test_end in [('2021-12-31','2022-01-01','2024-12-31'),('2024-12-31','2025-01-01',str(END))]:
        scores=[(statistics(daily.loc[:train_end,n])['sharpe'],-len(by_name[n].added),n) for n in eligible]
        chosen=sorted(scores,reverse=True)[0][2]
        t=data[chosen][1].reindex(columns=prices.columns,fill_value=0.)
        interval=(walk.index>=pd.Timestamp(test_start))&(walk.index<=pd.Timestamp(test_end))
        walk.loc[interval]=t.loc[interval]
        choices.append({'train_end':train_end,'test_start':test_start,'test_end':test_end,'chosen':chosen})
    account=simulate(prices,walk,pd.Timestamp('2018-01-01')).series()
    for choice in choices:
        for name,x in [('continuous_switch_account',account),('default',daily['default'])]:
            choice[name]=statistics(x.loc[choice['test_start']:choice['test_end']])
    json_file(OUT/'time_ordered_choices.json',choices);walk.to_csv(OUT/'time_ordered_targets.csv')
    diagnostics=[];changes=[];_,base_t,_,base_mask=build_targets(prices,by_name['default'])
    for name in representatives:
        if name=='default':continue
        case=by_name[name]
        for mechanism in ('fixed_old_spread','new_assets_not_selectable'):
            p,t,_,mask=build_targets(prices,case,diagnostic=mechanism)
            diagnostics.extend(evaluate(p,t,name,diagnostic=mechanism))
            if mechanism=='new_assets_not_selectable':
                shared=[s for s in CORE_NAMES if s in p]
                weeks=t.index[t.index>=pd.Timestamp('2018-01-01')]
                changed=(t[shared]-base_t[shared]).abs().max(axis=1).gt(1e-10)
                flips=(mask[shared]!=base_mask[shared]).sum(axis=1).reindex(weeks)
                changes.append({'case':name,'weeks_with_old_target_change_without_new_holdings':int(changed.reindex(weeks).sum()),
                                'weeks_with_old_ivol_eligibility_change':int(flips.gt(0).sum()),'weeks':len(weeks)})
    pd.DataFrame(diagnostics).to_csv(OUT/'mechanism_diagnostics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(changes).to_csv(OUT/'cross_section_changes.csv',index=False,encoding='utf-8-sig')
    attribution=[]
    for name in dict.fromkeys(['default',champion,nondefault,'all_main']):
        p,t=data[name]
        for start in ('2018-01-01','2022-01-01'):
            led=simulate(p,t,pd.Timestamp(start));w=pd.DataFrame(led.weights,index=led.dates,columns=p.columns).shift(1).fillna(0.)
            contribution=w*p.ffill(limit=3).pct_change(fill_method=None).reindex(led.dates).fillna(0.)
            scale=np.ones(len(led.dates));nz=np.abs(led.gross)>1e-14;scale[nz]=np.log1p(led.gross[nz])/led.gross[nz]
            totals=contribution.mul(scale,axis=0).sum();cost=float(np.log1p(-led.turnover*.0005).sum())
            err=abs(float(totals.sum())+cost-float(np.log1p(led.returns).sum()))
            checks.append({'check':name+'_'+start+'_log_attribution','error':err,'passed':err<1e-10})
            attribution.extend({'case':name,'start':start,'symbol':s,'log_growth_contribution':float(v)} for s,v in totals.items())
            attribution.append({'case':name,'start':start,'symbol':'cost','log_growth_contribution':cost})
    pd.DataFrame(attribution).to_csv(OUT/'log_attribution.csv',index=False,encoding='utf-8-sig')
    correlations=prices.loc['2022':].pct_change(fill_method=None).corr(min_periods=252)
    correlations.to_csv(OUT/'correlations_2022.csv',encoding='utf-8-sig')
    corr_rows=[]
    for s in included:
        c=correlations.loc[s,[x for x in CORE_NAMES if x!='511260']].dropna()
        top=c.idxmax()
        corr_rows.append({'symbol':s,'max_core_correlation':float(c[top]),'most_correlated_core':top,
                          'market_correlation':float(c['510300'])})
    pd.DataFrame(corr_rows).to_csv(OUT/'candidate_correlations.csv',index=False,encoding='utf-8-sig')
    for name in dict.fromkeys(['default','all_main',champion,nondefault]):
        case=by_name[name]
        for cutoff in (pd.Timestamp('2020-12-25'),pd.Timestamp('2023-12-29'),pd.Timestamp('2025-12-26')):
            p,t=data[name];_,prefix,_,_=build_targets(prices.loc[:cutoff],case)
            perturbed=prices.copy();future=perturbed.index>cutoff
            perturbed.loc[future]*=np.linspace(.4,3.1,prices.shape[1])
            _,changed,_,_=build_targets(perturbed,case)
            err=max(float((prefix-t.loc[:cutoff]).abs().to_numpy().max()),float((changed.loc[:cutoff]-t.loc[:cutoff]).abs().to_numpy().max()))
            checks.append({'check':name+'_future_'+str(cutoff.date()),'error':err,'passed':err<1e-10})
    json_file(OUT/'family_bootstrap.json',paired_bootstrap(daily.to_numpy(),reps=1000))
    passing=[]
    base=table.loc[table.case.eq('default')].set_index(['start','cost'])
    for name in eligible:
        if name=='default':continue
        rows=table.loc[table.case.eq(name)].set_index(['start','cost'])
        if bool((rows.annual>=base.annual-1e-12).all() and (rows.sharpe>=base.sharpe-1e-12).all()
                and rows.loc[pd.IndexSlice[:,5.],'drawdown'].ge(-.10-1e-12).all()):passing.append(name)
    checks.append({'check':'production_state_and_live_unchanged','passed':protected()==before})
    json_file(OUT/'checks.json',checks)
    json_file(OUT/'manifest.json',{'end':END,'cases':len(entries),'main_rows':len(table),'champion':champion,
                'best_nondefault':nondefault,'strict_passed':passing,'main_candidates':spec['main'],'observation_candidates':spec['observation'],
                'protected_before':before,'protected_after':protected(),'all_checks_passed':all(q['passed'] for q in checks),
                'price_hash':hashlib.sha256(pd.util.hash_pandas_object(prices,index=True).values.tobytes()).hexdigest(),
                'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'default_changed':False})
    if not all(q['passed'] for q in checks):raise AssertionError('Research checks failed')
    print(main.sort_values('sharpe',ascending=False)[['annual','sharpe','drawdown']].head(8).to_string(),flush=True)

def label(name):
    if name=='default':return 'CL120默认'
    if name=='all_main':return '七行业全部加入'
    if name=='all_with_observation':return '主层＋农业/基建观察'
    if name=='replace_securities_with_nonbank':return '非银金融替换证券'
    if name=='basket_resources':return '煤炭＋化工＋有色股票'
    if name.startswith('basket_'):return name
    return '加'+'＋'.join(SCREEN[s][0].replace('ETF','') for s in name.removeprefix('add_').split('_'))

def report():
    table=pd.read_csv(OUT/'all_results.csv');meta=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    screening=pd.read_csv(OUT/'screening.csv',dtype={'symbol':str});spec=json.loads((OUT/'selected_universe.json').read_text(encoding='utf-8'))
    a=table.query("start=='2018-01-01' and cost==5").set_index('case')
    b=table.query("start=='2022-01-01' and cost==5").set_index('case')
    stress=table.query("start=='2018-01-01' and cost==20").set_index('case')
    names=list(dict.fromkeys(['default',*['add_'+s for s in spec['main']+spec['observation']],
                             'replace_securities_with_nonbank','all_main','all_with_observation',meta['best_nondefault']]))
    names=[n for n in names if n in a.index]
    def results(frame):
        return markdown_table(['方案','年化','夏普','最大回撤','双边年换手倍数'],
          [[label(n),f'{frame.at[n,"annual"]:.2%}',f'{frame.at[n,"sharpe"]:.3f}',f'{frame.at[n,"drawdown"]:.2%}',f'{frame.at[n,"annual_turnover"]:.2f}'] for n in names])
    candidate_rows=[]
    for r in screening.itertuples():
        candidate_rows.append([r.symbol,r.name,str(r.found_date),f'{r.assets_yi:.2f}' if pd.notna(r.assets_yi) else '未筛规模',
                   f'{r.median_amount20_yuan/1e8:.2f}' if pd.notna(r.median_amount20_yuan) else '—',r.tier,r.reason])
    daily=pd.read_csv(OUT/'daily_2018.csv',index_col=0,parse_dates=True)
    recent=pd.read_csv(OUT/'daily_2022.csv',index_col=0,parse_dates=True)
    chart_names=list(dict.fromkeys(['default',meta['best_nondefault'],'all_main','add_159611','replace_securities_with_nonbank']))
    chart_names=[n for n in chart_names if n in daily]
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
    fig,axes=plt.subplots(2,2,figsize=(15,9),sharex='col')
    for col,(returns,start) in enumerate(((daily,'2018'),(recent,'2022'))):
        for n in chart_names:
            nav=(1+returns[n]).cumprod();dd=nav/nav.cummax().clip(lower=1)-1
            axes[0,col].plot(nav.index,nav,label=label(n),linewidth=2 if n=='default' else 1.2)
            axes[1,col].plot(dd.index,dd*100,label=label(n),linewidth=2 if n=='default' else 1.2)
        axes[0,col].set_title(f'{start}现金起点：扣费净值');axes[0,col].legend(fontsize=8)
        axes[1,col].set_title('收盘回撤路径');axes[1,col].set_ylabel('%')
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle(f'CL120行业扩池｜单边5bps｜截至{END}｜全历史假设，非实盘',fontsize=14);fig.tight_layout()
    assets=ROOT/'docs/assets';assets.mkdir(parents=True,exist_ok=True)
    fig.savefig(assets/'cl120_pool_expansion_20261109.png',dpi=150);plt.close(fig)
    main=spec['main'];fig,axes=plt.subplots(2,2,figsize=(16,13))
    cases=experiment_cases(spec);lookup={tuple(sorted(c.added)):c.name for c in cases if not c.removed}
    for ax,frame,key,title,scale in [(axes[0,0],a,'annual','2018以来：年化增量/百分点',100),
                                    (axes[0,1],a,'sharpe','2018以来：夏普增量',1),
                                    (axes[1,0],b,'annual','2022以来：年化增量/百分点',100),
                                    (axes[1,1],b,'drawdown','2022以来：回撤改善/百分点（正值较浅）',100)]:
        grid=np.zeros((len(main),len(main)))
        for i,s in enumerate(main):
            for j,t in enumerate(main):
                name=lookup[tuple(sorted({s,t}))]
                grid[i,j]=(frame.at[name,key]-frame.at['default',key])*scale
        maximum=max(float(np.abs(grid).max()),1e-9)
        ax.imshow(grid,cmap='RdYlGn',vmin=-maximum,vmax=maximum)
        ticks=[SCREEN[s][0].replace('ETF','\n') for s in main]
        ax.set(xticks=range(len(main)),xticklabels=ticks,yticks=range(len(main)),yticklabels=ticks,title=title)
        for i in range(len(main)):
            for j in range(len(main)):ax.text(j,i,f'{grid[i,j]:+.2f}',ha='center',va='center',fontsize=8)
    fig.suptitle('固定七行业单只与两两扩池：对角线为单只｜未搜索因子参数',fontsize=15);fig.tight_layout()
    fig.savefig(assets/'cl120_pool_expansion_pairs_20261109.png',dpi=150);plt.close(fig)
    holding=pd.read_csv(OUT/'holdings.csv',dtype={'symbol':str});correlation=pd.read_csv(OUT/'candidate_correlations.csv',dtype={'symbol':str,'most_correlated_core':str})
    holdrows=[]
    for s in spec['main']+spec['observation']:
        r=holding.loc[holding.case.eq('add_'+s)&holding.symbol.eq(s)].iloc[0]
        c=correlation.loc[correlation.symbol.eq(s)].iloc[0]
        holdrows.append([s,SCREEN[s][0],int(r.held_days),f'{r.held_fraction:.1%}',f'{r.average_weight:.2%}',
                         int(r.episodes_including_open),f'{r.median_episode_days:.0f}',c.most_correlated_core,f'{c.max_core_correlation:.3f}'])
    robust=pd.read_csv(OUT/'historical_admission.csv');rmain=robust.query("start=='2018-01-01' and cost==5").set_index('case')
    rrecent=robust.query("start=='2022-01-01' and cost==5").set_index('case')
    robustrows=[[label(n),f'{rmain.at[n,"annual"]:.2%}/{rmain.at[n,"sharpe"]:.3f}/{rmain.at[n,"drawdown"]:.2%}',
                 f'{rrecent.at[n,"annual"]:.2%}/{rrecent.at[n,"sharpe"]:.3f}/{rrecent.at[n,"drawdown"]:.2%}'] for n in names]
    mechanism=pd.read_csv(OUT/'mechanism_diagnostics.csv');mr=mechanism.query("start=='2018-01-01' and cost==5")
    diag_names=list(dict.fromkeys([meta['best_nondefault'],'all_main','add_512400','replace_securities_with_nonbank']))
    diagrows=[[label(r.case),r.diagnostic,f'{r.annual:.2%}',f'{r.sharpe:.3f}',f'{r.drawdown:.2%}']
              for r in mr.itertuples() if r.case in diag_names]
    family=json.loads((OUT/'family_bootstrap.json').read_text(encoding='utf-8'))
    choices=json.loads((OUT/'time_ordered_choices.json').read_text(encoding='utf-8'))
    choice_rows=[[r['train_end'],label(r['chosen']),r['test_start']+'至'+r['test_end'],
                  f'{r["continuous_switch_account"]["annual"]:.2%}/{r["continuous_switch_account"]["sharpe"]:.3f}',
                  f'{r["default"]["annual"]:.2%}/{r["default"]["sharpe"]:.3f}'] for r in choices]
    segments=pd.read_csv(OUT/'segments.csv');segrows=[[label(r.case),r.period,f'{r.annual:.2%}',f'{r.sharpe:.3f}',f'{r.drawdown:.2%}']
                   for r in segments.itertuples() if r.case in {'default',meta['best_nondefault'],'all_main'}]
    checks=json.loads((OUT/'checks.json').read_text(encoding='utf-8'))
    winner=meta['best_nondefault'];passing='、'.join(label(n) for n in meta['strict_passed']) or '无'
    text=f'''# CL120行业扩池研究：补行业缺口，保持默认不变

研究日期2026-11-09，完整行情统一截至{END}。当前实盘为澄流120（CL120）；本轮只研究，未修改默认池、参数、实盘计划或真实账户。

与上轮截至10月20日的13.49%年化、1.553夏普略有差异，是因为统一追加至11月6日，未更改默认参数。本轮默认为全历史假设回放，不是11月9日才启用的实际账户记录。

## 结论与筛选结果

主层{len(spec['main'])}只：医药卫生、电力公用事业、房地产、化工、煤炭、有色股票、证券保险；观察层{len(spec['observation'])}只：农业、基建。当前池通过沪深300、创业板、纳指、红利等已间接覆盖部分行业，“缺少独立ETF”不等于完全没有敞口。现有159980为商品期货ETF，不是有色矿业股票；159928偏主要消费，可选消费仍缺独立代表。

本轮最强非默认方案为{label(winner)}，2018现金起点年化{a.at[winner,'annual']:.2%}、夏普{a.at[winner,'sharpe']:.3f}、回撤{a.at[winner,'drawdown']:.2%}；CL120默认为{a.at['default','annual']:.2%}、{a.at['default','sharpe']:.3f}、{a.at['default','drawdown']:.2%}。主层方案按2018起点夏普排序的第一名（含默认）为{label(meta['champion'])}。

预设跨区间/费用筛选：2018与2022两个现金起点、5/20bps下年化和夏普均不低于默认，且两个5bps起点回撤≤10%。通过：{passing}。通过也不等于消除过拟合，本轮不据此切换实盘。

35种主表方案（含默认）中，34种扩池/替换在两个现金起点、两档费用下，均没有年化或夏普高于CL120。历史准入后方向相同：扩大行业覆盖在本套评分与固定3个核心风险名额下没有体现稳定收益增益。建议继续使用当前池；医药、电力与有色股票可保留行业覆盖观察价值，化工是最接近默认但尚未胜出的候选，不为追求行业齐全立即实盘加入。

## 候选先筛规模，后看回测

先固定17个行业产品检查对象。主层门槛为2021年末前成立、存续、资产估计≥20亿元、近20日成交额中位数≥1000万元；10—20亿元单列观察，其余不进入回测。没有按回测挑同一行业的最佳基金，也未放宽门槛挽救汽车、家电或材料。

门槛筛选的是本轮列明的代表产品，不是宣称该行业所有ETF都不合格；原池作为既定对照保留，也不据当前门槛二次筛选。候选产品以行业覆盖与历史确定，没有在回测后改换同类代码。

规模为截至11月6日已存在的最新同日份额×单位净值估计净资产，单位亿元；成交额单位亿元。份额来自Tushare，万份乘10000；净值只用于当前筛选，未输入历史信号。NAV披露日不晚于研究日，当前规模仍带有后视选择偏差。

{markdown_table(['代码','产品','成立日','估计净资产/亿元','20日成交额中位数/亿元','层级','原因'],candidate_rows)}

## 对照口径与IVOL范围

原17只核心加512890卫星，因子、持仓上限、仓位预算和周度节奏固定。两次独立现金启动为2018-01-02、2022-01-04，2018年前仅预热；年化252日、夏普无风险利率0。默认与扩池均为全历史假设采用当前规则，并非真实采纳日期或个人账户收益。每周最后开市日收盘出信号、下一交易日收盘成交，单边5/20bps；包含自然权重漂移及双边换手，不假设无成本每日再平衡。

新增候选都是A股股票ETF，因此同样加入120日市场IVOL比较，移除的证券ETF不再参与比较；不让新ETF自动享受IVOL豁免。门槛始终是归一化IVOL排名>75%过滤、至少3只有效资产、缺因子中性。扩大比较集合会改变原资产被过滤的名单，绝非仅增加一个可选代码。独立红利卫星不参与核心IVOL或排名；全账户仍可能超过核心8%/70%预算，最多4只风险ETF加国债。

固定{meta['cases']}种方案：默认、全部单只、主层两两组合、预设资源篮子、全主层/含观察层、证券保险替换原证券。重复池只算一次；没有搜索新因子、仓位、窗口或过滤阈值。全部{len(table)}行主结果保留。汽车和家电的行业逻辑合理，但本轮规模不足；交运产品成立较晚，未为凑齐行业而纳入。

## 2018以来：单边5bps

{results(a)}

## 2022以来：单边5bps

{results(b)}

## 2018以来：单边20bps

{results(stress)}

![净值与回撤](assets/cl120_pool_expansion_20261109.png)

![固定单只及两两组合](assets/cl120_pool_expansion_pairs_20261109.png)

## 分段与按过去选择

分段继承2018现金账本，不等同于2022独立现金启动：

{markdown_table(['方案','区间','年化','夏普','回撤'],segrows)}

仅在主层方案中，训练截至2021/2024年底，下一年首个周信号后切换。候选各自缓冲状态只使用过去数据持续计算，将所选目标接入同一账户并支付实际换仓费用，没有拼接不同账户的收益。后段已经在项目中观察过，只能称时间顺序诊断，不能称独立盲测：

{markdown_table(['训练截止','所选','测试区间','连续切换账户年化/夏普','默认年化/夏普'],choice_rows)}

本轮所有方案的配对循环块抽样，每个块长1000次；20/60/120日块长家族均值收益增益p分别为{', '.join(f"{x['family_mean_gain_p']:.3f}" for x in family)}。这不是夏普差异检验，也不覆盖此前多轮因子、参数和选池搜索。

## 历史准入与份额延迟

新增候选要求真实报价至少252日、前一交易日已知的20日成交额中位数≥1000万元、前一日份额×原始价≥2亿元。份额只向前沿用最多5个交易日，不向后填充；未准入新资产也不影响评分D或IVOL比较集合。该对照只作用于新增候选，原池不另做全市场清盘池重建，因此不是完全无偏的历史选池。

{markdown_table(['方案','2018起点年化/夏普/回撤','2022起点年化/夏普/回撤'],robustrows)}

另将份额估计市值滞后2个交易日复验代表方案，结果留在`two_day_share_lag.csv`；不搜索滞后天数。日频历史份额仍可能修订，未拥有全部逐时发布时间，不能保证1/2日滞后覆盖所有信息延迟。

## 持有时间、相关性与增益来源

各行是仅新增该ETF时的2018账本。持有占比和平均仓位以全部交易日为分母，包括上市前；周期含末端开放周期，不把每次加减仓误算成独立完整获利回合。相关性用2022以来至少252个共同真实日收益，只作事后诊断，不进入选仓。

{markdown_table(['代码','产品','持有日','日占比','平均账户权重','周期数含开放','周期中位日','原池最相关','相关系数'],holdrows)}

扩池会通过两条路径改变旧仓：均线反转缩放D包含新增原始评分，IVOL尾部过滤的参照集合也变大。预先列明的解释对照如下；固定旧D仍使用扩展IVOL，禁止新增入选对照则只保留其横截面影响，不将这些对照另作优化冠军：

{markdown_table(['扩池方案','解释对照','年化','夏普','回撤'],diagrows)}

几个有实际意义的发现：

- 化工单独加入长期年化13.30%、夏普1.531，最接近默认；2022独立现金起点回撤却由4.25%扩大到6.06%。按过去选池时，2022—2024的连续账户略优，2025以后又落后，不能仅保留有利年份。
- 有色股票将长期最深回撤由9.62%降至8.81%，但2022起点回撤由4.25%升至7.35%，没有跨时期一致的对冲改善。
- 证券保险与原证券ETF日收益相关约0.935；“包含保险、覆盖更宽”不等于提供独立风险来源。单独加入和直接替换原证券，两者都较弱。
- 电力与原核心风险ETF的最高相关仅约0.496，有分散逻辑，但单独加入年化12.37%、夏普1.411，仍弱于默认。相关性低无法补偿本轮选时和机会成本。
- 主层全扩池禁止买入新增ETF，只保留其对D和IVOL比较的影响时，年化12.83%、夏普1.458；允许新ETF入选后降至9.60%、1.124。固定旧D也没有挽回全扩池，约9.39%、1.099。因此不能只把拖累归因于D缩放，也不能忽略IVOL参照集合变化。
- 煤炭单独加入仅49个持有日、4个周期，平均账户权重约0.24%；其表现变动还涉及旧仓被重新排序，不能用较少持仓样本认定因子对煤炭有稳定选择能力。

`cross_section_changes.csv`记录在完全不买新增资产时，旧目标或IVOL名单仍改变的周数。`log_attribution.csv`将默认、最强非默认、全主层等的累计对数净增长精确拆为各ETF与交易成本；它不是ETF自身年化，也不是各资产买入持有收益相加。完整持仓和相关性矩阵均保留，便于识别增益是否依赖单一资产或仅来自重排原仓。

## 验证、限制与复现

{len(checks)}项研究检查全部通过：CL120默认与当前生产整合函数对账、单只及代表性扩池目标与生产函数对账、5/20bps逐日引擎复核、上市前/仓位/名额/IVOL约束、历史准入、多个真实周末截点的价格截断与未来扰动、对数归因对账、默认配置与实盘计划指纹未变。测试验证实现，不能证明策略没有任何未知问题。

现存产品和当前规模筛选仍有幸存者与后视偏差；本轮识别了终止产品但未重建全市场历史清盘ETF。此前已反复开发并观察历史，不能将默认胜出或局部冠军当作充分的可盈利证明。理想回测允许分数单位，未模拟整手、最低佣金、盘中缺流动性、真实竞价价格和分红现金到账。保持CL120默认；行业覆盖合理性与提高回测收益是两项不同目标。

```powershell
python research_cl120_pool_expansion.py --screen
python research_cl120_pool_expansion.py --download
python research_cl120_pool_expansion.py
python research_cl120_pool_expansion.py --report-only
```

缓存`data/cl120_pool_expansion_20261109/`，全部结果`output/cl120_pool_expansion_20261109/`，均Git忽略。源代码固定CL120基线与截至11月6日的数据，后续重跑不会自动扩展到今天或修改实盘。

本轮基线行情复用`data/factor_extensions_20261109/`。新克隆环境若缺少该缓存，先运行`python research_factor_extensions.py --download-only`准备完整18只基线，再按上述顺序运行；每人使用自己的Tushare权限。只运行`--report-only`需要保留已生成的本轮结果，不请求新行情。

## 公开来源

- [易方达512070官方产品页：证券、保险及非银金融范围](https://www.efunds.com.cn/fund/512070_org.shtml)
- [上交所512400产品信息](https://www.sse.com.cn/assortment/fund/list/etfinfo/basic/index.shtml?FUNDID=512400)
- [富国159825官方产品与公告](https://wap.fullgoal.com.cn/fundDetail/159825/index.html)
- [上交所516110产品概要：汽车及零部件范围与上市日期](https://www.sse.com.cn/disclosure/fund/announcement/c/new/2026-03-19/516110_20260319_22W5.pdf)
- [前轮行业扩池报告与官方医药、电力、地产、化工资料](pool_expansion_20261031.md)
- [Tushare基金份额接口：单位万份](https://tushare.pro/document/2?doc_id=207)
'''
    (ROOT/'docs/cl120_pool_expansion_20261109.md').write_text(text,encoding='utf-8')
    print('Report and two figures complete.',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group();group.add_argument('--screen',action='store_true')
    group.add_argument('--download',action='store_true');group.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if args.screen:screen()
    elif args.download:download()
    elif args.report_only:report()
    else:execute();report()

if __name__=='__main__':main()
