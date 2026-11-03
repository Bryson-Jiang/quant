"""境外指数ETF扩池研究：默认策略不变，行情固定至2026-10-20。

python research_overseas_expansion.py --download
python research_overseas_expansion.py
python research_overseas_expansion.py --report-only
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import time
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etf_rotation.backtest import run_backtest
from etf_rotation.config import LEGACY_CORE_NAMES as ETF_NAMES
from etf_rotation.data import _client, _download_table, _token, load_symbol, tushare_code
from etf_rotation.factors import FactorSpec, compute_factor
from etf_rotation.mean_reversion import ReversionRule, reversion_states, defensive_funding_targets
from etf_rotation.portfolio_strategy import portfolio_signals
from etf_rotation.risk_strategy import RiskConfig, risk_controlled_signals, weekly_signal_dates
from etf_rotation.legacy_research import load_rotation_state
from research_overfitting_audit import CoreAudit, simulate, statistics
from research_pool_expansion import historical_eligibility, protected_fingerprints, markdown_table

ROOT=Path(__file__).resolve().parent
CACHE=ROOT/'data/overseas_expansion_20261031'
OUT=ROOT/'output/overseas_expansion_20261031'
END=date(2026,10,20)
SAT='512890'
BASE=RiskConfig.current_default()
CANDIDATES={'513500':'标普500ETF博时','159920':'恒生ETF华夏','513030':'德国DAXETF华安',
            '513080':'法国CAC40ETF华安','513180':'恒生科技ETF华夏',
            '513050':'中概互联网ETF易方达','513800':'日本东证指数ETF南方'}
BROAD=('513500','159920','513030','513080')
FOREIGN={'159941','513520',*CANDIDATES}
RULE=ReversionRule(200,0,.04,None)


def write_json(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8')


def experiment_cases():
    cases=[{'case':'default' if not c else 'add_'+'_'.join(c),'added':list(c),'removed':[],'family':'regional_grid'}
           for n in range(5) for c in itertools.combinations(BROAD,n)]
    cases += [{'case':'add_'+s,'added':[s],'removed':[],'family':'style_single'} for s in ('513180','513050')]
    cases += [{'case':'replace_nasdaq_sp500','added':['513500'],'removed':['159941'],'family':'substitution'},
              {'case':'replace_nikkei_topix','added':['513800'],'removed':['513520'],'family':'substitution'}]
    return cases


def plan():
    return {'research_date':'2026-10-31','end':END,'candidates':CANDIDATES,'cases':experiment_cases(),
            'selection':'2021年末前成立；优先当前规模20亿元以上，欧洲及TOPIX作为10亿元以上观察候选；不按回测结果换代表',
            'universe':'境内场内交易的跨境ETF；港股归境外市场，不称为外国股票；不需要海外券商账户',
            'primary':'2018年现金启动；2022年独立现金启动辅助；5bps和20bps；当前红利全历史假设',
            'robustness':'新增252真实报价、前1交易日20日成交额中位数>=1000万元、前1交易日份额市值>=2亿元；不寻优阈值',
            'overlap_control':'预设全体境外核心风险目标合计上限30%对照；只减持、不再分配名额，优先补国债否则现金',
            'data_timing':'只使用境内ETF交易价格形成信号；不拼接同日未来海外收盘；NAV只做规模统计，不输入历史信号',
            'excluded':{'513400':'道琼斯ETF2024年成立，历史太短','159329':'沙特ETF2024年成立，历史太短',
                        '513730':'东南亚科技2023年成立、历史较短且非地区宽基',
                        '159655':'标普500另一代表2022年成立，选更长历史的513500',
                        'other_nasdaq_nikkei':'现有纳指及日经，不按同指数换多个代码扩大排名票数'},
            'limitations':'现存基金及当前规模筛选仍有后视偏差，所有历史已观察；QDII份额及NAV存在发布延迟，不能用同日NAV估计为实时溢价',
            'protected_before':protected_fingerprints()}


def download():
    CACHE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'plan.json').exists():write_json(OUT/'plan.json',plan())
    for symbol in [*ETF_NAMES,SAT]:
        for suffix in ('.csv','.json'):
            source=ROOT/'data/audit_history'/f'tushare_{symbol}_hfq{suffix}'
            if not (CACHE/source.name).exists():shutil.copy2(source,CACHE/source.name)
    pro=_client();status=[]
    for s in CANDIDATES:
        for attempt in range(3):
            try:
                p=load_symbol(s,date(2010,1,1),END,CACHE)
                for kind,method in [('daily','fund_daily'),('share','fund_share')]:
                    path=CACHE/f'{kind}_{s}.csv'
                    if not path.exists():
                        frame=_download_table(pro,method,tushare_code(s),date(2010,1,1),END)
                        if frame.empty:raise ValueError('Empty '+method)
                        frame.to_csv(path,index=False,encoding='utf-8-sig')
                path=CACHE/f'nav_{s}.csv'
                if not path.exists():
                    nav=pro.fund_nav(ts_code=tushare_code(s),start_date='20260901',end_date='20261030')
                    if nav.empty:raise ValueError('Empty fund_nav')
                    nav.to_csv(path,index=False,encoding='utf-8-sig')
                status.append({'symbol':s,'ok':True,'first':str(p.index[0].date()),'last':str(p.index[-1].date()),'rows':len(p)})
                print(f'download {s}: {len(p)} quotes {p.index[0].date()}..{p.index[-1].date()}',flush=True)
                break
            except Exception as exc:
                status.append({'symbol':s,'ok':False,'attempt':attempt+1,'type':type(exc).__name__,
                               'diagnostic':str(exc).replace(_token(),'[REDACTED]')})
                write_json(OUT/'download_status.json',status)
                print(f'download {s}: {type(exc).__name__}, attempt {attempt+1}; details withheld',flush=True)
                if attempt==2:raise RuntimeError('Research download failed; see local redacted diagnostic') from None
                time.sleep(2)
    basic=pd.read_csv(ROOT/'data/audit_history/fund_basic_current_snapshot.csv',dtype=str)
    basic[basic.ts_code.isin([tushare_code(s) for s in CANDIDATES])].to_csv(CACHE/'candidate_basic.csv',index=False,encoding='utf-8-sig')
    write_json(OUT/'download_status.json',status)


def read_data():
    prices={s:load_symbol(s,date(2010,1,1),END,CACHE,offline=True) for s in [*ETF_NAMES,*CANDIDATES,SAT]}
    calendar=prices['510300'].index
    return pd.DataFrame({s:p.reindex(calendar) for s,p in prices.items()},index=calendar).rename_axis('date')


def known_nav(nav,day):
    """NAV按实际公告日期而非净值归属日期判断可得性；仅用于规模说明。"""
    f=nav.copy()
    for c in ('nav_date','ann_date'):f[c]=pd.to_datetime(f[c].astype(str),format='%Y%m%d',errors='coerce')
    f=f.loc[f.nav_date.le(day)&f.ann_date.le(day)].sort_values(['nav_date','ann_date'])
    if f.empty:raise ValueError('No NAV published by research date')
    return f.iloc[-1]


def metadata_and_allowed(prices):
    allowed=pd.DataFrame(True,index=prices.index,columns=prices.columns);records=[]
    basic=pd.read_csv(CACHE/'candidate_basic.csv',dtype=str).set_index('ts_code')
    for s in CANDIDATES:
        daily=pd.read_csv(CACHE/f'daily_{s}.csv',dtype={'trade_date':str})
        daily.index=pd.to_datetime(daily.trade_date,format='%Y%m%d');daily=daily.sort_index()
        shares=pd.read_csv(CACHE/f'share_{s}.csv',dtype={'trade_date':str})
        shares.index=pd.to_datetime(shares.trade_date,format='%Y%m%d');shares=shares.sort_index()
        if daily.index.has_duplicates or shares.index.has_duplicates:raise ValueError('Duplicate history dates')
        amount=pd.to_numeric(daily.amount).reindex(prices.index)*1000
        unit_shares=pd.to_numeric(shares.fd_share).reindex(prices.index).ffill(limit=5)*10000
        market_value=unit_shares*pd.to_numeric(daily.close).reindex(prices.index)
        allowed[s]=historical_eligibility(prices[s],amount,market_value)
        nav=pd.read_csv(CACHE/f'nav_{s}.csv',dtype={'nav_date':str,'ann_date':str})
        # QDII同日净值可能晚几天公布，规模估算也明确对应已公布的净值日与该日份额。
        row=known_nav(nav,pd.Timestamp(END))
        share_day=unit_shares.loc[:row.nav_date].dropna()
        if share_day.empty:raise ValueError('No shares for NAV size estimate')
        if (row.nav_date-share_day.index[-1]).days>7:raise ValueError('Stale shares for NAV size estimate')
        meta=basic.loc[tushare_code(s)]
        scale=float(share_day.iloc[-1])*float(row.unit_nav)/1e8
        if scale<10:raise ValueError(f'{s} 未满足事先固定的10亿元观察门槛')
        records.append({'symbol':s,'name':CANDIDATES[s],'found_date':meta.found_date,'list_date':meta.list_date,
                        'first_quote':str(prices[s].first_valid_index().date()),'quotes':int(prices[s].notna().sum()),
                        'nav_date_used':str(row.nav_date.date()),'nav_ann_date':str(row.ann_date.date()),
                        'scale_yi':scale,'median_amount20_yuan':float(amount.tail(20).median()),
                        'tier':'large' if scale>=20 else 'observation_10_to_20_yi',
                        'benchmark':meta.benchmark})
    return pd.DataFrame(records),allowed


def filtered_state_kwargs(state,prices):
    kwargs=state.signal_kwargs(prices)
    if 'initial_risky_selection' in kwargs:
        kwargs['initial_risky_selection']=[s for s in kwargs['initial_risky_selection'] if s in prices]
    if 'selection_resets' in kwargs:
        kwargs['selection_resets']=[(d,[s for s in selection if s in prices]) for d,selection in kwargs['selection_resets']]
    return kwargs


def make_targets(prices,case,state,sat_state,*,allowed=None,production=False):
    cols=[s for s in ETF_NAMES if s not in case['removed']]+case['added']
    core=prices[cols]
    if not production and allowed is None:
        signals=CoreAudit(core,state).signals(BASE)
    else:
        frames=None
        if allowed is not None:
            specs=[FactorSpec('momentum',20),FactorSpec('momentum',40),FactorSpec('trend',40),
                   FactorSpec('momentum',60),FactorSpec('volatility',60),FactorSpec('ma_zscore',20)]
            frames={x:compute_factor(core,x) for x in specs}
            frames[FactorSpec('trend',40)]=frames[FactorSpec('trend',40)].where(allowed[cols])
        signals=risk_controlled_signals(core,BASE,factor_frames=frames,**filtered_state_kwargs(state,core))
    p=core.assign(**{SAT:prices[SAT]})
    return p,defensive_funding_targets(p,signals,sat_state,SAT,.30)


def cap_foreign(targets,cap=.30):
    """固定境外核心合计上限；不增加总风险，削减部分补回原防守方向。"""
    out=targets.copy();columns=[s for s in out if s in FOREIGN]
    for day,row in out.iterrows():
        total=float(row[columns].sum())
        if total>cap:
            row.loc[columns]*=cap/total
            if row['511260']>0:row['511260']+=total-cap
            out.loc[day]=row
    return out


def holding_rows(p,led,label):
    w=pd.DataFrame(led.weights,index=led.dates,columns=p.columns);rows=[]
    for s in p:
        active=w[s].gt(1e-10);groups=active.ne(active.shift(fill_value=False)).cumsum()
        lengths=active[active].groupby(groups[active]).size()
        available=led.dates>=p[s].first_valid_index()
        rows.append({'case':label,'symbol':s,'name':CANDIDATES.get(s,ETF_NAMES.get(s,'红利低波')),
                     'held_days':int(active.sum()),'held_fraction_full':float(active.mean()),
                     'held_fraction_since_listing':float(active.loc[available].mean()) if available.any() else 0.,
                     'mean_weight':float(w[s].mean()),'max_weight':float(w[s].max()),'episodes':len(lengths),
                     'median_episode_days':float(lengths.median()) if len(lengths) else 0})
    return rows


def execute():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'plan.json').exists():write_json(OUT/'plan.json',plan())
    before=protected_fingerprints();prices=read_data();metadata,allowed=metadata_and_allowed(prices)
    metadata.to_csv(OUT/'candidate_metadata.csv',index=False,encoding='utf-8-sig')
    state=load_rotation_state(ROOT/'data/rotation_state.json')
    sat_state=reversion_states(prices[[SAT]],SAT,RULE,weekly_signal_dates(prices.index))
    cases=experiment_cases();results=[];daily={};holdings=[];checks=[];cache={};segments=[]
    baseline=portfolio_signals(prices[list(ETF_NAMES)],prices[[SAT]],state,hypothetical_satellite_history=True)
    full='add_'+'_'.join(BROAD)
    verify={'default',full,'add_513500','replace_nasdaq_sp500','replace_nikkei_topix'}
    for i,case in enumerate(cases):
        label=case['case'];p,t=make_targets(prices,case,state,sat_state);cache[label]=(p,t)
        t.to_csv(OUT/f'{label}_targets.csv',encoding='utf-8-sig')
        led=simulate(p,t,pd.Timestamp('2018-01-01'));daily[label]=led.series()
        holdings+=holding_rows(p,led,label)
        for start in ('2018-01-01','2022-01-01'):
            for cost in (5.,20.):
                ledger=led if start=='2018-01-01' and cost==5 else simulate(p,t,pd.Timestamp(start),cost_bps=cost)
                results.append({'case':label,'family':case['family'],'added':','.join(case['added']),
                                'removed':','.join(case['removed']),'start':start,'cost':cost,'skipped':ledger.skipped,
                                'annual_turnover':float(ledger.turnover.sum()*252/len(ledger.dates)),**statistics(ledger.returns)})
        for name,left,right in [('2018-2021','2018-01-01','2021-12-31'),('2022-2024','2022-01-01','2024-12-31'),('2025+','2025-01-01',str(END))]:
            segments.append({'case':label,'segment':name,**statistics(led.series().loc[left:right])})
        if label in verify:
            pp,tt=make_targets(prices,case,state,sat_state,production=True)
            err=float(np.max(np.abs(tt.to_numpy()-t.to_numpy())))
            checks.append({'check':label+'_production_signal_match','error':err,'passed':err<1e-10})
        if label in {'default',full}:
            for cost in (5.,20.):
                actual=run_backtest(p,t,date(2018,1,1),END,fee_bps=0,slippage_bps=cost)
                expected=simulate(p,t,pd.Timestamp('2018-01-01'),cost_bps=cost)
                err=float(np.max(np.abs(actual.equity.daily_return.to_numpy()-expected.returns)))
                checks.append({'check':label+f'_production_ledger_{cost:g}','error':err,'passed':err<1e-10})
        invalid=False
        for s in case['added']:
            invalid|=bool(t.loc[t.index<p[s].first_valid_index(),s].gt(0).any())
        core_risky=t.drop(columns=['511260',SAT])
        valid=(not invalid and core_risky.max().max()<=.30+1e-10 and core_risky.sum(axis=1).max()<=.70+1e-10
               and core_risky.gt(1e-10).sum(axis=1).max()<=3 and t.sum(axis=1).max()<=1+1e-10)
        checks.append({'check':label+'_listing_and_limits','passed':bool(valid)})
        print(f'case {i+1}/{len(cases)} {label}: complete',flush=True)
    err=float((cache['default'][1]-baseline.signals).abs().to_numpy().max())
    checks.append({'check':'integrated_default_match','error':err,'passed':err<1e-10})
    table=pd.DataFrame(results);table.to_csv(OUT/'all_results.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(daily).to_csv(OUT/'daily_returns.csv',encoding='utf-8-sig')
    pd.DataFrame(holdings).to_csv(OUT/'holding_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(segments).to_csv(OUT/'segment_results.csv',index=False,encoding='utf-8-sig')
    robust=[];country=[]
    # 对全部20个固定场景施加相同的新候选准入条件；没有挑选有效的门槛。
    for i,case in enumerate(cases):
        p,t=make_targets(prices,case,state,sat_state,allowed=allowed)
        for start in ('2018-01-01','2022-01-01'):
            for cost in (5.,20.):
                led=simulate(p,t,pd.Timestamp(start),cost_bps=cost)
                robust.append({'case':case['case'],'start':start,'cost':cost,**statistics(led.returns)})
        original_p,original_t=cache[case['case']]
        capped=cap_foreign(original_t)
        if capped[[s for s in capped if s in FOREIGN]].sum(axis=1).max()>.30+1e-10:raise ValueError('Foreign cap failed')
        for start in ('2018-01-01','2022-01-01'):
            for cost in (5.,20.):
                led=simulate(original_p,capped,pd.Timestamp(start),cost_bps=cost)
                country.append({'case':case['case'],'start':start,'cost':cost,**statistics(led.returns)})
        print(f'robust {i+1}/{len(cases)}: complete',flush=True)
    pd.DataFrame(robust).to_csv(OUT/'historical_admission_results.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(country).to_csv(OUT/'foreign_total30_results.csv',index=False,encoding='utf-8-sig')
    selected=table.query("start=='2018-01-01' and cost==5").sort_values('sharpe',ascending=False).iloc[0]['case']
    lookup={c['case']:c for c in cases}
    for label in dict.fromkeys(['default',full,selected,'replace_nasdaq_sp500']):
        for cutoff in (pd.Timestamp('2020-12-25'),pd.Timestamp('2023-12-29'),pd.Timestamp('2025-12-26')):
            partial=prices.loc[:cutoff]
            st=reversion_states(partial[[SAT]],SAT,RULE,weekly_signal_dates(partial.index))
            _,prefix=make_targets(partial,lookup[label],state,st)
            changed=prices.copy();changed.loc[changed.index>cutoff]*=np.linspace(.5,2.4,changed.shape[1])
            st=reversion_states(changed[[SAT]],SAT,RULE,weekly_signal_dates(changed.index))
            _,future=make_targets(changed,lookup[label],state,st)
            err=max(float((prefix-cache[label][1].loc[:cutoff]).abs().to_numpy().max()),
                    float((future.loc[:cutoff]-cache[label][1].loc[:cutoff]).abs().to_numpy().max()))
            checks.append({'check':label+'_future_'+str(cutoff.date()),'error':err,'passed':err<1e-10})
    prices.loc['2022-01-01':].pct_change(fill_method=None).corr().to_csv(OUT/'correlations_2022.csv',encoding='utf-8-sig')
    from research_pool_expansion import paired_bootstrap
    family=paired_bootstrap(pd.DataFrame(daily).to_numpy())
    write_json(OUT/'family_bootstrap.json',family)
    checks.append({'check':'production_and_live_unchanged','passed':before==protected_fingerprints()})
    write_json(OUT/'checks.json',checks)
    write_json(OUT/'manifest.json',{'plan':json.loads((OUT/'plan.json').read_text(encoding='utf-8')),
               'best_ex_post':selected,'checks_passed':all(r['passed'] for r in checks),'protected_before':before,
               'protected_after':protected_fingerprints(),
               'data_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in CACHE.glob('*.csv')}})
    if not all(r['passed'] for r in checks):raise AssertionError('Research validation failed')
    print(table.query("start=='2018-01-01' and cost==5")[['case','annual','sharpe','drawdown']].sort_values('sharpe',ascending=False).to_string(index=False),flush=True)


def supplementary_diagnostics():
    """主试验后的解释诊断；不混入事先20场景、不自动选择新参数。"""
    prices=read_data();state=load_rotation_state(ROOT/'data/rotation_state.json')
    sat_state=reversion_states(prices[[SAT]],SAT,RULE,weekly_signal_dates(prices.index))
    from research_pool_expansion import targets_for
    lookup={c['case']:c for c in experiment_cases()};rows=[]
    for label in ['default','add_513180','add_513500','add_513500_159920']:
        case=lookup[label]
        for mode in (['fixed_original_D'] if label=='default' else ['fixed_original_D','candidate_blocked_expanded_D']):
            if mode=='fixed_original_D':
                p,t=targets_for(prices,tuple(case['added']),state,sat_state,fixed_spread=True)
            else:
                allowed=pd.DataFrame(True,index=prices.index,columns=prices.columns)
                allowed[case['added']]=False
                p,t=make_targets(prices,case,state,sat_state,allowed=allowed)
            for start in ('2018-01-01','2022-01-01'):
                for cost in (5.,20.):
                    led=simulate(p,t,pd.Timestamp(start),cost_bps=cost)
                    rows.append({'case':label,'mode':mode,'start':start,'cost':cost,**statistics(led.returns)})
    pd.DataFrame(rows).to_csv(OUT/'posthoc_dispersion_diagnostics.csv',index=False,encoding='utf-8-sig')
    daily=pd.read_csv(OUT/'daily_returns.csv',index_col=0,parse_dates=True)
    confidence=[];rng=np.random.default_rng(20261031)
    for label in ('add_513180','add_513500_159920'):
        x=daily[['default',label]].to_numpy()
        for block in (20,60,120):
            values=[]
            for _ in range(2000):
                starts=rng.integers(0,len(x),size=int(np.ceil(len(x)/block)))
                idx=((starts[:,None]+np.arange(block))%len(x)).ravel()[:len(x)]
                a,b=statistics(x[idx,0]),statistics(x[idx,1])
                values.append([b['annual']-a['annual'],b['sharpe']-a['sharpe']])
            bounds=np.quantile(values,[.025,.975],axis=0)
            confidence.append({'case':label,'block':block,'reps':2000,
                               'annual_difference_ci95':bounds[:,0].tolist(),'sharpe_difference_ci95':bounds[:,1].tolist()})
    write_json(OUT/'conditional_paired_intervals.json',confidence)
    segments=pd.read_csv(OUT/'segment_results.csv');cases=pd.DataFrame(experiment_cases())
    choices=[]
    for train,test in [('2018-2021','2022-2024'),('2022-2024','2025+')]:
        frame=segments.loc[segments.segment.eq(train)].copy()
        frame['complexity']=frame['case'].map({c['case']:len(c['added'])+len(c['removed']) for c in experiment_cases()})
        winner=frame.sort_values(['sharpe','complexity','case'],ascending=[False,True,True]).iloc[0]['case']
        choices.append({'train':train,'test':test,'chosen':winner,
                        'next_segment':segments.loc[segments.segment.eq(test)&segments.case.isin(['default',winner])].to_dict('records')})
    write_json(OUT/'time_ordered_choices.json',choices)
    # 恒科自身8段价格路径仅用于理解信号，不称为扣费交易胜率。
    case=lookup['add_513180'];p,t=make_targets(prices,case,state,sat_state)
    led=simulate(p,t,pd.Timestamp('2018-01-01'));w=pd.DataFrame(led.weights,index=led.dates,columns=p.columns)
    active=w['513180'].gt(1e-10);groups=active.ne(active.shift(fill_value=False)).cumsum();episodes=[]
    for _,part in active[active].groupby(groups[active]):
        entry=part.index[0];last=part.index[-1];pos=led.dates.get_loc(last)+1
        exit_day=led.dates[pos] if pos<len(led.dates) else last
        episodes.append({'entry_execution':str(entry.date()),'exit_execution_or_end':str(exit_day.date()),
                         'closed':pos<len(led.dates),'held_days':len(part),
                         'etf_gross_price_return':float(p.at[exit_day,'513180']/p.at[entry,'513180']-1)})
    write_json(OUT/'hstech_episodes.json',episodes)
    print('Supplementary diagnostics saved',flush=True)


def plot_comparison():
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    prices=read_data();state=load_rotation_state(ROOT/'data/rotation_state.json')
    sat_state=reversion_states(prices[[SAT]],SAT,RULE,weekly_signal_dates(prices.index))
    labels={'default':'当前默认','add_513180':'加恒生科技','add_513500':'加标普500',
            'add_513500_159920':'加标普500＋恒生','add_'+'_'.join(BROAD):'四地区宽基全扩池'}
    cases={c['case']:c for c in experiment_cases()}
    fig,axes=plt.subplots(2,2,figsize=(15,10))
    fig.subplots_adjust(top=.90,bottom=.08,left=.06,right=.98,hspace=.30,wspace=.23)
    for label,name in labels.items():
        p,t=make_targets(prices,cases[label],state,sat_state)
        for j,start in enumerate(('2018-01-01','2022-01-01')):
            led=simulate(p,t,pd.Timestamp(start));nav=(1+led.series()).cumprod()
            axes[0,j].plot(nav.index,nav,label=name,lw=1.5)
            axes[1,j].plot(nav.index,(nav/nav.cummax().clip(lower=1)-1)*100,lw=1.2)
    for j,year in enumerate((2018,2022)):
        axes[0,j].set_title(f'{year}年现金启动：扣费净值')
        axes[1,j].set_title(f'{year}年现金启动：收盘回撤（%）')
    axes[0,0].legend(fontsize=9)
    for ax in axes.ravel():ax.grid(alpha=.22)
    fig.suptitle('境外指数ETF扩池 | 行情至2026-10-20 | 单边5bps | 同一默认规则，非盲测',fontsize=14,y=.97)
    fig.savefig(ROOT/'docs/assets/overseas_expansion_20261031.png',dpi=160,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)


def build_report():
    table=pd.read_csv(OUT/'all_results.csv');segments=pd.read_csv(OUT/'segment_results.csv')
    admission=pd.read_csv(OUT/'historical_admission_results.csv');capped=pd.read_csv(OUT/'foreign_total30_results.csv')
    meta=pd.read_csv(OUT/'candidate_metadata.csv',dtype={'symbol':str,'found_date':str})
    hold=pd.read_csv(OUT/'holding_summary.csv',dtype={'symbol':str})
    diag=pd.read_csv(OUT/'posthoc_dispersion_diagnostics.csv')
    intervals=json.loads((OUT/'conditional_paired_intervals.json').read_text(encoding='utf-8'))
    family=json.loads((OUT/'family_bootstrap.json').read_text(encoding='utf-8'))
    episodes=json.loads((OUT/'hstech_episodes.json').read_text(encoding='utf-8'))
    checks=json.loads((OUT/'checks.json').read_text(encoding='utf-8'))
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    if protected_fingerprints()!=manifest['protected_before'] or not all(r['passed'] for r in checks):
        raise ValueError('Protected files changed or checks failed')
    full='add_'+'_'.join(BROAD)
    names={'default':'当前默认','add_513500':'仅加标普500','add_159920':'仅加恒生',
           'add_513030':'仅加德国DAX','add_513080':'仅加法国CAC40','add_513180':'仅加恒生科技',
           'add_513050':'仅加中概互联','replace_nasdaq_sp500':'标普500替换纳指','replace_nikkei_topix':'东证替换日经',
           'add_513500_159920':'加标普500＋恒生',full:'四地区宽基全扩池'}
    def label(case):
        if case in names:return names[case]
        return '加'+'＋'.join(CANDIDATES[s] for s in case.removeprefix('add_').split('_'))
    def rows(source,start,cost=5,selected=None):
        f=source.loc[source.start.eq(start)&source.cost.eq(cost)].set_index('case')
        keys=list(names) if selected is None else selected
        return [[label(c),f'{f.at[c,"annual"]:.2%}',f'{f.at[c,"sharpe"]:.3f}',f'{f.at[c,"drawdown"]:.2%}'] for c in keys if c in f.index]
    headers=['方案','年化收益','零利率夏普','最大回撤']
    metadata_rows=[[r.symbol,CANDIDATES[r.symbol],pd.Timestamp(r.found_date).date(),r.first_quote,
                    f'{r.scale_yi:.2f}',r.nav_date_used,'较大规模' if r.scale_yi>=20 else '观察（不足20亿）'] for r in meta.itertuples()]
    segment_rows=[[label(r.case),r.segment,f'{r.annual:.2%}',f'{r.sharpe:.3f}',f'{r.drawdown:.2%}']
                  for r in segments.loc[segments.case.isin(['default','add_513180','add_513500_159920'])].itertuples()]
    h=hold.loc[hold.case.eq('add_513180')&hold.symbol.eq('513180')].iloc[0]
    cycle_rows=[[r['entry_execution'],r['exit_execution_or_end'],r['held_days'],f'{r["etf_gross_price_return"]:.2%}'] for r in episodes]
    corr=pd.read_csv(OUT/'correlations_2022.csv',index_col=0);corr.index=corr.index.astype(str)
    corr_rows=[[s,CANDIDATES[s],f'{corr.at[s,"159941"]:.3f}',f'{corr.at[s,"513520"]:.3f}',f'{corr.at[s,"510300"]:.3f}']
               for s in CANDIDATES]
    ci=[r for r in intervals if r['case']=='add_513180' and r['block']==60][0]
    fixed=diag.loc[diag.case.eq('add_513180')&diag['mode'].eq('fixed_original_D')&diag.start.eq('2018-01-01')&diag.cost.eq(5)].iloc[0]
    block=diag.loc[diag.case.eq('add_513180')&diag['mode'].eq('candidate_blocked_expanded_D')&diag.start.eq('2018-01-01')&diag.cost.eq(5)].iloc[0]
    text=f'''# 境外指数ETF扩池验证

研究日期2026-10-31，行情固定截至2026-10-20，与此前审计使用同一截止日。用户要求暂不改变策略；本轮默认池、参数、初始化状态、实盘信号及个人账户数据库均未调整。

## 结论与候选定位

本轮最值得观察的是513180恒生科技：2018主区间年化12.07%、夏普1.376，略高于默认11.55%、1.322；2022以来为18.63%、2.117，回撤4.60%略深于默认4.43%。历史准入及高成本对照仍有增益，但配对抽样区间包含零、家族校正不显著，暂不能认定具有稳定优势。

513500标普500与159920恒生补充较宽的市场覆盖，合加后2022年以来年化18.90%、夏普2.154，但2018主区间11.27%、1.300，低于默认。其优势依赖观察区间，不能只选近年结果。德国/法国没有显示稳定组合改善。用标普替换纳指可降低主区间最大回撤至9.45%，同时收益降至11.16%；东证替换日经未改善。所有方案仅为研究，不上线。

## 候选、规模与市场范围

研究对象仍是沪深交易所上市、人民币交易的跨境ETF，不要求使用境外券商账户。香港属于境外市场，不称为外国市场。4个地区/大盘宽基全部16个子集（含原池）＋2个风格单独新增＋2个指数替换，共20个事先固定场景。没有因子窗口、风险预算或调仓频率搜索，也没有临时加入此前行业候选。

{markdown_table(['代码','候选','成立日','首个真实报价','估算规模/亿元','规模净值归属日','规模层级'],metadata_rows)}

规模来自Tushare份额×单位净值估算，仅使用研究截止日已经公告的NAV，并配对该净值日已知的份额；不是以未公布的同日境外净值估算。全部在2021年末前成立。优先20亿元以上；德国、法国、TOPIX体量较小，事先明确放在10—20亿元观察层，不沿用上一轮行业扩池的20亿元上线偏好。并未因回测结果更换代表。

- 513500标普500：美股大盘行业覆盖比现有纳指更广，但与纳指仍高度重叠，新增代码不等于新增独立风险。
- 159920恒生：补充香港市场的大盘综合指数，包含科技和金融等；仍受中国经济、政策及全球流动性影响。
- 513030德国DAX、513080法国CAC40：补充欧元区国家股票；二者也有共同欧洲/汇率风险，不代表整个欧洲或全球市场。
- 513180恒生科技、513050中概互联：风格/行业补充，不是两个新国家宽基。分别单独添加，没有同时塞入它们来增加相似资产的排名票数。
- 513800东证：仅测试替换现有日经，比较更宽日本股票指数；不重复增加同一国家的仓位名额。

沙特159329、道琼斯513400等2024年成立、东南亚科技513730于2023年成立，本轮因历史短不进入回测。标普的其他较新代表不重复测试；现有纳指和日经也不换多个相同指数代码去挑历史赢家。

## 方法与跨时区处理

2018主账户从首个交易日现金启动；2018前价格只预热。2022辅助账户另从现金启动，不能与2018账本分段混用。核心仍为20/40日风险调整动量＋20日均线反转0.5、40日趋势、8%核心波动预算、70%核心风险仓、单只30%、最多3只核心风险ETF、前4缓冲且必选第1名。周末端开市日收盘出目标，下一交易日收盘成交。

卫星仍为512890 MA200／入0%／出+4%／先现金后国债／最高30%，明确采用全历史假设，与之前审计一致，不是实际历史收益。全账户风险仓可能100%、最多4只风险ETF加国债。单边5bps（3费＋2滑点预算），另测20bps；年化按252交易日，夏普无风险利率为零，回撤按收盘净值。

全部收益和动量来自境内ETF真实交易日的人民币复权收盘价，未使用当日晚间才形成的美国/欧洲指数收盘。人民币价格已包含汇率及ETF市场溢折价影响，不额外重复叠加汇率。没有把境外指数历史回填到ETF上市前。估值最多沿用3个过去交易日，实际成交必须当日有报价；缺价取消整批调仓，长期缺价报错。

QDII净值的归属日期不等于发布时间，`known_nav`按`ann_date`限制可得性。NAV不输入选仓、风险或收益，也没有拿同日期境内收盘与稍后形成的境外NAV计算“实时溢价”。公告只有日期、没有时刻，不能把本次规模估算当作盘中可用净值。报价仍有价差、限额、停牌和溢折价风险；5/20bps压力对照不是完整盘口模型。未模拟整数手、成交容量或实际现金分红到账。

## 2018年以来主结果

{markdown_table(headers,rows(table,'2018-01-01'))}

## 2022年以来辅助结果

{markdown_table(headers,rows(table,'2022-01-01'))}

![境外扩池净值与回撤](assets/overseas_expansion_20261031.png)

## 分段与稳定性

以下全部继承同一2018账本仓位，不是重新选择最有利现金启动日。

{markdown_table(['方案','区间','年化','夏普','最大回撤'],segment_rows)}

恒科主区间最大回撤与默认相同，主要因为全期最深峰谷位于2018年，当时恒科ETF尚未上市，不能据此推断加入恒科完全不增加熊市风险。2022后回撤4.60%对比4.43%，显示实际仍可能加深。

2018—2021选出法国ETF后，2022—2024夏普2.046，低于默认2.084；2022—2024选出恒科后，2025以后夏普2.068，高于默认1.965。时间顺序结果有好有坏。所有年份已被研究者观察，不能称为独立盲测。

## 费用、历史准入与境外集中度

单边20bps，2018以来：

{markdown_table(headers,rows(table,'2018-01-01',20,['default','add_513180','add_513500','add_513500_159920']))}

新候选固定准入：真实报价满252日、前1交易日20日成交额中位数至少1000万元、前1交易日份额×原始价估计市值至少2亿元、当天有真实报价。份额最多只向前沿用5日，不向后填充。对新增候选收紧准入，不改原池既有准入，也不保证完全解决历史份额发布/修订问题。

{markdown_table(headers,rows(admission,'2018-01-01',5,['default','add_513180','add_513500','add_513500_159920']))}

境外核心目标合计30%对照同时限制原纳指、日经及新境外ETF。削减部分在原本有国债时补回国债，否则留现金，不增加总风险或名额。必须与同样受限的默认作比较，而非只与不受限默认比较：

{markdown_table(headers,rows(capped,'2018-01-01',5,['default','add_513180','add_513500','add_513500_159920']))}

该合计限制是事先固定的风险对照，不是本轮新默认；卫星红利及国内资产不计入境外额度。

## 持仓、相关与缩放诊断

恒科扩池方案实际持有恒科{int(h.held_days)}个交易日，占全期{h.held_fraction_full:.2%}、上市后交易日{h.held_fraction_since_listing:.2%}；全期平均账户权重{h.mean_weight:.2%}、最大漂移权重{h.max_weight:.2%}，共{int(h.episodes)}段连续持有，周期中位{h.median_episode_days:g}天。不是长期重仓恒科获取的收益。

{markdown_table(['进入执行日','退出执行日','持有天数','ETF价格毛回报'],cycle_rows)}

周期回报是ETF自身复权价格变化，未扣费用、不含账户权重再平衡，不冒充实际交易胜率或相对默认的精确贡献。存在单周期约5%—7%的价格下跌；较大上涨集中在2022年反弹、2024年9月后及2025年初，8个周期样本仍小。

2022以来境内交易价格日收益的成对相关（仅事后诊断，不输入信号）：

{markdown_table(['候选','名称','与现有纳指','与现有日经','与沪深300'],corr_rows)}

现有均线反转项受池内原始评分离散度D缩放。主试验后另做固定原15只风险ETF的D，以及“保留扩池D但禁止买新增ETF”解释对照，共7种×2起点×2费用，单独保存28个结果。恒科固定原D仍为年化{fixed.annual:.2%}、夏普{fixed.sharpe:.3f}；保留扩池D却禁止买恒科时为{block.annual:.2%}、{block.sharpe:.3f}。增益未完全由缩放解释，但不能据此证明因果稳定优势。

## 统计证据与实现检查

20个主场景×2起点×2费用＝80个结果；历史准入另80个；境外30%上限另80个。全部保留，不删除失败候选。主场景的家族内均值增益循环块抽样20/60/120日、各1500次，探索性p约{', '.join(f'{r["family_mean_gain_p"]:.3f}' for r in family)}，未达到5%显著性。检验不涵盖项目之前所有选池/因子搜索，不能作为全项目过拟合概率。

恒科相对默认的配对60日块抽样、2000次，年化增益95%条件区间约[{ci['annual_difference_ci95'][0]*100:+.2f}, {ci['annual_difference_ci95'][1]*100:+.2f}]个百分点；夏普增益约[{ci['sharpe_difference_ci95'][0]:+.3f}, {ci['sharpe_difference_ci95'][1]:+.3f}]。区间包含零，而且条件于已开发策略与当前样本，未校正事后挑选恒科的全部偏差，不是未来收益范围。

全部{len(checks)}项研究检查通过：生产整合默认对账；默认、全宽基、标普新增及两个替换的生产信号对账；5/20bps逐日账本对账；20池上市前及仓位名额检查；多个历史截点前缀截断和未来价格扰动；实盘文件指纹不变。新增NAV公告日、替换初始化和境外总仓约束测试，完整110项测试通过。仍保留当前大规模存续ETF选择、交易流动性、时区、报价、幸存者及多轮调参偏差。

建议保持现有默认，恒科列为首要观察候选，标普500/恒生列为较宽市场覆盖候选；欧洲和TOPIX暂无收益证据支持纳入。下一步若继续研究，应预先固定候选和验收标准，在后续新样本验证，不能将本轮最好的区间或场景当作已知可实现收益。

## 复现

```powershell
python research_overseas_expansion.py --download
python research_overseas_expansion.py
python research_overseas_expansion.py --diagnostics
python research_overseas_expansion.py --report-only
```

独立行情在`data/overseas_expansion_20261031/`；全部数表、每日收益、目标、持仓、时间顺序诊断、条件区间与指纹在`output/overseas_expansion_20261031/`。研究代码复用此前已验证的账本与历史准入函数，不改生产配置。研究数据不会自动更新到当前日期。

## 公开资料

- [博时标普500产品概要：2013年成立、2014年上市](https://www.sse.com.cn/disclosure/fund/announcement/c/new/2026-06-26/513500_20260626_PDT5.pdf)
- [华夏恒生官方介绍](https://fund.chinaamc.com/portal/cn/hxxc/2013/ETF-cxyx/index4.html)
- [华安德国DAX官方产品](https://wap.huaan.com.cn/funds/513030/index.shtml)
- [华安法国CAC40官方产品](https://www.huaan.com.cn/funds/513080/index.shtml)
- [华夏恒生科技产品概要](https://www.sse.com.cn/disclosure/fund/announcement/c/new/2026-03-23/513180_20260323_XSB4.pdf)
- [易方达中概互联年度报告：指数与成立日](https://www.sse.com.cn/disclosure/fund/announcement/c/2018-03-30/513050_2017_n.pdf)
- [南方TOPIX官方产品](https://www.southernfund.com/main/jjcp/fundproduct/513800.shtml)
'''
    (ROOT/'docs/overseas_expansion_20261031.md').write_text(text,encoding='utf-8')
    write_json(OUT/'final_validation.json',{'checks':len(checks),'unit_tests':110,'all_checks_passed':True,
               'production_unchanged':True,'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                 for p in ['research_overseas_expansion.py','tests/test_overseas_expansion.py','research_pool_expansion.py','research_overfitting_audit.py']}})
    print('Report saved: docs/overseas_expansion_20261031.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--diagnostics',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if args.download:download()
    elif args.diagnostics:supplementary_diagnostics()
    elif args.report_only:
        plot_comparison();build_report()
    else:execute()
